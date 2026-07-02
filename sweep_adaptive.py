"""
Fast vectorized sweep: adaptive confirmation modes across 2024+2025 XAUUSD.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.signal_engine import SignalEngine
from app.bias_engine import BiasEngine
from app.risk_manager import EquityScaler
from app.dukascopy_client import DukascopyClient
import config as cfg

INITIAL_BALANCE = 20.0
CONTRACT_SIZE = 100
SPREAD_PER_LOT_USD = 25.0
SLIPPAGE_PER_LOT_USD = 3.0
SPREAD_ROUND_TRIP = 2
LEVERAGE = 200
MARGIN_MAX_PCT = 1.0
BIAS_WARMUP_DAYS = 14

ADAPTIVE_MODES = {
    "baseline":       {"type": "none"},
    "vreg_loose":     {"type": "vreg", "style": "loose",  "p_low": 30,  "p_norm": None, "p_high": None},
    "vreg_normal":    {"type": "vreg", "style": "normal", "p_low": 30,  "p_norm": None, "p_high": None},
    "vreg_tight":     {"type": "vreg", "style": "tight",  "p_low": 50,  "p_norm": None, "p_high": None},
    "vreg_vtight":    {"type": "vreg", "style": "vtight", "p_low": 60,  "p_norm": 40,   "p_high": None},
    "vreg_xtight":    {"type": "vreg", "style": "xtight", "p_low": 70,  "p_norm": 50,   "p_high": 30},
    "vreg_maxtight":  {"type": "vreg", "style": "maxtight", "p_low": 80, "p_norm": 60,   "p_high": 40},
}


def _session_allowed(ts):
    s = cfg.ALLOWED_SESSIONS.upper()
    if s == 'ALL': return True
    h = ts.hour
    parts = [x.strip() for x in s.split(',')]
    if 'ASIA' in parts and 0 <= h < 8: return True
    if 'LONDON' in parts and 8 <= h < 17: return True
    if 'NEW_YORK' in parts and 13 <= h < 22: return True
    return False

def _spread_mult(ts):
    h = ts.hour
    if 13 <= h < 17: return 0.9
    if 8 <= h < 13: return 1.0
    if 0 <= h < 8: return 1.6
    return 1.3

def _pnl(ep, xp, direction, lot, n, et=None):
    delta = xp - ep if direction == 'BUY' else ep - xp
    gross = delta * CONTRACT_SIZE * lot * n
    sm = _spread_mult(et) if et is not None else 1.0
    sc = SPREAD_PER_LOT_USD * lot * n * SPREAD_ROUND_TRIP * sm
    slip = SLIPPAGE_PER_LOT_USD * lot * n
    return gross - sc - slip

def load_data(from_dt, to_dt):
    t0 = time.time()
    client = DukascopyClient()
    m1 = client.download_range(from_dt.year - 1, to_dt.year)
    if len(m1) == 0: return None, None
    warmup = from_dt - timedelta(days=BIAS_WARMUP_DAYS)
    m1 = m1[(m1['time'] >= warmup) & (m1['time'] <= to_dt)]
    if len(m1) == 0: return None, None
    print(f'M1: {len(m1)} bars', flush=True)
    h1 = client.resample_to(m1, 16385)
    m5 = client.resample_to(m1, 5)
    h1 = h1[h1['time'] >= warmup].copy()
    m5 = m5[m5['time'] >= from_dt].copy()
    print(f'  -> M5: {len(m5)} H1: {len(h1)} [{time.time()-t0:.0f}s]', flush=True)
    return m5, h1

def vectorized_pre_compute(m5, h1):
    t0 = time.time()
    N = len(m5)
    print(f'Vectorized pre-compute {N} bars...', flush=True)

    m5_times = m5['time'].values
    m5_close = m5['close'].values
    m5_open = m5['open'].values
    m5_high = m5['high'].values
    m5_low = m5['low'].values
    h1_times = h1['time'].values.astype(np.int64)
    m5_tns = m5_times.astype(np.int64)
    h1_idx = np.searchsorted(h1_times, m5_tns, side='right') - 2
    h1_idx = np.clip(h1_idx, 0, len(h1)-1)
    h1_high = h1['high'].values[h1_idx]
    h1_low = h1['low'].values[h1_idx]
    h1_range = h1_high - h1_low
    h1_range = np.maximum(h1_range, 1e-10)

    max_breakout = np.maximum(m5_high - h1_high, h1_low - m5_low)
    breakout_score = max_breakout / h1_range
    breakout_direction = np.where(m5_high - h1_high > h1_low - m5_low, 'BUY', 'SELL')

    body_ratio = np.abs(m5_close - m5_open) / np.maximum(m5_high - m5_low, 1e-10)

    tr = np.maximum(m5_high[1:] - m5_low[1:],
                    np.maximum(np.abs(m5_high[1:] - m5_close[:-1]),
                               np.abs(m5_low[1:] - m5_close[:-1])))
    atr_series = np.zeros(N)
    for i in range(14, N):
        atr_series[i] = np.mean(tr[i-14:i])

    def calc_momentum(idx):
        if idx < 6: return 0.0
        c = m5_close[:idx+1]
        o = m5_open[:idx+1]
        h = m5_high[:idx+1]
        l = m5_low[:idx+1]
        recent = c[-3:]; older = c[-6:-3]
        rc = abs(recent[-1] - recent[0])
        oc = abs(older[-1] - older[0]) if len(older) >= 2 else rc
        atr_v = atr_series[idx]
        ref = max(atr_v, c[-1] * 0.0001)
        avg_body = np.mean(np.abs(c[-5:] - o[-5:]))
        if avg_body == 0 or ref == 0: return 0.0
        raw = (rc / (oc + 1e-10)) * (avg_body / (ref + 1e-10))
        return min(abs(raw), 1.0)

    momentum = np.array([calc_momentum(i) for i in range(N)])

    be = BiasEngine()
    bias_arr = [None] * N
    last_bh = -1
    bias_cache = {}
    for idx in range(N):
        ts = m5['time'].iloc[idx]
        bh = ts.hour
        if bh == last_bh and bias_arr[idx-1] is not None:
            bias_arr[idx] = bias_arr[idx-1]
            continue
        last_bh = bh
        key = ts.replace(minute=0, second=0, microsecond=0)
        if key in bias_cache:
            bias_arr[idx] = bias_cache[key]
            continue
        hs = h1[h1['time'] < key].tail(96)
        if len(hs) >= 20:
            bs = be.update(hs)
            bias_arr[idx] = bs
            bias_cache[key] = bs
        elif idx > 0:
            bias_arr[idx] = bias_arr[idx-1]

    print(f'  biases done', flush=True)

    se = SignalEngine()
    sig_list = [None] * N
    for idx in range(10, N):
        bs = bias_arr[idx]
        if bs is None: continue
        if bs.get('bias') not in ('BULLISH', 'BEARISH') or bs.get('strength', 0) < cfg.BIAS_STRENGTH_MIN: continue
        if breakout_score[idx] < cfg.SIGNAL_ENTRY_THRESHOLD: continue
        if idx % 15000 == 0: print(f'  signal: {idx}/{N}', flush=True)
        win = m5.iloc[max(0, idx-50):idx]
        sig = se.evaluate(win, bs, float(m5_close[idx]),
                          h1_high=float(h1_high[idx]),
                          h1_low=float(h1_low[idx]))
        if sig:
            sig_list[idx] = sig

    sig_cnt = sum(1 for s in sig_list if s is not None)
    print(f'  {sig_cnt} signals [{time.time()-t0:.0f}s]', flush=True)
    return {
        'sig_df': m5, 'sig_signal': sig_list, 'breakout_score': breakout_score,
        'body_ratios': body_ratio, 'momentums': momentum, 'atr': atr_series,
        'bias_arr': bias_arr, 'signal_engine': se,
    }


def check_conf(br, mom, atr, conf, br_win, atr_win):
    t = conf['type']
    if t == 'none': return True
    if len(br_win) < 10: return True
    if t == 'pbr':
        return br >= np.percentile(br_win, conf['p'])
    if t == 'mom_atr':
        if atr <= 0: return False
        return (mom / (atr + 1e-10)) >= conf['t']
    if t == 'vreg':
        if len(atr_win) < 50: return True
        lo, hi = np.percentile(atr_win, [25, 75])
        p_low = conf.get('p_low')
        p_norm = conf.get('p_norm')
        p_high = conf.get('p_high')
        if atr <= lo and p_low is not None:
            return br >= np.percentile(br_win, p_low)
        elif lo < atr < hi and p_norm is not None:
            return br >= np.percentile(br_win, p_norm)
        elif atr >= hi and p_high is not None:
            return br >= np.percentile(br_win, p_high)
        return True
    if t == 'combined':
        ok = True
        if conf.get('pbr_p') and br < np.percentile(br_win, conf['pbr_p']): ok = False
        if conf.get('moma_t') and atr > 0 and (mom / (atr + 1e-10)) < conf['moma_t']: ok = False
        return ok
    return True


def run_trades(data, conf):
    df = data['sig_df']
    sigs = data['sig_signal']
    scores = data['breakout_score']
    brs = data['body_ratios']
    moms = data['momentums']
    atrs = data['atr']
    bias_arr = data['bias_arr']
    N = len(df)

    scaler = EquityScaler(); scaler.initialize(INITIAL_BALANCE)
    balance = INITIAL_BALANCE
    trades = []; cur = None; cl = 0; dt = 0
    cur_day = None; peak = INITIAL_BALANCE; cd = None; dpnl = 0.0
    br_win = []; atr_win = []

    for idx in range(N):
        ts = df['time'].iloc[idx]
        row = df.iloc[idx]; px = float(row['open'])
        if cur_day != ts.date():
            cur_day = ts.date(); dt = 0; dpnl = 0.0; cl = 0
        br = float(brs[idx]); mom = float(moms[idx]); atr = float(atrs[idx])
        br_win.append(br); atr_win.append(atr)
        if len(br_win) > 100: br_win.pop(0)
        if len(atr_win) > 100: atr_win.pop(0)
        bs = bias_arr[idx]
        if bs is None: continue

        if cur is not None:
            xp = float(row['close'])
            epnl = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], et=cur['entry_time'])
            ew = df.iloc[max(0, idx-20):idx]
            sx, sv, sr = data['signal_engine'].evaluate_exit(
                ew, cur['entry_price'], cur['direction'], cur.get('entry_score'),
                exit_threshold=cfg.EXIT_THRESHOLD_TIGHT, exit_mode=6, signal=cur.get('entry_signal'))
            if sx:
                p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], et=cur['entry_time'])
                balance += p; dpnl += p
                if balance > peak: peak = balance
                scaler.update_peak(balance)
                trades.append({'p': round(p, 2)})
                cl = 0 if p > 0 else cl + 1
                cd = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + cl))
                cur = None; continue
            if epnl <= -cfg.MAX_EVENT_LOSS_USD:
                p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], et=cur['entry_time'])
                balance += p; dpnl += p
                if balance > peak: peak = balance
                trades.append({'p': round(p, 2)})
                cl += 1; cd = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + cl))
                cur = None; continue

        if cd is not None and ts < cd: continue
        cd = None

        if cur is None:
            if not _session_allowed(ts): continue
            if cl >= cfg.MAX_CONSECUTIVE_LOSSES or balance < cfg.MIN_BALANCE: continue
            if dpnl <= -cfg.MAX_DAILY_LOSS_USD: continue
            if dt >= cfg.MAX_TRADES_PER_SESSION: continue
            if bs.get('bias') not in ('BULLISH', 'BEARISH') or bs.get('strength', 0) < cfg.BIAS_STRENGTH_MIN: continue

            sig = sigs[idx]
            if sig is None: continue
            sc = scores[idx]
            if sc < cfg.SIGNAL_ENTRY_THRESHOLD: continue
            atr_th = sig.get('atr_entry_threshold')
            min_et = max(atr_th if atr_th else 0, cfg.SIGNAL_ENTRY_THRESHOLD)
            if sc < min_et: continue

            if not check_conf(br, mom, atr, conf, br_win, atr_win): continue

            lot = min(scaler.get_lot(balance) * cfg.LOT_MULTIPLIER, cfg.MAX_LOT)
            lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP
            lot = max(cfg.MIN_LOT, lot)
            num_tr = min(cfg.MAX_TRADES_PER_EVENT, scaler.get_trades_per_event(balance, sc))
            margin_per = (float(row['close']) * CONTRACT_SIZE) / LEVERAGE
            max_by_margin = (balance * MARGIN_MAX_PCT) / margin_per
            if max_by_margin < cfg.MIN_LOT: continue
            lot = min(lot, max_by_margin)
            lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP
            lot = max(cfg.MIN_LOT, lot)
            num_tr = max(1, min(num_tr, int(max_by_margin / max(lot, 1e-9))))
            if lot * num_tr <= 0: continue
            cur = {'entry_time': ts, 'entry_price': px, 'direction': sig['direction'],
                   'lot': lot, 'num_trades': num_tr, 'entry_score': float(sc), 'entry_signal': sig}
            dt += 1

    if cur is not None:
        xp = float(df['close'].iloc[-1])
        p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], et=cur['entry_time'])
        balance += p
        trades.append({'p': round(p, 2)})

    if not trades: return None
    pnls = np.array([t['p'] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    wr = len(wins)/len(pnls)*100 if len(pnls) > 0 else 0
    gp = wins.sum() if len(wins) > 0 else 0
    gl = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gp/gl if gl > 0 else (999 if gp > 0 else 0)
    cum = np.cumsum(pnls)
    dd = (np.maximum.accumulate(cum) - cum).max()
    return {'trades': len(pnls), 'net_pnl': round(pnls.sum(), 2), 'win_rate': round(wr, 1),
            'profit_factor': round(pf, 2), 'max_dd': round(dd, 2), 'final_bal': round(balance, 2)}


def main():
    dt_from = datetime(2025, 1, 1)
    dt_to = datetime(2025, 12, 31, 23, 59)
    print(f'Sweep: {dt_from.date()} to {dt_to.date()}', flush=True)
    m5, h1 = load_data(dt_from, dt_to)
    if m5 is None: print('No data'); return
    comp = vectorized_pre_compute(m5, h1)

    results = []
    t0 = time.time()
    for name, conf in ADAPTIVE_MODES.items():
        r = run_trades(comp, conf)
        if r is None:
            r = dict.fromkeys(['trades','net_pnl','win_rate','profit_factor','max_dd','final_bal'], 0)
        r['name'] = name; r.update(conf)
        results.append(r)

    results.sort(key=lambda x: x['profit_factor'], reverse=True)
    print(f'{"="*95}', flush=True)
    print(f'  ADAPTIVE CONFIRMATION SWEEP (2025)', flush=True)
    print(f'{"="*95}', flush=True)
    print(f'  {"Mode":<18} {"Trades":>6} {"Net PnL":>10} {"WR%":>6} {"PF":>6} {"Max DD":>10} {"Final Bal":>9}', flush=True)
    print(f'  {"-"*70}', flush=True)
    for r in results:
        print(f'  {r["name"]:<18} {r["trades"]:>6} {r["net_pnl"]:>+10.2f} {r["win_rate"]:>5.1f}% {r["profit_factor"]:>6.2f} {r["max_dd"]:>10.2f} {r["final_bal"]:>9.2f}', flush=True)

    # Extend to combined 2024+2025 for top 3
    print(f'\n{"="*95}', flush=True)
    print(f'  VALIDATE TOP 3 ON 2024+2025 COMBINED', flush=True)
    print(f'{"="*95}', flush=True)
    top3 = results[:3]
    m5_2, h1_2 = load_data(datetime(2024, 1, 1), datetime(2025, 12, 31, 23, 59))
    if m5_2 is not None:
        comp2 = vectorized_pre_compute(m5_2, h1_2)
        for r in top3:
            conf = {k: r[k] for k in ('type','p','t','style','pbr_p','moma_t','p_low','p_norm','p_high') if k in r}
            vr = run_trades(comp2, conf)
            if vr is None:
                vr = dict.fromkeys(['trades','net_pnl','win_rate','profit_factor','max_dd','final_bal'], 0)
            print(f'  {r["name"]:<18} 2025: PnL={r["net_pnl"]:>+8.2f} WR={r["win_rate"]:>5.1f}% {r["trades"]:>4}t PF={r["profit_factor"]:.2f} DD={r["max_dd"]:>7.2f}', flush=True)
            print(f'  {"":<18} 24-25: PnL={vr["net_pnl"]:>+8.2f} WR={vr["win_rate"]:>5.1f}% {vr["trades"]:>4}t PF={vr["profit_factor"]:.2f} DD={vr["max_dd"]:>7.2f} Bal={vr["final_bal"]:>8.2f}', flush=True)

    with open('sweep_adaptive_results.json', 'w') as f:
        clean = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in results]
        json.dump(clean, f, indent=2)
    print(f'\nDone [{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
