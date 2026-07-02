"""
Fast sweep: test entry confirmation modes on 1-month XAUUSD, validate top on full year.
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
CASH_FLOWS = []

CONF_MODES = {
    "baseline":       {"br": None, "cons": 1, "mom": None},
    "body_ratio_04":  {"br": 0.40, "cons": 1, "mom": None},
    "body_ratio_05":  {"br": 0.50, "cons": 1, "mom": None},
    "consecutive_2":  {"br": None, "cons": 2, "mom": None},
    "consecutive_3":  {"br": None, "cons": 3, "mom": None},
    "momentum_010":   {"br": None, "cons": 1, "mom": 0.10},
    "momentum_015":   {"br": None, "cons": 1, "mom": 0.15},
    "momentum_020":   {"br": None, "cons": 1, "mom": 0.20},
    "br04_mom010":    {"br": 0.40, "cons": 1, "mom": 0.10},
    "br04_mom015":    {"br": 0.40, "cons": 1, "mom": 0.15},
    "cons2_mom010":   {"br": None, "cons": 2, "mom": 0.10},
    "cons2_mom015":   {"br": None, "cons": 2, "mom": 0.15},
}
def _session_allowed(ts, sessions_str):
    if sessions_str.upper() == 'ALL':
        return True
    h = ts.hour
    sessions = [s.strip().upper() for s in sessions_str.split(',')]
    if 'ASIA' in sessions and 0 <= h < 8: return True
    if 'LONDON' in sessions and 8 <= h < 17: return True
    if 'NEW_YORK' in sessions and 13 <= h < 22: return True
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

def _body_ratio(row):
    rg = float(row['high'] - row['low'])
    if rg <= 0: return 0.0
    return abs(float(row['close'] - row['open'])) / rg

def _atr(high, low, close, period=14):
    if len(high) < period + 1: return 0.0
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    return float(np.mean(tr[-period:]))

def _momentum_at(df, idx, period=14):
    if idx < 6: return 0.0
    closes = df['close'].values[:idx+1]
    opens_ = df['open'].values[:idx+1]
    highs = df['high'].values[:idx+1]
    lows = df['low'].values[:idx+1]
    if len(closes) < 6: return 0.0
    recent = closes[-3:]; older = closes[-6:-3]
    rc = abs(recent[-1] - recent[0])
    oc = abs(older[-1] - older[0]) if len(older) >= 2 else rc
    atr_v = _atr(highs, lows, closes, period)
    ref = max(atr_v, closes[-1] * 0.0001)
    avg_body = np.mean(np.abs(closes[-5:] - opens_[-5:]))
    if avg_body == 0 or ref == 0: return 0.0
    raw = (rc / (oc + 1e-10)) * (avg_body / (ref + 1e-10))
    return min(abs(raw), 1.0)

def load_data(from_dt, to_dt):
    t0 = time.time()
    client = DukascopyClient()
    m1 = client.download_range(from_dt.year - 1, to_dt.year)
    if len(m1) == 0: return None
    warmup = from_dt - timedelta(days=BIAS_WARMUP_DAYS)
    m1 = m1[(m1['time'] >= warmup) & (m1['time'] <= to_dt)]
    if len(m1) == 0: return None
    print(f'M1: {len(m1)} bars ({m1["time"].min()} to {m1["time"].max()}) [{time.time()-t0:.0f}s]')
    h1 = client.resample_to(m1, 16385)
    m5 = client.resample_to(m1, 5)
    h1 = h1[h1['time'] >= warmup].copy()
    m5 = m5[m5['time'] >= from_dt].copy()
    h1.reset_index(drop=True, inplace=True)
    m5.reset_index(drop=True, inplace=True)
    return m5, h1

def pre_compute(sig_df, h1):
    t0 = time.time()
    N = len(sig_df)
    print(f'Pre-computing {N} bars...')
    h1_tns = h1['time'].values.astype(np.int64)
    sig_tns = sig_df['time'].values.astype(np.int64)
    h1_idx = np.clip(np.searchsorted(h1_tns, sig_tns, side='right') - 2, 0, len(h1) - 1)
    h1_high_arr = h1['high'].values[h1_idx]
    h1_low_arr = h1['low'].values[h1_idx]

    be = BiasEngine()
    biases = [None] * N
    last_bh = -1
    for idx in range(N):
        ts = sig_df['time'].iloc[idx]
        bh = ts.hour
        prev = biases[idx - 1] if idx > 0 else None
        if bh != last_bh or prev is None:
            last_bh = bh
            hs = h1[h1['time'] < ts.replace(minute=0, second=0, microsecond=0)].tail(96)
            if len(hs) >= 20: biases[idx] = be.update(hs)
            elif idx > 0: biases[idx] = prev
        else: biases[idx] = prev

    se = SignalEngine()
    sig_signal = [None] * N
    sig_score = np.zeros(N)
    brs = np.zeros(N)
    moms = np.zeros(N)

    for idx in range(10, N):
        if idx % 10000 == 0: print(f'  signal: {idx}/{N}')
        bsum = biases[idx]
        if bsum is None: continue
        if bsum.get('bias') not in ('BULLISH', 'BEARISH') or bsum.get('strength', 0) < cfg.BIAS_STRENGTH_MIN: continue
        row = sig_df.iloc[idx]
        win = sig_df.iloc[max(0, idx - 50):idx]
        sig = se.evaluate(win, bsum, row['close'], h1_high=float(h1_high_arr[idx]), h1_low=float(h1_low_arr[idx]))
        if sig:
            sig_signal[idx] = sig
            sig_score[idx] = sig['score']
            brs[idx] = _body_ratio(row)
            moms[idx] = _momentum_at(sig_df, idx)

    cnt = sum(1 for s in sig_signal if s is not None)
    print(f'  {cnt} signals [{time.time()-t0:.0f}s]')
    return {'sig_df': sig_df, 'biases': biases, 'sig_signal': sig_signal, 'sig_score': sig_score,
            'body_ratios': brs, 'momentums': moms, 'signal_engine': se}

def run_trades(data, conf):
    sig_df = data['sig_df']; biases = data['biases']; sigs = data['sig_signal']
    scores = data['sig_score']; brs = data['body_ratios']; moms = data['momentums']
    N = len(sig_df)
    br_min = conf.get('br'); cons = conf.get('cons', 1); mom_min = conf.get('mom')
    scaler = EquityScaler(); scaler.initialize(INITIAL_BALANCE)
    trades = []; cur = None; balance = INITIAL_BALANCE; cl = 0; dt = 0
    cur_day = None; peak = INITIAL_BALANCE; cd = None; epnl = 0.0; dpnl = 0.0

    for idx in range(N):
        ts = sig_df['time'].iloc[idx]
        row = sig_df.iloc[idx]; px = row['open']
        if cur_day != ts.date():
            cur_day = ts.date(); dt = 0; dpnl = 0.0; cl = 0
        bsum = biases[idx]
        if bsum is None: continue

        # Exit
        if cur is not None:
            xp = row['close']
            epnl = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], et=cur['entry_time'])
            ew = sig_df.iloc[max(0, idx - 20):idx]
            sx, sv, sr = data['signal_engine'].evaluate_exit(
                ew, cur['entry_price'], cur['direction'], cur.get('entry_score'),
                exit_threshold=cfg.EXIT_THRESHOLD_TIGHT, exit_mode=6, signal=cur.get('entry_signal'))
            if sx:
                p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], et=cur['entry_time'])
                balance += p; dpnl += p
                if balance > peak: peak = balance
                scaler.update_peak(balance)
                trades.append({'dir': cur['direction'], 'pnl': round(p, 2), 'lot': cur['lot'], 'n': cur['num_trades'], 'r': sr})
                cl = 0 if p > 0 else cl + 1
                cd = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + cl))
                cur = None; continue
            if epnl <= -cfg.MAX_EVENT_LOSS_USD:
                p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], et=cur['entry_time'])
                balance += p; dpnl += p
                if balance > peak: peak = balance
                trades.append({'dir': cur['direction'], 'pnl': round(p, 2), 'lot': cur['lot'], 'n': cur['num_trades'], 'r': 'el'})
                cl += 1; cd = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + cl))
                cur = None; continue

        if cd is not None and ts < cd: continue
        cd = None

        # Entry
        if cur is None:
            if not _session_allowed(ts, cfg.ALLOWED_SESSIONS): continue
            if cl >= cfg.MAX_CONSECUTIVE_LOSSES or balance < cfg.MIN_BALANCE: continue
            if dpnl <= -cfg.MAX_DAILY_LOSS_USD: continue
            if dt >= cfg.MAX_TRADES_PER_SESSION: continue
            if bsum.get('bias') not in ('BULLISH', 'BEARISH'): continue
            if bsum.get('strength', 0) < cfg.BIAS_STRENGTH_MIN: continue

            sig = sigs[idx]; sc = scores[idx]
            if sig is None or sc < cfg.SIGNAL_ENTRY_THRESHOLD: continue
            atr_th = sig.get('atr_entry_threshold')
            min_et = max(atr_th if atr_th else 0, cfg.SIGNAL_ENTRY_THRESHOLD)
            if sc < min_et: continue

            # Confirmation gate
            if br_min is not None and brs[idx] < br_min: continue
            if cons > 1:
                ok = True
                for off in range(1, cons):
                    if idx - off < 0 or sigs[idx - off] is None or scores[idx - off] < cfg.SIGNAL_ENTRY_THRESHOLD:
                        ok = False; break
                if not ok: continue
            if mom_min is not None and moms[idx] < mom_min: continue

            lot = min(scaler.get_lot(balance) * cfg.LOT_MULTIPLIER, cfg.MAX_LOT)
            lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP
            lot = max(cfg.MIN_LOT, lot)
            num_tr = min(cfg.MAX_TRADES_PER_EVENT, scaler.get_trades_per_event(balance, sc))
            margin_per = (row['close'] * CONTRACT_SIZE) / LEVERAGE
            max_by_margin = (balance * MARGIN_MAX_PCT) / margin_per
            if max_by_margin < cfg.MIN_LOT: continue
            lot = min(lot, max_by_margin)
            lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP
            lot = max(cfg.MIN_LOT, lot)
            num_tr = max(1, min(num_tr, int(max_by_margin / max(lot, 1e-9))))
            total_lot = lot * num_tr
            if total_lot <= 0: continue
            cur = {'entry_time': ts, 'entry_price': px, 'direction': sig['direction'],
                   'lot': lot, 'num_trades': num_tr, 'entry_score': float(sc), 'entry_signal': sig}
            epnl = 0.0; dt += 1

    if cur is not None:
        xp = float(sig_df['close'].iloc[-1])
        p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], et=cur['entry_time'])
        balance += p
        trades.append({'dir': cur['direction'], 'pnl': round(p, 2), 'lot': cur['lot'], 'n': cur['num_trades'], 'r': 'end'})

    if not trades: return None
    arr = np.array([t['pnl'] for t in trades])
    wins = arr[arr > 0]; losses = arr[arr < 0]
    wr = len(wins) / len(arr) * 100 if len(arr) > 0 else 0
    gp = wins.sum() if len(wins) > 0 else 0
    gl = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)
    cum = np.cumsum(arr)
    peak_cum = np.maximum.accumulate(cum)
    dd = (peak_cum - cum).max()
    return { 'trades': len(arr), 'net_pnl': round(arr.sum(), 2), 'win_rate': round(wr, 1),
             'profit_factor': round(pf, 2), 'max_dd': round(dd, 2), 'final_bal': round(balance, 2) }

def main():
    dt_from = datetime(2024, 3, 1)
    dt_to = datetime(2024, 3, 31, 23, 59)
    print(f'Sweep: {dt_from.date()} to {dt_to.date()}')
    data = load_data(dt_from, dt_to)
    if data is None: print('No data'); return
    m5, h1 = data
    comp = pre_compute(m5, h1)

    results = []
    t0 = time.time()
    for name, conf in CONF_MODES.items():
        r = run_trades(comp, conf)
        if r is None: r = {'trades': 0, 'net_pnl': 0, 'win_rate': 0, 'profit_factor': 0, 'max_dd': 0, 'final_bal': INITIAL_BALANCE}
        results.append({'name': name, **conf, **r})

    results.sort(key=lambda x: x['net_pnl'], reverse=True)
    print(f'{"="*100}')
    print(f'  SWEEP RESULTS (1 month)')
    print(f'{"="*100}')
    hdr = f'  {"Mode":<18} {"Trades":>6} {"Net":>8} {"WR%":>6} {"PF":>6} {"DD":>6} {"Bal":>8}  {"Config"}'
    print(hdr)
    print(f'  {"-"*92}')
    for rank, r in enumerate(results, 1):
        cfg_str = f'br={r["br"] or "-"} cons={r["cons"]} mom={r["mom"] or "-"}'
        print(f'  {r["name"]:<18} {r["trades"]:>6}  {r["net_pnl"]:>+8.2f}  {r["win_rate"]:>5.1f}%  {r["profit_factor"]:>5.2f}  {r["max_dd"]:>8.2f}  {r["final_bal"]:>8.2f}  {cfg_str}')

    # Validate top 3 on full 2025
    print(f'{"="*100}')
    print(f'  VALIDATE TOP 3 ON FULL 2025')
    print(f'{"="*100}')
    top3 = results[:3]
    baseline_r = [r for r in results if r['name'] == 'baseline']
    if baseline_r: top3.append(baseline_r[0])
    print('Loading full 2025 data...')
    vdata = load_data(datetime(2025, 1, 1), datetime(2025, 12, 31, 23, 59))
    if vdata:
        vm5, vh1 = vdata
        vcomp = pre_compute(vm5, vh1)
        for r in top3:
            conf = {'br': r['br'], 'cons': r['cons'], 'mom': r['mom']}
            vr = run_trades(vcomp, conf)
            if vr is None: vr = {'trades': 0, 'net_pnl': 0, 'win_rate': 0, 'profit_factor': 0, 'max_dd': 0, 'final_bal': INITIAL_BALANCE}
            print(f'  {r["name"]:<18} 1mo: PnL={r["net_pnl"]:>+8.2f} WR={r["win_rate"]:>5.1f}% {r["trades"]:>4}t PF={r["profit_factor"]:.2f} DD={r["max_dd"]:>7.2f} Bal={r["final_bal"]:>8.2f}')
            print(f'  {"":<18} yr:  PnL={vr["net_pnl"]:>+8.2f} WR={vr["win_rate"]:>5.1f}% {vr["trades"]:>4}t PF={vr["profit_factor"]:.2f} DD={vr["max_dd"]:>7.2f} Bal={vr["final_bal"]:>8.2f}')

    # Save
    with open('sweep_confirmation_results.json', 'w') as f:
        clean = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in results]
        json.dump(clean, f, indent=2)
    print(f'Saved sweep_confirmation_results.json [{time.time()-t0:.0f}s total]')

if __name__ == '__main__':
    main()
