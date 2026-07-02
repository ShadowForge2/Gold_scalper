import sys, os, time; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.dukascopy_client import DukascopyClient
from app.signal_engine import SignalEngine
from app.bias_engine import BiasEngine
from app.risk_manager import EquityScaler
from app.meta_strategy import MetaStrategy
import config as cfg

INITIAL_BALANCE = 20.0; CONTRACT_SIZE = 100; LEVERAGE = 200; MARGIN_MAX_PCT = 1.0
SPREAD_PER_LOT_USD = 25.0; SLIPPAGE_PER_LOT_USD = 3.0; SPREAD_ROUND_TRIP = 2

def _pnl(ep, xp, d, lot, n, et=None):
    delta = xp - ep if d == 'BUY' else ep - xp
    gross = delta * CONTRACT_SIZE * lot * n
    sm = (0.9 if et and 13 <= et.hour < 17 else 1.0 if et and 8 <= et.hour < 13 else 1.6 if et and 0 <= et.hour < 8 else 1.3)
    sc = SPREAD_PER_LOT_USD * lot * n * SPREAD_ROUND_TRIP * sm
    slip = SLIPPAGE_PER_LOT_USD * lot * n
    return gross - sc - slip

def check_conf(br, atr, br_win, atr_win):
    if len(br_win) < 10 or len(atr_win) < 50: return True
    lo, hi = np.percentile(atr_win, [25, 75])
    if atr <= lo: return br >= np.percentile(br_win, 60)
    if lo < atr < hi: return br >= np.percentile(br_win, 40)
    return True

def run_year(label, m5, h1, trade_start):
    t0 = time.time()
    N = len(m5)
    trade_offset = 0
    if trade_start:
        trade_times = m5['time'].values
        trade_offset = np.searchsorted(trade_times, np.datetime64(trade_start), side='right')
    h1_tns = h1['time'].values.astype(np.int64); m5_tns = m5['time'].values.astype(np.int64)
    h1_idx = np.clip(np.searchsorted(h1_tns, m5_tns, side='right') - 2, 0, len(h1) - 1)
    h1h = h1['high'].values[h1_idx]; h1l = h1['low'].values[h1_idx]; h1r = np.maximum(h1h - h1l, 1e-10)
    mc = m5['close'].values; mh = m5['high'].values; ml = m5['low'].values; mo = m5['open'].values
    bs = np.maximum(mh - h1h, h1l - ml) / h1r
    br = np.abs(mc - mo) / np.maximum(mh - ml, 1e-10)
    tr = np.maximum(mh[1:] - ml[1:], np.maximum(np.abs(mh[1:] - mc[:-1]), np.abs(ml[1:] - mc[:-1])))
    atrs = np.zeros(N)
    for i in range(14, N): atrs[i] = np.mean(tr[i-14:i])
    be = BiasEngine(); bias = [None] * N; cache = {}; last_bh = -1
    for idx in range(N):
        ts = m5['time'].iloc[idx]; bh = ts.hour
        if bh == last_bh and bias[idx - 1] is not None: bias[idx] = bias[idx - 1]; continue
        last_bh = bh; key = ts.replace(minute=0, second=0, microsecond=0)
        if key in cache: bias[idx] = cache[key]; continue
        hs = h1[h1['time'] < key].tail(96)
        if len(hs) >= 20: bias[idx] = be.update(hs); cache[key] = bias[idx]
        elif idx > 0: bias[idx] = bias[idx - 1]
    se = SignalEngine(); sigs = [None] * N
    for idx in range(10, N):
        b = bias[idx]
        if b is None or b.get('bias') not in ('BULLISH', 'BEARISH') or b.get('strength', 0) < cfg.BIAS_STRENGTH_MIN: continue
        if bs[idx] < cfg.SIGNAL_ENTRY_THRESHOLD: continue
        win = m5.iloc[max(0, idx - 50):idx]
        s = se.evaluate(win, b, float(mc[idx]), h1_high=float(h1h[idx]), h1_low=float(h1l[idx]))
        if s: sigs[idx] = s
    sig_count = sum(1 for s in sigs if s is not None)
    print(f"{label}: sigs={sig_count} [{time.time()-t0:.0f}s]", flush=True)

    meta = MetaStrategy(); sc = EquityScaler(); sc.initialize(INITIAL_BALANCE)
    bal = INITIAL_BALANCE; trds = []; cur = None; cl = 0; dt = 0; cd = None
    cur_day = None; dpnl = 0.0; br_win = []; atr_win = []
    dbg = {'sig':0,'thr':0,'conf':0,'bal':0,'dpnl':0,'enter':0}
    print(f"  offset={trade_offset} N={N} bars={N-trade_offset}", flush=True)
    for idx in range(trade_offset, N):
        ts = m5['time'].iloc[idx]; row = m5.iloc[idx]; px = float(row['open'])
        if cur_day != ts.date(): cur_day = ts.date(); dt = 0; dpnl = 0.0; cl = 0
        brv = float(br[idx]); atv = float(atrs[idx]); br_win.append(brv); atr_win.append(atv)
        if len(br_win) > 100: br_win.pop(0)
        if len(atr_win) > 100: atr_win.pop(0)
        b = bias[idx]
        if b is None: continue
        if cur is not None:
            xp = float(row['close'])
            ew = m5.iloc[max(0, idx - 20):idx]
            sx, sv, sr = se.evaluate_exit(ew, cur['entry_price'], cur['direction'], cur.get('entry_score'),
                                           exit_threshold=cfg.EXIT_THRESHOLD_TIGHT, exit_mode=6, signal=cur.get('entry_signal'))
            if sx:
                p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], cur['entry_time'])
                bal += p; dpnl += p; sc.update_peak(bal); trds.append(round(p, 2))
                cl = 0 if p > 0 else cl + 1
                cd = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + cl))
                meta.record_trade(p, b.get('strength', 0)); meta.update(bal, {'direction': b.get('bias'), 'strength': b.get('strength')})
                cur = None; continue
            if _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades']) <= -cfg.MAX_EVENT_LOSS_USD:
                p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], cur['entry_time'])
                bal += p; dpnl += p; trds.append(round(p, 2)); cl += 1
                cd = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + cl))
                meta.record_trade(p, b.get('strength', 0)); meta.update(bal, {'direction': b.get('bias'), 'strength': b.get('strength')})
                cur = None; continue
        if cd and ts < cd: continue
        cd = None
        if cur is None:
            if cl >= cfg.MAX_CONSECUTIVE_LOSSES or bal < cfg.MIN_BALANCE: continue
            if dpnl <= -cfg.MAX_DAILY_LOSS_USD: continue
            sig = sigs[idx]; dbg['sig'] += 1
            if sig is None: continue
            scv = bs[idx]
            if scv < meta.current_threshold: dbg['thr'] += 1; continue
            if not check_conf(brv, atv, br_win, atr_win): dbg['conf'] += 1; continue
            dbg['enter'] += 1
            lot_mult = meta.current_lot_mult
            if cfg.AGGRESSIVE_SIZING_ENABLED:
                if scv >= cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD: lot_mult *= cfg.AGGRESSIVE_VERY_STRONG_LOT_MULT
                elif scv >= cfg.AGGRESSIVE_STRONG_THRESHOLD: lot_mult *= cfg.AGGRESSIVE_STRONG_LOT_MULT
            lot = min(sc.get_lot(bal) * lot_mult, cfg.MAX_LOT)
            lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP; lot = max(cfg.MIN_LOT, lot)
            num_tr = meta.current_trades_per_event
            margin_per = (float(row['close']) * CONTRACT_SIZE) / LEVERAGE
            max_by_margin = (bal * MARGIN_MAX_PCT) / margin_per
            if max_by_margin < cfg.MIN_LOT: continue
            lot = min(lot, max_by_margin); lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP; lot = max(cfg.MIN_LOT, lot)
            num_tr = max(1, min(num_tr, int(max_by_margin / max(lot, 1e-9))))
            if lot * num_tr <= 0: continue
            cur = {'entry_time': ts, 'entry_price': px, 'direction': sig['direction'],
                   'lot': lot, 'num_trades': num_tr, 'entry_score': float(scv), 'entry_signal': sig}
            dt += 1
    if cur:
        p = _pnl(cur['entry_price'], float(m5['close'].iloc[-1]), cur['direction'], cur['lot'], cur['num_trades'], cur['entry_time'])
        bal += p; trds.append(round(p, 2))
        meta.record_trade(p, 0)
    if not trds: return None
    pnls = np.array(trds); wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    wr = len(wins) / len(pnls) * 100; gp = wins.sum() if len(wins) > 0 else 0; gl = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)
    cum = np.cumsum(pnls); dd = (np.maximum.accumulate(cum) - cum).max()
    print(f"  dbg: sigs_reached={dbg['sig']} thr_blocked={dbg['thr']} conf_blocked={dbg['conf']} entered={dbg['enter']}", flush=True)
    t = time.time() - t0
    return {'label': label, 'trades': len(pnls), 'pnl': round(pnls.sum(), 2), 'wr': round(wr, 1),
            'pf': round(pf, 2), 'dd': round(dd, 2), 'bal': round(bal, 2), 'time': round(t)}

client = DukascopyClient(); t_all = time.time()
years = [2023, 2024, 2025]; all_results = []

for yr in years:
    print(f"\n=== {yr} ===")
    m1 = client.download_range(yr - 1, yr)
    from_dt = datetime(yr, 1, 1); to_dt = datetime(yr, 12, 31, 23, 59)
    warmup = from_dt - timedelta(days=14)
    m1 = m1[(m1['time'] >= warmup) & (m1['time'] <= to_dt)]
    h1 = client.resample_to(m1, 16385); m5 = client.resample_to(m1, 5)
    print(f"M5: {len(m5)} H1: {len(h1)}  range: {m5['time'].min()} to {m5['time'].max()}", flush=True)
    r = run_year(str(yr), m5, h1, from_dt)
    if r: all_results.append(r)

print(f"\n{'='*70}")
print(f"  FULL STRATEGY CONFIRMATION — Meta+vreg+Aggressive")
print(f"{'='*70}")
print(f"  {'Year':<8} {'Trades':>6} {'PnL':>12} {'WR%':>5} {'PF':>6} {'Max DD':>10} {'Final':>10} {'Time':>6}")
print(f"  {'-'*69}")
for r in all_results:
    print(f"  {r['label']:<8} {r['trades']:>6} ${r['pnl']:>+9.2f} {r['wr']:>5.1f} {r['pf']:>6.2f} ${r['dd']:>8.2f} ${r['bal']:>8.2f} {r['time']:>5}s")
if all_results:
    tot = {'trades': sum(r['trades'] for r in all_results), 'pnl': sum(r['pnl'] for r in all_results),
           'dd': max(r['dd'] for r in all_results), 'time': sum(r['time'] for r in all_results)}
    pnls_all = np.concatenate([np.full(r['trades'], r['pnl']/r['trades']) for r in all_results])  # rough
    print(f"  {'-'*69}")
    print(f"  {'TOTAL':<8} {tot['trades']:>6} ${tot['pnl']:>+9.2f} {'':>12} {tot['dd']:>10.2f} {'':>10} {tot['time']:>5}s")
print(f"{'='*70}")
print(f"  Total time: {time.time()-t_all:.0f}s")
