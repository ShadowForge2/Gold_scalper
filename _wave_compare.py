"""Compare old vs new wave-scalper behavior on real data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _train_candle_h1 import load_m1_data, resample_h1
from app.candle_engine import compute_atr


def run_candle_wave_OLD(m1, o, atr, entry_r, cut_r, profit_r, cost_r,
                        jump_break_r, jump_body_r, trail_r, reversal_r,
                        rider_enabled=True):
    """OLD version: exit checks run on the same bar that triggered entry."""
    trades = []
    base = o
    flat = True
    pos = 0
    entry = 0.0
    peak = 0.0
    rider = False
    for bo, bh, bl, bc in m1:
        if flat and not rider:
            if bh >= base + entry_r * atr:
                entry = base + entry_r * atr
                pos = 1
                peak = entry
                flat = False
            elif bl <= base - entry_r * atr:
                entry = base - entry_r * atr
                pos = -1
                peak = entry
                flat = False
            elif rider_enabled:
                if (bh - o) >= jump_break_r * atr and (bh - bl) > 0:
                    body = bh - o
                    if body / (bh - bl) >= jump_body_r:
                        entry = o
                        pos = 1
                        peak = o
                        flat = False
                        rider = True
                elif (o - bl) >= jump_break_r * atr and (bh - bl) > 0:
                    body = o - bl
                    if body / (bh - bl) >= jump_body_r:
                        entry = o
                        pos = -1
                        peak = o
                        flat = False
                        rider = True
        if pos == 1:
            if bh > peak:
                peak = bh
            stop = entry - cut_r * atr
            lock = peak - profit_r * atr
            if bl <= stop:
                trades.append((entry - stop) / atr - cost_r)
                base = stop
                flat = True
                pos = 0
            elif bl <= lock:
                trades.append((lock - entry) / atr - cost_r)
                base = lock
                flat = True
                pos = 0
            elif rider:
                ts = peak - trail_r * atr
                if bl <= ts:
                    trades.append((ts - entry) / atr - cost_r)
                    flat = True
                    pos = 0
                    rider = False
                    base = ts
        elif pos == -1:
            if bl < peak:
                peak = bl
            stop = entry + cut_r * atr
            lock = peak + profit_r * atr
            if bh >= stop:
                trades.append((entry - stop) / atr - cost_r)
                base = stop
                flat = True
                pos = 0
            elif bh >= lock:
                trades.append((entry - lock) / atr - cost_r)
                base = lock
                flat = True
                pos = 0
            elif rider:
                ts = peak + trail_r * atr
                if bh >= ts:
                    trades.append((entry - ts) / atr - cost_r)
                    flat = True
                    pos = 0
                    rider = False
                    base = ts
    if pos != 0:
        r = (bc - entry) * pos / atr - cost_r
        trades.append(r)
    return trades


from _sweep_candle_wave import run_candle_wave

m1 = load_m1_data('XAUUSD', start_year=2023, end_year=2024)
h1 = resample_h1(m1, 60)
atr_all = compute_atr(h1, 14)
mi = m1.set_index('time') if 'time' in m1.columns else m1
mi = mi[~mi.index.duplicated(keep='first')].sort_index()
bucket = mi.index.floor('1h')
mask = (bucket >= h1.index[100]) & (bucket <= h1.index[-1])
m1t = mi[mask]
bt = bucket[mask]
o_arr = m1t['open'].values
h_arr = m1t['high'].values
l_arr = m1t['low'].values
c_arr = m1t['close'].values
grp = pd.Series(np.arange(len(bt)), index=m1t.index).groupby(
    pd.Series(bt.values, index=m1t.index), sort=True)
candle_m1 = {}
for ts, idx_ in grp.indices.items():
    idx_ = np.asarray(idx_)
    candle_m1[ts] = np.column_stack([o_arr[idx_], h_arr[idx_], l_arr[idx_], c_arr[idx_]])

old_r, new_r = [], []
for k in range(100, len(h1) - 1):
    ts = h1.index[k]
    sub = candle_m1.get(ts)
    if sub is None or len(sub) < 2:
        continue
    o = float(h1['open'].iloc[k])
    atr = float(atr_all.iloc[k - 1]) if atr_all.iloc[k - 1] > 0 else 1e-9
    old_r += run_candle_wave_OLD(sub, o, atr, 0.20, 0.03, 0.05, 0.05, 1.5, 0.7, 0.5, 0.5)
    new_r += run_candle_wave(sub, o, atr, 0.20, 0.03, 0.05, 0.05, 1.5, 0.7, 0.5, 0.5)

old_r = np.asarray(old_r); new_r = np.asarray(new_r)
print(f'OLD: n={len(old_r)} net={old_r.sum():.2f} PF={old_r[old_r>0].sum()/max(-old_r[old_r<0].sum(),1e-9):.2f}')
print(f'NEW: n={len(new_r)} net={new_r.sum():.2f} PF={new_r[new_r>0].sum()/max(-new_r[new_r<0].sum(),1e-9):.2f}')
