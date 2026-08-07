"""Full-model-gated comparison: CURRENT wave engine vs STRICT (no same-bar peak).
Runs the exact sweep pipeline (train model, chop gate, jump flags) for XAUUSD
and reports PF/net/trades for both engine variants.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import xgboost as xgb
from _train_candle_h1 import load_m1_data, resample_h1
from app.candle_engine import (FEATURE_COLS, compute_features, compute_atr, generate_labels)
from _sweep_candle_wave import train_model, compute_jump_flags, run_candle_wave

ATR_PERIOD = 14


def run_candle_wave_strict(m1, o, atr, entry_r, cut_r, profit_r, cost_r,
                           jump_break_r, jump_body_r, trail_r, reversal_r,
                           rider_enabled=True):
    """STRICT: peak is updated only AFTER the exit check on each bar, so a bar
    can never set its own high as the sell level (no buy-low/sell-high of the
    same bar)."""
    trades = []
    base = o
    flat = True
    pos = 0
    entry = 0.0
    peak = 0.0
    rider = False
    jump_dir = 0
    just_entered = False

    for bo, bh, bl, bc in m1:
        if flat and not rider:
            if bh >= base + entry_r * atr:
                entry = base + entry_r * atr
                pos = 1; peak = entry; flat = False; just_entered = True
            elif bl <= base - entry_r * atr:
                entry = base - entry_r * atr
                pos = -1; peak = entry; flat = False; just_entered = True
            elif rider_enabled:
                if (bh - o) >= jump_break_r * atr and (bh - bl) > 0:
                    body = bh - o
                    if body / (bh - bl) >= jump_body_r:
                        entry = o; pos = 1; peak = o; flat = False
                        rider = True; jump_dir = 1; just_entered = True
                elif (o - bl) >= jump_break_r * atr and (bh - bl) > 0:
                    body = o - bl
                    if body / (bh - bl) >= jump_body_r:
                        entry = o; pos = -1; peak = o; flat = False
                        rider = True; jump_dir = -1; just_entered = True
            if just_entered:
                continue
        elif pos == 1:
            stop = entry - cut_r * atr
            lock = peak - profit_r * atr
            if bl <= stop:
                trades.append((stop - entry) / atr - cost_r)
                base = stop; flat = True; pos = 0
            elif bl <= lock:
                trades.append((lock - entry) / atr - cost_r)
                base = lock; flat = True; pos = 0
            elif rider:
                ts = peak - trail_r * atr
                if bl <= ts:
                    trades.append((ts - entry) / atr - cost_r)
                    flat = True; pos = 0; rider = False; base = ts
            if bh > peak:
                peak = bh
        elif pos == -1:
            stop = entry + cut_r * atr
            lock = peak + profit_r * atr
            if bh >= stop:
                trades.append((entry - stop) / atr - cost_r)
                base = stop; flat = True; pos = 0
            elif bh >= lock:
                trades.append((entry - lock) / atr - cost_r)
                base = lock; flat = True; pos = 0
            elif rider:
                ts = peak + trail_r * atr
                if bh >= ts:
                    trades.append((entry - ts) / atr - cost_r)
                    flat = True; pos = 0; rider = False; base = ts
            if bl < peak:
                peak = bl
    if pos != 0:
        r = (bc - entry) * pos / atr - cost_r
        trades.append(r)
    return trades


symbol = "XAUUSD"
train_start, train_end, test_start, test_end = 2018, 2022, 2023, 2025
m1 = load_m1_data(symbol, start_year=train_start, end_year=test_end)
h1_all = resample_h1(m1, 60)
feats_all = compute_features(h1_all)
atr_all = compute_atr(h1_all, ATR_PERIOD)
tr_mask = (h1_all.index.year >= train_start) & (h1_all.index.year <= train_end)
te_mask = (h1_all.index.year >= test_start) & (h1_all.index.year <= test_end)
tr_idx = np.where(tr_mask)[0]
te_idx = np.where(te_mask)[0]
trade_params = dict(
    sl_r=1.0, reversal_r=0.5, trail_r=0.5, max_hold=24, cost_r=0.05,
    entry_min_r=0.90, edge_margin=1.75,
)
X = feats_all[FEATURE_COLS].fillna(0.0).values
Yl = generate_labels(feats_all, atr_all, **trade_params)["entry_label"].values
rng = np.random.default_rng(42)

def balanced(idx, per_class):
    parts = []
    for cls in (0, 1, 2):
        sel = idx[Yl[idx] == cls]
        if len(sel) > per_class:
            sel = rng.choice(sel, size=per_class, replace=False)
        parts.append(sel)
    out = np.concatenate(parts); out.sort()
    return out

per_class = 15000
tr_sel = balanced(tr_idx, per_class)
va_sel = tr_idx[int(0.85 * len(tr_idx)):]
model = train_model(X[tr_sel], Yl[tr_sel], X[va_sel], Yl[va_sel])

h1_test = feats_all.iloc[te_idx]
atr_test = atr_all.iloc[te_idx]
X_test = h1_test[FEATURE_COLS].fillna(0.0).values
probs = model.predict(xgb.DMatrix(X_test))
jf = compute_jump_flags(h1_test, atr_test, 1.5, 0.70)

# bucket M1
mi = m1.set_index("time") if "time" in m1.columns else m1
mi = mi[~mi.index.duplicated(keep="first")].sort_index()
bucket = mi.index.floor("1h")
mask = (bucket >= h1_test.index[0]) & (bucket <= h1_test.index[-1])
m1t = mi[mask]; bt = bucket[mask]
o_arr = m1t["open"].values; h_arr = m1t["high"].values
l_arr = m1t["low"].values; c_arr = m1t["close"].values
grp = pd.Series(np.arange(len(bt)), index=m1t.index).groupby(
    pd.Series(bt.values, index=m1t.index), sort=True)
candle_m1 = {}
for ts, idx_ in grp.indices.items():
    idx_ = np.asarray(idx_)
    candle_m1[ts] = np.column_stack([o_arr[idx_], h_arr[idx_], l_arr[idx_], c_arr[idx_]])

for combo in [(0.20, 0.03, 0.05), (0.50, 0.03, 0.01), (0.30, 0.03, 0.05), (0.10, 0.03, 0.01)]:
    entry_r, cut_r, profit_r = combo
    out = {"cur": [], "strict": []}
    for k, ts in enumerate(h1_test.index):
        if k == 0:
            continue
        prev = k - 1
        pb, ps, pn = float(probs[prev][0]), float(probs[prev][1]), float(probs[prev][2])
        if pn > max(pb, ps) and not jf[prev]:
            continue
        sub = candle_m1.get(ts)
        if sub is None or len(sub) < 2:
            continue
        o = float(h1_test["open"].iloc[k])
        atr = float(atr_test.iloc[prev]) if atr_test.iloc[prev] > 0 else 1e-9
        out["cur"] += run_candle_wave(sub, o, atr, entry_r, cut_r, profit_r, 0.05, 1.5, 0.70, 0.5, 0.5)
        out["strict"] += run_candle_wave_strict(sub, o, atr, entry_r, cut_r, profit_r, 0.05, 1.5, 0.70, 0.5, 0.5)
    for name, rs in out.items():
        rs = np.asarray(rs)
        wins = float(rs[rs > 0].sum()); losses = float((-rs[rs < 0]).sum())
        pf = wins / losses if losses > 0 else float("inf")
        eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
        dd = float((peak - eq).max())
        print(f"entry={entry_r} cut={cut_r} profit={profit_r} [{name:>6}] "
              f"n={len(rs):>7} WR={100*(rs>0).mean():.1f}% exp={rs.mean():+.3f} "
              f"PF={pf:.2f} net={rs.sum():+.1f} dd={dd:.1f}")
