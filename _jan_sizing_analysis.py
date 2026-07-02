"""
Jan 2025 — Detailed position sizing, trades per event, and profitability analysis.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features
from collections import defaultdict, Counter

CS = 100; LEV = 200
TRADE_MAX_BARS = cfg.PEAK_HARVEST_MAX_HOLD_BARS

pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
client = DukascopyClient()
print("Downloading 2025 data...", flush=True)
m1 = client.download_year(2025)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
m5 = client.resample_to(m1, 5).set_index("time")
h1 = client.resample_to(m1, 16385).set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]
print(f"  M5: {len(m5)}, H1: {len(h1)}", flush=True)

print("Computing features...", flush=True)
ft = compute_features(m5, h1)
model = pred.model
X = ft[ft.columns.intersection(pred._feature_cols)]
for c in pred._feature_cols:
    if c not in X.columns: X[c] = 0.0
X = X[pred._feature_cols]
X_aligned = X.reindex(m5.index, method="ffill")
ml_mask = ~X_aligned.isna().any(axis=1)
pb_u = np.full(len(m5), np.nan)
pb_d = np.full(len(m5), np.nan)
if ml_mask.any():
    valid_idx = np.where(ml_mask)[0]
    probs = model.predict_proba(X_aligned[ml_mask].values)
    pb_u[valid_idx] = np.array([p[1] for p in probs])
    pb_d[valid_idx] = np.array([p[0] for p in probs])

print("Computing ATR/bias...", flush=True)
tr = np.maximum(m5["high"].values - m5["low"].values,
    np.maximum(np.abs(m5["high"].values - np.roll(m5["close"].values, 1)),
               np.abs(m5["low"].values - np.roll(m5["close"].values, 1))))
tr[0] = m5["high"].values[0] - m5["low"].values[0]
m5_atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
m5_atr = np.where(np.isnan(m5_atr), 0.0, m5_atr)

h1_c = h1["close"].values
fast = pd.Series(h1_c).ewm(span=20, adjust=False).mean().values
slow = pd.Series(h1_c).ewm(span=50, adjust=False).mean().values
h1_bias = np.where(fast > slow, 1, -1)
h1_idx_map = h1.index.get_indexer(m5.index, method="ffill")
h1_idx_map = np.clip(h1_idx_map, 0, len(h1_bias) - 1)
m5_bias = h1_bias[h1_idx_map]
m5_bias_arr = np.full(len(m5_bias), None, dtype=object)
m5_bias_arr[m5_bias == 1] = "BUY"
m5_bias_arr[m5_bias == -1] = "SELL"

h1_h = h1["high"].values; h1_l = h1["low"].values
m5_h1h = np.where(h1_idx_map > 0, h1_h[h1_idx_map - 1], h1_h[h1_idx_map])
m5_h1l = np.where(h1_idx_map > 0, h1_l[h1_idx_map - 1], h1_l[h1_idx_map])

m5_close = m5["close"].values; m5_open = m5["open"].values
m5_high = m5["high"].values; m5_low = m5["low"].values

jan_start = pd.Timestamp("2025-01-01")
jan_end = pd.Timestamp("2025-02-01")
jan_indices = [i for i in range(len(m5)) if jan_start <= m5.index[i] < jan_end]
print(f"January bars: {len(jan_indices)}", flush=True)

bal = 20.0
override_count = 0
override_day = None

trades = []

for idx in jan_indices:
    i = idx
    expected = m5_bias_arr[i]
    if expected is None: continue
    p = float(m5_close[i])
    h1h = float(m5_h1h[i]); h1l = float(m5_h1l[i])
    if h1h <= h1l: continue
    if expected == "BUY" and p <= h1h: continue
    if expected == "SELL" and p >= h1l: continue
    range_sz = h1h - h1l
    breakout_dist = p - h1h if expected == "BUY" else h1l - p
    score = min(breakout_dist / range_sz, 1.0) if range_sz > 0 else 0.0
    if score < cfg.MIN_BREAKOUT_SCORE: continue

    ts_i = m5.index[i]
    day_key = ts_i.date()
    if override_day != day_key:
        override_count = 0
        override_day = day_key

    entry_dir = expected
    was_overridden = False
    if not np.isnan(pb_d[i]):
        if expected == "BUY" and pb_d[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_d[i] > pb_u[i] and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
            entry_dir = "SELL"; was_overridden = True
        elif expected == "SELL" and pb_u[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_u[i] > pb_d[i] and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
            entry_dir = "BUY"; was_overridden = True
        if was_overridden: override_count += 1

    if not was_overridden:
        if not np.isnan(pb_u[i]) and not np.isnan(pb_d[i]):
            if expected == "BUY" and pb_u[i] < cfg.ML_CONFIDENCE_THRESHOLD: continue
            if expected == "SELL" and pb_d[i] < cfg.ML_CONFIDENCE_THRESHOLD: continue

    n = max(0.01, bal * LEV / p / CS)
    n = min(n, 1.0)
    mg = p * CS / LEV
    mx = bal / mg
    if mx < cfg.MIN_LOT: continue
    n = min(n, mx)

    ep = p
    peak_profit = 0.0

    for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, len(m5) - 1)):
        fp = float(m5_close[j]); fh = float(m5_high[j]); fl = float(m5_low[j])
        diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
        prof_val = diff * CS * n
        if entry_dir == "BUY": peak_profit = max(peak_profit, float(fh - ep))
        else: peak_profit = max(peak_profit, float(ep - fl))
        atr_j = float(m5_atr[j])

        ml_hold = False
        if not np.isnan(pb_u[j]):
            if entry_dir == "BUY" and pb_u[j] >= cfg.ML_HOLD_CONFIDENCE: ml_hold = True
            elif entry_dir == "SELL" and pb_d[j] >= cfg.ML_HOLD_CONFIDENCE: ml_hold = True

        exit_now = False; exit_reason = None
        if peak_profit > 0:
            trail_trigger = atr_j * cfg.PEAK_HARVEST_TRAIL_TRIGGER
            if peak_profit >= trail_trigger:
                pullback = peak_profit - max(0, diff)
                if pullback / peak_profit > cfg.PEAK_HARVEST_TRAIL_RETRACE:
                    exit_now = True; exit_reason = "trail_stop"

        if not exit_now and not ml_hold and (j - i) >= 4:
            streak = 0
            lookback = min(cfg.DIRECTION_LOSS_LOOKBACK, j - i)
            for k in range(lookback):
                idx_k = j - k
                if entry_dir == "BUY":
                    if m5_close[idx_k] < m5_open[idx_k]: streak += 1
                    else: break
                else:
                    if m5_close[idx_k] > m5_open[idx_k]: streak += 1
                    else: break
            if streak >= cfg.DIRECTION_LOSS_STREAK:
                exit_now = True; exit_reason = "direction_loss"

        if not exit_now and not ml_hold and (j - i) >= cfg.PEAK_HARVEST_MIN_BARS_EXIT:
            start = max(0, j - 19)
            c_win = m5_close[start:j + 1]; o_win = m5_open[start:j + 1]
            if len(c_win) >= 5:
                recent = c_win[-3:]; older = c_win[-6:-3] if len(c_win) >= 6 else c_win[:3]
                recent_chg = abs(recent[-1] - recent[0])
                older_chg = abs(older[-1] - older[0]) if len(older) >= 2 else recent_chg
                avg_body = float(np.mean(np.abs(c_win[-5:] - o_win[-5:])))
                if avg_body > 0:
                    window_atr = float(np.mean(m5_atr[max(0, j - 13):j + 1]))
                    ref = max(window_atr, fp * 0.0001)
                    raw = (recent_chg / (older_chg + 1e-10)) * (avg_body / (ref + 1e-10))
                    if (1.0 - min(abs(raw), 1.0)) > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
                        exit_now = True; exit_reason = "momentum_decay"

        if not exit_now and not ml_hold:
            if not np.isnan(pb_d[j]) and not np.isnan(pb_u[j]):
                if entry_dir == "BUY" and pb_d[j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                    exit_now = True; exit_reason = "ml_reversal"
                elif entry_dir == "SELL" and pb_u[j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                    exit_now = True; exit_reason = "ml_reversal"

        if not exit_now and (j - i) >= TRADE_MAX_BARS:
            exit_now = True; exit_reason = "max_hold"

        if exit_now:
            bal += prof_val
            trades.append((n, entry_dir, score, was_overridden, round(prof_val, 2), exit_reason, day_key, i, ts_i))
            break

print(f"\nTrades recorded: {len(trades)}", flush=True)

# === ANALYSIS ===
pnls = [t[4] for t in trades]
lots = [t[0] for t in trades]

print("=" * 65)
print("  Jan 2025 — Position Sizing & Trade Detail")
print("=" * 65)
print(f"Total trades:         {len(trades)}")
print(f"Wins:                 {sum(1 for t in trades if t[4] > 0)} ({sum(1 for t in trades if t[4] > 0)/len(trades)*100:.1f}%)")
print(f"Losses:               {sum(1 for t in trades if t[4] <= 0)} ({sum(1 for t in trades if t[4] <= 0)/len(trades)*100:.1f}%)")
print(f"Net PnL:              ${sum(pnls):+.2f}")
print(f"Starting balance:     $20.00")
print(f"Ending balance:       ${bal:.2f}")
print()

print("--- Lot Size Distribution ---")
buckets = [(0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
for lo, hi in buckets:
    group = [t for t in trades if lo <= t[0] < hi]
    if group:
        w = sum(1 for t in group if t[4] > 0)
        avg_pnl = sum(t[4] for t in group) / len(group)
        print(f"  {lo:.3f}-{hi:<5} lot: {len(group):3d} trades, {w:3d}W ({w/len(group)*100:.0f}%), avg PnL ${avg_pnl:+.2f}")
print(f"  Avg lot:      {np.mean(lots):.4f}")
print(f"  Median lot:   {np.median(lots):.4f}")
print(f"  Min lot:      {min(lots):.4f}")
print(f"  Max lot:      {max(lots):.4f}")
print()

# Per-day stats
day_data = defaultdict(lambda: {"trades": [], "lots": [], "pnls": [], "overrides": 0})
for t in trades:
    d = t[6]
    day_data[d]["trades"].append(t)
    day_data[d]["lots"].append(t[0])
    day_data[d]["pnls"].append(t[4])
    if t[3]: day_data[d]["overrides"] += 1

print("--- Daily Summary ---")
print(f"  {'Date':12s} {'Trades':>6s} {'W/L':>8s} {'WR':>5s} {'AvgLot':>7s} {'AvgPnL':>8s} {'MaxLot':>7s} {'TotPnL':>8s} {'Ovr':>4s}")
print("  " + "-" * 70)
sorted_days = sorted(day_data.keys())
for d in sorted_days:
    data = day_data[d]
    lots_d = data["lots"]
    pnls_d = data["pnls"]
    w = sum(1 for p in pnls_d if p > 0)
    l = len(pnls_d) - w
    wr = f"{w/(w+l)*100:.0f}%" if (w+l) > 0 else "-"
    print(f"  {str(d):12s} {len(lots_d):6d} {f'{w}/{l}':>8s} {wr:>5s} {np.mean(lots_d):7.4f} {np.mean(pnls_d):>+8.2f} {max(lots_d):7.4f} {sum(pnls_d):>+8.2f} {day_data[d]['overrides']:4d}")

tpd = [len(day_data[d]["trades"]) for d in sorted_days]
print()
print(f"  Trading days:     {len(tpd)} / 31")
print(f"  Avg trades/day:   {np.mean(tpd):.1f}")
print(f"  Min trades/day:   {min(tpd)}")
print(f"  Max trades/day:   {max(tpd)}")
print()

# Trades per event (consecutive same-direction = same breakout opportunity)
events = []
current = []
for t in trades:
    if not current or t[1] == current[-1][1]:
        current.append(t)
    else:
        events.append(current)
        current = [t]
if current: events.append(current)

print("--- Trades Per Event (same-direction streak) ---")
tpe = [len(e) for e in events]
print(f"  Total events:        {len(events)}")
print(f"  Avg trades/event:    {np.mean(tpe):.1f}")
print(f"  Max trades/event:    {max(tpe)}")
print(f"  Min trades/event:    {min(tpe)}")
print(f"  1-trade events:      {sum(1 for e in events if len(e) == 1)} ({sum(1 for e in events if len(e) == 1)/len(events)*100:.0f}%)")
print(f"  2-3 trade events:    {sum(1 for e in events if 2 <= len(e) <= 3)} ({sum(1 for e in events if 2 <= len(e) <= 3)/len(events)*100:.0f}%)")
print(f"  4+ trade events:     {sum(1 for e in events if len(e) >= 4)} ({sum(1 for e in events if len(e) >= 4)/len(events)*100:.0f}%)")
print()

if len(events) >= 3:
    print("--- Top 3 Multi-Trade Events ---")
    sorted_events = sorted(events, key=len, reverse=True)[:3]
    for ei, e in enumerate(sorted_events):
        ew = sum(1 for t in e if t[4] > 0)
        lots_str = ", ".join(f"{t[0]:.4f}" for t in e)
        pnls_str = ", ".join(f"{t[4]:+.2f}" for t in e)
        times_str = ", ".join(f"{t[8].strftime('%H:%M')}" for t in e)
        print(f"  Event {ei+1}: {len(e)} trades, {ew}W ({ew/len(e)*100:.0f}%), dir={e[0][1]}")
        print(f"    Times: [{times_str}]")
        print(f"    Lots:  [{lots_str}]")
        print(f"    PnLs:  [{pnls_str}]")
    print()

print("--- ML Override vs Normal ---")
override_trades = [t for t in trades if t[3]]
normal_trades = [t for t in trades if not t[3]]
if override_trades:
    ow = sum(1 for t in override_trades if t[4] > 0)
    print(f"  Override: {len(override_trades)} trades, {ow}W ({ow/len(override_trades)*100:.0f}%), total PnL ${sum(t[4] for t in override_trades):+.2f}")
if normal_trades:
    nw = sum(1 for t in normal_trades if t[4] > 0)
    print(f"  Normal:   {len(normal_trades)} trades, {nw}W ({nw/len(normal_trades)*100:.0f}%), total PnL ${sum(t[4] for t in normal_trades):+.2f}")
print()

print("--- Score Distribution ---")
scores_arr = [t[2] for t in trades]
print(f"  Mean score: {np.mean(scores_arr):.4f}")
print(f"  Median:     {np.median(scores_arr):.4f}")
for pctile in [10, 25, 50, 75, 90, 95]:
    print(f"  P{pctile}:       {np.percentile(scores_arr, pctile):.4f}")
print()

print("--- Win Rate by Score Range ---")
for lo, hi, label in [(0.00, 0.02, "0.00-0.02"), (0.02, 0.05, "0.02-0.05"), (0.05, 0.10, "0.05-0.10"),
                       (0.10, 0.20, "0.10-0.20"), (0.20, 0.50, "0.20-0.50"), (0.50, 1.00, "0.50-1.00")]:
    group = [t for t in trades if lo <= t[2] < hi]
    if group:
        w = sum(1 for t in group if t[4] > 0)
        avg_lot = np.mean([t[0] for t in group])
        print(f"  {label:>10s}: {len(group):3d} trades, {w:3d}W ({w/len(group)*100:.0f}%), avg lot {avg_lot:.4f}, avg PnL ${np.mean([t[4] for t in group]):+.2f}")

print()
print("--- Exit Reason Breakdown ---")
er_counts = Counter(t[5] for t in trades)
for reason, count in sorted(er_counts.items(), key=lambda x: -x[1]):
    g = [t for t in trades if t[5] == reason]
    gw = sum(1 for t in g if t[4] > 0)
    total_pnl = sum(t[4] for t in g)
    print(f"  {reason:20s}: {count:4d} ({count/len(trades)*100:.1f}%), {gw:3d}W ({gw/count*100:.0f}%), PnL ${total_pnl:+.2f}")
