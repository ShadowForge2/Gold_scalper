import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features

CS = 100; LEV = 200
TRADE_MAX_BARS = cfg.PEAK_HARVEST_MAX_HOLD_BARS

pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")

client = DukascopyClient()
m1 = client.download_year(2025)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
m5 = client.resample_to(m1, 5).set_index("time")
h1 = client.resample_to(m1, 16385).set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]

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

print(f"January M5 bars: {len(jan_indices)}")
print(f"Range: {m5.index[min(jan_indices)]} to {m5.index[max(jan_indices)]}")
print()

from collections import defaultdict

daily_trades = defaultdict(int)
daily_overrides = defaultdict(int)
daily_wins = defaultdict(int)
daily_losses = defaultdict(int)

bal = 20.0; total = 0; wins = 0
override_count = 0; override_day = None
total_overrides = 0
exit_reasons = {}

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
        if was_overridden:
            override_count += 1
            total_overrides += 1

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

    ep = p; total += 1
    daily_trades[day_key] += 1
    if was_overridden:
        daily_overrides[day_key] += 1

    peak_profit = 0.0
    for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, len(m5) - 1)):
        fp = float(m5_close[j])
        fh = float(m5_high[j])
        fl = float(m5_low[j])
        diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
        prof = diff * CS * n

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
                    momentum = min(abs(raw), 1.0)
                    exit_score = 1.0 - momentum
                    if exit_score > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
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
            bal += prof
            if prof > 0:
                wins += 1; daily_wins[day_key] += 1
            else:
                daily_losses[day_key] += 1
            exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
            bal = max(bal, 0.01)
            break
    else:
        j = min(i + TRADE_MAX_BARS, len(m5) - 1)
        fp = float(m5_close[j])
        prof = (fp - ep) * CS * n if entry_dir == "BUY" else (ep - fp) * CS * n
        bal += prof
        exit_reasons["max_hold"] = exit_reasons.get("max_hold", 0) + 1
        if prof > 0: wins += 1
        bal = max(bal, 0.01)

print("=" * 60)
print("  January 2025 — ML Override + ML Exit Hold")
print("=" * 60)
print(f"Total trades:        {total}")
print(f"Wins:                {wins} ({wins/total*100:.1f}%)")
print(f"Losses:              {total - wins} ({(total-wins)/total*100:.1f}%)")
print(f"Net PnL:             ${bal - 20:>+.2f}")
print(f"ML overrides:        {total_overrides} ({total_overrides/total*100:.1f}%)")
print(f"Exit reasons:        {dict(exit_reasons)}")
print()

print("=== Per-Day Breakdown ===")
print(f"{'Date':12s} {'Trades':>6s} {'W/L':>8s} {'WR':>5s} {'Ovr':>4s}")
print("-" * 40)
total_w = 0
total_l = 0
for d in sorted(daily_trades.keys()):
    w = daily_wins.get(d, 0)
    l = daily_losses.get(d, 0)
    ov = daily_overrides.get(d, 0)
    wr = f"{w/(w+l)*100:.0f}%" if (w+l) > 0 else "-"
    print(f"{str(d):12s} {daily_trades[d]:6d} {f'{w}/{l}':>8s} {wr:>5s} {ov:4d}")
    total_w += w
    total_l += l

days_traded = len(daily_trades)
print("-" * 40)
print(f"{'TOTAL':12s} {total:6d} {f'{total_w}/{total_l}':>8s} {f'{total_w/total*100:.0f}%':>5s} {total_overrides:4d}")
print()
print(f"Trading days:        {days_traded} / 31")
print(f"Avg trades/day:     {total/days_traded:.1f}")
print(f"Min trades/day:     {min(daily_trades.values())}")
print(f"Max trades/day:     {max(daily_trades.values())}")
print(f"Avg overrides/day:  {total_overrides/days_traded:.1f}")
