import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features

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

jan_indices = [i for i in range(len(m5)) if pd.Timestamp("2025-01-01") <= m5.index[i] < pd.Timestamp("2025-02-01")]

def run_test(max_override):
    bal = 20.0; total = 0; wins = 0
    override_count = 0; override_day = None; total_overrides = 0
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

        day_key = m5.index[i].date()
        if override_day != day_key:
            override_count = 0
            override_day = day_key

        entry_dir = expected; was_overridden = False
        if not np.isnan(pb_d[i]):
            cond = (expected == "BUY" and pb_d[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_d[i] > pb_u[i] and override_count < max_override)
            if cond:
                entry_dir = "SELL"; was_overridden = True
            cond2 = (expected == "SELL" and pb_u[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_u[i] > pb_d[i] and override_count < max_override)
            if cond2:
                entry_dir = "BUY"; was_overridden = True
            if was_overridden:
                override_count += 1
                total_overrides += 1

        if not was_overridden:
            if not np.isnan(pb_u[i]) and not np.isnan(pb_d[i]):
                if expected == "BUY" and pb_u[i] < cfg.ML_CONFIDENCE_THRESHOLD: continue
                if expected == "SELL" and pb_d[i] < cfg.ML_CONFIDENCE_THRESHOLD: continue

        n = max(0.01, bal * 200 / p / 100); n = min(n, 1.0)
        mg = p * 100 / 200; mx = bal / mg
        if mx < cfg.MIN_LOT: continue
        n = min(n, mx)
        ep = p; total += 1

        peak_profit = 0.0
        for j in range(i + 1, min(i + cfg.PEAK_HARVEST_MAX_HOLD_BARS + 1, len(m5) - 1)):
            fp = float(m5_close[j]); fh = float(m5_high[j]); fl = float(m5_low[j])
            diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
            prof = diff * 100 * n
            if entry_dir == "BUY": peak_profit = max(peak_profit, float(fh - ep))
            else: peak_profit = max(peak_profit, float(ep - fl))
            atr_j = float(m5_atr[j])

            ml_hold = False
            if not np.isnan(pb_u[j]):
                if entry_dir == "BUY" and pb_u[j] >= cfg.ML_HOLD_CONFIDENCE: ml_hold = True
                elif entry_dir == "SELL" and pb_d[j] >= cfg.ML_HOLD_CONFIDENCE: ml_hold = True

            ex = False; er = None
            if peak_profit > 0:
                if peak_profit >= atr_j * cfg.PEAK_HARVEST_TRAIL_TRIGGER:
                    pull = peak_profit - max(0, diff)
                    if pull / peak_profit > cfg.PEAK_HARVEST_TRAIL_RETRACE: ex = True; er = "trail"
            if not ex and not ml_hold and (j-i) >= 4:
                streak = 0
                for k in range(min(cfg.DIRECTION_LOSS_LOOKBACK, j-i)):
                    ik = j - k
                    if entry_dir == "BUY":
                        if m5_close[ik] < m5_open[ik]: streak += 1
                        else: break
                    else:
                        if m5_close[ik] > m5_open[ik]: streak += 1
                        else: break
                if streak >= cfg.DIRECTION_LOSS_STREAK: ex = True; er = "dl"
            if not ex and not ml_hold and (j-i) >= cfg.PEAK_HARVEST_MIN_BARS_EXIT:
                start = max(0, j - 19)
                cw = m5_close[start:j+1]; ow = m5_open[start:j+1]
                if len(cw) >= 5:
                    rc = cw[-3:]; oc = cw[-6:-3] if len(cw) >= 6 else cw[:3]
                    avg_body = float(np.mean(np.abs(cw[-5:] - ow[-5:])))
                    if avg_body > 0:
                        window_atr = float(np.mean(m5_atr[max(0,j-13):j+1]))
                        ref = max(window_atr, fp * 0.0001)
                        raw = (abs(rc[-1]-rc[0]) / (abs(oc[-1]-oc[0]) + 1e-10)) * (avg_body / (ref + 1e-10))
                        momentum = min(abs(raw), 1.0)
                        if (1.0 - momentum) > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
                            ex = True; er = "md"
            if not ex and not ml_hold:
                if not np.isnan(pb_d[j]) and not np.isnan(pb_u[j]):
                    if (entry_dir == "BUY" and pb_d[j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD) or \
                       (entry_dir == "SELL" and pb_u[j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD):
                        ex = True; er = "mlr"
            if not ex and (j-i) >= cfg.PEAK_HARVEST_MAX_HOLD_BARS:
                ex = True; er = "mh"
            if ex:
                bal += prof
                if prof > 0: wins += 1
                bal = max(bal, 0.01)
                break
        else:
            j = min(i + cfg.PEAK_HARVEST_MAX_HOLD_BARS, len(m5)-1)
            prof = (float(m5_close[j])-ep)*100*n if entry_dir == "BUY" else (ep-float(m5_close[j]))*100*n
            bal += prof
            if prof > 0: wins += 1
            bal = max(bal, 0.01)

    wr = wins/total*100 if total else 0
    return total, wr, bal-20, total_overrides

print(f"{'Limit':>6s} {'Trades':>8s} {'WR':>6s} {'PnL':>12s} {'Ovr':>6s} {'Ovr%':>6s}")
print("-" * 50)
for lim in [3, 5, 10, 20, 99]:
    t, wr, pnl, ov = run_test(lim)
    print(f"{lim:>6d} {t:>8d} {wr:>5.1f}% {pnl:>+10.2f} {ov:>6d} {ov/t*100:>5.1f}%")
