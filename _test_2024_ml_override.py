"""
Backtest 2024: Meta-Aggressive + ML Bias Override + ML-Guided Exit Hold.
Peak harvest exit (no SL). Uses pre-computed ML probabilities from direction model.
ML Override: flip entry dir if ML predicts opposite at >=70% (max 3/day).
ML Exit Hold: suppress momentum_decay/direction_loss if ML still predicts same dir >=50%.
"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features

CS = 100; LEV = 200


def compute_bias_vectorized(h1: pd.DataFrame) -> np.ndarray:
    c = h1["close"].values.astype(np.float64)
    h = h1["high"].values.astype(np.float64)
    l = h1["low"].values.astype(np.float64)
    fast = pd.Series(c).ewm(span=20, adjust=False).mean().values
    slow = pd.Series(c).ewm(span=50, adjust=False).mean().values
    fast_slope = np.full(len(c), 0.0)
    slow_slope = np.full(len(c), 0.0)
    if len(c) >= 6:
        fast_slope[5:] = fast[5:] - fast[:-5]
        slow_slope[5:] = slow[5:] - slow[:-5]
    votes = np.zeros(len(c))
    cross = (fast > slow) & (fast_slope > 0)
    votes[cross] += 1.0
    cross = (fast < slow) & (fast_slope < 0)
    votes[cross] -= 1.0
    price_above = c > slow
    votes[price_above & (slow_slope >= 0)] += 0.5
    votes[~price_above & (slow_slope <= 0)] -= 0.5
    lookback = 5
    n = len(c)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        is_high[i] = all(h[i] >= h[i - j] for j in range(1, lookback + 1)) and \
                     all(h[i] >= h[i + j] for j in range(1, lookback + 1))
        is_low[i] = all(l[i] <= l[i - j] for j in range(1, lookback + 1)) and \
                    all(l[i] <= l[i + j] for j in range(1, lookback + 1))
    swing_score = np.zeros(n)
    for i in range(n):
        hi = np.where(is_high[:i + 1])[0]
        lo = np.where(is_low[:i + 1])[0]
        if len(hi) >= 3 and len(lo) >= 3:
            rh = hi[-3:]; rl = lo[-3:]
            h_up = sum(1 for k in range(1, len(rh)) if h[rh[k]] > h[rh[k - 1]])
            h_dn = sum(1 for k in range(1, len(rh)) if h[rh[k]] < h[rh[k - 1]])
            l_up = sum(1 for k in range(1, len(rl)) if l[rl[k]] > l[rl[k - 1]])
            l_dn = sum(1 for k in range(1, len(rl)) if l[rl[k]] < l[rl[k - 1]])
            swing_score[i] = ((h_up - h_dn) + (l_up - l_dn)) / max(1, (len(rh) - 1) + (len(rl) - 1))
    total = votes + swing_score
    bias = np.zeros(n, dtype=np.int8)
    bias[total >= 0.75] = 1
    bias[total <= -0.75] = -1
    return bias


def compute_atr_series(high, low, close, period=14):
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
    return np.where(np.isnan(atr), 0.0, atr)


def run_bt(year, pred):
    client = DukascopyClient()
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    print(f"  [{year}] Features ({len(m5)} bars)...", end=" ", flush=True)
    ft = compute_features(m5, h1)
    print(f"OK ({len(ft)} rows)")

    # Pre-compute ML probabilities for all bars
    model = pred.model
    X = ft[ft.columns.intersection(pred._feature_cols)]
    for c in pred._feature_cols:
        if c not in X.columns:
            X[c] = 0.0
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

    print(f"  [{year}] ATR...", end=" ", flush=True)
    m5_atr = compute_atr_series(m5["high"].values, m5["low"].values, m5["close"].values)
    print("OK")

    print(f"  [{year}] Bias...", end=" ", flush=True)
    h1_bias = compute_bias_vectorized(h1)
    h1_idx_map = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx_map = np.clip(h1_idx_map, 0, len(h1_bias) - 1)
    m5_bias = h1_bias[h1_idx_map]
    m5_bias_arr = np.full(len(m5_bias), None, dtype=object)
    m5_bias_arr[m5_bias == 1] = "BUY"
    m5_bias_arr[m5_bias == -1] = "SELL"

    h1_h = h1["high"].values; h1_l = h1["low"].values
    m5_h1h = h1_h[h1_idx_map]
    m5_h1l = h1_l[h1_idx_map]
    m5_h1h = np.where(h1_idx_map > 0, h1_h[h1_idx_map - 1], m5_h1h)
    m5_h1l = np.where(h1_idx_map > 0, h1_l[h1_idx_map - 1], m5_h1l)
    print("OK")

    m5_close = m5["close"].values
    m5_open = m5["open"].values

    bal = 20.0; total = 0; wins = 0; dd = 0; peak_bal = 20.0
    t0 = time.time()
    override_count = 0
    override_day = None
    total_overrides = 0
    TRADE_MAX_BARS = cfg.PEAK_HARVEST_MAX_HOLD_BARS

    for i in range(60, len(m5) - 15):
        expected = m5_bias_arr[i]
        if expected is None:
            continue

        p = float(m5_close[i])
        h1h = float(m5_h1h[i]); h1l = float(m5_h1l[i])
        if h1h <= h1l: continue
        if expected == "BUY" and p <= h1h: continue
        if expected == "SELL" and p >= h1l: continue

        # --- ML Bias Override ---
        ts_i = m5.index[i]
        day_key = ts_i.date()
        if override_day != day_key:
            override_count = 0
            override_day = day_key

        entry_dir = expected
        was_overridden = False
        if not np.isnan(pb_d[i]):
            if expected == "BUY" and pb_d[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_d[i] > pb_u[i] and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
                entry_dir = "SELL"
                was_overridden = True
            elif expected == "SELL" and pb_u[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_u[i] > pb_d[i] and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
                entry_dir = "BUY"
                was_overridden = True
            if was_overridden:
                override_count += 1
                total_overrides += 1

        if not was_overridden:
            if not np.isnan(pb_u[i]) and not np.isnan(pb_d[i]):
                if expected == "BUY" and pb_u[i] < cfg.ML_CONFIDENCE_THRESHOLD:
                    continue
                if expected == "SELL" and pb_d[i] < cfg.ML_CONFIDENCE_THRESHOLD:
                    continue

        ep = p; total += 1
        n = max(0.01, bal * LEV / ep / CS)
        n = min(n, 1.0)

        # --- Trade exit (peak harvest, no SL) ---
        peak_profit = 0.0

        for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, len(m5) - 1)):
            fp = float(m5_close[j])
            fh = float(m5["high"].iloc[j])
            fl = float(m5["low"].iloc[j])
            bars_held = j - i
            diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
            prof = diff * CS * n

            if entry_dir == "BUY":
                peak_profit = max(peak_profit, float(fh - ep))
            else:
                peak_profit = max(peak_profit, float(ep - fl))

            atr_j = float(m5_atr[j])

            # ML hold: does ML still predict same direction at >=50%?
            ml_hold = False
            if not np.isnan(pb_u[j]):
                if entry_dir == "BUY" and pb_u[j] >= cfg.ML_HOLD_CONFIDENCE:
                    ml_hold = True
                elif entry_dir == "SELL" and pb_d[j] >= cfg.ML_HOLD_CONFIDENCE:
                    ml_hold = True

            # 1. Trail stop (always checked)
            if peak_profit > 0:
                trail_trigger = atr_j * cfg.PEAK_HARVEST_TRAIL_TRIGGER
                if peak_profit >= trail_trigger:
                    pullback = peak_profit - max(0, diff)
                    if pullback / peak_profit > cfg.PEAK_HARVEST_TRAIL_RETRACE:
                        bal += prof
                        if prof > 0: wins += 1
                        break

            # 2. Direction loss (only if NOT ml_hold)
            if not ml_hold and bars_held >= 4:
                streak = 0
                lookback = min(cfg.DIRECTION_LOSS_LOOKBACK, j - i)
                for k in range(lookback):
                    idx_k = j - k
                    if entry_dir == "BUY":
                        if m5_close[idx_k] < m5_open[idx_k]:
                            streak += 1
                        else:
                            break
                    else:
                        if m5_close[idx_k] > m5_open[idx_k]:
                            streak += 1
                        else:
                            break
                if streak >= cfg.DIRECTION_LOSS_STREAK:
                    bal += prof
                    if prof > 0: wins += 1
                    break

            # 3. Momentum decay (only if NOT ml_hold)
            if not ml_hold and bars_held >= cfg.PEAK_HARVEST_MIN_BARS_EXIT:
                start = max(0, j - 19)
                c_win = m5_close[start:j + 1]
                o_win = m5_open[start:j + 1]
                if len(c_win) >= 5:
                    recent = c_win[-3:]
                    older = c_win[-6:-3] if len(c_win) >= 6 else c_win[:3]
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
                            bal += prof
                            if prof > 0: wins += 1
                            break

            # 4. Max hold (always checked)
            if bars_held >= TRADE_MAX_BARS:
                bal += prof
                if prof > 0: wins += 1
                break
        else:
            fb = m5.iloc[min(i + TRADE_MAX_BARS, len(m5) - 1)]
            fp = float(fb["close"])
            prof = (fp - ep) * CS * n if entry_dir == "BUY" else (ep - fp) * CS * n
            bal += prof
            if prof > 0: wins += 1

        peak_bal = max(peak_bal, bal)
        dd = max(dd, peak_bal - bal)

    elapsed = time.time() - t0
    wr = (wins / total * 100) if total else 0
    pf = wins / max(1, total - wins) if (total - wins) > 0 else float("inf")
    print(f"{year}: {total:>4} tr  WR={wr:>5.1f}%  PF={pf:>5.2f}  "
          f"PnL=${bal - 20:>+8.2f}  DD=${dd:>8.2f}  Bal=${bal:>8.2f}  "
          f"[{elapsed:.0f}s]  overrides={total_overrides}")
    return total, wr, bal, dd, total_overrides


if __name__ == "__main__":
    pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
    print("Model loaded.\n")
    run_bt(2024, pred)
