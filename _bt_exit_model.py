"""
Backtest comparison: EXIT_MODE=5 baseline vs exit model.
Runs 2022-2024, same entries, compares exit quality.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import (
    compute_features, FEATURE_COLS, ExitPredictor, EXIT_FEATURE_COLS,
)

CS = 100; LEV = 200; TRADE_MAX_BARS = 25
SESSION_HOURS = {"ASIA": (0, 8), "LONDON": (7, 17), "NEW_YORK": (12, 22)}

def in_allowed_session(ts):
    h = ts.hour
    for s in SESSION_HOURS:
        lo, hi = SESSION_HOURS[s]
        if lo <= h < hi:
            return True
    return False

def compute_atr(high, low, close, period=14):
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period, min_periods=period).mean().bfill().fillna(0.01).values

def compute_bias_vectorized(h1):
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
    cross = (fast > slow) & (fast_slope > 0); votes[cross] += 1.0
    cross = (fast < slow) & (fast_slope < 0); votes[cross] -= 1.0
    price_above = c > slow
    votes[price_above & (slow_slope >= 0)] += 0.5
    votes[~price_above & (slow_slope <= 0)] -= 0.5
    n = len(c); lookback = 5
    is_high = np.zeros(n, dtype=bool); is_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        is_high[i] = all(h[i] >= h[i - j] for j in range(1, lookback + 1)) and all(h[i] >= h[i + j] for j in range(1, lookback + 1))
        is_low[i] = all(l[i] <= l[i - j] for j in range(1, lookback + 1)) and all(l[i] <= l[i + j] for j in range(1, lookback + 1))
    swing_score = np.zeros(n)
    for i in range(n):
        hi = np.where(is_high[:i + 1])[0]; lo = np.where(is_low[:i + 1])[0]
        if len(hi) >= 3 and len(lo) >= 3:
            rh = hi[-3:]; rl = lo[-3:]
            h_up = sum(1 for k in range(1, len(rh)) if h[rh[k]] > h[rh[k - 1]])
            h_dn = sum(1 for k in range(1, len(rh)) if h[rh[k]] < h[rh[k - 1]])
            l_up = sum(1 for k in range(1, len(rl)) if l[rl[k]] > l[rl[k - 1]])
            l_dn = sum(1 for k in range(1, len(rl)) if l[rl[k]] < l[rl[k - 1]])
            swing_score[i] = ((h_up - h_dn) + (l_up - l_dn)) / max(1, (len(rh) - 1) + (len(rl) - 1))
    total = votes + swing_score
    bias = np.zeros(n, dtype=np.int8)
    bias[total >= 0.75] = 1; bias[total <= -0.75] = -1
    return bias


def run_bt(year, exit_predictor=None):
    client = DukascopyClient()
    m1 = client.download_year(year).sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = client.resample_to(m1, 16385).set_index("time")
    h1 = h1[~h1.index.duplicated(keep="first")]

    close = m5["close"].values.astype(float)
    high = m5["high"].values.astype(float)
    low = m5["low"].values.astype(float)
    open_ = m5["open"].values.astype(float)
    n = len(m5)

    atr = compute_atr(high, low, close)

    h1_idx = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx = np.clip(h1_idx, 0, len(h1) - 1)
    h1_h = np.where(h1_idx > 0, h1["high"].values[h1_idx - 1], h1["high"].values[h1_idx])
    h1_l = np.where(h1_idx > 0, h1["low"].values[h1_idx - 1], h1["low"].values[h1_idx])

    bias_arr = compute_bias_vectorized(h1)
    bias_map = bias_arr[np.clip(h1_idx, 0, len(bias_arr) - 1)]

    # Precompute market features for exit model
    feat_all = None
    if exit_predictor is not None:
        h1_a = h1.reindex(m5.index, method="ffill")
        feat_all = compute_features(m5, h1_a)

    # Precompute entry features for each bar (used by direction model)
    h1_feat = h1.reindex(m5.index, method="ffill")
    m5_feat = m5.copy()
    market_feat = compute_features(m5_feat, h1_feat)

    # Stats
    total_trades = 0; wins = 0; total_pnl = 0.0; dd = 0; peak = 0.0
    bal = 20.0
    exit_reasons = {}
    exit_model_holds = 0
    exit_model_exits = 0

    for i in range(100, n - 30):
        if bias_map[i] == 0 or not in_allowed_session(m5.index[i]):
            continue
        direction = "BUY" if bias_map[i] == 1 else "SELL"
        p = close[i]; hh = h1_h[i]; hl = h1_l[i]
        if hh <= hl: continue
        rs = hh - hl
        bd = (p - hh) if direction == "BUY" else (hl - p)
        if bd <= 0: continue
        score = min(bd / rs, 1.0)
        if score < 0.02: continue

        # Entry
        ep = p; eb = i
        best_pnl = 0.0; w_streak = 0; exited = False

        for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, n - 1)):
            diff = (close[j] - ep) if direction == "BUY" else (ep - close[j])
            bars = j - i
            hi_diff = (high[j] - ep) if direction == "BUY" else (ep - low[j])
            best_pnl = max(best_pnl, hi_diff)
            w_streak = (w_streak + 1) if (direction == "BUY" and close[j] < open_[j]) or \
                        (direction == "SELL" and close[j] > open_[j]) else 0

            # --- Exit logic ---
            atr_j = max(atr[j], 0.01)
            px_j = close[j]

            # Exit model check
            if exit_predictor is not None and feat_all is not None and j < len(feat_all):
                mf = feat_all.iloc[j].to_dict()
                pnl_atr = diff / atr_j
                peak_atr = best_pnl / atr_j
                drawdown = (best_pnl - max(0, diff)) / max(best_pnl, 0.001)
                ts = {
                    "bars_held": bars, "pnl_atr": round(pnl_atr, 4),
                    "peak_atr": round(peak_atr, 4),
                    "drawdown_pct": round(drawdown, 4),
                    "entry_score": round(score, 4), "atr_change": 1.0,
                    "wrong_streak": w_streak,
                }
                hold_prob = exit_predictor.predict_hold_prob(mf, ts)
                if hold_prob >= 0.70:
                    if bars > TRADE_MAX_BARS:
                        exit_pnl = diff; reason = "max_hold"
                        exit_model_holds += 1; exited = True
                    else:
                        continue  # strong hold
                elif hold_prob >= 0.60:
                    # moderate hold - only ML reversal can exit
                    # skip mechanical exits, let it run
                    if bars > TRADE_MAX_BARS:
                        exit_pnl = diff; reason = "max_hold"
                        exit_model_holds += 1; exited = True
                    else:
                        continue
                elif hold_prob <= 0.30:
                    exit_pnl = diff; reason = "exit_model_exit"
                    exit_model_exits += 1; exited = True
                else:
                    # uncertain - use mechanical exits
                    pass

            if not exited:
                # Mechanical exit (baseline logic)
                peak_check = best_pnl
                if direction == "BUY":
                    peak_check = float(np.max(high[i:j+1] - ep))
                else:
                    peak_check = float(np.max(ep - low[i:j+1]))

                trail_trigger = atr_j * 1.5 * (1.0 + (1.0 - score))
                trail_active = peak_check >= trail_trigger
                if trail_active and diff > 0 and peak_check > 0:
                    pullback = peak_check - diff
                    if pullback / peak_check > 0.3:
                        exit_pnl = diff; reason = "trail_stop"; exited = True

            if not exited and bars >= 4:
                streak = 0
                for k in range(min(5, j - i)):
                    idx_k = j - k
                    if direction == "BUY" and close[idx_k] < open_[idx_k]: streak += 1
                    elif direction == "SELL" and close[idx_k] > open_[idx_k]: streak += 1
                    else: break
                if streak >= 3:
                    exit_pnl = diff; reason = "direction_loss"; exited = True

            if not exited and bars >= 10:
                mom = 0.5  # simplified
                if (1.0 - mom) > 0.85:
                    exit_pnl = diff; reason = "momentum_decay"; exited = True

            if not exited and bars > TRADE_MAX_BARS:
                exit_pnl = diff; reason = "max_hold"; exited = True

            if exited:
                total_trades += 1; total_pnl += exit_pnl
                if exit_pnl > 0: wins += 1
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                bal += exit_pnl
                peak = max(peak, bal)
                dd = max(dd, peak - bal)
                break

    return {
        "trades": total_trades, "wins": wins,
        "net_pnl": round(total_pnl, 2),
        "dd": round(dd, 2),
        "exit_reasons": exit_reasons,
        "exit_model_holds": exit_model_holds,
        "exit_model_exits": exit_model_exits,
    }


print("Loading exit model...")
ep = ExitPredictor(model_path="models/exit_xgb_m5.joblib")

for year in [2022, 2023, 2024]:
    print(f"\n=== {year} ===")
    t0 = time.time()

    base = run_bt(year, exit_predictor=None)
    print(f"Baseline:   {base['trades']} trades WR={base['wins']/max(1,base['trades']):.1%} "
          f"PnL=${base['net_pnl']} DD=${base['dd']}")

    ex = run_bt(year, exit_predictor=ep)
    print(f"+ExitModel: {ex['trades']} trades WR={ex['wins']/max(1,ex['trades']):.1%} "
          f"PnL=${ex['net_pnl']} DD=${ex['dd']} "
          f"holds={ex['exit_model_holds']} exits={ex['exit_model_exits']}")

    print(f"  Time: {time.time()-t0:.1f}s")
    if base["trades"] > 0:
        improvement = ((ex["net_pnl"] - base["net_pnl"]) / abs(base["net_pnl"])) * 100 if base["net_pnl"] != 0 else 0
        print(f"  PnL change: {improvement:+.1f}%")
