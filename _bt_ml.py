"""
Backtest: Meta-Aggressive strategy WITH ML direction validation.
Fast vectorized bias computation.
"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features
from app.bias_engine import BiasEngine

CS = 100; LEV = 200

def compute_bias_vectorized(h1: pd.DataFrame) -> np.ndarray:
    """Return bias for every H1 bar: 1=BULLISH, -1=BEARISH, 0=NEUTRAL. Vectorized."""
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

    # Vectorized swing detection with lookback=5
    lookback = 5
    n = len(c)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        is_high[i] = all(h[i] >= h[i - j] for j in range(1, lookback + 1)) and \
                     all(h[i] >= h[i + j] for j in range(1, lookback + 1))
        is_low[i] = all(l[i] <= l[i - j] for j in range(1, lookback + 1)) and \
                    all(l[i] <= l[i + j] for j in range(1, lookback + 1))

    # Compute swing scores incrementally
    swing_score = np.zeros(n)
    for i in range(n):
        hi = np.where(is_high[:i+1])[0]
        lo = np.where(is_low[:i+1])[0]
        if len(hi) >= 3 and len(lo) >= 3:
            rh = hi[-3:]; rl = lo[-3:]
            h_up = sum(1 for k in range(1, len(rh)) if h[rh[k]] > h[rh[k-1]])
            h_dn = sum(1 for k in range(1, len(rh)) if h[rh[k]] < h[rh[k-1]])
            l_up = sum(1 for k in range(1, len(rl)) if l[rl[k]] > l[rl[k-1]])
            l_dn = sum(1 for k in range(1, len(rl)) if l[rl[k]] < l[rl[k-1]])
            swing_score[i] = ((h_up - h_dn) + (l_up - l_dn)) / max(1, (len(rh)-1) + (len(rl)-1))

    total = votes + swing_score
    bias = np.zeros(n, dtype=np.int8)
    bias[total >= 0.75] = 1   # BULLISH
    bias[total <= -0.75] = -1 # BEARISH
    return bias


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

    # Pre-compute ML predictions
    model = pred.model
    X = ft[ft.columns.intersection(pred._feature_cols)]
    for c in pred._feature_cols:
        if c not in X.columns: X[c] = 0.0
    X = X[pred._feature_cols]
    X_aligned = X.reindex(m5.index, method="ffill")
    ml_mask = ~X_aligned.isna().any(axis=1)
    ml_dir = np.full(len(m5), None, dtype=object)
    if ml_mask.any():
        valid_idx = np.where(ml_mask)[0]
        probs = model.predict_proba(X_aligned[ml_mask].values)
        pb_u = np.array([p[1] for p in probs])
        pb_d = np.array([p[0] for p in probs])
        for k, idx in enumerate(valid_idx):
            if pb_u[k] >= 0.55: ml_dir[idx] = "BUY"
            elif pb_d[k] >= 0.55: ml_dir[idx] = "SELL"

    # Compute bias for all H1 bars, then map to M5 bars by forward fill
    print(f"  [{year}] Bias compute...", end=" ", flush=True)
    h1_bias = compute_bias_vectorized(h1)
    h1_idx_map = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx_map = np.clip(h1_idx_map, 0, len(h1_bias) - 1)
    m5_bias = h1_bias[h1_idx_map]
    m5_bias_arr = np.full(len(m5_bias), None, dtype=object)
    m5_bias_arr[m5_bias == 1] = "BUY"
    m5_bias_arr[m5_bias == -1] = "SELL"
    print("OK")

    # Also pre-compute H1 high/low for each M5 bar
    m5_h1h = h1["high"].values[h1_idx_map]
    m5_h1l = h1["low"].values[h1_idx_map]
    m5_h1h = np.where(h1_idx_map > 0, h1["high"].values[h1_idx_map - 1], m5_h1h)
    m5_h1l = np.where(h1_idx_map > 0, h1["low"].values[h1_idx_map - 1], m5_h1l)

    # Backtest loop
    CONTRACT_SIZE = 100  # oz per standard lot
    LEV = 200
    bal = 20.0; total = 0; wins = 0; dd = 0; peak = 20.0
    t0 = time.time()

    for i in range(60, len(m5) - 15):
        expected = m5_bias_arr[i]
        if expected is None: continue
        if ml_dir[i] != expected: continue

        p = float(m5.iloc[i]["close"])
        h1h = float(m5_h1h[i]); h1l = float(m5_h1l[i])
        if h1h <= h1l: continue
        if expected == "BUY" and p <= h1h: continue
        if expected == "SELL" and p >= h1l: continue

        ep = p; total += 1

        # Full margin sizing: max_oz = bal * LEV / ep
        n = max(0.01, bal * LEV / ep / CONTRACT_SIZE)
        n = min(n, 1.0)  # cap at 1 standard lot

        atr_val = float(m5.iloc[i]["close"]) * 0.002
        if expected == "BUY":
            sl = ep - atr_val; tp1 = ep + atr_val * 2; tp2 = ep + atr_val * 4; tp3 = ep + atr_val * 6
        else:
            sl = ep + atr_val; tp1 = ep - atr_val * 2; tp2 = ep - atr_val * 4; tp3 = ep - atr_val * 6

        for j in range(i + 1, min(i + 48, len(m5) - 1)):
            fb = m5.iloc[j]; fp = float(fb["close"]); fh = float(fb["high"]); fl = float(fb["low"])
            prof = (fp - ep) * CONTRACT_SIZE * n if expected == "BUY" else (ep - fp) * CONTRACT_SIZE * n

            if expected == "BUY":
                if fl <= sl: bal += prof; break
                if fp >= tp3: bal += prof; wins += 1; break
                if fp >= tp2: bal += prof; wins += 1; break
                if fp >= tp1: bal += prof; wins += 1; break
            else:
                if fh >= sl: bal += prof; break
                if fp <= tp3: bal += prof; wins += 1; break
                if fp <= tp2: bal += prof; wins += 1; break
                if fp <= tp1: bal += prof; wins += 1; break
        else:
            fb = m5.iloc[min(i + 48, len(m5) - 1)]
            fp = float(fb["close"])
            prof = (fp - ep) * CONTRACT_SIZE * n if expected == "BUY" else (ep - fp) * CONTRACT_SIZE * n
            bal += prof
            if prof > 0: wins += 1

        peak = max(peak, bal)
        dd = max(dd, peak - bal)

    elapsed = time.time() - t0
    wr = (wins / total * 100) if total else 0
    pf = wins / max(1, total - wins)
    print(f"{year}: {total:>4} tr WR={wr:>5.1f}% PF={pf:>5.2f} PnL=${bal-20:>+8.2f} DD=${dd:>8.2f} Bal=${bal:>8.2f} [{elapsed:.0f}s]")
    return total, wr, bal, dd

if __name__ == "__main__":
    pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
    print("Model loaded.\n")
    for y in [2022, 2023, 2024, 2025]:
        run_bt(y, pred)
