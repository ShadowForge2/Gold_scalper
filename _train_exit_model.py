"""
Train XGBoost exit-timing model for XAUUSD.
Streamlined: uses bias direction, 2020-2021 train + 2022 val.
"""
import sys, os, time, gc
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import compute_features, FEATURE_COLS

MODEL_PATH = "models/exit_xgb_m5.joblib"
EXIT_FEATURE_COLS = FEATURE_COLS + [
    "bars_held", "pnl_atr", "peak_atr", "drawdown_pct",
    "entry_score", "atr_change", "wrong_streak",
]
TRADE_MAX_BARS = 25
SESSION_HOURS = {"ASIA": (0, 8), "LONDON": (7, 17), "NEW_YORK": (12, 22)}

def in_allowed_session(ts):
    h = ts.hour
    for s in SESSION_HOURS:
        lo, hi = SESSION_HOURS[s]
        if lo <= h < hi:
            return True
    return False

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
        is_high[i] = all(h[i] >= h[i - j] for j in range(1, lookback + 1)) and all(h[i] >= h[i + j] for j in range(1, lookback + 1))
        is_low[i] = all(l[i] <= l[i - j] for j in range(1, lookback + 1)) and all(l[i] <= l[i + j] for j in range(1, lookback + 1))
    swing_score = np.zeros(n)
    for i in range(n):
        hi = np.where(is_high[:i + 1])[0]
        lo = np.where(is_low[:i + 1])[0]
        if len(hi) >= 3 and len(lo) >= 3:
            rh = hi[-3:]
            rl = lo[-3:]
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

def process_year(year, client=None):
    if client is None:
        client = DukascopyClient()
    t0 = time.time()
    m1 = client.download_year(year).sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = client.resample_to(m1, 16385).set_index("time")
    h1 = h1[~h1.index.duplicated(keep="first")]
    del m1; gc.collect()
    print(f"  [{year}] Downloaded ({time.time()-t0:.1f}s)", flush=True)

    n = len(m5)
    close = m5["close"].values.astype(float)
    high = m5["high"].values.astype(float)
    low = m5["low"].values.astype(float)
    open_ = m5["open"].values.astype(float)

    # ATR
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().bfill().fillna(0.01).values

    # H1 alignment
    h1_idx = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx = np.clip(h1_idx, 0, len(h1) - 1)
    h1_h = np.where(h1_idx > 0, h1["high"].values[h1_idx - 1], h1["high"].values[h1_idx])
    h1_l = np.where(h1_idx > 0, h1["low"].values[h1_idx - 1], h1["low"].values[h1_idx])

    # Bias
    bias_arr = compute_bias_vectorized(h1)
    bias_map = bias_arr[np.clip(h1_idx, 0, len(bias_arr) - 1)]

    # Market features
    t1 = time.time()
    h1_a = h1.reindex(m5.index, method="ffill")
    feat_all = compute_features(m5, h1_a)
    print(f"  [{year}] Features ({time.time()-t1:.1f}s)", flush=True)

    # Find entries filtered by bias
    entries = []
    for i in range(100, n - 30):
        if not in_allowed_session(m5.index[i]):
            continue
        b = bias_map[i]
        if b == 0:
            continue
        direction = "BUY" if b == 1 else "SELL"
        p = close[i]; hh = h1_h[i]; hl = h1_l[i]
        if hh <= hl: continue
        rs = hh - hl
        bd = (p - hh) if direction == "BUY" else (hl - p)
        if bd <= 0: continue
        score = min(bd / rs, 1.0)
        if score < 0.02: continue
        entries.append((i, direction, p, score))

    print(f"  [{year}] {len(entries)} entries ({time.time()-t0:.1f}s)", flush=True)

    # Generate training rows
    rows = []
    for ei, (ebar, direction, ep, score) in enumerate(entries):
        if ei % 2000 == 0 and ei > 0:
            print(f"    [{year}] {ei}/{len(entries)} rows={len(rows)} ({time.time()-t0:.1f}s)", flush=True)
        best_pnl = 0.0
        w_streak = 0
        atr_e = max(atr[ebar], 0.01)
        max_j = min(ebar + TRADE_MAX_BARS + 1, n - 1)

        for j in range(ebar + 1, max_j):
            diff = (close[j] - ep) if direction == "BUY" else (ep - close[j])
            bars = j - ebar
            hi = (high[j] - ep) if direction == "BUY" else (ep - low[j])
            best_pnl = max(best_pnl, hi)
            w_streak = (w_streak + 1) if (direction == "BUY" and close[j] < open_[j]) or \
                        (direction == "SELL" and close[j] > open_[j]) else 0

            # Label
            la = min(TRADE_MAX_BARS - bars, n - 1 - j)
            if la > 0:
                bf = max(np.max(high[j:j+la] - ep) if direction == "BUY" else np.max(ep - low[j:j+la]), 0)
            else:
                bf = max(diff, 0)
            remaining = bf - max(diff, 0)

            if remaining > 0.5 * atr_e:
                label = 1
            elif remaining < 0.1 * atr_e:
                label = 0
            else:
                continue

            if j >= len(feat_all):
                continue
            mf = feat_all.iloc[j].to_dict()
            mf["bars_held"] = bars
            mf["pnl_atr"] = round(diff / atr_e, 4)
            mf["peak_atr"] = round(best_pnl / atr_e, 4)
            mf["drawdown_pct"] = round((best_pnl - max(0, diff)) / max(best_pnl, 0.001), 4)
            mf["entry_score"] = round(score, 4)
            mf["atr_change"] = round(atr[j] / atr_e, 4)
            mf["wrong_streak"] = w_streak
            mf["target"] = label
            mf["year"] = year
            rows.append(mf)

    print(f"  [{year}] Done: {len(rows)} rows ({time.time()-t0:.1f}s)", flush=True)
    return rows


def main():
    all_rows = []
    for year in range(2007, 2022):
        t0 = time.time()
        try:
            rows = process_year(year)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  {year}: ERROR - {e}", flush=True)
            import traceback; traceback.print_exc()

    print(f"\nTotal: {len(all_rows)} rows", flush=True)
    df = pd.DataFrame(all_rows)

    for c in EXIT_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df = df.dropna(subset=EXIT_FEATURE_COLS + ["target"])
    print(f"After NaN drop: {len(df)}", flush=True)
    print(f"  Hold(1): {(df['target']==1).sum()}  Exit(0): {(df['target']==0).sum()}", flush=True)

    train_mask = df["year"] < 2020
    X_train = df[EXIT_FEATURE_COLS][train_mask].values
    y_train = df["target"][train_mask].values
    X_val = df[EXIT_FEATURE_COLS][~train_mask].values
    y_val = df["target"][~train_mask].values
    print(f"Train: {len(X_train)}  Val: {len(X_val)}", flush=True)

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=0.1,
        eval_metric="logloss", early_stopping_rounds=30,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_val)[:, 1]
    print(f"\nVal AUC: {roc_auc_score(y_val, y_prob):.4f}", flush=True)

    imp = pd.DataFrame({"feature": EXIT_FEATURE_COLS,
                        "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    print(f"\nTop 10:\n{imp.head(10).to_string(index=False)}", flush=True)

    joblib.dump(model, MODEL_PATH)
    np.save(MODEL_PATH.replace(".joblib", "_features.npy"), EXIT_FEATURE_COLS)
    print(f"Saved to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
