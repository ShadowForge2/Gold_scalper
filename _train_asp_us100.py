"""Retrain ASP direction model for US100 (XGBoost)."""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features, ASP_FEATURE_COLS

SYMBOL = "US100"
CACHE_DIR = "data/dukascopy_us100"
client = DukascopyClient(symbol=SYMBOL, cache_dir=CACHE_DIR)
MODEL_PATH = "models/asp_swing_xgb_m5_US100.joblib"
FEATURE_PATH = "models/asp_swing_m5_features_US100.npy"

TRAIN_YEARS = list(range(2020, 2023))
TEST_YEARS = [2023, 2024, 2025]
FWD_BARS = 3
ATR_MULT = 0.3


def load_year(year):
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    return (client.resample_to(m1, 5),
            client.resample_to(m1, 16385))


def make_labels(m5):
    """3-bar forward return: BUY if > +ATR*0.3, SELL if < -ATR*0.3, else NEUTRAL."""
    closes = m5["close"].values.astype(np.float64)
    highs = m5["high"].values.astype(np.float64)
    lows = m5["low"].values.astype(np.float64)
    n = len(closes)

    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=2).mean().values

    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - FWD_BARS):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        future_ret = closes[i + FWD_BARS] - closes[i]
        threshold = atr[i] * ATR_MULT
        if future_ret > threshold:
            labels[i] = 1   # BUY
        elif future_ret < -threshold:
            labels[i] = -1  # SELL
    return labels


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"  ASP Direction Model Training — {SYMBOL} (XGBoost)")
    print("=" * 60)

    print(f"\n[1] Loading training data ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})...")
    all_m5, all_h1 = [], []
    for y in TRAIN_YEARS:
        print(f"  {y}...", end=" ", flush=True)
        m5, h1 = load_year(y)
        print(f"{len(m5)} bars {time.time()-t0:.0f}s")
        all_m5.append(m5); all_h1.append(h1)

    m5_all = pd.concat(all_m5, ignore_index=True)
    h1_all = pd.concat(all_h1, ignore_index=True)

    print(f"\n[2] Features + labels...")
    t_f = time.time()
    features = compute_asp_features(m5_all, h1_all)
    print(f"  Features: {time.time()-t_f:.0f}s")

    labels = make_labels(m5_all)
    valid_mask = ~(np.isnan(features.values).any(axis=1) | np.isinf(features.values).any(axis=1))
    valid_mask &= labels != 0
    print(f"  Labels: {(labels==1).sum()} BUY, {(labels==-1).sum()} SELL, {(labels==0).sum()} NEUTRAL")
    print(f"  Valid: {valid_mask.sum()}/{len(labels)}")

    vidx = np.where(valid_mask)[0]
    split = int(len(vidx) * 0.85)
    X_tr = features.iloc[vidx[:split]][ASP_FEATURE_COLS].values.astype(np.float32)
    Y_tr = labels[vidx[:split]]
    X_val = features.iloc[vidx[split:]][ASP_FEATURE_COLS].values.astype(np.float32)
    Y_val = labels[vidx[split:]]
    print(f"  Train: {len(X_tr)}, Val: {len(X_val)}")

    print(f"\n[3] Training...")
    t_train = time.time()
    y_tr_enc = np.where(Y_tr == -1, 0, 1)
    y_val_enc = np.where(Y_val == -1, 0, 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", early_stopping_rounds=30,
        tree_method="hist", random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr_enc, eval_set=[(X_val, y_val_enc)], verbose=False)
    print(f"  Done: {model.best_iteration+1} trees, {time.time()-t_train:.0f}s")

    pred_val_enc = model.predict(X_val)
    pred_val = np.where(pred_val_enc == 0, -1, 1)
    acc = accuracy_score(Y_val, pred_val)
    base = max((Y_val==1).sum(), (Y_val==-1).sum(), (Y_val==0).sum()) / max(len(Y_val), 1)
    print(f"  Val accuracy: {acc:.3f} (baseline: {base:.3f})")

    label_map = {0: -1, 1: 1}
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "label_map": label_map}, MODEL_PATH)
    np.save(FEATURE_PATH, np.array(ASP_FEATURE_COLS))
    print(f"  Saved: {MODEL_PATH}")

    print(f"\n[4] OOS ({TEST_YEARS[0]}-{TEST_YEARS[-1]})...")
    for y in TEST_YEARS:
        print(f"  {y}...", end=" ", flush=True)
        try:
            m5, h1 = load_year(y)
            feat = compute_asp_features(m5, h1)
            lab = make_labels(m5)
            valid = ~(np.isnan(feat.values).any(axis=1) | np.isinf(feat.values).any(axis=1))
            valid &= lab != 0
            vi = np.where(valid)[0]
            X_t = feat.iloc[vi][ASP_FEATURE_COLS].values.astype(np.float32)
            Y_t = lab[vi]
            pred_enc = model.predict(X_t)
            pred = np.where(pred_enc == 0, -1, 1)
            acc_t = accuracy_score(Y_t, pred)
            n_buy = (pred == 1).sum()
            n_sell = (pred == -1).sum()
            print(f"acc={acc_t:.3f} buy={n_buy} sell={n_sell}")
        except Exception as e:
            print(f"error: {e}")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
