"""
Final Swing Quality XGBoost — ZigZag H4 labels, full 2007-2021 training.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features, ASP_FEATURE_COLS

client = DukascopyClient()
MODEL_PATH = os.path.join("models", "swing_quality_xgb.json")

TRAIN_YEARS = list(range(2007, 2022))
TEST_YEARS = [2022, 2023, 2024, 2025]


def load_year(year):
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    return (client.resample_to(m1, 5),
            client.resample_to(m1, 16385),
            client.resample_to(m1, 16408))


def zigzag(highs, lows, pct=0.5):
    n = len(highs)
    direction = 0
    last_high = highs[0]
    last_low = lows[0]
    last_turn_idx = 0
    turns = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if direction == 0:
            if highs[i] >= last_high * (1 + pct / 100):
                direction = 1; turns[last_turn_idx] = True; last_turn_idx = i
            elif lows[i] <= last_low * (1 - pct / 100):
                direction = -1; turns[last_turn_idx] = True; last_turn_idx = i
            last_high = max(last_high, highs[i]); last_low = min(last_low, lows[i])
        elif direction == 1:
            if highs[i] > last_high: last_high = highs[i]; last_turn_idx = i
            elif lows[i] <= last_high * (1 - pct / 100):
                direction = -1; turns[last_turn_idx] = True; last_turn_idx = i; last_low = lows[i]
        else:
            if lows[i] < last_low: last_low = lows[i]; last_turn_idx = i
            elif highs[i] >= last_low * (1 + pct / 100):
                direction = 1; turns[last_turn_idx] = True; last_turn_idx = i; last_high = highs[i]
    turns[last_turn_idx] = True
    return turns


def label_zigzag_h4(m5, h4):
    h4_turns = zigzag(h4["high"].values.astype(np.float64), h4["low"].values.astype(np.float64), 0.5)
    h4_ts = h4["time"].values.astype(np.int64) // 10**9
    m5_ts = m5["time"].values.astype(np.int64) // 10**9
    h4_idx = np.clip(np.searchsorted(h4_ts, m5_ts, side='right') - 1, 0, len(h4["time"]) - 1)
    return h4_turns[h4_idx]


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Final Swing Quality XGBoost — ZigZag H4 0.5%")
    print("=" * 60)
    sys.stdout.flush()

    print(f"\n[1] Loading data ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})...")
    all_m5, all_h1, all_h4 = [], [], []
    for y in TRAIN_YEARS:
        print(f"  {y}...", end=" ", flush=True)
        m5, h1, h4 = load_year(y)
        print(f"{len(m5)} bars {time.time()-t0:.0f}s")
        all_m5.append(m5); all_h1.append(h1); all_h4.append(h4)

    m5_all = pd.concat(all_m5, ignore_index=True)
    h1_all = pd.concat(all_h1, ignore_index=True)
    h4_all = pd.concat(all_h4, ignore_index=True)

    print(f"\n[2] Features + labels...")
    t_f = time.time()
    features = compute_asp_features(m5_all, h1_all)
    feat_arr = features.values.astype(np.float32)
    valid = ~(np.isnan(feat_arr).any(axis=1) | np.isinf(feat_arr).any(axis=1))
    print(f"  Features: {time.time()-t_f:.0f}s")

    t_l = time.time()
    labels = label_zigzag_h4(m5_all, h4_all)
    print(f"  Labels: {int(labels.sum())}/{len(labels)} ({labels.mean():.3f}) {time.time()-t_l:.1f}s")

    n = min(len(feat_arr), len(valid), len(labels))
    feat_arr, valid, labels = feat_arr[:n], valid[:n], labels[:n]
    vidx = np.where(valid)[0]

    split = int(len(vidx) * 0.85)
    X_tr, X_val = feat_arr[vidx[:split]], feat_arr[vidx[split:]]
    Y_tr, Y_val = labels[vidx[:split]], labels[vidx[split:]]
    print(f"  Train: {len(X_tr)} ({int(Y_tr.sum())} pos), Val: {len(X_val)} ({int(Y_val.sum())} pos)")

    print(f"\n[3] Training...")
    t_train = time.time()
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(len(Y_tr) - Y_tr.sum()) / max(Y_tr.sum(), 1),
        eval_metric="auc", early_stopping_rounds=30,
        tree_method="hist", random_state=42, verbosity=0,
    )
    model.fit(X_tr, Y_tr, eval_set=[(X_val, Y_val)], verbose=False)
    print(f"  Done: {model.best_iteration+1} trees, {time.time()-t_train:.0f}s")

    prob = model.predict_proba(X_val)[:, 1]
    for t in [0.3, 0.5, 0.6, 0.7]:
        sig = prob > t
        n_s = sig.sum()
        if n_s > 0:
            print(f"  Threshold {t}: {n_s} signals, accuracy={Y_val[sig].mean():.3f}")

    importances = model.feature_importances_
    top10 = [(ASP_FEATURE_COLS[i], importances[i]) for i in np.argsort(importances)[::-1][:10]]
    print(f"\n  Top 10 features:")
    for name, imp in top10:
        print(f"    {name:<25} {imp:.4f}")

    print(f"\n[4] OOS (2022-2025)...")
    for y in TEST_YEARS:
        print(f"  {y}...", end=" ", flush=True)
        m5, h1, h4 = load_year(y)
        feat = compute_asp_features(m5, h1)
        lab = label_zigzag_h4(m5, h4)
        fa = feat.values.astype(np.float32)
        v = ~(np.isnan(fa).any(axis=1) | np.isinf(fa).any(axis=1))
        n2 = min(len(fa), len(v), len(lab))
        fa, v, lab = fa[:n2], v[:n2], lab[:n2]
        vi = np.where(v)[0]
        X_t, Y_t = fa[vi], lab[vi]
        prob_t = model.predict_proba(X_t)[:, 1]
        sig = prob_t > 0.7
        n_s = sig.sum()
        acc = Y_t[sig].mean() if n_s > 0 else 0
        print(f"sigs={n_s:>5} acc={acc:.3f} base={Y_t.mean():.3f} ({n_s/len(Y_t)*100:.1f}%)")

    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
