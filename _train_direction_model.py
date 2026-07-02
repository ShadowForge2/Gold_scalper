"""
Train XGBoost direction model for XAUUSD M5.
Trains on 2007-2021, tests on each subsequent year individually.
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import (
    compute_features, create_target, prepare_dataset,
    train_model, FEATURE_COLS, DirectionPredictor,
)

TRAIN_END = 2021
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
HORIZON = 3
ATR_THRESHOLD = 0.3
MODEL_PATH = "models/direction_xgb_m5.joblib"

client = DukascopyClient()

def load_year_df(year: int) -> pd.DataFrame:
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    return m5, h1

# Load all training data
print("Loading training data (2007-2021)...")
t0 = time.time()
train_parts = []
h1_train_parts = []
for y in range(2007, TRAIN_END + 1):
    m5, h1 = load_year_df(y)
    train_parts.append(m5)
    h1_train_parts.append(h1)
    print(f"  {y}: {len(m5)} M5 bars, {len(h1)} H1 bars")
train_m5 = pd.concat(train_parts).sort_values("time").drop_duplicates(subset="time").set_index("time")
h1_train = pd.concat(h1_train_parts).sort_values("time").drop_duplicates(subset="time").set_index("time")
train_m5 = train_m5[~train_m5.index.duplicated(keep="first")]
h1_train = h1_train[~h1_train.index.duplicated(keep="first")]

print(f"Train M5: {len(train_m5)} bars, {time.time()-t0:.1f}s")

# Feature engineering for training
print("Engineering features (train)...")
t0 = time.time()
train_features = compute_features(train_m5, h1_train)
print(f"  {len(train_features)} rows, {len(train_features.columns)} features, {time.time()-t0:.1f}s")

print("Creating target...")
t0 = time.time()
train_target = create_target(train_m5, horizon=HORIZON, atr_threshold=ATR_THRESHOLD)
train_target.name = "target"
n_ups = (train_target == 1).sum()
n_downs = (train_target == 0).sum()
print(f"  UP: {n_ups}, DOWN: {n_downs}, Ratio: {n_ups/max(n_downs,1):.3f}, {time.time()-t0:.1f}s")

print("Preparing dataset...")
t0 = time.time()
X_train, y_train = prepare_dataset(train_features, train_target)
print(f"  {len(X_train)} rows, {len(X_train.columns)} features, {time.time()-t0:.1f}s")

# Train model
print("Training XGBoost...")
t0 = time.time()
# split train into train/val for early stopping
split_idx = int(len(X_train) * 0.85)
X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

model = train_model(X_tr, y_tr, X_val, y_val)
train_probs = model.predict_proba(X_train.values)
train_preds = model.predict(X_train.values)
train_acc = accuracy_score(y_train, train_preds)
print(f"  Train accuracy: {train_acc:.4f} ({len(X_train)} rows, {time.time()-t0:.1f}s)")

# Save model
os.makedirs("models", exist_ok=True)
predictor = DirectionPredictor(model=model)
predictor.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Test on each subsequent year
print("\n--- Out-of-Sample Test ---")
all_results = {}
for ty in TEST_YEARS:
    print(f"\n=== Testing on {ty} ===")
    t0 = time.time()
    test_m5, h1_test = load_year_df(ty)
    test_m5 = test_m5.set_index("time")
    h1_test = h1_test.set_index("time")
    test_m5 = test_m5[~test_m5.index.duplicated(keep="first")]

    test_features = compute_features(test_m5, h1_test)
    test_target = create_target(test_m5, horizon=HORIZON, atr_threshold=ATR_THRESHOLD)
    X_test, y_test = prepare_dataset(test_features, test_target)
    print(f"  {len(X_test)} test rows, {time.time()-t0:.1f}s")

    if len(X_test) == 0:
        print("  No test samples")
        continue

    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)
    y_conf = np.max(y_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    n_pos = int((y_pred == 1).sum())
    n_neg = int((y_pred == 0).sum())
    n_up_actual = int((y_test == 1).sum())
    n_down_actual = int((y_test == 0).sum())

    # High-confidence subset
    high_conf_mask = y_conf >= 0.60
    if high_conf_mask.sum() > 0:
        hc_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
        hc_pct = high_conf_mask.mean() * 100
    else:
        hc_acc = 0
        hc_pct = 0

    # Confusion matrix details
    tp = cm[1, 1] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    total_signals = n_pos + n_neg

    print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    print(f"  Pred UP={n_pos} DOWN={n_neg} (Actual UP={n_up_actual} DOWN={n_down_actual})")
    print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  High-conf (>=60%): {high_conf_mask.sum()}/{len(y_pred)} ({hc_pct:.1f}%) Acc={hc_acc:.4f}")
    print(f"  Total signals: {total_signals}")

    all_results[ty] = {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "n_pos": n_pos, "n_neg": n_neg,
        "n_up": n_up_actual, "n_down": n_down_actual,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "hc_acc": hc_acc, "hc_pct": hc_pct,
        "total": len(y_test),
    }

# Summary table
print("\n\n=== Summary ===")
print(f"{'Year':>6} {'Total':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'PredUP':>8} {'PredDN':>8} {'HC%':>8} {'HCAcc':>8}")
print("-" * 88)
for ty in TEST_YEARS:
    if ty in all_results:
        r = all_results[ty]
        print(f"{ty:>6} {r['total']:>8} {r['acc']:>8.4f} {r['prec']:>8.4f} {r['rec']:>8.4f} {r['f1']:>8.4f} {r['n_pos']:>8} {r['n_neg']:>8} {r['hc_pct']:>8.1f} {r['hc_acc']:>8.4f}")
    else:
        print(f"{ty:>6} {'no data':>8}")
