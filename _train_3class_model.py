"""
Train 3-class XGBoost direction model for XAUUSD M5.
Classes: 0=DOWN, 1=UP, 2=NO_TRADE (|move| <= 0.3*ATR, previously dropped).
Includes 4 new chop features + 3 calendar features (hardcoded).
Trains on 2007-2021, tests on each subsequent year individually.
"""
import os, sys, time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta, date
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
MODEL_PATH = "models/direction_xgb_m5_3class.joblib"

client = DukascopyClient()

EVENT_IMPACT = 2  # all hardcoded events treated as HIGH

def _generate_all_events(start_date: date, end_date: date) -> list:
    """Pre-generate all hardcoded event datetimes between start and end dates."""
    events = []
    d = start_date
    while d <= end_date:
        y, m, day, wd, dom = d.year, d.month, d.day, d.weekday(), d.day
        if wd == 3:
            events.append((datetime(y, m, day, 12, 30, tzinfo=timezone.utc), EVENT_IMPACT))
        if wd == 4 and dom <= 7:
            events.append((datetime(y, m, day, 12, 30, tzinfo=timezone.utc), EVENT_IMPACT))
        if dom in (13, 14, 15) and wd <= 4:
            events.append((datetime(y, m, day, 12, 30, tzinfo=timezone.utc), EVENT_IMPACT))
        if dom in (12, 13, 14) and wd <= 4:
            events.append((datetime(y, m, day, 12, 30, tzinfo=timezone.utc), EVENT_IMPACT))
        if dom in (14, 15, 16) and wd <= 4:
            events.append((datetime(y, m, day, 12, 30, tzinfo=timezone.utc), EVENT_IMPACT))
        if dom in (1, 2, 3) and wd <= 4:
            events.append((datetime(y, m, day, 15, 0, tzinfo=timezone.utc), EVENT_IMPACT))
        if m in {1, 3, 5, 6, 8, 9, 11, 12} and wd == 2 and 15 <= dom <= 21:
            events.append((datetime(y, m, day, 18, 0, tzinfo=timezone.utc), EVENT_IMPACT))
        if m in (1, 4, 7, 10) and dom >= 25 and wd <= 4:
            events.append((datetime(y, m, day, 12, 30, tzinfo=timezone.utc), EVENT_IMPACT))
        d += timedelta(days=1)
    events.sort(key=lambda x: x[0])
    return events


def add_calendar_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Fully vectorized calendar features using np.searchsorted."""
    feats = features.copy()
    idx_series = feats.index
    start_d = idx_series.min().date() - timedelta(days=5)
    end_d = idx_series.max().date() + timedelta(days=32)
    event_list = _generate_all_events(start_d, end_d)
    ev_dts = np.array([e[0] for e in event_list], dtype="datetime64[us]")
    ev_imp = np.array([e[1] for e in event_list])
    n = len(idx_series)
    mins_until = np.full(n, 999.0, dtype=np.float64)
    mins_since = np.full(n, -999.0, dtype=np.float64)
    impacts = np.zeros(n, dtype=np.int32)
    # Convert index to datetime64[us] for searchsorted
    ts = idx_series.values.astype("datetime64[us]")
    p = np.searchsorted(ev_dts, ts, side="right")
    # Next event
    nxt_mask = p < len(ev_dts)
    if nxt_mask.any():
        pi = p[nxt_mask]
        diff = (ev_dts[pi] - ts[nxt_mask]).astype(np.float64) / 6e7  # us -> min
        mins_until[nxt_mask] = np.abs(diff)
        impacts[nxt_mask] = np.maximum(impacts[nxt_mask], ev_imp[pi])
    # Previous event
    prv_mask = p > 0
    if prv_mask.any():
        pi = p[prv_mask] - 1
        diff = (ts[prv_mask] - ev_dts[pi]).astype(np.float64) / 6e7  # us -> min
        mins_since[prv_mask] = np.abs(diff)
        impacts[prv_mask] = np.maximum(impacts[prv_mask], ev_imp[pi])
    feats["minutes_until_event"] = mins_until
    feats["minutes_since_event"] = mins_since
    feats["event_impact"] = impacts
    return feats


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
train_features = add_calendar_features(train_features, train_m5)
print(f"  {len(train_features)} rows, {len(train_features.columns)} features, {time.time()-t0:.1f}s")

print("Creating 3-class target...")
t0 = time.time()
train_target = create_target(train_m5, horizon=HORIZON, atr_threshold=ATR_THRESHOLD)
n_ups = int((train_target == 1).sum())
n_downs = int((train_target == 0).sum())
n_notrade = int((train_target == 2).sum())
print(f"  UP={n_ups} DOWN={n_downs} NO_TRADE={n_notrade} ({time.time()-t0:.1f}s)")

print("Preparing dataset...")
t0 = time.time()
X_train, y_train = prepare_dataset(train_features, train_target)
print(f"  {len(X_train)} rows, {len(X_train.columns)} features, {time.time()-t0:.1f}s")

# Train model
print("Training 3-class XGBoost...")
t0 = time.time()
split_idx = int(len(X_train) * 0.85)
X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

model = train_model(X_tr, y_tr, X_val, y_val)
train_preds = model.predict(X_train.values)
train_acc = accuracy_score(y_train, train_preds)

# Per-class metrics
cm_train = confusion_matrix(y_train, train_preds, labels=[0, 1, 2])
print(f"  Train accuracy: {train_acc:.4f} ({len(X_train)} rows)")
print(f"  Confusion matrix (0=DOWN, 1=UP, 2=NO_TRADE):")
print(f"    {cm_train}")
print(f"  Time: {time.time()-t0:.1f}s")

# Save model
os.makedirs("models", exist_ok=True)
predictor = DirectionPredictor(model=model)
predictor.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Test on each subsequent year
print("\n--- Out-of-Sample 3-Class Test ---")
all_results = {}
for ty in TEST_YEARS:
    print(f"\n=== Testing on {ty} ===")
    t0 = time.time()
    test_m5, h1_test = load_year_df(ty)
    test_m5 = test_m5.set_index("time")
    h1_test = h1_test.set_index("time")
    test_m5 = test_m5[~test_m5.index.duplicated(keep="first")]

    test_features = compute_features(test_m5, h1_test)
    test_features = add_calendar_features(test_features, test_m5)
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
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    # Per-class stats
    n_cls = {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum()), 2: int((y_test == 2).sum())}
    n_pred = {0: int((y_pred == 0).sum()), 1: int((y_pred == 1).sum()), 2: int((y_pred == 2).sum())}
    
    # Direction accuracy (UP/DOWN only, ignore NO_TRADE)
    dir_mask = y_test != 2
    if dir_mask.sum() > 0:
        dir_acc = accuracy_score(y_test[dir_mask], y_pred[dir_mask])
    else:
        dir_acc = 0.0

    # High-confidence subset (any class)
    hc_mask = y_conf >= 0.60
    if hc_mask.sum() > 0:
        hc_acc = accuracy_score(y_test[hc_mask], y_pred[hc_mask])
        hc_pct = hc_mask.mean() * 100
    else:
        hc_acc = 0.0
        hc_pct = 0.0

    # NO_TRADE accuracy when model predicts NO_TRADE
    no_trade_pred_mask = y_pred == 2
    if no_trade_pred_mask.sum() > 0:
        no_trade_acc = (y_test[no_trade_pred_mask] == 2).mean() * 100
    else:
        no_trade_acc = 0.0

    print(f"  Acc={acc:.4f} | DirAcc={dir_acc:.4f}")
    print(f"  Actual:   UP={n_cls[1]} DOWN={n_cls[0]} NO_TRADE={n_cls[2]}")
    print(f"  Predicted: UP={n_pred[1]} DOWN={n_pred[0]} NO_TRADE={n_pred[2]}")
    print(f"  Confusion matrix:")
    print(f"    {cm}")
    print(f"  High-conf (>=60%): {hc_mask.sum()}/{len(y_pred)} ({hc_pct:.1f}%) Acc={hc_acc:.4f}")
    print(f"  NO_TRADE accuracy (when predicted): {no_trade_acc:.1f}%")

    all_results[ty] = {
        "acc": acc, "dir_acc": dir_acc, "cm": cm,
        "n_act": n_cls, "n_pred": n_pred,
        "hc_acc": hc_acc, "hc_pct": hc_pct,
        "no_trade_acc": no_trade_acc,
        "total": len(y_test),
    }

# Summary
print("\n\n=== Summary ===")
print(f"{'Year':>6} {'Total':>8} {'Acc':>8} {'DirAcc':>8} {'PredUP':>8} {'PredDN':>8} {'PredNT':>8} {'HC%':>8} {'HCAcc':>8}")
print("-" * 80)
for ty in TEST_YEARS:
    if ty in all_results:
        r = all_results[ty]
        print(f"{ty:>6} {r['total']:>8} {r['acc']:>8.4f} {r['dir_acc']:>8.4f} "
              f"{r['n_pred'][1]:>8} {r['n_pred'][0]:>8} {r['n_pred'][2]:>8} "
              f"{r['hc_pct']:>7.1f}% {r['hc_acc']:>8.4f}")
    else:
        print(f"{ty:>6} {'no data':>8}")

print(f"\nModel saved with {len(FEATURE_COLS)} features")
