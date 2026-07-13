"""
Train ASP (Adaptive Swing Probability) XGBoost model.

Pipeline:
  1. Load M5+H1 data via DukascopyClient (2007-2021 train, 2022-2026 OOS)
  2. Compute ASP features (56 features, 7 systems)
  3. Create swing labels (BUY/SELL/NO_TRADE)
  4. Train 3-class XGBoost (DOWN=-1, UP=1, NEUTRAL=0)
  5. Evaluate OOS per year
  6. Save model + feature list
"""
import os, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features, ASP_FEATURE_COLS
from app.asp_labels import create_asp_labels

# Config
TRAIN_END = 2021
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
FORWARD_BARS = 12       # look 12 M5 bars ahead (1 hr)
MIN_TARGET_ATR = 1.0    # target = 1x ATR profit
SL_ATR_MULT = 2.0       # SL = 2x ATR
SWING_LOOKBACK = 5      # fractal detection window
MODEL_PATH = "models/asp_swing_xgb_m5.joblib"
FEATURE_PATH = "models/asp_swing_m5_features.npy"

client = DukascopyClient()


def load_year_df(year):
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    return m5, h1


def main():
    print("=" * 60)
    print("  ASP Swing Probability Model — Training")
    print("=" * 60)

    # ── Load training data ──
    print(f"\nLoading training data (2007-{TRAIN_END})...")
    t0 = time.time()
    all_rows = []

    for y in range(2007, TRAIN_END + 1):
        t_y = time.time()
        print(f"  Processing {y}...", end=" ", flush=True)
        try:
            m5, h1 = load_year_df(y)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if len(m5) == 0 or len(h1) == 0:
            print("no data")
            continue

        m5 = m5.set_index("time")
        h1 = h1.set_index("time")
        m5 = m5[~m5.index.duplicated(keep="first")]
        h1 = h1[~h1.index.duplicated(keep="first")]

        # Compute features
        features = compute_asp_features(m5, h1)
        if features is None or len(features) == 0:
            print("no features")
            continue

        # Create labels
        labels = create_asp_labels(m5, forward_bars=FORWARD_BARS,
                                   min_target_atr=MIN_TARGET_ATR,
                                   sl_atr_mult=SL_ATR_MULT,
                                   swing_lookback=SWING_LOOKBACK)

        # Align
        data = features.copy()
        data["target"] = labels
        data["year"] = y
        data = data.dropna(subset=ASP_FEATURE_COLS + ["target"])

        n_buy = (data["target"] == 1).sum()
        n_sell = (data["target"] == -1).sum()
        n_neutral = (data["target"] == 0).sum()
        print(f"{len(data)} rows, BUY={n_buy}, SELL={n_sell}, NEUTRAL={n_neutral} ({time.time()-t_y:.1f}s)")

        if len(data) > 0:
            all_rows.append(data)

    if not all_rows:
        print("No training data!")
        return

    df = pd.concat(all_rows, ignore_index=True)
    print(f"\nTotal training data: {len(df)} rows ({time.time()-t0:.1f}s)")

    # Class distribution
    print(f"\n  Class distribution:")
    for cls, label in [(1, "BUY"), (-1, "SELL"), (0, "NEUTRAL")]:
        cnt = (df["target"] == cls).sum()
        print(f"    {label:8s}: {cnt:8d} ({cnt/len(df)*100:.1f}%)")

    # ── Train/test split ──
    train_mask = df["year"] < 2022
    X_train = df[ASP_FEATURE_COLS][train_mask].values
    y_train = df["target"][train_mask].values
    print(f"\n  Train set: {len(X_train)} rows")

    # Train/val split (85/15)
    split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # ── Train XGBoost ──
    print("\nTraining XGBoost (3-class: SELL=-1, NEUTRAL=0, BUY=1)...")
    t0 = time.time()

    # Map labels: -1→0, 0→1, 1→2 (XGBoost needs 0-based classes)
    label_map = {-1: 0, 0: 1, 1: 2}
    y_tr_mapped = np.array([label_map[y] for y in y_tr])
    y_val_mapped = np.array([label_map[y] for y in y_val])

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    # Compute class weights for balancing
    class_counts = np.bincount(y_tr_mapped)
    total = len(y_tr_mapped)
    class_weights = {i: total / (3 * cnt) for i, cnt in enumerate(class_counts) if cnt > 0}
    sample_weights = np.array([class_weights[y] for y in y_tr_mapped])
    print(f"  Class weights: {class_counts} -> {[f'{w:.2f}' for w in [class_weights.get(i,0) for i in range(3)]]}")

    model.fit(X_tr, y_tr_mapped, eval_set=[(X_val, y_val_mapped)],
              sample_weight=sample_weights, verbose=True)
    print(f"  Trained in {time.time()-t0:.1f}s")

    inv_map = {v: k for k, v in label_map.items()}

    # Train accuracy
    train_preds_mapped = model.predict(X_train)
    train_preds = np.array([inv_map[p] for p in train_preds_mapped])
    train_acc = accuracy_score(y_train, train_preds)
    print(f"  Train accuracy: {train_acc:.4f}")

    # Save model + label mapping
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "label_map": label_map}, MODEL_PATH)
    np.save(FEATURE_PATH, np.array(ASP_FEATURE_COLS))
    print(f"\n  Model saved: {MODEL_PATH}")
    print(f"  Features saved: {FEATURE_PATH}")

    # Feature importance
    imp = pd.DataFrame({"feature": ASP_FEATURE_COLS,
                         "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    print(f"\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    # ── Out-of-Sample Test ──
    print("\n" + "=" * 60)
    print("  OUT-OF-SAMPLE TEST")
    print("=" * 60)
    print(f"{'Year':>6} {'Total':>8} {'Acc':>8} {'BUY%':>8} {'SELL%':>8} {'NEUT%':>8}")
    print("-" * 56)

    for ty in TEST_YEARS:
        t_y = time.time()
        print(f"\n=== Testing {ty} ===", flush=True)
        try:
            test_m5, test_h1 = load_year_df(ty)
            if len(test_m5) == 0 or len(test_h1) == 0:
                print(f"  {ty}: no data")
                continue
            test_m5 = test_m5.set_index("time")
            test_h1 = test_h1.set_index("time")
            test_m5 = test_m5[~test_m5.index.duplicated(keep="first")]
            test_h1 = test_h1[~test_h1.index.duplicated(keep="first")]

            features = compute_asp_features(test_m5, test_h1)
            labels = create_asp_labels(test_m5, forward_bars=FORWARD_BARS,
                                       min_target_atr=MIN_TARGET_ATR,
                                       sl_atr_mult=SL_ATR_MULT,
                                       swing_lookback=SWING_LOOKBACK)

            data = features.copy()
            data["target"] = labels
            data = data.dropna(subset=ASP_FEATURE_COLS + ["target"])

            if len(data) == 0:
                print(f"  {ty}: no rows")
                continue

            X_t = data[ASP_FEATURE_COLS].values
            y_t = data["target"].values

            y_pred_mapped = model.predict(X_t)
            y_pred = np.array([inv_map[p] for p in y_pred_mapped])

            acc = accuracy_score(y_t, y_pred)
            n_buy = (y_pred == 1).sum() / len(y_pred) * 100
            n_sell = (y_pred == -1).sum() / len(y_pred) * 100
            n_neut = (y_pred == 0).sum() / len(y_pred) * 100
            print(f"  {ty:>6} {len(y_t):>8} {acc:>8.4f} {n_buy:>7.1f}% {n_sell:>7.1f}% {n_neut:>7.1f}% ({time.time()-t_y:.1f}s)")

            # Detailed report for test years
            print(f"\n  Classification report ({ty}):")
            print(classification_report(y_t, y_pred, target_names=["SELL", "NEUTRAL", "BUY"], zero_division=0))

        except Exception as e:
            print(f"  {ty}: ERROR {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
