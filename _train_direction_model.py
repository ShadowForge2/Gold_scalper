import os, sys, time, warnings, argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import (
    compute_features, compute_chop_score, create_target,
    prepare_dataset, train_model, FEATURE_COLS,
)

SYMBOLS = ["XAUUSD", "US100"]
TRAIN_START = 2022
TRAIN_SPLIT = 0.85
CHOP_PCT = 0.33
HORIZON = 3
ATR_THRESHOLD = 0.3


def load_year(client, year):
    m1 = client.download_year(year)
    if m1 is None or len(m1) == 0:
        return None, None
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    if len(m5) == 0:
        return None, None
    return m5, h1


def train_for_symbol(cfg):
    symbol = cfg.get("symbol", "?")
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"  Direction Predictor Training — {symbol}")
    print(f"{'=' * 60}")

    client = DukascopyClient(symbol=symbol, cache_dir=cfg["cache_dir"])

    print(f"\n[1] Loading training data ({cfg['train_years'][0]}-{cfg['train_years'][-1]})...")
    all_m5, all_h1 = [], []
    for y in cfg["train_years"]:
        print(f"  {y}...", end=" ", flush=True)
        m5, h1 = load_year(client, y)
        if m5 is not None:
            print(f"{len(m5)} bars {time.time()-t0:.0f}s")
            all_m5.append(m5)
            all_h1.append(h1)
        else:
            print("no data")

    if not all_m5:
        print("  No training data available")
        return

    m5_all = pd.concat(all_m5, ignore_index=True)
    h1_all = pd.concat(all_h1, ignore_index=True)

    print(f"\n[2] Computing features...")
    t_f = time.time()
    features = compute_features(m5_all, h1_all)
    print(f"  Features: {len(features)} rows x {len(features.columns)} cols ({time.time()-t_f:.0f}s)")

    print(f"[3] Computing chop score...")
    chop = compute_chop_score(m5_all)

    print(f"[4] Creating target labels (horizon={HORIZON}, atr_thresh={ATR_THRESHOLD})...")
    target = create_target(m5_all, horizon=HORIZON, atr_threshold=ATR_THRESHOLD, chop_score=chop, chop_pct=CHOP_PCT)

    print(f"[5] Preparing dataset...")
    X, y = prepare_dataset(features, target)
    print(f"  Samples: {len(X)}")
    counts = np.bincount(y.astype(int))
    for i, c in enumerate(counts):
        labels = {0: "DOWN", 1: "UP", 2: "NO_TRADE"}
        print(f"    {labels.get(i, i)}: {c}")

    split = int(len(X) * TRAIN_SPLIT)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    print(f"  Train: {len(X_tr)}, Val: {len(X_val)}")

    print(f"\n[6] Training XGBoost...")
    t_train = time.time()
    n_classes = len(np.unique(y_tr))
    model = train_model(X_tr, y_tr, X_val, y_val)
    print(f"  Done: {model.best_iteration + 1} trees ({time.time()-t_train:.0f}s)")

    proba = model.predict_proba(X_val)
    classes = list(model.classes_)
    pred = model.predict(X_val)
    acc = (pred == y_val.values).mean()
    print(f"  Val accuracy: {acc:.3f}")

    for label_name, label_val in [("DOWN", 0), ("UP", 1), ("NO_TRADE", 2)]:
        mask = y_val.values == label_val
        if mask.sum() > 0:
            lab_acc = (pred[mask] == label_val).mean()
            print(f"    {label_name}: {mask.sum()} samples, accuracy {lab_acc:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, cfg["model_path"])
    print(f"  Saved: {cfg['model_path']}")

    if cfg["test_years"]:
        print(f"\n[7] OOS ({cfg['test_years'][0]}-{cfg['test_years'][-1]})...")
    for y in cfg["test_years"]:
        print(f"  {y}...", end=" ", flush=True)
        try:
            m5, h1 = load_year(client, y)
            if m5 is None:
                print("no data")
                continue
            feat = compute_features(m5, h1)
            ch = compute_chop_score(m5)
            tgt = create_target(m5, horizon=HORIZON, atr_threshold=ATR_THRESHOLD, chop_score=ch, chop_pct=CHOP_PCT)
            X_t, y_t = prepare_dataset(feat, tgt)
            if len(X_t) == 0:
                print("no valid samples")
                continue
            prob_t = model.predict_proba(X_t)
            pred_t = model.predict(X_t)
            acc_t = (pred_t == y_t.values).mean()
            n_up = (pred_t == 1).sum()
            n_down = (pred_t == 0).sum()
            n_nt = (pred_t == 2).sum()
            conf_up = prob_t[classes.index(1)].mean() if 1 in classes else 0
            conf_down = prob_t[classes.index(0)].mean() if 0 in classes else 0
            print(f"acc={acc_t:.3f} up={n_up} down={n_down} nt={n_nt} (conf_up={conf_up:.3f} conf_down={conf_down:.3f})")
        except Exception as e:
            print(f"error: {e}")

    print(f"\n  Total for {symbol}: {time.time()-t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Train Direction Predictor models")
    parser.add_argument("--year", type=int, default=datetime.now(timezone.utc).year, help="Training end year (default: current)")
    args = parser.parse_args()

    train_years = list(range(TRAIN_START, args.year + 1))
    test_years = [] if args.year >= TRAIN_START else [args.year]
    if len(train_years) < 2:
        print(f"ERROR: Need at least 2 years of training data (got {train_years})")
        return

    # Build per-symbol config dynamically
    SYMBOL_CONFIG = {
        "XAUUSD": {
            "symbol": "XAUUSD",
            "cache_dir": "data/dukascopy",
            "model_path": "models/direction_xgb_m5_XAUUSD.joblib",
            "train_years": train_years,
            "test_years": test_years,
        },
        "US100": {
            "symbol": "US100",
            "cache_dir": "data/dukascopy_us100",
            "model_path": "models/direction_xgb_m5_US100.joblib",
            "train_years": train_years,
            "test_years": test_years,
        },
    }

    for symbol in SYMBOLS:
        train_for_symbol(SYMBOL_CONFIG[symbol])
    print(f"\n{'=' * 60}")
    print(f"  All models trained successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
