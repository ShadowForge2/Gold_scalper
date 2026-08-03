"""Train Candle ML — multi-timeframe M5 direction prediction.

Uses M1 data for all three timeframes. Fast vectorized feature computation.
"""
import os, sys, time, warnings, argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.candle_ml import (
    compute_candle_features, create_candle_target, CANDLE_FEATURE_COLS,
)

SYMBOLS = ["XAUUSD", "US100", "US500", "US30"]
TRAIN_START = 2022
TRAIN_SPLIT = 0.85


def load_year_m1(client, year):
    m1 = client.download_year(year)
    if m1 is None or len(m1) == 0:
        return None
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m1 = m1.set_index("time") if "time" in m1.columns else m1
    return m1


def train_for_symbol(cfg):
    symbol = cfg.get("symbol", "?")
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"  Candle ML — {symbol}")
    print(f"{'=' * 60}")

    client = DukascopyClient(symbol=symbol, cache_dir=cfg["cache_dir"])

    print(f"\n[1] Loading M1 data ({cfg['train_years'][0]}-{cfg['train_years'][-1]})...")
    all_m1 = []
    for y in cfg["train_years"]:
        print(f"  {y}...", end=" ", flush=True)
        m1 = load_year_m1(client, y)
        if m1 is not None:
            print(f"{len(m1)} bars {time.time()-t0:.0f}s")
            all_m1.append(m1)
        else:
            print("no data")

    if not all_m1:
        print("  No training data available")
        return

    m1_all = pd.concat(all_m1).sort_index()
    print(f"  Total M1 bars: {len(m1_all)}")

    print(f"\n[2] Computing features...")
    t_f = time.time()
    features = compute_candle_features(m1_all)
    print(f"  {len(features)} rows x {len(features.columns)} cols ({time.time()-t_f:.0f}s)")

    print(f"[3] Creating target...")
    target = create_candle_target(m1_all)

    print(f"[4] Aligning & cleaning...")
    feature_cols = [c for c in CANDLE_FEATURE_COLS if c in features.columns]
    X = features[feature_cols].copy()
    y = target.copy()

    # Align on common index
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    # Drop rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    print(f"  Samples: {len(X)}")
    counts = np.bincount(y.astype(int))
    for i, c in enumerate(counts):
        print(f"    {'DOWN' if i == 0 else 'UP'}: {c}")

    split = int(len(X) * TRAIN_SPLIT)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    print(f"  Train: {len(X_tr)}, Val: {len(X_val)}")

    print(f"\n[5] Training XGBoost...")
    t_train = time.time()
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(X_tr.values, y_tr.values, eval_set=[(X_val.values, y_val.values)], verbose=False)
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else "N/A"
    print(f"  Done: {best_iter} trees ({time.time()-t_train:.0f}s)")

    proba = model.predict_proba(X_val.values)
    pred = model.predict(X_val.values)
    acc = (pred == y_val.values).mean()
    print(f"  Val accuracy: {acc:.3f} (baseline: {max((y_val==0).mean(), (y_val==1).mean()):.3f})")

    classes = list(model.classes_)
    for label_name, label_val in [("DOWN", 0), ("UP", 1)]:
        mask_l = y_val.values == label_val
        if mask_l.sum() > 0:
            lab_acc = (pred[mask_l] == label_val).mean()
            print(f"    {label_name}: {mask_l.sum()} samples, acc {lab_acc:.3f}")

    prob_pos = proba[:, classes.index(1)] if 1 in classes else proba[:, 1]
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        confident = prob_pos >= thresh
        if confident.sum() > 0:
            conf_acc = (pred[confident] == y_val.values[confident]).mean()
            print(f"    Conf>={thresh:.2f}: {confident.sum()} samples, acc {conf_acc:.3f}")

    # Also check confident SELL predictions
    prob_neg = 1 - prob_pos
    for thresh in [0.55, 0.60, 0.65]:
        confident_sell = prob_neg >= thresh
        if confident_sell.sum() > 0:
            conf_acc_s = (pred[confident_sell] == y_val.values[confident_sell]).mean()
            print(f"    Sell conf>={thresh:.2f}: {confident_sell.sum()} samples, acc {conf_acc_s:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, cfg["model_path"])
    print(f"  Saved: {cfg['model_path']}")

    if cfg["test_years"]:
        print(f"\n[6] OOS ({cfg['test_years'][0]}-{cfg['test_years'][-1]})...")
    for y in cfg["test_years"]:
        print(f"  {y}...", end=" ", flush=True)
        try:
            m1 = load_year_m1(client, y)
            if m1 is None:
                print("no data")
                continue
            feat = compute_candle_features(m1)
            tgt = create_candle_target(m1)
            X_t = feat[feature_cols].copy()
            y_t = tgt.copy()
            common_t = X_t.index.intersection(y_t.index)
            X_t = X_t.loc[common_t]
            y_t = y_t.loc[common_t]
            mask_t = ~(X_t.isna().any(axis=1) | y_t.isna())
            X_t = X_t[mask_t]
            y_t = y_t[mask_t]
            if len(X_t) == 0:
                print("no valid samples")
                continue
            prob_t = model.predict_proba(X_t.values)
            pred_t = model.predict(X_t.values)
            acc_t = (pred_t == y_t.values).mean()
            n_up = (pred_t == 1).sum()
            n_down = (pred_t == 0).sum()
            prob_pos_t = prob_t[:, classes.index(1)] if 1 in classes else prob_t[:, 1]
            n_conf = (prob_pos_t >= 0.65).sum()
            print(f"acc={acc_t:.3f} up={n_up} down={n_down} (conf>=0.65: {n_conf})")
        except Exception as e:
            print(f"error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Total for {symbol}: {time.time()-t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Train Candle ML models")
    parser.add_argument("--year", type=int, default=datetime.now(timezone.utc).year, help="Training end year (default: current)")
    parser.add_argument("--symbol", type=str, default=None, help="Train only this symbol (XAUUSD|US100|US500|US30). Default: all")
    args = parser.parse_args()

    train_years = list(range(TRAIN_START, args.year + 1))
    test_years = [] if args.year >= TRAIN_START else [args.year]
    if len(train_years) < 2:
        print(f"ERROR: Need at least 2 years of training data (got {train_years})")
        return

    SYMBOL_CONFIG = {
        "XAUUSD": {
            "symbol": "XAUUSD",
            "cache_dir": "data/dukascopy",
            "model_path": "models/candle_xgb_m5_XAUUSD.joblib",
            "train_years": train_years,
            "test_years": test_years,
        },
        "US100": {
            "symbol": "US100",
            "cache_dir": "data/dukascopy_us100",
            "model_path": "models/candle_xgb_m5_US100.joblib",
            "train_years": train_years,
            "test_years": test_years,
        },
        "US500": {
            "symbol": "US500",
            "cache_dir": "data/dukascopy_us500",
            "model_path": "models/candle_xgb_m5_US500.joblib",
            "train_years": train_years,
            "test_years": test_years,
        },
        "US30": {
            "symbol": "US30",
            "cache_dir": "data/dukascopy_us30",
            "model_path": "models/candle_xgb_m5_US30.joblib",
            "train_years": train_years,
            "test_years": test_years,
        },
    }

    symbols = [args.symbol.upper()] if args.symbol else list(SYMBOL_CONFIG)
    for symbol in symbols:
        if symbol not in SYMBOL_CONFIG:
            print(f"ERROR: unknown symbol {symbol} (choose {list(SYMBOL_CONFIG)})")
            return
        train_for_symbol(SYMBOL_CONFIG[symbol])
    print(f"\n{'=' * 60}")
    print(f"  All Candle ML models trained successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
