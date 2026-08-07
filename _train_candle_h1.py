"""
Train per-pair XGBoost H1 candle-following models.

Reuses the profit-based labeling approach from _train_candle_brain.py but
adapts the trade simulation to the H1 candle strategy (CANDLE_STRATEGY.md):
enter at candle commit (close), SL = SL_ATR*ATR, close on reversal
(price retraces REVERSAL_ATR*ATR past entry), trail once past open, cap at
MAX_HOLD_BARS, subtract COST_R. Labels: BUY/SELL/NONE (chop -> NONE).

XGBoost (chosen over LightGBM): same gradient-boosting family, but XGBoost
grows trees level-wise (safer on small data), LightGBM leaf-wise can overfit
~40k rows, and the project already has a proven XGBoost pipeline.

Walk-forward split per pair: train = older years, val = newest years.
Model selection uses simulated net R (profit, not accuracy) minus a
drawdown penalty, mirroring the candle-brain model-selection metric.

Usage:
  python _train_candle_h1.py --symbols XAUUSD,US100,JP225,DE40,US500,US30
  python _train_candle_h1.py --symbols XAGUSD,BRENT,WTI --start-year 2015
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from app.candle_engine import (
    FEATURE_COLS, compute_features, generate_labels, compute_atr,
)

ATR_PERIOD = 14
DD_PENALTY = 2.0
MIN_VAL_TRADES = 30
SEED = 42

SYMBOL_MAP = {
    "XAUUSD": "data/dukascopy",
    "XAGUSD": "data/dukascopy_xag",
    "XAG": "data/dukascopy_xag",
    "BRENT": "data/dukascopy_brent",
    "WTI": "data/dukascopy_wti",
    "GAS": "data/dukascopy_gas",
    "COPPER": "data/dukascopy_copper",
    "XPT": "data/dukascopy_xpt",
    "US100": "data/dukascopy_us100",
    "US500": "data/dukascopy_us500",
    "US30": "data/dukascopy_us30",
    "JP225": "data/dukascopy_jp225",
    "DE40": "data/dukascopy_de40",
}


def load_m1_data(symbol: str, start_year: int = None, end_year: int = None) -> pd.DataFrame:
    data_dir = SYMBOL_MAP.get(symbol)
    if data_dir is None:
        raise ValueError(f"Unknown symbol: {symbol}")
    path = Path(data_dir)
    frames = []
    prefix = symbol + "_"
    for f in sorted(path.glob(f"{prefix}*.parquet")):
        try:
            y = int(str(f.name).split("_")[-1].split(".")[0])
        except Exception:
            y = None
        if y is not None:
            if start_year is not None and y < start_year:
                continue
            if end_year is not None and y > end_year:
                continue
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "tick_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def resample_h1(m1: pd.DataFrame, tf_min: int = 60) -> pd.DataFrame:
    rule = f"{tf_min}min" if tf_min != 60 else "1h"
    idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
    h1 = idx.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum",
    }).dropna()
    return h1


def simulate_trades(pred_side: np.ndarray, label_idx: np.ndarray, edges: np.ndarray, conf: np.ndarray):
    """Replay the trades the model would take on validation (net R, DD, PF).

    `pred_side` is the model's predicted direction (0=BUY, 1=SELL) aligned
    with `label_idx`; each trade's realized R is edges[bar, side].
    """
    trades = []
    for k, i in enumerate(label_idx):
        trades.append(float(edges[i, int(pred_side[k])]))
    rs = np.asarray(trades)
    wins = float(rs[rs > 0].sum())
    losses = float((-rs[rs < 0]).sum())
    equity = np.cumsum(rs)
    peak = np.maximum.accumulate(equity)
    dd = float((peak - equity).max())
    return {
        "net_r": float(rs.sum()),
        "trades": int(len(rs)),
        "wr": 100.0 * float((rs > 0).mean()) if len(rs) else 0.0,
        "exp": float(rs.mean()) if len(rs) else 0.0,
        "pf": (wins / losses) if losses > 0 else float("inf"),
        "dd": dd,
    }


def train_symbol(
    symbol: str,
    start_year: int = None,
    end_year: int = None,
    train_frac: float = 0.70,
    conf_thresh: float = 0.0,
    tf_min: int = 60,
):
    import xgboost as xgb
    from sklearn.metrics import classification_report
    t0 = time.time()
    sl_r = float(getattr(cfg, "CANDLE_ENGINE_SL_ATR", 1.0))
    reversal_r = float(getattr(cfg, "CANDLE_ENGINE_REVERSAL_ATR", 0.5))
    trail_r = float(getattr(cfg, "CANDLE_ENGINE_TRAIL_ATR", 0.5))
    max_hold = int(getattr(cfg, "CANDLE_ENGINE_MAX_HOLD_BARS", 24))
    cost_r = float(getattr(cfg, "CANDLE_ENGINE_COST_R", 0.05))
    entry_min_r = float(getattr(cfg, "CANDLE_ENGINE_ENTRY_MIN_R", 0.35))
    edge_margin = float(getattr(cfg, "CANDLE_ENGINE_EDGE_MARGIN", 1.0))
    label_sl_r = float(getattr(cfg, "CANDLE_ENGINE_LABEL_SL_ATR", 1.0))

    # Labels are simulated with the TIGHT label stop (early-strength candles);
    # the live engine exits with the WIDER sl_r. Decoupling these was the key
    # to OOS PF 1.67 (coupling at 1.5R dropped it to 1.07).
    label_params = dict(
        sl_r=label_sl_r, reversal_r=reversal_r, trail_r=trail_r,
        max_hold=max_hold, cost_r=cost_r,
        entry_min_r=entry_min_r, edge_margin=edge_margin,
    )
    trade_params = dict(
        sl_r=sl_r, reversal_r=reversal_r, trail_r=trail_r,
        max_hold=max_hold, cost_r=cost_r,
        entry_min_r=entry_min_r, edge_margin=edge_margin,
    )

    print(f"\n=== {symbol} ===", flush=True)
    print(f"Loading M1 (years {start_year or 'start'}..{end_year or 'end'})...", flush=True)
    m1 = load_m1_data(symbol, start_year=start_year, end_year=end_year)
    print(f"  {len(m1):,} M1 bars", flush=True)

    h1 = resample_h1(m1, tf_min)
    print(f"  {len(h1):,} {tf_min}min bars", flush=True)
    if len(h1) < 2000:
        print(f"  SKIP: too few {tf_min}min bars", flush=True)
        return None

    feats = compute_features(h1)
    atr = compute_atr(h1, ATR_PERIOD)
    feats = generate_labels(feats, atr, **label_params)

    buy_count = int((feats["entry_label"] == 0).sum())
    sell_count = int((feats["entry_label"] == 1).sum())
    none_count = int((feats["entry_label"] == 2).sum())
    total = buy_count + sell_count + none_count
    print(f"  labels  BUY {buy_count} ({100*buy_count/total:.1f}%) "
          f"SELL {sell_count} ({100*sell_count/total:.1f}%) "
          f"NONE {none_count} ({100*none_count/total:.1f}%)", flush=True)

    X = feats[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
    Y = feats["entry_label"].values.astype(np.int64)
    edges = np.column_stack([feats["edge_long"].values, feats["edge_short"].values]).astype(np.float32)
    confs = feats["entry_conf"].values.astype(np.float32)

    n = len(X)
    # Skip warmup + bars whose future window is incomplete.
    valid = np.arange(max(60, ATR_PERIOD * 2), n - max_hold)

    # Time-aware split: train = older, val = newest (no leakage).
    split_i = int(len(valid) * train_frac)
    train_idx = valid[:split_i]
    val_idx = valid[split_i:]

    # Class-balanced subsample for training (keep NONE minority but enough).
    rng = np.random.default_rng(SEED)

    def balanced(idx, per_class):
        parts = []
        for cls in (0, 1, 2):
            sel = idx[Y[idx] == cls]
            if len(sel) > per_class:
                sel = rng.choice(sel, size=per_class, replace=False)
            parts.append(sel)
        out = np.concatenate(parts)
        out.sort()
        return out

    per_class = int(getattr(cfg, "CANDLE_ENGINE_TRAIN_PER_CLASS", 15000))
    tr_sel = balanced(train_idx, per_class)
    # Early-stopping val = newest 15% of the train window, RAW distribution
    # (not class-balanced). The verified OOS recipe (PF 1.67) uses this; a
    # balanced val over-represented NONE and starved the model of trades.
    va_sel = train_idx[int(0.85 * len(train_idx)):]

    print(f"  train {len(tr_sel):,} val {len(va_sel):,}", flush=True)

    # Class weights applied ONLY to the early-stopping eval set (weighted val
    # mlogloss rewards correct BUY/SELL more than NONE). Training is unweighted
    # softprob — upweighting trade classes in training made the model trade too
    # aggressively (PF 1.06). Profit comes from the label simulation.
    label_w = np.ones(len(Y), dtype=np.float32)
    label_w[Y == 0] = 1.0
    label_w[Y == 1] = 1.0
    label_w[Y == 2] = 0.5
    dtrain = xgb.DMatrix(X[tr_sel], label=Y[tr_sel])
    dval_w = xgb.DMatrix(X[va_sel], label=Y[va_sel], weight=label_w[va_sel])

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 50,
        "lambda": 2.0,
        "eval_metric": "mlogloss",
        "seed": SEED,
    }
    watchlist = [(dtrain, "train"), (dval_w, "val")]
    evals_result = {}

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=800,
        evals=watchlist,
        evals_result=evals_result,
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    best_it = booster.best_iteration if booster.best_iteration is not None else 800
    booster = xgb.train(
        params, dtrain, num_boost_round=best_it + 1,
        evals=watchlist, evals_result=evals_result, verbose_eval=False,
    )
    model = booster

    # Simulate what the model would actually trade on validation.
    proba = model.predict(xgb.DMatrix(X[va_sel]))
    pred = proba.argmax(axis=1)
    pconf = proba.max(axis=1)

    # conf_thresh=0 uses all raw calls (model-selection metric).
    use = pred != 2
    if conf_thresh > 0:
        use = use & (pconf >= conf_thresh)
    va_idx = va_sel[use]
    metrics = simulate_trades(pred[use], va_idx, edges, confs)
    best_metrics = metrics
    score = metrics["net_r"] - DD_PENALTY * metrics["dd"] if metrics["trades"] >= MIN_VAL_TRADES else -float("inf")

    print(
        f"  val ({time.time()-t0:.0f}s): trades={metrics['trades']} "
        f"WR={metrics['wr']:.1f}% exp={metrics['exp']:+.3f}R PF={metrics['pf']:.2f} "
        f"net={metrics['net_r']:+.1f}R dd={metrics['dd']:.1f}R score={score:+.1f}",
        flush=True,
    )

    if metrics["trades"] < MIN_VAL_TRADES:
        print("  SKIP: not enough val trades", flush=True)
        return None

    out_dir = Path(cfg.CANDLE_ENGINE_MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.joblib"
    joblib_data = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "trade_params": trade_params,
        "symbol": symbol,
        "tf_min": tf_min,
    }
    import joblib
    joblib.dump(joblib_data, str(out_path))
    print(f"  Saved: {out_path} (score {score:+.1f})", flush=True)

    # Report class accuracy for diagnostics.
    report = classification_report(Y[va_sel], pred, labels=[0, 1, 2],
                                   target_names=["BUY", "SELL", "NONE"],
                                   zero_division=0)
    print(report, flush=True)
    return score


def main():
    parser = argparse.ArgumentParser(description="Train H1 candle XGBoost per pair")
    parser.add_argument("--symbols", default=None,
                        help="comma list (default: all in SYMBOL_MAP with data)")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--tf", type=int, default=60,
                        help="bar size in minutes (60 = H1, 30 = 30m)")
    args = parser.parse_args()

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        syms = list(SYMBOL_MAP.keys())

    results = {}
    for sym in syms:
        try:
            results[sym] = train_symbol(
                sym, start_year=args.start_year, end_year=args.end_year,
                train_frac=args.train_frac, tf_min=args.tf,
            )
        except Exception as e:
            print(f"\n=== {sym} FAILED: {e} ===", flush=True)
            results[sym] = None

    print("\n==== SUMMARY ====", flush=True)
    for sym, score in results.items():
        print(f"  {sym}: {'OK ' + f'(score {score:+.1f})' if score is not None else 'FAIL/SKIP'}", flush=True)


if __name__ == "__main__":
    main()
