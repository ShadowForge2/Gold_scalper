"""
Evaluate CandleBrain on REAL simulated trading (profit, not win rate).

Runs the trained model over an out-of-sample period and simulates the exact
trades it would take (SL = 1R, TP = 2R, 2h cap, cost subtracted). Reports:

  - trades, win rate, expectancy (R/trade), profit factor, net R, max drawdown
  - threshold sweep (confidence)
  - monthly R consistency  <-- the "consistent profit" proof
  - per-regime breakdown   (trend up / trend down / range)

Usage:
  python _eval_candle_brain.py --symbol XAUUSD --limit-files 2
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cap BLAS/OMP threads BEFORE torch imports so a tiny transformer on a
# low-RAM machine doesn't spawn a full thread pool and thrash to swap.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import _train_candle_brain as t
from app.candle_brain import CandleBrainTransformer, FEATURE_COLS, N_FEATURES, SEQ_LEN

SYMBOL_MAP = {
    "XAUUSD": "data/dukascopy",
    "US100": "data/dukascopy_us100",
    "US500": "data/dukascopy_us500",
    "US30": "data/dukascopy_us30",
}


def _load_last(symbol: str, limit_files: int) -> pd.DataFrame:
    data_dir = SYMBOL_MAP[symbol]
    prefix = symbol + "_"
    files = sorted(Path(data_dir).glob(f"{prefix}*.parquet"))
    frames = []
    for f in files[-limit_files:]:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True).sort_values("time")
    return df


def simulate(rs: np.ndarray, times: np.ndarray) -> dict:
    n = len(rs)
    if n == 0:
        return {"trades": 0, "wr": 0, "exp": 0, "pf": 0, "net_r": 0, "dd": 0,
                "avg_win": 0, "avg_loss": 0, "max_win": 0, "max_loss": 0}
    net = float(rs.sum())
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    equity = np.cumsum(rs)
    peak = np.maximum.accumulate(equity)
    dd = float((peak - equity).max())
    return {
        "trades": n,
        "wr": 100.0 * float((rs > 0).mean()),
        "exp": net / n,
        "pf": (wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        "net_r": net,
        "dd": dd,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "max_win": float(wins.max()) if len(wins) else 0.0,
        "max_loss": float(losses.min()) if len(losses) else 0.0,
    }


def main(symbol: str, limit_files: int = 2):
    print("Loading torch (takes ~1-2 min on this machine)...", flush=True)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)

    model_path = Path("models") / f"candle_brain_{symbol}.pt"
    if not model_path.exists():
        print(f"No model at {model_path} — train first")
        return

    print(f"Loading {symbol} data (last {limit_files} file(s), out-of-sample)...", flush=True)
    m1 = _load_last(symbol, limit_files)
    print(f"  {len(m1):,} M1 bars loaded", flush=True)
    m5 = t.resample_m5(m1)
    print("Computing features (vectorized, fast)...", flush=True)
    m5 = t.compute_features(m5)
    m5 = t.add_h1_context(m5, m1)
    m5 = t.add_swing_features(m5)
    m5 = t.add_time_features(m5)
    del m1
    gc.collect()
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    print("Simulating labels (SL=1R TP=2R)...", flush=True)
    m5 = t.generate_labels(m5, atr)
    print(f"  {len(m5):,} M5 bars "
          f"({m5.index.min()} .. {m5.index.max()})", flush=True)

    missing = [c for c in FEATURE_COLS if c not in m5.columns]
    for c in missing:
        m5[c] = 0.0
    m5 = m5.dropna(subset=["entry_label"])

    features = m5[FEATURE_COLS].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0)
    edges = np.column_stack([
        m5["edge_long"].values, m5["edge_short"].values
    ]).astype(np.float32)
    n = len(features)

    valid = np.zeros(n, dtype=bool)
    valid[SEQ_LEN:n - t.LABEL_WINDOW] = True
    idx = np.where(valid)[0]

    model = CandleBrainTransformer()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print("Running inference over all bars...", flush=True)
    probs, confs, mgmts = [], [], []
    for s in range(0, len(idx), 20_000):
        e = min(s + 20_000, len(idx))
        sel = idx[s:e]
        X = np.empty((len(sel), SEQ_LEN, N_FEATURES), dtype=np.float32)
        for j, i in enumerate(sel):
            X[j] = features[i-SEQ_LEN:i]
        with torch.no_grad():
            entry_logits, conf, mgmt_logits = model(torch.from_numpy(X))
            p = F.softmax(entry_logits, dim=-1).numpy()
            probs.append(p)
            confs.append(conf.squeeze(-1).numpy())
            mgmts.append(F.softmax(mgmt_logits, dim=-1).numpy())
        if (s // 20_000) % 5 == 4:
            print(f"  {e:,}/{len(idx):,} bars done...", flush=True)
    probs = np.concatenate(probs)
    confs = np.concatenate(confs)
    mgmts = np.concatenate(mgmts)
    pred = probs.argmax(axis=1)
    pconf = probs.max(axis=1)

    times = m5.index.values[idx]
    r_up = m5["regime_trend"].values[idx]
    label = m5["entry_label"].values[idx]
    times_dt = pd.to_datetime(times)

    # ── THE STRATEGY OBJECTIVE: raw directional calls, no confidence filter ──
    # Training optimises realised R of these calls (profit, not accuracy).
    # Confidence-gated thresholds below are a LIVE filter on top of this.
    mask_raw = pred != 2
    rs_raw = edges[idx][mask_raw, pred[mask_raw]]
    m_raw = simulate(rs_raw, times[mask_raw])

    print("\n-- RAW directional calls (training objective: edge in R) --")
    print(f"  Trades:        {m_raw['trades']}")
    print(f"  Win rate:      {m_raw['wr']:.1f}%")
    print(f"  Expectancy:    {m_raw['exp']:+.3f} R per trade")
    print(f"  Profit factor: {m_raw['pf']:.2f}")
    print(f"  Net result:    {m_raw['net_r']:+.1f} R")
    print(f"  Max drawdown:  {m_raw['dd']:.1f} R")

    monthly_raw = {}
    for i in range(len(rs_raw)):
        key = times_dt[mask_raw][i].strftime("%Y-%m")
        monthly_raw.setdefault(key, []).append(rs_raw[i])
    if monthly_raw:
        print(f"\n  -- Monthly R (consistency check, raw calls) --")
        pos_months = 0
        for key in sorted(monthly_raw):
            r = np.array(monthly_raw[key])
            net = float(r.sum())
            if net > 0:
                pos_months += 1
            print(f"  {key:>8} | {len(r):>6} trades | WR={100.0*(r>0).mean():4.1f}% | net {net:+7.1f}R")
        print(f"  Positive months: {pos_months}/{len(monthly_raw)} "
              f"({100.0*pos_months/max(len(monthly_raw),1):.0f}%)")

    print("\n-- Threshold sweep (simulated trading: SL=1R TP=2R 2h cap) --")
    print(f"{'thr':>5} | {'calls':>6} | {'WR':>5} | {'exp R':>6} | "
          f"{'PF':>5} | {'net R':>7} | {'maxDD R':>7}")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = (pred != 2) & (pconf >= thr)
        rs = edges[idx][mask, pred[mask]]
        m = simulate(rs, times[mask])
        if m["trades"] == 0:
            print(f"{thr:.2f} | {0:>6} | {'-':>5} | {'-':>6} | {'-':>5} | {'-':>7} | {'-':>7}")
            continue
        print(f"{thr:.2f} | {m['trades']:>6} | {m['wr']:4.1f}% | {m['exp']:+.3f} | "
              f"{m['pf']:4.2f} | {m['net_r']:+7.1f} | {m['dd']:6.1f}")

    thr = float(getattr(__import__("config"), "CANDLE_BRAIN_ENTRY_THRESHOLD", 0.60))
    mask = (pred != 2) & (pconf >= thr)
    rs = edges[idx][mask, pred[mask]]
    m = simulate(rs, times[mask])

    print(f"\n-- Summary @ conf >= {thr:.2f} --")
    if m["trades"]:
        print(f"  Trades:        {m['trades']}")
        print(f"  Win rate:      {m['wr']:.1f}%")
        print(f"  Expectancy:    {m['exp']:+.3f} R per trade")
        print(f"  Profit factor: {m['pf']:.2f}")
        print(f"  Net result:    {m['net_r']:+.1f} R")
        print(f"  Max drawdown:  {m['dd']:.1f} R")
        print(f"  Avg win / loss:{m['avg_win']:+.2f}R / {m['avg_loss']:+.2f}R "
              f"(max {m['max_win']:+.2f}R / {m['max_loss']:+.2f}R)")
    else:
        print("  No trades.")

    print("\n-- Monthly R (consistency check) --")
    monthly = {}
    for i in range(len(rs)):
        key = times_dt[mask][i].strftime("%Y-%m")
        monthly.setdefault(key, []).append(rs[i])
    if monthly:
        print(f"{'month':>8} | {'trades':>6} | {'WR':>5} | {'net R':>7}")
        pos_months = 0
        for key in sorted(monthly):
            r = np.array(monthly[key])
            net = float(r.sum())
            if net > 0:
                pos_months += 1
            print(f"{key:>8} | {len(r):>6} | {100.0*(r>0).mean():4.1f}% | {net:+7.1f}")
        print(f"Positive months: {pos_months}/{len(monthly)} "
              f"({100.0*pos_months/max(len(monthly),1):.0f}%)")

    print("\n-- Regime breakdown (does the brain switch by regime?) --")
    regime_names = {1.0: "UP-TREND", -1.0: "DOWN-TREND", 0.0: "RANGE"}
    for rv, name in [(1.0, "UP-TREND"), (-1.0, "DOWN-TREND"), (0.0, "RANGE")]:
        rmask = (r_up == rv)
        amask = mask & rmask
        n_calls = int(amask.sum())
        n_bars = int(rmask.sum())
        if n_calls:
            r2 = rs[amask]
            w = 100.0 * float((r2 > 0).mean())
            net = float(r2.sum())
            print(f"  {name:>11}: {n_calls:>5} calls / {n_bars:>7} bars "
                  f"({100.0*n_calls/max(n_bars,1):.1f}%) WR={w:4.1f}% net={net:+.1f}R")
        else:
            print(f"  {name:>11}: {0:>5} calls / {n_bars:>7} bars (0.0%)")

    print("\n-- Baseline label stats (what was available) --")
    cnt = np.bincount(label, minlength=3)
    print(f"BUY={cnt[0]} SELL={cnt[1]} NONE={cnt[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--limit-files", type=int, default=2)
    args = parser.parse_args()
    main(args.symbol, args.limit_files)
