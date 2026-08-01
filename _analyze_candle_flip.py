"""Measure CandleML flip rigidity.

Extends the flip analysis to answer: can we trust a single opposite signal as a
rigid "no", or do we need 2+ consecutive opposite signals?

Reports, for BUY and SELL entries:
  1. Signal stats + flip rate between consecutive signals
  2. Opposite-run length distribution (how long a "no" persists)
  3. Entry survival to a single flip (1-consecutive) vs a PERSISTENT flip
     (2+ consecutive opposite) — the "rigid no"
  4. Confidence of the first opposite signal (would a higher exit threshold help?)
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
os.chdir(base)

import config as cfg
from app.candle_ml import compute_candle_features, CandleML

CONF = getattr(cfg, "CANDLE_ML_CONFIDENCE_THRESHOLD", 0.65)

DATA = {
    "XAUUSD": ("data/dukascopy", "models/candle_xgb_m5_XAUUSD.joblib"),
    "US100": ("data/dukascopy_us100", "models/candle_xgb_m5_US100.joblib"),
    "US500": ("data/dukascopy_us500", "models/candle_xgb_m5_US500.joblib"),
}


def load_bars(cache_dir, symbol, year):
    path = os.path.join(cache_dir, f"{symbol}_M1_{year}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df = df.sort_values("time").drop_duplicates(subset="time")
    idx = df.set_index("time") if "time" in df.columns else df
    return idx[~idx.index.duplicated(keep="first")].sort_index()


def out(msg):
    print(msg, flush=True)


def analyze(bars, model, symbol):
    out(f"  computing features over {len(bars):,} M1 bars ...")
    feats = compute_candle_features(bars)
    if feats is None or len(feats) == 0:
        out(f"  {symbol}: no features")
        return
    m1_dir = feats["m1_first_dir"].fillna(0).astype(int)
    preds = {}   # boundary -> direction
    confs = {}   # boundary -> confidence of that prediction
    for ts, row in feats.iterrows():
        prob_up = model.predict_proba(row.to_frame().T)
        d = m1_dir.loc[ts]
        pred = model.predict(prob_up, d, confidence_threshold=CONF)
        t = int(ts.timestamp())
        preds[t] = pred
        confs[t] = float(max(prob_up, 1 - prob_up)) if pred is not None else None

    buckets = sorted(preds)
    sigs = [(b, preds[b]) for b in buckets if preds[b] is not None]
    if len(sigs) < 2:
        out(f"  {symbol}: too few signals")
        return

    n_all = len(buckets)
    n_sig = len(sigs)
    buys = [p for _, p in sigs if p == "BUY"]
    sells = [p for _, p in sigs if p == "SELL"]
    flips = sum(1 for a, b in zip(sigs, sigs[1:]) if a[1] != b[1])

    out(f"\n  {symbol}: boundaries={n_all} signals={n_sig} "
        f"({n_sig / max(1, n_all) * 100:.1f}%) BUY={len(buys)} SELL={len(sells)}")
    out(f"  Consecutive-signal flip rate: {flips / max(1, n_sig - 1) * 100:.1f}%")

    # ── Opposite-run length distribution ──
    runs = []
    cur, run = sigs[0][1], 1
    for _, p in sigs[1:]:
        if p == cur:
            run += 1
        else:
            runs.append((cur, run))
            cur, run = p, 1
    runs.append((cur, run))

    for d in ("BUY", "SELL"):
        d_runs = [r for c, r in runs if c == d]
        if not d_runs:
            continue
        a = np.array(d_runs, dtype=float)
        blip = (a == 1).mean() * 100
        out(f"  {d}-signal run length: mean={a.mean():.2f} median={np.median(a):.1f} "
            f"blips(len1)={blip:.1f}% max={int(a.max())}")

    # ── Entry survival: single flip vs persistent (2+ consecutive) flip ──
    surv1 = {"BUY": [], "SELL": []}
    surv2 = {"BUY": [], "SELL": []}
    never1 = {"BUY": 0, "SELL": 0}
    never2 = {"BUY": 0, "SELL": 0}
    flip_conf = {"BUY": [], "SELL": []}
    for i, (b0, p0) in enumerate(sigs):
        target = "SELL" if p0 == "BUY" else "BUY"
        t1 = t2 = None
        c1 = None
        for j in range(i + 1, len(sigs)):
            if sigs[j][1] == target:
                if t1 is None:
                    t1 = (sigs[j][0] - b0) // 300
                    c1 = confs.get(sigs[j][0])
                if j + 1 < len(sigs) and sigs[j + 1][1] == target:
                    t2 = (sigs[j][0] - b0) // 300
                    break
        if t1 is None:
            never1[p0] += 1
        else:
            surv1[p0].append(t1)
            if c1 is not None:
                flip_conf[p0].append(c1)
        if t2 is None:
            never2[p0] += 1
        else:
            surv2[p0].append(t2)

    for d in ("BUY", "SELL"):
        s1 = np.array(surv1[d], dtype=float)
        s2 = np.array(surv2[d], dtype=float)
        fc = np.array(flip_conf[d], dtype=float)
        tot1 = len(s1) + never1[d]
        tot2 = len(s2) + never2[d]
        if tot1 == 0:
            continue
        p1 = {b: int((s1 <= b).sum()) / tot1 * 100 for b in (1, 2, 3, 5, 10)}
        p2 = {b: int((s2 <= b).sum()) / tot2 * 100 for b in (1, 2, 3, 5, 10)}
        out(f"\n  {d} entries={tot1} | single-flip within 1/2/3/5/10: "
            f"{p1[1]:.1f}/{p1[2]:.1f}/{p1[3]:.1f}/{p1[5]:.1f}/{p1[10]:.1f}% "
            f"(median {np.median(s1):.1f})")
        out(f"  {d}           | PERSISTENT-flip(2+) within 1/2/3/5/10: "
            f"{p2[1]:.1f}/{p2[2]:.1f}/{p2[3]:.1f}/{p2[5]:.1f}/{p2[10]:.1f}% "
            f"(median {np.median(s2):.1f})" if len(s2) else f"  {d} | persistent: no samples")
        blip_rate = (tot1 - tot2) / tot1 * 100 if tot1 else 0
        out(f"  {d} blips avoided by requiring 2 consecutive: {blip_rate:.1f}% "
            f"| first-flip conf median {np.median(fc):.2f} "
            f"(conf>=0.75 would filter {(fc >= 0.75).mean() * 100:.1f}%)" if len(fc) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="US100", help="XAUUSD|US100|US500")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()

    symbol = args.symbol.upper()
    if symbol not in DATA:
        print(f"unknown symbol {symbol}")
        return
    cache_dir, model_path = DATA[symbol]
    if not os.path.exists(model_path):
        print(f"model not found: {model_path}")
        return

    model = CandleML(model_path)
    if model.model is None:
        print("model failed to load")
        return

    bars = load_bars(cache_dir, symbol, args.year)
    if bars is None or len(bars) == 0:
        print(f"no data for {symbol} {args.year}")
        return
    print(f"Loaded {len(bars):,} M1 bars ({symbol} {args.year})", flush=True)
    analyze(bars, model, symbol)


if __name__ == "__main__":
    main()
