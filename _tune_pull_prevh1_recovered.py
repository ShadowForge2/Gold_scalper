"""
_tune_pull_prevh1.py ΓÇö per-pair tuning of pull/prevh1/trail25.

Grid over (pullback depth, trail giveback, max hold). TRAIN on 2023-2024,
pick the best config per pair by PF, then report that config on 2025
(validation) and 2026 (true out-of-sample). Same runtime for every symbol.

Usage:
  python _tune_pull_prevh1.py [--symbols ...] [--oos 2026]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _data_loader import load_m1_data
from app.candle_engine import compute_atr

ATR_PERIOD = 14

SYMBOL_COSTS = {
    "XAUUSD": {"spread": 0.30, "commission": 0.04},
    "US100":  {"spread": 2.00, "commission": 0.00},
    "JP225":  {"spread": 10.0, "commission": 0.00},
    "DE40":   {"spread": 2.00, "commission": 0.00},
    "US500":  {"spread": 0.80, "commission": 0.00},
    "US30":   {"spread": 2.00, "commission": 0.00},
}

PULLS = [0.05, 0.10, 0.15, 0.20, 0.30]
TRAILS = [0.15, 0.25, 0.35, 0.50]
HORIZONS = [6, 12, 24]   # M5 bars


def build(symbol, start, end):
    m1 = load_m1_data(symbol, start_year=start - 1, end_year=end)
    if m1 is None or len(m1) == 0:
        return None
    idx = m1.set_index("time") if "time" in m1.columns else m1
    idx = idx[~idx.index.duplicated(keep="first")].sort_index()
    h1 = idx.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    atr = compute_atr(h1, ATR_PERIOD)
    m5 = idx.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    c = m5["close"].values
    t = m5.index
    pa = atr.reindex(m5.index, method="ffill").shift(1).values
    h1dir = np.sign(h1["close"] - h1["open"]).reindex(m5.index, method="ffill").shift(1).values
    cost = SYMBOL_COSTS.get(symbol, {"spread": 0.0, "commission": 0.0})
    rtp = cost["spread"] + 2 * cost["commission"]
    cost_r = rtp / pa
    year_mask = (t.year >= start) & (t.year <= end)
    return dict(c=c, t=t, pa=pa, h1dir=h1dir, cost_r=cost_r,
                year_mask=year_mask, n=len(c))


def run(symbol, b, pull, trail, horizon):
    c = b["c"]
    pa = b["pa"]
    h1dir = b["h1dir"]
    cost_r = b["cost_r"]
    n = b["n"]
    out = []
    for i in range(3, n - 1):
        if not b["year_mask"][i] or pa[i] <= 0 or np.isnan(pa[i]):
            continue
        d0 = h1dir[i]
        if d0 == 0:
            continue
        up = (c[i - 1] - c[i - 2]) > 0 and (c[i - 2] - c[i - 3]) > 0
        dn = (c[i - 1] - c[i - 2]) < 0 and (c[i - 2] - c[i - 3]) < 0
        if (up and d0 < 0) or (dn and d0 > 0) or (not up and not dn):
            continue
        dirn = 1 if up else -1
        dip = (c[i - 1] - c[i]) * dirn
        if dip < pull * pa[i]:
            continue
        if (c[i + 1] - c[i]) * dirn <= 0:
            continue
        e = c[i + 1]
        ei = i + 1
        ep = None
        run_ex = e
        for k in range(1, horizon + 1):
            j = ei + k
            if j >= n:
                break
            run_ex = max(run_ex, c[j]) if dirn > 0 else min(run_ex, c[j])
            wave = (run_ex - e) * dirn
            back = (run_ex - c[j]) * dirn
            if wave > 0 and back >= trail * wave:
                ep = c[j]
                break
        if ep is None:
            ep = c[min(ei + horizon, n - 1)]
        r = (ep - e) * dirn / pa[ei] - cost_r[ei]
        out.append(r)
    if len(out) < 30:
        return None
    arr = np.asarray(out, dtype=float)
    wins = float(arr[arr > 0].sum())
    losses = float((-arr[arr < 0]).sum())
    return dict(trd=len(arr), wr=100 * float((arr > 0).mean()),
                exp=float(arr.mean()),
                pf=(wins / losses if losses > 0 else float("inf")),
                net=float(arr.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="XAUUSD,US100,JP225,DE40,US500,US30")
    p.add_argument("--train-start", type=int, default=2023)
    p.add_argument("--train-end", type=int, default=2024)
    p.add_argument("--valid", type=int, default=2025)
    p.add_argument("--oos", type=int, default=2026)
    p.add_argument("--min-trades", type=int, default=300)
    args = p.parse_args()

    print(f"\nper-pair tuning of pull/prevh1/trail25 "
          f"| train {args.train_start}-{args.train_end} "
          f"| valid {args.valid} | OOS {args.oos}")
    print(f"grid: pull {PULLS} trail {TRAILS} hold {HORIZONS}")
    best_global = None
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        b = build(sym, args.train_start, args.valid)
        if b is None:
            print(f"\n  {sym}: no data")
            continue
        # tune on train window only
        btr = dict(b, year_mask=(b["t"].year >= args.train_start)
                   & (b["t"].year <= args.train_end))
        results = []
        for pull in PULLS:
            for trail in TRAILS:
                for hz in HORIZONS:
                    r = run(sym, btr, pull, trail, hz)
                    if r and r["trd"] >= args.min_trades and r["pf"] > 0:
                        results.append((r["pf"], r["net"], r["trd"],
                                        pull, trail, hz, r))
        if not results:
            print(f"\n  {sym}: no qualifying configs")
            continue
        results.sort(key=lambda x: (-x[0], -x[1]))
        pf, net, trd, pull, trail, hz, r = results[0]
        cfg = (pull, trail, hz)
        # validate on the validation year
        bv = dict(b, year_mask=(b["t"].year == args.valid))
        rv = run(sym, bv, *cfg)
        # out-of-sample on the OOS year
        bo = None
        ro = None
        try:
            bo = build(sym, args.oos, args.oos)
            if bo is not None:
                bo = dict(bo, year_mask=(bo["t"].year == args.oos))
                ro = run(sym, bo, *cfg)
        except Exception:
            ro = None
        line = (f"  {sym:>7} | BEST train PF {pf:.2f} ({net:+.0f}R "
                f"pull {pull} trail {trail} hold {hz})")
        if rv:
            line += f" | valid {args.valid} PF {rv['pf']:.2f} ({rv['net']:+.0f}R)"
        if ro:
            line += f" | OOS {args.oos} PF {ro['pf']:.2f} ({ro['net']:+.0f}R)"
        else:
            line += f" | OOS {args.oos}: no data"
        print(line)
        if ro and ro["pf"] >= 1.3:
            best_global = best_global or []
            best_global.append((sym, cfg, ro, rv))
        # also show top-5 train table
        print(f"    top5 (train):")
        for pf, net, trd, pull, trail, hz, r in results[:5]:
            print(f"      pull {pull} trail {trail} hold {hz}: "
                  f"PF {pf:.2f} {net:+.0f}R wr {r['wr']:.0f}% trd {trd}")

    if best_global:
        print(f"\n  configs that survived OOS at PF>=1.3:")
        for sym, cfg, ro, rv in best_global:
            print(f"    {sym}: pull {cfg[0]} trail {cfg[1]} hold {cfg[2]} "
                  f"-> OOS PF {ro['pf']:.2f} {ro['net']:+.0f}R")


if __name__ == "__main__":
    main()
