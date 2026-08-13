"""
_sweep_pull_params.py — robust, dense sweep of pull/prevh1/trail params.

Improves on _tune_pull_prevh1.py in two ways:
  1. Denser grid over (pullback depth, trail giveback, max hold).
  2. Anti-overfit selection. Instead of picking the single best TRAIN-PF config
     (which overfits the train window), every config is scored across THREE
     independent periods:
         TRAIN 2023-2024   (in-sample)
         VALID 2025        (walk-forward validation)
         OOS   2026        (true out-of-sample)
     and we keep only configs that are profitable in ALL three periods
     (PF >= oos_floor each) with enough trades, then rank by the geometric mean
     of the three PFs (so a config that collapses in any one period is punished).
     We also report the "consistency" = min(train,valid,oos) PF.

For each symbol we compare the recommended config against the CURRENT hardcoded
default (config.PULL_SYMBOL_DEFAULTS) so you can see the PF lift.

Usage:
  python _sweep_pull_params.py                 # full dense sweep, all 6 symbols
  python _sweep_pull_params.py --quick         # smaller grid, fast sanity check
  python _sweep_pull_params.py --symbols XAUUSD,US30
  python _sweep_pull_params.py --emit-env      # also print env var assignments
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _tune_pull_prevh1 import run, SYMBOL_COSTS   # reuse honest run(); we cache build()
import config as cfg

import pandas as pd
from pathlib import Path

# Cache the expensive M1->M5 build once per symbol (npz). Reloads are instant,
# so re-running the sweep (or tweaking the grid) does not re-read parquet.
_CACHE_DIR = Path("data/_sweep_cache")
_CACHE_YEARS = (2022, 2026)   # covers train 2023-24, valid 2025, oos 2026


def build_all(symbol: str) -> dict:
    """Load M1 once for all needed years, resample to M5, and return the arrays
    the sweep needs. Cached to an npz keyed by symbol."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _CACHE_DIR / f"{symbol}.npz"
    if cache.exists():
        try:
            d = np.load(cache)
            return dict(c=d["c"], t=d["t"], pa=d["pa"], h1dir=d["h1dir"],
                        cost_r=d["cost_r"], year=d["year"], n=len(d["c"]))
        except Exception:
            pass
    from _data_loader import load_m1_data
    from app.candle_engine import compute_atr
    m1 = load_m1_data(symbol, start_year=_CACHE_YEARS[0], end_year=_CACHE_YEARS[1])
    if m1 is None or len(m1) == 0:
        return None
    idx = m1.set_index("time") if "time" in m1.columns else m1
    idx = idx[~idx.index.duplicated(keep="first")].sort_index()
    h1 = idx.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    atr = compute_atr(h1, 14)
    m5 = idx.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    c = m5["close"].values.astype(float)
    t = m5.index
    pa = atr.reindex(m5.index, method="ffill").shift(1).values.astype(float)
    h1dir = np.sign(h1["close"] - h1["open"]).reindex(m5.index, method="ffill").shift(1).values.astype(float)
    cost = SYMBOL_COSTS.get(symbol, {"spread": 0.0, "commission": 0.0})
    rtp = cost["spread"] + 2 * cost["commission"]
    cost_r = (rtp / pa).astype(float)
    year = t.year.values.astype(int)
    np.savez(cache, c=c, t=t.values, pa=pa, h1dir=h1dir, cost_r=cost_r, year=year)
    return dict(c=c, t=t, pa=pa, h1dir=h1dir, cost_r=cost_r, year=year, n=len(c))

# ── Grids ──────────────────────────────────────────────────────────────
PULLS_FULL = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]
TRAILS_FULL = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
HORIZONS_FULL = [4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48]

PULLS_QUICK = [0.10, 0.15, 0.20, 0.30]
TRAILS_QUICK = [0.15, 0.25, 0.35, 0.50]
HORIZONS_QUICK = [6, 12, 24]


def _period(b, start, end):
    if b is None:
        return None
    year_mask = (b["year"] >= start) & (b["year"] <= end)
    return dict(b, year_mask=year_mask)


def eval_cfg(symbol, pull, trail, hz, bt, bv, bo):
    """Return dict with PF/net/trades for each period, or None if any period
    has too few trades to be meaningful."""
    rt = run(symbol, bt, pull, trail, hz)
    rv = run(symbol, bv, pull, trail, hz)
    ro = run(symbol, bo, pull, trail, hz) if bo is not None else None
    if rt is None or rv is None:
        return None
    if ro is None:
        return None
    return {"train": rt, "valid": rv, "oos": ro}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(cfg.SYMBOLS))
    p.add_argument("--train-start", type=int, default=2023)
    p.add_argument("--train-end", type=int, default=2024)
    p.add_argument("--valid", type=int, default=2025)
    p.add_argument("--oos", type=int, default=2026)
    p.add_argument("--min-trades", type=int, default=150)
    p.add_argument("--oos-floor", type=float, default=1.0,
                   help="every selected config must hit PF >= this in all 3 periods")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--emit-env", action="store_true")
    args = p.parse_args()

    pulls = PULLS_QUICK if args.quick else PULLS_FULL
    trails = TRAILS_QUICK if args.quick else TRAILS_FULL
    horizons = HORIZONS_QUICK if args.quick else HORIZONS_FULL

    print(f"\nROBUST PULL-PARAM SWEEP | grid pull={len(pulls)} trail={len(trails)} "
          f"hold={len(horizons)} = {len(pulls)*len(trails)*len(horizons)} combos/sym")
    print(f"periods: train {args.train_start}-{args.train_end} | valid {args.valid} "
          f"| oos {args.oos} | min_trades={args.min_trades} | oos_floor={args.oos_floor}")
    print("selection: profitable in ALL 3 periods, ranked by geometric-mean PF\n")

    recs = []
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        print(f"  {sym:>7}: loading/caching M5 data...", flush=True)
        ball = build_all(sym)
        if ball is None:
            print(f"  {sym:>7}: no data")
            continue
        bt = _period(ball, args.train_start, args.train_end)
        bv = _period(ball, args.valid, args.valid)
        bo = _period(ball, args.oos, args.oos)
        if bt is None or bv is None or bo is None:
            print(f"  {sym:>7}: missing data (train/valid/oos)")
            continue

        rows = []
        for pull in pulls:
            for trail in trails:
                for hz in horizons:
                    r = eval_cfg(sym, pull, trail, hz, bt, bv, bo)
                    if r is None:
                        continue
                    tr, va, oo = r["train"], r["valid"], r["oos"]
                    if min(tr["trd"], va["trd"], oo["trd"]) < args.min_trades:
                        continue
                    if min(tr["pf"], va["pf"], oo["pf"]) < args.oos_floor:
                        continue
                    gmean = (tr["pf"] * va["pf"] * oo["pf"]) ** (1.0 / 3.0)
                    cons = min(tr["pf"], va["pf"], oo["pf"])
                    rows.append((gmean, cons, tr["pf"], va["pf"], oo["pf"],
                                 tr["net"], va["net"], oo["net"],
                                 pull, trail, hz, tr["trd"], va["trd"], oo["trd"]))
        if not rows:
            print(f"  {sym:>7}: NO config survived all 3 periods at PF>={args.oos_floor}")
            continue
        rows.sort(key=lambda x: (-x[0], -x[1]))
        g, cons, tpf, vpf, opf, tnet, vnet, onet, pull, trail, hz, tt, vt, ot = rows[0]

        # current default for comparison
        d = cfg.PULL_SYMBOL_DEFAULTS.get(sym, {})
        rd = eval_cfg(sym, d.get("pull_r", 0.30), d.get("trail_r", 0.35),
                      d.get("max_hold", 24), bt, bv, bo)
        if rd is not None:
            dstr = (f"DEFAULT pull {d.get('pull_r')} trail {d.get('trail_r')} "
                    f"hold {d.get('max_hold')}: "
                    f"TRN {rd['train']['pf']:.2f} VAL {rd['valid']['pf']:.2f} "
                    f"OOS {rd['oos']['pf']:.2f}")
            dlift = opf - rd["oos"]["pf"]
        else:
            dstr = "DEFAULT: n/a"
            dlift = float("nan")

        print(f"  {sym:>7}: RECOMMENDED pull {pull} trail {trail} hold {hz}")
        print(f"           PF  TRN {tpf:.2f} ({tnet:+.0f}R) "
              f"VAL {vpf:.2f} ({vnet:+.0f}R) OOS {opf:.2f} ({onet:+.0f}R) "
              f"| gmean {g:.2f} cons {cons:.2f} | trd {tt}/{vt}/{ot}")
        print(f"           {dstr}  ->  OOS lift {dlift:+.2f}")
        print(f"           top-3 (geom-mean):")
        for r in rows[:3]:
            print(f"             pull {r[8]} trail {r[9]} hold {r[10]}: "
                  f"gmean {r[0]:.2f} | TRN {r[2]:.2f} VAL {r[3]:.2f} OOS {r[4]:.2f}")
        recs.append((sym, pull, trail, hz, opf, dlift))

    print("\n=== RECOMMENDED SYMBOL_PULL_PARAMS ===")
    for sym, pull, trail, hz, opf, dlift in recs:
        flag = "" if opf >= 1.3 else "  (OOS<1.3: keep disabled)"
        print(f"  {sym}: pull_r={pull} trail_r={trail} max_hold={hz}{flag}")
    if args.emit_env:
        print("\n=== ENV VAR ASSIGNMENTS ===")
        for sym, pull, trail, hz, opf, dlift in recs:
            print(f"PULL_PULL_R_{sym}={pull}")
            print(f"PULL_TRAIL_R_{sym}={trail}")
            print(f"PULL_MAX_HOLD_{sym}={hz}")
    print()


if __name__ == "__main__":
    main()
