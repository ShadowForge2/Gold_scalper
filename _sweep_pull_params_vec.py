"""
_sweep_pull_params_vec.py — FULLY VECTORIZED robust pull-param sweep.

Same design as _sweep_pull_params.py (train 2023-24 / valid 2025 / oos 2026,
keep configs profitable in all three periods, rank by geometric-mean PF) but
the per-config simulation is numpy-vectorized over all bars and all
(trail, horizon) combos at once, so a 11x10x11 grid over 6 symbols completes
in well under a minute of compute (data load/cache is the only slow part).

For each PULL value the entry set is computed once; then for every (trail, H)
the exit is a single vectorized "first index where required-trail is met"
lookup. Reuses the npz cache from _sweep_pull_params.py.

Usage:
  python _sweep_pull_params_vec.py                # strict 3-period, full grid
  python _sweep_pull_params_vec.py --quick        # small grid, sanity check
  python _sweep_pull_params_vec.py --wide --relaxed --symbols DE40,JP225
                                                  # separate test: wider grid,
                                                  # walk-forward valid+OOS only
  python _sweep_pull_params_vec.py --emit-env     # also print env assignments
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _sweep_pull_params import build_all   # cached M1->M5 build
import config as cfg

HMAX = 48   # max forward window (bars) to precompute per entry

PULLS_FULL = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]
TRAILS_FULL = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
HORIZONS_FULL = [4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48]

PULLS_QUICK = [0.10, 0.15, 0.20, 0.30]
TRAILS_QUICK = [0.15, 0.25, 0.35, 0.50]
HORIZONS_QUICK = [6, 12, 24]

# Wider grid for the DE40/JP225 separate test (deeper pulls, looser trails,
# much longer holds — those markets run in longer waves).
PULLS_WIDE = PULLS_FULL + [0.45, 0.50, 0.60, 0.70]
TRAILS_WIDE = TRAILS_FULL + [0.70, 0.80]
HORIZONS_WIDE = HORIZONS_FULL + [60, 72, 96]


def build_req(ball, start, end, pull):
    """Return per-entry forward arrays for a (period, pull). None if no entries."""
    c = ball["c"]
    n = len(c)
    pa = ball["pa"]
    h1 = ball["h1dir"]
    cost_r = ball["cost_r"]
    y = ball["year"]

    # bars i in [3, n-1) (i+1 must exist for the entry turn + fwd window).
    # All windows have length n-4.
    w0 = c[0:n - 4]
    w1 = c[1:n - 3]
    w2 = c[2:n - 2]
    w3 = c[3:n - 1]
    w4 = c[4:n]

    d0 = h1[3:n - 1]
    pa_i = pa[3:n - 1]
    base = (y[3:n - 1] >= start) & (y[3:n - 1] <= end) & (pa_i > 0) & (d0 != 0)

    up = ((w2 - w1) > 0) & ((w1 - w0) > 0)   # two consecutive closes in direction
    dn = ((w2 - w1) < 0) & ((w1 - w0) < 0)
    dirn = np.zeros(len(w2))
    dirn[up] = 1.0
    dirn[dn] = -1.0

    aligned = (dirn * d0) > 0
    dip = (w2 - w3) * dirn >= pull * pa_i
    turn = (w4 - w3) * dirn > 0
    m = base & aligned & (dirn != 0) & dip & turn

    e = 4 + np.nonzero(m)[0]
    if len(e) == 0:
        return None
    keep = (e + HMAX) <= n - 1
    e = e[keep]
    dirn_e = dirn[m][keep]
    if len(e) == 0:
        return None

    entry_c = c[e]
    pa_e = pa[e]
    cost_e = cost_r[e]
    entry_z = (entry_c * dirn_e)[:, None]

    z = c[np.minimum(e[:, None] + np.arange(1, HMAX + 1), n - 1)] * dirn_e[:, None]
    ext = np.maximum.accumulate(z, axis=1)
    wave = ext - entry_z
    back = ext - z
    req = np.full_like(wave, np.nan, dtype=float)
    ok = wave > 0
    req[ok] = back[ok] / wave[ok]
    return dict(e=e, entry_c=entry_c, dirn=dirn_e, pa_e=pa_e,
                cost_e=cost_e, req=req, n=n)


def eval_vec_full(ball, start, end, pull, trail, H):
    """Vectorized per-config stats (trades / wr / pf / net R)."""
    r = build_req(ball, start, end, pull)
    if r is None:
        return None
    req = r["req"][:, :H]
    hit = req >= trail
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1) + 1
    exit_k = np.where(any_hit, first, H).astype(int)
    idx = np.minimum(r["e"] + exit_k, r["n"] - 1)
    c = ball["c"]
    ep = c[idx]
    rr = (ep - r["entry_c"]) * r["dirn"] / r["pa_e"] - r["cost_e"]
    wins = float(rr[rr > 0].sum())
    losses = float((-rr[rr < 0]).sum())
    return dict(trd=len(rr), wr=100.0 * float((rr > 0).mean()),
                exp=float(rr.mean()),
                pf=(wins / losses if losses > 0 else float("inf")),
                net=float(rr.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(cfg.SYMBOLS))
    p.add_argument("--train-start", type=int, default=2023)
    p.add_argument("--train-end", type=int, default=2024)
    p.add_argument("--valid", type=int, default=2025)
    p.add_argument("--oos", type=int, default=2026)
    p.add_argument("--min-trades", type=int, default=150)
    p.add_argument("--oos-floor", type=float, default=1.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--wide", action="store_true",
                   help="use the wider grid (deeper pulls, longer holds)")
    p.add_argument("--relaxed", action="store_true",
                   help="require valid+OOS PF >= floor (ignore the train period)")
    p.add_argument("--emit-env", action="store_true")
    args = p.parse_args()

    pulls = PULLS_QUICK if args.quick else (PULLS_WIDE if args.wide else PULLS_FULL)
    trails = TRAILS_QUICK if args.quick else (TRAILS_WIDE if args.wide else TRAILS_FULL)
    horizons = HORIZONS_QUICK if args.quick else (HORIZONS_WIDE if args.wide else HORIZONS_FULL)

    mode = "RELAXED(valid+oos)" if args.relaxed else "STRICT(all 3 periods)"
    print(f"\nVECTORIZED ROBUST PULL SWEEP ({mode}) | "
          f"{len(pulls)}x{len(trails)}x{len(horizons)} = "
          f"{len(pulls)*len(trails)*len(horizons)} combos/sym", flush=True)
    print(f"periods: train {args.train_start}-{args.train_end} | "
          f"valid {args.valid} | oos {args.oos} | min_trades={args.min_trades} "
          f"| oos_floor={args.oos_floor}", flush=True)

    recs = []
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        print(f"  {sym:>7}: building/caching data...", flush=True)
        ball = build_all(sym)
        if ball is None:
            print(f"  {sym:>7}: no data", flush=True)
            continue

        best = None
        rows = []
        for pull in pulls:
            r = build_req(ball, args.train_start, args.train_end, pull)
            rv = build_req(ball, args.valid, args.valid, pull)
            ro = build_req(ball, args.oos, args.oos, pull)
            for trail in trails:
                for H in horizons:
                    res = []
                    ok = True
                    for rp in (r, rv, ro):
                        if rp is None:
                            ok = False
                            break
                        hit = rp["req"][:, :H] >= trail
                        any_hit = hit.any(axis=1)
                        first = hit.argmax(axis=1) + 1
                        exit_k = np.where(any_hit, first, H).astype(int)
                        idx = np.minimum(rp["e"] + exit_k, rp["n"] - 1)
                        ep = ball["c"][idx]
                        rr = (ep - rp["entry_c"]) * rp["dirn"] / rp["pa_e"] - rp["cost_e"]
                        wins = float(rr[rr > 0].sum())
                        losses = float((-rr[rr < 0]).sum())
                        res.append(dict(trd=len(rr),
                                        pf=(wins / losses if losses > 0 else float("inf")),
                                        net=float(rr.sum()),
                                        wr=100.0 * float((rr > 0).mean())))
                    if not ok:
                        continue
                    if args.relaxed:
                        # Walk-forward: only the valid + oos windows must be
                        # profitable; the 2023-24 train period is informational.
                        if min(res[1]["trd"], res[2]["trd"]) < args.min_trades:
                            continue
                        if min(res[1]["pf"], res[2]["pf"]) < args.oos_floor:
                            continue
                        g = (res[1]["pf"] * res[2]["pf"]) ** 0.5
                        cons = min(res[1]["pf"], res[2]["pf"])
                    else:
                        if min(res[0]["trd"], res[1]["trd"], res[2]["trd"]) < args.min_trades:
                            continue
                        if min(res[0]["pf"], res[1]["pf"], res[2]["pf"]) < args.oos_floor:
                            continue
                        g = (res[0]["pf"] * res[1]["pf"] * res[2]["pf"]) ** (1.0 / 3.0)
                        cons = min(res[0]["pf"], res[1]["pf"], res[2]["pf"])
                    rows.append((g, cons, res[0], res[1], res[2], pull, trail, H))

        if not rows:
            what = "valid+OOS" if args.relaxed else "all 3 periods"
            print(f"  {sym:>7}: no config survived {what} at PF>={args.oos_floor}",
                  flush=True)
            continue
        rows.sort(key=lambda x: (-x[0], -x[1]))
        g, cons, tr, va, oo, pull, trail, H = rows[0]

        d = cfg.PULL_SYMBOL_DEFAULTS.get(sym, {})
        dlift = float("nan")
        try:
            rd = [eval_vec_full(ball, args.train_start, args.train_end,
                                d.get("pull_r", 0.30), d.get("trail_r", 0.35),
                                d.get("max_hold", 24)),
                  eval_vec_full(ball, args.valid, args.valid,
                                d.get("pull_r", 0.30), d.get("trail_r", 0.35),
                                d.get("max_hold", 24)),
                  eval_vec_full(ball, args.oos, args.oos,
                                d.get("pull_r", 0.30), d.get("trail_r", 0.35),
                                d.get("max_hold", 24))]
            if all(x is not None for x in rd):
                dstr = (f"DEFAULT pull {d.get('pull_r')} trail {d.get('trail_r')} "
                        f"hold {d.get('max_hold')}: "
                        f"TRN {rd[0]['pf']:.2f} VAL {rd[1]['pf']:.2f} "
                        f"OOS {rd[2]['pf']:.2f}")
                dlift = oo["pf"] - rd[2]["pf"]
            else:
                dstr = "DEFAULT: n/a"
        except Exception:
            dstr = "DEFAULT: n/a"

        print(f"  {sym:>7}: RECOMMENDED pull {pull} trail {trail} hold {H}", flush=True)
        print(f"           PF  TRN {tr['pf']:.2f} ({tr['net']:+.0f}R) "
              f"VAL {va['pf']:.2f} ({va['net']:+.0f}R) "
              f"OOS {oo['pf']:.2f} ({oo['net']:+.0f}R) "
              f"| gmean {g:.2f} cons {cons:.2f} | "
              f"trd {tr['trd']}/{va['trd']}/{oo['trd']}", flush=True)
        print(f"           {dstr}  ->  OOS lift {dlift:+.2f}", flush=True)
        print(f"           top-3 (geom-mean):", flush=True)
        for r0 in rows[:3]:
            print(f"             pull {r0[5]} trail {r0[6]} hold {r0[7]}: "
                  f"gmean {r0[0]:.2f} | TRN {r0[2]['pf']:.2f} "
                  f"VAL {r0[3]['pf']:.2f} OOS {r0[4]['pf']:.2f}", flush=True)
        recs.append((sym, pull, trail, H, oo["pf"], dlift))

    print("\n=== RECOMMENDED SYMBOL_PULL_PARAMS ===", flush=True)
    for sym, pull, trail, H, opf, dlift in recs:
        flag = "" if opf >= 1.3 else "  (OOS<1.3: keep disabled)"
        print(f"  {sym}: pull_r={pull} trail_r={trail} max_hold={H}{flag}", flush=True)
    if args.emit_env:
        print("\n=== ENV VAR ASSIGNMENTS ===", flush=True)
        for sym, pull, trail, H, opf, dlift in recs:
            print(f"PULL_PULL_R_{sym}={pull}", flush=True)
            print(f"PULL_TRAIL_R_{sym}={trail}", flush=True)
            print(f"PULL_MAX_HOLD_{sym}={H}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
