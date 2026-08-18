"""
_sweep_pf.py — sweep pull/trail/hold to find the highest PF configs for XAUUSD.

Tests every combo under realistic adverse-side fills. Reports:
- train/valid/OOS PF, win rate, expected R, net R
- avg win / avg loss ratio (R:R)
- worst-day drawdown

Usage:
  python _sweep_pf.py --symbols XAUUSD
  python _sweep_pf.py --symbols XAUUSD --min-pf 1.5
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _tune_recovered import build, SYMBOL_COSTS

PULLS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TRAILS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
HORIZONS = [4, 6, 8, 10, 12, 16, 20, 24]
HMAX = 48


def eval_config(ball, start, end, pull, trail, H, slip_r=0.01):
    """Full eval with per-trade R values for win/loss ratio analysis."""
    c = ball["c"]
    t = ball["t"]
    pa = ball["pa"]
    h1 = ball["h1dir"]
    n = ball["n"]
    y = ball.get("year")
    if y is None:
        y = np.asarray([ts.year for ts in t])

    cost = SYMBOL_COSTS.get("XAUUSD", {"spread": 0.30, "commission": 0.04})
    spread = cost["spread"]
    comm = cost["commission"]

    w0 = c[0:n - 4]
    w1 = c[1:n - 3]
    w2 = c[2:n - 2]
    w3 = c[3:n - 1]
    w4 = c[4:n]

    d0 = h1[3:n - 1]
    pa_i = pa[3:n - 1]
    base = (y[3:n - 1] >= start) & (y[3:n - 1] <= end) & (pa_i > 0) & (d0 != 0)

    up = ((w2 - w1) > 0) & ((w1 - w0) > 0)
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
    dirn_e = dirn[m]

    entry_c = c[e]
    pa_e = pa[e]
    slip_per_side = slip_r * pa_e
    half_adv = spread / 2.0 + slip_per_side

    long_mask = dirn_e > 0
    entry_z = entry_c.copy()
    entry_z[long_mask] += half_adv[long_mask]
    entry_z[~long_mask] -= half_adv[~long_mask]

    z = c[np.minimum(e[:, None] + np.arange(1, HMAX + 1), n - 1)] * dirn_e[:, None]
    ext = np.maximum.accumulate(z, axis=1)
    wave = ext - entry_z[:, None]
    back = ext - z
    req = np.full_like(wave, np.nan, dtype=float)
    ok = wave > 0
    req[ok] = back[ok] / wave[ok]

    req_h = req[:, :H]
    hit = req_h >= trail
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1) + 1
    exit_k = np.where(any_hit, first, H).astype(int)
    idx = np.minimum(e + exit_k, n - 1)
    exit_c = c[idx]

    exit_z = exit_c.copy()
    exit_z[long_mask] -= half_adv[long_mask]
    exit_z[~long_mask] += half_adv[~long_mask]

    rr = (exit_z - entry_z) * dirn_e / pa_e

    if len(rr) < 30:
        return None

    wins = rr[rr > 0]
    losses = rr[rr < 0]
    wr = 100.0 * float((rr > 0).mean())
    pf = float(wins.sum() / (-losses.sum())) if len(losses) > 0 and losses.sum() < 0 else float("inf")
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0
    rr_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")

    eq = np.cumsum(rr)
    dd = float((np.maximum.accumulate(eq) - eq).max())

    years = {}
    for idx_y, r_val in zip(y[3:n - 1][m], rr):
        years.setdefault(idx_y, []).append(r_val)

    return dict(
        trd=len(rr), wr=wr, pf=pf, net=float(rr.sum()),
        exp=float(rr.mean()), dd=dd,
        avg_win=avg_win, avg_loss=avg_loss, rr_ratio=rr_ratio,
        years=years,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="XAUUSD")
    p.add_argument("--train-start", type=int, default=2023)
    p.add_argument("--train-end", type=int, default=2024)
    p.add_argument("--valid", type=int, default=2025)
    p.add_argument("--oos", type=int, default=2026)
    p.add_argument("--min-trades", type=int, default=100)
    p.add_argument("--min-pf", type=float, default=1.0)
    args = p.parse_args()

    sym = args.symbols.strip().upper()

    print(f"\n{'='*90}")
    print(f"  PF SWEEP: {sym} | train {args.train_start}-{args.train_end} "
          f"| valid {args.valid} | OOS {args.oos}")
    print(f"  grid: pull {len(PULLS)} x trail {len(TRAILS)} x hold {len(HORIZONS)} "
          f"= {len(PULLS)*len(TRAILS)*len(HORIZONS)} combos")
    print(f"{'='*90}\n")

    print("  Loading data...", flush=True)
    b = build(sym, args.train_start, args.oos)
    if b is None:
        print("  No data!")
        return

    btr = dict(b, year_mask=(b["t"].year >= args.train_start)
               & (b["t"].year <= args.train_end))
    bv = dict(b, year_mask=(b["t"].year == args.valid))
    try:
        bo = build(sym, args.oos, args.oos)
        bo = dict(bo, year_mask=(bo["t"].year >= args.oos))
    except Exception:
        bo = None

    results = []
    total = len(PULLS) * len(TRAILS) * len(HORIZONS)
    done = 0
    for pull in PULLS:
        for trail in TRAILS:
            for hz in HORIZONS:
                done += 1
                rt = eval_config(btr, args.train_start, args.train_end,
                                 pull, trail, hz)
                if not rt or rt["trd"] < args.min_trades or rt["pf"] < args.min_pf:
                    continue
                rv = eval_config(bv, args.train_start, args.valid,
                                 pull, trail, hz)
                if not rv or rv["pf"] < 1.0:
                    continue
                ro = eval_config(bo, args.oos, args.oos, pull, trail, hz) if bo else None
                if not ro or ro["pf"] < 1.0:
                    continue

                geo = (rt["pf"] * rv["pf"] * ro["pf"]) ** (1.0 / 3.0)
                results.append((geo, rt, rv, ro, pull, trail, hz))

    results.sort(key=lambda x: -x[0])

    print(f"\n  {len(results)} configs survived all 3 periods at PF>=1.0\n")
    print(f"  {'pull':>5} {'trail':>5} {'hold':>4} | {'geo':>5} "
          f"{'tPF':>5} {'tWR':>5} {'tRR':>5} {'tExR':>6} {'tNet':>8} "
          f"| {'vPF':>5} {'vWR':>5} "
          f"| {'oPF':>5} {'oWR':>5} {'oExR':>6} {'oNet':>8}")
    print(f"  {'-'*90}")

    for geo, rt, rv, ro, pull, trail, hz in results[:20]:
        print(f"  {pull:>5.2f} {trail:>5.2f} {hz:>4} | {geo:>5.2f} "
              f"{rt['pf']:>5.2f} {rt['wr']:>5.1f} {rt['rr_ratio']:>5.2f} "
              f"{rt['exp']:>+6.3f} {rt['net']:>+8.1f} "
              f"| {rv['pf']:>5.2f} {rv['wr']:>5.1f} "
              f"| {ro['pf']:>5.2f} {ro['wr']:>5.1f} {ro['exp']:>+6.3f} "
              f"{ro['net']:>+8.1f}")

    if results:
        print(f"\n  TOP 5 RECOMMENDED CONFIGS:")
        for i, (geo, rt, rv, ro, pull, trail, hz) in enumerate(results[:5]):
            print(f"\n  #{i+1}: pull={pull} trail={trail} hold={hz}")
            print(f"    Train: PF {rt['pf']:.2f} WR {rt['wr']:.1f}% "
                  f"R:R {rt['rr_ratio']:.2f} expR {rt['exp']:+.3f} "
                  f"net {rt['net']:+.0f}R ({rt['trd']} trades)")
            print(f"    Valid: PF {rv['pf']:.2f} WR {rv['wr']:.1f}% "
                  f"R:R {rv['rr_ratio']:.2f} net {rv['net']:+.0f}R")
            if ro:
                print(f"    OOS:   PF {ro['pf']:.2f} WR {ro['wr']:.1f}% "
                      f"R:R {ro['rr_ratio']:.2f} net {ro['net']:+.0f}R")

    print(f"\n  DEPLOYED config (pull=0.30 trail=0.15 hold=12):")
    rt_d = eval_config(btr, args.train_start, args.train_end, 0.30, 0.15, 12)
    rv_d = eval_config(bv, args.train_start, args.valid, 0.30, 0.15, 12)
    ro_d = eval_config(bo, args.oos, args.oos, 0.30, 0.15, 12) if bo else None
    if rt_d:
        print(f"    Train: PF {rt_d['pf']:.2f} WR {rt_d['wr']:.1f}% "
              f"R:R {rt_d['rr_ratio']:.2f} net {rt_d['net']:+.0f}R "
              f"({rt_d['trd']} trades)")
    if rv_d:
        print(f"    Valid: PF {rv_d['pf']:.2f} WR {rv_d['wr']:.1f}% "
              f"net {rv_d['net']:+.0f}R")
    if ro_d:
        print(f"    OOS:   PF {ro_d['pf']:.2f} WR {ro_d['wr']:.1f}% "
              f"net {ro_d['net']:+.0f}R")


if __name__ == "__main__":
    main()
