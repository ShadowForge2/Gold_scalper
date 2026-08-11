"""
_h1_hypotheses.py — validate wave-scalping hypotheses directly on data.

Unit of edge = R (per prev completed H1 ATR), with REAL Capital.com costs in R.
The EquityScaler reproduces "many positions" (size grows with equity), so we
only measure per-R expectancy. If per-R edge is positive, the scaler turns it
into compounding profit.

Hypotheses measured (XAUUSD etc.):
  H1  M5 continuation: after a 3-close M5 impulse aligned with the FORMING H1
      candle, does M5 continue? P(cont), avg next-1/3/5 bar move in R (net).
  H2  Acceleration -> amplitude: bigger impulse steepness => bigger total wave
      to the first counter-move (bucketed, R).
  H3  Peak proxy: exiting at the first M5 counter-close keeps how much of the
      true peak? (retention %, and net R after cost).
  H5  Fade: after the wave exhausts (first counter-close), does the opposite
      side make a net positive move over next 3 M5 bars?

Usage:
  python _h1_hypotheses.py --symbol XAUUSD --start 2023 --end 2025
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _train_candle_h1 import load_m1_data
from app.candle_engine import compute_atr

ATR_PERIOD = 14
IMPULSE = 3       # M5 closes in a row, same direction
LOOKAHEAD = 6     # M5 bars ahead = 30 minutes for the "true wave"

SYMBOL_COSTS = {
    "XAUUSD": {"spread": 0.30, "commission": 0.04},
    "US100":  {"spread": 2.00, "commission": 0.00},
    "JP225":  {"spread": 10.0, "commission": 0.00},
    "DE40":   {"spread": 2.00, "commission": 0.00},
    "US500":  {"spread": 0.80, "commission": 0.00},
    "US30":   {"spread": 2.00, "commission": 0.00},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", type=int, default=2023)
    p.add_argument("--end", type=int, default=2025)
    args = p.parse_args()
    sym = args.symbol.upper()

    m1 = load_m1_data(sym, start_year=args.start - 1, end_year=args.end)
    if m1 is None or len(m1) == 0:
        print("no data")
        return

    idx = m1.set_index("time") if "time" in m1.columns else m1
    idx = idx[~idx.index.duplicated(keep="first")].sort_index()

    h1 = idx.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    atr = compute_atr(h1, ATR_PERIOD)

    m5 = idx.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    # Forming-H1 reference for every M5 bar: hour open (from M1) + prev H1 ATR.
    m1o = idx["open"].resample("1h").first().dropna()
    hour_open = m1o.reindex(m5.index, method="ffill").ffill()
    prev_atr = atr.reindex(m5.index, method="ffill").shift(1)

    o = m5["open"].values
    h = m5["high"].values
    l = m5["low"].values
    c = m5["close"].values
    ho = hour_open.values
    pa = prev_atr.values
    t = m5.index

    cost = SYMBOL_COSTS.get(sym, {"spread": 0.0, "commission": 0.0})
    rtp = cost["spread"] + 2 * cost["commission"]
    cost_r = rtp / pa

    year_mask = (t.year >= args.start) & (t.year <= args.end)
    n = len(c)

    h1_cont = {"n": 0, "cont": 0}
    h1_moves = {1: [], 3: [], 5: []}
    h2_buckets = {}
    h3_retain = []
    h3_netR = []
    h5_netR = []
    h5_pos = 0

    # H6: buy the pullback in an H1-favoring hour (mean reversion into favour)
    h6_fwd = {1: [], 3: [], 5: []}
    h6_resume = []
    h6_resume_by_depth = {}
    h6_cont = 0
    h6_n = 0

    # trailing-exit variants (H3b): net R and retained fraction of the wave
    exit_net = {}      # giveback fraction -> list of net R
    exit_ret = {}      # giveback fraction -> retained fraction of wave
    exit_net_by_steep = {}

    for i in range(1, n - 1):
        if not year_mask[i]:
            continue
        if np.isnan(pa[i]) or pa[i] <= 0:
            continue
        # forming H1 favour
        fav = c[i] - ho[i]
        if fav == 0:
            continue
        dirn = 1 if fav > 0 else -1
        # impulse: IMPULSE consecutive closes same direction, aligned with favour
        ok = True
        for k in range(1, IMPULSE):
            if i - k < 0:
                ok = False
                break
            if (c[i - k] - c[i - k - 1]) * dirn <= 0:
                ok = False
                break
        if not ok:
            continue

        # ---- H1 continuation ----
        h1_cont["n"] += 1
        if (c[i + 1] - c[i]) * dirn > 0:
            h1_cont["cont"] += 1
        for k in (1, 3, 5):
            j = min(i + k, n - 1)
            r = (c[j] - c[i]) * dirn / pa[i] - cost_r[i]
            h1_moves[k].append(r)

        # ---- H2: impulse steepness (R per bar over the impulse) ----
        steep = (c[i] - c[i - IMPULSE]) * dirn / pa[i]
        lo, hi_ = min(l[i:i + LOOKAHEAD + 1]), max(h[i:i + LOOKAHEAD + 1])
        amp = (hi_ - c[i]) * (1 if dirn > 0 else -1) if dirn > 0 else (c[i] - lo)
        amp = abs(amp) / pa[i]
        if not np.isnan(steep) and not np.isnan(amp):
            h2_buckets.setdefault(round(steep, 1), []).append(amp)

        # ---- H3: exit at first M5 counter-close vs true peak ----
        peak = (max(h[i + 1:i + LOOKAHEAD + 1]) if dirn > 0
                else min(l[i + 1:i + LOOKAHEAD + 1]))
        wave = (peak - c[i]) * dirn
        # first counter-close within lookahead
        exit_px = None
        for k in range(1, LOOKAHEAD + 1):
            if (c[i + k] - c[i]) * dirn < 0:
                exit_px = c[i + k]
                break
        if exit_px is not None and wave > 0 and pa[i] > 0:
            gain = (exit_px - c[i]) * dirn
            h3_retain.append(gain / wave)
            h3_netR.append(gain / pa[i] - cost_r[i])

        # ---- H3b: trailing giveback exits from the running extreme ----
        if wave > 0 and pa[i] > 0:
            for f in (0.15, 0.25, 0.40):
                run_ext = c[i]
                ep = None
                for k in range(1, LOOKAHEAD + 1):
                    j = i + k
                    if dirn > 0:
                        run_ext = max(run_ext, h[j])
                        pullback = run_ext - c[j]
                        frac = run_ext - c[i]
                    else:
                        run_ext = min(run_ext, l[j])
                        pullback = c[j] - run_ext
                        frac = c[i] - run_ext
                    if frac > 0 and pullback >= f * frac:
                        ep = c[j]
                        break
                if ep is None:
                    ep = c[i + LOOKAHEAD]
                r_net = (ep - c[i]) * dirn / pa[i] - cost_r[i]
                exit_net.setdefault(f, []).append(r_net)
                exit_ret.setdefault(f, []).append((ep - c[i]) * dirn / wave)
                if f == 0.25 and not np.isnan(steep):
                    exit_net_by_steep.setdefault(round(steep, 1), []).append(r_net)

        # ---- H5: fade after exhaustion ----
        if exit_px is not None:
            fade = (c[i + 1] - exit_px) * (-dirn)
            # measure opposite move over next 3 bars from the counter-close
            idx_exit = i + 1
            fade_px = None
            for k in range(1, 4):
                if idx_exit + k < n:
                    fade_px = c[idx_exit + k]
            if fade_px is not None:
                r = (fade_px - exit_px) * (-dirn) / pa[i] - cost_r[i]
                h5_netR.append(r)
                if r > 0:
                    h5_pos += 1

        # ---- H6 must run on ALL bars (pullback is the opposite of an impulse),
        # so re-run it in its own pass, independent of the impulse gate. ----
    h6_n = 0
    h6_cont = 0
    h6_fwd = {1: [], 3: [], 5: []}
    h6_resume = []
    h6_resume_by_depth = {}
    for i in range(3, n - 2):
        if not year_mask[i]:
            continue
        if np.isnan(pa[i]) or pa[i] <= 0:
            continue
        fav = c[i] - ho[i]
        if fav == 0:
            continue
        dirn = 1 if fav > 0 else -1
        j = i - 1
        depth = (c[j - 1] - c[j]) * dirn
        was_up = (c[j - 1] - ho[i]) * dirn > 0
        turning = (c[i] - c[j]) * dirn > 0
        if depth < 0.05 * pa[i] or not was_up or not turning:
            continue
        h6_n += 1
        if (c[i + 1] - c[i]) * dirn > 0:
            h6_cont += 1
        for k in (1, 3, 5):
            jj = min(i + k, n - 1)
            h6_fwd[k].append((c[jj] - c[i]) * dirn / pa[i] - cost_r[i])
        ep = None
        for k in range(1, LOOKAHEAD + 1):
            if (c[i + k] - c[i + k - 1]) * dirn > 0 and \
               (c[i + k - 1] - c[i + k - 2]) * dirn > 0:
                ep = c[i + k]
                break
        if ep is None:
            ep = c[i + LOOKAHEAD]   # force-close: count EVERY signal
        r = (ep - c[i]) * dirn / pa[i] - cost_r[i]
        h6_resume.append(r)
        h6_resume_by_depth.setdefault(round(depth / pa[i], 1), []).append(r)

    def stat(name, arr):
        arr = np.asarray(arr)
        if len(arr) == 0:
            print(f"  {name:<28} n=0")
            return
        pos = float((arr > 0).mean())
        print(f"  {name:<28} n={len(arr):>6} mean={arr.mean():>+8.3f}R "
              f"median={np.median(arr):>+8.3f}R p(pos)={pos*100:>5.1f}% "
              f"sum={arr.sum():>+9.1f}R")

    print(f"\n{sym} | M5 wave-scalp hypotheses | {args.start}-{args.end} | "
          f"cost {rtp:.2f} units/roundtrip")
    print(f"\n-- H1: M5 continuation after {IMPULSE}-close impulse aligned "
          f"with forming H1 --")
    if h1_cont["n"]:
        print(f"  P(continue next M5 close)  = "
              f"{100*h1_cont['cont']/h1_cont['n']:.1f}%  (n={h1_cont['n']})")
    for k in (1, 3, 5):
        stat(f"fwd {k} M5 move (net R)", h1_moves[k])

    print(f"\n-- H2: impulse steepness vs wave amplitude to first counter-move --")
    if h2_buckets:
        keys = sorted(h2_buckets)
        print(f"  {'steep (R)':>10} {'n':>7} {'mean amp (R)':>12}")
        for k in keys:
            arr = np.asarray(h2_buckets[k])
            print(f"  {k:>10.1f} {len(arr):>7} {arr.mean():>12.2f}")

    print(f"\n-- H3: exit at first M5 counter-close vs true peak --")
    stat("retained fraction of peak", h3_retain)
    stat("net R captured", h3_netR)

    print(f"\n-- H3b: trailing giveback exits (ride the wave, bail on X% giveback) --")
    for f in (0.15, 0.25, 0.40):
        stat(f"giveback {int(f*100)}% net R", exit_net.get(f, []))
        stat(f"giveback {int(f*100)}% retained", exit_ret.get(f, []))

    if exit_net_by_steep:
        print(f"\n-- best trailing exit, net R by impulse steepness --")
        print(f"  {'steep (R)':>10} {'n':>7} {'net R':>8}")
        for k in sorted(exit_net_by_steep):
            arr = np.asarray(exit_net_by_steep[k])
            print(f"  {k:>10.1f} {len(arr):>7} {arr.mean():>+8.3f}")

    print(f"\n-- H5: fade the opposite side after exhaustion (next 3 M5) --")
    stat("fade net R", h5_netR)
    if h5_netR:
        print(f"  P(fade > 0) = {100*h5_pos/len(h5_netR):.1f}%")

    print(f"\n-- H6: buy the pullback in an H1-favouring hour --")
    if h6_n:
        print(f"  P(next bar resumes favour) = {100*h6_cont/h6_n:.1f}%  (n={h6_n})")
    for k in (1, 3, 5):
        stat(f"fwd {k} M5 move (net R)", h6_fwd[k])
    stat("exit @ 2-close impulse resume (net R)", h6_resume)
    if h6_resume_by_depth:
        print(f"\n-- H6 resume net R by pullback depth --")
        print(f"  {'depth (R)':>10} {'n':>7} {'net R':>8}")
        for k in sorted(h6_resume_by_depth):
            arr = np.asarray(h6_resume_by_depth[k])
            print(f"  {k:>10.1f} {len(arr):>7} {arr.mean():>+8.3f}")


if __name__ == "__main__":
    main()
