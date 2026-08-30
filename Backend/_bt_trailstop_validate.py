"""
_bt_trailstop_validate.py — validate the broker-side trailing-stop fix against
M1 data, contrasting the exit-fill model that caused the live bleed vs. the new
broker-stop model, on IDENTICAL entries and costs.

Live bleed recap (Aug 24-28, 32 trades, -4.82 USD):
  - all trail exits were closed by a market DELETE on a 15s poll loop
  - 11 of 21 trail exits closed WORSE than entry (fill-gap blow-through)
  - trail = -4.87 (entire bleed), pump_tip = +0.27 (ok), max_hold = -0.22

Why backtest can decide this:
  Engine exits at the CLOSE of an M5 bar (trail test). On fast M5 bars that
  close is far from the eventual fill:
    OLD (live):  bot polls every 15s -> market DELETE fills somewhere past the
                 bar close, untracked. Model: fill = worst of (trigger-close,
                 adverse M1 extreme within the next poll window) => the gap.
    NEW (fixed): broker stopLevel sits at trail_stop_level(); it fills the
                 INSTANT an M1 low/high touches it, exactly at the stop. No
                 polling latency, no gap. Model: fill = stop level on touch.

Both models run on the SAME entries (engine logic from
_tune_pull_prevh1_recovered.py) and the same round-trip costs.

Usage:
  python _bt_trailstop_validate.py [--symbols XAUUSD,US100,US30] [--year 2025]
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

SYMBOL_PARAMS = {
    "XAUUSD": dict(pull_r=0.30, trail_r=0.15, max_hold=12, rtp=0.38),
    "US100":  dict(pull_r=0.30, trail_r=0.50, max_hold=6,  rtp=2.00),
    "US30":   dict(pull_r=0.30, trail_r=0.35, max_hold=24, rtp=2.00),
}


def build_m1(symbol, year):
    """Return M1 arrays: time, open, high, low, close + M5/H1 context in price terms."""
    m1 = load_m1_data(symbol, start_year=year, end_year=year)
    idx = m1.set_index("time") if "time" in m1.columns else m1
    idx = idx[~idx.index.duplicated(keep="last")].sort_index()
    m1o = idx["open"].values
    m1h = idx["high"].values
    m1l = idx["low"].values
    m1c = idx["close"].values
    m1t = idx.index

    # H1 ATR + direction at each M1 bar (use last completed H1, ffill + shift 1)
    h1 = idx.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    atr_s = compute_atr(h1, ATR_PERIOD)
    h1dir_s = np.sign(h1["close"] - h1["open"])
    atr = atr_s.reindex(idx.index, method="ffill").shift(1).values
    h1dir = h1dir_s.reindex(idx.index, method="ffill").shift(1).values

    # M5 close context (entry trigger uses M5 closes)
    m5c = idx["close"].resample("5min").last().dropna()

    return dict(t=m1t, o=m1o, h=m1h, l=m1l, c=m1c,
                atr=atr, h1dir=h1dir, m5c=m5c)


def find_m5_triggers(b, pull_r, atr_pull_mult):
    """Replicate the engine's entry-trigger detection on M5 closes.

    Returns a list of (entry_idx, dirn) where entry_idx is the M1 index of the
    M5 trigger/entry close. Mirrors _tune_pull_prevh1_recovered.run().
    """
    m5c = b["m5c"]
    t_arr = m5c.index
    c = m5c.values
    n = len(c)
    # Map each M5 close timestamp -> M1 index of that same timestamp (last M1 of the bar)
    keys = b["t"].floor("5min")
    m1_idx_by_bar = pd.Series(np.arange(len(b["t"])), index=keys).groupby(level=0).last()
    m5_to_m1 = {ts: int(m1_idx_by_bar.get(ts, -1)) for ts in t_arr}
    # Build context arrays aligned to M5 bars
    atr_m5 = pd.Series(b["atr"], index=b["t"]).reindex(t_arr, method="ffill").values
    h1_m5 = pd.Series(b["h1dir"], index=b["t"]).reindex(t_arr, method="ffill").values
    triggers = []
    for i in range(3, n - 1):
        a0 = atr_m5[i]
        d0 = h1_m5[i]
        if a0 <= 0 or np.isnan(a0) or d0 == 0 or np.isnan(d0):
            continue
        up = (c[i - 1] - c[i - 2]) > 0 and (c[i - 2] - c[i - 3]) > 0
        dn = (c[i - 1] - c[i - 2]) < 0 and (c[i - 2] - c[i - 3]) < 0
        if (up and d0 < 0) or (dn and d0 > 0) or (not up and not dn):
            continue
        dirn = 1 if up else -1
        dip = (c[i - 1] - c[i]) * dirn
        if dip < pull_r * a0:
            continue
        if (c[i + 1] - c[i]) * dirn <= 0:
            continue
        # entry is the M5 bar at i+1's close
        entry_ts = t_arr[i + 1]
        ei = m5_to_m1.get(entry_ts)
        if ei is None or ei >= len(b["t"]) - 1:
            continue
        triggers.append((ei, dirn))
    return triggers


def simulate(b, params, triggers, model):
    """Run exits on identical triggers under a given fill model.

    model = "OLD" (bar-close trail + poll-gap slippage) | "NEW" (broker stop touch)
    """
    pull_r, trail_r, max_hold, rtp = (
        params["pull_r"], params["trail_r"], params["max_hold"], params["rtp"])
    t, h, l, c, atr = b["t"], b["h"], b["l"], b["c"], b["atr"]
    n = len(c)
    out = []
    for entry_idx, dirn in triggers:
        entry = c[entry_idx]
        atr_e = atr[entry_idx]
        if atr_e <= 0 or np.isnan(atr_e):
            continue
        cost_r = rtp / atr_e
        run_ext = entry
        hold = 0
        stop = 0.0          # NEW: broker stop level
        # step M1 bars forward up to max_hold M5 bars
        max_m1 = int(max_hold) * 5
        m1_step = 0
        exited = None
        exit_k = entry_idx
        while m1_step < max_m1:
            k = entry_idx + m1_step + 1
            if k >= n:
                break
            m1_step += 1
            # NEW: ratchet broker stop (same formula as trail_stop_level) using
            # running M1 extreme, then test intra-bar touch against M1 high/low.
            if model == "NEW":
                if dirn > 0:
                    run_ext_m1 = max(run_ext, h[k]) if m1_step == 1 else max(run_ext, h[k])
                    wave = (run_ext_m1 - entry) * dirn
                    if wave > 0:
                        newstop = run_ext_m1 - (1.0 - trail_r) * wave
                        if stop == 0.0 or newstop > stop:
                            stop = newstop
                    if stop > 0 and l[k] <= stop:
                        exited = stop
                        exit_k = k
                        break
                else:
                    run_ext_m1 = min(run_ext, l[k]) if m1_step == 1 else min(run_ext, l[k])
                    wave = (run_ext_m1 - entry) * dirn
                    if wave > 0:
                        newstop = run_ext_m1 + (1.0 - trail_r) * wave
                        if stop == 0.0 or newstop < stop:
                            stop = newstop
                    if stop > 0 and h[k] >= stop:
                        exited = stop
                        exit_k = k
                        break
            # OLD model: evaluate the trail at the M5-bar close; if it triggers,
            # exit with gap slippage to the adverse M1 extreme of that bar.
            is_bar_close = (b["t"][k].floor("5min") != b["t"][k - 1].floor("5min"))
            if is_bar_close:
                cc = c[k]
                hold += 1
                run_ext = max(run_ext, cc) if dirn > 0 else min(run_ext, cc)
                wave = (run_ext - entry) * dirn
                back = (run_ext - cc) * dirn
                if wave > 0 and back >= trail_r * wave:
                    # fill gap: adverse extreme of the current M1 bar
                    gap = (l[k] - cc) if dirn > 0 else (h[k] - cc)
                    exited = cc + min(0.0, gap) if dirn > 0 else cc + max(0.0, gap)
                    exit_k = k
                    if exited == cc + 0.0 and dirn > 0:
                        exited = cc
                    break
                if hold >= max_hold:
                    exited = cc
                    exit_k = k
                    break
        if exited is None:
            # forced close at end of horizon
            kf = entry_idx + m1_step
            if kf >= n:
                kf = n - 1
            exited = c[kf]
        rr = (exited - entry) * dirn / atr_e - cost_r
        reason = "trail"  # simplified; max_hold folds in here for OLD
        out.append((rr, reason))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="XAUUSD,US100,US30")
    p.add_argument("--year", type=int, default=2025)
    args = p.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"\n  Broker trailing-stop validation (M1 fill modeling) | year {args.year}")
    print(f"  OLD = bar-close trail + poll-gap slippage   NEW = broker stop touch-fill\n")
    for sym in syms:
        params = SYMBOL_PARAMS.get(sym)
        if params is None:
            continue
        b = build_m1(sym, args.year)
        triggers = find_m5_triggers(b, params["pull_r"], 1.0)
        print(f"  {sym:>7} | M1 bars {len(b['t']):,} | triggers {len(triggers)}")
        for model in ("OLD", "NEW"):
            out = simulate(b, params, triggers, model)
            if not out:
                print(f"    {model}: no trades"); continue
            arr = np.asarray([r for r, _ in out], dtype=float)
            wins = float(arr[arr > 0].sum())
            losses = float((-arr[arr < 0]).sum())
            pf = wins / losses if losses > 0 else float("inf")
            wr = 100 * float((arr > 0).mean())
            print(f"    {model}: {len(arr)} trd | WR {wr:.1f}% | PF {pf:.2f} "
                  f"| net {arr.sum():+.1f}R | exp {arr.mean():+.3f}R")
        print()


if __name__ == "__main__":
    main()
