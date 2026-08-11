"""
_scan_strategies.py — broad honest scan for ANY scalping rule with real edge.

Enumerates a grid of mechanical M5 scalping strategies (entry signal x exit
rule x direction filter) over all 6 live symbols, every trade forced to close
within a horizon, real Capital.com costs, fills at bar close. Reports the top
combos by net R / PF. A combo only "works" if it is net positive on MULTIPLE
pairs (an artifact pops on one pair only).

Usage:
  python _scan_strategies.py --symbols XAUUSD,US100,US30 --cost 1
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
HORIZON = 12          # max M5 bars held = 1 hour
OUT_MIN_BARS = 50     # require >= this many trades to count a combo

SYMBOL_COSTS = {
    "XAUUSD": {"spread": 0.30, "commission": 0.04},
    "US100":  {"spread": 2.00, "commission": 0.00},
    "JP225":  {"spread": 10.0, "commission": 0.00},
    "DE40":   {"spread": 2.00, "commission": 0.00},
    "US500":  {"spread": 0.80, "commission": 0.00},
    "US30":   {"spread": 2.00, "commission": 0.00},
}


def build_arrays(symbol, start, end):
    m1 = load_m1_data(symbol, start_year=start - 1, end_year=end)
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
    ho = idx["open"].resample("1h").first().dropna()
    ho = ho.reindex(m5.index, method="ffill").ffill().values
    pa = atr.reindex(m5.index, method="ffill").shift(1).values
    # prev completed H1 body direction (close vs open of the LAST closed H1)
    h1dir = np.sign(h1["close"] - h1["open"]).reindex(m5.index, method="ffill").shift(1).values
    cost = SYMBOL_COSTS.get(symbol, {"spread": 0.0, "commission": 0.0})
    rtp = cost["spread"] + 2 * cost["commission"]
    cost_r = rtp / pa
    year_mask = (t.year >= start) & (t.year <= end)
    return dict(c=c, t=t, ho=ho, pa=pa, h1dir=h1dir, cost_r=cost_r,
                year_mask=year_mask, n=len(c))


def entry_signals(b, entry_type):
    """Return per-bar direction (0/±1) for the given entry type."""
    n = b["n"]
    c = b["c"]
    pa = b["pa"]
    d = np.zeros(n, dtype=np.int8)
    if entry_type == "imp2":
        for i in range(2, n):
            if (c[i] - c[i - 1]) * (c[i - 1] - c[i - 2]) > 0:
                d[i] = 1 if c[i] - c[i - 1] > 0 else -1
    elif entry_type == "imp3":
        for i in range(3, n):
            if (c[i] - c[i - 1]) * (c[i - 1] - c[i - 2]) > 0 and \
               (c[i - 1] - c[i - 2]) * (c[i - 2] - c[i - 3]) > 0:
                d[i] = 1 if c[i] - c[i - 1] > 0 else -1
    elif entry_type == "pull":
        for i in range(3, n - 1):
            if pa[i] <= 0 or np.isnan(pa[i]):
                continue
            up = (c[i - 1] - c[i - 2]) > 0 and (c[i - 2] - c[i - 3]) > 0
            dn = (c[i - 1] - c[i - 2]) < 0 and (c[i - 2] - c[i - 3]) < 0
            if up:
                dip = c[i - 1] - c[i]
                if dip > 0.1 * pa[i] and (c[i + 1] - c[i]) > 0:
                    d[i + 1] = 1
            elif dn:
                rip = c[i] - c[i - 1]
                if rip > 0.1 * pa[i] and (c[i + 1] - c[i]) < 0:
                    d[i + 1] = -1
    elif entry_type == "break":
        for i in range(6, n):
            if pa[i] <= 0 or np.isnan(pa[i]):
                continue
            hh = max(c[i - 6:i])
            ll = min(c[i - 6:i])
            if c[i] > hh and (c[i] - hh) > 0.1 * pa[i]:
                d[i] = 1
            elif c[i] < ll and (ll - c[i]) > 0.1 * pa[i]:
                d[i] = -1
    return d


def apply_direction(d, b, mode):
    if mode == "any":
        return d
    out = np.zeros_like(d)
    if mode == "h1fav":
        fav = np.sign(b["c"] - b["ho"])
        for i in range(b["n"]):
            if fav[i] != 0 and d[i] == fav[i]:
                out[i] = d[i]
    elif mode == "prevh1":
        for i in range(b["n"]):
            if b["h1dir"][i] != 0 and d[i] == b["h1dir"][i]:
                out[i] = d[i]
    return out


def simulate(d, b, exit_rule):
    """Return array of net R per trade; force-close at HORIZON."""
    n = b["n"]
    c = b["c"]
    pa = b["pa"]
    cost_r = b["cost_r"]
    out = []
    for i in range(n):
        if d[i] == 0 or not b["year_mask"][i]:
            continue
        dirn = int(d[i])
        e = c[i]
        if pa[i] <= 0 or np.isnan(pa[i]):
            continue
        ep = None
        for k in range(1, HORIZON + 1):
            j = i + k
            if j >= n:
                break
            px = c[j]
            r = (px - e) * dirn / pa[i]
            if exit_rule == "time":
                ep = px
                break
            if exit_rule == "counter" and r < 0:
                ep = px
                break
            if exit_rule == "dirclose" and r > 0:
                ep = px
                break
            if exit_rule == "trail25":
                # running extreme close so far (from i+1..j); giveback check
                if k == 1:
                    run = px
                else:
                    run = run if dirn > 0 else run
                    run = max(run, px) if dirn > 0 else min(run, px)
                wave = (run - e) * dirn
                back = (run - px) * dirn
                if wave > 0 and back >= 0.25 * wave:
                    ep = px
                    break
            if exit_rule == "tpsl":
                if r >= 0.3 or r <= -0.3:
                    ep = px
                    break
        if ep is None:
            ep = c[min(i + HORIZON, n - 1)]
        out.append((ep - e) * dirn / pa[i] - cost_r[i])
    return np.asarray(out, dtype=float)


def stats(arr):
    if len(arr) < OUT_MIN_BARS:
        return None
    wins = float(arr[arr > 0].sum())
    losses = float((-arr[arr < 0]).sum())
    pf = wins / losses if losses > 0 else float("inf")
    return dict(trd=len(arr), wr=100 * float((arr > 0).mean()),
                exp=float(arr.mean()), pf=pf, net=float(arr.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="XAUUSD,US100,JP225,DE40,US500,US30")
    p.add_argument("--start", type=int, default=2023)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--cost", type=int, default=1)
    args = p.parse_args()

    entries = ["imp2", "imp3", "pull", "break"]
    exits = ["time", "counter", "dirclose", "trail25", "tpsl"]
    dirs = ["any", "h1fav", "prevh1"]

    results = {}
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        b = build_arrays(sym, args.start, args.end)
        print(f"\n== {sym} ==", flush=True)
        for et in entries:
            d = entry_signals(b, et)
            for dm in dirs:
                dd = apply_direction(d, b, dm)
                for ex in exits:
                    arr = simulate(dd, b, ex)
                    s = stats(arr)
                    if s is None:
                        continue
                    key = (et, dm, ex)
                    r = results.setdefault(key, [])
                    r.append((sym, s))
                    if s["net"] > 0:
                        print(f"  +{et}/{dm}/{ex}: net {s['net']:+.1f}R "
                              f"PF {s['pf']:.2f} WR {s['wr']:.0f}% "
                              f"trd {s['trd']}  [{sym}]", flush=True)

    print(f"\n\n==== TOP COMBOS BY POSITIVE-PAIR COUNT ====", flush=True)
    ranked = []
    for key, lst in results.items():
        pos = [x for x in lst if x[1]["net"] > 0]
        neg = [x for x in lst if x[1]["net"] <= 0]
        ranked.append((len(pos), sum(x[1]["net"] for x in pos),
                       key, pos, neg))
    for npairs, total, key, pos, neg in sorted(ranked, key=lambda x: (-x[0], -x[1])):
        if npairs == 0:
            continue
        print(f"\n  {key[0]}/{key[1]}/{key[2]}: + on {npairs} pair(s) "
              f"(sum {total:+.1f}R)", flush=True)
        for sym, s in pos:
            print(f"    {sym}: net {s['net']:+.1f}R PF {s['pf']:.2f} "
                  f"WR {s['wr']:.0f}% trd {s['trd']}", flush=True)
        for sym, s in neg:
            print(f"    {sym}: net {s['net']:+.1f}R (neg)", flush=True)


if __name__ == "__main__":
    main()
