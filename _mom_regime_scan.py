"""Find the BEST regime gate PER SYMBOL for the momentum-jump engine.

Each pair has its own "conditions" that suit momentum:
  - US100/JP225: edge comes from high volatility (ATR percentile of last 8h).
  - DE40:        edge comes from trending (efficiency ratio), not raw vol.
We sweep per-symbol gates and report the best 2022-25 netR for each.

Gate definitions (M5, forward-safe):
  vol_pct(N): current ATR percentile vs last N M5 bars. High = volatile.
  er(N):       Kaufman efficiency ratio |Pn| / sum|P| over N bars. High = trending.
"""
import os
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t
from _two_engine import us100_jump_trades

SYMBOLS = ["US100", "JP225", "DE40"]
YEARS = [2022, 2023, 2024, 2025]
ATR_N = 96  # M5 bars in 8 hours
ER_N = 96


def build_gates(m5, atr):
    n = len(m5)
    one = np.ones(n, dtype=bool)
    gates = {"none": one}

    # Vectorized volatility percentile: for each bar, fraction of last ATR_N
    # bars with ATR <= current ATR.
    a = atr.values.astype(np.float64)
    vol = np.full(n, 0.5)
    if n > ATR_N:
        win = sliding_window_view(a, ATR_N + 1)
        cur = win[:, -1]
        vol[ATR_N:] = (win[:, :-1] <= cur[:, None]).mean(axis=1)
    for th in (0.50, 0.55, 0.60, 0.65, 0.70):
        gates[f"vol>={int(th*100)}"] = vol >= th

    # Efficiency ratio.
    c = m5["close"]
    net = (c - c.shift(ER_N)).abs()
    gross = c.diff().abs().rolling(ER_N).sum()
    er = (net / gross.replace(0, 1e-10)).values
    er = np.nan_to_num(er, nan=0.0)
    for th in (0.20, 0.25, 0.30, 0.35, 0.40):
        gates[f"er>={th:.2f}"] = er >= th

    # Combined: volatile AND trending.
    for vt in (0.55, 0.60):
        for et in (0.20, 0.25):
            gates[f"vol{int(vt*100)}+er{int(et*100)}"] = (vol >= vt) & (er >= et)

    return gates


def run(sym, year, gate_name, cache):
    m1 = t.load_m1_data(sym, start_year=year, end_year=year)
    m5 = t.resample_m5(m1)
    m5 = t.compute_features(m5)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    key = (sym, year)
    if key not in cache:
        cache[key] = (m5, atr, build_gates(m5, atr))
    _, _, gates = cache[key]
    trades = us100_jump_trades(m5, atr, gate=gates[gate_name])
    rs = [x["r"] for x in trades]
    n = len(rs)
    if n == 0:
        return n, 0.0, 0.0, 0.0
    wins = sum(r for r in rs if r > 0)
    losses = -sum(r for r in rs if r < 0)
    pf = wins / losses if losses else float("inf")
    return n, 100.0 * np.mean([r > 0 for r in rs]), pf, sum(rs)


cache = {}
GATE_NAMES = None
for sym in SYMBOLS:
    print(f"\n=== {sym} ===")
    print(f"{'gate':>14} | {'2022':>7} | {'2023':>7} | {'2024':>7} | {'2025':>7} | {'TOTAL':>7}")
    print("-" * 66)
    best = (None, -1e18)
    for g in ["none", "vol>=50", "vol>=55", "vol>=60", "vol>=65", "vol>=70",
              "er>=0.20", "er>=0.25", "er>=0.30", "er>=0.35", "er>=0.40",
              "vol55+er20", "vol55+er25", "vol60+er20", "vol60+er25"]:
        row = []
        total = 0.0
        for y in YEARS:
            n, wr, pf, r = run(sym, y, g, cache)
            row.append(f"{r:+7.1f}")
            total += r
        if total > best[1]:
            best = (g, total)
        print(f"{g:>14} | " + " | ".join(row) + f" | {total:+7.1f}")
    print(f"  BEST: {best[0]} -> {best[1]:+.1f}R 2022-25")
