"""Per-symbol parameter adaptation for the momentum-jump engine.

US100/JP225 get a vol>=55 gate (validated +127.8R / +47.8R). DE40 doesn't
respond to gates, so we sweep strategy parameters (surge strength mz_min,
jump target, max hold, retrace) combined with the er gate to find a positive
2022-25 profile with a reasonable trade count.
"""
import os
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t
from _two_engine import us100_jump_trades

SYMBOL = "DE40"
YEARS = [2022, 2023, 2024, 2025]
ATR_N = 96
ER_N = 96


def build_er_gate(m5):
    c = m5["close"]
    net = (c - c.shift(ER_N)).abs()
    gross = c.diff().abs().rolling(ER_N).sum()
    er = (net / gross.replace(0, 1e-10)).values
    return np.nan_to_num(er, nan=0.0)


def load_year(year):
    m1 = t.load_m1_data(SYMBOL, start_year=year, end_year=year)
    m5 = t.resample_m5(m1)
    m5 = t.compute_features(m5)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    er = build_er_gate(m5)
    return m5, atr, er


def scan(params, er_th, cache):
    mz_min, jump, hold, retr = params
    total_r = 0.0
    total_n = 0
    years_r = []
    for y in YEARS:
        m5, atr, er = cache[y]
        gate = er >= er_th
        trades = us100_jump_trades(m5, atr, jump_target=jump, max_hold=hold,
                                   retr_dist=retr, mz_min=mz_min, gate=gate)
        rs = [x["r"] for x in trades]
        total_n += len(rs)
        r = sum(rs)
        total_r += r
        years_r.append(r)
    return total_r, total_n, years_r


cache = {}
for y in YEARS:
    cache[y] = load_year(y)

print(f"{'mz':>3} {'jump':>5} {'hold':>4} {'retr':>5} {'er>=':>4} | "
      f"{'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'TOT':>7} {'n':>4}")
print("-" * 78)
results = []
for mz_min in (2.0, 2.5, 3.0):
    for jump in (0.75, 1.0, 1.25):
        for hold in (12, 18, 24):
            for retr in (0.25, 0.5):
                for er_th in (0.25, 0.35, 0.45):
                    r, n, yrs = scan((mz_min, jump, hold, retr), er_th, cache)
                    results.append((r, n, (mz_min, jump, hold, retr, er_th), yrs))

results.sort(reverse=True)
for r, n, p, yrs in results[:20]:
    print(f"{p[0]:>3.1f} {p[1]:>5.2f} {p[2]:>4d} {p[3]:>5.2f} {p[4]:>4.2f} | "
          + " ".join(f"{v:+7.1f}" for v in yrs) + f" | {r:+7.1f} | {n:>4}")
