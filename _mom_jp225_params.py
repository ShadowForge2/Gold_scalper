"""Per-symbol parameter adaptation for JP225 momentum-jump.

JP225 responds to the vol>=55 gate (+47.8R vs -48.7), but 2023/24 still
bleed slightly. Sweep strategy params (surge strength, jump target, max
hold, retrace) + vol-gate threshold to find a profile with all years
positive (or near-zero) and a reasonable trade count.
"""
import os
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t
from _two_engine import us100_jump_trades

SYMBOL = "JP225"
YEARS = [2022, 2023, 2024, 2025]
ATR_N = 96


def build_vol_gate(atr):
    a = atr.values.astype(np.float64)
    n = len(a)
    vol = np.full(n, 0.5)
    if n > ATR_N:
        win = sliding_window_view(a, ATR_N + 1)
        cur = win[:, -1]
        vol[ATR_N:] = (win[:, :-1] <= cur[:, None]).mean(axis=1)
    return vol


def load_year(year):
    m1 = t.load_m1_data(SYMBOL, start_year=year, end_year=year)
    m5 = t.resample_m5(m1)
    m5 = t.compute_features(m5)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    vol = build_vol_gate(atr)
    return m5, atr, vol


def scan(params, vol_th, cache):
    mz_min, jump, hold, retr = params
    total_r = 0.0
    total_n = 0
    years_r = []
    for y in YEARS:
        m5, atr, vol = cache[y]
        gate = vol >= vol_th
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

print(f"{'mz':>3} {'jump':>5} {'hold':>4} {'retr':>5} {'vol>=':>4} | "
      f"{'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'TOT':>7} {'n':>4}")
print("-" * 78)
results = []
for mz_min in (2.0, 2.5, 3.0):
    for jump in (0.75, 1.0, 1.25):
        for hold in (12, 18, 24):
            for retr in (0.25, 0.5):
                for vol_th in (0.50, 0.55, 0.60):
                    r, n, yrs = scan((mz_min, jump, hold, retr), vol_th, cache)
                    # prefer all-years >= 0, then higher total
                    results.append((r, n, (mz_min, jump, hold, retr, vol_th), yrs))

def score(row):
    r, n, p, yrs = row
    neg = sum(1 for v in yrs if v < -1.0)
    return (neg, -r)

results.sort(key=score)
print("\nAll-years-positive (or nearly), best total first:")
for r, n, p, yrs in results:
    if all(v >= -1.0 for v in yrs):
        print(f"{p[0]:>3.1f} {p[1]:>5.2f} {p[2]:>4d} {p[3]:>5.2f} {p[4]:>4.2f} | "
              + " ".join(f"{v:+7.1f}" for v in yrs) + f" | {r:+7.1f} | {n:>4}")

print("\nTop 15 by total (any shape):")
for r, n, p, yrs in sorted(results, reverse=True)[:15]:
    print(f"{p[0]:>3.1f} {p[1]:>5.2f} {p[2]:>4d} {p[3]:>5.2f} {p[4]:>4.2f} | "
          + " ".join(f"{v:+7.1f}" for v in yrs) + f" | {r:+7.1f} | {n:>4}")
