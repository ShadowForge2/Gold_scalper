"""Per-symbol US100 momentum-jump backtest to find which pairs actually have edge."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t
from _two_engine import us100_jump_trades

SYMBOLS = ["US100", "JP225", "DE40", "US500", "US30", "XAUUSD"]
YEARS = [2022, 2023, 2024, 2025]


def scan(symbol, year):
    m1 = t.load_m1_data(symbol, start_year=year, end_year=year)
    m5 = t.resample_m5(m1)
    m5 = t.compute_features(m5)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    trades = us100_jump_trades(m5, atr)
    rs = [x["r"] for x in trades]
    n = len(rs)
    if n == 0:
        return 0, 0, 0.0, 0.0
    wins = [r for r in rs if r > 0]
    losses = [-r for r in rs if r < 0]
    pf = sum(wins) / sum(losses) if losses else float("inf")
    return n, len(wins) / n, pf, sum(rs)


print(f"{'symbol':>7} | {'year':>4} | {'n':>4} | {'WR%':>5} | {'PF':>6} | {'netR':>8}")
print("-" * 48)
tot = {}
for sym in SYMBOLS:
    tot[sym] = {"n": 0, "r": 0.0}
    for y in YEARS:
        n, wr, pf, net = scan(sym, y)
        tot[sym]["n"] += n
        tot[sym]["r"] += net
        pf_s = "inf" if pf == float("inf") else f"{pf:6.2f}"
        print(f"{sym:>7} | {y:>4} | {n:>4} | {100*wr:>5.1f} | {pf_s:>6} | {net:+8.1f}")
    print(f"{'TOTAL':>7} | {'':>4} | {tot[sym]['n']:>4} | {'':>5} | {'':>6} | {tot[sym]['r']:+8.1f}")
    print("-" * 48)
