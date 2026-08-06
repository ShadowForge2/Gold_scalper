"""All-symbol momentum, each scanned independently, one shared $20 balance (2025)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t
from _two_engine import us100_jump_trades, AggressiveScaler, run_usd, report

SYMBOLS = ["US100", "US500", "US30", "XAUUSD"]
YEAR = 2025


def symbol_trades(symbol):
    m1 = t.load_m1_data(symbol, start_year=YEAR, end_year=YEAR)
    m5 = t.resample_m5(m1)
    m5 = t.compute_features(m5)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    trs = us100_jump_trades(m5, atr)
    for x in trs:
        x["symbol"] = symbol
    return trs


all_trades = []
for sym in SYMBOLS:
    trs = symbol_trades(sym)
    net = sum(x["r"] for x in trs)
    all_trades += trs
    print(f"  {sym}: {len(trs)} trades, net {net:+.1f}R", flush=True)

scaler = AggressiveScaler(base_lots={s: 0.02 for s in SYMBOLS},
                          lot_mult=2.0, aggr=1.0, use_score_mult=False)
lev = {s: 20.0 for s in SYMBOLS}
rows, endb, mineq, below, first, blocked = run_usd(all_trades, 20.0, scaler, lev)

print(f"\ncombined (all scanned independently, one $20 balance):")
report("all-mom-2025", rows, endb, 20.0, mineq, below, first, blocked)
if below and first is not None:
    print(f"  -> equity went <0 at {first}, {below} trades below zero", flush=True)

for sym in SYMBOLS:
    d = pd.DataFrame([r for r in rows if r["symbol"] == sym])
    if len(d):
        usd = d["usd"].sum()
        print(f"  {sym:>6}: {len(d)} trades, ${usd:+.2f}")
