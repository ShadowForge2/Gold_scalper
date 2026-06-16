#!/usr/bin/env python3
"""Walk-forward optimisation: test same strategy across multiple time periods."""

import sys, os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import backtest as bt
from backtest import run_backtest

PERIODS = [
    ("2023 Full Year", datetime(2023, 1, 1),           datetime(2023, 12, 31, 23, 59)),
    ("2024 Full Year", datetime(2024, 1, 1),           datetime(2024, 12, 31, 23, 59)),
    ("2025 H1",        datetime(2025, 1, 1),           datetime(2025, 6, 30, 23, 59)),
    ("2025 H2",        datetime(2025, 7, 1),           datetime(2025, 12, 31, 23, 59)),
    ("2025 Full Year", datetime(2025, 1, 1),           datetime(2025, 12, 31, 23, 59)),
]

results = []

for label, start, end in PERIODS:
    bt.BACKTEST_START = start
    bt.BACKTEST_END = end
    bt.BIAS_WARMUP_DAYS = 14

    print(f"\n{'='*60}")
    print(f"  PERIOD: {label}   ({start.date()} -> {end.date()})")
    print(f"{'='*60}")

    d = bt.load_and_compute()
    if d is None:
        print(f"  SKIP: no data for {label}")
        results.append((label, None))
        continue

    r = run_backtest(d, verbose=False)
    if r is None:
        print(f"  SKIP: no trades for {label}")
        results.append((label, None))
        continue

    results.append((label, r))
    print(f"  => ${r['net_pnl']:>8.2f}   WR={r['win_rate']:>5.1f}%   PF={r['profit_factor']:>5.2f}   "
          f"Trades={r['trades']:>4}   MaxDD=${r['max_dd']:>7.2f}   Ret={r['return_pct']:>9.2f}%")
    print(f"     AvgWin=${r['avg_win']:>6.2f}   AvgLoss=${r['avg_loss']:>6.2f}   "
          f"GP=${r['gross_profit']:>8.2f}   GL=${r['gross_loss']:>8.2f}")

print(f"\n{'='*100}")
print(f"  WALK-FORWARD COMPARISON")
print(f"{'='*100}")
hdr = f"{'Period':<16} {'Trades':<7} {'WR%':<6} {'PF':<7} {'Net P&L':<12} {'Max DD':<12} {'AvgWin':<10} {'AvgLoss':<10} {'Return%':<12}"
print(hdr)
print('-' * len(hdr))
for label, r in results:
    if r is None:
        print(f"{label:<16} {'N/A':<59}")
    else:
        print(f"{label:<16} {r['trades']:<7} {r['win_rate']:<6} {r['profit_factor']:<7.2f} "
              f"${r['net_pnl']:<9.2f} ${r['max_dd']:<9.2f} ${r['avg_win']:<7.2f} ${r['avg_loss']:<7.2f} "
              f"{r['return_pct']:<10.2f}%")
print("-" * len(hdr))
print("Note: 2023/2024 results assume same $20 starting balance for comparison.")
print("Done.")
