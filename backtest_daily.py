"""
Backtest 2025 and report every trade individually with timestamps.
Saves full trade log to CSV + prints daily performance summary.
"""

import sys, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from app.dukascopy_client import DukascopyClient
import backtest as bt

INITIAL_BALANCE = 20.0
WARMUP_DAYS = 21
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "dukascopy")


def main():
    client = DukascopyClient(cache_dir=CACHE_DIR)

    print("=" * 70)
    print("GOLD SCALPER - 2025 Daily Trade Report")
    print(f"Balance: ${INITIAL_BALANCE:.2f}")
    print(f"Config: entry={cfg.SIGNAL_ENTRY_THRESHOLD} exit={cfg.EXIT_THRESHOLD_TIGHT}")
    print(f"        lot_mult={cfg.LOT_MULTIPLIER} sessions={cfg.ALLOWED_SESSIONS}")
    print("=" * 70)

    print("\nLoading 2025 data...")
    all_m1 = client.download_range(2024, 2025)
    print(f"Total M1 bars: {len(all_m1)}")
    if len(all_m1) < 1000:
        print("Insufficient data."); return

    year_start = datetime(2025, 1, 1)
    year_end = datetime(2025, 12, 31, 23, 59)
    warmup_start = year_start - timedelta(days=WARMUP_DAYS)

    mask = (all_m1["time"] >= warmup_start) & (all_m1["time"] < year_end)
    m1 = all_m1[mask].copy()
    h1 = client.resample_to(m1, 16385)
    m5 = client.resample_to(m1, 5)
    print(f"M5 bars: {len(m5)}  H1 bars: {len(h1)}")

    print("\nPre-computing signals...")
    data = bt._pre_compute(m5, h1)
    if data is None:
        print("Pre-compute failed."); return

    bt.BACKTEST_START = year_start
    bt.BACKTEST_END = year_end
    bt.INITIAL_BALANCE = INITIAL_BALANCE

    params = {
        "entry_threshold": cfg.SIGNAL_ENTRY_THRESHOLD,
        "exit_threshold": cfg.EXIT_THRESHOLD_TIGHT,
        "base_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
        "sessions": cfg.ALLOWED_SESSIONS,
        "bias_strength_min": cfg.BIAS_STRENGTH_MIN,
        "lot_multiplier": cfg.LOT_MULTIPLIER,
        "max_total_lot_per_event": cfg.MAX_LOT,
        "max_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
        "event_loss_limit": cfg.MAX_EVENT_LOSS_USD,
        "consecutive_loss_limit": cfg.MAX_CONSECUTIVE_LOSSES,
        "daily_loss_limit": cfg.MAX_DAILY_LOSS_USD,
        "max_trades_per_session": cfg.MAX_TRADES_PER_SESSION,
        "min_balance": cfg.MIN_BALANCE,
        "cash_flows": [],
        "exit_mode": 1,
    }

    print("\nRunning backtest...")
    t0 = datetime.now()
    result = bt.run_backtest(data, params=params, verbose=False)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Done in {elapsed:.0f}s\n")

    if result is None:
        print("No trades."); return

    trades_df = result["trades_df"]
    # Keep copies of the raw datetime columns for CSV
    trades_df["entry_time_raw"] = trades_df["entry_time"]
    trades_df["exit_time_raw"] = trades_df["exit_time"]
    # Format for display
    trades_df["entry_time"] = trades_df["entry_time"].apply(
        lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, 'strftime') else str(t)
    )
    trades_df["exit_time"] = trades_df["exit_time"].apply(
        lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, 'strftime') else str(t)
    )

    # ── Summary ──
    print("=" * 90)
    print("  FULL YEAR SUMMARY")
    print("=" * 90)
    print(f"  Total trades:    {result['trades']}")
    print(f"  Wins:            {result['wins']}")
    print(f"  Losses:          {result['losses']}")
    print(f"  Win rate:        {result['win_rate']}%")
    print(f"  Net P&L:         ${result['net_pnl']:.2f}")
    print(f"  Profit factor:   {result['profit_factor']}")
    print(f"  Max drawdown:    ${result['max_dd']:.2f}")
    print(f"  Ending balance:  ${result['ending_balance']:.2f}")
    print(f"  Avg bars held:   {result['avg_bars_held']}")
    print()

    # ── Daily breakdown ──
    trades_df["day"] = trades_df["exit_time"].str[:10]
    daily = trades_df.groupby("day").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
    )
    daily["losses"] = daily["trades"] - daily["wins"]
    daily["wr"] = (daily["wins"] / daily["trades"] * 100).round(1)
    daily = daily.sort_index()

    print("=" * 100)
    print(f"  {'Date':<12} {'Trades':<7} {'Wins':<5} {'Losses':<7} {'WR%':<6} {'PnL':<12} {'Avg PnL':<10}")
    print("=" * 100)
    for day, row in daily.iterrows():
        avg_pnl = row["pnl"] / row["trades"] if row["trades"] > 0 else 0
        print(f"  {day:<12} {int(row['trades']):<7} {int(row['wins']):<5} {int(row['losses']):<7} "
              f"{row['wr']:<5.1f}% ${row['pnl']:<+8.2f} ${avg_pnl:<+7.2f}")
    print("=" * 100)
    tot_avg = result['net_pnl'] / result['trades'] if result['trades'] > 0 else 0
    print(f"  {'TOTAL':<12} {result['trades']:<7} {result['wins']:<5} {result['losses']:<7} "
          f"{result['win_rate']:<5.1f}% ${result['net_pnl']:<+8.2f} ${tot_avg:<+7.2f}")
    print()

    # ── Monthly breakdown ──
    trades_df["month"] = trades_df["exit_time"].str[:7]
    monthly = trades_df.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
    )
    monthly["losses"] = monthly["trades"] - monthly["wins"]
    monthly["wr"] = (monthly["wins"] / monthly["trades"] * 100).round(1)

    print("=" * 100)
    print(f"  {'Month':<10} {'Trades':<7} {'Wins':<5} {'Losses':<7} {'WR%':<6} {'PnL':<12} {'Avg PnL':<10}")
    print("=" * 100)
    for month, row in monthly.iterrows():
        avg_pnl = row["pnl"] / row["trades"] if row["trades"] > 0 else 0
        print(f"  {month:<10} {int(row['trades']):<7} {int(row['wins']):<5} {int(row['losses']):<7} "
              f"{row['wr']:<5.1f}% ${row['pnl']:<+8.2f} ${avg_pnl:<+7.2f}")
    print("=" * 100)
    print(f"  {'TOTAL':<10} {result['trades']:<7} {result['wins']:<5} {result['losses']:<7} "
          f"{result['win_rate']:<5.1f}% ${result['net_pnl']:<+8.2f} ${tot_avg:<+7.2f}")
    print()

    # ── All trades log ──
    print("=" * 120)
    print("  TRADE-BY-TRADE LOG")
    print("=" * 120)
    hdr = (f"  {'#':<4} {'Entry':<16} {'Exit':<16} {'Dir':<5} {'EntryPx':<10} "
           f"{'ExitPx':<10} {'PnL':<12} {'Lot':<8} {'N':<3} {'Bars':<5} {'Reason':<16} {'Bal':<10}")
    print(hdr)
    print("  " + "-" * 116)

    for i, (_, t) in enumerate(trades_df.iterrows()):
        print(f"  {i+1:<4} {t['entry_time']:<16} {t['exit_time']:<16} {t['direction']:<5} "
              f"{t['entry_price']:<10} {t['exit_price']:<10} ${t['pnl']:<+8.2f} "
              f"{t['lot']:<8.5f} {int(t['num_trades']):<3} {int(t['bars_held']):<5} "
               f"{t['exit_reason']:<16} "
              f"${t['balance']:<7.2f}")

    print()

    # ── Save to CSV ──
    csv_path = os.path.join(os.path.dirname(__file__), "backtest_2025_trades.csv")
    csv_cols = [c for c in trades_df.columns if not c.endswith("_raw")]
    trades_df[csv_cols].to_csv(csv_path, index=False)
    print(f"Full trade log saved: {csv_path}")

    csv_daily = os.path.join(os.path.dirname(__file__), "backtest_2025_daily.csv")
    daily.to_csv(csv_daily)
    print(f"Daily breakdown saved: {csv_daily}")

    csv_monthly = os.path.join(os.path.dirname(__file__), "backtest_2025_monthly.csv")
    monthly.to_csv(csv_monthly)
    print(f"Monthly breakdown saved: {csv_monthly}")

    print("\nDone.")


if __name__ == "__main__":
    main()
