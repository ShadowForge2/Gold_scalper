"""
2025 backtest matching LIVE config:
  - EXIT_THRESHOLD_TIGHT=0.75
  - EquityScaler lot sizing (balance / 20 reference, max_by_equity = balance/5000)
  - MAX_LOT=100, MARGIN_MAX_PCT=1.0 (no margin override)
Reports every trade with daily/monthly breakdown.
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from app.dukascopy_client import DukascopyClient
from datetime import datetime, timedelta
import backtest as bt
import numpy as np

INITIAL_BALANCE = 20.0
WARMUP_DAYS = 21
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "dukascopy")

# Live config values
cfg.EXIT_THRESHOLD_TIGHT = 0.75
cfg.SIGNAL_ENTRY_THRESHOLD = 0.50
cfg.LOT_MULTIPLIER = 2
cfg.MAX_LOT = 100.00
cfg.MIN_LOT = 0.01
cfg.LOT_SIZE = 0.01
cfg.MIN_BALANCE = 10.0
cfg.MAX_TRADES_PER_EVENT = 10
cfg.MAX_TRADES_PER_SESSION = 10
cfg.MAX_CONSECUTIVE_LOSSES = 3
cfg.MAX_DAILY_LOSS_USD = 10.00
cfg.MAX_EVENT_LOSS_USD = 5.00
cfg.RE_ENTRY_COOLDOWN_SEC = 60
cfg.ALLOWED_SESSIONS = "ASIA,LONDON,NEW_YORK"
cfg.BIAS_STRENGTH_MIN = 0.30

# Use bot's own margin logic (no override)
bt.MARGIN_MAX_PCT = 1.0

print("=" * 70)
print("GOLD SCALPER - 2023 Backtest (Live Config)")
print(f"Balance: ${INITIAL_BALANCE:.2f}")
print(f"Exit threshold: {cfg.EXIT_THRESHOLD_TIGHT}  Entry threshold: {cfg.SIGNAL_ENTRY_THRESHOLD}")
print(f"Lot mult: {cfg.LOT_MULTIPLIER}  Max lot: {cfg.MAX_LOT}  Margin: {bt.MARGIN_MAX_PCT*100:.0f}%")
print(f"Sessions: {cfg.ALLOWED_SESSIONS}")
print("=" * 70)

client = DukascopyClient(cache_dir=CACHE_DIR)
print("\nLoading data...")
all_m1 = client.download_range(2022, 2023)
print(f"M1 bars: {len(all_m1)}")

year_start = datetime(2023, 1, 1)
year_end = datetime(2023, 12, 31, 23, 59)
warmup_start = year_start - timedelta(days=WARMUP_DAYS)

mask = (all_m1["time"] >= warmup_start) & (all_m1["time"] < year_end)
m1 = all_m1[mask].copy()
h1 = client.resample_to(m1, 16385)
m5 = client.resample_to(m1, 5)
print(f"M5: {len(m5)}  H1: {len(h1)}")

print("\nPre-computing signals...")
data = bt._pre_compute(m5, h1)
if data is None:
    print("Pre-compute failed."); sys.exit(1)

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
    print("No trades."); sys.exit(1)

trades_df = result["trades_df"].copy()
trades_df["entry_time"] = trades_df["entry_time"].apply(
    lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, 'strftime') else str(t)
)
trades_df["exit_time"] = trades_df["exit_time"].apply(
    lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, 'strftime') else str(t)
)

# ── Summary ──
print("=" * 80)
print("  FULL YEAR SUMMARY")
print("=" * 80)
print(f"  Total trades:    {result['trades']}")
print(f"  Wins:            {result['wins']}")
print(f"  Losses:          {result['losses']}")
print(f"  Win rate:        {result['win_rate']}%")
print(f"  Gross profit:    ${result['gross_profit']:,.2f}")
print(f"  Gross loss:      ${result['gross_loss']:,.2f}")
print(f"  Net P&L:         ${result['net_pnl']:,.2f}")
print(f"  Profit factor:   {result['profit_factor']}")
print(f"  Max drawdown:    ${result['max_dd']:,.2f}")
print(f"  Avg win:         ${result['avg_win']:,.2f}")
print(f"  Avg loss:        ${result['avg_loss']:,.2f}")
print(f"  Ending balance:  ${result['ending_balance']:,.2f}")
print(f"  Return:          {result['return_pct']:,.2f}%")
print(f"  Avg bars held:   {result['avg_bars_held']}")
print()

# ── Monthly ──
trades_df["month"] = trades_df["exit_time"].str[:7]
monthly = trades_df.groupby("month").agg(
    trades=("pnl", "count"),
    pnl=("pnl", "sum"),
    wins=("pnl", lambda x: (x > 0).sum()),
)
monthly["losses"] = monthly["trades"] - monthly["wins"]
monthly["wr"] = (monthly["wins"] / monthly["trades"] * 100).round(1)

print("=" * 105)
print(f"  {'Month':<8} {'Trades':<7} {'Wins':<5} {'Losses':<7} {'WR%':<6} {'PnL':<14} {'Avg PnL':<12} {'Cumul Bal':<14}")
print("=" * 105)
cumul = INITIAL_BALANCE
for month, row in monthly.iterrows():
    avg_pnl = row["pnl"] / row["trades"] if row["trades"] > 0 else 0
    cumul += row["pnl"]
    print(f"  {month:<8} {int(row['trades']):<7} {int(row['wins']):<5} {int(row['losses']):<7} "
          f"{row['wr']:<5.1f}% ${row['pnl']:<+10.2f} ${avg_pnl:<+9.2f} ${cumul:<10.2f}")
print("=" * 105)
tot_avg = result['net_pnl'] / result['trades'] if result['trades'] > 0 else 0
print(f"  {'TOTAL':<8} {result['trades']:<7} {result['wins']:<5} {result['losses']:<7} "
      f"{result['win_rate']:<5.1f}% ${result['net_pnl']:<+10.2f} ${tot_avg:<+9.2f} ${result['ending_balance']:<10.2f}")
print()

# ── Daily stats ──
trades_df["day"] = trades_df["exit_time"].str[:10]
daily = trades_df.groupby("day").agg(
    trades=("pnl", "count"),
    pnl=("pnl", "sum"),
    wins=("pnl", lambda x: (x > 0).sum()),
)
daily["losses"] = daily["trades"] - daily["wins"]
win_days = len(daily[daily["pnl"] > 0])
loss_days = len(daily[daily["pnl"] < 0])
print(f"  Trading days: {len(daily)}  Win days: {win_days} ({win_days/len(daily)*100:.1f}%)  Loss days: {loss_days}")

# ── Top/bottom trades ──
print("\n" + "=" * 105)
print("  TOP 10 TRADES")
print("=" * 105)
print(f"  {'#':<4} {'Date':<16} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'PnL':<14} {'Lot':<10} {'Reason':<20}")
print("  " + "-" * 89)
for i, (_, t) in enumerate(trades_df.nlargest(10, "pnl").iterrows()):
    print(f"  {i+1:<4} {t['exit_time']:<16} {t['direction']:<5} {t['entry_price']:<10} "
          f"{t['exit_price']:<10} ${t['pnl']:<+10.2f} {t['lot']:<10.5f} {t['exit_reason']:<20}")

print("\n" + "=" * 105)
print("  BOTTOM 10 TRADES")
print("=" * 105)
print(f"  {'#':<4} {'Date':<16} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'PnL':<14} {'Lot':<10} {'Reason':<20}")
print("  " + "-" * 89)
for i, (_, t) in enumerate(trades_df.nsmallest(10, "pnl").iterrows()):
    print(f"  {i+1:<4} {t['exit_time']:<16} {t['direction']:<5} {t['entry_price']:<10} "
          f"{t['exit_price']:<10} ${t['pnl']:<+10.2f} {t['lot']:<10.5f} {t['exit_reason']:<20}")

# ── Lot size distribution ──
print("\n" + "=" * 60)
print("  LOT SIZE DISTRIBUTION")
print("=" * 60)
lot_bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100]
labels = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "50", "100"]
for i in range(len(lot_bins)-1):
    lo, hi = lot_bins[i], lot_bins[i+1]
    count = ((trades_df["lot"] > lo) & (trades_df["lot"] <= hi)).sum()
    if count > 0:
        print(f"  {lo:<5}-{hi:<5}: {count} trades")
last = (trades_df["lot"] > 100).sum()
if last > 0:
    print(f"  >100    : {last} trades")

print(f"\n  Max lot used:  {trades_df['lot'].max():.5f}")
print(f"  Avg lot used:  {trades_df['lot'].mean():.5f}")
print(f"  Min lot used:  {trades_df['lot'].min():.5f}")

# ── Save ──
csv_path = os.path.join(os.path.dirname(__file__), "backtest_2023_live.csv")
trades_df.to_csv(csv_path, index=False)
print(f"\nTrades saved: {csv_path}")
print("Done.")
