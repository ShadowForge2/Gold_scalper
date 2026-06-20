"""
Realistic 2025 backtest with proper margin & position sizing.
Overrides config to enforce realistic constraints.
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from app.dukascopy_client import DukascopyClient
from datetime import datetime, timedelta
from app.risk_manager import EquityScaler
import backtest as bt
import numpy as np

INITIAL_BALANCE = 20.0
WARMUP_DAYS = 21
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "dukascopy")

# ── Override config with realistic constraints ──
cfg.MIN_BALANCE = 10.0
cfg.LOT_SIZE = 0.01
cfg.MIN_LOT = 0.01
cfg.MAX_LOT = 0.50              # 50 micro lots max
cfg.LOT_STEP = 0.01
cfg.LOT_MULTIPLIER = 1
cfg.MAX_TRADES_PER_EVENT = 2
cfg.MAX_TRADES_PER_SESSION = 5
cfg.MAX_CONSECUTIVE_LOSSES = 3
cfg.MAX_DAILY_LOSS_USD = 5.00
cfg.MAX_EVENT_LOSS_USD = 2.00

# key change: limit margin usage to 20% of equity
bt.MARGIN_MAX_PCT = 0.20

print("=" * 70)
print("GOLD SCALPER - Realistic 2025 Backtest")
print(f"Balance: ${INITIAL_BALANCE:.2f}")
print(f"Max lot: {cfg.MAX_LOT}  Margin limit: {bt.MARGIN_MAX_PCT*100:.0f}% of equity")
print(f"Daily loss limit: ${cfg.MAX_DAILY_LOSS_USD}  Event loss: ${cfg.MAX_EVENT_LOSS_USD}")
print("=" * 70)

client = DukascopyClient(cache_dir=CACHE_DIR)

print("\nLoading 2025 data...")
all_m1 = client.download_range(2024, 2025)
print(f"M1 bars: {len(all_m1)}")

year_start = datetime(2025, 1, 1)
year_end = datetime(2025, 12, 31, 23, 59)
warmup_start = year_start - timedelta(days=WARMUP_DAYS)

mask = (all_m1["time"] >= warmup_start) & (all_m1["time"] < year_end)
m1 = all_m1[mask].copy()
h1 = client.resample_to(m1, 16385)
m5 = client.resample_to(m1, 5)
print(f"M5: {len(m5)}  H1: {len(h1)}")

print("\nPre-computing signals...")
data = bt._pre_compute(m5, h1)
if data is None:
    print("Pre-compute failed."); exit(1)

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
    print("No trades."); exit(1)

trades_df = result["trades_df"]
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
print(f"  Gross profit:    ${result['gross_profit']:.2f}")
print(f"  Gross loss:      ${result['gross_loss']:.2f}")
print(f"  Net P&L:         ${result['net_pnl']:.2f}")
print(f"  Profit factor:   {result['profit_factor']}")
print(f"  Max drawdown:    ${result['max_dd']:.2f}")
print(f"  Avg win:         ${result['avg_win']:.2f}")
print(f"  Avg loss:        ${result['avg_loss']:.2f}")
print(f"  Ending balance:  ${result['ending_balance']:.2f}")
print(f"  Return:          {result['return_pct']:.2f}%")
print(f"  Avg bars held:   {result['avg_bars_held']}")
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
print(f"  {'Month':<8} {'Trades':<7} {'Wins':<5} {'Losses':<7} {'WR%':<6} {'PnL':<12} {'Avg PnL':<10} {'Cumul Bal':<11}")
print("=" * 100)
cumul = INITIAL_BALANCE
for month, row in monthly.iterrows():
    avg_pnl = row["pnl"] / row["trades"] if row["trades"] > 0 else 0
    cumul += row["pnl"]
    print(f"  {month:<8} {int(row['trades']):<7} {int(row['wins']):<5} {int(row['losses']):<7} "
          f"{row['wr']:<5.1f}% ${row['pnl']:<+8.2f} ${avg_pnl:<+7.2f} ${cumul:<8.2f}")
print("=" * 100)
tot_avg = result['net_pnl'] / result['trades'] if result['trades'] > 0 else 0
print(f"  {'TOTAL':<8} {result['trades']:<7} {result['wins']:<5} {result['losses']:<7} "
      f"{result['win_rate']:<5.1f}% ${result['net_pnl']:<+8.2f} ${tot_avg:<+7.2f} ${result['ending_balance']:<8.2f}")
print()

# ── Daily winners/losers summary ──
trades_df["day"] = trades_df["exit_time"].str[:10]
daily = trades_df.groupby("day").agg(
    trades=("pnl", "count"),
    pnl=("pnl", "sum"),
    wins=("pnl", lambda x: (x > 0).sum()),
)
daily["losses"] = daily["trades"] - daily["wins"]
daily["wr"] = (daily["wins"] / daily["trades"] * 100).round(1)
win_days = len(daily[daily["pnl"] > 0])
loss_days = len(daily[daily["pnl"] < 0])
flat_days = len(daily[daily["pnl"] == 0])

print(f"  Trading days: {len(daily)}  Win days: {win_days}  Loss days: {loss_days}  Flat: {flat_days}")
print(f"  Day win rate: {win_days/len(daily)*100:.1f}%")
print()

# ── Top 10 best/worst trades ──
print("=" * 100)
print("  TOP 10 BEST TRADES")
print("=" * 100)
print(f"  {'#':<4} {'Date':<16} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'PnL':<12} {'Reason':<20} {'Bal':<10}")
print("  " + "-" * 87)
for i, (_, t) in enumerate(trades_df.nlargest(10, "pnl").iterrows()):
    print(f"  {i+1:<4} {t['exit_time']:<16} {t['direction']:<5} {t['entry_price']:<10} {t['exit_price']:<10} ${t['pnl']:<+8.2f} {t['exit_reason']:<20} ${t['balance']:<7.2f}")

print()
print("=" * 100)
print("  TOP 10 WORST TRADES")
print("=" * 100)
print(f"  {'#':<4} {'Date':<16} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'PnL':<12} {'Reason':<20} {'Bal':<10}")
print("  " + "-" * 87)
for i, (_, t) in enumerate(trades_df.nsmallest(10, "pnl").iterrows()):
    print(f"  {i+1:<4} {t['exit_time']:<16} {t['direction']:<5} {t['entry_price']:<10} {t['exit_price']:<10} ${t['pnl']:<+8.2f} {t['exit_reason']:<20} ${t['balance']:<7.2f}")

print()
print(f"Results saved to backtest_2025_realistic.csv")
trades_df.to_csv(os.path.join(os.path.dirname(__file__), "backtest_2025_realistic.csv"), index=False)
print("Done.")
