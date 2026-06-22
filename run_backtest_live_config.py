"""
Backtest wrapper using live Render env settings.
Uses Yahoo Finance data (no pre-download needed).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override backtest.py constants BEFORE importing it
import backtest as bt
bt.BACKTEST_BROKER = "DUKASCOPY"
bt.BACKTEST_YEAR = 2024
bt.BACKTEST_START = __import__('datetime').datetime(2024, 1, 1)
bt.BACKTEST_END = __import__('datetime').datetime(2024, 12, 31, 23, 59)
bt.INITIAL_BALANCE = 20.0

import config as cfg

print("=" * 70)
print("BACKTEST: LIVE ENV CONFIGURATION")
print("=" * 70)
print(f"  BROKER:                  {cfg.BROKER}")
print(f"  SYMBOL:                  {cfg.SYMBOL}")
print(f"  SIGNAL_ENTRY_THRESHOLD:  {cfg.SIGNAL_ENTRY_THRESHOLD}")
print(f"  EXIT_THRESHOLD_TIGHT:    {cfg.EXIT_THRESHOLD_TIGHT}")
print(f"  LOT_MULTIPLIER:          {cfg.LOT_MULTIPLIER}")
print(f"  MAX_DAILY_LOSS_USD:      {cfg.MAX_DAILY_LOSS_USD}")
print(f"  MAX_EVENT_LOSS_USD:      {cfg.MAX_EVENT_LOSS_USD}")
print(f"  MAX_TRADES_PER_EVENT:    {cfg.MAX_TRADES_PER_EVENT}")
print(f"  MAX_TRADES_PER_SESSION:  {cfg.MAX_TRADES_PER_SESSION}")
print(f"  MAX_CONSECUTIVE_LOSSES:  {cfg.MAX_CONSECUTIVE_LOSSES}")
print(f"  RE_ENTRY_COOLDOWN_SEC:   {cfg.RE_ENTRY_COOLDOWN_SEC}")
print(f"  MIN_BALANCE:             {cfg.MIN_BALANCE}")
print(f"  MAX_LOT:                 {cfg.MAX_LOT}")
print(f"  MIN_LOT:                 {cfg.MIN_LOT}")
print(f"  LOT_SIZE:                {cfg.LOT_SIZE}")
print(f"  MAX_SPREAD_PIPS:         {cfg.MAX_SPREAD_PIPS}")
print(f"  SIGNAL_TIMEFRAME:        {cfg.SIGNAL_TIMEFRAME}")
print(f"  ALLOWED_SESSIONS:        {cfg.ALLOWED_SESSIONS}")
print(f"\n  Data source: Yahoo Finance (GC=F)")
print(f"  Period: {bt.BACKTEST_START.date()} to {bt.BACKTEST_END.date()}")
print(f"  Initial balance: ${bt.INITIAL_BALANCE:.2f}")
print("=" * 70)

print("\nLoading data and computing signals...")
data = bt.load_and_compute()
if data is None:
    print("FAILED: Could not load data")
    sys.exit(1)

print("\nRunning backtest...")
result = bt.run_backtest(data, verbose=True)

if result is None:
    print("No trades generated.")
    sys.exit(0)

print("\n" + "=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)
print(f"  Total Trades:      {result['trades']}")
print(f"  Wins:              {result['wins']}")
print(f"  Losses:            {result['losses']}")
print(f"  Win Rate:          {result['win_rate']}%")
print(f"  Net P&L:           ${result['net_pnl']:.2f}")
print(f"  Profit Factor:     {result['profit_factor']}")
print(f"  Max Drawdown:      ${result['max_dd']:.2f}")
print(f"  Avg Win:           ${result['avg_win']:.2f}")
print(f"  Avg Loss:          ${result['avg_loss']:.2f}")
print(f"  Ending Balance:    ${result['ending_balance']:.2f}")
print(f"  Return:            {result['return_pct']}%")
print(f"  Avg Bars Held:     {result['avg_bars_held']}")

print("\n  Monthly Breakdown:")
for m, v in result['monthly'].items():
    print(f"    {m}: {v['trades']:3d} tr  ${v['pnl']:+.2f}  WR={v['wr']:.1f}%")
total_m = sum(v['pnl'] for v in result['monthly'].values())
print(f"    TOTAL: ${total_m:+.2f}")

# Save trades to CSV
trades_df = result['trades_df']
csv_path = os.path.join(os.path.dirname(__file__), "backtest_live_config_2025.csv")
trades_df.to_csv(csv_path, index=False)
print(f"\n  Trades saved to: {csv_path}")
print("=" * 70)
