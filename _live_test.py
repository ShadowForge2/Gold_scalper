"""Proper test: user's live config. Jan 1-10, 2025 only for speed."""
import sys, os, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
import backtest

# User's live config
backtest.INITIAL_BALANCE = 20.0
backtest.LEVERAGE = 100
backtest.MARGIN_RATE = 0.01  # broker's margin_rate for gold at 100:1
backtest.BACKTEST_START = datetime(2025, 1, 1)
backtest.BACKTEST_END = datetime(2025, 1, 10, 23, 59)

# Live features ON
cfg.ML_ENABLED = True
cfg.META_ENABLED = True

print("Loading (10 days only)...", flush=True); t0 = time.time()
data = backtest.load_and_compute()
print(f"  Done in {time.time()-t0:.0f}s\n", flush=True)

r = backtest.run_backtest(data, {"exit_mode": 5}, verbose=True)
if r:
    df = r['trades_df']
    print(f"\n{'='*60}")
    print(f"  RESULTS: {r['trades']} trades  WR={r['win_rate']:.1f}%  PF={r['profit_factor']:.2f}")
    print(f"  PnL=${r['net_pnl']:+.2f}  Ret={r['return_pct']:.1f}%  DD=${r['max_dd']:.2f}")
    print(f"  AvgBars={r['avg_bars_held']:.1f}")
    print(f"  Exit reasons: {df['exit_reason'].value_counts().to_dict()}")
    print(f"  Avg lot={df['lot'].mean():.4f}  Avg n={df['num_trades'].mean():.1f}")
