import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import backtest as bt
from datetime import datetime
import config as cfg

bt.BACKTEST_BROKER = 'YAHOO'
bt.BACKTEST_YEAR = 2025
bt.BACKTEST_START = datetime(2025, 1, 1)
bt.BACKTEST_END = datetime(2025, 12, 31, 23, 59)
bt.INITIAL_BALANCE = 20.0

print(f'Current config:')
print(f'  SIGNAL_ENTRY_THRESHOLD = {cfg.SIGNAL_ENTRY_THRESHOLD}')
print(f'  EXIT_THRESHOLD_TIGHT = {cfg.EXIT_THRESHOLD_TIGHT}')
print(f'  LOT_MULTIPLIER = {cfg.LOT_MULTIPLIER}')
print(f'  MAX_TRADES_PER_EVENT = {cfg.MAX_TRADES_PER_EVENT}')
print(f'  ALLOWED_SESSIONS = {cfg.ALLOWED_SESSIONS}')
print(f'  BIAS_STRENGTH_MIN = {cfg.BIAS_STRENGTH_MIN}')
print(f'  MAX_DAILY_LOSS_USD = {cfg.MAX_DAILY_LOSS_USD}')
print(f'  MAX_EVENT_LOSS_USD = {cfg.MAX_EVENT_LOSS_USD}')
print(f'  SIGNAL_TIMEFRAME = {cfg.SIGNAL_TIMEFRAME}')

print(f'\nBroker={bt.BACKTEST_BROKER} Year={bt.BACKTEST_YEAR}')
print('Loading and pre-computing...')
d = bt.load_and_compute()
if d:
    print('\nRunning backtest...')
    r = bt.run_backtest(d, {'exit_mode': 1}, verbose=True)
    if r:
        print('\nDone.')
    else:
        print('No trades generated')
else:
    print('Data loading failed')
