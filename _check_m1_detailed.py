"""Detailed M1 data check - try various chunk sizes and timeframes."""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

mt5.initialize(login=49717207, password='Thunder@g1', server='HFMarketsGlobal-Demo')

# Try M1 data for just the last 7 days
today = datetime.now()
for days_ago in [1, 3, 7, 14, 30]:
    start = today - timedelta(days=days_ago)
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, start, today)
    if rates is not None and len(rates) > 0:
        print(f"M1 last {days_ago:2d}d: {len(rates):>6} candles  "
              f"({pd.to_datetime(rates[0][0], unit='s')} -> {pd.to_datetime(rates[-1][0], unit='s')})")
    else:
        print(f"M1 last {days_ago:2d}d: NO DATA")

# Also try downloading M1 via copy_rates_from_pos (might work differently)
rates2 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 100)
if rates2 is not None and len(rates2) > 0:
    print(f"M1 last 100 from pos: {len(rates2)} candles  "
          f"({pd.to_datetime(rates2[0][0], unit='s')} -> {pd.to_datetime(rates2[-1][0], unit='s')})")
else:
    print(f"M1 last 100 from pos: NO DATA")

# Check M5 vs M1 availability in general
rates_m5 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 100)
rates_m1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 100)
print(f"M5 last 100: {len(rates_m5) if rates_m5 is not None else 0} candles")
print(f"M1 last 100: {len(rates_m1) if rates_m1 is not None else 0} candles")

mt5.shutdown()
