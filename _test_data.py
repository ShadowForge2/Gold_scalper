"""Test MT5 data download for 2025 M5."""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import config as cfg

acct = cfg.MT5_ACCOUNT
if not mt5.initialize(login=int(acct), password=cfg.MT5_PASSWORD, server=cfg.MT5_SERVER):
    print(f"MT5 init failed: {mt5.last_error()}")
    mt5.shutdown()
    exit(1)

TIMEFRAME_M5 = 5

# Check what data is available
rates = mt5.copy_rates_from("XAUUSD", TIMEFRAME_M5, datetime.now(), 10000)
if rates is not None:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"Got {len(df)} M5 candles")
    print(f"From: {df['time'].min()} To: {df['time'].max()}")
else:
    print(f"No M5 data: {mt5.last_error()}")

# Try Jan 2025
start = datetime(2025, 1, 1)
end = datetime(2025, 1, 31)
rates2 = mt5.copy_rates_range("XAUUSD", TIMEFRAME_M5, start, end)
if rates2 is not None:
    df2 = pd.DataFrame(rates2)
    print(f"\nJan 2025: {len(df2)} M5 candles")
else:
    print(f"\nJan 2025: No data: {mt5.last_error()}")

mt5.shutdown()
