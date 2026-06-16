import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

mt5.initialize(login=49717207, password='Thunder@g1', server='HFMarketsGlobal-Demo')

for label, start, end in [
    ("Jan 2025", datetime(2025, 1, 1), datetime(2025, 1, 31)),
    ("Mar 2025", datetime(2025, 3, 1), datetime(2025, 3, 31)),
    ("Jun 2025", datetime(2025, 6, 1), datetime(2025, 6, 12)),
    ("Oct 2024", datetime(2024, 10, 1), datetime(2024, 10, 31)),
]:
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, start, end)
    if rates is not None and len(rates) > 0:
        first = pd.to_datetime(rates[0][0], unit="s")
        last = pd.to_datetime(rates[-1][0], unit="s")
        print(f"M1 {label}: {len(rates):>6} candles  ({first} -> {last})")
    else:
        err = mt5.last_error()
        print(f"M1 {label}: NO DATA  ({err})")

mt5.shutdown()
