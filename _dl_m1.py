"""Download M1 XAUUSD data from Dukascopy, convert to CSV."""
import urllib.request
import lzma
import struct
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

SYMBOL = "XAUUSD"
OUTPUT = os.path.join(os.path.dirname(__file__), "xauusd_m1.csv")

def download_hour(symbol, year, month, day, hour):
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        compressed = resp.read()
        if len(compressed) < 10:
            return None
        data = lzma.decompress(compressed)
        return data
    except Exception as e:
        return None

def parse_ticks(binary, base_dt):
    """Parse binary bi5 data into tick rows."""
    n = len(binary) // 20
    ticks = []
    for i in range(n):
        off = i * 20
        delta_ms = struct.unpack(">I", binary[off:off+4])[0]
        ask = struct.unpack(">f", binary[off+4:off+8])[0]
        bid = struct.unpack(">f", binary[off+8:off+12])[0]
        ask_vol = struct.unpack(">f", binary[off+12:off+16])[0]
        bid_vol = struct.unpack(">f", binary[off+16:off+20])[0]
        ts = base_dt + timedelta(milliseconds=delta_ms)
        ticks.append((ts, ask, bid, ask_vol, bid_vol))
    return ticks

def ticks_to_m1(ticks):
    """Aggregate ticks into M1 OHLCV bars."""
    if not ticks:
        return []
    df = pd.DataFrame(ticks, columns=["time", "ask", "bid", "ask_vol", "bid_vol"])
    mid = (df["ask"] + df["bid"]) / 2
    df["price"] = mid
    df["time"] = df["time"].dt.floor("1min")
    ohlc = df.groupby("time").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        tick_volume=("price", "count"),
    ).reset_index()
    return ohlc.to_records(index=False)

# Try downloading a few days to see if it works
start = datetime(2025, 1, 2)
end = datetime(2025, 1, 5)

all_ticks = []
cur = start
while cur < end:
    for h in range(24):
        print(f"  {cur.date()} {h:02d}h...", end=" ", flush=True)
        raw = download_hour(SYMBOL, cur.year, cur.month, cur.day, h)
        if raw and len(raw) >= 20:
            ticks = parse_ticks(raw, cur.replace(hour=h))
            all_ticks.extend(ticks)
            print(f"{len(ticks)} ticks")
        else:
            print("no data")
    cur += timedelta(days=1)

print(f"\nTotal ticks: {len(all_ticks)}")
if all_ticks:
    m1 = ticks_to_m1(all_ticks)
    df = pd.DataFrame(m1, columns=["time", "open", "high", "low", "close", "tick_volume"])
    print(f"M1 bars: {len(df)}")
    print(df.head(10))
    print(f"\n... to {df.tail(3)}")
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")
else:
    print("No data downloaded - Dukascopy may be blocked")
