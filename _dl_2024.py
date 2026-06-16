"""Download 2024 XAUUSD ticks from Dukascopy, aggregate to M5 CSV."""
import sys, os, lzma, struct, urllib.request, calendar
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

YEAR = 2024
SYMBOL = "XAUUSD"
OUT = os.path.join(os.path.dirname(__file__), f"_xauusd_m5_{YEAR}.csv")

def build_hours(year):
    hours = []
    for m in range(1, 13):
        for d in range(1, calendar.monthrange(year, m)[1] + 1):
            for h in range(24):
                hours.append((year, m, d, h))
    return hours

def dl(args):
    y, m, d, h = args
    url = f"https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=30).read()
        if len(raw) < 20: return None
        return (m, d, h, lzma.decompress(raw))
    except: return None

hours = build_hours(YEAR)
print(f"Downloading {YEAR} ({len(hours)} hours) from Dukascopy (30 threads)...")

results = {}
with ThreadPoolExecutor(max_workers=30) as ex:
    fut_map = {ex.submit(dl, h): h for h in hours}
    done = 0
    for fut in as_completed(fut_map):
        r = fut.result()
        if r:
            m, d, h, data = r
            results[(m, d, h)] = data
        done += 1
        if done % 300 == 0:
            print(f"  {done}/{len(hours)} ({len(results)} with data)")

print(f"  {len(results)}/{len(hours)} hours with data")

print("Parsing and aggregating to M5 by month...")
chunks = []
for m in range(1, 13):
    days_in = calendar.monthrange(YEAR, m)[1]
    month_ticks = []
    for d in range(1, days_in + 1):
        for h in range(24):
            raw = results.get((m, d, h))
            if raw:
                base = datetime(YEAR, m, d, h)
                n = len(raw) // 20
                for i in range(n):
                    off = i * 20
                    delta = struct.unpack(">I", raw[off:off+4])[0]
                    ask = struct.unpack(">f", raw[off+4:off+8])[0]
                    bid = struct.unpack(">f", raw[off+8:off+12])[0]
                    ts = base + timedelta(milliseconds=delta)
                    month_ticks.append((ts, (ask + bid) / 2))
    if month_ticks:
        df = pd.DataFrame(month_ticks, columns=["time", "price"])
        df["time"] = pd.to_datetime(df["time"]).dt.floor("5min")
        m5 = df.groupby("time").agg(
            open=("price", "first"), high=("price", "max"),
            low=("price", "min"), close=("price", "last"),
            tick_volume=("price", "count"),
        ).reset_index()
        m5.columns = ["time", "open", "high", "low", "close", "tick_volume"]
        m5["spread"] = 0
        m5["real_volume"] = 0
        chunks.append(m5)
        print(f"  {YEAR}-{m:02d}: {len(m5)} M5 bars ({len(month_ticks):,} ticks)")

if chunks:
    full = pd.concat(chunks, ignore_index=True)
    full.drop_duplicates(subset="time", keep="last", inplace=True)
    full.sort_values("time", inplace=True)
    full.reset_index(drop=True, inplace=True)
    full.to_csv(OUT, index=False)
    print(f"\nSaved {len(full)} M5 bars to {OUT}")
else:
    print("No data downloaded!")
