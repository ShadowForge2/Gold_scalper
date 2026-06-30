import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from app.dukascopy_client import DukascopyClient

client = DukascopyClient(symbol="EURUSD")
df = client.download_range(2024, 2024)

def ema(values, period):
    if len(values) < period:
        return np.full_like(values, np.nan)
    mult = 2.0/(period+1)
    res = np.empty_like(values)
    res[:period] = np.mean(values[:period])
    for i in range(period, len(values)):
        res[i] = (values[i] - res[i-1]) * mult + res[i-1]
    return res

def count_bias(idx_from, idx_to, label):
    chunk = df.iloc[idx_from:idx_to]
    c = chunk["close"].values
    e5 = ema(c, 5)
    e13 = ema(c, 13)
    b, s, n = 0, 0, 0
    for i in range(20, len(c)):
        if e5[i] > e13[i] and c[i] > e13[i]: b += 1
        elif e5[i] < e13[i] and c[i] < e13[i]: s += 1
        else: n += 1
    total = b + s + n
    print(f"{label}: B={b} S={s} N={n} (tradeable={b+s}/{total})")

# Jan 2 full day
jan2 = df[df["time"].dt.date == pd.Timestamp("2024-01-02").date()]
i0 = jan2.index[0]
i1 = jan2.index[-1] + 1
count_bias(i0, i1, "Jan 2 full day")

# Jan 3 full day
jan3 = df[df["time"].dt.date == pd.Timestamp("2024-01-03").date()]
i0 = jan3.index[0]
i1 = jan3.index[-1] + 1
count_bias(i0, i1, "Jan 3 full day")

# March 8 (NFP)
mar8 = df[df["time"].dt.date == pd.Timestamp("2024-03-08").date()]
if len(mar8) > 0:
    i0 = mar8.index[0]
    i1 = mar8.index[-1] + 1
    count_bias(i0, i1, "Mar 8 full day")

# Any day in June
jun12 = df[df["time"].dt.date == pd.Timestamp("2024-06-12").date()]
if len(jun12) > 0:
    i0 = jun12.index[0]
    i1 = jun12.index[-1] + 1
    count_bias(i0, i1, "Jun 12 full day")

# Sample every 100th bar across full year
print("\nFull year sample (every 100 bars):")
c = df["close"].values
e5 = ema(c, 5)
e13 = ema(c, 13)
b, s, n = 0, 0, 0
skip = 100
for i in range(20, len(c), skip):
    if e5[i] > e13[i] and c[i] > e13[i]: b += 1
    elif e5[i] < e13[i] and c[i] < e13[i]: s += 1
    else: n += 1
total = b + s + n
print(f"Sampled (every 100): B={b} S={s} N={n} tradeable={b+s}/{total} pct={(b+s)/total*100:.1f}%")
