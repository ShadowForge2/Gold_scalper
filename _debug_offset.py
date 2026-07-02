import sys; sys.path.insert(0, '.')
import numpy as np
from datetime import datetime
from app.dukascopy_client import DukascopyClient

client = DukascopyClient()
m1 = client.download_range(2021, 2022)
from_dt = datetime(2022, 1, 1)
warmup = from_dt - __import__('datetime').timedelta(days=14)
m1 = m1[(m1['time'] >= warmup) & (m1['time'] <= datetime(2022,12,31,23,59))]
h1 = client.resample_to(m1, 16385)
m5 = client.resample_to(m1, 5)

m5_t = m5['time'].values
print(f'Total M5 bars: {len(m5_t)}')
print(f'First bar: {m5_t[0]}')

jan1_idx = np.searchsorted(m5_t, np.datetime64('2022-01-01'), side='right')
print(f'searchsorted(..., side=right) idx: {jan1_idx}')
if jan1_idx > 0 and jan1_idx < len(m5_t):
    print(f'Bar before 2022-01-01: {m5_t[jan1_idx-1]}')
    print(f'First bar >= 2022-01-01: {m5_t[jan1_idx]}')
elif jan1_idx == 0:
    print('All bars are >= 2022-01-01')
elif jan1_idx >= len(m5_t):
    print('All bars are < 2022-01-01')

int_t = m5_t.astype(np.int64)
target = np.datetime64(from_dt).astype(np.int64)
offset = np.searchsorted(int_t, target, side='right')
print(f'\nInt64 offset: {offset}')
print(f'Int64 first: {int_t[0]}')
print(f'Int64 target: {target}')
print(f'Int64 bar at offset-1: {m5_t[offset-1] if offset > 0 else "N/A"}')
print(f'Int64 bar at offset: {m5_t[offset] if offset < len(m5_t) else "N/A"}')

# Try different conversions
offset2 = np.searchsorted(m5_t.astype('datetime64[us]').astype(np.int64), np.datetime64(from_dt).astype('datetime64[us]').astype(np.int64), side='right')
print(f'\nMicrosecond offset: {offset2}')

offset3 = np.searchsorted(m5_t.astype('datetime64[s]').astype(np.int64), np.datetime64(from_dt).astype('datetime64[s]').astype(np.int64), side='right')
print(f'Second offset: {offset3}')

offset4 = np.searchsorted(m5_t.astype('datetime64[ms]').astype(np.int64), np.datetime64(from_dt).astype('datetime64[ms]').astype(np.int64), side='right')
print(f'Millisecond offset: {offset4}')
