import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from app.dukascopy_client import DukascopyClient

client = DukascopyClient()
data_dir = "data/dukascopy"
os.makedirs(data_dir, exist_ok=True)

for year in range(2000, 2027):
    fname = f"{data_dir}/XAUUSD_M1_{year}.parquet"
    if os.path.exists(fname):
        sz = os.path.getsize(fname)
        if sz > 1000:
            print(f"  {year}: already cached ({sz:,} bytes)")
            continue
    print(f"  Downloading {year}...")
    df = client.download_year(year)
    print(f"  {year}: {len(df)} bars")
