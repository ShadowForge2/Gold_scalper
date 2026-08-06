"""Download M1 data for candidate momentum pairs (JP225, DE40) from Dukascopy."""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.dukascopy_client import DukascopyClient

PAIRS = ["JP225", "DE40"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2022)
    p.add_argument("--end-year", type=int, default=2025)
    args = p.parse_args()

    for sym in PAIRS:
        cache_dir = f"data/dukascopy_{sym.lower()}"
        client = DukascopyClient(symbol=sym, cache_dir=cache_dir)
        for year in range(args.start_year, args.end_year + 1):
            df = client.download_year(year)
            if len(df) > 0:
                print(f"  {sym} {year}: {len(df):,} M1 bars "
                      f"{df['time'].min()} -> {df['time'].max()}", flush=True)
            else:
                print(f"  {sym} {year}: NO DATA", flush=True)


if __name__ == "__main__":
    main()
