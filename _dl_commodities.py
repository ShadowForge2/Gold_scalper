import sys, time, os
sys.path.insert(0, "app")
from dukascopy_client import DukascopyClient

SYMS = {
    "XAGUSD": "data/dukascopy_xag",
    "BRENT": "data/dukascopy_brent",
    "WTI": "data/dukascopy_wti",
    "GAS": "data/dukascopy_gas",
    "COPPER": "data/dukascopy_copper",
    "XPT": "data/dukascopy_xpt",
}

FROM_YEAR = int(os.getenv("FROM_YEAR", "2015"))
TO_YEAR = int(os.getenv("TO_YEAR", "2026"))
SYM_FILTER = os.getenv("SYM_FILTER", "")

for sym, cache_dir in SYMS.items():
    if SYM_FILTER and sym not in SYM_FILTER.split(","):
        continue
    c = DukascopyClient(sym, cache_dir=cache_dir)
    for year in range(FROM_YEAR, TO_YEAR + 1):
        p = os.path.join(cache_dir, f"{sym}_M1_{year}.parquet")
        if os.path.exists(p):
            print(f"[skip] {sym} {year}", flush=True)
            continue
        t0 = time.time()
        df = c.download_year(year)
        if len(df) == 0:
            print(f"[no-data] {sym} {year} (%.0fs)" % (time.time() - t0), flush=True)
        else:
            print(f"[ok] {sym} {year}: {len(df)} bars (%.0fs)" % (time.time() - t0), flush=True)
print("DONE")
