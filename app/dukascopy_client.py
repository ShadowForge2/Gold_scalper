"""
Dukascopy historical data client.
Wraps dukascopy-python with local caching and standard format.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from dukascopy_python import fetch, OFFER_SIDE_BID
from dukascopy_python import (
    INTERVAL_MIN_1, INTERVAL_MIN_5, INTERVAL_MIN_10, INTERVAL_MIN_15,
    INTERVAL_MIN_30, INTERVAL_HOUR_1, INTERVAL_HOUR_4, INTERVAL_DAY_1,
)
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD, INSTRUMENT_FX_MAJORS_EUR_USD

SYMBOL_MAP = {
    "XAUUSD": INSTRUMENT_FX_METALS_XAU_USD,
    "GOLD": INSTRUMENT_FX_METALS_XAU_USD,
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
}

TF_MAP = {
    1: INTERVAL_MIN_1,
    5: INTERVAL_MIN_5,
    10: INTERVAL_MIN_10,
    15: INTERVAL_MIN_15,
    30: INTERVAL_MIN_30,
    16385: INTERVAL_HOUR_1,
    16408: INTERVAL_HOUR_4,
    16415: INTERVAL_DAY_1,
}


class DukascopyClient:
    def __init__(self, symbol: str = "XAUUSD", cache_dir: str = "data/dukascopy"):
        self.symbol = symbol.upper()
        self.cache_dir = cache_dir
        self._cache: dict = {}

    def _cache_path(self, year: int) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{self.symbol}_M1_{year}.parquet")

    def download_year(self, year: int) -> pd.DataFrame:
        """Download M1 data for a full year + 2-day buffer."""
        cache_file = self._cache_path(year)
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)

        instr = SYMBOL_MAP.get(self.symbol)
        if instr is None:
            print(f"  ERROR: symbol {self.symbol} not in SYMBOL_MAP", flush=True)
            return pd.DataFrame()

        print(f"  Downloading {self.symbol} M1 for {year}...", flush=True)
        try:
            df = fetch(
                instrument=instr,
                interval=INTERVAL_MIN_1,
                offer_side=OFFER_SIDE_BID,
                start=start,
                end=end,
                max_retries=7,
                limit=30000,
                debug=False,
            )
        except Exception as e:
            print(f"  ERROR downloading {year}: {e}", flush=True)
            return pd.DataFrame()

        if df is None or len(df) == 0:
            print(f"  No data for {year}", flush=True)
            return pd.DataFrame()

        df = df.reset_index()
        df.rename(columns={"timestamp": "time"}, inplace=True)
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["tick_volume"] = (df["volume"] * 1000).astype(np.int32)
        df["spread"] = 0
        df["real_volume"] = 0
        df.drop(columns=["volume"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df.to_parquet(cache_file, index=False)
        print(f"  Cached {len(df)} M1 bars to {cache_file}", flush=True)
        return df

    def download_range(self, from_year: int, to_year: int) -> pd.DataFrame:
        """Download M1 data for a range of years, returns combined DataFrame."""
        chunks = []
        for year in range(from_year, to_year + 1):
            df = self.download_year(year)
            if len(df) > 0:
                chunks.append(df)
        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks, ignore_index=True)
        combined.drop_duplicates(subset="time", keep="last", inplace=True)
        combined.sort_values("time", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined

    def resample_to(self, m1_df: pd.DataFrame, timeframe: int) -> pd.DataFrame:
        """Resample M1 data to a higher timeframe.
        timeframe: 5 (M5), 16385 (H1), 16408 (H4), 16415 (D1)
        """
        if len(m1_df) == 0:
            return m1_df

        tf_map_str = {
            5: "5min",
            16385: "1h",
            16408: "4h",
            16415: "1D",
        }

        rule = tf_map_str.get(timeframe)
        if rule is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        df = m1_df.copy()
        df.set_index("time", inplace=True)

        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "spread": "mean",
            "real_volume": "sum",
        }).dropna()

        resampled.reset_index(inplace=True)
        resampled["open"] = resampled["open"].round(3)
        resampled["high"] = resampled["high"].round(3)
        resampled["low"] = resampled["low"].round(3)
        resampled["close"] = resampled["close"].round(3)
        resampled["spread"] = resampled["spread"].round(0).astype(int)
        resampled["tick_volume"] = resampled["tick_volume"].astype(np.int32)
        resampled["real_volume"] = resampled["real_volume"].astype(np.int32)
        resampled.sort_values("time", inplace=True)
        resampled.reset_index(drop=True, inplace=True)
        return resampled
