"""Shared M1 parquet loader for the surviving backtest scripts.

Moved out of `_train_candle_h1.py` (deleted with the dead H1 candle-engine
strategy) so the live pull-scalper backtests keep working:
  _bt_pull_prevh1.py, _tune_pull_prevh1.py
"""

import pandas as pd
from pathlib import Path

SYMBOL_MAP = {
    "XAUUSD": "data/dukascopy",
    "XAGUSD": "data/dukascopy_xag",
    "XAG": "data/dukascopy_xag",
    "BRENT": "data/dukascopy_brent",
    "WTI": "data/dukascopy_wti",
    "GAS": "data/dukascopy_gas",
    "COPPER": "data/dukascopy_copper",
    "XPT": "data/dukascopy_xpt",
    "US100": "data/dukascopy_us100",
    "US500": "data/dukascopy_us500",
    "US30": "data/dukascopy_us30",
    "JP225": "data/dukascopy_jp225",
    "DE40": "data/dukascopy_de40",
}


def load_m1_data(symbol: str, start_year: int = None, end_year: int = None) -> pd.DataFrame:
    data_dir = SYMBOL_MAP.get(symbol)
    if data_dir is None:
        raise ValueError(f"Unknown symbol: {symbol}")
    path = Path(data_dir)
    frames = []
    prefix = symbol + "_"
    for f in sorted(path.glob(f"{prefix}*.parquet")):
        try:
            y = int(str(f.name).split("_")[-1].split(".")[0])
        except Exception:
            y = None
        if y is not None:
            if start_year is not None and y < start_year:
                continue
            if end_year is not None and y > end_year:
                continue
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "tick_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def resample_h1(m1: pd.DataFrame, tf_min: int = 60) -> pd.DataFrame:
    rule = f"{tf_min}min" if tf_min != 60 else "1h"
    idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
    h1 = idx.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum",
    }).dropna()
    return h1
