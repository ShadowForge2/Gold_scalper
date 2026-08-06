import pandas as pd
import numpy as np
from typing import Optional


class SignalEngine:
    def __init__(self, logger=None):
        self._logger = logger

    def _resample_to_m5(self, m1_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            if "time" in m1_data.columns:
                m1_idx = m1_data.set_index("time")
            else:
                m1_idx = m1_data
            m5 = m1_idx.resample("5min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "tick_volume": "sum",
            }).dropna()
            if len(m5) < 5:
                return None
            return m5
        except Exception:
            return None

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]),
                                   np.abs(l[1:] - c[:-1])))
        atr = float(np.mean(tr[-period:]))
        return 0.0 if np.isnan(atr) else atr

    def _compute_atr_m5(self, m1_data: pd.DataFrame, period: int = 14) -> float:
        m5 = self._resample_to_m5(m1_data)
        if m5 is None or len(m5) < period + 2:
            return 0.0
        return self._compute_atr(m5.iloc[:-1], period)
