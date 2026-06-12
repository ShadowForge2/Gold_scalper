import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime


class BiasEngine:
    def __init__(self):
        self.bias: str = "NEUTRAL"
        self.strength: float = 0.0
        self.last_updated: Optional[datetime] = None
        self.summary: Dict = {}

    def update(self, h1_data: pd.DataFrame) -> Dict:
        trend = self._detect_trend(h1_data)
        self.bias = trend
        self.strength = 0.3 if trend != "NEUTRAL" else 0.0
        self.last_updated = datetime.now()

        self.summary = {
            "bias": self.bias,
            "strength": round(self.strength, 3),
            "primary_trend": trend,
            "secondary_trend": "NEUTRAL",
            "key_levels": [],
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
        return self.summary

    def _detect_trend(self, df: pd.DataFrame, lookback: int = 5) -> str:
        if df is None or len(df) < lookback * 3:
            return "NEUTRAL"

        highs, lows = [], []
        for i in range(lookback, len(df) - lookback):
            if all(df["high"].iloc[i] >= df["high"].iloc[i - j] for j in range(1, lookback + 1)) and \
               all(df["high"].iloc[i] >= df["high"].iloc[i + j] for j in range(1, lookback + 1)):
                highs.append(df["high"].iloc[i])
            if all(df["low"].iloc[i] <= df["low"].iloc[i - j] for j in range(1, lookback + 1)) and \
               all(df["low"].iloc[i] <= df["low"].iloc[i + j] for j in range(1, lookback + 1)):
                lows.append(df["low"].iloc[i])

        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL"

        recent_h = highs[-3:] if len(highs) >= 3 else highs
        recent_l = lows[-3:] if len(lows) >= 3 else lows

        h_up = sum(1 for i in range(1, len(recent_h)) if recent_h[i] >= recent_h[i-1])
        h_dn = sum(1 for i in range(1, len(recent_h)) if recent_h[i] <= recent_h[i-1])
        l_up = sum(1 for i in range(1, len(recent_l)) if recent_l[i] >= recent_l[i-1])
        l_dn = sum(1 for i in range(1, len(recent_l)) if recent_l[i] <= recent_l[i-1])

        score = (h_up - h_dn) + (l_up - l_dn)
        total = max(1, (len(recent_h)-1) + (len(recent_l)-1))

        if score / total >= 0.3:
            return "BULLISH"
        if score / total <= -0.3:
            return "BEARISH"
        return "NEUTRAL"

    def get_bias_summary(self) -> Dict:
        return self.summary

    def is_tradeable(self) -> bool:
        return self.bias in ("BULLISH", "BEARISH") and self.strength >= 0.3
