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
        self._last_votes: Optional[float] = None

    def update(self, h1_data: pd.DataFrame) -> Dict:
        trend = self._detect_trend(h1_data)
        self.bias = trend
        if self._last_votes is not None:
            raw = abs(self._last_votes) / 1.5
            self.strength = min(raw, 1.0)
        else:
            self.strength = 0.0
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
            self._last_votes = None
            return "NEUTRAL"

        close = df["close"].astype(float)
        fast = close.ewm(span=20, adjust=False).mean()
        slow = close.ewm(span=50, adjust=False).mean()
        fast_slope = fast.iloc[-1] - fast.iloc[-6] if len(fast) >= 6 else 0.0
        slow_slope = slow.iloc[-1] - slow.iloc[-6] if len(slow) >= 6 else 0.0

        votes = 0.0
        if fast.iloc[-1] > slow.iloc[-1] and fast_slope > 0:
            votes += 1.0
        elif fast.iloc[-1] < slow.iloc[-1] and fast_slope < 0:
            votes -= 1.0

        if close.iloc[-1] > slow.iloc[-1] and slow_slope >= 0:
            votes += 0.5
        elif close.iloc[-1] < slow.iloc[-1] and slow_slope <= 0:
            votes -= 0.5

        highs, lows = [], []
        for i in range(lookback, len(df) - lookback):
            if all(df["high"].iloc[i] >= df["high"].iloc[i - j] for j in range(1, lookback + 1)) and \
               all(df["high"].iloc[i] >= df["high"].iloc[i + j] for j in range(1, lookback + 1)):
                highs.append(df["high"].iloc[i])
            if all(df["low"].iloc[i] <= df["low"].iloc[i - j] for j in range(1, lookback + 1)) and \
               all(df["low"].iloc[i] <= df["low"].iloc[i + j] for j in range(1, lookback + 1)):
                lows.append(df["low"].iloc[i])

        if len(highs) < 2 or len(lows) < 2:
            self._last_votes = votes
            if votes >= 1.0:
                return "BULLISH"
            if votes <= -1.0:
                return "BEARISH"
            return "NEUTRAL"

        recent_h = highs[-3:] if len(highs) >= 3 else highs
        recent_l = lows[-3:] if len(lows) >= 3 else lows

        h_up = sum(1 for i in range(1, len(recent_h)) if recent_h[i] > recent_h[i-1])
        h_dn = sum(1 for i in range(1, len(recent_h)) if recent_h[i] < recent_h[i-1])
        l_up = sum(1 for i in range(1, len(recent_l)) if recent_l[i] > recent_l[i-1])
        l_dn = sum(1 for i in range(1, len(recent_l)) if recent_l[i] < recent_l[i-1])

        swing_score = (h_up - h_dn) + (l_up - l_dn)
        swing_total = max(1, (len(recent_h)-1) + (len(recent_l)-1))
        votes += swing_score / swing_total

        self._last_votes = votes

        if votes >= 0.75:
            return "BULLISH"
        if votes <= -0.75:
            return "BEARISH"
        return "NEUTRAL"

    def get_bias_summary(self) -> Dict:
        return self.summary

    def is_tradeable(self) -> bool:
        return self.bias in ("BULLISH", "BEARISH") and self.strength >= 0.3
