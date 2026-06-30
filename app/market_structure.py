import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


class MarketStructure:
    def __init__(self):
        self.swing_highs: List[Dict] = []
        self.swing_lows: List[Dict] = []
        self.key_levels: List[float] = []
        self.trend: str = "NEUTRAL"
        self.structure_broken: bool = False
        self.last_swing_idx: int = 0

    def analyze(self, df: pd.DataFrame, lookback: int = 5) -> Dict:
        if df is None or len(df) < lookback * 3:
            return self._summary(price=df["close"].iloc[-1] if df is not None and len(df) else None)

        highs, lows = self._find_swing_points(df, lookback)
        self.swing_highs = highs
        self.swing_lows = lows
        self.trend = self._determine_trend()
        self.key_levels = self._identify_key_levels(df)
        self.structure_broken = self._detect_structure_break()

        return self._summary(price=df["close"].iloc[-1])

    def _find_swing_points(self, df: pd.DataFrame, lookback: int = 5
                           ) -> Tuple[List[Dict], List[Dict]]:
        highs = []
        lows = []
        for i in range(lookback, len(df) - lookback):
            if all(df["high"].iloc[i] >= df["high"].iloc[i - j]
                   for j in range(1, lookback + 1)) and \
               all(df["high"].iloc[i] >= df["high"].iloc[i + j]
                   for j in range(1, lookback + 1)):
                highs.append({
                    "index": i,
                    "price": df["high"].iloc[i],
                    "time": df["time"].iloc[i],
                })
            if all(df["low"].iloc[i] <= df["low"].iloc[i - j]
                   for j in range(1, lookback + 1)) and \
               all(df["low"].iloc[i] <= df["low"].iloc[i + j]
                   for j in range(1, lookback + 1)):
                lows.append({
                    "index": i,
                    "price": df["low"].iloc[i],
                    "time": df["time"].iloc[i],
                })
        return highs, lows

    def _determine_trend(self) -> str:
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return "NEUTRAL"

        recent_highs = self.swing_highs[-3:] if len(self.swing_highs) >= 3 else self.swing_highs
        recent_lows = self.swing_lows[-3:] if len(self.swing_lows) >= 3 else self.swing_lows

        highs_up = all(
            recent_highs[i]["price"] >= recent_highs[i - 1]["price"]
            for i in range(1, len(recent_highs))
        ) if len(recent_highs) >= 2 else False

        lows_up = all(
            recent_lows[i]["price"] >= recent_lows[i - 1]["price"]
            for i in range(1, len(recent_lows))
        ) if len(recent_lows) >= 2 else False

        highs_down = all(
            recent_highs[i]["price"] <= recent_highs[i - 1]["price"]
            for i in range(1, len(recent_highs))
        ) if len(recent_highs) >= 2 else False

        lows_down = all(
            recent_lows[i]["price"] <= recent_lows[i - 1]["price"]
            for i in range(1, len(recent_lows))
        ) if len(recent_lows) >= 2 else False

        if (highs_up or lows_up) and not (highs_down or lows_down):
            return "BULLISH"
        elif (highs_down or lows_down) and not (highs_up or lows_up):
            return "BEARISH"
        return "NEUTRAL"

    def _identify_key_levels(self, df: pd.DataFrame) -> List[float]:
        levels = set()
        for swing in self.swing_highs[-6:]:
            levels.add(round(swing["price"], 3))
        for swing in self.swing_lows[-6:]:
            levels.add(round(swing["price"], 3))
        return sorted(levels)

    def _detect_structure_break(self) -> bool:
        if len(self.swing_highs) < 3 or len(self.swing_lows) < 3:
            return False
        if self.trend == "BULLISH":
            return self.swing_lows[-1]["price"] < self.swing_lows[-2]["price"]
        elif self.trend == "BEARISH":
            return self.swing_highs[-1]["price"] > self.swing_highs[-2]["price"]
        return False

    def is_near_level(self, price: float, pips: float = 10,
                      point: float = 0.0001) -> bool:
        for level in self.key_levels:
            if abs(price - level) <= pips * point:
                return True
        return False

    def nearest_level(self, price: float) -> Optional[float]:
        if not self.key_levels:
            return None
        return min(self.key_levels, key=lambda l: abs(price - l))

    def nearest_level_distance(self, price: float,
                               point: float = 0.0001) -> Optional[float]:
        level = self.nearest_level(price)
        if level is None:
            return None
        return abs(price - level) / point

    def _summary(self, price: float = None) -> Dict:
        return {
            "trend": self.trend,
            "swing_highs": self.swing_highs[-5:],
            "swing_lows": self.swing_lows[-5:],
            "key_levels": self.key_levels,
            "structure_broken": self.structure_broken,
            "near_level": self.is_near_level(price) if price is not None else False,
        }
