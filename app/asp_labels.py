"""
ASP Swing Labeling Engine

Detects historical swing turning points and labels them as:
  1 = BUY  (swing low that led to profitable upside)
  -1 = SELL (swing high that led to profitable downside)
  0 = NO_TRADE

Labeling logic:
  1. Detect fractal swing highs and lows (lookback=5)
  2. At each swing, look forward up to `forward_bars` bars
  3. If price moved >= min_target ATR in the expected direction before
     hitting the SL distance in the opposite direction → label = good swing
  4. Otherwise → NO_TRADE

This creates a dataset where each bar with a detected swing gets a label
that the ML model learns to predict.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def detect_swings(highs: np.ndarray, lows: np.ndarray,
                  lookback: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Detect fractal swing highs and lows.

    Returns:
        swing_high_mask: bool array, True at swing high bars
        swing_low_mask: bool array, True at swing low bars
    """
    n = len(highs)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        window_l = lows[i - lookback:i + lookback + 1]
        if highs[i] == np.max(window_h):
            swing_high[i] = True
        if lows[i] == np.min(window_l):
            swing_low[i] = True

    return swing_high, swing_low


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Compute ATR array."""
    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
    return atr


def label_swings(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 forward_bars: int = 12, min_target_atr: float = 1.0,
                 sl_atr_mult: float = 2.0, swing_lookback: int = 5,
                 atr_period: int = 14) -> np.ndarray:
    """Label each bar: 1=BUY(swing low), -1=SELL(swing high), 0=NO_TRADE.

    For each detected swing:
      - Swing Low → BUY if price rises >= min_target_atr * ATR within forward_bars
                     before falling >= sl_atr_mult * ATR
      - Swing High → SELL if price falls >= min_target_atr * ATR within forward_bars
                      before rising >= sl_atr_mult * ATR
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)

    swing_high, swing_low = detect_swings(highs, lows, swing_lookback)
    atr = compute_atr(highs, lows, closes, atr_period)
    atr = np.nan_to_num(atr, nan=np.nanmean(atr))
    atr = np.where(atr <= 0, 1.0, atr)

    for i in range(n):
        if swing_low[i]:
            a = atr[i]
            target = min_target_atr * a
            sl_dist = sl_atr_mult * a
            # Look forward
            for j in range(1, min(forward_bars + 1, n - i)):
                high_j = highs[i + j]
                low_j = lows[i + j]
                # Check if target hit
                if high_j - closes[i] >= target:
                    labels[i] = 1  # BUY — swing low is good entry
                    break
                # Check if SL hit first
                if closes[i] - low_j >= sl_dist:
                    break  # NO_TRADE — swing low failed

        elif swing_high[i]:
            a = atr[i]
            target = min_target_atr * a
            sl_dist = sl_atr_mult * a
            for j in range(1, min(forward_bars + 1, n - i)):
                high_j = highs[i + j]
                low_j = lows[i + j]
                if closes[i] - low_j >= target:
                    labels[i] = -1  # SELL — swing high is good entry
                    break
                if high_j - closes[i] >= sl_dist:
                    break  # NO_TRADE — swing high failed

    return labels


def create_asp_labels(m5_df: pd.DataFrame, forward_bars: int = 12,
                      min_target_atr: float = 1.0, sl_atr_mult: float = 2.0,
                      swing_lookback: int = 5) -> pd.Series:
    """Create ML labels from M5 OHLCV data.

    Returns Series aligned with m5_df index:
      1  = BUY  (good swing low)
      -1 = SELL (good swing high)
      0  = NO_TRADE (no swing, or bad swing)
    """
    highs = m5_df["high"].values.astype(np.float64)
    lows = m5_df["low"].values.astype(np.float64)
    closes = m5_df["close"].values.astype(np.float64)

    labels = label_swings(highs, lows, closes,
                          forward_bars=forward_bars,
                          min_target_atr=min_target_atr,
                          sl_atr_mult=sl_atr_mult,
                          swing_lookback=swing_lookback)

    result = pd.Series(labels, index=m5_df.index, name="asp_target")
    return result
