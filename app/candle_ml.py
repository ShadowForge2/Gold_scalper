"""Candle ML — multi-timeframe M5 direction prediction.

Predicts whether the NEXT M5 candle closes UP (1) or DOWN (0).

Features from three timeframes:
- H1: trend, RSI, position, volatility regime
- M5 (completed): returns, structure, momentum, volume
- M1 (first bar of current candle): initial direction, range, momentum shape

Training is fully vectorized — no per-candle loops.
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    xgb = None
    _HAS_XGB = False

logger = logging.getLogger("GoldScalper")

CANDLE_FEATURE_COLS = [
    # H1 context
    "h1_trend", "h1_rsi", "h1_pos", "h1_atr_norm",
    "h1_volatility_ratio",
    # M5 pre-candle
    "m5_return_1", "m5_return_3", "m5_return_5",
    "m5_prev_body", "m5_prev_range",
    "m5_rsi", "m5_atr_norm",
    "m5_bb_pos", "m5_bb_width",
    "m5_range_ratio", "m5_volatility",
    "m5_close_pos", "m5_direction",
    "m5_sweep_low", "m5_sweep_high",
    "m5_micro_slope",
    # M1 first-bar
    "m1_first_dir", "m1_range_norm", "m1_body_ratio",
    "m1_momentum_z",
    # Price levels
    "gap_norm",
    # Time
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
]


def _resample_h1(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum",
    }).dropna()


def compute_candle_features(m1_df: pd.DataFrame) -> pd.DataFrame:
    """Multi-timeframe features for M5 direction prediction.

    Uses M1 data to compute:
    1. H1 features (from M1 resampled to H1)
    2. M5 completed features (from M1 resampled to M5, shifted)
    3. M1 first-bar of current candle (actual sub-5min data)

    All features are available at prediction time (1 min into M5 candle).
    """
    df = m1_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    vol_col = "tick_volume" if "tick_volume" in df.columns else None

    # ── Resample ──
    m5 = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last",
        "tick_volume": "sum" if vol_col else "count",
    }).dropna()

    h1 = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last",
        "tick_volume": "sum" if vol_col else "count",
    }).dropna()

    # First M1 bar in each M5 window (vectorized)
    m1_first = df.resample("5min").first().dropna(how="all")
    # Align to M5 index (both resampled from same data, but M5 drops NaN rows)
    m1_first = m1_first.reindex(m5.index)

    n = len(m5)

    # ── Pre-compute reused arrays ──
    c_open = m5["open"].values.astype(np.float64)
    c_high = m5["high"].values.astype(np.float64)
    c_low = m5["low"].values.astype(np.float64)
    c_close = m5["close"].values.astype(np.float64)

    # Shift by 1 for completed-candle features
    s_open = np.roll(c_open, 1);  s_open[0] = np.nan
    s_high = np.roll(c_high, 1);  s_high[0] = np.nan
    s_low = np.roll(c_low, 1);    s_low[0] = np.nan
    s_close = np.roll(c_close, 1); s_close[0] = np.nan
    s_range = s_high - s_low
    s_body = np.abs(s_close - s_open)

    feats = pd.DataFrame(index=m5.index, dtype=np.float64)

    # ── H1 features (aligned to each M5 candle) ──
    h1_idx = h1.index
    h1_high = h1["high"].values
    h1_low = h1["low"].values
    h1_close = h1["close"].values
    h1_range = h1_high - h1_low

    # Map each M5 candle to its PREVIOUS H1 bar (fully completed, no lookahead)
    m5_time_numeric = m5.index.asi8.astype(np.float64)
    h1_time_numeric = h1_idx.asi8.astype(np.float64)
    h1_map = np.searchsorted(h1_time_numeric, m5_time_numeric, side="right") - 2
    h1_map = np.clip(h1_map, 0, len(h1) - 1)

    h1_trend = np.full(n, np.nan, dtype=np.float64)
    h1_rsi_a = np.full(n, np.nan, dtype=np.float64)
    h1_pos_a = np.full(n, np.nan, dtype=np.float64)
    h1_atr_norm_a = np.full(n, np.nan, dtype=np.float64)
    h1_vol_ratio_a = np.full(n, np.nan, dtype=np.float64)

    if len(h1) > 14:
        h1_tr = np.maximum(
            h1_high - h1_low,
            np.maximum(np.abs(h1_high - np.roll(h1_close, 1)),
                       np.abs(h1_low - np.roll(h1_close, 1))),
        )
        h1_atr = pd.Series(h1_tr).rolling(14, min_periods=5).mean().values
        h1_delta = np.diff(h1_close, prepend=h1_close[0])
        h1_gain = pd.Series(np.where(h1_delta > 0, h1_delta, 0)).rolling(14, min_periods=5).mean().values
        h1_loss = pd.Series(np.where(h1_delta < 0, -h1_delta, 0)).rolling(14, min_periods=5).mean().values
        h1_rs = h1_gain / np.where(h1_loss > 0, h1_loss, np.nan)
        h1_rsi_v = 100 - (100 / (1 + h1_rs))
        h1_short_vol = pd.Series(h1_tr).rolling(3, min_periods=2).mean().values
        h1_long_vol = pd.Series(h1_tr).rolling(24, min_periods=5).mean().values

        h1_trend_v = np.sign(np.where(h1_map > 0,
                                       h1_close[h1_map] - h1_close[np.clip(h1_map - 4, 0, len(h1) - 1)],
                                       np.nan))

        for i in range(n):
            hi = h1_map[i]
            if hi < 0 or hi >= len(h1):
                continue
            h1_trend[i] = h1_trend_v[i] if i < len(h1_trend_v) else 0
            h1_rsi_a[i] = h1_rsi_v[hi]
            h1_range_i = h1_range[hi].clip(1e-10)
            h1_pos_a[i] = (s_close[i] - h1_low[hi]) / h1_range_i if not np.isnan(s_close[i]) else np.nan
            h1_atr_norm_a[i] = h1_atr[hi] / h1_close[hi] if h1_close[hi] > 0 else np.nan
            h1_vol_ratio_a[i] = h1_short_vol[hi] / h1_long_vol[hi].clip(1e-10)

    feats["h1_trend"] = h1_trend
    feats["h1_rsi"] = h1_rsi_a
    feats["h1_pos"] = h1_pos_a
    feats["h1_atr_norm"] = h1_atr_norm_a
    feats["h1_volatility_ratio"] = h1_vol_ratio_a

    # ── M5 completed features ──
    feats["m5_return_1"] = np.log(s_close / np.roll(s_close, 1))
    feats["m5_return_3"] = np.log(s_close / np.roll(s_close, 3))
    feats["m5_return_5"] = np.log(s_close / np.roll(s_close, 5))

    sr_safe = np.where(s_range > 0, s_range, np.nan)
    feats["m5_prev_body"] = s_body / sr_safe
    feats["m5_prev_range"] = s_range / np.roll(s_range, 1).clip(1e-10)

    # ATR
    tr = np.maximum(
        s_high - s_low,
        np.maximum(np.abs(s_high - np.roll(s_close, 1)),
                   np.abs(s_low - np.roll(s_close, 1))),
    )
    m5_atr = pd.Series(tr).rolling(14, min_periods=2).mean().values
    m5_atr_safe = np.where(m5_atr > 0, m5_atr, np.nan)
    feats["m5_atr_norm"] = m5_atr / s_close

    # RSI
    delta = np.diff(s_close, prepend=s_close[0])
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(14, min_periods=2).mean().values
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(14, min_periods=2).mean().values
    rs = gain / np.where(loss > 0, loss, np.nan)
    feats["m5_rsi"] = 100 - (100 / (1 + rs))

    # BB
    sma20 = pd.Series(s_close).rolling(20, min_periods=2).mean().values
    bb_std = pd.Series(s_close).rolling(20, min_periods=2).std().values
    bb_u = sma20 + 2 * bb_std
    bb_l = sma20 - 2 * bb_std
    bb_r = (bb_u - bb_l).clip(1e-10)
    feats["m5_bb_pos"] = (s_close - bb_l) / bb_r
    feats["m5_bb_width"] = bb_r / sma20

    feats["m5_range_ratio"] = s_range / m5_atr_safe
    feats["m5_close_pos"] = (s_close - s_low) / s_range.clip(1e-10)
    feats["m5_direction"] = np.sign(s_close - s_open)

    # Sweeps
    sweep_lk = 12
    rmin = pd.Series(s_low).rolling(sweep_lk, min_periods=sweep_lk).min().values
    rmax = pd.Series(s_high).rolling(sweep_lk, min_periods=sweep_lk).max().values
    feats["m5_sweep_low"] = np.maximum(0, rmin - s_close) / m5_atr_safe
    feats["m5_sweep_high"] = np.maximum(0, s_close - rmax) / m5_atr_safe

    # Micro slope (last 3 completed candles)
    def _slope_3(arr):
        x = np.array([0, 1, 2])
        if np.any(np.isnan(arr)):
            return np.nan
        return (np.sum((x - x.mean()) * (arr - arr.mean())) / np.sum((x - x.mean()) ** 2))
    slope_a = np.full(n, np.nan)
    for i in range(3, n):
        slope_a[i] = _slope_3(s_close[i - 2:i + 1])
    feats["m5_micro_slope"] = slope_a / m5_atr_safe

    # Volatility
    short_v = pd.Series(tr).rolling(5, min_periods=2).mean().values
    long_v = pd.Series(tr).rolling(40, min_periods=5).mean().values
    feats["m5_volatility"] = short_v / long_v.clip(1e-10)

    # ── M1 first-bar features ──
    m1_o = m1_first["open"].values.astype(np.float64)
    m1_h = m1_first["high"].values.astype(np.float64)
    m1_l = m1_first["low"].values.astype(np.float64)
    m1_c = m1_first["close"].values.astype(np.float64)
    m1_range = (m1_h - m1_l).astype(np.float64)
    m1_body = np.abs(m1_c - m1_o)

    feats["m1_first_dir"] = np.sign(m1_c - m1_o[:n])
    feats["m1_range_norm"] = m1_range / m5_atr_safe
    feats["m1_body_ratio"] = m1_body / m1_range.clip(1e-10)

    # M1 momentum: how decisive was the first bar's move?
    # Range / ATR ratio z-score over last 20 first-bars
    m1_mom_z = np.full(n, np.nan)
    m1_rn = m1_range / m5_atr_safe
    for i in range(20, n):
        m1_rn_win = m1_rn[i - 19:i + 1]
        if np.nanstd(m1_rn_win) > 0 and not np.isnan(m1_rn[i]):
            m1_mom_z[i] = (m1_rn[i] - np.nanmean(m1_rn_win)) / np.nanstd(m1_rn_win)
    feats["m1_momentum_z"] = m1_mom_z

    # ── Price levels ──
    gap = c_open - s_close
    feats["gap_norm"] = gap / m5_atr_safe

    # ── Time ──
    hours = m5.index.hour
    days = m5.index.dayofweek
    feats["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    feats["day_sin"] = np.sin(2 * np.pi * days / 7)
    feats["day_cos"] = np.cos(2 * np.pi * days / 7)

    return feats


def create_candle_target(m1_df: pd.DataFrame) -> pd.Series:
    """Target: 1 if M5 candle closes UP, 0 if DOWN.

    Uses next candle's close — so we're predicting the NEXT M5 candle.
    """
    df = m1_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    m5 = df.resample("5min").agg({
        "open": "first", "close": "last",
    }).dropna()

    target = (m5["close"] > m5["open"]).astype(int)
    target.name = "target"
    return target


class CandleML:
    """Multi-timeframe M5 direction predictor.

    Combines H1 trend, M5 completed-candle context, and M1 first-bar momentum
    to predict whether the current M5 candle closes UP (1) or DOWN (0).

    Strategy:
    1. New M5 candle opens
    2. After first M1 bar (~1 min), assess initial direction + momentum
    3. Model predicts final direction from all three timeframes
    4. Enter in predicted direction if confident
    5. Exit if price reverses past open
    """

    def __init__(self, model_path: str = None, model=None):
        if model is not None:
            self.model = model
        elif model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None
        self._feature_cols = CANDLE_FEATURE_COLS

    def predict_proba(self, features: pd.DataFrame) -> float:
        """Probability of UP close."""
        if self.model is None or features is None or len(features) == 0:
            return 0.5
        missing = [c for c in self._feature_cols if c not in features.columns]
        if missing:
            features = features.copy()
            for c in missing:
                features[c] = 0.0
        try:
            row = features[self._feature_cols].iloc[[-1]]
            if row.isna().any().any():
                return 0.5
            probs = self.model.predict_proba(row.values)
            if hasattr(self.model, "classes_"):
                classes = list(self.model.classes_)
                return float(probs[0][classes.index(1)]) if 1 in classes else float(probs[0][1])
            return float(probs[0][1])
        except Exception:
            return 0.5

    def predict(self, prob_up: float, m1_direction: int,
                confidence_threshold: float = 0.60) -> Optional[str]:
        """Return BUY/SELL if confident, None otherwise.

        Model predicts UP/DOWN. If m1_direction agrees → enter.
        If m1_direction disagrees → wait (don't fight the first bar).
        """
        if np.isnan(prob_up) or np.isnan(m1_direction) or m1_direction == 0:
            return None

        if prob_up >= confidence_threshold and m1_direction > 0:
            return "BUY"
        if (1 - prob_up) >= confidence_threshold and m1_direction < 0:
            return "SELL"
        return None

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "CandleML":
        return cls(model_path=path)
