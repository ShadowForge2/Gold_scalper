"""
CandleBrain — Transformer model for trading decisions.

Processes a sequence of M5 bars through self-attention to make
context-aware entry and management decisions.

Architecture:
  Input:  [batch, seq_len, n_features]  (40 M5 bars × 30 features)
  Output: entry_logits [batch, 3], confidence [batch, 1], mgmt_logits [batch, 2]

Output heads:
  Entry:   P(BUY), P(SELL), P(NONE) — enter long, short, or stay flat
  Conf:    0-1 confidence score
  Mgmt:    P(HOLD), P(CLOSE) — trade management when in position
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


N_FEATURES = 34
SEQ_LEN = 25
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 1
DIM_FF = 256
DROPOUT = 0.1

FEATURE_COLS = [
    # Price in ATR units vs 50-EMA (4) — scale-free, not raw $ prices
    "open_norm", "high_norm", "low_norm", "close_norm",
    "volume",
    # Derived M5 (15)
    "rsi", "atr_norm", "bb_pos", "bb_width",
    "body_ratio", "range_ratio", "momentum_z",
    "close_pos", "direction", "sweep_high", "sweep_low",
    "micro_slope", "volatility_ratio", "trend_strength",
    "return_1",
    # Context (10)
    "h1_trend", "h1_rsi", "dist_h1_high", "dist_h1_low",
    "bars_since_swing_h", "bars_since_swing_l",
    "session_enc", "hour_sin", "hour_cos",
    "day_sin",
    # Regime (4)
    "adx_norm", "vol_regime", "squeeze", "regime_trend",
]

ENTRY_ACTIONS = {0: "BUY", 1: "SELL", 2: "NONE"}
MGMT_ACTIONS = {0: "HOLD", 1: "CLOSE"}


# ── Feature pipeline (shared with _train_candle_brain.py) ──────────

def resample_m5(m1: pd.DataFrame) -> pd.DataFrame:
    idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
    m5 = idx.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum",
    }).dropna()
    return m5


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_bb(close: pd.Series, period: int = 20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / sma.replace(0, 1e-10)
    pos = (close - lower) / (upper - lower).replace(0, 1e-10)
    return pos.clip(0, 1), width


def compute_features(m5: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    df = m5.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    rng = (h - l).replace(0, 1e-10)
    body = (c - o).abs()

    df["rsi"] = compute_rsi(c).fillna(50) / 100
    atr = compute_atr(df, atr_period).fillna(rng.rolling(atr_period).mean().fillna(rng))
    df["atr_norm"] = (atr / c.replace(0, 1e-10)).fillna(0)
    bb_pos, bb_width = compute_bb(c)
    df["bb_pos"] = bb_pos.fillna(0.5)
    df["bb_width"] = bb_width.fillna(0)
    df["body_ratio"] = (body / rng).fillna(0)
    df["range_ratio"] = (rng / atr.replace(0, 1e-10)).fillna(1).clip(0, 5)
    df["momentum_z"] = (c.pct_change(3) / c.pct_change(3).rolling(20).std().replace(0, 1e-10)).fillna(0).clip(-3, 3)
    df["close_pos"] = ((c - l) / rng).fillna(0.5)
    df["direction"] = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(float)
    df["sweep_high"] = ((h.rolling(12).max() - h) / atr.replace(0, 1e-10)).fillna(0).clip(0, 10)
    df["sweep_low"] = ((l - l.rolling(12).min()) / atr.replace(0, 1e-10)).fillna(0).clip(0, 10)
    df["micro_slope"] = (c.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.mean(x) if len(x) == 5 else 0,
        raw=True
    )).fillna(0).clip(-0.01, 0.01) * 100

    short_vol = atr.rolling(3).mean()
    long_vol = atr.rolling(24).mean()
    df["volatility_ratio"] = (short_vol / long_vol.replace(0, 1e-10)).fillna(1).clip(0.2, 3)

    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    df["trend_strength"] = ((ema20 - ema50) / atr.replace(0, 1e-10)).fillna(0).clip(-3, 3)

    # Scale-free price representation: distance from the 50-EMA, in ATR units.
    atr_safe = atr.replace(0, 1e-10)
    df["open_norm"] = ((o - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["high_norm"] = ((h - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["low_norm"] = ((l - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["close_norm"] = ((c - ema50) / atr_safe).fillna(0).clip(-8, 8)

    df["return_1"] = c.pct_change(1).fillna(0).clip(-0.05, 0.05) * 100

    df["volume"] = df.get("tick_volume", pd.Series(0, index=df.index)).fillna(0)
    df["volume"] = (df["volume"] / df["volume"].rolling(20).mean().replace(0, 1)).fillna(1).clip(0, 5)

    # ── Regime features ──────────────────────────────────────────────────
    # ADX: trend strength (0..1, 60+ = very strong trend).
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_s = atr.replace(0, 1e-10)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr_s
    minus_di = 100 * minus_dm.rolling(14).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(14).mean()
    df["adx_norm"] = (adx / 60.0).clip(0, 1).fillna(0)

    # Volatility regime: ATR vs its rolling 96-bar median (8h lookback).
    vol_med = atr.rolling(96).median()
    df["vol_regime"] = (((atr / vol_med.replace(0, 1e-10)) - 0.5).clip(0, 1.5) / 1.5).fillna(0.5)

    # Squeeze: Bollinger width well below recent average (compression).
    bb_avg = df["bb_width"].rolling(96).mean()
    df["squeeze"] = (df["bb_width"] < bb_avg * 0.9).astype(float).fillna(0)

    # Regime direction: 1 = up-trend, -1 = down-trend, 0 = range.
    df["regime_trend"] = np.where(
        (ema20 > ema50) & (adx > 20), 1.0,
        np.where((ema20 < ema50) & (adx > 20), -1.0, 0.0),
    ).astype(float)

    return df


def add_h1_context(m5: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
    h1 = idx.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()

    h1_close = h1["close"]
    h1_ema20 = h1_close.ewm(span=20).mean()
    h1_ema50 = h1_close.ewm(span=50).mean()
    h1_trend = ((h1_ema20 - h1_ema50) / h1_close.replace(0, 1e-10)).fillna(0).clip(-3, 3)
    h1_rsi = compute_rsi(h1_close).fillna(50) / 100

    m5["h1_trend"] = h1_trend.reindex(m5.index, method="ffill").fillna(0)
    m5["h1_rsi"] = h1_rsi.reindex(m5.index, method="ffill").fillna(0.5)

    h1_hh = h1["high"].rolling(20).max().reindex(m5.index, method="ffill")
    h1_ll = h1["low"].rolling(20).min().reindex(m5.index, method="ffill")
    h1_range = (h1_hh - h1_ll).replace(0, 1e-10)

    m5["dist_h1_high"] = ((h1_hh - m5["close"]) / h1_range).fillna(0.5).clip(0, 1)
    m5["dist_h1_low"] = ((m5["close"] - h1_ll) / h1_range).fillna(0.5).clip(0, 1)

    return m5


def add_swing_features(m5: pd.DataFrame) -> pd.DataFrame:
    h = m5["high"].values
    l = m5["low"].values
    n = len(m5)
    since_h = np.zeros(n)
    since_l = np.zeros(n)
    last_h, last_l = 0, 0
    for i in range(1, n):
        if h[i] > h[i-1] and h[i] > (h[i+1] if i+1 < n else h[i]):
            last_h = i
        if l[i] < l[i-1] and l[i] < (l[i+1] if i+1 < n else l[i]):
            last_l = i
        since_h[i] = i - last_h if last_h > 0 else min(i, 12)
        since_l[i] = i - last_l if last_l > 0 else min(i, 12)

    m5["bars_since_swing_h"] = np.clip(since_h / 12, 0, 1)
    m5["bars_since_swing_l"] = np.clip(since_l / 12, 0, 1)
    return m5


def add_time_features(m5: pd.DataFrame) -> pd.DataFrame:
    if hasattr(m5.index, "hour"):
        hours = m5.index.hour
        days = m5.index.dayofweek
    else:
        hours = pd.Series(12, index=m5.index)
        days = pd.Series(0, index=m5.index)

    m5["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    m5["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    m5["day_sin"] = np.sin(2 * np.pi * days / 7)

    session = pd.Series(1.0, index=m5.index)
    session[(hours >= 0) & (hours < 8)] = 0.0
    session[(hours >= 8) & (hours < 14)] = 0.5
    session[(hours >= 14) & (hours < 22)] = 1.0
    session[(hours >= 22)] = 0.0
    m5["session_enc"] = session

    return m5


def build_features_from_m1(
    m1: pd.DataFrame, seq_len: int = SEQ_LEN
) -> Optional[np.ndarray]:
    """Convert raw M1 DataFrame → (seq_len × n_features) model input."""
    try:
        m5 = resample_m5(m1)
        m5 = compute_features(m5)
        m5 = add_h1_context(m5, m1)
        m5 = add_swing_features(m5)
        m5 = add_time_features(m5)
        missing = [c for c in FEATURE_COLS if c not in m5.columns]
        for c in missing:
            m5[c] = 0.0
        arr = m5[FEATURE_COLS].values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        if len(arr) < seq_len:
            pad = np.zeros((seq_len - len(arr), N_FEATURES), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)
        return arr[-seq_len:]
    except Exception:
        return None


# torch-based CandleBrain transformer/predictor classes (PositionalEncoding,
# CandleBrainTransformer, CandleBrain, CandleBrainPredictor) were removed -- torch is
# no longer used at runtime. Only the feature pipeline below is retained for research
# scripts (_train_candle_brain.py, momentum sweeps).
