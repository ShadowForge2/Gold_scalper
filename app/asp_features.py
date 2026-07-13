"""
ASP Feature Pipeline — 7 analysis systems producing ML features.

Systems:
  1. Swing Structure    — HH/HL/LH/LL, swing strength, duration, amplitude
  2. Fractal Detection  — fractal turning points, proximity to fractal levels
  3. Volatility         — ATR, BB width, range expansion/compression, vol ratio
  4. Market Geometry    — pullback depth, swing angle, acceleration, symmetry
  5. Liquidity          — sweeps, stop hunts, failed breakouts/breakdowns
  6. Candle Behaviour   — wick rejection, body dominance, inside/outside bars
  7. HTF Context        — higher timeframe trend, distance to HTF levels

All features are computed per-bar on M5 data (scalping timeframe).
H1 data provides HTF context.
"""
import numpy as np
import pandas as pd
from typing import Optional


# =========================================================================
# 1. Swing Structure Analysis
# =========================================================================

def swing_structure_features(df: pd.DataFrame, swing_lookback: int = 5) -> pd.DataFrame:
    """Detect swing highs/lows and compute structural features.

    Features:
      swing_high_dist, swing_low_dist    — distance to nearest swing high/low (ATR-normalized)
      hh, hl, lh, ll                     — higher-high, higher-low, lower-high, lower-low (binary)
      swing_strength                     — amplitude of last swing / ATR
      swing_duration                     — bars since last swing high/low
      swing_amplitude_pct                — swing range as % of price
      consecutive_swings_dir             — direction of last N swings (+1/-1 sum)
      swing_frequency                    — swings per 60 bars (volatility-adjusted)
    """
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    n = len(closes)

    feats = pd.DataFrame(index=df.index)

    # Compute ATR for normalization
    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean()
    atr_safe = atr.replace(0, np.nan)

    # Detect raw swings
    swing_h = np.zeros(n, dtype=bool)
    swing_l = np.zeros(n, dtype=bool)
    for i in range(swing_lookback, n - swing_lookback):
        window_h = highs[i - swing_lookback:i + swing_lookback + 1]
        window_l = lows[i - swing_lookback:i + swing_lookback + 1]
        if highs[i] == np.max(window_h):
            swing_h[i] = True
        if lows[i] == np.min(window_l):
            swing_l[i] = True

    # Distance to nearest swing high/low
    sh_dist = np.full(n, np.nan)
    sl_dist = np.full(n, np.nan)
    last_sh_price = np.nan
    last_sl_price = np.nan
    for i in range(n):
        if swing_h[i]:
            last_sh_price = highs[i]
        if swing_l[i]:
            last_sl_price = lows[i]
        sh_dist[i] = abs(closes[i] - last_sh_price) if not np.isnan(last_sh_price) else np.nan
        sl_dist[i] = abs(closes[i] - last_sl_price) if not np.isnan(last_sl_price) else np.nan

    feats["swing_high_dist"] = sh_dist / atr_safe.values
    feats["swing_low_dist"] = sl_dist / atr_safe.values

    # HH/HL/LH/LL — compare last 2 swing highs and last 2 swing lows
    sh_prices = []
    sl_prices = []
    hh_arr = np.zeros(n)
    hl_arr = np.zeros(n)
    lh_arr = np.zeros(n)
    ll_arr = np.zeros(n)
    for i in range(n):
        if swing_h[i]:
            sh_prices.append(highs[i])
        if swing_l[i]:
            sl_prices.append(lows[i])
        if len(sh_prices) >= 2:
            hh_arr[i] = 1.0 if sh_prices[-1] > sh_prices[-2] else 0.0
            lh_arr[i] = 1.0 if sh_prices[-1] < sh_prices[-2] else 0.0
        if len(sl_prices) >= 2:
            hl_arr[i] = 1.0 if sl_prices[-1] > sl_prices[-2] else 0.0
            ll_arr[i] = 1.0 if sl_prices[-1] < sl_prices[-2] else 0.0

    feats["hh"] = hh_arr
    feats["hl"] = hl_arr
    feats["lh"] = lh_arr
    feats["ll"] = ll_arr

    # Swing strength — amplitude of last completed swing / ATR
    swing_strength = np.zeros(n)
    last_sh_idx = -1
    last_sl_idx = -1
    for i in range(n):
        if swing_h[i]:
            last_sh_idx = i
        if swing_l[i]:
            last_sl_idx = i
        if last_sh_idx >= 0 and last_sl_idx >= 0:
            amp = abs(highs[last_sh_idx] - lows[last_sl_idx])
            swing_strength[i] = amp / max(atr_safe.iloc[i], 0.01) if not np.isnan(atr_safe.iloc[i]) else 0.0
    feats["swing_strength"] = swing_strength

    # Swing duration — bars since last swing point (either direction)
    swing_any = swing_h | swing_l
    last_swing_idx = np.full(n, -1, dtype=int)
    for i in range(n):
        if swing_any[i]:
            last_swing_idx[i] = i
        elif i > 0:
            last_swing_idx[i] = last_swing_idx[i - 1]
    feats["swing_duration"] = np.where(last_swing_idx >= 0, np.arange(n) - last_swing_idx, swing_lookback).astype(float)

    # Swing amplitude as % of price
    swing_amp = np.zeros(n)
    for i in range(n):
        if last_sh_idx >= 0 and last_sl_idx >= 0:
            swing_amp[i] = abs(highs[max(last_sh_idx, 0)] - lows[max(last_sl_idx, 0)]) / max(closes[i], 1.0)
    feats["swing_amplitude_pct"] = swing_amp

    # Swing frequency — number of swings in last 60 bars
    swing_cumsum = np.cumsum(swing_any.astype(int))
    freq = np.zeros(n)
    for i in range(60, n):
        freq[i] = (swing_cumsum[i] - swing_cumsum[i - 60]) / 60.0
    feats["swing_frequency"] = freq

    return feats


# =========================================================================
# 2. Fractal Detection
# =========================================================================

def fractal_features(df: pd.DataFrame, fractal_period: int = 5) -> pd.DataFrame:
    """Fractal-based features for swing turning point proximity.

    Features:
      fractal_bullish      — bullish fractal formed (close near fractal low)
      fractal_bearish      — bearish fractal formed (close near fractal high)
      fractal_distance     — distance to nearest fractal level / ATR
      fractal_agreement    — fractals on M5 align with H1 fractals (if available)
      price_near_fractal   — 1 if price within 0.3 ATR of a fractal level
    """
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean()
    atr_safe = atr.replace(0, np.nan)

    feats = pd.DataFrame(index=df.index)

    # Bullish fractal: low is lowest of 5 bars (2 before, 2 after)
    bull_frac = np.zeros(n, dtype=float)
    bear_frac = np.zeros(n, dtype=float)
    frac_low_price = np.full(n, np.nan)
    frac_high_price = np.full(n, np.nan)

    for i in range(fractal_period, n - fractal_period):
        window_l = lows[i - fractal_period:i + fractal_period + 1]
        window_h = highs[i - fractal_period:i + fractal_period + 1]
        if lows[i] == np.min(window_l):
            bull_frac[i] = 1.0
            frac_low_price[i] = lows[i]
        if highs[i] == np.max(window_h):
            bear_frac[i] = 1.0
            frac_high_price[i] = highs[i]

    feats["fractal_bullish"] = bull_frac
    feats["fractal_bearish"] = bear_frac

    # Distance to nearest fractal level
    last_frac_low = np.nan
    last_frac_high = np.nan
    frac_dist = np.zeros(n)
    price_near = np.zeros(n)
    for i in range(n):
        if not np.isnan(frac_low_price[i]):
            last_frac_low = frac_low_price[i]
        if not np.isnan(frac_high_price[i]):
            last_frac_high = frac_high_price[i]
        d_low = abs(closes[i] - last_frac_low) if not np.isnan(last_frac_low) else 999.0
        d_high = abs(closes[i] - last_frac_high) if not np.isnan(last_frac_high) else 999.0
        frac_dist[i] = min(d_low, d_high)
        a = atr_safe.iloc[i] if not np.isnan(atr_safe.iloc[i]) else 1.0
        price_near[i] = 1.0 if frac_dist[i] < 0.3 * a else 0.0

    feats["fractal_distance"] = frac_dist / atr_safe.values
    feats["price_near_fractal"] = price_near

    return feats


# =========================================================================
# 3. Volatility Analysis
# =========================================================================

def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility regime features.

    Features:
      atr_norm                — ATR / close (normalized)
      atr_ratio               — short ATR / long ATR (expansion/compression)
      bb_width                — Bollinger bandwidth
      bb_position             — position within BB bands (0=lower, 1=upper)
      range_expansion         — current range / avg range
      vol_regime              — 0=low, 1=normal, 2=high (ATR percentile)
      vol_zscore              — z-score of ATR relative to rolling mean
    """
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    n = len(closes)

    feats = pd.DataFrame(index=df.index)

    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr14 = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean()
    atr5 = pd.Series(tr, index=df.index).rolling(5, min_periods=2).mean()
    atr50 = pd.Series(tr, index=df.index).rolling(50, min_periods=2).mean()
    atr_safe = atr14.replace(0, np.nan)

    feats["atr_norm"] = atr14 / np.where(closes > 0, closes, 1)
    feats["atr_ratio"] = atr5 / atr_safe
    feats["vol_zscore"] = (atr14 - atr14.rolling(50, min_periods=2).mean()) / atr14.rolling(50, min_periods=2).std().replace(0, np.nan)

    # Bollinger
    sma20 = pd.Series(closes, index=df.index).rolling(20, min_periods=2).mean()
    bb_std = pd.Series(closes, index=df.index).rolling(20, min_periods=2).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    feats["bb_width"] = bb_range / sma20
    feats["bb_position"] = (closes - bb_lower) / bb_range

    # Range expansion
    candle_range = highs - lows
    avg_range = pd.Series(candle_range, index=df.index).rolling(20, min_periods=2).mean()
    feats["range_expansion"] = candle_range / avg_range.replace(0, np.nan)

    # Vol regime (0=low, 1=normal, 2=high)
    atr_pct = atr14.rank(pct=True)
    vol_regime = np.where(atr_pct < 0.33, 0, np.where(atr_pct < 0.67, 1, 2))
    feats["vol_regime"] = vol_regime.astype(float)

    return feats


# =========================================================================
# 4. Market Geometry
# =========================================================================

def geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market geometry — pullback depth, angle, acceleration, symmetry.

    Features:
      pullback_depth       — retracement of last swing as fraction (0-1)
      swing_angle          — slope of price over last swing / ATR
      price_acceleration   — second derivative of price (curvature)
      symmetry             — ratio of up-swings to down-swings amplitude
      retracement_level    — Fibonacci-like level of current price vs last swing
    """
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    n = len(closes)

    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean()
    atr_safe = atr.replace(0, np.nan)

    feats = pd.DataFrame(index=df.index)

    # Pullback depth: retracement of last 20-bar swing
    swing_high_20 = pd.Series(highs, index=df.index).rolling(20, min_periods=5).max()
    swing_low_20 = pd.Series(lows, index=df.index).rolling(20, min_periods=5).min()
    swing_range = (swing_high_20 - swing_low_20).replace(0, np.nan)
    feats["pullback_depth"] = (closes - swing_low_20) / swing_range

    # Swing angle: linear regression slope of closes over 10 bars / ATR
    slope = np.full(n, np.nan)
    x = np.arange(10, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(10, n):
        y = closes[i - 9:i + 1]
        y_mean = y.mean()
        slope[i] = (np.sum((x - x_mean) * (y - y_mean)) / x_var)
    feats["swing_angle"] = slope / atr_safe.values

    # Price acceleration: change in slope
    slope_series = pd.Series(slope, index=df.index)
    feats["price_acceleration"] = slope_series.diff(5) / atr_safe

    # Symmetry: ratio of avg up-move to avg down-move over last 20 bars
    returns = np.diff(closes, prepend=closes[0])
    up_avg = pd.Series(np.where(returns > 0, returns, 0), index=df.index).rolling(20, min_periods=5).mean()
    dn_avg = pd.Series(np.where(returns < 0, -returns, 0), index=df.index).rolling(20, min_periods=5).mean()
    feats["symmetry"] = up_avg / dn_avg.replace(0, np.nan)

    # Retracement level (Fibonacci-like)
    feats["retracement_level"] = (closes - swing_low_20) / swing_range

    return feats


# =========================================================================
# 5. Liquidity Analysis
# =========================================================================

def liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Liquidity sweep and stop-hunt detection.

    Features:
      sweep_high           — price spiked above recent high then closed below (bull trap)
      sweep_low            — price spiked below recent low then closed above (bear trap)
      failed_breakout      — broke above range but closed back inside
      failed_breakdown     — broke below range but closed back inside
      wick_above_range     — wick extension above recent high / ATR
      wick_below_range     — wick extension below recent low / ATR
      liquidity_score      — composite score of sweep + failed breakout events
    """
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    opens = df["open"].values.astype(np.float64)
    n = len(closes)

    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean()
    atr_safe = atr.replace(0, np.nan)

    feats = pd.DataFrame(index=df.index)

    lookback = 20
    recent_high = pd.Series(highs, index=df.index).rolling(lookback, min_periods=5).max().shift(1)
    recent_low = pd.Series(lows, index=df.index).rolling(lookback, min_periods=5).min().shift(1)

    # Sweep high: high > recent_high but close < recent_high (bearish fakeout)
    sweep_high = ((highs > recent_high.values) & (closes < recent_high.values)).astype(float)
    # Sweep low: low < recent_low but close > recent_low (bullish fakeout)
    sweep_low = ((lows < recent_low.values) & (closes > recent_low.values)).astype(float)

    feats["sweep_high"] = sweep_high
    feats["sweep_low"] = sweep_low

    # Failed breakout/breakdown: close outside range, then next bar closes back inside
    broke_above = closes > recent_high.values
    broke_below = closes < recent_low.values
    feats["failed_breakout"] = (broke_above & (np.roll(closes, -1) < recent_high.values)).astype(float)
    feats["failed_breakdown"] = (broke_below & (np.roll(closes, -1) > recent_low.values)).astype(float)
    # Fix last bar
    feats.iloc[-1, feats.columns.get_loc("failed_breakout")] = 0.0
    feats.iloc[-1, feats.columns.get_loc("failed_breakdown")] = 0.0

    # Wick extensions
    a = atr_safe.values
    a = np.where(np.isnan(a) | (a <= 0), 1.0, a)
    feats["wick_above_range"] = np.maximum(0, highs - recent_high.values) / a
    feats["wick_below_range"] = np.maximum(0, recent_low.values - lows) / a

    # Composite liquidity score
    feats["liquidity_score"] = (sweep_high + sweep_low +
                                feats["failed_breakout"] + feats["failed_breakdown"]).clip(0, 1)

    return feats


# =========================================================================
# 6. Candle Behaviour
# =========================================================================

def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candle pattern features.

    Features:
      wick_ratio            — (upper_wick + lower_wick) / total_range
      body_ratio            — body / range
      wick_rejection_upper  — upper wick / range (selling pressure)
      wick_rejection_lower  — lower wick / range (buying pressure)
      inside_bar            — current range within previous range
      outside_bar           — current range engulfs previous range
      consec_bull           — consecutive bullish candles
      consec_bear           — consecutive bearish candles
      body_vs_avg           — current body / avg body
      range_vs_avg          — current range / avg range
    """
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    opens = df["open"].values.astype(np.float64)
    n = len(closes)

    feats = pd.DataFrame(index=df.index)

    body = np.abs(closes - opens)
    candle_range = highs - lows
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows
    range_safe = np.where(candle_range > 0, candle_range, 1.0)

    feats["body_ratio"] = body / range_safe
    feats["wick_ratio"] = (upper_wick + lower_wick) / range_safe
    feats["wick_rejection_upper"] = upper_wick / range_safe
    feats["wick_rejection_lower"] = lower_wick / range_safe

    # Inside/outside bars
    prev_range = np.concatenate([[candle_range[0]], candle_range[:-1]])
    feats["inside_bar"] = (candle_range <= prev_range).astype(float)
    feats["outside_bar"] = (candle_range > prev_range * 1.5).astype(float)

    # Consecutive direction
    bull = closes > opens
    cb = np.zeros(n)
    cc = np.zeros(n)
    for i in range(1, n):
        cb[i] = (cb[i - 1] + 1) if bull[i] else 0
        cc[i] = (cc[i - 1] + 1) if not bull[i] else 0
    feats["consec_bull"] = cb
    feats["consec_bear"] = cc

    # Body and range vs average
    avg_body = pd.Series(body, index=df.index).rolling(20, min_periods=2).mean()
    avg_range = pd.Series(candle_range, index=df.index).rolling(20, min_periods=2).mean()
    feats["body_vs_avg"] = body / avg_body.replace(0, np.nan)
    feats["range_vs_avg"] = candle_range / avg_range.replace(0, np.nan)

    return feats


# =========================================================================
# 7. HTF Context (H1)
# =========================================================================

def htf_context_features(m5_df: pd.DataFrame, h1_df: pd.DataFrame) -> pd.DataFrame:
    """Higher timeframe context aligned to M5 bars.

    Features:
      h1_trend_dir         — H1 close vs H1 EMA20 direction (+1/-1/0)
      h1_rsi               — H1 RSI(14)
      h1_distance_to_high  — distance to H1 session high / ATR
      h1_distance_to_low   — distance to H1 session low / ATR
      h1_in_range          — price within H1 range (0-1, 0.5=middle)
      h1_momentum          — H1 MACD histogram sign
      h1_vol_ratio         — H1 ATR / M5 ATR (timeframe vol ratio)
    """
    feats = pd.DataFrame(index=m5_df.index)

    if h1_df is None or len(h1_df) < 20:
        for c in ["h1_trend_dir", "h1_rsi", "h1_distance_to_high", "h1_distance_to_low",
                   "h1_in_range", "h1_momentum", "h1_vol_ratio"]:
            feats[c] = 0.0
        return feats

    h1 = h1_df.copy()
    if not isinstance(h1.index, pd.DatetimeIndex):
        h1.index = pd.to_datetime(h1.index)
    h1 = h1[~h1.index.duplicated(keep="first")]

    c = h1["close"].values.astype(np.float64)
    h = h1["high"].values.astype(np.float64)
    lo = h1["low"].values.astype(np.float64)
    o = h1["open"].values.astype(np.float64)

    # H1 RSI
    delta = np.diff(c, prepend=np.nan)
    gain = pd.Series(np.where(delta > 0, delta, 0), index=h1.index).rolling(14, min_periods=2).mean()
    loss_s = pd.Series(np.where(delta < 0, -delta, 0), index=h1.index).rolling(14, min_periods=2).mean()
    rs = gain / loss_s.replace(0, np.nan)
    h1_rsi = (100 - (100 / (1 + rs))).fillna(50.0)

    # H1 trend direction (close vs EMA20)
    ema20 = pd.Series(c, index=h1.index).ewm(span=20, adjust=False).mean()
    h1_dir = np.where(c > ema20, 1.0, np.where(c < ema20, -1.0, 0.0))

    # H1 MACD
    ema12 = pd.Series(c, index=h1.index).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(c, index=h1.index).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    h1_macd_hist = (macd_line - macd_sig).fillna(0.0)

    # H1 range (rolling 24 bars = session)
    h1_high_roll = pd.Series(h, index=h1.index).rolling(24, min_periods=1).max()
    h1_low_roll = pd.Series(lo, index=h1.index).rolling(24, min_periods=1).min()
    h1_range = (h1_high_roll - h1_low_roll).replace(0, np.nan)

    h1_in_range = ((c - h1_low_roll) / h1_range).fillna(0.5)

    # H1 ATR
    prev_hc = np.concatenate([[c[0]], c[:-1]])
    h1_tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_hc), np.abs(lo - prev_hc)))
    h1_atr = pd.Series(h1_tr, index=h1.index).rolling(14, min_periods=2).mean()

    # Build aligned DataFrame
    h1_feats = pd.DataFrame(index=h1.index)
    h1_feats["h1_trend_dir"] = h1_dir
    h1_feats["h1_rsi"] = h1_rsi.values
    h1_feats["h1_distance_to_high"] = ((h1_high_roll - c) / h1_atr).fillna(0).values
    h1_feats["h1_distance_to_low"] = ((c - h1_low_roll) / h1_atr).fillna(0).values
    h1_feats["h1_in_range"] = h1_in_range.values
    h1_feats["h1_momentum"] = np.sign(h1_macd_hist.values)
    h1_feats["h1_vol_ratio"] = h1_atr.fillna(1.0).values

    # Align to M5 index
    aligned = h1_feats.reindex(m5_df.index, method="ffill").fillna(0.0)
    return aligned


# =========================================================================
# Time Features
# =========================================================================

def time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time features."""
    feats = pd.DataFrame(index=df.index)
    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    hours = times.hour
    days = times.dayofweek
    feats["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    feats["day_sin"] = np.sin(2 * np.pi * days / 7)
    feats["day_cos"] = np.cos(2 * np.pi * days / 7)
    return feats


# =========================================================================
# Master Feature Pipeline
# =========================================================================

ASP_FEATURE_COLS = [
    # Swing Structure (12)
    "swing_high_dist", "swing_low_dist", "hh", "hl", "lh", "ll",
    "swing_strength", "swing_duration", "swing_amplitude_pct", "swing_frequency",
    # Fractal (4)
    "fractal_bullish", "fractal_bearish", "fractal_distance", "price_near_fractal",
    # Volatility (7)
    "atr_norm", "atr_ratio", "bb_width", "bb_position",
    "range_expansion", "vol_regime", "vol_zscore",
    # Geometry (5)
    "pullback_depth", "swing_angle", "price_acceleration", "symmetry", "retracement_level",
    # Liquidity (7)
    "sweep_high", "sweep_low", "failed_breakout", "failed_breakdown",
    "wick_above_range", "wick_below_range", "liquidity_score",
    # Candle (10)
    "body_ratio", "wick_ratio", "wick_rejection_upper", "wick_rejection_lower",
    "inside_bar", "outside_bar", "consec_bull", "consec_bear",
    "body_vs_avg", "range_vs_avg",
    # HTF (7)
    "h1_trend_dir", "h1_rsi", "h1_distance_to_high", "h1_distance_to_low",
    "h1_in_range", "h1_momentum", "h1_vol_ratio",
    # Time (4)
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

TOTAL_FEATURES = len(ASP_FEATURE_COLS)


def compute_asp_features(m5_df: pd.DataFrame, h1_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute all ASP features from M5 + H1 OHLCV data.

    Returns DataFrame with same index as m5_df, containing all feature columns.
    """
    m5 = m5_df.copy()
    if not isinstance(m5.index, pd.DatetimeIndex):
        if "time" in m5.columns:
            m5 = m5.set_index("time")
        else:
            m5.index = pd.to_datetime(m5.index)

    s1 = swing_structure_features(m5)
    s2 = fractal_features(m5)
    s3 = volatility_features(m5)
    s4 = geometry_features(m5)
    s5 = liquidity_features(m5)
    s6 = candle_features(m5)
    s7 = htf_context_features(m5, h1_df)
    s8 = time_features(m5)

    combined = pd.concat([s1, s2, s3, s4, s5, s6, s7, s8], axis=1)

    # Ensure all expected columns exist
    for c in ASP_FEATURE_COLS:
        if c not in combined.columns:
            combined[c] = 0.0

    combined = combined[ASP_FEATURE_COLS]
    combined = combined[~combined.index.duplicated(keep="first")]

    return combined
