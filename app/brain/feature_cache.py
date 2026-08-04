"""FeatureCache — shared data layer with bounded ring buffers.

Computes 100 features once per tick and provides them to all Brain layers.
Uses fixed-size numpy arrays (ring buffers) to guarantee constant memory usage.
"""
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger("GoldScalper.Brain")

BRAIN_FEATURE_COLS = [
    # === PRICE ACTION (15) ===
    "pnl_absolute", "pnl_atr_ratio", "pnl_percent",
    "distance_to_entry", "distance_to_sl", "distance_to_tp",
    "price_vs_session_high", "price_vs_session_low",
    "candle_body_ratio", "candle_range_atr",
    "price_velocity_5m", "price_velocity_15m",
    "price_acceleration", "consecutive_bars_in_direction",
    "bars_since_entry",

    # === ML CONFIDENCE (8) ===
    "ml_confidence_entry", "ml_confidence_current",
    "ml_confidence_delta", "ml_confidence_trend",
    "ml_confidence_rolling_avg", "ml_confidence_std",
    "ml_agreement_with_bias", "ml_confidence_percentile",

    # === MARKET STRUCTURE (12) ===
    "adx_current", "adx_entry", "adx_delta",
    "regime_current", "regime_entry", "regime_changed",
    "higher_highs_count", "higher_lows_count",
    "support_distance", "resistance_distance",
    "structure_break_detected", "structure_quality_score",

    # === VOLATILITY (10) ===
    "atr_current", "atr_entry", "atr_ratio",
    "volatility_percentile", "bollinger_width",
    "bollinger_position", "keltner_width",
    "regime_volatility_match", "implied_move",
    "realized_vs_implied",

    # === TIME & SESSION (8) ===
    "hour_of_day_utc", "minute_of_hour",
    "session_active", "minutes_to_session_close",
    "minutes_to_daily_close", "minutes_to_weekly_close",
    "day_of_week", "is_friday",

    # === OPPORTUNITY COST (15) ===
    "best_pending_score", "best_pending_atr_setup",
    "best_pending_session_match", "best_pending_regime_fit",
    "opportunity_count", "opportunity_avg_score",
    "opportunity_max_expected_pnl",
    "current_trade_rank_vs_opportunities",
    "correlated_symbol_trend_alignment",
    "portfolio_heat", "portfolio_exposure",
    "portfolio_heat_percentile",
    "marginal_value_of_capital",
    "opportunity_cost_per_hour",
    "opportunity_decay_rate",

    # === HISTORICAL PERFORMANCE (12) ===
    "recent_win_rate_20", "recent_win_rate_50",
    "recent_avg_pnl_20", "recent_avg_pnl_50",
    "max_drawdown_recent", "sharpe_ratio_recent",
    "regime_win_rate", "session_win_rate",
    "trade_count_today", "pnl_today",
    "consecutive_wins", "consecutive_losses",

    # === MARKET MICROSTRUCTURE (20) ===
    "spread_pips", "spread_vs_avg",
    "bid_ask_imbalance", "volume_spike",
    "order_flow_bias", "liquidity_score",
    "tick_direction_consensus", "price_impact",
    "momentum_1m", "momentum_5m",
    "momentum_15m", "momentum_1h",
    "rsi_14", "rsi_divergence",
    "macd_histogram", "macd_crossover",
    "stochastic_k", "stochastic_d",
    "cci_20", "williams_r",
]

assert len(BRAIN_FEATURE_COLS) == 100, f"Expected 100 features, got {len(BRAIN_FEATURE_COLS)}"


class FeatureCache:
    """Bounded ring-buffer feature cache shared across all Brain layers.

    Memory budget: ~10 MB
      - Per-symbol feature vectors: 100 features × 4 symbols × 8 bytes = 3.2 KB
      - History ring buffers: 100 features × 200 bars × 4 symbols × 8 bytes = 6.4 MB
      - M1/M5/H1 data caches: ~4 MB (bounded by bar count limits)

    All arrays are pre-allocated at init. No append() calls — only index writes.
    """

    RING_SIZE = 200  # history window per symbol

    def __init__(self):
        self._current: Dict[str, np.ndarray] = {}
        self._history: Dict[str, np.ndarray] = {}
        self._history_idx: Dict[str, int] = {}
        self._m1_cache: Dict[str, pd.DataFrame] = {}
        self._m5_cache: Dict[str, pd.DataFrame] = {}
        self._h1_cache: Dict[str, pd.DataFrame] = {}
        self._h4_cache: Dict[str, pd.DataFrame] = {}
        self._last_update: Dict[str, float] = {}
        self._regime_cache: Dict[str, str] = {}
        self._adx_cache: Dict[str, float] = {}
        self._atr_cache: Dict[str, float] = {}
        self._ml_confidence_history: Dict[str, np.ndarray] = {}
        self._ml_confidence_idx: Dict[str, int] = {}
        self._pending_signals: Dict[str, Dict] = {}

    def init_symbol(self, symbol: str):
        n_feat = len(BRAIN_FEATURE_COLS)
        self._current[symbol] = np.full(n_feat, np.nan, dtype=np.float64)
        self._history[symbol] = np.full((self.RING_SIZE, n_feat), np.nan, dtype=np.float64)
        self._history_idx[symbol] = 0
        self._ml_confidence_history[symbol] = np.full(50, np.nan, dtype=np.float64)
        self._ml_confidence_idx[symbol] = 0
        self._last_update[symbol] = 0.0
        self._regime_cache[symbol] = "UNKNOWN"
        self._adx_cache[symbol] = 0.0
        self._atr_cache[symbol] = 0.0

    def update_tick(self, symbol: str, m1_data: pd.DataFrame,
                    current_price: float, entry_info: Optional[Dict] = None,
                    force: bool = False):
        """Recompute all features for a symbol from fresh M1 data.

        Called once per tick per symbol. All layers read from get_features().
        `force=True` bypasses the 1-second throttle for offline replay.
        """
        if symbol not in self._current:
            self.init_symbol(symbol)

        now = time.time()
        if not force and now - self._last_update.get(symbol, 0) < 1.0:
            return
        self._last_update[symbol] = now

        if m1_data is None or len(m1_data) < 20:
            return

        try:
            m1 = m1_data.copy()
            if "time" in m1.columns:
                m1 = m1.set_index("time")
            m1 = m1.sort_index()

            self._m1_cache[symbol] = m1

            m5 = m1.resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "tick_volume": "sum",
            }).dropna()
            self._m5_cache[symbol] = m5

            h1 = m1.resample("1h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "tick_volume": "sum",
            }).dropna()
            self._h1_cache[symbol] = h1

            h4 = m1.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "tick_volume": "sum",
            }).dropna()
            self._h4_cache[symbol] = h4

            feat = np.full(len(BRAIN_FEATURE_COLS), np.nan, dtype=np.float64)
            col_idx = {c: i for i, c in enumerate(BRAIN_FEATURE_COLS)}

            closes = m1["close"].values.astype(np.float64)
            highs = m1["high"].values.astype(np.float64)
            lows = m1["low"].values.astype(np.float64)
            opens = m1["open"].values.astype(np.float64)

            m5_closes = m5["close"].values.astype(np.float64)
            m5_highs = m5["high"].values.astype(np.float64)
            m5_lows = m5["low"].values.astype(np.float64)
            m5_opens = m5["open"].values.astype(np.float64)

            h1_closes = h1["close"].values.astype(np.float64)
            h1_highs = h1["high"].values.astype(np.float64)
            h1_lows = h1["low"].values.astype(np.float64)

            px = current_price
            entry = entry_info or {}
            entry_px = entry.get("entry_price", px)
            direction = entry.get("direction", "BUY")
            sl = entry.get("sl", 0)
            tp = entry.get("tp1", 0)
            atr_at_entry = entry.get("atr", 0)
            ml_conf_entry = entry.get("ml_confidence", 0.5)

            prev_close = np.concatenate([[np.nan], closes[:-1]])
            tr_m1 = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(np.abs(highs[1:] - closes[:-1]),
                           np.abs(lows[1:] - closes[:-1])),
            )
            atr_14 = float(np.mean(tr_m1[-14:])) if len(tr_m1) >= 14 else 1.0
            if atr_14 <= 0:
                atr_14 = 1.0
            self._atr_cache[symbol] = atr_14

            # --- Price Action ---
            if direction == "BUY":
                pnl_abs = px - entry_px if entry_px > 0 else 0
            else:
                pnl_abs = entry_px - px if entry_px > 0 else 0
            pnl_atr = pnl_abs / atr_14 if atr_14 > 0 else 0
            pnl_pct = pnl_abs / entry_px * 100 if entry_px > 0 else 0

            feat[col_idx["pnl_absolute"]] = pnl_abs
            feat[col_idx["pnl_atr_ratio"]] = pnl_atr
            feat[col_idx["pnl_percent"]] = pnl_pct
            feat[col_idx["distance_to_entry"]] = (px - entry_px) / atr_14 if atr_14 > 0 else 0

            dist_sl = abs(px - sl) / atr_14 if sl > 0 and atr_14 > 0 else 0
            dist_tp = abs(px - tp) / atr_14 if tp > 0 and atr_14 > 0 else 0
            feat[col_idx["distance_to_sl"]] = dist_sl
            feat[col_idx["distance_to_tp"]] = dist_tp

            if len(m5_highs) > 0 and len(m5_lows) > 0:
                sess_high = float(m5_highs[-20:].max()) if len(m5_highs) >= 20 else float(m5_highs.max())
                sess_low = float(m5_lows[-20:].min()) if len(m5_lows) >= 20 else float(m5_lows.min())
            else:
                sess_high = px
                sess_low = px
            sess_range = sess_high - sess_low
            feat[col_idx["price_vs_session_high"]] = (px - sess_high) / atr_14 if atr_14 > 0 else 0
            feat[col_idx["price_vs_session_low"]] = (px - sess_low) / atr_14 if atr_14 > 0 else 0

            if len(m5_closes) >= 2:
                last_body = abs(m5_closes[-1] - m5_opens[-1]) if len(m5_opens) > 0 else 0
                last_range = m5_highs[-1] - m5_lows[-1] if len(m5_highs) > 0 and len(m5_lows) > 0 else 1
                feat[col_idx["candle_body_ratio"]] = last_body / last_range if last_range > 0 else 0
                feat[col_idx["candle_range_atr"]] = last_range / atr_14 if atr_14 > 0 else 0

            if len(m5_closes) >= 3:
                feat[col_idx["price_velocity_5m"]] = (m5_closes[-1] - m5_closes[-3]) / atr_14 if atr_14 > 0 else 0
            if len(m5_closes) >= 15:
                feat[col_idx["price_velocity_15m"]] = (m5_closes[-1] - m5_closes[-15]) / atr_14 if atr_14 > 0 else 0

            if len(m5_closes) >= 3:
                v1 = m5_closes[-1] - m5_closes[-2]
                v2 = m5_closes[-2] - m5_closes[-3]
                feat[col_idx["price_acceleration"]] = (v1 - v2) / atr_14 if atr_14 > 0 else 0

            bars_dir = 0
            if len(m5_closes) >= 2:
                sign = 1 if m5_closes[-1] > m5_opens[-1] else -1
                bars_dir = 1
                for i in range(len(m5_closes) - 2, max(0, len(m5_closes) - 20), -1):
                    s = 1 if m5_closes[i] > m5_opens[i] else -1
                    if s == sign:
                        bars_dir += 1
                    else:
                        break
                bars_dir *= sign
            feat[col_idx["consecutive_bars_in_direction"]] = bars_dir

            event_start = entry.get("event_start")
            if event_start:
                mins_held = (time.time() - event_start) / 60.0
                feat[col_idx["bars_since_entry"]] = mins_held / 5.0
            else:
                feat[col_idx["bars_since_entry"]] = 0

            # --- ML Confidence ---
            ml_now = entry.get("ml_confidence_current", ml_conf_entry)
            feat[col_idx["ml_confidence_entry"]] = ml_conf_entry
            feat[col_idx["ml_confidence_current"]] = ml_now
            feat[col_idx["ml_confidence_delta"]] = ml_now - ml_conf_entry

            conf_hist = self._ml_confidence_history[symbol]
            conf_idx = self._ml_confidence_idx[symbol]
            conf_hist[conf_idx % 50] = ml_now
            self._ml_confidence_idx[symbol] = conf_idx + 1

            valid_conf = conf_hist[~np.isnan(conf_hist)]
            if len(valid_conf) >= 3:
                feat[col_idx["ml_confidence_rolling_avg"]] = float(np.mean(valid_conf[-10:]))
                feat[col_idx["ml_confidence_std"]] = float(np.std(valid_conf[-10:]))
                feat[col_idx["ml_confidence_trend"]] = float(np.mean(np.diff(valid_conf[-5:]))) if len(valid_conf) >= 5 else 0
                feat[col_idx["ml_confidence_percentile"]] = float(np.searchsorted(np.sort(valid_conf), ml_now) / len(valid_conf))
            else:
                feat[col_idx["ml_confidence_rolling_avg"]] = ml_now
                feat[col_idx["ml_confidence_std"]] = 0
                feat[col_idx["ml_confidence_trend"]] = 0
                feat[col_idx["ml_confidence_percentile"]] = 0.5

            feat[col_idx["ml_agreement_with_bias"]] = entry.get("bias_agreement", 0.5)

            # --- Market Structure ---
            adx = self._compute_adx(m5_highs, m5_lows, m5_closes, 14) if len(m5_closes) > 15 else 0
            self._adx_cache[symbol] = adx
            feat[col_idx["adx_current"]] = adx
            feat[col_idx["adx_entry"]] = entry.get("adx_at_entry", adx)
            feat[col_idx["adx_delta"]] = adx - entry.get("adx_at_entry", adx)

            regime = self._classify_regime(m5_closes, m5_highs, m5_lows, atr_14) if len(m5_closes) > 20 else "UNKNOWN"
            self._regime_cache[symbol] = regime
            regime_map = {"TRENDING": 1, "RANGING": -1, "VOLATILE": 0.5, "STAGNANT": -0.5, "UNKNOWN": 0}
            feat[col_idx["regime_current"]] = regime_map.get(regime, 0)
            feat[col_idx["regime_entry"]] = regime_map.get(entry.get("regime_at_entry", "UNKNOWN"), 0)
            feat[col_idx["regime_changed"]] = 1.0 if regime != entry.get("regime_at_entry", "UNKNOWN") else 0.0

            hh, hl, lh, ll = self._count_structure(m5_highs, m5_lows)
            feat[col_idx["higher_highs_count"]] = hh
            feat[col_idx["higher_lows_count"]] = hl

            support_dist, resist_dist = self._support_resistance_dist(px, m5_highs, m5_lows)
            feat[col_idx["support_distance"]] = support_dist / atr_14 if atr_14 > 0 else 0
            feat[col_idx["resistance_distance"]] = resist_dist / atr_14 if atr_14 > 0 else 0

            feat[col_idx["structure_break_detected"]] = 1.0 if self._detect_structure_break(m5_highs, m5_lows, m5_closes) else 0.0
            feat[col_idx["structure_quality_score"]] = self._structure_quality(m5_highs, m5_lows, m5_closes)

            # --- Volatility ---
            feat[col_idx["atr_current"]] = atr_14
            feat[col_idx["atr_entry"]] = atr_at_entry
            feat[col_idx["atr_ratio"]] = atr_14 / atr_at_entry if atr_at_entry > 0 else 1.0

            atr_history = self._atr_cache.get(f"{symbol}_history", np.full(200, np.nan))
            if not isinstance(atr_history, np.ndarray):
                atr_history = np.full(200, np.nan, dtype=np.float64)
            atr_hist_idx = int(self._atr_cache.get(f"{symbol}_hist_idx", 0))
            atr_history[atr_hist_idx % 200] = atr_14
            self._atr_cache[f"{symbol}_history"] = atr_history
            self._atr_cache[f"{symbol}_hist_idx"] = atr_hist_idx + 1
            valid_atr = atr_history[~np.isnan(atr_history)]
            if len(valid_atr) > 10:
                feat[col_idx["volatility_percentile"]] = float(np.searchsorted(np.sort(valid_atr), atr_14) / len(valid_atr))
            else:
                feat[col_idx["volatility_percentile"]] = 0.5

            if len(m5_closes) >= 20:
                sma20 = float(np.mean(m5_closes[-20:]))
                std20 = float(np.std(m5_closes[-20:]))
                bb_upper = sma20 + 2 * std20
                bb_lower = sma20 - 2 * std20
                bb_width = (bb_upper - bb_lower) / sma20 if sma20 > 0 else 0
                feat[col_idx["bollinger_width"]] = bb_width
                feat[col_idx["bollinger_position"]] = (px - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            else:
                feat[col_idx["bollinger_width"]] = 0
                feat[col_idx["bollinger_position"]] = 0.5

            if len(m5_closes) >= 20:
                keltner_mid = float(np.mean(m5_closes[-20:]))
                keltner_range = atr_14 * 1.5
                feat[col_idx["keltner_width"]] = keltner_range * 2 / keltner_mid if keltner_mid > 0 else 0
            else:
                feat[col_idx["keltner_width"]] = 0

            feat[col_idx["regime_volatility_match"]] = 1.0 if regime == "VOLATILE" and atr_14 > atr_at_entry else 0.0
            feat[col_idx["implied_move"]] = atr_14
            feat[col_idx["realized_vs_implied"]] = atr_14 / atr_at_entry if atr_at_entry > 0 else 1.0

            # --- Time & Session ---
            now_utc = time.gmtime()
            feat[col_idx["hour_of_day_utc"]] = now_utc.tm_hour + now_utc.tm_min / 60.0
            feat[col_idx["minute_of_hour"]] = now_utc.tm_min
            feat[col_idx["day_of_week"]] = now_utc.tm_wday
            feat[col_idx["is_friday"]] = 1.0 if now_utc.tm_wday == 4 else 0.0

            h = now_utc.tm_hour
            if 0 <= h < 8:
                feat[col_idx["session_active"]] = 0  # ASIA
            elif 8 <= h < 17:
                feat[col_idx["session_active"]] = 1  # LONDON
            elif 12 < h < 22:
                feat[col_idx["session_active"]] = 2  # NEW_YORK
            else:
                feat[col_idx["session_active"]] = 3  # OUTSIDE

            if h >= 17:
                feat[col_idx["minutes_to_session_close"]] = 0
            elif h >= 12:
                feat[col_idx["minutes_to_session_close"]] = (17 - h - 1) * 60 + (60 - now_utc.tm_min)
            elif h >= 8:
                feat[col_idx["minutes_to_session_close"]] = (17 - h - 1) * 60 + (60 - now_utc.tm_min)
            else:
                feat[col_idx["minutes_to_session_close"]] = (8 - h - 1) * 60 + (60 - now_utc.tm_min)

            from config import is_market_open_for_symbol, minutes_to_friday_close
            mins_friday = minutes_to_friday_close(symbol)
            feat[col_idx["minutes_to_weekly_close"]] = mins_friday if mins_friday is not None else 999

            feat[col_idx["minutes_to_daily_close"]] = max(0, (22 - h - 1) * 60 + (60 - now_utc.tm_min)) if h < 22 else 0

            # --- Market Microstructure ---
            if "tick_volume" in m1.columns:
                vol_arr = m1["tick_volume"].values.astype(np.float64)
                vol_avg = float(np.mean(vol_arr[-50:])) if len(vol_arr) >= 50 else 1
                feat[col_idx["volume_spike"]] = vol_arr[-1] / vol_avg if vol_avg > 0 else 1
            else:
                feat[col_idx["volume_spike"]] = 1.0

            feat[col_idx["spread_pips"]] = 0
            feat[col_idx["spread_vs_avg"]] = 0
            feat[col_idx["bid_ask_imbalance"]] = 0
            feat[col_idx["order_flow_bias"]] = 0
            feat[col_idx["liquidity_score"]] = 0.5
            feat[col_idx["tick_direction_consensus"]] = 0
            feat[col_idx["price_impact"]] = 0

            if len(closes) >= 60:
                feat[col_idx["momentum_1m"]] = (closes[-1] - closes[-2]) / atr_14 if atr_14 > 0 else 0
                feat[col_idx["momentum_5m"]] = (closes[-1] - closes[-6]) / atr_14 if atr_14 > 0 and len(closes) >= 6 else 0
                feat[col_idx["momentum_15m"]] = (closes[-1] - closes[-16]) / atr_14 if atr_14 > 0 and len(closes) >= 16 else 0
                feat[col_idx["momentum_1h"]] = (closes[-1] - closes[-61]) / atr_14 if atr_14 > 0 and len(closes) >= 61 else 0

            if len(closes) >= 15:
                delta = np.diff(closes[-15:], prepend=closes[-15])
                gain = np.mean(np.where(delta > 0, delta, 0)[-14:])
                loss = np.mean(np.where(delta < 0, -delta, 0)[-14:])
                rs = gain / loss if loss > 0 else 100
                feat[col_idx["rsi_14"]] = 100 - (100 / (1 + rs))
            else:
                feat[col_idx["rsi_14"]] = 50.0

            feat[col_idx["rsi_divergence"]] = 0
            feat[col_idx["macd_histogram"]] = 0
            feat[col_idx["macd_crossover"]] = 0
            feat[col_idx["stochastic_k"]] = 50.0
            feat[col_idx["stochastic_d"]] = 50.0
            feat[col_idx["cci_20"]] = 0
            feat[col_idx["williams_r"]] = -50.0

            self._current[symbol] = feat
            idx = self._history_idx[symbol]
            self._history[symbol][idx % self.RING_SIZE] = feat
            self._history_idx[symbol] = idx + 1

        except Exception as e:
            logger.warning(f"[{symbol}] FeatureCache update failed: {e}")

    def get_features(self, symbol: str) -> np.ndarray:
        return self._current.get(symbol, np.full(len(BRAIN_FEATURE_COLS), np.nan))

    def get_feature_dict(self, symbol: str) -> Dict[str, float]:
        feat = self.get_features(symbol)
        return {col: float(feat[i]) for i, col in enumerate(BRAIN_FEATURE_COLS) if not np.isnan(feat[i])}

    def get_history(self, symbol: str, n: int = 20) -> np.ndarray:
        hist = self._history.get(symbol)
        if hist is None:
            return np.full((n, len(BRAIN_FEATURE_COLS)), np.nan)
        idx = self._history_idx.get(symbol, 0)
        count = min(idx, self.RING_SIZE)
        n = min(n, count)
        if n == 0:
            return np.full((0, len(BRAIN_FEATURE_COLS)), np.nan)
        start = (idx - n) % self.RING_SIZE
        if start + n <= self.RING_SIZE:
            return hist[start:start + n].copy()
        else:
            return np.concatenate([hist[start:], hist[:n - (self.RING_SIZE - start)]], axis=0)

    def get_m5_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._m5_cache.get(symbol)

    def get_h1_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._h1_cache.get(symbol)

    def get_h4_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._h4_cache.get(symbol)

    def get_regime(self, symbol: str) -> str:
        return self._regime_cache.get(symbol, "UNKNOWN")

    def get_adx(self, symbol: str) -> float:
        return self._adx_cache.get(symbol, 0.0)

    def get_atr(self, symbol: str) -> float:
        return self._atr_cache.get(symbol, 0.0)

    def set_pending_signals(self, signals: Dict[str, Dict]):
        self._pending_signals = signals

    def get_pending_signals(self) -> Dict[str, Dict]:
        return self._pending_signals

    def _compute_adx(self, highs, lows, closes, period=14):
        n = len(closes)
        if n < period + 2:
            return 0.0
        h, l, c = np.asarray(highs, dtype=float), np.asarray(lows, dtype=float), np.asarray(closes, dtype=float)
        tr = np.maximum(h[1:] - l[1:],
              np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        up = h[1:] - h[:-1]
        dn = l[:-1] - l[1:]
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        alpha = 1.0 / period
        atr_s, pdm_s, mdm_s = float(tr[0]), float(plus_dm[0]), float(minus_dm[0])
        dx_vals = []
        for i in range(1, len(tr)):
            atr_s = atr_s * (1 - alpha) + tr[i] * alpha
            pdm_s = pdm_s * (1 - alpha) + plus_dm[i] * alpha
            mdm_s = mdm_s * (1 - alpha) + minus_dm[i] * alpha
            if atr_s <= 0:
                dx_vals.append(0.0)
                continue
            pdi = (pdm_s / atr_s) * 100
            mdi = (mdm_s / atr_s) * 100
            s = pdi + mdi
            dx_vals.append(abs(pdi - mdi) / s * 100 if s > 0 else 0.0)
        if len(dx_vals) < period:
            return dx_vals[-1] if dx_vals else 0.0
        adx = sum(dx_vals[:period]) / period
        for dx in dx_vals[period:]:
            adx = adx * (1 - alpha) + dx * alpha
        return round(adx, 2)

    def _classify_regime(self, closes, highs, lows, atr):
        if len(closes) < 40:
            return "UNKNOWN"
        short_vol = float(np.mean(np.abs(np.diff(closes[-10:]))))
        long_vol = float(np.mean(np.abs(np.diff(closes[-40:]))))
        vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
        direction_changes = 0
        for i in range(len(closes) - 10, len(closes)):
            if i > 0 and np.sign(closes[i] - closes[i - 1]) != np.sign(closes[i - 1] - closes[i - 2]):
                direction_changes += 1
        avg_range = float(np.mean(highs[-20:] - lows[-20:]))
        atr_ratio = avg_range / atr if atr > 0 else 1.0
        if vol_ratio > 1.5 and atr_ratio > 1.2:
            return "VOLATILE"
        if direction_changes > 12:
            return "STAGNANT"
        if vol_ratio > 1.2 and atr_ratio > 0.8:
            return "TRENDING"
        return "RANGING"

    def _count_structure(self, highs, lows, lookback=5):
        n = len(highs)
        if n < lookback * 2 + 1:
            return 0, 0, 0, 0
        hh, hl, lh, ll = 0, 0, 0, 0
        for i in range(lookback, n - lookback):
            if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
                if i >= 2 * lookback:
                    prev_hh = False
                    for k in range(i - lookback, i - 2 * lookback, -1):
                        if k >= lookback and all(highs[k] >= highs[k - j] for j in range(1, lookback + 1)):
                            prev_hh = True
                            break
                    if prev_hh and highs[i] > highs[k]:
                        hh += 1
            if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
                if i >= 2 * lookback:
                    prev_ll = False
                    for k in range(i - lookback, i - 2 * lookback, -1):
                        if k >= lookback and all(lows[k] <= lows[k - j] for j in range(1, lookback + 1)):
                            prev_ll = True
                            break
                    if prev_ll and lows[i] < lows[k]:
                        ll += 1
                else:
                    hl += 1
        return hh, hl, lh, ll

    def _support_resistance_dist(self, price, highs, lows):
        if len(highs) < 10:
            return price, price
        recent_h = highs[-20:]
        recent_l = lows[-20:]
        support = float(np.min(recent_l))
        resistance = float(np.max(recent_h))
        return price - support, resistance - price

    def _detect_structure_break(self, highs, lows, closes):
        if len(closes) < 15:
            return False
        recent_high = float(np.max(highs[-10:]))
        prev_high = float(np.max(highs[-20:-10])) if len(highs) >= 20 else recent_high
        recent_low = float(np.min(lows[-10:]))
        prev_low = float(np.min(lows[-20:-10])) if len(lows) >= 20 else recent_low
        if closes[-1] > prev_high or closes[-1] < prev_low:
            return True
        return False

    def _structure_quality(self, highs, lows, closes):
        if len(closes) < 20:
            return 0.5
        ranges = highs[-20:] - lows[-20:]
        avg_range = float(np.mean(ranges))
        direction = np.mean(np.diff(closes[-10:]))
        consistency = 1.0 - min(1.0, float(np.std(np.diff(closes[-10:]))) / (avg_range if avg_range > 0 else 1))
        return float(np.clip(consistency + abs(direction) / avg_range if avg_range > 0 else 0.5, 0, 1))
