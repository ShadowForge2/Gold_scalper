"""PerceptionEngine — Layer 1: Market State Classification.

Classifies the current market into regimes, maps structure, and detects momentum.
Runs every tick. Cost: ~3ms. RAM: ~8MB.

This is the "eyes" of the Brain — it sees what the market looks like right now.
"""
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from app.brain.feature_cache import FeatureCache, BRAIN_FEATURE_COLS

logger = logging.getLogger("GoldScalper.Brain.Perception")


class PerceptionEngine:
    """Lightweight market state classifier. Reads from FeatureCache."""

    def __init__(self, cache: FeatureCache):
        self.cache = cache
        self._structure_maps: Dict[str, Dict] = {}
        self._momentum_states: Dict[str, Dict] = {}

    def perceive(self, symbol: str) -> Dict:
        """Full perception snapshot for a symbol.

        Returns:
            {
                "regime": "TRENDING|RANGING|VOLATILE|STAGNANT|REVERSAL|BREAKOUT",
                "regime_confidence": 0.0-1.0,
                "structure": {"trend": ..., "quality": ..., "break_detected": ...},
                "momentum": {"velocity": ..., "acceleration": ..., "state": ...},
                "volatility": {"current": ..., "percentile": ..., "expanding": ...},
                "time_context": {"session": ..., "minutes_to_close": ..., "is_prime": ...},
                "composite_score": float  # 0-100, overall market quality for trading
            }
        """
        feat = self.cache.get_feature_dict(symbol)
        if not feat:
            return self._empty_perception()

        regime = self._classify_regime_deep(symbol, feat)
        structure = self._map_structure(symbol, feat)
        momentum = self._assess_momentum(symbol, feat)
        volatility = self._assess_volatility(symbol, feat)
        time_ctx = self._time_context(symbol, feat)

        composite = self._compute_composite(regime, structure, momentum, volatility, time_ctx)

        return {
            "regime": regime["type"],
            "regime_confidence": regime["confidence"],
            "regime_features": regime,
            "structure": structure,
            "momentum": momentum,
            "volatility": volatility,
            "time_context": time_ctx,
            "composite_score": composite,
        }

    def _classify_regime_deep(self, symbol: str, feat: Dict) -> Dict:
        cache_regime = self.cache.get_regime(symbol)
        adx = feat.get("adx_current", 0)
        vol_pct = feat.get("volatility_percentile", 0.5)
        vol_ratio = feat.get("realized_vs_implied", 1.0)
        bb_width = feat.get("bollinger_width", 0)
        momentum = feat.get("momentum_5m", 0)

        confidence = 0.5
        regime_type = cache_regime

        if adx > 25 and abs(momentum) > 0.3:
            regime_type = "TRENDING"
            confidence = min(1.0, 0.5 + adx / 100)
        elif vol_pct > 0.75 and vol_ratio > 1.3:
            regime_type = "VOLATILE"
            confidence = min(1.0, 0.5 + vol_pct * 0.5)
        elif adx < 15 and abs(momentum) < 0.1:
            regime_type = "STAGNANT"
            confidence = min(1.0, 0.6 + (1 - adx / 50) * 0.4)
        elif adx < 20 and bb_width < 0.02:
            regime_type = "RANGING"
            confidence = 0.6

        hist = self.cache.get_history(symbol, 12)
        if hist is not None and len(hist) >= 6:
            mom_idx = BRAIN_FEATURE_COLS.index("momentum_5m")
            recent_momentum = hist[:, mom_idx]
            recent_momentum = recent_momentum[~np.isnan(recent_momentum)]
            if len(recent_momentum) >= 4:
                mom_trend = recent_momentum[-1] - recent_momentum[-4]
                if regime_type == "STAGNANT" and abs(mom_trend) > 0.3:
                    regime_type = "REVERSAL"
                    confidence = 0.65

        return {
            "type": regime_type,
            "confidence": confidence,
            "adx": adx,
            "vol_percentile": vol_pct,
        }

    def _map_structure(self, symbol: str, feat: Dict) -> Dict:
        adx = feat.get("adx_current", 0)
        hh_count = feat.get("higher_highs_count", 0)
        hl_count = feat.get("higher_lows_count", 0)
        support_dist = feat.get("support_distance", 0)
        resist_dist = feat.get("resistance_distance", 0)
        break_detected = feat.get("structure_break_detected", 0)
        quality = feat.get("structure_quality_score", 0.5)

        if hh_count > 0 and hl_count > 0:
            trend = "BULLISH"
        elif hh_count < 0 and hl_count < 0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        return {
            "trend": trend,
            "higher_highs": hh_count,
            "higher_lows": hl_count,
            "support_distance_atr": support_dist,
            "resistance_distance_atr": resist_dist,
            "break_detected": bool(break_detected),
            "quality_score": quality,
            "adx": adx,
        }

    def _assess_momentum(self, symbol: str, feat: Dict) -> Dict:
        m1 = feat.get("momentum_1m", 0)
        m5 = feat.get("momentum_5m", 0)
        m15 = feat.get("momentum_15m", 0)
        m1h = feat.get("momentum_1h", 0)
        accel = feat.get("price_acceleration", 0)
        consec = feat.get("consecutive_bars_in_direction", 0)

        velocity = (m5 * 0.4 + m15 * 0.3 + m1h * 0.2 + m1 * 0.1)

        if velocity > 0.2 and accel > 0:
            state = "ACCELERATING_UP"
        elif velocity < -0.2 and accel < 0:
            state = "ACCELERATING_DOWN"
        elif velocity > 0.1:
            state = "DRIFTING_UP"
        elif velocity < -0.1:
            state = "DRIFTING_DOWN"
        elif abs(velocity) < 0.05:
            state = "STALLED"
        else:
            state = "NEUTRAL"

        return {
            "velocity": velocity,
            "acceleration": accel,
            "state": state,
            "momentum_1m": m1,
            "momentum_5m": m5,
            "momentum_15m": m15,
            "momentum_1h": m1h,
            "consecutive_bars": consec,
        }

    def _assess_volatility(self, symbol: str, feat: Dict) -> Dict:
        atr = feat.get("atr_current", 0)
        atr_pct = feat.get("volatility_percentile", 0.5)
        ratio = feat.get("realized_vs_implied", 1.0)
        bb_width = feat.get("bollinger_width", 0)

        expanding = ratio > 1.2
        contracting = ratio < 0.8

        return {
            "current_atr": atr,
            "percentile": atr_pct,
            "realized_vs_implied": ratio,
            "bollinger_width": bb_width,
            "expanding": expanding,
            "contracting": contracting,
            "state": "EXPANDING" if expanding else "CONTRACTING" if contracting else "STEADY",
        }

    def _time_context(self, symbol: str, feat: Dict) -> Dict:
        hour = feat.get("hour_of_day_utc", 12)
        session_val = feat.get("session_active", 3)
        session_names = {0: "ASIA", 1: "LONDON", 2: "NEW_YORK", 3: "OUTSIDE"}
        session = session_names.get(int(session_val), "UNKNOWN")

        is_friday = feat.get("is_friday", 0) > 0.5
        mins_to_daily = feat.get("minutes_to_daily_close", 0)
        mins_to_weekly = feat.get("minutes_to_weekly_close", 999)

        is_prime = session in ("LONDON", "NEW_YORK") and not is_friday
        is_close = mins_to_daily < 60

        return {
            "hour_utc": hour,
            "session": session,
            "is_prime_hours": is_prime,
            "is_market_close": is_close,
            "is_friday": is_friday,
            "minutes_to_daily_close": mins_to_daily,
            "minutes_to_weekly_close": mins_to_weekly,
        }

    def _compute_composite(self, regime, structure, momentum, volatility, time_ctx) -> float:
        score = 50.0

        regime_bonus = {
            "TRENDING": 15, "VOLATILE": 10, "REVERSAL": 8,
            "RANGING": 0, "STAGNANT": -15, "UNKNOWN": 0,
        }
        score += regime_bonus.get(regime["type"], 0) * regime["confidence"]

        if structure["quality_score"] > 0.6:
            score += 10
        if structure["break_detected"]:
            score -= 10

        vel = abs(momentum["velocity"])
        if vel > 0.3:
            score += 10
        elif vel < 0.05:
            score -= 15

        if volatility["expanding"]:
            score += 5
        elif volatility["contracting"]:
            score -= 5

        if time_ctx["is_prime_hours"]:
            score += 10
        if time_ctx["is_market_close"]:
            score -= 10

        return float(np.clip(score, 0, 100))

    def _empty_perception(self) -> Dict:
        return {
            "regime": "UNKNOWN",
            "regime_confidence": 0,
            "regime_features": {},
            "structure": {},
            "momentum": {},
            "volatility": {},
            "time_context": {},
            "composite_score": 0,
        }
