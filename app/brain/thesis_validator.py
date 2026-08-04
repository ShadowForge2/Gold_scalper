"""ThesisValidator — Layer 2: Entry Condition Tracking.

Tracks why a trade was entered and evaluates whether the thesis is still valid.
Causal reasoning: "I entered because X. Is X still true?"

Runs every M5 boundary. Cost: ~2ms. RAM: ~6MB.
"""
import os
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger("GoldScalper.Brain.Thesis")

THESIS_FEATURE_COLS = [
    "pnl_atr_ratio", "bars_since_entry",
    "ml_confidence_entry", "ml_confidence_current", "ml_confidence_delta",
    "adx_entry", "adx_current", "adx_delta",
    "regime_entry", "regime_current", "regime_changed",
    "momentum_5m", "momentum_15m", "price_acceleration",
    "consecutive_bars_in_direction",
    "volatility_percentile", "atr_ratio",
    "structure_quality_score", "structure_break_detected",
    "distance_to_sl", "distance_to_tp",
    "session_active", "is_friday",
    "opportunity_cost_per_hour",
    "recent_win_rate_20",
]


class ThesisValidator:
    """Evaluates whether the original trade thesis is still valid.

    Two modes:
    1. Fast path (rule-based): instant, no model needed
    2. ML path (ONNX model): more accurate, used when uncertain

    Output: thesis_validity (0-100%), confidence
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self._onnx_session = None
        if model_path:
            candidates = [model_path]
            if model_path.endswith(".onnx"):
                candidates.append(model_path.replace(".onnx", ".joblib"))
            for p in candidates:
                if os.path.exists(p):
                    self._load_model(p)
                    if self.model is not None or self._onnx_session is not None:
                        break
        self._entry_conditions: Dict[str, Dict] = {}

    def _load_model(self, path: str):
        try:
            if path.endswith(".onnx"):
                try:
                    import onnxruntime as ort
                    self._onnx_session = ort.InferenceSession(path)
                    logger.info("ThesisValidator ONNX model loaded: %s", path)
                    return
                except ImportError:
                    logger.warning("onnxruntime not installed, falling back to rule-based")
                    return
            import joblib
            self.model = joblib.load(path)
            logger.info("ThesisValidator model loaded: %s", path)
        except Exception as e:
            logger.warning("ThesisValidator model load failed: %s", e)

    def record_entry(self, symbol: str, entry_data: Dict):
        """Record entry conditions for later comparison."""
        self._entry_conditions[symbol] = {
            "entry_time": time.time(),
            "direction": entry_data.get("direction", "BUY"),
            "entry_price": entry_data.get("price", 0),
            "ml_confidence": entry_data.get("ml_confidence", entry_data.get("score", 0.5)),
            "adx_at_entry": entry_data.get("adx_at_entry", 0),
            "regime_at_entry": entry_data.get("regime_at_entry", "UNKNOWN"),
            "atr_at_entry": entry_data.get("atr", 0),
            "sl": entry_data.get("sl", 0),
            "tp1": entry_data.get("tp1", 0),
            "score": entry_data.get("score", 0),
            "signal_type": entry_data.get("signal_type", ""),
            "bias_agreement": entry_data.get("bias_agreement", 0.5),
        }

    def evaluate(self, symbol: str, current_features: Dict, perception: Dict) -> Dict:
        """Evaluate thesis validity.

        Returns:
            {
                "validity": 0-100 (higher = more valid),
                "confidence": 0-1,
                "method": "rule" | "ml",
                "degraded_conditions": [...],
                "strong_conditions": [...],
                "recommendation": "HOLD" | "EXIT" | "UNCERTAIN"
            }
        """
        entry = self._entry_conditions.get(symbol)
        if not entry:
            return {"validity": 50, "confidence": 0.3, "method": "none",
                    "degraded_conditions": [], "strong_conditions": [],
                    "recommendation": "UNCERTAIN"}

        rule_result = self._rule_based_evaluate(symbol, entry, current_features, perception)

        if self.model is not None or self._onnx_session is not None:
            ml_result = self._ml_evaluate(symbol, entry, current_features, perception)
            if ml_result and ml_result.get("confidence", 0) > 0.5:
                combined_validity = 0.4 * rule_result["validity"] + 0.6 * ml_result["validity"]
                combined_conf = max(rule_result["confidence"], ml_result["confidence"])
                rule_result["validity"] = combined_validity
                rule_result["confidence"] = combined_conf
                rule_result["method"] = "combined"

        if rule_result["validity"] > 70:
            rule_result["recommendation"] = "HOLD"
        elif rule_result["validity"] < 30:
            rule_result["recommendation"] = "EXIT"
        else:
            rule_result["recommendation"] = "UNCERTAIN"

        return rule_result

    def _rule_based_evaluate(self, symbol: str, entry: Dict, features: Dict, perception: Dict) -> Dict:
        score = 50.0
        degraded = []
        strong = []

        ml_now = features.get("ml_confidence_current", entry["ml_confidence"])
        ml_entry = entry["ml_confidence"]
        ml_delta = ml_now - ml_entry
        ml_rolling = features.get("ml_confidence_rolling_avg", ml_now)
        ml_trend = features.get("ml_confidence_trend", 0)

        if ml_delta < -0.15:
            score -= 20
            degraded.append(f"ML confidence dropped {ml_delta:.2f}")
        elif ml_delta < -0.08:
            score -= 10
            degraded.append(f"ML confidence declining {ml_delta:.2f}")
        elif ml_delta > 0.05:
            score += 10
            strong.append(f"ML confidence improved +{ml_delta:.2f}")

        if ml_trend < -0.02:
            score -= 10
            degraded.append(f"ML trend negative ({ml_trend:.3f})")

        adx_now = features.get("adx_current", 0)
        adx_entry = entry.get("adx_at_entry", 0)
        adx_delta = adx_now - adx_entry

        if adx_delta < -10:
            score -= 15
            degraded.append(f"ADX dropped {adx_delta:.1f} (trend weakening)")
        elif adx_delta < -5:
            score -= 8
            degraded.append(f"ADX declining ({adx_delta:.1f})")
        elif adx_delta > 5:
            score += 8
            strong.append(f"ADX improving +{adx_delta:.1f}")

        regime_now = perception.get("regime", "UNKNOWN")
        regime_entry = entry.get("regime_at_entry", "UNKNOWN")
        if regime_now != regime_entry:
            score -= 12
            degraded.append(f"Regime shift: {regime_entry} -> {regime_now}")
        elif regime_now == "TRENDING":
            score += 5
            strong.append("Market still trending")
        elif regime_now == "STAGNANT":
            score -= 15
            degraded.append("Market stagnant — no momentum")

        momentum = features.get("momentum_5m", 0)
        accel = features.get("price_acceleration", 0)
        direction = entry.get("direction", "BUY")
        if direction == "BUY" and momentum < -0.2:
            score -= 10
            degraded.append(f"Momentum against trade: {momentum:.3f}")
        elif direction == "SELL" and momentum > 0.2:
            score -= 10
            degraded.append(f"Momentum against trade: {momentum:.3f}")
        elif abs(momentum) > 0.2:
            score += 5
            strong.append(f"Momentum aligned: {momentum:.3f}")

        if accel < -0.1:
            score -= 5
            degraded.append(f"Decelerating ({accel:.3f})")

        pnl_atr = features.get("pnl_atr_ratio", 0)
        if pnl_atr < -1.0:
            score -= 15
            degraded.append(f"Heavy loss: {pnl_atr:.2f} ATR")
        elif pnl_atr < -0.5:
            score -= 8
            degraded.append(f"Losing: {pnl_atr:.2f} ATR")
        elif pnl_atr > 0.5:
            score += 10
            strong.append(f"In profit: +{pnl_atr:.2f} ATR")

        is_friday = features.get("is_friday", 0) > 0.5
        mins_to_close = features.get("minutes_to_daily_close", 999)
        if is_friday and mins_to_close < 120:
            score -= 10
            degraded.append(f"Friday close in {mins_to_close:.0f}m")

        time_ctx = perception.get("time_context", {})
        if not time_ctx.get("is_prime_hours", True):
            score -= 8
            degraded.append("Outside prime trading hours")

        volatility = perception.get("volatility", {})
        if volatility.get("contracting", False):
            score -= 5
            degraded.append("Volatility contracting")

        return {
            "validity": float(np.clip(score, 0, 100)),
            "confidence": 0.7,
            "method": "rule",
            "degraded_conditions": degraded,
            "strong_conditions": strong,
            "recommendation": "HOLD",
            "details": {
                "ml_delta": ml_delta,
                "adx_delta": adx_delta,
                "regime_shift": regime_now != regime_entry,
                "pnl_atr": pnl_atr,
                "momentum": momentum,
            }
        }

    def _ml_evaluate(self, symbol: str, entry: Dict, features: Dict, perception: Dict) -> Optional[Dict]:
        try:
            feat_vec = []
            for col in THESIS_FEATURE_COLS:
                val = features.get(col, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                feat_vec.append(float(val))

            x = np.array([feat_vec], dtype=np.float32)

            if self._onnx_session is not None:
                input_name = self._onnx_session.get_inputs()[0].name
                probs = self._onnx_session.run(None, {input_name: x})[0]
            elif self.model is not None:
                probs = self.model.predict_proba(x)[0]
            else:
                return None

            if hasattr(probs, '__len__') and len(probs) >= 2:
                hold_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            else:
                hold_prob = float(probs[0])

            validity = hold_prob * 100
            return {
                "validity": validity,
                "confidence": 0.8,
                "method": "ml",
            }
        except Exception as e:
            logger.debug("ThesisValidator ML eval failed: %s", e)
            return None

    def clear_entry(self, symbol: str):
        self._entry_conditions.pop(symbol, None)
