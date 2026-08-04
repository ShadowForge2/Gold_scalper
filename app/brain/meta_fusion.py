"""MetaFusion — Layer 5: 5-Model Ensemble Decision Maker.

Combines outputs from all layers into a final HOLD/EXIT decision.
Uses a stacking ensemble: XGBoost + LightGBM + CatBoost + Neural Net + Random Forest.

Runs every M5 boundary. Cost: ~5ms. RAM: ~18MB.
"""
import os
import logging
import numpy as np
from typing import Dict, Optional, Tuple

import config as cfg

logger = logging.getLogger("GoldScalper.Brain.MetaFusion")

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

META_FEATURE_COLS = [
    "thesis_validity",
    "thesis_confidence",
    "perception_composite",
    "regime_type_encoded",
    "opportunity_cost",
    "opportunity_count",
    "dt_ev_diff",
    "dt_hold_win_pct",
    "dt_confidence",
    "pnl_atr_ratio",
    "ml_confidence_current",
    "adx_current",
    "momentum_5m",
    "volatility_percentile",
    "bars_since_entry",
    "session_active",
    "is_friday",
    "recent_win_rate_20",
    "consecutive_losses",
    "portfolio_heat",
]

REGIME_ENCODING = {
    "TRENDING": 1, "RANGING": -1, "VOLATILE": 0.5,
    "STAGNANT": -0.5, "REVERSAL": 0.75, "UNKNOWN": 0,
}


class MetaFusion:
    """5-model ensemble for final HOLD/EXIT decision.

    Models are loaded as ONNX (primary) or joblib (fallback).
    If no models are trained yet, falls back to rule-based fusion.
    """

    def __init__(self, model_dir: str = "models/brain"):
        self.model_dir = model_dir
        self._models = {}
        self._model_weights = {
            "xgb": getattr(cfg, "BRAIN_FUSION_WEIGHT_XGB", 0.25),
            "lgb": getattr(cfg, "BRAIN_FUSION_WEIGHT_LGB", 0.25),
            "cat": getattr(cfg, "BRAIN_FUSION_WEIGHT_CAT", 0.20),
            "nn": getattr(cfg, "BRAIN_FUSION_WEIGHT_NN", 0.20),
            "rf": getattr(cfg, "BRAIN_FUSION_WEIGHT_RF", 0.10),
        }
        self._load_models()
        self._recent_decisions = []
        self._decision_accuracy = 0.5

    def _load_models(self):
        model_files = {
            "xgb": "meta_fusion_xgb.onnx",
            "lgb": "meta_fusion_lgb.onnx",
            "cat": "meta_fusion_cat.onnx",
            "nn": "meta_fusion_nn.onnx",
            "rf": "meta_fusion_rf.onnx",
        }

        for name, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            if not os.path.exists(path):
                fallback = path.replace(".onnx", ".joblib")
                if os.path.exists(fallback):
                    try:
                        import joblib
                        self._models[name] = joblib.load(fallback)
                        logger.info("MetaFusion loaded %s (joblib): %s", name, fallback)
                    except Exception as e:
                        logger.warning("MetaFusion failed to load %s: %s", name, e)
                continue

            if _HAS_ONNX:
                try:
                    self._models[name] = ort.InferenceSession(path)
                    logger.info("MetaFusion loaded %s (ONNX): %s", name, path)
                except Exception as e:
                    logger.warning("MetaFusion ONNX load failed for %s: %s", name, e)
            else:
                try:
                    import joblib
                    self._models[name] = joblib.load(path.replace(".onnx", ".joblib"))
                except Exception:
                    pass

        logger.info("MetaFusion: %d/%d models loaded", len(self._models), len(model_files))

    def decide(self, thesis: Dict, perception: Dict, opportunity: Dict,
               digital_twin: Dict, features: Dict, portfolio_heat: float = 0.0,
               consecutive_losses: int = 0) -> Dict:
        """Final decision from all layer inputs.

        Returns:
            {
                "action": "HOLD" | "EXIT",
                "confidence": 0-1,
                "method": "ensemble" | "rule",
                "model_votes": {...},
                "vote_weights": {...},
                "consensus_strength": float,
                "dissent_detected": bool,
            }
        """
        meta_features = self._build_meta_features(
            thesis, perception, opportunity, digital_twin, features,
            portfolio_heat, consecutive_losses,
        )

        if len(self._models) >= 3:
            return self._ensemble_decide(meta_features)
        else:
            return self._rule_decide(thesis, perception, opportunity, digital_twin, features)

    def _build_meta_features(self, thesis, perception, opportunity, dt, features,
                              portfolio_heat, consecutive_losses) -> np.ndarray:
        regime_str = perception.get("regime", "UNKNOWN") if isinstance(perception, dict) else "UNKNOWN"

        vals = [
            thesis.get("validity", 50) / 100.0,
            thesis.get("confidence", 0.5),
            perception.get("composite_score", 50) / 100.0 if isinstance(perception, dict) else 0.5,
            REGIME_ENCODING.get(regime_str, 0),
            opportunity.get("opportunity_cost", 0),
            opportunity.get("opportunity_count", 0) / 10.0,
            dt.get("ev_diff", 0),
            dt.get("hold_win_pct", 0.5),
            dt.get("confidence", 0.5),
            features.get("pnl_atr_ratio", 0),
            features.get("ml_confidence_current", 0.5),
            features.get("adx_current", 0) / 50.0,
            features.get("momentum_5m", 0),
            features.get("volatility_percentile", 0.5),
            features.get("bars_since_entry", 0) / 24.0,
            features.get("session_active", 3) / 3.0,
            features.get("is_friday", 0),
            features.get("recent_win_rate_20", 0.5),
            consecutive_losses / 5.0,
            portfolio_heat,
        ]

        return np.array([vals], dtype=np.float32)

    def _ensemble_decide(self, features: np.ndarray) -> Dict:
        votes = {}
        probabilities = {}

        for name, model in self._models.items():
            try:
                if hasattr(model, 'run') and _HAS_ONNX and isinstance(model, ort.InferenceSession):
                    input_name = model.get_inputs()[0].name
                    probs = model.run(None, {input_name: features})[0]
                elif hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features)[0]
                elif hasattr(model, 'predict'):
                    pred = model.predict(features)[0]
                    probs = np.array([1 - pred, pred])
                else:
                    continue

                if hasattr(probs, '__len__') and len(probs) >= 2:
                    exit_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                else:
                    exit_prob = float(probs[0])

                probabilities[name] = exit_prob
                votes[name] = "EXIT" if exit_prob > 0.5 else "HOLD"
            except Exception as e:
                logger.debug("MetaFusion model %s failed: %s", name, e)
                continue

        if not votes:
            return self._rule_decide({}, {}, {}, {}, {})

        weighted_exit_prob = 0.0
        total_weight = 0.0
        for name, prob in probabilities.items():
            w = self._model_weights.get(name, 0.2)
            weighted_exit_prob += prob * w
            total_weight += w

        if total_weight > 0:
            weighted_exit_prob /= total_weight

        hold_count = sum(1 for v in votes.values() if v == "HOLD")
        exit_count = sum(1 for v in votes.values() if v == "EXIT")
        total = len(votes)

        consensus = max(hold_count, exit_count) / total if total > 0 else 0.5
        dissent = min(hold_count, exit_count) / total > 0.3 if total >= 4 else False

        confidence = abs(weighted_exit_prob - 0.5) * 2 * consensus
        confidence = min(1.0, confidence)

        action = "EXIT" if weighted_exit_prob > 0.5 else "HOLD"

        if consensus < 0.6 and total >= 3:
            confidence *= 0.7

        return {
            "action": action,
            "confidence": confidence,
            "method": "ensemble",
            "model_votes": votes,
            "model_probabilities": probabilities,
            "vote_weights": self._model_weights,
            "consensus_strength": consensus,
            "dissent_detected": dissent,
            "weighted_exit_prob": weighted_exit_prob,
        }

    def _rule_decide(self, thesis, perception, opportunity, dt, features) -> Dict:
        score = 0.0
        weights = 0.0

        thesis_valid = thesis.get("validity", 50)
        score += (100 - thesis_valid) / 100.0 * 0.35
        weights += 0.35

        composite = perception.get("composite_score", 50) if isinstance(perception, dict) else 50
        score += (100 - composite) / 100.0 * 0.15
        weights += 0.15

        opp_cost = opportunity.get("opportunity_cost", 0)
        if opp_cost > 0:
            score += min(1.0, opp_cost) * 0.20
        weights += 0.20

        dt_rec = dt.get("recommendation", "HOLD")
        if dt_rec == "EXIT":
            score += 0.20 * dt.get("confidence", 0.5)
        weights += 0.20

        pnl_atr = features.get("pnl_atr_ratio", 0)
        if pnl_atr < -1.0:
            score += 0.10
        weights += 0.10

        exit_prob = score / weights if weights > 0 else 0.5
        action = "EXIT" if exit_prob > 0.5 else "HOLD"
        confidence = abs(exit_prob - 0.5) * 2

        return {
            "action": action,
            "confidence": confidence,
            "method": "rule",
            "model_votes": {},
            "vote_weights": {},
            "consensus_strength": 1.0,
            "dissent_detected": False,
            "weighted_exit_prob": exit_prob,
        }

    def record_outcome(self, decision_correct: bool):
        self._recent_decisions.append(decision_correct)
        if len(self._recent_decisions) > 100:
            self._recent_decisions = self._recent_decisions[-100:]
        self._decision_accuracy = sum(self._recent_decisions) / len(self._recent_decisions)

    @property
    def accuracy(self) -> float:
        return self._decision_accuracy
