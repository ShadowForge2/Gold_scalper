"""BrainOrchestrator — Cascade Intelligence Engine.

The main entry point for the Brain. Implements the cascade logic:
  Layer 1 (Perception) → ALWAYS
  Layer 2 (Thesis) → ALWAYS
  → If HIGH confidence: DECIDE NOW (skip rest)
  → If UNCERTAIN: Layer 3 (Opportunity) → Layer 4 (Digital Twin)
  Layer 5 (MetaFusion) → ALWAYS
  Layer 6 (Reflection) → POST-TRADE

Memory budget: ~440 MB total (142 MB Brain portion)
CPU budget: ~8ms avg, ~70ms worst case
"""
import os
import time
import logging
from typing import Dict, List, Optional

import numpy as np

import config as cfg

from app.brain.feature_cache import FeatureCache
from app.brain.perception import PerceptionEngine
from app.brain.thesis_validator import ThesisValidator
from app.brain.opportunity import OpportunityScanner
from app.brain.digital_twin import DigitalTwin
from app.brain.meta_fusion import MetaFusion
from app.brain.reflection import ReflectionEngine

logger = logging.getLogger("GoldScalper.Brain")


class BrainOrchestrator:
    """The Brain — cascade intelligence for trade management.

    Usage:
        brain = BrainOrchestrator(symbols=["XAUUSD", "US100", "US500", "US30"])
        brain.init_symbol("US100")

        # Every tick:
        brain.update_features(symbol, m1_data, current_price, entry_info)

        # Every M5 boundary when IN_TRADE:
        decision = brain.evaluate(symbol, entry_info, portfolio_heat)
        if decision["action"] == "EXIT":
            # close the trade
    """

    def __init__(self, symbols: List[str], model_dir: str = "models/brain",
                 candle_models: Optional[Dict] = None):
        self.symbols = symbols
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.cache = FeatureCache()
        self.perception = PerceptionEngine(self.cache)
        self.thesis = ThesisValidator(
            model_path=os.path.join(model_dir, "thesis_validator.onnx")
        )
        self.opportunity = OpportunityScanner(self.cache, symbols)
        self.twin = DigitalTwin(
            calibrator_path=os.path.join(model_dir, "twin_calibrator.onnx"),
            n_sims=getattr(cfg, "BRAIN_SIM_COUNT", 800),
        )
        self.fusion = MetaFusion(model_dir=model_dir)
        self.reflection = ReflectionEngine(log_dir="data/brain_decisions")

        # Candle ML models — the Brain monitors trades using the SAME Candle ML
        # model that opened them, so the two work together (live confidence and
        # direction feed the thesis every boundary instead of a frozen entry
        # score). Callers (the bot) pass their already-loaded instances; if none
        # are given, the Brain loads them itself from config paths.
        self._candle_models: Dict[str, object] = {}
        if candle_models:
            for sym in symbols:
                cm = candle_models.get(sym)
                if cm is not None and getattr(cm, "model", None) is not None:
                    self._candle_models[sym] = cm
        else:
            self._load_candle_models()
        logger.info("Brain: %d/%d symbols have a Candle ML model loaded",
                    len(self._candle_models), len(symbols))

        # Cascade thresholds from config (env-tunable on Render).
        self._thesis_hold_thr = getattr(cfg, "BRAIN_THESIS_HOLD_THRESHOLD", 75)
        self._thesis_exit_thr = getattr(cfg, "BRAIN_THESIS_EXIT_THRESHOLD", 25)
        self._thesis_conf_min = getattr(cfg, "BRAIN_THESIS_CONFIDENCE_MIN", 0.6)

        self._last_eval_time: Dict[str, float] = {}
        self._eval_interval = 280  # ~M5 boundary (seconds)
        self._entry_info_cache: Dict[str, Dict] = {}
        self._trade_results: Dict[str, Dict] = {}

        for sym in symbols:
            self.cache.init_symbol(sym)

        logger.info(
            "Brain initialized: %d symbols, %d models loaded",
            len(symbols),
            len(self.fusion._models),
        )

    def init_symbol(self, symbol: str):
        self.cache.init_symbol(symbol)

    def _load_candle_models(self):
        """Load the per-symbol Candle ML models (XAUUSD fallback) so the Brain
        can watch the same model that opened each trade."""
        try:
            from app.candle_ml import CandleML
        except Exception as exc:
            logger.warning("Brain: CandleML import failed (%s) — live candle read disabled", exc)
            return
        paths = getattr(cfg, "CANDLE_ML_MODEL_PATHS", {})
        fallback = paths.get("XAUUSD")
        for sym in self.symbols:
            path = paths.get(sym) or fallback
            if not path or not os.path.exists(path):
                logger.warning("Brain: no Candle ML model for %s at %s", sym, path)
                continue
            try:
                self._candle_models[sym] = CandleML(model_path=path)
            except Exception as exc:
                logger.warning("Brain: failed to load Candle ML for %s (%s)", sym, exc)

    def _live_candle_read(self, symbol: str, m1_data) -> Optional[Dict]:
        """Run the symbol's Candle ML model on current M1 data.

        Returns the live read used to supervise the open trade:
            {"prob_up": float, "m1_direction": int,
             "confidence": float, "call": "BUY"|"SELL"|None}
        so the Brain and Candle ML work together on the same signal.
        """
        cm = self._candle_models.get(symbol)
        if cm is None:
            return None
        try:
            from app.candle_ml import compute_candle_features
            feats = compute_candle_features(m1_data)
            if feats is None or len(feats) == 0:
                return None
            prob_up = cm.predict_proba(feats)
            if prob_up is None or np.isnan(prob_up):
                return None
            m1_dir = int(np.sign(feats["m1_first_dir"].iloc[-1])) if "m1_first_dir" in feats else 0
            if np.isnan(m1_dir):
                m1_dir = 0
            thr = getattr(cfg, "CANDLE_ML_CONFIDENCE_THRESHOLDS", {}).get(symbol, 0.60)
            call = cm.predict(prob_up, m1_dir, confidence_threshold=thr)
            return {
                "prob_up": float(prob_up),
                "m1_direction": int(m1_dir),
                "confidence": float(max(prob_up, 1 - prob_up)),
                "call": call,
            }
        except Exception as exc:
            logger.debug("Brain: live candle read failed for %s (%s)", symbol, exc)
            return None

    def update_features(self, symbol: str, m1_data, current_price: float,
                        entry_info: Optional[Dict] = None):
        """Update feature cache. Called every tick."""
        if entry_info:
            self._entry_info_cache[symbol] = entry_info
        # Live Candle ML read — let the Brain monitor the trade with the same
        # model that generated the entry signal.
        live = self._live_candle_read(symbol, m1_data)
        if live is not None and entry_info:
            pos_dir = str(entry_info.get("direction", "BUY"))
            bias_agree = 1.0 if (live["call"] == pos_dir) else (0.0 if live["call"] is not None else 0.5)
            entry_info = dict(entry_info)
            entry_info["ml_confidence_current"] = live["confidence"]
            entry_info["bias_agreement"] = bias_agree
            entry_info["live_prob_up"] = live["prob_up"]
            self._entry_info_cache[symbol] = entry_info
        self.cache.update_tick(symbol, m1_data, current_price, entry_info)

    def record_entry(self, symbol: str, entry_data: Dict):
        """Record trade entry for thesis tracking."""
        self.thesis.record_entry(symbol, entry_data)
        self._entry_info_cache[symbol] = entry_data

    def evaluate(self, symbol: str, current_pnl: float = 0.0,
                 current_price: float = 0.0, entry_price: float = 0.0,
                 direction: str = "BUY", atr: float = 0.0,
                 minutes_held: float = 0.0, portfolio_heat: float = 0.0,
                 consecutive_losses: int = 0,
                 pending_signals: Optional[Dict] = None) -> Dict:
        """Full Brain evaluation. Call on M5 boundary when IN_TRADE.

        Returns:
            {
                "action": "HOLD" | "EXIT",
                "confidence": 0-1,
                "reason": str,
                "cascade_path": str,
                "layers_evaluated": int,
                "layer_results": {...},
                "timing_ms": float,
            }
        """
        t0 = time.time()
        layer_results = {}
        layers_evaluated = 0

        # ── LAYER 1: PERCEPTION (always runs, ~3ms) ──
        perception = self.perception.perceive(symbol)
        layer_results["perception"] = perception
        layers_evaluated += 1

        # ── LAYER 2: THESIS VALIDATOR (always runs, ~2ms) ──
        features = self.cache.get_feature_dict(symbol)
        thesis = self.thesis.evaluate(symbol, features, perception)
        layer_results["thesis"] = thesis
        layers_evaluated += 1

        thesis_validity = thesis.get("validity", 50)
        thesis_confidence = thesis.get("confidence", 0.5)

        # ── CASCADE DECISION POINT ──
        if thesis_validity > self._thesis_hold_thr and thesis_confidence > self._thesis_conf_min:
            # HIGH CONFIDENCE HOLD — skip expensive layers
            result = self._fast_hold(symbol, thesis, perception, features,
                                     current_pnl, minutes_held)
            result["cascade_path"] = "fast_hold"
            result["layers_evaluated"] = layers_evaluated
            result["layer_results"] = layer_results
            result["timing_ms"] = (time.time() - t0) * 1000
            return result

        if thesis_validity < self._thesis_exit_thr and thesis_confidence > self._thesis_conf_min:
            # HIGH CONFIDENCE EXIT — skip expensive layers
            result = self._fast_exit(symbol, thesis, perception, features,
                                     current_pnl, minutes_held, "thesis_failing")
            result["cascade_path"] = "fast_exit"
            result["layers_evaluated"] = layers_evaluated
            result["layer_results"] = layer_results
            result["timing_ms"] = (time.time() - t0) * 1000
            return result

        # ── UNCERTAIN PATH — run full cascade ──

        # ── LAYER 3: OPPORTUNITY SCANNER (~8ms) ──
        if pending_signals:
            self.opportunity.update_signals(pending_signals)
        opportunity = self.opportunity.scan(symbol, features, perception.get("regime", "UNKNOWN"))
        layer_results["opportunity"] = opportunity
        layers_evaluated += 1

        # ── LAYER 4: DIGITAL TWIN (~35ms) ──
        dt_result = self.twin.simulate(
            current_price=current_price if current_price > 0 else entry_price + current_pnl * atr,
            atr=atr if atr > 0 else 1.0,
            direction=direction,
            entry_price=entry_price if entry_price > 0 else current_price,
            current_pnl=current_pnl,
            regime=perception.get("regime", "UNKNOWN"),
            minutes_held=minutes_held,
            opportunity_data=opportunity.get("best_opportunity"),
            features=features,
        )
        layer_results["digital_twin"] = dt_result
        layers_evaluated += 1

        # ── LAYER 5: META-FUSION (always runs, ~5ms) ──
        fusion_decision = self.fusion.decide(
            thesis=thesis,
            perception=perception,
            opportunity=opportunity,
            digital_twin=dt_result,
            features=features,
            portfolio_heat=portfolio_heat,
            consecutive_losses=consecutive_losses,
        )
        layer_results["meta_fusion"] = fusion_decision
        layers_evaluated += 1

        # ── COMPOSE FINAL RESULT ──
        action = fusion_decision["action"]
        confidence = fusion_decision["confidence"]
        method = fusion_decision["method"]

        reasons = []
        if thesis_validity < 40:
            reasons.append(f"thesis weak ({thesis_validity:.0f}%)")
        if opportunity.get("should_switch"):
            reasons.append(f"better opp: {opportunity.get('reason', '')}")
        if dt_result.get("recommendation") == "EXIT":
            reasons.append(f"DT favors exit (EV diff={dt_result.get('ev_diff', 0):.2f})")
        if fusion_decision.get("dissent_detected"):
            reasons.append("ensemble dissent")

        reason = " | ".join(reasons) if reasons else f"ensemble {action.lower()} ({confidence:.0%})"

        result = {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "method": method,
            "thesis_validity": thesis_validity,
            "perception_composite": perception.get("composite_score", 0),
            "opportunity_cost": opportunity.get("opportunity_cost", 0),
            "dt_ev_diff": dt_result.get("ev_diff", 0),
        }

        # Log decision for reflection
        if action == "EXIT":
            self.reflection.record_decision(symbol, fusion_decision, {
                "thesis_validity": thesis_validity,
                "perception_score": perception.get("composite_score", 0),
                "opportunity_cost": opportunity.get("opportunity_cost", 0),
                "dt_recommendation": dt_result.get("recommendation"),
                "pnl_atr": current_pnl,
                "minutes_held": minutes_held,
                "regime": perception.get("regime"),
            })

        result["cascade_path"] = "full_cascade"
        result["layers_evaluated"] = layers_evaluated
        result["layer_results"] = layer_results
        result["timing_ms"] = (time.time() - t0) * 1000

        return result

    def record_exit(self, symbol: str, exit_data: Dict):
        """Record trade exit for reflection."""
        self.reflection.record_exit(symbol, exit_data)
        self.thesis.clear_entry(symbol)
        self._trade_results[symbol] = exit_data

    def _fast_hold(self, symbol, thesis, perception, features, pnl, minutes_held) -> Dict:
        return {
            "action": "HOLD",
            "confidence": min(1.0, thesis.get("confidence", 0.7) + 0.2),
            "reason": f"thesis strong ({thesis['validity']:.0f}%)",
            "method": "fast_hold",
        }

    def _fast_exit(self, symbol, thesis, perception, features, pnl, minutes_held,
                   reason: str = "thesis_failing") -> Dict:
        return {
            "action": "EXIT",
            "confidence": min(1.0, thesis.get("confidence", 0.7) + 0.1),
            "reason": f"thesis failing ({thesis['validity']:.0f}%) — {reason}",
            "method": "fast_exit",
        }

    def get_stats(self) -> Dict:
        return {
            "fusion_accuracy": self.fusion.accuracy,
            "reflection_stats": self.reflection.get_stats(),
            "models_loaded": len(self.fusion._models),
        }
