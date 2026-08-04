"""ReflectionEngine — Layer 6: Post-Trade Learning.

Logs every decision and its outcome. Computes counterfactuals.
Feeds into offline retraining cycles.

Runs post-trade only. Cost: ~0.1ms. RAM: ~3MB.
"""
import os
import json
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger("GoldScalper.Brain.Reflection")


class ReflectionEngine:
    """Post-trade analysis and learning engine.

    Logs decision context, tracks outcomes, computes counterfactuals.
    Data is written to disk for offline retraining.
    """

    def __init__(self, log_dir: str = "data/brain_decisions"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._pending_decisions: Dict[str, Dict] = {}
        self._trade_log: List[Dict] = []
        self._log_file = os.path.join(log_dir, "decisions.jsonl")
        self._stats_file = os.path.join(log_dir, "stats.json")
        self._load_stats()

    def _load_stats(self):
        try:
            if os.path.exists(self._stats_file):
                with open(self._stats_file, "r") as f:
                    self._stats = json.load(f)
            else:
                self._stats = {
                    "total_decisions": 0,
                    "correct_decisions": 0,
                    "total_exits": 0,
                    "exits_that_should_have_held": 0,
                    "holds_that_should_have_exited": 0,
                    "avg_confidence_correct": 0,
                    "avg_confidence_incorrect": 0,
                }
        except Exception:
            self._stats = {}

    def record_decision(self, symbol: str, decision: Dict, context: Dict):
        """Record a Brain decision for later evaluation.

        Called when the Brain makes an EXIT decision.
        """
        self._pending_decisions[symbol] = {
            "decision_time": time.time(),
            "symbol": symbol,
            "action": decision.get("action"),
            "confidence": decision.get("confidence"),
            "method": decision.get("method"),
            "thesis_validity": context.get("thesis_validity"),
            "perception_score": context.get("perception_score"),
            "opportunity_cost": context.get("opportunity_cost"),
            "dt_recommendation": context.get("dt_recommendation"),
            "ensemble_votes": decision.get("model_votes"),
            "pnl_atr": context.get("pnl_atr"),
            "minutes_held": context.get("minutes_held"),
            "regime": context.get("regime"),
        }

    def record_exit(self, symbol: str, exit_data: Dict):
        """Record the actual exit outcome."""
        pending = self._pending_decisions.pop(symbol, None)
        if pending is None:
            return

        record = {
            **pending,
            "exit_time": time.time(),
            "exit_reason": exit_data.get("exit_reason"),
            "exit_pnl": exit_data.get("pnl"),
            "exit_pnl_atr": exit_data.get("pnl", 0) / max(1, exit_data.get("atr", 1)),
            "actual_exit_price": exit_data.get("exit_price"),
        }

        self._write_decision(record)
        self._trade_log.append(record)

    def record_hold_outcome(self, symbol: str, outcome_data: Dict):
        """Record what happened after a HOLD decision."""
        pending = self._pending_decisions.get(symbol)
        if pending is None:
            return

        record = {
            **pending,
            "evaluation_time": time.time(),
            "outcome": "evaluated",
            "pnl_then": outcome_data.get("pnl_atr"),
            "pnl_now": outcome_data.get("current_pnl_atr"),
            "regime_then": outcome_data.get("regime_at_decision"),
            "regime_now": outcome_data.get("current_regime"),
        }

        self._write_decision(record)

    def compute_counterfactual(self, symbol: str, actual_exit_pnl: float,
                                held_pnl: float, decision: Dict) -> Dict:
        """Was the decision correct?"""
        action = decision.get("action", "HOLD")

        if action == "EXIT":
            correct = actual_exit_pnl > held_pnl
            if not correct:
                self._stats["exits_that_should_have_held"] = \
                    self._stats.get("exits_that_should_have_held", 0) + 1
        else:
            correct = held_pnl > actual_exit_pnl
            if not correct:
                self._stats["holds_that_should_have_exited"] = \
                    self._stats.get("holds_that_should_have_exited", 0) + 1

        self._stats["total_decisions"] = self._stats.get("total_decisions", 0) + 1
        if correct:
            self._stats["correct_decisions"] = self._stats.get("correct_decisions", 0) + 1
        self._stats["total_exits"] = self._stats.get("total_exits", 0) + (1 if action == "EXIT" else 0)

        self._save_stats()

        return {
            "correct": correct,
            "actual_pnl": actual_exit_pnl,
            "counterfactual_pnl": held_pnl,
            "edge": actual_exit_pnl - held_pnl,
            "action": action,
        }

    def _write_decision(self, record: Dict):
        try:
            record["timestamp"] = datetime.utcnow().isoformat()
            with open(self._log_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.debug("ReflectionEngine write failed: %s", e)

    def _save_stats(self):
        try:
            total = self._stats.get("total_decisions", 0)
            if total > 0:
                self._stats["accuracy"] = self._stats.get("correct_decisions", 0) / total
            with open(self._stats_file, "w") as f:
                json.dump(self._stats, f, indent=2)
        except Exception:
            pass

    def get_stats(self) -> Dict:
        return self._stats.copy()

    def get_recent_decisions(self, n: int = 20) -> List[Dict]:
        return self._trade_log[-n:]

    def get_training_data(self, min_confidence: float = 0.3) -> List[Dict]:
        """Export decisions for retraining."""
        try:
            if not os.path.exists(self._log_file):
                return []
            records = []
            with open(self._log_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            r = json.loads(line)
                            if r.get("confidence", 0) >= min_confidence:
                                records.append(r)
                        except json.JSONDecodeError:
                            continue
            return records[-10000:]
        except Exception:
            return []
