"""OpportunityScanner — Layer 3: Cross-Symbol Opportunity Cost.

Scans all symbols for pending setups and calculates the opportunity cost
of staying in the current trade vs switching.

Runs when thesis is uncertain. Cost: ~8ms. RAM: ~10MB.
"""
import time
import logging
import numpy as np
from typing import Dict, List, Optional
from app.brain.feature_cache import FeatureCache

logger = logging.getLogger("GoldScalper.Brain.Opportunity")


class OpportunityScanner:
    """Cross-symbol opportunity cost calculator."""

    def __init__(self, cache: FeatureCache, symbols: List[str]):
        self.cache = cache
        self.symbols = symbols
        self._pending_signals: Dict[str, Dict] = {}
        self._last_scan: Dict[str, float] = {}

    def update_signals(self, signals: Dict[str, Dict]):
        """Update with latest signal data from the bot."""
        self._pending_signals = signals
        self.cache.set_pending_signals(signals)

    def scan(self, current_symbol: str, current_features: Dict,
             current_regime: str) -> Dict:
        """Scan all other symbols for better opportunities.

        Returns:
            {
                "best_opportunity": {...} | None,
                "opportunity_count": int,
                "avg_score": float,
                "max_expected_pnl": float,
                "current_rank": int,  # 1=best, N=worst
                "opportunity_cost": float,  # expected value difference
                "should_switch": bool,
                "reason": str,
            }
        """
        opportunities = []
        current_score = current_features.get("ml_confidence_current", 0.5)

        for sym in self.symbols:
            if sym == current_symbol:
                continue

            sig = self._pending_signals.get(sym)
            if sig is None:
                continue

            opp_features = self.cache.get_feature_dict(sym)
            opp_regime = self.cache.get_regime(sym)
            opp_adx = self.cache.get_adx(sym)

            opp_score = sig.get("score", 0)
            opp_direction = sig.get("direction", "")
            opp_atr = sig.get("atr", 0)
            opp_signal_type = sig.get("signal_type", "")

            session_match = self._check_session_match(sym)
            regime_fit = self._check_regime_fit(opp_regime, opp_adx)
            expected_pnl = self._estimate_expected_pnl(opp_score, opp_atr, regime_fit, session_match)

            opportunities.append({
                "symbol": sym,
                "direction": opp_direction,
                "score": opp_score,
                "signal_type": opp_signal_type,
                "atr": opp_atr,
                "regime": opp_regime,
                "adx": opp_adx,
                "session_match": session_match,
                "regime_fit": regime_fit,
                "expected_pnl": expected_pnl,
            })

        if not opportunities:
            return {
                "best_opportunity": None,
                "opportunity_count": 0,
                "avg_score": 0,
                "max_expected_pnl": 0,
                "current_rank": 1,
                "opportunity_cost": 0,
                "should_switch": False,
                "reason": "No pending signals on other symbols",
            }

        opportunities.sort(key=lambda x: x["expected_pnl"], reverse=True)
        best = opportunities[0]
        avg_score = float(np.mean([o["score"] for o in opportunities]))
        max_pnl = best["expected_pnl"]

        all_scores = sorted([o["score"] for o in opportunities] + [current_score], reverse=True)
        current_rank = all_scores.index(current_score) + 1

        holding_ev = self._estimate_hold_ev(current_features)
        switching_ev = max_pnl
        opportunity_cost = switching_ev - holding_ev

        should_switch = opportunity_cost > 0.3 and best["score"] > current_score + 0.1

        reason = ""
        if should_switch:
            reason = (f"{best['symbol']} {best['direction']} "
                     f"(score={best['score']:.2f}) expected +{switching_ev:.2f} ATR "
                     f"vs current holding +{holding_ev:.2f} ATR "
                     f"(cost={opportunity_cost:.2f} ATR)")

        return {
            "best_opportunity": best,
            "opportunity_count": len(opportunities),
            "avg_score": avg_score,
            "max_expected_pnl": max_pnl,
            "current_rank": current_rank,
            "opportunity_cost": opportunity_cost,
            "should_switch": should_switch,
            "reason": reason,
            "all_opportunities": opportunities,
        }

    def _check_session_match(self, symbol: str) -> bool:
        feat = self.cache.get_feature_dict(symbol)
        session = feat.get("session_active", 3)
        return session in (1, 2)

    def _check_regime_fit(self, regime: str, adx: float) -> float:
        if regime == "TRENDING" and adx > 20:
            return 0.9
        elif regime == "VOLATILE":
            return 0.7
        elif regime == "RANGING":
            return 0.4
        elif regime == "STAGNANT":
            return 0.2
        return 0.5

    def _estimate_expected_pnl(self, score: float, atr: float,
                                regime_fit: float, session_match: bool) -> float:
        base = (score - 0.5) * 2.0
        regime_mult = 0.7 + 0.3 * regime_fit
        session_mult = 1.1 if session_match else 0.8
        return base * regime_mult * session_mult

    def _estimate_hold_ev(self, features: Dict) -> float:
        pnl_atr = features.get("pnl_atr_ratio", 0)
        momentum = features.get("momentum_5m", 0)
        ml_conf = features.get("ml_confidence_current", 0.5)

        base = (ml_conf - 0.5) * 2.0
        momentum_bonus = momentum * 0.3
        pnl_signal = pnl_atr * 0.2

        return base + momentum_bonus + pnl_signal
