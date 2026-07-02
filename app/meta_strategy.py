import numpy as np
from collections import deque
from typing import Dict, Optional, List
from datetime import datetime

import config as cfg


class RollingTracker:
    def __init__(self, window: int = 20):
        self.window = window
        self._trades: deque = deque(maxlen=window)

    def add_trade(self, pnl: float, bias_strength: float):
        self._trades.append({
            "pnl": pnl,
            "won": pnl > 0,
            "bias_strength": bias_strength,
        })

    @property
    def count(self) -> int:
        return len(self._trades)

    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        return sum(1 for t in self._trades if t["won"]) / len(self._trades)

    def profit_factor(self) -> float:
        wins = [t["pnl"] for t in self._trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in self._trades if t["pnl"] < 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    def avg_pnl(self) -> float:
        if not self._trades:
            return 0.0
        return np.mean([t["pnl"] for t in self._trades])

    def consecutive_losses(self) -> int:
        count = 0
        for t in reversed(self._trades):
            if t["won"]:
                break
            count += 1
        return count

    def avg_bias_strength(self) -> float:
        if not self._trades:
            return 0.0
        return np.mean([t["bias_strength"] for t in self._trades])


class MetaStrategy:
    def __init__(self):
        self.tracker = RollingTracker(window=cfg.META_LOOKBACK_WINDOW)
        self.base_entry_threshold = cfg.SIGNAL_ENTRY_THRESHOLD
        self.base_lot_multiplier = cfg.LOT_MULTIPLIER
        self.base_trades_per_event = cfg.MAX_TRADES_PER_EVENT
        self.base_bias_strength_min = cfg.BIAS_STRENGTH_MIN

        self._current_threshold = self.base_entry_threshold
        self._current_lot_mult = self.base_lot_multiplier
        self._current_trades_per_event = self.base_trades_per_event
        self._current_bias_min = self.base_bias_strength_min

        self._last_regime: str = "UNKNOWN"
        self._regime_since: datetime = datetime.now()
        self._total_trades = 0
        self._peak_balance: Optional[float] = None
        self._last_threshold_change: Optional[datetime] = None
        self._cooldown_trades = 0

    @property
    def current_threshold(self) -> float:
        return self._current_threshold

    @property
    def current_lot_mult(self) -> float:
        return self._current_lot_mult

    @property
    def current_trades_per_event(self) -> int:
        return self._current_trades_per_event

    @property
    def current_bias_min(self) -> float:
        return self._current_bias_min

    @property
    def regime(self) -> str:
        return self._last_regime

    def update_peak(self, balance: float):
        if self._peak_balance is None or balance > self._peak_balance:
            self._peak_balance = balance

    def dd_pct(self, balance: float) -> float:
        if not self._peak_balance or self._peak_balance <= 0:
            return 0.0
        return (self._peak_balance - balance) / self._peak_balance * 100

    def _detect_regime(self, bias_summary: Dict, balance: float = 0.0) -> str:
        strength = bias_summary.get("strength", 0.0)

        if self.tracker.count < cfg.META_MIN_TRADES_FOR_REGIME:
            if strength >= 0.7:
                return "TRENDING_STRONG"
            elif strength >= 0.4:
                return "TRENDING_WEAK"
            return "RANGING"

        wr = self.tracker.win_rate()
        pf = self.tracker.profit_factor()

        if strength >= 0.7 and wr >= 0.4 and pf >= 1.5:
            return "TRENDING_STRONG"
        if strength >= 0.4 and wr >= 0.35 and pf >= 1.2:
            return "TRENDING_WEAK"
        if self.dd_pct(balance) > 15:
            return "DRAWDOWN"

        if strength < 0.4 or (wr < 0.3 and pf < 1.0):
            return "CHOPPY"

        return "RANGING"

    def _compute_threshold_adjustment(self, regime: str, balance: float = 0.0) -> float:
        adjust = 0.0
        wr = self.tracker.win_rate()
        consec_losses = self.tracker.consecutive_losses()
        dd = self.dd_pct(balance)

        if regime == "TRENDING_STRONG" and wr >= 0.45:
            adjust = -0.10
        elif regime == "TRENDING_STRONG" and wr >= 0.35:
            adjust = -0.05
        elif regime == "TRENDING_WEAK" and wr >= 0.40:
            adjust = -0.05
        elif regime == "CHOPPY":
            adjust = 0.10
        elif regime == "RANGING" and wr < 0.35:
            adjust = 0.05
        elif regime == "DRAWDOWN":
            adjust = 0.10

        if consec_losses >= 4:
            adjust = max(adjust, 0.15)
        elif consec_losses >= 2:
            adjust = max(adjust, 0.05)

        if dd > 20:
            adjust = max(adjust, 0.10)

        return adjust

    def _compute_lot_mult_adjustment(self, regime: str, balance: float = 0.0) -> float:
        mult = 1.0
        wr = self.tracker.win_rate()
        dd = self.dd_pct(balance)

        if regime == "TRENDING_STRONG" and wr >= 0.45:
            mult = 1.5
        elif regime == "TRENDING_WEAK" and wr >= 0.35:
            mult = 1.0
        elif regime == "CHOPPY":
            mult = 0.5
        elif regime == "RANGING":
            mult = 0.75

        if dd > 15:
            mult = min(mult, 0.5)
        if dd > 25:
            mult = min(mult, 0.25)

        return mult

    def _compute_trades_adjustment(self, regime: str, balance: float = 0.0) -> int:
        if self.tracker.count < 5:
            return self.base_trades_per_event

        wr = self.tracker.win_rate()
        consec = self.tracker.consecutive_losses()
        dd = self.dd_pct(balance)

        if consec >= 3 or dd > 15:
            return 1
        if regime == "CHOPPY" or wr < 0.3:
            return max(1, self.base_trades_per_event // 2)
        if regime == "TRENDING_STRONG" and wr >= 0.45:
            return min(self.base_trades_per_event + 2, 5)

        return self.base_trades_per_event

    def _compute_bias_min_adjustment(self, regime: str) -> float:
        if regime == "CHOPPY":
            return 0.5
        if regime == "TRENDING_STRONG":
            return 0.2
        if regime == "DRAWDOWN":
            return 0.4
        return self.base_bias_strength_min

    def update(self, balance: float, bias_summary: Dict) -> Dict:
        self.update_peak(balance)
        regime = self._detect_regime(bias_summary, balance)

        if regime != self._last_regime:
            self._regime_since = datetime.now()
            self._last_regime = regime

        thresh_adj = self._compute_threshold_adjustment(regime, balance)
        new_threshold = self.base_entry_threshold + thresh_adj
        new_threshold = max(cfg.META_THRESHOLD_MIN, min(cfg.META_THRESHOLD_MAX, new_threshold))
        self._current_threshold = new_threshold

        lot_factor = self._compute_lot_mult_adjustment(regime, balance)
        self._current_lot_mult = max(1, round(self.base_lot_multiplier * lot_factor))

        self._current_trades_per_event = self._compute_trades_adjustment(regime, balance)
        self._current_bias_min = self._compute_bias_min_adjustment(regime)

        return self.summary(balance, regime)

    def record_trade(self, pnl: float, bias_strength: float):
        self.tracker.add_trade(pnl, bias_strength)
        self._total_trades += 1

    def summary(self, balance: float, regime: str = None) -> Dict:
        r = regime or self._last_regime
        dd = self.dd_pct(balance)
        return {
            "regime": r,
            "entry_threshold": round(self._current_threshold, 3),
            "lot_multiplier": self._current_lot_mult,
            "trades_per_event": self._current_trades_per_event,
            "bias_strength_min": round(self._current_bias_min, 2),
            "rolling_win_rate": round(self.tracker.win_rate() * 100, 1),
            "rolling_profit_factor": round(self.tracker.profit_factor(), 2),
            "rolling_avg_pnl": round(self.tracker.avg_pnl(), 2),
            "consecutive_losses": self.tracker.consecutive_losses(),
            "trades_tracked": self.tracker.count,
            "drawdown_pct": round(dd, 1),
            "peak_balance": round(self._peak_balance, 2) if self._peak_balance else None,
        }
