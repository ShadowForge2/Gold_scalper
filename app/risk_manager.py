from typing import Dict, Tuple, Optional
from datetime import datetime
import config as cfg


class EquityScaler:
    """Progressive money machine — scales aggression as equity grows."""

    def __init__(self):
        self.starting_balance: Optional[float] = None
        self.peak_balance: Optional[float] = None
        self.base_lot = cfg.LOT_SIZE
        self.base_trades = cfg.ML_POSITIONS_STRONG

    def initialize(self, balance: float):
        self.starting_balance = balance
        self.peak_balance = balance

    def growth_pct(self, balance: float) -> float:
        if not self.starting_balance or self.starting_balance <= 0:
            return 0.0
        return (balance - self.starting_balance) / self.starting_balance * 100

    def _tier(self, balance: float) -> int:
        if balance >= 100:
            return 3
        if balance >= 50:
            return 2
        return 1

    def update_peak(self, balance: float):
        if self.peak_balance is None or balance > self.peak_balance:
            self.peak_balance = balance

    def get_lot(self, balance: float) -> float:
        if not self.starting_balance or self.starting_balance <= 0:
            return self.base_lot
        self.update_peak(balance)
        reference = 20.0  # fixed reference; all accounts scale identically
        lot = self.base_lot * (balance / reference)
        max_by_equity = balance / 5000.0
        lot = min(lot, max_by_equity)
        if self.in_drawdown(balance):
            lot *= 0.5
        lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP
        return max(cfg.MIN_LOT, min(lot, cfg.MAX_LOT))

    def get_trades_per_event(self, balance: float, signal_score: float, ml_confidence: float = 0.0) -> int:
        t = self._tier(balance)
        tier_mults = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        tm = tier_mults[min(t - 1, len(tier_mults) - 1)]

        if signal_score >= 0.50:
            cm = 2.0
        elif signal_score >= 0.30:
            cm = 1.5
        else:
            cm = 1.0

        if ml_confidence >= 0.88:
            ml_cm = 3.0
        elif ml_confidence >= 0.75:
            ml_cm = 2.0
        else:
            ml_cm = 1.0

        trades = int(self.base_trades * tm * cm * ml_cm)
        trades = max(1, trades)
        if self.in_drawdown(balance):
            trades = 1
        return trades

    def in_drawdown(self, balance: float) -> bool:
        if not self.peak_balance or self.peak_balance <= 0:
            return False
        dd = (self.peak_balance - balance) / self.peak_balance * 100
        return dd > 15

    def summary(self, balance: float) -> Dict:
        g = self.growth_pct(balance)
        t = self._tier(balance)
        return {
            "starting_balance": round(self.starting_balance, 2) if self.starting_balance else None,
            "peak_balance": round(self.peak_balance, 2) if self.peak_balance else None,
            "current_balance": round(balance, 2),
            "growth_pct": round(g, 2),
            "tier": t,
            "lot_size": round(self.get_lot(balance), 4),
            "trades_per_event": self.get_trades_per_event(balance, 0.65),
            "in_drawdown": self.in_drawdown(balance),
        }


class RiskManager:
    def __init__(self):
        self.max_spread = cfg.MAX_SPREAD_PIPS
        self.allowed_sessions = [s.strip().upper() for s in cfg.ALLOWED_SESSIONS.split(",")]

    def can_enter_trade(self, symbol_info: Dict,
                        current_time: datetime) -> Tuple[bool, str]:
        now = current_time
        point = symbol_info.get("point", 0.0001)
        spread_pips = float(symbol_info.get("spread", 0)) / point if point > 0 else 0
        if spread_pips > self.max_spread:
            return False, f"spread_too_high ({spread_pips:.1f} > {self.max_spread})"

        session_ok, session_name = self._check_session(now)
        if not session_ok:
            return False, f"session_not_allowed ({session_name})"

        return True, "ok"

    def check_event_loss(self, event_pnl: float) -> Tuple[bool, str]:
        return True, "ok"

    def calculate_lot_size(self, balance: float,
                           stop_distance_pips: float,
                           point: float = 0.0001) -> float:
        if stop_distance_pips <= 0:
            return cfg.LOT_SIZE
        risk_per_trade = balance * 0.01
        lot = risk_per_trade / (stop_distance_pips * 10)
        lot = max(cfg.MIN_LOT, min(lot, cfg.MAX_LOT))
        step = cfg.LOT_STEP
        lot = round(lot / step) * step
        return lot

    def _check_session(self, dt: datetime) -> Tuple[bool, str]:
        hour = dt.hour
        minute = dt.minute
        time_decimal = hour + minute / 60.0

        sessions = {
            "ASIA": (0, 8),
            "LONDON": (7, 17),
            "NEW_YORK": (12, 22),
        }

        for name in self.allowed_sessions:
            if name in sessions:
                start, end = sessions[name]
                if start <= time_decimal < end:
                    return True, name

        return False, "outside_allowed_sessions"
