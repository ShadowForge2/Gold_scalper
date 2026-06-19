from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import config as cfg


class EquityScaler:
    """Progressive money machine — scales aggression as equity grows."""

    def __init__(self):
        self.starting_balance: Optional[float] = None
        self.peak_balance: Optional[float] = None
        self.base_lot = cfg.LOT_SIZE
        self.base_trades = cfg.MAX_TRADES_PER_EVENT

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
        lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP
        return max(cfg.MIN_LOT, min(lot, cfg.MAX_LOT))

    def get_trades_per_event(self, balance: float, signal_score: float) -> int:
        t = self._tier(balance)
        tier_mults = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        tm = tier_mults[min(t, len(tier_mults) - 1)]

        if signal_score >= 0.85:
            cm = 2.0
        elif signal_score >= 0.75:
            cm = 1.5
        else:
            cm = 1.0

        trades = int(self.base_trades * tm * cm)
        return max(1, trades)

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
        self.max_daily_loss = cfg.MAX_DAILY_LOSS_USD
        self.max_event_loss = cfg.MAX_EVENT_LOSS_USD
        self.max_trades_per_event = cfg.MAX_TRADES_PER_EVENT
        self.max_trades_per_session = cfg.MAX_TRADES_PER_SESSION
        self.max_consecutive_losses = cfg.MAX_CONSECUTIVE_LOSSES
        self.max_spread = cfg.MAX_SPREAD_PIPS
        self.min_volatility = cfg.MIN_VOLATILITY_PIPS
        self.max_volatility = cfg.MAX_VOLATILITY_PIPS
        self.min_body_ratio = cfg.MIN_CANDLE_BODY_RATIO
        self.cooldown_seconds = cfg.RE_ENTRY_COOLDOWN_SEC

        self.last_exit_time: Optional[datetime] = None
        self.consecutive_losses = 0
        self.session_trades = 0
        self.daily_pnl = 0.0
        self.event_trades = 0
        self._session_reset_day: Optional[int] = None

    def can_enter_trade(self, symbol_info: Dict,
                        current_time: datetime) -> Tuple[bool, str]:
        now = current_time

        if self._session_reset_day != now.day:
            self.session_trades = 0
            self.consecutive_losses = 0
            self._session_reset_day = now.day

        if self.session_trades >= self.max_trades_per_session:
            return False, f"session_trade_limit ({self.session_trades}/{self.max_trades_per_session})"

        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"consecutive_losses ({self.consecutive_losses})"

        if self.last_exit_time and self.cooldown_seconds > 0:
            elapsed = (now - self.last_exit_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - elapsed)
                return False, f"cooldown ({remaining}s remaining)"

        spread_pips = symbol_info.get("spread", 0)
        if spread_pips > self.max_spread:
            return False, f"spread_too_high ({spread_pips:.1f} > {self.max_spread})"

        session_ok, session_name = self._check_session(now)
        if not session_ok:
            return False, f"session_not_allowed ({session_name})"

        return True, "ok"

    def can_add_to_event(self) -> Tuple[bool, str]:
        if self.event_trades >= self.max_trades_per_event:
            return False, f"event_trade_limit ({self.event_trades}/{self.max_trades_per_event})"
        return True, "ok"

    def check_daily_loss(self, daily_pnl: float) -> Tuple[bool, str]:
        if daily_pnl <= -self.max_daily_loss:
            return False, f"daily_loss_limit (${daily_pnl:.2f} <= -${self.max_daily_loss:.2f})"
        return True, "ok"

    def check_event_loss(self, event_pnl: float) -> Tuple[bool, str]:
        if event_pnl <= -self.max_event_loss:
            return False, f"event_loss_limit (${event_pnl:.2f} <= -${self.max_event_loss:.2f})"
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

    def record_entry(self):
        self.event_trades += 1
        self.session_trades += 1

    def record_exit(self, profit: float):
        self.last_exit_time = datetime.now()
        self.event_trades = 0
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def record_daily_pnl_reset(self):
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.session_trades = 0

    def _check_session(self, dt: datetime) -> Tuple[bool, str]:
        hour = dt.hour
        minute = dt.minute
        time_decimal = hour + minute / 60.0

        sessions = {
            "ASIA": (0, 8),
            "LONDON": (8, 17),
            "NEW_YORK": (13, 22),
        }

        allowed = getattr(self, 'allowed_sessions', None)
        if allowed is None:
            allowed = [s.strip().upper() for s in cfg.ALLOWED_SESSIONS.split(",")]

        for name in allowed:
            if name in sessions:
                start, end = sessions[name]
                if start <= time_decimal <= end:
                    return True, name

        return False, "outside_allowed_sessions"
