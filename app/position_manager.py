from typing import List, Dict, Optional, Any
from datetime import datetime
import config as cfg


class PositionManager:
    def __init__(self, client: Any):
        self.mt5 = client
        self.magic = cfg.MAGIC_NUMBER
        self.open_positions: List[Dict] = []
        self.event_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self.in_event: bool = False

    def refresh(self) -> Dict:
        self.open_positions = self.mt5.get_positions(magic=self.magic)
        self.event_pnl = sum(p["profit"] for p in self.open_positions)
        self.daily_pnl = self.mt5.get_total_daily_pnl(self.magic)
        self.in_event = len(self.open_positions) > 0
        return self.summary()

    def summary(self) -> Dict:
        return {
            "open_count": len(self.open_positions),
            "positions": self.open_positions,
            "event_pnl": round(self.event_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "in_event": self.in_event,
        }

    def has_position(self, ticket: int) -> bool:
        return any(p["ticket"] == ticket for p in self.open_positions)

    def get_total_volume(self) -> float:
        return sum(p["volume"] for p in self.open_positions)

    def get_direction_counts(self) -> Dict[str, int]:
        buys = sum(1 for p in self.open_positions if p["type"] == "BUY")
        sells = sum(1 for p in self.open_positions if p["type"] == "SELL")
        return {"BUY": buys, "SELL": sells}

    def get_average_entry(self, direction: str) -> Optional[float]:
        dir_positions = [
            p for p in self.open_positions if p["type"] == direction
        ]
        if not dir_positions:
            return None
        total_vol = sum(p["volume"] for p in dir_positions)
        if total_vol == 0:
            return None
        weighted = sum(p["price_open"] * p["volume"] for p in dir_positions)
        return weighted / total_vol
