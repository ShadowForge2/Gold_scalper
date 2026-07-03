import time
from typing import List, Dict, Optional, Any
from datetime import datetime
import config as cfg


class PositionManager:
    def __init__(self, client: Any):
        self.client = client
        self.magic = cfg.MAGIC_NUMBER
        self.open_positions: List[Dict] = []
        self.event_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self.in_event: bool = False
        self.open_count: int = 0
        self._closed_tickets: Dict[str, float] = {}
        self.closed_history: List[Dict] = []
        self._event_start_ts: Optional[float] = None

    def note_closed(self, pos_data: Dict) -> None:
        ticket = pos_data["ticket"]
        self._closed_tickets[str(ticket)] = time.time()
        self.closed_history.append({
            "ticket": ticket,
            "type": pos_data.get("type", "UNKNOWN"),
            "volume": pos_data.get("volume", 0),
            "entry_price": pos_data.get("price_open", 0),
            "exit_price": pos_data.get("price_current", 0),
            "profit": pos_data.get("profit", 0),
            "swap": pos_data.get("swap", 0),
            "symbol": pos_data.get("symbol", ""),
            "closed_at": datetime.utcnow().isoformat(),
        })
        if len(self.closed_history) > 500:
            self.closed_history[:100] = []

    def refresh(self) -> Dict:
        raw_positions = self.client.get_positions(symbol=cfg.SYMBOL) or []
        now = time.time()
        cutoff = now - 30.0
        self._closed_tickets = {
            tid: ts for tid, ts in self._closed_tickets.items()
            if ts > cutoff
        }
        self.open_positions = [
            p for p in raw_positions
            if str(p.get("ticket", "")) not in self._closed_tickets
        ]
        self.event_pnl = sum(p.get("profit", 0) for p in self.open_positions)
        self.daily_pnl = self.client.get_total_daily_pnl(self.magic) or 0.0
        self.in_event = len(self.open_positions) > 0
        self.open_count = len(self.open_positions)
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
