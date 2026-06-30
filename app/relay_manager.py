"""
Relay Manager — bridges remote instances to the bot.
Stores relay data, queues pending orders, manages WebSocket connections.
"""
import asyncio
import json
from typing import Optional, Dict, List, Callable
from datetime import datetime


class RelayManager:
    def __init__(self):
        self._relay_data: Optional[Dict] = None
        self._last_tick: Optional[Dict] = None
        self._last_positions: List[Dict] = []
        self._relay_connected = False
        self._relay_ws = None
        self._pending_orders: List[Dict] = []
        self._order_results: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._relay_connected

    @property
    def account(self) -> Optional[Dict]:
        return self._relay_data

    @property
    def tick(self) -> Optional[Dict]:
        return self._last_tick

    @property
    def positions(self) -> List[Dict]:
        return self._last_positions

    def set_relay(self, ws):
        self._relay_ws = ws
        self._relay_connected = True

    def clear_relay(self):
        self._relay_ws = None
        self._relay_connected = False
        self._relay_data = None
        self._last_tick = None
        self._last_positions = []

    def update_account(self, data: dict):
        self._relay_data = data

    def update_tick(self, data: dict):
        self._last_tick = data

    def update_positions(self, data: list):
        self._last_positions = data

    async def send_order(self, action: str, volume: float, symbol: str = "XAUUSD",
                         close_ticket: Optional[int] = None) -> Optional[Dict]:
        oid = f"o{datetime.now().timestamp():.0f}"
        msg = {
            "type": "order", "id": oid,
            "action": action, "volume": volume,
            "symbol": symbol, "close_ticket": close_ticket,
        }
        if self._relay_ws:
            try:
                await self._relay_ws.send_json(msg)
                for _ in range(30):
                    await asyncio.sleep(0.5)
                    if oid in self._order_results:
                        return self._order_results.pop(oid)
                return {"success": False, "error": "timeout"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "no_relay"}

    async def send_close_all(self) -> Optional[Dict]:
        if self._relay_ws:
            try:
                await self._relay_ws.send_json({"type": "close_all"})
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "no_relay"}

    def store_result(self, result: dict):
        oid = result.get("id")
        if oid:
            self._order_results[oid] = result
