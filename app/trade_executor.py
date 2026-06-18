from typing import Optional, Dict, List, Any
from app.logger import BotLogger
import config as cfg


class TradeExecutor:
    def __init__(self, client: Any, logger: BotLogger):
        self.mt5 = client
        self.logger = logger

    def open_market(self, symbol: str, direction: str,
                    volume: float, magic: int = cfg.MAGIC_NUMBER,
                    comment: str = cfg.COMMENT,
                    slippage: int = cfg.MAX_SLIPPAGE_PIPS) -> Optional[Any]:

        ticket = self.mt5.open_position(
            symbol=symbol, direction=direction, volume=volume,
            magic=magic, comment=comment, slippage=slippage
        )
        if ticket is not None:
            self.logger.trade(
                f"Opened {direction} {volume:.2f} {symbol} "
                f"@ ticket {ticket}"
            )
            return ticket
        else:
            detail = ""
            if hasattr(self.mt5, "last_order_error"):
                detail = self.mt5.last_order_error()
            suffix = f": {detail}" if detail else ""
            self.logger.error(f"Order failed for {symbol} {direction}{suffix}")
            return None

    def close_position(self, ticket: int) -> bool:
        success = self.mt5.close_position(ticket)
        if not success:
            self.logger.warning(f"Position {ticket} not found or close failed")
            return False
        self.logger.trade(f"Closed position {ticket}")
        return True

    def close_all_bot_positions(self) -> List:
        positions = self.mt5.get_positions(magic=cfg.MAGIC_NUMBER)
        closed_tickets = []
        for pos in positions:
            if self.close_position(pos["ticket"]):
                closed_tickets.append(pos["ticket"])
        if closed_tickets:
            self.logger.info(f"Closed {len(closed_tickets)} bot position(s)")
        return closed_tickets

    def close_all_positions(self, symbol: Optional[str] = None) -> int:
        positions = self.mt5.get_positions()
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]
        closed = 0
        for pos in positions:
            if self.close_position(pos["ticket"]):
                closed += 1
        return closed
