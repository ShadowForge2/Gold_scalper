from typing import Optional, Dict, List, Any
from app.logger import BotLogger
import config as cfg


class TradeExecutor:
    def __init__(self, client: Any, logger: BotLogger):
        self.client = client
        self.logger = logger

    async def open_market(self, symbol: str, direction: str,
                          volume: float, magic: int = cfg.MAGIC_NUMBER,
                          comment: str = cfg.COMMENT,
                          slippage: int = cfg.MAX_SLIPPAGE_PIPS) -> Optional[Any]:

        try:
            ticket = await self.client.open_position(
                symbol=symbol, direction=direction, volume=volume,
                magic=magic, comment=comment, slippage=slippage
            )
        except Exception as e:
            self.logger.error(f"Order exception for {symbol} {direction}: {e}")
            return None
        if ticket is not None:
            self.logger.trade(
                f"Opened {direction} {volume:.2f} {symbol} "
                f"@ ticket {ticket}"
            )
            return ticket
        else:
            err_attr = getattr(self.client, "last_order_error", None)
            err_detail = ""
            if err_attr is not None:
                try:
                    err_detail = err_attr() if callable(err_attr) else str(err_attr)
                except Exception:
                    err_detail = str(getattr(self.client, "last_order_error", ""))
            suffix = f": {err_detail}" if err_detail else ""
            self.logger.error(f"Order failed for {symbol} {direction}{suffix}")
            return None

    def close_position(self, ticket: int) -> bool:
        try:
            success = self.client.close_position(ticket)
            if not success:
                self.logger.warning(f"Position {ticket} not found or close failed")
                return False
            self.logger.trade(f"Closed position {ticket}")
            return True
        except Exception as e:
            self.logger.error(f"Exception closing position {ticket}: {e}")
            return False

    def close_all_bot_positions(self) -> List[Dict]:
        positions = self.client.get_positions(symbol=cfg.SYMBOL) or []
        closed = []
        for pos in positions:
            ticket = pos.get("ticket")
            if ticket and self.close_position(ticket):
                closed.append(pos)
        if closed:
            self.logger.info(f"Closed {len(closed)} bot position(s)")
        return closed

    def close_all_positions(self, symbol: Optional[str] = None) -> int:
        positions = self.client.get_positions() or []
        if symbol:
            positions = [p for p in positions if p.get("symbol") == symbol]
        closed = 0
        for pos in positions:
            ticket = pos.get("ticket")
            if ticket and self.close_position(ticket):
                closed += 1
        return closed
