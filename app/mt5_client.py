try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    _MT5_AVAILABLE = False
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime


class MT5Client:
    def __init__(self):
        self.connected = False
        self._account_info: Optional[mt5.AccountInfo] = None

    def initialize(self, login: Optional[int] = None,
                   password: Optional[str] = None,
                   server: Optional[str] = None) -> bool:
        if login and password:
            self.connected = mt5.initialize(
                login=login, password=password, server=server
            )
        else:
            self.connected = mt5.initialize()
        if self.connected:
            self._account_info = mt5.account_info()
        return self.connected

    def shutdown(self):
        mt5.shutdown()
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected and mt5.terminal_info() is not None

    def last_error(self) -> Tuple[int, str]:
        err = mt5.last_error()
        return (err[0], err[1]) if err else (0, "No error")

    def reconnect(self, server: str, account: str, password: str) -> bool:
        mt5.shutdown()
        import time
        time.sleep(2)
        login = int(account) if account else None
        self.connected = mt5.initialize(login=login, password=password, server=server)
        if self.connected:
            self._account_info = mt5.account_info()
        return self.connected

    def get_account_info(self) -> Optional[Dict]:
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "account_number": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level,
            "leverage": info.leverage,
            "currency": info.currency,
            "profit": info.profit,
            "server": info.server,
            "name": info.name,
        }

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "name": info.name,
            "spread": info.spread,
            "digits": info.digits,
            "point": info.point,
            "bid": info.bid,
            "ask": info.ask,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_mode": info.trade_mode,
            "filling_mode": info.filling_mode,
            "trade_stops_level": getattr(info, 'trade_stops_level', 0),
        }

    def select_symbol(self, symbol: str) -> bool:
        return mt5.symbol_select(symbol, True)

    def get_rates(self, symbol: str, timeframe: int,
                  count: int) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_rates_range(self, symbol: str, timeframe: int,
                        from_dt: datetime, to_dt: datetime
                        ) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_range(symbol, timeframe,
                                     from_dt, to_dt)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_positions(self, magic: Optional[int] = None
                      ) -> List[Dict]:
        positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for pos in positions:
            if magic is None or pos.magic == magic:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "magic": pos.magic,
                    "comment": pos.comment,
                    "time": datetime.fromtimestamp(pos.time),
                })
        return result

    def get_history_deals(self, from_dt: datetime, to_dt: datetime,
                          magic: Optional[int] = None) -> List[Dict]:
        deals = mt5.history_deals_get(from_dt, to_dt)
        if deals is None:
            return []
        result = []
        for deal in deals:
            if magic is None or deal.magic == magic:
                result.append({
                    "ticket": deal.ticket,
                    "symbol": deal.symbol,
                    "type": deal.type,
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "magic": deal.magic,
                    "comment": deal.comment,
                    "time": datetime.fromtimestamp(deal.time),
                })
        return result

    def get_filling_type(self, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_FOK
        filling = info.filling_mode
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def order_send(self, request: dict) -> Dict:
        result = mt5.order_send(request)
        return {
            "retcode": result.retcode,
            "order": result.order if hasattr(result, "order") else 0,
            "comment": result.comment if hasattr(result, "comment") else "",
            "volume": result.volume if hasattr(result, "volume") else 0.0,
            "price": result.price if hasattr(result, "price") else 0.0,
            "bid": result.bid if hasattr(result, "bid") else 0.0,
            "ask": result.ask if hasattr(result, "ask") else 0.0,
            "success": result.retcode == mt5.TRADE_RETCODE_DONE,
        }

    def open_position(self, symbol: str, direction: str, volume: float,
                      price: Optional[float] = None,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None,
                      comment: str = "",
                      magic: int = 0,
                      slippage: int = 30) -> Optional[int]:
        if not self.select_symbol(symbol):
            return None
        info = self.get_symbol_info(symbol)
        if info is None:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        if price is None:
            price = tick.ask if direction == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        filling = self.get_filling_type(symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": slippage,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        if stop_loss is not None:
            request["sl"] = stop_loss
        if take_profit is not None:
            request["tp"] = take_profit
        result = self.order_send(request)
        if result["success"]:
            return result["order"]
        return None

    def close_position(self, ticket: int) -> bool:
        positions = self.get_positions()
        pos = next((p for p in positions if p["ticket"] == ticket), None)
        if pos is None:
            return False
        symbol = pos["symbol"]
        info = self.get_symbol_info(symbol)
        if info is None:
            return False
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        is_buy = pos["type"] == "BUY"
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy else tick.ask
        filling = self.get_filling_type(symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos["volume"],
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 30,
            "magic": 0,
            "comment": "Close by bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = self.order_send(request)
        return result["success"]

    def get_total_daily_pnl(self, magic: int) -> float:
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day)
        positions = self.get_positions(magic)
        open_pnl = sum(p["profit"] for p in positions)
        deals = self.get_history_deals(today_start, now, magic)
        closed_pnl = sum(d["profit"] for d in deals)
        return closed_pnl + open_pnl
