import requests
import pandas as pd
import time
from typing import Optional, List, Dict, Tuple
from datetime import datetime


TIMEFRAME_MAP = {
    1: "MINUTE",
    5: "MINUTE_5",
    15: "MINUTE_15",
    30: "MINUTE_30",
    16385: "HOUR",
    16408: "HOUR_4",
    16415: "DAY",
    32769: "WEEK",
}

EPIC_MAP = {
    "XAUUSD": "GOLD",
    "GOLD": "GOLD",
    "XAGUSD": "SILVER",
    "SILVER": "SILVER",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
}


class CapitalClient:
    def __init__(self):
        self.connected = False
        self.api_key = None
        self.identifier = None
        self.password = None
        self.demo = True
        self.base_url = None
        self.cst = None
        self.security_token = None
        self._session = requests.Session()
        self._last_activity = 0.0
        self._symbol_info_cache: Dict[str, Dict] = {}
        self._prev_balance = None
        self._daily_pnl_date = None
        self._realized_daily_pnl = 0.0
        self._last_position_pnl: Dict[str, float] = {}

    def initialize(self, api_key: Optional[str] = None,
                   identifier: Optional[str] = None,
                   password: Optional[str] = None,
                   demo: bool = True) -> bool:
        if api_key:
            self.api_key = api_key
        if identifier:
            self.identifier = identifier
        if password:
            self.password = password
        self.demo = demo
        self.base_url = "https://demo-api-capital.backend-capital.com" if demo else "https://api-capital.backend-capital.com"
        self._symbol_info_cache.clear()
        return self._login()

    def _login(self) -> bool:
        headers = {'X-CAP-API-KEY': self.api_key, 'Content-Type': 'application/json'}
        body = {'identifier': self.identifier, 'password': self.password, 'encryptedPassword': False}
        try:
            r = self._session.post(f"{self.base_url}/api/v1/session", headers=headers, json=body)
            if r.ok:
                self.cst = r.headers.get("CST")
                self.security_token = r.headers.get("X-SECURITY-TOKEN")
                self.connected = True
                self._last_activity = time.time()
                data = r.json()
                self._prev_balance = data.get("accountInfo", {}).get("balance", 0)
                return True
        except Exception:
            pass
        self.connected = False
        return False

    def _auth_headers(self) -> Dict:
        return {"CST": self.cst or "", "X-SECURITY-TOKEN": self.security_token or "", "Content-Type": "application/json"}

    def _ensure_session(self) -> bool:
        if not self.connected:
            return False
        if time.time() - self._last_activity > 280:
            try:
                r = self._session.get(f"{self.base_url}/api/v1/ping", headers=self._auth_headers())
                if not r.ok:
                    self._login()
            except Exception:
                self._login()
        self._last_activity = time.time()
        return self.connected

    def shutdown(self):
        if self.connected:
            try:
                self._session.delete(f"{self.base_url}/api/v1/session", headers=self._auth_headers())
            except Exception:
                pass
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def reconnect(self, server: str, account: str, password: str) -> bool:
        self.shutdown()
        self.demo = "demo" in server.lower() or "DEMO" in server
        return self.initialize(self.api_key, self.identifier, self.password)

    def last_error(self) -> Tuple[int, str]:
        return 0, self._last_error if hasattr(self, '_last_error') else "No error"

    def select_symbol(self, symbol: str) -> bool:
        info = self.get_symbol_info(symbol)
        if info is None:
            return False
        return info.get("market_status") == "TRADEABLE"

    def get_account_info(self) -> Optional[Dict]:
        if not self._ensure_session():
            return None
        try:
            r = self._session.get(f"{self.base_url}/api/v1/accounts", headers=self._auth_headers())
            if r.ok:
                accounts = r.json().get("accounts", [])
                for acct in accounts:
                    if acct.get("preferred"):
                        bal = acct.get("balance", {})
                        return {
                            "account_number": acct.get("accountId", ""),
                            "balance": float(bal.get("balance", 0)),
                            "equity": float(bal.get("balance", 0)),
                            "margin": 0,
                            "free_margin": float(bal.get("available", 0)),
                            "margin_level": 0,
                            "leverage": 100,
                            "currency": acct.get("currency", "USD"),
                            "profit": float(bal.get("profitLoss", 0)),
                            "server": "Capital.com Demo" if self.demo else "Capital.com Live",
                            "name": "Capital.com",
                        }
        except Exception:
            pass
        return None

    def _resolve_epic(self, symbol: str) -> str:
        return EPIC_MAP.get(symbol.upper(), symbol.upper())

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        epic = self._resolve_epic(symbol)
        if epic in self._symbol_info_cache:
            return self._symbol_info_cache[epic]
        if not self._ensure_session():
            return None
        try:
            r = self._session.get(f"{self.base_url}/api/v1/markets/{epic}", headers=self._auth_headers())
            if r.ok:
                data = r.json()
                snap = data.get("snapshot", {})
                inst = data.get("instrument", {})
                dr = data.get("dealingRules", {})
                dpf = int(snap.get("decimalPlacesFactor", 2))
                info = {
                    "name": inst.get("name", symbol),
                    "epic": epic,
                    "spread": max(0, float(snap.get("offer", 0) or 0) - float(snap.get("bid", 0) or 0)),
                    "digits": dpf,
                    "point": 10 ** -dpf,
                    "bid": float(snap.get("bid", 0) or 0),
                    "ask": float(snap.get("offer", 0) or 0),
                    "high": float(snap.get("high", 0) or 0),
                    "low": float(snap.get("low", 0) or 0),
                    "volume_min": float(dr.get("minDealSize", {}).get("value", 0.01)),
                    "volume_max": float(dr.get("maxDealSize", {}).get("value", 100)),
                    "volume_step": float(dr.get("minSizeIncrement", {}).get("value", 0.01)),
                    "trade_mode": "ENABLED" if snap.get("marketStatus") == "TRADEABLE" else "DISABLED",
                    "market_status": snap.get("marketStatus", ""),
                    "filling_mode": 0,
                    "trade_stops_level": 0,
                }
                self._symbol_info_cache[epic] = info
                return info
        except Exception:
            pass
        return None

    def get_rates(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        epic = self._resolve_epic(symbol)
        resolution = TIMEFRAME_MAP.get(timeframe, "MINUTE")
        if not self._ensure_session():
            return None
        try:
            r = self._session.get(f"{self.base_url}/api/v1/prices/{epic}",
                                  params={"resolution": resolution, "max": count},
                                  headers=self._auth_headers())
            if r.ok:
                prices = r.json().get("prices", [])
                if not prices:
                    return None
                return pd.DataFrame(self._parse_prices(prices))
        except Exception:
            pass
        return None

    def get_rates_range(self, symbol: str, timeframe: int,
                        from_dt: datetime, to_dt: datetime) -> Optional[pd.DataFrame]:
        epic = self._resolve_epic(symbol)
        resolution = TIMEFRAME_MAP.get(timeframe, "MINUTE")
        if not self._ensure_session():
            return None

        # Try range-based query first
        try:
            r = self._session.get(f"{self.base_url}/api/v1/prices/{epic}",
                                  params={"resolution": resolution,
                                          "from": from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "to": to_dt.strftime("%Y-%m-%dT%H:%M:%S")},
                                  headers=self._auth_headers())
            if r.ok:
                prices = r.json().get("prices", [])
                if prices:
                    rows = self._parse_prices(prices)
                    return pd.DataFrame(rows)
        except Exception:
            pass

        # Fallback: paginate count-based API (max=1000 per call)
        # First call without `to` returns latest candles
        all_rows = []
        target_from = from_dt.strftime("%Y-%m-%dT%H:%M:%S")
        cursor_to = None
        for _ in range(20):
            try:
                params = {"resolution": resolution, "max": 1000}
                if cursor_to:
                    params["to"] = cursor_to
                r = self._session.get(f"{self.base_url}/api/v1/prices/{epic}",
                                      params=params,
                                      headers=self._auth_headers())
                if not r.ok:
                    break
                prices = r.json().get("prices", [])
                if not prices:
                    break
                rows = self._parse_prices(prices)
                all_rows.extend(rows)
                cursor_to = prices[0].get("snapshotTime", "")
                if prices[-1].get("snapshotTime", "") <= target_from:
                    break
            except Exception:
                break

        if not all_rows:
            return None
        df = pd.DataFrame(all_rows)
        df.drop_duplicates(subset="time", keep="last", inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        mask = (df["time"] >= from_dt) & (df["time"] <= to_dt)
        return df[mask].copy() if mask.any() else df

    def _parse_prices(self, prices: list) -> list:
        rows = []
        for p in prices:
            t = p.get("snapshotTime", "").replace("Z", "+00:00")
            if "." not in t and "+" not in t:
                t += "+00:00"
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            rows.append({
                "time": dt,
                "open": float(p.get("openPrice", {}).get("bid", 0)),
                "high": float(p.get("highPrice", {}).get("bid", 0)),
                "low": float(p.get("lowPrice", {}).get("bid", 0)),
                "close": float(p.get("closePrice", {}).get("bid", 0)),
                "tick_volume": int(p.get("lastTradedVolume", 0)),
                "spread": 0,
                "real_volume": int(p.get("lastTradedVolume", 0)),
            })
        return rows

    def get_positions(self, magic: Optional[int] = None) -> List[Dict]:
        if not self._ensure_session():
            return []
        try:
            r = self._session.get(f"{self.base_url}/api/v1/positions", headers=self._auth_headers())
            if r.ok:
                result = []
                for pos_data in r.json().get("positions", []):
                    p = pos_data.get("position", {})
                    mkt = pos_data.get("market", {})
                    epic = mkt.get("epic", self._resolve_epic("XAUUSD"))
                    deal_id = p.get("dealId", "")
                    comment = p.get("reference", "")
                    if magic is not None:
                        if not comment.startswith(str(magic)):
                            continue
                    result.append({
                        "ticket": deal_id,
                        "symbol": mkt.get("instrumentName", epic),
                        "type": "BUY" if p.get("direction") == "BUY" else "SELL",
                        "volume": float(p.get("size", 0)),
                        "price_open": float(p.get("level", 0)),
                        "price_current": float(p.get("level", 0)),
                        "sl": float(p.get("stopLevel", 0)) if p.get("stopLevel") else 0.0,
                        "tp": float(p.get("profitLevel", 0)) if p.get("profitLevel") else 0.0,
                        "profit": float(p.get("upl", 0)),
                        "swap": 0,
                        "magic": magic or 0,
                        "comment": comment,
                        "time": datetime.fromisoformat(p.get("createdDateUTC", "").replace("Z", "+00:00")) if p.get("createdDateUTC") else datetime.now(),
                    })
                if magic is not None:
                    self._update_position_pnl_cache(result)
                return result
        except Exception:
            pass
        return []

    def _update_position_pnl_cache(self, positions: List[Dict]):
        current = {str(p["ticket"]): float(p.get("profit", 0.0)) for p in positions}
        for ticket, last_pnl in list(self._last_position_pnl.items()):
            if ticket not in current:
                self._realized_daily_pnl += last_pnl
                del self._last_position_pnl[ticket]
        self._last_position_pnl.update(current)

    def get_history_deals(self, from_dt: datetime, to_dt: datetime,
                          magic: Optional[int] = None) -> List[Dict]:
        if not self._ensure_session():
            return []
        try:
            r = self._session.get(f"{self.base_url}/api/v1/history/activity",
                                  params={"from": from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "to": to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "detailed": "true"},
                                  headers=self._auth_headers())
            if r.ok:
                result = []
                for act in r.json().get("activities", []):
                    if act.get("type") == "POSITION":
                        result.append({
                            "ticket": act.get("dealId", ""),
                            "symbol": act.get("epic", ""),
                            "type": act.get("status", ""),
                            "volume": 0,
                            "price": 0,
                            "profit": 0,
                            "magic": magic or 0,
                            "comment": "",
                            "time": datetime.fromisoformat(act.get("dateUTC", "").replace("Z", "+00:00")) if act.get("dateUTC") else datetime.now(),
                        })
                return result
        except Exception:
            pass
        return []

    def get_filling_type(self, symbol: str) -> int:
        return 0

    def get_total_daily_pnl(self, magic: int) -> float:
        now = datetime.now()
        today = now.date()

        if self._daily_pnl_date != today:
            info = self.get_account_info()
            if info:
                self._prev_balance = info.get("balance", 0)
            self._daily_pnl_date = today
            self._realized_daily_pnl = 0.0
            self._last_position_pnl.clear()

        info = self.get_account_info()
        if info is None or self._prev_balance is None:
            return 0.0

        positions = self.get_positions(magic)
        open_pnl = sum(p["profit"] for p in positions)
        return self._realized_daily_pnl + open_pnl

    def order_send(self, request: dict) -> Dict:
        epic = request.get("epic", self._resolve_epic(request.get("symbol", "GOLD")))
        direction = "BUY" if request.get("type") == 0 else "SELL"
        volume = request.get("volume", 0.01)

        body = {"epic": epic, "direction": direction, "size": volume,
                "orderType": "MARKET", "guaranteedStop": False, "forceOpen": True}

        sl = request.get("sl")
        tp = request.get("tp")
        if sl:
            body["stopLevel"] = float(sl)
        if tp:
            body["profitLevel"] = float(tp)

        result = self._open_position_raw(epic, direction, volume, sl, tp)
        if result:
            return {"retcode": 10009, "order": result.get("dealReference", ""), "comment": "Done",
                    "volume": volume, "price": 0, "bid": 0, "ask": 0, "success": True}
        return {"retcode": 10004, "order": 0, "comment": "Open failed", "volume": 0, "price": 0,
                "bid": 0, "ask": 0, "success": False}

    def _open_position_raw(self, epic: str, direction: str, volume: float,
                           stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None,
                           force_open: bool = True,
                           reference: str = "") -> Optional[Dict]:
        if not self._ensure_session():
            return None
        body = {"epic": epic, "direction": direction.upper(), "size": volume,
                "orderType": "MARKET", "guaranteedStop": False, "forceOpen": force_open}
        if stop_loss is not None:
            body["stopLevel"] = stop_loss
        if take_profit is not None:
            body["profitLevel"] = take_profit
        if reference:
            body["reference"] = reference
        try:
            r = self._session.post(f"{self.base_url}/api/v1/positions",
                                   headers=self._auth_headers(), json=body)
            if r.ok:
                time.sleep(0.5)
                return r.json()
        except Exception:
            pass
        return None

    def open_position(self, symbol: str, direction: str, volume: float,
                      price: Optional[float] = None,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None,
                      comment: str = "",
                      magic: int = 0,
                      slippage: int = 30) -> Optional[str]:
        epic = self._resolve_epic(symbol)
        reference = str(magic) + ":" + comment if comment else str(magic)
        result = self._open_position_raw(epic, direction, volume, stop_loss, take_profit, reference=reference)
        if result:
            return result.get("dealReference")
        return None

    def close_position(self, ticket) -> bool:
        if isinstance(ticket, str):
            deal_id = ticket
        else:
            positions = self.get_positions()
            pos = next((p for p in positions if p["ticket"] == ticket), None)
            if pos is None:
                return False
            deal_id = pos["ticket"]
        if not self._ensure_session():
            return False
        try:
            r = self._session.delete(f"{self.base_url}/api/v1/positions/{deal_id}",
                                     headers=self._auth_headers())
            return r.ok
        except Exception:
            return False

    def get_tick(self, symbol: str) -> Optional[Dict]:
        info = self.get_symbol_info(symbol)
        if info is None:
            return None
        return {
            "bid": info.get("bid", 0),
            "ask": info.get("ask", 0),
            "time": datetime.now(),
        }

    def modify_position(self, deal_id: str, stop_loss: Optional[float] = None,
                        take_profit: Optional[float] = None) -> bool:
        if not self._ensure_session():
            return False
        body = {}
        if stop_loss is not None:
            body["stopLevel"] = stop_loss
        if take_profit is not None:
            body["profitLevel"] = take_profit
        if not body:
            return False
        try:
            r = self._session.put(f"{self.base_url}/api/v1/positions/{deal_id}",
                                  headers=self._auth_headers(), json=body)
            return r.ok
        except Exception:
            return False
