import requests
import pandas as pd
import time
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from collections import deque
import config as cfg


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
        self._last_order_error = ""
        self._request_times: deque = deque(maxlen=20)
        self._max_requests_per_sec = 8
        self._timeout = 15

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
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self._throttle()
                r = self._session.post(
                    f"{self.base_url}/api/v1/session",
                    headers=headers,
                    json=body,
                    timeout=self._timeout,
                )
                if r.ok:
                    self.cst = r.headers.get("CST")
                    self.security_token = r.headers.get("X-SECURITY-TOKEN")
                    self.connected = True
                    self._last_activity = time.time()
                    data = r.json()
                    self._prev_balance = data.get("accountInfo", {}).get("balance", 0)
                    self._last_order_error = ""
                    return True
                self._last_order_error = f"HTTP {r.status_code}: {r.text[:500]}"
                if r.status_code not in (429, 502, 503, 504):
                    break
            except Exception as exc:
                self._last_order_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                backoff = min(2 ** attempt, 30)
                time.sleep(backoff)
        self.connected = False
        return False

    def _auth_headers(self) -> Dict:
        return {"CST": self.cst or "", "X-SECURITY-TOKEN": self.security_token or "", "Content-Type": "application/json"}

    def _ensure_session(self) -> bool:
        if not self.connected:
            return False
        if time.time() - self._last_activity > 280:
            try:
                r = self._session.get(
                    f"{self.base_url}/api/v1/ping",
                    headers=self._auth_headers(),
                    timeout=self._timeout,
                )
                if r.status_code in (401, 403):
                    self._login()
            except Exception:
                pass
        self._last_activity = time.time()
        return self.connected

    def _throttle(self):
        now = time.time()
        while len(self._request_times) > 0 and now - self._request_times[0] > 1.0:
            self._request_times.popleft()
        if len(self._request_times) >= self._max_requests_per_sec:
            sleep_for = 1.0 - (now - self._request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._request_times.append(time.time())

    def _request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        self._throttle()
        kwargs.setdefault("timeout", self._timeout)
        for attempt in range(2):
            try:
                r = self._session.request(method, url, **kwargs)
                # 429: rate limited — sleep Retry-After then retry
                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", 5))
                    self._last_order_error = f"HTTP 429: rate limited, retrying after {retry_after}s"
                    time.sleep(retry_after)
                    self._throttle()
                    continue
                # 401: session expired — re-login and retry once
                if r.status_code == 401 and attempt == 0:
                    self._last_order_error = "HTTP 401: unauthorized, re-authenticating"
                    if self._login():
                        self._throttle()
                        continue
                    self._last_order_error = "Re-authentication failed"
                return r
            except Exception:
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise
        return None

    def shutdown(self):
        if self.connected:
            try:
                self._session.delete(
                    f"{self.base_url}/api/v1/session",
                    headers=self._auth_headers(),
                    timeout=self._timeout,
                )
            except Exception:
                pass
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def reconnect(self, server: str, account: str, password: str) -> bool:
        self.shutdown()
        self.demo = "demo" in server.lower() or "DEMO" in server
        return self.initialize(self.api_key, account, password)

    def last_error(self) -> Tuple[int, str]:
        return 0, self._last_order_error or "No error"

    def last_order_error(self) -> str:
        return self._last_order_error

    def select_symbol(self, symbol: str) -> bool:
        info = self.get_symbol_info(symbol)
        if info is None:
            return False
        return info.get("market_status") == "TRADEABLE"

    def get_account_info(self) -> Optional[Dict]:
        if not self._ensure_session():
            self._last_order_error = "session_not_connected"
            return None
        try:
            r = self._request("GET", f"{self.base_url}/api/v1/accounts", headers=self._auth_headers())
            if r.ok:
                accounts = r.json().get("accounts", [])
                for acct in accounts:
                    if acct.get("preferred"):
                        bal = acct.get("balance", {})
                        raw_balance = float(bal.get("balance", 0))
                        raw_profit = float(bal.get("profitLoss", 0))
                        return {
                            "account_number": acct.get("accountId", ""),
                            "balance": raw_balance,
                            "equity": raw_balance + raw_profit,
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
        if not self._ensure_session():
            return None
        try:
            r = self._request("GET", f"{self.base_url}/api/v1/markets/{epic}", headers=self._auth_headers())
            if r.ok:
                data = r.json()
                snap = data.get("snapshot", {})
                inst = data.get("instrument", {})
                dr = data.get("dealingRules", {})
                dpf = int(snap.get("decimalPlacesFactor", 2))
                return {
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
                    "margin_rate": float(dr.get("marginFactor", {}).get("value", 0.05)),
                    "trade_mode": "ENABLED" if snap.get("marketStatus") == "TRADEABLE" else "DISABLED",
                    "market_status": snap.get("marketStatus", ""),
                    "filling_mode": 0,
                    "trade_stops_level": 0,
                }
        except Exception:
            pass
        return None

    def get_rates(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        epic = self._resolve_epic(symbol)
        resolution = TIMEFRAME_MAP.get(timeframe, "MINUTE")
        if not self._ensure_session():
            return None
        try:
            r = self._request("GET", f"{self.base_url}/api/v1/prices/{epic}",
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
            r = self._request("GET", f"{self.base_url}/api/v1/prices/{epic}",
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
                r = self._request("GET", f"{self.base_url}/api/v1/prices/{epic}",
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

    @staticmethod
    def _extract_price(price_field, side: str = "bid") -> float:
        if isinstance(price_field, dict):
            return float(price_field.get(side if side == "bid" else "offer", 0))
        try:
            return float(price_field)
        except (TypeError, ValueError):
            return 0.0

    def _parse_prices(self, prices: list) -> list:
        rows = []
        for p in prices:
            t = p.get("snapshotTime", "").replace("Z", "+00:00")
            if "." not in t and "+" not in t:
                t += "+00:00"
            dt = datetime.fromisoformat(t)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            open_price = p.get("openPrice", {})
            high_price = p.get("highPrice", {})
            low_price = p.get("lowPrice", {})
            close_price = p.get("closePrice", {})
            open_bid = self._extract_price(open_price, "bid")
            open_ask = self._extract_price(open_price, "ask")
            high_bid = self._extract_price(high_price, "bid")
            high_ask = self._extract_price(high_price, "ask")
            low_bid = self._extract_price(low_price, "bid")
            low_ask = self._extract_price(low_price, "ask")
            close_bid = self._extract_price(close_price, "bid")
            close_ask = self._extract_price(close_price, "ask")
            rows.append({
                "time": dt,
                "open": open_bid,
                "high": high_bid,
                "low": low_bid,
                "close": close_bid,
                "open_bid": open_bid,
                "open_ask": open_ask,
                "high_bid": high_bid,
                "high_ask": high_ask,
                "low_bid": low_bid,
                "low_ask": low_ask,
                "close_bid": close_bid,
                "close_ask": close_ask,
                "tick_volume": int(p.get("lastTradedVolume", 0)),
                "spread": round(max(0, open_ask - open_bid), 5),
                "real_volume": int(p.get("lastTradedVolume", 0)),
            })
        return rows

    def get_positions(self, magic: Optional[int] = None, symbol: Optional[str] = None) -> List[Dict]:
        if not self._ensure_session():
            return []
        try:
            target_epic = self._resolve_epic(symbol or cfg.SYMBOL) if magic is not None else None
            r = self._request("GET", f"{self.base_url}/api/v1/positions", headers=self._auth_headers())
            if r.ok:
                result = []
                for pos_data in r.json().get("positions", []):
                    p = pos_data.get("position", {})
                    mkt = pos_data.get("market", {})
                    epic = mkt.get("epic", "")
                    deal_id = p.get("dealId", "")
                    comment = p.get("reference", "")
                    if magic is not None:
                        if not comment.startswith(str(magic)) or epic != target_epic:
                            continue
                    result.append({
                        "ticket": deal_id,
                        "symbol": mkt.get("instrumentName", epic),
                        "type": "BUY" if p.get("direction") == "BUY" else "SELL",
                        "volume": float(p.get("size", 0)),
                        "price_open": float(p.get("level", 0)),
                        "price_current": float(mkt.get("bid", p.get("level", 0))),
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
            r = self._request("GET", f"{self.base_url}/api/v1/history/activity",
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
        now = datetime.utcnow()
        today = now.date()

        if self._daily_pnl_date != today:
            info = self.get_account_info()
            if info:
                self._prev_balance = info.get("balance", 0)
            self._daily_pnl_date = today

        info = self.get_account_info()
        if info is None or self._prev_balance is None:
            return 0.0

        current_balance = info.get("balance", 0)
        positions = self.get_positions(magic)
        open_pnl = sum(p["profit"] for p in positions)
        return (current_balance - self._prev_balance) + open_pnl

    async def order_send(self, request: dict) -> Dict:
        epic = request.get("epic", self._resolve_epic(request.get("symbol", "GOLD")))
        direction = "BUY" if request.get("type") == 0 else "SELL"
        volume = request.get("volume", 0.01)

        sl = request.get("sl")
        tp = request.get("tp")

        result = await self._open_position_raw(epic, direction, volume, sl, tp)
        if result:
            return {"retcode": 10009, "order": result.get("dealReference", ""), "comment": "Done",
                    "volume": volume, "price": 0, "bid": 0, "ask": 0, "success": True}
        return {"retcode": 10004, "order": 0, "comment": "Open failed", "volume": 0, "price": 0,
                "bid": 0, "ask": 0, "success": False}

    async def _open_position_raw(self, epic: str, direction: str, volume: float,
                                   stop_loss: Optional[float] = None,
                                   take_profit: Optional[float] = None,
                                   force_open: bool = True,
                                   reference: str = "") -> Optional[Dict]:
        import asyncio
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
            r = self._request("POST", f"{self.base_url}/api/v1/positions",
                                   headers=self._auth_headers(), json=body)
            if r.ok:
                self._last_order_error = ""
                await asyncio.sleep(0.5)
                return r.json()
            self._last_order_error = f"HTTP {r.status_code}: {r.text[:500]}"
        except Exception as exc:
            self._last_order_error = f"{type(exc).__name__}: {exc}"
        return None

    async def open_position(self, symbol: str, direction: str, volume: float,
                            price: Optional[float] = None,
                            stop_loss: Optional[float] = None,
                            take_profit: Optional[float] = None,
                            comment: str = "",
                            magic: int = 0,
                            slippage: int = 30) -> Optional[str]:
        import asyncio
        epic = self._resolve_epic(symbol)
        reference = str(magic) + ":" + comment if comment else str(magic)
        positions = self.get_positions()
        for p in positions:
            p_epic = EPIC_MAP.get(p.get("symbol", ""), p.get("symbol", ""))
            if p_epic != epic:
                continue
            p_dir = p.get("type", "")
            if p_dir and p_dir != direction.upper():
                self._last_order_error = (
                    f"Opposing position exists ({p_dir}) for {epic} "
                    f"cannot open {direction.upper()}"
                )
                return None
        result = await self._open_position_raw(epic, direction, volume, stop_loss, take_profit, reference=reference)
        if result is None:
            return None
        deal_ref = result.get("dealReference", "")
        expected_prefix = str(magic)
        for _ in range(24):
            await asyncio.sleep(0.5)
            fresh = self.get_positions()
            for p in fresh:
                if p.get("comment", "").startswith(expected_prefix):
                    return p.get("ticket")
        self._last_order_error = "Order submitted but position not confirmed"
        return None

    def close_position(self, ticket) -> bool:
        ticket_str = str(ticket)
        if isinstance(ticket, str):
            deal_id = ticket
        else:
            positions = self.get_positions()
            pos = next((p for p in positions if str(p["ticket"]) == ticket_str), None)
            if pos is None:
                return False
            deal_id = pos["ticket"]
        if not self._ensure_session():
            return False
        try:
            r = self._request("DELETE", f"{self.base_url}/api/v1/positions/{deal_id}",
                                     headers=self._auth_headers())
            return r is not None and r.ok
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
            r = self._request("PUT", f"{self.base_url}/api/v1/positions/{deal_id}",
                                  headers=self._auth_headers(), json=body)
            return r is not None and r.ok
        except Exception:
            return False
