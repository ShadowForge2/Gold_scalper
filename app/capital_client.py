import requests
import math
import json
import pandas as pd
import time
import asyncio
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from collections import deque
import logging
import config as cfg


logger = logging.getLogger(__name__)


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
    "US100": "US100",
    "NASDAQ": "US100",
    "NAS100": "US100",
    "US500": "US500",
    "SP500": "US500",
    "US30": "US30",
    "DOW": "US30",
    "DJ30": "US30",
    "DJIA": "US30",
    "JP225": "J225",
    "JPN225": "J225",
    "N225": "J225",
    "J225": "J225",
    "NIKKEI": "J225",
    "DE40": "DE40",
    "GER40": "DE40",
    "GERMANY40": "DE40",
    "DAX": "DE40",
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
        self._last_error_code = ""
        self._last_error_hint = ""
        self._request_times: deque = deque(maxlen=20)
        self._max_requests_per_sec = 8
        self._timeout = 15
        self._preferences_cache: Optional[Dict] = None
        self._preferences_ts = 0.0
        self._preferences_ttl = 300.0

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
                    self._last_error_code = ""
                    self._last_error_hint = ""
                    return True
                self._record_login_error(r.status_code, r.text)
                if r.status_code not in (429, 502, 503, 504):
                    break
            except Exception as exc:
                self._last_order_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                backoff = min(2 ** attempt, 30)
                time.sleep(backoff)
        self.connected = False
        return False

    def _record_login_error(self, status: int, text: str) -> None:
        """Parse a Capital.com login failure into a stable code + friendly hint.

        Capital returns e.g. {'errorCode': 'error.invalid.details', ...} or a
        bare string body. Keep the raw HTTP text too so callers can fall back.
        """
        code = ""
        message = ""
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    code = str(data.get("errorCode") or "")
                    message = str(data.get("errorMessage") or data.get("message") or "")
            except (ValueError, TypeError):
                pass
        if not message:
            message = text[:500]
        self._last_error_code = code
        self._last_error_hint = self._friendly_auth_error(code, message, status)
        self._last_order_error = f"HTTP {status}: {message}"

    @staticmethod
    def _friendly_auth_error(code: str, message: str, status: int) -> str:
        """Map Capital login errors to actionable, human-readable messages."""
        known = {
            "error.invalid.api-key": "The API key is invalid or was revoked. Generate a new key in Capital.com → Settings → API integrations.",
            "error.invalid.details": "The identifier or password is incorrect. Capital.com uses the same credentials for demo and live — double-check your email and API key password.",
            "error.invalid.identifier": "No Capital.com account matches this identifier/email.",
            "error.invalid.password": "The password is incorrect.",
            "error.account.disabled": "This Capital.com account is disabled. Contact Capital.com support.",
            "error.too.many.requests": "Capital.com is rate-limiting login attempts. Wait a few minutes and try again.",
            "error.invalid.session": "Session rejected. Try again.",
        }
        lowered = code.lower()
        if lowered in known:
            return known[lowered]
        if "api" in lowered and "key" in lowered:
            return known["error.invalid.api-key"]
        if "invalid" in lowered:
            return "The identifier or password is incorrect for this account type."
        if status == 401:
            return "Authentication failed — check your email and API key password (same for demo and live)."
        if status == 403:
            return "Access denied by Capital.com. The API key may lack trading permissions or the account is restricted."
        if status in (429, 502, 503, 504):
            return f"Capital.com is temporarily unavailable (HTTP {status}). Try again shortly."
        return message or f"Broker authentication failed (HTTP {status})."

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
                elif r.status_code != 200:
                    self.connected = False
                    return False
            except Exception:
                self.connected = False
                return False
        if self.connected:
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
                # 429: rate limited — sleep Retry-After then retry once
                if r.status_code == 429:
                    try:
                        retry_after = int(r.headers.get("Retry-After", 5))
                    except (ValueError, TypeError):
                        retry_after = 5
                    self._last_order_error = f"HTTP 429: rate limited, retrying after {retry_after}s"
                    time.sleep(retry_after)
                    self._throttle()
                    if attempt == 0:
                        continue
                    return r
                # 401: session expired — re-login and retry once
                if r.status_code == 401 and attempt == 0:
                    self._last_order_error = "HTTP 401: unauthorized, re-authenticating"
                    if self._login():
                        kwargs["headers"] = self._auth_headers()
                        self._throttle()
                        continue
                    self._last_order_error = "Re-authentication failed"
                return r
            except Exception:
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise

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

    def reconnect(self, server: str, account: str, password: str, api_key: Optional[str] = None, demo: Optional[bool] = None) -> bool:
        self.shutdown()
        if demo is not None:
            self.demo = demo
        else:
            self.demo = "demo" in server.lower() if server else self.demo
        return self.initialize(api_key or self.api_key, account, password)

    def last_error(self) -> Tuple[int, str]:
        return 0, self._last_order_error or "No error"

    def last_error_hint(self) -> str:
        return self._last_error_hint or self._last_order_error or ""

    def last_error_code(self) -> str:
        return self._last_error_code

    def last_order_error(self) -> str:
        return self._last_order_error

    def select_symbol(self, symbol: str) -> bool:
        info = self.get_symbol_info(symbol)
        if info is None:
            return False
        return info.get("market_status") == "TRADEABLE"

    def get_account_preferences(self) -> Optional[Dict]:
        """Fetch leverage per asset class from /accounts/preferences.

        Returns {"leverages": {"COMMODITIES": {"current": 5, ...}, ...}} or None.
        Cached for _preferences_ttl seconds to avoid hammering the endpoint.
        """
        if not self._ensure_session():
            return self._preferences_cache
        now = time.time()
        if self._preferences_cache is not None and now - self._preferences_ts < self._preferences_ttl:
            return self._preferences_cache
        try:
            r = self._request("GET", f"{self.base_url}/api/v1/accounts/preferences",
                              headers=self._auth_headers())
            if r is not None and r.ok:
                data = r.json() or {}
                self._preferences_cache = data
                self._preferences_ts = now
                return data
        except Exception:
            pass
        return self._preferences_cache

    def get_leverage_for_class(self, asset_class: str) -> float:
        """Current leverage for an asset class (e.g. COMMODITIES, INDICES).

        Falls back to 1 / margin_factor style defaults if preferences are
        unavailable. Capital.com caps commodities/indices leverage (<=20), so a
        missing value defaults to 20 (the least-margin requirement) rather than
        the old hardcoded 100.
        """
        pref = self.get_account_preferences()
        if pref:
            lev = (pref.get("leverages") or {}).get(asset_class, {})
            current = lev.get("current") if isinstance(lev, dict) else None
            if current:
                try:
                    return float(current)
                except (TypeError, ValueError):
                    pass
        return 20.0

    def _max_account_leverage(self) -> float:
        """Largest current leverage across asset classes (for display only)."""
        pref = self.get_account_preferences()
        if pref:
            levs = pref.get("leverages") or {}
            values = []
            for lev in levs.values():
                if isinstance(lev, dict):
                    try:
                        values.append(float(lev.get("current") or 0))
                    except (TypeError, ValueError):
                        pass
            if values:
                return max(values)
        return 20.0

    def estimate_margin(self, symbol: str, lot: float, price: float) -> float:
        """Estimate required margin for an order before submitting it.

        margin = (lot * contract_size * price) / leverage

        Uses the real per-asset-class leverage from /accounts/preferences and
        the per-symbol contract size (config). This replaces the old formula
        `lot * price * margin_rate * 1`, which (a) ignored contract size (the
        `* 1`), (b) read marginFactor from the wrong location, and (c) treated
        a PERCENTAGE value as a decimal fraction.
        """
        info = self.get_symbol_info(symbol) or {}
        asset_class = info.get("asset_class") or getattr(cfg, "SYMBOL_ASSET_CLASS", {}).get(symbol, "COMMODITIES")
        contract_size = info.get("contract_size") or float(getattr(cfg, "SYMBOL_CONTRACT_SIZE", {}).get(symbol, 1))
        leverage = self.get_leverage_for_class(asset_class)
        try:
            return max(0.0, float(lot)) * max(0.0, float(contract_size)) * max(0.0, float(price)) / max(1.0, leverage)
        except (TypeError, ValueError):
            return 0.0

    def get_account_info(self) -> Optional[Dict]:
        if not self._ensure_session():
            self._last_order_error = "session_not_connected"
            return None
        try:
            r = self._request("GET", f"{self.base_url}/api/v1/accounts", headers=self._auth_headers())
            if r is not None and r.ok:
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
                            "leverage": self._max_account_leverage(),
                            "leverages": (self.get_account_preferences() or {}).get("leverages", {}),
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
            if r is not None and r.ok:
                data = r.json()
                snap = data.get("snapshot", {})
                inst = data.get("instrument", {})
                dr = data.get("dealingRules", {})
                dpf = int(snap.get("decimalPlacesFactor", 2))
                # Capital.com returns marginFactor under instrument (e.g. 10 =
                # 10%), NOT under dealingRules. It is a PERCENTAGE, so convert to
                # a decimal fraction of notional for the margin estimate.
                mf_raw = float(inst.get("marginFactor", 0) or 0)
                mf_unit = inst.get("marginFactorUnit", "PERCENTAGE")
                if mf_raw > 0 and mf_unit == "PERCENTAGE":
                    margin_rate = mf_raw / 100.0
                else:
                    margin_rate = float(getattr(cfg, "SYMBOL_MARGIN_FACTOR", {}).get(symbol, mf_raw or 0.01))
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
                    "margin_rate": margin_rate,
                    "margin_factor_raw": mf_raw,
                    "margin_factor_unit": mf_unit,
                    "contract_size": float(getattr(cfg, "SYMBOL_CONTRACT_SIZE", {}).get(symbol, 1)),
                    "asset_class": inst.get("type", getattr(cfg, "SYMBOL_ASSET_CLASS", {}).get(symbol, "COMMODITIES")),
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
            if r is not None and r.ok:
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

        # Always page BACKWARD from the most recent candle. Capital.com caps a
        # single from/to range response (default/max candles per request), and a
        # capped response is NOT guaranteed to end at the newest bar — if it is
        # truncated to the OLDEST candles in range, the tail goes stale and any
        # candle-window detection silently never fires. Paginating from the
        # latest bar guarantees fresh data plus the requested history.
        page_max = 1000
        need = int((to_dt - from_dt).total_seconds() // 60) + 1
        pages = max(1, min(int(math.ceil(need / page_max)), 20))

        all_rows = []
        cursor_to = None
        for _ in range(pages):
            try:
                params = {"resolution": resolution, "max": page_max}
                if cursor_to:
                    params["to"] = cursor_to
                r = self._request("GET", f"{self.base_url}/api/v1/prices/{epic}",
                                  params=params,
                                  headers=self._auth_headers())
                if r is None or not r.ok:
                    break
                prices = r.json().get("prices", [])
                if not prices:
                    break
                all_rows.extend(self._parse_prices(prices))
                snap_times = [p.get("snapshotTime", "") for p in prices if p.get("snapshotTime")]
                if not snap_times:
                    break
                # Order-independent cursor: the oldest candle in this page is
                # the `to` bound for the next (older) page. Using min() rather
                # than prices[0] handles either ascending or descending order.
                oldest = min(snap_times)
                if oldest == cursor_to:
                    break  # no older data — start of available history reached
                cursor_to = oldest
            except Exception:
                break

        if not all_rows:
            # `to`-cursor pagination failed entirely (e.g. bad cursor rejected).
            # Fall back to the plain most-recent-N fetch used everywhere else —
            # returns the latest candles only, but at least we return data.
            try:
                r = self._request("GET", f"{self.base_url}/api/v1/prices/{epic}",
                                  params={"resolution": resolution, "max": page_max},
                                  headers=self._auth_headers())
                if r is not None and r.ok:
                    prices = r.json().get("prices", [])
                    if prices:
                        all_rows = self._parse_prices(prices)
            except Exception:
                pass

        if not all_rows:
            return None
        df = pd.DataFrame(all_rows)
        df.drop_duplicates(subset="time", keep="last", inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

        mask = (df["time"] >= from_dt) & (df["time"] <= to_dt)
        return df[mask].copy() if mask.any() else df

    def get_all_markets(self) -> Optional[List[Dict]]:
        """Fetch the full tradable-market catalog in ONE request.

        Capital.com returns every market (4000+) with live snapshot fields
        (bid/offer/status/type/high/low/pct change) when a pageSize param is
        sent, regardless of its value. Used by the pair scanner to pick
        tradeable instruments on the whole board each scan."""
        if not self._ensure_session():
            return None
        try:
            r = self._request(
                "GET", f"{self.base_url}/api/v1/markets",
                params={"pageSize": 500},
                headers=self._auth_headers(),
            )
            if r is not None and r.ok:
                return r.json().get("markets") or []
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_price(price_field, side: str = "bid") -> float:
        if isinstance(price_field, dict):
            return float(price_field.get(side if side == "bid" else "ask", 0))
        try:
            return float(price_field)
        except (TypeError, ValueError):
            return 0.0

    def _parse_prices(self, prices: list) -> list:
        rows = []
        for p in prices:
            t = p.get("snapshotTime", "").replace("Z", "+00:00") if p.get("snapshotTime") else ""
            if not t:
                continue
            if "." not in t and "+" not in t:
                t += "+00:00"
            try:
                dt = datetime.fromisoformat(t)
            except ValueError:
                continue
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
            filter_symbol = symbol or cfg.SYMBOL
            target_epic = self._resolve_epic(filter_symbol)
            r = self._request("GET", f"{self.base_url}/api/v1/positions", headers=self._auth_headers())
            if r is not None and r.ok:
                result = []
                for pos_data in r.json().get("positions", []):
                    p = pos_data.get("position", {})
                    mkt = pos_data.get("market", {})
                    epic = mkt.get("epic", "")
                    if epic != target_epic:
                        continue
                    if magic is not None:
                        ref = p.get("reference", "") or p.get("dealReference", "")
                        if str(magic) not in ref:
                            continue
                    deal_id = p.get("dealId", "")
                    comment = p.get("reference", "") or p.get("dealReference", "")
                    result.append({
                        "ticket": deal_id,
                        "symbol": mkt.get("instrumentName", epic),
                        "type": "BUY" if p.get("direction") == "BUY" else "SELL",
                        "volume": float(p.get("size", 0)),
                        "price_open": float(p.get("level", 0)),
                        "price_current": float(mkt.get("bid") or p.get("level", 0)),
                        "sl": float(p.get("stopLevel", 0)) if p.get("stopLevel") else 0.0,
                        "tp": float(p.get("profitLevel", 0)) if p.get("profitLevel") else 0.0,
                        "profit": float(p.get("upl") or 0),
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
        open_pnl = sum(p.get("profit", 0) for p in positions)
        return (current_balance - self._prev_balance) + open_pnl

    async def order_send(self, request: dict) -> Dict:
        epic = request.get("epic") or self._resolve_epic(request.get("symbol") or "GOLD")
        req_type = request.get("type")
        if req_type is None:
            logger.error("Missing 'type' in request")
            return {"retcode": 10004, "order": 0, "comment": "missing_order_type",
                    "volume": 0, "price": 0, "bid": 0, "ask": 0, "success": False}
        direction = "BUY" if req_type == 0 else "SELL"
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
            if r is not None and r.ok:
                self._last_order_error = ""
                await asyncio.sleep(0.5)
                return r.json()
            self._last_order_error = f"HTTP {r.status_code}: {r.text[:500]}" if r is not None else "Request returned None"
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
        epic = self._resolve_epic(symbol)
        reference = str(magic) + ":" + comment if comment else str(magic)
        positions = self.get_positions(symbol=symbol)
        for p in positions:
            p_sym = p.get("symbol", "")
            p_epic = EPIC_MAP.get(p_sym, p_sym)
            if p_epic != epic:
                continue
            p_dir = p.get("type", "")
            if p_dir and direction and p_dir != direction.upper():
                self._last_order_error = (
                    f"Opposing position exists ({p_dir}) for {epic} "
                    f"cannot open {direction.upper()}"
                )
                return None
        existing_tickets = {str(p.get("ticket")) for p in positions if p.get("ticket")}
        result = await self._open_position_raw(epic, direction, volume, stop_loss, take_profit, reference=reference)
        if result is None:
            return None
        for _ in range(24):
            await asyncio.sleep(0.5)
            fresh = self.get_positions(symbol=symbol)
            for p in fresh:
                ticket = p.get("ticket")
                if ticket and str(ticket) not in existing_tickets:
                    return ticket
        self._last_order_error = "Order submitted but position not confirmed"
        return None

    def close_position(self, ticket) -> bool:
        ticket_str = str(ticket)
        if isinstance(ticket, str):
            deal_id = ticket
        else:
            found = False
            for sym in getattr(cfg, 'SYMBOLS', [cfg.SYMBOL]):
                positions = self.get_positions(symbol=sym)
                pos = next((p for p in positions if str(p.get("ticket", "")) == ticket_str), None)
                if pos is not None:
                    deal_id = pos["ticket"]
                    found = True
                    break
            if not found:
                return False
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
