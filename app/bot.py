import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

import config as cfg
from app.logger import BotLogger
from app.capital_client import CapitalClient
from app.risk_manager import RiskManager
from app.trade_executor import TradeExecutor
from app.position_manager import PositionManager
from app.economic_calendar import EconomicCalendar
from app.news_state_machine import NewsStateMachine
from app.candle_engine import compute_atr
from app.universal_pull_scanner import UniversalPullScanner

try:
    from app.pull_h1_scalper import PullPrevH1Scalper
    _HAS_PULL = True
except Exception:
    PullPrevH1Scalper = None
    _HAS_PULL = False

try:
    from app import pull_auto_tune
    _HAS_AUTO_TUNE = True
except Exception:
    pull_auto_tune = None
    _HAS_AUTO_TUNE = False

try:
    from app.pair_scanner import PairScanner
    _HAS_SCANNER = True
except Exception:
    PairScanner = None
    _HAS_SCANNER = False


class Bot:
    STATES = {
        "IDLE": "IDLE",
        "ENTERING": "ENTERING",
        "IN_TRADE": "IN_TRADE",
        "STOPPED": "STOPPED",
        "WAITING_FOR_FUNDS": "WAITING_FOR_FUNDS",
        "MARKET_CLOSED": "MARKET_CLOSED",
    }

    def __init__(self, logger: Optional[BotLogger] = None):
        self.logger = logger or BotLogger()
        self.client: object = None
        self.symbols: List[str] = cfg.SYMBOLS
        self._symbol_states: Dict[str, str] = {}
        self._symbol_signals: Dict[str, Optional[Dict]] = {}
        self._symbol_event_start_ts: Dict[str, Optional[float]] = {}
        self._symbol_atr_history: Dict[str, list] = {}  # rolling ATR for vol detection
        # Pull-into-H1 scalper per-symbol state
        self._symbol_pull_engine: Dict[str, Optional[object]] = {}  # PullPrevH1Scalper instances
        self._symbol_pull_entry: Dict[str, bool] = {}   # current trade is a pull entry
        self._symbol_pull_cache_ts: Dict[str, float] = {}  # M1 history fetch time
        self._last_pull_warn_ts: float = 0.0  # throttle for pull history warnings
        self._symbol_engine_failures: Dict[str, int] = {}  # consecutive pull engine failures per symbol
        # Whole-board scanner: the active tradeable universe is the scanner's
        # top-K momentum leaders (no fixed symbol configuration — scan and see).
        self.scanner: Optional[object] = (
            PairScanner(None, self.logger)
            if _HAS_SCANNER and getattr(cfg, "SCANNER_ENABLED", False) else None
        )
        self._scanner_pull_syms: set = set()  # dynamic/configured syms armed with the general pull engine
        self._candidate_map: Dict[str, Dict] = {}  # canonical sym -> scanner candidate row
        # Universal pull scanner — scans ALL symbols using proven pull-into-H1 strategy
        self.universal_pull_scanner = UniversalPullScanner(self.logger)

        # Load per-symbol pull scalper engines. The pull scalper owns entry/exit
        # for every symbol — there is no other strategy path.
        for sym in self.symbols:
            pull_owns = bool(getattr(cfg, "PULL_ENGINE_ENABLED", {}).get(sym, False)) and _HAS_PULL

            self._symbol_states[sym] = self.STATES["IDLE"]
            self._symbol_signals[sym] = None
            self._symbol_event_start_ts[sym] = None
            self._symbol_atr_history[sym] = []
            self._symbol_engine_failures[sym] = 0  # consecutive pull engine failures

            # ── Pull-into-H1 scalper engine (owns entry/exit when enabled) ──
            pull_eng = None
            if pull_owns:
                try:
                    pp = dict(getattr(cfg, "SYMBOL_PULL_PARAMS", {}).get(sym, {}) or {})
                    pull_eng = PullPrevH1Scalper(
                        symbol=sym,
                        logger=self.logger,
                        pull_r=float(pp.get("pull_r", 0.30)),
                        trail_r=float(pp.get("trail_r", 0.35)),
                        max_hold_bars=int(pp.get("max_hold", 24)),
                        round_trip_price=float(pp.get("round_trip", 0.0)),
                        min_h1_bars=int(getattr(cfg, "PULL_MIN_H1_BARS", 30)),
                        daily_target_r=float(pp.get("daily_target_r", 0.0)),
                        daily_max_loss_r=float(pp.get("daily_max_loss_r", 0.0)),
                        giveback_cap=float(pp.get("giveback_cap", getattr(cfg, "PULL_GIVEBACK_CAP", 0.30))),
                        pump_atr=float(pp.get("pump_atr", getattr(cfg, "PULL_PUMP_ATR", 0.5))),
                    )
                    self.logger.info(
                        f"[{sym}] Pull scalper enabled (pull {pull_eng.pull_r}R "
                        f"trail {pull_eng.trail_r}R max_hold {pull_eng.max_hold_bars} "
                        f"cost {pull_eng.round_trip_price:.2f})"
                    )
                except Exception as e:
                    self.logger.warning(f"[{sym}] Pull scalper init failed: {e}")
                    pull_eng = None
            self._symbol_pull_engine[sym] = pull_eng
            self._symbol_pull_entry[sym] = False
            self._symbol_pull_cache_ts[sym] = 0.0

        # Legacy single-symbol reference (use first symbol for backward compat)
        first_sym = self.symbols[0] if self.symbols else cfg.SYMBOL

        self.risk_manager = RiskManager()

        self.state: str = self.STATES["IDLE"]
        self.symbol: str = first_sym
        self.magic: int = cfg.MAGIC_NUMBER
        self._running = False
        self._starting_balance: Optional[float] = None
        self._lot_multiplier_override: Optional[float] = None  # per-bot, kept for pool compat
        self.is_demo: bool = True  # set to False by initialize() or initialize_with_credentials()

        self._current_signal: Optional[Dict] = None
        self._last_tick: Optional[Dict] = None
        self._last_signal_diag_key: Optional[str] = None
        self._last_signal_diag_time = 0.0
        self._last_signal_found_key: Optional[str] = None
        self._last_signal_found_time = 0.0
        self._last_signal_blocked_key: Optional[str] = None
        self._last_signal_blocked_time = 0.0

        self._accounts_file: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "accounts.json")
        self._account_id: Optional[str] = None
        self._state_file: Optional[str] = None
        self._last_state_write = 0.0
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0.0
        self._reconnect_backoff_max = 60.0

        self._can_trade_cb = None
        self._winding_down = False

        # News-aware trading
        self.news_calendar = EconomicCalendar(
            cache_path="data/calendar_cache.pkl",
            cache_ttl_hours=cfg.NEWS_CACHE_TTL_HOURS,
            user_events_path=cfg.NEWS_USER_EVENTS_PATH,
            jblanked_api_key=cfg.JBLANKED_API_KEY,
            finnhub_api_key=cfg.FINNHUB_API_KEY,
        ) if cfg.NEWS_AWARE_ENABLED else None
        self.news_state = NewsStateMachine(
            pre_window_min=cfg.NEWS_PRE_WINDOW_MINUTES,
            spike_window_min=cfg.NEWS_SPIKE_WINDOW_MINUTES,
            post_window_min=cfg.NEWS_POST_WINDOW_MINUTES,
        ) if cfg.NEWS_AWARE_ENABLED else None
        if self.news_state and self.news_calendar:
            self.news_state.set_calendar(self.news_calendar)
            ev = self.news_calendar.get_next_event()
            if ev:
                self.logger.info(f"[NEWS] Next event: '{ev['title']}' @ {ev['datetime'].strftime('%H:%M UTC %d-%b')}")
            else:
                self.logger.info("[NEWS] No upcoming events found")
        self._shutdown_deadline = None
        self._last_sub_check = 0.0
        self._creds: Optional[Dict] = None
        self._last_market_status_check = 0.0
        self._market_check_interval = 60


    def _check_market_dynamic(self) -> Optional[bool]:
        now = time.time()
        if now - self._last_market_status_check < self._market_check_interval:
            return None
        self._last_market_status_check = now
        if not self.client:
            return None
        try:
            info = self.client.get_symbol_info(self.symbol)
            if info is None:
                return None
            return info.get("market_status") == "TRADEABLE"
        except Exception:
            return None

    @staticmethod
    def _fmt_leverages(levs: dict) -> str:
        """Render the per-asset-class leverage map (leverage is dynamic per
        instrument type, NOT a single account value)."""
        if not levs:
            return "per-instrument (n/a)"
        parts = []
        for cls, lev in levs.items():
            cur = lev.get("current") if isinstance(lev, dict) else None
            if cur:
                parts.append(f"{cls} 1:{cur}")
        return " | ".join(parts) if parts else "per-instrument (n/a)"


    async def _notify(self, ntype: str, title: str, message: str, data: Optional[Dict] = None):
        ident = getattr(self, '_account_id', None) or cfg.CAPITAL_IDENTIFIER or "unknown"
        try:
            from app.subscription import create_notification
            await create_notification(ident, ntype, title, message, data)
        except Exception as e:
            self.logger.warning(f"Notification failed ({ntype}): {e}")

    async def initialize(self) -> bool:
        n_cfg = len(self.symbols)
        scanner_on = bool(getattr(cfg, "SCANNER_ENABLED", False))
        self.is_demo = cfg.CAPITAL_DEMO
        self.logger.info(
            f"Initializing multi-symbol pull-into-H1 scalper (whole-board scanner) — "
            f"{n_cfg} configured symbol(s), board scanner {'ENABLED' if scanner_on else 'disabled'}, "
            f"auto-tune {'ON' if getattr(cfg, 'PULL_AUTO_TUNE_ENABLED', True) else 'OFF'}"
        )
        self.logger.info(f"Config: LOT_MULTIPLIER={cfg.LOT_MULTIPLIER}")
        self.logger.warning(
            f"*** ACCOUNT TYPE: {'DEMO' if cfg.CAPITAL_DEMO else 'LIVE'} *** "
            f"Signals computed from {'DEMO' if cfg.CAPITAL_DEMO else 'LIVE'} chart data. "
            f"Base URL: {'demo-api-capital' if cfg.CAPITAL_DEMO else 'api-capital'}.backend-capital.com"
        )

        self.logger.info("Broker: Capital.com (REST API)")
        self.client = CapitalClient()
        if self.scanner is not None:
            self.scanner.client = self.client
        success = self.client.initialize(
            api_key=cfg.CAPITAL_API_KEY,
            identifier=cfg.CAPITAL_IDENTIFIER,
            password=cfg.CAPITAL_PASSWORD,
            demo=cfg.CAPITAL_DEMO,
        )

        self.trade_executor = TradeExecutor(self.client, self.logger)
        self.position_manager = PositionManager(self.client)

        if success:
            self._reconnect_attempts = 0
            info = self.client.get_account_info()
            if info:
                if self._starting_balance is None:
                    self._starting_balance = info["balance"]
                self.logger.info(
                    f"Connected: {info['name']} | "
                    f"Balance: ${info['balance']:.2f} | "
                    f"Leverage (per instrument): {self._fmt_leverages(info.get('leverages'))}"
                )
                if info["balance"] < cfg.MIN_BALANCE:
                    self.logger.warning(
                        f"Balance ${info['balance']:.2f} below minimum ${cfg.MIN_BALANCE:.2f}. "
                        f"Bot waiting for funds..."
                    )
                    self.state = self.STATES["WAITING_FOR_FUNDS"]
                    for sym in self.symbols:
                        self._symbol_states[sym] = self.STATES["WAITING_FOR_FUNDS"]
                else:
                    self.state = self.STATES["IDLE"]
                    for sym in self.symbols:
                        self._symbol_states[sym] = self.STATES["IDLE"]
            else:
                self.state = self.STATES["IDLE"]
            return True
        else:
            err = self.client.last_error()
            self.logger.error(f"Connection failed: {err}")
            self.state = self.STATES["IDLE"]
            return False

    async def initialize_with_credentials(self, api_key: str, identifier: str, password: str, demo: bool = True) -> bool:
        self._creds = {"api_key": api_key, "identifier": identifier, "password": password, "demo": demo}
        self.is_demo = demo
        self.client = CapitalClient()
        if self.scanner is not None:
            self.scanner.client = self.client
        self.logger.info(f"Connecting to Capital.com — account type: {'DEMO' if demo else 'LIVE'}")
        self.logger.warning(
            f"*** ACCOUNT TYPE: {'DEMO' if demo else 'LIVE'} *** "
            f"Signals computed from {'DEMO' if demo else 'LIVE'} chart data. "
            f"Base URL: {'demo-api-capital' if demo else 'api-capital'}.backend-capital.com"
        )
        success = self.client.initialize(
            api_key=api_key,
            identifier=identifier,
            password=password,
            demo=demo,
        )
        self.trade_executor = TradeExecutor(self.client, self.logger)
        self.position_manager = PositionManager(self.client)

        if success:
            self._reconnect_attempts = 0
            info = self.client.get_account_info()
            if info:
                if self._starting_balance is None:
                    self._starting_balance = info["balance"]
                self.logger.info(f"Account connected ✓")
                self.logger.info(
                    f"Balance: ${info['balance']:.2f} | "
                    f"Leverage (per instrument): {self._fmt_leverages(info.get('leverages'))}"
                )
                if info["balance"] < cfg.MIN_BALANCE:
                    self.logger.warning(f"Balance ${info['balance']:.2f} below minimum ${cfg.MIN_BALANCE:.2f}")
                    self.state = self.STATES["WAITING_FOR_FUNDS"]
                    for sym in self.symbols:
                        self._symbol_states[sym] = self.STATES["WAITING_FOR_FUNDS"]
                else:
                    self.state = self.STATES["IDLE"]
                    for sym in self.symbols:
                        self._symbol_states[sym] = self.STATES["IDLE"]
                self._write_state()
                self._verify_symbol_epics()
                return True
            self.state = self.STATES["IDLE"]
            for sym in self.symbols:
                self._symbol_states[sym] = self.STATES["IDLE"]
            self._write_state()
            self._verify_symbol_epics()
            return True
        else:
            err = self.client.last_error()
            self.logger.error(f"Authentication failed: {err}")
            self.state = self.STATES["STOPPED"]
            for sym in self.symbols:
                self._symbol_states[sym] = self.STATES["STOPPED"]
            return False

    def _verify_symbol_epics(self):
        """Log the resolved epic + market status for every configured symbol."""
        try:
            for sym in self.symbols:
                info = self.client.get_symbol_info(sym)
                if info is None:
                    self.logger.warning(
                        f"[{sym}] Epic check: '{self.client._resolve_epic(sym)}' "
                        f"NOT resolvable — market_status unknown, symbol cannot trade"
                    )
                else:
                    self.logger.info(
                        f"[{sym}] Epic check: '{self.client._resolve_epic(sym)}' -> "
                        f"name={info.get('name')} status={info.get('market_status')}"
                    )
        except Exception as e:
            self.logger.warning(f"Epic verification failed: {e}")

    # ── Whole-board scanner: dynamic tradeable universe ──────────────
    # The bot does NOT rely on the configured symbol list. Each scan the
    # PairScanner pulls the full market board, screens it, ranks by momentum
    # and keeps the top-K as the active universe. Symbols are canonicalized to
    # configured codes (GOLD -> XAUUSD) so positions/state never double-key.

    @staticmethod
    def _is_blacklisted(sym: str) -> bool:
        return str(sym or "").strip().upper() in getattr(cfg, "BLACKLIST_SYMBOLS", set())

    def _canon_sym(self, key: str) -> str:
        key = str(key or "").strip().upper()
        if not key:
            return ""
        if self.client is None:
            return key
        try:
            epic = self.client._resolve_epic(key)
        except Exception:
            return key
        for s in self.symbols:
            try:
                if self.client._resolve_epic(s) == epic:
                    return s
            except Exception:
                continue
        return key

    def _make_general_pull_engine(self, sym: str) -> Optional[object]:
        if not _HAS_PULL or PullPrevH1Scalper is None:
            return None
        round_trip = 0.0
        cand = self._candidate_map.get(sym) or {}
        try:
            spread = float(cand.get("spread") or 0)
            if spread > 0:
                round_trip = spread
        except (TypeError, ValueError):
            round_trip = 0.0
        # Self-calibrate: use explicit SYMBOL_PULL_PARAMS if present, else
        # auto-tune from the symbol's own recent structure (cached/refreshed).
        pp = {}
        if _HAS_AUTO_TUNE and pull_auto_tune is not None:
            try:
                pp = pull_auto_tune.get_pull_params(sym, self.client, self.logger)
            except Exception as e:
                self.logger.debug(f"[{sym}] auto-tune failed: {e}")
                pp = {}
        if not pp:
            pp = {
                "pull_r": float(getattr(cfg, "SCANNER_PULL_R", 0.30)),
                "trail_r": float(getattr(cfg, "SCANNER_TRAIL_R", 0.35)),
                "max_hold": int(getattr(cfg, "SCANNER_MAX_HOLD", 12)),
            }
        return PullPrevH1Scalper(
            symbol=sym,
            logger=self.logger,
            pull_r=float(pp.get("pull_r", getattr(cfg, "SCANNER_PULL_R", 0.30))),
            trail_r=float(pp.get("trail_r", getattr(cfg, "SCANNER_TRAIL_R", 0.35))),
            max_hold_bars=int(pp.get("max_hold", getattr(cfg, "SCANNER_MAX_HOLD", 12))),
            round_trip_price=round_trip,
            min_h1_bars=int(getattr(cfg, "PULL_MIN_H1_BARS", 30)),
            giveback_cap=float(pp.get("giveback_cap", getattr(cfg, "PULL_GIVEBACK_CAP", 0.30))),
            pump_atr=float(pp.get("pump_atr", getattr(cfg, "PULL_PUMP_ATR", 0.5))),
        )

    def _ensure_symbol_state(self, sym: str) -> None:
        """Create per-symbol state + a general pull engine for a scanner
        candidate. The pull engine owns entry/exit for EVERY scan trade."""
        if self._is_blacklisted(sym):
            # Blacklisted pair (DE40/JP225/US500): never arm a pull engine, so
            # no NEW entries ever fire here. State is kept only so any
            # already-open position is still recovered and managed (never
            # orphan a live trade).
            if sym not in self._symbol_states:
                self._symbol_states[sym] = self.STATES["IDLE"]
                self._symbol_signals[sym] = None
                self._symbol_event_start_ts[sym] = None
                self._symbol_atr_history[sym] = []
                self._symbol_pull_engine[sym] = None
                self._symbol_pull_entry[sym] = False
                self._symbol_pull_cache_ts[sym] = 0.0
            return

        if sym in self._symbol_states:
            # Configured symbol that is not pull-owned (e.g. US500): arm the
            # general engine once it becomes an active scanner candidate.
            if (self._symbol_pull_engine.get(sym) is None
                    and not getattr(cfg, "PULL_ENGINE_ENABLED", {}).get(sym, False)):
                eng = self._make_general_pull_engine(sym)
                self._symbol_pull_engine[sym] = eng
                self._symbol_pull_entry[sym] = False
                self._symbol_pull_cache_ts[sym] = 0.0
                if eng is not None:
                    self._scanner_pull_syms.add(sym)
                self.logger.info(
                    f"[SCANNER] Armed general pull engine for {sym} "
                    f"(pull {eng.pull_r}R trail {eng.trail_r}R hold {eng.max_hold_bars})"
                    if eng is not None else f"[SCANNER] No pull engine for {sym} (module missing)"
                )
            return

        self._symbol_states[sym] = self.STATES["IDLE"]
        self._symbol_signals[sym] = None
        self._symbol_event_start_ts[sym] = None
        self._symbol_atr_history[sym] = []
        self._symbol_pull_engine[sym] = self._make_general_pull_engine(sym)
        self._symbol_pull_entry[sym] = False
        self._symbol_pull_cache_ts[sym] = 0.0
        self._scanner_pull_syms.add(sym)
        eng = self._symbol_pull_engine.get(sym)
        self.logger.info(
            f"[SCANNER] Active candidate {sym}: pull scalper armed "
            f"(pull {eng.pull_r}R trail {eng.trail_r}R hold {eng.max_hold_bars})"
            if eng is not None else f"[SCANNER] Active candidate {sym}: no pull engine (module missing)"
        )

    def _current_candidates(self) -> List[str]:
        """Scanner top-K epics (canonicalized), or [] if the scanner is not ready."""
        if self.scanner is None or getattr(self.scanner, "client", None) is None:
            return []
        try:
            cands = self.scanner.scan() or []
        except Exception as e:
            self.logger.debug(f"[SCANNER] scan failed: {e}")
            return []
        self._candidate_map = {}
        out, seen = [], set()
        for c in cands:
            s = self._canon_sym(c.get("epic", ""))
            if not s or s in seen or self._is_blacklisted(s):
                continue
            seen.add(s)
            out.append(s)
            self._candidate_map[s] = c
        return out

    async def shutdown(self, grace_period: float = 25.0, close_positions: bool = True):
        self._winding_down = True
        self._shutdown_deadline = time.time() + grace_period
        self.logger.info(
            f"Shutdown requested, managing {self.position_manager.open_count} open position(s) "
            f"gracefully (timeout={grace_period:.0f}s)..."
        )
        while self._running and self.position_manager.open_count > 0:
            if time.time() > self._shutdown_deadline:
                if close_positions:
                    self.logger.warning(
                        f"Grace period expired, force-closing {self.position_manager.open_count} position(s)"
                    )
                    if hasattr(self, 'trade_executor'):
                        for pos_data in self.trade_executor.close_all_bot_positions():
                            self.position_manager.note_closed(pos_data, exit_reason="shutdown")
                else:
                    self.logger.warning(
                        f"Grace period expired, leaving {self.position_manager.open_count} position(s) "
                        f"for the new leader to recover"
                    )
                break
            await asyncio.sleep(0.5)
        self._running = False
        if hasattr(self, 'client'):
            self.client.shutdown()
        self.logger.info("Bot shutdown complete")

    def set_can_trade_callback(self, cb):
        self._can_trade_cb = cb

    async def run(self):
        self._running = True
        self.logger.info("Bot loop started")

        while self._running:
            try:
                await self._tick()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Bot loop error: {e}")
                await asyncio.sleep(5)

        self.logger.info("Bot loop ended")

    async def _tick(self):
        failover = getattr(self, '_failover', None)
        if failover and not await failover.can_manage():
            await asyncio.sleep(5)
            return

        active_syms = self._current_candidates()
        refresh_syms = list(dict.fromkeys(list(self.symbols) + active_syms))
        pnl_data = self.position_manager.refresh(symbols=refresh_syms)
        self.risk_manager.daily_pnl = pnl_data["daily_pnl"]

        if self.state == self.STATES["STOPPED"]:
            return

        now_t = time.time()
        if self._can_trade_cb and now_t - self._last_sub_check >= 30.0:
            self._last_sub_check = now_t
            can_trade = await self._can_trade_cb()
            if not can_trade and not self._winding_down:
                self._winding_down = True
                self.logger.warning("Subscription expired. Completing open trades, then stopping.")
            if self._winding_down and pnl_data["open_count"] == 0:
                reason = "shutdown" if self._shutdown_deadline is not None else "expired subscription"
                self.logger.warning(f"All trades closed. Stopping bot due to {reason}.")
                self.state = self.STATES["STOPPED"]
                self._running = False
                return

        if getattr(self, '_shutdown_deadline', None) and now_t > self._shutdown_deadline:
            if pnl_data["open_count"] > 0:
                self.logger.warning(
                    f"Shutdown deadline passed, force-closing {pnl_data['open_count']} position(s)"
                )
                for pos_data in self.trade_executor.close_all_bot_positions():
                    self.position_manager.note_closed(pos_data, exit_reason="shutdown")

        if not self.client.is_connected():
            now_t = time.time()
            backoff = min(2 ** self._reconnect_attempts, self._reconnect_backoff_max)
            if now_t - self._last_reconnect_time >= backoff:
                self.logger.warning(
                    f"Connection lost, reconnecting "
                    f"(attempt {self._reconnect_attempts + 1}, "
                    f"backoff={backoff:.0f}s)..."
                )
                if self._creds:
                    ok = await self.initialize_with_credentials(**self._creds)
                else:
                    ok = await self.initialize()
                if ok:
                    self._reconnect_attempts = 0
                    self.logger.info("Reconnected successfully")
                else:
                    self._reconnect_attempts += 1
                self._last_reconnect_time = now_t
            max_reconnect = getattr(cfg, "MAX_RECONNECT_ATTEMPTS", 10)
            if self._reconnect_attempts >= max_reconnect:
                self.logger.warning(
                    f"Connection lost after {self._reconnect_attempts} attempts — "
                    f"force-closing all positions to prevent unprotected exposure"
                )
                for sym in self.symbols:
                    closed = self.trade_executor.close_all_bot_positions(symbol=sym)
                    for pos_data in closed:
                        self.position_manager.note_closed(pos_data, exit_reason="reconnect_fail")
                self.state = self.STATES["STOPPED"]
                for sym in self.symbols:
                    self._symbol_states[sym] = self.STATES["STOPPED"]
                self._running = False
            self._write_state()
            return

        acct = self.client.get_account_info()
        balance = acct.get("balance", 0) if acct else 0

        if self.state not in (self.STATES["IN_TRADE"], self.STATES["WAITING_FOR_FUNDS"]):
            if balance < cfg.MIN_BALANCE:
                self.logger.warning(
                    f"Balance ${balance:.2f} below minimum "
                    f"${cfg.MIN_BALANCE:.2f}. Waiting for funds..."
                )
                self.state = self.STATES["WAITING_FOR_FUNDS"]
                for sym in self.symbols:
                    sym_positions = [p for p in pnl_data.get("positions", []) if p.get("_symbol_code") == sym]
                    if sym_positions:
                        self._symbol_states[sym] = self.STATES["IN_TRADE"]
                    else:
                        self._symbol_states[sym] = self.STATES["WAITING_FOR_FUNDS"]
                return

        await self._update_news_state()

        # ── Dynamic universe: ONE scanner ranks ALL tradable pairs by the
        # proven pull-into-H1 edge. The whole-board momentum scan is the cheap
        # first stage that narrows ~4000 markets to the top-K leaders; those
        # candidates are merged with the configured symbols and handed to the
        # universal pull scanner, which evaluates every pair the same way.
        # Eligible pairs are then traded by the per-symbol pull engines (single
        # source of truth for entry/exit state). ─────────────────────────────
        # 1. Whole-board momentum leaders (cheap pre-filter across all markets).
        scan_candidates = self._current_candidates()

        # 2. Universal pull-into-H1 scanner: ranks the FULL universe (configured
        #    symbols + whole-board candidates) by the proven strategy's edge.
        scan_rank = []
        try:
            scan_rank = await self.universal_pull_scanner.scan_all(
                self.client, symbols=scan_candidates,
            )
        except Exception as e:
            self.logger.debug(f"[UNIVERSAL PULL] scan failed: {e}")
            scan_rank = []

        # 3. Combine into one priority-ordered active universe: proven-strategy
        #    eligible symbols first, then any symbol still holding an open
        #    position (never orphan a live trade).
        active_syms: List[str] = []
        seen = set()

        def _add(sym: str):
            if not sym or sym in seen:
                return
            seen.add(sym)
            active_syms.append(sym)

        for r in scan_rank:
            _add(self._canon_sym(r.get("symbol", "")))
        for sym in list(self._symbol_states.keys()):
            sym_open = any(p.get("_symbol_code") == sym for p in pnl_data.get("positions", []))
            if sym_open:
                _add(sym)
        if not active_syms:
            active_syms = [s for s in self.symbols if not self._is_blacklisted(s)]
            self.logger.debug("[SCANNER] No candidates yet — falling back to configured symbols")

        # The proven PullPrevH1Scalper lives on the per-symbol engine. Feeding the
        # scanner's best signal back into that engine keeps one state machine.
        best_signal = scan_rank[0].get("entry_signal") if scan_rank else None
        if best_signal and best_signal.get("direction"):
            best_sym = self._canon_sym(scan_rank[0].get("symbol", ""))
            self.logger.info(
                f"[UNIVERSAL PULL] Best edge: {best_sym} "
                f"{best_signal['direction']} @ {best_signal.get('entry', 0):.2f} "
                f"(atp={scan_rank[0].get('atr', 0):.2f})"
            )

        for sym in active_syms:
            self._ensure_symbol_state(sym)
            await self._tick_symbol(sym, pnl_data, balance)

        if self.state not in (self.STATES["STOPPED"], self.STATES["WAITING_FOR_FUNDS"]):
            has_trade = any(s == self.STATES["IN_TRADE"] for s in self._symbol_states.values())
            self.state = self.STATES["IN_TRADE"] if has_trade else self.STATES["IDLE"]

        self._write_state()

    # ── Global margin guard ─────────────────────────────────────────
    # When a NEW entry is rejected for insufficient margin we stop firing further
    # NEW entries everywhere and wait until free margin recovers (an open trade
    # closing frees margin). Existing positions are always managed/exited.

    @staticmethod
    def _is_margin_error(text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False
        return any(k in t for k in (
            "margin", "not enough", "insufficient", "available funds",
            "rejected_margin", "free margin", "not sufficient",
        ))

    async def _tick_symbol(self, sym: str, pnl_data: Dict, balance: float = 0.0):
        state = self._symbol_states[sym]

        if state == self.STATES["STOPPED"]:
            return

        sym_open = any(
            p.get("_symbol_code") == sym
            for p in pnl_data.get("positions", [])
        )

        if sym_open and state not in (self.STATES["IN_TRADE"], self.STATES["STOPPED"]):
            self.logger.info(f"[{sym}] Recovered open position(s), resuming management")
            self._symbol_states[sym] = self.STATES["IN_TRADE"]
            state = self.STATES["IN_TRADE"]
            if not self._symbol_event_start_ts.get(sym):
                self._symbol_event_start_ts[sym] = time.time()
            if not self._symbol_signals.get(sym):
                sym_positions = [p for p in pnl_data.get("positions", []) if p.get("_symbol_code") == sym]
                if sym_positions:
                    pos = sym_positions[0]
                    direction = pos.get("type", "BUY")
                    entry_price = pos.get("price_open", 0)
                    broker_sl = pos.get("sl") or 0
                    broker_tp = pos.get("tp") or 0
                    atr_val = 0.0
                    try:
                        m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, 500)
                        if m1_data is not None and len(m1_data) >= 20:
                            atr_val = self._compute_atr_m5(m1_data, cfg.ATR_PERIOD)
                    except Exception:
                        pass
                    if atr_val <= 0:
                        atr_val = entry_price * 0.001
                    sl_dist = atr_val * getattr(cfg, "SL_ATR_MULTIPLIER", 1.0)
                    tp_dist = atr_val * getattr(cfg, "TP1_MULTIPLIER", 2.0)
                    if direction == "BUY":
                        sl = broker_sl if broker_sl > 0 else entry_price - sl_dist
                        tp = broker_tp if broker_tp > 0 else entry_price + tp_dist
                    else:
                        sl = broker_sl if broker_sl > 0 else entry_price + sl_dist
                        tp = broker_tp if broker_tp > 0 else entry_price - tp_dist
                    # Every recovered position is a pull trade — the pull engine
                    # owns its exit path (trailing giveback / max hold / daily guards).
                    pull_owns_rec = (bool(getattr(cfg, "PULL_ENGINE_ENABLED", {}).get(sym, False))
                                     or sym in self._scanner_pull_syms)
                    self._symbol_signals[sym] = {
                        "signal_type": "pull",
                        "direction": direction,
                        "sl": sl,
                        "tp1": tp,
                        "atr": atr_val,
                        "score": 0.5,
                        "recovered": True,
                    }
                    if pull_owns_rec:
                        pull_eng = self._symbol_pull_engine.get(sym)
                        if pull_eng is not None:
                            pull_eng.adopt_position(direction, entry_price, atr=atr_val)
                            self._symbol_pull_entry[sym] = True
                        else:
                            self.logger.warning(
                                f"[{sym}] Recovered position but pull engine missing — closing"
                            )
                            try:
                                await self.trade_executor.close_all_bot_positions(symbol=sym)
                            except Exception:
                                pass
                    self.logger.info(
                        f"[{sym}] Reconstructed signal: {direction} entry={entry_price:.2f} "
                        f"sl={sl:.2f} tp={tp if tp else 'n/a'} atr={atr_val:.2f} type=pull"
                    )

        info = self.client.get_symbol_info(sym)
        market_open = info is not None and info.get("market_status") == "TRADEABLE"

        if not market_open:
            if state == self.STATES["IN_TRADE"]:
                if sym_open:
                    self.logger.info(f"[{sym}] Market closed, managing open positions only")
                else:
                    self.logger.info(f"[{sym}] Market closed, no open positions, pausing")
                    self._symbol_states[sym] = self.STATES["MARKET_CLOSED"]
                    self._last_market_status_check = 0.0
            elif state != self.STATES["MARKET_CLOSED"]:
                self.logger.info(f"[{sym}] Market closed, pausing until reopen")
                self._symbol_states[sym] = self.STATES["MARKET_CLOSED"]
                self._last_market_status_check = 0.0
            return
        elif state == self.STATES["MARKET_CLOSED"]:
            info = self.client.get_symbol_info(sym)
            if info is not None and info.get("market_status") == "TRADEABLE":
                self.logger.info(f"[{sym}] Market reopened, resuming normal operation")
                self._symbol_states[sym] = self.STATES["IDLE"]
            return

        if self._winding_down:
            if state == self.STATES["IN_TRADE"]:
                await self._handle_in_trade(sym, pnl_data)
            return

        # Per-symbol balance gate: each symbol only trades once balance reaches
        # its minimum (e.g. US500 unlocks at $30). Open positions are never
        # orphaned — if a trade is open, it stays IN_TRADE and is managed.
        sym_min_balance = cfg.SYMBOL_MIN_BALANCE.get(sym, cfg.MIN_BALANCE)
        if balance < sym_min_balance:
            if state != self.STATES["WAITING_FOR_FUNDS"] and not sym_open:
                self.logger.info(
                    f"[{sym}] Balance ${balance:.2f} below ${sym_min_balance:.2f} "
                    f"threshold — waiting for funds"
                )
                self._symbol_states[sym] = self.STATES["WAITING_FOR_FUNDS"]
            if not sym_open:
                return

        state = self._symbol_states[sym]

        if state == self.STATES["IN_TRADE"]:
            await self._handle_in_trade(sym, pnl_data)
        elif state == self.STATES["WAITING_FOR_FUNDS"]:
            await self._handle_waiting_for_funds(sym)
        else:
            failover = getattr(self, '_failover', None)
            if failover and not await failover.can_trade():
                return
            await self._search_symbol(sym, pnl_data)

    async def _update_news_state(self):
        if self.news_state is None:
            return
        try:
            state = self.news_state.update()
            prev = getattr(self, '_prev_news_state', None)
            if state != prev:
                self.logger.info(f"[NEWS] State: {state}")
                self._prev_news_state = state
                info = self.news_state.get_state_info()
                event = info.get("last_event")
                event_title = event.get("title", "Unknown") if event else None
                if state == "PRE_NEWS" and event_title:
                    mins_until = int((event['datetime'] - datetime.now(timezone.utc)).total_seconds() / 60)
                    await self._notify(
                        "news_alert",
                        "News Alert",
                        f"{event_title} in ~{mins_until} min — bot continues trading normally",
                        {"state": state, "event": event_title},
                    )
                elif state == "SPIKE":
                    await self._notify(
                        "news_alert",
                        "Volatility Spike",
                        f"High volatility detected ({'event' if event_title else 'unknown cause'}) — bot continues trading normally",
                        {"state": state, "event": event_title},
                    )
                elif state == "POST_NEWS" and event_title:
                    await self._notify(
                        "news_alert",
                        "Post-News Window",
                        f"{event_title} passed — elevated volatility possible for ~60 min",
                        {"state": state, "event": event_title},
                    )
        except Exception as e:
            self.logger.debug(f"[NEWS] State update failed: {e}")

    @staticmethod
    def _compute_atr_m5(m1_data: pd.DataFrame, period: int = 14) -> float:
        """Resample M1 -> completed M5 bars and return the trailing ATR."""
        try:
            m1_idx = m1_data.set_index("time") if "time" in m1_data.columns else m1_data
            m5 = m1_idx.resample("5min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
            }).dropna()
            if len(m5) < period + 2:
                return 0.0
            atr = compute_atr(m5.iloc[:-1], period)
            val = float(atr.iloc[-1])
            return 0.0 if val != val else val
        except Exception:
            return 0.0

    @staticmethod
    def _compute_adx(highs, lows, closes, period=14):
        n = len(closes)
        if n < period + 1:
            return 0.0
        h, l, c = np.asarray(highs, dtype=float), np.asarray(lows, dtype=float), np.asarray(closes, dtype=float)
        tr = np.maximum(h[1:] - l[1:],
              np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        up = h[1:] - h[:-1]
        dn = l[:-1] - l[1:]
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        alpha = 1.0 / period
        atr_s = tr[0]
        pdm_s = plus_dm[0]
        mdm_s = minus_dm[0]
        dx_vals = []
        for i in range(1, len(tr)):
            atr_s = atr_s * (1 - alpha) + tr[i] * alpha
            pdm_s = pdm_s * (1 - alpha) + plus_dm[i] * alpha
            mdm_s = mdm_s * (1 - alpha) + minus_dm[i] * alpha
            if atr_s <= 0:
                dx_vals.append(0.0)
                continue
            pdi = (pdm_s / atr_s) * 100
            mdi = (mdm_s / atr_s) * 100
            s = pdi + mdi
            dx_vals.append(abs(pdi - mdi) / s * 100 if s > 0 else 0.0)
        if len(dx_vals) < period:
            return dx_vals[-1] if dx_vals else 0.0
        adx = sum(dx_vals[:period]) / period
        for dx in dx_vals[period:]:
            adx = adx * (1 - alpha) + dx * alpha
        return round(adx, 2)

    def _feed_m5_volatility(self, m1_data):
        if self.news_state is None or m1_data is None or len(m1_data) < 5:
            return
        try:
            if "time" not in m1_data.columns:
                return
            m1_idx = m1_data.set_index("time") if m1_data.index.name != "time" else m1_data
            m5 = m1_idx.resample("5min").agg({"high": "max", "low": "min", "close": "last"}).dropna()
            for _, row in m5.iterrows():
                self.news_state.feed_m5_bar(
                    float(row["high"]), float(row["low"]), float(row["close"])
                )
        except Exception as e:
            self.logger.debug(f"[NEWS] M5 feed failed: {e}")


    def _pull_refresh(self, sym: str, pull_eng) -> Optional[pd.DataFrame]:
        """Fetch fresh M1 bars for the pull engine (long history on a refresh
        timer, short tail every tick). Returns the recent frame or None."""
        now = time.time()
        refresh = getattr(cfg, "PULL_REFRESH_SEC", 60)
        try:
            if now - self._symbol_pull_cache_ts.get(sym, 0.0) >= refresh:
                from datetime import timedelta
                hist = int(getattr(cfg, "PULL_M1_HISTORY_BARS", 8000))
                window_min = hist + 600
                to_dt = datetime.utcnow()
                from_dt = to_dt - timedelta(seconds=window_min * 60)
                fetched = self.client.get_rates_range(sym, 1, from_dt, to_dt)
                self._symbol_pull_cache_ts[sym] = now
                if fetched is not None and len(fetched) > 0:
                    pull_eng.set_history(fetched)
        except Exception as e:
            self.logger.warning(f"[{sym}] Pull M1 history refresh failed: {e}")
        try:
            recent = self.client.get_rates(sym, 1, 30)
            if recent is not None and len(recent) > 0:
                return recent
        except Exception as e:
            if time.time() - self._last_pull_warn_ts > 600:
                self._last_pull_warn_ts = time.time()
                self.logger.warning(f"[{sym}] Pull M1 fetch failed: {e}")
        return None

    def _pull_entry_signal(self, sym: str, current_price: float, symbol_info: Dict) -> Optional[Dict]:
        """Pull scalper entry: feed fresh M1 bars; return a signal dict when the
        engine fires a pullback-into-H1-trend entry trigger (order still needs
        confirmation)."""
        eng = self._symbol_pull_engine.get(sym)
        if eng is None:
            return None
        recent = self._pull_refresh(sym, eng)
        now = time.time()
        if recent is None:
            return None
        if not eng.warm_up(now):
            return None
        try:
            action = eng.feed(recent, now_ts=now)
        except Exception as e:
            self.logger.warning(f"[{sym}] Pull eval failed: {e}")
            return None
        if action is None or action.get("type") != "enter":
            return None
        atr = eng.atr if eng.atr > 0 else (current_price * 0.001)
        pull_quality = min(1.0, (eng.pull_r / max(eng.atr, 1e-6)) * 10) if eng.atr > 0 else 0.5
        return {
            "direction": action["direction"],
            "score": round(pull_quality, 3),
            "price": current_price,
            "candle_open": current_price,
            "sl": None,
            "tp1": None,
            "signal_type": "pull",
            "atr": atr,
            "high_volatility": False,
            "bar_time": now,
            "gate_ok": True,
        }

    async def _search_symbol(self, sym: str, pnl_data: Dict):
        """Search for a pull-scalper entry signal on a single symbol."""
        pull_eng = self._symbol_pull_engine.get(sym)
        if pull_eng is None:
            return

        m1_data = None
        current_atr = 0.0
        high_vol = False
        symbol_info = None
        current_price = 0.0

        m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, 500)
        if m1_data is None or len(m1_data) < 10:
            return

        self._feed_m5_volatility(m1_data)

        symbol_info = self.client.get_symbol_info(sym)
        if symbol_info is None:
            return

        current_price = symbol_info.get("ask", 0)

        current_atr = self._compute_atr_m5(m1_data, cfg.ATR_PERIOD)

        signal = self._pull_entry_signal(sym, current_price, symbol_info)
        if signal is not None:
            self.logger.signal(
                f"[{sym}] Pull: {signal['direction']} score={signal['score']:.2f} "
                f"px={current_price:.2f} atr={signal.get('atr', 0):.4f}"
            )

        if signal and m1_data is not None and len(m1_data) >= 15:
            signal["adx_at_entry"] = self._compute_adx(
                m1_data["high"].values, m1_data["low"].values, m1_data["close"].values
            )

        if signal:
            sig_dir = signal.get('direction', 'UNKNOWN')
            sf_key = f"{sym}|{sig_dir}|{signal['score']:.2f}"
            sf_now = time.monotonic()
            if sf_key != self._last_signal_found_key or sf_now - self._last_signal_found_time >= 30:
                self._last_signal_found_key = sf_key
                self._last_signal_found_time = sf_now
                sig_type = "PULL"
                self.logger.signal(
                    f"[{sym}] Signal found: [{sig_type}] {sig_dir} | "
                    f"{sig_type} prediction at ${current_price:.2f} "
                    f"(score={signal['score']:.3f} SL={signal.get('sl', 'n/a')} TP={signal.get('tp1', 'n/a')})"
                )

        if signal:
            can_enter, reason = self.risk_manager.can_enter_trade(
                symbol_info, datetime.utcnow(), symbol=sym,
            )
            if not can_enter:
                bk_key = f"{sym}|blocked_{signal['direction']}|{reason}"
                bk_now = time.monotonic()
                if bk_key != self._last_signal_blocked_key or bk_now - self._last_signal_blocked_time >= 30:
                    self._last_signal_blocked_key = bk_key
                    self._last_signal_blocked_time = bk_now
                    self.logger.signal(
                        f"[{sym}] Signal {signal.get('direction', 'UNKNOWN')} "
                        f"blocked: {reason} | "
                        f"price={current_price:.2f} "
                        f"spread={symbol_info.get('spread', 0)}"
                    )
                return

            self._current_signal = signal
            self._symbol_signals[sym] = signal
            self.logger.signal(
                f"[{sym}] Signal triggered: [PULL] {signal.get('direction', 'UNKNOWN')} "
                f"score={signal['score']:.2f}"
            )
            await self._execute_entry(signal, symbol_info, symbol=sym)

    def _is_high_volatility(self, sym: str, current_atr: float) -> bool:
        if not getattr(cfg, "VOLATILITY_REGIME_ENABLED", True):
            return False
        hist = self._symbol_atr_history.get(sym, [])
        if len(hist) < 20:
            self._symbol_atr_history[sym] = hist + [current_atr]
            return False
        avg_atr = sum(hist[-50:]) / len(hist[-50:])
        self._symbol_atr_history[sym] = hist[-99:] + [current_atr]
        if avg_atr <= 0:
            return False
        return current_atr > avg_atr * cfg.VOLATILITY_ATR_MULT

    def _log_signal_diagnostic(self, reason: str, context: Dict):
        now = time.monotonic()
        key = f"{reason}|{context.get('direction')}|{context.get('price')}"
        if key == self._last_signal_diag_key and now - self._last_signal_diag_time < 30:
            return

        self._last_signal_diag_key = key
        self._last_signal_diag_time = now

        def fmt_num(value, decimals=2):
            if value is None:
                return "n/a"
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return str(value)

        self.logger.signal(
            f"No entry: {reason} | "
            f"direction={context.get('direction', 'n/a')} "
            f"price={fmt_num(context.get('price'))} "
            f"score={fmt_num(context.get('score'), 3)} "
            f"threshold={fmt_num(context.get('threshold', 0.75), 3)}"
        )

    async def _handle_in_trade(self, sym: str, pnl_data: Dict):
        sym_positions = [p for p in pnl_data.get("positions", []) if p.get("_symbol_code") == sym]

        if not sym_positions:
            event_ts = self._symbol_event_start_ts.get(sym)
            grace = getattr(cfg, "POSITION_GRACE_SECONDS", 30)
            if event_ts is not None and time.time() - event_ts < grace:
                self.logger.debug(f"[{sym}] Waiting for positions to appear (API delay grace period {grace}s)")
                return
            self.logger.info(f"[{sym}] Position gone (external close / margin call) — resetting state")
            eng = self._symbol_pull_engine.get(sym)
            if eng is not None and eng.in_position:
                eng.confirm_exit()
            self._symbol_states[sym] = self.STATES["IDLE"]
            self._symbol_event_start_ts[sym] = None
            self._symbol_pull_entry[sym] = False
            self._symbol_engine_failures[sym] = 0
            return

        acct = self.client.get_account_info()
        balance = acct.get("balance", 0) if acct else 0

        # ── Blacklisted pair: never hold — force-close any open position ──
        if self._is_blacklisted(sym):
            self.logger.warning(f"[{sym}] Blacklisted pair — force-closing open position(s)")
            closed = self.trade_executor.close_all_bot_positions(symbol=sym)
            for pos_data in closed:
                self.position_manager.note_closed(
                    pos_data, exit_reason="blacklist", score=0, balance=balance)
            if closed:
                pnl = sum(p.get("profit", 0) for p in closed)
                await self._notify(
                    "trade_close",
                    f"Trade Closed — {sym}",
                    f"Blacklisted pair closed | PnL: ${pnl:+.2f}",
                    {"symbol": sym, "exit_reason": "blacklist", "pnl": pnl},
                )
            self._symbol_states[sym] = self.STATES["IDLE"]
            self._symbol_event_start_ts[sym] = None
            self._symbol_pull_entry[sym] = False
            return

        pos = sym_positions[0]
        direction = pos.get("type", "BUY")
        entry_price = pos.get("price_open", 0)
        current_px = pos.get("price_current", pos.get("current_price", entry_price))
        event_start = self._symbol_event_start_ts.get(sym)

        pos_signal = self._symbol_signals.get(sym)
        minutes_held = (time.time() - event_start) / 60.0 if event_start else 0.0

        # ── Pull scalper exit — engine-driven (trailing giveback / max hold) ──
        # No timeout, no event loss. The engine emits a pending exit action on a
        # completed M5 bar; we close at market and confirm_exit() so it can
        # re-arm the next pullback pattern from the exit price.
        if self._symbol_pull_entry.get(sym, False):
            eng = self._symbol_pull_engine.get(sym)
            if eng is None:
                self.logger.warning(
                    f"[{sym}] Pull engine missing for open pull trade — force-closing to prevent orphan"
                )
                closed = self.trade_executor.close_all_bot_positions(symbol=sym)
                for pos_data in closed:
                    self.position_manager.note_closed(
                        pos_data, exit_reason="engine_missing",
                        score=0, balance=balance)
                if closed:
                    pnl = sum(p.get("profit", 0) for p in closed)
                self._symbol_states[sym] = self.STATES["IDLE"]
                self._symbol_event_start_ts[sym] = None
                self._symbol_pull_entry[sym] = False
                return
            else:
                recent = self._pull_refresh(sym, eng)
                action = None
                if recent is not None:
                    try:
                        action = eng.feed(recent, now_ts=time.time())
                        self._symbol_engine_failures[sym] = 0
                    except Exception as e:
                        self._symbol_engine_failures[sym] = self._symbol_engine_failures.get(sym, 0) + 1
                        self.logger.warning(
                            f"[{sym}] Pull exit eval failed ({self._symbol_engine_failures[sym]}x): {e}"
                        )
                        max_fails = getattr(cfg, "MAX_ENGINE_FAILURES", 5)
                        if self._symbol_engine_failures[sym] >= max_fails:
                            self.logger.warning(
                                f"[{sym}] Pull engine failed {max_fails}x consecutively — force-closing"
                            )
                            closed = self.trade_executor.close_all_bot_positions(symbol=sym)
                            for pos_data in closed:
                                self.position_manager.note_closed(
                                    pos_data, exit_reason="engine_failure",
                                    score=0, balance=balance)
                            if closed:
                                pnl = sum(p.get("profit", 0) for p in closed)
                            self._symbol_states[sym] = self.STATES["IDLE"]
                            self._symbol_event_start_ts[sym] = None
                            self._symbol_pull_entry[sym] = False
                            return
                if action is None or action.get("type") != "exit":
                    return
                exit_reason = action.get("reason", "pull_exit")
                self.logger.signal(
                    f"[{sym}] Pull exit: {exit_reason} | dir={direction} entry={entry_price:.2f} "
                    f"px={current_px:.2f} atr={eng.atr:.4f} held={minutes_held:.1f}m "
                    f"daily_r={eng.daily_r:+.3f}"
                )
                closed = self.trade_executor.close_all_bot_positions(symbol=sym)
                for pos_data in closed:
                    self.position_manager.note_closed(
                        pos_data, exit_reason=exit_reason,
                        score=pos_signal.get("score", 0) if pos_signal else 0,
                        balance=balance)
                if closed:
                    pnl = sum(p.get("profit", 0) for p in closed)
                    eng.confirm_exit()
                    self._symbol_states[sym] = self.STATES["IDLE"]
                    self._symbol_event_start_ts[sym] = None
                    self._symbol_pull_entry[sym] = False
                    await self._notify(
                        "trade_close",
                        f"Trade Closed — {sym}",
                        f"{direction} {sym} closed ({exit_reason}) | PnL: ${pnl:+.2f} | held {int(minutes_held)}m",
                        {"symbol": sym, "direction": direction, "exit_reason": exit_reason, "pnl": pnl, "held_minutes": minutes_held},
                    )
                else:
                    self.logger.warning(f"[{sym}] Pull exit close failed, retrying next tick")
                return

    async def _handle_waiting_for_funds(self, sym: str = None):
        info = self.client.get_account_info()
        min_balance = cfg.SYMBOL_MIN_BALANCE.get(sym, cfg.MIN_BALANCE) if sym else cfg.MIN_BALANCE
        if info and info["balance"] >= min_balance:
            self.logger.info(
                f"Funds detected: ${info['balance']:.2f} (need ${min_balance:.2f}). Starting bot."
            )
            self.state = self.STATES["IDLE"]
            if sym:
                self._symbol_states[sym] = self.STATES["IDLE"]

    async def _execute_entry(self, signal: Dict, symbol_info: Dict, symbol: str = None):
        sym = symbol or self.symbol
        self._symbol_states[sym] = self.STATES["ENTERING"]
        direction = signal["direction"]
        score = signal["score"]

        stale_positions = [p for p in self.position_manager.open_positions if p.get("_symbol_code") == sym]
        if stale_positions:
            self.logger.warning(f"[{sym}] Entry blocked: {len(stale_positions)} existing position(s) still open")
            self._symbol_states[sym] = self.STATES["IN_TRADE"]
            return

        fresh_info = self.client.get_symbol_info(sym)
        if fresh_info is None:
            self.logger.warning(f"[{sym}] Entry blocked: cannot fetch symbol info")
            self._symbol_states[sym] = self.STATES["IDLE"]
            return

        if fresh_info.get("market_status") != "TRADEABLE":
            self.logger.warning(f"[{sym}] Entry blocked: market {fresh_info.get('market_status')}, pausing")
            self._symbol_states[sym] = self.STATES["MARKET_CLOSED"]
            self._last_market_status_check = 0.0
            return

        # Friday close awareness: no NEW positions inside the last
        # FRIDAY_CLOSE_BLOCK_MIN minutes before the weekly close, so no fresh
        # trade gets stuck over the weekend. Existing positions are unaffected.
        mins_left = cfg.minutes_to_friday_close(sym)
        block_min = getattr(cfg, "FRIDAY_CLOSE_BLOCK_MIN", 60)
        if mins_left is not None and mins_left < block_min:
            self.logger.info(
                f"[{sym}] Entry blocked: Friday close in {mins_left:.0f}m "
                f"(block window {block_min}m) — no new positions"
            )
            self._symbol_states[sym] = self.STATES["IDLE"]
            return

        account = self.client.get_account_info()
        if account is None:
            self.logger.warning(f"[{sym}] Entry blocked: cannot fetch account info")
            self._symbol_states[sym] = self.STATES["IDLE"]
            return
        balance = account.get("balance", 0)
        min_balance = cfg.SYMBOL_MIN_BALANCE.get(sym, cfg.MIN_BALANCE)
        if balance < min_balance:
            self.logger.warning(
                f"[{sym}] Entry blocked: balance ${balance:.2f} below minimum ${min_balance:.2f}"
            )
            self._symbol_states[sym] = self.STATES["WAITING_FOR_FUNDS"]
            return

        risk_pct = getattr(cfg, "RISK_PCT", 0.03)
        risk_amount = balance * risk_pct
        atr_at_entry = signal.get("atr", 0)
        contract_size = fresh_info.get("contract_size", 1) or 1
        if atr_at_entry > 0 and contract_size > 0:
            lot = risk_amount / (atr_at_entry * contract_size)
        else:
            lot = cfg.LOT_SIZE
        vol_step = max(fresh_info.get("volume_step", cfg.LOT_STEP) or cfg.LOT_STEP, cfg.LOT_STEP)
        lot = round(lot / vol_step) * vol_step
        lot = max(fresh_info.get("volume_min", cfg.MIN_LOT), min(lot, fresh_info.get("volume_max", cfg.MAX_LOT)))
        max_trades = 1

        current_price = fresh_info.get("bid", fresh_info.get("price", 0)) if direction == "SELL" else fresh_info.get("ask", fresh_info.get("price", 0))
        if current_price <= 0:
            self.logger.warning(f"[{sym}] Entry blocked: current price is ${current_price} (no valid bid/ask)")
            self._symbol_states[sym] = self.STATES["IDLE"]
            return
        signal_price = symbol_info.get("bid", 0) if direction == "SELL" else symbol_info.get("ask", 0)
        if signal_price <= 0:
            self.logger.warning(f"[{sym}] Entry blocked: signal price is ${signal_price} (no valid bid/ask)")
            self._symbol_states[sym] = self.STATES["IDLE"]
            return
        drift_amount = abs(current_price - signal_price)
        drift_pct = drift_amount / signal_price * 100
        max_drift = cfg.SYMBOL_MAX_DRIFT.get(sym, 5.00)
        if drift_amount > max_drift:
            self.logger.warning(
                f"[{sym}] Entry blocked: price drifted ${drift_amount:.2f} "
                f"from signal price ${signal_price:.2f} to ${current_price:.2f} "
                f"(max drift ${max_drift:.2f})"
            )
            self._symbol_states[sym] = self.STATES["IDLE"]
            return

        free_margin = account.get("free_margin", 0)
        single_margin = self.client.estimate_margin(sym, lot, current_price)
        if free_margin < single_margin:
            contract_size = fresh_info.get("contract_size", 1)
            leverage = self.client.get_leverage_for_class(fresh_info.get("asset_class", "COMMODITIES"))
            notional_per_lot = contract_size * current_price
            max_lot_by_margin = free_margin * 0.8 / (notional_per_lot / max(1.0, leverage or 1.0))
            vol_step = max(fresh_info.get("volume_step", cfg.LOT_STEP) or cfg.LOT_STEP, cfg.LOT_STEP)
            max_lot_by_margin = round(max_lot_by_margin / vol_step) * vol_step
            max_lot_by_margin = max(fresh_info.get("volume_min", cfg.MIN_LOT), max_lot_by_margin)
            if max_lot_by_margin < lot:
                self.logger.info(
                    f"[{sym}] Reducing lot from {lot:.4f} to {max_lot_by_margin:.4f} "
                    f"(margin: ${free_margin:.2f} available, "
                    f"${single_margin:.2f} needed)"
                )
                lot = max_lot_by_margin
                single_margin = self.client.estimate_margin(sym, lot, current_price)
            if free_margin < single_margin:
                self.logger.warning(
                    f"[{sym}] Entry blocked: insufficient margin "
                    f"(est ${single_margin:.2f} needed, ${free_margin:.2f} available)"
                )
                self._symbol_states[sym] = self.STATES["IDLE"]
                return

        ml_conf = signal.get("ml_confidence", signal.get("score", 0))
        entry_lev = self.client.get_leverage_for_class(fresh_info.get("asset_class", "COMMODITIES"))

        self.logger.info(
            f"[{sym}] Entry: {direction} | score={score:.2f} | "
            f"balance=${balance:.2f} | lot={lot:.2f} | "
            f"max_trades={max_trades} | drift={drift_pct:.3f}% | "
            f"leverage=1:{entry_lev:.0f} (per-instrument) | "
            f"margin=${free_margin:.2f} | "
            f"risk={risk_pct*100:.1f}% (${risk_amount:.2f}) | "
            f"ML conf={ml_conf:.2f}"
        )

        any_opened = False
        for i in range(max_trades):
            if i > 0:
                acct = self.client.get_account_info()
                fm = acct.get("free_margin", 0) if acct else 0
                if fm < single_margin:
                    self.logger.info(
                        f"[{sym}] Margin exhausted after {i} trade(s) "
                        f"(${fm:.2f} < ${single_margin:.2f} needed for next)"
                    )
                    break

            ticket = await self.trade_executor.open_market(
                sym, direction, lot, expected_price=current_price
            )
            if ticket is not None:
                any_opened = True
                await asyncio.sleep(0.3)
            else:
                err_detail = ""
                err_attr = getattr(self.client, "last_order_error", None)
                if err_attr is not None:
                    try:
                        err_detail = err_attr() if callable(err_attr) else str(err_attr)
                    except Exception:
                        pass
                if "currently closed" in err_detail.lower():
                    self.logger.info(f"[{sym}] Market closed detected, pausing until reopen")
                    self._symbol_states[sym] = self.STATES["MARKET_CLOSED"]
                    return
                if self._is_margin_error(err_detail):
                    self.logger.warning(
                        f"[{sym}] Broker rejected for margin — will retry next tick"
                    )
                    return
                break

        fresh_acct = self.client.get_account_info()
        if fresh_acct:
            self.position_manager.refresh(symbols=list(self.symbols) + [sym])

        if any_opened:
            self._symbol_states[sym] = self.STATES["IN_TRADE"]
            self._symbol_event_start_ts[sym] = time.time()
            actual_fill = current_price
            fill_positions = [p for p in self.position_manager.open_positions if p.get("_symbol_code") == sym]
            if fill_positions:
                actual_fill = fill_positions[0].get("price_open", current_price)
            if signal.get("signal_type") == "pull":
                # Pull trade: the engine state machine owns all exits from here
                # (trailing giveback / max-hold), anchored at the ACTUAL fill.
                pull_eng = self._symbol_pull_engine.get(sym)
                if pull_eng is not None:
                    pull_eng.confirm_entry(actual_fill)
                    self._symbol_pull_entry[sym] = True
                else:
                    self._symbol_pull_entry[sym] = False
            self.logger.info(
                f"[{sym}] Entered {direction} with {max_trades} position(s) "
                f"(ML conf={ml_conf:.2f})"
            )
            await self._notify(
                "trade_open",
                f"Trade Opened — {sym}",
                f"{direction} {lot:.2f} lot {sym} @ ${current_price:.2f} | score={score:.2f}",
                {"symbol": sym, "direction": direction, "lot": lot, "price": current_price, "score": score},
            )
        else:
            self._symbol_states[sym] = self.STATES["IDLE"]
            if signal.get("signal_type") == "pull":
                pull_eng = self._symbol_pull_engine.get(sym)
                if pull_eng is not None:
                    pull_eng.cancel_entry()

    def _write_state(self):
        if not self._state_file:
            return
        now = time.time()
        if now - self._last_state_write < 1.0:
            return
        self._last_state_write = now
        account = self.client.get_account_info() or {"error": "No connection"}
        symbol_info = self.client.get_symbol_info(self.symbol) if self.client else {}
        if symbol_info:
            account["bid"] = symbol_info.get("bid", 0)
            account["ask"] = symbol_info.get("ask", 0)
        state = self.get_state_summary()
        payload = {
            "account": account,
            "bot": state,
            "logs": self.logger.logs[-50:],
            "timestamp": datetime.now().isoformat(),
        }
        try:
            tmp = self._state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self._state_file)
        except IOError:
            pass

    def get_state_summary(self) -> Dict:
        account = self.client.get_account_info()
        current_balance = account["balance"] if account else 0

        signal = self._current_signal or {}
        news = self.news_state.get_state_info() if self.news_state else {"state": "DISABLED"}

        return {
            "state": self.state,
            "is_demo": self.is_demo,
            "symbol": self.symbol,
            "symbol_states": dict(self._symbol_states),
            "magic": self.magic,
            "signal": signal,
            "news": news,
            "positions": self.position_manager.summary(),
            "risk": {},
            "starting_balance": round(self._starting_balance, 2) if self._starting_balance else None,
            "last_logs": self.logger.logs[-50:],
            "closed_trades": self.position_manager.closed_history[-100:],
        }

    def start(self):
        if self.state == self.STATES["STOPPED"]:
            self.risk_manager.reset_daily_pnl()
        self.state = self.STATES["IDLE"]
        for sym in self.symbols:
            self._symbol_states[sym] = self.STATES["IDLE"]
            self._symbol_event_start_ts[sym] = None
            self._symbol_signals[sym] = None
        self.logger.info("Bot manually started")

    def stop(self):
        self.state = self.STATES["STOPPED"]
        for sym in self.symbols:
            self._symbol_states[sym] = self.STATES["STOPPED"]
        self.logger.warning("Bot manually stopped")

    async def emergency_close(self):
        self.logger.warning("Emergency close triggered")
        total = 0
        for sym in self.symbols:
            closed = self.trade_executor.close_all_bot_positions(symbol=sym)
            for pos_data in closed:
                self.position_manager.note_closed(pos_data, exit_reason="emergency")
            total += len(closed)
            if closed:
                self._symbol_states[sym] = self.STATES["IDLE"]
                self._symbol_event_start_ts[sym] = None
        self.position_manager.refresh(symbols=self.symbols)
        await self._notify(
            "trade_close",
            "Emergency Close",
            f"Closed {total} position(s) manually",
            {"count": total, "reason": "emergency"},
        )
        self.state = self.STATES["IDLE"]
        return total

    def update_settings(self, settings: Dict):
        clamped = {}
        if "lot_multiplier" in settings:
            clamped["lot_multiplier"] = max(0.1, min(float(settings["lot_multiplier"]), 100.0))
            self._lot_multiplier_override = clamped["lot_multiplier"]
        if "max_spread_pips" in settings:
            clamped["max_spread_pips"] = max(1.0, min(float(settings["max_spread_pips"]), 500.0))
            self.risk_manager.max_spread = clamped["max_spread_pips"]
        self.logger.info(f"Bot settings updated: {clamped}")

    def login(self, server: str, account: str, password: str) -> Dict:
        self.logger.info(f"Logging into {server} account {account}...")
        ok = self.client.reconnect(server, account, password)
        if ok:
            info = self.client.get_account_info()
            if info:
                if self._starting_balance is None:
                    self._starting_balance = info["balance"]
                self.state = self.STATES["IDLE"]
                for sym in self.symbols:
                    self._symbol_states[sym] = self.STATES["IDLE"]
                self.logger.info(
                    f"Reconnected: {info['name']} | Balance: ${info['balance']:.2f} | "
                    f"Leverage (per instrument): {self._fmt_leverages(info.get('leverages'))}"
                )
                return {"success": True, "account": info}
        err = self.client.last_error()
        self.logger.error(f"Login failed: {err}")
        return {"success": False, "error": str(err)}

    def _load_accounts(self) -> List[Dict]:
        if not os.path.exists(self._accounts_file):
            return []
        try:
            with open(self._accounts_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_accounts(self, accounts: List[Dict]):
        with open(self._accounts_file, "w") as f:
            json.dump(accounts, f, indent=2)

    def _anonymize(self, acct: Dict) -> Dict:
        pw = acct.get("password", "")
        masked = pw[:1] + "****" + pw[-1:] if len(pw) > 4 else "****"
        return {k: v if k != "password" else masked for k, v in acct.items()}

    def list_accounts(self) -> List[Dict]:
        return [self._anonymize(a) for a in self._load_accounts()]

    def add_account(self, label: str, server: str, account: str, password: str) -> Dict:
        accounts = self._load_accounts()
        for a in accounts:
            if a["account"] == account and a["server"] == server:
                a["password"] = password
                a["label"] = label
                self._save_accounts(accounts)
                self.logger.info(f"Updated account {account}")
                return {"success": True, "message": "Account updated"}
        accounts.append({"label": label, "server": server, "account": account, "password": password})
        self._save_accounts(accounts)
        self.logger.info(f"Added account {account}")
        return {"success": True, "message": "Account added"}

    def remove_account(self, account_id: str) -> Dict:
        accounts = self._load_accounts()
        filtered = [a for a in accounts if a["account"] != account_id]
        if len(filtered) == len(accounts):
            return {"success": False, "message": "Account not found"}
        self._save_accounts(filtered)
        self.logger.info(f"Removed account {account_id}")
        return {"success": True, "message": "Account removed"}
