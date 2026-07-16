import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List

import config as cfg
from app.logger import BotLogger
from app.capital_client import CapitalClient
from app.signal_engine import SignalEngine
from app.risk_manager import RiskManager, EquityScaler
from app.trade_executor import TradeExecutor
from app.position_manager import PositionManager
from app.economic_calendar import EconomicCalendar
from app.news_state_machine import NewsStateMachine

try:
    from app.asp_predictor import ASPPredictor
    _HAS_ASP = True
except ImportError:
    ASPPredictor = None
    _HAS_ASP = False

try:
    from app.swing_quality_predictor import SwingQualityPredictor
    _HAS_SQ = True
except ImportError:
    SwingQualityPredictor = None
    _HAS_SQ = False


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
        self._symbol_engines: Dict[str, SignalEngine] = {}
        self._symbol_states: Dict[str, str] = {}
        self._symbol_signals: Dict[str, Optional[Dict]] = {}
        self._symbol_event_start_ts: Dict[str, Optional[float]] = {}
        self._symbol_exit_confirms: Dict[str, int] = {}

        # Load per-symbol models and signal engines
        for sym in self.symbols:
            asp_pred = None
            sq_pred = None

            # ASP model
            if _HAS_ASP and cfg.ASP_ENABLED:
                asp_path = cfg.ASP_MODEL_PATHS.get(sym, cfg.ASP_MODEL_PATHS.get("XAUUSD"))
                feat_path = cfg.ASP_FEATURE_PATHS.get(sym, cfg.ASP_FEATURE_PATHS.get("XAUUSD"))
                try:
                    asp_pred = ASPPredictor(model_path=asp_path, feature_path=feat_path)
                    if asp_pred.ready:
                        self.logger.info(f"[{sym}] ASP model loaded from {asp_path}")
                    else:
                        self.logger.warning(f"[{sym}] ASP model failed to initialize")
                        asp_pred = None
                except Exception as e:
                    self.logger.warning(f"[{sym}] Failed to load ASP model: {e}")

            # Swing quality model
            if _HAS_SQ:
                sq_path = cfg.SWING_QUALITY_MODEL_PATHS.get(sym, cfg.SWING_QUALITY_MODEL_PATHS.get("XAUUSD"))
                try:
                    sq_pred = SwingQualityPredictor(model_path=sq_path)
                    if sq_pred.ready:
                        self.logger.info(f"[{sym}] Swing quality model loaded")
                    else:
                        sq_pred = None
                except Exception as e:
                    self.logger.warning(f"[{sym}] Failed to load swing quality model: {e}")

            engine = SignalEngine(
                asp_predictor=asp_pred,
                swing_quality_predictor=sq_pred,
                logger=self.logger,
            )
            self._symbol_engines[sym] = engine
            self._symbol_states[sym] = self.STATES["IDLE"]
            self._symbol_signals[sym] = None
            self._symbol_event_start_ts[sym] = None
            self._symbol_exit_confirms[sym] = 0

        # Legacy single-symbol references (use first symbol for backward compat)
        first_sym = self.symbols[0] if self.symbols else cfg.SYMBOL
        self.signal_engine = self._symbol_engines.get(first_sym, SignalEngine(logger=self.logger))
        self._asp_predictor = getattr(self.signal_engine, '_asp_predictor', None)
        self._swing_quality_predictor = getattr(self.signal_engine, '_swing_quality', None)

        self.risk_manager = RiskManager()
        self.scaler = EquityScaler()

        self.state: str = self.STATES["IDLE"]
        self.symbol: str = first_sym
        self.magic: int = cfg.MAGIC_NUMBER
        self._running = False

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


    async def _notify(self, ntype: str, title: str, message: str, data: Optional[Dict] = None):
        try:
            from app.subscription import create_notification
            await create_notification(
                getattr(self, '_account_id', None) or "unknown",
                ntype, title, message, data,
            )
        except Exception:
            pass

    async def initialize(self) -> bool:
        self.logger.info(f"Initializing {self.symbol} scalping bot (ASP-only mode)...")
        self.logger.info(f"Config: ASP_TIMEOUT_BARS={cfg.SYMBOL_ASP_TIMEOUT_BARS} LOT_MULTIPLIER={cfg.LOT_MULTIPLIER}")

        self.logger.info("Broker: Capital.com (REST API)")
        self.client = CapitalClient()
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
                self.scaler.initialize(info["balance"])
                self.logger.info(
                    f"Connected: {info['name']} | "
                    f"Balance: ${info['balance']:.2f} | "
                    f"Leverage: 1:{info['leverage']}"
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
        self.client = CapitalClient()
        self.logger.info("Connecting to Capital.com...")
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
                self.scaler.initialize(info["balance"])
                self.logger.info(f"Account connected ✓")
                self.logger.info(f"Balance: ${info['balance']:.2f} | Leverage: 1:{info['leverage']}")
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
                return True
            self.state = self.STATES["IDLE"]
            for sym in self.symbols:
                self._symbol_states[sym] = self.STATES["IDLE"]
            self._write_state()
            return True
        else:
            err = self.client.last_error()
            self.logger.error(f"Authentication failed: {err}")
            self.state = self.STATES["STOPPED"]
            for sym in self.symbols:
                self._symbol_states[sym] = self.STATES["STOPPED"]
            return False

    async def shutdown(self, grace_period: float = 25.0):
        self._winding_down = True
        self._shutdown_deadline = time.time() + grace_period
        self.logger.info(
            f"Shutdown requested, managing {self.position_manager.open_count} open position(s) "
            f"gracefully (timeout={grace_period:.0f}s)..."
        )
        while self._running and self.position_manager.open_count > 0:
            if time.time() > self._shutdown_deadline:
                self.logger.warning(
                    f"Grace period expired, force-closing {self.position_manager.open_count} position(s)"
                )
                if hasattr(self, 'trade_executor'):
                    for pos_data in self.trade_executor.close_all_bot_positions():
                        self.position_manager.note_closed(pos_data)
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
        pnl_data = self.position_manager.refresh(symbols=self.symbols)
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
                        self.position_manager.note_closed(pos_data)

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
            self._write_state()
            return

        if self.state not in (self.STATES["IN_TRADE"], self.STATES["WAITING_FOR_FUNDS"]):
            info = self.client.get_account_info()
            if info and info["balance"] < cfg.MIN_BALANCE:
                self.logger.warning(
                    f"Balance ${info['balance']:.2f} below minimum "
                    f"${cfg.MIN_BALANCE:.2f}. Waiting for funds..."
                )
                self.state = self.STATES["WAITING_FOR_FUNDS"]
                return

        await self._update_news_state()

        for sym in self.symbols:
            await self._tick_symbol(sym, pnl_data)

        self._write_state()

    async def _tick_symbol(self, sym: str, pnl_data: Dict):
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
            if not self._symbol_event_start_ts.get(sym):
                self._symbol_event_start_ts[sym] = time.time()

        market_open = cfg.is_market_open_for_symbol(sym)
        if market_open and state != self.STATES["MARKET_CLOSED"]:
            info = self.client.get_symbol_info(sym)
            if info is not None:
                market_open = info.get("market_status") == "TRADEABLE"

        if not market_open:
            if state == self.STATES["IN_TRADE"]:
                self.logger.info(f"[{sym}] Market closed, managing open positions only")
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
            return

        state = self._symbol_states[sym]
        if state == self.STATES["IN_TRADE"]:
            await self._handle_in_trade(sym, pnl_data)
        elif state == self.STATES["WAITING_FOR_FUNDS"]:
            await self._handle_waiting_for_funds(sym)
        else:
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

    async def _handle_search(self, pnl_data: Dict):
        if self._winding_down:
            return

        for sym in self.symbols:
            await self._search_symbol(sym, pnl_data)

    async def _search_symbol(self, sym: str, pnl_data: Dict):
        """Search for entry signal on a single symbol."""
        engine = self._symbol_engines.get(sym)
        if engine is None:
            return

        m1_bars = getattr(cfg, "ASP_M1_HISTORY_BARS", 300)
        m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, m1_bars)
        if m1_data is None or len(m1_data) < 10:
            return

        self._feed_m5_volatility(m1_data)

        symbol_info = self.client.get_symbol_info(sym)
        if symbol_info is None:
            return

        current_price = symbol_info.get("ask", 0)

        signal = None
        asp_pred = getattr(engine, '_asp_predictor', None)
        if asp_pred and asp_pred.ready and cfg.ASP_ENABLED:
            h1_for_asp = None
            try:
                h1_for_asp = self.client.get_rates(sym, cfg.BIAS_TIMEFRAME, 96)
            except Exception:
                pass
            signal = engine.evaluate_asp_entry(
                m1_data, current_price,
                h1_data=h1_for_asp,
            )

        if signal:
            sig_dir = signal.get('direction', 'UNKNOWN')
            sf_key = f"{sym}|{sig_dir}|{signal['score']:.2f}"
            sf_now = time.monotonic()
            if sf_key != self._last_signal_found_key or sf_now - self._last_signal_found_time >= 30:
                self._last_signal_found_key = sf_key
                self._last_signal_found_time = sf_now
                self.logger.signal(
                    f"[{sym}] Signal found: [ASP] {sig_dir} | "
                    f"ASP swing prediction at ${current_price:.2f} "
                    f"(score={signal['score']:.3f} SL={signal.get('sl', 'n/a')} TP={signal.get('tp1', 'n/a')})"
                )

        if signal:
            can_enter, reason = self.risk_manager.can_enter_trade(
                symbol_info, datetime.utcnow()
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
                f"[{sym}] Signal triggered: [ASP] {signal.get('direction', 'UNKNOWN')} "
                f"score={signal['score']:.2f}"
            )
            await self._execute_entry(signal, symbol_info, symbol=sym)

    def _current_session(self) -> str:
        h = datetime.utcnow().hour
        if 0 <= h < 8:
            return "ASIA"
        if 8 <= h < 17:
            return "LONDON"
        if 12 <= h < 22:
            return "NEW_YORK"
        return "OUTSIDE"

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
            if event_ts is not None and time.time() - event_ts < 10:
                self.logger.debug(f"[{sym}] Waiting for positions to appear (API delay grace period)")
                return
            self._symbol_states[sym] = self.STATES["IDLE"]
            self._symbol_event_start_ts[sym] = None
            self._symbol_exit_confirms[sym] = 0
            return

        acct = self.client.get_account_info()
        balance = acct.get("balance", 0) if acct else 0

        event_pnl = sum(p.get("profit", 0) for p in sym_positions)
        event_ok, event_msg = self.risk_manager.check_event_loss(event_pnl)
        if not event_ok:
            self.logger.warning(f"[{sym}] Event stop: {event_msg}")
            closed = self.trade_executor.close_all_bot_positions(symbol=sym)
            for pos_data in closed:
                self.position_manager.note_closed(pos_data)
            if closed:
                self._symbol_states[sym] = self.STATES["IDLE"]
                self._symbol_event_start_ts[sym] = None
                self._symbol_exit_confirms[sym] = 0
            else:
                self.logger.warning(f"[{sym}] Event stop close failed, retrying next tick")
            return

        pos = sym_positions[0]
        direction = pos.get("type", "BUY")
        entry_price = pos.get("price_open", 0)
        current_px = pos.get("current_price", entry_price)
        event_start = self._symbol_event_start_ts.get(sym)
        exit_interval = cfg.EXIT_CHECK_INTERVAL or 300
        bars_held = max(0, int((time.time() - event_start) / exit_interval)) if event_start else 0
        timeout_bars = cfg.SYMBOL_ASP_TIMEOUT_BARS.get(sym, 24)

        pos_signal = self._symbol_signals.get(sym)
        is_asp = pos_signal and pos_signal.get("asp_model")

        if is_asp:
            asp_sl = pos_signal.get("sl")
            asp_tp = pos_signal.get("tp1")
            atr_val = pos_signal.get("atr_value", 0)

            if getattr(cfg, "ASP_TRAILING_ENABLED", False) and atr_val > 0:
                best_key = f"_best_price_{sym}"
                if not hasattr(self, best_key):
                    setattr(self, best_key, entry_price)
                best = getattr(self, best_key)

                if direction == "BUY":
                    best = max(best, current_px)
                    trigger = entry_price + atr_val * cfg.ASP_TRAILING_TRIGGER_ATR
                    if best >= trigger:
                        trail_sl = best - atr_val * cfg.ASP_TRAILING_RETRACE_ATR
                        if trail_sl > asp_sl:
                            asp_sl = trail_sl
                            pos_signal["sl"] = asp_sl
                else:
                    best = min(best, current_px)
                    trigger = entry_price - atr_val * cfg.ASP_TRAILING_TRIGGER_ATR
                    if best <= trigger:
                        trail_sl = best + atr_val * cfg.ASP_TRAILING_RETRACE_ATR
                        if trail_sl < asp_sl:
                            asp_sl = trail_sl
                            pos_signal["sl"] = asp_sl
                setattr(self, best_key, best)

            should_exit = False
            exit_reason = ""

            if direction == "BUY":
                if asp_sl and current_px <= asp_sl:
                    should_exit = True
                    exit_reason = "asp_trail_sl" if getattr(cfg, "ASP_TRAILING_ENABLED", False) else "asp_sl_hit"
                elif asp_tp and current_px >= asp_tp:
                    should_exit = True
                    exit_reason = "asp_tp_hit"
            else:
                if asp_sl and current_px >= asp_sl:
                    should_exit = True
                    exit_reason = "asp_trail_sl" if getattr(cfg, "ASP_TRAILING_ENABLED", False) else "asp_sl_hit"
                elif asp_tp and current_px <= asp_tp:
                    should_exit = True
                    exit_reason = "asp_tp_hit"

            if should_exit:
                self.logger.signal(f"[{sym}] ASP exit: {exit_reason} | dir={direction} entry={entry_price:.2f} px={current_px:.2f} sl={asp_sl:.2f}")
                closed = self.trade_executor.close_all_bot_positions(symbol=sym)
                for pos_data in closed:
                    self.position_manager.note_closed(pos_data)
                if closed:
                    self._symbol_states[sym] = self.STATES["IDLE"]
                    self._symbol_event_start_ts[sym] = None
                    self._symbol_exit_confirms[sym] = 0
                else:
                    self.logger.warning(f"[{sym}] ASP exit close failed, retrying next tick")
                return

        if bars_held >= timeout_bars:
            engine = self._symbol_engines.get(sym)
            m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, 20)
            fresh_signal = None
            if engine and m1_data is not None and len(m1_data) >= 10:
                fresh_signal = engine.evaluate_asp_entry(m1_data, current_px)

            fresh_dir = fresh_signal.get("direction") if fresh_signal else None
            if fresh_dir == direction:
                self._symbol_exit_confirms[sym] = self._symbol_exit_confirms.get(sym, 0) + 1
                confirms = self._symbol_exit_confirms[sym]
                self.logger.signal(
                    f"[{sym}] Timeout hold: same-direction re-scan #{confirms} "
                    f"(dir={direction} held {bars_held}bars, need ≥2 confirms)"
                )
                if confirms >= 2:
                    self._symbol_event_start_ts[sym] = time.time()
                    self._symbol_exit_confirms[sym] = 0
                    self.logger.signal(f"[{sym}] Timeout reset: model confirms move still valid, extending")
            else:
                self._symbol_reversal_confirms[sym] = self._symbol_reversal_confirms.get(sym, 0) + 1
                rev_confirms = self._symbol_reversal_confirms[sym]
                reason = f"asp_timeout_{bars_held}bars_reversal" if fresh_dir else f"asp_timeout_{bars_held}bars_no_signal"
                self.logger.signal(
                    f"[{sym}] Timeout reversal re-scan #{rev_confirms}: {reason} "
                    f"(dir={direction} held {bars_held}bars, need ≥2 reversals to exit)"
                )
                if rev_confirms < 2:
                    self._symbol_exit_confirms[sym] = 0
                    return
                self._symbol_exit_confirms[sym] = 0
                self._symbol_reversal_confirms[sym] = 0
                self.logger.signal(f"[{sym}] Timeout exit: {reason} | dir={direction} entry={entry_price:.2f} px={current_px:.2f}")
                closed = self.trade_executor.close_all_bot_positions(symbol=sym)
                for pos_data in closed:
                    self.position_manager.note_closed(pos_data)
                if closed:
                    self._symbol_states[sym] = self.STATES["IDLE"]
                    self._symbol_event_start_ts[sym] = None
                else:
                    self.logger.warning(f"[{sym}] Timeout exit close failed, retrying next tick")
                return

    async def _handle_waiting_for_funds(self, sym: str = None):
        info = self.client.get_account_info()
        if info and info["balance"] >= cfg.MIN_BALANCE:
            self.scaler.update_peak(info["balance"])
            if self.scaler.starting_balance is None:
                self.scaler.initialize(info["balance"])
            self.logger.info(
                f"Funds detected: ${info['balance']:.2f}. Starting bot."
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

        account = self.client.get_account_info()
        if account is None:
            self.logger.warning(f"[{sym}] Entry blocked: cannot fetch account info")
            self._symbol_states[sym] = self.STATES["IDLE"]
            return
        balance = account.get("balance", 0)
        if balance < cfg.MIN_BALANCE:
            self.logger.warning(
                f"[{sym}] Entry blocked: balance ${balance:.2f} below minimum ${cfg.MIN_BALANCE:.2f}"
            )
            self._symbol_states[sym] = self.STATES["WAITING_FOR_FUNDS"]
            return

        self.scaler.update_peak(balance)

        lot = self.scaler.get_lot(balance, symbol=sym)
        vol_step = fresh_info.get("volume_step", cfg.LOT_STEP)
        lot = round(lot / vol_step) * vol_step
        lot = max(fresh_info.get("volume_min", cfg.MIN_LOT), min(lot, fresh_info.get("volume_max", cfg.MAX_LOT)))
        max_trades = 1

        current_price = fresh_info.get("bid", fresh_info.get("price", 0)) if direction == "SELL" else fresh_info.get("ask", fresh_info.get("price", 0))
        signal_price = symbol_info.get("bid", 0) if direction == "SELL" else symbol_info.get("ask", 0)
        point = fresh_info.get("point", 0.01)
        drift_amount = abs(current_price - signal_price)
        drift_pct = drift_amount / signal_price * 100
        sym_max_spread = cfg.SYMBOL_MAX_SPREAD.get(sym, cfg.MAX_SPREAD_PIPS)
        max_drift = sym_max_spread * point
        if drift_amount > max_drift:
            self.logger.warning(
                f"[{sym}] Entry blocked: price drifted ${drift_amount:.2f} "
                f"from signal price ${signal_price:.2f} to ${current_price:.2f} "
                f"(max drift ${max_drift:.2f})"
            )
            self._symbol_states[sym] = self.STATES["IDLE"]
            return

        margin_rate = fresh_info.get("margin_rate", 0.05)
        free_margin = account.get("free_margin", 0)
        single_margin = lot * current_price * margin_rate * 1
        if free_margin < single_margin:
            max_lot_by_margin = free_margin * 0.9 / (current_price * margin_rate * 1)
            vol_step = fresh_info.get("volume_step", cfg.LOT_STEP)
            max_lot_by_margin = round(max_lot_by_margin / vol_step) * vol_step
            max_lot_by_margin = max(fresh_info.get("volume_min", cfg.MIN_LOT), max_lot_by_margin)
            if max_lot_by_margin < lot:
                self.logger.info(
                    f"[{sym}] Reducing lot from {lot:.2f} to {max_lot_by_margin:.2f} "
                    f"(margin: ${free_margin:.2f} available, "
                    f"${single_margin:.2f} needed)"
                )
                lot = max_lot_by_margin
                single_margin = lot * current_price * margin_rate * 1
            if free_margin < single_margin:
                self.logger.warning(
                    f"[{sym}] Entry blocked: insufficient margin "
                    f"(est ${single_margin:.2f} needed, ${free_margin:.2f} available)"
                )
                self._symbol_states[sym] = self.STATES["IDLE"]
                return

        ml_conf = signal.get("ml_confidence", signal.get("score", 0))

        self.logger.info(
            f"[{sym}] Entry: {direction} | score={score:.2f} | "
            f"ASP={signal.get('asp_model', False)} | "
            f"balance=${balance:.2f} | lot={lot:.2f} | "
            f"max_trades={max_trades} | drift={drift_pct:.3f}% | "
            f"margin=${free_margin:.2f} | "
            f"tier={self.scaler._tier(balance)} "
            f"({self.scaler.growth_pct(balance):.1f}% growth) | "
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
                sym, direction, lot
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
                break

        fresh_acct = self.client.get_account_info()
        if fresh_acct:
            self.scaler.update_peak(fresh_acct["balance"])
        self.position_manager.refresh()

        if any_opened:
            self._symbol_states[sym] = self.STATES["IN_TRADE"]
            self._symbol_event_start_ts[sym] = time.time()
            self.logger.info(
                f"[{sym}] Entered {direction} with {max_trades} position(s) "
                f"(ML conf={ml_conf:.2f})"
            )
        else:
            self._symbol_states[sym] = self.STATES["IDLE"]

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
            "symbol": self.symbol,
            "symbol_states": dict(self._symbol_states),
            "magic": self.magic,
            "signal": signal,
            "news": news,
            "positions": self.position_manager.summary(),
            "risk": {},
            "scaler": self.scaler.summary(current_balance) if self.scaler.starting_balance else None,
            "last_logs": self.logger.logs[-50:],
            "closed_trades": self.position_manager.closed_history[-100:],
        }

    def start(self):
        if self.state == self.STATES["STOPPED"]:
            self.risk_manager.record_daily_pnl_reset()
        self.state = self.STATES["IDLE"]
        for sym in self.symbols:
            self._symbol_states[sym] = self.STATES["IDLE"]
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
                self.position_manager.note_closed(pos_data)
            total += len(closed)
            if closed:
                self._symbol_states[sym] = self.STATES["IDLE"]
                self._symbol_event_start_ts[sym] = None
                self._symbol_exit_confirms[sym] = 0
        self.position_manager.refresh()
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
        if "allowed_sessions" in settings:
            sessions_raw = settings["allowed_sessions"]
            if isinstance(sessions_raw, list):
                sessions_str = ",".join(str(s) for s in sessions_raw)
            else:
                sessions_str = str(sessions_raw)
            self.risk_manager.allowed_sessions = [s.strip().upper() for s in sessions_str.split(",") if s.strip()]
        self.logger.info(f"Bot settings updated: {clamped}")

    def login(self, server: str, account: str, password: str) -> Dict:
        self.logger.info(f"Logging into {server} account {account}...")
        ok = self.client.reconnect(server, account, password)
        if ok:
            info = self.client.get_account_info()
            if info:
                self.scaler.initialize(info["balance"])
                self.state = self.STATES["IDLE"]
                for sym in self.symbols:
                    self._symbol_states[sym] = self.STATES["IDLE"]
                self.logger.info(f"Reconnected: {info['name']} | Balance: ${info['balance']:.2f} | Leverage: 1:{info['leverage']}")
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
