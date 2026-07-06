import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import config as cfg
from app.logger import BotLogger
from app.capital_client import CapitalClient
from app.bias_engine import BiasEngine
from app.signal_engine import SignalEngine
from app.risk_manager import RiskManager, EquityScaler
from app.trade_executor import TradeExecutor
from app.position_manager import PositionManager
from app.gemini_advisor import GeminiAdvisor
from app.meta_strategy import MetaStrategy
from app.adaptive_confirmation import AdaptiveConfirmation

try:
    from app.direction_predictor import DirectionPredictor, SLTPredictor
    _HAS_ML = True
except ImportError:
    DirectionPredictor = None
    SLTPredictor = None
    _HAS_ML = False


class Bot:
    STATES = {
        "IDLE": "IDLE",
        "BIAS_ANALYSIS": "BIAS_ANALYSIS",
        "AWAITING_SIGNAL": "AWAITING_SIGNAL",
        "ENTERING": "ENTERING",
        "IN_TRADE": "IN_TRADE",
        "EXITING": "EXITING",
        "COOLDOWN": "COOLDOWN",
        "STOPPED": "STOPPED",
        "WAITING_FOR_FUNDS": "WAITING_FOR_FUNDS",
        "MARKET_CLOSED": "MARKET_CLOSED",
    }

    def __init__(self, logger: Optional[BotLogger] = None):
        self.logger = logger or BotLogger()
        self.client: object = None
        self.bias_engine = BiasEngine()
        self._direction_predictor = None
        self._slt_predictor = None
        if _HAS_ML:
            try:
                if os.path.exists(cfg.ML_MODEL_PATH):
                    self._direction_predictor = DirectionPredictor.load(cfg.ML_MODEL_PATH)
                    self.logger.info(f"Direction model loaded from {cfg.ML_MODEL_PATH}")
            except Exception as e:
                self.logger.warning(f"Failed to load direction model: {e}")
            try:
                buy_path = cfg.ML_BUY_MODEL_PATH
                sell_path = cfg.ML_SELL_MODEL_PATH
                if os.path.exists(buy_path) and os.path.exists(sell_path):
                    self._slt_predictor = SLTPredictor(buy_model_path=buy_path, sell_model_path=sell_path)
                    self.logger.info(f"SL/TP models loaded (buy={buy_path}, sell={sell_path})")
            except Exception as e:
                self.logger.warning(f"Failed to load SL/TP models: {e}")
        self.signal_engine = SignalEngine(
            direction_predictor=self._direction_predictor,
            slt_predictor=self._slt_predictor,
            logger=self.logger,
        )
        ml_status = []
        if self._direction_predictor:
            ml_status.append("Direction model loaded")
        if self._slt_predictor:
            ml_status.append("SL/TP models loaded")
        if ml_status:
            self.logger.info(f"[ML] Active: {', '.join(ml_status)} | "
                             f"confidence_threshold={cfg.ML_CONFIDENCE_THRESHOLD} "
                             f"override_max=20")
        else:
            self.logger.info("[ML] Not available")
        self.risk_manager = RiskManager()
        self.scaler = EquityScaler()
        self.meta = MetaStrategy() if cfg.META_ENABLED else None
        self.gemini_advisor = GeminiAdvisor()
        if self.gemini_advisor.enabled:
            self.logger.info("Gemini advisor enabled")
        self.adaptive_conf = AdaptiveConfirmation()

        self.state: str = self.STATES["IDLE"]
        self.symbol: str = cfg.SYMBOL
        self.magic: int = cfg.MAGIC_NUMBER
        self._running = False
        self._cooldown_until: Optional[datetime] = None

        self._current_signal: Optional[Dict] = None
        self._bias_summary: Dict = {}
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
        self._shutdown_deadline = None
        self._last_sub_check = 0.0
        self._ml_heartbeat_ticks = 0
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
        self.logger.info(f"Initializing {self.symbol} scalping bot...")
        self.logger.info(f"Config: EXIT_THRESHOLD_TIGHT={cfg.EXIT_THRESHOLD_TIGHT} LOT_MULTIPLIER={cfg.LOT_MULTIPLIER} ENTRY_THRESHOLD={cfg.SIGNAL_ENTRY_THRESHOLD}")

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
                else:
                    self.state = self.STATES["IDLE"]
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
        self.logger.info("Connecting to broker...")
        self.client = CapitalClient()
        self.logger.info("Establishing secure connection...")
        self.logger.info("Authenticating client...")
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
                else:
                    self.state = self.STATES["IDLE"]
                self._write_state()
                return True
            self.state = self.STATES["IDLE"]
            self._write_state()
            return True
        else:
            err = self.client.last_error()
            self.logger.error(f"Authentication failed: {err}")
            self.state = self.STATES["STOPPED"]
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
        pnl_data = self.position_manager.refresh()
        self.risk_manager.daily_pnl = pnl_data["daily_pnl"]

        if pnl_data["open_count"] > 0 and self.state not in (
            self.STATES["IN_TRADE"],
            self.STATES["STOPPED"],
        ):
            self.logger.info(
                f"Recovered {pnl_data['open_count']} open position(s) "
                f"(PnL=${pnl_data['event_pnl']:.2f}). Resuming management."
            )
            self.state = self.STATES["IN_TRADE"]
            if not getattr(self.position_manager, '_event_start_ts', None):
                self.position_manager._event_start_ts = time.time()

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
                reason = "shutdown" if hasattr(self, '_shutdown_deadline') else "expired subscription"
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

        daily_ok, daily_msg = self.risk_manager.check_daily_loss(
            pnl_data["daily_pnl"]
        )
        if not daily_ok:
            self.logger.warning(f"Stopping bot: {daily_msg}")
            for pos_data in self.trade_executor.close_all_bot_positions():
                self.position_manager.note_closed(pos_data)
            self.state = self.STATES["STOPPED"]
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

        market_open = cfg.is_market_open()
        if market_open:
            dyn = self._check_market_dynamic()
            if dyn is not None:
                market_open = dyn

        if not market_open:
            if self.state == self.STATES["IN_TRADE"]:
                self.logger.info("Market closed, managing open positions only")
            elif self.state != self.STATES["MARKET_CLOSED"]:
                self.logger.info("Market closed, pausing until reopen")
                self.state = self.STATES["MARKET_CLOSED"]
                return
            else:
                return
        elif self.state == self.STATES["MARKET_CLOSED"]:
            dyn = self._check_market_dynamic()
            if dyn is True:
                self.logger.info("Market reopened, resuming normal operation")
                self.state = self.STATES["IDLE"]
            else:
                return

        self._ml_heartbeat_ticks += 1
        if self._ml_heartbeat_ticks >= 100 and self._direction_predictor is not None:
            self._ml_heartbeat_ticks = 0
            n_override = getattr(self.signal_engine, '_ml_override_count', 0)
            self.logger.info(f"[ML] Heartbeat: overrides today={n_override} | models=active")

        if self.state == self.STATES["IN_TRADE"]:
            await self._handle_in_trade(pnl_data)
        elif self.state == self.STATES["COOLDOWN"]:
            await self._handle_cooldown()
        elif self.state == self.STATES["WAITING_FOR_FUNDS"]:
            await self._handle_waiting_for_funds()
        else:
            await self._handle_search(pnl_data)

        self._write_state()

    async def _handle_search(self, pnl_data: Dict):
        if self._winding_down:
            return

        m1_count = cfg.ML_M1_HISTORY_BARS if (hasattr(self, '_direction_predictor') and self._direction_predictor is not None) else 100
        m1_data = self.client.get_rates(
            self.symbol, cfg.SIGNAL_TIMEFRAME, m1_count
        )
        if m1_data is None or len(m1_data) < 10:
            self._log_signal_diagnostic(
                "insufficient_m1_data",
                {"bars": 0 if m1_data is None else len(m1_data)},
            )
            return

        self.adaptive_conf.update(m1_data)

        symbol_info = self.client.get_symbol_info(self.symbol)
        if symbol_info is None:
            self._log_signal_diagnostic("symbol_info_unavailable", {})
            return

        now = time.monotonic()
        if not hasattr(self, '_last_bias_time'):
            self._last_bias_time = 0.0
        bias_interval = getattr(self, '_bias_update_interval', cfg.BIAS_UPDATE_INTERVAL_SEC)
        if now - self._last_bias_time >= bias_interval or \
           self.state == self.STATES["IDLE"]:
            self._last_bias_time = now
            if await self._update_bias():
                self.state = self.STATES["AWAITING_SIGNAL"]
                self.logger.info("Looking for clear setup...")

        if self.state != self.STATES["AWAITING_SIGNAL"]:
            return

        if not self.bias_engine.is_tradeable():
            self._log_signal_diagnostic(
                "bias_not_tradeable",
                {"bias": self._bias_summary.get("bias", "UNKNOWN"),
                 "strength": self._bias_summary.get("strength", 0)},
            )
            return

        bias_dir = self._bias_summary.get("bias", "NEUTRAL")
        current_price = symbol_info["ask"] if bias_dir == "BULLISH" else symbol_info["bid"]

        h1_high = None
        h1_low = None
        try:
            if m1_data is not None and len(m1_data) >= 60:
                m1_idx = m1_data.set_index("time")
                has_bid_ask_m1 = "high_ask" in m1_idx.columns
                if has_bid_ask_m1:
                    h1_agg = m1_idx.resample("1h").agg({
                        "high_ask": "max", "low_ask": "min",
                        "high_bid": "max", "low_bid": "min",
                        "high": "max", "low": "min",
                    }).dropna()
                else:
                    h1_agg = m1_idx.resample("1h").agg({
                        "high": "max", "low": "min",
                    }).dropna()
                if len(h1_agg) >= 2:
                    bar = h1_agg.iloc[-2]
                    h1_high = bar.get("high_ask" if (has_bid_ask_m1 and bias_dir == "BULLISH") else "high_bid" if (has_bid_ask_m1 and bias_dir == "BEARISH") else "high")
                    h1_low = bar.get("low_ask" if (has_bid_ask_m1 and bias_dir == "BULLISH") else "low_bid" if (has_bid_ask_m1 and bias_dir == "BEARISH") else "low")
                    if h1_high is None or h1_low is None or not (h1_high > 0 and h1_low > 0):
                        h1_high = None
                        h1_low = None
        except Exception as e:
            self.logger.warning(f"H1 range computation failed: {e}")

        if h1_high is None or h1_low is None or h1_high <= h1_low:
            h1_data = self.client.get_rates(
                self.symbol, cfg.BIAS_TIMEFRAME, 3
            )
            if h1_data is not None and len(h1_data) >= 2:
                has_bid_ask = "high_ask" in h1_data.columns
                if has_bid_ask and bias_dir == "BULLISH":
                    h1_high = h1_data["high_ask"].iloc[-2]
                    h1_low = h1_data["low_ask"].iloc[-2]
                elif has_bid_ask and bias_dir == "BEARISH":
                    h1_high = h1_data["high_bid"].iloc[-2]
                    h1_low = h1_data["low_bid"].iloc[-2]
                else:
                    h1_high = h1_data["high"].iloc[-2]
                    h1_low = h1_data["low"].iloc[-2]
            else:
                h1_high = None
                h1_low = None

        signal = self.signal_engine.evaluate(
            m1_data, self._bias_summary, current_price,
            h1_high=h1_high, h1_low=h1_low
        )
        if signal:
            ml_tag = " [ML OVERRIDE]" if signal.get("ml_override") else ""
            sig_dir = signal.get('direction', 'UNKNOWN')
            reason = (
                f"Price broke above H1 high (${h1_high:.2f}) with bullish momentum"
                if sig_dir == 'BUY'
                else f"Price broke below H1 low (${h1_low:.2f}) with bearish momentum"
            )
            sf_key = f"{signal['direction']}|{signal.get('ml_override')}|{signal['score']:.2f}"
            sf_now = time.monotonic()
            if sf_key != self._last_signal_found_key or sf_now - self._last_signal_found_time >= 30:
                self._last_signal_found_key = sf_key
                self._last_signal_found_time = sf_now
                self.logger.signal(
                    f"Signal found: {signal['direction']}{ml_tag} | {reason} "
                    f"(score={signal['score']:.3f})"
                )
        else:
            rejection = self.signal_engine.last_rejection or {}
            self._log_signal_diagnostic(
                rejection.get("reason", "signal_not_generated"),
                {"bias": bias_dir, "price": current_price,
                 "h1_high": h1_high, "h1_low": h1_low, **rejection},
            )

        # Shared entry logic
        override = getattr(self, '_signal_entry_threshold_override', None)
        effective_threshold = override if override is not None else (self.meta.current_threshold if self.meta else cfg.SIGNAL_ENTRY_THRESHOLD)
        if signal:
            atr_thresh = signal.get("atr_entry_threshold")
            if atr_thresh is not None:
                effective_threshold = max(atr_thresh, effective_threshold)
        if signal and signal["score"] >= effective_threshold:
            can_enter, reason = self.risk_manager.can_enter_trade(
                symbol_info, datetime.utcnow(),
                ml_override=signal.get('ml_override', False)
            )
            if not can_enter:
                bk_key = f"blocked_{signal['direction']}|{reason}|{signal.get('ml_override')}"
                bk_now = time.monotonic()
                if bk_key != self._last_signal_blocked_key or bk_now - self._last_signal_blocked_time >= 30:
                    self._last_signal_blocked_key = bk_key
                    self._last_signal_blocked_time = bk_now
                    self.logger.signal(
                f"Signal {signal.get('direction', 'UNKNOWN')} (score={signal['score']:.2f}) "
                f"blocked: {reason} | "
                f"price={current_price:.2f} "
                f"spread={symbol_info.get('spread', 0)}"
            )
                return

            if not self.adaptive_conf.should_enter():
                self.logger.signal(
                    f"Signal {signal.get('direction', 'UNKNOWN')} score={signal['score']:.2f} "
                    f"blocked: adaptive confirmation (low vol filter)"
                )
                return

            gemini_advice = await self.gemini_advisor.advise_entry({
                "bias": self._bias_summary.get("bias", "UNKNOWN"),
                "direction": signal.get("direction", "UNKNOWN"),
                "score": signal["score"],
                "atr": signal.get("atr_value"),
                "breakout_dist": signal.get("breakout_dist"),
                "range_size": signal.get("range_size"),
                "spread": symbol_info.get("spread", 0),
                "consecutive_losses": self.risk_manager.consecutive_losses,
                "session": self._current_session(),
                "momentum": round(
                    self.signal_engine.compute_momentum(m1_data), 3
                ) if hasattr(self.signal_engine, 'compute_momentum') else None,
                "volatility": symbol_info.get("volatility", 0),
            })

            if gemini_advice:
                action = gemini_advice.get("action", "proceed")
                reason = gemini_advice.get("reason", "")
                self.logger.signal(
                    f"Gemini: {action} | {reason}"
                )
                if action == "skip":
                    return
                if action == "caution":
                    tp_mod = float(gemini_advice.get("tp_modifier", 1.0))
                    if "tp1" in signal:
                        signal["tp1"] = signal["entry_price"] + (
                            signal["tp1"] - signal["entry_price"]
                        ) * tp_mod
                    if "tp2" in signal:
                        signal["tp2"] = signal["entry_price"] + (
                            signal["tp2"] - signal["entry_price"]
                        ) * tp_mod
                    if "tp3" in signal:
                        signal["tp3"] = signal["entry_price"] + (
                            signal["tp3"] - signal["entry_price"]
                        ) * tp_mod

            self._current_signal = signal
            self.logger.signal(
                f"Signal triggered: {signal.get('direction', 'UNKNOWN')} "
                f"score={signal['score']:.2f}"
            )
            await self._execute_entry(signal, symbol_info)
        elif signal:
            self._log_signal_diagnostic(
                "score_below_entry_threshold",
                {"direction": signal["direction"],
                 "price": current_price,
                 "score": signal["score"],
                 "threshold": effective_threshold},
            )

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
        key = (
            f"{reason}|{context.get('bias')}|{context.get('direction')}|"
            f"{context.get('h1_high')}|{context.get('h1_low')}"
        )
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
            f"bias={context.get('bias', 'UNKNOWN')} "
            f"direction={context.get('direction', 'n/a')} "
            f"price={fmt_num(context.get('price'))} "
            f"h1_high={fmt_num(context.get('h1_high'))} "
            f"h1_low={fmt_num(context.get('h1_low'))} "
            f"breakout={fmt_num(context.get('breakout_dist'))} "
            f"range={fmt_num(context.get('range_size'))} "
            f"score={fmt_num(context.get('score'), 3)} "
            f"threshold={fmt_num(context.get('threshold', cfg.SIGNAL_ENTRY_THRESHOLD), 3)}"
        )

    async def _handle_in_trade(self, pnl_data: Dict):
        if pnl_data["open_count"] == 0:
            # Grace period: if we just entered, API may not show positions yet
            if getattr(self.position_manager, "_event_start_ts", None) is not None and time.time() - self.position_manager._event_start_ts < 10:
                self.logger.debug("Waiting for positions to appear (API delay grace period)")
                return
            self.risk_manager.record_exit(pnl_data["event_pnl"])
            if self.meta:
                self.meta.record_trade(pnl_data["event_pnl"], self._bias_summary.get("strength", 0))
                balance = (self.client.get_account_info() or {}).get("balance", 0)
                self.meta.update(balance, self._bias_summary)
            self._enter_cooldown()
            return

        acct = self.client.get_account_info()
        balance = acct.get("balance", 0) if acct else 0

        event_ok, event_msg = self.risk_manager.check_event_loss(
            pnl_data["event_pnl"]
        )
        if not event_ok:
            self.logger.warning(f"Event stop: {event_msg}")
            closed = self.trade_executor.close_all_bot_positions()
            realized_pnl = 0.0
            for pos_data in closed:
                self.position_manager.note_closed(pos_data)
                realized_pnl += pos_data.get("profit", 0)
            self.risk_manager.record_exit(realized_pnl, len(closed))
            if self.meta:
                self.meta.record_trade(realized_pnl, self._bias_summary.get("strength", 0))
                self.meta.update(balance, self._bias_summary)
            await self._notify(
                "trade_close",
                "Trade Closed (Event Loss)",
                f"PnL: ${pnl_data['event_pnl']:.2f} | {self.position_manager.open_count} position(s)",
                {"pnl": pnl_data["event_pnl"], "reason": "event_loss"},
            )
            self._enter_cooldown()
            return

        m1_count = cfg.ML_M1_HISTORY_BARS if (hasattr(self, '_direction_predictor') and self._direction_predictor is not None) else 20
        m1_data = self.client.get_rates(
            self.symbol, cfg.SIGNAL_TIMEFRAME, m1_count
        )
        if m1_data is None:
            return

        positions = self.position_manager.summary().get("positions", [])
        if not positions:
            return

        direction = positions[0].get("type", "BUY")
        entry_price = self.position_manager.get_average_entry(direction)
        if entry_price is None:
            return

        entry_score = self._current_signal.get("score") if self._current_signal else None
        is_recovery = self._current_signal is None
        if is_recovery:
            if not hasattr(self, '_recovery_ticks'):
                self._recovery_ticks = 0
                self.logger.info("Recovery: warming up M1 window before exit checks")
            self._recovery_ticks += 1
            if self._recovery_ticks < 5:
                return
        else:
            self._recovery_ticks = 0

        exit_thresh = getattr(self, '_exit_threshold_override', cfg.EXIT_THRESHOLD_TIGHT)
        effective_exit_mode = getattr(self, '_exit_mode_override', cfg.EXIT_MODE)
        event_start = getattr(self.position_manager, '_event_start_ts', None)
        bars_since_entry = max(0, int((time.time() - event_start) / (cfg.EXIT_CHECK_INTERVAL or 300))) if event_start else 0
        should_exit, exit_score, reason = self.signal_engine.evaluate_exit(
                m1_data, entry_price, direction, entry_score,
                exit_mode=effective_exit_mode, exit_threshold=exit_thresh,
                signal=self._current_signal, bars_since_entry=bars_since_entry,
            )

        if not should_exit and reason == "ml_hold":
            now_t = time.time()
            if not hasattr(self, '_last_ml_hold_log'):
                self._last_ml_hold_log = 0.0
            if now_t - self._last_ml_hold_log >= 30.0:
                self._last_ml_hold_log = now_t
                self.logger.signal(f"ML hold: suppressing exit, ML still confident in {direction}")

        gemini_exit_advice = None
        if not should_exit:
            gemini_exit_advice = await self.gemini_advisor.advise_exit({
                "direction": direction,
                "pnl": round(pnl_data["event_pnl"], 2),
                "run_duration_min": round(
                    (time.time() - getattr(self.position_manager, '_event_start_ts', 0)) / 60
                ) if getattr(self.position_manager, '_event_start_ts', None) else None,
                "momentum": round(
                    self.signal_engine.compute_momentum(m1_data), 3
                ) if hasattr(self.signal_engine, 'compute_momentum') else None,
                "distance_from_entry": round(
                    m1_data["close"].iloc[-1] - entry_price
                ) if direction == "BUY" else round(
                    entry_price - m1_data["close"].iloc[-1]
                ),
                "tp1": self._current_signal.get("tp1") if self._current_signal else None,
                "tp2": self._current_signal.get("tp2") if self._current_signal else None,
                "spread": (self.client.get_symbol_info(self.symbol) or {}).get("spread", 0)
                if self.client else 0,
            })
            if gemini_exit_advice and gemini_exit_advice.get("action") == "exit_early":
                self.logger.signal(
                    f"Gemini suggests early exit: {gemini_exit_advice.get('reason', '')}"
                )
                should_exit = True

        if should_exit:
            self.logger.signal(
                f"Exit signal: score={exit_score:.2f} reason={reason}"
            )
            closed = self.trade_executor.close_all_bot_positions()
            realized_pnl = 0.0
            for pos_data in closed:
                self.position_manager.note_closed(pos_data)
                realized_pnl += pos_data.get("profit", 0)
            self.risk_manager.record_exit(realized_pnl, len(closed))
            if self.meta:
                self.meta.record_trade(realized_pnl, self._bias_summary.get("strength", 0))
                self.meta.update(balance, self._bias_summary)
            await self._notify(
                "trade_close",
                "Trade Closed",
                f"PnL: ${pnl_data['event_pnl']:.2f} | {reason}",
                {"pnl": pnl_data["event_pnl"], "reason": reason},
            )
            self._enter_cooldown()
            return

    async def _handle_cooldown(self):
        if self._winding_down:
            self.state = self.STATES["IDLE"]
            self._cooldown_until = None
            return
        if self._cooldown_until and datetime.now() >= self._cooldown_until:
            self.logger.info("Cooldown expired, resuming search")
            self.state = self.STATES["IDLE"]
            self._cooldown_until = None

    async def _handle_waiting_for_funds(self):
        info = self.client.get_account_info()
        if info and info["balance"] >= cfg.MIN_BALANCE:
            self.scaler.update_peak(info["balance"])
            if self.scaler.starting_balance is None:
                self.scaler.initialize(info["balance"])
            self.logger.info(
                f"Funds detected: ${info['balance']:.2f}. Starting bot."
            )
            self.state = self.STATES["IDLE"]

    async def _update_bias(self) -> bool:
        h1_data = self.client.get_rates(
            self.symbol, cfg.BIAS_TIMEFRAME, 96
        )
        if h1_data is None or len(h1_data) < 20:
            self.logger.warning("Cannot update bias: insufficient H1 data")
            return False

        summary = self.bias_engine.update(h1_data)
        self._bias_summary = summary

        if self.meta:
            info = self.client.get_account_info()
            balance = info["balance"] if info else 0
            self.meta.update(balance, summary)

        tradeable = self.bias_engine.is_tradeable()
        self.logger.bias(
            f"{summary['bias']} (strength={summary['strength']:.2f}, "
            f"H1={summary['primary_trend']}) "
            f"{'TRADEABLE' if tradeable else 'WAITING'}"
        )
        return True

    async def _execute_entry(self, signal: Dict, symbol_info: Dict):
        self.state = self.STATES["ENTERING"]
        direction = signal["direction"]
        score = signal["score"]

        # Re-fetch fresh info for confirmation checks
        fresh_info = self.client.get_symbol_info(self.symbol)
        if fresh_info is None:
            self.logger.warning("Entry blocked: cannot fetch symbol info")
            self.state = self.STATES["IDLE"]
            return

        if fresh_info.get("market_status") != "TRADEABLE":
            self.logger.warning(f"Entry blocked: market {fresh_info.get('market_status')}")
            self.state = self.STATES["IDLE"]
            return

        account = self.client.get_account_info()
        if account is None:
            self.logger.warning("Entry blocked: cannot fetch account info")
            self.state = self.STATES["IDLE"]
            return
        balance = account.get("balance", 0)
        if balance < cfg.MIN_BALANCE:
            self.logger.warning(
                f"Entry blocked: balance ${balance:.2f} below minimum ${cfg.MIN_BALANCE:.2f}"
            )
            self.state = self.STATES["WAITING_FOR_FUNDS"]
            return

        self.scaler.update_peak(balance)

        lot_mult = getattr(self, '_lot_multiplier_override', None)
        if lot_mult is None:
            lot_mult = self.meta.current_lot_mult if self.meta else cfg.LOT_MULTIPLIER

        if cfg.AGGRESSIVE_SIZING_ENABLED:
            if score >= cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD:
                lot_mult *= cfg.AGGRESSIVE_VERY_STRONG_LOT_MULT
                self.logger.info(
                    f"Aggressive LOT: score={score:.2f} >= {cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD} "
                    f"→ lot_mult={lot_mult:.1f}x"
                )
            elif score >= cfg.AGGRESSIVE_STRONG_THRESHOLD:
                lot_mult *= cfg.AGGRESSIVE_STRONG_LOT_MULT
                self.logger.info(
                    f"Aggressive LOT: score={score:.2f} >= {cfg.AGGRESSIVE_STRONG_THRESHOLD} "
                    f"→ lot_mult={lot_mult:.1f}x"
                )

        lot = min(self.scaler.get_lot(balance) * lot_mult, cfg.MAX_LOT)
        vol_step = fresh_info.get("volume_step", cfg.LOT_STEP)
        lot = round(lot / vol_step) * vol_step
        lot = max(fresh_info.get("volume_min", cfg.MIN_LOT), min(lot, fresh_info.get("volume_max", cfg.MAX_LOT)))
        max_trades = self.scaler.get_trades_per_event(balance, score)
        if self.meta:
            max_trades = self.meta.current_trades_per_event

        # Price drift check — reject if price moved outside allowed drift
        current_price = fresh_info["bid"] if direction == "SELL" else fresh_info["ask"]
        signal_price = symbol_info["bid"] if direction == "SELL" else symbol_info["ask"]
        point = fresh_info.get("point", 0.01)
        drift_amount = abs(current_price - signal_price)
        drift_pct = drift_amount / signal_price * 100
        max_drift = cfg.MAX_SPREAD_PIPS * point
        if drift_amount > max_drift:
            self.logger.warning(
                f"Entry blocked: price drifted ${drift_amount:.2f} "
                f"from signal price ${signal_price:.2f} to ${current_price:.2f} "
                f"(max drift ${max_drift:.2f})"
            )
            self.state = self.STATES["IDLE"]
            return

        # Margin check — ensure at least 1 trade fits, then fire sequentially
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
                    f"Reducing lot from {lot:.2f} to {max_lot_by_margin:.2f} "
                    f"(margin: ${free_margin:.2f} available, "
                    f"${single_margin:.2f} needed)"
                )
                lot = max_lot_by_margin
                single_margin = lot * current_price * margin_rate * 1
            if free_margin < single_margin:
                self.logger.warning(
                    f"Entry blocked: insufficient margin "
                    f"(est ${single_margin:.2f} needed, ${free_margin:.2f} available)"
                )
                self.state = self.STATES["IDLE"]
                return

        self.logger.info(
            f"Entry: {direction} | score={score:.2f} | "
            f"balance=${balance:.2f} | lot={lot:.2f} | "
            f"max_trades={max_trades} | drift={drift_pct:.3f}% | "
            f"margin=${free_margin:.2f} | "
            f"tier={self.scaler._tier(balance)} "
            f"({self.scaler.growth_pct(balance):.1f}% growth)"
        )

        any_opened = False
        for i in range(max_trades):
            can_add, add_msg = self.risk_manager.can_add_to_event()
            if not can_add:
                break

            # Recheck margin before each subsequent trade
            if i > 0:
                acct = self.client.get_account_info()
                fm = acct.get("free_margin", 0) if acct else 0
                if fm < single_margin:
                    self.logger.info(
                        f"Margin exhausted after {i} trade(s) "
                        f"(${fm:.2f} < ${single_margin:.2f} needed for next)"
                    )
                    break

            ticket = await self.trade_executor.open_market(
                self.symbol, direction, lot
            )
            if ticket is not None:
                any_opened = True
                self.risk_manager.record_entry(ml_override=signal.get('ml_override', False))
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
                    self.logger.info("Market closed detected, pausing until reopen")
                    self.state = self.STATES["MARKET_CLOSED"]
                    return
                break

        # Re-read balance and refresh positions after trades
        fresh_acct = self.client.get_account_info()
        if fresh_acct:
            self.scaler.update_peak(fresh_acct["balance"])
        self.position_manager.refresh()

        if self.position_manager.in_event or any_opened:
            self.state = self.STATES["IN_TRADE"]
            if any_opened:
                self.position_manager._event_start_ts = time.time()
            open_cnt = self.position_manager.open_count or max_trades
            self.logger.info(
                f"Entered {direction} mode with "
                f"{open_cnt} position(s)"
            )
            await self._notify(
                "trade_open",
                "Trade Opened",
                f"{direction} {lot} {self.symbol} @ ${current_price:.2f}",
                {"direction": direction, "lot": lot, "price": current_price, "symbol": self.symbol},
            )
        else:
            self.state = self.STATES["IDLE"]

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

    def _enter_cooldown(self):
        self.position_manager._event_start_ts = None
        self.state = self.STATES["COOLDOWN"]
        mult = 1 + self.risk_manager.consecutive_losses
        duration = self.risk_manager.cooldown_seconds * mult
        self._cooldown_until = datetime.now() + \
            timedelta(seconds=duration)
        self.logger.info(
            f"Cooldown for {duration}s (losses={self.risk_manager.consecutive_losses}) "
            f"until {self._cooldown_until.strftime('%H:%M:%S')}"
        )

    def get_state_summary(self) -> Dict:
        account = self.client.get_account_info()
        current_balance = account["balance"] if account else 0

        signal = self._current_signal or {}
        return {
            "state": self.state,
            "symbol": self.symbol,
            "magic": self.magic,
            "bias": self._bias_summary,
            "signal": signal,
            "positions": self.position_manager.summary(),
            "risk": {
                "consecutive_losses": self.risk_manager.consecutive_losses,
                "session_trades": self.risk_manager.session_trades,
                "event_trades": self.risk_manager.event_trades,
                "cooldown_until": self._cooldown_until.isoformat()
                if self._cooldown_until else None,
                "cooldown_active": self.state == self.STATES["COOLDOWN"],
                "max_daily_loss": self.risk_manager.max_daily_loss,
                "max_event_loss": self.risk_manager.max_event_loss,
                "max_trades_per_event": self.risk_manager.max_trades_per_event,
                "max_trades_per_session": self.risk_manager.max_trades_per_session,
                "cooldown_seconds": self.risk_manager.cooldown_seconds,
                "daily_pnl": self.risk_manager.daily_pnl,
            },
            "scaler": self.scaler.summary(current_balance) if self.scaler.starting_balance else None,
            "cooldown_active": self.state == self.STATES["COOLDOWN"],
            "last_logs": self.logger.logs[-50:],
            "closed_trades": self.position_manager.closed_history[-100:],
        }

    def start(self):
        if self.state == self.STATES["STOPPED"]:
            self.risk_manager.record_daily_pnl_reset()
        self.state = self.STATES["IDLE"]
        self.logger.info("Bot manually started")

    def stop(self):
        self.state = self.STATES["STOPPED"]
        self.logger.warning("Bot manually stopped")

    async def emergency_close(self):
        self.logger.warning("Emergency close triggered")
        closed = self.trade_executor.close_all_bot_positions()
        for pos_data in closed:
            self.position_manager.note_closed(pos_data)
        self.position_manager.refresh()
        await self._notify(
            "trade_close",
            "Emergency Close",
            f"Closed {len(closed)} position(s) manually",
            {"count": len(closed), "reason": "emergency"},
        )
        self.state = self.STATES["COOLDOWN"]
        self._enter_cooldown()
        return len(closed)

    def update_settings(self, settings: Dict):
        clamped = {}
        if "max_daily_loss" in settings:
            clamped["max_daily_loss"] = max(0.01, min(float(settings["max_daily_loss"]), 10000.0))
            self.risk_manager.max_daily_loss = clamped["max_daily_loss"]
        if "max_event_loss" in settings:
            clamped["max_event_loss"] = max(0.01, min(float(settings["max_event_loss"]), 1000.0))
            self.risk_manager.max_event_loss = clamped["max_event_loss"]
        if "max_trades_per_event" in settings:
            clamped["max_trades_per_event"] = max(1, min(int(settings["max_trades_per_event"]), 50))
            self.risk_manager.max_trades_per_event = clamped["max_trades_per_event"]
        if "max_trades_per_session" in settings:
            clamped["max_trades_per_session"] = max(1, min(int(settings["max_trades_per_session"]), 100))
            self.risk_manager.max_trades_per_session = clamped["max_trades_per_session"]
        if "cooldown_seconds" in settings:
            clamped["cooldown_seconds"] = max(0, min(int(settings["cooldown_seconds"]), 86400))
            self.risk_manager.cooldown_seconds = clamped["cooldown_seconds"]
        if "consecutive_loss_limit" in settings:
            clamped["consecutive_loss_limit"] = max(1, min(int(settings["consecutive_loss_limit"]), 100))
            self.risk_manager.max_consecutive_losses = clamped["consecutive_loss_limit"]
        if "lot_multiplier" in settings:
            clamped["lot_multiplier"] = max(0.1, min(float(settings["lot_multiplier"]), 100.0))
            self._lot_multiplier_override = clamped["lot_multiplier"]
        if "signal_entry_threshold" in settings:
            clamped["signal_entry_threshold"] = max(0.1, min(float(settings["signal_entry_threshold"]), 1.0))
            self._signal_entry_threshold_override = clamped["signal_entry_threshold"]
        if "exit_threshold_tight" in settings:
            clamped["exit_threshold_tight"] = max(0.01, min(float(settings["exit_threshold_tight"]), 1.0))
            self._exit_threshold_override = clamped["exit_threshold_tight"]
        if "max_spread_pips" in settings:
            clamped["max_spread_pips"] = max(1.0, min(float(settings["max_spread_pips"]), 500.0))
            self.risk_manager.max_spread = clamped["max_spread_pips"]
        if "bias_update_interval_sec" in settings:
            clamped["bias_update_interval_sec"] = max(1, min(float(settings["bias_update_interval_sec"]), 3600.0))
            self._bias_update_interval = clamped["bias_update_interval_sec"]
        if "allowed_sessions" in settings:
            sessions_raw = settings["allowed_sessions"]
            if isinstance(sessions_raw, list):
                sessions_str = ",".join(str(s) for s in sessions_raw)
            else:
                sessions_str = str(sessions_raw)
            self.risk_manager.allowed_sessions = [s.strip().upper() for s in sessions_str.split(",") if s.strip()]
        if "exit_mode" in settings:
            clamped["exit_mode"] = max(1, min(int(settings["exit_mode"]), 6))
            self._exit_mode_override = clamped["exit_mode"]
        self.logger.info(f"Bot settings updated: {clamped}")

    def login(self, server: str, account: str, password: str) -> Dict:
        self.logger.info(f"Logging into {server} account {account}...")
        ok = self.client.reconnect(server, account, password)
        if ok:
            info = self.client.get_account_info()
            if info:
                self.scaler.initialize(info["balance"])
                self.state = self.STATES["IDLE"]
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
