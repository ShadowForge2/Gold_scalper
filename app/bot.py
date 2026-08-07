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
from app.signal_engine import SignalEngine
from app.risk_manager import RiskManager, EquityScaler
from app.trade_executor import TradeExecutor
from app.position_manager import PositionManager
from app.economic_calendar import EconomicCalendar
from app.news_state_machine import NewsStateMachine

try:
    from app.candle_ml import CandleML
    _HAS_CANDLE = True
except Exception:
    CandleML = None
    _HAS_CANDLE = False

try:
    from app.candle_brain import CandleBrainPredictor
    _HAS_CANDLE_BRAIN = True
except Exception:
    CandleBrainPredictor = None
    _HAS_CANDLE_BRAIN = False

try:
    from app.momentum_engine import MomentumEngine
    _HAS_MOMENTUM = True
except Exception:
    MomentumEngine = None
    _HAS_MOMENTUM = False

try:
    from app.wave_scalper import WaveScalper
    _HAS_WAVE = True
except Exception:
    WaveScalper = None
    _HAS_WAVE = False

try:
    from app.brain import BrainOrchestrator
    _HAS_BRAIN = True
except Exception:
    BrainOrchestrator = None
    _HAS_BRAIN = False


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
        self._symbol_reversal_confirms: Dict[str, int] = {}
        self._symbol_consecutive_losses: Dict[str, int] = {}
        self._symbol_last_loss_ts: Dict[str, float] = {}
        self._symbol_regime_skipped: Dict[str, Optional[str]] = {}  # "high"|"normal"|None
        self._symbol_atr_history: Dict[str, list] = {}  # rolling ATR for vol detection
        self._symbol_vol_regime: Dict[str, bool] = {}  # track current vol regime per symbol
        self._symbol_candle_ml: Dict[str, Optional[CandleML]] = {}
        self._symbol_candle_brain: Dict[str, Optional[object]] = {}  # CandleBrainPredictor per symbol
        self._symbol_last_rescan_ts: Dict[str, float] = {}
        self._symbol_rescan_count: Dict[str, int] = {}
        self._symbol_candle_entry: Dict[str, bool] = {}
        self._symbol_flip_streak: Dict[str, int] = {}  # consecutive strong-opposite calls for flip-cut
        # Candle ML backtest-parity exit state
        self._symbol_candle_entry_open: Dict[str, Optional[float]] = {}  # entry candle open (bt reference)
        self._symbol_candle_tp: Dict[str, Optional[float]] = {}          # TP price (bt: entry + 2*ATR)
        self._symbol_candle_last_boundary: Dict[str, Optional[int]] = {} # last M5 boundary evaluated
        # Momentum-jump engine (US100) per-symbol state
        self._symbol_momentum_engine: Dict[str, Optional[object]] = {}  # MomentumEngine instances
        self._symbol_momentum_entry: Dict[str, bool] = {}   # current trade was a momentum entry
        self._symbol_momentum_peak: Dict[str, Optional[float]] = {}  # best price since entry
        self._symbol_momentum_last_boundary: Dict[str, Optional[int]] = {}  # last M5 boundary exit-checked
        self._symbol_momentum_last_signal_ts: Dict[str, Optional[float]] = {}  # candle ts of last momentum entry
        self._symbol_momentum_eval_boundary: Dict[str, Optional[int]] = {}  # last M5 boundary scanned for entry
        self._symbol_momentum_m1_cache: Dict[str, Optional[pd.DataFrame]] = {}  # cached long M1 history
        self._symbol_momentum_m1_cache_ts: Dict[str, float] = {}  # cache fetch time
        self._last_momentum_warn_ts: float = 0.0  # throttle for short-window warnings
        # Wave-scalper per-symbol state
        self._symbol_wave_engine: Dict[str, Optional[object]] = {}  # WaveScalper instances
        self._symbol_wave_entry: Dict[str, bool] = {}   # current trade is a wave entry
        self._symbol_wave_cache_ts: Dict[str, float] = {}  # M1 history fetch time
        self._last_wave_warn_ts: float = 0.0  # throttle for wave history warnings

        # Load per-symbol models and signal engines
        # ML (CandleML/CandleBrain) models are skipped for momentum/wave-owned
        # pairs — those engines own entry/exit, so the models are never used.
        for sym in self.symbols:
            candle_pred = None
            wave_owns = bool(getattr(cfg, "WAVE_ENGINE_ENABLED", {}).get(sym, False)) and _HAS_WAVE
            momentum_owns = (not wave_owns
                             and bool(getattr(cfg, "MOMENTUM_ENGINE_ENABLED", {}).get(sym, False)))
            if (not wave_owns and not momentum_owns
                    and _HAS_CANDLE and getattr(cfg, "CANDLE_ML_ENABLED", True)):
                candle_path = cfg.CANDLE_ML_MODEL_PATHS.get(sym, cfg.CANDLE_ML_MODEL_PATHS.get("XAUUSD"))
                try:
                    candle_pred = CandleML(model_path=candle_path)
                    if candle_pred.model is not None:
                        self.logger.info(f"[{sym}] Candle ML loaded from {candle_path}")
                    else:
                        self.logger.warning(f"[{sym}] Candle ML model not found at {candle_path}")
                        candle_pred = None
                except Exception as e:
                    self.logger.warning(f"[{sym}] Failed to load Candle ML: {e}")

            engine = SignalEngine(logger=self.logger)
            self._symbol_engines[sym] = engine
            self._symbol_candle_ml[sym] = candle_pred

            # Load CandleBrain Transformer model
            brain_pred = None
            if (not wave_owns and not momentum_owns
                    and _HAS_CANDLE_BRAIN and getattr(cfg, "CANDLE_BRAIN_ENABLED", False)):
                brain_path = cfg.CANDLE_BRAIN_MODEL_PATHS.get(sym)
                if brain_path:
                    try:
                        brain_pred = CandleBrainPredictor(model_path=brain_path)
                        self.logger.info(f"[{sym}] CandleBrain loaded from {brain_path}")
                    except Exception as e:
                        self.logger.warning(f"[{sym}] CandleBrain load failed: {e}")
                        brain_pred = None
            self._symbol_candle_brain[sym] = brain_pred
            self._symbol_states[sym] = self.STATES["IDLE"]
            self._symbol_signals[sym] = None
            self._symbol_event_start_ts[sym] = None
            self._symbol_exit_confirms[sym] = 0
            self._symbol_reversal_confirms[sym] = 0
            self._symbol_consecutive_losses[sym] = 0
            self._symbol_last_loss_ts[sym] = 0.0
            self._symbol_regime_skipped[sym] = None
            self._symbol_atr_history[sym] = []
            self._symbol_vol_regime[sym] = False  # False = normal vol
            self._symbol_candle_entry[sym] = False
            self._symbol_candle_entry_open[sym] = None
            self._symbol_candle_tp[sym] = None
            self._symbol_candle_last_boundary[sym] = 0
            self._symbol_last_rescan_ts[sym] = 0.0
            self._symbol_rescan_count[sym] = 0

            # ── Momentum-jump engine (per-pair adapted params + regime gate) ──
            mom = None
            if _HAS_MOMENTUM and getattr(cfg, "MOMENTUM_ENGINE_ENABLED", {}).get(sym, False):
                try:
                    pp = dict(getattr(cfg, "MOMENTUM_PAIR_PARAMS", {}).get(sym, {}) or {})
                    mom = MomentumEngine(
                        mz_min=pp.get("mz_min", cfg.MOMENTUM_MZ_MIN),
                        body_min=pp.get("body_min", cfg.MOMENTUM_BODY_RATIO_MIN),
                        ts_min=pp.get("ts_min", cfg.MOMENTUM_TS_MIN),
                        ema_span=pp.get("ema_span", cfg.MOMENTUM_EMA_SPAN),
                        sl_r=pp.get("sl_r", cfg.MOMENTUM_SL_R),
                        jump_target=pp.get("jump_target", cfg.MOMENTUM_JUMP_TARGET_R),
                        retr_r=pp.get("retr_r", cfg.MOMENTUM_RETRACE_R),
                        max_hold=pp.get("max_hold", cfg.MOMENTUM_MAX_HOLD_BARS),
                        atr_period=pp.get("atr_period", cfg.ATR_PERIOD),
                        logger=self.logger,
                        gate=cfg.MOMENTUM_GATE.get(sym, "none"),
                        gate_threshold=cfg.MOMENTUM_GATE_THRESHOLD.get(sym, 0.55),
                        gate_window=getattr(cfg, "MOMENTUM_GATE_WINDOW", 96),
                    )
                    self.logger.info(
                        f"[{sym}] Momentum engine enabled (jump {mom.jump_target}R "
                        f"+ {mom.retr_r}R retrace, SL {mom.sl_r}R, max hold {mom.max_hold} bars, "
                        f"gate {mom.gate}>={mom.gate_threshold:.2f})"
                    )
                except Exception as e:
                    self.logger.warning(f"[{sym}] Momentum engine init failed: {e}")
                    mom = None
            self._symbol_momentum_engine[sym] = mom
            self._symbol_momentum_entry[sym] = False
            self._symbol_momentum_peak[sym] = None
            self._symbol_momentum_last_boundary[sym] = 0
            self._symbol_momentum_last_signal_ts[sym] = 0.0
            self._symbol_momentum_eval_boundary[sym] = 0
            self._symbol_momentum_m1_cache[sym] = None
            self._symbol_momentum_m1_cache_ts[sym] = 0.0

            # ── Wave scalper engine (owns entry/exit when enabled) ──
            wave_eng = None
            if wave_owns:
                try:
                    import joblib
                    model_dir = getattr(cfg, "CANDLE_ENGINE_MODEL_DIR", "models/candle_h1")
                    model_path = os.path.join(model_dir, f"{sym}.joblib")
                    loaded = None
                    if os.path.exists(model_path):
                        loaded = joblib.load(model_path)
                    model = loaded.get("model") if isinstance(loaded, dict) else loaded
                    feature_cols = loaded.get("feature_cols") if isinstance(loaded, dict) else None
                    if model is None:
                        self.logger.warning(f"[{sym}] Wave scalper: no model at {model_path} — chop gate disabled")
                    wave_eng = WaveScalper(
                        symbol=sym,
                        model=model,
                        feature_cols=feature_cols,
                        logger=self.logger,
                        entry_r=float(getattr(cfg, "CANDLE_ENGINE_WAVE_ENTRY_R", 0.50)),
                        cut_r=float(getattr(cfg, "CANDLE_ENGINE_WAVE_CUT_R", 0.03)),
                        profit_r=float(getattr(cfg, "CANDLE_ENGINE_WAVE_PROFIT_R", 0.05)),
                        cost_r=float(getattr(cfg, "CANDLE_ENGINE_COST_R", 0.05)),
                        jump_break_r=float(getattr(cfg, "CANDLE_ENGINE_WAVE_JUMP_BREAK_R", getattr(cfg, "CANDLE_ENGINE_JUMP_BREAK_R", 1.5))),
                        jump_body_r=float(getattr(cfg, "CANDLE_ENGINE_WAVE_JUMP_BODY_R", getattr(cfg, "CANDLE_ENGINE_JUMP_BODY_R", 0.70))),
                        trail_r=float(getattr(cfg, "CANDLE_ENGINE_WAVE_TRAIL_R", 0.5)),
                        reversal_r=float(getattr(cfg, "CANDLE_ENGINE_WAVE_REVERSAL_R", 0.5)),
                        rider_enabled=not bool(getattr(cfg, "CANDLE_ENGINE_WAVE_NO_RIDER", False)),
                    )
                    self.logger.info(
                        f"[{sym}] Wave scalper enabled (entry {wave_eng.entry_r}R cut "
                        f"{wave_eng.cut_r}R profit {wave_eng.profit_r}R rider "
                        f"{wave_eng.jump_break_r}R/{wave_eng.jump_body_r:.2f})"
                    )
                except Exception as e:
                    self.logger.warning(f"[{sym}] Wave scalper init failed: {e}")
                    wave_eng = None
            self._symbol_wave_engine[sym] = wave_eng
            self._symbol_wave_entry[sym] = False
            self._symbol_wave_cache_ts[sym] = 0.0

        # ── BRAIN — hierarchical trade decision intelligence ──
        self.brain = None
        if _HAS_BRAIN and getattr(cfg, "BRAIN_ENABLED", True):
            try:
                self.brain = BrainOrchestrator(
                    symbols=self.symbols,
                    model_dir=getattr(cfg, "BRAIN_MODEL_DIR", "models/brain"),
                    candle_models=self._symbol_candle_ml,
                )
                self.logger.info(f"Brain initialized: {len(self.symbols)} symbols")
            except Exception as e:
                self.logger.warning(f"Brain init failed: {e}")
                self.brain = None

        # Legacy single-symbol references (use first symbol for backward compat)
        first_sym = self.symbols[0] if self.symbols else cfg.SYMBOL
        self.signal_engine = self._symbol_engines.get(first_sym, SignalEngine(logger=self.logger))

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
        ident = getattr(self, '_account_id', None) or cfg.CAPITAL_IDENTIFIER or "unknown"
        try:
            from app.subscription import create_notification
            await create_notification(ident, ntype, title, message, data)
        except Exception as e:
            self.logger.warning(f"Notification failed ({ntype}): {e}")

    async def initialize(self) -> bool:
        self.logger.info(f"Initializing {self.symbol} scalping bot (Candle ML mode)...")
        self.logger.info(f"Config: LOT_MULTIPLIER={cfg.LOT_MULTIPLIER}")

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

        for sym in self.symbols:
            await self._tick_symbol(sym, pnl_data, balance)

        if self.state not in (self.STATES["STOPPED"], self.STATES["WAITING_FOR_FUNDS"]):
            has_trade = any(s == self.STATES["IN_TRADE"] for s in self._symbol_states.values())
            self.state = self.STATES["IN_TRADE"] if has_trade else self.STATES["IDLE"]

        self._write_state()

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
                    broker_sl = pos.get("sl", 0)
                    broker_tp = pos.get("tp", 0)
                    atr_val = 0.0
                    try:
                        m1_bars = getattr(cfg, 'CANDLE_ML_M1_HISTORY_BARS', 500)
                        m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, m1_bars)
                        if m1_data is not None and len(m1_data) >= 20:
                            engine = self._symbol_engines.get(sym)
                            if engine:
                                atr_val = engine._compute_atr_m5(m1_data, cfg.ATR_PERIOD)
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
                    # Recovered momentum trades keep their momentum exit path —
                    # SL (per-pair) + jump target + retrace + max hold.
                    is_mom = bool(getattr(cfg, "MOMENTUM_ENGINE_ENABLED", {}).get(sym, False))
                    wave_owns_rec = bool(getattr(cfg, "WAVE_ENGINE_ENABLED", {}).get(sym, False))
                    if is_mom:
                        mom_eng = self._symbol_momentum_engine.get(sym)
                        sl_r = float(mom_eng.sl_r) if mom_eng is not None else float(getattr(cfg, "MOMENTUM_SL_R", 1.0))
                        sl = broker_sl if broker_sl > 0 else (entry_price - sl_r * atr_val if direction == "BUY" else entry_price + sl_r * atr_val)
                        tp = None
                    self._symbol_signals[sym] = {
                        "signal_type": "wave" if wave_owns_rec else ("momentum" if is_mom else "candle_ml"),
                        "direction": direction,
                        "sl": sl,
                        "tp1": tp,
                        "atr": atr_val,
                        "score": 0.5,
                        "recovered": True,
                    }
                    if wave_owns_rec:
                        wave_eng = self._symbol_wave_engine.get(sym)
                        if wave_eng is not None:
                            wave_eng.adopt_position(direction, entry_price, atr=atr_val)
                            self._symbol_wave_entry[sym] = True
                    self._symbol_momentum_entry[sym] = is_mom and not wave_owns_rec
                    self._symbol_momentum_peak[sym] = entry_price if (is_mom and not wave_owns_rec) else None
                    self._symbol_momentum_last_boundary[sym] = 0
                    self._symbol_momentum_last_signal_ts[sym] = time.time()
                    self.logger.info(
                        f"[{sym}] Reconstructed signal: {direction} entry={entry_price:.2f} "
                        f"sl={sl:.2f} tp={tp if tp else 'n/a'} atr={atr_val:.2f} "
                        f"type={'wave' if wave_owns_rec else ('momentum' if is_mom else 'candle_ml')}"
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

        # ── BRAIN: update feature cache every tick ──
        if self.brain is not None and state == self.STATES["IN_TRADE"]:
            try:
                m1_bars_brain = getattr(cfg, "CANDLE_ML_M1_HISTORY_BARS", 500)
                m1_brain = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, m1_bars_brain)
                if m1_brain is not None and len(m1_brain) > 10:
                    sym_positions_brain = [p for p in pnl_data.get("positions", []) if p.get("_symbol_code") == sym]
                    entry_info = None
                    if sym_positions_brain:
                        pos_b = sym_positions_brain[0]
                        entry_info = {
                            "direction": pos_b.get("type", "BUY"),
                            "entry_price": pos_b.get("price_open", 0),
                            "sl": pos_b.get("sl", 0),
                            "tp1": pos_b.get("tp", 0),
                            "atr": self._symbol_signals.get(sym, {}).get("atr_value", 0),
                            "ml_confidence": self._symbol_signals.get(sym, {}).get("score", 0.5),
                            "adx_at_entry": self._symbol_signals.get(sym, {}).get("adx_at_entry", 0),
                            "regime_at_entry": self._symbol_signals.get(sym, {}).get("regime_at_entry", "UNKNOWN"),
                            "event_start": self._symbol_event_start_ts.get(sym),
                        }
                    symbol_info_b = self.client.get_symbol_info(sym)
                    price_b = symbol_info_b.get("ask", 0) if symbol_info_b else 0
                    self.brain.update_features(sym, m1_brain, price_b, entry_info)
            except Exception as e:
                self.logger.debug(f"[{sym}] Brain feature update failed: {e}")

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


    def _momentum_entry_signal(self, sym: str, current_price: float, high_vol: bool) -> Optional[Dict]:
        """US100 momentum-jump entry on the last completed M5 candle.

        The long M1 history (EMA480 warmup) is refetched at most once per
        MOMENTUM_REFRESH_SEC and re-evaluated every tick, so a signal fires as
        soon as its candle data is available. Mirrors _two_engine.us100_jump_trades.
        """
        mom = self._symbol_momentum_engine.get(sym)
        if mom is None:
            return None
        now = time.time()
        cache = self._symbol_momentum_m1_cache.get(sym)
        refresh = getattr(cfg, "MOMENTUM_REFRESH_SEC", 60)
        # The engine is M1-contract (it resamples M1->M5 for detection), so we
        # MUST fetch MINUTE candles here regardless of cfg.SIGNAL_TIMEFRAME.
        # Rendering with SIGNAL_TIMEFRAME=5 (M5) makes the cache M5 bars, and
        # then `need_bars` (2425 "M1" bars) can never be met by an 8000-minute
        # window (~1600 M5 bars) — the cache never fills and no signal ever fires.
        mom_timeframe = 1
        need_bars = (mom.ema_span + 5) * 5  # >= ema_span+5 M5 buckets, in M1 bars
        if cache is None or now - self._symbol_momentum_m1_cache_ts.get(sym, 0.0) >= refresh:
            try:
                from datetime import timedelta
                hist = int(getattr(cfg, "MOMENTUM_M1_HISTORY_BARS", 8000))
                window_min = max(hist, need_bars + 600)
                to_dt = datetime.utcnow()
                from_dt = to_dt - timedelta(seconds=window_min * 60)
                fetched = self.client.get_rates_range(sym, mom_timeframe, from_dt, to_dt)
                self._symbol_momentum_m1_cache_ts[sym] = now
                if fetched is not None and len(fetched) >= need_bars:
                    self._symbol_momentum_m1_cache[sym] = fetched
                    cache = fetched
                    if now - self._last_momentum_warn_ts > 600:
                        self._last_momentum_warn_ts = now
                        t0 = fetched["time"].iloc[0]
                        t1 = fetched["time"].iloc[-1]
                        self.logger.info(
                            f"[{sym}] Momentum M1 history ready: {len(fetched)} bars "
                            f"({t0} .. {t1})"
                        )
                elif fetched is not None:
                    # Too-short window is the #1 silent no-signal cause — make it visible.
                    if now - self._last_momentum_warn_ts > 600:
                        self._last_momentum_warn_ts = now
                        self.logger.warning(
                            f"[{sym}] Momentum M1 window short: {len(fetched)} bars, "
                            f"need >= {need_bars} — no signals until history fills"
                        )
                else:
                    # get_rates_range returned None (fetch/parse failure) — this
                    # used to be a fully silent no-signal path.
                    if now - self._last_momentum_warn_ts > 600:
                        self._last_momentum_warn_ts = now
                        self.logger.warning(
                            f"[{sym}] Momentum M1 fetch returned no data "
                            f"(epic={self.client._resolve_epic(sym)}), retrying in {refresh}s"
                        )
            except Exception as e:
                self.logger.warning(f"[{sym}] Momentum M1 fetch failed: {e}")
                self._symbol_momentum_m1_cache_ts[sym] = now
        if cache is None or len(cache) == 0:
            return None
        try:
            sig = mom.detect(cache, now_ts=now)
        except Exception as e:
            self.logger.warning(f"[{sym}] Momentum eval failed: {e}")
            return None
        if sig is None:
            reason = getattr(mom, "last_reason", "unknown")
            if now - self._last_momentum_warn_ts > 300:
                self._last_momentum_warn_ts = now
                self.logger.debug(f"[{sym}] Momentum no signal: {reason}")
            return None
        if sig.get("bar_time", 0) <= self._symbol_momentum_last_signal_ts.get(sym, 0.0):
            return None  # never re-enter the same candle we already traded
        sig["price"] = current_price
        sig["high_volatility"] = high_vol if high_vol else False
        sig["sl"] = None       # anchored to the actual fill in _execute_entry
        sig["tp1"] = None      # dynamic exit (jump target + retrace)
        sig["candle_open"] = sig.get("close", current_price)
        return sig

    def _wave_refresh(self, sym: str, wave_eng) -> Optional[pd.DataFrame]:
        """Fetch fresh M1 bars for the wave engine (long history on a refresh
        timer, short tail every tick). Returns the recent frame or None."""
        now = time.time()
        refresh = getattr(cfg, "WAVE_REFRESH_SEC", 60)
        try:
            if now - self._symbol_wave_cache_ts.get(sym, 0.0) >= refresh:
                from datetime import timedelta
                hist = int(getattr(cfg, "WAVE_M1_HISTORY_BARS", 8000))
                window_min = hist + 600
                to_dt = datetime.utcnow()
                from_dt = to_dt - timedelta(seconds=window_min * 60)
                fetched = self.client.get_rates_range(sym, 1, from_dt, to_dt)
                self._symbol_wave_cache_ts[sym] = now
                if fetched is not None and len(fetched) > 0:
                    wave_eng.set_history(fetched)
        except Exception as e:
            self.logger.warning(f"[{sym}] Wave M1 history refresh failed: {e}")
        try:
            recent = self.client.get_rates(sym, 1, 30)
            if recent is not None and len(recent) > 0:
                return recent
        except Exception as e:
            if time.time() - self._last_wave_warn_ts > 600:
                self._last_wave_warn_ts = time.time()
                self.logger.warning(f"[{sym}] Wave M1 fetch failed: {e}")
        return None

    def _wave_entry_signal(self, sym: str, current_price: float, symbol_info: Dict) -> Optional[Dict]:
        """Wave-scalper entry: feed fresh M1 bars; return a signal dict when the
        engine fires a wave/jump entry trigger (order still needs confirmation)."""
        eng = self._symbol_wave_engine.get(sym)
        if eng is None:
            return None
        recent = self._wave_refresh(sym, eng)
        now = time.time()
        if recent is None:
            return None
        try:
            action = eng.feed(recent, now_ts=now)
        except Exception as e:
            self.logger.warning(f"[{sym}] Wave eval failed: {e}")
            return None
        if action is None or action.get("type") != "enter":
            return None
        atr = eng.atr if eng.atr > 0 else (current_price * 0.001)
        return {
            "direction": action["direction"],
            "score": 0.5,
            "price": current_price,
            "candle_open": eng.state_dict().get("candle_open", 0) or current_price,
            "sl": None,
            "tp1": None,
            "signal_type": "wave",
            "atr": atr,
            "high_volatility": False,
            "bar_time": now,
            "gate_ok": eng.gate_ok,
        }

    async def _search_symbol(self, sym: str, pnl_data: Dict):
        """Search for entry signal on a single symbol."""
        engine = self._symbol_engines.get(sym)
        if engine is None:
            return

        momentum_enabled = (not bool(getattr(cfg, "WAVE_ENGINE_ENABLED", {}).get(sym, False))
                            and bool(getattr(cfg, "MOMENTUM_ENGINE_ENABLED", {}).get(sym, False)))
        wave_owns = (bool(getattr(cfg, "WAVE_ENGINE_ENABLED", {}).get(sym, False))
                     and self._symbol_wave_engine.get(sym) is not None)

        m1_data = None
        current_atr = 0.0
        high_vol = False
        symbol_info = None
        current_price = 0.0

        if not momentum_enabled:
            m1_bars = getattr(cfg, "CANDLE_ML_M1_HISTORY_BARS", 500)
            m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, m1_bars)
            if m1_data is None or len(m1_data) < 10:
                return

            self._feed_m5_volatility(m1_data)

            symbol_info = self.client.get_symbol_info(sym)
            if symbol_info is None:
                return

            current_price = symbol_info.get("ask", 0)

            current_atr = engine._compute_atr_m5(m1_data, cfg.ATR_PERIOD)
            high_vol = self._is_high_volatility(sym, current_atr)

            if self._should_skip_regime(sym, high_vol):
                return

            if high_vol:
                self.logger.debug(f"[{sym}] High volatility regime — tighter filters active")
        else:
            # Momentum symbols pull their own long M1 history inside
            # _momentum_entry_signal, so a short M1 fetch failure must NOT
            # block the momentum evaluation. Best-effort volatility regime only.
            try:
                m1_bars = getattr(cfg, "CANDLE_ML_M1_HISTORY_BARS", 500)
                m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, m1_bars)
                if m1_data is not None and len(m1_data) >= 10:
                    self._feed_m5_volatility(m1_data)
                    current_atr = engine._compute_atr_m5(m1_data, cfg.ATR_PERIOD)
                    high_vol = self._is_high_volatility(sym, current_atr)
                    if self._should_skip_regime(sym, high_vol):
                        return
            except Exception:
                pass

            symbol_info = self.client.get_symbol_info(sym)
            if symbol_info is None:
                if time.time() - self._last_momentum_warn_ts > 600:
                    self._last_momentum_warn_ts = time.time()
                    self.logger.warning(
                        f"[{sym}] Momentum skipped: symbol_info=None "
                        f"(epic={self.client._resolve_epic(sym)} unresolved or market closed)"
                    )
                return
            current_price = symbol_info.get("ask", 0)

        signal = None

        # ── Wave scalper — owns entry for its symbols ──
        if wave_owns:
            signal = self._wave_entry_signal(sym, current_price, symbol_info)
            if signal is not None:
                self.logger.signal(
                    f"[{sym}] Wave: {signal['direction']} score={signal['score']:.2f} "
                    f"px={current_price:.2f} atr={signal.get('atr', 0):.4f} "
                    f"gate={signal.get('gate_ok', True)}"
                )

        # ── Momentum-jump engine — owns entry for its symbols ──
        elif momentum_enabled:
            signal = self._momentum_entry_signal(sym, current_price, high_vol)
            if signal is not None:
                self.logger.signal(
                    f"[{sym}] Momentum jump: {signal['direction']} score={signal['score']:.2f} "
                    f"px={current_price:.2f} atr={signal.get('atr', 0):.4f}"
                )

        candle_ml = self._symbol_candle_ml.get(sym)
        candle_brain = self._symbol_candle_brain.get(sym)

        # ── CandleBrain Transformer entry (priority over CandleML) ──
        if not wave_owns and not momentum_enabled and candle_brain is not None and getattr(cfg, "CANDLE_BRAIN_ENABLED", False) and signal is None:
            try:
                brain_result = candle_brain.predict(m1_data)
                if brain_result is not None:
                    action = brain_result.get("action")  # BUY / SELL / NONE
                    confidence = brain_result.get("confidence", 0)
                    expectancy = brain_result.get("expectancy", 0.0)
                    entry_thresh = getattr(cfg, "CANDLE_BRAIN_ENTRY_THRESHOLD", 0.60)
                    exp_min = getattr(cfg, "CANDLE_BRAIN_EXPECTANCY_MIN", 0.30)
                    if (
                        action in ("BUY", "SELL")
                        and confidence >= entry_thresh
                        and expectancy >= exp_min
                    ):
                        _atr = current_atr if current_atr and current_atr > 0 else 0.5
                        sl_dist = _atr * getattr(cfg, "SL_ATR_MULTIPLIER", 1.0)
                        tp_dist = _atr * getattr(cfg, "TP1_MULTIPLIER", 2.0)
                        try:
                            _co = m1_data.set_index("time").resample("5min")["open"].first().dropna() if "time" in m1_data.columns else m1_data.resample("5min")["open"].first().dropna()
                            candle_open = float(_co.iloc[-1]) if len(_co) else current_price
                        except Exception:
                            candle_open = current_price
                        direction = action
                        signal = {
                            "direction": direction,
                            "score": confidence,
                            "ml_confidence": confidence,
                            "price": current_price,
                            "candle_open": candle_open,
                            "sl": candle_open - sl_dist if direction == "BUY" else candle_open + sl_dist,
                            "tp1": candle_open + tp_dist if direction == "BUY" else candle_open - tp_dist,
                            "signal_type": "candle_brain",
                            "atr": _atr,
                            "high_volatility": high_vol if high_vol else False,
                        }
                        self.logger.info(
                            f"[{sym}] CandleBrain: {direction} conf={confidence:.3f} "
                            f"exp={expectancy:.3f} px={current_price:.2f} atr={_atr:.4f}"
                        )
                    elif time.time() - getattr(self, '_last_brain_log_ts', 0) > 30:
                        self._last_brain_log_ts = time.time()
                        self.logger.info(
                            f"[{sym}] CandleBrain: no entry (action={action} "
                            f"conf={confidence:.3f} exp={expectancy:.3f} "
                            f"thresh={entry_thresh} exp_min={exp_min})"
                        )
            except Exception as e:
                self.logger.warning(f"[{sym}] CandleBrain eval failed: {e}")

        # ── CandleML XGBoost fallback entry ──
        def _use_candle_ml():
            if candle_ml is None or candle_ml.model is None:
                return False
            mode = cfg.CANDLE_ML_MODE.get(sym, "volatility")
            if mode == "always":
                return True
            if mode == "volatility":
                return high_vol
            return False

        if not wave_owns and not momentum_enabled and _use_candle_ml():
            try:
                from app.candle_ml import compute_candle_features
                m1_idx = m1_data.set_index("time") if "time" in m1_data.columns else m1_data
                feats = compute_candle_features(m1_idx)
                if feats is not None and len(feats) > 0:
                    prob_up = candle_ml.predict_proba(feats)
                    # Get M1 first-bar direction from last feature row
                    last_row = feats.iloc[[-1]]
                    m1_dir = last_row.get("m1_first_dir", pd.Series([0.0])).values[0]
                    if np.isnan(m1_dir):
                        m1_dir = 0
                    m1_dir = int(m1_dir)
                    conf_thresh = getattr(cfg, "CANDLE_ML_CONFIDENCE_THRESHOLDS", {}).get(
                        sym, getattr(cfg, "CANDLE_ML_CONFIDENCE_THRESHOLD", 0.65))
                    pred = candle_ml.predict(prob_up, m1_dir, confidence_threshold=conf_thresh)
                    if pred is not None:
                        direction = pred
                        score = max(prob_up, 1 - prob_up)
                        _atr = current_atr if current_atr and current_atr > 0 else 0.5
                        # Entry quality filter: trade only best candles at best times
                        pat_mode = getattr(cfg, "CANDLE_ML_PATTERN_FILTERS", {}).get(
                            sym, getattr(cfg, "CANDLE_ML_PATTERN_FILTER", "strict"))
                        from app.candle_ml import candle_pattern_gate
                        pat_ok, pat_name = candle_pattern_gate(m1_idx, direction, _atr, pat_mode)
                        hour = datetime.now(timezone.utc).hour
                        allowed = getattr(cfg, "CANDLE_ML_ALLOWED_HOURS", {}).get(sym, "")
                        hour_ok = True
                        if allowed:
                            allowed_list = [int(x) for x in str(allowed).split(",") if str(x).strip() != ""]
                            hour_ok = hour in allowed_list
                        if not pat_ok:
                            self.logger.info(
                                f"[{sym}] Candle ML: skip pattern={pat_name} dir={direction} "
                                f"conf={score:.3f} m1_dir={m1_dir}"
                            )
                        elif not hour_ok:
                            self.logger.info(
                                f"[{sym}] Candle ML: skip hour={hour} (allowed={allowed}) dir={direction} conf={score:.3f}"
                            )
                        else:
                            sl_dist = _atr * getattr(cfg, "SL_ATR_MULTIPLIER", 1.0)
                            tp_dist = _atr * getattr(cfg, "TP1_MULTIPLIER", 2.0)
                            # Entry candle open — matches backtest (entry = M5 candle open).
                            # SL/TP are referenced from the candle open, not the 1-min-late fill.
                            try:
                                _co = m1_idx.resample("5min")["open"].first().dropna()
                                candle_open = float(_co.iloc[-1]) if len(_co) else current_price
                            except Exception:
                                candle_open = current_price
                            signal = {
                                "direction": direction,
                                "score": score,
                                "ml_confidence": score,
                                "price": current_price,
                                "candle_open": candle_open,
                                "sl": candle_open - sl_dist if direction == "BUY" else candle_open + sl_dist,
                                "tp1": candle_open + tp_dist if direction == "BUY" else candle_open - tp_dist,
                                "signal_type": "candle_ml",
                                "atr": _atr,
                                "high_volatility": high_vol if high_vol else False,
                            }
                            self.logger.info(
                                f"[{sym}] Candle ML: {direction} pattern={pat_name} (conf={score:.3f}, "
                                f"m1_dir={m1_dir}, prob_up={prob_up:.3f})"
                            )
                    elif time.time() - getattr(self, '_last_candle_log_ts', 0) > 30:
                        self._last_candle_log_ts = time.time()
                        self.logger.info(
                            f"[{sym}] Candle ML: no entry (prob_up={prob_up:.3f} "
                            f"m1_dir={m1_dir} conf={max(prob_up, 1-prob_up):.3f})"
                        )
            except Exception as e:
                self.logger.warning(f"[{sym}] Candle ML eval failed: {e}")

        if signal:
            if high_vol:
                signal["high_volatility"] = True

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
                sig_type = signal.get("signal_type", "CANDLE_ML").upper()
                if sig_type == "candle_ml":
                    sig_type = "CNDL"
                elif sig_type == "candle_brain":
                    sig_type = "BRAIN"
                self.logger.signal(
                    f"[{sym}] Signal found: [{sig_type}] {sig_dir} | "
                    f"{sig_type} prediction at ${current_price:.2f} "
                    f"(score={signal['score']:.3f} SL={signal.get('sl', 'n/a')} TP={signal.get('tp1', 'n/a')})"
                )

        if signal:
            can_enter, reason = self.risk_manager.can_enter_trade(
                symbol_info, datetime.utcnow(), symbol=sym
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
                f"[{sym}] Signal triggered: [{signal.get('signal_type', 'CANDLE_ML').upper()}] {signal.get('direction', 'UNKNOWN')} "
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

    def _should_skip_regime(self, sym: str, high_vol: bool = False) -> bool:
        skipped = self._symbol_regime_skipped.get(sym)
        if not skipped:
            return False
        current_regime = "high" if high_vol else "normal"
        if skipped != current_regime:
            return False
        reset_hours = getattr(cfg, "CONSECUTIVE_LOSS_RESET_HOURS", 4.0)
        last_loss = self._symbol_last_loss_ts.get(sym, 0)
        if time.time() - last_loss > reset_hours * 3600:
            self._symbol_regime_skipped[sym] = None
            self._symbol_consecutive_losses[sym] = 0
            self.logger.info(f"[{sym}] Regime skip ({skipped}) reset after {reset_hours}h cooldown")
            return False
        return True

    def _record_trade_result(self, sym: str, pnl: float, high_vol: bool = False):
        prev_vol = self._symbol_vol_regime.get(sym, False)
        if high_vol != prev_vol:
            self._symbol_consecutive_losses[sym] = 0
            self._symbol_vol_regime[sym] = high_vol
            if high_vol:
                self.logger.info(f"[{sym}] Regime changed to HIGH VOL — loss counter reset")

        if pnl < 0:
            self._symbol_consecutive_losses[sym] = self._symbol_consecutive_losses.get(sym, 0) + 1
            self._symbol_last_loss_ts[sym] = time.time()
            losses = self._symbol_consecutive_losses[sym]
            skip_thresh = getattr(cfg, "CONSECUTIVE_LOSS_SKIP", 5)
            if losses >= skip_thresh:
                self._symbol_regime_skipped[sym] = "high" if high_vol else "normal"
                regime = "HIGH VOL" if high_vol else "NORMAL"
                self.logger.warning(
                    f"[{sym}] {losses} consecutive losses in {regime} regime — "
                    f"skipping for {getattr(cfg, 'CONSECUTIVE_LOSS_RESET_HOURS', 4.0)}h"
                )
            else:
                self.logger.info(f"[{sym}] Consecutive losses: {losses}/{skip_thresh}")
        else:
            if self._symbol_consecutive_losses.get(sym, 0) > 0:
                self.logger.info(f"[{sym}] Win resets consecutive loss counter")
            self._symbol_consecutive_losses[sym] = 0

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
            self._symbol_candle_entry[sym] = False
            self._symbol_exit_confirms[sym] = 0
            self._symbol_reversal_confirms[sym] = 0
            self._symbol_momentum_entry[sym] = False
            self._symbol_momentum_peak[sym] = None
            self._symbol_momentum_last_boundary[sym] = 0
            self._symbol_wave_entry[sym] = False
            return

        acct = self.client.get_account_info()
        balance = acct.get("balance", 0) if acct else 0

        direction = sym_positions[0].get("type", "BUY") if sym_positions else "BUY"

        # Candle ML / wave trades skip event loss — those backtests have no event-loss stop.
        is_candle = (self._symbol_candle_entry.get(sym, False)
                     or self._symbol_wave_entry.get(sym, False))
        if not is_candle:
            # Correlated symbols (EVENT_LOSS_GROUP) share ONE combined event-loss
            # budget. US100 + US500 move ~95% together, so pool their PnL so a
            # single drawdown trips one stop instead of two separate budgets.
            group = getattr(cfg, "EVENT_LOSS_GROUP", {}).get(sym, sym)
            group_syms = [
                s for s in self.symbols
                if getattr(cfg, "EVENT_LOSS_GROUP", {}).get(s, s) == group
            ]
            group_positions = [
                p for p in pnl_data.get("positions", [])
                if p.get("_symbol_code") in group_syms
            ]
            event_pnl = sum(p.get("profit", 0) for p in group_positions)
            event_ok, event_msg = self.risk_manager.check_event_loss(event_pnl, balance)
            if not event_ok:
                self.logger.warning(f"[{sym}] Event stop: {event_msg}")
                closed = []
                for g_sym in group_syms:
                    closed += self.trade_executor.close_all_bot_positions(symbol=g_sym) or []
                sig = self._symbol_signals.get(sym) or {}
                for pos_data in closed:
                    self.position_manager.note_closed(pos_data, exit_reason="event_loss", score=sig.get("score", 0), balance=balance)
                if closed:
                    pnl = sum(p.get("profit", 0) for p in closed)
                    self._record_trade_result(sym, pnl, sig.get("high_volatility", False))
                    for g_sym in group_syms:
                        self._symbol_states[g_sym] = self.STATES["IDLE"]
                        self._symbol_event_start_ts[g_sym] = None
                        self._symbol_candle_entry[g_sym] = False
                        self._symbol_exit_confirms[g_sym] = 0
                        self._symbol_reversal_confirms[g_sym] = 0
                        self._symbol_last_rescan_ts[g_sym] = 0.0
                        self._symbol_rescan_count[g_sym] = 0
                        self._symbol_momentum_entry[g_sym] = False
                        self._symbol_momentum_peak[g_sym] = None
                        self._symbol_momentum_last_boundary[g_sym] = 0
                        self._symbol_wave_entry[g_sym] = False
                    await self._notify(
                        "trade_close",
                        f"Trade Closed — {sym}",
                        f"{direction} {sym} closed (event_loss) | PnL: ${pnl:+.2f}",
                        {"symbol": sym, "direction": direction, "exit_reason": "event_loss", "pnl": pnl},
                    )
                else:
                    self.logger.warning(f"[{sym}] Event stop close failed, retrying next tick")
                return

        pos = sym_positions[0]
        direction = pos.get("type", "BUY")
        entry_price = pos.get("price_open", 0)
        current_px = pos.get("price_current", pos.get("current_price", entry_price))
        event_start = self._symbol_event_start_ts.get(sym)
        exit_interval = cfg.EXIT_CHECK_INTERVAL or 300
        bars_held = max(0, int((time.time() - event_start) / exit_interval)) if event_start else 0

        pos_signal = self._symbol_signals.get(sym)
        minutes_held = (time.time() - event_start) / 60.0 if event_start else 0.0

        # ── Wave scalper exit — engine-driven (cut / lock / rider trail / candle end) ──
        # No timeout, no event loss. The engine emits a pending exit action on a
        # completed M1 bar; we close at market and confirm_exit() so it can
        # re-enter the next wave from the exit price.
        if self._symbol_wave_entry.get(sym, False):
            eng = self._symbol_wave_engine.get(sym)
            if eng is None:
                self.logger.warning(f"[{sym}] Wave engine missing for open wave trade")
                self._symbol_wave_entry[sym] = False
            else:
                recent = self._wave_refresh(sym, eng)
                action = None
                if recent is not None:
                    try:
                        action = eng.feed(recent, now_ts=time.time())
                    except Exception as e:
                        self.logger.warning(f"[{sym}] Wave exit eval failed: {e}")
                if action is None or action.get("type") != "exit":
                    return
                exit_reason = action.get("reason", "wave_exit")
                self.logger.signal(
                    f"[{sym}] Wave exit: {exit_reason} | dir={direction} entry={entry_price:.2f} "
                    f"px={current_px:.2f} atr={eng.atr:.4f} held={minutes_held:.1f}m"
                )
                closed = self.trade_executor.close_all_bot_positions(symbol=sym)
                for pos_data in closed:
                    self.position_manager.note_closed(
                        pos_data, exit_reason=exit_reason,
                        score=pos_signal.get("score", 0) if pos_signal else 0,
                        balance=balance)
                if closed:
                    pnl = sum(p.get("profit", 0) for p in closed)
                    self._record_trade_result(sym, pnl, False)
                    eng.confirm_exit()
                    self._symbol_states[sym] = self.STATES["IDLE"]
                    self._symbol_event_start_ts[sym] = None
                    self._symbol_wave_entry[sym] = False
                    await self._notify(
                        "trade_close",
                        f"Trade Closed — {sym}",
                        f"{direction} {sym} closed ({exit_reason}) | PnL: ${pnl:+.2f} | held {int(minutes_held)}m",
                        {"symbol": sym, "direction": direction, "exit_reason": exit_reason, "pnl": pnl, "held_minutes": minutes_held},
                    )
                else:
                    self.logger.warning(f"[{sym}] Wave exit close failed, retrying next tick")
                return

        if is_candle:
            # ── Candle ML exit — trailing reversal line + TP only ──
            # The candle backtest evaluates ONLY completed M5 candles at 5-min
            # boundaries and has exactly two live exits:
            #   1. candle_reversal — completed candle traded back through the
            #      trailing reversal line → exit at market. The line trails the
            #      best price by CANDLE_ML_TRAIL_ATR x ATR, floored at the
            #      fill/break-even, so pullbacks don't close a valid trade.
            #   2. candle_tp_hit — completed candle reached TP (2×ATR from fill).
            # The SL branch is dead code in the backtest (1×ATR SL always
            # sits inside the reversal zone), so there is NO SL exit.
            # No timeout, no event loss.
            now = time.time()
            cur_boundary = int(now) - (int(now) % 300)
            last_boundary = self._symbol_candle_last_boundary.get(sym, 0)
            if last_boundary == 0:
                self._symbol_candle_last_boundary[sym] = cur_boundary
                return
            if cur_boundary <= last_boundary:
                return  # still inside the same M5 candle
            self._symbol_candle_last_boundary[sym] = cur_boundary

            # Skip the ENTRY candle — matches backtest (entry bar not re-checked).
            if event_start is not None:
                entry_boundary = int(event_start) - (int(event_start) % 300)
                if entry_boundary == last_boundary:
                    return

            candle_open = self._symbol_candle_entry_open.get(sym) or entry_price
            candle_tp = self._symbol_candle_tp.get(sym)
            if candle_tp is None:
                candle_tp = pos_signal.get("tp1") if pos_signal else None
            if candle_tp is None:
                self._symbol_candle_last_boundary[sym] = last_boundary
                return

            # Completed candle = M1 bars in [last_boundary, cur_boundary).
            # If the M1 data lags the boundary (bar not published yet), revert
            # last_boundary so the next tick retries the SAME candle — otherwise
            # the boundary is consumed and never re-evaluated.
            m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, 12)
            if m1_data is None or len(m1_data) == 0:
                self._symbol_candle_last_boundary[sym] = last_boundary
                return
            m1_idx = m1_data.set_index("time") if "time" in m1_data.columns else m1_data
            try:
                lo_ts = pd.Timestamp(last_boundary, unit="s")
                hi_ts = pd.Timestamp(cur_boundary, unit="s")
                cb = m1_idx[(m1_idx.index >= lo_ts) & (m1_idx.index < hi_ts)]
            except Exception:
                cb = pd.DataFrame()
            if len(cb) == 0:
                self._symbol_candle_last_boundary[sym] = last_boundary
                return
            c_high = float(cb["high"].max())
            c_low = float(cb["low"].min())

            atr_val = pos_signal.get("atr", 0) if pos_signal else 0
            trailing_enabled = getattr(cfg, "CANDLE_ML_TRAILING_ENABLED", True) and atr_val > 0
            trail_mult = getattr(cfg, "CANDLE_ML_TRAIL_ATR", 1.5)

            best_key = f"_best_price_{sym}"
            if not hasattr(self, best_key):
                setattr(self, best_key, entry_price)
            best = getattr(self, best_key)

            exit_reason = ""
            peak_retrace = getattr(cfg, "CANDLE_ML_PEAK_RETRACE_ENABLED", True)
            retrace_frac = getattr(cfg, "CANDLE_ML_PEAK_RETRACE_FRAC", 0.5)
            if direction == "BUY":
                best = max(best, c_high)
                setattr(self, best_key, best)
                anchor = candle_open
                if trailing_enabled:
                    if peak_retrace and retrace_frac > 0 and best > entry_price:
                        anchor = max(candle_open, best - retrace_frac * (best - entry_price))
                    else:
                        anchor = max(candle_open, best - atr_val * trail_mult)
                if c_low < anchor:
                    exit_reason = "candle_reversal"
                elif c_high >= candle_tp:
                    exit_reason = "candle_tp_hit"
            else:
                best = min(best, c_low)
                setattr(self, best_key, best)
                anchor = candle_open
                if trailing_enabled:
                    if peak_retrace and retrace_frac > 0 and best < entry_price:
                        anchor = min(candle_open, best + retrace_frac * (entry_price - best))
                    else:
                        anchor = min(candle_open, best + atr_val * trail_mult)
                if c_high > anchor:
                    exit_reason = "candle_reversal"
                elif c_low <= candle_tp:
                    exit_reason = "candle_tp_hit"

            # Max-loss time guard — force close a Candle ML trade that has been
            # underwater past CANDLE_ML_MAX_LOSS_MINUTES so it can't block better
            # opportunities indefinitely (config.py:393).
            if not exit_reason and getattr(cfg, "CANDLE_ML_MAX_LOSS_MINUTES", 0) > 0:
                max_loss_min = cfg.CANDLE_ML_MAX_LOSS_MINUTES
                underwater = (current_px < entry_price) if direction == "BUY" else (current_px > entry_price)
                if minutes_held >= max_loss_min and underwater:
                    exit_reason = "candle_max_loss_time"
                    self.logger.signal(
                        f"[{sym}] Candle ML max-loss time: {direction} px={current_px:.2f} "
                        f"entry={entry_price:.2f} held={minutes_held:.1f}m (limit={max_loss_min:.0f}m)"
                    )

            # ── CandleBrain model re-eval at M5 boundary ──
            # For CandleBrain trades, re-run the Transformer each boundary.
            # If it says CLOSE with enough confidence → exit.
            if not exit_reason and pos_signal and pos_signal.get("signal_type") == "candle_brain":
                cb = self._symbol_candle_brain.get(sym)
                if cb is not None and atr_val > 0:
                    try:
                        exit_bars = self.client.get_rates(
                            sym, cfg.SIGNAL_TIMEFRAME,
                            getattr(cfg, "CANDLE_BRAIN_M1_HISTORY", 500))
                        if exit_bars is not None and len(exit_bars) > 0:
                            brain_exit = cb.predict(exit_bars)
                            if brain_exit is not None:
                                mgmt_action = brain_exit.get("mgmt_action", "HOLD")
                                mgmt_conf = brain_exit.get("mgmt_confidence", 0)
                                exit_thresh = getattr(cfg, "CANDLE_BRAIN_EXIT_THRESHOLD", 0.60)
                                if mgmt_action == "CLOSE" and mgmt_conf >= exit_thresh:
                                    exit_reason = "candle_brain_exit"
                                    self.logger.signal(
                                        f"[{sym}] CandleBrain EXIT: mgmt={mgmt_action} "
                                        f"conf={mgmt_conf:.3f} dir={direction} "
                                        f"px={current_px:.2f} entry={entry_price:.2f} "
                                        f"held={minutes_held:.1f}m"
                                    )
                    except Exception as e:
                        self.logger.warning(f"[{sym}] CandleBrain mgmt eval failed: {e}")

            if not exit_reason and getattr(cfg, "CANDLE_ML_FLIP_EXIT_ENABLED", True):
                cm = self._symbol_candle_ml.get(sym)
                if cm is not None and cm.model is not None and atr_val > 0 and entry_price > 0:
                    try:
                        flip_bars = self.client.get_rates(
                            sym, cfg.SIGNAL_TIMEFRAME,
                            getattr(cfg, "CANDLE_ML_M1_HISTORY_BARS", 500))
                        if flip_bars is not None and len(flip_bars) > 0:
                            from app.candle_ml import compute_candle_features
                            f_idx = flip_bars.set_index("time") if "time" in flip_bars.columns else flip_bars
                            f = compute_candle_features(f_idx)
                            if f is not None and len(f) > 0:
                                prob_up = cm.predict_proba(f)
                                last_row = f.iloc[[-1]]
                                m1_dir = last_row.get("m1_first_dir", pd.Series([0.0])).values[0]
                                if np.isnan(m1_dir):
                                    m1_dir = 0
                                # Rigid "no": only strong opposite signals count,
                                # and only after CANDLE_ML_FLIP_CONSECUTIVE of them
                                # in a row (single blips are noise — 48% of runs).
                                flip_conf = getattr(cfg, "CANDLE_ML_FLIP_CONFS", {}).get(
                                    sym, getattr(cfg, "CANDLE_ML_FLIP_CONF", 0.70))
                                pred = cm.predict(
                                    prob_up, int(m1_dir), confidence_threshold=flip_conf)
                                streak = self._symbol_flip_streak.get(sym, 0)
                                if pred is None:
                                    pass  # no strong opposite call — keep streak
                                elif pred == direction:
                                    streak = 0  # model agrees again — flip cancelled
                                else:
                                    streak += 1  # strong opposite call
                                self._symbol_flip_streak[sym] = streak
                                need = getattr(cfg, "CANDLE_ML_FLIP_CONSECUTIVES", {}).get(
                                    sym, getattr(cfg, "CANDLE_ML_FLIP_CONSECUTIVE", 2))
                                if streak >= need:
                                    loss_atr = getattr(cfg, "CANDLE_ML_FLIP_LOSS_ATR", 0.25) * atr_val
                                    if direction == "BUY":
                                        losing = current_px <= entry_price - loss_atr
                                    else:
                                        losing = current_px >= entry_price + loss_atr
                                    if losing:
                                        exit_reason = "candle_model_flip"
                                        self.logger.signal(
                                            f"[{sym}] Candle ML flip-cut: {direction} px={current_px:.2f} "
                                            f"entry={entry_price:.2f} loss_thr={loss_atr:.2f} "
                                            f"streak={streak}/{need} pred={pred} prob_up={prob_up:.3f} "
                                            f"held={minutes_held:.1f}m"
                                        )
                    except Exception as e:
                        self.logger.warning(f"[{sym}] Candle ML flip-check failed: {e}")

            # ── BRAIN EVALUATION for Candle ML trades ──
            # If no Candle ML exit triggered, let the Brain decide.
            if not exit_reason and self.brain is not None:
                try:
                    brain_decision = self.brain.evaluate(
                        symbol=sym,
                        current_pnl=(current_px - entry_price) / atr_val if direction == "BUY" and atr_val > 0 else (entry_price - current_px) / atr_val if atr_val > 0 else 0,
                        current_price=current_px,
                        entry_price=entry_price,
                        direction=direction,
                        atr=atr_val,
                        minutes_held=minutes_held,
                        pending_signals=self._symbol_signals,
                    )
                    if brain_decision.get("action") == "EXIT" and brain_decision.get("confidence", 0) > 0.4:
                        exit_reason = f"brain_exit({brain_decision.get('reason', 'cascade')})"
                        self.logger.signal(
                            f"[{sym}] Brain EXIT: {brain_decision.get('reason', '')} | "
                            f"conf={brain_decision['confidence']:.2f} method={brain_decision.get('method', '?')} | "
                            f"dir={direction} entry={entry_price:.2f} px={current_px:.2f} "
                            f"held={minutes_held:.1f}m cascade={brain_decision.get('cascade_path', '?')} "
                            f"layers={brain_decision.get('layers_evaluated', 0)} "
                            f"timing={brain_decision.get('timing_ms', 0):.1f}ms"
                        )
                except Exception as e:
                    self.logger.warning(f"[{sym}] Brain evaluation failed: {e}")

            if not exit_reason:
                return  # candle trades use backtest logic only — no other exits

            self.logger.signal(
                f"[{sym}] Candle exit: {exit_reason} | dir={direction} entry={entry_price:.2f} "
                f"px={current_px:.2f} candle_open={candle_open:.2f} tp={candle_tp:.2f} "
                f"anchor={anchor:.2f} best={best:.2f} held={minutes_held:.1f}m"
            )
            closed = self.trade_executor.close_all_bot_positions(symbol=sym)
            for pos_data in closed:
                self.position_manager.note_closed(pos_data, exit_reason=exit_reason, score=pos_signal.get("score", 0) if pos_signal else 0, balance=balance)
            if closed:
                pnl = sum(p.get("profit", 0) for p in closed)
                self._record_trade_result(sym, pnl, pos_signal.get("high_volatility", False) if pos_signal else False)
                self._symbol_states[sym] = self.STATES["IDLE"]
                self._symbol_event_start_ts[sym] = None
                self._symbol_candle_entry[sym] = False
                self._symbol_candle_entry_open[sym] = None
                self._symbol_candle_tp[sym] = None
                self._symbol_candle_last_boundary[sym] = 0
                self._symbol_flip_streak[sym] = 0
                if self.brain:
                    self.brain.record_exit(sym, {
                        "exit_reason": exit_reason, "pnl": pnl,
                        "exit_price": current_px, "atr": atr_val,
                    })
                if hasattr(self, f"_best_price_{sym}"):
                    delattr(self, f"_best_price_{sym}")
                await self._notify(
                    "trade_close",
                    f"Trade Closed — {sym}",
                    f"{direction} {sym} closed ({exit_reason}) | PnL: ${pnl:+.2f} | held {int(minutes_held)}m",
                    {"symbol": sym, "direction": direction, "exit_reason": exit_reason, "pnl": pnl, "held_minutes": minutes_held},
                )
            else:
                self.logger.warning(f"[{sym}] Candle exit close failed, retrying next tick")
            return

        # ── Momentum-jump exit — SL 1R / jump-target 1R + 0.25R retrace / max hold ──
        # Mirrors _two_engine.us100_jump_trades: evaluated only on completed M5
        # candles; the SL check runs BEFORE the retrace check on the same bar.
        if self._symbol_momentum_entry.get(sym, False):
            now = time.time()
            cur_boundary = int(now) - (int(now) % 300)
            last_boundary = self._symbol_momentum_last_boundary.get(sym, 0)
            if last_boundary == 0:
                self._symbol_momentum_last_boundary[sym] = cur_boundary
                return
            if cur_boundary <= last_boundary:
                return  # still inside the same M5 candle
            self._symbol_momentum_last_boundary[sym] = cur_boundary

            # Skip the ENTRY candle — matches backtest (exits start at bar i+1).
            if event_start is not None:
                entry_boundary = int(event_start) - (int(event_start) % 300)
                if entry_boundary == last_boundary:
                    return

            m1_data = self.client.get_rates(sym, cfg.SIGNAL_TIMEFRAME, 12)
            if m1_data is None or len(m1_data) == 0:
                self._symbol_momentum_last_boundary[sym] = last_boundary
                return
            m1_idx = m1_data.set_index("time") if "time" in m1_data.columns else m1_data
            try:
                lo_ts = pd.Timestamp(last_boundary, unit="s")
                hi_ts = pd.Timestamp(cur_boundary, unit="s")
                cb = m1_idx[(m1_idx.index >= lo_ts) & (m1_idx.index < hi_ts)]
            except Exception:
                cb = pd.DataFrame()
            if len(cb) == 0:
                self._symbol_momentum_last_boundary[sym] = last_boundary
                return
            c_high = float(cb["high"].max())
            c_low = float(cb["low"].min())
            c_close = float(cb["close"].iloc[-1])

            pos_signal = self._symbol_signals.get(sym) or {}
            atr_i = pos_signal.get("atr", 0) or (entry_price * 0.001)
            sl = pos_signal.get("sl", 0) or 0
            mom = self._symbol_momentum_engine.get(sym)
            if mom is not None:
                jump_target = float(mom.jump_target)
                retr_r = float(mom.retr_r)
                max_hold = int(mom.max_hold)
            else:
                jump_target = float(getattr(cfg, "MOMENTUM_JUMP_TARGET_R", 1.0))
                retr_r = float(getattr(cfg, "MOMENTUM_RETRACE_R", 0.25))
                max_hold = int(getattr(cfg, "MOMENTUM_MAX_HOLD_BARS", 12))

            peak = self._symbol_momentum_peak.get(sym, entry_price)
            exit_reason = ""
            if direction == "BUY":
                if sl > 0 and c_low <= sl:
                    exit_reason = "momentum_sl"
                else:
                    if c_high > peak:
                        peak = c_high
                    bfe = (peak - entry_price) / atr_i
                    if bfe >= jump_target and c_close <= peak - retr_r * atr_i:
                        exit_reason = "momentum_retrace"
            else:
                if sl > 0 and c_high >= sl:
                    exit_reason = "momentum_sl"
                else:
                    if c_low < peak:
                        peak = c_low
                    bfe = (entry_price - peak) / atr_i
                    if bfe >= jump_target and c_close >= peak + retr_r * atr_i:
                        exit_reason = "momentum_retrace"
            self._symbol_momentum_peak[sym] = peak

            if not exit_reason and event_start is not None:
                entry_boundary = int(event_start) - (int(event_start) % 300)
                # Candle j ends at entry_boundary + (j - i + 1)*300 and is evaluated
                # during candle j+1's window, so subtract 1: bars_held = j - i.
                bars_held = int((cur_boundary - entry_boundary) / 300) - 1
                if bars_held >= max_hold:
                    exit_reason = "momentum_timeout"
                    self.logger.signal(
                        f"[{sym}] Momentum max hold: {direction} held={bars_held} bars "
                        f"(limit={max_hold}) px={current_px:.2f} entry={entry_price:.2f}"
                    )

            if not exit_reason:
                return

            self.logger.signal(
                f"[{sym}] Momentum exit: {exit_reason} | dir={direction} entry={entry_price:.2f} "
                f"px={current_px:.2f} peak={peak:.2f} sl={sl:.2f} atr={atr_i:.4f} "
                f"held={minutes_held:.0f}m"
            )
            closed = self.trade_executor.close_all_bot_positions(symbol=sym)
            for pos_data in closed:
                self.position_manager.note_closed(pos_data, exit_reason=exit_reason, score=pos_signal.get("score", 0), balance=balance)
            if closed:
                pnl = sum(p.get("profit", 0) for p in closed)
                self._record_trade_result(sym, pnl, pos_signal.get("high_volatility", False))
                self._symbol_states[sym] = self.STATES["IDLE"]
                self._symbol_event_start_ts[sym] = None
                self._symbol_candle_entry[sym] = False
                self._symbol_momentum_entry[sym] = False
                self._symbol_momentum_peak[sym] = None
                self._symbol_momentum_last_boundary[sym] = 0
                if self.brain:
                    self.brain.record_exit(sym, {
                        "exit_reason": exit_reason, "pnl": pnl,
                        "exit_price": current_px, "atr": atr_i,
                    })
                await self._notify(
                    "trade_close",
                    f"Trade Closed — {sym}",
                    f"{direction} {sym} closed ({exit_reason}) | PnL: ${pnl:+.2f} | held {int(minutes_held)}m",
                    {"symbol": sym, "direction": direction, "exit_reason": exit_reason, "pnl": pnl, "held_minutes": minutes_held},
                )
            else:
                self.logger.warning(f"[{sym}] Momentum exit close failed, retrying next tick")
            return

    async def _handle_waiting_for_funds(self, sym: str = None):
        info = self.client.get_account_info()
        min_balance = cfg.SYMBOL_MIN_BALANCE.get(sym, cfg.MIN_BALANCE) if sym else cfg.MIN_BALANCE
        if info and info["balance"] >= min_balance:
            self.scaler.update_peak(info["balance"])
            if self.scaler.starting_balance is None:
                self.scaler.initialize(info["balance"])
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

        self.scaler.update_peak(balance)

        lot = self.scaler.get_lot(balance, symbol=sym)
        vol_step = fresh_info.get("volume_step", cfg.LOT_STEP)
        lot = round(lot / vol_step) * vol_step
        lot = max(fresh_info.get("volume_min", cfg.MIN_LOT), min(lot, fresh_info.get("volume_max", cfg.MAX_LOT)))

        high_vol = signal.get("high_volatility", False)
        if high_vol:
            lot_reduction = getattr(cfg, "VOLATILITY_LOT_REDUCTION", 0.5)
            original_lot = lot
            lot = round(lot * lot_reduction / vol_step) * vol_step
            lot = max(fresh_info.get("volume_min", cfg.MIN_LOT), lot)
            self.logger.info(
                f"[{sym}] High vol: lot reduced {original_lot:.2f} → {lot:.2f} "
                f"({lot_reduction:.0%} reduction)"
            )
        max_trades = 1

        current_price = fresh_info.get("bid", fresh_info.get("price", 0)) if direction == "SELL" else fresh_info.get("ask", fresh_info.get("price", 0))
        signal_price = symbol_info.get("bid", 0) if direction == "SELL" else symbol_info.get("ask", 0)
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
            max_lot_by_margin = free_margin * 0.9 / (notional_per_lot / max(1.0, leverage))
            vol_step = fresh_info.get("volume_step", cfg.LOT_STEP)
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

        self.logger.info(
            f"[{sym}] Entry: {direction} | score={score:.2f} | "
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
            self._symbol_candle_entry[sym] = signal.get("signal_type") in ("candle_ml", "candle_brain")
            self._symbol_flip_streak[sym] = 0  # fresh trade — no stale flip streak
            if signal.get("signal_type") == "wave":
                # Wave trade: anchor cut/lock/trail at the ACTUAL fill (the
                # engine state machine owns all exits from here).
                wave_eng = self._symbol_wave_engine.get(sym)
                if wave_eng is not None:
                    wave_eng.confirm_entry(current_price)
                    self._symbol_wave_entry[sym] = True
                else:
                    self._symbol_wave_entry[sym] = False
            if signal.get("signal_type") == "momentum":
                # Momentum-jump trade: SL anchored at the ACTUAL fill (like the
                # candle TP anchor), peak starts at the fill, exits evaluated on
                # M5 boundaries from the next candle.
                self._symbol_momentum_entry[sym] = True
                self._symbol_momentum_peak[sym] = current_price
                self._symbol_momentum_last_boundary[sym] = int(time.time()) - (int(time.time()) % 300)
                self._symbol_momentum_last_signal_ts[sym] = signal.get("bar_time", time.time())
                atr_i = signal.get("atr", 0) or (current_price * 0.001)
                sl_r = float(getattr(cfg, "MOMENTUM_SL_R", 1.0))
                signal["atr"] = atr_i
                signal["sl"] = current_price - sl_r * atr_i if direction == "BUY" else current_price + sl_r * atr_i
                signal["tp1"] = None
            else:
                self._symbol_momentum_entry[sym] = False
                self._symbol_momentum_peak[sym] = None
            # Backtest-parity candle state: reference open, TP, and entry M5 boundary.
            # Anchor at the ACTUAL fill (break-even), not the candle open — matches
            # _bt_live_sim ANCHOR="fill". TP shifted by the same delta so it is
            # measured from the real entry, not the entry candle's open.
            sig_open = signal.get("candle_open") or current_price
            sig_tp = signal.get("tp1")
            self._symbol_candle_entry_open[sym] = current_price
            self._symbol_candle_tp[sym] = (current_price + (sig_tp - sig_open)) if sig_tp is not None else None
            self._symbol_candle_last_boundary[sym] = int(time.time()) - (int(time.time()) % 300)
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
            # ── BRAIN: record entry for thesis tracking ──
            if self.brain is not None:
                try:
                    self.brain.record_entry(sym, {
                        "direction": direction,
                        "price": current_price,
                        "sl": signal.get("sl", 0),
                        "tp1": signal.get("tp1", 0),
                        "atr": signal.get("atr", signal.get("atr_value", 0)),
                        "score": score,
                        "ml_confidence": ml_conf,
                        "adx_at_entry": signal.get("adx_at_entry", 0),
                        "regime_at_entry": self.brain.cache.get_regime(sym) if self.brain else "UNKNOWN",
                        "signal_type": signal.get("signal_type", ""),
                        "bias_agreement": signal.get("bias_agreement", 0.5),
                    })
                except Exception:
                    pass
        else:
            self._symbol_states[sym] = self.STATES["IDLE"]
            if signal.get("signal_type") == "wave":
                wave_eng = self._symbol_wave_engine.get(sym)
                if wave_eng is not None:
                    wave_eng.cancel_entry()

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

        momentum = {}
        for sym in self.symbols:
            mom = self._symbol_momentum_engine.get(sym)
            if mom is None:
                continue
            cache = self._symbol_momentum_m1_cache.get(sym)
            last_reason = getattr(mom, "last_reason", "unknown")
            if cache is not None and len(cache) > 0:
                try:
                    t0 = str(cache["time"].iloc[0])
                    t1 = str(cache["time"].iloc[-1])
                except Exception:
                    t0 = t1 = "?"
                bars = len(cache)
            else:
                bars, t0, t1 = 0, "-", "-"
            refresh_age = int(time.time() - self._symbol_momentum_m1_cache_ts.get(sym, 0.0))
            momentum[sym] = {
                "enabled": True,
                "gate": mom.gate,
                "gate_threshold": mom.gate_threshold,
                "cache_bars": bars,
                "cache_from": t0,
                "cache_to": t1,
                "refresh_age_sec": refresh_age,
                "last_reason": last_reason,
            }

        return {
            "state": self.state,
            "symbol": self.symbol,
            "symbol_states": dict(self._symbol_states),
            "magic": self.magic,
            "signal": signal,
            "news": news,
            "momentum": momentum,
            "positions": self.position_manager.summary(),
            "risk": {},
            "scaler": self.scaler.summary(current_balance) if self.scaler.starting_balance else None,
            "last_logs": self.logger.logs[-50:],
            "closed_trades": self.position_manager.closed_history[-100:],
            "brain": self.brain.get_stats() if self.brain else None,
        }

    def start(self):
        if self.state == self.STATES["STOPPED"]:
            self.risk_manager.reset_daily_pnl()
        self.state = self.STATES["IDLE"]
        for sym in self.symbols:
            self._symbol_states[sym] = self.STATES["IDLE"]
            self._symbol_event_start_ts[sym] = None
            self._symbol_exit_confirms[sym] = 0
            self._symbol_reversal_confirms[sym] = 0
            self._symbol_consecutive_losses[sym] = 0
            self._symbol_last_loss_ts[sym] = 0.0
            self._symbol_regime_skipped[sym] = None
            self._symbol_vol_regime[sym] = False
            self._symbol_signals[sym] = None
            self._symbol_candle_entry[sym] = False
            self._symbol_candle_entry_open[sym] = None
            self._symbol_candle_tp[sym] = None
            self._symbol_candle_last_boundary[sym] = 0
            self._symbol_last_rescan_ts[sym] = 0.0
            self._symbol_rescan_count[sym] = 0
            self._symbol_momentum_entry[sym] = False
            self._symbol_momentum_peak[sym] = None
            self._symbol_momentum_last_boundary[sym] = 0
            if hasattr(self, f"_best_price_{sym}"):
                delattr(self, f"_best_price_{sym}")
        self.logger.info("Bot manually started")

    def stop(self):
        self.state = self.STATES["STOPPED"]
        for sym in self.symbols:
            self._symbol_states[sym] = self.STATES["STOPPED"]
            self._symbol_candle_entry[sym] = False
            self._symbol_candle_entry_open[sym] = None
            self._symbol_candle_tp[sym] = None
            self._symbol_candle_last_boundary[sym] = 0
            self._symbol_momentum_entry[sym] = False
            self._symbol_momentum_peak[sym] = None
            self._symbol_momentum_last_boundary[sym] = 0
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
                self._symbol_exit_confirms[sym] = 0
                self._symbol_reversal_confirms[sym] = 0
                self._symbol_consecutive_losses[sym] = 0
                self._symbol_regime_skipped[sym] = None
                self._symbol_candle_entry[sym] = False
                self._symbol_candle_entry_open[sym] = None
                self._symbol_candle_tp[sym] = None
                self._symbol_candle_last_boundary[sym] = 0
                self._symbol_last_rescan_ts[sym] = 0.0
                self._symbol_rescan_count[sym] = 0
                self._symbol_momentum_entry[sym] = False
                self._symbol_momentum_peak[sym] = None
                self._symbol_momentum_last_boundary[sym] = 0
                if hasattr(self, f"_best_price_{sym}"):
                    delattr(self, f"_best_price_{sym}")
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
            cfg.LOT_MULTIPLIER = clamped["lot_multiplier"]
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
