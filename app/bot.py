import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import config as cfg
from app.logger import BotLogger
from app.mt5_client import MT5Client
from app.capital_client import CapitalClient
from app.bias_engine import BiasEngine
from app.signal_engine import SignalEngine
from app.risk_manager import RiskManager, EquityScaler
from app.trade_executor import TradeExecutor
from app.position_manager import PositionManager


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
    }

    def __init__(self, logger: Optional[BotLogger] = None):
        self.logger = logger or BotLogger()
        self.client: object = None
        self.bias_engine = BiasEngine()
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.scaler = EquityScaler()

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

        self._accounts_file: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "accounts.json")
        self._account_id: Optional[str] = None
        self._state_file: Optional[str] = None
        self._last_state_write = 0.0
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0.0
        self._reconnect_backoff_max = 60.0

    async def initialize(self) -> bool:
        self.logger.info(f"Initializing {self.symbol} scalping bot...")

        if cfg.BROKER == "CAPITAL":
            self.logger.info("Broker: Capital.com (REST API)")
            self.client = CapitalClient()
            success = self.client.initialize(
                api_key=cfg.CAPITAL_API_KEY,
                identifier=cfg.CAPITAL_IDENTIFIER,
                password=cfg.CAPITAL_PASSWORD,
                demo=cfg.CAPITAL_DEMO,
            )
        else:
            self.logger.info("Broker: MetaTrader 5")
            self.client = MT5Client()
            login = None
            password = None
            server = cfg.MT5_SERVER
            if cfg.MT5_ACCOUNT:
                try:
                    login = int(cfg.MT5_ACCOUNT)
                    password = cfg.MT5_PASSWORD
                except ValueError:
                    self.logger.warning("MT5_ACCOUNT is not a valid integer")
            success = self.client.initialize(login, password, server)

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
            self.state = self.STATES["STOPPED"]
            return False

    async def initialize_with_credentials(self, api_key: str, identifier: str, password: str, demo: bool = True) -> bool:
        self.logger.info(f"Initializing user bot for Capital.com (demo={demo})...")
        self.client = CapitalClient()
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
                self.logger.info(f"Connected: ${info['balance']:.2f} | Leverage: 1:{info['leverage']}")
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
            self.logger.error(f"Connection failed: {err}")
            self.state = self.STATES["STOPPED"]
            return False

    async def shutdown(self):
        self._running = False
        self.trade_executor.close_all_bot_positions()
        self.client.shutdown()
        self.logger.info("Bot shutdown complete")

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

        if self.state == self.STATES["STOPPED"]:
            return

        if not self.client.is_connected():
            now_t = time.time()
            backoff = min(2 ** self._reconnect_attempts, self._reconnect_backoff_max)
            if now_t - self._last_reconnect_time >= backoff:
                self.logger.warning(
                    f"Connection lost, reconnecting "
                    f"(attempt {self._reconnect_attempts + 1}, "
                    f"backoff={backoff:.0f}s)..."
                )
                await self.initialize()
                if self.client.is_connected():
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
            self.trade_executor.close_all_bot_positions()
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
        now = time.monotonic()
        if not hasattr(self, '_last_bias_time'):
            self._last_bias_time = 0.0
        if now - self._last_bias_time >= cfg.BIAS_UPDATE_INTERVAL_SEC or \
           self.state == self.STATES["IDLE"]:
            self._last_bias_time = now
            await self._update_bias()
            self.state = self.STATES["AWAITING_SIGNAL"]

        if self.state != self.STATES["AWAITING_SIGNAL"]:
            return

        if not self.bias_engine.is_tradeable():
            self._log_signal_diagnostic(
                "bias_not_tradeable",
                {
                    "bias": self._bias_summary.get("bias", "UNKNOWN"),
                    "strength": self._bias_summary.get("strength", 0),
                },
            )
            return

        m1_data = self.client.get_rates(
            self.symbol, cfg.SIGNAL_TIMEFRAME, 50
        )
        if m1_data is None or len(m1_data) < 10:
            self._log_signal_diagnostic(
                "insufficient_m1_data",
                {"bars": 0 if m1_data is None else len(m1_data)},
            )
            return

        symbol_info = self.client.get_symbol_info(self.symbol)
        if symbol_info is None:
            self._log_signal_diagnostic("symbol_info_unavailable", {})
            return

        bias_dir = self._bias_summary.get("bias", "NEUTRAL")
        current_price = symbol_info["bid"] if bias_dir == "BEARISH" else symbol_info["ask"]

        h1_data = self.client.get_rates(
            self.symbol, cfg.BIAS_TIMEFRAME, 3
        )
        h1_high = h1_data["high"].iloc[-2] if h1_data is not None and len(h1_data) >= 2 else None
        h1_low = h1_data["low"].iloc[-2] if h1_data is not None and len(h1_data) >= 2 else None

        signal = self.signal_engine.evaluate(
            m1_data, self._bias_summary, current_price,
            h1_high=h1_high, h1_low=h1_low
        )
        self._current_signal = signal

        if signal:
            self.logger.info(
                f"[DEBUG] Signal evaluated: {signal['direction']} "
                f"score={signal['score']:.3f} "
                f"breakout_dist={signal.get('breakout_dist', 0):.2f} "
                f"range={signal.get('range_size', 0):.2f} "
                f"price={current_price:.2f} "
                f"h1_high={h1_high:.2f} h1_low={h1_low:.2f}"
            )
        else:
            rejection = self.signal_engine.last_rejection or {}
            self._log_signal_diagnostic(
                rejection.get("reason", "signal_not_generated"),
                {
                    "bias": bias_dir,
                    "price": current_price,
                    "h1_high": h1_high,
                    "h1_low": h1_low,
                    **rejection,
                },
            )

        if signal and signal["score"] >= cfg.SIGNAL_ENTRY_THRESHOLD:
            can_enter, reason = self.risk_manager.can_enter_trade(
                symbol_info, datetime.now()
            )
            if not can_enter:
                self.logger.signal(
                    f"Signal {signal['direction']} (score={signal['score']:.2f}) "
                    f"blocked: {reason} | "
                    f"bias={bias_dir} price={current_price:.2f} "
                    f"h1_high={h1_high:.2f} h1_low={h1_low:.2f} "
                    f"breakout={signal.get('breakout_dist', 0):.2f} "
                    f"range={signal.get('range_size', 0):.2f} "
                    f"spread={symbol_info.get('spread', 0)}"
                )
                return

            self.logger.signal(
                f"Signal triggered: {signal['direction']} "
                f"score={signal['score']:.2f} | "
                f"breakout={signal.get('breakout_dist', 0):.2f}pt"
            )
            await self._execute_entry(signal, symbol_info)
        elif signal:
            self._log_signal_diagnostic(
                "score_below_entry_threshold",
                {
                    "direction": signal["direction"],
                    "bias": bias_dir,
                    "price": current_price,
                    "h1_high": h1_high,
                    "h1_low": h1_low,
                    "breakout_dist": signal.get("breakout_dist", 0),
                    "range_size": signal.get("range_size", 0),
                    "score": signal["score"],
                    "threshold": cfg.SIGNAL_ENTRY_THRESHOLD,
                },
            )

    def _log_signal_diagnostic(self, reason: str, context: Dict):
        now = time.monotonic()
        key = (
            f"{reason}|{context.get('bias')}|{context.get('direction')}|"
            f"{context.get('score')}|{context.get('h1_high')}|{context.get('h1_low')}"
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
            self.state = self.STATES["IDLE"]
            return

        event_ok, event_msg = self.risk_manager.check_event_loss(
            pnl_data["event_pnl"]
        )
        if not event_ok:
            self.logger.warning(f"Event stop: {event_msg}")
            self.trade_executor.close_all_bot_positions()
            self.risk_manager.record_exit(pnl_data["event_pnl"])
            self._enter_cooldown()
            return

        m1_data = self.client.get_rates(
            self.symbol, cfg.SIGNAL_TIMEFRAME, 20
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
        should_exit, exit_score, reason = self.signal_engine.evaluate_exit(
            m1_data, entry_price, direction, entry_score, exit_mode=1
        )

        if should_exit:
            self.logger.signal(
                f"Exit signal: score={exit_score:.2f} reason={reason}"
            )
            self.trade_executor.close_all_bot_positions()
            self.risk_manager.record_exit(pnl_data["event_pnl"])
            self._enter_cooldown()
            return

    async def _handle_cooldown(self):
        if self._cooldown_until and datetime.now() >= self._cooldown_until:
            self.logger.info("Cooldown expired, resuming search")
            self.state = self.STATES["IDLE"]
            self._cooldown_until = None

    async def _handle_waiting_for_funds(self):
        info = self.client.get_account_info()
        if info and info["balance"] >= cfg.MIN_BALANCE:
            self.scaler.initialize(info["balance"])
            self.logger.info(
                f"Funds detected: ${info['balance']:.2f}. Starting bot."
            )
            self.state = self.STATES["IDLE"]

    async def _update_bias(self):
        h1_data = self.client.get_rates(
            self.symbol, cfg.BIAS_TIMEFRAME, 96
        )
        if h1_data is None or len(h1_data) < 20:
            self.logger.warning("Cannot update bias: insufficient H1 data")
            return

        summary = self.bias_engine.update(h1_data)
        self._bias_summary = summary

        tradeable = self.bias_engine.is_tradeable()
        self.logger.bias(
            f"{summary['bias']} (strength={summary['strength']:.2f}, "
            f"H1={summary['primary_trend']}) "
            f"{'TRADEABLE' if tradeable else 'WAITING'}"
        )

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
        balance = account.get("balance", 0) if account else fresh_info.get("balance", 0)
        if balance < cfg.MIN_BALANCE:
            self.logger.warning(
                f"Entry blocked: balance ${balance:.2f} below minimum ${cfg.MIN_BALANCE:.2f}"
            )
            self.state = self.STATES["WAITING_FOR_FUNDS"]
            return

        self.scaler.update_peak(balance)

        lot = min(self.scaler.get_lot(balance) * cfg.LOT_MULTIPLIER, cfg.MAX_LOT)
        vol_step = fresh_info.get("volume_step", cfg.LOT_STEP)
        lot = round(lot / vol_step) * vol_step
        lot = max(fresh_info.get("volume_min", cfg.MIN_LOT), min(lot, fresh_info.get("volume_max", cfg.MAX_LOT)))
        max_trades = self.scaler.get_trades_per_event(balance, score)

        # Price drift check — reject if price moved outside signal range
        current_price = fresh_info["bid"] if direction == "SELL" else fresh_info["ask"]
        signal_price = symbol_info["bid"] if direction == "SELL" else symbol_info["ask"]
        drift_pct = abs(current_price - signal_price) / signal_price * 100 if signal_price else 0
        if drift_pct > cfg.MAX_SPREAD_PIPS * 0.1:
            self.logger.warning(
                f"Entry blocked: price drifted {drift_pct:.3f}% "
                f"from signal price ${signal_price:.2f} to ${current_price:.2f}"
            )
            self.state = self.STATES["IDLE"]
            return

        # Margin check — verify sufficient free margin for the position
        margin_rate = fresh_info.get("margin_rate", 0.05)
        est_margin = lot * current_price * margin_rate * max_trades
        free_margin = account.get("free_margin", 0) if account else 0
        if free_margin < est_margin:
            self.logger.warning(
                f"Entry blocked: insufficient margin "
                f"(est ${est_margin:.2f} needed, ${free_margin:.2f} available)"
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

        for i in range(max_trades):
            can_add, add_msg = self.risk_manager.can_add_to_event()
            if not can_add:
                break

            ticket = self.trade_executor.open_market(
                self.symbol, direction, lot
            )
            if ticket is not None:
                self.risk_manager.record_entry()
                await asyncio.sleep(0.2)
            else:
                break

        # Re-read balance and refresh positions after trades
        fresh_acct = self.client.get_account_info()
        if fresh_acct:
            self.scaler.update_peak(fresh_acct["balance"])
        self.position_manager.refresh()

        if self.position_manager.in_event:
            self.state = self.STATES["IN_TRADE"]
            self.logger.info(
                f"Entered {direction} mode with "
                f"{self.position_manager.open_count} position(s)"
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
        account = self.client.get_account_info()
        state = self.get_state_summary()
        payload = {
            "account": account or {"error": "No connection"},
            "bot": state,
            "logs": self.logger.logs[-50:],
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(self._state_file, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except IOError:
            pass

    def _enter_cooldown(self):
        self.state = self.STATES["COOLDOWN"]
        self._cooldown_until = datetime.now() + \
            timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC)
        self.logger.info(
            f"Cooldown for {cfg.RE_ENTRY_COOLDOWN_SEC}s until "
            f"{self._cooldown_until.strftime('%H:%M:%S')}"
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
        count = self.trade_executor.close_all_bot_positions()
        self.position_manager.refresh()
        self.state = self.STATES["COOLDOWN"]
        self._enter_cooldown()
        return count

    def update_settings(self, settings: Dict):
        if "max_daily_loss" in settings:
            self.risk_manager.max_daily_loss = float(settings["max_daily_loss"])
        if "max_event_loss" in settings:
            self.risk_manager.max_event_loss = float(settings["max_event_loss"])
        if "max_trades_per_event" in settings:
            self.risk_manager.max_trades_per_event = int(settings["max_trades_per_event"])
        if "max_trades_per_session" in settings:
            self.risk_manager.max_trades_per_session = int(settings["max_trades_per_session"])
        if "cooldown_seconds" in settings:
            self.risk_manager.cooldown_seconds = int(settings["cooldown_seconds"])
        self.logger.info("Bot settings updated")

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
