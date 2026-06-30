import asyncio
import json
import os
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List

from app.bot import Bot
from app.capital_client import CapitalClient
from app.logger import BotLogger
from app.subscription import can_start_live


STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bot_states")


def _fmt_id(raw: str) -> str:
    return raw.strip().lower()


class BotPool:
    def __init__(self):
        self._bots: Dict[str, Bot] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._loops: Dict[str, asyncio.AbstractEventLoop] = {}
        self._device_logs: Dict[str, List[Dict]] = {}
        self._lock = threading.Lock()
        self._last_sub_warning: Dict[str, str] = {}
        self._state_cache: Dict[str, Dict] = {}
        self._state_cache_ts: Dict[str, float] = {}
        os.makedirs(STATE_DIR, exist_ok=True)

    def start(self, identifier: str, api_key: str, password: str, demo: bool = True) -> Dict:
        ident = _fmt_id(identifier)
        with self._lock:
            if ident in self._bots:
                return {"success": False, "error": "Bot already running for this account"}

        temp = CapitalClient()
        ok = temp.initialize(api_key=api_key, identifier=identifier, password=password, demo=demo)
        err_msg = temp.last_error()[1] if not ok else ""
        temp.shutdown()
        if not ok:
            return {"success": False, "error": f"Broker authentication failed: {err_msg}"}

        with self._lock:
            if ident in self._bots:
                return {"success": False, "error": "Bot already running for this account"}

            loop = asyncio.new_event_loop()
            bot = Bot(logger=BotLogger(f"Bot-{ident[:8]}"))
            bot._account_id = ident
            bot._state_file = os.path.join(STATE_DIR, f"{ident}.json")

            self._bots[ident] = bot
            self._loops[ident] = loop

            creds = {"api_key": api_key, "identifier": identifier, "password": password, "demo": demo}
            t = threading.Thread(target=self._run_bot_thread, args=(ident, bot, loop, creds), daemon=True)
            self._threads[ident] = t
            t.start()

            return {"success": True, "message": "Bot started"}

    def stop(self, identifier: str) -> Dict:
        ident = _fmt_id(identifier)
        with self._lock:
            bot = self._bots.pop(ident, None)
            loop = self._loops.pop(ident, None)
            thread = self._threads.pop(ident, None)
            if bot is None:
                return {"success": False, "error": "No bot running for this account"}
            if loop and loop.is_running():
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                loop.call_soon_threadsafe(loop.stop)
            return {"success": True, "message": "Bot stopped"}

    def is_running(self, identifier: str) -> bool:
        ident = _fmt_id(identifier)
        with self._lock:
            return ident in self._bots

    def open_count(self, identifier: str) -> int:
        ident = _fmt_id(identifier)
        with self._lock:
            bot = self._bots.get(ident)
            if bot is None:
                return 0
            return getattr(bot.position_manager, "open_count", 0)

    def get_state(self, identifier: str) -> Optional[Dict]:
        ident = _fmt_id(identifier)
        now = time.time()
        with self._lock:
            cached = self._state_cache.get(ident)
            ts = self._state_cache_ts.get(ident, 0)
            if cached is not None and now - ts < 2.0:
                return cached
        state_file = os.path.join(STATE_DIR, f"{ident}.json")
        if not os.path.exists(state_file):
            return None
        try:
            with open(state_file) as f:
                data = json.load(f)
            with self._lock:
                self._state_cache[ident] = data
                self._state_cache_ts[ident] = now
            return data
        except (json.JSONDecodeError, IOError):
            return None

    def add_log_once(self, identifier: str, message: str, level: str = "INFO") -> bool:
        ident = _fmt_id(identifier)
        with self._lock:
            key = f"{ident}|{level}|{message}"
            last = self._last_sub_warning.get(ident)
            if last == key:
                return False
            self._last_sub_warning[ident] = key
        self.add_log(identifier, message, level)
        return True

    def add_log(self, identifier: str, message: str, level: str = "INFO"):
        ident = _fmt_id(identifier)
        with self._lock:
            bot = self._bots.get(ident)
            if bot:
                getattr(bot.logger, level.lower(), bot.logger.info)(message)
            else:
                if ident not in self._device_logs:
                    self._device_logs[ident] = []
                self._device_logs[ident].append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "message": message,
                    "level": level,
                })
                if len(self._device_logs[ident]) > 200:
                    self._device_logs[ident][:100] = []

    def get_logs(self, identifier: str, count: int = 50) -> List[Dict]:
        ident = _fmt_id(identifier)
        with self._lock:
            bot_logs = []
            bot = self._bots.get(ident)
            if bot:
                bot_logs = bot.logger.logs[-count:]
            dev_logs = self._device_logs.get(ident, [])
            merged = sorted(dev_logs + bot_logs, key=lambda e: e.get("time", ""))
            return merged[-count:]

    def list(self) -> List[Dict]:
        with self._lock:
            result = []
            for ident, bot in self._bots.items():
                state = self.get_state(ident)
                result.append({
                    "identifier": ident,
                    "running": True,
                    "state": (state or {}).get("state", "UNKNOWN"),
                })
            return result

    async def emergency_close(self, identifier: str) -> int:
        ident = _fmt_id(identifier)
        with self._lock:
            bot = self._bots.get(ident)
            if bot is None:
                return 0
        return await bot.emergency_close()

    def update_settings(self, identifier: str, settings: Dict) -> Dict:
        ident = _fmt_id(identifier)
        with self._lock:
            bot = self._bots.get(ident)
            if bot is None:
                return {"success": False, "error": "Bot not running"}
            bot.update_settings(settings)
            return {"success": True, "message": "Settings updated"}

    def _remove_bot(self, ident: str):
        with self._lock:
            self._bots.pop(ident, None)
            self._loops.pop(ident, None)
            self._threads.pop(ident, None)

    def stop_all(self):
        with self._lock:
            ids = list(self._bots.keys())
        for ident in ids:
            self.stop(ident)

    def _run_bot_thread(self, ident: str, bot: Bot, loop: asyncio.AbstractEventLoop, creds: Dict):
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_bot(ident, bot, creds))
        except Exception as e:
            bot.logger.error(f"Bot thread error: {e}")
        finally:
            self._remove_bot(ident)
            loop.close()

    async def _run_bot(self, ident: str, bot: Bot, creds: Dict):
        ok = await bot.initialize_with_credentials(
            api_key=creds["api_key"],
            identifier=creds["identifier"],
            password=creds["password"],
            demo=creds["demo"],
        )
        if not ok:
            bot.logger.error("Failed to initialize bot with credentials")
            self._write_state(ident, {"state": "STOPPED", "error": "init_failed"})
            self._remove_bot(ident)
            return

        if not creds.get("demo", True):
            orig_ident = creds.get("identifier", ident)
            async def _sub_check():
                try:
                    return await can_start_live(orig_ident, 0.0)
                except Exception:
                    return True
            bot.set_can_trade_callback(_sub_check)

        self._write_state(ident, {
            "state": bot.state,
            "started_at": datetime.utcnow().isoformat(),
            "symbol": bot.symbol,
        })

        await bot.run()
        self._remove_bot(ident)

    def _write_state(self, ident: str, extra: Optional[Dict] = None):
        with self._lock:
            bot = self._bots.get(ident)
        if bot is None:
            return
        account = bot.client.get_account_info() or {"error": "No connection"}
        symbol_info = bot.client.get_symbol_info(bot.symbol) if bot.client else {}
        if symbol_info:
            account["bid"] = symbol_info.get("bid", 0)
            account["ask"] = symbol_info.get("ask", 0)
        state = bot.get_state_summary() if hasattr(bot, 'get_state_summary') else {}
        payload = {
            "account": account,
            "bot": state,
            "logs": bot.logger.logs[-50:],
            "timestamp": datetime.utcnow().isoformat(),
        }
        if extra:
            payload.update(extra)
        state_file = os.path.join(STATE_DIR, f"{ident}.json")
        try:
            tmp = state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, state_file)
        except IOError:
            pass
        with self._lock:
            self._state_cache[ident] = payload
            self._state_cache_ts[ident] = time.time()
