import asyncio
import json
import os
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List

from app.bot import Bot
from app.logger import BotLogger


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
        os.makedirs(STATE_DIR, exist_ok=True)

    def start(self, identifier: str, api_key: str, password: str, demo: bool = True) -> Dict:
        ident = _fmt_id(identifier)
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
                loop.call_soon_threadsafe(loop.stop)
            return {"success": True, "message": "Bot stopped"}

    def get_state(self, identifier: str) -> Optional[Dict]:
        ident = _fmt_id(identifier)
        state_file = os.path.join(STATE_DIR, f"{ident}.json")
        if not os.path.exists(state_file):
            return None
        try:
            with open(state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

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
                    "time": datetime.utcnow().strftime("%H:%M:%S"),
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
            merged = dev_logs + bot_logs
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

    def is_running(self, identifier: str) -> bool:
        return _fmt_id(identifier) in self._bots

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
            return

        self._write_state(ident, {
            "state": bot.state,
            "started_at": datetime.utcnow().isoformat(),
            "symbol": bot.symbol,
        })

        await bot.run()

    def _write_state(self, ident: str, extra: Optional[Dict] = None):
        with self._lock:
            bot = self._bots.get(ident)
            if bot is None:
                return
            account = bot.client.get_account_info()
            state = bot.get_state_summary() if hasattr(bot, 'get_state_summary') else {}
            payload = {
                "account": account or {"error": "No connection"},
                "bot": state,
                "logs": bot.logger.logs[-50:],
                "timestamp": datetime.utcnow().isoformat(),
            }
            if extra:
                payload.update(extra)
            state_file = os.path.join(STATE_DIR, f"{ident}.json")
            try:
                with open(state_file, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
            except IOError:
                pass
