import logging
from datetime import datetime
from typing import List, Dict


class BotLogger:
    def __init__(self, name: str = "GoldScalper"):
        self.logs: List[Dict] = []
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self._logger.addHandler(handler)

    def _log(self, msg: str, level: str = "INFO"):
        getattr(self._logger, level.lower(), self._logger.info)(msg)
        self.logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": msg,
            "level": level,
        })
        if len(self.logs) > 200:
            self.logs[:100] = []

    def info(self, msg: str):
        self._log(msg, "INFO")

    def warning(self, msg: str):
        self._log(msg, "WARNING")

    def error(self, msg: str):
        self._log(msg, "ERROR")

    def trade(self, msg: str):
        self._log(f"[TRADE] {msg}", "INFO")

    def signal(self, msg: str):
        self._log(f"[SIGNAL] {msg}", "INFO")

    def bias(self, msg: str):
        self._log(f"[BIAS] {msg}", "INFO")
