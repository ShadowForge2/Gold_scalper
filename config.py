import os
from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


# Broker selection: "MT5" or "CAPITAL"
BROKER = _env_str("BROKER", "MT5").upper()

# MT5 Account (only used if BROKER=MT5)
MT5_ACCOUNT = _env_str("MT5_ACCOUNT", "49717207")
MT5_PASSWORD = _env_str("MT5_PASSWORD", "Thunder@g1")
MT5_SERVER = _env_str("MT5_SERVER", "HFMarketsGlobal-Demo")

# Capital.com Account (only used if BROKER=CAPITAL)
CAPITAL_API_KEY = _env_str("CAPITAL_API_KEY", "")
CAPITAL_IDENTIFIER = _env_str("CAPITAL_IDENTIFIER", "")
CAPITAL_PASSWORD = _env_str("CAPITAL_PASSWORD", "")
CAPITAL_DEMO = _env_bool("CAPITAL_DEMO", True)
CAPITAL_EPIC = _env_str("CAPITAL_EPIC", "GOLD")

# Trading symbol
SYMBOL = _env_str("SYMBOL", "XAUUSD")
MAGIC_NUMBER = _env_int("MAGIC_NUMBER", 123456)
COMMENT = _env_str("COMMENT", "Gold Scalper")

# Lot sizing
LOT_SIZE = _env_float("LOT_SIZE", 0.01)
MIN_LOT = _env_float("MIN_LOT", 0.01)
MAX_LOT = _env_float("MAX_LOT", 1.0)
LOT_STEP = _env_float("LOT_STEP", 0.01)
LOT_MULTIPLIER = _env_int("LOT_MULTIPLIER", 5)

# Risk limits
MAX_DAILY_LOSS_USD = _env_float("MAX_DAILY_LOSS_USD", 2.00)
MAX_EVENT_LOSS_USD = _env_float("MAX_EVENT_LOSS_USD", 1.00)
MAX_TRADES_PER_EVENT = _env_int("MAX_TRADES_PER_EVENT", 1)
MAX_TRADES_PER_SESSION = _env_int("MAX_TRADES_PER_SESSION", 3)
MAX_CONSECUTIVE_LOSSES = _env_int("MAX_CONSECUTIVE_LOSSES", 2)

# Cooldown
RE_ENTRY_COOLDOWN_SEC = _env_int("RE_ENTRY_COOLDOWN_SEC", 600)
BIAS_UPDATE_INTERVAL_SEC = _env_int("BIAS_UPDATE_INTERVAL_SEC", 60)

# Signal thresholds
SIGNAL_ENTRY_THRESHOLD = _env_float("SIGNAL_ENTRY_THRESHOLD", 0.50)
BIAS_STRENGTH_MIN = _env_float("BIAS_STRENGTH_MIN", 0.3)

# Exit thresholds
EXIT_THRESHOLD_TIGHT = _env_float("EXIT_THRESHOLD_TIGHT", 0.50)
EXIT_MOMENTUM_THRESHOLD = _env_float("EXIT_MOMENTUM_THRESHOLD", 0.30)

# Filters
MAX_SPREAD_PIPS = _env_float("MAX_SPREAD_PIPS", 50.0)
MIN_VOLATILITY_PIPS = _env_float("MIN_VOLATILITY_PIPS", 5.0)
MAX_VOLATILITY_PIPS = _env_float("MAX_VOLATILITY_PIPS", 200.0)
ALLOWED_SESSIONS = _env_str("ALLOWED_SESSIONS", "LONDON,NEW_YORK")
MIN_CANDLE_BODY_RATIO = _env_float("MIN_CANDLE_BODY_RATIO", 0.4)

# Deviation / slippage
MAX_SLIPPAGE_PIPS = _env_int("MAX_SLIPPAGE_PIPS", 10)

# API
API_HOST = _env_str("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)

# Timeframes used by the system
BIAS_TIMEFRAME = _env_int("BIAS_TIMEFRAME", 16385)  # mt5.TIMEFRAME_H1
SIGNAL_TIMEFRAME = _env_int("SIGNAL_TIMEFRAME", 1)  # mt5.TIMEFRAME_M1
STRUCTURE_TIMEFRAMES = [16385, 16408]  # H1, H4
