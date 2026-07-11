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


# Broker
BROKER = _env_str("BROKER", "CAPITAL").upper()

# Capital.com Account
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
MAX_LOT = 1.0
LOT_STEP = _env_float("LOT_STEP", 0.01)
LOT_MULTIPLIER = _env_int("LOT_MULTIPLIER", 5)

# Minimum balance to start trading
MIN_BALANCE = _env_float("MIN_BALANCE", 10.0)

# Signal — ML confidence is the only entry gate
MIN_BREAKOUT_SCORE = _env_float("MIN_BREAKOUT_SCORE", 0.02)
ATR_MULTIPLIER = _env_float("ATR_MULTIPLIER", 1.0)
ATR_PERIOD = _env_int("ATR_PERIOD", 14)
BIAS_UPDATE_INTERVAL_SEC = _env_int("BIAS_UPDATE_INTERVAL_SEC", 60)

# Exit mode and thresholds
EXIT_MODE = _env_int("EXIT_MODE", 5)  # 5=peak harvest (no hard SL), 6=multi-TP zone
EXIT_THRESHOLD_TIGHT = _env_float("EXIT_THRESHOLD_TIGHT", 0.50)
EXIT_MOMENTUM_THRESHOLD = _env_float("EXIT_MOMENTUM_THRESHOLD", 0.30)

# Peak harvest exit (mode 5)
PEAK_HARVEST_TRAIL_TRIGGER = _env_float("PEAK_HARVEST_TRAIL_TRIGGER", 2.0)
PEAK_HARVEST_TRAIL_RETRACE = _env_float("PEAK_HARVEST_TRAIL_RETRACE", 0.50)
PEAK_HARVEST_MIN_BARS_EXIT = _env_int("PEAK_HARVEST_MIN_BARS_EXIT", 10)
PEAK_HARVEST_MOMENTUM_THRESHOLD = _env_float("PEAK_HARVEST_MOMENTUM_THRESHOLD", 0.85)
PEAK_HARVEST_MAX_HOLD_BARS = _env_int("PEAK_HARVEST_MAX_HOLD_BARS", 48)
DIRECTION_LOSS_LOOKBACK = _env_int("DIRECTION_LOSS_LOOKBACK", 5)
DIRECTION_LOSS_STREAK = _env_int("DIRECTION_LOSS_STREAK", 3)

EXIT_CHECK_INTERVAL = _env_int("EXIT_CHECK_INTERVAL", 60)

# Multi-TP zone exit (mode 6)
SL_ATR_MULTIPLIER = _env_float("SL_ATR_MULTIPLIER", 1.0)
TP1_MULTIPLIER = _env_float("TP1_MULTIPLIER", 2.0)
TP2_MULTIPLIER = _env_float("TP2_MULTIPLIER", 4.0)
TP3_MULTIPLIER = _env_float("TP3_MULTIPLIER", 6.0)
TP_CLOSE_THRESHOLD = _env_float("TP_CLOSE_THRESHOLD", 0.8)
TP_CLOSE_MOMENTUM_MIN = _env_float("TP_CLOSE_MOMENTUM_MIN", 0.25)

# Multi-symbol scanning (comma-separated list, e.g. "XAUUSD,XAGUSD")
SYMBOLS = [s.strip() for s in _env_str("SYMBOLS", "XAUUSD").split(",")]

# Filters
MAX_SPREAD_PIPS = _env_float("MAX_SPREAD_PIPS", 35.0)
ALLOWED_SESSIONS = _env_str("ALLOWED_SESSIONS", "LONDON,NEW_YORK")

# Deviation / slippage
MAX_SLIPPAGE_PIPS = _env_int("MAX_SLIPPAGE_PIPS", 10)

# API
API_HOST = _env_str("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)

# Timeframes used by the system (Capital.com API IDs: 16385=HOUR, 16408=4HOUR)
BIAS_TIMEFRAME = _env_int("BIAS_TIMEFRAME", 16385)
SIGNAL_TIMEFRAME = _env_int("SIGNAL_TIMEFRAME", 1)
STRUCTURE_TIMEFRAMES = [16385, 16408]

# Market hours (Capital.com XAUUSD: Sun 23:00 UTC - Fri 21:00 UTC)
MARKET_OPEN_SUNDAY_UTC = _env_int("MARKET_OPEN_SUNDAY_UTC", 23)
MARKET_CLOSE_FRIDAY_UTC = _env_int("MARKET_CLOSE_FRIDAY_UTC", 21)

# Daily close window (XAUUSD closes 20:59-22:00 UTC Mon-Thu)
MARKET_DAILY_CLOSE_START = _env_float("MARKET_DAILY_CLOSE_START", 20.9833)  # 20:59 UTC
MARKET_DAILY_CLOSE_END = _env_float("MARKET_DAILY_CLOSE_END", 22.0)  # 22:00 UTC


# ML direction prediction
ML_CONFIDENCE_THRESHOLD = _env_float("ML_CONFIDENCE_THRESHOLD", 0.75)
ML_NO_TRADE_THRESHOLD = _env_float("ML_NO_TRADE_THRESHOLD", 0.50)
ML_MODEL_PATH = _env_str("ML_MODEL_PATH", "models/direction_xgb_m5.joblib")
ML_M1_HISTORY_BARS = _env_int("ML_M1_HISTORY_BARS", 500)
ML_EXIT_MODEL_PATH = _env_str("ML_EXIT_MODEL_PATH", "models/exit_xgb_m5.joblib")
ML_EXIT_HOLD_THRESHOLD = _env_float("ML_EXIT_HOLD_THRESHOLD", 0.60)
ML_BIAS_OVERRIDE_THRESHOLD = _env_float("ML_BIAS_OVERRIDE_THRESHOLD", 0.60)
ML_HOLD_CONFIDENCE = _env_float("ML_HOLD_CONFIDENCE", 0.50)

# Conviction tiers — ML confidence determines position count (no other filters)
ML_CONF_STRONG_THRESHOLD = _env_float("ML_CONF_STRONG_THRESHOLD", 0.82)
ML_CONF_VERY_STRONG_THRESHOLD = _env_float("ML_CONF_VERY_STRONG_THRESHOLD", 0.92)
ML_POSITIONS_STRONG = _env_int("ML_POSITIONS_STRONG", 3)
ML_POSITIONS_VERY_STRONG = _env_int("ML_POSITIONS_VERY_STRONG", 5)
ML_POSITIONS_MAX = _env_int("ML_POSITIONS_MAX", 7)

# Conviction sizing (lot multiplier per confidence tier)
ML_LOT_MULTIPLIER = _env_float("ML_LOT_MULTIPLIER", 1.5)
ML_LOT_MULT_STRONG = _env_float("ML_LOT_MULT_STRONG", 2.0)
ML_LOT_MULT_VERY_STRONG = _env_float("ML_LOT_MULT_VERY_STRONG", 3.0)

# MaxelPay
MAXELPAY_API_KEY = _env_str("MAXELPAY_API_KEY", "")

# USD to NGN exchange rate for Paystack payments
USD_TO_NGN_RATE = _env_float("USD_TO_NGN_RATE", 1500.0)


# News-aware trading
NEWS_AWARE_ENABLED = _env_bool("NEWS_AWARE_ENABLED", True)
NEWS_PRE_WINDOW_MINUTES = _env_int("NEWS_PRE_WINDOW_MINUTES", 15)
NEWS_SPIKE_WINDOW_MINUTES = _env_int("NEWS_SPIKE_WINDOW_MINUTES", 3)
NEWS_POST_WINDOW_MINUTES = _env_int("NEWS_POST_WINDOW_MINUTES", 60)
NEWS_CACHE_TTL_HOURS = _env_int("NEWS_CACHE_TTL_HOURS", 6)
NEWS_WIDER_SL_MULT = _env_float("NEWS_WIDER_SL_MULT", 1.5)
NEWS_WIDER_TP_MULT = _env_float("NEWS_WIDER_TP_MULT", 1.5)
NEWS_USER_EVENTS_PATH = _env_str("NEWS_USER_EVENTS_PATH", "data/user_events.json")
JBLANKED_API_KEY = _env_str("JBLANKED_API_KEY", "")
FINNHUB_API_KEY = _env_str("FINNHUB_API_KEY", "")

def is_market_open() -> bool:
    from datetime import datetime as _dt
    now = _dt.utcnow()
    wd = now.weekday()
    h = now.hour + now.minute / 60.0
    if wd == 6:
        return h >= MARKET_OPEN_SUNDAY_UTC
    if wd == 4:
        return h < MARKET_CLOSE_FRIDAY_UTC
    if 0 <= wd <= 3:
        if MARKET_DAILY_CLOSE_START <= h < MARKET_DAILY_CLOSE_END:
            return False
        return True
    return False
