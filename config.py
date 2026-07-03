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
MAX_LOT = _env_float("MAX_LOT", 1.0)
LOT_STEP = _env_float("LOT_STEP", 0.01)
LOT_MULTIPLIER = _env_int("LOT_MULTIPLIER", 5)

# Minimum balance to start trading
MIN_BALANCE = _env_float("MIN_BALANCE", 10.0)

# Risk limits
MAX_DAILY_LOSS_USD = _env_float("MAX_DAILY_LOSS_USD", 2.00)
MAX_EVENT_LOSS_USD = _env_float("MAX_EVENT_LOSS_USD", 1.00)
MAX_TRADES_PER_EVENT = _env_int("MAX_TRADES_PER_EVENT", 3)
MAX_TRADES_PER_SESSION = _env_int("MAX_TRADES_PER_SESSION", 3)
MAX_CONSECUTIVE_LOSSES = _env_int("MAX_CONSECUTIVE_LOSSES", 2)

# Cooldown
RE_ENTRY_COOLDOWN_SEC = _env_int("RE_ENTRY_COOLDOWN_SEC", 600)
BIAS_UPDATE_INTERVAL_SEC = _env_int("BIAS_UPDATE_INTERVAL_SEC", 60)

# Signal thresholds
MIN_BREAKOUT_SCORE = _env_float("MIN_BREAKOUT_SCORE", 0.02)
REQUIRE_RECENT_PULLBACK = _env_bool("REQUIRE_RECENT_PULLBACK", False)
RECENT_PULLBACK_LOOKBACK = _env_int("RECENT_PULLBACK_LOOKBACK", 5)
BIAS_STRENGTH_MIN = _env_float("BIAS_STRENGTH_MIN", 0.3)
ATR_MULTIPLIER = _env_float("ATR_MULTIPLIER", 1.0)
ATR_PERIOD = _env_int("ATR_PERIOD", 14)

# Exit mode and thresholds
EXIT_MODE = _env_int("EXIT_MODE", 5)  # 5=peak harvest (no hard SL), 6=multi-TP zone
EXIT_THRESHOLD_TIGHT = _env_float("EXIT_THRESHOLD_TIGHT", 0.50)
EXIT_MOMENTUM_THRESHOLD = _env_float("EXIT_MOMENTUM_THRESHOLD", 0.30)

# Peak harvest exit (mode 5) — no hard SL, trail only after significant profit
PEAK_HARVEST_TRAIL_TRIGGER = _env_float("PEAK_HARVEST_TRAIL_TRIGGER", 2.0)
PEAK_HARVEST_TRAIL_RETRACE = _env_float("PEAK_HARVEST_TRAIL_RETRACE", 0.50)
PEAK_HARVEST_MIN_BARS_EXIT = _env_int("PEAK_HARVEST_MIN_BARS_EXIT", 10)
PEAK_HARVEST_MOMENTUM_THRESHOLD = _env_float("PEAK_HARVEST_MOMENTUM_THRESHOLD", 0.85)
PEAK_HARVEST_MAX_HOLD_BARS = _env_int("PEAK_HARVEST_MAX_HOLD_BARS", 48)
DIRECTION_LOSS_LOOKBACK = _env_int("DIRECTION_LOSS_LOOKBACK", 5)
DIRECTION_LOSS_STREAK = _env_int("DIRECTION_LOSS_STREAK", 3)

# Multi-TP zone exit (mode 6) — structural SL + TP1/TP2/TP3 targets
SIGNAL_ENTRY_THRESHOLD = _env_float("SIGNAL_ENTRY_THRESHOLD", 0.10)
SL_ATR_MULTIPLIER = _env_float("SL_ATR_MULTIPLIER", 1.0)
TP1_MULTIPLIER = _env_float("TP1_MULTIPLIER", 2.0)
TP2_MULTIPLIER = _env_float("TP2_MULTIPLIER", 4.0)
TP3_MULTIPLIER = _env_float("TP3_MULTIPLIER", 6.0)
TP_CLOSE_THRESHOLD = _env_float("TP_CLOSE_THRESHOLD", 0.8)
TP_CLOSE_MOMENTUM_MIN = _env_float("TP_CLOSE_MOMENTUM_MIN", 0.25)

# Gemini AI advisor
GEMINI_API_KEY = _env_str("GEMINI_API_KEY", "")
GEMINI_ENABLED = _env_bool("GEMINI_ENABLED", False)
GEMINI_MODEL = _env_str("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_ADVICE_WEIGHT = _env_float("GEMINI_ADVICE_WEIGHT", 1.0)

# Multi-symbol scanning (comma-separated list, e.g. "XAUUSD,XAGUSD")
SYMBOLS = [s.strip() for s in _env_str("SYMBOLS", "XAUUSD").split(",")]

# Filters
MAX_SPREAD_PIPS = _env_float("MAX_SPREAD_PIPS", 35.0)
MIN_VOLATILITY_PIPS = _env_float("MIN_VOLATILITY_PIPS", 5.0)
MAX_VOLATILITY_PIPS = _env_float("MAX_VOLATILITY_PIPS", 200.0)
MIN_ATR_PIPS = _env_float("MIN_ATR_PIPS", 5.0)
ALLOWED_SESSIONS = _env_str("ALLOWED_SESSIONS", "LONDON,NEW_YORK")
MIN_CANDLE_BODY_RATIO = _env_float("MIN_CANDLE_BODY_RATIO", 0.4)

# Deviation / slippage
MAX_SLIPPAGE_PIPS = _env_int("MAX_SLIPPAGE_PIPS", 10)

# API
API_HOST = _env_str("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)

# Timeframes used by the system (Capital.com API IDs: 16385=HOUR, 16408=4HOUR)
BIAS_TIMEFRAME = _env_int("BIAS_TIMEFRAME", 16385)
SIGNAL_TIMEFRAME = _env_int("SIGNAL_TIMEFRAME", 1)
STRUCTURE_TIMEFRAMES = [16385, 16408]

# Market hours (spot gold: Sun 23:00 UTC - Fri 22:00 UTC)
MARKET_OPEN_SUNDAY_UTC = _env_int("MARKET_OPEN_SUNDAY_UTC", 23)
MARKET_CLOSE_FRIDAY_UTC = _env_int("MARKET_CLOSE_FRIDAY_UTC", 22)


# Meta-strategy (adaptive threshold / regime switching)
META_ENABLED = _env_bool("META_ENABLED", True)
META_LOOKBACK_WINDOW = _env_int("META_LOOKBACK_WINDOW", 20)
META_MIN_TRADES_FOR_REGIME = _env_int("META_MIN_TRADES_FOR_REGIME", 10)
META_THRESHOLD_MIN = _env_float("META_THRESHOLD_MIN", 0.03)
META_THRESHOLD_MAX = _env_float("META_THRESHOLD_MAX", 0.30)

# ML direction prediction
ML_CONFIDENCE_THRESHOLD = _env_float("ML_CONFIDENCE_THRESHOLD", 0.60)
ML_MODEL_PATH = _env_str("ML_MODEL_PATH", "models/direction_xgb_m5.joblib")
ML_BUY_MODEL_PATH = _env_str("ML_BUY_MODEL_PATH", "models/buy_sltp_xgb.joblib")
ML_SELL_MODEL_PATH = _env_str("ML_SELL_MODEL_PATH", "models/sell_sltp_xgb.joblib")
ML_M1_HISTORY_BARS = _env_int("ML_M1_HISTORY_BARS", 500)
ML_BIAS_OVERRIDE_THRESHOLD = _env_float("ML_BIAS_OVERRIDE_THRESHOLD", 0.70)
ML_HOLD_CONFIDENCE = _env_float("ML_HOLD_CONFIDENCE", 0.50)  # min ML confidence to hold exit (suppress momentum_decay/direction_loss)


# Adaptive confirmation (vreg_vtight)
ADAPTIVE_CONFIRMATION_ENABLED = _env_bool("ADAPTIVE_CONFIRMATION_ENABLED", True)
ADAPTIVE_CONF_WINDOW = _env_int("ADAPTIVE_CONF_WINDOW", 100)
ADAPTIVE_CONF_P_LOW = _env_int("ADAPTIVE_CONF_P_LOW", 60)
ADAPTIVE_CONF_P_NORM = _env_int("ADAPTIVE_CONF_P_NORM", 40)
ADAPTIVE_CONF_P_HIGH = _env_int("ADAPTIVE_CONF_P_HIGH", -1)  # -1 = no filter

# Aggressive sizing (score-based lot boost)
AGGRESSIVE_SIZING_ENABLED = _env_bool("AGGRESSIVE_SIZING_ENABLED", True)
AGGRESSIVE_STRONG_THRESHOLD = _env_float("AGGRESSIVE_STRONG_THRESHOLD", 0.30)
AGGRESSIVE_VERY_STRONG_THRESHOLD = _env_float("AGGRESSIVE_VERY_STRONG_THRESHOLD", 0.50)
AGGRESSIVE_STRONG_LOT_MULT = _env_float("AGGRESSIVE_STRONG_LOT_MULT", 2.0)
AGGRESSIVE_VERY_STRONG_LOT_MULT = _env_float("AGGRESSIVE_VERY_STRONG_LOT_MULT", 3.0)

# MaxelPay
MAXELPAY_API_KEY = _env_str("MAXELPAY_API_KEY", "")

# USD to NGN exchange rate for Paystack payments
USD_TO_NGN_RATE = _env_float("USD_TO_NGN_RATE", 1500.0)


def is_market_open() -> bool:
    from datetime import datetime as _dt
    now = _dt.utcnow()
    wd = now.weekday()
    h = now.hour + now.minute / 60.0
    if wd == 6:
        return h >= MARKET_OPEN_SUNDAY_UTC
    if wd == 4:
        return h < MARKET_CLOSE_FRIDAY_UTC
    return 0 <= wd <= 3
