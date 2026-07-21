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
LOT_SIZE = _env_float("LOT_SIZE", 0.02)
MIN_LOT = _env_float("MIN_LOT", 0.02)
MAX_LOT = 9999.0
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

EXIT_CHECK_INTERVAL = _env_int("EXIT_CHECK_INTERVAL", 300)  # 5 min (M5 bar)

# Multi-TP zone exit (mode 6)
SL_ATR_MULTIPLIER = _env_float("SL_ATR_MULTIPLIER", 1.0)
TP1_MULTIPLIER = _env_float("TP1_MULTIPLIER", 2.0)
TP2_MULTIPLIER = _env_float("TP2_MULTIPLIER", 4.0)
TP3_MULTIPLIER = _env_float("TP3_MULTIPLIER", 6.0)
TP_CLOSE_THRESHOLD = _env_float("TP_CLOSE_THRESHOLD", 0.8)
TP_CLOSE_MOMENTUM_MIN = _env_float("TP_CLOSE_MOMENTUM_MIN", 0.25)

# Event loss — percentage of balance (scales with equity)
MAX_EVENT_LOSS_PCT = _env_float("MAX_EVENT_LOSS_PCT", 5.0)  # 5% of balance

# Meta strategy
META_LOOKBACK_WINDOW = _env_int("META_LOOKBACK_WINDOW", 20)
META_THRESHOLD_MIN = _env_float("META_THRESHOLD_MIN", 0.03)
META_THRESHOLD_MAX = _env_float("META_THRESHOLD_MAX", 0.30)
META_ENABLED = _env_bool("META_ENABLED", False)
META_MIN_TRADES_FOR_REGIME = _env_int("META_MIN_TRADES_FOR_REGIME", 5)
SIGNAL_ENTRY_THRESHOLD = _env_float("SIGNAL_ENTRY_THRESHOLD", 0.10)
MAX_TRADES_PER_EVENT = _env_int("MAX_TRADES_PER_EVENT", 6)
BIAS_STRENGTH_MIN = _env_float("BIAS_STRENGTH_MIN", 0.3)
MAX_LOT = _env_float("MAX_LOT", 9999.0)

# Aggressive sizing
AGGRESSIVE_SIZING_ENABLED = _env_bool("AGGRESSIVE_SIZING_ENABLED", True)
AGGRESSIVE_VERY_STRONG_THRESHOLD = _env_float("AGGRESSIVE_VERY_STRONG_THRESHOLD", 0.50)
AGGRESSIVE_VERY_STRONG_LOT_MULT = _env_float("AGGRESSIVE_VERY_STRONG_LOT_MULT", 2.0)
AGGRESSIVE_STRONG_THRESHOLD = _env_float("AGGRESSIVE_STRONG_THRESHOLD", 0.30)
AGGRESSIVE_STRONG_LOT_MULT = _env_float("AGGRESSIVE_STRONG_LOT_MULT", 1.5)

# ML override
ML_OVERRIDE_MAX_PER_SESSION = _env_int("ML_OVERRIDE_MAX_PER_SESSION", 3)

# Multi-symbol: bot trades both XAUUSD and US100 automatically
SYMBOLS = [s.strip() for s in _env_str("SYMBOLS", "XAUUSD,US100").split(",")]

# Per-symbol model paths
ASP_MODEL_PATHS = {
    "XAUUSD": _env_str("ASP_MODEL_PATH", "models/asp_swing_xgb_m5.joblib"),
    "US100": _env_str("ASP_MODEL_PATH_US100", "models/asp_swing_xgb_m5_US100.joblib"),
}
ASP_FEATURE_PATHS = {
    "XAUUSD": _env_str("ASP_FEATURE_PATH", "models/asp_swing_m5_features.npy"),
    "US100": _env_str("ASP_FEATURE_PATH_US100", "models/asp_swing_m5_features_US100.npy"),
}
SWING_QUALITY_MODEL_PATHS = {
    "XAUUSD": _env_str("SWING_QUALITY_MODEL_PATH", "models/swing_quality_xgb.json"),
    "US100": _env_str("SWING_QUALITY_MODEL_PATH_US100", "models/swing_quality_xgb_US100.json"),
}

# Per-symbol lot sizes
SYMBOL_LOT_SIZES = {
    "XAUUSD": _env_float("LOT_SIZE_XAUUSD", 0.02),
    "US100": _env_float("LOT_SIZE_US100", 0.02),
}

# Per-symbol spread limits (pips)
SYMBOL_MAX_SPREAD = {
    "XAUUSD": _env_float("MAX_SPREAD_PIPS", 35.0),
    "US100": _env_float("MAX_SPREAD_PIPS_US100", 50.0),
}

# Per-symbol max drift (absolute price units, not pips)
# XAUUSD: $0.50 (gold H1 range ~$7-10)
# US100: $5.00 (US100 H1 range ~50-100 points)
SYMBOL_MAX_DRIFT = {
    "XAUUSD": _env_float("MAX_DRIFT_XAUUSD", 0.50),
    "US100": _env_float("MAX_DRIFT_US100", 5.00),
}

# Per-symbol ASP timeout (in M5 bars; 10 = 50 minutes)
SYMBOL_ASP_TIMEOUT_BARS = {
    "XAUUSD": _env_int("ASP_TIMEOUT_BARS_XAUUSD", 10),
    "US100": _env_int("ASP_TIMEOUT_BARS_US100", 10),
}

# Adaptive confirmation
ADAPTIVE_CONFIRMATION_ENABLED = _env_bool("ADAPTIVE_CONFIRMATION_ENABLED", True)
ADAPTIVE_CONF_WINDOW = _env_int("ADAPTIVE_CONF_WINDOW", 200)
ADAPTIVE_CONF_P_LOW = _env_int("ADAPTIVE_CONF_P_LOW", 60)
ADAPTIVE_CONF_P_NORM = _env_int("ADAPTIVE_CONF_P_NORM", 40)
ADAPTIVE_CONF_P_HIGH = _env_int("ADAPTIVE_CONF_P_HIGH", 0)

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

# US100 market hours (Mon-Fri 14:30-21:00 UTC)
US100_OPEN_HOUR_UTC = 14
US100_OPEN_MINUTE_UTC = 30
US100_CLOSE_HOUR_UTC = 21


# ASP (Adaptive Swing Probability) model
ASP_ENABLED = _env_bool("ASP_ENABLED", True)
ASP_MODEL_PATH = _env_str("ASP_MODEL_PATH", "models/asp_swing_xgb_m5.joblib")
ASP_FEATURE_PATH = _env_str("ASP_FEATURE_PATH", "models/asp_swing_m5_features.npy")
ASP_SL_ATR_MULTIPLIER = _env_float("ASP_SL_ATR_MULTIPLIER", 2.0)
ASP_TP_ATR_MULTIPLIER = _env_float("ASP_TP_ATR_MULTIPLIER", 1.0)
ASP_TIMEOUT_BARS = _env_int("ASP_TIMEOUT_BARS", 5)  # 5 x 5min = 25min
ASP_MIN_ATR_DIST = _env_float("ASP_MIN_ATR_DIST", 0.50)
ASP_MIN_CONFIDENCE = _env_float("ASP_MIN_CONFIDENCE", 0.65)
ASP_M1_HISTORY_BARS = _env_int("ASP_M1_HISTORY_BARS", 300)

# Trailing stop for ASP trades
ASP_TRAILING_ENABLED = _env_bool("ASP_TRAILING_ENABLED", True)
ASP_TRAILING_TRIGGER_ATR = _env_float("ASP_TRAILING_TRIGGER_ATR", 1.5)  # activate after 1.5x ATR profit (~1R)
ASP_TRAILING_RETRACE_ATR = _env_float("ASP_TRAILING_RETRACE_ATR", 2.5)  # trail 2.5x ATR behind best (2-3x ATR)
ASP_TRAIL_MIN_BARS = _env_int("ASP_TRAIL_MIN_BARS", 4)  # minimum M1 bars held before trailing activates
ASP_TRAIL_ADX_MIN = _env_float("ASP_TRAIL_ADX_MIN", 20.0)  # minimum ADX for trailing (0 = disabled)

# Chop filter — reject ASP signals when market is stagnant
CHOP_FILTER_ENABLED = _env_bool("CHOP_FILTER_ENABLED", True)
CHOP_THRESHOLD = _env_float("CHOP_THRESHOLD", 0.70)  # validated: 1.59x movement ratio at this cutoff

# Swing quality XGBoost model — confirmation gate
SWING_QUALITY_ENABLED = _env_bool("SWING_QUALITY_ENABLED", True)
SWING_QUALITY_MODEL_PATH = _env_str("SWING_QUALITY_MODEL_PATH", "models/swing_quality_xgb.json")
SWING_QUALITY_THRESHOLD = _env_float("SWING_QUALITY_THRESHOLD", 0.40)

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


def is_market_open_for_symbol(sym: str) -> bool:
    """Check if market is open for a specific symbol."""
    from datetime import datetime as _dt
    now = _dt.utcnow()
    wd = now.weekday()
    h = now.hour + now.minute / 60.0

    if sym in ("US100", "NASDAQ", "NAS100"):
        # US100: Mon-Fri 14:30-21:00 UTC
        if 0 <= wd <= 4:
            open_t = US100_OPEN_HOUR_UTC + US100_OPEN_MINUTE_UTC / 60.0
            close_t = US100_CLOSE_HOUR_UTC
            return open_t <= h < close_t
        return False

    # XAUUSD: Sun 23:00 - Fri 21:00 UTC, closed 20:59-22:00 Mon-Thu
    if wd == 6:
        return h >= MARKET_OPEN_SUNDAY_UTC
    if wd == 4:
        return h < MARKET_CLOSE_FRIDAY_UTC
    if 0 <= wd <= 3:
        if MARKET_DAILY_CLOSE_START <= h < MARKET_DAILY_CLOSE_END:
            return False
        return True
    return False
