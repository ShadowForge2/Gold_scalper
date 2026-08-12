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
LOT_MULTIPLIER = _env_int("LOT_MULTIPLIER", 2)
# Hard safety cap on the lot multiplier regardless of env/API value.
# Backtest (1:100 lev, $20 start): mult>2 blows the account up
# (x3 +8%, x4 -43%, x5 -70%). EquityScaler.get_lot() clamps to this.
MAX_LOT_MULTIPLIER = _env_float("MAX_LOT_MULTIPLIER", 2.0)

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

# Correlated symbols share ONE combined event-loss budget. US100 and US500
# move ~95% together, so two separate 5% budgets would let a single US-equity
# drawdown hit both symbols before tripping a stop. Grouping them means one
# budget of MAX_EVENT_LOSS_PCT covers both. Symbols not listed here get their
# own budget.
EVENT_LOSS_GROUP = {
    "XAUUSD": "XAUUSD",
    "US100": "US_EQUITY",
    "US500": "US_EQUITY",
    "US30": "US_EQUITY",
}

# Volatility regime — adaptive filters during high volatility
VOLATILITY_REGIME_ENABLED = _env_bool("VOLATILITY_REGIME_ENABLED", True)
VOLATILITY_ATR_MULT = _env_float("VOLATILITY_ATR_MULT", 1.5)  # current ATR > 1.5x avg = high vol
VOLATILITY_LOT_REDUCTION = _env_float("VOLATILITY_LOT_REDUCTION", 0.5)  # 50% lot during high vol
CONSECUTIVE_LOSS_SKIP = _env_int("CONSECUTIVE_LOSS_SKIP", 5)  # skip regime after N consecutive losses
CONSECUTIVE_LOSS_RESET_HOURS = _env_float("CONSECUTIVE_LOSS_RESET_HOURS", 4.0)  # reset counter after N hours

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

# Multi-symbol: bot trades XAUUSD, US100, US500, and US30 automatically.
# All pairs trade from MIN_BALANCE ($10) — the $30 gate was for the old
# single-pair strategy; we now poll all pairs and trade the best one.
# JP225/DE40 are momentum-engine pairs (per-pair adapted params + regime gate).
SYMBOLS = [s.strip() for s in _env_str("SYMBOLS", "XAUUSD,US100,JP225,DE40,US500,US30").split(",")]

# Per-symbol minimum balance before that symbol becomes tradeable.
SYMBOL_MIN_BALANCE = {
    "XAUUSD": _env_float("MIN_BALANCE_XAUUSD", MIN_BALANCE),
    "US100": _env_float("MIN_BALANCE_US100", MIN_BALANCE),
    "JP225": _env_float("MIN_BALANCE_JP225", MIN_BALANCE),
    "DE40": _env_float("MIN_BALANCE_DE40", MIN_BALANCE),
    "US500": _env_float("MIN_BALANCE_US500", MIN_BALANCE),
    "US30": _env_float("MIN_BALANCE_US30", MIN_BALANCE),
}

# Per-symbol lot sizes
SYMBOL_LOT_SIZES = {
    "XAUUSD": _env_float("LOT_SIZE_XAUUSD", 0.02),
    "US100": _env_float("LOT_SIZE_US100", 0.02),
    "JP225": _env_float("LOT_SIZE_JP225", 0.02),
    "DE40": _env_float("LOT_SIZE_DE40", 0.02),
    "US500": _env_float("LOT_SIZE_US500", 0.02),
    "US30": _env_float("LOT_SIZE_US30", 0.02),
}

# Per-symbol contract size (underlying units per 1.0 lot).
# Capital.com CFDs are micro-style: position payloads show contractSize=1 for
# commodities AND indices. So XAUUSD 1.0 lot = 1 troy oz (NOT 100 oz), and
# indices 1.0 lot = $1 per index point.
SYMBOL_CONTRACT_SIZE = {
    "XAUUSD": _env_float("CONTRACT_SIZE_XAUUSD", 1),
    "US100": _env_float("CONTRACT_SIZE_US100", 1),
    "JP225": _env_float("CONTRACT_SIZE_JP225", 1),
    "DE40": _env_float("CONTRACT_SIZE_DE40", 1),
    "US500": _env_float("CONTRACT_SIZE_US500", 1),
    "US30": _env_float("CONTRACT_SIZE_US30", 1),
}

# Per-symbol Capital.com asset class (leverage grouping from /accounts/preferences).
SYMBOL_ASSET_CLASS = {
    "XAUUSD": _env_str("ASSET_CLASS_XAUUSD", "COMMODITIES"),
    "US100": _env_str("ASSET_CLASS_US100", "INDICES"),
    "JP225": _env_str("ASSET_CLASS_JP225", "INDICES"),
    "DE40": _env_str("ASSET_CLASS_DE40", "INDICES"),
    "US500": _env_str("ASSET_CLASS_US500", "INDICES"),
    "US30": _env_str("ASSET_CLASS_US30", "INDICES"),
}

# Fallback margin requirement (decimal fraction of notional) if the API omits it.
SYMBOL_MARGIN_FACTOR = {
    "XAUUSD": _env_float("MARGIN_FACTOR_XAUUSD", 0.01),
    "US100": _env_float("MARGIN_FACTOR_US100", 0.01),
    "JP225": _env_float("MARGIN_FACTOR_JP225", 0.01),
    "DE40": _env_float("MARGIN_FACTOR_DE40", 0.01),
    "US500": _env_float("MARGIN_FACTOR_US500", 0.01),
    "US30": _env_float("MARGIN_FACTOR_US30", 0.01),
}

# Per-symbol spread limits (pips)
SYMBOL_MAX_SPREAD = {
    "XAUUSD": _env_float("MAX_SPREAD_PIPS_XAUUSD", 35.0),
    "US100": _env_float("MAX_SPREAD_PIPS_US100", 50.0),
    "JP225": _env_float("MAX_SPREAD_PIPS_JP225", 50.0),
    "DE40": _env_float("MAX_SPREAD_PIPS_DE40", 50.0),
    "US500": _env_float("MAX_SPREAD_PIPS_US500", 50.0),
    "US30": _env_float("MAX_SPREAD_PIPS_US30", 50.0),
}

# Per-symbol max drift (absolute price units, not pips)
# XAUUSD: $0.50 (gold H1 range ~$7-10)
# US100: $5.00 (US100 H1 range ~50-100 points)
# US500: $1.50 (S&P prices ~3.5x lower than Nasdaq, so the same relative
# drift is a much smaller absolute number — $5 would be ~0.08% of US500)
# US30: $10.00 (Dow prices ~2x Nasdaq / ~7x S&P, so drift scales up)
SYMBOL_MAX_DRIFT = {
    "XAUUSD": _env_float("MAX_DRIFT_XAUUSD", 0.50),
    "US100": _env_float("MAX_DRIFT_US100", 5.00),
    "JP225": _env_float("MAX_DRIFT_JP225", 10.00),
    "DE40": _env_float("MAX_DRIFT_DE40", 5.00),
    "US500": _env_float("MAX_DRIFT_US500", 1.50),
    "US30": _env_float("MAX_DRIFT_US30", 10.00),
}

# Candle ML smart timeout — own rescan cycle matching its ~4-min prediction horizon
# First re-scan ~1 candle after entry, then every CANDLE_ML_TIMEOUT_RESCAN_SEC
CANDLE_ML_TIMEOUT_MIN = _env_float("CANDLE_ML_TIMEOUT_MIN", 3.8)
CANDLE_ML_TIMEOUT_RESCAN_SEC = _env_int("CANDLE_ML_TIMEOUT_RESCAN_SEC", 60)

# Adaptive confirmation
ADAPTIVE_CONFIRMATION_ENABLED = _env_bool("ADAPTIVE_CONFIRMATION_ENABLED", True)
ADAPTIVE_CONF_WINDOW = _env_int("ADAPTIVE_CONF_WINDOW", 200)
ADAPTIVE_CONF_P_LOW = _env_int("ADAPTIVE_CONF_P_LOW", 60)
ADAPTIVE_CONF_P_NORM = _env_int("ADAPTIVE_CONF_P_NORM", 40)
ADAPTIVE_CONF_P_HIGH = _env_int("ADAPTIVE_CONF_P_HIGH", 0)

# Filters
MAX_SPREAD_PIPS = _env_float("MAX_SPREAD_PIPS", 35.0)
ALLOWED_SESSIONS = _env_str("ALLOWED_SESSIONS", "LONDON,NEW_YORK")
SYMBOL_ALLOWED_SESSIONS = {
    "XAUUSD": _env_str("SESSIONS_XAUUSD", "ASIA,LONDON,NEW_YORK"),
    "US100": _env_str("SESSIONS_US100", "LONDON,NEW_YORK"),
    "JP225": _env_str("SESSIONS_JP225", "ASIA"),
    "DE40": _env_str("SESSIONS_DE40", "LONDON"),
    "US500": _env_str("SESSIONS_US500", "LONDON,NEW_YORK"),
    "US30": _env_str("SESSIONS_US30", "LONDON,NEW_YORK"),
}

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

# JP225 (Nikkei 225) market hours — Asia session (Mon-Fri 23:00-06:00 UTC).
# JP225 is an overnight (Asia) pair vs the US session pairs, keeping the bot
# busy across sessions.
JP225_OPEN_HOUR_UTC = 23
JP225_OPEN_MINUTE_UTC = 0
JP225_CLOSE_HOUR_UTC = 6

# DE40 (Germany 40) market hours — Europe session (Mon-Fri 07:00-21:00 UTC).
DE40_OPEN_HOUR_UTC = 7
DE40_OPEN_MINUTE_UTC = 0
DE40_CLOSE_HOUR_UTC = 21

# Friday close awareness: block NEW entries within this many minutes before the
# symbol's weekly (Friday) close so no fresh position gets stuck over the
# weekend. Existing positions are never force-closed — they stay open and are
# managed by the normal exit logic (trailing reversal line, TP, model-flip cut).
FRIDAY_CLOSE_BLOCK_MIN = _env_int("FRIDAY_CLOSE_BLOCK_MIN", 60)


# Candle ML — multi-timeframe M5 direction prediction
CANDLE_ML_ENABLED = _env_bool("CANDLE_ML_ENABLED", True)
CANDLE_ML_CONFIDENCE_THRESHOLD = _env_float("CANDLE_ML_CONFIDENCE_THRESHOLD", 0.65)
# Per-symbol entry confidence override (defaults to CANDLE_ML_CONFIDENCE_THRESHOLD).
# US500 (S&P 500) is less volatile and more mean-reverting than US100 (Nasdaq),
# so it may want a different bar than the US100-tuned global default.
CANDLE_ML_CONFIDENCE_THRESHOLDS = {
    "XAUUSD": _env_float("CANDLE_ML_CONFIDENCE_XAUUSD", CANDLE_ML_CONFIDENCE_THRESHOLD),
    "US100": _env_float("CANDLE_ML_CONFIDENCE_US100", CANDLE_ML_CONFIDENCE_THRESHOLD),
    "JP225": _env_float("CANDLE_ML_CONFIDENCE_JP225", CANDLE_ML_CONFIDENCE_THRESHOLD),
    "DE40": _env_float("CANDLE_ML_CONFIDENCE_DE40", CANDLE_ML_CONFIDENCE_THRESHOLD),
    "US500": _env_float("CANDLE_ML_CONFIDENCE_US500", CANDLE_ML_CONFIDENCE_THRESHOLD),
    "US30": _env_float("CANDLE_ML_CONFIDENCE_US30", CANDLE_ML_CONFIDENCE_THRESHOLD),
}
CANDLE_ML_M1_HISTORY_BARS = _env_int("CANDLE_ML_M1_HISTORY_BARS", 500)
CANDLE_ML_MODEL_PATHS = {
    "XAUUSD": _env_str("CANDLE_ML_MODEL_PATH_XAUUSD", "models/candle_xgb_m5_XAUUSD.joblib"),
    "US100": _env_str("CANDLE_ML_MODEL_PATH_US100", "models/candle_xgb_m5_US100.joblib"),
    "JP225": _env_str("CANDLE_ML_MODEL_PATH_JP225", "models/candle_xgb_m5_JP225.joblib"),
    "DE40": _env_str("CANDLE_ML_MODEL_PATH_DE40", "models/candle_xgb_m5_DE40.joblib"),
    "US500": _env_str("CANDLE_ML_MODEL_PATH_US500", "models/candle_xgb_m5_US500.joblib"),
    "US30": _env_str("CANDLE_ML_MODEL_PATH_US30", "models/candle_xgb_m5_US30.joblib"),
}
# Mode: "always" = use Candle ML for all entries
#       "volatility" = use Candle ML when vol_ratio > threshold
CANDLE_ML_MODE = {
    "XAUUSD": _env_str("CANDLE_ML_MODE_XAUUSD", "always"),
    "US100": _env_str("CANDLE_ML_MODE_US100", "always"),
    "JP225": _env_str("CANDLE_ML_MODE_JP225", "off"),
    "DE40": _env_str("CANDLE_ML_MODE_DE40", "off"),
    "US500": _env_str("CANDLE_ML_MODE_US500", "off"),
    "US30": _env_str("CANDLE_ML_MODE_US30", "always"),
}
# For "volatility" mode: switch to Candle ML when short-term / long-term ATR > this
CANDLE_ML_VOL_THRESHOLD = _env_float("CANDLE_ML_VOL_THRESHOLD", 1.3)
CANDLE_ML_VOL_WINDOW_SHORT = _env_int("CANDLE_ML_VOL_WINDOW_SHORT", 5)   # M5 bars
CANDLE_ML_VOL_WINDOW_LONG = _env_int("CANDLE_ML_VOL_WINDOW_LONG", 40)    # M5 bars
# Entry quality filter (research-backed): only trade on strong candles at best sessions.
# mode: "strict" (strong_body/pin_bar/engulfing only) | "basic" (+normal, skip doji/tiny) | "off"
CANDLE_ML_PATTERN_FILTER = _env_str("CANDLE_ML_PATTERN_FILTER", "strict")
# Per-symbol pattern filter override (defaults to CANDLE_ML_PATTERN_FILTER).
CANDLE_ML_PATTERN_FILTERS = {
    "XAUUSD": _env_str("CANDLE_ML_PATTERN_XAUUSD", CANDLE_ML_PATTERN_FILTER),
    "US100": _env_str("CANDLE_ML_PATTERN_US100", CANDLE_ML_PATTERN_FILTER),
    "JP225": _env_str("CANDLE_ML_PATTERN_JP225", CANDLE_ML_PATTERN_FILTER),
    "DE40": _env_str("CANDLE_ML_PATTERN_DE40", CANDLE_ML_PATTERN_FILTER),
    "US500": _env_str("CANDLE_ML_PATTERN_US500", CANDLE_ML_PATTERN_FILTER),
    "US30": _env_str("CANDLE_ML_PATTERN_US30", CANDLE_ML_PATTERN_FILTER),
}
# Best hours per symbol (UTC). Empty string = all hours. XAUUSD backtest: 13-18 UTC = London/NY (WR 55-61%).
CANDLE_ML_ALLOWED_HOURS = {
    "XAUUSD": _env_str("CANDLE_ML_ALLOWED_HOURS_XAUUSD", "13,14,15,16,17,18"),
    "US100": _env_str("CANDLE_ML_ALLOWED_HOURS_US100", ""),
    "JP225": _env_str("CANDLE_ML_ALLOWED_HOURS_JP225", ""),
    "DE40": _env_str("CANDLE_ML_ALLOWED_HOURS_DE40", ""),
    "US500": _env_str("CANDLE_ML_ALLOWED_HOURS_US500", ""),
    "US30": _env_str("CANDLE_ML_ALLOWED_HOURS_US30", ""),
}
# Exit: the candle_reversal line trails the best price by CANDLE_ML_TRAIL_ATR x
# ATR (floored at the fill/break-even), so normal pullbacks don't close a trade
# CandleML still agrees with. Wider value = more room to breathe, less locked in.
CANDLE_ML_TRAILING_ENABLED = _env_bool("CANDLE_ML_TRAILING_ENABLED", True)
CANDLE_ML_TRAIL_ATR = _env_float("CANDLE_ML_TRAIL_ATR", 1.5)
# Peak-retrace trailing (profitable trades only): instead of trailing a fixed ATR
# distance behind the best price, trail a fraction of the peak profit already made
# (CANDLE_ML_PEAK_RETRACE_FRAC). Auto-adapts to lot/equity scaling — small moves
# keep most of their profit (70%), big moves lock in proportionally. Only applied
# while the trade is in profit; losing trades keep the existing flip-cut logic.
CANDLE_ML_PEAK_RETRACE_ENABLED = _env_bool("CANDLE_ML_PEAK_RETRACE_ENABLED", True)
CANDLE_ML_PEAK_RETRACE_FRAC = _env_float("CANDLE_ML_PEAK_RETRACE_FRAC", 0.3)
# Model-flip loss cut: at each M5 boundary re-run the candle model for the open
# symbol. If it now predicts the OPPOSITE direction AND the trade is already
# underwater by CANDLE_ML_FLIP_LOSS_ATR x ATR, exit immediately
# ("candle_model_flip"). Never triggers on winners — only cuts losers short.
#
# Rigidity (the model is flippy: ~48% of runs are single-boundary blips, and
# first-flip confidence is only ~0.73-0.74 median):
#   - CANDLE_ML_FLIP_CONSECUTIVE (default 2): the opposite call must repeat for
#     2 consecutive M5 boundaries (a single blip is ignored; a real reversal
#     persists). Halves the flip-trigger rate at every horizon.
#   - CANDLE_ML_FLIP_CONF (default 0.70): the opposite call must be strong,
#     above the 0.65 entry threshold, so weak "no"s don't count.
CANDLE_ML_FLIP_EXIT_ENABLED = _env_bool("CANDLE_ML_FLIP_EXIT_ENABLED", True)
CANDLE_ML_FLIP_LOSS_ATR = _env_float("CANDLE_ML_FLIP_LOSS_ATR", 0.25)
CANDLE_ML_FLIP_CONSECUTIVE = _env_int("CANDLE_ML_FLIP_CONSECUTIVE", 2)
CANDLE_ML_FLIP_CONF = _env_float("CANDLE_ML_FLIP_CONF", 0.70)
# Per-symbol flip-cut overrides (default to the global values above).
CANDLE_ML_FLIP_CONSECUTIVES = {
    "XAUUSD": _env_int("CANDLE_ML_FLIP_CONSECUTIVE_XAUUSD", CANDLE_ML_FLIP_CONSECUTIVE),
    "US100": _env_int("CANDLE_ML_FLIP_CONSECUTIVE_US100", CANDLE_ML_FLIP_CONSECUTIVE),
    "JP225": _env_int("CANDLE_ML_FLIP_CONSECUTIVE_JP225", CANDLE_ML_FLIP_CONSECUTIVE),
    "DE40": _env_int("CANDLE_ML_FLIP_CONSECUTIVE_DE40", CANDLE_ML_FLIP_CONSECUTIVE),
    "US500": _env_int("CANDLE_ML_FLIP_CONSECUTIVE_US500", CANDLE_ML_FLIP_CONSECUTIVE),
    "US30": _env_int("CANDLE_ML_FLIP_CONSECUTIVE_US30", CANDLE_ML_FLIP_CONSECUTIVE),
}
CANDLE_ML_FLIP_CONFS = {
    "XAUUSD": _env_float("CANDLE_ML_FLIP_CONF_XAUUSD", CANDLE_ML_FLIP_CONF),
    "US100": _env_float("CANDLE_ML_FLIP_CONF_US100", CANDLE_ML_FLIP_CONF),
    "JP225": _env_float("CANDLE_ML_FLIP_CONF_JP225", CANDLE_ML_FLIP_CONF),
    "DE40": _env_float("CANDLE_ML_FLIP_CONF_DE40", CANDLE_ML_FLIP_CONF),
    "US500": _env_float("CANDLE_ML_FLIP_CONF_US500", CANDLE_ML_FLIP_CONF),
    "US30": _env_float("CANDLE_ML_FLIP_CONF_US30", CANDLE_ML_FLIP_CONF),
}

# Max loss time — force close Candle ML trades that are underwater after this many minutes.
# Prevents losing positions from blocking better opportunities indefinitely.
CANDLE_ML_MAX_LOSS_MINUTES = _env_float("CANDLE_ML_MAX_LOSS_MINUTES", 120.0)

# ── H1 Candle Engine (candle-following, XGBoost per-pair) ────────────
# Follows the H1 candle instead of predicting it (see CANDLE_STRATEGY.md).
# Enter at candle commit, ride the full move, trail once past open into
# profit, close on reversal/confusion and wait for a fresh candle. Small
# fixed bleeds are covered by the rare full move (+3.00$). Each enabled pair
# gets its own model + dynamic threshold; a pair-selection layer only lets
# the currently-best-moving pairs fire (jump away from idle symbols).
CANDLE_ENGINE_ENABLED = _env_bool("CANDLE_ENGINE_ENABLED", True)
CANDLE_ENGINE_PAIRS = [s.strip() for s in _env_str("CANDLE_ENGINE_PAIRS", "XAUUSD,US100,JP225,DE40,US500,US30").split(",")]
CANDLE_ENGINE_TF = _env_int("CANDLE_ENGINE_TF", 60)  # minutes: 60 = H1, 30 = 30m
CANDLE_ENGINE_MODEL_DIR = _env_str("CANDLE_ENGINE_MODEL_DIR", "models/candle_h1")
# Profit-based labels: simulate the candle-following trade per bar (enter at
# commit, SL = SL_ATR*ATR, ride until reversal/trail, cap at MAX_HOLD, cost
# COST_R). A bar is BUY/SELL only if its realized R beats the other side by
# EDGE_MARGIN (chop => NONE => no trade => no bleed).
CANDLE_ENGINE_SL_ATR = _env_float("CANDLE_ENGINE_SL_ATR", 1.5)
CANDLE_ENGINE_REVERSAL_ATR = _env_float("CANDLE_ENGINE_REVERSAL_ATR", 0.5)
CANDLE_ENGINE_TRAIL_ATR = _env_float("CANDLE_ENGINE_TRAIL_ATR", 0.5)
CANDLE_ENGINE_MAX_HOLD_BARS = _env_int("CANDLE_ENGINE_MAX_HOLD_BARS", 24)
CANDLE_ENGINE_COST_R = _env_float("CANDLE_ENGINE_COST_R", 0.05)
# Label stop for TRAINING (decoupled from the live exit SL): labels are
# simulated with a TIGHT 1.0R stop so the model learns early-strength candles,
# while the live engine rides with a WIDER 1.5R stop. Coupling them (labels
# @ 1.5R) dropped OOS PF 1.67 -> 1.07, so keep them separate.
CANDLE_ENGINE_LABEL_SL_ATR = _env_float("CANDLE_ENGINE_LABEL_SL_ATR", 1.0)
# Label strictness (the model only fires on the strong, clearly-directional
# moves): a bar is BUY/SELL only if the winning side realizes >= ENTRY_MIN_R
# AND beats the losing side by >= EDGE_MARGIN. Sweep on H1 OOS 2023-25:
#   (0.35, 1.0) -> PF 1.18   (0.60, 1.0) -> PF 1.31
#   (0.90, 1.75) -> PF 1.67  (broad plateau, WR ~69%, dd ~7R)
# Looser labels teach the model to chase weak-move noise; the tight labels
# confine trades to the ~15% of bars with a real directional edge.
CANDLE_ENGINE_ENTRY_MIN_R = _env_float("CANDLE_ENGINE_ENTRY_MIN_R", 0.90)
CANDLE_ENGINE_EDGE_MARGIN = _env_float("CANDLE_ENGINE_EDGE_MARGIN", 1.75)
CANDLE_ENGINE_MIN_CONF = _env_float("CANDLE_ENGINE_MIN_CONF", 0.60)
CANDLE_ENGINE_TRAIN_PER_CLASS = _env_int("CANDLE_ENGINE_TRAIN_PER_CLASS", 15000)
# Jump-candle scan (add-on): a candle that leaves its starting point and jumps
# in its full direction gets a higher-conviction entry. Detected when close has
# moved JUMP_BREAK_R*ATR past the open AND the body covers >= JUMP_BODY_R of
# the H1 range.
CANDLE_ENGINE_JUMP_ENABLED = _env_bool("CANDLE_ENGINE_JUMP_ENABLED", True)
CANDLE_ENGINE_JUMP_BREAK_R = _env_float("CANDLE_ENGINE_JUMP_BREAK_R", 1.5)
CANDLE_ENGINE_JUMP_BODY_R = _env_float("CANDLE_ENGINE_JUMP_BODY_R", 0.70)
# Pair-selection layer: only the top-K pairs by live candle momentum score may
# fire at any time. Each pair's own threshold adapts to its 30-60d score
# percentile (dynamic per-pair), so idle pairs are automatically muted.
CANDLE_ENGINE_PAIR_TOP_K = _env_int("CANDLE_ENGINE_PAIR_TOP_K", 2)
CANDLE_ENGINE_PAIR_SCORE_WINDOW = _env_int("CANDLE_ENGINE_PAIR_SCORE_WINDOW", 720)  # ~30d H1 bars
CANDLE_ENGINE_PAIR_PCT_MIN = _env_float("CANDLE_ENGINE_PAIR_PCT_MIN", 0.70)

# ── Pull-into-H1 scalper (prevh1 / pull / trail) — LIVE execution ────
# Trades the M5 pullback in the direction of the last COMPLETED H1 candle
# body, trailing a giveback fraction of the wave, force-closing at a fixed
# horizon (in M5 bars). This is the ONLY strategy that survived honest
# backtests with real costs (see _bt_pull_prevh1.py, _tune_pull_prevh1.py).
# Per-symbol tuned configs (train 2023-24 -> 2025 -> OOS 2026):
#   US30   pull .30 trail .35 hold 24  PF 1.73 -> 1.83 -> 1.81  (best)
#   XAUUSD pull .30 trail .15 hold 12  PF 1.22 -> 1.96 -> 1.40
#   US100  pull .30 trail .50 hold 6   PF 1.38 -> 1.86 -> 1.34
#   DE40   pull .30 trail .50 hold 24  (no 2026 data)
#   US500  pull .30 trail .35 hold 24  (marginal after costs)
#   JP225  pull .30 trail .35 hold 24  (marginal after costs — avoid)
# Enabled per symbol (ON for the 3 OOS-validated pairs, OFF by default for
# the rest). Flip PULL_XAUUSD=false etc. to disable a pair.
PULL_ENGINE_ENABLED = {
    "XAUUSD": _env_bool("PULL_XAUUSD", True),
    "US100": _env_bool("PULL_US100", True),
    "JP225": _env_bool("PULL_JP225", False),
    "DE40": _env_bool("PULL_DE40", False),
    "US500": _env_bool("PULL_US500", False),
    "US30": _env_bool("PULL_US30", True),
}
PULL_M1_HISTORY_BARS = _env_int("PULL_M1_HISTORY_BARS", 8000)
PULL_REFRESH_SEC = _env_int("PULL_REFRESH_SEC", 60)
PULL_MIN_H1_BARS = _env_int("PULL_MIN_H1_BARS", 30)  # completed H1 candles before trading
# Round-trip cost in price units = spread + 2 x commission (matches the
# backtest's cost model; used to score trade R for the daily guard).
PULL_SYMBOL_ROUND_TRIP = {
    "XAUUSD": _env_float("PULL_ROUND_TRIP_XAUUSD", 0.38),
    "US100": _env_float("PULL_ROUND_TRIP_US100", 2.00),
    "JP225": _env_float("PULL_ROUND_TRIP_JP225", 10.0),
    "DE40": _env_float("PULL_ROUND_TRIP_DE40", 2.00),
    "US500": _env_float("PULL_ROUND_TRIP_US500", 0.80),
    "US30": _env_float("PULL_ROUND_TRIP_US30", 2.00),
}
PULL_SYMBOL_DEFAULTS = {
    "XAUUSD": dict(pull_r=0.30, trail_r=0.15, max_hold=12),
    "US100": dict(pull_r=0.30, trail_r=0.50, max_hold=6),
    "JP225": dict(pull_r=0.30, trail_r=0.35, max_hold=24),
    "DE40": dict(pull_r=0.30, trail_r=0.50, max_hold=24),
    "US500": dict(pull_r=0.30, trail_r=0.35, max_hold=24),
    "US30": dict(pull_r=0.30, trail_r=0.35, max_hold=24),
}
# "Daily profit bot" guards (per symbol, R in entry ATR units, UTC midnight
# rollover): stop NEW entries once the day's net R reaches the target (lock in
# the day) or hits the max loss (stop the bleed). 0.0 = guard disabled. An open
# position is never force-closed by these — only new entries are blocked.
PULL_DAILY_TARGET_R = {
    "XAUUSD": _env_float("PULL_DAILY_TARGET_R_XAUUSD", 0.0),
    "US100": _env_float("PULL_DAILY_TARGET_R_US100", 0.0),
    "JP225": _env_float("PULL_DAILY_TARGET_R_JP225", 0.0),
    "DE40": _env_float("PULL_DAILY_TARGET_R_DE40", 0.0),
    "US500": _env_float("PULL_DAILY_TARGET_R_US500", 0.0),
    "US30": _env_float("PULL_DAILY_TARGET_R_US30", 3.0),
}
PULL_DAILY_MAX_LOSS_R = {
    "XAUUSD": _env_float("PULL_DAILY_MAX_LOSS_R_XAUUSD", 0.0),
    "US100": _env_float("PULL_DAILY_MAX_LOSS_R_US100", 0.0),
    "JP225": _env_float("PULL_DAILY_MAX_LOSS_R_JP225", 0.0),
    "DE40": _env_float("PULL_DAILY_MAX_LOSS_R_DE40", 0.0),
    "US500": _env_float("PULL_DAILY_MAX_LOSS_R_US500", 0.0),
    "US30": _env_float("PULL_DAILY_MAX_LOSS_R_US30", 2.5),
}
SYMBOL_PULL_PARAMS = {
    sym: {
        "pull_r": _env_float(f"PULL_PULL_R_{sym}", PULL_SYMBOL_DEFAULTS.get(sym, {}).get("pull_r", 0.30)),
        "trail_r": _env_float(f"PULL_TRAIL_R_{sym}", PULL_SYMBOL_DEFAULTS.get(sym, {}).get("trail_r", 0.35)),
        "max_hold": _env_int(f"PULL_MAX_HOLD_{sym}", PULL_SYMBOL_DEFAULTS.get(sym, {}).get("max_hold", 24)),
        "round_trip": PULL_SYMBOL_ROUND_TRIP.get(sym, 0.0),
        "daily_target_r": PULL_DAILY_TARGET_R.get(sym, 0.0),
        "daily_max_loss_r": PULL_DAILY_MAX_LOSS_R.get(sym, 0.0),
    }
    for sym in SYMBOLS
}

# ── Momentum-jump engine (per-pair adapted) ──────────────────────────
# Validated OOS on M5 2022-2025 with per-pair regime gates:
#   US100: vol>=55 gate  -> +127.8R  (all 4 years positive)
#   JP225: vol>=55 gate + mz2.5/jump1.0/hold18/retr0.5 -> +111.8R (all 4 years positive)
#   DE40:  er>=0.35 gate + mz2.5/jump1.25/hold18 -> +6.2R (all years ~flat/+)
# Signal (long example): surge bar (momentum_z >= MZ_MIN, body_ratio >= BODY_MIN,
# |trend_strength| >= TS_MIN) inside the long-term EMA480 regime (long above /
# short below) entered at bar close; SL = SL_R x ATR; exits on jump-target
# JUMP_TARGET_R + RETRACE_R retrace from the peak, or after MAX_HOLD_BARS.
# Regime gate (M5, forward-safe): only trade when the market's current ATR is
# in the top (1-GATE_THRESHOLD) of the last GATE_WINDOW M5 bars ("vol", good for
# US100/JP225) OR when the Kaufman efficiency ratio >= GATE_THRESHOLD over the
# window ("er", trending-only, good for DE40). "none" disables the gate.
MOMENTUM_ENGINE_ENABLED = {
    "XAUUSD": _env_bool("MOMENTUM_XAUUSD", False),
    "US100": _env_bool("MOMENTUM_US100", False),
    "JP225": _env_bool("MOMENTUM_JP225", False),
    "DE40": _env_bool("MOMENTUM_DE40", False),
    "US500": _env_bool("MOMENTUM_US500", False),
    "US30": _env_bool("MOMENTUM_US30", False),
}
MOMENTUM_MZ_MIN = _env_float("MOMENTUM_MZ_MIN", 2.0)
MOMENTUM_BODY_RATIO_MIN = _env_float("MOMENTUM_BODY_RATIO_MIN", 0.60)
MOMENTUM_TS_MIN = _env_float("MOMENTUM_TS_MIN", 0.50)
MOMENTUM_EMA_SPAN = _env_int("MOMENTUM_EMA_SPAN", 480)
MOMENTUM_SL_R = _env_float("MOMENTUM_SL_R", 1.0)
MOMENTUM_JUMP_TARGET_R = _env_float("MOMENTUM_JUMP_TARGET_R", 1.0)
MOMENTUM_RETRACE_R = _env_float("MOMENTUM_RETRACE_R", 0.25)
MOMENTUM_MAX_HOLD_BARS = _env_int("MOMENTUM_MAX_HOLD_BARS", 12)

# Per-pair parameter overrides (backtest-adapted). Keys omitted from the dict
# fall back to the global MOMENTUM_* values above.
MOMENTUM_PAIR_PARAMS = {
    "US100": {"mz_min": 2.0, "jump_target": 1.0, "retr_r": 0.25, "max_hold": 12},
    "JP225": {"mz_min": 2.5, "jump_target": 1.0, "retr_r": 0.50, "max_hold": 18},
    "DE40": {"mz_min": 2.5, "jump_target": 1.25, "retr_r": 0.25, "max_hold": 18},
}

# Per-pair regime gate ("none" | "vol" | "er").
#   vol: ATR percentile >= threshold (fraction), window = MOMENTUM_GATE_WINDOW.
#   er:  Kaufman efficiency ratio >= threshold (fraction), window = same.
MOMENTUM_GATE = {
    "US100": _env_str("MOMENTUM_GATE_US100", "vol"),
    "JP225": _env_str("MOMENTUM_GATE_JP225", "vol"),
    "DE40": _env_str("MOMENTUM_GATE_DE40", "er"),
}
MOMENTUM_GATE_THRESHOLD = {
    "US100": _env_float("MOMENTUM_GATE_THRESHOLD_US100", 0.55),
    "JP225": _env_float("MOMENTUM_GATE_THRESHOLD_JP225", 0.55),
    "DE40": _env_float("MOMENTUM_GATE_THRESHOLD_DE40", 0.35),
}
# M5 bars in the regime window (96 M5 = 8 hours of market data).
MOMENTUM_GATE_WINDOW = _env_int("MOMENTUM_GATE_WINDOW", 96)
# How often the bot refetches M1 history for the EMA480 regime (seconds).
MOMENTUM_REFRESH_SEC = _env_int("MOMENTUM_REFRESH_SEC", 60)
# M1 bars fetched for the EMA480 regime. 485 M5 buckets are needed for the EMA480
# warmup (480 + 5) = ~2425 M1 bars of market data; 8000 bars (~5.5 trading days)
# guarantees that window even across market breaks/weekend gaps. 2500 was too
# small (max 500 buckets, real windows averaged ~408 -> detect() never fired).
MOMENTUM_M1_HISTORY_BARS = _env_int("MOMENTUM_M1_HISTORY_BARS", 8000)

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

# Failover
FAILOVER_ENABLED = _env_bool("FAILOVER_ENABLED", False)
FAILOVER_ROLE = _env_str("FAILOVER_ROLE", "primary")

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

    if sym in ("US100", "NASDAQ", "NAS100", "US500", "SP500", "US30", "DOW", "DJ30"):
        # US100: Mon-Fri 14:30-21:00 UTC
        if 0 <= wd <= 4:
            open_t = US100_OPEN_HOUR_UTC + US100_OPEN_MINUTE_UTC / 60.0
            close_t = US100_CLOSE_HOUR_UTC
            return open_t <= h < close_t
        return False

    if sym in ("JP225", "NIKKEI", "N225", "JPN225"):
        # JP225: Asia overnight session (Mon 23:00 UTC - Sat 06:00 UTC, with a
        # daily break 06:00-23:00). Across midnight: active when h >= 23 OR h < 6.
        open_t = JP225_OPEN_HOUR_UTC + JP225_OPEN_MINUTE_UTC / 60.0
        close_t = JP225_CLOSE_HOUR_UTC
        if 0 <= wd <= 4:
            return h >= open_t or h < close_t
        if wd == 6:  # Sunday night = start of Monday's Asia session
            return h >= open_t
        return False  # Saturday daytime closed

    if sym in ("DE40", "GER40", "DAX", "GERMANY40"):
        # DE40: Europe session (Mon-Fri 07:00-21:00 UTC).
        if 0 <= wd <= 4:
            open_t = DE40_OPEN_HOUR_UTC + DE40_OPEN_MINUTE_UTC / 60.0
            close_t = DE40_CLOSE_HOUR_UTC
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


def minutes_to_friday_close(sym: str):
    """Minutes until this symbol's weekly (Friday) close, or None if not Friday.

    Friday-aware entry gate: on Friday the bot stops opening NEW positions once
    the remaining time drops below FRIDAY_CLOSE_BLOCK_MIN. Existing positions
    are never force-closed by this.
    """
    from datetime import datetime as _dt
    now = _dt.utcnow()
    if now.weekday() != 4:
        return None
    h = now.hour + now.minute / 60.0
    if sym in ("US100", "NASDAQ", "NAS100", "US500", "SP500", "US30", "DOW", "DJ30"):
        close_t = US100_CLOSE_HOUR_UTC
    elif sym in ("JP225", "NIKKEI", "N225", "JPN225"):
        # Nikkei's weekly close is Friday 06:00 UTC (end of the Asia session
        # that started Thu 23:00). On Friday the session runs 00:00-06:00 UTC.
        if h >= JP225_CLOSE_HOUR_UTC:
            return None  # Friday session already closed this morning
        close_t = JP225_CLOSE_HOUR_UTC
    elif sym in ("DE40", "GER40", "DAX", "GERMANY40"):
        close_t = DE40_CLOSE_HOUR_UTC
    else:
        close_t = MARKET_CLOSE_FRIDAY_UTC
    return (close_t - h) * 60.0
