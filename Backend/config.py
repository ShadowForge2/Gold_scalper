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
LOT_STEP = _env_float("LOT_STEP", 0.01)
LOT_MULTIPLIER = _env_int("LOT_MULTIPLIER", 2)
# Hard safety cap on the lot multiplier regardless of env/API value.
# Backtest (1:100 lev, $20 start): mult>2 blows the account up
# (x3 +8%, x4 -43%, x5 -70%).
MAX_LOT_MULTIPLIER = _env_float("MAX_LOT_MULTIPLIER", 2.0)

# Fixed fractional risk per trade (matches backtest: lot_value = balance * RISK_PCT)
RISK_PCT = _env_float("RISK_PCT", 0.04)

# Minimum balance to start trading
MIN_BALANCE = _env_float("MIN_BALANCE", 10.0)

# ATR used by the pull scalper (and as the SL/TP basis).
ATR_PERIOD = _env_int("ATR_PERIOD", 14)

# Trailing SL / TP (used for recovered-position fallback sizing).
SL_ATR_MULTIPLIER = _env_float("SL_ATR_MULTIPLIER", 1.0)
TP1_MULTIPLIER = _env_float("TP1_MULTIPLIER", 2.0)

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

# Hard safety cap on lot size regardless of API/equity scaling.
MAX_LOT = _env_float("MAX_LOT", 9999.0)

# Multi-symbol: bot trades XAUUSD, US100, US500, and US30 automatically.
# All pairs trade from MIN_BALANCE ($10) — the $30 gate was for the old
# single-pair strategy; we now poll all pairs and trade the best one.
# JP225/DE40 are momentum-engine pairs (per-pair adapted params + regime gate).
SYMBOLS = [s.strip() for s in _env_str("SYMBOLS", "XAUUSD,US100,JP225,DE40,US500,US30").split(",")]

# Pairs that must NEVER be traded with the pull strategy (the sweep showed they
# fail under realistic fills). They are excluded from the scan universe, never
# get a pull engine armed, and any leftover open position is force-closed.
# Override via BLACKLIST_SYMBOLS (comma-separated) in the env.
BLACKLIST_SYMBOLS = {
    s.strip().upper()
    for s in _env_str("BLACKLIST_SYMBOLS", "DE40,JP225,US500").split(",")
    if s.strip()
}

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

# Filters
MAX_SPREAD_PIPS = _env_float("MAX_SPREAD_PIPS", 35.0)

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
PULL_REFRESH_SEC = _env_int("PULL_REFRESH_SEC", 15)
PULL_MIN_H1_BARS = _env_int("PULL_MIN_H1_BARS", 30)  # completed H1 candles before trading
# Profit lock-in for the PullPrevH1 engine:
#   GIVEBACK_CAP — a live trade never gives back more than this fraction of the
#                  peak profit it has touched (tight trailing stop that rises
#                  with the run; 0.30 = keep at least 70% of the best run).
#   PUMP_ATR     — a single M5 candle whose favourable body exceeds this many
#                  H1-ATR is treated as a sudden pump; the trade exits at that
#                  candle's close to capture the tip instead of waiting to trail.
PULL_GIVEBACK_CAP = _env_float("PULL_GIVEBACK_CAP", 0.30)
PULL_PUMP_ATR = _env_float("PULL_PUMP_ATR", 0.5)
# Broker-side trailing stop (fixes the live fill-gap bleed). When enabled for a
# symbol, the bot ratchets a real Capital.com stopLevel on each completed M5 bar
# via modify_position() so the broker closes AT the engine's intended trail price
# instead of a market DELETE that slips badly on fast moves (US30/US100 live
# trail exits were filled far past the intended price -> the entire -4.82 bleed).
# Falls back to a market close if the broker rejects the level; pump-atr tip
# exits are untouched.
# PER-SYMBOL, evidence-backed by _bt_trailstop_validate.py (2025 + 2026, M1
# touch-fill modeling):
#   US30 / US100  -> ON.  Broker stop lifts WR ~60% and PF ~2.1-2.6 vs ~1.0-1.4
#                    bar-close. These two bled live; the stop fixes it.
#   XAUUSD        -> OFF. Gold's tight trail_r=0.15 gets chopped by intra-bar
#                    noise under a real broker stop (PF drops to ~0.9-0.04 in
#                    backtest); gold did NOT bleed live, keep its bar-close trail.
PULL_TRAIL_STOP_ENABLED = {
    "XAUUSD": _env_bool("PULL_TRAIL_STOP_XAUUSD", False),
    "US100": _env_bool("PULL_TRAIL_STOP_US100", True),
    "JP225": _env_bool("PULL_TRAIL_STOP_JP225", False),
    "DE40": _env_bool("PULL_TRAIL_STOP_DE40", False),
    "US500": _env_bool("PULL_TRAIL_STOP_US500", False),
    "US30": _env_bool("PULL_TRAIL_STOP_US30", True),
}
# Legacy global switch: if explicitly set, it forces the per-symbol value ON for
# every enabled pull pair (used by deployments that only know the old knob).
PULL_TRAIL_STOP_DEFAULT = _env_bool("PULL_TRAIL_STOP_ENABLED", True)
# Percentage-of-ATR guard used when ratcheting the broker stop: we never set a
# stopLevel closer to market than this fraction of ATR, because Capital.com
# rejects stops too close to the current price (min-stop-distance rule). 0.0
# disables the floor. Only a safety valve — it does not change the exit model.
PULL_SLIP_GUARD_ATR = _env_float("PULL_SLIP_GUARD_ATR", 0.75)
# Global margin guard: if a NEW entry is rejected for insufficient margin, the
# bot pauses ALL new entries (open positions keep trading/exiting) and only
# resumes once free margin recovers past the blocked level (a position closing
# frees margin). BUFFER adds hysteresis so tiny fluctuations don't flap.
MARGIN_BLOCK_ENABLED = _env_bool("MARGIN_BLOCK_ENABLED", True)
MARGIN_BLOCK_RETRY_SEC = _env_float("MARGIN_BLOCK_RETRY_SEC", 15)
MARGIN_BLOCK_BUFFER = _env_float("MARGIN_BLOCK_BUFFER", 1.05)
# Per-symbol auto-calibration for the PullPrevH1 engine: any pair without an
# explicit SYMBOL_PULL_PARAMS entry derives its own pull_r/trail_r/max_hold from
# its recent M5/H1 structure. Results are cached and refreshed every TTL_SEC.
PULL_AUTO_TUNE_ENABLED = _env_bool("PULL_AUTO_TUNE_ENABLED", True)
PULL_AUTO_TUNE_BARS_M5 = _env_int("PULL_AUTO_TUNE_BARS_M5", 600)   # ~50h of M5
PULL_AUTO_TUNE_TTL_SEC = _env_int("PULL_AUTO_TUNE_TTL_SEC", 21600)  # 6h
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
    "US30": _env_float("PULL_DAILY_TARGET_R_US30", 0.0),
}
PULL_DAILY_MAX_LOSS_R = {
    "XAUUSD": _env_float("PULL_DAILY_MAX_LOSS_R_XAUUSD", 0.0),
    "US100": _env_float("PULL_DAILY_MAX_LOSS_R_US100", 0.0),
    "JP225": _env_float("PULL_DAILY_MAX_LOSS_R_JP225", 0.0),
    "DE40": _env_float("PULL_DAILY_MAX_LOSS_R_DE40", 0.0),
    "US500": _env_float("PULL_DAILY_MAX_LOSS_R_US500", 0.0),
    "US30": _env_float("PULL_DAILY_MAX_LOSS_R_US30", 0.0),
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



# ── Pair scanner — whole-board trading (daily-returns bot) ───────────
# The bot scans ALL ~4000 Capital.com markets in one request every
# SCANNER_SCAN_SEC and takes the top-K highest-momentum TRADEABLE instruments
# that pass the tradability screen (status, spread, ATR band, streaming).
# Trades are simple trend+breakout with a fixed ATR SL/TP ("simple sl, take
# profit at max"). Honest fills (entry at ask for BUY / bid for SELL), cost =
# live spread + SCANNER_COST_R of price, validated by _bt_pair_scanner.py.
SCANNER_ENABLED = _env_bool("SCANNER_ENABLED", True)
SCANNER_SCAN_SEC = _env_int("SCANNER_SCAN_SEC", 120)
SCANNER_TOP_K = _env_int("SCANNER_TOP_K", 4)
SCANNER_TIMEFRAME = _env_int("SCANNER_TIMEFRAME", 15)        # M15 analysis bars
SCANNER_LOOKBACK_BARS = _env_int("SCANNER_LOOKBACK_BARS", 300)
# Tradable universe filter. Real Capital.com instrumentType values are plural:
# "INDICES", "CURRENCIES", "COMMODITIES" (covers metals AND energy), and
# "CRYPTOCURRENCIES". "SHARES"/"ETFS"/"BONDS" can be added to trade stocks.
SCANNER_TYPES = _env_str("SCANNER_TYPES",
    "INDICES,CURRENCIES,COMMODITIES,CRYPTOCURRENCIES").split(",")
SCANNER_MAX_SPREAD_R = _env_float("SCANNER_MAX_SPREAD_R", 0.0004)   # spread <= 4bps of price
SCANNER_MIN_ATR_R = _env_float("SCANNER_MIN_ATR_R", 0.0002)         # min volatility 2bps/bar
SCANNER_MAX_ATR_R = _env_float("SCANNER_MAX_ATR_R", 0.01)           # max volatility 1%/bar
# Momentum ranking (snapshot-based; M15 6-bar ROC used for the entry filter).
SCANNER_MIN_PCT_CHANGE = _env_float("SCANNER_MIN_PCT_CHANGE", 0.10) # abs daily move >= 0.10%
# Entry / exit rule (R in SCANNER_TIMEFRAME ATR units).
SCANNER_SL_R = _env_float("SCANNER_SL_R", 1.5)
SCANNER_TP_R = _env_float("SCANNER_TP_R", 2.0)
SCANNER_TRAIL_AT_R = _env_float("SCANNER_TRAIL_AT_R", 1.0)   # start trailing once >= 1R
SCANNER_TRAIL_R = _env_float("SCANNER_TRAIL_R", 0.5)         # giveback allowed before exit
SCANNER_MAX_HOLD_BARS = _env_int("SCANNER_MAX_HOLD_BARS", 16)   # M15 bars (~4h)
SCANNER_EMA_FAST = _env_int("SCANNER_EMA_FAST", 20)
SCANNER_EMA_SLOW = _env_int("SCANNER_EMA_SLOW", 50)
SCANNER_SWING_BACK = _env_int("SCANNER_SWING_BACK", 10)      # breakout vs max high of last N bars
SCANNER_ROC_BARS = _env_int("SCANNER_ROC_BARS", 6)
SCANNER_BREAKOUT_MIN_R = _env_float("SCANNER_BREAKOUT_MIN_R", 0.0)  # fresh breakout: close must pierce swing by >= X*ATR
SCANNER_COOLDOWN_SEC = _env_int("SCANNER_COOLDOWN_SEC", 1800)   # no re-entry on same epic
SCANNER_COST_R = _env_float("SCANNER_COST_R", 0.0001)           # extra round-trip cost f(bps)
SCANNER_HIST_SPREAD_R = _env_float("SCANNER_HIST_SPREAD_R", 0.00008)  # per-bar spread cost f(price) for offline M1-parquet backtests
SCANNER_MAX_TRADES_DAY = _env_int("SCANNER_MAX_TRADES_DAY", 6)  # per-epic daily cap

# General scalper fixed pull params (validated in _bt_general_scalper.py):
# ONE parameter set for EVERY pair — no per-symbol tuning. The scanner's top-K
# momentum ranking is the only selector: the bot trades whatever pair is
# currently leading the board (scan and see, no fixed symbol configuration).
SCANNER_PULL_R = _env_float("SCANNER_PULL_R", 0.30)
SCANNER_TRAIL_R = _env_float("SCANNER_TRAIL_R", 0.35)
SCANNER_MAX_HOLD = _env_int("SCANNER_MAX_HOLD", 12)

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
