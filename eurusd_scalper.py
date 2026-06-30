"""
EURUSD Scalper — Standalone v1
================================
A self-contained scalping strategy for EURUSD.

Strategy logic:
  - H1: 50 EMA trend filter (bias)
  - M5: RSI(14) regime
  - M1: 5/13 EMA crossover + volume surge entry
  - SL: 1.5 × ATR(14,M1)
  - TP1: 1.5 × SL  (close 60%)
  - TP2: 2.5 × SL  (close 40%)
  - Sessions: London (07-16 UTC), NY (12-21 UTC)
  - Max 3 trades / session | Max 2 consecutive losses

Usage:
  python eurusd_scalper.py                   # paper trade live
  python eurusd_scalper.py --backtest        # run backtest
  python eurusd_scalper.py --symbol EURUSD   # custom symbol

Requirements:
  pip install pandas numpy python-dotenv

Integration goals (future):
  - Import EURUSDStrategy from app.eurusd_strategy into main Bot class
  - Config via config.py EURUSD_* env vars
  - Live execution via existing TradeExecutor
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────

class Config:
    SYMBOL = "EURUSD"
    LOT_SIZE = 0.01
    MIN_LOT = 0.01
    MAX_LOT = 1.0
    LOT_STEP = 0.01

    # Risk
    MAX_DAILY_LOSS_USD = 5.0
    MAX_EVENT_LOSS_USD = 2.0
    MAX_TRADES_PER_EVENT = 2
    MAX_TRADES_PER_SESSION = 3
    MAX_CONSECUTIVE_LOSSES = 2
    RE_ENTRY_COOLDOWN_SEC = 60

    # Signal
    SIGNAL_ENTRY_THRESHOLD = 0.55
    EXIT_THRESHOLD = 0.50
    ATR_PERIOD = 14
    SL_ATR = 1.5
    TP1_ATR = 2.25
    TP2_ATR = 3.75

    # Filters
    MIN_SPREAD_PIPS = 0.1
    MAX_SPREAD_PIPS = 3.0
    MIN_VOLATILITY_PIPS = 2.0
    MAX_VOLATILITY_PIPS = 25.0
    ALLOWED_SESSIONS = ["LONDON", "NEW_YORK"]

    # Bias
    BIAS_UPDATE_INTERVAL_SEC = 300
    BIAS_STRENGTH_MIN = 0.25

    # Account
    MIN_BALANCE = 10.0

    # Data
    DATA_PROVIDER = "SIMULATED"  # SIMULATED | MT5 | CAPITAL | YAHOO


config = Config()


# ────────────────────────────────────────────────────────────────────
#  Indicators
# ────────────────────────────────────────────────────────────────────

def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.full_like(values, values[-1] if len(values) > 0 else 0)
    multiplier = 2.0 / (period + 1)
    result = np.empty_like(values)
    result[:period] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
    return result


def rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


def momentum(df: pd.DataFrame) -> float:
    closes = df["close"].values
    if len(closes) < 6:
        return 0.0
    recent = closes[-3:]
    older = closes[-6:-3]
    rc = abs(recent[-1] - recent[0])
    oc = abs(older[-1] - older[0]) if len(older) >= 2 else rc
    atr_val = atr(df, 14) or 0.0001
    avg_body = np.mean(np.abs(
        df["close"].iloc[-5:].values - df["open"].iloc[-5:].values
    ))
    raw = (rc / (oc + 1e-10)) * (avg_body / (atr_val + 1e-10))
    return min(abs(raw), 1.0)


def candle_direction(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and last["close"] > prev["close"]:
        return 1
    if last["close"] < last["open"] and last["close"] < prev["close"]:
        return -1
    return 1 if last["close"] > last["open"] else -1 if last["close"] < last["open"] else 0


def volume_surge(df: pd.DataFrame, lookback: int = 10) -> bool:
    if "tick_volume" not in df.columns or len(df) < lookback + 1:
        return True
    vols = df["tick_volume"].values
    recent = vols[-1]
    avg_vol = np.mean(vols[-(lookback+1):-1])
    if avg_vol <= 0:
        return True
    return recent > avg_vol * 1.2


# ────────────────────────────────────────────────────────────────────
#  Trade Record
# ────────────────────────────────────────────────────────────────────

class Trade:
    MIN_HOLD_BARS = 3  # minimum bars before momentum decay exit
    MIN_COUNTER_CANDLES = 2  # consecutive counter-candles needed for exit

    def __init__(self, direction: str, entry_price: float, lot: float,
                 sl: float, tp1: float, tp2: float, signal_score: float):
        self.direction = direction
        self.entry_price = entry_price
        self.lot = lot
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.signal_score = signal_score
        self.entry_time = datetime.now()
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.exit_reason: Optional[str] = None
        self.pnl: float = 0.0
        self.pips: float = 0.0
        self.status = "OPEN"
        self.bars_held = 0
        self._peak_profit = 0.0

    def update(self, bid: float, ask: float, df: pd.DataFrame, atr_val: float):
        self.bars_held += 1
        mid = (bid + ask) / 2

        # Track peak profit for trail
        diff = mid - self.entry_price if self.direction == "BUY" else self.entry_price - mid
        if diff > self._peak_profit:
            self._peak_profit = diff

        # Check SL/TP (use close price for realistic fill)
        exit_price = mid  # close price
        if self.direction == "BUY":
            if bid <= self.sl:
                self.close(bid, "stop_loss")
                return
            if ask >= self.tp2:
                self.close(ask, "take_profit_full")
                return
            if ask >= self.tp1:
                self.close(ask, "take_profit_partial")
                return
        else:
            if ask >= self.sl:
                self.close(ask, "stop_loss")
                return
            if bid <= self.tp2:
                self.close(bid, "take_profit_full")
                return
            if bid <= self.tp1:
                self.close(bid, "take_profit_partial")
                return

        # Trailing stop: lock in profits after ATR * 1.5 peak
        trail_level = atr_val * 1.5
        if self._peak_profit >= trail_level:
            trail_back = self._peak_profit - atr_val * 0.8
            if self.direction == "BUY":
                new_sl = self.entry_price + trail_back
                if new_sl > self.sl:
                    self.sl = new_sl
            else:
                new_sl = self.entry_price - trail_back
                if new_sl < self.sl:
                    self.sl = new_sl

        # N consecutive counter-candles
        if len(df) >= self.MIN_COUNTER_CANDLES + 2:
            streak = 0
            for i in range(1, self.MIN_COUNTER_CANDLES + 3):
                candle = df.iloc[-i]
                d = 1 if candle["close"] > candle["open"] else -1 if candle["close"] < candle["open"] else 0
                if self.direction == "BUY" and d == -1:
                    streak += 1
                elif self.direction == "SELL" and d == 1:
                    streak += 1
                else:
                    break
            if streak >= self.MIN_COUNTER_CANDLES:
                self.close(mid, "counter_candle_streak")

        # Momentum fade exit (only after minimum hold)
        if self.bars_held >= self.MIN_HOLD_BARS:
            mom = momentum(df)
            if mom < (1.0 - config.EXIT_THRESHOLD):
                self.close(mid, "momentum_decay")

    def close(self, price: float, reason: str):
        self.exit_price = price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.status = "CLOSED"

        if self.direction == "BUY":
            self.pnl = (self.exit_price - self.entry_price) / 0.0001 * self.lot * 10
            self.pips = (self.exit_price - self.entry_price) / 0.0001
        else:
            self.pnl = (self.entry_price - self.exit_price) / 0.0001 * self.lot * 10
            self.pips = (self.entry_price - self.exit_price) / 0.0001

    def to_dict(self) -> Dict:
        return {
            "direction": self.direction,
            "entry_price": round(self.entry_price, 5),
            "exit_price": round(self.exit_price, 5) if self.exit_price else None,
            "lot": self.lot,
            "sl": round(self.sl, 5),
            "tp1": round(self.tp1, 5),
            "tp2": round(self.tp2, 5),
            "signal_score": self.signal_score,
            "pips": round(self.pips, 1),
            "pnl": round(self.pnl, 2),
            "bars_held": self.bars_held,
            "exit_reason": self.exit_reason,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status,
        }


# ────────────────────────────────────────────────────────────────────
#  Strategy Engine
# ────────────────────────────────────────────────────────────────────

class EURUSDEngine:
    def __init__(self):
        self.bias = "NEUTRAL"
        self.bias_strength = 0.0
        self.last_bias_time = 0.0
        self.current_signal: Optional[Dict] = None
        self.last_rejection: Optional[Dict] = None
        self.trades: List[Trade] = []
        self.active_trades: List[Trade] = []

        # Risk state
        self.daily_pnl = 0.0
        self.daily_date = None
        self.session_trades = 0
        self.consecutive_losses = 0
        self.last_exit_time: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None

    def update_bias(self, df_h1: pd.DataFrame) -> Dict:
        if df_h1 is None or len(df_h1) < 26:
            return self._bias_summary()

        closes = df_h1["close"].values
        ema21 = ema(closes, 21)
        ema50 = ema(closes, 50)
        price = closes[-1]
        above_ema21 = price > ema21[-1]
        above_ema50 = price > ema50[-1]
        ema21_slope = ema21[-1] - ema21[-4] if len(ema21) >= 4 else 0
        ema50_slope = ema50[-1] - ema50[-6] if len(ema50) >= 6 else 0
        atr_val = atr(df_h1, 14)
        point = 0.0001

        votes = 0.0
        votes += 0.5 if above_ema21 else -0.5
        votes += 0.5 if above_ema50 else -0.5
        votes += 0.3 if ema21_slope > 0 else -0.3 if ema21_slope < 0 else 0
        votes += 0.2 if ema50_slope > 0 else -0.2 if ema50_slope < 0 else 0

        # Swing structure
        highs, lows = [], []
        lb = 5
        for i in range(lb, len(df_h1) - lb):
            if all(df_h1["high"].iloc[i] >= df_h1["high"].iloc[i-j] for j in range(1, lb+1)) and \
               all(df_h1["high"].iloc[i] >= df_h1["high"].iloc[i+j] for j in range(1, lb+1)):
                highs.append(df_h1["high"].iloc[i])
            if all(df_h1["low"].iloc[i] <= df_h1["low"].iloc[i-j] for j in range(1, lb+1)) and \
               all(df_h1["low"].iloc[i] <= df_h1["low"].iloc[i+j] for j in range(1, lb+1)):
                lows.append(df_h1["low"].iloc[i])

        if len(highs) >= 2 and len(lows) >= 2:
            h_up = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            h_dn = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
            l_up = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
            l_dn = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            swing = (h_up - h_dn) + (l_up - l_dn)
            votes += swing / max(1, (len(highs)-1) + (len(lows)-1))

        self.bias_strength = min(abs(votes) / 2.0, 1.0)
        self.bias = "BULLISH" if votes >= 0.8 else "BEARISH" if votes <= -0.8 else "NEUTRAL"
        self.last_bias_time = time.time()
        self._bias_df = df_h1

        summary = self._bias_summary()
        summary.update({
            "ema21": round(ema21[-1], 5),
            "ema50": round(ema50[-1], 5),
            "price_to_ema21_pips": round(abs(price - ema21[-1]) / point, 1),
            "atr_pips": round(atr_val / point, 1),
        })
        return summary

    def _bias_summary(self) -> Dict:
        return {
            "bias": self.bias,
            "strength": round(self.bias_strength, 3),
            "last_updated": datetime.now().isoformat(),
        }

    def is_tradeable(self) -> bool:
        return self.bias in ("BULLISH", "BEARISH") and self.bias_strength >= config.BIAS_STRENGTH_MIN

    def can_trade(self, now: datetime) -> Tuple[bool, str]:
        # Session check — London 07-16 UTC, NY 12-21 UTC
        h = now.hour + now.minute / 60.0
        in_london = 7 <= h < 16
        in_ny = 12 <= h < 21
        in_session = in_london or in_ny
        if not in_session:
            return False, "outside_trading_sessions"

        # Daily reset
        if self.daily_date != now.date():
            self.daily_pnl = 0.0
            self.session_trades = 0
            self.consecutive_losses = 0
            self.daily_date = now.date()

        if self.session_trades >= config.MAX_TRADES_PER_SESSION:
            return False, f"session_limit ({self.session_trades}/{config.MAX_TRADES_PER_SESSION})"
        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            return False, f"consecutive_losses ({self.consecutive_losses})"
        if self.daily_pnl <= -config.MAX_DAILY_LOSS_USD:
            return False, f"daily_loss_limit (${self.daily_pnl:.2f})"

        if self.cooldown_until and now < self.cooldown_until:
            remaining = int((self.cooldown_until - now).total_seconds())
            return False, f"cooldown ({remaining}s)"

        return True, "ok"

    def find_signal(self, df_m1: pd.DataFrame, df_m5: pd.DataFrame,
                    bid: float, ask: float) -> Optional[Dict]:
        if df_m1 is None or len(df_m1) < 14:
            return None
        if self.bias not in ("BULLISH", "BEARISH"):
            return None

        direction = "BUY" if self.bias == "BULLISH" else "SELL"
        entry = ask if direction == "BUY" else bid
        closes = df_m1["close"].values

        e5 = ema(closes, 5)
        e13 = ema(closes, 13)
        e21 = ema(closes, 21)
        atr_val = atr(df_m1, 14)
        if atr_val <= 0:
            atr_val = 0.0001

        rsi_m5 = rsi(df_m5["close"].values, 14) if df_m5 is not None and len(df_m5) >= 14 else 50.0

        cross_over = e5[-1] > e13[-1] and e5[-2] <= e13[-2]
        cross_under = e5[-1] < e13[-1] and e5[-2] >= e13[-2]
        near_ema13 = abs(closes[-1] - e13[-1]) <= atr_val * 1.5
        vol_ok = volume_surge(df_m1)

        if direction == "BUY":
            if not (cross_over or (e5[-1] > e13[-1] and near_ema13)):
                return None
            if not (closes[-1] > e21[-1]):
                return None
            if rsi_m5 < 45:
                return None
            if not vol_ok:
                return None
        else:
            if not (cross_under or (e5[-1] < e13[-1] and near_ema13)):
                return None
            if not (closes[-1] < e21[-1]):
                return None
            if rsi_m5 > 55:
                return None
            if not vol_ok:
                return None

        sl_pips_points = atr_val * config.SL_ATR
        tp1_pips_points = atr_val * config.TP1_ATR
        tp2_pips_points = atr_val * config.TP2_ATR

        if direction == "BUY":
            sl = entry - sl_pips_points
            tp1 = entry + tp1_pips_points
            tp2 = entry + tp2_pips_points
        else:
            sl = entry + sl_pips_points
            tp1 = entry - tp1_pips_points
            tp2 = entry - tp2_pips_points

        score = 0.5
        spread_ema = abs(e5[-1] - e13[-1]) / atr_val
        score += min(spread_ema, 0.3)
        if cross_over or cross_under:
            score += 0.15
        if 40 <= rsi_m5 <= 60:
            score += 0.05
        mom = momentum(df_m1)
        score += mom * 0.1
        score = min(score, 1.0)

        if score < config.SIGNAL_ENTRY_THRESHOLD:
            return None

        return {
            "direction": direction,
            "score": round(score, 3),
            "entry_price": round(entry, 5),
            "sl": round(sl, 5),
            "tp1": round(tp1, 5),
            "tp2": round(tp2, 5),
            "sl_pips": round(sl_pips_points / 0.0001, 1),
            "tp1_pips": round(tp1_pips_points / 0.0001, 1),
            "tp2_pips": round(tp2_pips_points / 0.0001, 1),
            "rsi_m5": round(rsi_m5, 1),
        }

    def open_trade(self, signal: Dict, balance: float) -> Optional[Trade]:
        lot = config.LOT_SIZE * max(1, balance / 100.0)
        lot = round(lot / config.LOT_STEP) * config.LOT_STEP
        lot = max(config.MIN_LOT, min(lot, config.MAX_LOT))

        trade = Trade(
            direction=signal["direction"],
            entry_price=signal["entry_price"],
            lot=lot,
            sl=signal["sl"],
            tp1=signal["tp1"],
            tp2=signal["tp2"],
            signal_score=signal["score"],
        )
        self.active_trades.append(trade)
        self.trades.append(trade)
        self.session_trades += 1
        return trade

    def close_all(self, bid: float, ask: float, reason: str = "manual"):
        mid = (bid + ask) / 2
        for t in self.active_trades:
            if t.status == "OPEN":
                t.close(mid, reason)
        self.active_trades = [t for t in self.active_trades if t.status == "OPEN"]

    def record_exit(self, trade: Trade):
        self.daily_pnl += trade.pnl
        if trade.pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.cooldown_until = datetime.now() + timedelta(seconds=config.RE_ENTRY_COOLDOWN_SEC)

    def summary(self) -> Dict:
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.status == "CLOSED"]
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        total_pnl = sum(t.pnl for t in closed_trades)
        total_pips = sum(t.pips for t in closed_trades)

        return {
            "total_trades": total_trades,
            "closed_trades": len(closed_trades),
            "open_trades": len([t for t in self.active_trades if t.status == "OPEN"]),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / max(len(closed_trades), 1) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "total_pips": round(total_pips, 1),
            "daily_pnl": round(self.daily_pnl, 2),
            "session_trades": self.session_trades,
            "consecutive_losses": self.consecutive_losses,
            "avg_pnl": round(total_pnl / max(len(closed_trades), 1), 2),
            "avg_pips": round(total_pips / max(len(closed_trades), 1), 1),
            "bias": self.bias,
            "bias_strength": round(self.bias_strength, 3),
        }


# ────────────────────────────────────────────────────────────────────
#  Data Provider (Simulated Market)
# ────────────────────────────────────────────────────────────────────

class SimulatedMarket:
    """Generates realistic EURUSD M1 data for paper trading / backtest."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.price = 1.0800
        self._records: List[Dict] = []
        self._generate_history()

    def _generate_history(self):
        np.random.seed(42)
        n = 5000
        dt_base = pd.Timestamp("2025-01-01")

        prices = np.zeros(n)
        prices[0] = 1.0800

        hours = np.arange(n) / 60 % 24
        london = (hours >= 7) & (hours < 16)
        ny = (hours >= 13) & (hours < 22)
        overlap = (hours >= 13) & (hours < 16)
        daily_vol = np.where(overlap, 0.00025, np.where(london | ny, 0.00018, 0.00010))

        t = np.linspace(0, 6 * np.pi, n)
        tw = np.sin(t) * 0.003 + np.sin(t * 0.42) * 0.0015

        for i in range(1, n):
            r = np.random.normal(tw[i] - tw[i - 1], daily_vol[i])
            prices[i] = prices[i - 1] + r

        prices = np.clip(prices, 1.0700, 1.1000)

        for i in range(n - 1):
            now = dt_base + timedelta(minutes=i)
            h_ = now.hour + now.minute / 60.0
            ov = 13 <= h_ < 16
            ln = (7 <= h_ < 16) or (13 <= h_ < 22)
            vol = 0.00025 if ov else (0.00018 if ln else 0.00010)
            sp = 0.3 if ov else (0.5 if ln else 1.0)
            vm = 2.5 if ov else (1.5 if ln else 0.8)
            o = prices[i]
            c = prices[i + 1]
            self._records.append({
                "time": now,
                "open": o,
                "high": max(o, c) + abs(np.random.normal(0, vol)),
                "low": min(o, c) - abs(np.random.normal(0, vol)),
                "close": c,
                "tick_volume": int(np.random.randint(10, 200) * vm),
                "spread": round(sp, 1),
            })

    def _to_df(self, bars: int) -> pd.DataFrame:
        return pd.DataFrame(self._records[-bars:])

    def get_h1(self, bars: int = 100) -> pd.DataFrame:
        return self._resample("1h").iloc[-bars:]

    def get_m5(self, bars: int = 100) -> pd.DataFrame:
        return self._resample("5min").iloc[-bars:]

    def get_m1(self, bars: int = 100) -> pd.DataFrame:
        return self._to_df(bars)

    def get_bid_ask(self) -> Tuple[float, float]:
        last = self._records[-1]
        spread = last["spread"] * 0.0001
        mid = last["close"]
        return mid - spread/2, mid + spread/2

    def tick(self):
        """Advance one M1 bar."""
        last = self._records[-1]
        now = last["time"] + timedelta(minutes=1)
        h = now.hour + now.minute / 60.0

        overlap = 13 <= h < 16
        london_ny = (7 <= h < 16) or (13 <= h < 22)
        vol = 0.00025 if overlap else (0.00018 if london_ny else 0.00010)
        spread = 0.3 if overlap else (0.5 if london_ny else 1.0)
        vol_mult = 2.5 if overlap else (1.5 if london_ny else 0.8)

        ret = np.random.normal(0, vol)
        new_close = last["close"] + ret
        new_close = max(1.0700, min(1.1000, new_close))

        self._records.append({
            "time": now,
            "open": last["close"],
            "high": max(new_close, last["close"]) + abs(np.random.normal(0, vol)),
            "low": min(new_close, last["close"]) - abs(np.random.normal(0, vol)),
            "close": new_close,
            "tick_volume": int(np.random.randint(10, 200) * vol_mult),
            "spread": round(spread, 1),
        })

    def _resample(self, rule: str) -> pd.DataFrame:
        df = pd.DataFrame(self._records)
        df = df.set_index("time")
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "spread": "mean",
        }).dropna()
        return resampled.reset_index()


# ────────────────────────────────────────────────────────────────────
#  Main Loop
# ────────────────────────────────────────────────────────────────────

def print_state(engine: EURUSDEngine, bid: float, ask: float):
    s = engine.summary()
    bar = "=" * 50
    print(f"\n{bar}")
    print(f"  EURUSD Scalper  |  Bias: {s['bias']} ({s['bias_strength']})")
    print(f"  Price: {bid:.5f} / {ask:.5f}  |  Spread: {(ask-bid)/0.0001:.1f} pips")
    print(f"  Trades: {s['closed_trades']} closed / {s['open_trades']} open")
    print(f"  Win Rate: {s['win_rate']}%  |  PnL: ${s['total_pnl']:.2f} ({s['total_pips']:.1f} pips)")
    print(f"  Daily: ${s['daily_pnl']:.2f}  |  Session: {s['session_trades']}/{config.MAX_TRADES_PER_SESSION}")
    print(f"  Consecutive Losses: {s['consecutive_losses']}")
    if engine.active_trades:
        for t in engine.active_trades:
            if t.status == "OPEN":
                print(f"  >> OPEN: {t.direction} {t.lot} lot @ {t.entry_price:.5f} | SL: {t.sl:.5f} TP1: {t.tp1:.5f} TP2: {t.tp2:.5f}")
    print(bar)


def run_live(duration_minutes: int = 60, show_chart: bool = False):
    engine = EURUSDEngine()
    market = SimulatedMarket()

    print(f"\n{'='*60}")
    print(f"  EURUSD Scalper — Live Paper Trading")
    print(f"  Sessions: {', '.join(config.ALLOWED_SESSIONS)}")
    print(f"  Max trades/session: {config.MAX_TRADES_PER_SESSION}")
    print(f"  SL: {config.SL_ATR}x ATR | TP1: {config.TP1_ATR}x ATR | TP2: {config.TP2_ATR}x ATR")
    print(f"  Max daily loss: ${config.MAX_DAILY_LOSS_USD}")
    print(f"{'='*60}\n")

    bias_update_counter = 0
    balance = 100.0

    start = time.time()
    while (time.time() - start) < duration_minutes * 60:
        now = datetime.now()

        # Advance market
        market.tick()
        bid, ask = market.get_bid_ask()
        df_m1 = market.get_m1(100)
        df_m5 = market.get_m5(50)

        # Update bias periodically
        bias_update_counter += 1
        if bias_update_counter % 5 == 0:
            df_h1 = market.get_h1(100)
            engine.update_bias(df_h1)

        # Strategy tick
        can, reason = engine.can_trade(now)
        if can:
            if engine.is_tradeable() and len(engine.active_trades) < config.MAX_TRADES_PER_EVENT:
                signal = engine.find_signal(df_m1, df_m5, bid, ask)
                if signal:
                    trade = engine.open_trade(signal, balance)
                    if trade:
                        print(f"\n  >> ENTRY: {trade.direction} {trade.lot:.2f} lot @ {trade.entry_price:.5f} "
                              f"(score={signal['score']:.3f} | SL={signal['sl_pips']:.1f}p TP1={signal['tp1_pips']:.1f}p TP2={signal['tp2_pips']:.1f}p)")

        # Update open trades
        for trade in engine.active_trades[:]:
            if trade.status == "OPEN":
                trade.update(bid, ask, df_m1, atr(df_m1))
                if trade.status == "CLOSED":
                    engine.record_exit(trade)
                    print(f"  << EXIT: {trade.direction} | {trade.exit_reason} | "
                          f"PnL=${trade.pnl:.2f} ({trade.pips:.1f}p) | "
                          f"bars={trade.bars_held}")

        # Print state every 10 ticks
        if bias_update_counter % 10 == 0:
            print_state(engine, bid, ask)

        time.sleep(0.05)

    engine.close_all(bid, ask, "session_end")
    print(f"\n{'='*60}")
    print(f"  SESSION END — Final Summary")
    print(f"{'='*60}")
    s = engine.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    return engine


def run_backtest(days: int = 30):
    """Run backtest over historical data."""
    engine = EURUSDEngine()
    market = SimulatedMarket()

    print(f"\n{'='*60}")
    print(f"  EURUSD Scalper — Backtest ({days} days simulated)")
    print(f"{'='*60}\n")

    balance = 100.0
    total_bars = days * 24 * 60
    report_interval = max(1, total_bars // 20)

    for i in range(total_bars):
        now = datetime.now()
        market.tick()
        bid, ask = market.get_bid_ask()
        df_m1 = market.get_m1(100)
        df_m5 = market.get_m5(50)

        if i % 5 == 0:
            df_h1 = market.get_h1(100)
            engine.update_bias(df_h1)

        can, reason = engine.can_trade(now)
        if can:
            if engine.is_tradeable() and len(engine.active_trades) < config.MAX_TRADES_PER_EVENT:
                signal = engine.find_signal(df_m1, df_m5, bid, ask)
                if signal:
                    engine.open_trade(signal, balance)

        for trade in engine.active_trades[:]:
            if trade.status == "OPEN":
                trade.update(bid, ask, df_m1, atr(df_m1))
                if trade.status == "CLOSED":
                    engine.record_exit(trade)

        if (i + 1) % report_interval == 0:
            s = engine.summary()
            print(f"  [{i+1:>6}/{total_bars}]  Trades: {s['total_trades']}  "
                  f"Win Rate: {s['win_rate']}%  PnL: ${s['total_pnl']:.2f}  "
                  f"Pips: {s['total_pips']:.1f}  "
                  f"Bias: {s['bias']}")

    engine.close_all(bid, ask, "backtest_end")
    s = engine.summary()
    print(f"\n{'='*60}")
    print(f"  BACKTEST COMPLETE — Final Results")
    print(f"{'='*60}")
    for k, v in s.items():
        print(f"  {k}: {v}")

    # Print trade list
    if engine.trades:
        print(f"\n  Trade Log:")
        print(f"  {'#':>3} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'Pips':>7} {'PnL':>8} {'Reason':<22} {'Bars':>4}")
        print(f"  {'-'*70}")
        for i, t in enumerate(engine.trades):
            if t.status == "CLOSED":
                print(f"  {i+1:>3} {t.direction:<5} {t.entry_price:>10.5f} {t.exit_price:>10.5f} "
                      f"{t.pips:>7.1f} ${t.pnl:>7.2f} {t.exit_reason:<22} {t.bars_held:>4}")

    return engine


# ────────────────────────────────────────────────────────────────────
#  Entry Point
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EURUSD Scalper")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--days", type=int, default=30, help="Backtest days")
    parser.add_argument("--duration", type=int, default=60, help="Live duration (minutes)")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol (default EURUSD)")
    args = parser.parse_args()

    config.SYMBOL = args.symbol.upper()

    if args.backtest:
        run_backtest(days=args.days)
    else:
        run_live(duration_minutes=args.duration)
