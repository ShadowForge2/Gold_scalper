import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
from datetime import datetime


DEFAULT_CONFIG = {
    "ema_fast": 5,
    "ema_slow": 8,
    "ema_trend_period": 21,
    "atr_period": 14,
    "sl_atr_mult": 1.0,
    "tp_atr_mult": 3.0,
    "rsi_buy_min": 35,
    "rsi_sell_max": 65,
    "rsi_period": 14,
    "volume_surge_mult": 1.2,
    "volume_lookback": 10,
    "max_dist_atr": 3.5,
    "candle_dir_lookback": 2,
    "max_bars_hold": 240,
    "retest_range_atr": 0.2,
    "cooldown_seconds": 30,
    "consec_loss_limit": 4,
    "max_daily_trades": 10,
    "max_daily_loss": 5.0,
    "min_balance": 10.0,
    "use_candle_filter": True,
    "use_volume_filter": False,
    "score_threshold": 0.70,
    "lot_size": 0.01,
    "point": 0.0001,
}


class EURUSDStrategy:
    """
    EURUSD Scalping Strategy — configurable parameterized version.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.current_signal: Optional[Dict] = None
        self.last_rejection: Optional[Dict] = None
        self.atr_value: float = 0.0
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

    def _m5_trend(self, m5_data: pd.DataFrame) -> str:
        if m5_data is None or len(m5_data) < 30:
            return "NEUTRAL"
        closes = m5_data["close"].values
        ema = self._ema(closes, self.cfg["ema_trend_period"])
        atr = self._compute_atr(m5_data, self.cfg["atr_period"]) or 0.0001
        price_dist = abs(closes[-1] - ema[-1]) / atr
        if price_dist > self.cfg["max_dist_atr"]:
            return "NEUTRAL"
        if closes[-1] > ema[-1]:
            return "BULLISH"
        if closes[-1] < ema[-1]:
            return "BEARISH"
        return "NEUTRAL"

    def _volume_surge(self, df: pd.DataFrame) -> bool:
        if not self.cfg["use_volume_filter"]:
            return True
        lookback = self.cfg["volume_lookback"]
        if "tick_volume" not in df.columns or len(df) < lookback + 1:
            return True
        vols = df["tick_volume"].values
        recent = vols[-1]
        avg_vol = np.mean(vols[-(lookback + 1):-1])
        if avg_vol <= 0:
            return True
        return recent > avg_vol * self.cfg["volume_surge_mult"]

    def evaluate(self, m1_data: pd.DataFrame, m5_data: pd.DataFrame,
                 bid: float, ask: float) -> Optional[Dict]:
        c = self.cfg
        if m1_data is None or len(m1_data) < 20:
            return self._reject("insufficient_m1_data")
        if m5_data is None or len(m5_data) < 30:
            return self._reject("insufficient_m5_data")

        trend = self._m5_trend(m5_data)
        if trend == "NEUTRAL":
            return self._reject("m5_trend_neutral")

        closes = m1_data["close"].values
        ema_fast = self._ema(closes, c["ema_fast"])
        ema_slow = self._ema(closes, c["ema_slow"])

        atr = self._compute_atr(m1_data, c["atr_period"])
        if atr <= 0:
            atr = 0.0001
        self.atr_value = atr

        rsi_m5 = self._compute_rsi(m5_data, c["rsi_period"])
        vol_ok = self._volume_surge(m1_data)

        # Confirm last N candle direction matches trade direction
        if c["use_candle_filter"] and len(m1_data) >= c["candle_dir_lookback"] + 1:
            look = c["candle_dir_lookback"]
            bullish_candles = sum(
                1 for i in range(1, look + 1)
                if m1_data["close"].iloc[-i] > m1_data["open"].iloc[-i]
            )
            c1_dir = bullish_candles > 0
            c2_dir = bullish_candles >= look // 2 + 1 if look > 1 else True
        else:
            c1_dir = True
            c2_dir = True

        entry_mode = None

        if trend == "BULLISH":
            if c["use_candle_filter"] and not (c1_dir or c2_dir):
                return self._reject("buy_bearish_candle_streak")
            crossover = ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]
            retest = (ema_fast[-1] > ema_slow[-1] and
                      closes[-1] <= ema_slow[-1] + atr * c["retest_range_atr"] and
                      closes[-1] >= ema_slow[-1] - atr * c["retest_range_atr"])
            if not (crossover or retest):
                return self._reject("buy_no_entry_setup")
            if rsi_m5 < c["rsi_buy_min"]:
                return self._reject("buy_rsi_too_low", rsi=rsi_m5)
            if not vol_ok:
                return self._reject("buy_no_volume")
            entry_mode = "crossover" if crossover else "retest"
            sl_price = ask - atr * c["sl_atr_mult"]
            tp = ask + atr * c["tp_atr_mult"]
        else:
            if c["use_candle_filter"] and not ((not c1_dir) or (not c2_dir)):
                return self._reject("sell_bullish_candle_streak")
            crossover = ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]
            retest = (ema_fast[-1] < ema_slow[-1] and
                      closes[-1] >= ema_slow[-1] - atr * c["retest_range_atr"] and
                      closes[-1] <= ema_slow[-1] + atr * c["retest_range_atr"])
            if not (crossover or retest):
                return self._reject("sell_no_entry_setup")
            if rsi_m5 > c["rsi_sell_max"]:
                return self._reject("sell_rsi_too_high", rsi=rsi_m5)
            if not vol_ok:
                return self._reject("sell_no_volume")
            entry_mode = "crossover" if crossover else "retest"
            sl_price = bid + atr * c["sl_atr_mult"]
            tp = bid - atr * c["tp_atr_mult"]

        direction = "BUY" if trend == "BULLISH" else "SELL"
        entry_price = ask if direction == "BUY" else bid
        sl_pips = atr / c["point"] * c["sl_atr_mult"]
        tp_pips = atr / c["point"] * c["tp_atr_mult"]
        score = self._compute_entry_score(m1_data, rsi_m5, entry_mode)

        signal = {
            "direction": direction,
            "score": round(score, 3),
            "entry_price": round(entry_price, 5),
            "sl": round(sl_price, 5),
            "tp1": round(tp, 5),
            "tp2": round(tp, 5),
            "atr_pips": round(sl_pips, 1),
            "tp1_pips": round(tp_pips, 1),
            "tp2_pips": round(tp_pips, 1),
            "rsi_m5": round(rsi_m5, 1),
            "ema3": round(ema_fast[-1], 5),
            "ema8": round(ema_slow[-1], 5),
            "entry_mode": entry_mode,
            "timestamp": datetime.now().isoformat(),
        }

        self.current_signal = signal
        self.last_rejection = None
        return signal

    def evaluate_exit(self, m1_data: pd.DataFrame, entry_price: float,
                      direction: str, signal: Dict) -> Tuple[bool, float, str]:
        if m1_data is None or len(m1_data) < 5:
            return False, 0.0, "insufficient_data"

        atr = self._compute_atr(m1_data, self.cfg["atr_period"])
        if atr <= 0:
            atr = 0.0001

        px = m1_data["close"].iloc[-1]
        sl_price = signal.get("sl")
        tp_price = signal.get("tp1")

        if direction == "BUY":
            if px <= sl_price:
                return True, 1.0, "stop_loss"
            if px >= tp_price:
                return True, 1.0, "take_profit"
            if len(m1_data) >= self.cfg["max_bars_hold"]:
                return True, 0.5, "max_hold"
        else:
            if px >= sl_price:
                return True, 1.0, "stop_loss"
            if px <= tp_price:
                return True, 1.0, "take_profit"
            if len(m1_data) >= self.cfg["max_bars_hold"]:
                return True, 0.5, "max_hold"

        return False, 0.0, "holding"

    def _compute_entry_score(self, m1_data: pd.DataFrame, rsi: float, entry_mode: str = "crossover") -> float:
        c = self.cfg
        atr = self._compute_atr(m1_data, c["atr_period"]) or 0.0001
        closes = m1_data["close"].values
        ema_fast_arr = self._ema(closes, c["ema_fast"])
        ema_slow_arr = self._ema(closes, c["ema_slow"])
        score = 0.5
        if entry_mode == "crossover":
            score += 0.15
        spread_ema = abs(ema_fast_arr[-1] - ema_slow_arr[-1]) / atr
        score += min(spread_ema * 0.3, 0.2)
        if c["rsi_buy_min"] <= rsi <= c["rsi_sell_max"]:
            score += 0.1
        mom = self._compute_momentum(m1_data)
        score += mom * 0.1
        return min(score, 1.0)

    def _compute_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 50.0
        closes = df["close"].values
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_momentum(self, df: pd.DataFrame) -> float:
        closes = df["close"].values
        if len(closes) < 6:
            return 0.0
        recent = closes[-3:]
        older = closes[-6:-3]
        recent_change = abs(recent[-1] - recent[0])
        older_change = abs(older[-1] - older[0]) if len(older) >= 2 else recent_change
        atr = self._compute_atr(df, 14) or 0.0001
        raw = (recent_change / (older_change + 1e-10)) * \
              (np.mean(np.abs(df["close"].iloc[-5:].values - df["open"].iloc[-5:].values)) / (atr + 1e-10))
        return min(abs(raw), 1.0)

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]),
                                   np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:]))

    def _ema(self, values: np.ndarray, period: int) -> np.ndarray:
        if len(values) < period:
            return np.full_like(values, values[-1] if len(values) > 0 else 0)
        multiplier = 2.0 / (period + 1)
        result = np.empty_like(values)
        result[:period] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
        return result

    def _reject(self, reason: str, **context) -> None:
        self.last_rejection = {
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            **context,
        }
        self.current_signal = None
        return None
