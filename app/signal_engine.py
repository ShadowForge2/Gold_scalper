import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime


class SignalEngine:
    def __init__(self):
        self.current_signal: Optional[Dict] = None
        self.last_signal_time: Optional[datetime] = None

    def evaluate(self, m1_data: pd.DataFrame, bias: Dict,
                 current_price: float,
                 h1_high: float = None, h1_low: float = None) -> Optional[Dict]:

        if m1_data is None or len(m1_data) < 10:
            return None

        if bias.get("bias") not in ("BULLISH", "BEARISH"):
            return None

        if h1_high is None or h1_low is None or h1_high <= h1_low:
            return None

        bias_dir = bias["bias"]
        direction = "BUY" if bias_dir == "BULLISH" else "SELL"
        closes = m1_data["close"].values
        range_size = h1_high - h1_low

        if direction == "BUY":
            if current_price <= h1_high:
                return None
            breakout_dist = current_price - h1_high
            recent_in_range = any(h1_low <= closes[-i] <= h1_high for i in range(2, min(6, len(closes))))
            if not recent_in_range:
                return None
        else:
            if current_price >= h1_low:
                return None
            breakout_dist = h1_low - current_price
            recent_in_range = any(h1_low <= closes[-i] <= h1_high for i in range(2, min(6, len(closes))))
            if not recent_in_range:
                return None

        score = min(breakout_dist / range_size, 1.0)
        if score < 0.02:
            return None

        signal = {
            "direction": direction,
            "bias": bias_dir,
            "score": round(score, 3),
            "conviction": round(score, 3),
            "breakout_dist": round(breakout_dist, 2),
            "range_size": round(range_size, 2),
            "entry_price": current_price,
            "timestamp": datetime.now().isoformat(),
        }

        self.current_signal = signal
        return signal

    def evaluate_exit(self, m1_data: pd.DataFrame,
                      entry_price: float,
                      direction: str,
                      entry_score: float = None,
                      exit_threshold: float = 0.50,
                      exit_mode: int = 1) -> Tuple[bool, float, str]:

        if m1_data is None or len(m1_data) < 5:
            return False, 0.0, "insufficient_data"

        if exit_mode == 4:
            return self._exit_mode_breakout(m1_data, entry_price, direction, entry_score)
        if exit_mode == 1:
            return self._exit_mode_trail_sl(m1_data, entry_price, direction, exit_threshold)
        elif exit_mode == 2:
            return self._exit_mode_fixed_tp_sl(m1_data, entry_price, direction, exit_threshold)
        elif exit_mode == 3:
            return self._exit_mode_atr_anticandle(m1_data, entry_price, direction, exit_threshold)
        return self._exit_mode_trail_sl(m1_data, entry_price, direction, exit_threshold)

    def _exit_mode_breakout(self, df, entry, direction, entry_score):
        """Breakout exit: SL using high/low, ATR trailing, counter-candle."""
        momentum = self._compute_momentum(df)
        atr = self._compute_atr(df, 14)

        last_high = df["high"].iloc[-1]
        last_low = df["low"].iloc[-1]
        last_close = df["close"].iloc[-1]

        if atr <= 0:
            return False, 0.0, "atr_zero"

        sl_dist = max(atr * 1.0, 0.20)

        if direction == "BUY":
            stop_price = entry - sl_dist
            if last_low <= stop_price:
                return True, 1.0, "stop_loss"
            diff = last_close - entry
            worst = entry - min(df["low"].values[-5:])
        else:
            stop_price = entry + sl_dist
            if last_high >= stop_price:
                return True, 1.0, "stop_loss"
            diff = entry - last_close
            worst = float(np.max(df["high"].values[-5:])) - entry

        if worst >= atr * 1.0:
            pullback = worst - max(0, diff)
            if pullback >= atr * 0.5:
                return True, 0.85, "atr_trail"

        if len(df) >= 4:
            last_dir = self._candle_direction(df)
            prev_dir = self._candle_direction(df.iloc[:-1])
            if (direction == "BUY" and last_dir == -1 and prev_dir == -1) or \
               (direction == "SELL" and last_dir == 1 and prev_dir == 1):
                return True, 0.75, "double_counter_candle"

        exit_score = 1.0 - momentum
        if exit_score > 0.65:
            return True, round(exit_score, 3), "momentum_decay"
        return False, round(exit_score, 3), "holding"

    def _exit_mode_trail_sl(self, df, entry, direction, exit_threshold):
        momentum = self._compute_momentum(df)
        px = df["close"].iloc[-1]
        diff = px - entry if direction == "BUY" else entry - px

        stop_loss = -0.15
        trail_trigger = 0.20
        trail_retrace_pct = 0.50

        if diff <= stop_loss:
            return True, 1.0, "stop_loss"

        if len(df) >= 5:
            highs = df["high"].values[-5:] if direction == "BUY" else df["close"].values[-5:]
            refs = df["low"].values[-5:] if direction == "SELL" else df["close"].values[-5:]
            peak = float(np.max(highs - entry)) if direction == "BUY" else float(np.max(entry - refs))
            if peak >= trail_trigger:
                pullback = peak - max(0, diff)
                if pullback / peak > trail_retrace_pct:
                    return True, 0.85, "trail_stop"

        if len(df) >= 4:
            last_dir = self._candle_direction(df)
            prev_dir = self._candle_direction(df.iloc[:-1])
            if (direction == "BUY" and last_dir == -1 and prev_dir == -1) or \
               (direction == "SELL" and last_dir == 1 and prev_dir == 1):
                return True, 0.75, "double_counter_candle"

        exit_score = 1.0 - momentum
        if exit_score > exit_threshold:
            return True, round(exit_score, 3), "momentum_decay"
        return False, round(exit_score, 3), "holding"

    def _exit_mode_fixed_tp_sl(self, df, entry, direction, exit_threshold):
        px = df["close"].iloc[-1]
        diff = px - entry if direction == "BUY" else entry - px

        if diff <= -0.15:
            return True, 1.0, "stop_loss"
        if diff >= 0.60:
            return True, 0.0, "take_profit"
        return False, 0.0, "holding"

    def _exit_mode_atr_anticandle(self, df, entry, direction, exit_threshold):
        momentum = self._compute_momentum(df)
        px = df["close"].iloc[-1]
        diff = px - entry if direction == "BUY" else entry - px

        atr = self._compute_atr(df, 14)

        if diff <= -0.15:
            return True, 1.0, "stop_loss"

        if atr > 0 and len(df) >= 5:
            highs = df["high"].values[-5:] if direction == "BUY" else df["close"].values[-5:]
            refs = df["low"].values[-5:] if direction == "SELL" else df["close"].values[-5:]
            peak = float(np.max(highs - entry)) if direction == "BUY" else float(np.max(entry - refs))
            if peak >= atr:
                pullback = peak - max(0, diff)
                if pullback >= atr * 0.5:
                    return True, 0.85, "atr_trail"

        if len(df) >= 3:
            last_dir = self._candle_direction(df)
            if (direction == "BUY" and last_dir == -1) or (direction == "SELL" and last_dir == 1):
                return True, 0.67, "counter_candle"

        exit_score = 1.0 - momentum
        if exit_score > exit_threshold:
            return True, round(exit_score, 3), "momentum_decay"
        return False, round(exit_score, 3), "holding"

    def _candle_direction(self, df: pd.DataFrame) -> int:
        if len(df) < 2:
            return 0
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last["close"] > last["open"] and last["close"] > prev["close"]:
            return 1
        if last["close"] < last["open"] and last["close"] < prev["close"]:
            return -1
        return 1 if last["close"] > last["open"] else -1 if last["close"] < last["open"] else 0

    def _compute_momentum(self, df: pd.DataFrame) -> float:
        closes = df["close"].values
        if len(closes) < 5:
            return 0.0

        recent = closes[-3:]
        older = closes[-6:-3] if len(closes) >= 6 else closes[:3]

        recent_change = abs(recent[-1] - recent[0])
        older_change = abs(older[-1] - older[0]) if len(older) >= 2 else recent_change

        avg_body = np.mean(np.abs(
            df["close"].iloc[-5:].values - df["open"].iloc[-5:].values
        ))
        if avg_body == 0:
            return 0.0

        raw = (recent_change / (older_change + 1e-10)) * \
               (avg_body / (df["close"].iloc[-1] * 0.001 + 1e-10))
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

    def _candle_strength(self, df: pd.DataFrame) -> float:
        if len(df) < 3:
            return 0.0
        candle = df.iloc[-1]
        body = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]
        if total_range == 0:
            return 0.0
        return min(body / total_range, 1.0)
