import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime

import config as cfg

try:
    from app.direction_predictor import DirectionPredictor, SLTPredictor, compute_features
    _HAS_ML = True
except ImportError:
    DirectionPredictor = None
    SLTPredictor = None
    compute_features = None
    _HAS_ML = False


class SignalEngine:
    def __init__(self,
                 direction_predictor: Optional["DirectionPredictor"] = None,
                 slt_predictor: Optional["SLTPredictor"] = None):
        self.current_signal: Optional[Dict] = None
        self.last_signal_time: Optional[datetime] = None
        self.last_rejection: Optional[Dict] = None
        self._direction_predictor = direction_predictor
        self._slt_predictor = slt_predictor
        self._ml_override_count = 0
        self._ml_override_day = None

    def _reject(self, reason: str, **context) -> None:
        self.last_rejection = {
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            **context,
        }
        return None

    def _get_features(self, m1_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample M1 to M5+H1, compute features. Returns feature DataFrame or None."""
        try:
            if "time" in m1_data.columns:
                m1_idx = m1_data.set_index("time")
            else:
                m1_idx = m1_data
            if len(m1_idx) < 10:
                return None
            m5 = m1_idx.resample("5min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "tick_volume": "sum",
            }).dropna()
            if len(m5) < 5:
                return None
            h1 = m1_idx.resample("1h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
            }).dropna()
            return compute_features(m5, h1)
        except Exception:
            return None

    def _get_ml_direction(self, m1_data: pd.DataFrame, expected_direction: str = None) -> Optional[str]:
        """Use ML model to predict trade viability. Returns 'BUY', 'SELL', or None.
        - If SL/TP model available: checks if expected direction will hit TP before SL
        - Falls back to direction model: checks if price direction matches expected
        """
        features = self._get_features(m1_data)
        if features is None or len(features) == 0:
            return None

        try:
            # SL/TP model: directly predicts trade outcome
            if self._slt_predictor is not None and expected_direction is not None:
                if expected_direction == "BUY":
                    prob = self._slt_predictor.buy_win_prob(features)
                    return "BUY" if prob >= cfg.ML_CONFIDENCE_THRESHOLD else None
                elif expected_direction == "SELL":
                    prob = self._slt_predictor.sell_win_prob(features)
                    return "SELL" if prob >= cfg.ML_CONFIDENCE_THRESHOLD else None

            # Fallback: direction model predicts generic price direction
            if self._direction_predictor is not None:
                return self._direction_predictor.predict(
                    features, confidence_threshold=cfg.ML_CONFIDENCE_THRESHOLD
                )
        except Exception:
            pass

        return None

    def _get_ml_unbiased_prediction(self, features: pd.DataFrame) -> Tuple[Optional[str], float]:
        """Get ML prediction without bias filtering. Returns (direction, confidence) or (None, 0.0)."""
        if features is None or len(features) == 0:
            return None, 0.0
        try:
            override_threshold = getattr(cfg, 'ML_BIAS_OVERRIDE_THRESHOLD', 0.70)
            if self._slt_predictor is not None:
                buy_p = self._slt_predictor.buy_win_prob(features)
                sell_p = self._slt_predictor.sell_win_prob(features)
                if buy_p >= override_threshold and buy_p > sell_p:
                    return "BUY", buy_p
                if sell_p >= override_threshold and sell_p > buy_p:
                    return "SELL", sell_p
            if self._direction_predictor is not None:
                prob_down, prob_up = self._direction_predictor.predict_proba(features)
                if prob_up >= override_threshold and prob_up > prob_down:
                    return "BUY", prob_up
                if prob_down >= override_threshold and prob_down > prob_up:
                    return "SELL", prob_down
        except Exception:
            pass
        return None, 0.0

    def _get_ml_exit_signal(self, features: pd.DataFrame, trade_direction: str) -> Optional[str]:
        """ML-guided exit: 'hold' (ML agrees, suppress other exit signals),
        'exit' (ML reversal, exit now), or None (undecided, use normal logic)."""
        if features is None or len(features) == 0:
            return None
        try:
            if self._direction_predictor is not None:
                prob_down, prob_up = self._direction_predictor.predict_proba(features)
                hold_threshold = getattr(cfg, 'ML_HOLD_CONFIDENCE', 0.50)
                exit_threshold = getattr(cfg, 'ML_BIAS_OVERRIDE_THRESHOLD', 0.70)
                if trade_direction == "BUY":
                    if prob_up >= hold_threshold:
                        return "hold"
                    if prob_down >= exit_threshold:
                        return "exit"
                else:
                    if prob_down >= hold_threshold:
                        return "hold"
                    if prob_up >= exit_threshold:
                        return "exit"
        except Exception:
            pass
        return None

    def reset_ml_override_count(self):
        self._ml_override_count = 0
        self._ml_override_day = None

    def evaluate(self, m1_data: pd.DataFrame, bias: Dict,
                 current_price: float,
                 h1_high: float = None, h1_low: float = None) -> Optional[Dict]:

        if m1_data is None or len(m1_data) < 10:
            return self._reject(
                "insufficient_m1_data",
                bars=0 if m1_data is None else len(m1_data),
                bias=bias.get("bias", "UNKNOWN"),
                price=current_price,
                h1_high=h1_high,
                h1_low=h1_low,
            )

        if bias.get("bias") not in ("BULLISH", "BEARISH"):
            return self._reject(
                "bias_not_tradeable",
                bias=bias.get("bias", "UNKNOWN"),
                price=current_price,
                h1_high=h1_high,
                h1_low=h1_low,
            )

        if h1_high is None or h1_low is None or h1_high <= h1_low:
            return self._reject(
                "invalid_h1_range",
                bias=bias.get("bias", "UNKNOWN"),
                price=current_price,
                h1_high=h1_high,
                h1_low=h1_low,
            )

        bias_dir = bias["bias"]
        direction = "BUY" if bias_dir == "BULLISH" else "SELL"
        closes = m1_data["close"].values
        range_size = h1_high - h1_low
        atr = self._compute_atr(m1_data, cfg.ATR_PERIOD)
        atr_entry_threshold = round(atr * cfg.ATR_MULTIPLIER / range_size, 4) if range_size > 0 and atr > 0 else None

        # ML bias override: if ML confidently predicts opposite direction, trust ML
        ml_override = False
        if (self._direction_predictor is not None or self._slt_predictor is not None) and _HAS_ML:
            today = datetime.now().day
            if today != self._ml_override_day:
                self._ml_override_count = 0
                self._ml_override_day = today
            if self._ml_override_count < getattr(cfg, 'ML_OVERRIDE_MAX_PER_SESSION', 3):
                ml_features = self._get_features(m1_data)
                if ml_features is not None and len(ml_features) > 0:
                    ml_dir, ml_conf = self._get_ml_unbiased_prediction(ml_features)
                    if ml_dir is not None and ml_dir != direction:
                        direction = ml_dir
                        ml_override = True

        if direction == "BUY":
            if ml_override:
                breakout_dist = current_price - h1_low
                score = breakout_dist / range_size if range_size > 0 else 0.0
                if breakout_dist <= 0:
                    return self._reject(
                        "buy_price_at_or_below_h1_low",
                        direction=direction, bias=bias_dir,
                        price=current_price, h1_high=h1_high, h1_low=h1_low,
                        breakout_dist=round(breakout_dist, 2),
                        range_size=round(range_size, 2),
                        score=round(score, 3),
                        ml_override=True,
                    )
            else:
                breakout_dist = current_price - h1_high
                score = breakout_dist / range_size if range_size > 0 else 0.0
                if current_price <= h1_high:
                    return self._reject(
                        "buy_price_not_above_h1_high",
                        direction=direction, bias=bias_dir,
                        price=current_price, h1_high=h1_high, h1_low=h1_low,
                        breakout_dist=round(breakout_dist, 2),
                        range_size=round(range_size, 2),
                        score=round(score, 3),
                    )
                lookback = max(2, cfg.RECENT_PULLBACK_LOOKBACK + 1)
                recent_in_range = any(h1_low <= closes[-i] <= h1_high for i in range(2, min(lookback, len(closes))))
                if cfg.REQUIRE_RECENT_PULLBACK and not recent_in_range:
                    return self._reject(
                        "buy_no_recent_pullback_inside_h1_range",
                        direction=direction, bias=bias_dir,
                        price=current_price, h1_high=h1_high, h1_low=h1_low,
                        breakout_dist=round(breakout_dist, 2),
                        range_size=round(range_size, 2),
                        score=round(score, 3),
                    )
        else:
            if ml_override:
                breakout_dist = h1_high - current_price
                score = breakout_dist / range_size if range_size > 0 else 0.0
                if breakout_dist <= 0:
                    return self._reject(
                        "sell_price_at_or_above_h1_high",
                        direction=direction, bias=bias_dir,
                        price=current_price, h1_high=h1_high, h1_low=h1_low,
                        breakout_dist=round(breakout_dist, 2),
                        range_size=round(range_size, 2),
                        score=round(score, 3),
                        ml_override=True,
                    )
            else:
                breakout_dist = h1_low - current_price
                score = breakout_dist / range_size if range_size > 0 else 0.0
                if current_price >= h1_low:
                    return self._reject(
                        "sell_price_not_below_h1_low",
                        direction=direction, bias=bias_dir,
                        price=current_price, h1_high=h1_high, h1_low=h1_low,
                        breakout_dist=round(breakout_dist, 2),
                        range_size=round(range_size, 2),
                        score=round(score, 3),
                    )
                lookback = max(2, cfg.RECENT_PULLBACK_LOOKBACK + 1)
                recent_in_range = any(h1_low <= closes[-i] <= h1_high for i in range(2, min(lookback, len(closes))))
                if cfg.REQUIRE_RECENT_PULLBACK and not recent_in_range:
                    return self._reject(
                        "sell_no_recent_pullback_inside_h1_range",
                        direction=direction, bias=bias_dir,
                        price=current_price, h1_high=h1_high, h1_low=h1_low,
                        breakout_dist=round(breakout_dist, 2),
                        range_size=round(range_size, 2),
                        score=round(score, 3),
                    )

        score = min(breakout_dist / range_size, 1.0)
        if score < cfg.MIN_BREAKOUT_SCORE:
            return self._reject(
                "breakout_score_too_small",
                direction=direction,
                bias=bias_dir,
                price=current_price,
                h1_high=h1_high,
                h1_low=h1_low,
                breakout_dist=round(breakout_dist, 2),
                range_size=round(range_size, 2),
                score=round(score, 3),
                threshold=cfg.MIN_BREAKOUT_SCORE,
            )

        # ML direction validation (skip if already overridden by ML)
        if not ml_override and (self._direction_predictor is not None or self._slt_predictor is not None) and _HAS_ML:
            ml_direction = self._get_ml_direction(m1_data, expected_direction=direction)
            if ml_direction is None:
                return self._reject(
                    "ml_confidence_too_low",
                    direction=direction,
                    bias=bias_dir,
                    price=current_price,
                    h1_high=h1_high,
                    h1_low=h1_low,
                    score=round(score, 3),
                )
            if ml_direction != direction:
                return self._reject(
                    "ml_direction_conflict",
                    direction=direction,
                    ml_direction=ml_direction,
                    bias=bias_dir,
                    price=current_price,
                    h1_high=h1_high,
                    h1_low=h1_low,
                    score=round(score, 3),
                )

        atr_val = atr if atr > 0 else (current_price * 0.001)
        if direction == "BUY":
            sl_price = current_price - atr_val * cfg.SL_ATR_MULTIPLIER
            tp1 = current_price + atr_val * cfg.TP1_MULTIPLIER
            tp2 = current_price + atr_val * cfg.TP2_MULTIPLIER
            tp3 = current_price + atr_val * cfg.TP3_MULTIPLIER
        else:
            sl_price = current_price + atr_val * cfg.SL_ATR_MULTIPLIER
            tp1 = current_price - atr_val * cfg.TP1_MULTIPLIER
            tp2 = current_price - atr_val * cfg.TP2_MULTIPLIER
            tp3 = current_price - atr_val * cfg.TP3_MULTIPLIER

        signal = {
            "direction": direction,
            "bias": bias_dir,
            "score": round(score, 3),
            "conviction": round(score, 3),
            "breakout_dist": round(breakout_dist, 2),
            "range_size": round(range_size, 2),
            "atr_entry_threshold": atr_entry_threshold,
            "entry_price": current_price,
            "sl": round(sl_price, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "atr_value": round(atr_val, 4),
            "ml_override": ml_override,
            "timestamp": datetime.now().isoformat(),
        }

        self.current_signal = signal
        self.last_rejection = None
        if ml_override:
            self._ml_override_count += 1
        return signal

    def evaluate_exit(self, m1_data: pd.DataFrame,
                      entry_price: float,
                      direction: str,
                      entry_score: float = None,
                      exit_threshold: float = 0.50,
                      exit_mode: int = 1,
                      trail_sl_mult: float = None,
                      trail_trigger_mult: float = None,
                      trail_retrace: float = None,
                      trail_to_breakeven: bool = True,
                      signal: dict = None) -> Tuple[bool, float, str]:

        if m1_data is None or len(m1_data) < 5:
            return False, 0.0, "insufficient_data"

        if exit_mode == 6:
            return self._exit_mode_multi_tp(m1_data, entry_price, direction, entry_score, signal)
        if exit_mode == 5:
            return self._exit_mode_peak_harvest(m1_data, entry_price, direction, entry_score)
        if exit_mode == 4:
            return self._exit_mode_breakout(m1_data, entry_price, direction, entry_score)
        if exit_mode == 1:
            return self._exit_mode_trail_sl(m1_data, entry_price, direction, exit_threshold,
                                            trail_sl_mult, trail_trigger_mult, trail_retrace,
                                            trail_to_breakeven)
        elif exit_mode == 2:
            return self._exit_mode_fixed_tp_sl(m1_data, entry_price, direction, exit_threshold)
        elif exit_mode == 3:
            return self._exit_mode_atr_anticandle(m1_data, entry_price, direction, exit_threshold)
        return self._exit_mode_trail_sl(m1_data, entry_price, direction, exit_threshold)

    def _exit_mode_breakout(self, df, entry, direction, entry_score):
        """Breakout exit: SL using high/low, ATR trailing, counter-candle."""
        momentum = self.compute_momentum(df)
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

    def _exit_mode_trail_sl(self, df, entry, direction, exit_threshold,
                            sl_mult: float = None, trigger_mult: float = None,
                            retrace_pct: float = None,
                            trail_to_breakeven: bool = True):
        momentum = self.compute_momentum(df)
        px = df["close"].iloc[-1]
        diff = px - entry if direction == "BUY" else entry - px

        atr = self._compute_atr(df, 14)
        sl_m = sl_mult if sl_mult is not None else 0.5
        tr_m = trigger_mult if trigger_mult is not None else 0.7
        retrace = retrace_pct if retrace_pct is not None else 0.30
        trail_trigger = max(atr * tr_m, 0.20)
        trail_retrace_pct = retrace

        triggered = False
        if len(df) >= 5:
            if direction == "BUY":
                peaks = df["high"].values[-5:]
                peak = float(np.max(peaks - entry))
            else:
                troughs = df["low"].values[-5:]
                peak = float(np.max(entry - troughs))
            triggered = peak >= trail_trigger

        stop_loss = -(max(atr * sl_m, 0.15))
        if trail_to_breakeven and triggered:
            stop_loss = max(stop_loss, 0.0)

        if diff <= stop_loss:
            reason = "breakeven" if (trail_to_breakeven and triggered and stop_loss >= 0) else "stop_loss"
            return True, 1.0, reason

        if triggered:
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

        atr = self._compute_atr(df, 14)
        sl = -(max(atr * 0.5, 0.15))
        tp = max(atr * 2.0, 0.60)

        if diff <= sl:
            return True, 1.0, "stop_loss"
        if diff >= tp:
            return True, 0.0, "take_profit"
        return False, 0.0, "holding"

    def _exit_mode_atr_anticandle(self, df, entry, direction, exit_threshold):
        momentum = self.compute_momentum(df)
        px = df["close"].iloc[-1]
        diff = px - entry if direction == "BUY" else entry - px

        atr = self._compute_atr(df, 14)

        if diff <= -(max(atr * 0.5, 0.15)):
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

    def _exit_mode_peak_harvest(self, df, entry, direction, entry_score):
        """
        Peak-harvest exit — no hard stop loss, trail only after significant profit,
        close when directional confidence breaks down.
        ML-guided: if ML still predicts same direction, suppress momentum_decay/direction_loss.
        """
        momentum = self.compute_momentum(df)
        atr = self._compute_atr(df, 14)
        px = df["close"].iloc[-1]
        diff = px - entry if direction == "BUY" else entry - px
        bars = len(df)

        if atr <= 0:
            return False, 0.0, "holding"

        if direction == "BUY":
            peak = float(np.max(df["high"].values - entry))
        else:
            peak = float(np.max(entry - df["low"].values))

        # --- ML exit signal ---
        ml_features = self._get_features(df)
        ml_signal = self._get_ml_exit_signal(ml_features, direction)
        if ml_signal == "exit":
            return True, 0.90, "ml_reversal"
        if ml_signal == "hold":
            # ML still confident in direction — suppress momentum/direction exits
            # Only allow trail_stop and max_hold as hard limits
            confidence = entry_score if entry_score is not None else 0.7
            patience = 1.0 + (1.0 - confidence)
            trail_trigger = atr * cfg.PEAK_HARVEST_TRAIL_TRIGGER * patience
            trail_active = peak >= trail_trigger
            if trail_active and diff > 0:
                pullback = peak - diff
                if pullback / peak > cfg.PEAK_HARVEST_TRAIL_RETRACE:
                    return True, 0.85, "trail_stop"
            if bars > cfg.PEAK_HARVEST_MAX_HOLD_BARS:
                return True, 0.50, "max_hold"
            return False, 0.0, "ml_hold"

        # --- Normal (non-ML) exit logic ---
        confidence = entry_score if entry_score is not None else 0.7
        patience = 1.0 + (1.0 - confidence)

        trail_trigger = atr * cfg.PEAK_HARVEST_TRAIL_TRIGGER * patience
        trail_active = peak >= trail_trigger

        if trail_active and diff > 0:
            pullback = peak - diff
            if pullback / peak > cfg.PEAK_HARVEST_TRAIL_RETRACE:
                return True, 0.85, "trail_stop"

        if bars >= 4:
            streak = 0
            lookback = min(cfg.DIRECTION_LOSS_LOOKBACK, len(df))
            for i in range(lookback):
                c = df.iloc[-(i + 1)]
                if direction == "BUY":
                    if c["close"] < c["open"]:
                        streak += 1
                    else:
                        break
                else:
                    if c["close"] > c["open"]:
                        streak += 1
                    else:
                        break
            if streak >= cfg.DIRECTION_LOSS_STREAK:
                return True, 0.80, "direction_loss"

        if bars >= cfg.PEAK_HARVEST_MIN_BARS_EXIT:
            exit_score = 1.0 - momentum
            if exit_score > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
                return True, round(exit_score, 3), "momentum_decay"

        if bars > cfg.PEAK_HARVEST_MAX_HOLD_BARS:
            return True, 0.50, "max_hold"

        return False, 0.0, "holding"

    def _exit_mode_multi_tp(self, df, entry, direction, entry_score, signal):
        """
        Zone-aware multi-TP exit (mode 6) — v2 with dynamic SL, trail, and wick rejection.

        1. Static SL from signal (always active)
        2. Breakeven at 60% TP1 progress (lock out the loss)
        3. Lock 30% of TP1 profit when TP1 exceeded
        4. Trail 2 ATR after TP2 exceeded
        5. Wick rejection: if a candle near TP has >60% wick, exit early
        6. Original momentum-fade exit when near a TP level
        """
        momentum = self.compute_momentum(df)
        px = df["close"].iloc[-1]
        diff = px - entry if direction == "BUY" else entry - px

        if not signal:
            atr = self._compute_atr(df, 14)
            if atr <= 0:
                atr = abs(entry) * 0.001
            sl_price = entry - (2 * atr) if direction == "BUY" else entry + (2 * atr)
            return False, 0.0, "recovery_trail"

        sl_price = signal.get("sl")
        tp1 = signal.get("tp1")
        tp2 = signal.get("tp2")
        tp3 = signal.get("tp3")

        if None in (sl_price, tp1, tp2, tp3):
            atr = self._compute_atr(df, 14)
            if atr <= 0:
                atr = abs(entry) * 0.001
            sl_price = entry - (2 * atr) if direction == "BUY" else entry + (2 * atr)
            return False, 0.0, "recovery_trail"

        atr = self._compute_atr(df, 14)
        if atr <= 0:
            atr = abs(entry) * 0.001

        if direction == "BUY":
            # --- hard SL ---
            if px <= sl_price:
                return True, 1.0, "stop_loss"

            tp1_progress = (px - entry) / (tp1 - entry) if tp1 != entry else 0

            # --- dynamic SL management ---
            if px >= tp2:
                sl_price = max(sl_price, px - atr * 2)
            elif px >= tp1:
                sl_price = max(sl_price, entry + (tp1 - entry) * 0.3)
            elif tp1_progress >= 0.6:
                sl_price = max(sl_price, entry + atr * 0.3)

            if px <= sl_price:
                return True, 1.0, "stop_loss"

            tp_levels = [(tp1, "tp1"), (tp2, "tp2"), (tp3, "tp3")]
            active_tps = [(tp, name) for tp, name in tp_levels if px < tp]

            if not active_tps:
                # above all TPs — trail
                trail = px - atr * 2
                sl_price = max(sl_price, trail)
                if px <= sl_price:
                    return True, 1.0, "trailing_stop"
                if momentum < cfg.TP_CLOSE_MOMENTUM_MIN:
                    return True, round(1.0 - momentum, 3), "momentum_exhausted"
                return False, 0.0, "running"

            nearest_tp, tp_name = active_tps[0]
            progress = diff / (nearest_tp - entry) if nearest_tp != entry else 1.0

            # wick rejection near TP
            if progress >= 0.8:
                candle = df.iloc[-1]
                upper_wick = candle["high"] - max(candle["close"], candle["open"])
                lower_wick = min(candle["close"], candle["open"]) - candle["low"]
                total_range = candle["high"] - candle["low"]
                if total_range > 0 and upper_wick / total_range > 0.6:
                    return True, round(progress, 3), f"wick_rejection_{tp_name}"

        else:  # SELL
            if px >= sl_price:
                return True, 1.0, "stop_loss"

            tp1_progress = (entry - px) / (entry - tp1) if entry != tp1 else 0

            if px <= tp2:
                sl_price = min(sl_price, px + atr * 2)
            elif px <= tp1:
                sl_price = min(sl_price, entry - (entry - tp1) * 0.3)
            elif tp1_progress >= 0.6:
                sl_price = min(sl_price, entry - atr * 0.3)

            if px >= sl_price:
                return True, 1.0, "stop_loss"

            tp_levels = [(tp1, "tp1"), (tp2, "tp2"), (tp3, "tp3")]
            active_tps = [(tp, name) for tp, name in tp_levels if px > tp]

            if not active_tps:
                trail = px + atr * 2
                sl_price = min(sl_price, trail)
                if px >= sl_price:
                    return True, 1.0, "trailing_stop"
                if momentum < cfg.TP_CLOSE_MOMENTUM_MIN:
                    return True, round(1.0 - momentum, 3), "momentum_exhausted"
                return False, 0.0, "running"

            nearest_tp, tp_name = active_tps[0]
            progress = diff / (entry - nearest_tp) if entry != nearest_tp else 1.0

            if progress >= 0.8:
                candle = df.iloc[-1]
                upper_wick = candle["high"] - max(candle["close"], candle["open"])
                lower_wick = min(candle["close"], candle["open"]) - candle["low"]
                total_range = candle["high"] - candle["low"]
                if total_range > 0 and lower_wick / total_range > 0.6:
                    return True, round(progress, 3), f"wick_rejection_{tp_name}"

        progress = min(progress, 1.0)

        if progress >= cfg.TP_CLOSE_THRESHOLD:
            if momentum < cfg.TP_CLOSE_MOMENTUM_MIN:
                return True, round(progress, 3), f"take_profit_{tp_name}"
            return False, 0.0, f"holding_near_{tp_name}"

        return False, 0.0, "holding"

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

    def compute_momentum(self, df: pd.DataFrame) -> float:
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

        atr = self._compute_atr(df, 14)
        ref = max(atr, df["close"].iloc[-1] * 0.0001)
        raw = (recent_change / (older_change + 1e-10)) * \
               (avg_body / (ref + 1e-10))
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
        atr = float(np.mean(tr[-period:]))
        return 0.0 if np.isnan(atr) else atr

    def _candle_strength(self, df: pd.DataFrame) -> float:
        if len(df) < 3:
            return 0.0
        candle = df.iloc[-1]
        body = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]
        if total_range == 0:
            return 0.0
        return min(body / total_range, 1.0)
