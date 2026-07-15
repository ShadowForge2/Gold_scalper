import time as _time
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone

import config as cfg

try:
    from app.asp_predictor import ASPPredictor
    from app.asp_features import compute_asp_features
    _HAS_ASP = True
except ImportError:
    ASPPredictor = None
    compute_asp_features = None
    _HAS_ASP = False

try:
    from app.swing_quality_predictor import SwingQualityPredictor
    _HAS_SQ = True
except ImportError:
    SwingQualityPredictor = None
    _HAS_SQ = False

try:
    from app.direction_predictor import compute_chop_score
    _HAS_CHOP = True
except ImportError:
    compute_chop_score = None
    _HAS_CHOP = False


class SignalEngine:
    def __init__(self,
                 asp_predictor: Optional["ASPPredictor"] = None,
                 swing_quality_predictor: Optional["SwingQualityPredictor"] = None,
                 logger=None):
        self.current_signal: Optional[Dict] = None
        self.last_signal_time: Optional[datetime] = None
        self.last_rejection: Optional[Dict] = None
        self._asp_predictor = asp_predictor
        self._swing_quality = swing_quality_predictor
        self._logger = logger
        self._last_asp_log_time = 0.0

    def _reject(self, reason: str, **context) -> None:
        self.last_rejection = {
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            **context,
        }
        return None

    def evaluate_asp_entry(self, m1_data: pd.DataFrame, current_price: float,
                           events: Optional[list] = None,
                           h1_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """ASP-based entry: ML predicts swing turning points.

        SL = 2x ATR, TP = 1x ATR, timeout handled by bot.py.
        """
        if m1_data is None or len(m1_data) < 10:
            return self._reject("insufficient_m1_data", price=current_price)

        if not _HAS_ASP or self._asp_predictor is None or not self._asp_predictor.ready:
            return self._reject("no_asp_model", price=current_price)

        try:
            if "time" in m1_data.columns:
                m1_idx = m1_data.set_index("time")
            else:
                m1_idx = m1_data

            if len(m1_idx) < 10:
                return self._reject("insufficient_m1_data", price=current_price)

            m5 = m1_idx.resample("5min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "tick_volume": "sum",
            }).dropna()
            if len(m5) < 20:
                return self._reject("insufficient_m5_data", price=current_price)

            if h1_data is not None and len(h1_data) >= 20:
                h1_src = h1_data.set_index("time") if "time" in h1_data.columns else h1_data
                h1 = h1_src[["open", "high", "low", "close"]].dropna()
            else:
                h1 = m1_idx.resample("1h").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last",
                }).dropna()

            asp_feats = compute_asp_features(m5, h1)
            if asp_feats is None or len(asp_feats) == 0:
                return self._reject("no_asp_features", price=current_price)

            # Chop filter — reject when market is stagnant (validated: 1.59x movement ratio at 0.70)
            if (getattr(cfg, "CHOP_FILTER_ENABLED", True) and _HAS_CHOP and compute_chop_score is not None):
                chop = compute_chop_score(m5)
                if len(chop) > 0 and chop.iloc[-1] > getattr(cfg, "CHOP_THRESHOLD", 0.70):
                    if self._logger:
                        self._logger.info(
                            f"[ASP_ML] chop_filter_reject chop={chop.iloc[-1]:.4f} > "
                            f"{getattr(cfg, 'CHOP_THRESHOLD', 0.70):.2f}"
                        )
                    return self._reject("choppy_market", price=current_price,
                                        chop_score=round(float(chop.iloc[-1]), 4),
                                        threshold=getattr(cfg, "CHOP_THRESHOLD", 0.70))

            # Swing quality gate — reject if model says not at reversal
            if (getattr(cfg, "SWING_QUALITY_ENABLED", True) and
                    self._swing_quality is not None and self._swing_quality.ready):
                sq_prob, _ = self._swing_quality.predict_quality(asp_feats)
                if sq_prob is not None:
                    threshold = getattr(cfg, "SWING_QUALITY_THRESHOLD", 0.5)
                    if sq_prob < threshold:
                        if self._logger:
                            self._logger.info(
                                f"[ASP_ML] swing_quality_reject prob={sq_prob:.3f} < {threshold}"
                            )
                        return self._reject("swing_quality_low", price=current_price,
                                             swing_quality=round(sq_prob, 4))

            last_row = asp_feats.iloc[-1]
            direction, confidence = self._asp_predictor.predict(last_row)

            if self._logger:
                now = _time.monotonic()
                if now - self._last_asp_log_time >= 30:
                    self._last_asp_log_time = now
                    prob_dict = self._asp_predictor._get_raw_probs(last_row)
                    prob_str = ""
                    if prob_dict:
                        prob_str = f" B={prob_dict['buy']:.3f} S={prob_dict['sell']:.3f} N={prob_dict['neutral']:.3f}"
                    self._logger.info(
                        f"[ASP_ML] dir={direction} conf={confidence:.3f} "
                        f"price={current_price:.2f}{prob_str}"
                    )

            if direction is None:
                return self._reject("asp_no_signal", price=current_price,
                                    confidence=round(confidence, 4))

            min_conf = getattr(cfg, "ASP_MIN_CONFIDENCE", 0.55)
            if confidence < min_conf:
                if self._logger:
                    self._logger.info(
                        f"[ASP_ML] rejected {direction} conf={confidence:.3f} < min={min_conf:.3f}"
                    )
                return self._reject("asp_low_confidence", price=current_price,
                                    direction=direction,
                                    confidence=round(confidence, 4),
                                    min_confidence=min_conf)

            atr_val = self._compute_atr_m5(m1_data, cfg.ATR_PERIOD)
            if atr_val <= 0:
                atr_val = current_price * 0.001

            sl_distance = atr_val * cfg.ASP_SL_ATR_MULTIPLIER
            tp_distance = atr_val * cfg.ASP_TP_ATR_MULTIPLIER

            if direction == "BUY":
                sl_price = current_price - sl_distance
                tp_price = current_price + tp_distance
            else:
                sl_price = current_price + sl_distance
                tp_price = current_price - tp_distance

            if sl_distance < cfg.ASP_MIN_ATR_DIST:
                return self._reject("asp_atr_too_small", price=current_price,
                                    atr=round(atr_val, 4), dist=round(sl_distance, 4))

            signal = {
                "direction": direction,
                "score": round(confidence, 3),
                "conviction": round(confidence, 3),
                "breakout_dist": 0.0,
                "range_size": 0.0,
                "entry_price": current_price,
                "sl": round(sl_price, 2),
                "tp1": round(tp_price, 2),
                "tp2": None,
                "tp3": None,
                "atr_value": round(atr_val, 4),
                "ml_confidence": round(confidence, 4),
                "num_positions": 1,
                "lot_mult": 1.0,
                "asp_model": True,
                "timestamp": datetime.now().isoformat(),
            }

            self.current_signal = signal
            self.last_rejection = None
            return signal

        except Exception as e:
            if self._logger:
                self._logger.warning(f"[ASP_ML] entry error: {e}")
            return self._reject("asp_error", price=current_price, error=str(e))

    def _resample_to_m5(self, m1_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            if "time" in m1_data.columns:
                m1_idx = m1_data.set_index("time")
            else:
                m1_idx = m1_data
            m5 = m1_idx.resample("5min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "tick_volume": "sum",
            }).dropna()
            if len(m5) < 5:
                return None
            return m5
        except Exception:
            return None

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

    def _compute_atr_m5(self, m1_data: pd.DataFrame, period: int = 14) -> float:
        m5 = self._resample_to_m5(m1_data)
        if m5 is None or len(m5) < period + 1:
            return 0.0
        return self._compute_atr(m5, period)
