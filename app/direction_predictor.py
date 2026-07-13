import os, logging
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Tuple
import joblib

logger = logging.getLogger(__name__)


SWEEP_LOOKBACK = 12  # M5 bars (~1 hour)

FEATURE_COLS = [
    "return_1", "return_2", "return_3", "return_5", "return_10",
    "return_20",
    "rsi_14",
    "atr_norm",
    "bb_position",
    "bb_width",
    "range_ratio",
    "inside_bar_count",
    "micro_slope",
    "hl_ratio",
    "volume_ratio",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "sma_ratio",
    "macd", "macd_signal", "macd_hist",
    "h1_pos", "h1_dir", "above_h1h", "below_h1l",
    "volatility_ratio",
    "close_position",
    "return_vol_ratio",
    "sweep_low_atr", "sweep_high_atr", "close_vs_range_12",
    "minutes_until_event", "minutes_since_event", "event_impact",
]


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
    return df


def compute_features(df: pd.DataFrame, h1_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute feature matrix from M5 OHLCV data. Returns DataFrame with same index as df."""
    df = _ensure_dtindex(df.copy())
    if len(df) < 20:
        logger.warning("compute_features: insufficient data (%d rows)", len(df))
        return pd.DataFrame(index=df.index)
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    vols = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df.columns else np.ones(len(df))

    features = pd.DataFrame(index=df.index)

    features["return_1"] = np.log(closes / np.concatenate([[np.nan], closes[:-1]]))
    features["return_2"] = np.log(closes / np.concatenate([[np.nan, np.nan], closes[:-2]]))
    features["return_3"] = np.log(closes / np.concatenate([[np.nan, np.nan, np.nan], closes[:-3]]))
    features["return_5"] = np.log(closes / np.concatenate([[np.nan] * 5, closes[:-5]]))
    features["return_10"] = np.log(closes / np.concatenate([[np.nan] * 10, closes[:-10]]))
    features["return_20"] = np.log(closes / np.concatenate([[np.nan] * 20, closes[:-20]]))

    features["hl_ratio"] = (highs - lows) / closes

    prev_close = np.concatenate([[np.nan], closes[:-1]])
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - prev_close),
            np.abs(lows - prev_close),
        ),
    )
    atr_series = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean()
    features["atr_norm"] = atr_series / closes

    roll_vol = pd.Series(vols, index=df.index).rolling(20, min_periods=2).mean()
    roll_vol = roll_vol.replace(0, np.nan)
    features["volume_ratio"] = pd.Series(vols, index=df.index) / roll_vol

    delta = pd.Series(np.diff(closes, prepend=np.nan), index=df.index)
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=2).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    sma20 = pd.Series(closes, index=df.index).rolling(20, min_periods=2).mean()
    sma50 = pd.Series(closes, index=df.index).rolling(50, min_periods=2).mean()
    features["sma_ratio"] = sma20 / sma50.replace(0, np.nan)

    bb_mid = sma20
    bb_std = pd.Series(closes, index=df.index).rolling(20, min_periods=2).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    features["bb_position"] = (closes - bb_lower) / bb_range
    features["bb_width"] = bb_range / closes

    candle_range = highs - lows
    atr_safe_feat = atr_series.replace(0, np.nan)
    features["range_ratio"] = candle_range / atr_safe_feat.values

    inside = np.zeros(len(df), dtype=float)
    inside[1:] = ((highs[1:] <= highs[:-1]) & (lows[1:] >= lows[:-1])).astype(float)
    features["inside_bar_count"] = pd.Series(inside, index=df.index).rolling(5, min_periods=1).sum()

    n = len(closes)
    weights = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    weighted_sum = np.convolve(closes, weights, mode="valid")
    simple_sum = np.convolve(closes, np.ones(5, dtype=np.float64), mode="valid")
    slope_vals = np.full(n, np.nan, dtype=np.float64)
    slope_vals[4:] = (5.0 * weighted_sum - 10.0 * simple_sum) / 50.0
    features["micro_slope"] = pd.Series(slope_vals, index=df.index) / atr_safe_feat.values

    ema12 = pd.Series(closes, index=df.index).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(closes, index=df.index).ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    if h1_df is not None and len(h1_df) > 0:
        h1_indexed = _ensure_dtindex(h1_df)
        # Use reindex with method='ffill' instead of merge_asof to avoid dtype issues
        h1_aligned = h1_indexed[["high", "low", "close", "open"]].reindex(
            features.index, method="ffill"
        )
        h1h = h1_aligned["high"].values.astype(np.float64)
        h1l = h1_aligned["low"].values.astype(np.float64)
        h1c = h1_aligned["close"].values.astype(np.float64)
        h1o = h1_aligned["open"].values.astype(np.float64)

        h1_range = h1h - h1l
        h1_range_safe = np.where(h1_range > 0, h1_range, np.nan)
        features["h1_pos"] = np.where(
            h1_range_safe > 0, (closes - h1l) / h1_range_safe, np.nan
        )
        features["h1_dir"] = np.sign(h1c - h1o)
        features["above_h1h"] = (closes > h1h).astype(np.float64)
        features["below_h1l"] = (closes < h1l).astype(np.float64)
    else:
        features["h1_pos"] = np.nan
        features["h1_dir"] = 0.0
        features["above_h1h"] = 0.0
        features["below_h1l"] = 0.0

    sess_vol = pd.Series(tr, index=df.index).rolling(60, min_periods=2).mean()
    short_vol = pd.Series(tr, index=df.index).rolling(5, min_periods=2).mean()
    features["volatility_ratio"] = short_vol / sess_vol.replace(0, np.nan)

    features["close_position"] = np.divide(
        closes - lows, candle_range, where=candle_range > 0,
        out=np.full_like(closes, 0.5, dtype=float)
    )

    features["return_vol_ratio"] = (
        features["return_1"].rolling(5, min_periods=2).mean()
        / short_vol.replace(0, np.nan)
    )

    # SMC/manipulation features
    roll_min = pd.Series(lows, index=df.index).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min()
    roll_max = pd.Series(highs, index=df.index).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max()
    roll_range = roll_max - roll_min
    roll_range_safe = roll_range.replace(0, np.nan)
    atr_safe = atr_series.replace(0, np.nan)

    sweep_low = np.maximum(0, roll_min.values - closes)
    features["sweep_low_atr"] = sweep_low / atr_safe.values

    sweep_high = np.maximum(0, closes - roll_max.values)
    features["sweep_high_atr"] = sweep_high / atr_safe.values

    features["close_vs_range_12"] = (closes - roll_min.values) / roll_range_safe.values

    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    hours = times.hour
    days = times.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["day_sin"] = np.sin(2 * np.pi * days / 7)
    features["day_cos"] = np.cos(2 * np.pi * days / 7)

    # Default calendar features (neutral: no event near)
    features["minutes_until_event"] = 999.0
    features["minutes_since_event"] = -999.0
    features["event_impact"] = 0.0

    nan_pct = features.isna().mean()
    high_nan = nan_pct[nan_pct > 0.1]
    if len(high_nan):
        logger.debug("compute_features: cols with >10%% NaN: %s", dict(high_nan))
    nan_cols = features.columns[features.isna().all()]
    if len(nan_cols):
        logger.warning("compute_features: dropping %d all-NaN cols: %s", len(nan_cols), list(nan_cols))
    features = features.drop(columns=nan_cols)
    missing_from_schema = [c for c in FEATURE_COLS if c not in features.columns]
    if missing_from_schema:
        logger.debug("compute_features: %d cols missing from FEATURE_COLS: %s", len(missing_from_schema), missing_from_schema)
    features = features[~features.index.duplicated(keep="first")]
    return features


def compute_chop_score(df: pd.DataFrame) -> pd.Series:
    """Compute a composite chop score (0-1, higher = choppier) from OHLC data.

    Uses the same 4 chop features as the ML model: bb_width, range_ratio,
    inside_bar_count, micro_slope. Each is rank-normalized and averaged.
    """
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    prev_close = np.concatenate([[np.nan], closes[:-1]])
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    atr_series = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean()
    atr_safe = atr_series.replace(0, np.nan)

    # 1. bb_width: narrow → high chop
    sma20 = pd.Series(closes, index=df.index).rolling(20, min_periods=2).mean()
    bb_std = pd.Series(closes, index=df.index).rolling(20, min_periods=2).std()
    bb_width = (4 * bb_std / sma20.replace(0, np.nan))
    # Invert so narrow → high score
    bb_pct = bb_width.rank(pct=True)
    bb_chop = (1 - bb_pct).fillna(0.5)

    # 2. range_ratio: small candles relative to ATR → high chop
    candle_range = highs - lows
    rr = pd.Series(candle_range / atr_safe.values, index=df.index) if atr_safe is not None else pd.Series(candle_range, index=df.index)
    rr_chop = (1 - rr.rank(pct=True)).fillna(0.5)

    # 3. inside_bar_count: many inside bars → high chop
    inside = np.zeros(len(df), dtype=float)
    inside[1:] = ((highs[1:] <= highs[:-1]) & (lows[1:] >= lows[:-1])).astype(float)
    ibc = pd.Series(inside, index=df.index).rolling(5, min_periods=1).sum()
    ibc_chop = ibc.rank(pct=True).fillna(0.5)

    # 4. micro_slope: flat → high chop
    n = len(closes)
    weights = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    weighted_sum = np.convolve(closes, weights, mode="valid")
    simple_sum = np.convolve(closes, np.ones(5, dtype=np.float64), mode="valid")
    slope_vals = np.full(n, np.nan, dtype=np.float64)
    slope_vals[4:] = (5.0 * weighted_sum - 10.0 * simple_sum) / 50.0
    ms = pd.Series(np.abs(slope_vals) / atr_safe.values if atr_safe is not None else np.abs(slope_vals), index=df.index)
    ms_chop = (1 - ms.rank(pct=True)).fillna(0.5)

    composite = (bb_chop + rr_chop + ibc_chop + ms_chop) / 4
    return composite.clip(0, 1)


def create_target(df: pd.DataFrame, horizon: int = 3, atr_threshold: float = 0.3,
                  chop_score: Optional[pd.Series] = None, chop_pct: float = 0.33) -> pd.Series:
    """Create 3-class target: 1=UP, 0=DOWN, 2=NO_TRADE.

    When chop_score is provided, the most choppy `chop_pct` fraction of bars
    are labeled NO_TRADE based on market structure (narrow BB, small candles,
    inside bars, flat micro-trend). Remaining bars use forward return + ATR.

    When chop_score is None, falls back to |forward move| <= atr_threshold*ATR.
    """
    closes = df["close"].values.astype(np.float64)

    if chop_score is not None:
        chop_arr = chop_score.values.astype(np.float64)
        threshold = np.nanpercentile(chop_arr, (1 - chop_pct) * 100)
        target = np.full(len(closes), np.nan, dtype=np.float64)
        is_chop = chop_arr >= threshold
        target[is_chop] = 2.0  # NO_TRADE for choppy bars
        # UP/DOWN for the rest based on forward return
        future_close = np.roll(closes, -horizon)
        non_chop = ~is_chop
        if non_chop.any():
            tr = np.maximum(
                df["high"].values.astype(np.float64) - df["low"].values.astype(np.float64),
                np.maximum(
                    np.abs(df["high"].values.astype(np.float64) - np.roll(closes, 1)),
                    np.abs(df["low"].values.astype(np.float64) - np.roll(closes, 1)),
                ),
            )
            atr = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean().values
            thresh = atr_threshold * atr
            up = non_chop & (future_close > closes + thresh)
            down = non_chop & (future_close < closes - thresh)
            target[up] = 1.0
            target[down] = 0.0
    else:
        tr = np.maximum(
            df["high"].values.astype(np.float64) - df["low"].values.astype(np.float64),
            np.maximum(
                np.abs(df["high"].values.astype(np.float64) - np.roll(closes, 1)),
                np.abs(df["low"].values.astype(np.float64) - np.roll(closes, 1)),
            ),
        )
        atr = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean().values
        future_close = np.roll(closes, -horizon)
        threshold = atr_threshold * atr
        target = np.full(len(closes), 2.0)
        up = future_close > closes + threshold
        down = future_close < closes - threshold
        target[up] = 1.0
        target[down] = 0.0
        target[np.isnan(atr)] = np.nan

    target[np.isnan(target)] = np.nan
    target[-horizon:] = np.nan

    result = pd.Series(target, index=df.index, name="target")
    result = result[~result.index.duplicated(keep="first")]
    return result


def prepare_dataset(features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop NaN rows and align features with target."""
    data = features.copy()
    data["target"] = target
    data = data.dropna()
    return data[FEATURE_COLS], data["target"]


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series,
                **params) -> xgb.XGBClassifier:
    """Train XGBoost classifier (binary or multi-class) with early stopping."""
    n_classes = len(np.unique(y_train.dropna()))
    is_multi = n_classes >= 3
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="multi:softprob" if is_multi else "binary:logistic",
        num_class=n_classes if is_multi else None,
        eval_metric="mlogloss" if is_multi else "logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        **params,
    )
    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        verbose=False,
    )
    return model


class DirectionPredictor:
    def __init__(self, model_path: str = None, model: xgb.XGBClassifier = None):
        if model is not None:
            self.model = model
        elif model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None
        self._feature_cols = FEATURE_COLS

    def predict_proba(self, features: pd.DataFrame) -> Tuple[float, float, float]:
        """Return (prob_down, prob_up, prob_no_trade) for the most recent bar.
        For binary model: prob_no_trade = 0.0."""
        if self.model is None:
            logger.warning("predict_proba: model is None")
            return 0.5, 0.5, 0.0
        if features is None or len(features) == 0:
            logger.warning("predict_proba: empty features")
            return 0.5, 0.5, 0.0
        missing = [c for c in self._feature_cols if c not in features.columns]
        if missing:
            logger.warning("predict_proba: %d missing features filled with 0: %s", len(missing), missing)
            features = features.copy()
            for c in missing:
                features[c] = 0.0
        try:
            row = features[self._feature_cols].iloc[[-1]]
            nan_count = int(row.isna().sum().sum())
            if nan_count:
                logger.warning("predict_proba: %d NaN values in feature row", nan_count)
            probs = self.model.predict_proba(row.values)
            classes = list(self.model.classes_)
            if 0 in classes and 1 in classes and 2 in classes:
                result = (float(probs[0][classes.index(0)]), float(probs[0][classes.index(1)]), float(probs[0][classes.index(2)]))
                if result[2] > 0.40:
                    logger.info("predict_proba: NO_TRADE=%.4f (DOWN=%.4f UP=%.4f)", result[2], result[0], result[1])
                return result
            prob_down = float(probs[0][classes.index(0)]) if 0 in classes else float(probs[0][0])
            prob_up = float(probs[0][classes.index(1)]) if 1 in classes else float(probs[0][1])
            return prob_down, prob_up, 0.0
        except Exception as e:
            logger.error("predict_proba exception: %s", e, exc_info=True)
            return 0.5, 0.5, 0.0

    def predict(self, features: pd.DataFrame, confidence_threshold: float = 0.55) -> Optional[str]:
        """Return 'BUY', 'SELL', or 'NO_TRADE' based on confidence."""
        prob_down, prob_up, prob_no = self.predict_proba(features)
        if prob_up >= confidence_threshold and prob_up >= prob_down and prob_up >= prob_no:
            logger.debug("predict: BUY (UP=%.4f DOWN=%.4f NT=%.4f thr=%.2f)", prob_up, prob_down, prob_no, confidence_threshold)
            return "BUY"
        if prob_down >= confidence_threshold and prob_down >= prob_up and prob_down >= prob_no:
            logger.debug("predict: SELL (DOWN=%.4f UP=%.4f NT=%.4f thr=%.2f)", prob_down, prob_up, prob_no, confidence_threshold)
            return "SELL"
        if prob_no >= confidence_threshold:
            logger.info("predict: NO_TRADE (NT=%.4f thr=%.2f)", prob_no, confidence_threshold)
            return None
        return None

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "DirectionPredictor":
        return cls(model_path=path)


class SLTPredictor:
    """Predicts whether a trade will hit 2xATR TP before 1xATR SL.
    Uses two separate XGBoost models (BUY and SELL)."""
    def __init__(self, buy_model_path: str = None, sell_model_path: str = None):
        self.buy_model = joblib.load(buy_model_path) if buy_model_path and os.path.exists(buy_model_path) else None
        self.sell_model = joblib.load(sell_model_path) if sell_model_path and os.path.exists(sell_model_path) else None
        self._feature_cols = FEATURE_COLS

    def _prepare(self, features: pd.DataFrame) -> np.ndarray:
        if features is None or len(features) == 0:
            return np.zeros((1, len(self._feature_cols)))
        missing = [c for c in self._feature_cols if c not in features.columns]
        if missing:
            features = features.copy()
            for c in missing:
                features[c] = 0.0
        row = features[self._feature_cols].iloc[[-1]]
        return row.values

    def buy_win_prob(self, features: pd.DataFrame) -> float:
        if self.buy_model is None:
            return 0.5
        row = self._prepare(features)
        probs = self.buy_model.predict_proba(row)
        idx = list(self.buy_model.classes_).index(1) if 1 in self.buy_model.classes_ else 1
        return float(probs[0][idx])

    def sell_win_prob(self, features: pd.DataFrame) -> float:
        if self.sell_model is None:
            return 0.5
        row = self._prepare(features)
        probs = self.sell_model.predict_proba(row)
        idx = list(self.sell_model.classes_).index(1) if 1 in self.sell_model.classes_ else 1
        return float(probs[0][idx])

    def predict(self, features: pd.DataFrame, confidence_threshold: float = 0.60) -> Optional[str]:
        """Return 'BUY' if BUY likely to win, 'SELL' if SELL likely to win, None otherwise."""
        buy_p = self.buy_win_prob(features)
        sell_p = self.sell_win_prob(features)
        if buy_p >= confidence_threshold and buy_p >= sell_p:
            return "BUY"
        if sell_p >= confidence_threshold and sell_p > buy_p:
            return "SELL"
        return None


TRADE_STATE_COLS = [
    "bars_held", "pnl_atr", "peak_atr", "drawdown_pct",
    "entry_score", "atr_change", "wrong_streak",
    "sweep_atr", "recovery_pct",
]
EXIT_FEATURE_COLS = FEATURE_COLS + TRADE_STATE_COLS


# =========================================================================
# Trend model feature columns (35 base + 10 H1 structure = 45)
# =========================================================================
TREND_FEATURE_COLS = FEATURE_COLS + [
    "h1_rsi", "h1_macd_hist", "h1_ema_spread",
    "h1_consecutive_up", "h1_consecutive_down",
    "h1_swing_distance", "h1_atr_expanding",
    "m5_h1_alignment", "trend_duration", "h1_structure_break",
]

# Exit-trend model feature columns (35 base + 5 trade-state = 40)
TRADE_STATE_TREND_COLS = [
    "bars_held", "trend_pnl_atr", "trend_peak_atr", "trend_drawdown_pct",
    "trend_momentum_decay",
]
EXIT_TREND_FEATURE_COLS = FEATURE_COLS + TRADE_STATE_TREND_COLS


class ExitPredictor:
    """Predicts whether to hold or exit during an active trade.
    Features: market state (26 cols from compute_features) + trade state (7 cols)."""
    def __init__(self, model_path: str = None):
        self.model = joblib.load(model_path) if model_path and os.path.exists(model_path) else None

    def predict_hold_prob(self, market_features: pd.DataFrame, trade_state: dict) -> float:
        """Return probability that holding is better than exiting (0-1).
        market_features: output from compute_features (1 row DataFrame or dict-like)
        trade_state: dict with keys: bars_held, pnl_atr, peak_atr, drawdown_pct, entry_score, atr_change, wrong_streak
        """
        if self.model is None:
            return 0.5
        try:
            row = {}
            for c in FEATURE_COLS:
                if c in market_features:
                    row[c] = float(market_features[c]) if hasattr(market_features[c], '__float__') else 0.0
                else:
                    row[c] = 0.0
            for c in TRADE_STATE_COLS:
                row[c] = float(trade_state.get(c, 0.0))
            vec = np.array([[row[c] for c in EXIT_FEATURE_COLS]], dtype=np.float32)
            probs = self.model.predict_proba(vec)
            idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 1
            return float(probs[0][idx])
        except Exception:
            return 0.5


# =========================================================================
# H1 structure feature computation (for trend model)
# =========================================================================

def _compute_h1_structure_features(m5_df: pd.DataFrame, h1_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 10 H1 structure features aligned to M5 index.

    Features:
      h1_rsi, h1_macd_hist, h1_ema_spread,
      h1_consecutive_up, h1_consecutive_down,
      h1_swing_distance, h1_atr_expanding,
      m5_h1_alignment, trend_duration, h1_structure_break
    """
    if h1_df is None or len(h1_df) < 60:
        return pd.DataFrame(index=m5_df.index)

    h1 = h1_df.copy()
    if not isinstance(h1.index, pd.DatetimeIndex):
        h1.index = pd.to_datetime(h1.index)
    h1 = h1[~h1.index.duplicated(keep="first")]

    c = h1["close"].values.astype(np.float64)
    h = h1["high"].values.astype(np.float64)
    lo = h1["low"].values.astype(np.float64)
    o = h1["open"].values.astype(np.float64)
    n_h1 = len(c)

    feats = pd.DataFrame(index=h1.index)

    # 1. h1_rsi
    delta = np.diff(c, prepend=np.nan)
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(14, min_periods=2).mean()
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(14, min_periods=2).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["h1_rsi"] = (100 - (100 / (1 + rs))).fillna(50.0).values

    # 2. h1_macd_hist
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    feats["h1_macd_hist"] = (macd_line - macd_sig).fillna(0.0).values

    # 3. h1_ema_spread
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean()
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean()
    feats["h1_ema_spread"] = ((ema20 - ema50) / np.where(c > 0, c, 1)).fillna(0.0).values

    # 4/5. consecutive up/down
    bull = c > o
    bear = c < o
    cu = np.zeros(n_h1)
    cd = np.zeros(n_h1)
    for i in range(1, n_h1):
        cu[i] = (cu[i - 1] + 1) if bull[i] else 0
        cd[i] = (cd[i - 1] + 1) if bear[i] else 0
    feats["h1_consecutive_up"] = cu
    feats["h1_consecutive_down"] = cd

    # 6. h1_swing_distance
    tr_h1 = np.maximum(h[1:] - lo[1:],
                       np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))
    atr_h1 = np.concatenate([[tr_h1[0] if len(tr_h1) > 0 else 1.0], tr_h1])
    atr_h1 = pd.Series(atr_h1).rolling(14, min_periods=14).mean().bfill().fillna(1.0).values
    atr_h1 = np.where(atr_h1 <= 0, 1.0, atr_h1)

    lookback = 3
    swing_h = np.zeros(n_h1, dtype=bool)
    swing_l = np.zeros(n_h1, dtype=bool)
    for i in range(lookback, n_h1 - lookback):
        if h[i] == np.max(h[i - lookback:i + lookback + 1]):
            swing_h[i] = True
        if lo[i] == np.min(lo[i - lookback:i + lookback + 1]):
            swing_l[i] = True

    dist = np.zeros(n_h1)
    for i in range(n_h1):
        sh_pos = np.where(swing_h[:i + 1])[0]
        sl_pos = np.where(swing_l[:i + 1])[0]
        d_up = abs(c[i] - h[sh_pos[-1]]) if len(sh_pos) > 0 else atr_h1[i]
        d_dn = abs(c[i] - lo[sl_pos[-1]]) if len(sl_pos) > 0 else atr_h1[i]
        dist[i] = min(d_up, d_dn) / max(atr_h1[i], 0.01)
    feats["h1_swing_distance"] = dist

    # 7. h1_atr_expanding
    atr_ma = pd.Series(atr_h1).rolling(20, min_periods=2).mean().values
    feats["h1_atr_expanding"] = np.where(atr_ma > 0, atr_h1 / atr_ma, 1.0)

    # 8. trend_duration
    td = np.zeros(n_h1)
    for i in range(1, n_h1):
        if bull[i]:
            td[i] = td[i - 1] + 1 if bull[i - 1] else 1
        elif bear[i]:
            td[i] = td[i - 1] + 1 if bear[i - 1] else 1
        else:
            td[i] = 0
    feats["trend_duration"] = td

    # 9. h1_structure_break
    sb = np.zeros(n_h1)
    for i in range(3, n_h1):
        sh_pos = np.where(swing_h[:i])[0]
        sl_pos = np.where(swing_l[:i])[0]
        if len(sh_pos) > 0 and c[i] > h[sh_pos[-1]]:
            sb[i] = 1.0
        elif len(sl_pos) > 0 and c[i] < lo[sl_pos[-1]]:
            sb[i] = -1.1
    feats["h1_structure_break"] = sb

    # 10. m5_h1_alignment — H1 direction as proxy (filled later)
    feats["m5_h1_alignment"] = 0.0

    # Align to M5 index
    m5_idx = m5_df.index if isinstance(m5_df.index, pd.DatetimeIndex) else pd.to_datetime(m5_df.index)
    aligned = feats.reindex(m5_idx, method="ffill")
    return aligned


def compute_trend_features(m5_df: pd.DataFrame, h1_df: pd.DataFrame) -> pd.DataFrame:
    """Compute combined features: M5 base (35) + H1 structure (10) = 45 total."""
    m5_feats = compute_features(m5_df, h1_df)
    if m5_feats is None or len(m5_feats) == 0:
        return None

    h1_feats = _compute_h1_structure_features(m5_df, h1_df)
    if h1_feats is None or len(h1_feats) == 0:
        for c in TREND_FEATURE_COLS:
            if c not in m5_feats.columns:
                m5_feats[c] = 0.0
        return m5_feats

    combined = m5_feats.join(h1_feats, how="left", rsuffix="_dup")
    # m5_h1_alignment: use h1_dir from base features
    if "h1_dir" in combined.columns:
        combined["m5_h1_alignment"] = combined["h1_dir"]
    for c in TREND_FEATURE_COLS:
        if c not in combined.columns:
            combined[c] = 0.0
    return combined


# =========================================================================
# TrendPredictor — predicts H1 trend (UP_TREND/DOWN_TREND/RANGING)
# =========================================================================

class TrendPredictor:
    """Predicts H1 trend structure using M5+H1 features.

    Classes: 0=DOWN_TREND, 1=UP_TREND, 2=RANGING
    Model expects TREND_FEATURE_COLS (45 features).
    """
    def __init__(self, model_path: str = None, model: xgb.XGBClassifier = None):
        if model is not None:
            self.model = model
        elif model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None
        self._feature_cols = TREND_FEATURE_COLS

    def predict_proba(self, features: pd.DataFrame) -> Tuple[float, float, float]:
        """Return (prob_down, prob_up, prob_ranging) for the most recent bar."""
        if self.model is None:
            logger.warning("TrendPredictor.predict_proba: model is None")
            return 0.33, 0.33, 0.34
        if features is None or len(features) == 0:
            logger.warning("TrendPredictor.predict_proba: empty features")
            return 0.33, 0.33, 0.34

        missing = [c for c in self._feature_cols if c not in features.columns]
        if missing:
            features = features.copy()
            for c in missing:
                features[c] = 0.0

        try:
            row = features[self._feature_cols].iloc[[-1]]
            probs = self.model.predict_proba(row.values)
            classes = list(self.model.classes_)
            result = [0.33, 0.33, 0.34]
            for i, cls in enumerate(classes):
                if cls == 0:
                    result[0] = float(probs[0][i])
                elif cls == 1:
                    result[1] = float(probs[0][i])
                elif cls == 2:
                    result[2] = float(probs[0][i])
            return tuple(result)
        except Exception as e:
            logger.error("TrendPredictor.predict_proba error: %s", e)
            return 0.33, 0.33, 0.34

    def predict(self, features: pd.DataFrame, confidence_threshold: float = 0.55) -> Optional[str]:
        """Return 'UP_TREND', 'DOWN_TREND', or None (RANGING/too low confidence)."""
        prob_down, prob_up, prob_ranging = self.predict_proba(features)
        if prob_up >= confidence_threshold and prob_up >= prob_down and prob_up >= prob_ranging:
            return "UP_TREND"
        if prob_down >= confidence_threshold and prob_down >= prob_up and prob_down >= prob_ranging:
            return "DOWN_TREND"
        return None  # RANGING or low confidence

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "TrendPredictor":
        return cls(model_path=path)


# =========================================================================
# TrendExhaustionPredictor — predicts when to exit (ALIVE/EXHAUSTED/NEW_SETUP)
# =========================================================================

class TrendExhaustionPredictor:
    """Predicts trend exhaustion during open trades.

    Classes: 0=TREND_EXHAUSTED, 1=TREND_ALIVE, 2=NEW_SETUP
    Features: 35 base + 5 trade-state = 40 total.
    """
    def __init__(self, model_path: str = None):
        self.model = joblib.load(model_path) if model_path and os.path.exists(model_path) else None
        self._feature_cols = EXIT_TREND_FEATURE_COLS

    def predict_proba(self, market_features: pd.DataFrame, trade_state: dict) -> Tuple[float, float, float]:
        """Return (prob_exhausted, prob_alive, prob_new_setup)."""
        if self.model is None:
            return 0.2, 0.6, 0.2
        try:
            row = {}
            for c in FEATURE_COLS:
                if c in market_features:
                    row[c] = float(market_features[c]) if hasattr(market_features[c], '__float__') else 0.0
                else:
                    row[c] = 0.0
            for c in TRADE_STATE_TREND_COLS:
                row[c] = float(trade_state.get(c, 0.0))
            vec = np.array([[row.get(c, 0.0) for c in self._feature_cols]], dtype=np.float32)
            probs = self.model.predict_proba(vec)
            classes = list(self.model.classes_)
            result = [0.2, 0.6, 0.2]
            for i, cls in enumerate(classes):
                if cls == 0:
                    result[0] = float(probs[0][i])
                elif cls == 1:
                    result[1] = float(probs[0][i])
                elif cls == 2:
                    result[2] = float(probs[0][i])
            return tuple(result)
        except Exception:
            return 0.2, 0.6, 0.2

    def predict_exit(self, market_features: pd.DataFrame, trade_state: dict,
                     exhaustion_threshold: float = 0.70,
                     new_setup_threshold: float = 0.80) -> Optional[str]:
        """Return 'TREND_ALIVE', 'TREND_EXHAUSTED', 'NEW_SETUP', or None (undecided)."""
        prob_exhausted, prob_alive, prob_new_setup = self.predict_proba(market_features, trade_state)
        if prob_exhausted >= exhaustion_threshold:
            return "TREND_EXHAUSTED"
        if prob_new_setup >= new_setup_threshold:
            return "NEW_SETUP"
        if prob_alive >= 0.50:
            return "TREND_ALIVE"
        return None

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "TrendExhaustionPredictor":
        return cls(model_path=path)
