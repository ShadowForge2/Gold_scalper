import os
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Tuple
import joblib


FEATURE_COLS = [
    "return_1", "return_2", "return_3", "return_5", "return_10",
    "return_20",
    "rsi_14",
    "atr_norm",
    "bb_position",
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
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    vols = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df.columns else np.ones(len(df))

    features = pd.DataFrame(index=df.index)

    features["return_1"] = np.log(closes / np.roll(closes, 1))
    features["return_2"] = np.log(closes / np.roll(closes, 2))
    features["return_3"] = np.log(closes / np.roll(closes, 3))
    features["return_5"] = np.log(closes / np.roll(closes, 5))
    features["return_10"] = np.log(closes / np.roll(closes, 10))
    features["return_20"] = np.log(closes / np.roll(closes, 20))

    features["hl_ratio"] = (highs - lows) / closes

    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1)),
        ),
    )
    atr_series = pd.Series(tr, index=df.index).rolling(14, min_periods=2).mean()
    features["atr_norm"] = atr_series / closes

    roll_vol = pd.Series(vols, index=df.index).rolling(20, min_periods=2).mean()
    roll_vol = roll_vol.replace(0, np.nan)
    features["volume_ratio"] = pd.Series(vols, index=df.index) / roll_vol

    delta = pd.Series(closes - np.roll(closes, 1), index=df.index)
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

    candle_range = highs - lows
    features["close_position"] = np.where(
        candle_range > 0, (closes - lows) / candle_range, 0.5
    )

    features["return_vol_ratio"] = (
        features["return_1"].rolling(5, min_periods=2).mean()
        / short_vol.replace(0, np.nan)
    )

    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    hours = times.hour if hasattr(times, "hour") else times.hour
    days = times.dayofweek if hasattr(times, "dayofweek") else times.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["day_sin"] = np.sin(2 * np.pi * days / 7)
    features["day_cos"] = np.cos(2 * np.pi * days / 7)

    nan_cols = features.columns[features.isna().all()]
    features = features.drop(columns=nan_cols)

    features = features[~features.index.duplicated(keep="first")]
    return features


def create_target(df: pd.DataFrame, horizon: int = 3, atr_threshold: float = 0.3) -> pd.Series:
    """Create binary target: 1 if close[n+horizon] > close[n] + threshold, 0 if < -threshold, else NaN."""
    closes = df["close"].values.astype(np.float64)
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

    up = future_close > closes + threshold
    down = future_close < closes - threshold

    target = np.full(len(closes), np.nan)
    target[up] = 1.0
    target[down] = 0.0

    target[np.isnan(atr)] = np.nan
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
    """Train XGBoost classifier with early stopping."""
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        eval_metric="logloss",
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

    def predict_proba(self, features: pd.DataFrame) -> Tuple[float, float]:
        """Return (prob_down, prob_up) for the most recent bar."""
        if self.model is None:
            return 0.5, 0.5
        missing = [c for c in self._feature_cols if c not in features.columns]
        if missing:
            for c in missing:
                features[c] = 0.0
        row = features[self._feature_cols].iloc[[-1]]
        probs = self.model.predict_proba(row.values)
        if self.model.classes_[0] == 0:
            return float(probs[0][0]), float(probs[0][1])
        return float(probs[0][1]), float(probs[0][0])

    def predict(self, features: pd.DataFrame, confidence_threshold: float = 0.55) -> Optional[str]:
        """Return 'BUY' or 'SELL' if confidence exceeds threshold, else None."""
        prob_down, prob_up = self.predict_proba(features)
        if prob_up >= confidence_threshold:
            return "BUY"
        if prob_down >= confidence_threshold:
            return "SELL"
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
