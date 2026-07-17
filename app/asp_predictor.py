"""ASP (Adaptive Swing Probability) predictor.

Loads trained XGBoost model and predicts swing turning points.
Classes mapped via label_map from training (e.g. {0: -1, 1: 1} for binary).
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple

logger = logging.getLogger("GoldScalper")


class ASPPredictor:
    """Predicts swing turning points using 54 ASP features."""

    def __init__(self, model_path: str = None, feature_path: str = None):
        self.model = None
        self.label_map = {}
        self.inv_map = {}
        self.features = []
        self.ready = False

        if model_path and os.path.exists(model_path):
            try:
                saved = joblib.load(model_path)
                self.model = saved["model"]
                self.label_map = saved.get("label_map", {})
                self.inv_map = self.label_map  # {0: -1, 1: 1} maps model output to label
                logger.info("ASP model loaded: %s (classes=%s)", model_path, self.label_map)
            except Exception as e:
                logger.error("Failed to load ASP model: %s", e)
                return
        else:
            logger.warning("ASP model not found: %s", model_path)
            return

        if feature_path and os.path.exists(feature_path):
            try:
                self.features = list(np.load(feature_path, allow_pickle=True))
                logger.info("ASP features loaded: %d features", len(self.features))
            except Exception as e:
                logger.error("Failed to load ASP features: %s", e)
                return
        else:
            logger.warning("ASP features not found: %s", feature_path)
            return

        self.ready = True

    def _get_raw_probs(self, asp_features_row: pd.Series) -> Optional[dict]:
        """Get raw class probabilities from the model.

        Returns dict with keys 'buy', 'sell', 'neutral' or None on error.
        """
        if not self.ready or self.model is None:
            return None
        try:
            missing = [c for c in self.features if c not in asp_features_row.index]
            if missing:
                return None
            x = asp_features_row[self.features].values.reshape(1, -1)
            if np.any(np.isnan(x)):
                return None

            probs = self.model.predict_proba(x)[0]
            classes = list(self.model.classes_)
            result = {"buy": 0.0, "sell": 0.0, "neutral": 0.0}
            for i, cls in enumerate(classes):
                label = self.inv_map.get(cls, cls)
                if label == 1:
                    result["buy"] = float(probs[i])
                elif label == -1:
                    result["sell"] = float(probs[i])
                else:
                    result["neutral"] = float(probs[i])
            return result
        except Exception:
            return None

    def predict(self, asp_features_row: pd.Series) -> Tuple[Optional[str], float]:
        """Predict direction from a single ASP feature row.

        First checks argmax prediction. If NEUTRAL, falls back to
        probability-based: if P(BUY) or P(SELL) exceeds a threshold,
        use that direction instead.

        Returns:
            (direction, confidence) where direction is 'BUY', 'SELL', or None.
        """
        if not self.ready or self.model is None:
            return None, 0.0

        try:
            missing = [c for c in self.features if c not in asp_features_row.index]
            if missing:
                logger.warning("ASP predict: %d missing features: %s", len(missing), missing[:5])
                return None, 0.0

            x = asp_features_row[self.features].values.reshape(1, -1)
            nan_count = int(np.sum(np.isnan(x)))
            if nan_count > 0:
                logger.warning("ASP predict: %d NaN values in features", nan_count)
                return None, 0.0

            raw_pred = self.model.predict(x)[0]
            label = self.inv_map.get(raw_pred, raw_pred)

            probs = self.model.predict_proba(x)[0]
            confidence = float(np.max(probs))

            if label == 1:
                return "BUY", confidence
            elif label == -1:
                return "SELL", confidence

            # NEUTRAL argmax — check individual class probabilities
            prob_dict = self._get_raw_probs(asp_features_row)
            if prob_dict is not None:
                buy_p = prob_dict["buy"]
                sell_p = prob_dict["sell"]
                prob_threshold = 0.35
                margin = 0.10
                if buy_p >= prob_threshold and buy_p - sell_p >= margin:
                    return "BUY", buy_p
                if sell_p >= prob_threshold and sell_p - buy_p >= margin:
                    return "SELL", sell_p

            return None, confidence

        except Exception as e:
            logger.warning("ASP predict error: %s", e, exc_info=True)
            return None, 0.0

    def predict_batch(self, asp_features_df: pd.DataFrame) -> pd.Series:
        """Predict directions for a DataFrame of ASP features.

        Returns:
            Series of predictions: 0=BUY, -1=SELL, 1=NEUTRAL(=0 in signal).
        """
        if not self.ready or self.model is None:
            return pd.Series(1, index=asp_features_df.index, dtype=np.int8)

        try:
            valid_cols = [c for c in self.features if c in asp_features_df.columns]
            if len(valid_cols) != len(self.features):
                logger.warning("ASP features mismatch: %d/%d", len(valid_cols), len(self.features))
                return pd.Series(1, index=asp_features_df.index, dtype=np.int8)

            valid = asp_features_df[valid_cols].dropna()
            if len(valid) == 0:
                return pd.Series(1, index=asp_features_df.index, dtype=np.int8)

            feat_arr = valid.values
            raw_preds = self.model.predict(feat_arr)
            preds = np.array([self.inv_map.get(p, p) for p in raw_preds])

            signal_series = pd.Series(0, index=asp_features_df.index, dtype=np.int8)
            signal_series.loc[valid.index] = preds
            return signal_series

        except Exception as e:
            logger.warning("ASP batch predict error: %s", e)
            return pd.Series(1, index=asp_features_df.index, dtype=np.int8)

    def __bool__(self):
        return self.ready
