"""ASP (Adaptive Swing Probability) predictor.

Loads trained XGBoost model and predicts swing turning points.
Classes: 0=BUY, 1=NEUTRAL, 2=SELL (via label_map from training).
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


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
                self.inv_map = {v: k for k, v in self.label_map.items()}
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

    def predict(self, asp_features_row: pd.Series) -> Tuple[Optional[str], float]:
        """Predict direction from a single ASP feature row.

        Args:
            asp_features_row: Single row of ASP features (from compute_asp_features).

        Returns:
            (direction, confidence) where direction is 'BUY', 'SELL', or None.
        """
        if not self.ready or self.model is None:
            return None, 0.0

        try:
            missing = [c for c in self.features if c not in asp_features_row.index]
            if missing:
                return None, 0.0

            x = asp_features_row[self.features].values.reshape(1, -1)
            if np.any(np.isnan(x)):
                return None, 0.0

            raw_pred = self.model.predict(x)[0]
            label = self.inv_map.get(raw_pred, raw_pred)

            probs = self.model.predict_proba(x)[0]
            confidence = float(np.max(probs))

            if label == 1:
                return "BUY", confidence
            elif label == -1:
                return "SELL", confidence
            else:
                return None, confidence

        except Exception as e:
            logger.warning("ASP predict error: %s", e)
            return None, 0.0

    def predict_batch(self, asp_features_df: pd.DataFrame) -> pd.Series:
        """Predict directions for a DataFrame of ASP features.

        Returns:
            Series of predictions: 0=BUY, -1=SELL, 1=NEUTRAL(=0 in signal).
        """
        if not self.ready or self.model is None:
            return pd.Series(0, index=asp_features_df.index, dtype=np.int8)

        try:
            valid_cols = [c for c in self.features if c in asp_features_df.columns]
            if len(valid_cols) != len(self.features):
                logger.warning("ASP features mismatch: %d/%d", len(valid_cols), len(self.features))

            valid = asp_features_df[valid_cols].dropna()
            if len(valid) == 0:
                return pd.Series(0, index=asp_features_df.index, dtype=np.int8)

            feat_arr = valid.values
            raw_preds = self.model.predict(feat_arr)
            preds = np.array([self.inv_map.get(p, p) for p in raw_preds])

            signal_series = pd.Series(0, index=asp_features_df.index, dtype=np.int8)
            signal_series.loc[valid.index] = preds
            return signal_series

        except Exception as e:
            logger.warning("ASP batch predict error: %s", e)
            return pd.Series(0, index=asp_features_df.index, dtype=np.int8)

    def __bool__(self):
        return self.ready
