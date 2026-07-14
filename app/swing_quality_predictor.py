"""
Swing Quality Predictor — XGBoost model for detecting H4 reversal points.

Loaded as a confirmation gate in the signal engine. If the model predicts
the current bar is NOT at a reversal point, the signal is rejected.
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from app.asp_features import ASP_FEATURE_COLS


class SwingQualityPredictor:
    """Lightweight wrapper around trained XGBoost swing quality model."""

    def __init__(self, model_path: str = "models/swing_quality_xgb.json"):
        self.model = None
        self.ready = False
        self._load(model_path)

    def _load(self, path: str) -> None:
        if not _HAS_XGB:
            return
        if not os.path.exists(path):
            return
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(path)
            self.ready = True
        except Exception:
            self.ready = False

    def predict_quality(self, features: pd.DataFrame) -> Tuple[Optional[float], Optional[bool]]:
        """Predict swing quality probability for the last bar.

        Args:
            features: DataFrame with ASP_FEATURE_COLS columns (from compute_asp_features).

        Returns:
            (probability, passes_gate) or (None, None) if not ready.
        """
        if not self.ready or self.model is None:
            return None, None

        if features is None or len(features) == 0:
            return None, None

        try:
            row = features.iloc[-1:].values.astype(np.float32)
            if np.isnan(row).any() or np.isinf(row).any():
                return None, None

            prob = float(self.model.predict_proba(row)[0, 1])
            return prob, None  # caller decides threshold
        except Exception:
            return None, None

    def predict_batch(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict swing quality for multiple bars (for backtesting).

        Returns array of probabilities or None if not ready.
        """
        if not self.ready or self.model is None:
            return None

        if features is None or len(features) == 0:
            return None

        try:
            arr = features.values.astype(np.float32)
            valid = ~(np.isnan(arr).any(axis=1) | np.isinf(arr).any(axis=1))
            probs = np.full(len(arr), np.nan)
            if valid.any():
                probs[valid] = self.model.predict_proba(arr[valid])[:, 1]
            return probs
        except Exception:
            return None
