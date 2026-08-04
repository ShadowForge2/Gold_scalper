#!/usr/bin/env python3
"""
Brain Model Training Pipeline
=============================
Trains all Brain models from historical/synthetic trade data and exports
them to ONNX format (with joblib fallback).

Models:
  1. thesis_validator   – XGBoost classifier (HOLD vs EXIT)
  2. meta_fusion_xgb    – XGBoost ensemble member
  3. meta_fusion_lgb    – LightGBM ensemble member
  4. meta_fusion_cat    – CatBoost ensemble member
  5. meta_fusion_nn     – Neural Net ensemble member
  6. meta_fusion_rf     – Random Forest ensemble member
  7. twin_calibrator    – Volatility calibrator for Digital Twin

Usage:
  python _train_brain.py                         # synthetic data, default output
  python _train_brain.py --trades trades.csv     # use real trade log
  python _train_brain.py --output models/brain   # custom output dir
  python _train_brain.py --samples 50000         # custom sample count
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("brain_train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Feature lists -- imported from the RUNTIME modules so training and inference
# can never drift apart. If the import fails (partial install), fall back to
# embedded copies with the SAME content.

try:
    from app.brain.thesis_validator import THESIS_FEATURE_COLS
    from app.brain.meta_fusion import META_FEATURE_COLS, REGIME_ENCODING
    from app.brain.digital_twin import TWIN_INPUT_COLS, TWIN_REGIME_ENCODING
    _IMPORTED_FEATURE_LISTS = True
except Exception:
    _IMPORTED_FEATURE_LISTS = False
    REGIME_ENCODING = {
        "TRENDING": 1, "RANGING": -1, "VOLATILE": 0.5,
        "STAGNANT": -0.5, "REVERSAL": 0.75, "UNKNOWN": 0,
    }
    TWIN_REGIME_ENCODING = REGIME_ENCODING
    THESIS_FEATURE_COLS = [
        "pnl_atr_ratio", "bars_since_entry",
        "ml_confidence_entry", "ml_confidence_current", "ml_confidence_delta",
        "adx_entry", "adx_current", "adx_delta",
        "regime_entry", "regime_current", "regime_changed",
        "momentum_5m", "momentum_15m", "price_acceleration",
        "consecutive_bars_in_direction",
        "volatility_percentile", "atr_ratio",
        "structure_quality_score", "structure_break_detected",
        "distance_to_sl", "distance_to_tp",
        "session_active", "is_friday",
        "opportunity_cost_per_hour",
        "recent_win_rate_20",
    ]
    META_FEATURE_COLS = [
        "thesis_validity", "thesis_confidence", "perception_composite",
        "regime_type_encoded",
        "opportunity_cost", "opportunity_count",
        "dt_ev_diff", "dt_hold_win_pct", "dt_confidence",
        "pnl_atr_ratio", "ml_confidence_current", "adx_current",
        "momentum_5m", "volatility_percentile",
        "bars_since_entry", "session_active", "is_friday",
        "recent_win_rate_20", "consecutive_losses", "portfolio_heat",
    ]
    TWIN_INPUT_COLS = [
        "atr_current", "regime_type_encoded",
        "volatility_percentile", "adx_current",
    ]

if _IMPORTED_FEATURE_LISTS:
    log.info(
        "Feature lists imported from runtime modules "
        "(thesis=%d, meta=%d, twin=%d)",
        len(THESIS_FEATURE_COLS), len(META_FEATURE_COLS), len(TWIN_INPUT_COLS),
    )

FEATURE_IMPORTANCE_FILE = "feature_importance.json"
TWIN_TARGET_COLS = ["twin_drift_scale", "twin_vol_scale"]
METADATA_FILE = "training_metadata.json"


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

class SyntheticTradeGenerator:
    """Generates realistic synthetic M5 boundary trade data for training."""

    def __init__(self, n_samples: int = 15_000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def _regime(self) -> np.ndarray:
        """Simulated regime: 0=range, 1=trending, 2=volatile."""
        return self.rng.choice([0, 1, 2], size=self.n_samples, p=[0.4, 0.35, 0.25])

    def generate_thesis_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate data for the 25 runtime thesis features. Label 1=HOLD, 0=EXIT.

        Columns and units match app/brain/thesis_validator.THESIS_FEATURE_COLS
        and the values the runtime FeatureCache emits (unscaled).
        """
        rng = self.rng
        n = self.n_samples
        regime = self._regime()

        def _clip(a: np.ndarray) -> np.ndarray:
            return np.clip(a, -5.0, 5.0)

        # regime_entry/current must use the SAME encoding as the runtime
        # FeatureCache (TRENDING=1, RANGING=-1, VOLATILE=0.5, STAGNANT=-0.5, UNKNOWN=0).
        regime_enc = np.array([REGIME_ENCODING.get(
            {0: "RANGING", 1: "TRENDING", 2: "VOLATILE"}[r], 0.0) for r in regime])

        df = pd.DataFrame({
            "pnl_atr_ratio":              _clip(rng.normal(0.3, 1.2, n)),
            "bars_since_entry":           rng.poisson(6, n).clip(0, 50).astype(float),
            "ml_confidence_entry":        rng.uniform(0.3, 0.95, n),
            "ml_confidence_current":      rng.uniform(0.3, 0.95, n),
            "ml_confidence_delta":        rng.normal(0.0, 0.12, n),
            "adx_entry":                  rng.uniform(10, 60, n),
            "adx_current":                rng.uniform(10, 60, n),
            "adx_delta":                  rng.normal(0.0, 8.0, n),
            "regime_entry":               regime_enc,
            "regime_current":             np.roll(regime_enc, 1),
            "regime_changed":             (regime_enc != np.roll(regime_enc, 1)).astype(float),
            "momentum_5m":                _clip(rng.normal(0.0, 0.8, n) + 0.15 * regime),
            "momentum_15m":               _clip(rng.normal(0.0, 1.0, n) + 0.10 * regime),
            "price_acceleration":         _clip(rng.normal(0.0, 0.3, n)),
            "consecutive_bars_in_direction": rng.poisson(3, n).clip(0, 15).astype(float),
            "volatility_percentile":      rng.uniform(0, 1, n),
            "atr_ratio":                  rng.uniform(0.5, 2.0, n),
            "structure_quality_score":    rng.uniform(0, 1, n),
            "structure_break_detected":   rng.binomial(1, 0.15, n).astype(float),
            "distance_to_sl":             rng.exponential(1.5, n).clip(0.1, 8),
            "distance_to_tp":             rng.exponential(2.0, n).clip(0.1, 10),
            "session_active":             rng.binomial(1, 0.7, n).astype(float),
            "is_friday":                  rng.binomial(1, 0.2, n).astype(float),
            "opportunity_cost_per_hour":  rng.exponential(0.5, n).clip(0, 5),
            "recent_win_rate_20":         rng.uniform(0.2, 0.8, n),
        })

        # --- Label logic (HOLD=1, EXIT=0) ---
        # Holding is better when: positive pnl, ML confident, trending regime,
        # strong structure, fresh entry, active session.
        hold_score = (
            0.25 * (df["pnl_atr_ratio"] > 0.3).astype(float)
            + 0.20 * (df["ml_confidence_current"] > 0.6).astype(float)
            + 0.15 * (df["regime_current"] == 1).astype(float)
            + 0.10 * (df["adx_current"] > 25).astype(float)
            + 0.10 * (df["bars_since_entry"] < 12).astype(float)
            + 0.10 * (df["session_active"] == 1).astype(float)
            + 0.05 * (df["structure_quality_score"] > 0.6).astype(float)
            + 0.05 * (df["distance_to_tp"] > df["distance_to_sl"]).astype(float)
        )
        label = (hold_score + rng.normal(0, 0.12, n) > 0.50).astype(int)

        return df, pd.Series(label, name="label")

    def generate_meta_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate data for the 20 runtime meta-fusion features. Label 1=EXIT, 0=HOLD.

        Values are emitted in the SAME normalized units the runtime feeds the
        ensemble (opportunity_count/10, adx/50, bars_since_entry/24,
        session_active/3, consecutive_losses/5, and REGIME_ENCODING).
        """
        rng = self.rng
        n = self.n_samples
        regime = self._regime()
        regime_enc = np.array([REGIME_ENCODING.get(
            {0: "RANGING", 1: "TRENDING", 2: "VOLATILE"}[r], 0.0) for r in regime])

        opportunity_count = rng.poisson(3, n).clip(0, 15).astype(float)
        adx = rng.uniform(10, 60, n)
        bars = rng.poisson(6, n).clip(0, 50).astype(float)
        cons_loss = rng.poisson(2, n).clip(0, 10).astype(float)
        session = rng.binomial(1, 0.7, n).astype(float)

        df = pd.DataFrame({
            "thesis_validity":          rng.uniform(0, 1, n),
            "thesis_confidence":        rng.uniform(0.2, 1.0, n),
            "perception_composite":     rng.normal(0.5, 0.25, n).clip(0, 1),
            "regime_type_encoded":      regime_enc,
            "opportunity_cost":         rng.exponential(0.5, n).clip(0, 5),
            "opportunity_count":        opportunity_count / 10.0,
            "dt_ev_diff":               rng.normal(0, 1, n),
            "dt_hold_win_pct":          rng.uniform(0.3, 0.8, n),
            "dt_confidence":            rng.uniform(0.2, 0.95, n),
            "pnl_atr_ratio":            rng.normal(0.2, 1.5, n),
            "ml_confidence_current":    rng.uniform(0.3, 0.95, n),
            "adx_current":              adx / 50.0,
            "momentum_5m":              rng.normal(0, 0.6, n),
            "volatility_percentile":    rng.uniform(0, 1, n),
            "bars_since_entry":         bars / 24.0,
            "session_active":           session / 3.0,
            "is_friday":                rng.binomial(1, 0.2, n).astype(float),
            "recent_win_rate_20":       rng.uniform(0.2, 0.8, n),
            "consecutive_losses":       cons_loss / 5.0,
            "portfolio_heat":           rng.uniform(0, 1, n),
        })

        # --- Label logic (EXIT=1, HOLD=0) ---
        exit_score = (
            0.20 * (df["pnl_atr_ratio"] < -0.5).astype(float)
            + 0.20 * (df["thesis_validity"] < 0.4).astype(float)
            + 0.15 * (df["consecutive_losses"] >= 0.6).astype(float)
            + 0.15 * (df["portfolio_heat"] > 0.7).astype(float)
            + 0.10 * (df["dt_ev_diff"] < -0.5).astype(float)
            + 0.10 * (df["bars_since_entry"] > 0.83).astype(float)
            + 0.10 * (df["volatility_percentile"] > 0.85).astype(float)
        )
        label = (exit_score + rng.normal(0, 0.10, n) > 0.50).astype(int)

        return df, pd.Series(label, name="label")

    def generate_twin_data(self) -> pd.DataFrame:
        """Generate twin calibrator dataset matching TWIN_INPUT_COLS (4 features).

        Targets are drift_scale, vol_scale. regime_type_encoded uses the same
        REGIME_ENCODING as the runtime digital_twin feeder.
        """
        rng = self.rng
        n = self.n_samples
        regime = self._regime()
        regime_enc = np.array([REGIME_ENCODING.get(
            {0: "RANGING", 1: "TRENDING", 2: "VOLATILE"}[r], 0.0) for r in regime])

        df = pd.DataFrame({
            "atr_current":             rng.exponential(2, n).clip(0.3, 10),
            "regime_type_encoded":     regime_enc,
            "volatility_percentile":   rng.uniform(0, 1, n),
            "adx_current":             rng.uniform(10, 60, n),
        })

        # Drift scale: higher in trending, lower in range-bound
        drift = np.where(
            regime == 1,
            rng.uniform(0.8, 1.5, n),
            np.where(
                regime == 2,
                rng.uniform(0.3, 0.9, n),
                rng.uniform(0.5, 1.1, n),
            ),
        )
        drift += 0.3 * (df["adx_current"] - 30) / 30
        drift = np.clip(drift, 0.1, 2.0)

        # Vol scale: higher in volatile regime
        vol_scale = np.where(
            regime == 2,
            rng.uniform(1.2, 2.0, n),
            np.where(
                regime == 1,
                rng.uniform(0.8, 1.3, n),
                rng.uniform(0.5, 1.0, n),
            ),
        )
        vol_scale += 0.5 * (df["volatility_percentile"] - 0.5)
        vol_scale = np.clip(vol_scale, 0.3, 3.0)

        df["twin_drift_scale"] = drift
        df["twin_vol_scale"] = vol_scale

        return df

    def load_from_trades(self, path: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Attempt to load and parse a real trade CSV."""
        log.info("Loading real trades from %s", path)
        raw = pd.read_csv(path)
        log.info("Loaded %d rows, columns: %s", len(raw), list(raw.columns)[:20])
        # Placeholder: real pipeline would map columns → features.
        log.warning(
            "Real trade mapping not implemented yet – falling back to synthetic data."
        )
        return {}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _export_onnx(model: Any, X_sample: np.ndarray, out_path: Path, name: str) -> bool:
    """Try to export *model* to ONNX.  Returns True on success."""
    try:
        if name == "meta_fusion_lgb":
            from onnxmltools import convert_lightgbm
            from onnxconverter_common import FloatTensorType
            initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
            onx = convert_lightgbm(model, initial_types=initial_type)
        elif name == "meta_fusion_cat":
            from onnxmltools import convert_catboost
            from onnxconverter_common import FloatTensorType
            initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
            onx = convert_catboost(model, initial_types=initial_type)
        elif name in ("meta_fusion_nn", "meta_fusion_rf", "thesis_validator",
                       "meta_fusion_xgb"):
            from skl2onnx import convert_sklearn
            from onnxconverter_common import FloatTensorType
            initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
            onx = convert_sklearn(model, initial_types=initial_type)
        elif name == "twin_calibrator":
            from skl2onnx import convert_sklearn
            from onnxconverter_common import FloatTensorType
            initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
            onx = convert_sklearn(model, initial_types=initial_type)
        else:
            log.warning("No ONNX converter for %s", name)
            return False

        onx_path = out_path / f"{name}.onnx"
        with open(onx_path, "wb") as f:
            f.write(onx.SerializeToString())
        log.info("  → Exported ONNX: %s", onx_path)
        return True
    except Exception as exc:
        log.warning("  ONNX export failed for %s: %s", name, exc)
        return False


def _save_joblib(model: Any, out_path: Path, name: str) -> Path:
    jb_path = out_path / f"{name}.joblib"
    joblib.dump(model, jb_path)
    log.info("  → Saved joblib: %s", jb_path)
    return jb_path


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

class BrainTrainer:
    """Orchestrates training of all Brain sub-models."""

    def __init__(self, output_dir: str, n_samples: int = 15_000):
        self.out = _ensure_dir(output_dir)
        self.n_samples = n_samples
        self.metadata: Dict[str, Any] = {"models": {}, "n_samples": n_samples}

    # ---- Thesis Validator (XGBoost) ----
    def train_thesis_validator(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score

        log.info("Training thesis_validator (XGBoost, %d samples, %d features)",
                 X.shape[0], X.shape[1])

        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        t0 = time.time()
        model.fit(X, y, verbose=False)
        train_time = time.time() - t0

        # Quick cross-val estimate
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        auc = float(cv_scores.mean())

        log.info("  AUC: %.4f (+/- %.4f), train time: %.1fs", auc, cv_scores.std(), train_time)

        feat_imp = dict(zip(THESIS_FEATURE_COLS, model.feature_importances_.tolist()))

        _save_joblib(model, self.out, "thesis_validator")
        _export_onnx(model, X, self.out, "thesis_validator")

        return {"model": model, "auc": auc, "train_time": train_time,
                "feature_importance": feat_imp}

    # ---- Meta-fusion XGBoost ----
    def train_meta_xgb(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        from xgboost import XGBClassifier

        log.info("Training meta_fusion_xgb (%d samples)", X.shape[0])

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        t0 = time.time()
        model.fit(X, y, verbose=False)
        train_time = time.time() - t0

        _save_joblib(model, self.out, "meta_fusion_xgb")
        _export_onnx(model, X, self.out, "meta_fusion_xgb")

        return {"model": model, "train_time": train_time}

    # ---- Meta-fusion LightGBM ----
    def train_meta_lgb(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        import lightgbm as lgb

        log.info("Training meta_fusion_lgb (%d samples)", X.shape[0])

        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary",
            metric="binary_logloss",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        t0 = time.time()
        model.fit(X, y)
        train_time = time.time() - t0

        _save_joblib(model, self.out, "meta_fusion_lgb")
        _export_onnx(model, X, self.out, "meta_fusion_lgb")

        return {"model": model, "train_time": train_time}

    # ---- Meta-fusion CatBoost ----
    def train_meta_cat(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        from catboost import CatBoostClassifier

        log.info("Training meta_fusion_cat (%d samples)", X.shape[0])

        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0,
            loss_function="Logloss",
        )

        t0 = time.time()
        model.fit(X, y)
        train_time = time.time() - t0

        _save_joblib(model, self.out, "meta_fusion_cat")
        _export_onnx(model, X, self.out, "meta_fusion_cat")

        return {"model": model, "train_time": train_time}

    # ---- Meta-fusion Neural Net (MLPClassifier) ----
    def train_meta_nn(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        log.info("Training meta_fusion_nn (%d samples)", X.shape[0])

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ])

        t0 = time.time()
        model.fit(X, y)
        train_time = time.time() - t0

        _save_joblib(model, self.out, "meta_fusion_nn")
        _export_onnx(model, X, self.out, "meta_fusion_nn")

        return {"model": model, "train_time": train_time}

    # ---- Meta-fusion Random Forest ----
    def train_meta_rf(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier

        log.info("Training meta_fusion_rf (%d samples)", X.shape[0])

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        t0 = time.time()
        model.fit(X, y)
        train_time = time.time() - t0

        _save_joblib(model, self.out, "meta_fusion_rf")
        _export_onnx(model, X, self.out, "meta_fusion_rf")

        return {"model": model, "train_time": train_time}

    # ---- Twin Calibrator (Ridge regression → 2 outputs) ----
    def train_twin_calibrator(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        log.info("Training twin_calibrator (%d samples, %d targets)",
                 X.shape[0], y.shape[1])

        base = Ridge(alpha=1.0)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", MultiOutputRegressor(base)),
        ])

        t0 = time.time()
        model.fit(X, y)
        train_time = time.time() - t0

        train_pred = model.predict(X)
        mse = float(np.mean((train_pred - y) ** 2))
        log.info("  Train MSE: %.6f, time: %.1fs", mse, train_time)

        _save_joblib(model, self.out, "twin_calibrator")
        _export_onnx(model, X, self.out, "twin_calibrator")

        return {"model": model, "train_mse": mse, "train_time": train_time}

    # ---- Run all ----
    def run_all(
        self,
        trade_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry: generate/load data, train everything, save metadata."""
        log.info("=" * 60)
        log.info("Brain Training Pipeline – %d synthetic samples", self.n_samples)
        log.info("=" * 60)

        gen = SyntheticTradeGenerator(n_samples=self.n_samples, seed=42)

        # --- Thesis ---
        log.info("")
        log.info("--- Phase 1: Thesis Validator ---")
        X_th, y_th = gen.generate_thesis_data()
        th_result = self.train_thesis_validator(X_th.values, y_th.values)
        self.metadata["models"]["thesis_validator"] = {
            "auc": th_result["auc"],
            "train_time": th_result["train_time"],
            "features": THESIS_FEATURE_COLS,
            "n_features": len(THESIS_FEATURE_COLS),
        }

        # --- Meta-fusion ---
        log.info("")
        log.info("--- Phase 2: Meta-Fusion Ensemble ---")
        X_meta, y_meta = gen.generate_meta_data()
        Xm = X_meta.values
        ym = y_meta.values

        meta_results = {}
        for key, fn in (
            ("xgb", self.train_meta_xgb),
            ("lgb", self.train_meta_lgb),
            ("cat", self.train_meta_cat),
            ("nn", self.train_meta_nn),
            ("rf", self.train_meta_rf),
        ):
            try:
                meta_results[key] = fn(Xm, ym)
            except ImportError as exc:
                log.warning("Skipping meta_fusion_%s (dependency missing: %s)", key, exc)
            except Exception as exc:
                log.warning("Skipping meta_fusion_%s (training failed: %s)", key, exc)

        for key, res in meta_results.items():
            self.metadata["models"][f"meta_fusion_{key}"] = {
                "train_time": res["train_time"],
                "features": META_FEATURE_COLS,
                "n_features": len(META_FEATURE_COLS),
            }

        # --- Twin Calibrator ---
        log.info("")
        log.info("--- Phase 3: Twin Calibrator ---")
        twin_df = gen.generate_twin_data()
        Xt = twin_df[TWIN_INPUT_COLS].values
        yt = twin_df[TWIN_TARGET_COLS].values
        twin_result = self.train_twin_calibrator(Xt, yt)
        self.metadata["models"]["twin_calibrator"] = {
            "train_mse": twin_result["train_mse"],
            "train_time": twin_result["train_time"],
            "input_features": TWIN_INPUT_COLS,
            "target_features": TWIN_TARGET_COLS,
        }

        # --- Save global feature importance (from thesis XGB) ---
        fi_path = self.out / FEATURE_IMPORTANCE_FILE
        with open(fi_path, "w") as f:
            json.dump(th_result["feature_importance"], f, indent=2)
        log.info("Saved feature importance → %s", fi_path)

        # --- Save metadata ---
        self.metadata["total_train_time"] = sum(
            r.get("train_time", 0)
            for r in [th_result, *meta_results.values(), twin_result]
        )
        self.metadata["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta_path = self.out / METADATA_FILE
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        log.info("Saved metadata → %s", meta_path)

        self._summary()
        return self.metadata

    def _summary(self):
        log.info("")
        log.info("=" * 60)
        log.info("TRAINING COMPLETE")
        log.info("=" * 60)
        for name, info in self.metadata["models"].items():
            extras = ", ".join(f"{k}={v}" for k, v in info.items()
                               if k not in ("features",))
            log.info("  %-24s  %s", name, extras)
        log.info("  Total time: %.1fs", self.metadata["total_train_time"])
        log.info("  Output dir: %s", self.out)

        # List exported files
        onnx_files = sorted(self.out.glob("*.onnx"))
        joblib_files = sorted(self.out.glob("*.joblib"))
        log.info("  ONNX files:   %d", len(onnx_files))
        log.info("  Joblib files: %d", len(joblib_files))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Brain models and export to ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        default="models/brain",
        help="Output directory for trained models (default: models/brain)",
    )
    parser.add_argument(
        "--trades", "-t",
        default=None,
        help="Path to real trade CSV. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=15_000,
        help="Number of synthetic samples to generate (default: 15000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("Python %s", sys.version)
    log.info("Output directory: %s", args.output)
    if args.trades:
        log.info("Trade data: %s", args.trades)

    trainer = BrainTrainer(output_dir=args.output, n_samples=args.samples)

    try:
        trainer.run_all(trade_path=args.trades)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 130
    except Exception:
        log.exception("Training pipeline failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
