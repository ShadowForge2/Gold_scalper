"""DigitalTwin — Layer 4: Monte Carlo Simulation Engine.

Simulates 800 possible price paths from the current state and compares
the expected value of holding vs exiting + taking the best waiting trade.

Runs only when thesis is uncertain. Cost: ~35ms. RAM: ~25MB.
"""
import os
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger("GoldScalper.Brain.DigitalTwin")

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

# Canonical input columns for the twin_calibrator model.
# Only features the runtime FeatureCache actually computes are used, so the
# training pipeline (which imports this list) can never drift from inference.
TWIN_INPUT_COLS = [
    "atr_current",
    "regime_type_encoded",
    "volatility_percentile",
    "adx_current",
]

# Regime encoding must match meta_fusion.REGIME_ENCODING so the twin
# calibrator sees the same numeric regime code as the rest of the Brain.
TWIN_REGIME_ENCODING = {
    "TRENDING": 1, "RANGING": -1, "VOLATILE": 0.5,
    "STAGNANT": -0.5, "REVERSAL": 0.75, "UNKNOWN": 0,
}


class DigitalTwin:
    """Monte Carlo simulation engine for hold-vs-exit decisions.

    Pre-calibrates volatility surfaces per regime. Simulates price paths
    using regime-appropriate distributions (not just normal).
    """

    def __init__(self, calibrator_path: str = None, n_sims: int = 800):
        self.n_sims = n_sims
        self._calibrator = None
        self._onnx_session = None

        if calibrator_path:
            candidates = [calibrator_path]
            if calibrator_path.endswith(".onnx"):
                candidates.append(calibrator_path.replace(".onnx", ".joblib"))
            for p in candidates:
                if os.path.exists(p):
                    self._load_calibrator(p)
                    if self._onnx_session is not None or self._calibrator is not None:
                        break

        self._regime_params = {
            "TRENDING": {"drift_scale": 0.3, "vol_scale": 1.0, "skew": 0.1, "kurtosis": 3.5},
            "RANGING": {"drift_scale": 0.05, "vol_scale": 0.8, "skew": 0.0, "kurtosis": 4.0},
            "VOLATILE": {"drift_scale": 0.2, "vol_scale": 1.5, "skew": 0.15, "kurtosis": 5.0},
            "STAGNANT": {"drift_scale": 0.02, "vol_scale": 0.5, "skew": 0.0, "kurtosis": 3.0},
            "REVERSAL": {"drift_scale": 0.25, "vol_scale": 1.2, "skew": -0.2, "kurtosis": 4.5},
            "UNKNOWN": {"drift_scale": 0.15, "vol_scale": 1.0, "skew": 0.0, "kurtosis": 3.5},
        }

    def _load_calibrator(self, path: str):
        try:
            if path.endswith(".onnx"):
                if not _HAS_ONNX:
                    logger.warning("onnxruntime not installed, falling back to joblib")
                    return
                self._onnx_session = ort.InferenceSession(path)
                logger.info("DigitalTwin calibrator loaded: %s", path)
                return
            import joblib
            self._calibrator = joblib.load(path)
            logger.info("DigitalTwin calibrator loaded: %s", path)
        except Exception as e:
            logger.warning("DigitalTwin calibrator load failed: %s", e)

    def simulate(self, current_price: float, atr: float, direction: str,
                 entry_price: float, current_pnl: float, regime: str,
                 minutes_held: float, opportunity_data: Optional[Dict] = None,
                 features: Optional[Dict] = None,
                 n_sims: Optional[int] = None) -> Dict:
        """Run Monte Carlo simulation.

        Returns:
            {
                "ev_hold": float,      # expected value of holding (in ATR)
                "ev_exit": float,      # expected value of exiting now
                "ev_diff": float,      # ev_exit - ev_hold
                "hold_win_pct": float, # % of sims where holding wins
                "exit_win_pct": float, # % of sims where exiting wins
                "max_drawdown_hold": float,
                "recommendation": "HOLD" | "EXIT",
                "confidence": float,
                "n_sims": int,
                "sim_time_ms": float,
            }
        """
        t0 = time.time()
        sims = n_sims or self.n_sims

        params = self._get_regime_params(regime, atr, features)

        horizon_bars = max(6, min(48, int((120 - minutes_held) / 5)))
        if horizon_bars <= 0:
            horizon_bars = 6

        price_paths = self._generate_paths(
            current_price=current_price,
            atr=atr,
            params=params,
            n_sims=sims,
            n_bars=horizon_bars,
        )

        hold_pnls = np.zeros(sims)
        for i in range(sims):
            path = price_paths[i]
            final_price = path[-1]
            if direction == "BUY":
                hold_pnls[i] = (final_price - entry_price) / atr if atr > 0 else 0
            else:
                hold_pnls[i] = (entry_price - final_price) / atr if atr > 0 else 0

        exit_pnl = current_pnl
        if opportunity_data:
            opp_expected = opportunity_data.get("expected_pnl", 0)
            exit_pnls = np.full(sims, exit_pnl + opp_expected)
        else:
            exit_pnls = np.full(sims, exit_pnl)

        ev_hold = float(np.mean(hold_pnls))
        ev_exit = float(np.mean(exit_pnls))
        ev_diff = ev_exit - ev_hold

        hold_win_pct = float(np.mean(hold_pnls > exit_pnls))
        exit_win_pct = 1.0 - hold_win_pct

        peak_hold = np.maximum.accumulate(price_paths[:, :, 0] if price_paths.ndim == 3 else price_paths[:, -1:])
        if direction == "BUY":
            max_dd = float(np.min(hold_pnls - np.maximum(0, hold_pnls)))
        else:
            max_dd = float(np.min(hold_pnls - np.maximum(0, hold_pnls)))

        confidence = abs(ev_diff) / (atr / current_price) if current_price > 0 else 0
        confidence = min(1.0, confidence * 0.5)

        if ev_diff > 0.05 and exit_win_pct > 0.55:
            recommendation = "EXIT"
        elif ev_diff < -0.05 and hold_win_pct > 0.55:
            recommendation = "HOLD"
        else:
            recommendation = "HOLD" if hold_win_pct >= exit_win_pct else "EXIT"

        sim_time = (time.time() - t0) * 1000

        return {
            "ev_hold": ev_hold,
            "ev_exit": ev_exit,
            "ev_diff": ev_diff,
            "hold_win_pct": hold_win_pct,
            "exit_win_pct": exit_win_pct,
            "max_drawdown_hold": max_dd,
            "recommendation": recommendation,
            "confidence": confidence,
            "n_sims": sims,
            "sim_time_ms": sim_time,
        }

    def _get_regime_params(self, regime: str, atr: float,
                           features: Optional[Dict] = None) -> Dict:
        base = self._regime_params.get(regime, self._regime_params["UNKNOWN"])

        if self._onnx_session is not None or self._calibrator is not None:
            try:
                if features:
                    regime_raw = features.get("regime_current", regime)
                    if not isinstance(regime_raw, str):
                        regime_enc = float(regime_raw)
                    else:
                        regime_enc = TWIN_REGIME_ENCODING.get(regime_raw, 0.0)
                    x = np.array([[
                        float(features.get("atr_current", atr)),
                        regime_enc,
                        float(features.get("volatility_percentile", 0.5)),
                        float(features.get("adx_current", 0)),
                    ]], dtype=np.float32)
                else:
                    x = np.array([[atr, TWIN_REGIME_ENCODING.get(regime, 0), 0.5, 0]],
                                 dtype=np.float32)
                if self._onnx_session is not None:
                    input_name = self._onnx_session.get_inputs()[0].name
                    corrected = self._onnx_session.run(None, {input_name: x})[0]
                else:
                    corrected = self._calibrator.predict(x)[0]
                return {
                    "drift_scale": float(corrected[0]) if len(corrected) > 0 else base["drift_scale"],
                    "vol_scale": float(corrected[1]) if len(corrected) > 1 else base["vol_scale"],
                    "skew": base["skew"],
                    "kurtosis": base["kurtosis"],
                }
            except Exception:
                pass

        return base.copy()

    def _generate_paths(self, current_price: float, atr: float,
                         params: Dict, n_sims: int, n_bars: int) -> np.ndarray:
        """Generate price paths using regime-calibrated distributions.

        Uses skew-normal + fat tails (Student-t) for realistic path generation.
        """
        drift = params["drift_scale"] * atr / 5
        vol = params["vol_scale"] * atr / np.sqrt(5)
        skew = params["skew"]

        base_returns = np.random.standard_t(df=4, size=(n_sims, n_bars))
        base_returns = base_returns / np.sqrt(2.5)

        if skew != 0:
            alpha = skew / np.sqrt(1 + skew**2)
            base_returns = base_returns + skew * np.abs(base_returns) * 0.3

        returns = drift + vol * base_returns

        cum_returns = np.cumsum(returns, axis=1)
        paths = current_price + cum_returns

        first_col = np.full((n_sims, 1), current_price)
        paths = np.concatenate([first_col, paths], axis=1)

        return paths
