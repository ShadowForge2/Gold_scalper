"""Retrain Brain models on REAL generated trades (replaces synthetic training).

Reads the mechanical trade log (data/brain_decisions/decisions.jsonl) plus the
underlying M1 parquets and replays the real FeatureCache at every M5 boundary
during each trade, exactly like the live bot does (bot.py:1261-1282).

Labels come from the actual trade outcome:
  thesis_validator: HOLD=1 if holding to the real exit beat exiting at this
                    boundary, EXIT=0 otherwise.
  meta_fusion:      EXIT=1 if exiting now beat holding to the real exit.
  twin_calibrator:  supervised drift/vol params from the realized path from
                    this boundary to the exit.

This is what lets the Brain learn the user's manual win->retrace->loss
force-exit behavior instead of the synthetic "always valid" thesis.

Usage:
  python _train_brain_real.py --trades data/brain_decisions/decisions.jsonl --output models/brain
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("brain_train_real")

import config as cfg  # noqa: E402
from app.brain.feature_cache import FeatureCache  # noqa: E402
from app.brain.perception import PerceptionEngine  # noqa: E402
from app.brain.thesis_validator import ThesisValidator, THESIS_FEATURE_COLS  # noqa: E402
from app.brain.opportunity import OpportunityScanner  # noqa: E402
from app.brain.digital_twin import DigitalTwin, TWIN_INPUT_COLS  # noqa: E402
from app.brain.meta_fusion import MetaFusion, META_FEATURE_COLS, REGIME_ENCODING  # noqa: E402
from _train_brain import BrainTrainer, TWIN_TARGET_COLS  # noqa: E402

SYMBOL_DATA_DIRS = {
    "XAUUSD": "data/dukascopy",
    "US100": "data/dukascopy_us100",
    "US500": "data/dukascopy_us500",
    "US30": "data/dukascopy_us30",
}
M1_HIST = 500          # bars fed to the runtime cache (CANDLE_ML_M1_HISTORY_BARS)
BOUNDARY_SEC = 300     # M5 boundary
WARM_MINUTES = 1500    # warm-up history before each trade
WARM_STEP_BOUNDARIES = 2
TWIN_SIMS = 250        # faster sims for training features (serve uses cfg value)
HOLD_EPS_ATR = 0.02    # "holding beat exiting" tolerance (ATR units)
MAX_SAMPLES_PER_TRADE = 20


def load_bars(symbol: str) -> pd.DataFrame:
    data_dir = SYMBOL_DATA_DIRS.get(symbol, "data/dukascopy")
    frames = []
    for y in range(2020, 2027):
        path = os.path.join(data_dir, f"{symbol}_M1_{y}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            log.warning("  skip %s: %s", path, e)
            continue
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.sort_values("time").drop_duplicates(subset="time").set_index("time")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="first")].sort_index()
        frames.append(df)
    if not frames:
        sys.exit(f"no M1 data for {symbol}")
    return pd.concat(frames).sort_index()


def load_trades(path: str) -> List[Dict]:
    trades = []
    if not os.path.exists(path):
        sys.exit(f"trade file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["entry_ts"] = pd.Timestamp(rec.get("entry_time"), utc=True)
            rec["exit_ts"] = pd.Timestamp(rec.get("exit_time"), utc=True)
            rec["_atr_entry"] = float(rec.get("entry_features", {}).get("atr_current", 1.0) or 1.0)
            trades.append(rec)
    return trades


def _boundary_of(ts) -> int:
    return int(int(ts.timestamp()) // BOUNDARY_SEC * BOUNDARY_SEC)


class SampleExtractor:
    """Replays the real FeatureCache + cascade for every boundary of a trade."""

    def __init__(self, symbol: str, n_sims: int = TWIN_SIMS):
        self.symbol = symbol
        self.cache = FeatureCache()
        self.cache.init_symbol(symbol)
        self.perception = PerceptionEngine(self.cache)
        self.thesis_rules = ThesisValidator(model_path=None)
        self.opportunity = OpportunityScanner(self.cache, [symbol])
        self.twin = DigitalTwin(calibrator_path=None, n_sims=n_sims)

    def _warmup(self, bars: pd.DataFrame, entry_boundary: int):
        warm_start = entry_boundary - WARM_MINUTES * 60
        b = warm_start - WARM_STEP_BOUNDARIES * BOUNDARY_SEC
        while b < entry_boundary:
            b += WARM_STEP_BOUNDARIES * BOUNDARY_SEC
            window = bars[(bars.index >= warm_start - 2 * 3600) & (bars.index < b + BOUNDARY_SEC)]
            if len(window) < 50:
                continue
            window = window.tail(M1_HIST)
            self.cache.update_tick(self.symbol, window, float(window["close"].iloc[-1]),
                                   None, force=True)

    def extract(self, trade: Dict, bars: pd.DataFrame) -> Optional[List[Dict]]:
        entry_ts = trade["entry_ts"]
        exit_ts = trade["exit_ts"]
        if pd.isna(entry_ts) or pd.isna(exit_ts):
            return None
        entry_boundary = _boundary_of(entry_ts)
        exit_boundary = int((int(exit_ts.timestamp()) + 1) // BOUNDARY_SEC * BOUNDARY_SEC)
        if exit_boundary < entry_boundary + BOUNDARY_SEC:
            return None

        direction = trade.get("direction", "BUY")
        sign = 1.0 if direction == "BUY" else -1.0
        entry_px = float(trade.get("entry_price", 0))
        exit_px = float(trade.get("exit_price", 0))
        entry_atr = trade["_atr_entry"]
        tp_px = entry_px + sign * entry_atr * cfg.TP1_MULTIPLIER
        final_pnl = (exit_px - entry_px) * sign

        self._warmup(bars, entry_boundary)

        entry_info = {
            "entry_price": entry_px,
            "direction": direction,
            "sl": 0,
            "tp1": tp_px,
            "atr": entry_atr,
            "ml_confidence": float(trade.get("ml_confidence", 0.5)),
            "adx_at_entry": float(trade.get("adx_at_entry", 0)),
            "regime_at_entry": trade.get("regime_at_entry", "UNKNOWN"),
            "bias_agreement": 0.5,
            "event_start": None,
        }

        samples = []
        cur = entry_boundary + BOUNDARY_SEC
        warm_start = entry_boundary - WARM_MINUTES * 60
        while cur <= exit_boundary and len(samples) < MAX_SAMPLES_PER_TRADE:
            window = bars[(bars.index >= warm_start) & (bars.index < cur + BOUNDARY_SEC)]
            if len(window) >= 50:
                window = window.tail(M1_HIST)
                close = float(window["close"].iloc[-1])
                self.cache.update_tick(self.symbol, window, close, entry_info, force=True)
                feats = self.cache.get_feature_dict(self.symbol)
                feats["bars_since_entry"] = (cur - entry_boundary) / BOUNDARY_SEC

                cur_pnl = (close - entry_px) * sign
                atr = float(feats.get("atr_current", 1.0) or 1.0)
                if atr <= 0:
                    atr = 1.0
                hold_better = (final_pnl - cur_pnl) / atr >= -HOLD_EPS_ATR

                thesis_vec = np.array([float(feats.get(c, 0.0) or 0.0)
                                       for c in THESIS_FEATURE_COLS], dtype=np.float32)

                perc = self.perception.perceive(self.symbol)
                rule = self.thesis_rules._rule_based_evaluate(self.symbol, entry_info, feats, perc)
                op = self.opportunity.scan(self.symbol, feats, perc.get("regime", "UNKNOWN"))

                n_rem = max(1, (exit_boundary - cur) // BOUNDARY_SEC + 1)
                fav_delta = max(0.0, (final_pnl - cur_pnl) / atr / n_rem)
                drift_scale = float(np.clip(fav_delta * 5.0, 0.02, 2.5))
                vol_scale = float(np.clip(abs(final_pnl - cur_pnl) / atr / np.sqrt(n_rem) * np.sqrt(5.0),
                                          0.02, 3.0))
                regime_enc = float(REGIME_ENCODING.get(perc.get("regime", "UNKNOWN"), 0.0))

                dt = self.twin.simulate(
                    current_price=close,
                    atr=atr,
                    direction=direction,
                    entry_price=entry_px,
                    current_pnl=cur_pnl / atr,
                    regime=perc.get("regime", "UNKNOWN"),
                    minutes_held=(cur - entry_boundary) / 60.0,
                    opportunity_data=op.get("best_opportunity"),
                    features=feats,
                    n_sims=TWIN_SIMS,
                )

                samples.append({
                    "thesis_vec": thesis_vec,
                    "twin_input": np.array([feats.get("atr_current", atr),
                                            regime_enc,
                                            feats.get("volatility_percentile", 0.5),
                                            feats.get("adx_current", 0)], dtype=np.float32),
                    "twin_targets": np.array([drift_scale, vol_scale], dtype=np.float32),
                    "rule_validity": float(rule.get("validity", 50)),
                    "perception_composite": float(perc.get("composite_score", 50)),
                    "regime_enc": regime_enc,
                    "opportunity_cost": float(op.get("opportunity_cost", 0)),
                    "opportunity_count": float(op.get("opportunity_count", 0)),
                    "dt_ev_diff": float(dt.get("ev_diff", 0)),
                    "dt_hold_win_pct": float(dt.get("hold_win_pct", 0.5)),
                    "dt_confidence": float(dt.get("confidence", 0.5)),
                    "pnl_atr_ratio": float(feats.get("pnl_atr_ratio", 0)),
                    "ml_confidence_current": float(feats.get("ml_confidence_current", 0.5)),
                    "adx_current": float(feats.get("adx_current", 0)),
                    "momentum_5m": float(feats.get("momentum_5m", 0)),
                    "volatility_percentile": float(feats.get("volatility_percentile", 0.5)),
                    "bars_since_entry": float(feats.get("bars_since_entry", 0)),
                    "session_active": float(feats.get("session_active", 3)),
                    "is_friday": float(feats.get("is_friday", 0)),
                    "label_hold": 1 if hold_better else 0,
                    "symbol": self.symbol,
                })
            cur += BOUNDARY_SEC
        return samples


def build_datasets(trades: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    """Replay all trades and collect per-model datasets.

    Returns (thesis, meta_pre, twin): each maps sample-index -> dict, so the
    meta rows can be rebuilt with the freshly trained thesis model.
    """
    by_sym: Dict[str, List[Dict]] = {}
    for t in trades:
        by_sym.setdefault(t.get("symbol", "XAUUSD"), []).append(t)

    thesis = {}
    meta_pre = {}
    twin = {}
    count = 0
    for symbol, sym_trades in sorted(by_sym.items()):
        log.info("  Replaying %d %s trades", len(sym_trades), symbol)
        bars = load_bars(symbol)
        extractor = SampleExtractor(symbol)
        for i, t in enumerate(sym_trades):
            t0 = time.time()
            samples = extractor.extract(t, bars)
            if not samples:
                continue
            for s in samples:
                thesis[count] = {"vec": s["thesis_vec"], "label": s["label_hold"]}
                twin[count] = {"input": s["twin_input"], "targets": s["twin_targets"]}
                meta_pre[count] = s
                count += 1
            if (i + 1) % 50 == 0 or i == len(sym_trades) - 1:
                log.info("    %d/%d trades, %d samples (%.1fs)",
                         i + 1, len(sym_trades), len(thesis), time.time() - t0)
    return thesis, meta_pre, twin


def build_meta_rows(meta_pre: Dict, thesis_model) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    ids = sorted(meta_pre.keys())
    thesis_vectors = np.array([meta_pre[i]["thesis_vec"] for i in ids], dtype=np.float32)
    if hasattr(thesis_model, "predict_proba"):
        ml_validity = thesis_model.predict_proba(thesis_vectors)[:, 1] * 100.0
    else:
        ml_validity = np.full(len(ids), 0.0)
    for k, (i, mv) in enumerate(zip(ids, ml_validity)):
        s = meta_pre[i]
        rule_v = s["rule_validity"]
        validity = 0.4 * rule_v + 0.6 * float(mv) if mv > 0 else rule_v
        vals = [
            validity / 100.0,
            0.8,  # thesis_confidence (ML path) — matches runtime combined conf
            s["perception_composite"] / 100.0,
            s["regime_enc"],
            s["opportunity_cost"],
            s["opportunity_count"] / 10.0,
            s["dt_ev_diff"],
            s["dt_hold_win_pct"],
            s["dt_confidence"],
            s["pnl_atr_ratio"],
            s["ml_confidence_current"],
            s["adx_current"] / 50.0,
            s["momentum_5m"],
            s["volatility_percentile"],
            s["bars_since_entry"] / 24.0,
            s["session_active"] / 3.0,
            s["is_friday"],
            0.5,  # recent_win_rate_20 — not computed at runtime
            0.0,  # consecutive_losses / 5
            0.0,  # portfolio_heat
        ]
        X.append(vals)
        y.append(0 if s["label_hold"] else 1)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="data/brain_decisions/decisions.jsonl")
    ap.add_argument("--output", "-o", default="models/brain")
    ap.add_argument("--sims", type=int, default=TWIN_SIMS)
    args = ap.parse_args()

    t0 = time.time()
    log.info("=" * 70)
    log.info("BRAIN RETRAIN — REAL TRADES")
    log.info("=" * 70)
    log.info("Loading trades from %s", args.trades)
    trades = load_trades(args.trades)
    by_sym = {}
    for t in trades:
        by_sym[t.get("symbol", "XAUUSD")] = by_sym.get(t.get("symbol", "XAUUSD"), 0) + 1
    log.info("  %d trades: %s", len(trades), {k: v for k, v in sorted(by_sym.items())})
    if len(trades) < 100:
        log.warning("  Only %d trades — consider generating more before trusting the models", len(trades))

    log.info("Replaying trades through the real FeatureCache...")
    thesis, meta_pre, twin = build_datasets(trades)
    n = len(thesis)
    if n < 300:
        log.warning("  Only %d boundary samples. Retrain anyway? (Ctrl-C to abort)", n)
        time.sleep(2)
    log.info("Collected %d boundary samples", n)

    X_th = np.array([thesis[i]["vec"] for i in sorted(thesis)], dtype=np.float32)
    y_th = np.array([thesis[i]["label"] for i in sorted(thesis)], dtype=np.int8)
    log.info("  thesis: %d samples, %.1f%% HOLD labels",
             len(y_th), 100 * float(y_th.mean()))

    trainer = BrainTrainer(output_dir=args.output, n_samples=n)
    trainer.metadata["source"] = "real_m1_trades"
    trainer.metadata["n_samples"] = n
    trainer.metadata["trades_by_symbol"] = by_sym

    log.info("")
    log.info("--- Phase 1: Thesis Validator (real outcomes) ---")
    th_result = trainer.train_thesis_validator(X_th, y_th)
    trainer.metadata["models"]["thesis_validator"].update({
        "source": "real_m1_trades", "n_samples": n,
    })

    log.info("")
    log.info("--- Phase 2: Meta-Fusion Ensemble (real outcomes) ---")
    X_meta, y_meta = build_meta_rows(meta_pre, th_result["model"])
    log.info("  meta: %d samples, %.1f%% EXIT labels",
             len(y_meta), 100 * float(y_meta.mean()))
    meta_results = {}
    for key, fn in (("xgb", trainer.train_meta_xgb),
                    ("nn", trainer.train_meta_nn),
                    ("rf", trainer.train_meta_rf)):
        try:
            meta_results[key] = fn(X_meta, y_meta)
        except ImportError as exc:
            log.warning("Skipping meta_fusion_%s (dependency missing: %s)", key, exc)
        except Exception as exc:
            log.warning("Skipping meta_fusion_%s (training failed: %s)", key, exc)
    for key, res in meta_results.items():
        trainer.metadata["models"][f"meta_fusion_{key}"].update({
            "source": "real_m1_trades", "n_samples": len(y_meta),
        })

    log.info("")
    log.info("--- Phase 3: Twin Calibrator (realized drift/vol) ---")
    Xt = np.array([twin[i]["input"] for i in sorted(twin)], dtype=np.float32)
    yt = np.array([twin[i]["targets"] for i in sorted(twin)], dtype=np.float32)
    twin_result = trainer.train_twin_calibrator(Xt, yt)
    trainer.metadata["models"]["twin_calibrator"].update({
        "source": "real_m1_trades", "n_samples": len(yt),
    })

    trainer.metadata["total_train_time"] = sum(
        r.get("train_time", 0)
        for r in [th_result, *meta_results.values(), twin_result]
    )
    trainer.metadata["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(args.output, "training_metadata.json"), "w") as f:
        json.dump(trainer.metadata, f, indent=2, default=str)
    log.info("Saved metadata → %s/training_metadata.json", args.output)

    log.info("")
    log.info("--- Phase 4: Smoke test loaded Brain ---")
    from app.brain.orchestrator import BrainOrchestrator
    brain = BrainOrchestrator(symbols=["XAUUSD", "US100", "US500", "US30"])
    dummy = brain.evaluate(symbol="XAUUSD", current_pnl=0.3, current_price=3000.0,
                           entry_price=2995.0, direction="BUY", atr=5.0,
                           minutes_held=10.0)
    log.info("  smoke evaluate: action=%s conf=%.2f path=%s layers=%d reason=%s",
             dummy.get("action"), dummy.get("confidence", 0),
             dummy.get("cascade_path"), dummy.get("layers_evaluated", 0),
             dummy.get("reason"))

    log.info("")
    log.info("DONE in %.1fs — %d samples, output → %s", time.time() - t0, n, args.output)


if __name__ == "__main__":
    main()
