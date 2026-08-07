"""
CandleEngine — H1 candle-following strategy (see CANDLE_STRATEGY.md).

Follows the candle instead of predicting it:
  - ONE H1 (or 30m) candle at a time.
  - Enter at the candle's commit (after doji/hammer/star decides), in the side
    with the winning load.
  - Ride the full move with no retracement tolerance.
  - Trail once price has moved past the open into profit.
  - Close on reversal/confusion and wait for a fresh candle.

Model: per-pair XGBoost classifier (BUY / SELL / STAND-ASIDE) trained on
profit-based labels. Each bar simulates the exact candle-following trade:
enter at close, SL = SL_ATR*ATR, close on reversal (price retraces
REVERSAL_ATR*ATR past entry), trail once past open, cap at MAX_HOLD_BARS,
subtract COST_R. A bar is BUY/SELL only if its realized R beats the other
side by EDGE_MARGIN (chop => NONE => no trade => no bleed).

Two add-ons:
  1. Jump-candle scan: a candle that leaves its starting point and jumps in
     its full direction (close moved JUMP_BREAK_R*ATR past open, body covers
     >= JUMP_BODY_R of the range) is a higher-conviction entry.
  2. Pair-selection layer: only the top-K pairs by live candle-momentum score
     may fire; each pair's threshold adapts to its own recent score percentile,
     so idle pairs are muted and capital jumps to the pair that is paying.

Labels / features mirror _train_candle_brain.compute_features so live signals
match the backtest exactly.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional

FEATURE_COLS = [
    "rsi", "atr_norm", "bb_pos", "bb_width", "body_ratio", "range_ratio",
    "momentum_z", "close_pos", "direction", "sweep_high", "sweep_low",
    "micro_slope", "volatility_ratio", "trend_strength",
    "open_norm", "high_norm", "low_norm", "close_norm", "return_1",
    "volume", "adx_norm", "vol_regime", "squeeze", "regime_trend",
    "hour_sin", "hour_cos", "day_sin", "session_enc",
]

ATR_PERIOD = 14
TRAIL_ACTIVATE_R = 0.5  # trail once price is 0.5R past the open (in profit)


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_bb(close: pd.Series, period: int = 20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / sma.replace(0, 1e-10)
    pos = (close - lower) / (upper - lower).replace(0, 1e-10)
    return pos.clip(0, 1), width


def compute_features(h1: pd.DataFrame) -> pd.DataFrame:
    """H1 features (all computed on completed candles, no lookahead).

    Mirrors _train_candle_brain.compute_features so live == backtest.
    """
    df = h1.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    rng = (h - l).replace(0, 1e-10)
    body = (c - o).abs()

    df["rsi"] = compute_rsi(c).fillna(50) / 100
    atr = compute_atr(df, ATR_PERIOD).fillna(rng.rolling(ATR_PERIOD).mean().fillna(rng))
    df["atr_norm"] = (atr / c.replace(0, 1e-10)).fillna(0)
    bb_pos, bb_width = compute_bb(c)
    df["bb_pos"] = bb_pos.fillna(0.5)
    df["bb_width"] = bb_width.fillna(0)
    df["body_ratio"] = (body / rng).fillna(0)
    df["range_ratio"] = (rng / atr.replace(0, 1e-10)).fillna(1).clip(0, 5)
    df["momentum_z"] = (c.pct_change(3) / c.pct_change(3).rolling(20).std().replace(0, 1e-10)).fillna(0).clip(-3, 3)
    df["close_pos"] = ((c - l) / rng).fillna(0.5)
    df["direction"] = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(float)
    df["sweep_high"] = ((h.rolling(12).max() - h) / atr.replace(0, 1e-10)).fillna(0).clip(0, 10)
    df["sweep_low"] = ((l - l.rolling(12).min()) / atr.replace(0, 1e-10)).fillna(0).clip(0, 10)
    w = 5
    slope = (2.0 * c + c.shift(1) - c.shift(3) - 2.0 * c.shift(4)) / 10.0
    df["micro_slope"] = ((slope / c.rolling(w).mean().replace(0, 1e-10)).fillna(0).clip(-0.01, 0.01) * 100)

    short_vol = atr.rolling(3).mean()
    long_vol = atr.rolling(24).mean()
    df["volatility_ratio"] = (short_vol / long_vol.replace(0, 1e-10)).fillna(1).clip(0.2, 3)

    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    df["trend_strength"] = ((ema20 - ema50) / atr.replace(0, 1e-10)).fillna(0).clip(-3, 3)

    atr_safe = atr.replace(0, 1e-10)
    df["open_norm"] = ((o - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["high_norm"] = ((h - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["low_norm"] = ((l - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["close_norm"] = ((c - ema50) / atr_safe).fillna(0).clip(-8, 8)

    df["return_1"] = c.pct_change(1).fillna(0).clip(-0.05, 0.05) * 100

    df["volume"] = df["tick_volume"].fillna(0) if "tick_volume" in df.columns else 0
    df["volume"] = (df["volume"] / df["volume"].rolling(20).mean().replace(0, 1)).fillna(1).clip(0, 5)

    # ── Regime features ──────────────────────────────────────────────
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_s = atr.replace(0, 1e-10)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr_s
    minus_di = 100 * minus_dm.rolling(14).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(14).mean()
    df["adx_norm"] = (adx / 60.0).clip(0, 1).fillna(0)

    vol_med = atr.rolling(96).median()
    df["vol_regime"] = (((atr / vol_med.replace(0, 1e-10)) - 0.5).clip(0, 1.5) / 1.5).fillna(0.5)

    bb_avg = bb_width.rolling(96).mean()
    df["squeeze"] = (bb_width < bb_avg * 0.9).astype(float).fillna(0)

    df["regime_trend"] = np.where(
        (ema20 > ema50) & (adx > 20), 1.0,
        np.where((ema20 < ema50) & (adx > 20), -1.0, 0.0),
    ).astype(float)

    # ── Time features ────────────────────────────────────────────────
    if hasattr(df.index, "hour"):
        hours = df.index.hour
        days = df.index.dayofweek
    else:
        hours = pd.Series(12, index=df.index)
        days = pd.Series(0, index=df.index)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["day_sin"] = np.sin(2 * np.pi * days / 7)
    session = pd.Series(1.0, index=df.index)
    session[(hours >= 0) & (hours < 8)] = 0.0
    session[(hours >= 8) & (hours < 14)] = 0.5
    session[(hours >= 14) & (hours < 22)] = 1.0
    session[(hours >= 22)] = 0.0
    df["session_enc"] = session

    return df


def compute_candle_r(
    h1: pd.DataFrame, atr: pd.Series,
    sl_r: float = 1.0, reversal_r: float = 0.5, trail_r: float = 0.5,
    max_hold: int = 24, cost_r: float = 0.05,
) -> tuple:
    """Simulate the candle-following trade on every bar (long and short).

    Long example: enter at close[i]. SL = entry - SL_R*ATR. Once price moves
    TRAIL_ACTIVATE_R*ATR past the open the stop starts trailing at
    peak - TRAIL_R*ATR. If price closes REVERSAL_R*ATR back through entry ->
    exit (reversal). If nothing hits, exit at close[i+max_hold].

    Returns (long_r, short_r) realized-R arrays (R units, cost subtracted).
    """
    c = h1["close"].values.astype(np.float64)
    h = h1["high"].values.astype(np.float64)
    l = h1["low"].values.astype(np.float64)
    a = atr.values.astype(np.float64)
    n = len(c)
    w = int(max_hold)
    act = TRAIL_ACTIVATE_R

    long_r = np.zeros(n, dtype=np.float64)
    short_r = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        ai = a[i] if a[i] > 0 else 1e-9
        e = c[i]

        # ── Long ──
        sl = e - sl_r * ai
        peak = e
        trailing = False
        exit_px = None
        for j in range(i + 1, min(i + 1 + w, n)):
            hj, lj, cj = h[j], l[j], c[j]
            if lj <= sl:
                exit_px = sl
                break
            if not trailing and hj >= e + act * ai:
                trailing = True
            if trailing:
                if hj > peak:
                    peak = hj
                ts = peak - trail_r * ai
                if lj <= ts:
                    exit_px = ts
                    break
            if cj <= e - reversal_r * ai:
                exit_px = cj
                break
        if exit_px is None:
            exit_px = c[min(i + w, n - 1)]
        long_r[i] = (exit_px - e) / ai - cost_r

        # ── Short (mirror) ──
        sl = e + sl_r * ai
        trough = e
        trailing = False
        exit_px = None
        for j in range(i + 1, min(i + 1 + w, n)):
            hj, lj, cj = h[j], l[j], c[j]
            if hj >= sl:
                exit_px = sl
                break
            if not trailing and lj <= e - act * ai:
                trailing = True
            if trailing:
                if lj < trough:
                    trough = lj
                ts = trough + trail_r * ai
                if hj >= ts:
                    exit_px = ts
                    break
            if cj >= e + reversal_r * ai:
                exit_px = cj
                break
        if exit_px is None:
            exit_px = c[min(i + w, n - 1)]
        short_r[i] = (e - exit_px) / ai - cost_r

    return long_r, short_r


def generate_labels(h1: pd.DataFrame, atr: pd.Series, **trade_params) -> pd.DataFrame:
    """Turn realized R into BUY/SELL/NONE labels + confidence + edge columns."""
    sim_keys = {"sl_r", "reversal_r", "trail_r", "max_hold", "cost_r"}
    sim_params = {k: v for k, v in trade_params.items() if k in sim_keys}
    long_r, short_r = compute_candle_r(h1, atr, **sim_params)
    n = len(h1)

    entry_min_r = trade_params.get("entry_min_r", 0.35)
    edge_margin = trade_params.get("edge_margin", 1.0)

    entry_label = np.full(n, 2, dtype=np.int64)   # NONE default
    entry_conf = np.zeros(n, dtype=np.float32)

    buy_ok = (long_r >= entry_min_r) & ((long_r - short_r) >= edge_margin)
    sell_ok = (short_r >= entry_min_r) & ((short_r - long_r) >= edge_margin)

    entry_label[buy_ok] = 0
    entry_label[sell_ok] = 1
    # Confidence relative to a full move (~3R).
    entry_conf[buy_ok] = np.clip(long_r[buy_ok] / 3.0, 0, 1)
    entry_conf[sell_ok] = np.clip(short_r[sell_ok] / 3.0, 0, 1)

    h1["edge_long"] = long_r.astype(np.float32)
    h1["edge_short"] = short_r.astype(np.float32)
    h1["entry_label"] = entry_label
    h1["entry_conf"] = entry_conf
    return h1


def jump_signal(row: pd.Series, atr: float, break_r: float, body_r: float) -> bool:
    """Jump candle: close moved JUMP_BREAK_R*ATR past open AND the body covers
    >= JUMP_BODY_R of the H1 range (a candle that leaves its start point and
    runs in its full direction)."""
    if atr is None or atr <= 0 or "open" not in row or "close" not in row:
        return False
    o = float(row["open"])
    c = float(row["close"])
    h = float(row["high"])
    l = float(row["low"])
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    return (abs(c - o) >= break_r * atr) and (body / rng >= body_r)


def pair_momentum_score(row: pd.Series) -> float:
    """Live candle-momentum score used by the pair-selection layer.

    Magnitude of the committed/jump move: momentum_z * body_ratio * volume,
    normalized to ~[0,1].
    """
    try:
        mz = float(row.get("momentum_z", 0.0))
        br = float(row.get("body_ratio", 0.0))
        vol = float(row.get("volume", 1.0))
        ts = float(row.get("trend_strength", 0.0))
        return float(abs(mz) * br * min(vol, 2.0) * (1.0 + abs(ts) / 3.0) / 6.0)
    except Exception:
        return 0.0


class CandleEngine:
    """Per-pair XGBoost candle-following engine + jump scan + pair selection."""

    CLASSES = ["BUY", "SELL", "STAND_ASIDE"]

    def __init__(
        self,
        pairs: List[str],
        model_dir: str = "models/candle_h1",
        tf: int = 60,
        min_conf: float = 0.60,
        sl_r: float = 1.0, reversal_r: float = 0.5, trail_r: float = 0.5,
        max_hold: int = 24, cost_r: float = 0.05,
        entry_min_r: float = 0.35, edge_margin: float = 1.0,
        jump_enabled: bool = True, jump_break_r: float = 1.0, jump_body_r: float = 0.60,
        pair_top_k: int = 2, pair_score_window: int = 720, pair_pct_min: float = 0.70,
        logger=None,
    ):
        self.pairs = pairs
        self.model_dir = model_dir
        self.tf = int(tf)
        self.min_conf = min_conf
        self.trade_params = dict(
            sl_r=sl_r, reversal_r=reversal_r, trail_r=trail_r,
            max_hold=max_hold, cost_r=cost_r,
            entry_min_r=entry_min_r, edge_margin=edge_margin,
        )
        self.jump_enabled = jump_enabled
        self.jump_break_r = jump_break_r
        self.jump_body_r = jump_body_r
        self.pair_top_k = pair_top_k
        self.pair_score_window = max(int(pair_score_window), 50)
        self.pair_pct_min = pair_pct_min
        self._logger = logger
        self.models: Dict[str, object] = {}
        self._pair_scores: Dict[str, pd.Series] = {}
        self._pair_scores_ts: Dict[str, float] = {}
        self._log("", "CandleEngine initialized for pairs: %s" % ", ".join(pairs))

    def _log(self, sym: str, msg: str):
        if self._logger is not None:
            self._logger.info(f"CandleEngine[{sym}] {msg}".strip())

    # ── model loading ────────────────────────────────────────────────

    def load_models(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        for sym in self.pairs:
            p = os.path.join(self.model_dir, f"{sym}.joblib")
            if os.path.exists(p):
                try:
                    loaded = joblib.load(p)
                    if isinstance(loaded, dict) and "model" in loaded:
                        self.models[sym] = loaded["model"]
                    else:
                        self.models[sym] = loaded
                    self._log(sym, f"loaded model {p}")
                except Exception as e:
                    self._log(sym, f"load failed: {e}")
            else:
                self._log(sym, f"model missing ({p}) — pair muted")

    def has_model(self, sym: str) -> bool:
        return sym in self.models

    # ── feature / score plumbing ─────────────────────────────────────

    def pair_frame(self, sym: str, m1: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample M1 -> H1 (CANDLE_ENGINE_TF), compute features + labels."""
        if m1 is None or len(m1) == 0:
            return None
        idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
        rule = f"{self.tf}min" if self.tf != 60 else "1h"
        h1 = idx.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna()
        if len(h1) < 120:
            return None
        h1 = compute_features(h1)
        return h1

    def _predict(self, sym: str, h1: pd.DataFrame) -> Optional[Dict]:
        """Predict direction + confidence for the LAST completed candle."""
        model = self.models.get(sym)
        if model is None or len(h1) == 0:
            return None
        missing = [c for c in FEATURE_COLS if c not in h1.columns]
        if missing:
            return None
        row = h1[FEATURE_COLS].iloc[[-1]].fillna(0.0).values
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row)[0]
            else:
                import xgboost as xgb
                probs = model.predict(xgb.DMatrix(row))[0]
        except Exception as e:
            self._log(sym, f"predict error: {e}")
            return None
        # Column 0 = BUY, 1 = SELL, 2 = NONE (label 0/1/2 in training).
        pb = float(probs[0])
        ps = float(probs[1])
        pn = float(probs[2]) if len(probs) > 2 else 0.0
        conf = max(pb, ps)
        return {
            "prob_buy": pb,
            "prob_sell": ps,
            "prob_none": pn,
            "conf": float(conf),
            "direction": "BUY" if pb >= ps else "SELL",
            "atr": float(h1["atr_norm"].iloc[-1] * h1["close"].iloc[-1])
                   if "atr_norm" in h1 else 0.0,
            "close": float(h1["close"].iloc[-1]),
            "bar_time": float(h1.index[-1].value / 1e9),
        }

    def _live_score(self, sym: str, h1: pd.DataFrame) -> float:
        """Current pair momentum score + rolling percentile (forward-safe:
        percentile of the LAST score vs the PREVIOUS window only)."""
        if len(h1) < self.pair_score_window + 1:
            return 0.0
        scores = h1.apply(pair_momentum_score, axis=1).values
        cur = float(scores[-1])
        prior = scores[-self.pair_score_window - 1:-1]
        if len(prior) == 0 or cur <= 0:
            return 0.0
        pct = float((prior <= cur).mean())
        if pct >= self.pair_pct_min:
            return cur
        return 0.0

    # ── main entry point ─────────────────────────────────────────────

    def evaluate(
        self, sym: str, m1: pd.DataFrame, now_ts: Optional[float] = None,
    ) -> Optional[Dict]:
        """Evaluate the H1 candle signal for one pair.

        Returns a dict with direction/conf/atr/close/signal_type or None.
        The pair-selection gate (top-K / dynamic percentile) is applied via
        select_pairs(), which the caller runs across all enabled pairs first.
        """
        if sym not in self.pairs:
            return None
        h1 = self.pair_frame(sym, m1)
        if h1 is None:
            self._log(sym, "no H1 frame")
            return None

        pred = self._predict(sym, h1)
        if pred is None:
            return None

        # Jump-candle override: higher conviction when the candle leaves its
        # starting point and runs in its full direction.
        is_jump = False
        if self.jump_enabled and "open" in h1 and "high" in h1:
            last = h1.iloc[-1]
            is_jump = jump_signal(last, pred["atr"], self.jump_break_r, self.jump_body_r)

        conf = pred["conf"]
        if is_jump:
            conf = min(conf + 0.15, 0.99)

        if conf < self.min_conf:
            self._log(
                sym,
                f"low conf {conf:.2f} < {self.min_conf} (pb={pred['prob_buy']:.2f} "
                f"ps={pred['prob_sell']:.2f} jump={is_jump})",
            )
            return None

        # A STAND-ASIDE-leaning prediction that still clears min_conf is only
        # taken if the jump scan confirms full-direction commitment.
        if pred["prob_none"] > max(pred["prob_buy"], pred["prob_sell"]) and not is_jump:
            return None

        return {
            "direction": pred["direction"],
            "conf": conf,
            "score": pair_momentum_score(h1.iloc[-1]),
            "atr": pred["atr"],
            "close": pred["close"],
            "bar_time": pred["bar_time"],
            "is_jump": is_jump,
            "signal_type": "candle_jump" if is_jump else "candle",
            "prob_buy": pred["prob_buy"],
            "prob_sell": pred["prob_sell"],
        }

    def select_pairs(self, m1_by_sym: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Pair-selection layer: which pairs may fire at this tick.

        A pair is allowed only if (a) it has a live candle-momentum score above
        its own rolling percentile threshold AND (b) it is in the top-K pairs by
        raw score. Idle pairs are muted so capital jumps to the pair that pays.
        """
        scores = {}
        for sym in self.pairs:
            h1 = self.pair_frame(sym, m1_by_sym.get(sym))
            if h1 is None:
                scores[sym] = 0.0
                continue
            cur = float(h1.apply(pair_momentum_score, axis=1).iloc[-1])
            w = self.pair_score_window
            if len(h1) > w:
                prior = h1.apply(pair_momentum_score, axis=1).values[-w - 1:-1]
                pct = float((prior <= cur).mean()) if len(prior) and cur > 0 else 0.0
                if pct >= self.pair_pct_min:
                    scores[sym] = cur
                else:
                    scores[sym] = 0.0
            else:
                scores[sym] = cur if cur > 0 else 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        allowed = {sym: (score > 0 and rank < self.pair_top_k)
                   for rank, (sym, score) in enumerate(ranked)}
        self._log(
            "",
            "pair scores: " + ", ".join(f"{s}={v:.3f}{'*' if allowed.get(s) else ''}"
                                        for s, v in ranked),
        )
        return allowed
