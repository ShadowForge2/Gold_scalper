"""
MomentumEngine — live port of the validated US100 momentum-jump strategy
(_two_engine.us100_jump_trades, OOS-validated on M5 2025 data, US100 only).

Entry (long example): a completed M5 candle closes as a bullish surge bar —
  open < close, body_ratio >= 0.60, trend_strength >= 0.50, momentum_z >= 2.0 —
  while price sits above its long-term EMA480 (trend regime). The short is the
  mirror image (open > close, trend_strength <= -0.50, momentum_z <= -2.0)
  below the EMA480.

Exit (managed by the bot, mirrors the backtest):
  - SL 1 x ATR from the fill
  - jump target: after the best favorable excursion reaches 1.0R, exit when
    the close retraces 0.25R from the peak
  - max hold 12 M5 bars (60 min) -> exit at market

Feature formulas are identical to _train_candle_brain.compute_features so the
live signal matches the backtest exactly.
"""
import time
from typing import Optional, Dict

import numpy as np
import pandas as pd


class MomentumEngine:
    def __init__(self, mz_min: float = 2.0, body_min: float = 0.60,
                 ts_min: float = 0.50, ema_span: int = 480,
                 sl_r: float = 1.0, jump_target: float = 1.0,
                 retr_r: float = 0.25, max_hold: int = 12,
                 atr_period: int = 14, logger=None,
                 gate: str = "none", gate_threshold: float = 0.55,
                 gate_window: int = 96):
        self.mz_min = mz_min
        self.body_min = body_min
        self.ts_min = ts_min
        self.ema_span = int(ema_span)
        self.sl_r = sl_r
        self.jump_target = jump_target
        self.retr_r = retr_r
        self.max_hold = max_hold
        self.atr_period = atr_period
        self.gate = (gate or "none").lower()
        self.gate_threshold = gate_threshold
        self.gate_window = max(int(gate_window), 2)
        self._logger = logger
        self._last_warmup_log = 0.0

    # ── helpers (formulas mirror _train_candle_brain.py) ────────────────

    @staticmethod
    def resample_m5(m1: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
            m5 = idx.resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last",
            }).dropna()
            return m5 if len(m5) else None
        except Exception:
            return None

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def compute_features(self, m5: pd.DataFrame) -> pd.DataFrame:
        df = m5.copy()
        c, h, l, o = df["close"], df["high"], df["low"], df["open"]
        rng = (h - l).replace(0, 1e-10)
        body = (c - o).abs()
        atr = self.compute_atr(df).fillna(
            rng.rolling(self.atr_period).mean().fillna(rng))
        df["atr"] = atr
        df["body_ratio"] = (body / rng).fillna(0)
        df["momentum_z"] = (
            c.pct_change(3)
            / c.pct_change(3).rolling(20).std().replace(0, 1e-10)
        ).fillna(0).clip(-3, 3)
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        df["trend_strength"] = ((ema20 - ema50) / atr.replace(0, 1e-10)).fillna(0).clip(-3, 3)
        return df

    # ── regime gate (mirrors _mom_regime_scan.build_gates) ─────────────

    def gate_allows(self, f: pd.DataFrame, i: int) -> bool:
        """Forward-safe regime gate for the bar at index `i`.

        "vol": current ATR is in the top (1 - threshold) of the previous
               gate_window ATRs (high-volatility only). Percentile computed
               over the prior window ONLY (current bar excluded), so a quiet
               market can never satisfy it.
        "er":  Kaufman efficiency ratio |Pn| / sum|P| over the last
               gate_window bars >= threshold (trending only).
        """
        if self.gate == "none":
            return True
        w = self.gate_window
        if i < w:
            return False
        if self.gate == "vol":
            atr_vals = f["atr"].values
            cur = atr_vals[i]
            prior = atr_vals[i - w:i]
            if len(prior) == 0 or cur <= 0:
                return False
            pct = float((prior <= cur).mean())
            return pct >= self.gate_threshold
        if self.gate == "er":
            c = f["close"]
            net = abs(c.iloc[i] - c.iloc[i - w])
            gross = c.diff().abs().iloc[i - w + 1:i + 1].sum()
            if gross <= 0:
                return False
            return (net / gross) >= self.gate_threshold
        return True

    # ── signal detection ────────────────────────────────────────────────

    def detect(self, m1: pd.DataFrame, now_ts: Optional[float] = None) -> Optional[Dict]:
        """Evaluate the jump signal on the last completed M5 candle.

        Returns a signal dict (direction/score/atr/close/bar_time) or None.
        `now_ts` is a UTC epoch; only candles whose 5-min bucket has ended are
        considered, and candles older than two buckets are ignored (stale data).
        """
        m5 = self.resample_m5(m1)
        if m5 is None or len(m5) < self.ema_span + 5:
            # EMA480 warmup not satisfied — this is a *silent* no-signal unless
            # we log it, and a too-short M1 window is the usual cause live.
            if self._logger is not None:
                now = time.time()
                if now - self._last_warmup_log > 300:
                    self._last_warmup_log = now
                    have = len(m5) if m5 is not None else 0
                    self._logger.info(
                        f"Momentum warmup: {have}/{self.ema_span + 5} M5 bars "
                        f"(need ~{(self.ema_span + 5) * 5} M1 bars of market data) — no signal"
                    )
            return None

        f = self.compute_features(m5)
        c = f["close"].values
        o = f["open"].values
        atr = f["atr"].values
        br = f["body_ratio"].values
        ts = f["trend_strength"].values
        mz = f["momentum_z"].values
        n = len(f)

        closes = pd.Series(c, index=f.index)
        ema = closes.ewm(span=self.ema_span, adjust=False).mean().values
        regime_up = c > ema

        now_ts = now_ts if now_ts is not None else time.time()
        now_floor = int(now_ts) - (int(now_ts) % 300)
        starts = f.index.values.astype("datetime64[s]").astype("int64")
        ends = (f.index + pd.Timedelta(minutes=5)).values.astype("datetime64[s]").astype("int64")
        formed = np.where((ends <= now_floor) & (starts >= now_floor - 600))[0]
        if len(formed) == 0:
            return None
        i = int(formed[-1])
        if i <= self.ema_span:  # warmup — matches backtest idx > max(50, ema)
            return None

        if (o[i] < c[i] and br[i] >= self.body_min and ts[i] >= self.ts_min
                and mz[i] >= self.mz_min and regime_up[i]):
            direction = "BUY"
        elif (o[i] > c[i] and br[i] >= self.body_min and ts[i] <= -self.ts_min
              and mz[i] <= -self.mz_min and not regime_up[i]):
            direction = "SELL"
        else:
            return None

        # Regime gate — skip quiet/chop regimes (per-pair, validated per symbol).
        if not self.gate_allows(f, i):
            return None

        atr_i = float(atr[i]) if atr[i] > 0 else 1e-10
        return {
            "direction": direction,
            "score": float(abs(mz[i]) / 2.0),
            "atr": atr_i,
            "close": float(c[i]),
            "bar_time": float((f.index[i] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)),
            "signal_type": "momentum",
        }
