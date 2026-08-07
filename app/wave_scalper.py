"""
WaveScalper — live, stateful, incremental intra-candle M1 wave scalper.

Replicates `_sweep_candle_wave.run_candle_wave` EXACTLY (strict engine:
entry bar skipped for exits, peak updated only after exit checks, base = exit
price after each exit, jump-rider mode) as a persistent per-symbol state
machine so the bot can scalp each micro-wave of the forming H1 candle live.

The XGBoost model is ONLY a chop gate: if the PREVIOUS completed H1 candle was
NONE-leaning AND not a jump candle, sit out the whole forming candle. No
direction veto, no confidence floor.

Live fill handling:
  - The engine emits an ENTER action when a wave trigger fires on a completed
    M1 bar. The bot opens at market and calls confirm_entry(fill_price) with
    the ACTUAL fill, so cut/lock/trail are anchored to the real entry (the
    same fill-honest convention the momentum engine uses). cancel_entry()
    restores flat if the order is blocked.
  - Exits (cut/lock/rider-trail/candle-end) are emitted as pending actions;
    the bot closes at market and calls confirm_exit(). Until confirmed, the
    action is returned on every feed() so a failed close retries next tick.
"""

import os
import time

import numpy as np
import pandas as pd

from app.candle_engine import FEATURE_COLS, compute_features, compute_atr

ATR_PERIOD = 14


def _epoch(dt):
    return float(pd.Timestamp(dt).value) / 1e9


class WaveScalper:
    def __init__(
        self,
        symbol: str,
        model=None,
        feature_cols=None,
        logger=None,
        entry_r: float = 0.50,
        cut_r: float = 0.03,
        profit_r: float = 0.05,
        cost_r: float = 0.05,
        jump_break_r: float = 1.5,
        jump_body_r: float = 0.70,
        trail_r: float = 0.5,
        reversal_r: float = 0.5,
        rider_enabled: bool = True,
        gate_min_h1_bars: int = 60,
    ):
        self.symbol = symbol
        self.model = model
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self._logger = logger
        self.entry_r = float(entry_r)
        self.cut_r = float(cut_r)
        self.profit_r = float(profit_r)
        self.cost_r = float(cost_r)
        self.jump_break_r = float(jump_break_r)
        self.jump_body_r = float(jump_body_r)
        self.trail_r = float(trail_r)
        self.reversal_r = float(reversal_r)  # accepted for config parity (unused in strict engine)
        self.rider_enabled = bool(rider_enabled)
        self.gate_min_h1_bars = int(gate_min_h1_bars)

        # M1 cache (indexed by naive UTC datetime, sorted, deduped).
        self._cache = None
        self._warm = False
        # Bound the cache so a multi-month live run can't grow memory/CPU
        # without bound. 14 ATR bars + feature windows converge in ~60 H1 bars
        # (3600 M1 bars), so a few weeks of bars is far more than enough.
        self._max_cache_bars = int(
            os.environ.get("WAVE_MAX_CACHE_BARS", "20000"))

        # Wave state (mirrors run_candle_wave).
        self._candle_ts = None      # forming candle start (pd.Timestamp)
        self._candle_open = 0.0
        self._gate_ok = True
        self._atr = 0.0
        self._base = 0.0
        self._pos = 0               # 1 long / -1 short / 0 flat
        self._entry = 0.0
        self._peak = 0.0
        self._rider = False
        self._jump_dir = 0
        self._last_ts = None        # last fully processed completed M1 bar
        self._last_close = None     # close of the last processed completed bar
        self._pending = None        # {"type": "enter"|"exit", ...} awaiting confirm

    def _log(self, msg):
        if self._logger is not None:
            try:
                self._logger.info(f"Wave[{self.symbol}] {msg}")
            except Exception:
                pass

    # ── data plumbing ────────────────────────────────────────────────

    def set_history(self, df: pd.DataFrame) -> None:
        """Merge an M1 frame into the cache (dedupe by index)."""
        idx = self._normalize(df)
        if idx is None or len(idx) == 0:
            return
        if self._cache is None:
            self._cache = idx
        else:
            self._cache = pd.concat([self._cache, idx])
        self._cache = self._cache[~self._cache.index.duplicated(keep="last")].sort_index()
        if len(self._cache) > self._max_cache_bars:
            self._cache = self._cache.iloc[-self._max_cache_bars:]

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return None
        idx = df.set_index("time") if "time" in df.columns else df.copy()
        for col in ("open", "high", "low", "close"):
            if col not in idx.columns:
                return None
            idx[col] = pd.to_numeric(idx[col], errors="coerce")
        if "tick_volume" not in idx.columns:
            idx["tick_volume"] = 0
        if getattr(idx.index, "tz", None) is not None:
            idx.index = idx.index.tz_convert("UTC").tz_localize(None)
        idx = idx[~idx.index.duplicated(keep="last")].sort_index()
        return idx

    def _resample_h1(self):
        cache = self._cache
        if cache is None or len(cache) == 0:
            return None
        h1 = cache.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna()
        if len(h1) < 2:
            return None
        return h1

    # ── chop gate ────────────────────────────────────────────────────

    def _jump_flag(self, row: pd.Series, atr: float) -> bool:
        if atr is None or atr <= 0:
            return False
        o = float(row["open"]); c = float(row["close"])
        h = float(row["high"]); l = float(row["low"])
        rng = h - l
        if rng <= 0:
            return False
        body = abs(c - o)
        return (body >= self.jump_break_r * atr) and (body / rng >= self.jump_body_r)

    def _predict(self, h1: pd.DataFrame, idx: int):
        """Model probs (pb, ps, pn) for completed candle row `idx`."""
        if self.model is None:
            return None
        missing = [c for c in self.feature_cols if c not in h1.columns]
        if missing:
            return None
        row = h1[self.feature_cols].iloc[[idx]].fillna(0.0).values.astype(np.float32)
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(row)[0]
            else:
                import xgboost as xgb
                probs = self.model.predict(xgb.DMatrix(row))[0]
        except Exception as e:
            self._log(f"predict error: {e}")
            return None
        probs = np.asarray(probs, dtype=float)
        if len(probs) < 3:
            return None
        return float(probs[0]), float(probs[1]), float(probs[2])

    def _compute_gate(self, prev_idx: int):
        """Gate + ATR for the forming candle from the previous completed H1 row.

        Sets _gate_ok, _atr, _candle_open. Mirrors the sweep's per-candle loop.
        """
        h1 = self._resample_h1()
        if h1 is None or prev_idx < 0 or prev_idx >= len(h1):
            self._gate_ok = True
            return
        atr_ser = compute_atr(h1, ATR_PERIOD)
        prev_atr = float(atr_ser.iloc[prev_idx])
        self._atr = prev_atr if (prev_atr > 0 and not np.isnan(prev_atr)) else 0.0
        self._candle_open = float(h1["open"].iloc[prev_idx + 1])

        if self.model is None:
            self._gate_ok = True
            return
        if len(h1) < self.gate_min_h1_bars:
            self._gate_ok = True
            return
        try:
            feats = compute_features(h1)
        except Exception as e:
            self._log(f"feature compute failed: {e}")
            self._gate_ok = True
            return
        if prev_idx >= len(feats):
            self._gate_ok = True
            return
        pred = self._predict(feats, prev_idx)
        if pred is None:
            self._gate_ok = True
            return
        pb, ps, pn = pred
        is_jump = self._jump_flag(feats.iloc[prev_idx], self._atr)
        self._gate_ok = not (pn > max(pb, ps) and not is_jump)

    def _start_candle(self, ts: pd.Timestamp):
        """Begin a new forming candle: compute gate/atr/open from the previous
        completed candle and reset the wave state."""
        self._candle_ts = ts
        h1 = self._resample_h1()
        if h1 is None or ts not in h1.index:
            # Forming candle has no bars yet (or no data): stay flat this candle.
            self._gate_ok = True
            self._atr = 0.0
            self._candle_open = 0.0
        else:
            form_pos = h1.index.get_loc(ts)
            self._compute_gate(form_pos - 1)
        self._base = self._candle_open
        self._pos = 0
        self._entry = 0.0
        self._peak = 0.0
        self._rider = False
        self._jump_dir = 0
        self._pending = None

    # ── warm-up / adopt ──────────────────────────────────────────────

    def warm_up(self, now_ts: float) -> None:
        """Initialize state at the current forming candle (no replay)."""
        if self._cache is None or len(self._cache) < 3:
            return
        now = pd.Timestamp(now_ts, unit="s")
        current_minute = now.floor("min")
        completed = self._cache[self._cache.index < current_minute]
        if len(completed) == 0:
            return
        self._last_ts = completed.index[-1]
        self._last_close = float(completed["close"].iloc[-1])
        self._start_candle(now.floor("h"))
        self._warm = True
        self._log(
            f"warmed up at candle {self._candle_ts} gate_ok={self._gate_ok} "
            f"atr={self._atr:.4f} open={self._candle_open:.2f} "
            f"({len(self._cache)} M1 bars cached)"
        )

    def adopt_position(self, direction: str, entry_price: float, atr: float = 0.0) -> None:
        """Reconstruct wave state after a bot restart with an open position."""
        self._pos = 1 if str(direction).upper() == "BUY" else -1
        self._entry = float(entry_price)
        self._peak = float(entry_price)
        self._base = float(entry_price)
        self._rider = False
        self._jump_dir = 0
        self._pending = None
        if atr and atr > 0:
            self._atr = float(atr)
        self._candle_ts = pd.Timestamp(time.time(), unit="s").floor("h")
        self._log(f"adopted {direction} @ {entry_price:.2f} (atr={self._atr:.4f})")

    # ── core feed ────────────────────────────────────────────────────

    def feed(self, recent_df: pd.DataFrame, now_ts: float):
        """Process newly completed M1 bars; return an action dict or None.

        Actions: {"type":"enter","direction":..,"entry":..} or
                 {"type":"exit","reason":..,"price":..}.
        A pending action (awaiting confirm) is returned on every call so a
        failed broker close/entry retries without reprocessing bars.
        """
        self.set_history(recent_df)
        if self._pending is not None:
            return dict(self._pending)

        if not self._warm:
            self.warm_up(now_ts)
            if not self._warm:
                return None

        now = pd.Timestamp(now_ts, unit="s")
        current_minute = now.floor("min")
        cache = self._cache
        last = self._last_ts
        if last is None:
            completed = cache[cache.index < current_minute]
            if len(completed) == 0:
                return None
            self._last_ts = completed.index[-1]
            return None

        new = cache[(cache.index > last) & (cache.index < current_minute)]
        if len(new) == 0:
            return None

        for ts, row in new.iterrows():
            bo = float(row["open"]); bh = float(row["high"])
            bl = float(row["low"]); bc = float(row["close"])

            # Candle boundary rollover (force-close any open wave at the OLD
            # candle's final close, matching the backtest's end-of-candle exit).
            h1_floor = ts.floor("h")
            if self._candle_ts is None or h1_floor != self._candle_ts:
                if self._pos != 0:
                    # Hold the rollover bar back: it is the first bar of the NEW
                    # candle and must not be consumed by the exit detection (the
                    # backtest force-closes the old candle at its last close, then
                    # lets the new candle start on this same bar).
                    exit_px = self._last_close if self._last_close is not None else bc
                    self._pending = {"type": "exit", "reason": "candle_end", "price": exit_px}
                    return dict(self._pending)
                self._start_candle(h1_floor)
                # fall through: the rollover bar trades under the new candle

            self._last_ts = ts
            if not self._gate_ok:
                self._last_close = bc
                continue

            action = self._step(bo, bh, bl, bc)
            self._last_close = bc
            if action is not None:
                self._pending = action
                return dict(action)
        return None

    def _step(self, bo, bh, bl, bc):
        """One M1 bar of the strict wave state machine (mirrors run_candle_wave)."""
        atr = self._atr
        if atr <= 0:
            return None

        if self._pos == 0 and not self._rider:
            if bh >= self._base + self.entry_r * atr:
                return {"type": "enter", "direction": "BUY", "entry": self._base + self.entry_r * atr}
            if bl <= self._base - self.entry_r * atr:
                return {"type": "enter", "direction": "SELL", "entry": self._base - self.entry_r * atr}
            if self.rider_enabled:
                if (bh - self._candle_open) >= self.jump_break_r * atr and (bh - bl) > 0:
                    body = bh - self._candle_open
                    if body / (bh - bl) >= self.jump_body_r:
                        return {"type": "enter", "direction": "BUY", "entry": self._candle_open, "rider": True}
                if (self._candle_open - bl) >= self.jump_break_r * atr and (bh - bl) > 0:
                    body = self._candle_open - bl
                    if body / (bh - bl) >= self.jump_body_r:
                        return {"type": "enter", "direction": "SELL", "entry": self._candle_open, "rider": True}
            return None

        if self._pos == 1:
            stop = self._entry - self.cut_r * atr
            lock = self._peak - self.profit_r * atr
            if bl <= stop:
                return {"type": "exit", "reason": "cut", "price": stop}
            if bl <= lock:
                return {"type": "exit", "reason": "lock", "price": lock}
            if self._rider:
                ts_px = self._peak - self.trail_r * atr
                if bl <= ts_px:
                    return {"type": "exit", "reason": "rider_trail", "price": ts_px}
            if bh > self._peak:
                self._peak = bh
            return None

        if self._pos == -1:
            stop = self._entry + self.cut_r * atr
            lock = self._peak + self.profit_r * atr
            if bh >= stop:
                return {"type": "exit", "reason": "cut", "price": stop}
            if bh >= lock:
                return {"type": "exit", "reason": "lock", "price": lock}
            if self._rider:
                ts_px = self._peak + self.trail_r * atr
                if bh >= ts_px:
                    return {"type": "exit", "reason": "rider_trail", "price": ts_px}
            if bl < self._peak:
                self._peak = bl
            return None
        return None

    # ── confirm / cancel ─────────────────────────────────────────────

    def confirm_entry(self, fill_price: float) -> None:
        """Commit the pending entry at the actual fill price."""
        p = self._pending
        if p is None or p.get("type") != "enter":
            return
        self._pos = 1 if p["direction"] == "BUY" else -1
        self._entry = float(fill_price)
        self._peak = float(fill_price)
        self._rider = bool(p.get("rider", False))
        self._jump_dir = self._pos if self._rider else 0
        self._pending = None

    def cancel_entry(self) -> None:
        """Restore flat if the entry order was blocked."""
        if self._pending is not None and self._pending.get("type") == "enter":
            self._pending = None

    def confirm_exit(self) -> None:
        """Commit a pending exit: back to flat, base = exit price (matches the
        backtest's base = stop/lock/ts so the next wave re-enters from here).

        A cut/lock exit taken while in rider mode keeps _rider=True so the
        candle stays dead (no re-entry after a jump-rider exit, matching the
        sweep). Only a rider_trail exit re-enables scalping this candle.
        """
        p = self._pending
        if p is None or p.get("type") != "exit":
            return
        if p.get("reason") == "candle_end":
            # Roll straight into the new candle (state reset, gate recomputed).
            self._start_candle(self._candle_ts + pd.Timedelta(hours=1))
            return
        self._base = float(p.get("price", self._base))
        self._pos = 0
        self._entry = 0.0
        self._peak = 0.0
        if p.get("reason") == "rider_trail":
            self._rider = False
            self._jump_dir = 0
        self._pending = None

    def pending_action(self):
        return dict(self._pending) if self._pending else None

    @property
    def in_position(self) -> bool:
        return self._pos != 0

    @property
    def gate_ok(self) -> bool:
        return bool(self._gate_ok)

    @property
    def atr(self) -> float:
        return float(self._atr)

    def state_dict(self) -> dict:
        return {
            "candle": str(self._candle_ts) if self._candle_ts is not None else None,
            "candle_open": self._candle_open,
            "gate_ok": self._gate_ok,
            "atr": self._atr,
            "base": self._base,
            "pos": self._pos,
            "entry": self._entry,
            "peak": self._peak,
            "rider": self._rider,
            "pending": self._pending,
            "last_ts": str(self._last_ts) if self._last_ts is not None else None,
        }
