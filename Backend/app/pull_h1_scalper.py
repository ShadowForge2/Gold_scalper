"""
PullPrevH1Scalper — live, stateful M5 pullback-into-H1-trend scalper.

Replicates `_bt_pull_prevh1.py` (pull/prevh1/trail25) as a persistent
per-symbol state machine so the bot can trade it live:

  DIR   : direction of the last COMPLETED H1 candle body (h1dir).
  ENTRY : on M5, two consecutive closes in the h1dir direction, then a
          pullback >= pull_r * ATR against it, then the next M5 close turns
          back in the h1dir direction  ->  ENTER at that close.
  EXIT  : trailing giveback — exit at the close of the M5 bar that gives back
          >= trail_r of the wave from entry (running extreme close); force
          close after max_hold completed M5 bars.

Live fill handling (same convention as the wave/momentum engines):
  - The engine emits an ENTER action when a trigger fires on a completed M5
    bar. The bot opens at market and calls confirm_entry(fill_price) with the
    ACTUAL fill, so the trailing wave is anchored to the real entry.
    cancel_entry() restores flat if the order is blocked.
  - Exits are emitted as pending actions; the bot closes at market and calls
    confirm_exit(). Until confirmed, the action is returned on every feed()
    so a failed close retries next tick.

Daily guards (per symbol, "daily profit bot"):
  - daily_target_r  > 0: stop NEW entries once the day's net R reaches it.
  - daily_max_loss_r > 0: stop NEW entries once the day's net R hits -max.
  The day rolls over at UTC midnight. An open position is never force-closed
  by these guards (only new entries are blocked).
"""

import os

import numpy as np
import pandas as pd

from app.candle_engine import compute_atr

ATR_PERIOD = 14


class PullPrevH1Scalper:
    def __init__(
        self,
        symbol: str,
        logger=None,
        pull_r: float = 0.30,
        trail_r: float = 0.35,
        max_hold_bars: int = 24,
        round_trip_price: float = 0.0,
        min_h1_bars: int = 30,
        daily_target_r: float = 0.0,
        daily_max_loss_r: float = 0.0,
        giveback_cap: float = 0.30,
        pump_atr: float = 0.5,
    ):
        self.symbol = symbol
        self._logger = logger
        self.pull_r = float(pull_r)
        self.trail_r = float(trail_r)
        self.max_hold_bars = int(max_hold_bars)
        self.round_trip_price = float(round_trip_price)
        self.min_h1_bars = int(min_h1_bars)
        self.daily_target_r = float(daily_target_r)
        self.daily_max_loss_r = float(daily_max_loss_r)
        # Profit lock-in: never give back more than GIVEBACK_CAP of the max
        # profit the trade has touched (tight trailing stop that rises with the
        # peak). PUMP_ATR: a single M5 candle whose favorable body exceeds this
        # multiple of the H1 ATR is treated as a "sudden pump" — exit at that
        # candle's close to capture the tip instead of waiting for a retrace.
        self.giveback_cap = float(giveback_cap)
        self.pump_atr = float(pump_atr)

        # M1 cache (naive UTC, sorted, deduped); M5/H1 resampled on demand.
        self._cache = None
        self._max_cache_bars = int(os.environ.get("PULL_MAX_CACHE_BARS", "20000"))

        # State
        self._last_m5_ts = None      # last fully processed completed M5 bar
        self._last_m5_close = None
        self._c5 = []                # last 5 completed M5 closes (newest last)
        self._c5_ts = []
        self._atr = 0.0              # ATR of the last completed H1 candle
        self._atr_entry = 0.0        # ATR at entry (for daily-guard R bookkeeping)
        self._h1dir = 0              # sign of last completed H1 body
        self._pos = 0                # 1 long / -1 short / 0 flat
        self._entry = 0.0
        self._hold = 0               # completed M5 bars since entry
        self._run_ext = 0.0
        self._peak_profit = 0.0      # max favorable move (R) the trade has touched
        self._pending = None

        # Daily guard state
        self._daily_r = 0.0
        self._daily_day = None

    def _log(self, msg):
        if self._logger is not None:
            try:
                self._logger.info(f"Pull[{self.symbol}] {msg}")
            except Exception:
                pass

    # ── data plumbing ────────────────────────────────────────────────

    def set_history(self, df: pd.DataFrame) -> None:
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

    def _resample(self, rule):
        cache = self._cache
        if cache is None or len(cache) == 0:
            return None
        return cache.resample(rule).agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
        }).dropna()

    def _refresh_context(self, ts: pd.Timestamp):
        """Recompute h1dir + ATR from the last completed H1 candle before `ts`."""
        h1 = self._resample("1h")
        if h1 is None or len(h1) < 2:
            return
        hour = ts.floor("h")
        completed = h1[h1.index < hour]
        if len(completed) < self.min_h1_bars:
            return
        try:
            atr_ser = compute_atr(completed, ATR_PERIOD)
            self._atr = float(atr_ser.iloc[-1])
        except Exception:
            self._atr = 0.0
        row = completed.iloc[-1]
        self._h1dir = int(np.sign(float(row["close"]) - float(row["open"])))

    # ── warm-up ──────────────────────────────────────────────────────

    def warm_up(self, now_ts: float) -> bool:
        if self._cache is None or len(self._cache) < 3:
            return False
        now = pd.Timestamp(now_ts, unit="s")
        cur5 = now.floor("5min")
        m5 = self._resample("5min")
        if m5 is None:
            return False
        completed = m5[m5.index < cur5]
        if len(completed) == 0:
            return False
        self._refresh_context(completed.index[-1])
        if self._last_m5_ts is None:
            self._last_m5_ts = completed.index[-1]
            self._last_m5_close = float(completed["close"].iloc[-1])
            # Prime the rolling window with the last few completed closes.
            tail = completed["close"].iloc[-5:]
            self._c5 = [float(v) for v in tail.values]
            self._c5_ts = list(tail.index)
            self._log(
                f"warmed up: h1dir={self._h1dir} atr={self._atr:.4f} "
                f"last_m5={self._last_m5_ts} ({len(self._cache)} M1 cached)"
            )
        return self._atr > 0 and self._h1dir != 0

    def adopt_position(self, direction: str, entry_price: float, atr: float = 0.0) -> None:
        self._pos = 1 if str(direction).upper() == "BUY" else -1
        self._entry = float(entry_price)
        self._run_ext = float(entry_price)
        self._peak_profit = 0.0
        self._hold = 0
        self._pending = None
        if atr and atr > 0:
            self._atr = float(atr)
        self._atr_entry = self._atr
        self._log(f"adopted {direction} @ {entry_price:.2f} (atr={self._atr:.4f})")

    # ── core feed ────────────────────────────────────────────────────

    def _roll_day(self, now: pd.Timestamp):
        day = now.date()
        if self._daily_day is None:
            self._daily_day = day
        elif day != self._daily_day:
            self._daily_day = day
            self._daily_r = 0.0

    def feed(self, recent_df: pd.DataFrame, now_ts: float):
        self.set_history(recent_df)
        self._roll_day(pd.Timestamp(now_ts, unit="s"))
        if self._pending is not None:
            return dict(self._pending)

        now = pd.Timestamp(now_ts, unit="s")
        cur5 = now.floor("5min")
        m5 = self._resample("5min")
        if m5 is None:
            return None
        completed = m5[m5.index < cur5]
        if len(completed) == 0:
            return None
        if self._last_m5_ts is None:
            self._last_m5_ts = completed.index[-1]
            self._last_m5_close = float(completed["close"].iloc[-1])
            self._refresh_context(completed.index[-1])
            return None

        new = completed[completed.index > self._last_m5_ts]
        if len(new) == 0:
            return None

        for ts, row in new.iterrows():
            bc = float(row["close"])
            self._refresh_context(ts)
            self._c5.append(bc)
            self._c5_ts.append(ts)
            if len(self._c5) > 5:
                self._c5 = self._c5[-5:]
                self._c5_ts = self._c5_ts[-5:]
            if self._pos == 0:
                self._evaluate_entry(ts, bc)
            else:
                self._evaluate_exit(ts, row)
            self._last_m5_ts = ts
            self._last_m5_close = bc
            if self._pending is not None:
                return dict(self._pending)
        return None

    def _evaluate_entry(self, ts: pd.Timestamp, bc: float):
        if self._atr <= 0 or self._h1dir == 0:
            return
        if len(self._c5) < 5:
            return
        c0, c1, c2, c3, c4 = self._c5
        dirn = self._h1dir
        # 2-close run aligned with the completed H1 direction
        if (c2 - c1) * dirn <= 0 or (c1 - c0) * dirn <= 0:
            return
        # pullback bar (c3) against the favour, deep enough
        if (c2 - c3) * dirn < self.pull_r * self._atr:
            return
        # turn bar (c4) back in the favour direction -> ENTER
        if (c4 - c3) * dirn <= 0:
            return
        if not self.can_trade:
            self._log(f"entry blocked by daily guard (daily_r={self._daily_r:+.3f})")
            return
        self._pending = {
            "type": "enter",
            "direction": "BUY" if dirn > 0 else "SELL",
            "entry": bc,
        }

    def _evaluate_exit(self, ts: pd.Timestamp, row: pd.DataFrame):
        dirn = self._pos
        bc = float(row["close"])
        self._hold += 1
        if dirn > 0:
            self._run_ext = max(self._run_ext, bc)
        else:
            self._run_ext = min(self._run_ext, bc)
        cur = (bc - self._entry) * dirn           # current favorable move
        wave = (self._run_ext - self._entry) * dirn
        # Track the best favorable move the trade has touched so the trailing
        # lock-in can cap the giveback at a fraction of the peak.
        if cur > self._peak_profit:
            self._peak_profit = cur
        back = (self._run_ext - bc) * dirn
        # Sudden-pump tip exit: a single M5 candle whose favorable body exceeds
        # pump_atr * ATR is treated as the tip of a spike — exit at that close
        # immediately instead of waiting for a retrace.
        body = (float(row["close"]) - float(row["open"])) * dirn
        if self._atr > 0 and body >= self.pump_atr * self._atr:
            self._pending = {"type": "exit", "reason": "pump_tip", "price": bc}
            return
        # Tight trailing: never give back more than GIVEBACK_CAP of the peak
        # profit the trade has touched (measured from the current price, so the
        # stop rises with the run and locks in the bulk of a spike).
        # DISABLED — backtests showed giveback_cap kills winners too early.
        # The trail_r exit already handles retracement protection.
        # if self._peak_profit > 0 and cur <= self._peak_profit * (1 - self.giveback_cap):
        #     self._pending = {"type": "exit", "reason": "giveback_cap", "price": bc}
        #     return
        if wave > 0 and back >= self.trail_r * wave:
            self._pending = {"type": "exit", "reason": "trail", "price": bc}
        elif self._hold >= self.max_hold_bars:
            self._pending = {"type": "exit", "reason": "max_hold", "price": bc}

    # ── confirm / cancel ─────────────────────────────────────────────

    def confirm_entry(self, fill_price: float) -> None:
        p = self._pending
        if p is None or p.get("type") != "enter":
            return
        self._pos = 1 if p["direction"] == "BUY" else -1
        self._entry = float(fill_price)
        self._run_ext = float(fill_price)
        self._peak_profit = 0.0
        self._hold = 0
        self._atr_entry = self._atr
        self._pending = None

    def cancel_entry(self) -> None:
        if self._pending is not None and self._pending.get("type") == "enter":
            self._pending = None

    def trail_stop_level(self) -> float:
        """Broker-side trailing stop level for a long (+1) or short (-1) trade.

        Anchors the stop below/above entry by (1 - trail_r) of the current wave
        extreme, mirroring the in-engine giveback test (back >= trail_r * wave)
        so that a broker stop at this level fires close to where the engine
        would have emitted a trail exit. Returns 0.0 when there is nothing
        meaningful to ratchet to (flat, no wave yet).
        """
        if self._pos == 0 or self._run_ext == 0.0 or self._entry == 0.0:
            return 0.0
        wave = (self._run_ext - self._entry) * self._pos
        if wave <= 0:
            return 0.0
        if self._pos > 0:
            # long: stop stays below the running high, giving back (1-trail_r) of wave
            return float(self._run_ext - (1.0 - self.trail_r) * wave)
        else:
            # short: stop stays above the running low
            return float(self._run_ext + (1.0 - self.trail_r) * wave)

    def confirm_exit(self, exit_price: float = None) -> None:
        p = self._pending
        if p is None or p.get("type") != "exit":
            return
        # Prefer the ACTUAL market fill passed by the bot over the engine's
        # bar-close estimate, so daily-R bookkeeping reflects the real price.
        px = float(exit_price) if exit_price not in (None, 0) else float(p.get("price", 0.0) or self._last_m5_close or 0.0)
        if px and self._pos != 0 and self._atr_entry > 0:
            gross = (px - self._entry) * self._pos / self._atr_entry
            r = gross - (self.round_trip_price / self._atr_entry)
            self._daily_r += r
            self._log(f"closed {self._pos:+d} @ {px:.2f} r={r:+.3f} "
                      f"daily_r={self._daily_r:+.3f}")
        self._pos = 0
        self._entry = 0.0
        self._run_ext = 0.0
        self._hold = 0
        self._pending = None

    def pending_action(self):
        return dict(self._pending) if self._pending else None

    @property
    def can_trade(self) -> bool:
        if self._daily_r == 0 and self._daily_day is None:
            return True
        if self.daily_target_r > 0 and self._daily_r >= self.daily_target_r:
            return False
        if self.daily_max_loss_r > 0 and self._daily_r <= -self.daily_max_loss_r:
            return False
        return True

    @property
    def in_position(self) -> bool:
        return self._pos != 0

    @property
    def atr(self) -> float:
        return float(self._atr)

    @property
    def h1dir(self) -> int:
        return int(self._h1dir)

    @property
    def daily_r(self) -> float:
        return float(self._daily_r)

    def state_dict(self) -> dict:
        return {
            "h1dir": self._h1dir,
            "atr": self._atr,
            "pos": self._pos,
            "entry": self._entry,
            "hold": self._hold,
            "run_ext": self._run_ext,
            "daily_r": self._daily_r,
            "can_trade": self.can_trade,
            "pending": self._pending,
            "last_m5_ts": str(self._last_m5_ts) if self._last_m5_ts is not None else None,
        }
