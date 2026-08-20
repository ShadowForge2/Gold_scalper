"""
News-aware state machine for trading around economic events.
States: NORMAL, PRE_NEWS, SPIKE, POST_NEWS

Has two detection modes:
1. Calendar-aware: uses EconomicCalendar for anticipation (PRE_NEWS blocking)
2. Real-time: detects high-volatility bars as they happen (reactive)
"""
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# States
STATE_NORMAL = "NORMAL"
STATE_PRE_NEWS = "PRE_NEWS"
STATE_SPIKE = "SPIKE"
STATE_POST_NEWS = "POST_NEWS"

ALL_STATES = {STATE_NORMAL, STATE_PRE_NEWS, STATE_SPIKE, STATE_POST_NEWS}


class NewsStateMachine:
    def __init__(
        self,
        pre_window_min: int = 15,
        spike_window_min: int = 3,
        post_window_min: int = 60,
        vol_threshold: float = 3.0,
        atr_period: int = 14,
    ):
        self.pre_window_min = pre_window_min
        self.spike_window_min = spike_window_min
        self.post_window_min = post_window_min
        self.vol_threshold = vol_threshold
        self.atr_period = atr_period

        self.state = STATE_NORMAL
        self.calendar: object = None
        self._spike_start: Optional[float] = None
        self._post_start: Optional[float] = None
        self._last_event: Optional[dict] = None
        self._last_fired_event: Optional[dict] = None

        # Volatility tracking
        self._m5_highs: list[float] = []
        self._m5_lows: list[float] = []
        self._m5_closes: list[float] = []
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None
        self._detected_vol_event: bool = False
        self._vol_event_time: Optional[float] = None

    def set_calendar(self, calendar):
        self.calendar = calendar

    def reset(self):
        self.state = STATE_NORMAL
        self._spike_start = None
        self._post_start = None
        self._last_event = None
        self._detected_vol_event = False
        self._vol_event_time = None

    def update(self) -> str:
        now = datetime.now(timezone.utc)
        now_ts = time.time()

        # --- Calendar-aware detection ---
        calendar_event = None
        if self.calendar is not None:
            try:
                calendar_event = self.calendar.get_next_event()
            except Exception:
                calendar_event = None

        # --- Real-time volatility detection ---
        self._detected_vol_event = False
        if self._atr is not None and self._atr > 0 \
           and self._m5_closes and self._prev_close is not None:
            latest = self._m5_closes[-1]
            move = abs(latest - self._prev_close)
            if move > self._atr * self.vol_threshold:
                if self._vol_event_time is None or (now_ts - self._vol_event_time) > 120:
                    self._detected_vol_event = True
                    self._vol_event_time = now_ts
                    if calendar_event is None:
                        calendar_event = {
                            "datetime": now,
                            "title": "Volatility Event",
                            "currency": "XAU",
                            "impact": 2,
                            "source": "realtime",
                        }

        # --- State transitions ---
        if self.state == STATE_NORMAL:
            # Check if we should enter PRE_NEWS
            if calendar_event and calendar_event is not self._last_fired_event:
                mins_until = (calendar_event["datetime"] - now).total_seconds() / 60.0
                if 0 <= mins_until <= self.pre_window_min:
                    self.state = STATE_PRE_NEWS
                    self._last_event = calendar_event
                    logger.info(f"[NEWS] PRE_NEWS: '{calendar_event['title']}' in {mins_until:.0f}min")
                elif self._detected_vol_event:
                    self.state = STATE_SPIKE
                    self._spike_start = now_ts
                    self._last_event = calendar_event
                    self._last_fired_event = calendar_event
                    logger.info(f"[NEWS] SPIKE: real-time volatility detected ({move/self._atr:.1f}x ATR)")

        elif self.state == STATE_PRE_NEWS:
            if self._detected_vol_event or (
                calendar_event
                and calendar_event["datetime"] <= now
                and calendar_event is self._last_event
            ):
                self.state = STATE_SPIKE
                self._spike_start = now_ts
                self._last_fired_event = calendar_event
                logger.info(f"[NEWS] SPIKE: event '{calendar_event['title']}' fired")
            elif calendar_event is None or calendar_event is not self._last_event:
                self.state = STATE_NORMAL
                logger.info("[NEWS] Returned to NORMAL (event passed/cancelled)")

        elif self.state == STATE_SPIKE:
            if self._spike_start is not None:
                elapsed = (now_ts - self._spike_start) / 60.0
                if elapsed >= self.spike_window_min:
                    self.state = STATE_POST_NEWS
                    self._post_start = now_ts
                    logger.info(f"[NEWS] POST_NEWS: spike window ended ({elapsed:.0f}min)")

        elif self.state == STATE_POST_NEWS:
            if self._post_start is not None:
                elapsed = (now_ts - self._post_start) / 60.0
                if elapsed >= self.post_window_min:
                    self.state = STATE_NORMAL
                    self._last_fired_event = None
                    logger.info(f"[NEWS] NORMAL: post-news window ended ({elapsed:.0f}min)")
            # Check if new event is coming
            if calendar_event and calendar_event is not self._last_fired_event:
                mins_until = (calendar_event["datetime"] - now).total_seconds() / 60.0
                if 0 <= mins_until <= self.pre_window_min:
                    self.state = STATE_PRE_NEWS
                    self._last_event = calendar_event
                    logger.info(f"[NEWS] PRE_NEWS: next event '{calendar_event['title']}' in {mins_until:.0f}min")

        return self.state

    def feed_m5_bar(self, high: float, low: float, close: float):
        """Feed M5 bar data for real-time volatility detection."""
        self._m5_highs.append(high)
        self._m5_lows.append(low)
        self._m5_closes.append(close)

        if len(self._m5_highs) > self.atr_period + 5:
            self._m5_highs = self._m5_highs[-(self.atr_period + 5):]
            self._m5_lows = self._m5_lows[-(self.atr_period + 5):]
            self._m5_closes = self._m5_closes[-(self.atr_period + 5):]

        # Compute ATR
        old_atr = self._atr
        if len(self._m5_closes) >= 2:
            ranges = []
            prev_close = self._m5_closes[-2]
            for h, l in zip(self._m5_highs[-self.atr_period:], self._m5_lows[-self.atr_period:]):
                tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                ranges.append(tr)
            if ranges:
                self._atr = float(np.mean(ranges[-min(len(ranges), self.atr_period):]))

        # Log ATR changes >20%
        if old_atr is not None and self._atr is not None and old_atr > 0:
            pct = abs(self._atr - old_atr) / old_atr
            if pct > 0.20:
                logger.info("feed_m5_bar: ATR %.4f -> %.4f (%.0f%% change)", old_atr, self._atr, pct*100)

        # Log near-threshold volatility
        if self._atr is not None and self._atr > 0 and self._prev_close is not None:
            move = abs(close - self._prev_close)
            ratio = move / self._atr
            if ratio > 2.0:
                logger.info("feed_m5_bar: high vol bar %.1fx ATR (state=%s)", ratio, self.state)

        self._prev_close = self._m5_closes[-2] if len(self._m5_closes) >= 2 else close

    def should_block_entry(self) -> bool:
        """Block entry during PRE_NEWS (chop) and SPIKE (unpredictable)."""
        block = self.state in (STATE_PRE_NEWS, STATE_SPIKE)
        if block:
            logger.info("should_block_entry: True (state=%s)", self.state)
        return block

    def get_entry_multipliers(self) -> dict:
        """Return SL/TP multipliers based on current state."""
        if self.state == STATE_POST_NEWS:
            return {"sl_mult": 1.5, "tp_mult": 1.5}
        return {"sl_mult": 1.0, "tp_mult": 1.0}

    def get_state_info(self) -> dict:
        return {
            "state": self.state,
            "last_event": self._last_event,
            "spike_start": self._spike_start,
            "post_start": self._post_start,
            "atr_5m": round(self._atr, 2) if self._atr else None,
            "has_calendar": self.calendar is not None and self.calendar.is_available,
        }
