import numpy as np
import config as cfg


class AdaptiveConfirmation:
    """
    Volatility-regime adaptive entry confirmation.
    Gives stricter confirmation in low vol, relaxed in high vol.

    Low vol (ATR <= 25th pct):  body_ratio >= P_LOW  percentile
    Normal vol:                 body_ratio >= P_NORM percentile
    High vol (ATR >= 75th pct): body_ratio >= P_HIGH percentile (None = no filter)
    """

    def __init__(self):
        self.atr_window = []
        self.br_window = []
        self.atr_period = cfg.ATR_PERIOD
        self.max_window = cfg.ADAPTIVE_CONF_WINDOW
        self.p_low = cfg.ADAPTIVE_CONF_P_LOW
        self.p_norm = cfg.ADAPTIVE_CONF_P_NORM
        self.p_high = cfg.ADAPTIVE_CONF_P_HIGH if cfg.ADAPTIVE_CONF_P_HIGH >= 0 else None
        self._warmup_bars = min(self.max_window, 100)

    def update(self, m1_data):
        """Update rolling windows with latest M1 bar."""
        if len(m1_data) < 2:
            return

        latest = m1_data.iloc[-1]
        rg = float(latest['high'] - latest['low'])
        if rg <= 0 or np.isnan(rg):
            return
        br = abs(float(latest['close'] - latest['open'])) / rg
        self.br_window.append(br)
        if len(self.br_window) > self.max_window:
            self.br_window.pop(0)

        # Rolling ATR
        closes = m1_data['close'].values
        highs = m1_data['high'].values
        lows = m1_data['low'].values
        if len(closes) >= self.atr_period + 1:
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1]),
                ),
            )
            atr = float(np.mean(tr[-self.atr_period:]))
            self.atr_window.append(atr)
            if len(self.atr_window) > self.max_window:
                self.atr_window.pop(0)

    def should_enter(self) -> bool:
        """Returns True if confirmation passes."""
        if not cfg.ADAPTIVE_CONFIRMATION_ENABLED:
            return True

        if len(self.br_window) < 20 or len(self.atr_window) < self._warmup_bars:
            return True

        lo = float(np.nanpercentile(self.atr_window, 25))
        hi = float(np.nanpercentile(self.atr_window, 75))
        current_atr = self.atr_window[-1]
        current_br = self.br_window[-1]

        pct = None
        if current_atr <= lo and self.p_low is not None:
            pct = self.p_low
        elif lo < current_atr < hi and self.p_norm is not None:
            pct = self.p_norm
        elif current_atr >= hi and self.p_high is not None:
            pct = self.p_high

        if pct is None:
            return True

        threshold = float(np.nanpercentile(self.br_window, pct))
        return current_br >= threshold

    def reset(self):
        self.atr_window.clear()
        self.br_window.clear()
