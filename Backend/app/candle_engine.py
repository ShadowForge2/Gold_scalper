"""Candle helpers.

Only `compute_atr` is used by the live bot (the PullPrevH1Scalper engine and
the whole-board pair scanner). The old CandleEngine XGBoost class and its label
/ feature helpers were removed — that strategy did not survive honest backtests.
"""

import pandas as pd

ATR_PERIOD = 14


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
