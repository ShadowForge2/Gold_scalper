"""
pull_auto_tune — per-symbol self-calibration for the PullPrevH1Scalper.

For any pair WITHOUT an explicit SYMBOL_PULL_PARAMS entry, derive sensible
pull_r / trail_r / max_hold from that symbol's OWN recent structure:

  * pull_r   : typical M5 pullback depth (in H1-ATR) that precedes a
               continuation in the H1 direction  -> we key entries off the
               ~30th percentile so typical pullbacks trigger but noise doesn't.
  * trail_r  : typical giveback from the run's peak before the move reverses
               -> exit a bit tighter than the median reversal (x0.8).
  * max_hold : how long a favorable run usually lasts before reversing
               -> 80th percentile of run length (M5 bars), clamped 6..48.

Results are cached per symbol and refreshed every PULL_AUTO_TUNE_TTL_SEC so the
whole-board scan does not hammer the API on every cycle. Symbols with explicit
SYMBOL_PULL_PARAMS keep their hand-tuned (validated) values.
"""

import os
import time

import numpy as np
import pandas as pd

from app.candle_engine import compute_atr

ATR_PERIOD = 14

_TUNE_TTL = float(os.environ.get("PULL_AUTO_TUNE_TTL_SEC", "21600"))  # 6h
_TUNE_BARS = int(os.environ.get("PULL_AUTO_TUNE_BARS_M5", "600"))     # ~50h of M5

_CACHE: dict = {}  # (symbol, is_demo) -> (ts, params)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.set_index("time") if "time" in df.columns else df.copy()
    for col in ("open", "high", "low", "close"):
        if col not in idx.columns:
            return None
        idx[col] = pd.to_numeric(idx[col], errors="coerce")
    if getattr(idx.index, "tz", None) is not None:
        idx.index = idx.index.tz_convert("UTC").tz_localize(None)
    idx = idx[~idx.index.duplicated(keep="last")].sort_index()
    return idx


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()


def tune(symbol: str, client, logger=None) -> dict:
    """Return a dict of calibrated pull params for `symbol`, or None if the
    symbol lacks enough history to calibrate (caller falls back to defaults)."""
    try:
        df = client.get_rates(symbol, 5, _TUNE_BARS)
    except Exception:
        df = None
    if df is None or len(df) < 120:
        return None
    m5 = _normalize(df)
    if m5 is None or len(m5) < 120:
        return None
    h1 = _resample(m5, "1h")
    if len(h1) < 12:
        return None
    try:
        atr_ser = compute_atr(h1, ATR_PERIOD)
        atr = float(atr_ser.iloc[-1])
    except Exception:
        atr = 0.0
    if atr <= 0:
        return None

    # H1 direction of each completed H1 bar, aligned to each M5 bar by the
    # last COMPLETED H1 strictly before that M5 bar's hour (matches engine).
    h1dir = (h1["close"] - h1["open"]).apply(lambda x: int(np.sign(x)))
    hour_key = m5.index.floor("h") - pd.Timedelta(hours=1)
    m5 = m5.copy()
    m5["d"] = h1dir.reindex(hour_key).values

    closes = m5["close"].values
    dvals = m5["d"].values
    dvals = np.where(np.isnan(dvals), 0, dvals)
    dirs = dvals.astype(int)

    pull_depths = []   # pullback depth (ATR) that preceded a continuation
    run_fracs = []     # giveback/peak at the reversal
    run_bars = []      # M5 bars from entry to reversal

    c5 = []
    n = len(closes)
    for i in range(n):
        d = int(dirs[i])
        c5.append(float(closes[i]))
        if len(c5) > 5:
            c5 = c5[-5:]
        if len(c5) < 5 or d == 0:
            continue
        c0, c1, c2, c3, c4 = c5
        # Engine entry shape: two closes with the H1 dir, then a pullback, then
        # a turn back in the H1 dir.
        if (c2 - c1) * d <= 0 or (c1 - c0) * d <= 0:
            continue
        pull = (c2 - c3) * d
        if pull < 0:
            continue
        if (c4 - c3) * d <= 0:
            continue
        pull_depths.append(pull / atr)

        # Simulate the favorable run forward to learn its typical length/giveback.
        entry = c4
        run_ext = entry
        peak = 0.0
        bars = 0
        reversal = None
        for j in range(i + 1, min(n, i + 1 + 60)):
            bc = float(closes[j])
            run_ext = max(run_ext, bc) if d > 0 else min(run_ext, bc)
            wave = (run_ext - entry) * d
            if wave > peak:
                peak = wave
            back = (run_ext - bc) * d
            bars += 1
            if peak > 0 and back >= 0.5 * peak:
                reversal = back / peak
                break
        if reversal is not None:
            run_fracs.append(reversal)
            run_bars.append(bars)
        elif bars >= 60:
            run_bars.append(bars)
            if peak > 0:
                run_fracs.append(((run_ext - closes[min(n - 1, i + bars)]) * d) / peak)

    if len(pull_depths) < 8 or len(run_bars) < 8:
        return None

    pull_r = float(np.percentile(pull_depths, 30))
    if run_fracs:
        trail_r = float(np.median(run_fracs)) * 0.8
    else:
        trail_r = 0.35
    max_hold = int(np.percentile(run_bars, 80))

    # Clamp to sane, tradable bounds.
    pull_r = float(min(0.60, max(0.10, pull_r)))
    trail_r = float(min(0.90, max(0.10, trail_r)))
    max_hold = int(min(48, max(6, max_hold)))

    params = {
        "pull_r": round(pull_r, 3),
        "trail_r": round(trail_r, 3),
        "max_hold": max_hold,
        "round_trip": 0.0,
        "auto_tuned": True,
    }
    if logger is not None:
        try:
            logger.info(
                f"[AUTOTUNE {symbol}] pull_r={params['pull_r']} trail_r={params['trail_r']} "
                f"max_hold={params['max_hold']} (n={len(pull_depths)} pulls, "
                f"{len(run_bars)} runs, atr={atr:.4f})"
            )
        except Exception:
            pass
    return params


def get_pull_params(symbol: str, client, logger=None, cfg=None) -> dict:
    """Return tuned params for `symbol`, using explicit SYMBOL_PULL_PARAMS when
    present, else a cached/refreshed auto-tune. Falls back to SCANNER_* defaults
    if calibration is impossible.

    Cache is keyed by (symbol, is_demo) so demo-derived params never leak to
    live bots or vice versa — demo and live charts have different ticks and
    pullback characteristics."""
    import config as cfg_mod
    cfg = cfg or cfg_mod
    explicit = dict(getattr(cfg, "SYMBOL_PULL_PARAMS", {}).get(symbol, {}) or {})
    if explicit:
        return explicit
    if not getattr(cfg, "PULL_AUTO_TUNE_ENABLED", True):
        return {}
    is_demo = getattr(client, "demo", True)
    cache_key = (symbol, is_demo)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached is not None and now - cached[0] < _TUNE_TTL:
        return cached[1]
    params = tune(symbol, client, logger)
    if params is None:
        return {}
    _CACHE[cache_key] = (now, params)
    return params


def clear_cache(symbol: str = None, is_demo: bool = None) -> None:
    global _CACHE
    if symbol is None and is_demo is None:
        _CACHE.clear()
    elif symbol is None:
        _CACHE = {k: v for k, v in _CACHE.items() if k[1] != is_demo}
    elif is_demo is None:
        _CACHE = {k: v for k, v in _CACHE.items() if k[0] != symbol}
    else:
        _CACHE.pop((symbol, is_demo), None)
