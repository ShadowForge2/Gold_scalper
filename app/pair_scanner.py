"""Whole-board pair scanner for the Capital.com catalog.

Every SCANNER_SCAN_SEC the bot pulls the FULL market board (one
GET /api/v1/markets call -> ~4000 markets with live bid/offer/status/type),
screens it for tradability, ranks the survivors by momentum, and takes the
top-K as dynamic trading candidates. Entry/exit is a simple honest rule:

  * trend filter  : close > EMA(fast) > EMA(slow)   (BUY)
  * momentum      : close-to-close ROC > 0          (BUY)
  * breakout      : completed bar closes past the prior swing extreme
  * risk          : SL = 1.5R ATR, TP = 2.0R ATR, trail from 1R with 0.5R
                    giveback, force close at max-hold bars.
  * cost          : live spread (from the bar feed) + SCANNER_COST_R.

The same pure functions drive the serial backtest (_bt_pair_scanner.py) so
live behaviour is proven on history before money is at risk.
"""
import time
from typing import Dict, List, Optional

import pandas as pd

import config as cfg
from app.candle_engine import compute_atr


# ── Pure signal + exit logic (shared live / backtest) ─────────────────

def compute_scanner_signal(df: pd.DataFrame, price: float,
                           last_triggered_bar: Optional[str] = None,
                           i: Optional[int] = None,
                           pre: Optional[Dict] = None) -> Optional[Dict]:
    """Evaluate ONE completed M15 bar for an entry signal.

    df: OHLC M15 frame sorted ascending, 'time' column optional.
    price: current mid/ref price used for candle_open anchoring.
    last_triggered_bar: str time of the last bar that already fired — never
    re-fires the same bar (live guard).
    i/pre: fast-path — `df` is the FULL frame, `i` the bar index, `pre` holds
    full-length precomputed series ('ema_fast','ema_slow','roc','atr',
    'swing_high','swing_low'). Default (live) evaluates the last bar with
    window recompute. Results are identical either way.
    Returns a signal dict (signal_type='scanner') or None.
    """
    if pre is not None and i is not None:
        idx = i
        ema_fast = pre["ema_fast"]
        ema_slow = pre["ema_slow"]
        roc = pre["roc"]
        atr = pre["atr"]
        swing_high = pre["swing_high"]
        swing_low = pre["swing_low"]
    else:
        need = max(cfg.SCANNER_LOOKBACK_BARS, cfg.SCANNER_SWING_BACK + 5)
        if df is None or len(df) < need:
            return None
        idx = len(df) - 1
        ema_fast = df["close"].ewm(span=cfg.SCANNER_EMA_FAST, adjust=False).mean()
        ema_slow = df["close"].ewm(span=cfg.SCANNER_EMA_SLOW, adjust=False).mean()
        roc = df["close"].pct_change(cfg.SCANNER_ROC_BARS)
        atr = compute_atr(df, 14)
        back = cfg.SCANNER_SWING_BACK
        swing_high = df["high"].shift(1).rolling(back, min_periods=back).max()
        swing_low = df["low"].shift(1).rolling(back, min_periods=back).min()

    last = df.iloc[idx]
    bar_time = str(last.get("time", ""))
    if last_triggered_bar is not None and bar_time == last_triggered_bar:
        return None

    atr_val = float(atr.iloc[idx]) if atr.iloc[idx] > 0 else (price * 0.001)
    prev_swing_high = float(swing_high.iloc[idx])
    prev_swing_low = float(swing_low.iloc[idx])
    if prev_swing_high != prev_swing_high or prev_swing_high is None:
        prev_swing_high = float(df["high"].iloc[:idx].max())
    if prev_swing_low != prev_swing_low or prev_swing_low is None:
        prev_swing_low = float(df["low"].iloc[:idx].min())

    c = float(last["close"])
    direction = None
    reason = None
    brk_min = float(cfg.SCANNER_BREAKOUT_MIN_R) * atr_val
    if (c > ema_fast.iloc[idx] > ema_slow.iloc[idx]
            and roc.iloc[idx] > 0
            and c - prev_swing_high >= brk_min):
        direction = "BUY"
        reason = "breakout_up"
    elif (c < ema_fast.iloc[idx] < ema_slow.iloc[idx]
          and roc.iloc[idx] < 0
          and prev_swing_low - c >= brk_min):
        direction = "SELL"
        reason = "breakout_dn"

    if direction is None:
        return None

    return {
        "direction": direction,
        "score": 0.6,
        "price": price,
        "candle_open": c,
        "sl": None,   # anchored at the actual fill in _execute_entry
        "tp1": None,
        "signal_type": "scanner",
        "atr": atr_val,
        "high_volatility": False,
        "bar_time": time.time(),
        "gate_ok": True,
        "reason": reason,
        "scanner_sl_r": float(cfg.SCANNER_SL_R),
        "scanner_tp_r": float(cfg.SCANNER_TP_R),
        "scanner_trail_at": float(cfg.SCANNER_TRAIL_AT_R),
        "scanner_trail_r": float(cfg.SCANNER_TRAIL_R),
        "scanner_max_hold": int(cfg.SCANNER_MAX_HOLD_BARS),
        "last_bar": bar_time,
    }


def compute_exit(direction: str, entry: float, o: float, h: float, l: float,
                 close: float, sl: float, tp: float, atr: float,
                 trail_at: float, trail_r: float, max_hold: int,
                 bars_since_entry: int, peak: float,
                 trail: Optional[float]) -> Dict:
    """Bar-close exit decision (honest backtest semantics).

    Returns {exit_px, reason, peak, trail}. Intrabar SL has priority over TP.
    """
    if direction == "BUY":
        if l <= sl:
            return {"exit_px": sl, "reason": "scanner_sl", "peak": peak, "trail": trail}
        if h >= tp:
            return {"exit_px": tp, "reason": "scanner_tp", "peak": peak, "trail": trail}
        new_peak = max(peak, close)
        if new_peak - entry >= trail_at * atr:
            cand_trail = new_peak - trail_r * atr
            trail = cand_trail if trail is None else max(trail, cand_trail)
        if trail is not None and close <= trail:
            return {"exit_px": close, "reason": "scanner_trail", "peak": new_peak, "trail": trail}
        if bars_since_entry >= max_hold:
            return {"exit_px": close, "reason": "scanner_max_hold", "peak": new_peak, "trail": trail}
        return {"exit_px": None, "reason": None, "peak": new_peak, "trail": trail}
    else:
        if h >= sl:
            return {"exit_px": sl, "reason": "scanner_sl", "peak": peak, "trail": trail}
        if l <= tp:
            return {"exit_px": tp, "reason": "scanner_tp", "peak": peak, "trail": trail}
        new_peak = min(peak, close)
        if entry - new_peak >= trail_at * atr:
            cand_trail = new_peak + trail_r * atr
            trail = cand_trail if trail is None else min(trail, cand_trail)
        if trail is not None and close >= trail:
            return {"exit_px": close, "reason": "scanner_trail", "peak": new_peak, "trail": trail}
        if bars_since_entry >= max_hold:
            return {"exit_px": close, "reason": "scanner_max_hold", "peak": new_peak, "trail": trail}
        return {"exit_px": None, "reason": None, "peak": new_peak, "trail": trail}


# ── Live scanner (catalog + screening + rank) ─────────────────────────

class PairScanner:
    def __init__(self, client, logger=None):
        self.client = client
        self.logger = logger
        self._catalog: List[Dict] = []
        self._catalog_ts = 0.0
        self._bars: Dict[str, pd.DataFrame] = {}
        self._bars_ts: Dict[str, float] = {}
        self._last_scan = 0.0
        self.candidates: List[Dict] = []

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(f"[SCANNER] {msg}")

    def refresh_catalog(self, force: bool = False) -> List[Dict]:
        now = time.time()
        ttl = max(60, getattr(cfg, "SCANNER_SCAN_SEC", 120))
        if not force and now - self._catalog_ts < ttl and self._catalog:
            return self._catalog
        try:
            markets = self.client.get_all_markets()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[SCANNER] catalog fetch failed: {e}")
            return self._catalog
        if markets:
            self._catalog = markets
            self._catalog_ts = now
            self._log(f"catalog refreshed: {len(markets)} markets")
        return self._catalog

    def screen(self, force: bool = False) -> List[Dict]:
        """TRADEABLE + liquid + in-vol-band markets from the live snapshot."""
        catalog = self.refresh_catalog(force=force)
        allowed = {t.strip().upper() for t in cfg.SCANNER_TYPES}
        out = []
        for m in catalog:
            try:
                if m.get("marketStatus") != "TRADEABLE":
                    continue
                if not m.get("streamingPricesAvailable", True):
                    continue
                typ = str(m.get("instrumentType") or "").upper()
                if typ not in allowed:
                    continue
                # Capital.com nests the live quote/stat fields under `snapshot`
                # (bid/offer/high/low/dailyChangePct) for both the single-market
                # and the full-list endpoints. Flat fields are kept as a
                # fallback for proxies/tests that flatten the payload.
                snap = m.get("snapshot") or {}
                bid = float(snap.get("bid") or m.get("bid") or 0)
                offer = float(snap.get("offer") or m.get("offer") or 0)
                if bid <= 0 or offer <= 0 or offer < bid:
                    continue
                spread_r = (offer - bid) / bid
                if spread_r > cfg.SCANNER_MAX_SPREAD_R:
                    continue
                pct_raw = snap.get("dailyChangePct")
                if pct_raw is None:
                    pct_raw = m.get("percentageChange")
                pct = abs(float(pct_raw or 0))
                if pct < cfg.SCANNER_MIN_PCT_CHANGE:
                    continue
                hi = float(snap.get("high") or m.get("high") or bid)
                lo = float(snap.get("low") or m.get("low") or bid)
                rng = (hi - lo) / bid if bid else 0.0
                if rng < cfg.SCANNER_MIN_ATR_R or rng > cfg.SCANNER_MAX_ATR_R:
                    continue
                out.append({
                    "epic": m.get("epic"),
                    "name": m.get("instrumentName", m.get("epic")),
                    "type": typ,
                    "bid": bid,
                    "offer": offer,
                    "spread": offer - bid,
                    "spread_r": spread_r,
                    "pct_change": float(pct_raw or 0),
                    "high": hi,
                    "low": lo,
                    "range_r": rng,
                    "price": (bid + offer) / 2,
                })
            except (TypeError, ValueError):
                continue
        return out

    def scan(self, force: bool = False) -> List[Dict]:
        """Rank screened markets by daily momentum and keep the top-K."""
        now = time.time()
        if now - self._last_scan < max(30, cfg.SCANNER_SCAN_SEC) and not force:
            return self.candidates
        screened = self.screen(force=force)
        screened.sort(key=lambda m: abs(m["pct_change"]), reverse=True)
        self.candidates = screened[:cfg.SCANNER_TOP_K]
        self._last_scan = now
        if self.logger:
            self._log(
                f"scan: {len(screened)} tradeable -> top {len(self.candidates)} "
                f"({', '.join(m['epic'] for m in self.candidates)})"
            )
        return self.candidates

    def fetch_bars(self, epic: str, force: bool = False) -> Optional[pd.DataFrame]:
        now = time.time()
        cached = self._bars.get(epic)
        if cached is not None and not force and now - self._bars_ts.get(epic, 0) < cfg.SCANNER_SCAN_SEC:
            return cached
        try:
            df = self.client.get_rates(epic, cfg.SCANNER_TIMEFRAME, cfg.SCANNER_LOOKBACK_BARS)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[SCANNER] bars fetch failed for {epic}: {e}")
            return cached
        if df is not None and len(df) >= 10:
            df = df.reset_index(drop=True)
            self._bars[epic] = df
            self._bars_ts[epic] = now
            return df
        return cached

    def entry_signal(self, epic: str, price: float,
                     last_triggered_bar: Optional[str] = None) -> Optional[Dict]:
        df = self.fetch_bars(epic)
        if df is None:
            return None
        sig = compute_scanner_signal(df, price, last_triggered_bar=last_triggered_bar)
        if sig is not None:
            sig["epic"] = epic
        return sig
