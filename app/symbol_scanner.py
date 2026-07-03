import time
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

import config as cfg
from app.bias_engine import BiasEngine
from app.signal_engine import SignalEngine
from app.logger import BotLogger


class SymbolScanner:
    """Scans multiple symbols independently, ranks signals by quality."""

    def __init__(self, client, logger: Optional[BotLogger] = None):
        self.client = client
        self.logger = logger or BotLogger()
        self.symbols = cfg.SYMBOLS
        self.bias_engines: Dict[str, BiasEngine] = {}
        self.signal_engines: Dict[str, SignalEngine] = {}
        self.bias_summaries: Dict[str, Dict] = {}
        self.last_bias_times: Dict[str, float] = {}
        self.last_signals: Dict[str, Optional[Dict]] = {}
        self.symbol_info_cache: Dict[str, Dict] = {}

    async def scan_all(self, effective_threshold: Optional[float] = None) -> List[Dict]:
        """Scan all symbols, return signals ranked by quality (best first)."""
        results = []
        for symbol in self.symbols:
            try:
                signal = await self._evaluate(symbol, effective_threshold)
                if signal:
                    results.append(signal)
            except Exception as e:
                self.logger.warning(f"Scanner error on {symbol}: {e}")

        results.sort(key=lambda s: s["rank_score"], reverse=True)
        return results

    async def _evaluate(self, symbol: str, effective_threshold: Optional[float] = None) -> Optional[Dict]:
        m1_data = self.client.get_rates(symbol, cfg.SIGNAL_TIMEFRAME, 100)
        if m1_data is None or len(m1_data) < 10:
            self.last_signals[symbol] = None
            return None

        symbol_info = self.client.get_symbol_info(symbol)
        if symbol_info is None:
            self.last_signals[symbol] = None
            return None

        self.symbol_info_cache[symbol] = symbol_info

        now = time.monotonic()
        bias_interval = cfg.BIAS_UPDATE_INTERVAL_SEC
        do_bias = (symbol not in self.last_bias_times or
                   now - self.last_bias_times[symbol] >= bias_interval)

        if do_bias or symbol not in self.bias_engines:
            h1_data = self.client.get_rates(symbol, cfg.BIAS_TIMEFRAME, 96)
            if h1_data is not None and len(h1_data) >= 20:
                engine = self.bias_engines.setdefault(symbol, BiasEngine())
                summary = engine.update(h1_data)
                self.bias_summaries[symbol] = summary
                self.last_bias_times[symbol] = now
                self.logger.bias(
                    f"{symbol} bias: {summary['bias']} "
                    f"(strength={summary['strength']:.2f}, "
                    f"H1={summary['primary_trend']}) "
                    f"{'TRADEABLE' if engine.is_tradeable() else 'WAITING'}"
                )

        bias_summary = self.bias_summaries.get(symbol, {})
        bias_engine = self.bias_engines.get(symbol)
        if not bias_engine or not bias_engine.is_tradeable():
            self.last_signals[symbol] = None
            return None

        bias_dir = bias_summary.get("bias", "NEUTRAL")

        h1_high = None
        h1_low = None
        try:
            if m1_data is not None and len(m1_data) >= 60:
                m1_idx = m1_data.set_index("time")
                has_bid_ask_m1 = "high_ask" in m1_idx.columns
                if has_bid_ask_m1:
                    h1_agg = m1_idx.resample("1h").agg({
                        "high_ask": "max", "low_ask": "min",
                        "high_bid": "max", "low_bid": "min",
                        "high": "max", "low": "min",
                    }).dropna()
                else:
                    h1_agg = m1_idx.resample("1h").agg({
                        "high": "max", "low": "min",
                    }).dropna()
                if len(h1_agg) >= 2:
                    bar = h1_agg.iloc[-2]
                    h1_high = bar.get("high_ask" if (has_bid_ask_m1 and bias_dir == "BULLISH") else "high_bid" if (has_bid_ask_m1 and bias_dir == "BEARISH") else "high")
                    h1_low = bar.get("low_ask" if (has_bid_ask_m1 and bias_dir == "BULLISH") else "low_bid" if (has_bid_ask_m1 and bias_dir == "BEARISH") else "low")
                    if h1_high is None or h1_low is None or not (h1_high > 0 and h1_low > 0):
                        h1_high = None
                        h1_low = None
        except Exception:
            pass

        if h1_high is None or h1_low is None or h1_high <= h1_low:
            h1_data = self.client.get_rates(symbol, cfg.BIAS_TIMEFRAME, 3)
            if h1_data is not None and len(h1_data) >= 2:
                has_bid_ask = "high_ask" in h1_data.columns
                if has_bid_ask and bias_dir == "BULLISH":
                    h1_high = h1_data["high_ask"].iloc[-2]
                    h1_low = h1_data["low_ask"].iloc[-2]
                elif has_bid_ask and bias_dir == "BEARISH":
                    h1_high = h1_data["high_bid"].iloc[-2]
                    h1_low = h1_data["low_bid"].iloc[-2]
                else:
                    h1_high = h1_data["high"].iloc[-2]
                    h1_low = h1_data["low"].iloc[-2]

        current_price = symbol_info["bid"] if bias_dir == "BEARISH" else symbol_info["ask"]

        signal_engine = self.signal_engines.setdefault(symbol, SignalEngine())
        signal = signal_engine.evaluate(
            m1_data, bias_summary, current_price,
            h1_high=h1_high, h1_low=h1_low,
        )

        self.last_signals[symbol] = signal

        if not signal:
            return None

        entry_threshold = effective_threshold if effective_threshold is not None else cfg.SIGNAL_ENTRY_THRESHOLD
        if signal["score"] < entry_threshold:
            return None

        spread = symbol_info.get("spread", 999)
        if spread > cfg.MAX_SPREAD_PIPS:
            self.logger.signal(f"{symbol}: signal blocked — spread {spread:.1f} > max {cfg.MAX_SPREAD_PIPS}")
            return None

        atr_value = self._compute_atr(m1_data)
        atr_pips = atr_value / (symbol_info.get("point", 0.0001)) if symbol_info.get("point", 0) > 0 else 0
        if atr_pips < cfg.MIN_ATR_PIPS:
            self.logger.signal(f"{symbol}: skipped — low ATR ({atr_pips:.1f} pips < {cfg.MIN_ATR_PIPS})")
            return None
        if atr_pips > cfg.MAX_VOLATILITY_PIPS:
            self.logger.signal(f"{symbol}: skipped — high ATR ({atr_pips:.1f} pips > {cfg.MAX_VOLATILITY_PIPS})")
            return None

        bias_strength = bias_summary.get("strength", 0)
        spread_norm = max(0, 1.0 - (spread / cfg.MAX_SPREAD_PIPS))
        ideal_atr = (cfg.MIN_ATR_PIPS + cfg.MAX_VOLATILITY_PIPS) / 2
        atr_score = max(0, 1.0 - abs(atr_pips - ideal_atr) / ideal_atr)

        rank_score = (signal["score"] * 0.40 +
                      bias_strength * 0.25 +
                      spread_norm * 0.20 +
                      atr_score * 0.15)

        return {
            **signal,
            "symbol": symbol,
            "spread": spread,
            "bias_strength": bias_strength,
            "rank_score": round(rank_score, 4),
            "atr_pips": round(atr_pips, 1),
            "symbol_info": symbol_info,
        }

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]),
                                   np.abs(l[1:] - c[:-1])))
        return float(np.mean(tr[-period:]))

    def get_bias_summary(self, symbol: str) -> Dict:
        return self.bias_summaries.get(symbol, {})

    def get_all_bias_summaries(self) -> Dict[str, Dict]:
        return dict(self.bias_summaries)

    def get_last_signal(self, symbol: str) -> Optional[Dict]:
        return self.last_signals.get(symbol)

    def get_all_last_signals(self) -> Dict[str, Optional[Dict]]:
        return dict(self.last_signals)
