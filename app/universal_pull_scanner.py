"""Universal Pull-into-H1 Scalper Scanner.

Scans ALL configured symbols using the proven PullPrevH1Scalper strategy
(the only strategy that survived honest backtests with real costs, see
_bt_pull_prevh1.py / _tune_pull_prevh1.py and AGENTS.md) and returns the
eligible pairs ranked so the bot tries the strongest pull setups first.

This scanner is a stateless ranker: it rebuilds a throw-away scalper per
symbol each scan window (cached for SCANNER_SCAN_SEC) and reports eligibility
(h1 direction resolved, live ATR, tradeable market) + any fresh entry signal
the engine is detecting. It does NOT own live trade state — the per-symbol
PullPrevH1Scalper instances in Bot own all entry/exit state. The bot feeds the
ranked candidates into `_search_symbol` (the proven, state-synced entry path),
so every position is entered and exited by the same engine instance.

Per OOS 2026 validation (AGENTS.md):
  - US30   pull .30 / trail .35 / hold 24  PF train 1.73 valid 1.83 OOS 1.81
  - XAUUSD pull .30 / trail .15 / hold 12  PF train 1.22 valid 1.96 OOS 1.40
  - US100  pull .30 / trail .50 / hold 6   PF train 1.38 valid 1.86 OOS 1.34
"""

import asyncio
import time
from typing import Dict, List, Optional, Any

import pandas as pd

import config as cfg
from app.pull_h1_scalper import PullPrevH1Scalper
from app.logger import BotLogger

# Hard refresh throttle (seconds) between whole-board eligibility scans.
_SCAN_TTL = float(getattr(cfg, "PULL_REFRESH_SEC", 60))


def _build_scalper(symbol: str, logger: BotLogger) -> PullPrevH1Scalper:
    """Build a throw-away PullPrevH1Scalper for a symbol with its tuned params."""
    pp = dict(getattr(cfg, "SYMBOL_PULL_PARAMS", {}).get(symbol, {}) or {})
    return PullPrevH1Scalper(
        symbol=symbol,
        logger=logger,
        pull_r=float(pp.get("pull_r", 0.30)),
        trail_r=float(pp.get("trail_r", 0.35)),
        max_hold_bars=int(pp.get("max_hold", 24)),
        round_trip_price=float(pp.get("round_trip", 0.0)),
        min_h1_bars=int(getattr(cfg, "PULL_MIN_H1_BARS", 30)),
        daily_target_r=float(pp.get("daily_target_r", 0.0)),
        daily_max_loss_r=float(pp.get("daily_max_loss_r", 0.0)),
    )


class UniversalPullScanner:
    """Scans all configured symbols for PullPrevH1 entry opportunities."""

    def __init__(self, logger: Optional[BotLogger] = None, client: Optional[Any] = None):
        self.logger = logger or BotLogger()
        self.client = client
        self._cache: Dict[str, Dict] = {}      # symbol -> last ranking entry
        self._last_scan = 0.0

    def _log(self, msg: str):
        if self.logger is not None:
            try:
                self.logger.info(f"[UNIVERSAL] {msg}")
            except Exception:
                pass

    def refresh(self, client: Optional[Any] = None):
        if client is not None:
            self.client = client

    async def scan_symbol(self, symbol: str, client: Any) -> Optional[Dict]:
        """Run the proven pull engine on one symbol and return eligibility + any
        fresh entry signal, or None if the symbol is not currently eligible.

        NOTE: CapitalClient.get_rates is synchronous (per-app HTTP call), so it
        is NOT awaited here.
        """
        if client is None:
            return None
        info = client.get_symbol_info(symbol)
        if info is None or info.get("market_status") != "TRADEABLE":
            return None

        hist = int(getattr(cfg, "PULL_M1_HISTORY_BARS", 8000))
        try:
            df = client.get_rates(symbol, 1, hist)
        except Exception as e:
            self._log(f"{symbol}: history fetch failed: {e}")
            return None
        if df is None or len(df) == 0:
            return None

        eng = _build_scalper(symbol, self.logger)
        eng.set_history(df)
        ts = time.time()
        if not eng.warm_up(ts):
            return None  # not enough completed H1 bars yet

        # Feed the most recent tail so the engine can detect a fresh setup.
        try:
            recent = client.get_rates(symbol, 1, 60)
        except Exception as e:
            self._log(f"{symbol}: recent fetch failed: {e}")
            recent = None
        entry_signal = None
        if recent is not None and len(recent) > 0:
            try:
                eng.feed(recent, now_ts=ts)
            except Exception as e:
                self._log(f"{symbol}: feed failed: {e}")
                return None
            act = eng.pending_action()
            if act is not None and act.get("type") == "enter":
                entry_signal = {
                    "direction": act.get("direction"),
                    "entry": float(act.get("entry", 0.0)),
                }

        entry = {
            "symbol": symbol,
            "eligible": True,
            "atr": float(eng.atr) if eng.atr and eng.atr > 0 else 0.0,
            "h1dir": int(eng.h1dir),
            "can_trade": bool(eng.can_trade),
            "daily_r": float(eng.daily_r),
            "entry_signal": entry_signal,
        }
        if entry["atr"] > 0:
            entry["momentum_z"] = 0.0  # ATR-weighted rank key (see scan_all)
        return entry

    async def scan_all(self, client: Any = None, force: bool = False) -> List[Dict]:
        """Scan ALL configured symbols and return eligible pairs ranked by the
        proven pull-into-H1 edge (ATR-weighted). Cached for _SCAN_TTL seconds."""
        client = client or self.client
        now = time.time()
        if not force and client is not None and now - self._last_scan < _SCAN_TTL and self._cache:
            return list(self._cache.values())

        results: List[Dict] = []
        sem = asyncio.Semaphore(3)

        async def scan_one(sym: str):
            async with sem:
                try:
                    return await self.scan_symbol(sym, client)
                except Exception as e:
                    self._log(f"scan error for {sym}: {e}")
                    return None

        syms = list(getattr(cfg, "SYMBOLS", []))
        if not syms:
            return []
        try:
            raw = await asyncio.gather(*(scan_one(s) for s in syms))
        except Exception as e:
            self._log(f"scan_all gather failed: {e}")
            raw = []

        for r in raw:
            if r is not None and r.get("eligible"):
                results.append(r)

        # Rank: higher ATR (more volatility room) and a live entry signal first.
        def _rank(r: Dict) -> float:
            score = r.get("atr", 0.0) or 0.0
            if r.get("entry_signal"):
                score += score * 0.5  # +50% when the proven engine is actively signaling
            return score

        results.sort(key=_rank, reverse=True)

        self._cache = {r["symbol"]: r for r in results}
        self._last_scan = now
        self._log(
            f"scanned {len(syms)} symbols -> {len(results)} eligible "
            f"({', '.join(r['symbol'] for r in results[:5])})"
        )
        return results


# Convenience function for quick use / scripts.
async def scan_pull_opportunities(client: Any, force: bool = False) -> List[Dict]:
    """Quick scan of all symbols for pull-into-H1 entry opportunities."""
    return await UniversalPullScanner(client=client).scan_all(client, force=force)
