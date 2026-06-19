#!/usr/bin/env python3
"""Walk-forward optimisation: test same strategy across multiple time periods.
Uses Dukascopy M1 data (resampled to M5 + H1) for reliable multi-year coverage."""

import sys, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from app.dukascopy_client import DukascopyClient
import backtest as bt

INITIAL_BALANCE = 20.0
WARMUP_DAYS = 21
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "dukascopy")

# Live position sizing (matching live bot EquityScaler)
USE_LIVE_SIZING = True

PERIODS = [
    ("2023 Full Year", datetime(2023, 1, 1),           datetime(2023, 12, 31, 23, 59)),
    ("2024 Full Year", datetime(2024, 1, 1),           datetime(2024, 12, 31, 23, 59)),
    ("2025 H1",        datetime(2025, 1, 1),           datetime(2025, 6, 30, 23, 59)),
    ("2025 H2",        datetime(2025, 7, 1),           datetime(2025, 12, 31, 23, 59)),
    ("2025 Full Year", datetime(2025, 1, 1),           datetime(2025, 12, 31, 23, 59)),
]

PARAMS = {
    "entry_threshold": cfg.SIGNAL_ENTRY_THRESHOLD,
    "exit_threshold": cfg.EXIT_THRESHOLD_TIGHT,
    "base_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
    "sessions": cfg.ALLOWED_SESSIONS,
    "bias_strength_min": cfg.BIAS_STRENGTH_MIN,
    "lot_multiplier": cfg.LOT_MULTIPLIER,
    "max_total_lot_per_event": cfg.MAX_LOT if USE_LIVE_SIZING else max(1.0, INITIAL_BALANCE / 20.0),
    "max_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
    "event_loss_limit": cfg.MAX_EVENT_LOSS_USD,
    "consecutive_loss_limit": cfg.MAX_CONSECUTIVE_LOSSES,
    "daily_loss_limit": cfg.MAX_DAILY_LOSS_USD,
    "max_trades_per_session": cfg.MAX_TRADES_PER_SESSION,
    "min_balance": cfg.MIN_BALANCE,
    "cash_flows": [],
    "exit_mode": 1,
}


def load_data(client):
    """Download & cache M1 data for all needed years, return combined M1 DataFrame."""
    print("DOWNLOADING / LOADING DUKASCOPY DATA")
    print("-" * 50)
    needed = set()
    for _, start, end in PERIODS:
        needed.add(start.year)
        needed.add(end.year)
    needed.add(min(needed) - 1)  # warmup year
    for y in sorted(needed):
        df = client.download_year(y)
        print(f"  {y}: {len(df)} bars", flush=True)

    all_m1 = client.download_range(min(needed), max(needed))
    print(f"\nTotal M1 bars: {len(all_m1)}")
    if len(all_m1) > 0:
        print(f"Date range: {all_m1['time'].min()} to {all_m1['time'].max()}")
    print()
    return all_m1


def run_period(client, all_m1, label, start, end):
    """Run backtest for a single period."""
    warmup_start = start - timedelta(days=WARMUP_DAYS)
    mask = (all_m1["time"] >= warmup_start) & (all_m1["time"] < end)
    m1 = all_m1[mask].copy()
    if len(m1) < 100:
        print(f"  SKIP {label}: insufficient data ({len(m1)} bars)")
        return None

    h1 = client.resample_to(m1, 16385)
    m5 = client.resample_to(m1, 5)

    if len(h1) < 20 or len(m5) < 100:
        print(f"  SKIP {label}: insufficient resampled data (H1={len(h1)} M5={len(m5)})")
        return None

    data = bt._pre_compute(m5, h1)
    if data is None:
        print(f"  SKIP {label}: pre_compute failed")
        return None

    bt.BACKTEST_START = start
    bt.BACKTEST_END = end
    bt.INITIAL_BALANCE = INITIAL_BALANCE

    result = bt.run_backtest(data, params=PARAMS, verbose=False)
    if result is None or result.get("trades", 0) == 0:
        print(f"  SKIP {label}: no trades generated")
        return None

    return result


def main():
    print("=" * 80)
    sizing_label = "LIVE (MAX_LOT=%.2f)" % cfg.MAX_LOT if USE_LIVE_SIZING else "CONSERVATIVE (1 micro lot cap)"
    print(f"GOLD SCALPER - Walk-Forward Analysis ({sizing_label})")
    print(f"Balance: ${INITIAL_BALANCE:.2f} | Data: Dukascopy XAUUSD M1 -> M5 + H1")
    print(f"Config: entry={PARAMS['entry_threshold']} exit={PARAMS['exit_threshold']} "
          f"lot_mult={PARAMS['lot_multiplier']} sessions={PARAMS['sessions']} "
          f"max_lot={PARAMS['max_total_lot_per_event']}")
    print("=" * 80)
    print()

    client = DukascopyClient(cache_dir=CACHE_DIR)
    all_m1 = load_data(client)
    if len(all_m1) < 1000:
        print("ERROR: Insufficient data. Aborting.")
        return

    # Run each period
    print("RUNNING WALK-FORWARD PERIODS")
    print("-" * 50)
    results = []
    for label, start, end in PERIODS:
        print(f"\n{'=' * 60}")
        print(f"  PERIOD: {label}   ({start.date()} -> {end.date()})")
        print(f"{'=' * 60}")
        t0 = datetime.now()
        r = run_period(client, all_m1, label, start, end)
        elapsed = (datetime.now() - t0).total_seconds()
        if r:
            results.append((label, r))
            print(f"  => ${r['net_pnl']:>10.2f}  WR={r['win_rate']:>5.1f}%  "
                  f"PF={r['profit_factor']:>5.2f}  Trades={r['trades']:>4}  "
                  f"MaxDD=${r['max_dd']:>10.2f}  Ret={r['return_pct']:>9.2f}%  "
                  f"({elapsed:.0f}s)")
            print(f"     AvgWin=${r['avg_win']:>8.2f}  AvgLoss=${r['avg_loss']:>8.2f}  "
                  f"GP=${r['gross_profit']:>10.2f}  GL=${r['gross_loss']:>10.2f}")
        else:
            print(f"  SKIP ({elapsed:.0f}s)")

    # Summary table
    print(f"\n{'=' * 120}")
    print(f"  WALK-FORWARD COMPARISON ({sizing_label})")
    print(f"{'=' * 120}")
    hdr = (f"{'Period':<18} {'Trades':<7} {'WR%':<6} {'PF':<7} {'Net P&L':<14} "
           f"{'Max DD':<14} {'AvgWin':<10} {'AvgLoss':<10} {'Return%':<12} {'End Bal':<14}")
    print(hdr)
    print('-' * len(hdr))
    cum_balance = INITIAL_BALANCE
    for label, r in results:
        cum_balance += r['net_pnl']
        print(f"{label:<18} {r['trades']:<7} {r['win_rate']:<6} {r['profit_factor']:<7.2f} "
              f"${r['net_pnl']:<11.2f} ${r['max_dd']:<11.2f} ${r['avg_win']:<7.2f} "
              f"${r['avg_loss']:<7.2f} {r['return_pct']:<10.2f}% ${r['ending_balance']:<10.2f}")
    print("-" * len(hdr))
    print(f"{'Cumulative':<18} {'':<7} {'':<6} {'':<7} "
          f"${sum(r['net_pnl'] for _, r in results):<11.2f} {'':<14} {'':<10} "
          f"{'':<10} {'':<12} ${cum_balance:<10.2f}")
    print("=" * 120)

    # Monthly breakdown for each period
    print(f"\n{'=' * 100}")
    print("  MONTHLY BREAKDOWN BY PERIOD")
    print(f"{'=' * 100}")
    for label, r in results:
        monthly = r.get('monthly', {})
        if monthly:
            print(f"\n  {label}:")
            print(f"  {'Month':<10} {'Trades':<7} {'PnL':<12} {'WR%':<6}")
            print(f"  {'-' * 35}")
            for m, v in sorted(monthly.items()):
                print(f"  {m:<10} {v['trades']:<7} ${v['pnl']:<+9.2f} {v['wr']:<5.1f}%")
            total = sum(v['pnl'] for v in monthly.values())
            print(f"  {'TOTAL':<10} {'':<7} ${total:<+9.2f}")

    print(f"\n{'=' * 60}")
    print("  WALK-FORWARD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Sizing: {'LIVE (MAX_LOT=%.2f)' % cfg.MAX_LOT if USE_LIVE_SIZING else 'CONSERVATIVE'}")
    print(f"Starting balance: ${INITIAL_BALANCE:.2f} each period")
    print(f"Note: Each period starts fresh at ${INITIAL_BALANCE:.2f} (no compounding across periods).")
    print("Done.")


if __name__ == "__main__":
    main()
