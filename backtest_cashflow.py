"""
Backtest with random deposits & withdrawals to see how cash flows affect the system.
Uses local Dukascopy 2025 data, resampled to M5 + H1.
"""

import sys, os
import pandas as pd
import numpy as np
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

np.random.seed(42)


def generate_cash_flows(start, end, config_name="moderate"):
    configs = {
        "none": {
            "deposit_min": 0, "deposit_max": 0,
            "withdrawal_min": 0, "withdrawal_max": 0,
            "interval_days": 9999,
        },
        "light": {
            "deposit_min": 5, "deposit_max": 15,
            "withdrawal_min": -10, "withdrawal_max": -3,
            "interval_days": 45,
            "deposit_prob": 0.6,
        },
        "moderate": {
            "deposit_min": 10, "deposit_max": 30,
            "withdrawal_min": -20, "withdrawal_max": -5,
            "interval_days": 21,
            "deposit_prob": 0.55,
        },
        "aggressive": {
            "deposit_min": 20, "deposit_max": 50,
            "withdrawal_min": -30, "withdrawal_max": -10,
            "interval_days": 10,
            "deposit_prob": 0.5,
        },
        "erratic": {
            "deposit_min": 5, "deposit_max": 100,
            "withdrawal_min": -50, "withdrawal_max": -5,
            "interval_days": 7,
            "deposit_prob": 0.4,
        },
    }
    c = configs.get(config_name, configs["moderate"])

    if c["interval_days"] >= 9999:
        return []

    cash_flows = []
    cursor = start + timedelta(days=14)

    while cursor < end - timedelta(days=7):
        if np.random.random() < c.get("deposit_prob", 0.5):
            amount = round(np.random.uniform(c["deposit_min"], c["deposit_max"]), 2)
        else:
            amount = round(np.random.uniform(c["withdrawal_min"], c["withdrawal_max"]), 2)
        cash_flows.append((cursor, amount))
        interval = int(np.random.normal(c["interval_days"], c["interval_days"] * 0.3))
        interval = max(2, interval)
        cursor += timedelta(days=interval)

    return cash_flows


def cash_flow_stats(cash_flows):
    deposits = sum(a for _, a in cash_flows if a > 0)
    withdrawals = sum(abs(a) for _, a in cash_flows if a < 0)
    return {
        "count": len(cash_flows),
        "deposits": round(deposits, 2),
        "withdrawals": round(withdrawals, 2),
        "net": round(deposits - withdrawals, 2),
        "deposit_count": sum(1 for _, a in cash_flows if a > 0),
        "withdrawal_count": sum(1 for _, a in cash_flows if a < 0),
    }


def run_scenario(client, all_m1, label, cash_flows):
    year = 2025
    year_start = datetime(year, 1, 1)
    year_end = datetime(year, 12, 31, 23, 59)
    warmup_start = year_start - timedelta(days=WARMUP_DAYS)

    mask = (all_m1["time"] >= warmup_start) & (all_m1["time"] < year_end)
    m1 = all_m1[mask].copy()
    if len(m1) < 100:
        print(f"  SKIP: insufficient data ({len(m1)} bars)"); return None

    h1 = client.resample_to(m1, 16385)
    m5 = client.resample_to(m1, 5)
    if len(h1) < 20 or len(m5) < 100:
        print(f"  SKIP: insufficient resampled data"); return None

    data = bt._pre_compute(m5, h1)
    if data is None:
        print(f"  SKIP: pre_compute failed"); return None

    bt.BACKTEST_START = year_start
    bt.BACKTEST_END = year_end
    bt.INITIAL_BALANCE = INITIAL_BALANCE

    params = {
        "entry_threshold": cfg.SIGNAL_ENTRY_THRESHOLD,
        "exit_threshold": cfg.EXIT_THRESHOLD_TIGHT,
        "base_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
        "sessions": cfg.ALLOWED_SESSIONS,
        "bias_strength_min": cfg.BIAS_STRENGTH_MIN,
        "lot_multiplier": cfg.LOT_MULTIPLIER,
        "max_total_lot_per_event": cfg.MAX_LOT,
        "max_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
        "event_loss_limit": cfg.MAX_EVENT_LOSS_USD,
        "consecutive_loss_limit": cfg.MAX_CONSECUTIVE_LOSSES,
        "daily_loss_limit": cfg.MAX_DAILY_LOSS_USD,
        "max_trades_per_session": cfg.MAX_TRADES_PER_SESSION,
        "min_balance": cfg.MIN_BALANCE,
        "cash_flows": cash_flows,
        "exit_mode": 1,
    }

    return bt.run_backtest(data, params=params, verbose=False)


def main():
    print("=" * 90)
    print("GOLD SCALPER - Cash Flow Impact Analysis (2025)")
    print(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
    print(f"Config: entry={cfg.SIGNAL_ENTRY_THRESHOLD} exit={cfg.EXIT_THRESHOLD_TIGHT}")
    print(f"        lot_mult={cfg.LOT_MULTIPLIER} sessions={cfg.ALLOWED_SESSIONS}")
    print("=" * 90)

    client = DukascopyClient(cache_dir=CACHE_DIR)
    print("\nLOADING 2025 DATA...")
    all_m1 = client.download_range(2024, 2025)
    print(f"Total M1 bars: {len(all_m1)}")
    if len(all_m1) < 1000:
        print("ERROR: Insufficient data."); return

    scenarios = [
        ("Baseline (no cash flows)",     "none"),
        ("Light CF",                     "light"),
        ("Moderate CF",                  "moderate"),
        ("Aggressive CF",                "aggressive"),
        ("Erratic CF (mostly draws)",    "erratic"),
    ]

    results = []
    for label, config_name in scenarios:
        print(f"\n--- {label} ---")
        year_start = datetime(2025, 1, 1)
        year_end = datetime(2025, 12, 31, 23, 59)
        cash_flows = generate_cash_flows(year_start, year_end, config_name)
        stats = cash_flow_stats(cash_flows)

        if cash_flows:
            print(f"  Events: {stats['count']} (D:{stats['deposit_count']} W:{stats['withdrawal_count']})")
            print(f"  Deposits: ${stats['deposits']:.2f}  Withdrawals: ${stats['withdrawals']:.2f}  Net: ${stats['net']:+.2f}")
        else:
            print(f"  No cash flows")

        t0 = datetime.now()
        result = run_scenario(client, all_m1, label, cash_flows)
        elapsed = (datetime.now() - t0).total_seconds()

        if result:
            results.append((label, config_name, cash_flows, result))
            print(f"  => PnL=${result['net_pnl']:.2f}  WR={result['win_rate']:.1f}%  "
                  f"PF={result['profit_factor']:.2f}  Tr={result['trades']}  "
                  f"DD=${result['max_dd']:.2f}  End=${result['ending_balance']:.2f}  "
                  f"({elapsed:.0f}s)")
        else:
            print(f"  SKIP ({elapsed:.0f}s)")

    if not results:
        print("\nNo results."); return

    print(f"\n{'=' * 130}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 130}")
    hdr = (f"{'Scenario':<28} {'Trades':<7} {'WR%':<6} {'PF':<7} {'Net P&L':<11} "
           f"{'Max DD':<9} {'AvgWin':<9} {'AvgLoss':<9} {'End Bal':<11} "
           f"{'Deposits':<10} {'CF Net':<10}")
    print(hdr)
    print('-' * len(hdr))

    for label, cname, cf, r in results:
        stats = cash_flow_stats(cf)
        dips = stats['deposits'] if cname != "none" else 0
        cfnet = stats['net'] if cname != "none" else 0
        print(f"{label:<28} {r['trades']:<7} {r['win_rate']:<6} {r['profit_factor']:<7.2f} "
              f"${r['net_pnl']:<8.2f} ${r['max_dd']:<6.2f} ${r['avg_win']:<6.2f} "
              f"${r['avg_loss']:<6.2f} ${r['ending_balance']:<8.2f} "
              f"${dips:<8.2f} ${cfnet:<8.2f}")

    print("-" * len(hdr))

    # Relative to baseline
    bl = None
    for label, cname, cf, r in results:
        if cname == "none": bl = r
    if bl:
        print(f"\n  Difference from baseline:")
        print(f"  {'Scenario':<28} {'PnL delta':<12} {'DD delta':<12} {'Trades delta':<15}")
        print(f"  {'-' * 67}")
        for label, cname, cf, r in results:
            if cname == "none": continue
            dd_d = r['max_dd'] - bl['max_dd']
            pnl_d = r['net_pnl'] - bl['net_pnl']
            tr_d = r['trades'] - bl['trades']
            print(f"  {label:<28} ${pnl_d:<+9.2f} ${dd_d:<+9.2f} {tr_d:<+14d}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
