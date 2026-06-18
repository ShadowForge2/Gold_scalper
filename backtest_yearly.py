"""
Year-by-year comprehensive backtest from 2008 to present.
Uses Dukascopy M1 data, resampled to M5 (signals) and H1 (bias).
Reports yearly returns, win rate, profit factor, max drawdown.
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

START_YEAR = 2016
END_YEAR = datetime.now().year
INITIAL_BALANCE = 20.0
WARMUP_DAYS = 21
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "dukascopy")

PARAMS = {
    "entry_threshold": cfg.SIGNAL_ENTRY_THRESHOLD,
    "exit_threshold": cfg.EXIT_THRESHOLD_TIGHT,
    "base_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
    "sessions": cfg.ALLOWED_SESSIONS,
    "bias_strength_min": cfg.BIAS_STRENGTH_MIN,
    "lot_multiplier": cfg.LOT_MULTIPLIER,
    "max_total_lot_per_event": max(1.0, INITIAL_BALANCE / 20.0),
    "max_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
    "event_loss_limit": cfg.MAX_EVENT_LOSS_USD,
    "consecutive_loss_limit": cfg.MAX_CONSECUTIVE_LOSSES,
    "daily_loss_limit": cfg.MAX_DAILY_LOSS_USD,
    "max_trades_per_session": cfg.MAX_TRADES_PER_SESSION,
    "min_balance": cfg.MIN_BALANCE,
    "cash_flows": [],
    "exit_mode": 1,
}


def run_year(client, year, all_m1):
    year_start = datetime(year, 1, 1)
    year_end = datetime(year + 1, 1, 1)
    warmup_start = year_start - timedelta(days=WARMUP_DAYS)

    mask = (all_m1["time"] >= warmup_start) & (all_m1["time"] < year_end)
    m1 = all_m1[mask].copy()
    if len(m1) < 100:
        return None

    h1 = client.resample_to(m1, 16385)
    m5 = client.resample_to(m1, 5)

    if len(h1) < 20 or len(m5) < 100:
        return None

    data = bt._pre_compute(m5, h1)
    if data is None:
        return None

    bt.BACKTEST_START = year_start
    bt.BACKTEST_END = year_end
    bt.INITIAL_BALANCE = INITIAL_BALANCE

    result = bt.run_backtest(data, params=PARAMS, verbose=False)
    if result is None or result.get("trades", 0) == 0:
        return None

    return result


def main():
    print("=" * 80)
    print("GOLD SCALPER PRO - Comprehensive Yearly Backtest")
    print(f"Period: {START_YEAR} to {END_YEAR}")
    print(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
    print(f"Data: Dukascopy XAUUSD M1 -> M5 + H1")
    print(f"Config: entry_threshold={PARAMS['entry_threshold']}, "
          f"exit_threshold={PARAMS['exit_threshold']}, "
          f"lot_mult={PARAMS['lot_multiplier']}, "
          f"sessions={PARAMS['sessions']}")
    print("=" * 80)
    print()

    client = DukascopyClient(cache_dir=CACHE_DIR)

    # Download all years
    print("DOWNLOADING DATA")
    print("-" * 40)
    for y in range(START_YEAR - 1, END_YEAR + 1):
        df = client.download_year(y)
        status = f"{len(df)} bars" if len(df) > 0 else "no data"
        print(f"  {y}: {status}", flush=True)

    # Load combined
    print("\nLOADING CACHED DATA")
    print("-" * 40)
    all_m1 = client.download_range(START_YEAR - 1, END_YEAR)
    print(f"Total M1 bars: {len(all_m1)}")
    if len(all_m1) > 0:
        print(f"Date range: {all_m1['time'].min()} to {all_m1['time'].max()}")
    print()

    if len(all_m1) < 1000:
        print("ERROR: Insufficient data downloaded. Aborting.")
        return

    # Run yearly backtests
    print("RUNNING YEARLY BACKTESTS")
    print("-" * 40)

    results = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"  {year}...", end=" ", flush=True)
        t0 = datetime.now()
        r = run_year(client, year, all_m1)
        elapsed = (datetime.now() - t0).total_seconds()
        if r:
            results.append({"year": year, **r})
            print(
                f"tr={r['trades']} WR={r['win_rate']:.1f}% "
                f"PnL=${r['net_pnl']:.2f} PF={r['profit_factor']:.2f} "
                f"DD=${r['max_dd']:.2f} Bal=${r['ending_balance']:.2f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        else:
            print(f"SKIP ({elapsed:.0f}s)", flush=True)

    if not results:
        print("\nNo results generated for any year.")
        return

    # Print yearly results table
    print("\n" + "=" * 130)
    print("YEARLY RESULTS SUMMARY")
    print("=" * 130)
    header = (
        f"{'Year':>6} | {'Trades':>7} | {'WR%':>6} | {'Gross$':>10} | "
        f"{'Net$':>10} | {'PF':>7} | {'AvgW$':>8} | {'AvgL$':>8} | "
        f"{'MaxDD$':>8} | {'EndBal$':>10} | {'Ret%':>9}"
    )
    print(header)
    print("-" * 130)

    cum_balance = INITIAL_BALANCE
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_gp = 0.0
    total_gl = 0.0
    total_pnl = 0.0
    max_dd_overall = 0.0
    peak_balance = INITIAL_BALANCE

    for r in results:
        pnl = r["net_pnl"]
        cum_balance += pnl
        if cum_balance > peak_balance:
            peak_balance = cum_balance
        dd = peak_balance - cum_balance
        if dd > max_dd_overall:
            max_dd_overall = dd

        ret_pct = (pnl / INITIAL_BALANCE) * 100

        print(
            f"{r['year']:>6} | {r['trades']:>7} | {r['win_rate']:>5.1f}% | "
            f"${r['gross_profit']:>8.2f} | ${r['net_pnl']:>8.2f} | "
            f"{r['profit_factor']:>6.2f} | ${r['avg_win']:>6.2f} | "
            f"${r['avg_loss']:>6.2f} | ${r['max_dd']:>6.2f} | "
            f"${cum_balance:>8.2f} | {ret_pct:>7.2f}%",
            flush=True,
        )

        total_trades += r["trades"]
        total_wins += r["wins"]
        total_losses += r["losses"]
        total_gp += r["gross_profit"]
        total_gl += r["gross_loss"]
        total_pnl += pnl

    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    overall_pf = abs(total_gp / total_gl) if total_gl != 0 else float("inf")
    overall_ret = (total_pnl / INITIAL_BALANCE) * 100

    print("-" * 130)
    print(
        f"{'TOTAL':>6} | {total_trades:>7} | {overall_wr:>5.1f}% | "
        f"${total_gp:>8.2f} | ${total_pnl:>8.2f} | "
        f"{overall_pf:>6.2f} | {'':>8} | {'':>8} | "
        f"${max_dd_overall:>6.2f} | ${cum_balance:>8.2f} | {overall_ret:>7.2f}%",
        flush=True,
    )
    print("=" * 130)

    # Yearly returns breakdown
    print("\n" + "=" * 90)
    print("YEARLY RETURNS BREAKDOWN")
    print("=" * 90)
    print(f"{'Year':>6} | {'Return%':>9} | {'CAGR%':>8} | {'MaxDD%':>8} | {'Trades':>7}")
    print("-" * 90)

    running_balance = INITIAL_BALANCE
    returns_list = []
    years_list = []

    for r in results:
        pnl = r["net_pnl"]
        year_ret = (pnl / running_balance) * 100
        running_balance += pnl
        returns_list.append(year_ret)
        years_list.append(r["year"])

        years_in = r["year"] - START_YEAR + 1
        cagr = ((running_balance / INITIAL_BALANCE) ** (1.0 / max(0.01, years_in)) - 1) * 100
        dd_pct = (r["max_dd"] / max(r["ending_balance"], 0.01)) * 100

        print(
            f"{r['year']:>6} | {year_ret:>7.2f}% | {cagr:>6.2f}% | "
            f"{dd_pct:>6.2f}% | {r['trades']:>7}",
            flush=True,
        )

    if returns_list:
        avg_ret = np.mean(returns_list)
        std_ret = np.std(returns_list)
        sharpe = (avg_ret / std_ret) * np.sqrt(12) if std_ret > 0 else 0
        total_cagr = (
            (cum_balance / INITIAL_BALANCE) ** (1.0 / max(0.01, len(results))) - 1
        ) * 100
        avg_trades = total_trades // max(len(results), 1)
        print("-" * 90)
        print(
            f"{'AVG':>6} | {avg_ret:>7.2f}% | {total_cagr:>6.2f}% | "
            f"{'':>8} | {avg_trades:>7}",
            flush=True,
        )
        print(f"\nSharpe (annualized): {sharpe:.2f}")
        print(f"Volatility (yearly): {std_ret:.2f}%")

    print("=" * 90)

    # Save results
    csv_path = os.path.join(os.path.dirname(__file__), "backtest_yearly_results.csv")
    rows = []
    for r in results:
        rows.append({
            "year": r["year"], "trades": r["trades"], "wins": r["wins"],
            "losses": r["losses"], "win_rate": r["win_rate"],
            "gross_profit": r["gross_profit"], "gross_loss": r["gross_loss"],
            "net_pnl": r["net_pnl"], "profit_factor": r["profit_factor"],
            "avg_win": r["avg_win"], "avg_loss": r["avg_loss"],
            "max_dd": r["max_dd"], "ending_balance": r["ending_balance"],
            "return_pct": r["return_pct"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
