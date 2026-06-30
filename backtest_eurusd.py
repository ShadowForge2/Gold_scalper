"""
EURUSD Backtest — uses Dukascopy historical M1 data.

Runs the EURUSDStrategy over real market data and reports
daily / weekly / monthly P&L, win rate, profit factor.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

# Fix unicode on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict

from app.dukascopy_client import DukascopyClient
from app.eurusd_strategy import EURUSDStrategy, DEFAULT_CONFIG

# ── config ──

INITIAL_BALANCE = 20.0
LOT_SIZE = 0.01
POINT = 0.0001

BACKTEST_YEARS = [2024]
BIAS_WARMUP_DAYS = 14

# ── helpers ──

def _session_allowed(ts) -> bool:
    h = ts.hour + ts.minute / 60.0
    return (7 <= h < 16) or (12 <= h < 21)

def _pnl(entry, exit_, direction, lot):
    delta = exit_ - entry if direction == "BUY" else entry - exit_
    return delta / POINT * lot * 10  # USD for 0.01 lot (1 pip = $0.1)

# ── data loading ──

def load_eurusd_m1(years) -> Optional[pd.DataFrame]:
    client = DukascopyClient(symbol="EURUSD")
    fr = min(years)
    to = max(years)
    df = client.download_range(fr, to)
    if len(df) == 0:
        print("No data downloaded")
        return None
    print(f"Loaded {len(df)} M1 bars ({df['time'].min()} to {df['time'].max()})", flush=True)
    return df

# ── pre-compute H1 / M5 from M1 ──

def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.set_index("time")
    r = d.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum", "spread": "mean",
    }).dropna().reset_index()
    return r

# ── backtest ──

def run_backtest(m1: pd.DataFrame):
    strategy = EURUSDStrategy()
    m1 = m1.sort_values("time").reset_index(drop=True)

    m5 = resample(m1, "5min")
    print(f"M5: {len(m5)} bars", flush=True)

    trades = []
    cur_trade = None
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    consec_losses = 0
    daily_trades = 0
    cur_day = None
    daily_pnl = 0.0
    cooldown_until = None

    N = len(m1)
    report_interval = max(1, N // 50)

    m1_times = m1["time"].values.astype(np.int64)
    m5_times = m5["time"].values.astype(np.int64)
    m5_idx = np.searchsorted(m5_times, m1_times, side="right") - 2
    m5_idx = np.clip(m5_idx, 0, len(m5) - 1)

    for idx in range(N):
        if idx % report_interval == 0:
            pct = idx / N * 100
            print(f"  [{pct:5.1f}%] idx={idx}/{N} trades={len(trades)} bal=${balance:.2f}", flush=True)

        row = m1.iloc[idx]
        ts = row["time"]
        bid = row["close"]  # Dukascopy returns bid
        ask = bid  # simplified: no spread model for now
        mid = (bid + ask) / 2

        # Daily reset
        if cur_day != ts.date():
            if cur_day is not None:
                print(f"  DAY {cur_day}: {daily_trades} tr  daily=${daily_pnl:+.2f}  bal=${balance:.2f}", flush=True)
            cur_day = ts.date()
            daily_trades = 0
            daily_pnl = 0.0
            consec_losses = 0

        # Exit check
        if cur_trade is not None:
            mm = m5_idx[idx]
            if mm >= 0:
                # Get M1 window for exit evaluation
                window_start = max(0, idx - 20)
                window = m1.iloc[window_start:idx+1].reset_index(drop=True)
                should_exit, exit_score, reason = strategy.evaluate_exit(
                    window, cur_trade["entry_price"], cur_trade["direction"],
                    cur_trade["signal"]
                )
                if should_exit:
                    pnl = _pnl(cur_trade["entry_price"], mid, cur_trade["direction"], LOT_SIZE)
                    balance += pnl
                    daily_pnl += pnl
                    if balance > peak:
                        peak = balance
                    trades.append({
                        "entry_time": cur_trade["entry_time"],
                        "exit_time": ts,
                        "direction": cur_trade["direction"],
                        "entry_price": cur_trade["entry_price"],
                        "exit_price": mid,
                        "pnl": round(pnl, 2),
                        "pips": round(pnl / (LOT_SIZE * 10 / POINT), 1),
                        "reason": reason,
                        "bars_held": idx - cur_trade["entry_idx"],
                        "balance": round(balance, 2),
                        "date": ts.date().isoformat(),
                    })
                    if pnl < 0:
                        consec_losses += 1
                    else:
                        consec_losses = 0
                    cooldown_until = ts + timedelta(seconds=30)
                    cur_trade = None
                    continue

                if cur_trade.get("event_pnl", 0) <= -2.0:
                    pnl = _pnl(cur_trade["entry_price"], mid, cur_trade["direction"], LOT_SIZE)
                    balance += pnl
                    daily_pnl += pnl
                    if balance > peak:
                        peak = balance
                    trades.append({
                        "entry_time": cur_trade["entry_time"],
                        "exit_time": ts,
                        "direction": cur_trade["direction"],
                        "entry_price": cur_trade["entry_price"],
                        "exit_price": mid,
                        "pnl": round(pnl, 2),
                        "pips": round(pnl / (LOT_SIZE * 10 / POINT), 1),
                        "reason": "event_loss",
                        "bars_held": idx - cur_trade["entry_idx"],
                        "balance": round(balance, 2),
                        "date": ts.date().isoformat(),
                    })
                    consec_losses += 1
                    cooldown_until = ts + timedelta(seconds=30)
                    cur_trade = None
                    continue

                cur_trade["event_pnl"] = _pnl(cur_trade["entry_price"], mid, cur_trade["direction"], LOT_SIZE)

        # Entry check
        if cur_trade is None:
            # Cooldown
            if cooldown_until and ts < cooldown_until:
                continue
            cooldown_until = None

            # Session
            if not _session_allowed(ts):
                continue

            # Limits
            if consec_losses >= 4:
                continue
            if daily_trades >= 10:
                continue
            if daily_pnl <= -5.0:
                continue
            if balance < 10.0:
                continue

            # M5 data
            mm = m5_idx[idx]
            if mm < 0:
                continue
            m5_window = m5.iloc[max(0, mm - 48):mm+1].reset_index(drop=True)

            # M1 window
            m1_window = m1.iloc[max(0, idx - 100):idx+1].reset_index(drop=True)

            signal = strategy.evaluate(m1_window, m5_window, bid, ask)
            if signal and signal["score"] >= strategy.cfg["score_threshold"]:
                cur_trade = {
                    "entry_time": ts,
                    "entry_price": ask if signal["direction"] == "BUY" else bid,
                    "direction": signal["direction"],
                    "signal": signal,
                    "entry_idx": idx,
                    "event_pnl": 0.0,
                }
                daily_trades += 1
                print(f"  ENTRY {ts} {signal['direction']} @ {signal['entry_price']:.5f} "
                      f"score={signal['score']:.3f}", flush=True)

    # Close any open trade
    if cur_trade is not None:
        mid = (m1["close"].iloc[-1] + m1["close"].iloc[-1]) / 2
        pnl = _pnl(cur_trade["entry_price"], mid, cur_trade["direction"], LOT_SIZE)
        balance += pnl
        trades.append({
            "entry_time": cur_trade["entry_time"],
            "exit_time": m1["time"].iloc[-1],
            "direction": cur_trade["direction"],
            "entry_price": cur_trade["entry_price"],
            "exit_price": mid,
            "pnl": round(pnl, 2),
            "pips": round(pnl / (LOT_SIZE * 10 / POINT), 1),
            "reason": "end_of_data",
            "bars_held": N - cur_trade["entry_idx"],
            "balance": round(balance, 2),
            "date": m1["time"].iloc[-1].date().isoformat(),
        })

    if not trades:
        print("\nNo trades generated")
        return

    df_trades = pd.DataFrame(trades)

    # ── Results ──

    total_pnl = df_trades["pnl"].sum()
    wins = df_trades[df_trades["pnl"] > 0]
    losses = df_trades[df_trades["pnl"] < 0]
    wr = len(wins) / len(df_trades) * 100
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")

    df_trades["cumulative"] = df_trades["pnl"].cumsum()
    df_trades["peak"] = df_trades["cumulative"].cummax()
    df_trades["dd"] = df_trades["peak"] - df_trades["cumulative"]
    max_dd = df_trades["dd"].max()

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Period:       {m1['time'].min().date()} to {m1['time'].max().date()}")
    print(f"  Total Trades: {len(df_trades)}")
    print(f"  Wins:         {len(wins)}")
    print(f"  Losses:       {len(losses)}")
    print(f"  Win Rate:     {wr:.1f}%")
    print(f"  Gross Profit: ${gp:.2f}")
    print(f"  Gross Loss:   ${gl:.2f}")
    print(f"  Net P&L:      ${total_pnl:.2f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Max DD:       ${max_dd:.2f}")
    print(f"  Avg Win:      ${wins['pnl'].mean():.2f}" if len(wins) > 0 else "  Avg Win:      N/A")
    print(f"  Avg Loss:     ${losses['pnl'].mean():.2f}" if len(losses) > 0 else "  Avg Loss:     N/A")
    print(f"  Avg Bars Held: {df_trades['bars_held'].mean():.1f}")
    print(f"  Final Balance: ${balance:.2f}")
    print(f"  Return:       {(balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:.2f}%")

    # ── Daily breakdown ──

    print(f"\n{'='*60}")
    print(f"  DAILY BREAKDOWN")
    print(f"{'='*60}")
    df_trades["date"] = pd.to_datetime(df_trades["date"])
    daily = df_trades.groupby(df_trades["date"].dt.date).agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
    )
    daily["wr"] = (daily["wins"] / daily["trades"] * 100).round(1)
    print(f"  {'Date':<12} {'Trades':>7} {'P&L':>9} {'Wins':>5} {'WR':>6}")
    print(f"  {'-'*42}")
    profit_days = 0
    for date, row in daily.iterrows():
        label = "+" if row["pnl"] > 0 else "-"
        print(f"  {str(date):<12} {row['trades']:>7} ${row['pnl']:>+7.2f} {row['wins']:>5} {row['wr']:>5}%  {label}")
        if row["pnl"] > 0:
            profit_days += 1
    print(f"  {'-'*42}")
    print(f"  Profit Days: {profit_days}/{len(daily)} ({profit_days/len(daily)*100:.1f}%)")

    # Monthly
    print(f"\n{'='*60}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"{'='*60}")
    monthly = df_trades.groupby(df_trades["date"].dt.to_period("M")).agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
    )
    for period, row in monthly.iterrows():
        print(f"  {period}: {int(row['trades']):3d} trades  ${row['pnl']:+.2f}")

    # Trade log
    print(f"\n{'='*60}")
    print(f"  TRADE LOG")
    print(f"{'='*60}")
    print(f"  {'#':>3} {'Date':<12} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'Pips':>7} {'PnL':>8} {'Reason':<20}")
    print(f"  {'-'*75}")
    for i, (_, t) in enumerate(df_trades.iterrows()):
        d = t["entry_time"]
        d_str = d.strftime("%m/%d %H:%M") if hasattr(d, "strftime") else str(d)[:16]
        print(f"  {i+1:>3} {d_str:<12} {t['direction']:<5} {t['entry_price']:>10.5f} "
              f"{t['exit_price']:>10.5f} {t['pips']:>7.1f} ${t['pnl']:>+7.2f} {t['reason']:<20}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    years = BACKTEST_YEARS
    print(f"Loading EURUSD M1 data for {years}...", flush=True)
    m1 = load_eurusd_m1(years)
    if m1 is not None and len(m1) > 0:
        run_backtest(m1)
