"""
EURUSD parameter sweep on 2-week sample, then validate best on full year.
"""

import sys, os, time, itertools, json
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.dukascopy_client import DukascopyClient
from app.eurusd_strategy import EURUSDStrategy, DEFAULT_CONFIG

INITIAL_BALANCE = 20.0

def _pnl(entry, exit_, direction, lot):
    delta = exit_ - entry if direction == "BUY" else entry - exit_
    return delta / 0.0001 * lot * 10

def resample(df, rule):
    d = df.set_index("time")
    r = d.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum", "spread": "mean",
    }).dropna().reset_index()
    return r

def run_backtest(m1, cfg):
    strategy = EURUSDStrategy(cfg)
    m1 = m1.sort_values("time").reset_index(drop=True)
    m5 = resample(m1, "5min")

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
    m1_times = m1["time"].values.astype(np.int64)
    m5_times = m5["time"].values.astype(np.int64)
    m5_idx = np.searchsorted(m5_times, m1_times, side="right") - 2
    m5_idx = np.clip(m5_idx, 0, len(m5) - 1)

    session_ranges = []
    sess = cfg.get("session", "lon_ny")
    if sess == "lon_ny":
        session_ranges = [(7, 16), (12, 21)]
    elif sess == "london":
        session_ranges = [(7, 16)]
    elif sess == "ny":
        session_ranges = [(12, 21)]
    else:
        session_ranges = [(0, 24)]

    def session_ok(ts):
        h = ts.hour + ts.minute / 60.0
        return any(start <= h < end for start, end in session_ranges)

    for idx in range(N):
        row = m1.iloc[idx]
        ts = row["time"]
        bid = row["close"]
        ask = bid
        mid = bid

        if cur_day != ts.date():
            cur_day = ts.date()
            daily_trades = 0
            daily_pnl = 0.0
            consec_losses = 0

        # Exit
        if cur_trade is not None:
            mm = m5_idx[idx]
            if mm >= 0:
                window_start = max(0, idx - 20)
                window = m1.iloc[window_start:idx+1].reset_index(drop=True)
                should_exit, _, reason = strategy.evaluate_exit(
                    window, cur_trade["entry_price"], cur_trade["direction"],
                    cur_trade["signal"]
                )
                if should_exit:
                    pnl = _pnl(cur_trade["entry_price"], mid, cur_trade["direction"], cfg.get("lot_size", 0.01))
                    balance += pnl
                    daily_pnl += pnl
                    if balance > peak: peak = balance
                    trades.append({"pnl": round(pnl, 2), "date": ts.date().isoformat()})
                    if pnl < 0: consec_losses += 1
                    else: consec_losses = 0
                    cooldown_until = ts + timedelta(seconds=cfg.get("cooldown_seconds", 30))
                    cur_trade = None
                    continue

        # Entry
        if cur_trade is None:
            if cooldown_until and ts < cooldown_until: continue
            cooldown_until = None
            if not session_ok(ts): continue
            if consec_losses >= cfg.get("consec_loss_limit", 4): continue
            if daily_trades >= cfg.get("max_daily_trades", 10): continue
            if daily_pnl <= -cfg.get("max_daily_loss", 5.0): continue
            if balance < cfg.get("min_balance", 10.0): continue

            mm = m5_idx[idx]
            if mm < 0: continue
            m5_window = m5.iloc[max(0, mm - 48):mm+1].reset_index(drop=True)
            m1_window = m1.iloc[max(0, idx - 100):idx+1].reset_index(drop=True)

            signal = strategy.evaluate(m1_window, m5_window, bid, ask)
            if signal and signal["score"] >= cfg.get("score_threshold", 0.65):
                cur_trade = {
                    "entry_time": ts,
                    "entry_price": ask if signal["direction"] == "BUY" else bid,
                    "direction": signal["direction"],
                    "signal": signal,
                    "entry_idx": idx,
                }
                daily_trades += 1

    if cur_trade is not None:
        pnl = _pnl(cur_trade["entry_price"], m1["close"].iloc[-1], cur_trade["direction"], cfg.get("lot_size", 0.01))
        balance += pnl
        trades.append({"pnl": round(pnl, 2), "date": m1["time"].iloc[-1].date().isoformat()})

    if not trades:
        return {"trades": 0, "net_pnl": 0, "win_rate": 0, "profit_factor": 0,
                "max_dd": 0, "final_bal": balance, "profit_days": 0, "total_days": 0, "avg_daily_pnl": 0}

    df = pd.DataFrame(trades)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / len(df) * 100 if len(df) > 0 else 0
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

    df["cum"] = df["pnl"].cumsum()
    df["peak"] = df["cum"].cummax()
    dd = (df["peak"] - df["cum"]).max()

    df["date"] = pd.to_datetime(df["date"])
    daily_stats = df.groupby(df["date"].dt.date).agg(pnl=("pnl", "sum"))
    profit_days = (daily_stats["pnl"] > 0).sum()
    total_days = len(daily_stats)
    avg_daily = daily_stats["pnl"].mean()

    return {
        "trades": len(df),
        "net_pnl": round(df["pnl"].sum(), 2),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "max_dd": round(dd, 2),
        "final_bal": round(balance, 2),
        "profit_days": profit_days,
        "total_days": total_days,
        "avg_daily_pnl": round(avg_daily, 2),
    }

def main():
    print("Loading EURUSD M1 data...", flush=True)
    client = DukascopyClient(symbol="EURUSD")
    m1_full = client.download_range(2024, 2024)
    if len(m1_full) == 0:
        print("No data loaded")
        return
    print(f"Loaded {len(m1_full)} bars", flush=True)

    # Use first 2 weeks for sweep
    m1_full = m1_full.sort_values("time").reset_index(drop=True)
    start_date = m1_full["time"].min()
    end_date = start_date + timedelta(days=14)
    m1 = m1_full[m1_full["time"] < end_date].reset_index(drop=True)
    print(f"Sweep data: {m1['time'].min()} to {m1['time'].max()} ({len(m1)} bars)", flush=True)

    # Focused grid
    grid = {
        "ema_fast": [3, 5],
        "ema_slow": [8, 13],
        "sl_atr_mult": [0.5, 0.8, 1.0],
        "tp_atr_mult": [1.5, 2.0, 2.5, 3.0],
        "rsi_buy_min": [35, 40],
        "rsi_sell_max": [60, 65],
        "score_threshold": [0.50, 0.60, 0.70],
        "session": ["all", "lon_ny"],
        "use_candle_filter": [True, False],
    }

    keys = list(grid.keys())
    all_combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Total combos: {len(all_combos)}", flush=True)

    results = []
    start = time.time()

    for i, combo in enumerate(all_combos):
        cfg = dict(zip(keys, combo))
        r = run_backtest(m1, {**DEFAULT_CONFIG, **cfg})
        results.append({**cfg, **r})

        elapsed = time.time() - start
        if (i + 1) % 20 == 0 or i == 0 or i == len(all_combos) - 1:
            pct = (i + 1) / len(all_combos) * 100
            print(f"[{pct:5.1f}%] {i+1}/{len(all_combos)} elapsed={elapsed:.0f}s  "
                  f"best=${max(r['net_pnl'] for r in results):+.2f}", flush=True)

    # Sort by net P&L
    results.sort(key=lambda x: x["net_pnl"], reverse=True)

    print(f"\n{'='*100}")
    print(f"  TOP 20 CONFIGURATIONS (2-week sample)")
    print(f"{'='*100}")
    h = f"  {'Rank':>4} {'NetPnL':>8} {'WR%':>6} {'PF':>6} {'Trades':>6} {'AvgDay':>8} {'Config'}"
    print(h)
    print(f"  {'-'*92}")
    for rank, r in enumerate(results[:20], 1):
        c = (f"EMA{r['ema_fast']}/{r['ema_slow']} "
             f"SL{r['sl_atr_mult']} TP{r['tp_atr_mult']} "
             f"thr{r['score_threshold']} "
             f"rsi={r['rsi_buy_min']}-{r['rsi_sell_max']} "
             f"sess={r['session']} cdl={r['use_candle_filter']}")
        print(f"  {rank:>4} ${r['net_pnl']:>+7.2f} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
              f"{r['trades']:>5} ${r['avg_daily_pnl']:>+7.2f} {c}")

    # Save
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to sweep_results.json", flush=True)

    # Validate top 5 on full year
    print(f"\n{'='*100}")
    print(f"  VALIDATING TOP 5 ON FULL YEAR")
    print(f"{'='*100}")
    for rank, r in enumerate(results[:5], 1):
        cfg = {k: r[k] for k in keys}
        print(f"\n  [{rank}/5] {c}", flush=True)
        vr = run_backtest(m1_full, {**DEFAULT_CONFIG, **cfg})
        print(f"  FULL YEAR: {vr['trades']} trades, ${vr['net_pnl']:+.2f}, "
              f"{vr['win_rate']}% WR, PF={vr['profit_factor']}, "
              f"avg_daily=${vr['avg_daily_pnl']:+.2f}, "
              f"profit_days={vr['profit_days']}/{vr['total_days']}", flush=True)

    print(f"\n{'='*100}")
    print(f"  DONE")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
