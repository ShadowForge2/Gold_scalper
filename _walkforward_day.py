"""
Walk-forward single-day sim for the US100 momentum-jump engine.

Replicates the LIVE bot (app/bot.py), not the full-data backtest:
  - walks every M5 candle-completion boundary in a trading day's session
    (23:00Z prev day -> 21:00Z that day, matching the US100 CFD session)
  - at each boundary the entry scan uses a BOUNDED 8000-M1-bar window ending at
    that candle (same as MOMENTUM_M1_HISTORY_BARS=8000), resamples it to M5 and
    runs MomentumEngine.detect() -> fires only on the just-completed candle
  - entry fill = signal candle close; exit fill = exit candle close
  - exits step one candle per boundary: SL 1R (checked before retrace),
    jump 1R + 0.25R retrace from peak, max hold 12 bars — the exact exit replay
    verified bar-identical to the backtest in _mom_parity_test.py
  - cost 0.05R, EquityScaler-style sizing from a $20 start (margin-capped,
    lot = base_lot * (balance/20) * lot_mult, drawdown halves the lot)

Then compares the day's sim trades to the full-data backtest
(_two_engine.us100_jump_trades) restricted to the same session.

Usage:
  python _walkforward_day.py                     # random 2024 day, seed 2024
  python _walkforward_day.py --day 2024-06-03    # fixed day
  python _walkforward_day.py --start-equity 20 --lot-mult 2.0
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t
from app.momentum_engine import MomentumEngine
from _two_engine import us100_jump_trades, AggressiveScaler, run_usd, report

ENGINE = MomentumEngine(mz_min=2.0, body_min=0.60, ts_min=0.50, ema_span=480,
                        sl_r=1.0, jump_target=1.0, retr_r=0.25, max_hold=12,
                        atr_period=14)

M1_HISTORY_BARS = 8000
NEED_BARS = (ENGINE.ema_span + 5) * 5
COST_R = 0.05
YEAR = 2024
M5_OFFSET = pd.Timedelta(minutes=5)
EPOCH = pd.Timestamp("1970-01-01")


def load_full():
    m1 = t.load_m1_data("US100", start_year=YEAR, end_year=YEAR)
    m5 = t.resample_m5(m1)
    m5 = t.compute_features(m5)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    return m1, m5, atr


def session_bounds(day):
    start = pd.Timestamp(day) - pd.Timedelta(hours=1)   # 23:00Z prev day
    end = pd.Timestamp(day) + pd.Timedelta(hours=21)    # 21:00Z that day
    return start, end


BREAK_MIN = 45  # any gap >= 45 min = the daily/weekly break


def session_candle_idxs(ends, day):
    """Contiguous candle range for day D's session: from 23:00Z on D-1 up to
    the first data gap >= BREAK_MIN (the ~21:00-23:00 daily break; the actual
    close is data-driven, ~21:15)."""
    start = pd.Timestamp(day) - pd.Timedelta(hours=1)
    s_ep = int(start.value // 10**9)
    idxs = np.where((ends > s_ep) & (ends <= s_ep + 26 * 3600))[0]
    for k in range(1, len(idxs)):
        if ends[idxs[k]] - ends[idxs[k - 1]] >= BREAK_MIN * 60:
            idxs = idxs[:k]
            break
    return idxs


def bounded_window(m1, times, b_epoch):
    """Simulate the live M1 fetch: the last M1_HISTORY_BARS M1 bars ending at b."""
    pos = int(np.searchsorted(times, np.datetime64(pd.Timestamp(b_epoch, unit="s"))))
    lo = max(0, pos - M1_HISTORY_BARS)
    return m1.iloc[lo:pos]


def simulate_day(m1, m5, atr, day):
    c = m5["close"].values
    h = m5["high"].values
    l = m5["low"].values
    starts = m5.index.values.astype("datetime64[s]").astype("int64")
    ends = (m5.index + M5_OFFSET).values.astype("datetime64[s]").astype("int64")
    times = m1["time"].values.astype("datetime64[s]")

    start, end = session_bounds(day)
    idxs = session_candle_idxs(ends, day)
    if len(idxs) == 0:
        return [], 0, 0
    n = len(m5)
    # Entries only during the session; exits keep stepping past the session end
    # (the live bot holds positions through the 21:00-23:00 daily break and
    # manages them when the market reopens).
    session_mask = np.zeros(n, dtype=bool)
    session_mask[idxs] = True
    candles = np.concatenate([idxs, np.arange(int(idxs[-1]) + 1, n)])

    trades = []
    entry = None
    block_idx = -1
    short_windows = 0
    for i in candles:
        b_ep = int(ends[i])
        if entry is not None:
            d = entry["dir"]
            e0 = entry["fill"]
            a0 = entry["atr"]
            sl = entry["sl"]
            peak = entry["peak"]
            exit_ = None
            if d > 0:
                if l[i] <= sl:
                    exit_ = ("momentum_sl", -1.0)
                else:
                    if h[i] > peak:
                        peak = h[i]
                    bfe = (peak - e0) / a0
                    if bfe >= ENGINE.jump_target and c[i] <= peak - ENGINE.retr_r * a0:
                        exit_ = ("momentum_retrace", (c[i] - e0) / a0)
            else:
                if h[i] >= sl:
                    exit_ = ("momentum_sl", -1.0)
                else:
                    if l[i] < peak:
                        peak = l[i]
                    bfe = (e0 - peak) / a0
                    if bfe >= ENGINE.jump_target and c[i] >= peak + ENGINE.retr_r * a0:
                        exit_ = ("momentum_retrace", (e0 - c[i]) / a0)
            bars_held = i - entry["idx"]
            if exit_ is None and bars_held >= ENGINE.max_hold:
                r = (c[i] - e0) / a0 if d > 0 else (e0 - c[i]) / a0
                exit_ = ("momentum_timeout", r)
            if exit_ is not None:
                reason, r_out = exit_
                trades.append({
                    "time": m5.index[entry["idx"]],
                    "symbol": "US100",
                    "r": r_out - COST_R,
                    "atr": a0,
                    "price": e0,
                    "score": entry["score"],
                    "exit": str(m5.index[i]),
                    "reason": reason,
                    "dir": "BUY" if d > 0 else "SELL",
                    "held": bars_held,
                })
                block_idx = i
                entry = None
            else:
                entry["peak"] = peak
            continue

        if i <= block_idx:
            continue

        if not session_mask[i]:
            continue  # no new entries outside the session window

        win = bounded_window(m1, times, b_ep)
        if len(win) < NEED_BARS:
            short_windows += 1
            continue
        sig = ENGINE.detect(win, now_ts=b_ep + 1)
        if sig is None:
            continue
        if int(sig["bar_time"]) != int(starts[i]):
            continue  # safety: the live bot only enters the just-completed candle
        d = 1 if sig["direction"] == "BUY" else -1
        fill = sig["close"]
        a0 = sig["atr"] if sig["atr"] > 0 else max(atr.values[i], 1e-10)
        entry = {"dir": d, "fill": fill, "atr": a0,
                 "sl": fill - d * ENGINE.sl_r * a0, "peak": fill,
                 "idx": i, "score": sig["score"]}
    return trades, short_windows, len(idxs)


def backtest_day(m5, atr, day):
    trades = us100_jump_trades(m5, atr)
    ends = (m5.index + M5_OFFSET).values.astype("datetime64[s]").astype("int64")
    idxs = session_candle_idxs(ends, day)
    if len(idxs) == 0:
        return []
    lo = m5.index[idxs[0]]
    hi = m5.index[idxs[-1]]
    out = []
    for tr in trades:
        if lo <= tr["time"] <= hi:
            out.append(tr)
    return out


def eligible_days(m5):
    ends = (m5.index + M5_OFFSET).values.astype("datetime64[s]").astype("int64")
    days = []
    for d in pd.date_range("2024-01-10", "2024-12-31", freq="D"):
        if d.weekday() > 3:  # Mon-Thu (full 22h sessions)
            continue
        n = len(session_candle_idxs(ends, d))
        if n >= 200:
            days.append(d)
    return days


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--day", type=str, default=None)
    p.add_argument("--start-equity", type=float, default=20.0)
    p.add_argument("--lot-mult", type=float, default=2.0)
    p.add_argument("--aggr", type=float, default=1.0)
    p.add_argument("--use-score", action="store_true")
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    print("=== US100 momentum walk-forward: one 2024 trading day (bounded 8000-bar fetch) ===")
    print(f"session 23:00Z prev day -> 21:00Z | cost {COST_R}R | "
          f"start ${args.start_equity:.0f} | lot_mult x{args.lot_mult} | aggr x{args.aggr} | "
          f"score_mult={args.use_score}", flush=True)

    m1, m5, atr = load_full()
    print(f"full M1 {len(m1)} rows, M5 {len(m5)} bars  {m5.index[0]} .. {m5.index[-1]}", flush=True)

    days = eligible_days(m5)
    if args.day:
        day = pd.Timestamp(args.day)
    else:
        day = pd.to_datetime(days[int(np.random.default_rng(args.seed).integers(len(days)))]).date()
    day = pd.Timestamp(day)
    ends_arr = (m5.index + M5_OFFSET).values.astype("datetime64[s]").astype("int64")
    idxs = session_candle_idxs(ends_arr, day)
    print(f"\nrandom day (seed {args.seed}): {day.date()}  session "
          f"{m5.index[idxs[0]]} -> {m5.index[idxs[-1]] + M5_OFFSET}", flush=True)

    sim_trades, short_win, n_candles = simulate_day(m1, m5, atr, day)
    bt_trades = backtest_day(m5, atr, day)

    print(f"\nsession candles: {n_candles} | short windows (no warmup): {short_win} | "
          f"sim trades: {len(sim_trades)} | backtest trades: {len(bt_trades)}", flush=True)

    if sim_trades:
        print("\n--- sim trades (walk-forward) ---", flush=True)
        for tr in sim_trades:
            print(f"  {tr['time']} {tr['dir']:>4} entry {tr['price']:>10.2f} "
                  f"exit {tr['exit']} {tr['reason']:<16} held {tr['held']:>2} "
                  f"atr {tr['atr']:>7.2f} R {tr['r']:+7.2f}", flush=True)

    if bt_trades:
        print("\n--- backtest trades (full-data, same session) ---", flush=True)
        for tr in bt_trades:
            print(f"  {tr['time']} entry {tr['price']:>10.2f} R {tr['r']:+7.2f}", flush=True)

    sim_net = sum(x["r"] for x in sim_trades)
    bt_net = sum(x["r"] for x in bt_trades)
    print(f"\nR-level day total: sim {sim_net:+.2f}R  backtest {bt_net:+.2f}R  "
          f"delta {sim_net - bt_net:+.2f}R", flush=True)

    if sim_trades:
        sim_df = pd.DataFrame(sim_trades)
        wins = sim_df[sim_df["r"] > 0]
        losses = sim_df[sim_df["r"] < 0]
        pf = wins["r"].sum() / -losses["r"].sum() if len(losses) else float("inf")
        print(f"sim WR {100*(sim_df['r']>0).mean():.0f}%  PF {pf:.2f}  "
              f"exp {sim_df['r'].mean():+.3f}R", flush=True)

        # trade-by-trade parity vs backtest (same entry candle)
        bt_by_time = {pd.Timestamp(x["time"]): x["r"] for x in bt_trades}
        same = extra = missing = 0
        r_diff = 0.0
        for tr in sim_trades:
            if tr["time"] in bt_by_time:
                same += 1
                r_diff += tr["r"] - bt_by_time[tr["time"]]
            else:
                extra += 1
        for k, r in bt_by_time.items():
            if not any(tr["time"] == k for tr in sim_trades):
                missing += 1
        print(f"entry-candle parity: {same} matched, {extra} sim-only, "
              f"{missing} backtest-only (sum |R delta| on matched: {r_diff:+.2f}R)", flush=True)

        # USD equity walk with EquityScaler sizing from the $20 start
        scaler = AggressiveScaler(base_lots={"US100": 0.02}, lot_mult=args.lot_mult,
                                  aggr=args.aggr, use_score_mult=args.use_score)
        lev = {"US100": 20.0}
        rows, endb, mineq, below, first, blocked = run_usd(
            [dict(tr) for tr in sim_trades], args.start_equity, scaler, lev)
        print("\n--- USD walk ($20 start, EquityScaler sizing) ---", flush=True)
        for row in rows:
            print(f"  {row['time']} {row['dir']:>4} lot {row['lot']:.3f} "
                  f"R {row['r']:+7.2f}  $ {row['usd']:+8.2f}  equity {row['equity']:9.2f}",
                  flush=True)
        report("sim day", rows, endb, args.start_equity, mineq, below, first, blocked)


if __name__ == "__main__":
    main()
