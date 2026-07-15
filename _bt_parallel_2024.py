"""Unified Portfolio Backtest 2024 -- XAUUSD + US100, single $20 balance, both ML models live."""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features
from app.asp_labels import compute_atr

warnings.filterwarnings("ignore")

# ======================================================================
# SYMBOL CONFIGS
# ======================================================================
SYMBOLS = {
    "XAUUSD": {
        "asp_model": "models/asp_swing_xgb_m5.joblib",
        "asp_features": "models/asp_swing_m5_features.npy",
        "cache_dir": "data/dukascopy",
        "contract_size": 1,
        "spread_pts": 0.35,
        "base_lot": 0.02,
        "max_lot": 10.0,
        "lot_step": 0.01,
        "min_atr": 0.50,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 1.0,
    },
    "US100": {
        "asp_model": "models/asp_swing_xgb_m5_US100.joblib",
        "asp_features": "models/asp_swing_m5_features_US100.npy",
        "cache_dir": "data/dukascopy_us100",
        "contract_size": 1,
        "spread_pts": 1.5,
        "base_lot": 0.02,
        "max_lot": 20.0,
        "lot_step": 0.01,
        "min_atr": 5.0,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 1.0,
    },
}

TIMEOUT_BARS = 5
TRAILING_TRIGGER = 1.5
TRAILING_RETRACE = 1.0
ATR_PERIOD = 14
STARTING_BALANCE = 20.0
YEAR = 2025


def load_symbol_data(sym, cfg):
    print(f"  Loading {sym}...", end=" ", flush=True)
    t0 = time.time()
    client = DukascopyClient(symbol=sym, cache_dir=cfg["cache_dir"])
    m1 = client.download_year(YEAR)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    m5 = m5.set_index("time")
    h1 = h1.set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    m5_h = m5["high"].values.astype(np.float64)
    m5_l = m5["low"].values.astype(np.float64)
    m5_c = m5["close"].values.astype(np.float64)
    atr_arr = compute_atr(m5_h, m5_l, m5_c, ATR_PERIOD)

    asp_saved = joblib.load(cfg["asp_model"])
    asp_model = asp_saved["model"]
    inv_map = asp_saved["label_map"]
    asp_features_list = list(np.load(cfg["asp_features"], allow_pickle=True))

    features = compute_asp_features(m5, h1)
    valid = features[asp_features_list].dropna()
    signal_series = pd.Series(0, index=features.index, dtype=np.int8)
    if len(valid) > 0:
        feat_arr = valid[asp_features_list].values
        raw_preds = asp_model.predict(feat_arr)
        preds = np.array([inv_map.get(p, p) for p in raw_preds])
        signal_series.loc[valid.index] = preds.astype(np.int8)

    buy_n = (signal_series.values == 1).sum()
    sell_n = (signal_series.values == -1).sum()
    print(f"{len(m5)} M5 bars, BUY={buy_n} SELL={sell_n} [{time.time()-t0:.1f}s]")

    return {
        "m5": m5, "m5_c": m5_c, "m5_h": m5_h, "m5_l": m5_l,
        "atr": atr_arr, "signals": signal_series.values,
        "times": m5.index.values,
    }


def calc_lot(bal, peak_bal, cfg, start_bal):
    lot = cfg["base_lot"] * (bal / start_bal)
    if peak_bal and peak_bal > 0:
        dd = (peak_bal - bal) / peak_bal * 100
        if dd > 15:
            lot *= 0.5
    step = cfg["lot_step"]
    lot = round(lot / step) * step
    return max(step, min(lot, cfg["max_lot"]))


def main():
    t_start = time.time()
    print("=" * 70)
    print("  UNIFIED PORTFOLIO BACKTEST 2024")
    print("  XAUUSD + US100 -- single $20 balance, both ML models live")
    print("=" * 70)

    # Load both symbols
    print("\n[1] Loading data & ML predictions...")
    data = {}
    for sym, cfg in SYMBOLS.items():
        data[sym] = load_symbol_data(sym, cfg)

    # Build unified timeline: for each M5 bar on each symbol, store an event
    # Events are (timestamp, symbol, bar_index) sorted by time
    print("\n[2] Building unified timeline...")
    events = []
    for sym, d in data.items():
        for i in range(ATR_PERIOD + 5, len(d["m5_c"])):
            events.append((d["times"][i], sym, i))
    events.sort(key=lambda x: x[0])
    print(f"  Total events: {len(events)}")

    # State per symbol
    state = {}
    for sym in SYMBOLS:
        state[sym] = {
            "in_trade": False,
            "entry_price": 0.0, "entry_dir": "",
            "sl_price": 0.0, "tp_price": 0.0,
            "entry_lot": 0.0, "entry_time": None,
            "ticks": 0, "best_price": 0.0,
        }

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    trades = []

    # Run through unified timeline
    print("\n[3] Running backtest...")
    for ts, sym, bar_i in events:
        cfg = SYMBOLS[sym]
        d = data[sym]
        m5_c = d["m5_c"]
        m5_h = d["m5_h"]
        m5_l = d["m5_l"]
        atr_arr = d["atr"]
        m5_times = d["times"]
        asp_arr = d["signals"]
        st = state[sym]

        atr = atr_arr[bar_i]
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        if st["in_trade"]:
            st["ticks"] += 1
            hit_sl = hit_tp = False

            if st["entry_dir"] == "BUY":
                st["best_price"] = max(st["best_price"], m5_h[bar_i])
                trigger = st["entry_price"] + atr * TRAILING_TRIGGER
                if st["best_price"] >= trigger:
                    trail_sl = st["best_price"] - atr * TRAILING_RETRACE
                    if trail_sl > st["sl_price"]:
                        st["sl_price"] = trail_sl
                if m5_l[bar_i] <= st["sl_price"]:
                    hit_sl = True
                elif m5_h[bar_i] >= st["tp_price"]:
                    hit_tp = True
            else:
                st["best_price"] = min(st["best_price"], m5_l[bar_i])
                trigger = st["entry_price"] - atr * TRAILING_TRIGGER
                if st["best_price"] <= trigger:
                    trail_sl = st["best_price"] + atr * TRAILING_RETRACE
                    if trail_sl < st["sl_price"]:
                        st["sl_price"] = trail_sl
                if m5_h[bar_i] >= st["sl_price"]:
                    hit_sl = True
                elif m5_l[bar_i] <= st["tp_price"]:
                    hit_tp = True

            if hit_sl or hit_tp:
                ep = st["sl_price"] if hit_sl else st["tp_price"]
                if st["entry_dir"] == "BUY":
                    pnl = (ep - st["entry_price"]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
                else:
                    pnl = (st["entry_price"] - ep) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_time": st["entry_time"], "exit_time": m5_times[bar_i],
                    "dir": st["entry_dir"], "entry": st["entry_price"], "exit": ep,
                    "reason": "sl" if hit_sl else "tp", "pnl": pnl,
                    "lot": st["entry_lot"], "balance": balance, "ticks": st["ticks"],
                    "hold_min": st["ticks"] * 5, "symbol": sym,
                })
                st["in_trade"] = False
                st["ticks"] = 0
                continue

            if st["ticks"] >= TIMEOUT_BARS:
                if st["entry_dir"] == "BUY":
                    pnl = (m5_c[bar_i] - st["entry_price"]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
                else:
                    pnl = (st["entry_price"] - m5_c[bar_i]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_time": st["entry_time"], "exit_time": m5_times[bar_i],
                    "dir": st["entry_dir"], "entry": st["entry_price"], "exit": m5_c[bar_i],
                    "reason": "timeout", "pnl": pnl,
                    "lot": st["entry_lot"], "balance": balance, "ticks": st["ticks"],
                    "hold_min": st["ticks"] * 5, "symbol": sym,
                })
                st["in_trade"] = False
                st["ticks"] = 0
                continue
        else:
            d_sig = asp_arr[bar_i]
            if d_sig != 0:
                atr_val = atr_arr[bar_i]
                if np.isnan(atr_val) or atr_val <= 0 or atr_val < cfg["min_atr"]:
                    continue
                direction = "BUY" if d_sig == 1 else "SELL"
                st["entry_price"] = m5_c[bar_i]
                sl_dist = atr_val * cfg["sl_atr_mult"]
                tp_dist = atr_val * cfg["tp_atr_mult"]
                st["sl_price"] = st["entry_price"] - sl_dist if direction == "BUY" else st["entry_price"] + sl_dist
                st["tp_price"] = st["entry_price"] + tp_dist if direction == "BUY" else st["entry_price"] - tp_dist
                st["entry_lot"] = calc_lot(balance, peak_balance, cfg, STARTING_BALANCE)
                st["in_trade"] = True
                st["entry_dir"] = direction
                st["entry_time"] = m5_times[bar_i]
                st["ticks"] = 0
                st["best_price"] = st["entry_price"]

    # Close any open positions at EOY
    for sym, st in state.items():
        if st["in_trade"]:
            cfg = SYMBOLS[sym]
            d = data[sym]
            bar_i = len(d["m5_c"]) - 1
            if st["entry_dir"] == "BUY":
                pnl = (d["m5_c"][bar_i] - st["entry_price"]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
            else:
                pnl = (st["entry_price"] - d["m5_c"][bar_i]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
            balance += pnl
            trades.append({
                "entry_time": st["entry_time"], "exit_time": d["times"][bar_i],
                "dir": st["entry_dir"], "entry": st["entry_price"], "exit": d["m5_c"][bar_i],
                "reason": "eoy", "pnl": pnl,
                "lot": st["entry_lot"], "balance": balance, "ticks": st["ticks"],
                "hold_min": st["ticks"] * 5, "symbol": sym,
            })

    # ======================================================================
    # REPORTS
    # ======================================================================
    df = pd.DataFrame(trades)
    if len(df) > 0:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df["month"] = df["entry_time"].dt.month
        df["month_name"] = df["entry_time"].dt.strftime("%B")
        df["week"] = df["entry_time"].dt.isocalendar().week.astype(int)

    elapsed = time.time() - t_start

    # --- Per-Symbol Breakdown ---
    print(f"\n{'='*70}")
    print("  PER-SYMBOL BREAKDOWN")
    print(f"{'='*70}")

    for sym in SYMBOLS:
        sdf = df[df["symbol"] == sym] if len(df) > 0 else pd.DataFrame()
        if len(sdf) == 0:
            print(f"\n  {sym}: No trades")
            continue
        total = len(sdf)
        wins = (sdf["pnl"] > 0).sum()
        net = sdf["pnl"].sum()
        gp = sdf.loc[sdf["pnl"] > 0, "pnl"].sum()
        gl = abs(sdf.loc[sdf["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 99.9
        wr = wins / total * 100
        buy_t = sdf[sdf["dir"] == "BUY"]
        sell_t = sdf[sdf["dir"] == "SELL"]
        tp = sdf[sdf["reason"] == "tp"]
        sl = sdf[sdf["reason"] == "sl"]
        to = sdf[sdf["reason"] == "timeout"]

        print(f"\n  --- {sym} ---")
        print(f"  Trades: {total} | WR: {wr:.1f}% | PF: {pf:.2f} | Net: ${net:,.2f}")
        print(f"  BUY:  {len(buy_t):>4} ({(buy_t['pnl']>0).sum()} wins) | SELL: {len(sell_t):>4} ({(sell_t['pnl']>0).sum()} wins)")
        print(f"  TP: {len(tp):>4} | SL: {len(sl):>4} | Timeout: {len(to):>4}")

    # --- Monthly Portfolio ---
    if len(df) > 0:
        print(f"\n{'='*70}")
        print("  MONTHLY PORTFOLIO")
        print(f"{'='*70}")
        print(f"\n  {'Month':<12} {'XAUUSD':>10} {'US100':>10} {'Total':>10} {'Trades':>7} {'WR%':>6} {'Balance':>12}")
        print(f"  {'-'*70}")

        for m in range(1, 13):
            mc = df[df["month"] == m]
            if len(mc) == 0:
                continue
            xau_pnl = mc[mc["symbol"] == "XAUUSD"]["pnl"].sum()
            us_pnl = mc[mc["symbol"] == "US100"]["pnl"].sum()
            tot_pnl = xau_pnl + us_pnl
            m_wins = (mc["pnl"] > 0).sum()
            m_wr = m_wins / len(mc) * 100
            m_bal = mc["balance"].iloc[-1]
            mname = mc["month_name"].iloc[0]
            print(f"  {mname:<12} ${xau_pnl:>8,.2f} ${us_pnl:>8,.2f} ${tot_pnl:>8,.2f} {len(mc):>7} {m_wr:>5.1f}% ${m_bal:>10,.2f}")

    # --- Overall Summary ---
    print(f"\n{'='*70}")
    print("  OVERALL PORTFOLIO SUMMARY")
    print(f"{'='*70}")
    total_trades = len(df)
    if total_trades > 0:
        wins = (df["pnl"] > 0).sum()
        losses = total_trades - wins
        net = df["pnl"].sum()
        gp = df.loc[df["pnl"] > 0, "pnl"].sum()
        gl = abs(df.loc[df["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 99.9
        wr = wins / total_trades * 100
        avg_win = gp / wins if wins > 0 else 0
        avg_loss = gl / losses if losses > 0 else 0
        max_dd = 0
        peak = STARTING_BALANCE
        for _, row in df.iterrows():
            peak = max(peak, row["balance"])
            dd = (peak - row["balance"]) / peak * 100
            max_dd = max(max_dd, dd)

        # Streaks
        results_arr = (df["pnl"] > 0).astype(int).values
        max_win_streak = max_loss_streak = 0
        cur_win = cur_loss = 0
        for r in results_arr:
            if r == 1:
                cur_win += 1; cur_loss = 0
                max_win_streak = max(max_win_streak, cur_win)
            else:
                cur_loss += 1; cur_win = 0
                max_loss_streak = max(max_loss_streak, cur_loss)

        print(f"  Starting:   ${STARTING_BALANCE:.2f}")
        print(f"  Final:      ${balance:,.2f}")
        print(f"  Net PnL:    ${net:,.2f} ({net/STARTING_BALANCE*100:,.0f}%)")
        print(f"  Max DD:     {max_dd:.1f}%")
        print(f"")
        print(f"  Trades:     {total_trades}")
        print(f"  Wins:       {wins} ({wr:.1f}%)")
        print(f"  Losses:     {losses} ({100-wr:.1f}%)")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Avg Win:    ${avg_win:,.2f}")
        print(f"  Avg Loss:   ${avg_loss:,.2f}")
        print(f"  Max Win Streak:  {max_win_streak}")
        print(f"  Max Loss Streak: {max_loss_streak}")

        # Milestones
        print(f"\n  Milestones:")
        milestones = [50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000]
        for ms in milestones:
            hit = df[df["balance"] >= ms]
            if len(hit) > 0:
                first = hit.iloc[0]
                print(f"    ${ms:>10,} at trade #{df.index.get_loc(first.name)+1} ({first['entry_time'].strftime('%Y-%m-%d')}) lot={first['lot']:.2f}")
            else:
                print(f"    ${ms:>10,} NOT reached")
    else:
        print("  No trades executed.")

    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
