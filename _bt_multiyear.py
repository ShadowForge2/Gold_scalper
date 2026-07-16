"""Run unified portfolio backtest for multiple years: 2022, 2023, 2024."""
import sys, os, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

base = os.path.join(os.environ.get('USERPROFILE',''), 'OneDrive', 'Desktop', 'sir A.K hee', 'Dot Agni', 'Gold scalper')
sys.path.insert(0, base)
os.chdir(base)

from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features
from app.asp_labels import compute_atr

SYMBOLS = {
    "XAUUSD": {
        "asp_model": "models/asp_swing_xgb_m5.joblib",
        "asp_features": "models/asp_swing_m5_features.npy",
        "cache_dir": "data/dukascopy",
        "contract_size": 1, "spread_pts": 0.35, "base_lot": 0.02,
        "max_lot": 10.0, "lot_step": 0.01, "min_atr": 0.50,
        "sl_atr_mult": 2.0, "tp_atr_mult": 1.0,
    },
    "US100": {
        "asp_model": "models/asp_swing_xgb_m5_US100.joblib",
        "asp_features": "models/asp_swing_m5_features_US100.npy",
        "cache_dir": "data/dukascopy_us100",
        "contract_size": 1, "spread_pts": 1.5, "base_lot": 0.02,
        "max_lot": 20.0, "lot_step": 0.01, "min_atr": 5.0,
        "sl_atr_mult": 2.0, "tp_atr_mult": 1.0,
    },
}
TIMEOUT_BARS = 5
TRAILING_TRIGGER = 1.5
TRAILING_RETRACE = 1.0
ATR_PERIOD = 14
STARTING_BALANCE = 20.0

def calc_lot(bal, peak_bal, cfg, start_bal):
    lot = cfg["base_lot"] * (bal / start_bal)
    if peak_bal and peak_bal > 0:
        dd = (peak_bal - bal) / peak_bal * 100
        if dd > 15:
            lot *= 0.5
    step = cfg["lot_step"]
    lot = round(lot / step) * step
    return max(step, min(lot, cfg["max_lot"]))

def load_symbol_data(sym, cfg, year):
    client = DukascopyClient(symbol=sym, cache_dir=cfg["cache_dir"])
    m1 = client.download_year(year)
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
    return {"m5": m5, "m5_c": m5_c, "m5_h": m5_h, "m5_l": m5_l,
            "atr": atr_arr, "signals": signal_series.values, "times": m5.index.values,
            "buy_n": buy_n, "sell_n": sell_n, "bars": len(m5)}

def run_year(year):
    t_start = time.time()
    print()
    print("=" * 70)
    print("  BACKTEST %d -- $20 START, XAUUSD + US100" % year)
    print("=" * 70)

    print("\n  Loading data...")
    data = {}
    for sym, cfg in SYMBOLS.items():
        d = load_symbol_data(sym, cfg, year)
        data[sym] = d
        print("    %s: %d bars, BUY=%d SELL=%d" % (sym, d["bars"], d["buy_n"], d["sell_n"]))

    events = []
    for sym, d in data.items():
        for i in range(ATR_PERIOD + 5, len(d["m5_c"])):
            events.append((d["times"][i], sym, i))
    events.sort(key=lambda x: x[0])

    state = {}
    for sym in SYMBOLS:
        state[sym] = {"in_trade": False, "entry_price": 0.0, "entry_dir": "",
                      "sl_price": 0.0, "tp_price": 0.0, "entry_lot": 0.0,
                      "entry_time": None, "ticks": 0, "best_price": 0.0}

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    trades = []

    for ts, sym, bar_i in events:
        cfg = SYMBOLS[sym]
        d = data[sym]
        m5_c, m5_h, m5_l = d["m5_c"], d["m5_h"], d["m5_l"]
        atr_arr, m5_times, asp_arr = d["atr"], d["times"], d["signals"]
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
                trades.append({"entry_time": st["entry_time"], "exit_time": m5_times[bar_i],
                               "dir": st["entry_dir"], "entry": st["entry_price"], "exit": ep,
                               "reason": "sl" if hit_sl else "tp", "pnl": pnl,
                               "lot": st["entry_lot"], "balance": balance, "ticks": st["ticks"],
                               "hold_min": st["ticks"] * 5, "symbol": sym})
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
                trades.append({"entry_time": st["entry_time"], "exit_time": m5_times[bar_i],
                               "dir": st["entry_dir"], "entry": st["entry_price"], "exit": m5_c[bar_i],
                               "reason": "timeout", "pnl": pnl,
                               "lot": st["entry_lot"], "balance": balance, "ticks": st["ticks"],
                               "hold_min": st["ticks"] * 5, "symbol": sym})
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

    # Close EOY
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
            trades.append({"entry_time": st["entry_time"], "exit_time": d["times"][bar_i],
                           "dir": st["entry_dir"], "entry": st["entry_price"], "exit": d["m5_c"][bar_i],
                           "reason": "eoy", "pnl": pnl,
                           "lot": st["entry_lot"], "balance": balance, "ticks": st["ticks"],
                           "hold_min": st["ticks"] * 5, "symbol": sym})

    df = pd.DataFrame(trades)
    elapsed = time.time() - t_start

    if len(df) == 0:
        print("  No trades!")
        return None

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["month"] = df["entry_time"].dt.month
    df["month_name"] = df["entry_time"].dt.strftime("%B")

    # Per-symbol
    print("\n  PER-SYMBOL:")
    for sym in SYMBOLS:
        sdf = df[df["symbol"] == sym]
        if len(sdf) == 0:
            continue
        total = len(sdf)
        wins = (sdf["pnl"] > 0).sum()
        net = sdf["pnl"].sum()
        gp = sdf.loc[sdf["pnl"] > 0, "pnl"].sum()
        gl = abs(sdf.loc[sdf["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 99.9
        wr = wins / total * 100
        print("    %s: %d trades, WR=%.1f%%, PF=%.2f, Net=$%s" % (sym, total, wr, pf, "{:,.2f}".format(net)))

    # Monthly
    print("\n  MONTHLY:")
    print("  %-12s %10s %10s %10s %7s %6s %12s" % ("Month", "XAUUSD", "US100", "Total", "Trades", "WR%", "Balance"))
    print("  " + "-" * 70)
    for m in range(1, 13):
        mc = df[df["month"] == m]
        if len(mc) == 0:
            continue
        xau = mc[mc["symbol"] == "XAUUSD"]["pnl"].sum()
        us = mc[mc["symbol"] == "US100"]["pnl"].sum()
        tot = xau + us
        mw = (mc["pnl"] > 0).sum()
        mwr = mw / len(mc) * 100
        mbal = mc["balance"].iloc[-1]
        mn = mc["month_name"].iloc[0]
        print("  %-12s $%8s $%8s $%8s %7d %5.1f%% $%10s" % (
            mn, "{:,.2f}".format(xau), "{:,.2f}".format(us), "{:,.2f}".format(tot),
            len(mc), mwr, "{:,.2f}".format(mbal)))

    # Overall
    total_trades = len(df)
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

    print("\n  SUMMARY:")
    print("    Starting:  $20.00")
    print("    Final:     $%s" % "{:,.2f}".format(balance))
    print("    Net PnL:   $%s (%s%%)" % ("{:,.2f}".format(net), "{:,.0f}".format(net/STARTING_BALANCE*100)))
    print("    Max DD:    %.1f%%" % max_dd)
    print("    Trades:    %d" % total_trades)
    print("    Wins:      %d (%.1f%%)" % (wins, wr))
    print("    Losses:    %d (%.1f%%)" % (losses, 100-wr))
    print("    PF:        %.2f" % pf)
    print("    Avg Win:   $%s" % "{:,.2f}".format(avg_win))
    print("    Avg Loss:  $%s" % "{:,.2f}".format(avg_loss))
    print("    Max Win Streak:  %d" % max_win_streak)
    print("    Max Loss Streak: %d" % max_loss_streak)
    print("    Time:      %.0fs" % elapsed)

    return {"year": year, "final": balance, "net": net, "dd": max_dd,
            "trades": total_trades, "wr": wr, "pf": pf, "wins": wins, "losses": losses}

# Run all years
all_results = []
for yr in [2022, 2023, 2024]:
    r = run_year(yr)
    if r:
        all_results.append(r)

# Grand summary
print()
print("=" * 70)
print("  MULTI-YEAR COMPARISON")
print("=" * 70)
print()
print("  %-6s %12s %10s %7s %6s %6s %12s" % ("Year", "Final", "Net PnL", "Trades", "WR%", "PF", "Max DD"))
print("  " + "-" * 70)
for r in all_results:
    print("  %-6d $%10s $%10s %7d %5.1f%% %5.2f %11.1f%%" % (
        r["year"], "{:,.2f}".format(r["final"]), "{:,.2f}".format(r["net"]),
        r["trades"], r["wr"], r["pf"], r["dd"]))
total_final = sum(r["final"] for r in all_results)
total_net = sum(r["net"] for r in all_results)
total_trades = sum(r["trades"] for r in all_results)
print("  " + "-" * 70)
print("  %-6s $%10s $%10s %7d" % ("TOTAL", "{:,.2f}".format(total_final), "{:,.2f}".format(total_net), total_trades))
print()
print("=" * 70)
