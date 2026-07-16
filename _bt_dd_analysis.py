"""Find worst drawdown point in the unified backtest. Self-contained."""
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
YEAR = 2025

def calc_lot(bal, peak_bal, cfg, start_bal):
    lot = cfg["base_lot"] * (bal / start_bal)
    if peak_bal and peak_bal > 0:
        dd = (peak_bal - bal) / peak_bal * 100
        if dd > 15:
            lot *= 0.5
    step = cfg["lot_step"]
    lot = round(lot / step) * step
    return max(step, min(lot, cfg["max_lot"]))

def load_symbol_data(sym, cfg):
    print("  Loading %s..." % sym, end=" ", flush=True)
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
    print("%d M5 bars, BUY=%d SELL=%d [%.1fs]" % (len(m5), buy_n, sell_n, time.time()-t0))
    return {"m5": m5, "m5_c": m5_c, "m5_h": m5_h, "m5_l": m5_l,
            "atr": atr_arr, "signals": signal_series.values, "times": m5.index.values}

# Load data
print("=" * 80)
print("  DRAWDOWN ANALYSIS 2025")
print("=" * 80)
print("\n  Loading data...")
data = {}
for sym, cfg in SYMBOLS.items():
    data[sym] = load_symbol_data(sym, cfg)

# Build timeline
events = []
for sym, d in data.items():
    for i in range(ATR_PERIOD + 5, len(d["m5_c"])):
        events.append((d["times"][i], sym, i))
events.sort(key=lambda x: x[0])

# State
state = {}
for sym in SYMBOLS:
    state[sym] = {"in_trade": False, "entry_price": 0.0, "entry_dir": "",
                  "sl_price": 0.0, "tp_price": 0.0, "entry_lot": 0.0,
                  "entry_time": None, "ticks": 0, "best_price": 0.0}

balance = STARTING_BALANCE
peak_balance = STARTING_BALANCE
trades = []
trade_num = 0

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
            trade_num += 1
            peak_balance = max(peak_balance, balance)
            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
            trades.append({"n": trade_num, "time": m5_times[bar_i], "sym": sym, "dir": st["entry_dir"],
                           "pnl": pnl, "lot": st["entry_lot"], "reason": "sl" if hit_sl else "tp",
                           "balance": balance, "peak": peak_balance, "dd": dd})
            st["in_trade"] = False
            st["ticks"] = 0
        elif st["ticks"] >= TIMEOUT_BARS:
            if st["entry_dir"] == "BUY":
                pnl = (m5_c[bar_i] - st["entry_price"]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
            else:
                pnl = (st["entry_price"] - m5_c[bar_i]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
            balance += pnl
            trade_num += 1
            peak_balance = max(peak_balance, balance)
            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
            trades.append({"n": trade_num, "time": m5_times[bar_i], "sym": sym, "dir": st["entry_dir"],
                           "pnl": pnl, "lot": st["entry_lot"], "reason": "timeout",
                           "balance": balance, "peak": peak_balance, "dd": dd})
            st["in_trade"] = False
            st["ticks"] = 0
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

df = pd.DataFrame(trades)
if len(df) == 0:
    print("  No trades!")
    sys.exit(0)
df["time"] = pd.to_datetime(df["time"])

# Analysis
min_idx = df["balance"].idxmin()
min_row = df.loc[min_idx]
print()
print("  WORST DRAWDOWN ANALYSIS")
print("=" * 80)
print()
print("  Lowest balance:  $%s" % "{:,.2f}".format(min_row["balance"]))
print("  At trade #%d on %s" % (int(min_row["n"]), min_row["time"].strftime("%Y-%m-%d %H:%M")))
print("  Peak before:     $%s" % "{:,.2f}".format(min_row["peak"]))
print("  Drawdown:        %.1f%%" % min_row["dd"])
print("  That trade:      %s %s lot=%.2f pnl=$%s (%s)" % (
    min_row["sym"], min_row["dir"], min_row["lot"],
    "{:,.2f}".format(min_row["pnl"]), min_row["reason"]))

neg = df[df["balance"] <= 0]
if len(neg) > 0:
    first_neg = neg.iloc[0]
    print()
    print("  !!! BALANCE WENT NEGATIVE !!!")
    print("  First negative: trade #%d at %s" % (int(first_neg["n"]), first_neg["time"].strftime("%Y-%m-%d %H:%M")))
    print("  Balance: $%s" % "{:,.2f}".format(first_neg["balance"]))
    start_idx = max(0, int(first_neg["n"]) - 15)
    print()
    print("  Trades leading to blow-up:")
    sub = df[(df["n"] >= start_idx) & (df["n"] <= int(first_neg["n"]))]
    for _, row in sub.iterrows():
        w = "W" if row["pnl"] > 0 else "L"
        t = row["time"].strftime("%m-%d %H:%M")
        marker = " <-- NEGATIVE" if row["balance"] <= 0 else ""
        print("    #%-5d %s %-7s %-5s %-8s %s lot=%-7.2f pnl=$%10s bal=$%12s%s" % (
            int(row["n"]), t, row["sym"], row["dir"], row["reason"], w,
            row["lot"], "{:,.2f}".format(row["pnl"]),
            "{:,.2f}".format(row["balance"]), marker))
else:
    print()
    print("  Balance never went below $0")
    print("  Closest: $%s" % "{:,.2f}".format(min_row["balance"]))

# Top 10 worst
print()
print("  Top 10 worst drawdown moments:")
print("  %-6s %-18s %-7s %-5s %6s %12s %14s %7s" % ("Trade", "Time", "Sym", "Dir", "Lot", "PnL", "Balance", "DD%"))
print("  " + "-" * 75)
worst10 = df.nlargest(10, "dd")
for _, row in worst10.iterrows():
    t = row["time"].strftime("%Y-%m-%d %H:%M")
    print("  #%-5d %-18s %-7s %-5s %5.2f $%10s $%12s %5.1f%%" % (
        int(row["n"]), t, row["sym"], row["dir"], row["lot"],
        "{:,.2f}".format(row["pnl"]), "{:,.2f}".format(row["balance"]), row["dd"]))

# Lot sizes near worst
print()
print("  Lot sizes during worst DD period:")
worst_n = int(min_row["n"])
period = df[(df["n"] >= max(1, worst_n - 10)) & (df["n"] <= worst_n)]
for _, row in period.iterrows():
    t = row["time"].strftime("%m-%d %H:%M")
    print("    #%-5d %s %-7s lot=%.2f pnl=$%10s" % (
        int(row["n"]), t, row["sym"], row["lot"], "{:,.2f}".format(row["pnl"])))

# Consecutive losses
print()
print("  Consecutive losses around worst DD:")
window = df[(df["n"] >= max(1, worst_n - 20)) & (df["n"] <= worst_n + 5)]
losses_in_row = 0
max_streak_here = 0
for _, row in window.iterrows():
    if row["pnl"] <= 0:
        losses_in_row += 1
        max_streak_here = max(max_streak_here, losses_in_row)
    else:
        losses_in_row = 0
print("  Max consecutive losses near worst DD: %d" % max_streak_here)

# Balance timeline around worst
print()
print("  Balance timeline near worst DD (trades %d-%d):" % (max(1, worst_n - 8), min(len(df), worst_n + 3)))
sub2 = df[(df["n"] >= max(1, worst_n - 8)) & (df["n"] <= min(len(df), worst_n + 3))]
for _, row in sub2.iterrows():
    t = row["time"].strftime("%m-%d %H:%M")
    w = "W" if row["pnl"] > 0 else "L"
    marker = " *** LOWEST ***" if row["n"] == min_row["n"] else ""
    print("    #%-5d %s %s %-5s lot=%-6.2f pnl=$%10s  bal=$%12s  peak=$%12s%s" % (
        int(row["n"]), t, w, row["dir"], row["lot"],
        "{:,.2f}".format(row["pnl"]), "{:,.2f}".format(row["balance"]),
        "{:,.2f}".format(row["peak"]), marker))

# Broker simulation
print()
print("  Broker margin call simulation (50%% MC, 80%% SO, 100%% blow):")
bal_sim = STARTING_BALANCE
peak_sim = STARTING_BALANCE
mc_done = so_done = False
for _, row in df.iterrows():
    bal_sim = row["balance"]
    peak_sim = max(peak_sim, row["peak"])
    dd_now = (peak_sim - bal_sim) / peak_sim * 100 if peak_sim > 0 else 0
    if dd_now >= 50 and not mc_done:
        print("    MARGIN CALL 50%% DD: trade #%d bal=$%s peak=$%s" % (
            int(row["n"]), "{:,.2f}".format(bal_sim), "{:,.2f}".format(peak_sim)))
        mc_done = True
    if dd_now >= 80 and not so_done:
        print("    STOP-OUT    80%% DD: trade #%d bal=$%s peak=$%s" % (
            int(row["n"]), "{:,.2f}".format(bal_sim), "{:,.2f}".format(peak_sim)))
        so_done = True
    if bal_sim <= 0:
        print("    ACCOUNT BLOWN: trade #%d bal=$%s" % (int(row["n"]), "{:,.2f}".format(bal_sim)))
        break
if not mc_done and not so_done:
    print("    No margin call or stop-out triggered")

print()
print("=" * 80)
