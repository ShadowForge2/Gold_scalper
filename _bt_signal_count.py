"""Show signal counts and cumulative PnL from start to $100 milestone (trade #361)."""
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
    total_sig = buy_n + sell_n
    print("%d bars, BUY=%d SELL=%d Total=%d [%.1fs]" % (len(m5), buy_n, sell_n, total_sig, time.time()-t0))
    return {"m5": m5, "m5_c": m5_c, "m5_h": m5_h, "m5_l": m5_l,
            "atr": atr_arr, "signals": signal_series.values, "times": m5.index.values}

print("=" * 80)
print("  SIGNAL COUNT: START TO $100 MILESTONE")
print("=" * 80)
print("\n  Loading data...")
data = {}
for sym, cfg in SYMBOLS.items():
    data[sym] = load_symbol_data(sym, cfg)

# Count total signals generated per symbol across entire year
print("\n  Total signals generated in 2025:")
for sym in SYMBOLS:
    d = data[sym]
    sigs = d["signals"]
    buys = (sigs == 1).sum()
    sells = (sigs == -1).sum()
    print("    %s: BUY=%d  SELL=%d  Total=%d" % (sym, buys, sells, buys+sells))

# Now run backtest and track signals used up to trade #361
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
signals_seen = {"XAUUSD": {"BUY": 0, "SELL": 0, "filtered": 0},
                "US100": {"BUY": 0, "SELL": 0, "filtered": 0}}

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
            trades.append({"n": len(trades)+1, "time": m5_times[bar_i], "sym": sym,
                           "dir": st["entry_dir"], "pnl": pnl, "lot": st["entry_lot"],
                           "reason": "sl" if hit_sl else "tp", "balance": balance})
            st["in_trade"] = False
            st["ticks"] = 0
        elif st["ticks"] >= TIMEOUT_BARS:
            if st["entry_dir"] == "BUY":
                pnl = (m5_c[bar_i] - st["entry_price"]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
            else:
                pnl = (st["entry_price"] - m5_c[bar_i]) * st["entry_lot"] * cfg["contract_size"] - cfg["spread_pts"] * st["entry_lot"] * cfg["contract_size"]
            balance += pnl
            peak_balance = max(peak_balance, balance)
            trades.append({"n": len(trades)+1, "time": m5_times[bar_i], "sym": sym,
                           "dir": st["entry_dir"], "pnl": pnl, "lot": st["entry_lot"],
                           "reason": "timeout", "balance": balance})
            st["in_trade"] = False
            st["ticks"] = 0
    else:
        d_sig = asp_arr[bar_i]
        if d_sig != 0:
            # Signal was generated
            direction = "BUY" if d_sig == 1 else "SELL"
            if d_sig == 1:
                signals_seen[sym]["BUY"] += 1
            else:
                signals_seen[sym]["SELL"] += 1

            atr_val = atr_arr[bar_i]
            if np.isnan(atr_val) or atr_val <= 0 or atr_val < cfg["min_atr"]:
                signals_seen[sym]["filtered"] += 1
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

    if len(trades) >= 361:
        break

df = pd.DataFrame(trades)

# Find the exact trade #361
target = df[df["n"] == 361].iloc[0] if len(df) >= 361 else df.iloc[-1]
print()
print("=" * 80)
print("  FROM START TO $100 MILESTONE (trade #%d)" % int(target["n"]))
print("=" * 80)
print()
print("  Date range: %s to %s" % (
    pd.to_datetime(df.iloc[0]["time"]).strftime("%Y-%m-%d %H:%M"),
    pd.to_datetime(target["time"]).strftime("%Y-%m-%d %H:%M")))
print("  Final balance: $%s" % "{:,.2f}".format(target["balance"]))
print()

print("  Signals USED (entered trade):")
for sym in SYMBOLS:
    sdf = df[df["sym"] == sym]
    buys = len(sdf[sdf["dir"] == "BUY"])
    sells = len(sdf[sdf["dir"] == "SELL"])
    print("    %s: BUY=%d  SELL=%d  Total=%d" % (sym, buys, sells, len(sdf)))

print()
print("  Signals GENERATED by ML model (entire year):")
for sym in SYMBOLS:
    print("    %s: BUY=%d  SELL=%d  Total=%d  (filtered by ATR: %d)" % (
        sym, signals_seen[sym]["BUY"], signals_seen[sym]["SELL"],
        signals_seen[sym]["BUY"] + signals_seen[sym]["SELL"],
        signals_seen[sym]["filtered"]))

total_used = len(df)
print()
print("  Trade results in this period:")
wins = len(df[df["pnl"] > 0])
losses = len(df[df["pnl"] <= 0])
print("    Wins: %d (%.1f%%)" % (wins, wins/total_used*100))
print("    Losses: %d (%.1f%%)" % (losses, losses/total_used*100))
print("    Net PnL: $%s" % "{:,.2f}".format(target["balance"] - STARTING_BALANCE))

# Daily breakdown
df["date"] = pd.to_datetime(df["time"]).dt.date
print()
print("  Daily breakdown to $100:")
print("  %-12s %6s %6s %6s %12s %12s" % ("Date", "Trades", "Wins", "Losses", "PnL", "Balance"))
print("  " + "-" * 60)
for date, grp in df.groupby("date"):
    w = len(grp[grp["pnl"] > 0])
    l = len(grp) - w
    pnl = grp["pnl"].sum()
    bal = grp["balance"].iloc[-1]
    print("  %-12s %6d %6d %6d $%10s $%10s" % (str(date), len(grp), w, l,
        "{:,.2f}".format(pnl), "{:,.2f}".format(bal)))

print()
print("=" * 80)
