"""Find the absolute lowest balance across all 3 years."""
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
    return {"m5": m5, "m5_c": m5_c, "m5_h": m5_h, "m5_l": m5_l,
            "atr": atr_arr, "signals": signal_series.values, "times": m5.index.values}

all_balances = []

for YEAR in [2022, 2023, 2024]:
    print("Running %d..." % YEAR, end=" ", flush=True)
    t0 = time.time()
    data = {}
    for sym, cfg in SYMBOLS.items():
        data[sym] = load_symbol_data(sym, cfg, YEAR)

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
    lowest_bal = STARTING_BALANCE
    lowest_time = None
    lowest_peak = STARTING_BALANCE
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
                if balance < lowest_bal:
                    lowest_bal = balance
                    lowest_time = m5_times[bar_i]
                    lowest_peak = peak_balance
                trades.append({"time": m5_times[bar_i], "sym": sym, "dir": st["entry_dir"],
                               "pnl": pnl, "lot": st["entry_lot"], "reason": "sl" if hit_sl else "tp",
                               "balance": balance, "peak": peak_balance})
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
                if balance < lowest_bal:
                    lowest_bal = balance
                    lowest_time = m5_times[bar_i]
                    lowest_peak = peak_balance
                trades.append({"time": m5_times[bar_i], "sym": sym, "dir": st["entry_dir"],
                               "pnl": pnl, "lot": st["entry_lot"], "reason": "timeout",
                               "balance": balance, "peak": peak_balance})
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

    # EOY
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
            if balance < lowest_bal:
                lowest_bal = balance
                lowest_time = d["times"][bar_i]
                lowest_peak = peak_balance

    dd_from_start = (STARTING_BALANCE - lowest_bal) / STARTING_BALANCE * 100 if lowest_bal < STARTING_BALANCE else 0
    dd_from_peak = (lowest_peak - lowest_bal) / lowest_peak * 100 if lowest_peak > 0 else 0
    all_balances.append({"year": YEAR, "lowest": lowest_bal, "time": lowest_time,
                         "peak": lowest_peak, "dd_from_start": dd_from_start, "dd_from_peak": dd_from_peak,
                         "final": balance})
    print("done [%.1fs]" % (time.time() - t0))

print()
print("=" * 70)
print("  LOWEST BALANCE ACROSS ALL YEARS")
print("=" * 70)
print()
print("  %-6s %12s %12s %12s %8s %8s" % ("Year", "Lowest Bal", "Peak Before", "Final", "DD Peak", "DD Start"))
print("  " + "-" * 70)
for r in all_balances:
    t = pd.to_datetime(r["time"]).strftime("%Y-%m-%d %H:%M") if r["time"] is not None else "N/A"
    print("  %-6d $%10s $%10s $%10s %7.1f%% %7.1f%%" % (
        r["year"], "{:,.2f}".format(r["lowest"]), "{:,.2f}".format(r["peak"]),
        "{:,.2f}".format(r["final"]), r["dd_from_peak"], r["dd_from_start"]))
    print("           at %s" % t)

# Overall worst
worst = min(all_balances, key=lambda x: x["lowest"])
print()
print("  ABSOLUTE WORST: $%s in %d (peak was $%s, DD from peak=%.1f%%)" % (
    "{:,.2f}".format(worst["lowest"]), worst["year"],
    "{:,.2f}".format(worst["peak"]), worst["dd_from_peak"]))
print()
print("  Distance from $0: $%s" % "{:,.2f}".format(worst["lowest"]))
print("  As percentage of $20 start: %.1fx" % (worst["lowest"] / STARTING_BALANCE))
print()
print("=" * 70)
