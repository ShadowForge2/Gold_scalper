"""Full pipeline backtest using production models (trained 2022-2025).
Tests on 2025-2026, $20 start. Pipeline: Chop → SwingQuality → ASP → DirectionPredictor."""
import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
os.chdir(base)

from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features, ASP_FEATURE_COLS
from app.asp_predictor import ASPPredictor
from app.swing_quality_predictor import SwingQualityPredictor
from app.direction_predictor import DirectionPredictor, compute_features, compute_chop_score

SYMBOLS = ["XAUUSD", "US100"]

SYMBOL_CFG = {
    "XAUUSD": {
        "cache_dir": "data/dukascopy",
        "asp_model": "models/asp_swing_xgb_m5.joblib",
        "asp_features": "models/asp_swing_m5_features.npy",
        "sq_model": "models/swing_quality_xgb.json",
        "dp_model": "models/direction_xgb_m5_XAUUSD.joblib",
        "min_atr": 0.50, "sl_atr": 2.0, "tp_atr": 1.0,
        "timeout_bars": 10, "spread": 0.35,
    },
    "US100": {
        "cache_dir": "data/dukascopy_us100",
        "asp_model": "models/asp_swing_xgb_m5_US100.joblib",
        "asp_features": "models/asp_swing_m5_features_US100.npy",
        "sq_model": "models/swing_quality_xgb_US100.json",
        "dp_model": "models/direction_xgb_m5_US100.joblib",
        "min_atr": 5.0, "sl_atr": 2.0, "tp_atr": 1.0,
        "timeout_bars": 10, "spread": 1.5,
    },
}

ATR_PERIOD = 14
STARTING_BALANCE = 20.0
BASE_LOT = 0.02
MAX_LOT = 10.0
LOT_STEP = 0.01
TRAIL_TRIGGER = 1.5
TRAIL_RETRACE = 2.5

CHOP_THRESHOLD = 0.70
SQ_THRESHOLD = 0.40
ASP_MIN_CONF = 0.65
DP_CONF_THRESHOLD = 0.60


def load_and_precompute(sym, cfg, year):
    client = DukascopyClient(symbol=sym, cache_dir=cfg["cache_dir"])
    m1 = client.download_year(year)
    if m1 is None or len(m1) == 0:
        return None
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    m5 = m5.set_index("time") if "time" in m5.columns else m5
    h1 = h1.set_index("time") if "time" in h1.columns else h1
    m5 = m5[~m5.index.duplicated(keep="first")].sort_index()
    h1 = h1[~h1.index.duplicated(keep="first")].sort_index()

    print(f"    Loading models...", end=" ", flush=True)
    asp_pred = ASPPredictor(cfg["asp_model"], cfg["asp_features"])
    sq_pred = SwingQualityPredictor(cfg["sq_model"])
    dp_pred = DirectionPredictor(cfg["dp_model"])

    print(f"Computing features...", end=" ", flush=True)
    asp_feats = compute_asp_features(m5, h1)
    n = len(asp_feats)

    asp_dir = np.zeros(n, dtype=np.int8)
    asp_conf = np.zeros(n, dtype=np.float32)
    sq_prob = np.full(n, np.nan, dtype=np.float32)
    dp_dir = np.zeros(n, dtype=np.int8)

    print(f"Vectorized predictions...", end=" ", flush=True)
    feat_arr = asp_feats.values.astype(np.float32)
    valid = ~(np.isnan(feat_arr).any(axis=1) | np.isinf(feat_arr).any(axis=1))
    feat_clean = np.where(valid[:, None], feat_arr, 0)

    if sq_pred.ready and sq_pred.model is not None:
        sq_proba = sq_pred.model.predict_proba(feat_clean)
        sq_prob = sq_proba[:, 1] if sq_proba.shape[1] > 1 else sq_proba[:, 0]
    sq_prob[~valid] = np.nan

    if asp_pred.ready and asp_pred.model is not None:
        raw = asp_pred.model.predict(feat_clean)
        asp_dir = np.array([asp_pred.label_map.get(r, 0) for r in raw], dtype=np.int8)
        raw_proba = asp_pred.model.predict_proba(feat_clean)
        asp_conf = raw_proba.max(axis=1)
    asp_dir[~valid] = 0

    print(f"DP...", end=" ", flush=True)
    dp_feats = compute_features(m5, h1)
    if dp_feats is not None and len(dp_feats) == n:
        dp_arr = dp_feats.values.astype(np.float32)
        dp_valid = ~(np.isnan(dp_arr).any(axis=1) | np.isinf(dp_arr).any(axis=1))
        dp_clean = np.where(dp_valid[:, None], dp_arr, 0)
        dp_classes = list(dp_pred.model.classes_)
        dp_proba = dp_pred.model.predict_proba(dp_clean)
        dp_best = dp_proba.argmax(axis=1)
        if 1 in dp_classes:
            up_idx = dp_classes.index(1)
            up_prob = dp_proba[:, up_idx]
        else:
            up_prob = np.zeros(n)
        if 0 in dp_classes:
            dn_idx = dp_classes.index(0)
            dn_prob = dp_proba[:, dn_idx]
        else:
            dn_prob = np.zeros(n)
        for i in range(n):
            if not dp_valid[i]:
                continue
            best_class = dp_classes[dp_best[i]]
            if best_class == 1 and up_prob[i] >= DP_CONF_THRESHOLD:
                dp_dir[i] = 1
            elif best_class == 0 and dn_prob[i] >= DP_CONF_THRESHOLD:
                dp_dir[i] = -1

    print(f"ATR...", end=" ", flush=True)
    m5_c = m5["close"].values.astype(np.float64)
    m5_h = m5["high"].values.astype(np.float64)
    m5_l = m5["low"].values.astype(np.float64)
    tr = np.maximum(m5_h - m5_l,
                    np.maximum(np.abs(m5_h - np.concatenate([[m5_c[0]], m5_c[:-1]])),
                               np.abs(m5_l - np.concatenate([[m5_c[0]], m5_c[:-1]]))))
    atr = pd.Series(tr).rolling(14, min_periods=2).mean().values

    print(f"Chop...", end=" ")
    chop = compute_chop_score(m5)
    chop_arr = chop.values if chop is not None else np.full(n, np.nan)

    print(f"({n} bars)")
    return {
        "idx": asp_feats.index, "m5_c": m5_c, "m5_h": m5_h, "m5_l": m5_l,
        "atr": atr, "asp_dir": asp_dir, "asp_conf": asp_conf,
        "sq_prob": sq_prob, "dp_dir": dp_dir, "chop": chop_arr,
    }


def run_backtest(test_years):
    t_start = time.time()
    print(f"\n{'=' * 70}")
    print(f"  FULL PIPELINE BACKTEST")
    print(f"  Models: trained on 2022-2025 | Test: {test_years[0]}-{test_years[-1]}")
    print(f"  Equity: ${STARTING_BALANCE} | Pipeline: Chop->SQ->ASP->DP")
    print(f"{'=' * 70}")

    year_data = {}
    for sym in SYMBOLS:
        cfg = SYMBOL_CFG[sym]
        for y in test_years:
            print(f"\n  [{sym}] {y}...")
            d = load_and_precompute(sym, cfg, y)
            if d is not None:
                year_data[f"{sym}_{y}"] = d

    events = []
    for key, d in year_data.items():
        sym = key.split("_")[0]
        for i in range(ATR_PERIOD + 5, len(d["idx"])):
            events.append((d["idx"][i], sym, i, key))
    events.sort(key=lambda x: x[0])
    print(f"\n  Events: {len(events)}")

    state = {sym: {} for sym in SYMBOLS}
    for sym in SYMBOLS:
        state[sym] = {"in_trade": False, "entry_price": 0.0, "entry_dir": "",
                      "sl": 0.0, "tp": 0.0, "lot": 0.0, "entry_time": None,
                      "bars": 0, "best": 0.0}

    balance = STARTING_BALANCE
    peak = STARTING_BALANCE
    trades = []

    for ts, sym, bar_i, key in events:
        if balance <= 0:
            break
        d = year_data[key]
        atr = d["atr"][bar_i]
        atr = atr if np.isfinite(atr) and atr > 0 else 1.0
        st = state[sym]

        if st["in_trade"]:
            st["bars"] += 1
            cfg = SYMBOL_CFG[sym]
            hit_sl = hit_tp = False
            if st["entry_dir"] == "BUY":
                st["best"] = max(st["best"], d["m5_h"][bar_i])
                if st["best"] >= st["entry_price"] + atr * TRAIL_TRIGGER:
                    st["sl"] = max(st["sl"], st["best"] - atr * TRAIL_RETRACE)
                if d["m5_l"][bar_i] <= st["sl"]:
                    hit_sl = True
                elif d["m5_h"][bar_i] >= st["tp"]:
                    hit_tp = True
            else:
                st["best"] = min(st["best"], d["m5_l"][bar_i])
                if st["best"] <= st["entry_price"] - atr * TRAIL_TRIGGER:
                    st["sl"] = min(st["sl"], st["best"] + atr * TRAIL_RETRACE)
                if d["m5_h"][bar_i] >= st["sl"]:
                    hit_sl = True
                elif d["m5_l"][bar_i] <= st["tp"]:
                    hit_tp = True

            if hit_sl or hit_tp:
                ep = st["sl"] if hit_sl else st["tp"]
                pnl = (ep - st["entry_price"]) * st["lot"] - cfg["spread"] * st["lot"]
                if st["entry_dir"] == "SELL":
                    pnl = (st["entry_price"] - ep) * st["lot"] - cfg["spread"] * st["lot"]
                balance += pnl
                peak = max(peak, balance)
                trades.append({"entry_time": st["entry_time"], "exit_time": ts,
                               "dir": st["entry_dir"], "entry": st["entry_price"], "exit": ep,
                               "reason": "sl" if hit_sl else "tp", "pnl": pnl,
                               "lot": st["lot"], "balance": balance, "bars": st["bars"],
                               "hold_min": st["bars"] * 5, "symbol": sym})
                st["in_trade"] = False
                st["bars"] = 0
                continue

            if st["bars"] >= cfg["timeout_bars"]:
                ep = d["m5_c"][bar_i]
                pnl = (ep - st["entry_price"]) * st["lot"] - cfg["spread"] * st["lot"]
                if st["entry_dir"] == "SELL":
                    pnl = (st["entry_price"] - ep) * st["lot"] - cfg["spread"] * st["lot"]
                balance += pnl
                peak = max(peak, balance)
                trades.append({"entry_time": st["entry_time"], "exit_time": ts,
                               "dir": st["entry_dir"], "entry": st["entry_price"], "exit": ep,
                               "reason": "timeout", "pnl": pnl,
                               "lot": st["lot"], "balance": balance, "bars": st["bars"],
                               "hold_min": st["bars"] * 5, "symbol": sym})
                st["in_trade"] = False
                st["bars"] = 0
                continue

        else:
            chop = d["chop"][bar_i]
            if np.isfinite(chop) and chop > CHOP_THRESHOLD:
                continue
            sq = d["sq_prob"][bar_i]
            if np.isnan(sq) or sq < SQ_THRESHOLD:
                continue
            asp_d = d["asp_dir"][bar_i]
            asp_c = d["asp_conf"][bar_i]
            if asp_d == 0 or asp_c < ASP_MIN_CONF:
                continue
            dp_d = d["dp_dir"][bar_i]
            if dp_d != 0 and dp_d != asp_d:
                continue
            if atr < SYMBOL_CFG[sym]["min_atr"]:
                continue

            direction = "BUY" if asp_d == 1 else "SELL"
            cfg = SYMBOL_CFG[sym]
            lot = BASE_LOT * (balance / STARTING_BALANCE)
            dd_pct = (peak - balance) / peak * 100 if peak > 0 else 0
            if dd_pct > 15:
                lot *= 0.5
            lot = round(lot / LOT_STEP) * LOT_STEP
            lot = max(LOT_STEP, min(lot, MAX_LOT))

            entry = d["m5_c"][bar_i]
            sl_dist = atr * cfg["sl_atr"]
            tp_dist = atr * cfg["tp_atr"]
            st["entry_price"] = entry
            st["sl"] = entry - sl_dist if direction == "BUY" else entry + sl_dist
            st["tp"] = entry + tp_dist if direction == "BUY" else entry - tp_dist
            st["lot"] = lot
            st["in_trade"] = True
            st["entry_dir"] = direction
            st["entry_time"] = ts
            st["bars"] = 0
            st["best"] = entry

    for sym in SYMBOLS:
        st = state[sym]
        if st["in_trade"]:
            last_key = f"{sym}_{test_years[-1]}"
            if last_key in year_data:
                d = year_data[last_key]
                ep = d["m5_c"][-1]
                cfg = SYMBOL_CFG[sym]
                pnl = (ep - st["entry_price"]) * st["lot"] - cfg["spread"] * st["lot"]
                if st["entry_dir"] == "SELL":
                    pnl = (st["entry_price"] - ep) * st["lot"] - cfg["spread"] * st["lot"]
                balance += pnl
                trades.append({"entry_time": st["entry_time"], "exit_time": d["idx"][-1],
                               "dir": st["entry_dir"], "entry": st["entry_price"], "exit": ep,
                               "reason": "eoy", "pnl": pnl, "lot": st["lot"],
                               "balance": balance, "bars": st["bars"],
                               "hold_min": st["bars"] * 5, "symbol": sym})

    df = pd.DataFrame(trades)
    elapsed = time.time() - t_start

    if len(df) == 0:
        print("\n  No trades!")
        return

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["entry_time"].dt.year
    df["month_name"] = df["entry_time"].dt.strftime("%Y-%m")

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")

    for sym in SYMBOLS:
        sdf = df[df["symbol"] == sym]
        if len(sdf) == 0:
            continue
        n = len(sdf)
        wins = (sdf["pnl"] > 0).sum()
        net = sdf["pnl"].sum()
        gp = sdf.loc[sdf["pnl"] > 0, "pnl"].sum()
        gl = abs(sdf.loc[sdf["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 99.9
        wr = wins / n * 100
        print(f"\n  {sym}: {n} trades, WR={wr:.1f}%, PF={pf:.2f}, Net=${net:,.2f}")
        avg_win = gp / wins if wins > 0 else 0
        avg_loss = gl / (n - wins) if n - wins > 0 else 0
        print(f"         Avg Win=${avg_win:.2f} Avg Loss=${avg_loss:.2f}")

    months = sorted(df["month_name"].unique())
    print(f"\n  {'Month':<12} {'Trades':>7} {'WR%':>7} {'Net':>12} {'Balance':>12}")
    print(f"  {'-' * 52}")
    for mn in months:
        mc = df[df["month_name"] == mn]
        tot = mc["pnl"].sum()
        mw = (mc["pnl"] > 0).sum()
        mwr = mw / len(mc) * 100
        mbal = mc["balance"].iloc[-1]
        print(f"  {mn:<12} {len(mc):>7} {mwr:>6.1f}% ${tot:>8,.2f} ${mbal:>8,.2f}")

    total_trades = len(df)
    wins = (df["pnl"] > 0).sum()
    losses = total_trades - wins
    net = df["pnl"].sum()
    gp = df.loc[df["pnl"] > 0, "pnl"].sum()
    gl = abs(df.loc[df["pnl"] <= 0, "pnl"].sum())
    pf = gp / gl if gl > 0 else 99.9
    wr = wins / total_trades * 100
    max_dd = 0
    pk = STARTING_BALANCE
    for _, r in df.iterrows():
        pk = max(pk, r["balance"])
        dd = (pk - r["balance"]) / pk * 100
        max_dd = max(max_dd, dd)

    results = (df["pnl"] > 0).astype(int).values
    max_ws = max_ls = cw = cl = 0
    for r in results:
        if r == 1:
            cw += 1; cl = 0
            max_ws = max(max_ws, cw)
        else:
            cl += 1; cw = 0
            max_ls = max(max_ls, cl)

    print(f"\n  {'=' * 50}")
    print(f"  SUMMARY")
    print(f"  {'=' * 50}")
    print(f"    Period:       {test_years[0]}-{test_years[-1]}")
    print(f"    Starting:     ${STARTING_BALANCE:.2f}")
    print(f"    Final:        ${balance:,.2f}")
    print(f"    Net PnL:      ${net:,.2f} ({net/STARTING_BALANCE*100:.0f}%)")
    print(f"    Max DD:       {max_dd:.1f}%")
    print(f"    Trades:       {total_trades}")
    print(f"    Wins/Losses:  {wins}/{losses} ({wr:.1f}%)")
    print(f"    PF:           {pf:.2f}")
    print(f"    Win Streak:   {max_ws}")
    print(f"    Loss Streak:  {max_ls}")
    print(f"    Runtime:      {elapsed:.0f}s")
    print(f"  {'=' * 50}")

    csv_name = f"bt_full_{test_years[0]}_{test_years[-1]}.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n  Saved: {csv_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2026", help="Test years (comma-sep, e.g. 2025,2026)")
    args = parser.parse_args()
    test_yrs = [int(y.strip()) for y in args.years.split(",")]
    run_backtest(test_yrs)
