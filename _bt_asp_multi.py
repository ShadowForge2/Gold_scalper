"""ASP Backtest 2023-2025 with timeout=6, unlimited MAX_LOT."""
import os, sys, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features
from app.asp_labels import compute_atr

ASP_MODEL_PATH = "models/asp_swing_xgb_m5.joblib"
ASP_FEATURE_PATH = "models/asp_swing_m5_features.npy"

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.0
TIMEOUT_BARS = 6
ATR_PERIOD = 14
STARTING_BALANCE = 20.0
BASE_LOT = 0.01
CONTRACT_SIZE = 100
SPREAD_PTS = 0.30
LOT_STEP = 0.01
MIN_LOT = 0.01
MAX_LOT = 9999.0

YEARS = [2023, 2024, 2025]


def calc_lot(balance, peak_balance):
    reference = 20.0
    lot = BASE_LOT * (balance / reference)
    if peak_balance and balance < peak_balance * 0.9:
        lot *= 0.5
    lot = round(lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(lot, MAX_LOT))


asp_saved = joblib.load(ASP_MODEL_PATH)
asp_model = asp_saved["model"]
label_map = asp_saved["label_map"]
inv_map = {v: k for k, v in label_map.items()}
asp_features = list(np.load(ASP_FEATURE_PATH, allow_pickle=True))
print(f"ASP model loaded ({len(asp_features)} features)")

client = DukascopyClient()
all_results = []

for YEAR in YEARS:
    print(f"\n{'='*70}")
    print(f"  ASP BACKTEST {YEAR} (timeout={TIMEOUT_BARS}, no lot cap)")
    print(f"{'='*70}")

    t0 = time.time()
    m1 = client.download_year(YEAR)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    print(f"  M1={len(m1)}, M5={len(m5)}, H1={len(h1)} ({time.time()-t0:.1f}s)")

    m5 = m5.set_index("time")
    h1 = h1.set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    n = len(m5)
    m5_h = m5["high"].values.astype(np.float64)
    m5_l = m5["low"].values.astype(np.float64)
    m5_c = m5["close"].values.astype(np.float64)
    atr_arr = compute_atr(m5_h, m5_l, m5_c, ATR_PERIOD)

    t_f = time.time()
    features = compute_asp_features(m5, h1)
    print(f"  Features: {len(features)} rows ({time.time()-t_f:.1f}s)")

    valid = features[asp_features].dropna()
    signal_series = pd.Series(0, index=features.index, dtype=np.int8)
    if len(valid) > 0:
        feat_arr = valid[asp_features].values
        raw_preds = asp_model.predict(feat_arr)
        preds = np.array([inv_map[p] for p in raw_preds])
        signal_series.loc[valid.index] = preds
    signal_arr = signal_series.values

    buy_n = (signal_arr == 1).sum()
    sell_n = (signal_arr == -1).sum()
    print(f"  BUY={buy_n}, SELL={sell_n}")

    trades = []
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    entry_dir = ""
    sl_price = 0.0
    tp_price = 0.0
    entry_lot = 0.01
    ticks_in_trade = 0

    START_IDX = max(ATR_PERIOD + 5, 100)

    for i in range(START_IDX, n):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        if in_trade:
            ticks_in_trade += 1

            if entry_dir == "BUY" and m5_l[i] <= sl_price:
                pnl = (sl_price - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": sl_price,
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "sl_hit", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

            if entry_dir == "SELL" and m5_h[i] >= sl_price:
                pnl = (entry_price - sl_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": sl_price,
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "sl_hit", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

            if entry_dir == "BUY" and m5_h[i] >= tp_price:
                pnl = (tp_price - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": tp_price,
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "tp_hit", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

            if entry_dir == "SELL" and m5_l[i] <= tp_price:
                pnl = (entry_price - tp_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": tp_price,
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "tp_hit", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

            if ticks_in_trade >= TIMEOUT_BARS:
                if entry_dir == "BUY":
                    pnl = (m5_c[i] - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                else:
                    pnl = (entry_price - m5_c[i]) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": round(m5_c[i], 2),
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "timeout", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

        else:
            d = signal_arr[i]
            if d != 0:
                atr_val = atr_arr[i]
                if np.isnan(atr_val) or atr_val <= 0:
                    continue
                if atr_val < 0.50:
                    continue

                direction = "BUY" if d == 1 else "SELL"
                entry_price = m5_c[i]
                sl_dist = atr_val * SL_ATR_MULT
                tp_dist = atr_val * TP_ATR_MULT

                if direction == "BUY":
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                entry_lot = calc_lot(balance, peak_balance)
                in_trade = True
                entry_idx = i
                entry_dir = direction
                ticks_in_trade = 0

    if in_trade:
        if entry_dir == "BUY":
            pnl = (m5_c[-1] - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        else:
            pnl = (entry_price - m5_c[-1]) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        balance += pnl
        peak_balance = max(peak_balance, balance)
        trades.append({
            "entry_idx": entry_idx, "exit_idx": n - 1,
            "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[-1],
            "direction": entry_dir, "entry": entry_price, "exit": round(m5_c[-1], 2),
            "pnl_usd": round(pnl, 2), "lot": entry_lot,
            "reason": "eoy_close", "bars": ticks_in_trade,
        })

    df = pd.DataFrame(trades)
    if len(df) == 0:
        print("  No trades")
        continue

    wins = (df["pnl_usd"] > 0).sum()
    losses = (df["pnl_usd"] <= 0).sum()
    wr = wins / len(df) * 100
    gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = abs(df.loc[df["pnl_usd"] <= 0, "pnl_usd"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    net_pnl = df["pnl_usd"].sum()

    reasons = df.groupby("reason").agg(
        count=("pnl_usd", "size"),
        wins=("pnl_usd", lambda x: (x > 0).sum()),
        total_pnl=("pnl_usd", "sum"),
    ).reset_index()
    reasons["wr"] = reasons["wins"] / reasons["count"] * 100

    max_lot_used = df["lot"].max()
    avg_lot = df["lot"].mean()

    print(f"\n  Starting balance: ${STARTING_BALANCE:.2f}  |  SL: {SL_ATR_MULT}x ATR  |  TP: {TP_ATR_MULT}x ATR  |  Timeout: {TIMEOUT_BARS} bars")
    print(f"  Ending balance:   ${balance:,.2f}")
    print(f"  Net PnL:          ${net_pnl:,.2f}")
    print(f"  Total trades:     {len(df)}")
    print(f"  Wins / Losses:    {wins} / {losses}")
    print(f"  Win rate:         {wr:.1f}%")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Max lot used:     {max_lot_used:.2f}")
    print(f"  Avg lot:          {avg_lot:.2f}")
    print(f"\n  Exit breakdown:")
    for _, row in reasons.iterrows():
        print(f"    {row['reason']:>10}: {row['count']:>4} trades, WR={row['wr']:.1f}%, PnL=${row['total_pnl']:>12,.2f}")

    all_results.append({
        "year": YEAR, "balance": balance, "net_pnl": net_pnl,
        "trades": len(df), "wins": wins, "losses": losses,
        "wr": wr, "pf": pf, "max_lot": max_lot_used, "avg_lot": avg_lot,
    })

print(f"\n{'='*70}")
print(f"  MULTI-YEAR SUMMARY")
print(f"{'='*70}")
print(f"{'Year':>6} {'Balance':>14} {'Net PnL':>14} {'Trades':>8} {'WR%':>7} {'PF':>7} {'MaxLot':>8}")
print(f"{'-'*6} {'-'*14} {'-'*14} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")

for r in all_results:
    print(f"{r['year']:>6} ${r['balance']:>12,.2f} ${r['net_pnl']:>12,.2f} {r['trades']:>8} {r['wr']:>6.1f}% {r['pf']:>6.2f} {r['max_lot']:>7.2f}")

total_pnl = sum(r["net_pnl"] for r in all_results)
total_trades = sum(r["trades"] for r in all_results)
avg_wr = sum(r["wins"] for r in all_results) / total_trades * 100
print(f"{'TOTAL':>6} ${all_results[-1]['balance']:>12,.2f} ${total_pnl:>12,.2f} {total_trades:>8} {avg_wr:>6.1f}%")
