"""
Backtest: ASP Swing Probability ML system on M5 data.

Entry: ASP model predicts BUY (swing low) or SELL (swing high).
SL:    2x ATR (matching label definition).
TP:    1x ATR (matching label definition).
Exit:  Also allow exit after 12 bars (timeout, matching forward_bars).
Lot:   EquityScaler (0.01 base, scales with balance).
"""
import os, sys, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features, ASP_FEATURE_COLS
from app.asp_labels import compute_atr

# ── Config ──
YEARS = [2022, 2023, 2024, 2025]
ASP_MODEL_PATH = "models/asp_swing_xgb_m5.joblib"
ASP_FEATURE_PATH = "models/asp_swing_m5_features.npy"
SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.0
TIMEOUT_BARS = 12
ATR_PERIOD = 14
STARTING_BALANCE = 20.0
BASE_LOT = 0.01
CONTRACT_SIZE = 100
SPREAD_PTS = 0.30
LOT_STEP = 0.01
MIN_LOT = 0.01
MAX_LOT = 1.0

# ── Load ASP model ──
print("Loading ASP model...")
asp_saved = joblib.load(ASP_MODEL_PATH)
asp_model = asp_saved["model"]
label_map = asp_saved["label_map"]
inv_map = {v: k for k, v in label_map.items()}
asp_features = list(np.load(ASP_FEATURE_PATH, allow_pickle=True))
print(f"  ASP model: OK ({len(asp_features)} features)")


def calc_lot(balance, peak_balance):
    reference = 20.0
    lot = BASE_LOT * (balance / reference)
    if peak_balance and balance < peak_balance * 0.9:
        lot *= 0.5
    lot = round(lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(lot, MAX_LOT))


client = DukascopyClient()
all_results = []

for YEAR in YEARS:
    print(f"\n{'='*70}")
    print(f"  ASP BACKTEST -- {YEAR}")
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

    # Compute ASP features
    print("  Computing ASP features...")
    t_f = time.time()
    features = compute_asp_features(m5, h1)
    print(f"  Features: {len(features)} rows ({time.time()-t_f:.1f}s)")

    # Batch predict — features.index is aligned to m5.index
    print("  Batch-predicting...")
    t_p = time.time()
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
    print(f"  Predict: {time.time()-t_p:.1f}s, BUY={buy_n}, SELL={sell_n}")

    # ── Backtest loop ──
    print("  Running backtest...")
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

            # SL hit
            if entry_dir == "BUY" and m5_l[i] <= sl_price:
                exit_px = sl_price
                pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "sl_hit", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

            if entry_dir == "SELL" and m5_h[i] >= sl_price:
                exit_px = sl_price
                pnl = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "sl_hit", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

            # TP hit
            if entry_dir == "BUY" and m5_h[i] >= tp_price:
                exit_px = tp_price
                pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "tp_hit", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

            if entry_dir == "SELL" and m5_l[i] <= tp_price:
                exit_px = tp_price
                pnl = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": "tp_hit", "bars": ticks_in_trade,
                })
                in_trade = False
                ticks_in_trade = 0
                continue

            # Timeout
            if ticks_in_trade >= TIMEOUT_BARS:
                exit_px = m5_c[i]
                if entry_dir == "BUY":
                    pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                else:
                    pnl = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": entry_dir, "entry": entry_price, "exit": exit_px,
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

                if sl_dist < 0.50:
                    continue

                entry_lot = calc_lot(balance, peak_balance)
                in_trade = True
                entry_idx = i
                entry_dir = direction
                ticks_in_trade = 0

    # Close open trade at end
    if in_trade:
        exit_px = m5_c[-1]
        if entry_dir == "BUY":
            pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        else:
            pnl = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        balance += pnl
        peak_balance = max(peak_balance, balance)
        trades.append({
            "entry_idx": entry_idx, "exit_idx": n - 1,
            "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[-1],
            "direction": entry_dir, "entry": entry_price, "exit": exit_px,
            "pnl_usd": round(pnl, 2), "lot": entry_lot,
            "reason": "eoy_close", "bars": ticks_in_trade,
        })

    # ── Results ──
    df = pd.DataFrame(trades)
    print(f"\n  Starting balance: ${STARTING_BALANCE:.2f}  |  SL: {SL_ATR_MULT}x ATR  |  TP: {TP_ATR_MULT}x ATR  |  Timeout: {TIMEOUT_BARS} bars")

    if len(df) == 0:
        print("  NO TRADES")
        all_results.append({"year": YEAR, "trades": 0, "balance": balance, "pnl": 0, "wr": 0, "pf": 0})
    else:
        wins = df[df["pnl_usd"] > 0]
        losses = df[df["pnl_usd"] <= 0]
        wr = len(wins) / len(df) * 100
        avg_win = wins["pnl_usd"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_usd"].mean() if len(losses) > 0 else 0
        gross_profit = wins["pnl_usd"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl_usd"].sum()) if len(losses) > 0 else 0
        pf = gross_profit / max(gross_loss, 0.01)
        net_pnl = df["pnl_usd"].sum()
        avg_bars = df["bars"].mean()

        print(f"  Trades: {len(df)}  |  Wins: {len(wins)} ({wr:.1f}%)  |  Losses: {len(losses)} ({100-wr:.1f}%)")
        print(f"  Net PnL: ${net_pnl:+.2f}  |  Ending: ${balance:.2f}  |  Return: {(balance/STARTING_BALANCE - 1)*100:+.1f}%")
        print(f"  PF: {pf:.2f}  |  Avg win: ${avg_win:+.2f}  |  Avg loss: ${avg_loss:+.2f}  |  Avg bars: {avg_bars:.1f}")
        print(f"  Max DD: ${df['pnl_usd'].cumsum().min():.2f}")

        print(f"\n  Exit reasons:")
        for reason, group in df.groupby("reason"):
            wr_r = (group["pnl_usd"] > 0).sum() / len(group) * 100
            print(f"    {reason:12s}: {len(group):4d} trades, WR={wr_r:.1f}%, PnL=${group['pnl_usd'].sum():+.2f}")

        print(f"\n  By direction:")
        for d, group in df.groupby("direction"):
            wr_d = (group["pnl_usd"] > 0).sum() / len(group) * 100
            print(f"    {d:4s}: {len(group):4d} trades, WR={wr_d:.1f}%, PnL=${group['pnl_usd'].sum():+.2f}")

        df["month"] = pd.to_datetime(df["entry_ts"]).dt.to_period("M")
        print(f"\n  Monthly:")
        for m, group in df.groupby("month"):
            wr_m = (group["pnl_usd"] > 0).sum() / len(group) * 100
            print(f"    {str(m):8s}: {len(group):4d} trades, WR={wr_m:.1f}%, PnL=${group['pnl_usd'].sum():+.2f}")

        print(f"\n  First 15 trades:")
        print(f"  {'Dir':>4s} {'Entry':>10s} {'Exit':>10s} {'PnL':>8s} {'Reason':>12s} {'Bars':>5s}")
        for _, t in df.head(15).iterrows():
            print(f"  {t['direction']:>4s} {t['entry']:>10.2f} {t['exit']:>10.2f} {t['pnl_usd']:>+8.2f} {t['reason']:>12s} {t['bars']:>5d}")

        all_results.append({"year": YEAR, "trades": len(df), "balance": balance,
                            "pnl": net_pnl, "wr": wr, "pf": pf})

# ── Summary ──
print(f"\n{'='*70}")
print(f"  MULTI-YEAR SUMMARY")
print(f"{'='*70}")
print(f"  {'Year':>6} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>10} {'Balance':>10}")
print(f"  {'-'*52}")
for r in all_results:
    print(f"  {r['year']:>6} {r['trades']:>8} {r['wr']:>7.1f}% {r['pf']:>8.2f} ${r['pnl']:>+9.2f} ${r['balance']:>9.2f}")
total_pnl = sum(r["pnl"] for r in all_results)
total_trades = sum(r["trades"] for r in all_results)
final_bal = all_results[-1]["balance"] if all_results else STARTING_BALANCE
print(f"  {'-'*52}")
print(f"  {'TOTAL':>6} {total_trades:>8} {'':>8} {'':>8} ${total_pnl:>+9.2f} ${final_bal:>9.2f}")
print(f"\n  Return: {(final_bal/STARTING_BALANCE - 1)*100:+.1f}%")
