"""
Detailed ASP backtest for 2025 — Full January trade report.
Shows every January trade with entry/exit, lot size, position sizing, PnL.
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
YEAR = 2025
ASP_MODEL_PATH = "models/asp_swing_xgb_m5.joblib"
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

# ── Load model ──
print("Loading ASP model...")
asp_saved = joblib.load(ASP_MODEL_PATH)
asp_model = asp_saved["model"]
label_map = asp_saved["label_map"]
inv_map = {v: k for k, v in label_map.items()}
asp_features = list(np.load("models/asp_swing_m5_features.npy", allow_pickle=True))

def calc_lot(balance, peak_balance):
    reference = 20.0
    lot = BASE_LOT * (balance / reference)
    if peak_balance and balance < peak_balance * 0.9:
        lot *= 0.5
    lot = round(lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(lot, MAX_LOT))

client = DukascopyClient()

print(f"\nLoading {YEAR} data...")
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

print("Computing ASP features...")
features = compute_asp_features(m5, h1)

# Batch predict
print("Batch predicting...")
valid = features[asp_features].dropna()
signal_series = pd.Series(0, index=features.index, dtype=np.int64)
if len(valid) > 0:
    feat_arr = valid[asp_features].values
    raw_preds = asp_model.predict(feat_arr)
    preds = np.array([inv_map[p] for p in raw_preds])
    signal_series.loc[valid.index] = preds
signal_arr = signal_series.values

# ── Backtest loop ──
print("Running backtest...")
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
entry_balance_at_entry = 0.0
ticks_in_trade = 0

START_IDX = max(ATR_PERIOD + 5, 100)

for i in range(START_IDX, n):
    atr = atr_arr[i]
    if np.isnan(atr) or atr <= 0:
        atr = 1.0

    if in_trade:
        ticks_in_trade += 1

        if entry_dir == "BUY" and m5_l[i] <= sl_price:
            exit_px = sl_price
            pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            balance += pnl
            peak_balance = max(peak_balance, balance)
            trades.append({
                "entry_idx": entry_idx, "exit_idx": i,
                "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                "sl": sl_price, "tp": tp_price,
                "pnl_usd": round(pnl, 2), "lot": entry_lot,
                "balance_at_entry": round(entry_balance_at_entry, 2),
                "balance_after": round(balance, 2),
                "reason": "sl_hit", "bars": ticks_in_trade,
                "atr_at_entry": round(atr_arr[entry_idx], 4),
                "sl_dist": round(abs(entry_price - sl_price), 2),
                "tp_dist": round(abs(tp_price - entry_price), 2),
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
                "sl": sl_price, "tp": tp_price,
                "pnl_usd": round(pnl, 2), "lot": entry_lot,
                "balance_at_entry": round(entry_balance_at_entry, 2),
                "balance_after": round(balance, 2),
                "reason": "sl_hit", "bars": ticks_in_trade,
                "atr_at_entry": round(atr_arr[entry_idx], 4),
                "sl_dist": round(abs(entry_price - sl_price), 2),
                "tp_dist": round(abs(tp_price - entry_price), 2),
            })
            in_trade = False
            ticks_in_trade = 0
            continue

        if entry_dir == "BUY" and m5_h[i] >= tp_price:
            exit_px = tp_price
            pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            balance += pnl
            peak_balance = max(peak_balance, balance)
            trades.append({
                "entry_idx": entry_idx, "exit_idx": i,
                "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                "sl": sl_price, "tp": tp_price,
                "pnl_usd": round(pnl, 2), "lot": entry_lot,
                "balance_at_entry": round(entry_balance_at_entry, 2),
                "balance_after": round(balance, 2),
                "reason": "tp_hit", "bars": ticks_in_trade,
                "atr_at_entry": round(atr_arr[entry_idx], 4),
                "sl_dist": round(abs(entry_price - sl_price), 2),
                "tp_dist": round(abs(tp_price - entry_price), 2),
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
                "sl": sl_price, "tp": tp_price,
                "pnl_usd": round(pnl, 2), "lot": entry_lot,
                "balance_at_entry": round(entry_balance_at_entry, 2),
                "balance_after": round(balance, 2),
                "reason": "tp_hit", "bars": ticks_in_trade,
                "atr_at_entry": round(atr_arr[entry_idx], 4),
                "sl_dist": round(abs(entry_price - sl_price), 2),
                "tp_dist": round(abs(tp_price - entry_price), 2),
            })
            in_trade = False
            ticks_in_trade = 0
            continue

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
                "sl": sl_price, "tp": tp_price,
                "pnl_usd": round(pnl, 2), "lot": entry_lot,
                "balance_at_entry": round(entry_balance_at_entry, 2),
                "balance_after": round(balance, 2),
                "reason": "timeout", "bars": ticks_in_trade,
                "atr_at_entry": round(atr_arr[entry_idx], 4),
                "sl_dist": round(abs(entry_price - sl_price), 2),
                "tp_dist": round(abs(tp_price - entry_price), 2),
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

            entry_balance_at_entry = balance
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
        "sl": sl_price, "tp": tp_price,
        "pnl_usd": round(pnl, 2), "lot": entry_lot,
        "balance_at_entry": round(entry_balance_at_entry, 2),
        "balance_after": round(balance, 2),
        "reason": "eoy_close", "bars": ticks_in_trade,
        "atr_at_entry": round(atr_arr[entry_idx], 4),
        "sl_dist": round(abs(entry_price - sl_price), 2),
        "tp_dist": round(abs(tp_price - entry_price), 2),
    })

df = pd.DataFrame(trades)
df["month"] = pd.to_datetime(df["entry_ts"]).dt.month

# ── Full Year Summary ──
print(f"\n{'='*80}")
print(f"  ASP BACKTEST {YEAR} — FULL YEAR SUMMARY")
print(f"{'='*80}")
print(f"  Starting balance:  ${STARTING_BALANCE:.2f}")
print(f"  Ending balance:    ${balance:.2f}")
print(f"  Total return:      {(balance/STARTING_BALANCE - 1)*100:+.1f}%")
print(f"  Net PnL:           ${df['pnl_usd'].sum():+.2f}")
print(f"  Total trades:      {len(df)}")
print(f"  Max drawdown:      ${df['pnl_usd'].cumsum().min():.2f}")

wins = df[df["pnl_usd"] > 0]
losses = df[df["pnl_usd"] <= 0]
wr = len(wins) / len(df) * 100
gross_profit = wins["pnl_usd"].sum()
gross_loss = abs(losses["pnl_usd"].sum())
pf = gross_profit / max(gross_loss, 0.01)

print(f"  Win rate:          {wr:.1f}%")
print(f"  Profit factor:     {pf:.2f}")
print(f"  Avg win:           ${wins['pnl_usd'].mean():+.2f}")
print(f"  Avg loss:          ${losses['pnl_usd'].mean():+.2f}")
print(f"  Avg bars held:     {df['bars'].mean():.1f}")

print(f"\n  Exit reasons:")
for reason, group in df.groupby("reason"):
    wr_r = (group["pnl_usd"] > 0).sum() / len(group) * 100
    print(f"    {reason:12s}: {len(group):4d} trades, WR={wr_r:.1f}%, PnL=${group['pnl_usd'].sum():+.2f}, Avg lot={group['lot'].mean():.4f}")

# Monthly breakdown
print(f"\n  Monthly breakdown:")
print(f"  {'Month':>8s} {'Trades':>7s} {'WR%':>7s} {'PF':>7s} {'Net PnL':>12s} {'Avg Lot':>9s} {'End Bal':>12s}")
print(f"  {'-'*62}")
for m, group in df.groupby("month"):
    m_wins = group[group["pnl_usd"] > 0]
    m_losses = group[group["pnl_usd"] <= 0]
    m_wr = len(m_wins) / len(group) * 100
    m_gp = m_wins["pnl_usd"].sum()
    m_gl = abs(m_losses["pnl_usd"].sum())
    m_pf = m_gp / max(m_gl, 0.01)
    m_pnl = group["pnl_usd"].sum()
    m_lot = group["lot"].mean()
    m_bal = group["balance_after"].iloc[-1]
    print(f"  {m:>8d} {len(group):>7d} {m_wr:>6.1f}% {m_pf:>7.2f} ${m_pnl:>+11.2f} {m_lot:>9.4f} ${m_bal:>11.2f}")

# ── January Detailed Trades ──
jan = df[df["month"] == 1].copy()
print(f"\n{'='*80}")
print(f"  JANUARY {YEAR} — COMPLETE TRADE LIST ({len(jan)} trades)")
print(f"{'='*80}")
print(f"  Starting balance (Jan 1): ${STARTING_BALANCE:.2f}")
if len(jan) > 0:
    print(f"  Ending balance (Jan 31):  ${jan['balance_after'].iloc[-1]:.2f}")
    jan_wins = jan[jan["pnl_usd"] > 0]
    jan_losses = jan[jan["pnl_usd"] <= 0]
    print(f"  Jan trades: {len(jan)} | Wins: {len(jan_wins)} ({len(jan_wins)/len(jan)*100:.1f}%) | Losses: {len(jan_losses)}")
    print(f"  Jan PnL: ${jan['pnl_usd'].sum():+.2f}")
    print(f"  Jan PF:  {jan_wins['pnl_usd'].sum() / max(abs(jan_losses['pnl_usd'].sum()), 0.01):.2f}")

print(f"\n  {'#':>4s} {'Date':>12s} {'Time':>8s} {'Dir':>4s} {'Entry':>10s} {'Exit':>10s} {'SL':>10s} {'TP':>10s} {'ATR':>7s} {'Lot':>8s} {'BalBefore':>11s} {'BalAfter':>11s} {'PnL':>10s} {'Reason':>10s} {'Bars':>5s}")
print(f"  {'-'*145}")

for idx, (_, t) in enumerate(jan.iterrows(), 1):
    ts_entry = pd.Timestamp(t["entry_ts"])
    ts_exit = pd.Timestamp(t["exit_ts"])
    date_str = ts_entry.strftime("%Y-%m-%d")
    time_str = ts_entry.strftime("%H:%M")
    print(f"  {idx:>4d} {date_str:>12s} {time_str:>8s} {t['direction']:>4s} {t['entry']:>10.2f} {t['exit']:>10.2f} {t['sl']:>10.2f} {t['tp']:>10.2f} {t['atr_at_entry']:>7.2f} {t['lot']:>8.4f} ${t['balance_at_entry']:>10.2f} ${t['balance_after']:>10.2f} ${t['pnl_usd']:>+9.2f} {t['reason']:>10s} {t['bars']:>5d}")

# ── January Position Sizing Analysis ──
if len(jan) > 0:
    print(f"\n{'='*80}")
    print(f"  JANUARY POSITION SIZING ANALYSIS")
    print(f"{'='*80}")
    print(f"  Lot sizing method: EquityScaler (base={BASE_LOT}, ref=${20.00})")
    print(f"  Formula: lot = {BASE_LOT} * (balance / 20.00), halved if DD > 10%")
    print(f"  Contract size: {CONTRACT_SIZE} oz/lot")
    print(f"  Spread cost: ${SPREAD_PTS}/lot/trade = ${SPREAD_PTS * CONTRACT_SIZE:.2f}/lot round-trip")

    print(f"\n  Lot distribution:")
    print(f"    Min lot:     {jan['lot'].min():.4f}")
    print(f"    Max lot:     {jan['lot'].max():.4f}")
    print(f"    Mean lot:    {jan['lot'].mean():.4f}")
    print(f"    Median lot:  {jan['lot'].median():.4f}")

    # Position value at entry
    jan_copy = jan.copy()
    jan_copy["pos_value"] = jan_copy["entry"] * jan_copy["lot"] * CONTRACT_SIZE
    jan_copy["margin_used"] = jan_copy["pos_value"]  # all balance as margin
    jan_copy["risk_usd"] = jan_copy["sl_dist"] * jan_copy["lot"] * CONTRACT_SIZE
    jan_copy["reward_usd"] = jan_copy["tp_dist"] * jan_copy["lot"] * CONTRACT_SIZE
    jan_copy["risk_pct"] = jan_copy["risk_usd"] / jan_copy["balance_at_entry"] * 100
    jan_copy["spread_cost"] = SPREAD_PTS * jan_copy["lot"] * CONTRACT_SIZE

    print(f"\n  Position metrics:")
    print(f"    Avg position value:  ${jan_copy['pos_value'].mean():,.2f}")
    print(f"    Avg risk per trade:  ${jan_copy['risk_usd'].mean():.2f} ({jan_copy['risk_pct'].mean():.2f}% of balance)")
    print(f"    Avg reward target:   ${jan_copy['reward_usd'].mean():.2f}")
    print(f"    Avg spread cost:     ${jan_copy['spread_cost'].mean():.2f}")
    print(f"    Risk:Reward ratio:   1:{(jan_copy['reward_usd'].mean() / max(jan_copy['risk_usd'].mean(), 0.01)):.2f}")

    # Balance growth through January
    print(f"\n  Balance progression (first 20, last 10):")
    print(f"  {'#':>4s} {'Date':>12s} {'BalBefore':>11s} {'Lot':>8s} {'PnL':>10s} {'BalAfter':>11s} {'DD%':>7s}")
    print(f"  {'-'*65}")
    peak = STARTING_BALANCE
    for idx2, (_, t) in enumerate(jan.iterrows(), 1):
        peak = max(peak, t["balance_after"])
        dd_pct = (t["balance_after"] - peak) / peak * 100 if peak > 0 else 0
        ts = pd.Timestamp(t["entry_ts"])
        if idx2 <= 20 or idx2 > len(jan) - 10:
            print(f"  {idx2:>4d} {ts.strftime('%Y-%m-%d'):>12s} ${t['balance_at_entry']:>10.2f} {t['lot']:>8.4f} ${t['pnl_usd']:>+9.2f} ${t['balance_after']:>10.2f} {dd_pct:>6.1f}%")
        elif idx2 == 21:
            print(f"  {'...':>4s} {'...':>12s} {'...':>11s} {'...':>8s} {'...':>10s} {'...':>11s} {'...':>7s}")

    # Lot scaling over time
    print(f"\n  Lot scaling through January:")
    print(f"    Trade 1 lot:  {jan['lot'].iloc[0]:.4f} (balance: ${jan['balance_at_entry'].iloc[0]:.2f})")
    mid_idx = len(jan) // 2
    print(f"    Trade {mid_idx+1} lot:  {jan['lot'].iloc[mid_idx]:.4f} (balance: ${jan['balance_at_entry'].iloc[mid_idx]:.2f})")
    print(f"    Trade {len(jan)} lot:  {jan['lot'].iloc[-1]:.4f} (balance: ${jan['balance_at_entry'].iloc[-1]:.2f})")

    # Daily breakdown
    jan_copy["date"] = pd.to_datetime(jan_copy["entry_ts"]).dt.date
    print(f"\n  Daily breakdown:")
    print(f"  {'Date':>12s} {'Trades':>7s} {'WR%':>7s} {'PnL':>10s} {'EndBal':>11s} {'AvgLot':>8s}")
    print(f"  {'-'*58}")
    for date, group in jan_copy.groupby("date"):
        d_wr = (group["pnl_usd"] > 0).sum() / len(group) * 100
        d_pnl = group["pnl_usd"].sum()
        d_bal = group["balance_after"].iloc[-1]
        d_lot = group["lot"].mean()
        print(f"  {str(date):>12s} {len(group):>7d} {d_wr:>6.1f}% ${d_pnl:>+9.2f} ${d_bal:>10.2f} {d_lot:>8.4f}")
