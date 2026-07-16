"""US100 2024 — Signal accuracy + early-stage compounding ($20 start)."""
import os, sys, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features
from app.asp_labels import compute_atr

SYMBOL = "US100"
ASP_MODEL_PATH = "models/asp_swing_xgb_m5_US100.joblib"
ASP_FEATURE_PATH = "models/asp_swing_m5_features_US100.npy"

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.0
TIMEOUT_BARS = 5          # 25 min
TRAILING_TRIGGER = 1.5
TRAILING_RETRACE = 1.0
ATR_PERIOD = 14
STARTING_BALANCE = 20.0
CONTRACT_SIZE = 1
SPREAD_PTS = 1.5
BASE_LOT = 0.5
MAX_LOT = 20.0

asp_saved = joblib.load(ASP_MODEL_PATH)
asp_model = asp_saved["model"]
inv_map = asp_saved["label_map"]
asp_features_list = list(np.load(ASP_FEATURE_PATH, allow_pickle=True))

client = DukascopyClient(symbol=SYMBOL, cache_dir="data/dukascopy_us100")
m1 = client.download_year(2024)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
m5 = client.resample_to(m1, 5)
h1 = client.resample_to(m1, 16385)
m5 = m5.set_index("time")
h1 = h1.set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]
n = len(m5)
m5_h = m5["high"].values.astype(np.float64)
m5_l = m5["low"].values.astype(np.float64)
m5_c = m5["close"].values.astype(np.float64)
atr_arr = compute_atr(m5_h, m5_l, m5_c, ATR_PERIOD)

features = compute_asp_features(m5, h1)
valid = features[asp_features_list].dropna()
signal_series = pd.Series(0, index=features.index, dtype=np.int8)
if len(valid) > 0:
    feat_arr = valid[asp_features_list].values
    raw_preds = asp_model.predict(feat_arr)
    preds = np.array([inv_map.get(p, p) for p in raw_preds])
    signal_series.loc[valid.index] = preds.astype(np.int8)
asp_arr = signal_series.values
m5_times = m5.index.values

buy_n = (asp_arr == 1).sum()
sell_n = (asp_arr == -1).sum()
print(f"ASP signals: BUY={buy_n}, SELL={sell_n}, Total={buy_n+sell_n}\n")


def calc_lot(balance, peak_balance):
    lot = BASE_LOT * (balance / 20.0)
    if peak_balance and peak_balance > 0:
        dd = (peak_balance - balance) / peak_balance * 100
        if dd > 15:
            lot *= 0.5
    lot = round(lot / 0.1) * 0.1
    return max(0.1, min(lot, MAX_LOT))


trades = []
balance = STARTING_BALANCE
peak_balance = STARTING_BALANCE
in_trade = False
entry_price = 0.0
entry_dir = ""
sl_price = 0.0
tp_price = 0.0
entry_lot = 0.5
entry_time = None
ticks = 0
best_price = 0.0

for i in range(max(ATR_PERIOD + 5, 100), n):
    atr = atr_arr[i]
    if np.isnan(atr) or atr <= 0:
        atr = 1.0

    if in_trade:
        ticks += 1
        hit_sl = hit_tp = False

        if entry_dir == "BUY":
            best_price = max(best_price, m5_h[i])
            trigger = entry_price + atr * TRAILING_TRIGGER
            if best_price >= trigger:
                trail_sl = best_price - atr * TRAILING_RETRACE
                if trail_sl > sl_price:
                    sl_price = trail_sl
            if m5_l[i] <= sl_price:
                hit_sl = True
            elif m5_h[i] >= tp_price:
                hit_tp = True
        else:
            best_price = min(best_price, m5_l[i])
            trigger = entry_price - atr * TRAILING_TRIGGER
            if best_price <= trigger:
                trail_sl = best_price + atr * TRAILING_RETRACE
                if trail_sl < sl_price:
                    sl_price = trail_sl
            if m5_h[i] >= sl_price:
                hit_sl = True
            elif m5_l[i] <= tp_price:
                hit_tp = True

        if hit_sl or hit_tp:
            ep = sl_price if hit_sl else tp_price
            if entry_dir == "BUY":
                pnl = (ep - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            else:
                pnl = (entry_price - ep) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            balance += pnl
            peak_balance = max(peak_balance, balance)
            trades.append({
                "entry_time": entry_time,
                "exit_time": m5_times[i],
                "dir": entry_dir,
                "entry": entry_price,
                "exit": ep,
                "reason": "sl" if hit_sl else "tp",
                "pnl": pnl,
                "lot": entry_lot,
                "balance": balance,
                "ticks": ticks,
                "hold_min": ticks * 5,
                "atr_at_entry": atr_arr[i - ticks],
            })
            in_trade = False
            ticks = 0
            continue

        if ticks >= TIMEOUT_BARS:
            if entry_dir == "BUY":
                pnl = (m5_c[i] - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            else:
                pnl = (entry_price - m5_c[i]) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            balance += pnl
            peak_balance = max(peak_balance, balance)
            trades.append({
                "entry_time": entry_time,
                "exit_time": m5_times[i],
                "dir": entry_dir,
                "entry": entry_price,
                "exit": m5_c[i],
                "reason": "timeout",
                "pnl": pnl,
                "lot": entry_lot,
                "balance": balance,
                "ticks": ticks,
                "hold_min": ticks * 5,
                "atr_at_entry": atr_arr[i - ticks],
            })
            in_trade = False
            ticks = 0
            continue
    else:
        d = asp_arr[i]
        if d != 0:
            atr_val = atr_arr[i]
            if np.isnan(atr_val) or atr_val <= 0 or atr_val < 5.0:
                continue
            direction = "BUY" if d == 1 else "SELL"
            entry_price = m5_c[i]
            sl_dist = atr_val * SL_ATR_MULT
            tp_dist = atr_val * TP_ATR_MULT
            sl_price = entry_price - sl_dist if direction == "BUY" else entry_price + sl_dist
            tp_price = entry_price + tp_dist if direction == "BUY" else entry_price - tp_dist
            entry_lot = calc_lot(balance, peak_balance)
            in_trade = True
            entry_dir = direction
            entry_time = m5_times[i]
            ticks = 0
            best_price = entry_price

if in_trade:
    if entry_dir == "BUY":
        pnl = (m5_c[-1] - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
    else:
        pnl = (entry_price - m5_c[-1]) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
    balance += pnl
    trades.append({
        "entry_time": entry_time,
        "exit_time": m5_times[-1],
        "dir": entry_dir,
        "reason": "eoy",
        "pnl": pnl,
        "lot": entry_lot,
        "balance": balance,
        "ticks": ticks,
        "hold_min": ticks * 5,
        "atr_at_entry": atr_arr[-ticks] if ticks > 0 else 0,
    })

df = pd.DataFrame(trades)
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["exit_time"] = pd.to_datetime(df["exit_time"])
df["month"] = df["entry_time"].dt.month
df["month_name"] = df["entry_time"].dt.strftime("%B")
df["week"] = df["entry_time"].dt.isocalendar().week.astype(int)

# ── SIGNAL ACCURACY ──
print("=" * 80)
print("  SIGNAL ACCURACY ANALYSIS")
print("=" * 80)

# How many signals actually resulted in a trade entry vs were skipped (ATR < 5)
skipped = 0
entered = 0
for i in range(max(ATR_PERIOD + 5, 100), n):
    d = asp_arr[i]
    if d != 0:
        atr_val = atr_arr[i]
        if np.isnan(atr_val) or atr_val <= 0 or atr_val < 5.0:
            skipped += 1
        else:
            entered += 1

total_signals = buy_n + sell_n
print(f"  Total ASP signals generated: {total_signals}")
print(f"    BUY signals:  {buy_n} ({buy_n/total_signals*100:.1f}%)")
print(f"    SELL signals: {sell_n} ({sell_n/total_signals*100:.1f}%)")
print(f"  Skipped (ATR too low): {skipped} ({skipped/total_signals*100:.1f}%)")
print(f"  Actually entered: {entered} ({entered/total_signals*100:.1f}%)")

# TP vs SL vs Timeout
tp = df[df["reason"] == "tp"]
sl = df[df["reason"] == "sl"]
to = df[df["reason"] == "timeout"]

print(f"\n  Trade outcomes:")
print(f"    TP hits:     {len(tp):>6} ({len(tp)/len(df)*100:.1f}%) — 100% WR")
print(f"    SL hits:     {len(sl):>6} ({len(sl)/len(df)*100:.1f}%) — {(sl['pnl']>0).sum()} wins ({(sl['pnl']>0).sum()/len(sl)*100:.1f}% WR)")
print(f"    Timeouts:    {len(to):>6} ({len(to)/len(df)*100:.1f}%) — {(to['pnl']>0).sum()} wins ({(to['pnl']>0).sum()/len(to)*100:.1f}% WR)")

# Direction accuracy
buy_trades = df[df["dir"] == "BUY"]
sell_trades = df[df["dir"] == "SELL"]
buy_wr = (buy_trades["pnl"] > 0).sum() / len(buy_trades) * 100 if len(buy_trades) > 0 else 0
sell_wr = (sell_trades["pnl"] > 0).sum() / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
print(f"\n  Direction accuracy:")
print(f"    BUY trades:  {len(buy_trades):>5} | WR: {buy_wr:.1f}% | Net: ${buy_trades['pnl'].sum():>12,.2f}")
print(f"    SELL trades: {len(sell_trades):>5} | WR: {sell_wr:.1f}% | Net: ${sell_trades['pnl'].sum():>12,.2f}")

# ── EARLY STAGE COMPOUNDING ──
print(f"\n{'=' * 80}")
print("  EARLY-STAGE COMPOUNDING ($20 start)")
print("=" * 80)

print(f"\n  {'Week':<6} {'Trades':>7} {'WR%':>6} {'Net PnL':>14} {'Balance':>14} {'Lot Size':>10} {'Growth':>10}")
print(f"  {'-'*66}")

prev_bal = STARTING_BALANCE
for wk in range(1, 53):
    wk_df = df[(df["week"] == wk)]
    if len(wk_df) == 0:
        continue
    wk_wins = (wk_df["pnl"] > 0).sum()
    wk_wr = wk_wins / len(wk_df) * 100
    wk_net = wk_df["pnl"].sum()
    wk_bal = wk_df["balance"].iloc[-1]
    wk_lot = wk_df["lot"].mean()
    wk_growth = (wk_bal - prev_bal) / prev_bal * 100 if prev_bal > 0 else 0
    print(f"  W{wk:<5} {len(wk_df):>7} {wk_wr:>5.1f}% ${wk_net:>12,.2f} ${wk_bal:>12,.2f} {wk_lot:>8.2f} {wk_growth:>+8.0f}%")
    prev_bal = wk_bal
    if wk_bal > 1000:
        break

# ── MONTHLY DETAIL ──
print(f"\n{'=' * 80}")
print("  MONTHLY SUMMARY")
print("=" * 80)

print(f"\n  {'Month':<12} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'Avg Lot':>10} {'Net PnL':>14} {'End Bal':>14} {'Growth':>10}")
print(f"  {'-'*78}")
prev_bal = STARTING_BALANCE
for m in range(1, 13):
    md = df[df["month"] == m]
    if len(md) == 0:
        continue
    m_wins = (md["pnl"] > 0).sum()
    m_wr = m_wins / len(md) * 100
    m_net = md["pnl"].sum()
    m_bal = md["balance"].iloc[-1]
    m_lot = md["lot"].mean()
    m_growth = (m_bal - prev_bal) / prev_bal * 100 if prev_bal > 0 else 0
    mname = md["month_name"].iloc[0]
    print(f"  {mname:<12} {len(md):>7} {m_wins:>5} {m_wr:>5.1f}% {m_lot:>8.2f} ${m_net:>12,.2f} ${m_bal:>12,.2f} {m_growth:>+8.0f}%")
    prev_bal = m_bal

# ── LOT SCALING OVER TIME ──
print(f"\n{'=' * 80}")
print("  LOT SCALING PROGRESSION")
print("=" * 80)

milestones = [20, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000]
for ms in milestones:
    hit = df[df["balance"] >= ms]
    if len(hit) > 0:
        first_trade = hit.iloc[0]
        lot_at_milestone = first_trade["lot"]
        print(f"  ${ms:>10,} reached at trade #{df.index.get_loc(first_trade.name)+1} ({first_trade['entry_time'].strftime('%Y-%m-%d')}) — lot: {lot_at_milestone:.2f}")
    else:
        print(f"  ${ms:>10,} NOT reached")

# ── STREAK ANALYSIS ──
print(f"\n{'=' * 80}")
print("  WIN/LOSS STREAKS")
print("=" * 80)

results = (df["pnl"] > 0).astype(int).values
max_win_streak = max_loss_streak = 0
cur_win = cur_loss = 0
for r in results:
    if r == 1:
        cur_win += 1
        cur_loss = 0
        max_win_streak = max(max_win_streak, cur_win)
    else:
        cur_loss += 1
        cur_win = 0
        max_loss_streak = max(max_loss_streak, cur_loss)

print(f"  Max win streak:  {max_win_streak}")
print(f"  Max loss streak: {max_loss_streak}")

# ── OVERALL ──
total = len(df)
wins = (df["pnl"] > 0).sum()
losses = total - wins
net = df["pnl"].sum()
gp = df.loc[df["pnl"] > 0, "pnl"].sum()
gl = abs(df.loc[df["pnl"] <= 0, "pnl"].sum())
pf = gp / gl if gl > 0 else 99.9
wr = wins / total * 100

print(f"\n{'=' * 80}")
print(f"  FINAL: ${STARTING_BALANCE:.2f} -> ${balance:,.2f} ({net/STARTING_BALANCE*100:,.0f}% growth)")
print(f"  Trades: {total} | WR: {wr:.1f}% | PF: {pf:.2f}")
print(f"{'=' * 80}")
