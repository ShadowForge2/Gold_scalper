"""January 2025 deep-dive: per-trade detail, hold time, daily PnL."""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.direction_predictor import TrendPredictor, TrendExhaustionPredictor, compute_trend_features
from app.dukascopy_client import DukascopyClient

# Config
TREND_MODEL_PATH = "models/trend_xgb_h1.joblib"
EXIT_MODEL_PATH = "models/exit_trend_xgb_m5.joblib"
TREND_CONF_THRESHOLD = 0.55
EXHAUSTION_THRESHOLD = 0.70
NEW_SETUP_THRESHOLD = 0.80
SL_ATR_MULT = 2.0
ATR_PERIOD = 14
CHECK_INTERVAL = 3
STARTING_BALANCE = 20.0
BASE_LOT = 0.01
CONTRACT_SIZE = 100
SPREAD_PTS = 0.30
LOT_STEP = 0.01
MIN_LOT = 0.01
MAX_LOT = 1.0

def calc_lot(balance, peak_balance):
    lot = BASE_LOT * (balance / 20.0)
    if peak_balance and balance < peak_balance * 0.9:
        lot *= 0.5
    lot = round(lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(lot, MAX_LOT))

# Load
print("Loading models...")
trend_model = TrendPredictor.load(TREND_MODEL_PATH)
exit_model = TrendExhaustionPredictor.load(EXIT_MODEL_PATH)

client = DukascopyClient()
m1 = client.download_year(2025)
m5 = client.resample_to(m1, 5)
h1 = client.resample_to(m1, 16385)
m5 = m5.set_index("time"); h1 = h1.set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]

print("Computing features...")
m5_features = compute_trend_features(m5, h1)

# Batch predict trend
feat_arr = m5_features[trend_model._feature_cols].values
raw_probs = trend_model.model.predict_proba(feat_arr)
classes = list(trend_model.model.classes_)
m5_c = m5["close"].values.astype(np.float64)
prob_down_arr = np.zeros(len(m5_c))
prob_up_arr = np.zeros(len(m5_c))
prob_ranging_arr = np.zeros(len(m5_c))
for ci, cls in enumerate(classes):
    if cls == 0: prob_down_arr[:] = raw_probs[:, ci]
    elif cls == 1: prob_up_arr[:] = raw_probs[:, ci]
    elif cls == 2: prob_ranging_arr[:] = raw_probs[:, ci]

confidence_arr = np.maximum(prob_down_arr, np.maximum(prob_up_arr, prob_ranging_arr))
dir_arr = np.where(
    (prob_up_arr >= TREND_CONF_THRESHOLD) & (prob_up_arr >= prob_down_arr) & (prob_up_arr >= prob_ranging_arr), 1,
    np.where(
        (prob_down_arr >= TREND_CONF_THRESHOLD) & (prob_down_arr >= prob_up_arr) & (prob_down_arr >= prob_ranging_arr), -1, 0
    )
)

m5_h = m5["high"].values.astype(np.float64)
m5_l = m5["low"].values.astype(np.float64)
prev_c = np.concatenate([[m5_c[0]], m5_c[:-1]])
tr = np.maximum(m5_h - m5_l, np.maximum(np.abs(m5_h - prev_c), np.abs(m5_l - prev_c)))
atr_arr = pd.Series(tr, index=m5.index).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().values

# Backtest
print("Running backtest...")
n = len(m5_c)
trades = []
balance = STARTING_BALANCE
peak_bal = STARTING_BALANCE
in_trade = False
entry_idx = 0; entry_price = 0.0; entry_dir = ""
sl_price = 0.0; entry_lot = 0.01
peak_pnl_atr = 0.0; last_check = 0; ticks_in_trade = 0

for i in range(max(ATR_PERIOD + 5, 100), n):
    atr = atr_arr[i]
    if np.isnan(atr) or atr <= 0: atr = 1.0

    if in_trade:
        ticks_in_trade += 1
        # SL hits
        if entry_dir == "BUY" and m5_l[i] <= sl_price:
            exit_px = sl_price
            pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            balance += pnl; peak_bal = max(peak_bal, balance)
            trades.append({"entry_idx": entry_idx, "exit_idx": i, "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                "direction": entry_dir, "entry": entry_price, "exit": exit_px, "pnl_usd": round(pnl, 2), "lot": entry_lot,
                "reason": "sl_hit", "bars": ticks_in_trade, "p_exh": 0.0, "p_new": 0.0})
            in_trade = False; ticks_in_trade = 0; continue
        if entry_dir == "SELL" and m5_h[i] >= sl_price:
            exit_px = sl_price
            pnl = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            balance += pnl; peak_bal = max(peak_bal, balance)
            trades.append({"entry_idx": entry_idx, "exit_idx": i, "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                "direction": entry_dir, "entry": entry_price, "exit": exit_px, "pnl_usd": round(pnl, 2), "lot": entry_lot,
                "reason": "sl_hit", "bars": ticks_in_trade, "p_exh": 0.0, "p_new": 0.0})
            in_trade = False; ticks_in_trade = 0; continue

        # Exit model check
        if ticks_in_trade >= CHECK_INTERVAL and (ticks_in_trade - last_check) >= CHECK_INTERVAL:
            last_check = ticks_in_trade
            mf_row = m5_features.iloc[i] if i < len(m5_features) else None
            if mf_row is not None:
                if entry_dir == "BUY":
                    diff = m5_c[i] - entry_price
                    peak = float(np.max(m5_h[max(entry_idx,0):i+1] - entry_price))
                else:
                    diff = entry_price - m5_c[i]
                    peak = float(np.max(entry_price - m5_l[max(entry_idx,0):i+1]))
                peak_pnl_atr = max(peak_pnl_atr, peak / atr)
                drawdown = max(0, peak - diff) / max(peak, 0.001)
                momentum_decay = 0.5
                if ticks_in_trade > 5:
                    if entry_dir == "BUY":
                        recent = m5_c[i] - m5_c[max(i-5, 0)]
                        entry_m = m5_c[min(entry_idx + 5, n-1)] - entry_price
                    else:
                        recent = m5_c[max(i-5, 0)] - m5_c[i]
                        entry_m = entry_price - m5_c[min(entry_idx + 5, n-1)]
                    momentum_decay = 1.0 - min(abs(recent) / max(abs(entry_m), 0.001), 2.0)
                trade_state = {"bars_held": ticks_in_trade, "trend_pnl_atr": round(diff / atr, 4),
                    "trend_peak_atr": round(peak / atr, 4), "trend_drawdown_pct": round(drawdown, 4),
                    "trend_momentum_decay": round(momentum_decay, 4)}
                exit_signal = exit_model.predict_exit(mf_row, trade_state,
                    exhaustion_threshold=EXHAUSTION_THRESHOLD, new_setup_threshold=NEW_SETUP_THRESHOLD)
                p_exh, p_alive, p_new = exit_model.predict_proba(mf_row, trade_state)
                if exit_signal in ("TREND_EXHAUSTED", "NEW_SETUP"):
                    exit_px = m5_c[i]
                    if entry_dir == "BUY":
                        pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                    else:
                        pnl = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                    balance += pnl; peak_bal = max(peak_bal, balance)
                    reason = "exhaustion" if exit_signal == "TREND_EXHAUSTED" else "new_setup"
                    trades.append({"entry_idx": entry_idx, "exit_idx": i, "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                        "direction": entry_dir, "entry": entry_price, "exit": exit_px, "pnl_usd": round(pnl, 2), "lot": entry_lot,
                        "reason": reason, "bars": ticks_in_trade, "p_exh": round(p_exh, 4), "p_new": round(p_new, 4)})
                    in_trade = False; ticks_in_trade = 0; continue

        # Trail SL
        if entry_dir == "BUY" and m5_c[i] - entry_price > atr:
            new_sl = entry_price + 0.2 * atr
            if new_sl > sl_price: sl_price = new_sl
        elif entry_dir == "SELL" and entry_price - m5_c[i] > atr:
            new_sl = entry_price - 0.2 * atr
            if new_sl < sl_price: sl_price = new_sl
    else:
        d = dir_arr[i]
        if d != 0:
            direction = "BUY" if d == 1 else "SELL"
            atr_val = atr_arr[i]
            if np.isnan(atr_val) or atr_val <= 0: continue
            entry_price = m5_c[i]
            sl_dist = atr_val * SL_ATR_MULT
            if direction == "BUY": sl_price = entry_price - sl_dist
            else: sl_price = entry_price + sl_dist
            if sl_dist < 0.50: continue
            entry_lot = calc_lot(balance, peak_bal)
            in_trade = True; entry_idx = i; entry_dir = direction
            peak_pnl_atr = 0.0; last_check = 0; ticks_in_trade = 0

if in_trade:
    exit_px = m5_c[-1]
    if entry_dir == "BUY":
        pnl = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
    else:
        pnl = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
    balance += pnl
    trades.append({"entry_idx": entry_idx, "exit_idx": n-1, "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[-1],
        "direction": entry_dir, "entry": entry_price, "exit": exit_px, "pnl_usd": round(pnl, 2), "lot": entry_lot,
        "reason": "eoy_close", "bars": ticks_in_trade, "p_exh": 0.0, "p_new": 0.0})

# Analysis
df = pd.DataFrame(trades)
df["entry_ts"] = pd.to_datetime(df["entry_ts"])
df["exit_ts"] = pd.to_datetime(df["exit_ts"])
df["hold_mins"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 60.0
df["hold_hrs"] = df["hold_mins"] / 60.0
df["date"] = df["entry_ts"].dt.date

jan = df[df["entry_ts"].dt.month == 1].copy()
print()
print("=" * 70)
print("  JANUARY 2025 DEEP DIVE")
print("=" * 70)
print(f"  Total trades:   {len(jan)}")
print(f"  Wins:           {(jan.pnl_usd > 0).sum()}")
print(f"  Losses:         {(jan.pnl_usd <= 0).sum()}")
print(f"  Win rate:       {(jan.pnl_usd > 0).sum() / len(jan) * 100:.1f}%")
print(f"  Net PnL:        ${jan.pnl_usd.sum():+.2f}")
print(f"  Avg lot:        {jan.lot.mean():.4f}")
print()
print("  Hold time:")
print(f"    Avg:          {jan.hold_hrs.mean():.2f} hrs ({jan.hold_mins.mean():.1f} min)")
print(f"    Min:          {jan.hold_hrs.min():.2f} hrs")
print(f"    Max:          {jan.hold_hrs.max():.2f} hrs")
print(f"    Median:       {jan.hold_hrs.median():.2f} hrs")
print()
print("  Daily breakdown:")
for day, g in jan.groupby("date"):
    daily_pnl = g.pnl_usd.sum()
    avg_hold = g.hold_hrs.mean()
    wr = (g.pnl_usd > 0).sum() / len(g) * 100
    print(f"    {day}:  {len(g)} trades, WR={wr:.0f}%, PnL=${daily_pnl:+.2f}, avg_hold={avg_hold:.1f}h")

# Full year daily stats for comparison
print()
print("=" * 70)
print("  FULL YEAR DAILY STATS FOR COMPARISON")
print("=" * 70)
daily = df.groupby("date").agg(
    trades=("pnl_usd", "count"),
    pnl=("pnl_usd", "sum"),
    avg_hold_hrs=("hold_hrs", "mean"),
).reset_index()
print(f"  Avg trades/day:  {daily.trades.mean():.1f}")
print(f"  Avg PnL/day:     ${daily.pnl.mean():+.2f}")
print(f"  Avg hold time:   {daily.avg_hold_hrs.mean():.2f} hrs")
print(f"  Best day:        ${daily.pnl.max():+.2f}")
print(f"  Worst day:       ${daily.pnl.min():+.2f}")
print(f"  Profitable days: {(daily.pnl > 0).sum()}/{len(daily)} ({(daily.pnl > 0).sum()/len(daily)*100:.0f}%)")

# By exit reason
print()
print("  Exit reason breakdown (full year):")
for reason, g in df.groupby("reason"):
    wr = (g.pnl_usd > 0).sum() / len(g) * 100
    print(f"    {reason:15s}: {len(g):3d} trades, WR={wr:.1f}%, avg_hold={g.hold_hrs.mean():.1f}h, PnL=${g.pnl_usd.sum():+.2f}")

print()
print("  First 20 January trades:")
print(f"  {'Dir':>4s} {'Entry':>10s} {'Exit':>10s} {'PnL':>8s} {'Lot':>6s} {'Reason':>12s} {'Hold':>6s}")
for _, t in jan.head(20).iterrows():
    print(f"  {t.direction:>4s} {t.entry:>10.2f} {t.exit:>10.2f} {t.pnl_usd:>+8.2f} {t.lot:>6.2f} {t.reason:>12s} {t.hold_hrs:>5.1f}h")
