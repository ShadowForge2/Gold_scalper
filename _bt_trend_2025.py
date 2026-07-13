"""
Backtest: Trend-following ML system on 2025 M5 data.

Entry: TrendPredictor predicts UP_TREND or DOWN_TREND (conf >= threshold).
SL:    2x ATR (hard safety net).
Exit:  TrendExhaustionPredictor decides TREND_EXHAUSTED / NEW_SETUP.
       No fixed TP — let winners run.
Lot:   0.01 (minimum), single position per trade.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import (
    TrendPredictor, TrendExhaustionPredictor,
    compute_features, compute_trend_features,
    FEATURE_COLS, EXIT_TREND_FEATURE_COLS,
)

# ── Config ──────────────────────────────────────────────────────────────────
YEAR = 2025
TREND_MODEL_PATH = "models/trend_xgb_h1.joblib"
EXIT_MODEL_PATH = "models/exit_trend_xgb_m5.joblib"
TREND_CONF_THRESHOLD = 0.55
EXHAUSTION_THRESHOLD = 0.70
NEW_SETUP_THRESHOLD = 0.80
SL_ATR_MULT = 2.0          # SL = 2x ATR
ATR_PERIOD = 14
CHECK_INTERVAL = 3          # check exit every N M5 bars
STARTING_BALANCE = 20.0
BASE_LOT = 0.01             # base lot (scales with equity)
CONTRACT_SIZE = 100         # 100 oz per standard lot
SPREAD_PTS = 0.30           # estimated spread in USD for XAUUSD
LOT_STEP = 0.01
MIN_LOT = 0.01
MAX_LOT = 1.0

# ── Load models ─────────────────────────────────────────────────────────────
print("Loading models...")
trend_model = TrendPredictor.load(TREND_MODEL_PATH)
exit_model = TrendExhaustionPredictor.load(EXIT_MODEL_PATH)
print(f"  Trend model: {'OK' if trend_model.model is not None else 'MISSING'}")
print(f"  Exit model:  {'OK' if exit_model.model is not None else 'MISSING'}")

# ── Load data ───────────────────────────────────────────────────────────────
client = DukascopyClient()
print(f"\nLoading {YEAR} data...")
t0 = time.time()
m1 = client.download_year(YEAR)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
print(f"  M1 bars: {len(m1)} ({time.time()-t0:.1f}s)")

m5 = client.resample_to(m1, 5)
h1 = client.resample_to(m1, 16385)
print(f"  M5 bars: {len(m5)}")
print(f"  H1 bars: {len(h1)}")

m5 = m5.set_index("time")
h1 = h1.set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]

# Precompute M5 features
print("Computing M5 features...")
m5_features = compute_trend_features(m5, h1)
print(f"  Feature rows: {len(m5_features)}")

# Precompute ATR for SL
m5_h = m5["high"].values.astype(np.float64)
m5_l = m5["low"].values.astype(np.float64)
m5_c = m5["close"].values.astype(np.float64)
m5_o = m5["open"].values.astype(np.float64)
n = len(m5_c)

prev_c = np.concatenate([[m5_c[0]], m5_c[:-1]])
tr = np.maximum(m5_h - m5_l,
                np.maximum(np.abs(m5_h - prev_c), np.abs(m5_l - prev_c)))
atr_arr = pd.Series(tr, index=m5.index).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().values

# ── Batch-predict ALL trend signals upfront (1 call vs 70K) ─────────────────

def calc_lot(balance, peak_balance):
    """EquityScaler.get_lot() — scales lot with balance."""
    reference = 20.0
    lot = BASE_LOT * (balance / reference)
    # Drawdown half-size
    if peak_balance and balance < peak_balance * 0.9:
        lot *= 0.5
    lot = round(lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(lot, MAX_LOT))

print("Batch-predicting trend signals...")
t_pred = time.time()
feat_arr = m5_features[trend_model._feature_cols].values
raw_probs = trend_model.model.predict_proba(feat_arr)
classes = list(trend_model.model.classes_)
# Map class indices to (prob_down, prob_up, prob_ranging)
prob_down_arr = np.zeros(n)
prob_up_arr = np.zeros(n)
prob_ranging_arr = np.zeros(n)
for ci, cls in enumerate(classes):
    if cls == 0:
        prob_down_arr[:] = raw_probs[:, ci]
    elif cls == 1:
        prob_up_arr[:] = raw_probs[:, ci]
    elif cls == 2:
        prob_ranging_arr[:] = raw_probs[:, ci]
confidence_arr = np.maximum(prob_down_arr, np.maximum(prob_up_arr, prob_ranging_arr))
dir_arr = np.where(
    (prob_up_arr >= TREND_CONF_THRESHOLD) & (prob_up_arr >= prob_down_arr) & (prob_up_arr >= prob_ranging_arr),
    1,  # BUY
    np.where(
        (prob_down_arr >= TREND_CONF_THRESHOLD) & (prob_down_arr >= prob_up_arr) & (prob_down_arr >= prob_ranging_arr),
        -1,  # SELL
        0,   # no signal
    )
)
print(f"  Batch predict: {time.time()-t_pred:.1f}s ({n} bars)")
print(f"  BUY signals: {(dir_arr == 1).sum()}, SELL signals: {(dir_arr == -1).sum()}")

# ── Backtest loop ───────────────────────────────────────────────────────────
print("\nRunning backtest...")
trades = []
balance = STARTING_BALANCE
peak_balance = STARTING_BALANCE
in_trade = False
entry_idx = 0
entry_price = 0.0
entry_dir = ""
sl_price = 0.0
entry_lot = 0.01
peak_pnl_atr = 0.0
last_check = 0
ticks_in_trade = 0

for i in range(max(ATR_PERIOD + 5, 100), n):
    atr = atr_arr[i]
    if np.isnan(atr) or atr <= 0:
        atr = 1.0

    if in_trade:
        ticks_in_trade += 1

        # Hard SL check (bar-by-bar: check low for BUY, high for SELL)
        if entry_dir == "BUY" and m5_l[i] <= sl_price:
            exit_px = sl_price  # fill at SL
            pnl_usd = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            balance += pnl_usd
            peak_balance = max(peak_balance, balance)
            trades.append({
                "entry_idx": entry_idx, "exit_idx": i,
                "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                "sl": sl_price, "pnl_usd": round(pnl_usd, 2), "lot": entry_lot,
                "reason": "sl_hit", "bars": ticks_in_trade,
                "exit_prob_exh": 0.0, "exit_prob_new": 0.0,
            })
            in_trade = False
            ticks_in_trade = 0
            continue

        if entry_dir == "SELL" and m5_h[i] >= sl_price:
            exit_px = sl_price
            pnl_usd = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
            balance += pnl_usd
            trades.append({
                "entry_idx": entry_idx, "exit_idx": i,
                "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                "sl": sl_price, "pnl_usd": round(pnl_usd, 2), "lot": entry_lot,
                "reason": "sl_hit", "bars": ticks_in_trade,
                "exit_prob_exh": 0.0, "exit_prob_new": 0.0,
            })
            in_trade = False
            ticks_in_trade = 0
            continue

        # Exhaustion exit check every CHECK_INTERVAL bars
        if ticks_in_trade >= CHECK_INTERVAL and (ticks_in_trade - last_check) >= CHECK_INTERVAL:
            last_check = ticks_in_trade

            # Get current market features
            mf_row = m5_features.iloc[i] if i < len(m5_features) else None
            if mf_row is not None:
                # Compute trade state
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

                trade_state = {
                    "bars_held": ticks_in_trade,
                    "trend_pnl_atr": round(diff / atr, 4),
                    "trend_peak_atr": round(peak / atr, 4),
                    "trend_drawdown_pct": round(drawdown, 4),
                    "trend_momentum_decay": round(momentum_decay, 4),
                }

                exit_signal = exit_model.predict_exit(
                    mf_row, trade_state,
                    exhaustion_threshold=EXHAUSTION_THRESHOLD,
                    new_setup_threshold=NEW_SETUP_THRESHOLD,
                )

                p_exh, p_alive, p_new = exit_model.predict_proba(mf_row, trade_state)

                if exit_signal in ("TREND_EXHAUSTED", "NEW_SETUP"):
                    exit_px = m5_c[i]
                    if entry_dir == "BUY":
                        pnl_usd = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                    else:
                        pnl_usd = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                    balance += pnl_usd
                    peak_balance = max(peak_balance, balance)
                    reason = "exhaustion" if exit_signal == "TREND_EXHAUSTED" else "new_setup"
                    trades.append({
                        "entry_idx": entry_idx, "exit_idx": i,
                        "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                        "direction": entry_dir, "entry": entry_price, "exit": exit_px,
                        "sl": sl_price, "pnl_usd": round(pnl_usd, 2), "lot": entry_lot,
                        "reason": reason, "bars": ticks_in_trade,
                        "exit_prob_exh": round(p_exh, 4), "exit_prob_new": round(p_new, 4),
                    })
                    in_trade = False
                    ticks_in_trade = 0
                    continue

        # Trail SL to breakeven+ if peak > 1 ATR profit
        if entry_dir == "BUY" and m5_c[i] - entry_price > atr:
            new_sl = entry_price + 0.2 * atr  # small buffer above entry
            if new_sl > sl_price:
                sl_price = new_sl
        elif entry_dir == "SELL" and entry_price - m5_c[i] > atr:
            new_sl = entry_price - 0.2 * atr
            if new_sl < sl_price:
                sl_price = new_sl

    else:
        # ── Entry check (precomputed) ──
        d = dir_arr[i]
        if d != 0:
            direction = "BUY" if d == 1 else "SELL"
            confidence = confidence_arr[i]
            atr_val = atr_arr[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            entry_price = m5_c[i]
            sl_dist = atr_val * SL_ATR_MULT

            if direction == "BUY":
                sl_price = entry_price - sl_dist
            else:
                sl_price = entry_price + sl_dist

            # Check if trade is viable (SL not too tight)
            if sl_dist < 0.50:
                continue

            entry_lot = calc_lot(balance, peak_balance)

            in_trade = True
            entry_idx = i
            entry_dir = direction
            peak_pnl_atr = 0.0
            last_check = 0
            ticks_in_trade = 0

# ── Close open trade at end of year ─────────────────────────────────────────
if in_trade:
    exit_px = m5_c[-1]
    if entry_dir == "BUY":
        pnl_usd = (exit_px - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
    else:
        pnl_usd = (entry_price - exit_px) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
    balance += pnl_usd
    peak_balance = max(peak_balance, balance)
    trades.append({
        "entry_idx": entry_idx, "exit_idx": n - 1,
        "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[-1],
        "direction": entry_dir, "entry": entry_price, "exit": exit_px,
        "sl": sl_price, "pnl_usd": round(pnl_usd, 2), "lot": entry_lot,
        "reason": "eoy_close", "bars": ticks_in_trade,
        "exit_prob_exh": 0.0, "exit_prob_new": 0.0,
    })

# ── Results ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(trades)
print(f"\n{'='*60}")
print(f"  TREND FOLLOWING BACKTEST — {YEAR}")
print(f"{'='*60}")
print(f"  Starting balance:  ${STARTING_BALANCE:.2f}")
print(f"  Lot scaling:       EquityScaler ({BASE_LOT} base)")
print(f"  SL:                {SL_ATR_MULT}x ATR")
print(f"  Trend threshold:   {TREND_CONF_THRESHOLD}")
print(f"  Exhaustion thr:    {EXHAUSTION_THRESHOLD}")
print(f"  New setup thr:     {NEW_SETUP_THRESHOLD}")
print(f"  Check interval:    {CHECK_INTERVAL} M5 bars")
print(f"{'='*60}")

if len(df) == 0:
    print("  NO TRADES")
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

    print(f"  Trades:            {len(df)}")
    print(f"  Wins:              {len(wins)} ({wr:.1f}%)")
    print(f"  Losses:            {len(losses)} ({100-wr:.1f}%)")
    print(f"  Net PnL:           ${net_pnl:+.2f}")
    print(f"  Ending balance:    ${balance:.2f}")
    print(f"  Return:            {(balance/STARTING_BALANCE - 1)*100:+.1f}%")
    print(f"  Profit factor:     {pf:.2f}")
    print(f"  Avg win:           ${avg_win:+.2f}")
    print(f"  Avg loss:          ${avg_loss:+.2f}")
    print(f"  Avg bars held:     {avg_bars:.1f}")
    print(f"  Max drawdown:      ${df['pnl_usd'].cumsum().min():.2f}")

    # By exit reason
    print(f"\n  Exit reasons:")
    for reason, group in df.groupby("reason"):
        wr_r = (group["pnl_usd"] > 0).sum() / len(group) * 100
        print(f"    {reason:20s}: {len(group):4d} trades, WR={wr_r:.1f}%, PnL=${group['pnl_usd'].sum():+.2f}")

    # By direction
    print(f"\n  By direction:")
    for d, group in df.groupby("direction"):
        wr_d = (group["pnl_usd"] > 0).sum() / len(group) * 100
        print(f"    {d:6s}: {len(group):4d} trades, WR={wr_d:.1f}%, PnL=${group['pnl_usd'].sum():+.2f}")

    # Monthly breakdown
    df["month"] = pd.to_datetime(df["entry_ts"]).dt.to_period("M")
    print(f"\n  Monthly breakdown:")
    for m, group in df.groupby("month"):
        wr_m = (group["pnl_usd"] > 0).sum() / len(group) * 100
        print(f"    {str(m):8s}: {len(group):4d} trades, WR={wr_m:.1f}%, PnL=${group['pnl_usd'].sum():+.2f}")

    # Print first 20 trades
    print(f"\n  First 20 trades:")
    print(f"  {'Dir':>4s} {'Entry':>10s} {'Exit':>10s} {'PnL':>8s} {'Reason':>12s} {'Bars':>5s} {'P_exh':>6s} {'P_new':>6s}")
    for _, t in df.head(20).iterrows():
        print(f"  {t['direction']:>4s} {t['entry']:>10.2f} {t['exit']:>10.2f} {t['pnl_usd']:>+8.2f} {t['reason']:>12s} {t['bars']:>5d} {t['exit_prob_exh']:>6.3f} {t['exit_prob_new']:>6.3f}")
