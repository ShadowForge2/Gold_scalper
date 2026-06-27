"""
Fast H1 Breakout + M1 Momentum backtest with exit mode 6 (Multi-TP Zone).
Pre-computes all per-bar values, uses pure numpy for the main loop.
"""
import sys, os, json, time
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from app.bias_engine import BiasEngine
from app.dukascopy_client import DukascopyClient

CONTRACT_SIZE = 100
INITIAL_BALANCE = 20.0


def compute_atr(h, l, c, period=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[0.0], tr])
    atr = np.zeros(len(c))
    for i in range(period, len(c)):
        atr[i] = np.mean(tr[i - period + 1:i + 1])
    return atr


def compute_momentum(c, o, h, l, atr_arr):
    out = np.zeros(len(c))
    for i in range(5, len(c)):
        recent = c[i - 2:i + 1]
        older = c[max(0, i - 5):i - 2]
        rc = abs(recent[-1] - recent[0])
        oc = abs(older[-1] - older[0]) if len(older) >= 2 else rc
        bodies = np.abs(c[i - 4:i + 1] - o[i - 4:i + 1])
        avg_body = np.mean(bodies)
        if avg_body == 0:
            out[i] = 0.0
        else:
            ref = max(atr_arr[i], c[i] * 0.0001)
            raw = (rc / (oc + 1e-10)) * (avg_body / (ref + 1e-10))
            out[i] = min(abs(raw), 1.0)
    return out


def run_backtest(m1):
    n = len(m1)
    mv = {
        "t": m1["time"].values,
        "o": m1["open"].values.astype(float),
        "h": m1["high"].values.astype(float),
        "l": m1["low"].values.astype(float),
        "c": m1["close"].values.astype(float),
    }

    # Pre-compute ATR and momentum for every bar
    atr_arr = compute_atr(mv["h"], mv["l"], mv["c"], 14)
    mom_arr = compute_momentum(mv["c"], mv["o"], mv["h"], mv["l"], atr_arr)

    # Resample to H1 to get H1 high/low
    m1_df = m1.reset_index(drop=True)
    h1 = m1_df.resample("h", on="time").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna().reset_index()
    h1["time"] = pd.to_datetime(h1["time"])
    h_count = len(h1)

    # Map M1 -> H1 bar index
    h1_ns = h1["time"].values.astype(np.int64)
    m1_ns = mv["t"].astype(np.int64)
    m1_h = np.searchsorted(h1_ns, m1_ns, side="right") - 1
    m1_h = np.clip(m1_h, 0, h_count - 1)

    # Vectorized bias computation
    print(f"  Computing bias for {h_count} H1 bars...", flush=True)
    hclose = h1["close"].values.astype(float)
    hhigh = h1["high"].values.astype(float)
    hlow = h1["low"].values.astype(float)

    # EMA 20/50
    alpha20 = 2.0 / (20 + 1)
    alpha50 = 2.0 / (50 + 1)
    ema20 = np.zeros(h_count)
    ema50 = np.zeros(h_count)
    ema20[0] = hclose[0]
    ema50[0] = hclose[0]
    for i in range(1, h_count):
        ema20[i] = hclose[i] * alpha20 + ema20[i-1] * (1 - alpha20)
        ema50[i] = hclose[i] * alpha50 + ema50[i-1] * (1 - alpha50)

    fast_slope = np.zeros(h_count)
    slow_slope = np.zeros(h_count)
    for i in range(6, h_count):
        fast_slope[i] = ema20[i] - ema20[i-5]
        slow_slope[i] = ema50[i] - ema50[i-5]

    # Swing highs: local max in ±5 window
    swing_high = np.zeros(h_count, dtype=bool)
    swing_low = np.zeros(h_count, dtype=bool)
    for i in range(5, h_count - 5):
        swing_high[i] = all(hhigh[i] >= hhigh[i - j] for j in range(1, 6)) and \
                        all(hhigh[i] >= hhigh[i + j] for j in range(1, 6))
        swing_low[i] = all(hlow[i] <= hlow[i - j] for j in range(1, 6)) and \
                       all(hlow[i] <= hlow[i + j] for j in range(1, 6))

    # Compute bias per H1 bar using up-to-that-bar info
    lookback = 5
    biases = []
    for i in range(h_count):
        if i < lookback * 3:
            biases.append({"bias": "NEUTRAL", "strength": 0.0})
            continue

        votes = 0.0
        if ema20[i] > ema50[i] and fast_slope[i] > 0:
            votes += 1.0
        elif ema20[i] < ema50[i] and fast_slope[i] < 0:
            votes -= 1.0

        if hclose[i] > ema50[i] and slow_slope[i] >= 0:
            votes += 0.5
        elif hclose[i] < ema50[i] and slow_slope[i] <= 0:
            votes -= 0.5

        high_indices = np.where(swing_high[:i + 1])[0]
        low_indices = np.where(swing_low[:i + 1])[0]

        if len(high_indices) >= 2 and len(low_indices) >= 2:
            recent_h = high_indices[-3:] if len(high_indices) >= 3 else high_indices
            recent_l = low_indices[-3:] if len(low_indices) >= 3 else low_indices
            h_vals = hhigh[recent_h]
            l_vals = hlow[recent_l]
            h_up = sum(1 for j in range(1, len(h_vals)) if h_vals[j] > h_vals[j-1])
            h_dn = sum(1 for j in range(1, len(h_vals)) if h_vals[j] < h_vals[j-1])
            l_up = sum(1 for j in range(1, len(l_vals)) if l_vals[j] > l_vals[j-1])
            l_dn = sum(1 for j in range(1, len(l_vals)) if l_vals[j] < l_vals[j-1])
            swing_score = (h_up - h_dn) + (l_up - l_dn)
            swing_total = max(1, (len(h_vals)-1) + (len(l_vals)-1))
            votes += swing_score / swing_total

        strength = min(abs(votes) / 1.5, 1.0) if votes != 0 else 0.0

        if votes >= 0.75:
            b = "BULLISH"
        elif votes <= -0.75:
            b = "BEARISH"
        else:
            b = "NEUTRAL"

        biases.append({"bias": b, "strength": strength})

    h1_highs = h1["high"].values.astype(float)
    h1_lows = h1["low"].values.astype(float)

    # For each M1 bar, get the PREVIOUS H1 bar's data (completed bar)
    h1_idx_prev = m1_h - 1
    h1_high_arr = np.where(h1_idx_prev >= 0, h1_highs[h1_idx_prev], np.nan)
    h1_low_arr = np.where(h1_idx_prev >= 0, h1_lows[h1_idx_prev], np.nan)
    h1_range_arr = h1_high_arr - h1_low_arr

    bias_dir_arr = np.full(n, "NEUTRAL", dtype=object)
    bias_str_arr = np.zeros(n)
    for j in range(n):
        h_idx = m1_h[j]
        b = biases[min(h_idx, len(biases) - 1)]
        bias_dir_arr[j] = b["bias"]
        bias_str_arr[j] = b["strength"]

    # Debug: check ATR and range distribution
    valid_atr = atr_arr[atr_arr > 0]
    valid_range = h1_range_arr[~np.isnan(h1_range_arr) & (h1_range_arr > 0)]
    print(f"  M1 ATR(14) stats:  mean={np.mean(valid_atr):.4f}  median={np.median(valid_atr):.4f}  "
          f"p1={np.percentile(valid_atr,1):.4f}  p99={np.percentile(valid_atr,99):.4f}", flush=True)
    print(f"  H1 range stats:    mean={np.mean(valid_range):.4f}  median={np.median(valid_range):.4f}  "
          f"p1={np.percentile(valid_range,1):.4f}  p99={np.percentile(valid_range,99):.4f}", flush=True)

    # Score = breakout_dist / range, capped at 1.0
    buy_breakout = mv["c"] - h1_high_arr  # positive when price above H1 high
    sell_breakout = h1_low_arr - mv["c"]  # positive when price below H1 low
    scores = np.zeros(n)
    for j in range(n):
        if h1_range_arr[j] > 0:
            if bias_dir_arr[j] == "BULLISH" and buy_breakout[j] > 0:
                scores[j] = min(buy_breakout[j] / h1_range_arr[j], 1.0)
            elif bias_dir_arr[j] == "BEARISH" and sell_breakout[j] > 0:
                scores[j] = min(sell_breakout[j] / h1_range_arr[j], 1.0)

    # Effective entry threshold
    entry_act_thresh = np.zeros(n)
    for j in range(n):
        if h1_range_arr[j] > 0 and atr_arr[j] > 0:
            ath = atr_arr[j] * cfg.ATR_MULTIPLIER / h1_range_arr[j]
        else:
            ath = None
        entry_act_thresh[j] = ath if ath is not None else cfg.SIGNAL_ENTRY_THRESHOLD

    # Entry conditions
    can_enter = (bias_dir_arr != "NEUTRAL") & (bias_str_arr >= cfg.BIAS_STRENGTH_MIN) & \
                ~np.isnan(h1_high_arr) & ~np.isnan(h1_low_arr) & (h1_range_arr > 0) & \
                (scores >= cfg.MIN_BREAKOUT_SCORE) & (scores >= entry_act_thresh)

    # Risk management
    bal = INITIAL_BALANCE
    trades = []
    cur_day = None
    day_pnl = 0.0
    session_trades = 0
    consecutive_losses = 0
    cooldown_until = -1  # bar index
    entry_active = False

    allowed_sessions = [s.strip().upper() for s in cfg.ALLOWED_SESSIONS.split(",")]
    def in_session(h, minute):
        t = h + minute / 60.0
        for s in allowed_sessions:
            if s == "ASIA" and 0 <= t < 8:
                return True
            if s == "LONDON" and 8 <= t < 17:
                return True
            if s == "NEW_YORK" and 13 <= t < 22:
                return True
        return False

    for idx in range(200, n):
        # Day boundary reset
        ts = mv["t"][idx]
        dt = pd.Timestamp(ts)
        if cur_day != dt.date():
            cur_day = dt.date()
            day_pnl = 0.0
            session_trades = 0
            consecutive_losses = 0

        # Risk checks before entry
        if not entry_active:
            if not in_session(dt.hour, dt.minute):
                continue
            if session_trades >= cfg.MAX_TRADES_PER_SESSION:
                continue
            if consecutive_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
                continue
            if idx < cooldown_until:
                continue

        if not entry_active and can_enter[idx]:
            direction = "BUY" if bias_dir_arr[idx] == "BULLISH" else "SELL"
            price = mv["c"][idx]
            atr_v = atr_arr[idx]

            if atr_v <= 0:
                continue

            # Build signal for exit (pure ATR-based SL — no H1 override)
            if direction == "BUY":
                sl_p = price - atr_v * cfg.SL_ATR_MULTIPLIER
                tp1_p = price + atr_v * cfg.TP1_MULTIPLIER
                tp2_p = price + atr_v * cfg.TP2_MULTIPLIER
                tp3_p = price + atr_v * cfg.TP3_MULTIPLIER
            else:
                sl_p = price + atr_v * cfg.SL_ATR_MULTIPLIER
                tp1_p = price - atr_v * cfg.TP1_MULTIPLIER
                tp2_p = price - atr_v * cfg.TP2_MULTIPLIER
                tp3_p = price - atr_v * cfg.TP3_MULTIPLIER

            entry_info = {
                "time": mv["t"][idx],
                "price": price,
                "direction": direction,
                "score": scores[idx],
                "bar": idx,
                "sl": sl_p, "tp1": tp1_p, "tp2": tp2_p, "tp3": tp3_p,
                "atr": atr_v,
                "bias": bias_dir_arr[idx],
            }
            entry_active = True

        if entry_active:
            info = entry_info
            px = mv["c"][idx]
            direction = info["direction"]
            diff = px - info["price"] if direction == "BUY" else info["price"] - px
            mom = mom_arr[idx]
            sl = info["sl"]
            tp1 = info["tp1"]
            tp2 = info["tp2"]
            tp3 = info["tp3"]
            atr_v = atr_arr[idx]

            exited = False
            reason = ""

            if direction == "BUY":
                # hard SL
                if px <= sl:
                    exited, reason = True, "stop_loss"
                else:
                    tp1_progress = (px - info["price"]) / (tp1 - info["price"]) if tp1 != info["price"] else 0

                    # dynamic SL
                    if px >= tp2:
                        sl = max(sl, px - atr_v * 2)
                    elif px >= tp1:
                        sl = max(sl, info["price"] + (tp1 - info["price"]) * 0.3)
                    elif tp1_progress >= 0.6:
                        sl = max(sl, info["price"] + atr_v * 0.3)

                    if px <= sl:
                        exited, reason = True, "stop_loss"
                    else:
                        tp_levels = [(tp1, "tp1"), (tp2, "tp2"), (tp3, "tp3")]
                        active_tps = [(t, n) for t, n in tp_levels if px < t]
                        if not active_tps:
                            trail = px - atr_v * 2
                            sl = max(info["sl"], trail)
                            if px <= sl:
                                exited, reason = True, "trailing_stop"
                            elif mom < cfg.TP_CLOSE_MOMENTUM_MIN:
                                exited, reason = True, "momentum_exhausted"
                        else:
                            nearest_tp, tp_name = active_tps[0]
                            progress = diff / (nearest_tp - info["price"]) if nearest_tp != info["price"] else 1.0
                            progress = min(progress, 1.0)

                            # wick rejection near TP
                            if progress >= 0.8:
                                candle = mv
                                upper_wick = candle["h"][idx] - max(candle["c"][idx], candle["o"][idx])
                                lower_wick = min(candle["c"][idx], candle["o"][idx]) - candle["l"][idx]
                                total_range = candle["h"][idx] - candle["l"][idx]
                                if total_range > 0 and upper_wick / total_range > 0.6:
                                    exited, reason = True, f"wick_rejection_{tp_name}"

                            if not exited and progress >= cfg.TP_CLOSE_THRESHOLD and mom < cfg.TP_CLOSE_MOMENTUM_MIN:
                                exited, reason = True, f"take_profit_{tp_name}"
            else:
                if px >= sl:
                    exited, reason = True, "stop_loss"
                else:
                    tp1_progress = (info["price"] - px) / (info["price"] - tp1) if info["price"] != tp1 else 0

                    if px <= tp2:
                        sl = min(sl, px + atr_v * 2)
                    elif px <= tp1:
                        sl = min(sl, info["price"] - (info["price"] - tp1) * 0.3)
                    elif tp1_progress >= 0.6:
                        sl = min(sl, info["price"] - atr_v * 0.3)

                    if px >= sl:
                        exited, reason = True, "stop_loss"
                    else:
                        tp_levels = [(tp1, "tp1"), (tp2, "tp2"), (tp3, "tp3")]
                        active_tps = [(t, n) for t, n in tp_levels if px > t]
                        if not active_tps:
                            trail = px + atr_v * 2
                            sl = min(info["sl"], trail)
                            if px >= sl:
                                exited, reason = True, "trailing_stop"
                            elif mom < cfg.TP_CLOSE_MOMENTUM_MIN:
                                exited, reason = True, "momentum_exhausted"
                        else:
                            nearest_tp, tp_name = active_tps[0]
                            progress = diff / (info["price"] - nearest_tp) if info["price"] != nearest_tp else 1.0
                            progress = min(progress, 1.0)

                            if progress >= 0.8:
                                candle = mv
                                upper_wick = candle["h"][idx] - max(candle["c"][idx], candle["o"][idx])
                                lower_wick = min(candle["c"][idx], candle["o"][idx]) - candle["l"][idx]
                                total_range = candle["h"][idx] - candle["l"][idx]
                                if total_range > 0 and lower_wick / total_range > 0.6:
                                    exited, reason = True, f"wick_rejection_{tp_name}"

                            if not exited and progress >= cfg.TP_CLOSE_THRESHOLD and mom < cfg.TP_CLOSE_MOMENTUM_MIN:
                                exited, reason = True, f"take_profit_{tp_name}"

            if exited:
                delta = (px - info["price"]) if direction == "BUY" else (info["price"] - px)
                pnl = round(delta * CONTRACT_SIZE * cfg.LOT_SIZE, 2)
                bal += pnl

                sl_dist = abs(info["price"] - info["sl"])
                tp1_dist = abs(info["tp1"] - info["price"])
                avg_dist = (sl_dist + tp1_dist) / 2
                actual_dist = abs(px - info["price"])

                # Update risk tracking
                session_trades += 1
                day_pnl += pnl
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                cooldown_until = idx + cfg.RE_ENTRY_COOLDOWN_SEC // 60  # 600s = 10 M1 bars

                trades.append({
                    "entry_time": str(info["time"]),
                    "entry_price": round(info["price"], 2),
                    "direction": direction,
                    "exit_time": str(mv["t"][idx]),
                    "exit_price": round(px, 2),
                    "pnl": pnl,
                    "reason": reason,
                    "balance": round(bal, 2),
                    "score": round(info["score"], 3),
                    "hold_bars": idx - info["bar"],
                    "bias": info["bias"],
                    "atr": round(info["atr"], 4),
                    "sl_price": round(info["sl"], 2),
                    "tp1_price": round(info["tp1"], 2),
                    "tp2_price": round(info["tp2"], 2),
                    "tp3_price": round(info["tp3"], 2),
                    "sl_dist": round(sl_dist, 2),
                    "tp1_dist": round(tp1_dist, 2),
                    "actual_dist": round(actual_dist, 2),
                })
                entry_active = False

    return trades, bal


print("Loading XAUUSD M1 2024...", flush=True)
client = DukascopyClient()
m1 = client.download_range(2024, 2024)
if len(m1) == 0:
    print("No data loaded")
    exit()
print(f"  {len(m1):,} M1 bars", flush=True)

print("Running H1 Breakout + Multi-TP backtest...", flush=True)
t0 = time.time()
trades, end_bal = run_backtest(m1)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s, {len(trades)} trades", flush=True)

if not trades:
    print("No trades generated")
    exit()

df = pd.DataFrame(trades)
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["exit_time"] = pd.to_datetime(df["exit_time"])
df["month"] = df["entry_time"].dt.to_period("M")
df["entry_hour"] = df["entry_time"].dt.hour

total_pnl = df["pnl"].sum()
wins = df[df["pnl"] > 0]
losses = df[df["pnl"] < 0]
neutrals = df[df["pnl"] == 0]
wr = len(wins) / len(df) * 100
gp = wins["pnl"].sum() if len(wins) > 0 else 0
gl = losses["pnl"].sum() if len(losses) > 0 else 0
pf = abs(gp / gl) if gl != 0 else (999 if gp > 0 else 0)
df["cum"] = df["pnl"].cumsum()
dd = (df["cum"].cummax() - df["cum"]).max()
avg_w = wins["pnl"].mean() if len(wins) > 0 else 0
avg_l = losses["pnl"].mean() if len(losses) > 0 else 0

print(f"\n{'='*80}")
print(f"H1 BREAKOUT + M1 MOMENTUM — Exit Mode 6 (Multi-TP)")
print(f"{'='*80}")
print(f"Config: SL={cfg.SL_ATR_MULTIPLIER}x  TP1={cfg.TP1_MULTIPLIER}x  TP2={cfg.TP2_MULTIPLIER}x  TP3={cfg.TP3_MULTIPLIER}x")
print(f"        EntryThreshold={cfg.SIGNAL_ENTRY_THRESHOLD}  MinBreakoutScore={cfg.MIN_BREAKOUT_SCORE}")
print(f"        TP_CLOSE_THRESHOLD={cfg.TP_CLOSE_THRESHOLD}  TP_CLOSE_MOMENTUM_MIN={cfg.TP_CLOSE_MOMENTUM_MIN}")
print(f"        BIAS_STRENGTH_MIN={cfg.BIAS_STRENGTH_MIN}  LOT_SIZE={cfg.LOT_SIZE}")
print(f"Risk:    MaxSessionTrades={cfg.MAX_TRADES_PER_SESSION}  MaxConsecLoss={cfg.MAX_CONSECUTIVE_LOSSES}")
print(f"        DailyLossLimit=${cfg.MAX_DAILY_LOSS_USD:.0f}  EventLossLimit=${cfg.MAX_EVENT_LOSS_USD:.0f}")
print(f"        Cooldown={cfg.RE_ENTRY_COOLDOWN_SEC}s  Sessions={cfg.ALLOWED_SESSIONS}")
print(f"Trades:        {len(df)}")
print(f"Wins:          {len(wins)} ({wr:.1f}%)")
if len(neutrals) > 0:
    print(f"Breakeven:     {len(neutrals)} ({len(neutrals)/len(df)*100:.1f}%)")
print(f"Losses:        {len(losses)} ({100-wr:.1f}%)")
print(f"Net PnL:       ${total_pnl:+.2f}")
print(f"Profit Factor: {pf:.2f}")
print(f"Max Drawdown:  ${dd:.2f}")
print(f"Avg Win:       ${avg_w:.2f}")
print(f"Avg Loss:      ${avg_l:.2f}")
if avg_l != 0:
    print(f"Avg RR:        {abs(avg_w/avg_l):.2f}")
print(f"Return:        {total_pnl/INITIAL_BALANCE*100:.1f}%")
print(f"End Balance:   ${end_bal:.2f}")

# Monthly
print(f"\n{'='*80}")
print(f"MONTHLY")
print(f"{'='*80}")
for m, g in df.groupby("month"):
    w = g[g["pnl"] > 0]
    l = g[g["pnl"] < 0]
    wr_m = len(w)/len(g)*100 if len(g) > 0 else 0
    gp_m = w["pnl"].sum() if len(w) > 0 else 0
    gl_m = l["pnl"].sum() if len(l) > 0 else 0
    pf_m = abs(gp_m/gl_m) if gl_m != 0 else (999 if gp_m > 0 else 0)
    g_cum = g["pnl"].cumsum()
    dd_m = (g_cum.cummax()-g_cum).max()
    peak_bal = g_cum.max()
    print(f"  {str(m):<10} PnL=${g['pnl'].sum():+7.2f}  Trades={len(g):4}  WR={wr_m:.1f}%  PF={pf_m:.2f}  DD=${dd_m:<6.2f}")

# Exit reasons
print(f"\n{'='*80}")
print(f"EXIT REASONS")
print(f"{'='*80}")
for reason, g in df.groupby("reason"):
    w = g[g["pnl"] > 0]
    print(f"  {reason:<30} {len(g):4} trades  {len(w):3} wins  ${g['pnl'].sum():+7.2f} PnL")

# Hourly
print(f"\n{'='*80}")
print(f"BY HOUR")
print(f"{'='*80}")
for h, g in df.groupby("entry_hour"):
    w = g[g["pnl"] > 0]
    print(f"  {h:02d}:00  {len(g):4} trades  ${g['pnl'].sum():+7.2f} PnL  {len(w):3} wins")

# Direction
print(f"\n{'='*80}")
print(f"BY DIRECTION")
print(f"{'='*80}")
for d, g in df.groupby("direction"):
    w = g[g["pnl"] > 0]
    print(f"  {d:<6}  {len(g):4} trades  ${g['pnl'].sum():+7.2f} PnL  {len(w):3} wins")

# SL/TP distance stats
print(f"\n{'='*80}")
print(f"SL/TP DISTANCE ANALYSIS")
print(f"{'='*80}")
print(f"  Avg SL distance:   ${df['sl_dist'].mean():.2f}  (entry to SL)")
print(f"  Avg TP1 distance:  ${df['tp1_dist'].mean():.2f}")
print(f"  Avg actual exit:   ${df['actual_dist'].mean():.2f}  (entry to exit price)")
print(f"  Avg ATR at entry:  ${df['atr'].mean():.4f}")
print(f"  Avg hold bars:     {df['hold_bars'].mean():.0f}")
print()
# By exit reason
for reason in ["stop_loss", "take_profit_tp1", "take_profit_tp2", "take_profit_tp3", "momentum_exhausted"]:
    g = df[df["reason"] == reason]
    if len(g) > 0:
        print(f"  {reason:<25}  count={len(g):4}  "
              f"actual_dist=${g['actual_dist'].mean():.2f}  "
              f"hold_bars={g['hold_bars'].mean():.0f}  "
              f"atr=${g['atr'].mean():.2f}")

# Daily stats
df_day = df.groupby(df["entry_time"].dt.date)["pnl"].sum()
print(f"\n{'='*80}")
print(f"DAILY STATS")
print(f"{'='*80}")
print(f"Trading days:    {len(df_day)}")
print(f"Per day avg:     ${df_day.mean():+.2f}")
print(f"Winning days:    {(df_day>0).sum()} ({(df_day>0).sum()/len(df_day)*100:.1f}%)")
print(f"Losing days:     {(df_day<0).sum()} ({(df_day<0).sum()/len(df_day)*100:.1f}%)")
print(f"Best day:        ${df_day.max():+.2f}")
print(f"Worst day:       ${df_day.min():+.2f}")
print(f"Std dev:         ${df_day.std():.2f}")

# Best/worst
print(f"\n{'='*80}")
print(f"TOP TRADES")
print(f"{'='*80}")
for _, row in df.nlargest(5, "pnl").iterrows():
    et = row["entry_time"].strftime("%m/%d %H:%M")
    print(f"  BEST  {row['direction']} ${row['pnl']:+6.2f} on {et}  {row['reason']}  score={row['score']}")
print()
for _, row in df.nsmallest(5, "pnl").iterrows():
    et = row["entry_time"].strftime("%m/%d %H:%M")
    print(f"  WORST {row['direction']} ${row['pnl']:+6.2f} on {et}  {row['reason']}  score={row['score']}")

out_path = os.path.join(os.path.dirname(__file__), "backtest_h1_breakout_multi_tp_2024.json")
df.to_json(out_path, orient="records", date_format="iso", default_handler=str)
print(f"\nSaved to {out_path}")
