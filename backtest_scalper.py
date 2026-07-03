"""
Backtest on real Dukascopy data.
Faithfully replicates the H1 bias + M1 EMA5/13 entry + SL/TP/trail logic.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

from app.dukascopy_client import DukascopyClient

# ── config ──

LOT_SIZE = 0.01
POINT = 0.0001
INITIAL_BALANCE = 10.0

MAX_DAILY_LOSS = 5.0
MAX_TRADES_PER_SESSION = 3
MAX_CONSECUTIVE_LOSSES = 2
RE_ENTRY_COOLDOWN_SEC = 60

SIGNAL_ENTRY_THRESHOLD = 0.55
ATR_PERIOD = 14
SL_ATR = 1.5
TP1_ATR = 2.25
TP2_ATR = 3.75

BIAS_STRENGTH_MIN = 0.25

# ── indicators ──

def ema(values, period):
    if len(values) < period:
        return np.full_like(values, values[-1] if len(values) > 0 else 0)
    mult = 2.0 / (period + 1)
    result = np.empty_like(values)
    result[:period] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = (values[i] - result[i-1]) * mult + result[i-1]
    return result

def rsi(closes, period=14):
    N = len(closes)
    if N < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains[-period:])
    avg_l = np.mean(losses[-period:])
    if avg_l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)

def atr(high, low, close, period=14):
    N = len(high)
    if N < period + 1:
        return 0.0001
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    return float(np.mean(tr[-period:]))

def momentum(closes, high, low, open_, period=14):
    N = len(closes)
    if N < 6:
        return 0.0
    recent = closes[-3:]
    older = closes[-6:-3]
    rc = abs(recent[-1] - recent[0])
    oc = abs(older[-1] - older[0]) if len(older) >= 2 else rc
    atr_v = atr(high, low, closes, period) or 0.0001
    avg_body = np.mean(np.abs(closes[-5:] - open_[-5:]))
    raw = (rc / (oc + 1e-10)) * (avg_body / (atr_v + 1e-10))
    return min(abs(raw), 1.0)

def volume_surge(volumes, lookback=10):
    if len(volumes) < lookback + 1:
        return True
    recent = volumes[-1]
    avg_v = np.mean(volumes[-(lookback+1):-1])
    if avg_v <= 0:
        return True
    return recent > avg_v * 1.2

# ── helpers ──

def pnl(entry, exit_, direction, lot):
    delta = exit_ - entry if direction == "BUY" else entry - exit_
    return delta / POINT * lot * 10

def session_ok(ts):
    h = ts.hour + ts.minute / 60.0
    in_london = 7 <= h < 16
    in_ny = 12 <= h < 21
    return in_london or in_ny

def resample(df, rule):
    d = df.set_index("time")
    r = d.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum", "spread": "mean",
    }).dropna().reset_index()
    return r

# ── bias engine ──

def compute_bias(h1: pd.DataFrame) -> Tuple[str, float]:
    if h1 is None or len(h1) < 55:
        return "NEUTRAL", 0.0
    closes = h1["close"].values
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)
    price = closes[-1]
    above_ema21 = price > ema21[-1]
    above_ema50 = price > ema50[-1]
    ema21_slope = ema21[-1] - ema21[-4] if len(ema21) >= 4 else 0
    ema50_slope = ema50[-1] - ema50[-6] if len(ema50) >= 6 else 0

    votes = 0.0
    votes += 0.5 if above_ema21 else -0.5
    votes += 0.5 if above_ema50 else -0.5
    votes += 0.3 if ema21_slope > 0 else -0.3 if ema21_slope < 0 else 0
    votes += 0.2 if ema50_slope > 0 else -0.2 if ema50_slope < 0 else 0

    # Swing structure
    highs, lows = h1["high"].values, h1["low"].values
    n = len(h1)
    swing_highs, swing_lows = [], []
    lb = 5
    for i in range(lb, n - lb):
        if all(highs[i] >= highs[i-j] for j in range(1, lb+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, lb+1)):
            swing_highs.append(highs[i])
        if all(lows[i] <= lows[i-j] for j in range(1, lb+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, lb+1)):
            swing_lows.append(lows[i])

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        h_up = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] > swing_highs[i-1])
        h_dn = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] < swing_highs[i-1])
        l_up = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] > swing_lows[i-1])
        l_dn = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] < swing_lows[i-1])
        swing = (h_up - h_dn) + (l_up - l_dn)
        votes += swing / max(1, (len(swing_highs)-1) + (len(swing_lows)-1))

    strength = min(abs(votes) / 2.0, 1.0)
    bias = "BULLISH" if votes >= 0.8 else "BEARISH" if votes <= -0.8 else "NEUTRAL"
    return bias, strength


# ── entry signal ──

def find_signal(m1_win, m5_win, bid, ask, bias):
    if bias not in ("BULLISH", "BEARISH"):
        return None

    direction = "BUY" if bias == "BULLISH" else "SELL"
    entry = ask if direction == "BUY" else bid
    closes = m1_win["close"].values
    opens = m1_win["open"].values
    highs = m1_win["high"].values
    lows = m1_win["low"].values
    vols = m1_win["tick_volume"].values if "tick_volume" in m1_win.columns else np.ones(len(m1_win))

    e5 = ema(closes, 5)
    e13 = ema(closes, 13)
    e21 = ema(closes, 21)
    atr_v = atr(highs, lows, closes, 14)
    if atr_v <= 0:
        atr_v = 0.0001

    rsi_m5 = rsi(m5_win["close"].values, 14) if m5_win is not None and len(m5_win) >= 14 else 50.0

    cross_over = e5[-1] > e13[-1] and e5[-2] <= e13[-2]
    cross_under = e5[-1] < e13[-1] and e5[-2] >= e13[-2]
    near_ema13 = abs(closes[-1] - e13[-1]) <= atr_v * 1.5
    vol_ok = volume_surge(vols)

    if direction == "BUY":
        if not (cross_over or (e5[-1] > e13[-1] and near_ema13)):
            return None
        if not (closes[-1] > e21[-1]):
            return None
        if rsi_m5 < 45:
            return None
        if not vol_ok:
            return None
    else:
        if not (cross_under or (e5[-1] < e13[-1] and near_ema13)):
            return None
        if not (closes[-1] < e21[-1]):
            return None
        if rsi_m5 > 55:
            return None
        if not vol_ok:
            return None

    sl_pips_pts = atr_v * SL_ATR
    tp1_pts = atr_v * TP1_ATR
    tp2_pts = atr_v * TP2_ATR

    if direction == "BUY":
        sl = entry - sl_pips_pts
        tp1 = entry + tp1_pts
        tp2 = entry + tp2_pts
    else:
        sl = entry + sl_pips_pts
        tp1 = entry - tp1_pts
        tp2 = entry - tp2_pts

    # Score
    score = 0.5
    spread_ema = abs(e5[-1] - e13[-1]) / atr_v
    score += min(spread_ema, 0.3)
    if cross_over or cross_under:
        score += 0.15
    if 40 <= rsi_m5 <= 60:
        score += 0.05
    mom = momentum(closes, highs, lows, opens, 14)
    score += mom * 0.1
    score = min(score, 1.0)

    if score < SIGNAL_ENTRY_THRESHOLD:
        return None

    return {
        "direction": direction,
        "entry_price": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "score": score,
        "rsi_m5": rsi_m5,
    }


# ── trade class ──

class Trade:
    def __init__(self, direction, entry_price, sl, tp1, tp2, signal, entry_idx):
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.signal = signal
        self.entry_idx = entry_idx
        self.entry_time = None
        self.exit_idx = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0
        self.bars_held = 0
        self._peak_profit = 0.0
        self._trail_sl = sl
        self._tp1_hit = False  # partial close flag
        self._tp1_pnl = 0.0


# ── main backtest ──

def run_backtest(m1_full):
    m1 = m1_full.sort_values("time").reset_index(drop=True)
    m5 = resample(m1, "5min")
    h1 = resample(m1, "1h")

    print(f"M1: {len(m1)} bars, M5: {len(m5)} bars, H1: {len(h1)} bars", flush=True)

    # Pre-index M5/H1 lookup
    m1_t = m1["time"].values.astype(np.int64)
    m5_t = m5["time"].values.astype(np.int64)
    h1_t = h1["time"].values.astype(np.int64)
    m5_idx_map = np.searchsorted(m5_t, m1_t, side="right") - 1
    h1_idx_map = np.searchsorted(h1_t, m1_t, side="right") - 1

    # Pre-compute H1 bias for each H1 bar
    h1_bias = []
    for i in range(len(h1)):
        win = h1.iloc[max(0, i-120):i+1].reset_index(drop=True)
        b, s = compute_bias(win)
        h1_bias.append((b, s))
    print(f"Pre-computed {len(h1_bias)} H1 bias values", flush=True)

    trades = []
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    consec_losses = 0
    daily_trades = 0
    cur_day = None
    daily_pnl = 0.0
    cooldown_until = None
    in_trade = False
    cur_trade = None
    N = len(m1)

    report_interval = max(1, N // 50)

    for idx in range(N):
        ts = m1.iloc[idx]["time"]
        bid = m1.iloc[idx]["close"]
        ask = bid
        mid = bid

        if cur_day != ts.date():
            cur_day = ts.date()
            daily_trades = 0
            daily_pnl = 0.0
            consec_losses = 0

        # ── Exit logic ──
        if in_trade and cur_trade is not None:
            t = cur_trade
            t.bars_held += 1

            diff = (mid - t.entry_price) if t.direction == "BUY" else (t.entry_price - mid)
            if diff > t._peak_profit:
                t._peak_profit = diff

            exit_reason = None
            exit_px = mid

            if t.direction == "BUY":
                if bid <= t._trail_sl:
                    exit_reason = "stop_loss"
                    exit_px = bid
                elif ask >= t.tp2:
                    exit_reason = "take_profit_full"
                    exit_px = ask
                elif ask >= t.tp1 and not t._tp1_hit:
                    # Partial close: book 60% of TP1, continue with remaining
                    tp1_pnl = pnl(t.entry_price, t.tp1, t.direction, LOT_SIZE * 0.6)
                    t._tp1_pnl = tp1_pnl
                    t._tp1_hit = True
                    # Update SL to breakeven after TP1 hit
                    t._trail_sl = t.entry_price
                    # No exit yet — remaining 40% runs to TP2
            else:
                if ask >= t._trail_sl:
                    exit_reason = "stop_loss"
                    exit_px = ask
                elif bid <= t.tp2:
                    exit_reason = "take_profit_full"
                    exit_px = bid
                elif bid <= t.tp1 and not t._tp1_hit:
                    tp1_pnl = pnl(t.entry_price, t.tp1, t.direction, LOT_SIZE * 0.6)
                    t._tp1_pnl = tp1_pnl
                    t._tp1_hit = True
                    t._trail_sl = t.entry_price

            # Trailing stop (after 1.5×ATR profit, trail back 0.8×ATR)
            # Get ATR for trail calculation
            mm = h1_idx_map[idx]
            if mm >= 0:
                h1_win = h1.iloc[max(0, mm-24):mm+1].reset_index(drop=True)
                atr_v = atr(h1_win["high"].values, h1_win["low"].values, h1_win["close"].values, 14)
            else:
                atr_v = 0.0008

            trail_trigger = atr_v * 1.5
            if t._peak_profit >= trail_trigger:
                trail_back = t._peak_profit - atr_v * 0.8
                if t.direction == "BUY":
                    new_sl = t.entry_price + trail_back
                    if new_sl > t._trail_sl:
                        t._trail_sl = new_sl
                else:
                    new_sl = t.entry_price - trail_back
                    if new_sl < t._trail_sl:
                        t._trail_sl = new_sl

            # Counter-candle streak exit
            if not exit_reason and idx >= 3:
                streak = 0
                for i in range(1, 5):
                    candle = m1.iloc[idx - i]
                    d = 1 if candle["close"] > candle["open"] else -1 if candle["close"] < candle["open"] else 0
                    if t.direction == "BUY" and d == -1:
                        streak += 1
                    elif t.direction == "SELL" and d == 1:
                        streak += 1
                    else:
                        break
                if streak >= 2:
                    exit_reason = "counter_candle"
                    exit_px = mid

            # Momentum decay exit
            if not exit_reason and t.bars_held >= 3 and idx >= 5:
                win5 = m1.iloc[idx-5:idx+1]
                mom = momentum(win5["close"].values, win5["high"].values, win5["low"].values, win5["open"].values, 14)
                if mom < 0.5:
                    exit_reason = "momentum_decay"
                    exit_px = mid

            if exit_reason:
                # Final P&L = TP1 partial (if hit) + remaining full lot
                remaining_pnl = pnl(t.entry_price, exit_px, t.direction, LOT_SIZE * (0.4 if t._tp1_hit else 1.0))
                total_pnl = t._tp1_pnl + remaining_pnl
                balance += total_pnl
                daily_pnl += total_pnl
                if balance > peak:
                    peak = balance
                trades.append({
                    "entry_time": t.entry_time,
                    "exit_time": ts,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": exit_px,
                    "pnl": round(total_pnl, 2),
                    "reason": exit_reason,
                    "bars_held": t.bars_held,
                    "balance": round(balance, 2),
                    "date": ts.date().isoformat(),
                    "score": t.signal.get("score", 0),
                })
                if total_pnl < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0
                cooldown_until = ts + timedelta(seconds=RE_ENTRY_COOLDOWN_SEC)
                in_trade = False
                cur_trade = None
                continue

        # ── Entry logic ──
        if not in_trade:
            if cooldown_until and ts < cooldown_until:
                continue
            cooldown_until = None

            if not session_ok(ts):
                continue
            if consec_losses >= MAX_CONSECUTIVE_LOSSES:
                continue
            if daily_trades >= MAX_TRADES_PER_SESSION:
                continue
            if daily_pnl <= -MAX_DAILY_LOSS:
                continue
            if balance < 5.0:
                continue

            # Get H1 bias for current bar
            hi = h1_idx_map[idx]
            if hi < 0:
                continue
            bias, strength = h1_bias[hi]
            if bias == "NEUTRAL" or strength < BIAS_STRENGTH_MIN:
                continue

            # M5 window
            mi = m5_idx_map[idx]
            if mi < 0:
                continue
            m5_win = m5.iloc[max(0, mi-48):mi+1].reset_index(drop=True)

            # M1 window
            m1_win = m1.iloc[max(0, idx-100):idx+1].reset_index(drop=True)

            signal = find_signal(m1_win, m5_win, bid, ask, bias)
            if signal:
                in_trade = True
                cur_trade = Trade(
                    signal["direction"], signal["entry_price"],
                    signal["sl"], signal["tp1"], signal["tp2"],
                    signal, idx,
                )
                cur_trade.entry_time = ts
                daily_trades += 1

        if (idx + 1) % report_interval == 0:
            pct = (idx + 1) / N * 100
            print(f"  [{pct:5.1f}%] idx={idx+1}/{N} trades={len(trades)} bal=${balance:.2f}", flush=True)

    # Close any open trade
    if in_trade and cur_trade is not None:
        t = cur_trade
        exit_px = m1["close"].iloc[-1]
        total_pnl = t._tp1_pnl + pnl(t.entry_price, exit_px, t.direction, LOT_SIZE * (0.4 if t._tp1_hit else 1.0))
        balance += total_pnl
        trades.append({
            "entry_time": t.entry_time,
            "exit_time": m1["time"].iloc[-1],
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": exit_px,
            "pnl": round(total_pnl, 2),
            "reason": "end_of_data",
            "bars_held": t.bars_held,
            "balance": round(balance, 2),
            "date": m1["time"].iloc[-1].date().isoformat(),
            "score": t.signal.get("score", 0),
        })

    if not trades:
        print("\nNo trades generated")
        return

    df_trades = pd.DataFrame(trades)

    total_pnl = df_trades["pnl"].sum()
    wins = df_trades[df_trades["pnl"] > 0]
    losses = df_trades[df_trades["pnl"] < 0]
    wr = len(wins) / len(df_trades) * 100
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")

    df_trades["cum"] = df_trades["pnl"].cumsum()
    df_trades["peak_cum"] = df_trades["cum"].cummax()
    df_trades["dd"] = df_trades["peak_cum"] - df_trades["cum"]
    max_dd = df_trades["dd"].max()

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Period:       {m1['time'].min().date()} to {m1['time'].max().date()}")
    print(f"  Starting Bal: ${INITIAL_BALANCE:.2f}")
    print(f"  Total Trades: {len(df_trades)}")
    print(f"  Wins:         {len(wins)}")
    print(f"  Losses:       {len(losses)}")
    print(f"  Win Rate:     {wr:.1f}%")
    print(f"  Gross Profit: ${gp:.2f}")
    print(f"  Gross Loss:   ${gl:.2f}")
    print(f"  Net P&L:      ${total_pnl:.2f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Max DD:       ${max_dd:.2f}")
    print(f"  Avg Win:      ${wins['pnl'].mean():.2f}" if len(wins) > 0 else "  Avg Win:      N/A")
    print(f"  Avg Loss:     ${losses['pnl'].mean():.2f}" if len(losses) > 0 else "  Avg Loss:     N/A")
    print(f"  Avg Bars Held: {df_trades['bars_held'].mean():.1f}")
    print(f"  Final Balance: ${balance:.2f}")
    print(f"  Return:       {(balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100:.2f}%")
    print(f"  Avg Trade PnL: ${total_pnl/len(df_trades):.2f}")

    # Daily breakdown
    print(f"\n{'='*60}")
    print(f"  DAILY BREAKDOWN")
    print(f"{'='*60}")
    df_trades["dt"] = pd.to_datetime(df_trades["date"])
    daily = df_trades.groupby(df_trades["dt"].dt.date).agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
    )
    daily["wr"] = (daily["wins"] / daily["trades"] * 100).round(1)
    print(f"  {'Date':<12} {'Trades':>7} {'P&L':>9} {'Wins':>5} {'WR':>6}")
    print(f"  {'-'*42}")
    profit_days = 0
    for date, row in daily.iterrows():
        label = "+" if row["pnl"] > 0 else "-"
        print(f"  {str(date):<12} {row['trades']:>7} ${row['pnl']:>+7.2f} {row['wins']:>5} {row['wr']:>5}%  {label}")
        if row["pnl"] > 0:
            profit_days += 1
    print(f"  {'-'*42}")
    print(f"  Profit Days: {profit_days}/{len(daily)} ({profit_days/len(daily)*100:.1f}%)")
    print(f"  Avg Daily:   ${total_pnl/len(daily):.2f}")

    # Score distribution
    if "score" in df_trades.columns:
        print(f"\n{'='*60}")
        print(f"  SCORE DISTRIBUTION")
        print(f"{'='*60}")
        for thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
            subset = df_trades[df_trades["score"] >= thr]
            if len(subset) > 0:
                sw = subset[subset["pnl"] > 0]
                print(f"  Score >= {thr:.2f}: {len(subset):>4} trades, "
                      f"{len(sw)/len(subset)*100:.1f}% WR, "
                      f"${subset['pnl'].sum():.2f} net")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    years = [2024]
    print(f"Loading EURUSD M1 data for {years}...", flush=True)
    client = DukascopyClient(symbol="EURUSD")
    m1 = client.download_range(min(years), max(years))
    if m1 is not None and len(m1) > 0:
        print(f"Loaded {len(m1)} M1 bars", flush=True)
        run_backtest(m1)
    else:
        print("No data loaded")
