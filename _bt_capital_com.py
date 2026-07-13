"""
Live-constrained backtest with verified Capital.com costs:
- Spread: 0.30 points (confirmed on capital.com gold page)
- Commission: 0% (spread-only pricing)
- No slippage charges
- 1 lot = 100 oz, min 0.01 lots
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, time, random
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features, FEATURE_COLS

CS = 100
SPREAD = 0.30  # Capital.com XAUUSD spread (verified)

def compute_atr_series(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return np.where(np.isnan(pd.Series(tr).rolling(period, min_periods=period).mean().values), 0.0, tr)

def apply_spread(price, is_buy):
    """is_buy=True: pay ASK (enter BUY / exit SELL). is_buy=False: receive BID (enter SELL / exit BUY)."""
    return price + SPREAD / 2 if is_buy else price - SPREAD / 2

def _compute_bias(h1):
    c = h1["close"].values.astype(np.float64)
    h = h1["high"].values.astype(np.float64)
    l = h1["low"].values.astype(np.float64)
    fast = pd.Series(c).ewm(span=20, adjust=False).mean().values
    slow = pd.Series(c).ewm(span=50, adjust=False).mean().values
    fast_slope = np.full(len(c), 0.0)
    slow_slope = np.full(len(c), 0.0)
    if len(c) >= 6:
        fast_slope[5:] = fast[5:] - fast[:-5]
        slow_slope[5:] = slow[5:] - slow[:-5]
    votes = np.zeros(len(c))
    cross = (fast > slow) & (fast_slope > 0); votes[cross] += 1.0
    cross = (fast < slow) & (fast_slope < 0); votes[cross] -= 1.0
    price_above = c > slow
    votes[price_above & (slow_slope >= 0)] += 0.5
    votes[~price_above & (slow_slope <= 0)] -= 0.5
    n = len(c); lk = 5
    is_high = np.zeros(n, dtype=bool); is_low = np.zeros(n, dtype=bool)
    for i in range(lk, n - lk):
        is_high[i] = all(h[i] >= h[i - j] for j in range(1, lk + 1)) and all(h[i] >= h[i + j] for j in range(1, lk + 1))
        is_low[i] = all(l[i] <= l[i - j] for j in range(1, lk + 1)) and all(l[i] <= l[i + j] for j in range(1, lk + 1))
    swing_score = np.zeros(n)
    for i in range(n):
        hi = np.where(is_high[:i + 1])[0]; lo = np.where(is_low[:i + 1])[0]
        if len(hi) >= 3 and len(lo) >= 3:
            rh = hi[-3:]; rl = lo[-3:]
            h_up = sum(1 for k in range(1, len(rh)) if h[rh[k]] > h[rh[k - 1]])
            h_dn = sum(1 for k in range(1, len(rh)) if h[rh[k]] < h[rh[k - 1]])
            l_up = sum(1 for k in range(1, len(rl)) if l[rl[k]] > l[rl[k - 1]])
            l_dn = sum(1 for k in range(1, len(rl)) if l[rl[k]] < l[rl[k - 1]])
            swing_score[i] = ((h_up - h_dn) + (l_up - l_dn)) / max(1, (len(rh) - 1) + (len(rl) - 1))
    bias = np.zeros(n, dtype=np.int8)
    total = votes + swing_score
    bias[total >= 0.75] = 1; bias[total <= -0.75] = -1
    return bias

SESSION_HOURS = {"ASIA": (0, 8), "LONDON": (7, 17), "NEW_YORK": (12, 22)}

def run_bt(year, pred):
    client = DukascopyClient()
    m1 = client.download_year(year).sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    m1_close = m1["close"].values.astype(float)
    m1_time = m1["time"].values
    m5_close = m5["close"].values.astype(float)
    m5_open = m5["open"].values.astype(float)
    m5_high = m5["high"].values.astype(float)
    m5_low = m5["low"].values.astype(float)
    m5_atr = compute_atr_series(m5_high, m5_low, m5_close)

    ft = compute_features(m5, h1)
    feat_cols = pred._feature_cols if hasattr(pred, '_feature_cols') else FEATURE_COLS
    X = ft[ft.columns.intersection(feat_cols)]
    for c in feat_cols:
        if c not in X.columns: X[c] = 0.0
    X = X[feat_cols].reindex(m5.index, method="ffill")
    ml_mask = ~X.isna().any(axis=1)
    n_m5 = len(m5)
    pb_u = np.full(n_m5, np.nan); pb_d = np.full(n_m5, np.nan)
    if ml_mask.any():
        valid_idx = np.where(ml_mask)[0]
        probs = pred.model.predict_proba(X[ml_mask].values)
        pb_u[valid_idx] = np.array([p[1] for p in probs])
        pb_d[valid_idx] = np.array([p[0] for p in probs])

    h1_bias = _compute_bias(h1)
    h1_h = h1["high"].values.astype(float)
    h1_l = h1["low"].values.astype(float)
    m1_idx = pd.Index(m1_time)
    h1_idx_arr = np.clip(h1.index.get_indexer(m1_idx, method="ffill") - 1, 0, len(h1_bias) - 1)
    m5_idx_arr = m5.index.get_indexer(m1_idx, method="ffill")

    m1_hour = pd.DatetimeIndex(m1_time).hour.values
    session_mask = np.zeros(len(m1), dtype=bool)
    for s in cfg.ALLOWED_SESSIONS.split(","):
        s = s.strip().upper()
        if s in SESSION_HOURS:
            lo, hi = SESSION_HOURS[s]
            session_mask |= (m1_hour >= lo) & (m1_hour < hi)

    bal = 20.0; total_events = 0; wins = 0
    exit_reasons = {}
    event_log = []

    in_trade = False
    trade_ep = 0.0; trade_dir = ""; trade_lots = 0.0; trade_positions = 0
    trade_ml_conf = 0.0; trade_entry_ts = None; trade_entry_i = 0
    best_pnl = 0.0; wrong_streak = 0; wrong_candles = []

    for i in range(100, len(m1)):
        if not session_mask[i]: continue
        ts = m1_time[i]
        px = m1_close[i]
        hi = h1_idx_arr[i]
        mi = m5_idx_arr[i]
        if hi < 0 or mi < 0 or mi >= n_m5: continue

        if in_trade:
            bars = max(1, int((pd.Timestamp(ts) - pd.Timestamp(trade_entry_ts)).total_seconds() / 300))
            diff = (px - trade_ep) if trade_dir == "BUY" else (trade_ep - px)
            hi_diff = (m5_high[mi] - trade_ep) if trade_dir == "BUY" else (trade_ep - m5_low[mi])
            best_pnl = max(best_pnl, hi_diff)
            atr_j = max(m5_atr[mi], 0.01)
            is_wrong = (trade_dir == "BUY" and px < m5_open[mi]) or (trade_dir == "SELL" and px > m5_open[mi])
            wrong_streak = (wrong_streak + 1) if is_wrong else 0
            wrong_candles.append(1 if is_wrong else 0)
            if len(wrong_candles) > 7: wrong_candles.pop(0)

            exit_now = False; exit_reason = None
            if best_pnl > 0:
                tt = atr_j * cfg.PEAK_HARVEST_TRAIL_TRIGGER
                if best_pnl >= tt:
                    if (best_pnl - max(0, diff)) / best_pnl > cfg.PEAK_HARVEST_TRAIL_RETRACE:
                        exit_now = True; exit_reason = "trail_stop"
            if not exit_now and bars >= 4:
                streak = 0
                for k in range(min(cfg.DIRECTION_LOSS_LOOKBACK, mi)):
                    if trade_dir == "BUY" and m5_close[mi - k] < m5_open[mi - k]: streak += 1
                    elif trade_dir == "SELL" and m5_close[mi - k] > m5_open[mi - k]: streak += 1
                    else: break
                if streak >= cfg.DIRECTION_LOSS_STREAK:
                    exit_now = True; exit_reason = "direction_loss"
            if not exit_now and bars >= cfg.PEAK_HARVEST_MIN_BARS_EXIT and mi >= 10:
                recent = m5_close[mi - 2:mi + 1]
                older = m5_close[max(0, mi - 5):max(0, mi - 2)]
                if len(older) < 2: older = recent
                rc = abs(recent[-1] - recent[0])
                oc = abs(older[-1] - older[0]) if len(older) >= 2 else rc
                body = np.abs(m5_close[mi - 4:mi + 1] - m5_open[mi - 4:mi + 1]).mean()
                if body > 0:
                    watr = max(m5_atr[max(0, mi - 13):mi + 1].mean(), px * 0.0001)
                    mom = min(abs((rc / (oc + 1e-10)) * (body / (watr + 1e-10))), 1.0)
                    if (1.0 - mom) > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
                        exit_now = True; exit_reason = "momentum_decay"
            if not exit_now and not np.isnan(pb_d[mi]) and not np.isnan(pb_u[mi]):
                if trade_dir == "BUY" and pb_d[mi] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                    exit_now = True; exit_reason = "ml_reversal"
                elif trade_dir == "SELL" and pb_u[mi] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                    exit_now = True; exit_reason = "ml_reversal"
            if not exit_now and bars >= 25:
                exit_now = True; exit_reason = "max_hold"

            if exit_now:
                exit_px = apply_spread(px, trade_dir == "SELL")
                prof = diff * CS * trade_lots * trade_positions
                event_log.append({"event": total_events, "entry_time": trade_entry_ts, "exit_time": ts,
                                  "dir": trade_dir, "entry_px": round(trade_ep, 2),
                                  "exit_px": round(exit_px, 2),
                                  "reason": exit_reason, "profit": round(prof, 2),
                                  "bars": bars,
                                  "lots": trade_lots, "trades": trade_positions,
                                  "ml_conf": trade_ml_conf})
                bal += prof
                if prof > 0: wins += 1
                exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
                in_trade = False; total_events += 1
            continue

        # Entry
        bias_val = h1_bias[hi]
        if bias_val == 0: continue
        entry_dir = "BUY" if bias_val == 1 else "SELL"
        h1h = h1_h[hi]; h1l = h1_l[hi]
        if h1h <= h1l: continue
        if entry_dir == "BUY" and px <= h1h: continue
        if entry_dir == "SELL" and px >= h1l: continue
        if np.isnan(pb_u[mi]) or np.isnan(pb_d[mi]): continue
        if pb_u[mi] >= cfg.ML_CONFIDENCE_THRESHOLD and pb_u[mi] > pb_d[mi]:
            ml_dir, ml_c = "BUY", pb_u[mi]
        elif pb_d[mi] >= cfg.ML_CONFIDENCE_THRESHOLD and pb_d[mi] > pb_u[mi]:
            ml_dir, ml_c = "SELL", pb_d[mi]
        else: continue
        if ml_dir != entry_dir:
            new_bd = (px - h1h) if ml_dir == "BUY" else (h1l - px)
            if new_bd <= 0: continue
            entry_dir = ml_dir
        if ml_c >= cfg.ML_CONF_VERY_STRONG_THRESHOLD:
            n_pos = cfg.ML_POSITIONS_MAX; l_mul = cfg.ML_LOT_MULT_VERY_STRONG
        elif ml_c >= cfg.ML_CONF_STRONG_THRESHOLD:
            n_pos = cfg.ML_POSITIONS_VERY_STRONG; l_mul = cfg.ML_LOT_MULT_STRONG
        else:
            n_pos = cfg.ML_POSITIONS_STRONG; l_mul = cfg.ML_LOT_MULTIPLIER
        ep = apply_spread(px, entry_dir == "BUY")
        n_lots = max(cfg.MIN_LOT, min(1.0, round((bal * 0.001 * l_mul) / cfg.LOT_STEP) * cfg.LOT_STEP, cfg.MAX_LOT))
        n_lots = max(cfg.MIN_LOT, n_lots)
        in_trade = True
        trade_ep = ep; trade_dir = entry_dir
        trade_lots = n_lots; trade_positions = n_pos
        trade_ml_conf = ml_c; trade_entry_ts = ts; trade_entry_i = i
        best_pnl = 0.0; wrong_streak = 0; wrong_candles = []

    return {"events": total_events, "wins": wins, "net_pnl": round(bal - 20, 2),
            "exit_reasons": exit_reasons, "event_log": event_log}

if __name__ == "__main__":
    pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
    random.seed(42)

    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        t0 = time.time()
        r = run_bt(year, pred)
        el = time.time() - t0

        df = pd.DataFrame(r["event_log"])
        total = r["events"]; w = r["wins"]
        losses = total - w
        wr = w / total * 100 if total else 0
        gw = sum(e["profit"] for e in r["event_log"] if e["profit"] > 0)
        gl = abs(sum(e["profit"] for e in r["event_log"] if e["profit"] <= 0))
        pf = gw / gl if gl else float("inf")
        aw = gw / w if w else 0
        al = gl / losses if losses else 0
        at = sum(e["trades"] for e in r["event_log"]) / total if total else 0

        daily = df.groupby(pd.to_datetime(df["entry_time"]).dt.date).agg(
            events=("profit", "count"), pnl=("profit", "sum"))
        red = daily[daily["pnl"] <= 0]
        rp = len(red) / len(daily) * 100 if len(daily) else 0

        print(f"\n=== {year} (Capital.com costs) ===")
        print(f"  Events: {total} | WR: {wr:.1f}% | PF: {pf:.2f} | "
              f"Net: ${r['net_pnl']:+.2f} | "
              f"AvgW: ${aw:.2f} | AvgL: ${al:.2f} | AvgPos: {at:.1f}")
        print(f"  Days: {len(daily)} | Green: {len(daily)-len(red)} | Red: {len(red)} ({rp:.1f}%) | "
              f"Worst: ${daily['pnl'].min():.2f} | Best: ${daily['pnl'].max():.2f}")

        df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
        monthly = df.groupby("month")["profit"].sum()
        red_m = monthly[monthly <= 0]
        print(f"  Months: {len(monthly)} | Green: {len(monthly)-len(red_m)} | Red: {len(red_m)}")
        if len(red_m) > 0:
            for dt, p in red_m.items():
                print(f"    RED MONTH: {dt} | PnL=${p:+.2f}")
        print(f"  Time: {el:.1f}s")
