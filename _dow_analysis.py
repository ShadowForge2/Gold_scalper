"""
Day-of-week analysis for small account startups.
Runs ML backtest across years, tallies PnL by entry day-of-week.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, time, json
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features, FEATURE_COLS

CS = 100
SPREAD = 0.30

def compute_atr_series(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return np.where(np.isnan(pd.Series(tr).rolling(period, min_periods=period).mean().values), 0.0, tr)

def apply_spread(price, is_buy):
    return price + SPREAD / 2 if is_buy else price - SPREAD / 2

def compute_bias(h1):
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

INITIAL_BALANCE = 20.0

def run_bt_year(year, pred):
    client = DukascopyClient()
    m1 = client.download_year(year).sort_values("time").drop_duplicates(subset="time")
    if len(m1) == 0:
        return None
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

    h1_bias = compute_bias(h1)
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

    entries = []  # collect each entry with its profit and entry time

    bal = INITIAL_BALANCE
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
                dow = pd.Timestamp(trade_entry_ts).dayofweek
                entries.append({
                    "dow": dow,
                    "profit": round(prof, 2),
                    "year": pd.Timestamp(trade_entry_ts).year,
                    "month": pd.Timestamp(trade_entry_ts).month,
                    "entry_time": trade_entry_ts,
                    "reason": exit_reason,
                    "bars": bars,
                })
                bal += prof
                in_trade = False
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

        ml_conf = 0.0
        if pb_u[mi] >= cfg.ML_CONFIDENCE_THRESHOLD and pb_u[mi] > pb_d[mi]:
            ml_dir, ml_c = "BUY", pb_u[mi]
        elif pb_d[mi] >= cfg.ML_CONFIDENCE_THRESHOLD and pb_d[mi] > pb_u[mi]:
            ml_dir, ml_c = "SELL", pb_d[mi]
        else: continue
        if ml_dir != entry_dir:
            new_bd = (px - h1h) if ml_dir == "BUY" else (h1l - px)
            if new_bd <= 0: continue
            entry_dir = ml_dir

        ml_conf = ml_c
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

    if in_trade:
        exit_px = apply_spread(m1_close[-1], trade_dir == "SELL")
        diff = (m1_close[-1] - trade_ep) if trade_dir == "BUY" else (trade_ep - m1_close[-1])
        prof = diff * CS * trade_lots * trade_positions
        dow = pd.Timestamp(trade_entry_ts).dayofweek
        entries.append({
            "dow": dow, "profit": round(prof, 2),
            "year": pd.Timestamp(trade_entry_ts).year,
            "month": pd.Timestamp(trade_entry_ts).month,
            "entry_time": trade_entry_ts, "reason": "end_of_data", "bars": 0,
        })
        bal += prof

    if not entries:
        return None

    df = pd.DataFrame(entries)
    wins = df[df["profit"] > 0]
    losses = df[df["profit"] <= 0]
    wr = len(wins)/len(df)*100 if len(df) > 0 else 0
    gw = wins["profit"].sum() if len(wins) > 0 else 0
    gl = abs(losses["profit"].sum()) if len(losses) > 0 else 0
    pf = gw/gl if gl > 0 else (999 if gw > 0 else 0)
    net = df["profit"].sum()

    return {
        "entries": df,
        "net_pnl": round(net, 2),
        "trades": len(df),
        "wins": len(wins),
        "wr": round(wr, 1),
        "pf": round(pf, 2),
    }

def main():
    pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
    all_entries = []

    years = list(range(2007, 2026))  # 2007-2025 inclusive
    print(f"Running day-of-week analysis across {years[0]}-{years[-1]}...")
    print()

    for year in years:
        t0 = time.time()
        r = run_bt_year(year, pred)
        if r is None:
            print(f"  {year}: no data or no trades")
            continue
        all_entries.append(r["entries"])
        el = time.time() - t0
        print(f"  {year}: {r['trades']} trades, net=${r['net_pnl']:+.2f}, "
              f"WR={r['wr']}%, PF={r['pf']} [{el:.0f}s]")

    if not all_entries:
        print("\nNo trade data collected.")
        return

    combined = pd.concat(all_entries, ignore_index=True)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    print(f"\n{'='*95}")
    print(f"  DAY-OF-WEEK ANALYSIS: {years[0]}-{years[-1]} ({len(combined)} total trades)")
    print(f"{'='*95}")
    print(f"  {'Day':<12} {'Trades':>7} {'Net PnL':>10} {'Avg PnL':>9} {'WR%':>6} {'PF':>7} "
          f"{'GW$':>8} {'GL$':>8} {'Green Days':>11}")
    print(f"  {'-'*80}")

    results = []
    for dow in range(5):  # Mon-Fri only
        day_df = combined[combined["dow"] == dow]
        if len(day_df) == 0:
            continue
        wins = day_df[day_df["profit"] > 0]
        losses = day_df[day_df["profit"] <= 0]
        wr = len(wins)/len(day_df)*100
        gw = wins["profit"].sum() if len(wins) > 0 else 0
        gl = abs(losses["profit"].sum()) if len(losses) > 0 else 0
        pf = gw/gl if gl > 0 else (999 if gw > 0 else 0)
        net = day_df["profit"].sum()

        # Per-trade statistics (for small account)
        trade_net = day_df["profit"]
        green_days_pct = len(trade_net[trade_net > 0]) / len(trade_net) * 100

        results.append({
            "day": day_names[dow], "dow": dow,
            "trades": len(day_df), "net_pnl": round(net, 2),
            "avg_pnl": round(net / len(day_df), 2),
            "wr": round(wr, 1), "pf": round(pf, 2),
            "gw": round(gw, 2), "gl": round(gl, 2),
            "green_pct": round(green_days_pct, 1),
        })

    # Sort by net PnL descending
    results.sort(key=lambda x: x["net_pnl"], reverse=True)
    for r in results:
        print(f"  {r['day']:<12} {r['trades']:>7} ${r['net_pnl']:>+8.2f} ${r['avg_pnl']:>+7.2f} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.2f} ${r['gw']:>7.2f} ${r['gl']:>7.2f} {r['green_pct']:>9.1f}%")

    # Small account startup simulation
    print(f"\n{'='*95}")
    print(f"  SMALL ACCOUNT ($20) STARTUP SIMULATION — BEST/WORST START DAY")
    print(f"{'='*95}")
    print(f"  Simulates depositing $20 and running the bot from each day of week.")
    print(f"  Measures balance after 7, 14, 30 days of trading (same calendar).")
    print(f"{'='*95}")

    # Simulate: starting on each day-of-week, track balance progression
    for start_dow in range(5):
        day_df = combined[combined["dow"] == start_dow].copy()
        day_df = day_df.sort_values("entry_time")

        # Track balance as if we deposited $20 and only traded on entries from this DOW
        bal = INITIAL_BALANCE
        bal_history = [(day_df.iloc[0]["entry_time"], bal)]
        for _, trade in day_df.iterrows():
            bal += trade["profit"]
            bal_history.append((trade["entry_time"], bal))
        bal_hist_df = pd.DataFrame(bal_history, columns=["time", "balance"])

        # Check balance milestones
        if len(bal_hist_df) > 1:
            first = bal_hist_df["time"].min()
            for label, days in [("7 days", 7), ("14 days", 14), ("30 days", 30)]:
                cutoff = first + pd.Timedelta(days=days)
                snapshot = bal_hist_df[bal_hist_df["time"] <= cutoff]
                if len(snapshot) > 0:
                    final_bal = snapshot["balance"].iloc[-1]
                else:
                    final_bal = INITIAL_BALANCE
                print(f"  Start {day_names[start_dow]:<10}: {label:>7} balance = ${final_bal:>+7.2f}")

    # --- SIMULATION: deposit on each day and trade ALL following days ---
    print(f"\n{'='*95}")
    print(f"  LIVE SIMULATION: deposit $20 on a given day, trade all subsequent days")
    print(f"{'='*95}")
    print(f"  This is more realistic: you deposit on day X, then the bot trades normally")
    print(f"  on all days (not just entries from day X). This measures which START day")
    print(f"  gives the best trajectory.")
    print()

    # Get all trades sorted by time
    all_sorted = combined.sort_values("entry_time")

    # For each possible start day-of-week, simulate starting on the first occurrence
    # and measure balance over the next N calendar days
    first_day = all_sorted["entry_time"].min()
    last_day = all_sorted["entry_time"].max()

    for start_dow in range(5):
        # Find first trade on this DOW
        mask = all_sorted["dow"] == start_dow
        if not mask.any():
            continue
        first_trade_ts = all_sorted[mask]["entry_time"].iloc[0]

        # Simulate: start with $20 at first_trade_ts
        sim_start = pd.Timestamp(first_trade_ts)
        sim_bal = INITIAL_BALANCE
        sim_trades = all_sorted[all_sorted["entry_time"] >= sim_start].copy()

        for _, t in sim_trades.iterrows():
            sim_bal += t["profit"]

        total_net = sim_bal - INITIAL_BALANCE
        n_trades = len(sim_trades)
        actual_years = (last_day - sim_start).days / 365.25
        annualized = (total_net / INITIAL_BALANCE) / actual_years * 100 if actual_years > 0 else 0

        # Also compute first 7, 14, 30 days
        early = sim_trades[sim_trades["entry_time"] <= sim_start + pd.Timedelta(days=7)]
        early_bal = INITIAL_BALANCE + early["profit"].sum()
        early2 = sim_trades[sim_trades["entry_time"] <= sim_start + pd.Timedelta(days=14)]
        bal_14 = INITIAL_BALANCE + early2["profit"].sum()
        early3 = sim_trades[sim_trades["entry_time"] <= sim_start + pd.Timedelta(days=30)]
        bal_30 = INITIAL_BALANCE + early3["profit"].sum()

        # First trade performance
        first_profit = sim_trades.iloc[0]["profit"] if len(sim_trades) > 0 else 0

        print(f"  Start {day_names[start_dow]:<10}: "
              f"1st trade=${first_profit:>+5.2f} | "
              f"Day7=${early_bal:>5.2f} | "
              f"Day14=${bal_14:>5.2f} | "
              f"Day30=${bal_30:>5.2f} | "
              f"Final=${sim_bal:>+7.2f} ({n_trades}t) | "
              f"Ann={annualized:>+5.0f}%")

    print(f"\n{'='*95}")
    print(f"  SUMMARY: Best day to deposit = highest early balance + most consistent trajectory")
    print(f"{'='*95}")

    # Score each day by: avg(rank by 7d bal, rank by 30d bal, rank by final bal, rank by WR)
    day_scores = {}
    for start_dow in range(5):
        mask = all_sorted["dow"] == start_dow
        if not mask.any():
            continue
        first_trade_ts = all_sorted[mask]["entry_time"].iloc[0]
        sim_start = pd.Timestamp(first_trade_ts)
        sim_bal = INITIAL_BALANCE
        sim_trades = all_sorted[all_sorted["entry_time"] >= sim_start].copy()
        bal_7 = INITIAL_BALANCE + sim_trades[sim_trades["entry_time"] <= sim_start + pd.Timedelta(days=7)]["profit"].sum()
        bal_14 = INITIAL_BALANCE + sim_trades[sim_trades["entry_time"] <= sim_start + pd.Timedelta(days=14)]["profit"].sum()
        bal_30 = INITIAL_BALANCE + sim_trades[sim_trades["entry_time"] <= sim_start + pd.Timedelta(days=30)]["profit"].sum()
        final = sim_bal
        wr_entries = sim_trades
        wr = len(wr_entries[wr_entries["profit"] > 0]) / len(wr_entries) * 100 if len(wr_entries) > 0 else 0
        total_trades = len(sim_trades)
        day_scores[day_names[start_dow]] = {
            "bal_7": bal_7, "bal_14": bal_14, "bal_30": bal_30,
            "final": final, "wr": wr, "trades": total_trades
        }

    for day, scores in sorted(day_scores.items(),
                               key=lambda x: x[1]["bal_30"] * 0.4 + x[1]["wr"] * 0.3 + x[1]["bal_7"] * 0.3,
                               reverse=True):
        print(f"  {day:<12}: Day7=${scores['bal_7']:>5.2f} "
              f"Day14=${scores['bal_14']:>5.2f} "
              f"Day30=${scores['bal_30']:>5.2f} "
              f"Final=${scores['final']:>+7.2f} "
              f"WR={scores['wr']:.1f}% "
              f"Trades={scores['trades']}")

if __name__ == "__main__":
    main()
