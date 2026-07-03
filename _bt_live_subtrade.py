"""Backtest: Real leverage + sequential sub-trade logic matching live bot."""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features
from app.meta_strategy import MetaStrategy
from app.risk_manager import EquityScaler

CS = 1
MARGIN_RATE = 0.05

SESSION_HOURS = {"ASIA": (0, 8), "LONDON": (7, 17), "NEW_YORK": (12, 22)}
def in_allowed_session(ts) -> bool:
    h = ts.hour
    sessions = [s.strip().upper() for s in cfg.ALLOWED_SESSIONS.split(",") if s.strip()]
    for s in sessions:
        if s in SESSION_HOURS:
            lo, hi = SESSION_HOURS[s]
            if lo <= h < hi: return True
    return False

def session_of_hour(h: int) -> str:
    for name, (lo, hi) in SESSION_HOURS.items():
        if lo <= h < hi: return name
    return "OUTSIDE"

def compute_bias_vectorized(h1):
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
    votes[(fast > slow) & (fast_slope > 0)] += 1.0
    votes[(fast < slow) & (fast_slope < 0)] -= 1.0
    price_above = c > slow
    votes[price_above & (slow_slope >= 0)] += 0.5
    votes[~price_above & (slow_slope <= 0)] -= 0.5
    lookback = 5; n = len(c)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        is_high[i] = all(h[i] >= h[i - j] for j in range(1, lookback + 1)) and \
                     all(h[i] >= h[i + j] for j in range(1, lookback + 1))
        is_low[i] = all(l[i] <= l[i - j] for j in range(1, lookback + 1)) and \
                    all(l[i] <= l[i + j] for j in range(1, lookback + 1))
    swing_score = np.zeros(n)
    for i in range(n):
        hi = np.where(is_high[:i + 1])[0]
        lo = np.where(is_low[:i + 1])[0]
        if len(hi) >= 3 and len(lo) >= 3:
            rh = hi[-3:]; rl = lo[-3:]
            h_up = sum(1 for k in range(1, len(rh)) if h[rh[k]] > h[rh[k - 1]])
            h_dn = sum(1 for k in range(1, len(rh)) if h[rh[k]] < h[rh[k - 1]])
            l_up = sum(1 for k in range(1, len(rl)) if l[rl[k]] > l[rl[k - 1]])
            l_dn = sum(1 for k in range(1, len(rl)) if l[rl[k]] < l[rl[k - 1]])
            swing_score[i] = ((h_up - h_dn) + (l_up - l_dn)) / max(1, (len(rh) - 1) + (len(rl) - 1))
    total = votes + swing_score
    bias = np.zeros(n, dtype=np.int8)
    bias[total >= 0.75] = 1
    bias[total <= -0.75] = -1
    return bias

def compute_atr_series(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
    return np.where(np.isnan(atr), 0.0, atr)

def compute_body_ratios(m5_open, m5_high, m5_low, m5_close):
    body = np.abs(m5_close - m5_open)
    candle_range = np.maximum(m5_high - m5_low, 1e-10)
    return body / candle_range

def adaptive_conf_allowed(br_val, atr_val, bw, aw):
    if len(bw) < 10 or len(aw) < 50: return True
    p_low = cfg.ADAPTIVE_CONF_P_LOW / 100.0
    p_norm = cfg.ADAPTIVE_CONF_P_NORM / 100.0
    if p_low < 0: p_low = 0.0
    if p_norm < 0: p_norm = 0.0
    l, h = np.percentile(aw, [25, 75])
    if atr_val <= l: return br_val >= np.percentile(bw, int(p_low * 100))
    elif atr_val < h: return br_val >= np.percentile(bw, int(p_norm * 100))
    return True

def run_bt(year, pred):
    client = DukascopyClient()
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    print(f"  [{year}] Features ({len(m5)} bars)...", end=" ", flush=True)
    ft = compute_features(m5, h1)
    print(f"OK ({len(ft)} rows)")

    model = pred.model
    X = ft[ft.columns.intersection(pred._feature_cols)]
    for c in pred._feature_cols:
        if c not in X.columns: X[c] = 0.0
    X = X[pred._feature_cols]
    X_aligned = X.reindex(m5.index, method="ffill")
    ml_mask = ~X_aligned.isna().any(axis=1)
    pb_u = np.full(len(m5), np.nan)
    pb_d = np.full(len(m5), np.nan)
    if ml_mask.any():
        valid_idx = np.where(ml_mask)[0]
        probs = model.predict_proba(X_aligned[ml_mask].values)
        pb_u[valid_idx] = np.array([p[1] for p in probs])
        pb_d[valid_idx] = np.array([p[0] for p in probs])

    print(f"  [{year}] ATR + body ratios...", end=" ", flush=True)
    m5_atr = compute_atr_series(m5["high"].values, m5["low"].values, m5["close"].values)
    m5_br = compute_body_ratios(m5["open"].values, m5["high"].values, m5["low"].values, m5["close"].values)
    print("OK")

    print(f"  [{year}] Bias...", end=" ", flush=True)
    h1_bias = compute_bias_vectorized(h1)
    h1_idx_map = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx_map = np.clip(h1_idx_map, 0, len(h1_bias) - 1)
    m5_bias = h1_bias[h1_idx_map]
    m5_bias_arr = np.full(len(m5_bias), None, dtype=object)
    m5_bias_arr[m5_bias == 1] = "BUY"
    m5_bias_arr[m5_bias == -1] = "SELL"

    h1_h = h1["high"].values; h1_l = h1["low"].values
    m5_h1h = np.where(h1_idx_map > 0, h1_h[h1_idx_map - 1], h1_h[h1_idx_map])
    m5_h1l = np.where(h1_idx_map > 0, h1_l[h1_idx_map - 1], h1_l[h1_idx_map])
    print("OK")

    m5_close = m5["close"].values; m5_open = m5["open"].values
    m5_high = m5["high"].values; m5_low = m5["low"].values

    bal = 20.0; total_events = 0; total_sub_trades = 0; wins = 0; dd = 0; peak_bal = 20.0
    t0 = time.time()
    override_count = 0; override_day = None; total_overrides = 0
    TRADE_MAX_BARS = cfg.PEAK_HARVEST_MAX_HOLD_BARS
    exit_reasons = {}
    first_event = None

    meta = MetaStrategy()
    scaler = EquityScaler()
    scaler.initialize(bal)

    bw = []; aw = []
    current_day = None

    from datetime import timedelta
    start_date = m5.index[60].date()
    two_weeks = start_date + timedelta(days=14)
    session_log = {}  # {date: {session: {"events":0, "wins":0, "pnl":0.0, "sub":0}}}

    for i in range(60, len(m5) - 15):
        expected = m5_bias_arr[i]
        if expected is None: continue

        ts_i = m5.index[i]
        if not in_allowed_session(ts_i): continue

        p = float(m5_close[i])
        h1h = float(m5_h1h[i]); h1l = float(m5_h1l[i])
        if h1h <= h1l: continue
        if expected == "BUY" and p <= h1h: continue
        if expected == "SELL" and p >= h1l: continue

        range_sz = h1h - h1l
        breakout_dist = p - h1h if expected == "BUY" else h1l - p
        score = min(breakout_dist / range_sz, 1.0) if range_sz > 0 else 0.0
        if score < cfg.MIN_BREAKOUT_SCORE: continue

        br_val = float(m5_br[i]); atr_val = float(m5_atr[i])
        bw.append(br_val); aw.append(atr_val)
        if len(bw) > cfg.ADAPTIVE_CONF_WINDOW: bw.pop(0)
        if len(aw) > cfg.ADAPTIVE_CONF_WINDOW: aw.pop(0)
        if cfg.ADAPTIVE_CONFIRMATION_ENABLED:
            if not adaptive_conf_allowed(br_val, atr_val, bw, aw): continue

        meta_threshold = meta.current_threshold
        atr_entry_threshold = float(m5_atr[i] * cfg.ATR_MULTIPLIER / range_sz) if range_sz > 0 and m5_atr[i] > 0 else None
        effective_threshold = max(atr_entry_threshold if atr_entry_threshold else 0, meta_threshold)
        if score < effective_threshold: continue

        day_key = ts_i.date()
        if override_day != day_key:
            override_count = 0; override_day = day_key

        entry_dir = expected; was_overridden = False
        if not np.isnan(pb_d[i]):
            if expected == "BUY" and pb_d[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_d[i] > pb_u[i] and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
                entry_dir = "SELL"; was_overridden = True
            elif expected == "SELL" and pb_u[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_u[i] > pb_d[i] and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
                entry_dir = "BUY"; was_overridden = True
            if was_overridden: override_count += 1; total_overrides += 1

        if not was_overridden:
            if not np.isnan(pb_u[i]) and not np.isnan(pb_d[i]):
                if expected == "BUY" and pb_u[i] < cfg.ML_CONFIDENCE_THRESHOLD: continue
                if expected == "SELL" and pb_d[i] < cfg.ML_CONFIDENCE_THRESHOLD: continue

        # Sizing (live bot logic)
        lot_mult = meta.current_lot_mult
        if cfg.AGGRESSIVE_SIZING_ENABLED:
            if score >= cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD: lot_mult *= cfg.AGGRESSIVE_VERY_STRONG_LOT_MULT
            elif score >= cfg.AGGRESSIVE_STRONG_THRESHOLD: lot_mult *= cfg.AGGRESSIVE_STRONG_LOT_MULT

        lot = min(scaler.get_lot(bal) * lot_mult, cfg.MAX_LOT)
        vol_step = cfg.LOT_STEP
        lot = round(lot / vol_step) * vol_step
        lot = max(cfg.MIN_LOT, min(lot, cfg.MAX_LOT))

        # Real margin check
        margin_per_lot = p * MARGIN_RATE
        max_trades = scaler.get_trades_per_event(bal, score)
        if meta:
            max_trades = meta.current_trades_per_event
        free_margin = bal

        single_margin = lot * margin_per_lot
        if free_margin < single_margin:
            max_lot = free_margin * 0.9 / margin_per_lot
            max_lot = round(max_lot / vol_step) * vol_step
            max_lot = max(cfg.MIN_LOT, max_lot)
            if max_lot < cfg.MIN_LOT: continue
            lot = max_lot
            single_margin = lot * margin_per_lot

        if free_margin < single_margin: continue

        actual_trades = 0
        remaining = free_margin
        for _ in range(max_trades):
            if remaining >= single_margin:
                actual_trades += 1
                remaining -= single_margin
            else:
                break
        if actual_trades == 0: continue

        ep = p
        total_events += 1
        total_sub_trades += actual_trades

        if current_day != day_key:
            current_day = day_key

        peak_profit = 0.0
        event_prof = 0.0
        trade_exit_price = None
        trade_exit_bars = None
        trade_exit_reason = None

        for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, len(m5) - 1)):
            fp = float(m5_close[j]); fh = float(m5_high[j]); fl = float(m5_low[j])
            bars_held = j - i
            diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
            prof = diff * CS * lot * actual_trades

            if entry_dir == "BUY": peak_profit = max(peak_profit, float(fh - ep))
            else: peak_profit = max(peak_profit, float(ep - fl))

            atr_j = float(m5_atr[j])

            ml_hold = False
            if not np.isnan(pb_u[j]):
                if entry_dir == "BUY" and pb_u[j] >= cfg.ML_HOLD_CONFIDENCE: ml_hold = True
                elif entry_dir == "SELL" and pb_d[j] >= cfg.ML_HOLD_CONFIDENCE: ml_hold = True

            exit_now = False; exit_reason = None

            if peak_profit > 0:
                trail_trigger = atr_j * cfg.PEAK_HARVEST_TRAIL_TRIGGER
                if peak_profit >= trail_trigger:
                    pullback = peak_profit - max(0, diff)
                    if pullback / peak_profit > cfg.PEAK_HARVEST_TRAIL_RETRACE:
                        exit_now = True; exit_reason = "trail_stop"

            if not exit_now and not ml_hold and bars_held >= 4:
                streak = 0; lookback = min(cfg.DIRECTION_LOSS_LOOKBACK, j - i)
                for k in range(lookback):
                    idx_k = j - k
                    if entry_dir == "BUY":
                        if m5_close[idx_k] < m5_open[idx_k]: streak += 1
                        else: break
                    else:
                        if m5_close[idx_k] > m5_open[idx_k]: streak += 1
                        else: break
                if streak >= cfg.DIRECTION_LOSS_STREAK:
                    exit_now = True; exit_reason = "direction_loss"

            if not exit_now and not ml_hold and bars_held >= cfg.PEAK_HARVEST_MIN_BARS_EXIT:
                start = max(0, j - 19)
                c_win = m5_close[start:j + 1]; o_win = m5_open[start:j + 1]
                if len(c_win) >= 5:
                    recent = c_win[-3:]; older = c_win[-6:-3] if len(c_win) >= 6 else c_win[:3]
                    recent_chg = abs(recent[-1] - recent[0])
                    older_chg = abs(older[-1] - older[0]) if len(older) >= 2 else recent_chg
                    avg_body = float(np.mean(np.abs(c_win[-5:] - o_win[-5:])))
                    if avg_body > 0:
                        window_atr = float(np.mean(m5_atr[max(0, j - 13):j + 1]))
                        ref = max(window_atr, fp * 0.0001)
                        raw = (recent_chg / (older_chg + 1e-10)) * (avg_body / (ref + 1e-10))
                        momentum = min(abs(raw), 1.0)
                        if (1.0 - momentum) > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
                            exit_now = True; exit_reason = "momentum_decay"

            if not exit_now and not ml_hold:
                if not np.isnan(pb_d[j]) and not np.isnan(pb_u[j]):
                    if entry_dir == "BUY" and pb_d[j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                        exit_now = True; exit_reason = "ml_reversal"
                    elif entry_dir == "SELL" and pb_u[j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                        exit_now = True; exit_reason = "ml_reversal"

            if not exit_now and bars_held >= TRADE_MAX_BARS:
                exit_now = True; exit_reason = "max_hold"

            if exit_now:
                bal += prof
                event_prof = prof
                if prof > 0: wins += 1
                exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
                meta.record_trade(prof, abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5)
                trade_exit_price = fp
                trade_exit_bars = bars_held
                trade_exit_reason = exit_reason
                break
        else:
            fb = m5.iloc[min(i + TRADE_MAX_BARS, len(m5) - 1)]
            fp = float(fb["close"])
            prof = (fp - ep) * CS * lot * actual_trades if entry_dir == "BUY" else (ep - fp) * CS * lot * actual_trades
            bal += prof
            event_prof = prof
            exit_reasons["max_hold"] = exit_reasons.get("max_hold", 0) + 1
            if prof > 0: wins += 1
            meta.record_trade(prof, abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5)
            trade_exit_price = fp
            trade_exit_bars = TRADE_MAX_BARS
            trade_exit_reason = "max_hold"

        # Session log for first 2 weeks
        if start_date <= ts_i.date() < two_weeks:
            sess = session_of_hour(ts_i.hour)
            d = ts_i.date()
            if d not in session_log: session_log[d] = {}
            if sess not in session_log[d]: session_log[d][sess] = {"e":0,"w":0,"p":0.0,"s":0}
            session_log[d][sess]["e"] += 1
            session_log[d][sess]["w"] += 1 if event_prof > 0 else 0
            session_log[d][sess]["p"] += event_prof
            session_log[d][sess]["s"] += actual_trades

        if first_event is None:
            first_event = {
                "dir": entry_dir, "entry": ep, "exit": trade_exit_price,
                "sub_trades": actual_trades, "lot_per_trade": lot,
                "total_lots": lot * actual_trades,
                "bars": trade_exit_bars, "profit": event_prof,
                "reason": trade_exit_reason,
                "ts": ts_i, "bal_before": bal - event_prof, "override": was_overridden,
                "session": ts_i.hour,
            }

        peak_bal = max(peak_bal, bal)
        dd = max(dd, peak_bal - bal)
        bal = max(bal, 0.01)
        meta.update(bal, {"bias": "BULLISH" if h1_bias[h1_idx_map[i]] == 1 else "BEARISH",
                          "strength": abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5})

    elapsed = time.time() - t0
    wr = (wins / total_events * 100) if total_events else 0
    pf = wins / max(1, total_events - wins) if (total_events - wins) > 0 else float("inf")
    print()
    print(f"{'='*60}")
    print(f"  Live Sub-Trade Backtest — {year}")
    print(f"  (CS=1, margin_rate=0.05, sequential sub-trades)")
    print(f"{'='*60}")
    print(f"  Events:          {total_events}")
    print(f"  Sub-trades:      {total_sub_trades} (avg {total_sub_trades/max(1,total_events):.1f}/event)")
    print(f"  Wins:            {wins}")
    print(f"  Losses:          {total_events - wins}")
    print(f"  Win rate:        {wr:.1f}%")
    print(f"  Profit factor:   {pf:.2f}")
    print(f"  Net PnL:         ${bal - 20:>+8.2f}")
    print(f"  Final balance:   ${bal:.2f}")
    print(f"  Max drawdown:    ${dd:.2f}")
    print(f"  ML overrides:    {total_overrides}")
    if first_event:
        print(f"\n  First event:")
        print(f"    Direction:     {first_event['dir']}")
        print(f"    Entry:         ${first_event['entry']:.2f} at {first_event['ts']} (UTC+{first_event['session']})")
        print(f"    Exit:          ${first_event['exit']:.2f}")
        print(f"    Sub-trades:    {first_event['sub_trades']}")
        print(f"    Lot/trade:     {first_event['lot_per_trade']:.4f}")
        print(f"    Total lots:    {first_event['total_lots']:.4f}")
        print(f"    Hold:          {first_event['bars']} bars ({first_event['bars']*5} min)")
        print(f"    Profit:        ${first_event['profit']:.2f}")
        print(f"    Exit reason:   {first_event['reason']}")
        print(f"    ML Override:   {first_event['override']}")
        bal_after = first_event['bal_before'] + first_event['profit']
        print(f"    Bal before:    ${first_event['bal_before']:.2f}")
        print(f"    Bal after:     ${bal_after:.2f} (+{first_event['profit']/first_event['bal_before']*100:.1f}%)")
    print(f"  Runtime:         {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Exit reasons:    {exit_reasons}")
    print(f"  Meta regime:     {meta.regime}")
    print(f"  Meta threshold:  {meta.current_threshold:.3f}")
    print(f"  Meta lot mult:   {meta.current_lot_mult}")
    if session_log:
        print(f"\n  First 2 weeks — sessions:")
        print(f"  {'Date':<14} {'Session':<12} {'Events':<8} {'Wins':<6} {'WR':<6} {'PnL':<10} {'Sub-':<6}")
        print(f"  {'':<14} {'':<12} {'':<8} {'':<6} {'':<6} {'':<10} {'trades'}")
        print(f"  {'-'*56}")
        for d in sorted(session_log):
            for sess in ["ASIA", "LONDON", "NEW_YORK"]:
                if sess in session_log[d]:
                    s = session_log[d][sess]
                    wr_s = s["w"]/s["e"]*100 if s["e"] else 0
                    print(f"  {str(d):<14} {sess:<12} {s['e']:<8} {s['w']:<6} {wr_s:<5.1f}% ${s['p']:<+7.2f} {s['s']:<6}")
    print()
    return total_events, total_sub_trades, wr, pf, bal - 20, dd, total_overrides, elapsed

if __name__ == "__main__":
    pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
    print("Model loaded.\n")
    run_bt(2022, pred)
