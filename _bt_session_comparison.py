"""
Compare session limit strategies for ML overrides.
Downloads data once per year, runs all 4 scenarios on shared data.
"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features
from app.meta_strategy import MetaStrategy
from app.risk_manager import EquityScaler

CS = 100; LEV = 200

SESSION_HOURS = {"ASIA": (0, 8), "LONDON": (7, 17), "NEW_YORK": (12, 22)}

def in_allowed_session(ts) -> bool:
    h = ts.hour
    sessions = [s.strip().upper() for s in cfg.ALLOWED_SESSIONS.split(",") if s.strip()]
    for s in sessions:
        if s in SESSION_HOURS:
            lo, hi = SESSION_HOURS[s]
            if lo <= h < hi:
                return True
    return False

def compute_bias_vectorized(h1: pd.DataFrame) -> np.ndarray:
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
    cross = (fast > slow) & (fast_slope > 0)
    votes[cross] += 1.0
    cross = (fast < slow) & (fast_slope < 0)
    votes[cross] -= 1.0
    price_above = c > slow
    votes[price_above & (slow_slope >= 0)] += 0.5
    votes[~price_above & (slow_slope <= 0)] -= 0.5
    lookback = 5
    n = len(c)
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
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
    return np.where(np.isnan(atr), 0.0, atr)

def compute_body_ratios(m5_open, m5_high, m5_low, m5_close):
    body = np.abs(m5_close - m5_open)
    candle_range = np.maximum(m5_high - m5_low, 1e-10)
    return body / candle_range

def adaptive_conf_allowed(br_val, atr_val, bw, aw):
    if len(bw) < 10 or len(aw) < 50:
        return True
    p_low = max(0, cfg.ADAPTIVE_CONF_P_LOW / 100.0)
    p_norm = max(0, cfg.ADAPTIVE_CONF_P_NORM / 100.0)
    l, h = np.percentile(aw, [25, 75])
    if atr_val <= l:
        return br_val >= np.percentile(bw, int(p_low * 100))
    elif atr_val < h:
        return br_val >= np.percentile(bw, int(p_norm * 100))
    return True

def load_year_data(year, pred):
    """Load data and compute all shared arrays. Returns dict of arrays."""
    client = DukascopyClient()
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    ft = compute_features(m5, h1)

    model = pred.model
    X = ft[ft.columns.intersection(pred._feature_cols)]
    for c in pred._feature_cols:
        if c not in X.columns:
            X[c] = 0.0
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

    m5_atr = compute_atr_series(m5["high"].values, m5["low"].values, m5["close"].values)
    m5_br = compute_body_ratios(m5["open"].values, m5["high"].values, m5["low"].values, m5["close"].values)

    h1_bias = compute_bias_vectorized(h1)
    h1_idx_map = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx_map = np.clip(h1_idx_map, 0, len(h1_bias) - 1)
    m5_bias = h1_bias[h1_idx_map]
    m5_bias_arr = np.full(len(m5_bias), None, dtype=object)
    m5_bias_arr[m5_bias == 1] = "BUY"
    m5_bias_arr[m5_bias == -1] = "SELL"

    h1_h = h1["high"].values; h1_l = h1["low"].values
    m5_h1h = h1_h[h1_idx_map]
    m5_h1l = h1_l[h1_idx_map]
    m5_h1h = np.where(h1_idx_map > 0, h1_h[h1_idx_map - 1], m5_h1h)
    m5_h1l = np.where(h1_idx_map > 0, h1_l[h1_idx_map - 1], m5_h1l)

    return {
        "N": len(m5),
        "ts": m5.index,
        "close": m5["close"].values, "open": m5["open"].values,
        "high": m5["high"].values, "low": m5["low"].values,
        "bias_arr": m5_bias_arr, "h1_bias": h1_bias, "h1_idx_map": h1_idx_map,
        "h1h": m5_h1h, "h1l": m5_h1l,
        "pb_u": pb_u, "pb_d": pb_d,
        "m5_atr": m5_atr, "m5_br": m5_br,
    }

def run_bt_on_data(data, year, label, session_limit=None,
                   override_bypass_session=False,
                   override_counts_toward_session=True):
    N = data["N"]
    bal = 20.0; total_events = 0; total_sub_trades = 0; wins = 0; dd = 0; peak_bal = 20.0
    t0 = time.time()
    override_count = 0; override_day = None; total_overrides = 0
    TRADE_MAX_BARS = cfg.PEAK_HARVEST_MAX_HOLD_BARS
    exit_reasons = {}

    meta = MetaStrategy()
    scaler = EquityScaler()
    scaler.initialize(bal)
    bw = []; aw = []
    session_trades = 0; session_day = None

    for i in range(60, N - 15):
        expected = data["bias_arr"][i]
        if expected is None:
            continue

        ts_i = data["ts"][i]
        if not in_allowed_session(ts_i):
            continue

        p = float(data["close"][i])
        h1h = float(data["h1h"][i]); h1l = float(data["h1l"][i])
        if h1h <= h1l: continue
        if expected == "BUY" and p <= h1h: continue
        if expected == "SELL" and p >= h1l: continue

        range_sz = h1h - h1l
        if expected == "BUY":
            breakout_dist = p - h1h
        else:
            breakout_dist = h1l - p
        score = min(breakout_dist / range_sz, 1.0) if range_sz > 0 else 0.0
        if score < cfg.MIN_BREAKOUT_SCORE:
            continue

        br_val = float(data["m5_br"][i])
        atr_val = float(data["m5_atr"][i])
        bw.append(br_val); aw.append(atr_val)
        if len(bw) > cfg.ADAPTIVE_CONF_WINDOW: bw.pop(0)
        if len(aw) > cfg.ADAPTIVE_CONF_WINDOW: aw.pop(0)
        if cfg.ADAPTIVE_CONFIRMATION_ENABLED:
            if not adaptive_conf_allowed(br_val, atr_val, bw, aw):
                continue

        meta_threshold = meta.current_threshold
        atr_entry_threshold = float(data["m5_atr"][i] * cfg.ATR_MULTIPLIER / range_sz) if range_sz > 0 and data["m5_atr"][i] > 0 else None
        effective_threshold = max(atr_entry_threshold if atr_entry_threshold else 0, meta_threshold)
        if score < effective_threshold:
            continue

        day_key = ts_i.date()
        if session_day != day_key:
            session_trades = 0
            session_day = day_key

        if override_day != day_key:
            override_count = 0
            override_day = day_key

        entry_dir = expected
        was_overridden = False
        pb_d_i = data["pb_d"][i]; pb_u_i = data["pb_u"][i]
        if not np.isnan(pb_d_i):
            if expected == "BUY" and pb_d_i >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_d_i > pb_u_i and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
                entry_dir = "SELL"
                was_overridden = True
            elif expected == "SELL" and pb_u_i >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_u_i > pb_d_i and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
                entry_dir = "BUY"
                was_overridden = True
            if was_overridden:
                override_count += 1
                total_overrides += 1

        if not was_overridden:
            if not np.isnan(pb_u_i) and not np.isnan(pb_d_i):
                if expected == "BUY" and pb_u_i < cfg.ML_CONFIDENCE_THRESHOLD:
                    continue
                if expected == "SELL" and pb_d_i < cfg.ML_CONFIDENCE_THRESHOLD:
                    continue

        if session_limit is not None:
            if session_trades >= session_limit:
                if not (was_overridden and override_bypass_session):
                    continue

        if was_overridden and not override_counts_toward_session:
            pass
        else:
            session_trades += 1

        lot_mult = meta.current_lot_mult
        if cfg.AGGRESSIVE_SIZING_ENABLED:
            if score >= cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD:
                lot_mult *= cfg.AGGRESSIVE_VERY_STRONG_LOT_MULT
            elif score >= cfg.AGGRESSIVE_STRONG_THRESHOLD:
                lot_mult *= cfg.AGGRESSIVE_STRONG_LOT_MULT

        ep = p; total_events += 1
        n = max(0.01, bal * LEV / ep / CS)
        n = min(n, 1.0)
        scale_lot = scaler.get_lot(bal) * lot_mult
        n = min(n, max(cfg.MIN_LOT, min(scale_lot, cfg.MAX_LOT)))
        mg = ep * CS / LEV
        mx = bal / mg
        if mx < cfg.MIN_LOT:
            total_events -= 1; continue
        n = min(n, mx)

        trades_per_event = meta.current_trades_per_event
        margin_per_trade = n * ep * CS / LEV
        max_affordable = int(bal / margin_per_trade) if margin_per_trade > 0 else 0
        actual_trades = min(trades_per_event, max_affordable, cfg.MAX_TRADES_PER_EVENT)
        actual_trades = max(1, actual_trades)
        total_sub_trades += actual_trades

        peak_profit = 0.0
        for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, N - 1)):
            fp = float(data["close"][j])
            fh = float(data["high"][j])
            fl = float(data["low"][j])
            bars_held = j - i
            diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
            prof = diff * CS * n * actual_trades

            if entry_dir == "BUY":
                peak_profit = max(peak_profit, float(fh - ep))
            else:
                peak_profit = max(peak_profit, float(ep - fl))

            atr_j = float(data["m5_atr"][j])

            ml_hold = False
            if not np.isnan(data["pb_u"][j]):
                if entry_dir == "BUY" and data["pb_u"][j] >= cfg.ML_HOLD_CONFIDENCE:
                    ml_hold = True
                elif entry_dir == "SELL" and data["pb_d"][j] >= cfg.ML_HOLD_CONFIDENCE:
                    ml_hold = True

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
                        if data["close"][idx_k] < data["open"][idx_k]: streak += 1
                        else: break
                    else:
                        if data["close"][idx_k] > data["open"][idx_k]: streak += 1
                        else: break
                if streak >= cfg.DIRECTION_LOSS_STREAK:
                    exit_now = True; exit_reason = "direction_loss"

            if not exit_now and not ml_hold and bars_held >= cfg.PEAK_HARVEST_MIN_BARS_EXIT:
                start = max(0, j - 19)
                c_win = data["close"][start:j + 1]; o_win = data["open"][start:j + 1]
                if len(c_win) >= 5:
                    recent = c_win[-3:]; older = c_win[-6:-3] if len(c_win) >= 6 else c_win[:3]
                    recent_chg = abs(recent[-1] - recent[0]); older_chg = abs(older[-1] - older[0]) if len(older) >= 2 else recent_chg
                    avg_body = float(np.mean(np.abs(c_win[-5:] - o_win[-5:])))
                    if avg_body > 0:
                        window_atr = float(np.mean(data["m5_atr"][max(0, j - 13):j + 1]))
                        ref = max(window_atr, fp * 0.0001)
                        raw = (recent_chg / (older_chg + 1e-10)) * (avg_body / (ref + 1e-10))
                        momentum = min(abs(raw), 1.0)
                        exit_score = 1.0 - momentum
                        if exit_score > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
                            exit_now = True; exit_reason = "momentum_decay"

            if not exit_now and not ml_hold:
                if not np.isnan(data["pb_d"][j]) and not np.isnan(data["pb_u"][j]):
                    if entry_dir == "BUY" and data["pb_d"][j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                        exit_now = True; exit_reason = "ml_reversal"
                    elif entry_dir == "SELL" and data["pb_u"][j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                        exit_now = True; exit_reason = "ml_reversal"

            if not exit_now and bars_held >= TRADE_MAX_BARS:
                exit_now = True; exit_reason = "max_hold"

            if exit_now:
                bal += prof
                if prof > 0: wins += 1
                exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
                meta.record_trade(prof, abs(data["h1_bias"][data["h1_idx_map"][i]]) if abs(data["h1_bias"][data["h1_idx_map"][i]]) > 0 else 0.5)
                break
        else:
            fb_idx = min(i + TRADE_MAX_BARS, N - 1)
            fp = float(data["close"][fb_idx])
            prof = (fp - ep) * CS * n * actual_trades if entry_dir == "BUY" else (ep - fp) * CS * n * actual_trades
            bal += prof
            exit_reasons["max_hold"] = exit_reasons.get("max_hold", 0) + 1
            if prof > 0: wins += 1
            meta.record_trade(prof, abs(data["h1_bias"][data["h1_idx_map"][i]]) if abs(data["h1_bias"][data["h1_idx_map"][i]]) > 0 else 0.5)

        peak_bal = max(peak_bal, bal)
        dd = max(dd, peak_bal - bal)
        bal = max(bal, 0.01)
        meta.update(bal, {"bias": "BULLISH" if data["h1_bias"][data["h1_idx_map"][i]] == 1 else "BEARISH",
                          "strength": abs(data["h1_bias"][data["h1_idx_map"][i]]) if abs(data["h1_bias"][data["h1_idx_map"][i]]) > 0 else 0.5})

    elapsed = time.time() - t0
    wr = (wins / total_events * 100) if total_events else 0
    pf = wins / max(1, total_events - wins) if (total_events - wins) > 0 else float("inf")
    ovr_pct = total_overrides / max(1, total_events) * 100
    return total_events, total_sub_trades, wr, pf, bal - 20, dd, total_overrides, elapsed, ovr_pct

if __name__ == "__main__":
    pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
    print("Model loaded.\n")

    YEARS = [2022, 2023, 2024]

    scenarios = [
        ("A: No limit", dict(session_limit=None, override_bypass_session=False, override_counts_toward_session=True)),
        ("B: Limit=10, overrides bypass entry + count", dict(session_limit=10, override_bypass_session=True, override_counts_toward_session=True)),
        ("C: Limit=10, overrides bypass entry + don't count", dict(session_limit=10, override_bypass_session=True, override_counts_toward_session=False)),
        ("D: Limit=999 (effectively unlimited)", dict(session_limit=999, override_bypass_session=False, override_counts_toward_session=True)),
    ]

    # Pre-load all data
    year_data = {}
    for y in YEARS:
        print(f"Loading {y} data...", end=" ", flush=True)
        t0 = time.time()
        year_data[y] = load_year_data(y, pred)
        print(f"({time.time()-t0:.1f}s)")

    for label, kwargs in scenarios:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  params: {kwargs}")
        print(f"{'='*70}")

        all_results = []
        total_t_start = time.time()
        for y in YEARS:
            total, sub, wr, pf, net, dd, ovr, elapsed, ovr_pct = run_bt_on_data(year_data[y], y, label, **kwargs)
            all_results.append((y, total, sub, wr, pf, net, dd, ovr, ovr_pct))
            print(f"  {y}: {net:>+8.2f}  WR={wr:.1f}%  PF={pf:.2f}  "
                  f"DD=${dd:.0f}  tr={total}  ovr={ovr} ({ovr_pct:.0f}%)")

        total_tr = sum(r[1] for r in all_results)
        total_wins = sum(int(r[1] * r[3] / 100) for r in all_results)
        total_net = sum(r[5] for r in all_results)
        total_ovr = sum(r[7] for r in all_results)
        avg_wr = total_wins / total_tr * 100 if total_tr else 0
        avg_pf = total_wins / max(1, total_tr - total_wins) if total_tr > 0 else 0
        max_dd = max(r[6] for r in all_results)
        print(f"  {'TOTAL':>6}: {total_tr:>5}tr  WR={avg_wr:.1f}%  PF={avg_pf:.2f}  "
              f"PnL=${total_net:>+8.2f}  DD=${max_dd:.0f}  ovr={total_ovr}")
        print(f"  Time: {time.time()-total_t_start:.1f}s")
