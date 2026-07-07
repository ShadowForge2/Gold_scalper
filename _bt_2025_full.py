"""
Backtest 2025: exit model variants.
  - baseline: current live config (no exit model)
  - exit_narrow: exit model only prevents fallback exits (trail/dir_loss/mom_decay), NOT ml_reversal
  - exit_aggressive: exit model overrides everything (current prod)
"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, ExitPredictor, compute_features, FEATURE_COLS
from app.meta_strategy import MetaStrategy
from app.risk_manager import EquityScaler

CS = 100; LEV = 200; TRADE_MAX_BARS = 25
SESSION_HOURS = {"ASIA": (0, 8), "LONDON": (7, 17), "NEW_YORK": (12, 22)}

def in_allowed_session(ts):
    h = ts.hour
    for s in cfg.ALLOWED_SESSIONS.split(","):
        s = s.strip().upper()
        if s in SESSION_HOURS:
            lo, hi = SESSION_HOURS[s]
            if lo <= h < hi: return True
    return False

def compute_atr_series(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return np.where(np.isnan(pd.Series(tr).rolling(period, min_periods=period).mean().values), 0.0, tr)

def run_bt(year, pred, exit_predictor=None, mode="baseline"):
    client = DukascopyClient()
    m1 = client.download_year(year).sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    n = len(m5)
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
    X = X[feat_cols]
    X_aligned = X.reindex(m5.index, method="ffill")
    ml_mask = ~X_aligned.isna().any(axis=1)
    pb_u = np.full(n, np.nan); pb_d = np.full(n, np.nan)
    if ml_mask.any():
        valid_idx = np.where(ml_mask)[0]
        probs = pred.model.predict_proba(X_aligned[ml_mask].values)
        pb_u[valid_idx] = np.array([p[1] for p in probs])
        pb_d[valid_idx] = np.array([p[0] for p in probs])

    feat_all = None
    if exit_predictor is not None:
        h1_a = h1.reindex(m5.index, method="ffill")
        feat_all = compute_features(m5, h1_a)

    h1_bias = _compute_bias(h1)
    h1_idx_map = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx_map = np.clip(h1_idx_map, 0, len(h1_bias) - 1)
    m5_bias = h1_bias[h1_idx_map]
    h1_h = h1["high"].values; h1_l = h1["low"].values
    m5_h1h = np.where(h1_idx_map > 0, h1_h[h1_idx_map - 1], h1_h[h1_idx_map])
    m5_h1l = np.where(h1_idx_map > 0, h1_l[h1_idx_map - 1], h1_l[h1_idx_map])

    bal = 20.0; total_events = 0; wins = 0; dd = 0; peak_bal = 20.0
    exit_reasons = {}; meta = MetaStrategy()
    scaler = EquityScaler()
    scaler.initialize(bal)

    for i in range(100, n - 15):
        ts_i = m5.index[i]
        if not in_allowed_session(ts_i): continue
        bias_val = m5_bias[i]
        if bias_val == 0: continue
        entry_dir = "BUY" if bias_val == 1 else "SELL"
        p = float(m5_close[i]); h1h = float(m5_h1h[i]); h1l = float(m5_h1l[i])
        if h1h <= h1l: continue
        range_sz = h1h - h1l

        if entry_dir == "BUY":
            if p <= h1h: continue
        else:
            if p >= h1l: continue

        if not np.isnan(pb_u[i]) and not np.isnan(pb_d[i]):
            if entry_dir == "BUY" and pb_u[i] < cfg.ML_CONFIDENCE_THRESHOLD: continue
            if entry_dir == "SELL" and pb_d[i] < cfg.ML_CONFIDENCE_THRESHOLD: continue

        ep = p; total_events += 1
        n_lots = scaler.get_lot(bal); actual_trades = 1
        best_pnl = 0.0; w_streak = 0
        exited = False

        for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, n - 1)):
            fp = float(m5_close[j]); fh = float(m5_high[j]); fl = float(m5_low[j])
            diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
            bars = j - i
            hi_diff = (fh - ep) if entry_dir == "BUY" else (ep - fl)
            best_pnl = max(best_pnl, hi_diff)
            w_streak = (w_streak + 1) if (entry_dir == "BUY" and fp < m5_open[j]) or \
                        (entry_dir == "SELL" and fp > m5_open[j]) else 0
            atr_j = max(float(m5_atr[j]), 0.01)

            hp = 0.5
            if exit_predictor is not None and feat_all is not None and j < len(feat_all):
                mf = feat_all.iloc[j].to_dict()
                pnl_a = diff / atr_j; peak_a = best_pnl / atr_j
                dd_pct = (best_pnl - max(0, diff)) / max(best_pnl, 0.001)
                ts = {"bars_held": bars, "pnl_atr": round(pnl_a, 4), "peak_atr": round(peak_a, 4),
                      "drawdown_pct": round(dd_pct, 4), "entry_score": 0.5, "atr_change": 1.0, "wrong_streak": w_streak}
                hp = exit_predictor.predict_hold_prob(mf, ts)

            # --- Mechanical exits (always check first) ---
            mech_now = False; mech_reason = None
            if best_pnl > 0:
                trail_trigger = atr_j * cfg.PEAK_HARVEST_TRAIL_TRIGGER
                if best_pnl >= trail_trigger:
                    pullback = best_pnl - max(0, diff)
                    if pullback / best_pnl > cfg.PEAK_HARVEST_TRAIL_RETRACE:
                        mech_now = True; mech_reason = "trail_stop"

            if not mech_now and bars >= 4:
                streak = 0
                for k in range(min(cfg.DIRECTION_LOSS_LOOKBACK, bars)):
                    idx_k = j - k
                    if entry_dir == "BUY" and m5_close[idx_k] < m5_open[idx_k]: streak += 1
                    elif entry_dir == "SELL" and m5_close[idx_k] > m5_open[idx_k]: streak += 1
                    else: break
                if streak >= cfg.DIRECTION_LOSS_STREAK:
                    mech_now = True; mech_reason = "direction_loss"

            if not mech_now and bars >= cfg.PEAK_HARVEST_MIN_BARS_EXIT:
                start = max(0, j - 19)
                c_win = m5_close[start:j + 1]; o_win = m5_open[start:j + 1]
                if len(c_win) >= 5:
                    recent = c_win[-3:]; older = c_win[-6:-3] if len(c_win) >= 6 else c_win[:3]
                    recent_chg = abs(recent[-1] - recent[0]); older_chg = abs(older[-1] - older[0]) if len(older) >= 2 else recent_chg
                    avg_body = float(np.mean(np.abs(c_win[-5:] - o_win[-5:])))
                    if avg_body > 0:
                        window_atr = float(np.mean(m5_atr[max(0, j - 13):j + 1]))
                        ref = max(window_atr, fp * 0.0001)
                        raw = (recent_chg / (older_chg + 1e-10)) * (avg_body / (ref + 1e-10))
                        momentum = min(abs(raw), 1.0)
                        if (1.0 - momentum) > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
                            mech_now = True; mech_reason = "momentum_decay"

            # --- ML reversal check ---
            ml_rev = False
            if not np.isnan(pb_d[j]) and not np.isnan(pb_u[j]):
                if entry_dir == "BUY" and pb_d[j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                    ml_rev = True
                elif entry_dir == "SELL" and pb_u[j] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD:
                    ml_rev = True

            # --- Apply exit decision based on mode ---
            exit_now = False; exit_reason = None

            if mode == "baseline":
                # No exit model: mechanical exits run freely, ml_reversal overrides
                if mech_now:
                    exit_now = True; exit_reason = mech_reason
                elif ml_rev:
                    exit_now = True; exit_reason = "ml_reversal"
                elif bars >= TRADE_MAX_BARS:
                    exit_now = True; exit_reason = "max_hold"

            elif mode == "narrow":
                # Exit model only suppresses mechanical exits, NOT ml_reversal
                if hp >= 0.70:
                    # strong hold: suppress ALL mechanical fallsbacks
                    pass
                elif hp >= 0.60:
                    # moderate hold: suppress mechanical exits
                    pass
                else:
                    # uncertain or weak: allow mechanical exits
                    if mech_now:
                        exit_now = True; exit_reason = mech_reason
                # ML reversal always works
                if not exit_now and ml_rev:
                    exit_now = True; exit_reason = "ml_reversal"
                if not exit_now and bars >= TRADE_MAX_BARS:
                    exit_now = True; exit_reason = "max_hold"

            elif mode == "aggressive":
                # Exit model is primary: can override ml_reversal too
                if hp >= 0.70:
                    # strong hold: suppress everything, even ML reversal
                    pass
                elif hp >= 0.60:
                    # moderate hold: suppress mechanical, but allow ML reversal
                    if ml_rev:
                        exit_now = True; exit_reason = "ml_reversal"
                elif hp <= 0.30:
                    # strong exit: exit now via exit model
                    exit_now = True; exit_reason = "exit_model"
                else:
                    # uncertain: fallback to mechanical, then ML reversal
                    if mech_now:
                        exit_now = True; exit_reason = mech_reason
                    elif ml_rev:
                        exit_now = True; exit_reason = "ml_reversal"
                if not exit_now and bars >= TRADE_MAX_BARS:
                    exit_now = True; exit_reason = "max_hold"

            if exit_now:
                prof = diff * CS * n_lots * actual_trades
                bal += prof
                if prof > 0: wins += 1
                exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
                meta.record_trade(prof, abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5)
                break
        else:
            fb = m5.iloc[min(i + TRADE_MAX_BARS, len(m5) - 1)]
            fp = float(fb["close"])
            diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
            prof = diff * CS * n_lots * actual_trades
            bal += prof
            if prof > 0: wins += 1
            exit_reasons["max_hold"] = exit_reasons.get("max_hold", 0) + 1
            meta.record_trade(prof, abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5)

        peak_bal = max(peak_bal, bal)
        dd = max(dd, peak_bal - bal)
        bal = max(bal, 0.01)
        meta.update(bal, {"bias": "BULLISH" if h1_bias[h1_idx_map[i]] == 1 else "BEARISH",
                          "strength": abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5})

    return {"events": total_events, "wins": wins, "net_pnl": round(bal - 20, 2), "dd": round(dd, 2), "exit_reasons": exit_reasons}

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
    total = votes + swing_score
    bias = np.zeros(n, dtype=np.int8)
    bias[total >= 0.75] = 1; bias[total <= -0.75] = -1
    return bias

if __name__ == "__main__":
    pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
    ep = ExitPredictor(model_path="models/exit_xgb_m5.joblib")

    for year in [2025]:
        print(f"\n{'='*60}\n=== {year} ===\n{'='*60}")
        t0 = time.time()

        b = run_bt(year, pred, exit_predictor=None, mode="baseline")
        n = run_bt(year, pred, exit_predictor=ep, mode="narrow")

        print(f"{'':>20}  {'Events':>8} {'WR':>8} {'PnL':>14} {'DD':>10} {'AvgPnL':>10}")
        print(f"{'BASELINE (no exit model):':>20}  {b['events']:>8} {b['wins']/max(1,b['events']):>7.1%} "
              f"${b['net_pnl']:>10.2f} ${b['dd']:>7.2f} ${b['net_pnl']/max(1,b['events']):>8.2f}", flush=True)
        print(f"{'EXIT MODEL (narrow):':>20}  {n['events']:>8} {n['wins']/max(1,n['events']):>7.1%} "
              f"${n['net_pnl']:>10.2f} ${n['dd']:>7.2f} ${n['net_pnl']/max(1,n['events']):>8.2f}", flush=True)
        delta_pnl = n['net_pnl'] - b['net_pnl']
        delta_dd = n['dd'] - b['dd']
        print(f"{'DELTA:':>20}  {'':>8} {'':>8} ${delta_pnl:>+10.2f} ${delta_dd:>+8.2f}")
        print(f"\n  BASELINE exits: {dict(sorted(b['exit_reasons'].items(), key=lambda x: -x[1]))}")
        print(f"  EXIT MODEL exits: {dict(sorted(n['exit_reasons'].items(), key=lambda x: -x[1]))}")

        # Key metric: trail_stop + direction_loss + momentum_decay reduction
        base_mech = sum(v for k, v in b['exit_reasons'].items() if k in ('trail_stop','direction_loss','momentum_decay'))
        exit_mech = sum(v for k, v in n['exit_reasons'].items() if k in ('trail_stop','direction_loss','momentum_decay'))
        print(f"\n  Mechanical exits (trail+dir_loss+mom_decay): {base_mech} -> {exit_mech} ({((base_mech-exit_mech)/max(1,base_mech))*100:.0f}% reduction)")
        print(f"  ML reversal exits preserved: {b['exit_reasons'].get('ml_reversal',0)} → {n['exit_reasons'].get('ml_reversal',0)}")
        print(f"  Time: {time.time()-t0:.1f}s")
