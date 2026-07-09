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

CS = 100; LEV = 200; TRADE_MAX_BARS = 25; HP_COLLAPSE_THRESHOLD = 0.40
SESSION_HOURS = {"ASIA": (0, 8), "LONDON": (7, 17), "NEW_YORK": (12, 22)}

def in_allowed_session(ts, allow_all=False):
    if allow_all: return True
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

def run_bt(year, pred, exit_predictor=None, mode="baseline", allow_all=False, trace_first=False):
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
    event_log = []

    for i in range(100, n - 15):
        ts_i = m5.index[i]
        if not in_allowed_session(ts_i, allow_all=allow_all): continue
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
        ml_conf = pb_u[i] if entry_dir == "BUY" else pb_d[i]
        ml_conf = ml_conf if not np.isnan(ml_conf) else 0.5
        signal_score = ml_conf

        # --- Position sizing (matches live bot.py:861-899) ---
        lot_mult = float(meta.current_lot_mult)
        if cfg.AGGRESSIVE_SIZING_ENABLED:
            if signal_score >= cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD:
                lot_mult *= cfg.AGGRESSIVE_VERY_STRONG_LOT_MULT
            elif signal_score >= cfg.AGGRESSIVE_STRONG_THRESHOLD:
                lot_mult *= cfg.AGGRESSIVE_STRONG_LOT_MULT
        if ml_conf >= cfg.ML_CONF_VERY_STRONG_THRESHOLD:
            lot_mult *= cfg.ML_CONF_VERY_STRONG_LOT_MULT
        elif ml_conf >= cfg.ML_CONF_STRONG_THRESHOLD:
            lot_mult *= cfg.ML_CONF_STRONG_LOT_MULT
        n_lots = min(scaler.get_lot(bal) * lot_mult, cfg.MAX_LOT)
        n_lots = round(n_lots / cfg.LOT_STEP) * cfg.LOT_STEP
        n_lots = max(cfg.MIN_LOT, n_lots)

        actual_trades = meta.current_trades_per_event
        best_pnl = 0.0; w_streak = 0
        exited = False; hp_was_high = False
        wrong_candles = []  # rolling wrong-direction marker per bar

        for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, n - 1)):
            fp = float(m5_close[j]); fh = float(m5_high[j]); fl = float(m5_low[j])
            diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
            bars = j - i
            hi_diff = (fh - ep) if entry_dir == "BUY" else (ep - fl)
            best_pnl = max(best_pnl, hi_diff)
            is_wrong = (entry_dir == "BUY" and fp < m5_open[j]) or (entry_dir == "SELL" and fp > m5_open[j])
            w_streak = (w_streak + 1) if is_wrong else 0
            wrong_candles.append(1 if is_wrong else 0)
            atr_j = max(float(m5_atr[j]), 0.01)

            hp = 0.5
            if exit_predictor is not None and feat_all is not None and j < len(feat_all):
                mf = feat_all.iloc[j].to_dict()
                pnl_a = diff / atr_j; peak_a = best_pnl / atr_j
                dd_pct = (best_pnl - max(0, diff)) / max(best_pnl, 0.001)
                ts = {"bars_held": bars, "pnl_atr": round(pnl_a, 4), "peak_atr": round(peak_a, 4),
                      "drawdown_pct": round(dd_pct, 4), "entry_score": 0.5, "atr_change": 1.0, "wrong_streak": w_streak}
                hp = exit_predictor.predict_hold_prob(mf, ts)
                hp_was_high = hp_was_high or (hp >= 0.60)

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
                # Cumulative direction loss: 5+ wrong out of last 7
                if not mech_now and bars >= 7:
                    recent = wrong_candles[-7:]
                    if sum(recent) >= 5:
                        mech_now = True; mech_reason = "direction_loss_cum"

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
                    # strong hold: suppress all mechanical fallbacks
                    pass
                elif hp >= 0.60:
                    # moderate hold: suppress mechanical exits
                    pass
                else:
                    # HP collapse guard: if model was confident then lost conviction while in loss
                    if hp_was_high and hp < HP_COLLAPSE_THRESHOLD and diff <= 0 and bars >= 4:
                        exit_now = True; exit_reason = "hp_collapse"
                    elif mech_now:
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
                event_log.append({"event": total_events, "entry_time": ts_i, "exit_time": m5.index[j],
                                  "dir": entry_dir, "entry_px": ep, "exit_px": fp,
                                  "reason": exit_reason, "profit": prof,
                                  "bars": bars, "hp": hp, "lots": n_lots, "trades": actual_trades})
                if trace_first and total_events == 1:
                    entry_time = ts_i
                    exit_time = m5.index[j]
                    mins = (exit_time - entry_time).total_seconds() / 60
                    bar_duration = bars * 5
                    print(f"\n  FIRST EVENT:")
                    print(f"    Entry: {entry_time} {entry_dir} @ ${ep:.2f}")
                    print(f"    Exit:  {exit_time} ({exit_reason}) @ ${fp:.2f}")
                    print(f"    Duration: {bar_duration} min ({bars} bars)")
                    print(f"    Profit: ${prof:.2f}")
                    print(f"    Exit model hold_prob: {hp:.3f}")
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
            event_log.append({"event": total_events, "entry_time": ts_i, "exit_time": m5.index[min(i + TRADE_MAX_BARS, len(m5) - 1)],
                              "dir": entry_dir, "entry_px": ep, "exit_px": fp,
                              "reason": "max_hold", "profit": prof,
                              "bars": min(TRADE_MAX_BARS, len(m5) - 1 - i), "hp": 0.5,
                              "lots": n_lots, "trades": actual_trades})
            if trace_first and total_events == 1:
                entry_time = ts_i
                exit_time = m5.index[min(i + TRADE_MAX_BARS, len(m5) - 1)]
                mins = (exit_time - entry_time).total_seconds() / 60
                bar_duration = bars * 5
                print(f"\n  FIRST EVENT:")
                print(f"    Entry: {entry_time} {entry_dir} @ ${ep:.2f}")
                print(f"    Exit:  {exit_time} (max_hold) @ ${fp:.2f}")
                print(f"    Duration: {bar_duration} min ({bars} bars)")
                print(f"    Profit: ${prof:.2f}")
            bal += prof
            if prof > 0: wins += 1
            exit_reasons["max_hold"] = exit_reasons.get("max_hold", 0) + 1
            meta.record_trade(prof, abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5)

        peak_bal = max(peak_bal, bal)
        dd = max(dd, peak_bal - bal)
        bal = max(bal, 0.01)
        meta.update(bal, {"bias": "BULLISH" if h1_bias[h1_idx_map[i]] == 1 else "BEARISH",
                          "strength": abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5})

    return {"events": total_events, "wins": wins, "net_pnl": round(bal - 20, 2), "dd": round(dd, 2), "exit_reasons": exit_reasons, "event_log": event_log}

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

    for year in [2024, 2025]:
        print(f"\n{'='*80}\n=== {year} - baseline vs exit model comparison ===\n{'='*80}")
        t0 = time.time()

        b = run_bt(year, pred, exit_predictor=None, mode="baseline")
        n = run_bt(year, pred, exit_predictor=ep, mode="narrow")

        # Compare event logs to find disagreements
        min_events = min(len(b["event_log"]), len(n["event_log"]))
        disagreements = []
        for idx in range(min_events):
            be = b["event_log"][idx]
            ne = n["event_log"][idx]
            if be["reason"] != ne["reason"] or abs(be["profit"] - ne["profit"]) > 0.01:
                disagreements.append((be, ne))

        print(f"\nTotal events: baseline={b['events']}, exit_model={n['events']}")
        print(f"Disagreements found: {len(disagreements)}")

        if disagreements:
            # Filter to January only
            jan_dis = [(be, ne) for be, ne in disagreements if be['entry_time'].month == 1]
            show = len(jan_dis)
            print(f"\n{'='*100}")
            jan_count = len(jan_dis)
            print(f"{f'January disagreements (all {jan_count}):':^100}")
            print(f"{'='*100}")
            for idx, (be, ne) in enumerate(jan_dis):
                bals = "?"; nals = "?"
                print(f"\n--- Disagreement #{idx+1} (event #{be['event']}) ---")
                print(f"  Entry: {be['entry_time']} {be['dir']} @ ${be['entry_px']:.2f}")
                print(f"  {'':>10} {'BASELINE':>25} {'EXIT MODEL':>25}")
                print(f"  {'Exit time':>10} {str(be['exit_time']):>25} {str(ne['exit_time']):>25}")
                print(f"  {'Reason':>10} {be['reason']:>25} {ne['reason']:>25}")
                print(f"  {'Bars held':>10} {be['bars']:>25} {ne['bars']:>25}")
                print(f"  {'Lots':>10} {be['lots']:>25.3f} {ne['lots']:>25.3f}")
                print(f"  {'Trades':>10} {be['trades']:>25} {ne['trades']:>25}")
                print(f"  {'Profit':>10} ${be['profit']:>21.2f} ${ne['profit']:>21.2f}")
                if ne['hp'] != 0.5:
                    print(f"  {'ExitModel HP':>10} {'N/A':>25} {ne['hp']:>25.3f}")
                diff = ne['profit'] - be['profit']
                print(f"  {'Difference':>10} {'':>25} ${diff:>+21.2f}")

        # Summary stats
        for name, r in [("BASELINE", b), ("EXIT MODEL", n)]:
            total = r["events"]
            wins = r["wins"]
            losses = total - wins
            wr = wins / total * 100 if total else 0
            gross_win = sum(e["profit"] for e in r["event_log"] if e["profit"] > 0)
            gross_loss = abs(sum(e["profit"] for e in r["event_log"] if e["profit"] <= 0))
            pf = gross_win / gross_loss if gross_loss else float("inf")
            avg_win = gross_win / wins if wins else 0
            avg_loss = gross_loss / losses if losses else 0
            avg_trades = sum(e["trades"] for e in r["event_log"]) / total if total else 0
            max_dd = r["dd"]
            print(f"\n  [{name}] Net PnL: ${r['net_pnl']:+.2f} | WR: {wr:.1f}% | PF: {pf:.2f} | "
                  f"Wins: {wins}/{total} | AvgW: ${avg_win:.2f} | AvgL: ${avg_loss:.2f} | "
                  f"AvgTrades/Event: {avg_trades:.1f} | MaxDD: ${max_dd:.2f}")

        # Monthly comparison
        df_b = pd.DataFrame(b["event_log"])
        df_n = pd.DataFrame(n["event_log"])
        df_b["month"] = df_b["entry_time"].dt.to_period("M")
        df_n["month"] = df_n["entry_time"].dt.to_period("M")
        monthly_b = df_b.groupby("month").agg(events=("profit", "count"), net_pnl=("profit", "sum"),
                                              wins=("profit", lambda x: (x > 0).sum()))
        monthly_n = df_n.groupby("month").agg(events=("profit", "count"), net_pnl=("profit", "sum"),
                                              wins=("profit", lambda x: (x > 0).sum()))
        monthly_b["wr"] = monthly_b["wins"] / monthly_b["events"] * 100
        monthly_n["wr"] = monthly_n["wins"] / monthly_n["events"] * 100
        compare = pd.DataFrame({
            "BL_events": monthly_b["events"], "BL_PnL": monthly_b["net_pnl"], "BL_WR": monthly_b["wr"],
            "EX_events": monthly_n["events"], "EX_PnL": monthly_n["net_pnl"], "EX_WR": monthly_n["wr"],
        })
        compare["Diff_PnL"] = compare["EX_PnL"] - compare["BL_PnL"]
        print(f"\n  Monthly comparison:")
        print(compare.to_string())

        print(f"\n  Time: {time.time()-t0:.1f}s")
