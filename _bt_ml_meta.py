"""
Fast 2025 backtest: ML + MetaStrategy + multi-trade events.
Checks max event loss and frequency of $5+ event losses.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features
from app.bias_engine import BiasEngine
from app.risk_manager import EquityScaler
from app.meta_strategy import MetaStrategy

CS = 100; LEV = 200
INITIAL_BALANCE = 20.0
CONTRACT_SIZE = 100
EVENT_LOSS_LIMIT = cfg.MAX_EVENT_LOSS_USD  # $5.00

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
    is_high = np.zeros(n, dtype=bool); is_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        is_high[i] = all(h[i] >= h[i-j] for j in range(1, lookback+1)) and all(h[i] >= h[i+j] for j in range(1, lookback+1))
        is_low[i] = all(l[i] <= l[i-j] for j in range(1, lookback+1)) and all(l[i] <= l[i+j] for j in range(1, lookback+1))
    swing_score = np.zeros(n)
    for i in range(n):
        hi = np.where(is_high[:i+1])[0]; lo = np.where(is_low[:i+1])[0]
        if len(hi) >= 3 and len(lo) >= 3:
            rh = hi[-3:]; rl = lo[-3:]
            h_up = sum(1 for k in range(1, len(rh)) if h[rh[k]] > h[rh[k-1]])
            h_dn = sum(1 for k in range(1, len(rh)) if h[rh[k]] < h[rh[k-1]])
            l_up = sum(1 for k in range(1, len(rl)) if l[rl[k]] > l[rl[k-1]])
            l_dn = sum(1 for k in range(1, len(rl)) if l[rl[k]] < l[rl[k-1]])
            swing_score[i] = ((h_up - h_dn) + (l_up - l_dn)) / max(1, (len(rh)-1) + (len(rl)-1))
    total = votes + swing_score
    bias = np.zeros(n, dtype=np.int8)
    bias[total >= 0.75] = 1
    bias[total <= -0.75] = -1
    return bias

def _pnl(entry_px, exit_px, direction, lot, num_trades):
    delta = exit_px - entry_px if direction == "BUY" else entry_px - exit_px
    # simplified spread/slippage
    spread = 25.0 * lot * num_trades * 2
    slippage = 3.0 * lot * num_trades
    return delta * CONTRACT_SIZE * lot * num_trades - spread - slippage

def run():
    t0 = time.time()
    print("Loading data...", flush=True)
    client = DukascopyClient()
    m1 = client.download_year(2025)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]
    print(f"M1: {len(m1)} M5: {len(m5)} H1: {len(h1)} [{time.time()-t0:.0f}s]", flush=True)

    # Features
    print("Computing features...", flush=True)
    ft = compute_features(m5, h1)
    print(f"Features: {len(ft)} rows [{time.time()-t0:.0f}s]", flush=True)

    # ML predictions
    print("Loading ML model...", flush=True)
    pred = DirectionPredictor.load(cfg.ML_MODEL_PATH)
    X = ft[ft.columns.intersection(pred._feature_cols)]
    for c in pred._feature_cols:
        if c not in X.columns: X[c] = 0.0
    X = X[pred._feature_cols]
    X_aligned = X.reindex(m5.index, method="ffill")
    ml_mask = ~X_aligned.isna().any(axis=1)
    ml_dir = np.full(len(m5), None, dtype=object)
    if ml_mask.any():
        valid_idx = np.where(ml_mask)[0]
        probs = pred.model.predict_proba(X_aligned[ml_mask].values)
        pb_u = np.array([p[1] for p in probs])
        pb_d = np.array([p[0] for p in probs])
        for k, idx in enumerate(valid_idx):
            if pb_u[k] >= cfg.ML_CONFIDENCE_THRESHOLD:
                ml_dir[idx] = "BUY"
            elif pb_d[k] >= cfg.ML_CONFIDENCE_THRESHOLD:
                ml_dir[idx] = "SELL"
    print(f"ML predictions done [{time.time()-t0:.0f}s]", flush=True)

    # Bias
    print("Computing bias...", flush=True)
    h1_bias = compute_bias_vectorized(h1)
    h1_idx_map = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx_map = np.clip(h1_idx_map, 0, len(h1_bias) - 1)
    m5_bias_arr = np.full(len(h1_idx_map), None, dtype=object)
    m5_bias_arr[h1_bias[h1_idx_map] == 1] = "BUY"
    m5_bias_arr[h1_bias[h1_idx_map] == -1] = "SELL"
    print(f"Bias done [{time.time()-t0:.0f}s]", flush=True)

    # H1 high/low for each M5 bar
    m5_h1h = h1["high"].values[h1_idx_map]
    m5_h1l = h1["low"].values[h1_idx_map]
    m5_h1h = np.where(h1_idx_map > 0, h1["high"].values[h1_idx_map - 1], m5_h1h)
    m5_h1l = np.where(h1_idx_map > 0, h1["low"].values[h1_idx_map - 1], m5_h1l)

    # Backtest loop
    print("Running backtest...", flush=True)
    bal = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    meta = MetaStrategy()
    scaler = EquityScaler()
    scaler.initialize(INITIAL_BALANCE)

    event_trades = []
    event_pnl = 0.0
    consec_losses = 0
    max_event_loss = 0.0
    event_losses_5plus = 0
    total_events = 0
    all_events = []

    for i in range(60, len(m5) - 15):
        expected = m5_bias_arr[i]
        if expected is None:
            continue
        if ml_dir[i] is None or ml_dir[i] != expected:
            continue

        p = float(m5.iloc[i]["close"])
        h1h = float(m5_h1h[i]); h1l = float(m5_h1l[i])
        if h1h <= h1l:
            continue
        if expected == "BUY" and p <= h1h:
            continue
        if expected == "SELL" and p >= h1l:
            continue

        # Meta + sizing
        bsum = {"bias": "BULLISH" if expected == "BUY" else "BEARISH", "strength": 1.0}
        meta.update(bal, bsum)
        scaler.base_trades = cfg.MAX_TRADES_PER_EVENT
        lot_mult = meta.current_lot_mult
        max_tr = meta.current_trades_per_event
        et = meta.current_threshold

        # ATR-based threshold floor
        atr_val = float(m5.iloc[i]["close"]) * 0.002
        atr_et = atr_val / max(h1h - h1l, 0.01)
        effective_et = max(atr_et, et)

        # Simplified: entry threshold check (score = breakout dist / range)
        if expected == "BUY":
            score = (p - h1h) / max(h1h - h1l, 0.01)
        else:
            score = (h1l - p) / max(h1h - h1l, 0.01)
        score = min(score, 1.0)
        if score < effective_et:
            continue

        # Sizing
        n = max(0.01, min(1.0, bal * LEV / p / CONTRACT_SIZE))
        n_lot = max(cfg.MIN_LOT, min(n * lot_mult, cfg.MAX_LOT))
        num_tr = max(1, min(max_tr, int(bal * LEV / p / CONTRACT_SIZE / n_lot)))
        total_lot = n_lot * num_tr
        if total_lot <= 0 or total_lot <= cfg.MIN_LOT:
            continue

        ep = p
        # SL/TP
        if expected == "BUY":
            sl = ep - atr_val; tp1 = ep + atr_val*2; tp2 = ep + atr_val*4; tp3 = ep + atr_val*6
        else:
            sl = ep + atr_val; tp1 = ep - atr_val*2; tp2 = ep - atr_val*4; tp3 = ep - atr_val*6

        event_start = i
        event_entry_pnl = 0.0
        trade_results = []
        prev_event_pnl = event_pnl

        for j in range(i + 1, min(i + 48, len(m5) - 1)):
            fb = m5.iloc[j]
            fp = float(fb["close"]); fh = float(fb["high"]); fl = float(fb["low"])

            trade_pnl = _pnl(ep, fp, expected, n_lot, num_tr)

            hit = False
            if expected == "BUY":
                if fl <= sl:
                    event_entry_pnl += trade_pnl; hit = True
                elif fp >= tp3:
                    event_entry_pnl += trade_pnl; hit = True
                elif fp >= tp2:
                    event_entry_pnl += trade_pnl; hit = True
                elif fp >= tp1:
                    event_entry_pnl += trade_pnl; hit = True
            else:
                if fh >= sl:
                    event_entry_pnl += trade_pnl; hit = True
                elif fp <= tp3:
                    event_entry_pnl += trade_pnl; hit = True
                elif fp <= tp2:
                    event_entry_pnl += trade_pnl; hit = True
                elif fp <= tp1:
                    event_entry_pnl += trade_pnl; hit = True

            if hit:
                trade_results.append(trade_pnl)
                i = j
                break

        else:
            fb = m5.iloc[min(i + 48, len(m5) - 1)]
            fp = float(fb["close"])
            trade_pnl = _pnl(ep, fp, expected, n_lot, num_tr)
            event_entry_pnl += trade_pnl
            trade_results.append(trade_pnl)
            i = j

        bal += event_entry_pnl
        peak = max(peak, bal)
        meta.record_trade(event_entry_pnl, 1.0)

        event_pnl = 0.0
        event_loss = -event_entry_pnl if event_entry_pnl < 0 else 0
        total_events += 1
        all_events.append(event_entry_pnl)
        max_event_loss = max(max_event_loss, event_loss)
        if event_loss >= 5.0:
            event_losses_5plus += 1

        if event_entry_pnl < 0:
            consec_losses += 1
        else:
            consec_losses = 0

    elapsed = time.time() - t0
    events_np = np.array(all_events)
    wins = events_np[events_np > 0]
    losses = events_np[events_np < 0]
    wr = len(wins) / max(1, len(events_np)) * 100
    gross_p = wins.sum() if len(wins) > 0 else 0
    gross_l = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gross_p / max(gross_l, 1e-9)

    print(f"\n{'='*60}", flush=True)
    print(f"2025 ML + META BACKTEST RESULTS [{elapsed:.0f}s]", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total events:       {total_events}", flush=True)
    print(f"Win rate:           {wr:.1f}%", flush=True)
    print(f"Profit factor:      {pf:.2f}", flush=True)
    print(f"Net PnL:            ${bal - INITIAL_BALANCE:+.2f}", flush=True)
    print(f"Peak balance:       ${peak:.2f}", flush=True)
    print(f"Max event loss:     ${max_event_loss:.2f}", flush=True)
    print(f"Events >= $5 loss:  {event_losses_5plus}", flush=True)
    print(f"Pct events >= $5:   {event_losses_5plus/max(1,total_events)*100:.1f}%", flush=True)
    print(f"Ending balance:     ${bal:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    # Top 10 worst losses
    sorted_losses = sorted([l for l in all_events if l < 0])
    print(f"\nTop 10 worst event losses:", flush=True)
    for i, l in enumerate(sorted_losses[:10]):
        print(f"  {i+1}. ${l:.2f}", flush=True)

if __name__ == "__main__":
    run()
