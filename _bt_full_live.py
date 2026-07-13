"""Full backtest: ALL live models (ASP + Trend + Breakout + ML exits).

Entry priority: ASP → Trend → Breakout (same as bot.py)
Exit logic:
  - ASP trades: SL 2xATR, TP 1xATR, timeout 6 bars
  - Trend trades: TrendExhaustionPredictor, safety SL 2xATR
  - Breakout trades: trail SL (mode 1)
"""
import os, sys, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features
from app.asp_labels import compute_atr
from app.direction_predictor import (
    TrendPredictor, TrendExhaustionPredictor,
    ExitPredictor, compute_features, compute_trend_features,
    FEATURE_COLS, EXIT_TREND_FEATURE_COLS,
)

# ── Config (matching live bot) ──
ASP_MODEL_PATH = "models/asp_swing_xgb_m5.joblib"
ASP_FEATURE_PATH = "models/asp_swing_m5_features.npy"
TREND_MODEL_PATH = "models/trend_xgb_h1.joblib"
TREND_EXIT_MODEL_PATH = "models/exit_trend_xgb_m5.joblib"
EXIT_MODEL_PATH = "models/exit_xgb_m5.joblib"
DIRECTION_MODEL_PATH = "models/direction_xgb_m5.joblib"

ASP_SL_MULT = 2.0
ASP_TP_MULT = 1.0
ASP_TIMEOUT = 6
ASP_MIN_ATR = 0.50

TREND_SL_MULT = 1.0
TREND_EXIT_EXHAUSTION_THR = 0.70
TREND_EXIT_NEW_SETUP_THR = 0.80
TREND_CONF_THR = 0.55

BREAKOUT_SL_MULT = 1.0
BREAKOUT_ET_THR = 0.10
BREAKOUT_MIN_SCORE = 0.02

ATR_PERIOD = 14
STARTING_BALANCE = 20.0
BASE_LOT = 0.01
CONTRACT_SIZE = 100
SPREAD_PTS = 0.30
LOT_STEP = 0.01
MIN_LOT = 0.01
MAX_LOT = 9999.0

EXIT_TRAIL_SL_MULT = 0.50
EXIT_TRAIL_TRIGGER = 0.70
EXIT_TRAIL_RETRACE = 0.30

YEARS = [2023, 2024, 2025]


def calc_lot(balance, peak_balance):
    ref = 20.0
    lot = BASE_LOT * (balance / ref)
    if peak_balance and balance < peak_balance * 0.9:
        lot *= 0.5
    lot = round(lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(lot, MAX_LOT))


def load_models():
    print("Loading models...")
    asp_saved = joblib.load(ASP_MODEL_PATH)
    asp_model = asp_saved["model"]
    asp_label_map = asp_saved.get("label_map", {})
    asp_inv_map = {v: k for k, v in asp_label_map.items()}
    asp_features = list(np.load(ASP_FEATURE_PATH, allow_pickle=True))
    print(f"  ASP: {len(asp_features)} features")

    trend_predictor = TrendPredictor(TREND_MODEL_PATH)
    print(f"  Trend: {'loaded' if trend_predictor.model else 'MISSING'}")

    trend_exit = TrendExhaustionPredictor(TREND_EXIT_MODEL_PATH)
    print(f"  TrendExit: {'loaded' if trend_exit.model else 'MISSING'}")

    exit_pred = ExitPredictor(EXIT_MODEL_PATH)
    print(f"  Exit: {'loaded' if exit_pred.model else 'MISSING'}")

    direction_pred = None
    if os.path.exists(DIRECTION_MODEL_PATH):
        from app.direction_predictor import DirectionPredictor
        direction_pred = DirectionPredictor(DIRECTION_MODEL_PATH)
        print(f"  Direction: loaded")
    else:
        print(f"  Direction: MISSING")

    return asp_model, asp_inv_map, asp_features, trend_predictor, trend_exit, exit_pred, direction_pred


def compute_all_features(m5, h1):
    """Compute ASP features, trend features, and direction features."""
    asp_feats = compute_asp_features(m5, h1)
    trend_feats = compute_trend_features(m5, h1)
    dir_feats = compute_features(m5, h1)
    return asp_feats, trend_feats, dir_feats


def check_asp_entry(i, signal_arr, atr_arr, m5_c):
    """Check ASP entry at bar i. Returns entry dict or None."""
    d = signal_arr[i]
    if d == 0:
        return None
    atr_val = atr_arr[i]
    if np.isnan(atr_val) or atr_val <= 0:
        return None
    if atr_val < ASP_MIN_ATR:
        return None
    direction = "BUY" if d == 1 else "SELL"
    entry_price = m5_c[i]
    sl_dist = atr_val * ASP_SL_MULT
    tp_dist = atr_val * ASP_TP_MULT
    sl_price = entry_price - sl_dist if direction == "BUY" else entry_price + sl_dist
    tp_price = entry_price + tp_dist if direction == "BUY" else entry_price - tp_dist
    return {
        "model": "asp", "direction": direction, "entry_price": entry_price,
        "sl": sl_price, "tp": tp_price, "atr": atr_val,
    }


def check_trend_entry(i, trend_feats, trend_predictor, h1_high, h1_low, m5_c):
    """Check Trend entry at bar i. Returns entry dict or None."""
    if trend_feats is None or len(trend_feats) == 0:
        return None
    if i >= len(trend_feats):
        return None
    row = trend_feats.iloc[[i]]
    result = trend_predictor.predict(row, TREND_CONF_THR)
    if result is None:
        return None
    direction = "BUY" if result == "UP_TREND" else "SELL"
    entry_price = m5_c[i]
    return {
        "model": "trend", "direction": direction, "entry_price": entry_price,
        "confidence": TREND_CONF_THR, "atr": 0,
    }


def check_breakout_entry(i, m5_c, m5_h, m5_l, h1_high, h1_low, direction_pred, dir_feats):
    """Check Breakout entry at bar i. Returns entry dict or None."""
    if h1_high is None or h1_low is None or h1_high <= h1_low:
        return None
    range_size = h1_high - h1_low
    if range_size <= 0:
        return None
    price = m5_c[i]
    # Determine bias from recent price action
    if price > h1_high:
        direction = "BUY"
        breakout_dist = price - h1_high
    elif price < h1_low:
        direction = "SELL"
        breakout_dist = h1_low - price
    else:
        return None
    score = breakout_dist / range_size
    if score < BREAKOUT_MIN_SCORE:
        return None
    # ML direction validation
    if direction_pred is not None and dir_feats is not None and i < len(dir_feats):
        row = dir_feats.iloc[[i]]
        ml_dir = direction_pred.predict(row, 0.55)
        if ml_dir is None:
            return None
        if ml_dir == "BUY" and direction == "SELL":
            return None
        if ml_dir == "SELL" and direction == "BUY":
            return None
    entry_price = m5_c[i]
    return {
        "model": "breakout", "direction": direction, "entry_price": entry_price,
        "score": score, "atr": 0,
    }


def check_asp_exit(trade, m5_h, m5_l, m5_c, bars_held):
    """ASP exit: SL/TP/timeout."""
    direction = trade["direction"]
    sl, tp = trade["sl"], trade["tp"]
    if direction == "BUY":
        if m5_l <= sl:
            return "asp_sl", sl
        if m5_h >= tp:
            return "asp_tp", tp
    else:
        if m5_h >= sl:
            return "asp_sl", sl
        if m5_l <= tp:
            return "asp_tp", tp
    if bars_held >= ASP_TIMEOUT:
        return "asp_timeout", m5_c
    return None, None


def precompute_trend_exit_signals(trend_feats, trend_exit, m5_h, m5_l, m5_c, n):
    """Pre-compute TrendExhaustion exit signals for all bars to avoid per-bar ML calls."""
    exhausted = np.zeros(n, dtype=bool)
    new_setup = np.zeros(n, dtype=bool)
    if trend_exit.model is None or trend_feats is None or len(trend_feats) == 0:
        return exhausted, new_setup
    try:
        valid_cols = [c for c in trend_exit._feature_cols if c in trend_feats.columns]
        if len(valid_cols) != len(trend_exit._feature_cols):
            return exhausted, new_setup
        valid = trend_feats[valid_cols].dropna()
        if len(valid) == 0:
            return exhausted, new_setup
        feat_arr = valid.values.astype(np.float32)
        probs = trend_exit.model.predict_proba(feat_arr)
        classes = list(trend_exit.model.classes_)
        exh_idx = classes.index(0) if 0 in classes else -1
        new_idx = classes.index(2) if 2 in classes else -1
        if exh_idx >= 0:
            exh_probs = probs[:, exh_idx]
            for idx, ts in enumerate(valid.index):
                loc = trend_feats.index.get_loc(ts)
                if exh_probs[idx] >= TREND_EXIT_EXHAUSTION_THR:
                    exhausted[loc] = True
        if new_idx >= 0:
            new_probs = probs[:, new_idx]
            for idx, ts in enumerate(valid.index):
                loc = trend_feats.index.get_loc(ts)
                if new_probs[idx] >= TREND_EXIT_NEW_SETUP_THR:
                    new_setup[loc] = True
    except Exception as e:
        print(f"  Warning: trend exit precompute failed: {e}")
    return exhausted, new_setup


def check_breakout_exit(trade, m5_data, i, bars_held):
    """Breakout exit: trail SL mode."""
    direction = trade["direction"]
    entry_price = trade["entry_price"]
    atr_val = compute_atr(m5_data["high"].values, m5_data["low"].values, m5_data["close"].values, ATR_PERIOD)
    atr_now = atr_val[i] if i < len(atr_val) else 1.0
    if np.isnan(atr_now) or atr_now <= 0:
        atr_now = entry_price * 0.001

    px = m5_c[i]
    diff = (px - entry_price) if direction == "BUY" else (entry_price - px)

    sl_dist = max(atr_now * BREAKOUT_SL_MULT, 0.20)
    if direction == "BUY" and m5_l[i] <= entry_price - sl_dist:
        return "bo_stop_loss", entry_price - sl_dist
    if direction == "SELL" and m5_h[i] >= entry_price + sl_dist:
        return "bo_stop_loss", entry_price + sl_dist

    # Trail logic
    trail_trigger = max(atr_now * EXIT_TRAIL_TRIGGER, 0.20)
    peak = 0.0
    lookback = min(i + 1, 5)
    for j in range(i - lookback + 1, i + 1):
        if direction == "BUY":
            peak = max(peak, m5_h[j] - entry_price)
        else:
            peak = max(peak, entry_price - m5_l[j])

    triggered = peak >= trail_trigger
    if triggered and peak > 0 and diff > 0:
        pullback = peak - diff
        if pullback / peak > EXIT_TRAIL_RETRACE:
            return "bo_trail_stop", px

    # Breakeven stop
    sl_stop = -(max(atr_now * EXIT_TRAIL_SL_MULT, 0.15))
    if triggered and sl_stop < 0:
        sl_stop = 0.0
    if diff <= sl_stop:
        return "bo_breakeven" if sl_stop >= 0 else "bo_stop_loss", px

    return None, None


# ── Main ──
asp_model, asp_inv_map, asp_features, trend_predictor, trend_exit, exit_pred, direction_pred = load_models()
client = DukascopyClient()
all_results = []
all_trades = []

for YEAR in YEARS:
    print(f"\n{'='*70}")
    print(f"  FULL LIVE BACKTEST {YEAR}")
    print(f"{'='*70}")

    t0 = time.time()
    m1 = client.download_year(YEAR)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    print(f"  M1={len(m1)}, M5={len(m5)}, H1={len(h1)} ({time.time()-t0:.1f}s)")

    m5 = m5.set_index("time")
    h1 = h1.set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    n = len(m5)
    m5_h = m5["high"].values.astype(np.float64)
    m5_l = m5["low"].values.astype(np.float64)
    m5_c = m5["close"].values.astype(np.float64)
    atr_arr = compute_atr(m5_h, m5_l, m5_c, ATR_PERIOD)

    # Compute features
    t_f = time.time()
    asp_feats, trend_feats, dir_feats = compute_all_features(m5, h1)
    print(f"  Features computed ({time.time()-t_f:.1f}s)")

    # ASP signals
    asp_signal_arr = np.zeros(n, dtype=np.int8)
    if asp_feats is not None and len(asp_feats) > 0:
        valid_asp = asp_feats[asp_features].dropna()
        if len(valid_asp) > 0:
            feat_arr = valid_asp[asp_features].values
            raw_preds = asp_model.predict(feat_arr)
            preds = np.array([asp_inv_map.get(p, p) for p in raw_preds])
            asp_signal_arr[asp_feats.index.get_indexer(valid_asp.index)] = preds

    buy_n = (asp_signal_arr == 1).sum()
    sell_n = (asp_signal_arr == -1).sum()
    print(f"  ASP signals: BUY={buy_n}, SELL={sell_n}")

    # Pre-compute trend exit signals
    print(f"  Pre-computing trend exit signals...")
    t_e = time.time()
    exhausted_arr, new_setup_arr = precompute_trend_exit_signals(trend_feats, trend_exit, m5_h, m5_l, m5_c, n)
    exh_n = exhausted_arr.sum()
    new_n = new_setup_arr.sum()
    print(f"  Trend exit: exhausted={exh_n}, new_setup={new_n} ({time.time()-t_e:.1f}s)")

    # Simulation
    trades = []
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    in_trade = False
    trade = None
    entry_idx = 0
    bars_held = 0
    entry_lot = 0.01

    START_IDX = max(ATR_PERIOD + 5, 100)

    for i in range(START_IDX, n):
        atr_now = atr_arr[i]
        if np.isnan(atr_now) or atr_now <= 0:
            atr_now = 1.0

        # Get H1 range for breakout filter
        ts = m5.index[i]
        h1_slice = h1[h1.index <= ts]
        if len(h1_slice) >= 2:
            h1_high = float(h1_slice["high"].iloc[-2])  # previous completed H1
            h1_low = float(h1_slice["low"].iloc[-2])
        else:
            h1_high = None
            h1_low = None

        if in_trade:
            bars_held += 1
            model = trade["model"]
            exit_reason = None
            exit_price = None

            if model == "asp":
                exit_reason, exit_price = check_asp_exit(trade, m5_h[i], m5_l[i], m5_c[i], bars_held)
            elif model == "trend":
                direction = trade["direction"]
                entry_price = trade["entry_price"]
                # Safety SL at 2x ATR
                atr_now = atr_arr[i]
                if np.isnan(atr_now) or atr_now <= 0:
                    atr_now = 1.0
                px = m5_c[i]
                diff = (px - entry_price) if direction == "BUY" else (entry_price - px)
                if diff <= -(2.0 * atr_now):
                    exit_reason, exit_price = "trend_safety_sl", px
                elif i < len(exhausted_arr) and exhausted_arr[i]:
                    exit_reason, exit_price = "trend_exhaustion", px
                elif i < len(new_setup_arr) and new_setup_arr[i]:
                    exit_reason, exit_price = "trend_new_setup", px
                elif bars_held >= 12:
                    exit_reason, exit_price = "trend_timeout", px
            elif model == "breakout":
                exit_reason, exit_price = check_breakout_exit(trade, m5, i, bars_held)

            if exit_reason:
                direction = trade["direction"]
                ep = trade["entry_price"]
                if direction == "BUY":
                    pnl = (exit_price - ep) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                else:
                    pnl = (ep - exit_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[i],
                    "direction": direction, "entry": ep, "exit": round(exit_price, 2),
                    "pnl_usd": round(pnl, 2), "lot": entry_lot,
                    "reason": exit_reason, "bars": bars_held,
                    "model": model,
                })
                in_trade = False
                trade = None
                bars_held = 0
                continue

        else:
            # Entry priority: ASP → Trend → Breakout
            entry = None

            # 1. ASP
            entry = check_asp_entry(i, asp_signal_arr, atr_arr, m5_c)

            # 2. Trend
            if entry is None:
                entry = check_trend_entry(i, trend_feats, trend_predictor, h1_high, h1_low, m5_c)

            # 3. Breakout
            if entry is None:
                entry = check_breakout_entry(i, m5_c, m5_h, m5_l, h1_high, h1_low, direction_pred, dir_feats)

            if entry is not None:
                entry_price = entry["entry_price"]
                direction = entry["direction"]
                model = entry["model"]

                if model == "asp":
                    sl_price = entry["sl"]
                    tp_price = entry["tp"]
                elif model == "trend":
                    sl_dist = atr_now * TREND_SL_MULT
                    sl_price = entry_price - sl_dist if direction == "BUY" else entry_price + sl_dist
                    tp_price = None
                else:  # breakout
                    sl_dist = max(atr_now * BREAKOUT_SL_MULT, 0.20)
                    sl_price = entry_price - sl_dist if direction == "BUY" else entry_price + sl_dist
                    tp_price = None

                entry_lot = calc_lot(balance, peak_balance)
                in_trade = True
                trade = {**entry, "sl": sl_price, "tp": tp_price}
                entry_idx = i
                bars_held = 0

    # Close open trade at EOY
    if in_trade and trade:
        direction = trade["direction"]
        ep = trade["entry_price"]
        exit_price = m5_c[-1]
        if direction == "BUY":
            pnl = (exit_price - ep) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        else:
            pnl = (ep - exit_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        balance += pnl
        peak_balance = max(peak_balance, balance)
        trades.append({
            "entry_idx": entry_idx, "exit_idx": n - 1,
            "entry_ts": m5.index[entry_idx], "exit_ts": m5.index[-1],
            "direction": direction, "entry": ep, "exit": round(exit_price, 2),
            "pnl_usd": round(pnl, 2), "lot": entry_lot,
            "reason": "eoy_close", "bars": bars_held,
            "model": trade["model"],
        })

    df = pd.DataFrame(trades)
    all_trades.extend(trades)

    if len(df) == 0:
        print("  No trades")
        continue

    # Summary
    wins = (df["pnl_usd"] > 0).sum()
    losses = (df["pnl_usd"] <= 0).sum()
    wr = wins / len(df) * 100
    gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = abs(df.loc[df["pnl_usd"] <= 0, "pnl_usd"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    net_pnl = df["pnl_usd"].sum()

    print(f"\n  Ending balance: ${balance:,.2f}  |  Net PnL: ${net_pnl:,.2f}")
    print(f"  Trades: {len(df)}  |  WR: {wr:.1f}%  |  PF: {pf:.2f}")

    # Breakdown by model
    for model_name in ["asp", "trend", "breakout"]:
        mdf = df[df["model"] == model_name]
        if len(mdf) == 0:
            continue
        mw = (mdf["pnl_usd"] > 0).sum()
        mwr = mw / len(mdf) * 100
        mgp = mdf.loc[mdf["pnl_usd"] > 0, "pnl_usd"].sum()
        mgl = abs(mdf.loc[mdf["pnl_usd"] <= 0, "pnl_usd"].sum())
        mpf = mgp / mgl if mgl > 0 else 99.9
        print(f"    [{model_name.upper():>8}] {len(mdf):>4} trades, WR={mwr:.1f}%, PF={mpf:.2f}, PnL=${mdf['pnl_usd'].sum():>10,.2f}")

    # Breakdown by exit reason
    reasons = df.groupby("reason").agg(
        count=("pnl_usd", "size"),
        wins=("pnl_usd", lambda x: (x > 0).sum()),
        total_pnl=("pnl_usd", "sum"),
    ).reset_index()
    reasons["wr"] = reasons["wins"] / reasons["count"] * 100
    print(f"\n  Exit breakdown:")
    for _, row in reasons.iterrows():
        print(f"    {row['reason']:>22}: {row['count']:>4} trades, WR={row['wr']:.1f}%, PnL=${row['total_pnl']:>10,.2f}")

    all_results.append({
        "year": YEAR, "balance": balance, "net_pnl": net_pnl,
        "trades": len(df), "wins": wins, "losses": losses,
        "wr": wr, "pf": pf,
    })

# ── Multi-year summary ──
print(f"\n{'='*70}")
print(f"  FULL LIVE MODEL BACKTEST SUMMARY (2023-2025)")
print(f"{'='*70}")
print(f"{'Year':>6} {'Balance':>14} {'Net PnL':>14} {'Trades':>8} {'WR%':>7} {'PF':>7}")
print(f"{'-'*6} {'-'*14} {'-'*14} {'-'*8} {'-'*7} {'-'*7}")

for r in all_results:
    print(f"{r['year']:>6} ${r['balance']:>12,.2f} ${r['net_pnl']:>12,.2f} {r['trades']:>8} {r['wr']:>6.1f}% {r['pf']:>6.2f}")

total_pnl = sum(r["net_pnl"] for r in all_results)
total_trades = sum(r["trades"] for r in all_results)
total_wins = sum(r["wins"] for r in all_results)
avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print(f"{'TOTAL':>6} ${all_results[-1]['balance']:>12,.2f} ${total_pnl:>12,.2f} {total_trades:>8} {avg_wr:>6.1f}%")

# Full trade breakdown
if all_trades:
    fdf = pd.DataFrame(all_trades)
    print(f"\n  === MODEL BREAKDOWN (all years) ===")
    for model_name in ["asp", "trend", "breakout"]:
        mdf = fdf[fdf["model"] == model_name]
        if len(mdf) == 0:
            continue
        mw = (mdf["pnl_usd"] > 0).sum()
        mwr = mw / len(mdf) * 100
        mgp = mdf.loc[mdf["pnl_usd"] > 0, "pnl_usd"].sum()
        mgl = abs(mdf.loc[mdf["pnl_usd"] <= 0, "pnl_usd"].sum())
        mpf = mgp / mgl if mgl > 0 else 99.9
        print(f"    [{model_name.upper():>8}] {len(mdf):>5} trades, WR={mwr:.1f}%, PF={mpf:.2f}, PnL=${mdf['pnl_usd'].sum():>12,.2f}")

    print(f"\n  === EXIT REASON BREAKDOWN (all years) ===")
    reasons = fdf.groupby("reason").agg(
        count=("pnl_usd", "size"),
        wins=("pnl_usd", lambda x: (x > 0).sum()),
        total_pnl=("pnl_usd", "sum"),
    ).reset_index()
    reasons["wr"] = reasons["wins"] / reasons["count"] * 100
    for _, row in reasons.iterrows():
        print(f"    {row['reason']:>22}: {row['count']:>5} trades, WR={row['wr']:.1f}%, PnL=${row['total_pnl']:>12,.2f}")
