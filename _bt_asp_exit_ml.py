"""Fast A/B backtest: ASP-only vs ASP+TrendExhaustion exits.
Single data load, two modes. Target <60s.
"""
import os, sys, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features
from app.direction_predictor import compute_features, FEATURE_COLS, TRADE_STATE_TREND_COLS, EXIT_TREND_FEATURE_COLS
from app.asp_labels import compute_atr

ASP_MODEL_PATH = "models/asp_swing_xgb_m5.joblib"
ASP_FEATURE_PATH = "models/asp_swing_m5_features.npy"
EXIT_MODEL_PATH = "models/exit_trend_xgb_m5.joblib"

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.0
TIMEOUT_BARS = 6
MIN_EXIT_BARS = 2
ATR_PERIOD = 14
STARTING_BALANCE = 20.0
BASE_LOT = 0.01
CONTRACT_SIZE = 100
SPREAD_PTS = 0.30
LOT_STEP = 0.01
MIN_LOT = 0.01
MAX_LOT = 9999.0
EXHAUSTION_THRESHOLD = 0.70
NEW_SETUP_THRESHOLD = 0.80
YEAR = 2024


def calc_lot(balance, peak_balance):
    lot = BASE_LOT * (balance / 20.0)
    if peak_balance and balance < peak_balance * 0.9:
        lot *= 0.5
    lot = round(lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(lot, MAX_LOT))


def run_bt(asp_arr, atr_arr, m5_c, m5_h, m5_l, m5_index, m5_feats, exit_model,
           use_exit_ml, start_idx, n):
    trades = []
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    in_trade = False
    entry_idx = entry_lot = bars_in_trade = 0
    entry_price = sl_price = tp_price = 0.0
    entry_dir = ""

    for i in range(start_idx, n):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        if in_trade:
            bars_in_trade += 1
            exit_reason = ""
            exit_price = 0.0

            # SL
            if entry_dir == "BUY" and m5_l[i] <= sl_price:
                exit_reason, exit_price = "sl_hit", sl_price
            elif entry_dir == "SELL" and m5_h[i] >= sl_price:
                exit_reason, exit_price = "sl_hit", sl_price

            # TP
            if not exit_reason:
                if entry_dir == "BUY" and m5_h[i] >= tp_price:
                    exit_reason, exit_price = "tp_hit", tp_price
                elif entry_dir == "SELL" and m5_l[i] <= tp_price:
                    exit_reason, exit_price = "tp_hit", tp_price

            # TrendExhaustion ML exit
            if use_exit_ml and not exit_reason and bars_in_trade >= MIN_EXIT_BARS:
                ts = m5_index[i]
                if ts in m5_feats.index:
                    mf_row = m5_feats.loc[ts]
                    px = m5_c[i]
                    diff = px - entry_price if entry_dir == "BUY" else entry_price - px
                    cur_atr = max(atr, 0.01)
                    if entry_dir == "BUY":
                        peak = float(np.max(m5_h[max(0, i-20):i+1] - entry_price))
                    else:
                        peak = float(np.max(entry_price - m5_l[max(0, i-20):i+1]))
                    md = 0.5
                    if bars_in_trade > 5 and i >= 5:
                        if entry_dir == "BUY":
                            recent = m5_c[i] - m5_c[i-5]
                            entry_m = m5_c[i-4] - entry_price
                        else:
                            recent = m5_c[i-5] - m5_c[i]
                            entry_m = entry_price - m5_c[i-4]
                        md = 1.0 - min(abs(recent) / max(abs(entry_m), 0.001), 2.0)
                    trade_state = {
                        "bars_held": bars_in_trade,
                        "trend_pnl_atr": round(diff / cur_atr, 4),
                        "trend_peak_atr": round(peak / cur_atr, 4),
                        "trend_drawdown_pct": round(max(0, peak - diff) / max(peak, 0.001), 4),
                        "trend_momentum_decay": round(md, 4),
                    }
                    try:
                        row = {}
                        for c in FEATURE_COLS:
                            val = mf_row[c] if c in mf_row.index else 0.0
                            row[c] = float(val) if not np.isnan(val) else 0.0
                        for c in TRADE_STATE_TREND_COLS:
                            row[c] = float(trade_state.get(c, 0.0))
                        vec = np.array([[row.get(c, 0.0) for c in EXIT_TREND_FEATURE_COLS]], dtype=np.float32)
                        probs = exit_model.predict_proba(vec)[0]
                        classes = list(exit_model.classes_)
                        p_exh, p_alive, p_new = 0.2, 0.6, 0.2
                        for j, cls in enumerate(classes):
                            if cls == 0: p_exh = float(probs[j])
                            elif cls == 1: p_alive = float(probs[j])
                            elif cls == 2: p_new = float(probs[j])
                        if p_exh >= EXHAUSTION_THRESHOLD:
                            exit_reason, exit_price = "trend_exhaustion", m5_c[i]
                        elif p_new >= NEW_SETUP_THRESHOLD:
                            exit_reason, exit_price = "new_setup", m5_c[i]
                    except Exception:
                        pass

            # Safety SL 2x ATR
            if not exit_reason:
                if entry_dir == "BUY" and m5_l[i] <= entry_price - 2.0 * atr:
                    exit_reason, exit_price = "safety_sl", entry_price - 2.0 * atr
                elif entry_dir == "SELL" and m5_h[i] >= entry_price + 2.0 * atr:
                    exit_reason, exit_price = "safety_sl", entry_price + 2.0 * atr

            # Timeout
            if not exit_reason and bars_in_trade >= TIMEOUT_BARS:
                exit_reason, exit_price = "timeout", m5_c[i]

            if exit_reason:
                if entry_dir == "BUY":
                    pnl = (exit_price - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                else:
                    pnl = (entry_price - exit_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                balance += pnl
                peak_balance = max(peak_balance, balance)
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "direction": entry_dir,
                               "entry": entry_price, "exit": round(exit_price, 2),
                               "pnl_usd": round(pnl, 2), "lot": entry_lot,
                               "reason": exit_reason, "bars": bars_in_trade})
                in_trade = False
                bars_in_trade = 0
        else:
            d = asp_arr[i]
            if d != 0:
                atr_val = atr_arr[i]
                if np.isnan(atr_val) or atr_val <= 0 or atr_val < 0.50:
                    continue
                entry_dir = "BUY" if d == 1 else "SELL"
                entry_price = m5_c[i]
                sl_dist = atr_val * SL_ATR_MULT
                tp_dist = atr_val * TP_ATR_MULT
                sl_price = entry_price - sl_dist if entry_dir == "BUY" else entry_price + sl_dist
                tp_price = entry_price + tp_dist if entry_dir == "BUY" else entry_price - tp_dist
                entry_lot = calc_lot(balance, peak_balance)
                in_trade = True
                entry_idx = i
                bars_in_trade = 0

    if in_trade:
        if entry_dir == "BUY":
            pnl = (m5_c[-1] - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        else:
            pnl = (entry_price - m5_c[-1]) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        balance += pnl
        peak_balance = max(peak_balance, balance)
        trades.append({"entry_idx": entry_idx, "exit_idx": n-1, "direction": entry_dir,
                       "entry": entry_price, "exit": round(m5_c[-1], 2),
                       "pnl_usd": round(pnl, 2), "lot": entry_lot,
                       "reason": "eoy_close", "bars": bars_in_trade})

    return pd.DataFrame(trades), balance


def print_result(label, df, balance, elapsed):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    if len(df) == 0:
        print("  No trades!")
        return
    wins = (df["pnl_usd"] > 0).sum()
    losses = (df["pnl_usd"] <= 0).sum()
    wr = wins / len(df) * 100
    gp = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gl = abs(df.loc[df["pnl_usd"] <= 0, "pnl_usd"].sum())
    pf = gp / gl if gl > 0 else 99.9
    net = df["pnl_usd"].sum()
    ab = df["bars"].mean()
    print(f"  Balance:  ${STARTING_BALANCE:.2f} -> ${balance:,.2f}")
    print(f"  Net PnL:  ${net:,.2f}")
    print(f"  Trades:   {len(df)}")
    print(f"  Win rate: {wr:.1f}%")
    print(f"  PF:       {pf:.2f}")
    print(f"  Avg bars: {ab:.1f}")

    reasons = df.groupby("reason").agg(
        count=("pnl_usd", "size"), wins=("pnl_usd", lambda x: (x > 0).sum()),
        total_pnl=("pnl_usd", "sum"), avg_bars=("bars", "mean")).reset_index()
    reasons["wr"] = reasons["wins"] / reasons["count"] * 100
    print(f"\n  Exit breakdown:")
    for _, r in reasons.iterrows():
        print(f"    {r['reason']:>18}: {int(r['count']):>5} trades, WR={r['wr']:.1f}%, PnL=${r['total_pnl']:>12,.2f}, avg_bars={r['avg_bars']:.1f}")


# ============= MAIN =============
t_start = time.time()

asp_saved = joblib.load(ASP_MODEL_PATH)
asp_model = asp_saved["model"]
label_map = asp_saved["label_map"]
inv_map = {v: k for k, v in label_map.items()}
asp_features_list = list(np.load(ASP_FEATURE_PATH, allow_pickle=True))
exit_model = joblib.load(EXIT_MODEL_PATH)
print(f"Models loaded ({time.time()-t_start:.1f}s)")

client = DukascopyClient()
m1 = client.download_year(YEAR)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
m5 = client.resample_to(m1, 5)
h1 = client.resample_to(m1, 16385)
print(f"Data: M5={len(m5)} H1={len(h1)} ({time.time()-t_start:.1f}s)")

m5 = m5.set_index("time")
h1 = h1.set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]

n = len(m5)
m5_h = m5["high"].values.astype(np.float64)
m5_l = m5["low"].values.astype(np.float64)
m5_c = m5["close"].values.astype(np.float64)
atr_arr = compute_atr(m5_h, m5_l, m5_c, ATR_PERIOD)
m5_index = m5.index

t_f = time.time()
asp_feats = compute_asp_features(m5, h1)
valid_asp = asp_feats[asp_features_list].dropna()
asp_signal = pd.Series(0, index=asp_feats.index, dtype=np.int8)
if len(valid_asp) > 0:
    raw_preds = asp_model.predict(valid_asp[asp_features_list].values)
    preds = np.array([inv_map.get(p, p) for p in raw_preds])
    asp_signal.loc[valid_asp.index] = preds.astype(np.int8)
asp_arr = asp_signal.values
buy_n = (asp_arr == 1).sum()
sell_n = (asp_arr == -1).sum()
print(f"ASP features+predict: {len(asp_feats)} rows, BUY={buy_n} SELL={sell_n} ({time.time()-t_f:.1f}s)")

m5_feats = compute_features(m5, h1)
print(f"M5 features: {len(m5_feats)} rows")
print(f"Setup: {time.time()-t_start:.1f}s total")

START_IDX = max(ATR_PERIOD + 5, 100)

# Mode A: ASP only (SL/TP/timeout)
t_a = time.time()
df_a, bal_a = run_bt(asp_arr, atr_arr, m5_c, m5_h, m5_l, m5_index, m5_feats,
                      exit_model, use_exit_ml=False, start_idx=START_IDX, n=n)
t_a = time.time() - t_a
print_result(f"MODE A: ASP ONLY (SL/TP/timeout) {YEAR} [{t_a:.1f}s]", df_a, bal_a, t_a)

# Mode B: ASP + TrendExhaustion ML exits
t_b = time.time()
df_b, bal_b = run_bt(asp_arr, atr_arr, m5_c, m5_h, m5_l, m5_index, m5_feats,
                      exit_model, use_exit_ml=True, start_idx=START_IDX, n=n)
t_b = time.time() - t_b
print_result(f"MODE B: ASP + TrendExhaustion {YEAR} [{t_b:.1f}s]", df_b, bal_b, t_b)

print(f"\n{'='*70}")
print(f"  COMPARISON")
print(f"{'='*70}")
print(f"{'':>20} {'ASP Only':>14} {'ASP+ExitML':>14} {'Delta':>10}")
print(f"{'-'*20} {'-'*14} {'-'*14} {'-'*10}")
for metric, va, vb, fmt in [
    ("Trades", len(df_a), len(df_b), "d"),
    ("Win rate", (df_a["pnl_usd"]>0).sum()/max(len(df_a),1)*100, (df_b["pnl_usd"]>0).sum()/max(len(df_b),1)*100, ".1f"),
    ("Net PnL", df_a["pnl_usd"].sum(), df_b["pnl_usd"].sum(), ",.0f"),
]:
    d = vb - va
    arrow = "+" if d > 0 else ""
    if "%" in fmt:
        print(f"{metric:>20} {va:>13{fmt}}% {vb:>13{fmt}}% {arrow}{d:>8{fmt}}%")
    elif "$" in fmt:
        print(f"{metric:>20} ${va:>12{fmt}} ${vb:>12{fmt}} {arrow}${d:>9{fmt}}")
    else:
        print(f"{metric:>20} {va:>14{fmt}} {vb:>14{fmt}} {arrow}{d:>10{fmt}}")

print(f"\n  Total wall time: {time.time()-t_start:.1f}s")
