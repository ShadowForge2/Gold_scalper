"""
2025 Backtest: Daily PnL + Chop Score + NO_TRADE analysis.
Reports profit per day, identifies choppy days, and shows how ML handles them.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from app.direction_predictor import DirectionPredictor, compute_features, compute_chop_score
from app.meta_strategy import MetaStrategy
from app.risk_manager import EquityScaler
import config as cfg

CS = 100; LEV = 200
TRADE_MAX_BARS = cfg.PEAK_HARVEST_MAX_HOLD_BARS

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

pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
model = pred.model
print("Model loaded: classes =", model.classes_)
is_3class = (len(model.classes_) >= 3)
print(f"3-class model: {is_3class}")

# Load data
m1_path = os.path.join("C:\\Users\\1`030 G4\\data\\dukascopy", "XAUUSD_M1_2025.parquet")
m1 = pd.read_parquet(m1_path)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
print(f"Loaded {len(m1)} M1 bars")

m5 = m1.copy()
m5 = m5.set_index("time").resample("5min").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "tick_volume": "sum", "spread": "mean", "real_volume": "sum"
}).dropna().reset_index()
h1 = m1.copy()
h1 = h1.set_index("time").resample("1h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last",
    "tick_volume": "sum", "spread": "mean", "real_volume": "sum"
}).dropna().reset_index()

m5 = m5.set_index("time")
h1 = h1.set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]
print(f"M5: {len(m5)} bars, H1: {len(h1)} bars")

# Features
print("Computing features...")
ft = compute_features(m5, h1)
print(f"Features: {len(ft)} rows")

# Chop score
print("Computing chop scores...")
chop = compute_chop_score(m5)
daily_chop = chop.resample("1D").mean()

# Pre-compute ML probabilities
print("Pre-computing ML probabilities...")
X = ft[ft.columns.intersection(pred._feature_cols)]
for c in pred._feature_cols:
    if c not in X.columns:
        X[c] = 0.0
X = X[pred._feature_cols]
X_aligned = X.reindex(m5.index, method="ffill")
ml_mask = ~X_aligned.isna().any(axis=1)
pb_u = np.full(len(m5), np.nan)
pb_d = np.full(len(m5), np.nan)
pb_n = np.full(len(m5), np.nan)
if ml_mask.any():
    valid_idx = np.where(ml_mask)[0]
    probs = model.predict_proba(X_aligned[ml_mask].values)
    classes = list(model.classes_)
    pb_u[valid_idx] = np.array([p[classes.index(1)] for p in probs])
    pb_d[valid_idx] = np.array([p[classes.index(0)] for p in probs])
    if is_3class:
        pb_n[valid_idx] = np.array([p[classes.index(2)] for p in probs])

# ATR + body ratios
print("Computing ATR + body ratios...")
m5_atr = compute_atr_series(m5["high"].values, m5["low"].values, m5["close"].values)
m5_br = compute_body_ratios(m5["open"].values, m5["high"].values, m5["low"].values, m5["close"].values)

# H1 bias
print("Computing H1 bias...")
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

m5_close = m5["close"].values
m5_open = m5["open"].values
m5_high = m5["high"].values
m5_low = m5["low"].values

bal = 20.0; total = 0; wins = 0; dd = 0; peak_bal = 20.0
t0 = time.time()
override_count = 0
override_day = None
total_overrides = 0
exit_reasons = {}

meta = MetaStrategy()
scaler = EquityScaler()
scaler.initialize(bal)

bw = []; aw = []

# Daily tracking
daily_pnl = {}
daily_bar_count = {}  # total bars scanned per day
daily_nt_prob = {}    # avg NO_TRADE prob per day
daily_trades = {}     # number of trades per day
daily_wins = {}       # winning trades per day
daily_chop_record = {} # avg chop per day

for i in range(60, len(m5) - 15):
    expected = m5_bias_arr[i]
    if expected is None:
        continue

    p = float(m5_close[i])
    h1h = float(m5_h1h[i]); h1l = float(m5_h1l[i])
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

    br_val = float(m5_br[i])
    atr_val = float(m5_atr[i])
    bw.append(br_val); aw.append(atr_val)
    if len(bw) > cfg.ADAPTIVE_CONF_WINDOW: bw.pop(0)
    if len(aw) > cfg.ADAPTIVE_CONF_WINDOW: aw.pop(0)
    if cfg.ADAPTIVE_CONFIRMATION_ENABLED:
        if len(bw) >= 10 and len(aw) >= 50:
            p_low = cfg.ADAPTIVE_CONF_P_LOW / 100.0
            p_norm = cfg.ADAPTIVE_CONF_P_NORM / 100.0
            p_high = max(cfg.ADAPTIVE_CONF_P_HIGH, 0)
            l, h = np.percentile(aw, [25, 75])
            if atr_val <= l:
                allowed = br_val >= np.percentile(bw, int(p_low * 100))
            elif atr_val < h:
                allowed = br_val >= np.percentile(bw, int(p_norm * 100))
            else:
                allowed = True
            if not allowed:
                continue

    meta_threshold = meta.current_threshold
    atr_entry_threshold = float(m5_atr[i] * cfg.ATR_MULTIPLIER / range_sz) if range_sz > 0 and m5_atr[i] > 0 else None
    effective_threshold = max(atr_entry_threshold if atr_entry_threshold else 0, meta_threshold)
    if score < effective_threshold:
        continue

    ts_i = m5.index[i]
    day_key = ts_i.date()
    if override_day != day_key:
        override_count = 0
        override_day = day_key

    entry_dir = expected
    was_overridden = False
    if not np.isnan(pb_d[i]):
        if expected == "BUY" and pb_d[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_d[i] > pb_u[i] and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
            entry_dir = "SELL"; was_overridden = True
        elif expected == "SELL" and pb_u[i] >= cfg.ML_BIAS_OVERRIDE_THRESHOLD and pb_u[i] > pb_d[i] and override_count < cfg.ML_OVERRIDE_MAX_PER_SESSION:
            entry_dir = "BUY"; was_overridden = True
        if was_overridden:
            override_count += 1; total_overrides += 1

    if not was_overridden:
        if not np.isnan(pb_u[i]) and not np.isnan(pb_d[i]):
            if expected == "BUY" and pb_u[i] < cfg.ML_CONFIDENCE_THRESHOLD:
                continue
            if expected == "SELL" and pb_d[i] < cfg.ML_CONFIDENCE_THRESHOLD:
                continue

    lot_mult = meta.current_lot_mult
    if cfg.AGGRESSIVE_SIZING_ENABLED:
        if score >= cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD:
            lot_mult *= cfg.AGGRESSIVE_VERY_STRONG_LOT_MULT
        elif score >= cfg.AGGRESSIVE_STRONG_THRESHOLD:
            lot_mult *= cfg.AGGRESSIVE_STRONG_LOT_MULT

    ep = p; total += 1
    n = max(0.01, bal * LEV / ep / CS)
    n = min(n, 1.0)
    scale_lot = scaler.get_lot(bal) * lot_mult
    n = min(n, max(cfg.MIN_LOT, min(scale_lot, cfg.MAX_LOT)))
    mg = ep * CS / LEV
    mx = bal / mg
    if mx < cfg.MIN_LOT:
        total -= 1; continue
    n = min(n, mx)

    # Track daily chop and NO_TRADE prob at entry
    chop_i = float(chop.iloc[i]) if i < len(chop) else 0.5
    nt_i = float(pb_n[i]) if not np.isnan(pb_n[i]) else 0.0

    peak_profit = 0.0
    for j in range(i + 1, min(i + TRADE_MAX_BARS + 1, len(m5) - 1)):
        fp = float(m5_close[j])
        fh = float(m5_high[j])
        fl = float(m5_low[j])
        bars_held = j - i
        diff = (fp - ep) if entry_dir == "BUY" else (ep - fp)
        prof = diff * CS * n

        if entry_dir == "BUY":
            peak_profit = max(peak_profit, float(fh - ep))
        else:
            peak_profit = max(peak_profit, float(ep - fl))

        atr_j = float(m5_atr[j])
        ml_hold = False
        if not np.isnan(pb_u[j]):
            if entry_dir == "BUY" and pb_u[j] >= cfg.ML_HOLD_CONFIDENCE:
                ml_hold = True
            elif entry_dir == "SELL" and pb_d[j] >= cfg.ML_HOLD_CONFIDENCE:
                ml_hold = True

        exit_now = False; exit_reason = None

        if peak_profit > 0:
            trail_trigger = atr_j * cfg.PEAK_HARVEST_TRAIL_TRIGGER
            if peak_profit >= trail_trigger:
                pullback = peak_profit - max(0, diff)
                if pullback / peak_profit > cfg.PEAK_HARVEST_TRAIL_RETRACE:
                    exit_now = True; exit_reason = "trail_stop"

        if not exit_now and not ml_hold and bars_held >= 4:
            streak = 0
            lookback = min(cfg.DIRECTION_LOSS_LOOKBACK, j - i)
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
            c_win = m5_close[start:j + 1]
            o_win = m5_open[start:j + 1]
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
                    if 1.0 - momentum > cfg.PEAK_HARVEST_MOMENTUM_THRESHOLD:
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
            if prof > 0: wins += 1
            exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
            daily_pnl[day_key] = daily_pnl.get(day_key, 0) + prof
            daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
            if prof > 0: daily_wins[day_key] = daily_wins.get(day_key, 0) + 1
            daily_chop_record[day_key] = daily_chop_record.get(day_key, 0) + chop_i
            daily_bar_count[day_key] = daily_bar_count.get(day_key, 0) + 1
            daily_nt_prob[day_key] = daily_nt_prob.get(day_key, 0) + nt_i
            meta.record_trade(prof, abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5)
            break
    else:
        fb = m5.iloc[min(i + TRADE_MAX_BARS, len(m5) - 1)]
        fp = float(fb["close"])
        prof = (fp - ep) * CS * n if entry_dir == "BUY" else (ep - fp) * CS * n
        bal += prof
        exit_reasons["max_hold"] = exit_reasons.get("max_hold", 0) + 1
        if prof > 0: wins += 1
        daily_pnl[day_key] = daily_pnl.get(day_key, 0) + prof
        daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
        if prof > 0: daily_wins[day_key] = daily_wins.get(day_key, 0) + 1
        daily_chop_record[day_key] = daily_chop_record.get(day_key, 0) + chop_i
        daily_bar_count[day_key] = daily_bar_count.get(day_key, 0) + 1
        daily_nt_prob[day_key] = daily_nt_prob.get(day_key, 0) + nt_i
        meta.record_trade(prof, abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5)

    peak_bal = max(peak_bal, bal)
    dd = max(dd, peak_bal - bal)
    bal = max(bal, 0.01)
    meta.update(bal, {"bias": "BULLISH" if h1_bias[h1_idx_map[i]] == 1 else "BEARISH",
                      "strength": abs(h1_bias[h1_idx_map[i]]) if abs(h1_bias[h1_idx_map[i]]) > 0 else 0.5})

elapsed = time.time() - t0
wr = (wins / total * 100) if total else 0
pf = wins / max(1, total - wins)

# Normalize daily averages
for d in daily_chop_record:
    if daily_bar_count.get(d, 0) > 0:
        daily_chop_record[d] /= daily_bar_count[d]
        daily_nt_prob[d] /= daily_bar_count[d]

# Build daily report sorted by date
daily_dates = sorted(daily_pnl.keys())
print()
print("=" * 120)
print(f"  2025 Backtest — Daily PnL + Chop Analysis (start=$20, end=${bal:.2f})")
print("=" * 120)
print(f"{'Date':<14} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'PnL':>10} {'AvgChop':>8} {'AvgNT':>8} {'ChopDay':>8}")
print("-" * 120)

# Identify top 33% choppiest days
chop_values = [daily_chop_record.get(d, 0) for d in daily_dates]
chop_threshold = np.percentile(sorted(chop_values), 67) if chop_values else 1.0

total_pnl = 0
choppy_pnl = 0
non_choppy_pnl = 0
choppy_trades = 0
non_choppy_trades = 0

for d in daily_dates:
    t = daily_trades.get(d, 0)
    w = daily_wins.get(d, 0)
    p = daily_pnl.get(d, 0)
    ac = daily_chop_record.get(d, 0)
    an = daily_nt_prob.get(d, 0)
    wr_d = (w / t * 100) if t > 0 else 0
    is_chop = "*** CHOPPY ***" if ac >= chop_threshold else ""
    print(f"{str(d):<14} {t:>6} {w:>5} {wr_d:>5.0f}% {p:>+10.2f} {ac:>8.3f} {an:>8.3f} {is_chop:>14}")
    total_pnl += p
    if ac >= chop_threshold:
        choppy_pnl += p
        choppy_trades += t
    else:
        non_choppy_pnl += p
        non_choppy_trades += t

print("-" * 120)
print(f"{'TOTAL':<14} {total:>6} {wins:>5} {wr:>5.1f}% {total_pnl:>+10.2f}")
print()
print("=" * 60)
print("  Summary")
print("=" * 60)
print(f"  Final balance:    ${bal:.2f}")
print(f"  Net PnL:          ${bal - 20:>+10.2f}")
print(f"  Total trades:     {total}")
print(f"  Win rate:         {wr:.1f}%")
print(f"  Profit factor:    {pf:.2f}")
print(f"  Max drawdown:     ${dd:.2f}")
print(f"  ML overrides:     {total_overrides}")
print(f"  Exit reasons:     {exit_reasons}")
print(f"  Meta regime:      {meta.regime}")
print(f"  Choppy threshold: {chop_threshold:.3f} (top 33% daily avg chop)")
print(f"  Runtime:          {elapsed:.1f}s")
print()
print("-" * 60)
print("  Choppy Days Analysis")
print("-" * 60)
print(f"  Choppy days:      {len([d for d in daily_dates if daily_chop_record.get(d,0) >= chop_threshold])}")
print(f"  Choppy day PnL:   ${choppy_pnl:>+10.2f} ({choppy_trades} trades)")
print(f"  Normal day PnL:   ${non_choppy_pnl:>+10.2f} ({non_choppy_trades} trades)")
print(f"  Choppy vs Normal: {(choppy_pnl/total_pnl*100) if total_pnl!=0 else 0:.1f}% vs {(non_choppy_pnl/total_pnl*100) if total_pnl!=0 else 0:.1f}%")
print(f"  Avg chop score:   {np.mean(chop_values):.3f}")
print(f"  ML NO_TRADE prob on chop days vs normal:")
chop_nts = [daily_nt_prob.get(d,0) for d in daily_dates if daily_chop_record.get(d,0) >= chop_threshold]
norm_nts = [daily_nt_prob.get(d,0) for d in daily_dates if daily_chop_record.get(d,0) < chop_threshold]
print(f"    Choppy days avg NT prob:  {np.mean(chop_nts):.3f}" if chop_nts else "    (no data)")
print(f"    Normal days avg NT prob:  {np.mean(norm_nts):.3f}" if norm_nts else "    (no data)")
print(f"  Model:{' 3-class (NO_TRADE)' if is_3class else ' 2-class (binary)'}")
