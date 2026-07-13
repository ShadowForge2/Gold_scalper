"""
Train XGBoost H1 trend structure model for XAUUSD.
Predicts H1 trend (UP_TREND / DOWN_TREND / RANGING) looking 12 H1 bars forward.

Uses ATR-based swing detection for labels:
  - 3-bar local extremes + 1.0x ATR filter
  - Trend confirmed by forward 12 H1 bars

Trains on 2007-2021, tests on each subsequent year individually.
"""
import os, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient

TRAIN_END = 2021
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
FORWARD_BARS = 12  # H1 bars forward (12 hours)
SWING_LOOKBACK = 3  # bars on each side for local extreme
ATR_SWING_MULT = 1.0  # min distance between swings = 1.0x ATR
ATR_PERIOD = 14
MODEL_PATH = "models/trend_xgb_h1.joblib"
FEATURE_PATH = "models/trend_xgb_h1_features.npy"

client = DukascopyClient()

# Label mapping
LABEL_MAP = {0: "DOWN_TREND", 1: "UP_TREND", 2: "RANGING"}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

TREND_FEATURE_COLS = [
    # --- M5 base features (35) ---
    "return_1", "return_2", "return_3", "return_5", "return_10", "return_20",
    "rsi_14", "atr_norm", "bb_position", "bb_width", "range_ratio",
    "inside_bar_count", "micro_slope", "hl_ratio", "volume_ratio",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "sma_ratio", "macd", "macd_signal", "macd_hist",
    "h1_pos", "h1_dir", "above_h1h", "below_h1l",
    "volatility_ratio", "close_position", "return_vol_ratio",
    "sweep_low_atr", "sweep_high_atr", "close_vs_range_12",
    "minutes_until_event", "minutes_since_event", "event_impact",
    # --- H1 structure features (10) ---
    "h1_rsi", "h1_macd_hist", "h1_ema_spread",
    "h1_consecutive_up", "h1_consecutive_down",
    "h1_swing_distance", "h1_atr_expanding",
    "m5_h1_alignment", "trend_duration", "h1_structure_break",
]


def compute_atr(highs, lows, closes, period=14):
    """Compute ATR vectorized."""
    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr).rolling(period, min_periods=period).mean()
    return atr


def detect_swings(highs, lows, lookback=SWING_LOOKBACK):
    """Detect swing highs/lows using local extremes."""
    n = len(highs)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        window_l = lows[i - lookback:i + lookback + 1]
        if highs[i] == np.max(window_h):
            swing_high[i] = True
        if lows[i] == np.min(window_l):
            swing_low[i] = True
    return swing_high, swing_low


def filter_swings_by_atr(swings, prices, atr, min_mult=ATR_SWING_MULT):
    """Keep only swings that are >= min_mult * ATR apart from the previous same-type swing."""
    n = len(swings)
    filtered = np.zeros(n, dtype=bool)
    last_price = None
    for i in range(n):
        if not swings[i]:
            continue
        cur = prices[i]
        a = atr[i] if not np.isnan(atr[i]) else 1.0
        if last_price is None:
            filtered[i] = True
            last_price = cur
        else:
            if abs(cur - last_price) >= min_mult * a:
                filtered[i] = True
                last_price = cur
    return filtered


def detect_swings_h1(highs, lows, closes, atr_arr):
    """Full ATR-filtered swing detection on H1 data."""
    raw_highs, raw_lows = detect_swings(highs, lows, lookback=SWING_LOOKBACK)
    filt_highs = filter_swings_by_atr(raw_highs, highs, atr_arr, ATR_SWING_MULT)
    filt_lows = filter_swings_by_atr(raw_lows, lows, atr_arr, ATR_SWING_MULT)
    return filt_highs, filt_lows


def create_trend_labels_h1(h1_df, forward_bars=FORWARD_BARS):
    """Create UP/DOWN/RANGING labels on H1 using ATR-based swing structure.

    For each H1 bar i:
      1. Find the last 2 confirmed swing highs and swing lows at bar i
      2. Look forward up to `forward_bars` H1 bars
      3. Label based on swing structure + forward confirmation
    """
    highs = h1_df["high"].values.astype(np.float64)
    lows = h1_df["low"].values.astype(np.float64)
    closes = h1_df["close"].values.astype(np.float64)
    n = len(closes)

    atr = compute_atr(highs, lows, closes, ATR_PERIOD).values
    atr = np.nan_to_num(atr, nan=np.nanmean(atr))

    swing_highs, swing_lows = detect_swings_h1(highs, lows, closes, atr)

    labels = np.full(n, 2, dtype=np.int8)  # default RANGING

    for i in range(n - forward_bars):
        # Gather swing highs/lows up to bar i (confirmed)
        sh_idx = np.where(swing_highs[:i + 1])[0]
        sl_idx = np.where(swing_lows[:i + 1])[0]

        # Gather swing highs/lows in the forward window
        fwd_sh_idx = np.where(swing_highs[i + 1:i + forward_bars + 1])[0] + (i + 1)
        fwd_sl_idx = np.where(swing_lows[i + 1:i + forward_bars + 1])[0] + (i + 1)

        a = atr[i] if atr[i] > 0 else 1.0

        # Need at least 2 historical swing highs and 2 swing lows
        if len(sh_idx) < 2 or len(sl_idx) < 2:
            continue

        last_sh = highs[sh_idx[-1]]
        prev_sh = highs[sh_idx[-2]]
        last_sl = lows[sl_idx[-1]]
        prev_sl = lows[sl_idx[-2]]

        # Count forward swings
        n_fwd_sh = len(fwd_sh_idx)
        n_fwd_sl = len(fwd_sl_idx)

        # UP_TREND: historical structure ascending + forward confirms
        hh = last_sh > prev_sh + 0.5 * a  # higher high
        hl = last_sl > prev_sl + 0.5 * a  # higher low
        if hh and hl and n_fwd_sh >= 1 and n_fwd_sl >= 1:
            fwd_sh_price = highs[fwd_sh_idx[0]]
            fwd_sl_price = lows[fwd_sl_idx[0]]
            if fwd_sh_price > last_sh and fwd_sl_price > last_sl:
                labels[i] = 1  # UP_TREND
                continue

        # DOWN_TREND: historical structure descending + forward confirms
        lh = last_sh < prev_sh - 0.5 * a  # lower high
        ll = last_sl < prev_sl - 0.5 * a  # lower low
        if lh and ll and n_fwd_sh >= 1 and n_fwd_sl >= 1:
            fwd_sh_price = highs[fwd_sh_idx[0]]
            fwd_sl_price = lows[fwd_sl_idx[0]]
            if fwd_sh_price < last_sh and fwd_sl_price < last_sl:
                labels[i] = 0  # DOWN_TREND
                continue

        # Fallback: use forward price direction as tiebreaker
        future_close = closes[i + forward_bars]
        move = future_close - closes[i]
        if move > 1.5 * a:
            labels[i] = 1  # UP_TREND
        elif move < -1.5 * a:
            labels[i] = 0  # DOWN_TREND
        # else: stays RANGING (2)

    return labels


def compute_m5_features(m5_df, h1_df):
    """Compute M5 base features (same as existing direction_predictor)."""
    from app.direction_predictor import compute_features
    return compute_features(m5_df, h1_df)


def compute_h1_structure_features(m5_df, h1_df):
    """Compute 10 H1 structure features aligned to M5 index."""
    if h1_df is None or len(h1_df) < 60:
        return pd.DataFrame(index=m5_df.index)

    h1 = h1_df.copy()
    if not isinstance(h1.index, pd.DatetimeIndex):
        h1.index = pd.to_datetime(h1.index)
    h1 = h1[~h1.index.duplicated(keep="first")]

    c = h1["close"].values.astype(np.float64)
    h = h1["high"].values.astype(np.float64)
    lo = h1["low"].values.astype(np.float64)
    o = h1["open"].values.astype(np.float64)
    n_h1 = len(c)

    feats = pd.DataFrame(index=h1.index)

    # 1. h1_rsi
    delta = np.diff(c, prepend=np.nan)
    gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(14, min_periods=2).mean()
    loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(14, min_periods=2).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["h1_rsi"] = (100 - (100 / (1 + rs))).fillna(50.0).values

    # 2. h1_macd_hist
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    feats["h1_macd_hist"] = (macd_line - macd_sig).fillna(0.0).values

    # 3. h1_ema_spread
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean()
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean()
    feats["h1_ema_spread"] = ((ema20 - ema50) / np.where(c > 0, c, 1)).fillna(0.0).values

    # 4/5. h1_consecutive_up / h1_consecutive_down
    bull = c > o
    bear = c < o
    cu = np.zeros(n_h1)
    cd = np.zeros(n_h1)
    for i in range(1, n_h1):
        cu[i] = (cu[i - 1] + 1) if bull[i] else 0
        cd[i] = (cd[i - 1] + 1) if bear[i] else 0
    feats["h1_consecutive_up"] = cu
    feats["h1_consecutive_down"] = cd

    # 6. h1_swing_distance — distance from nearest confirmed swing high/low, normalized by ATR
    atr_h1 = compute_atr(h, lo, c, ATR_PERIOD).values
    atr_h1 = np.nan_to_num(atr_h1, nan=1.0)
    swing_h, swing_l = detect_swings(h, lo, lookback=3)
    dist = np.zeros(n_h1)
    for i in range(n_h1):
        sh_pos = np.where(swing_h[:i + 1])[0]
        sl_pos = np.where(swing_l[:i + 1])[0]
        d_up = abs(c[i] - h[sh_pos[-1]]) if len(sh_pos) > 0 else atr_h1[i]
        d_dn = abs(c[i] - lo[sl_pos[-1]]) if len(sl_pos) > 0 else atr_h1[i]
        dist[i] = min(d_up, d_dn) / max(atr_h1[i], 0.01)
    feats["h1_swing_distance"] = dist

    # 7. h1_atr_expanding
    atr_ma = pd.Series(atr_h1).rolling(20, min_periods=2).mean().values
    feats["h1_atr_expanding"] = np.where(atr_ma > 0, atr_h1 / atr_ma, 1.0)

    # 8. trend_duration — number of H1 bars since last direction change
    td = np.zeros(n_h1)
    for i in range(1, n_h1):
        if bull[i]:
            td[i] = td[i - 1] + 1 if (i > 0 and bull[i - 1]) else 1
        elif bear[i]:
            td[i] = td[i - 1] + 1 if (i > 0 and bear[i - 1]) else 1
        else:
            td[i] = 0
    feats["trend_duration"] = td

    # 9. h1_structure_break — did current close break above last swing high or below last swing low
    sb = np.zeros(n_h1)
    for i in range(3, n_h1):
        sh_pos = np.where(swing_h[:i])[0]
        sl_pos = np.where(swing_l[:i])[0]
        if len(sh_pos) > 0 and c[i] > h[sh_pos[-1]]:
            sb[i] = 1.0  # broke above
        elif len(sl_pos) > 0 and c[i] < lo[sl_pos[-1]]:
            sb[i] = -1.0  # broke below
    feats["h1_structure_break"] = sb

    # Align to M5 index (forward fill)
    m5_idx = m5_df.index if isinstance(m5_df.index, pd.DatetimeIndex) else pd.to_datetime(m5_df.index)
    aligned = feats.reindex(m5_idx, method="ffill")
    return aligned


def compute_trend_features(m5_df, h1_df):
    """Combine M5 base features + H1 structure features."""
    m5_feats = compute_m5_features(m5_df, h1_df)
    h1_feats = compute_h1_structure_features(m5_df, h1_df)
    if m5_feats is None or len(m5_feats) == 0:
        return None
    if h1_feats is None or len(h1_feats) == 0:
        # Fill missing H1 features with zeros
        for c in ["h1_rsi", "h1_macd_hist", "h1_ema_spread",
                   "h1_consecutive_up", "h1_consecutive_down",
                   "h1_swing_distance", "h1_atr_expanding",
                   "m5_h1_alignment", "trend_duration", "h1_structure_break"]:
            m5_feats[c] = 0.0
        return m5_feats
    combined = m5_feats.join(h1_feats, how="left", rsuffix="_h1_dup")
    # m5_h1_alignment: are M5 and H1 in same direction? Use h1_dir from base features
    if "h1_dir" in combined.columns:
        combined["m5_h1_alignment"] = combined["h1_dir"]
    else:
        combined["m5_h1_alignment"] = 0.0
    # Fill any remaining NaNs
    for c in TREND_FEATURE_COLS:
        if c not in combined.columns:
            combined[c] = 0.0
    return combined


def prepare_dataset(features, target):
    """Align and drop NaNs."""
    data = features.copy()
    data["target"] = target
    data = data.dropna(subset=TREND_FEATURE_COLS + ["target"])
    return data[TREND_FEATURE_COLS], data["target"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_year_df(year):
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    return m5, h1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load training data
    print("Loading training data (2007-2021)...")
    t0 = time.time()
    all_rows = []
    for y in range(2007, TRAIN_END + 1):
        print(f"  Processing {y}...", flush=True)
        t_y = time.time()
        m5, h1 = load_year_df(y)

        if len(m5) == 0 or len(h1) == 0:
            print(f"  {y}: no data, skipping")
            continue

        m5 = m5.set_index("time")
        h1 = h1.set_index("time")
        m5 = m5[~m5.index.duplicated(keep="first")]
        h1 = h1[~h1.index.duplicated(keep="first")]

        print(f"  {y}: {len(m5)} M5 bars, {len(h1)} H1 bars ({time.time()-t_y:.1f}s)")

        # Compute features on M5
        features = compute_trend_features(m5, h1)
        if features is None or len(features) == 0:
            print(f"  {y}: no features")
            continue

        # Create labels on H1
        h1_labels = create_trend_labels_h1(h1, forward_bars=FORWARD_BARS)

        # Align labels to M5 index (forward fill H1 labels to M5)
        h1_label_series = pd.Series(h1_labels, index=h1.index)
        m5_labels = h1_label_series.reindex(m5.index, method="ffill")

        # Build rows
        X, y_vec = prepare_dataset(features, m5_labels)
        if len(X) == 0:
            print(f"  {y}: 0 rows after prep")
            continue

        X["year"] = y
        X["target"] = y_vec.values
        all_rows.append(X)
        print(f"  {y}: {len(X)} rows ({time.time()-t_y:.1f}s)")

    if not all_rows:
        print("No training data!")
        return

    df = pd.concat(all_rows, ignore_index=True)
    print(f"\nTrain total: {len(df)} rows, {time.time()-t0:.1f}s")

    print(f"  DOWN_TREND: {(df['target']==0).sum()}")
    print(f"  UP_TREND:   {(df['target']==1).sum()}")
    print(f"  RANGING:    {(df['target']==2).sum()}")

    # Split train/test
    train_mask = df["year"] < 2022
    X_train = df[TREND_FEATURE_COLS][train_mask].values
    y_train = df["target"][train_mask].values
    print(f"\nTrain: {len(X_train)} rows")

    # Train/validation split within training set
    split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # Train XGBoost
    print("\nTraining XGBoost (3-class: DOWN=0, UP=1, RANGING=2)...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=0.1,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=50,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Train accuracy
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"  Train accuracy: {train_acc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    np.save(FEATURE_PATH, np.array(TREND_FEATURE_COLS))
    print(f"Model saved to {MODEL_PATH}")

    # Feature importance
    imp = pd.DataFrame({"feature": TREND_FEATURE_COLS,
                         "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    print(f"\nTop 15 features:\n{imp.head(15).to_string(index=False)}")

    # Out-of-sample test — load each test year separately
    print("\n--- Out-of-Sample Test ---")
    print(f"{'Year':>6} {'Total':>8} {'Acc':>8} {'UP%':>8} {'DN%':>8} {'RA%':>8}")
    print("-" * 52)

    for ty in TEST_YEARS:
        print(f"\n=== Testing on {ty} ===", flush=True)
        t_y = time.time()
        try:
            test_m5, test_h1 = load_year_df(ty)
            if len(test_m5) == 0 or len(test_h1) == 0:
                print(f"{ty:>6} {'no data':>8}")
                continue
            test_m5 = test_m5.set_index("time")
            test_h1 = test_h1.set_index("time")
            test_m5 = test_m5[~test_m5.index.duplicated(keep="first")]
            test_h1 = test_h1[~test_h1.index.duplicated(keep="first")]

            features = compute_trend_features(test_m5, test_h1)
            h1_labels = create_trend_labels_h1(test_h1, forward_bars=FORWARD_BARS)
            h1_label_series = pd.Series(h1_labels, index=test_h1.index)
            m5_labels = h1_label_series.reindex(test_m5.index, method="ffill")

            X_t, y_t = prepare_dataset(features, m5_labels)
            if len(X_t) == 0:
                print(f"{ty:>6} {'no rows':>8}")
                continue

            y_pred = model.predict(X_t.values)
            acc = accuracy_score(y_t, y_pred)
            n_up = (y_pred == 1).sum() / len(y_pred) * 100
            n_dn = (y_pred == 0).sum() / len(y_pred) * 100
            n_ra = (y_pred == 2).sum() / len(y_pred) * 100
            print(f"{ty:>6} {len(y_t):>8} {acc:>8.4f} {n_up:>7.1f}% {n_dn:>7.1f}% {n_ra:>7.1f}% ({time.time()-t_y:.1f}s)")
        except Exception as e:
            print(f"{ty:>6} ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
