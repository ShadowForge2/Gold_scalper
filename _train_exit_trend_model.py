"""
Train XGBoost trend exhaustion (exit) model for XAUUSD.
Predicts: TREND_ALIVE / TREND_EXHAUSTED / NEW_SETUP during simulated trades.

Simulates trades during detected H1 trends:
  - Enter when trend begins (detected by swing structure)
  - Label each bar in the trade based on trend health
  - TREND_ALIVE: trend still strong, keep holding
  - TREND_EXHAUSTED: momentum fading, close trade
  - NEW_SETUP: counter-trend opportunity forming, close trade

Uses 35 base features + 5 trade-state features = 40 total.
Trains on 2007-2021, tests on 2022-2026.
"""
import os, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import FEATURE_COLS, compute_features

TRAIN_END = 2021
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
ATR_PERIOD = 14
SWING_LOOKBACK = 3
MODEL_PATH = "models/exit_trend_xgb_m5.joblib"
FEATURE_PATH = "models/exit_trend_xgb_m5_features.npy"

client = DukascopyClient()

# Trade-state features appended to base features
TRADE_STATE_COLS = [
    "bars_held", "trend_pnl_atr", "trend_peak_atr", "trend_drawdown_pct",
    "trend_momentum_decay",
]
EXIT_TREND_FEATURE_COLS = FEATURE_COLS + TRADE_STATE_COLS

# Label mapping
LABEL_MAP = {0: "TREND_EXHAUSTED", 1: "TREND_ALIVE", 2: "NEW_SETUP"}


def compute_atr(highs, lows, closes, period=14):
    prev_c = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    return pd.Series(tr).rolling(period, min_periods=period).mean()


def detect_swings(highs, lows, lookback=SWING_LOOKBACK):
    n = len(highs)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        if highs[i] == np.max(highs[i - lookback:i + lookback + 1]):
            swing_high[i] = True
        if lows[i] == np.min(lows[i - lookback:i + lookback + 1]):
            swing_low[i] = True
    return swing_high, swing_low


def detect_trend_on_h1(h1_df):
    """Detect trend direction on H1 using swing structure.
    Returns array: 1=UP, -1=DOWN, 0=NEUTRAL per H1 bar.
    """
    highs = h1_df["high"].values.astype(np.float64)
    lows = h1_df["low"].values.astype(np.float64)
    closes = h1_df["close"].values.astype(np.float64)
    n = len(closes)

    atr = compute_atr(highs, lows, closes, ATR_PERIOD).values
    atr = np.nan_to_num(atr, nan=1.0)
    atr = np.where(atr <= 0, 1.0, atr)

    swing_h, swing_l = detect_swings(highs, lows, SWING_LOOKBACK)

    trend = np.zeros(n, dtype=np.int8)

    for i in range(SWING_LOOKBACK + 2, n):
        sh_idx = np.where(swing_h[:i + 1])[0]
        sl_idx = np.where(swing_l[:i + 1])[0]
        if len(sh_idx) < 2 or len(sl_idx) < 2:
            continue

        last_sh = highs[sh_idx[-1]]
        prev_sh = highs[sh_idx[-2]]
        last_sl = lows[sl_idx[-1]]
        prev_sl = lows[sl_idx[-2]]
        a = atr[i]

        hh = last_sh > prev_sh + 0.3 * a
        hl = last_sl > prev_sl + 0.3 * a
        lh = last_sh < prev_sh - 0.3 * a
        ll = last_sl < prev_sl - 0.3 * a

        if hh and hl:
            trend[i] = 1  # UP
        elif lh and ll:
            trend[i] = -1  # DOWN
        else:
            # Use momentum as tiebreaker
            if len(closes) > i >= 20:
                ema_fast = pd.Series(closes[:i + 1]).ewm(span=20, adjust=False).mean().values[-1]
                ema_slow = pd.Series(closes[:i + 1]).ewm(span=50, adjust=False).mean().values[-1]
                if closes[i] > ema_fast > ema_slow:
                    trend[i] = 1
                elif closes[i] < ema_fast < ema_slow:
                    trend[i] = -1

    return trend


def simulate_trades_h1(h1_df, m5_df, trend_arr):
    """Simulate trades on H1 trends, generate per-bar labels for M5 data.

    Trade lifecycle:
      1. Enter when trend changes from 0 to +1/-1 (or from opposite to +1/-1)
      2. Stay in trade while trend remains same direction
      3. Label bars as ALIVE/EXHAUSTED/NEW_SETUP based on trend health

    Labels (applied to M5 bars within each trade):
      - TREND_ALIVE (1): momentum strong, hold
      - TREND_EXHAUSTED (0): momentum fading, counter-candles, drawdown
      - NEW_SETUP (2): trend reversed or new counter-trend signal
    """
    h1_closes = h1_df["close"].values.astype(np.float64)
    h1_highs = h1_df["high"].values.astype(np.float64)
    h1_lows = h1_df["low"].values.astype(np.float64)
    h1_n = len(h1_closes)

    h1_atr = compute_atr(h1_highs, h1_lows, h1_closes, ATR_PERIOD).values
    h1_atr = np.nan_to_num(h1_atr, nan=1.0)

    # Identify trade entries: when trend changes direction
    trades = []
    in_trade = False
    trade_start_h1 = 0
    trade_dir = 0
    for i in range(SWING_LOOKBACK + 3, h1_n):
        if not in_trade:
            if trend_arr[i] != 0 and trend_arr[i] != trend_arr[i - 1]:
                in_trade = True
                trade_start_h1 = i
                trade_dir = trend_arr[i]
        else:
            # Trade ends when trend reverses or goes neutral
            if trend_arr[i] != trade_dir:
                trades.append({
                    "start_h1_idx": trade_start_h1,
                    "end_h1_idx": i,
                    "direction": trade_dir,
                    "bars": i - trade_start_h1,
                })
                in_trade = False
                # Check if new trade starts immediately
                if trend_arr[i] != 0:
                    trade_start_h1 = i
                    trade_dir = trend_arr[i]
                    in_trade = True

    # Close open trade at end
    if in_trade:
        trades.append({
            "start_h1_idx": trade_start_h1,
            "end_h1_idx": h1_n - 1,
            "direction": trade_dir,
            "bars": h1_n - 1 - trade_start_h1,
        })

    print(f"  Simulated {len(trades)} H1 trades")

    # Now map trades to M5 bars and generate labels
    m5_closes = m5_df["close"].values.astype(np.float64)
    m5_highs = m5_df["high"].values.astype(np.float64)
    m5_lows = m5_df["low"].values.astype(np.float64)
    m5_n = len(m5_closes)

    m5_atr = compute_atr(m5_highs, m5_lows, m5_closes, ATR_PERIOD).values
    m5_atr = np.nan_to_num(m5_atr, nan=0.01)
    m5_atr = np.where(m5_atr <= 0, 0.01, m5_atr)

    # Map M5 timestamps to H1 indices
    h1_idx = h1_df.index.get_indexer(m5_df.index, method="ffill")

    rows = []
    for trade in trades:
        h1_start = trade["start_h1_idx"]
        h1_end = trade["end_h1_idx"]
        direction = trade["direction"]

        # Find M5 bars that fall within this trade's H1 range
        start_ts = h1_df.index[h1_start]
        end_ts = h1_df.index[min(h1_end + 1, h1_n - 1)]

        mask = (m5_df.index >= start_ts) & (m5_df.index < end_ts)
        m5_indices = np.where(mask)[0]

        if len(m5_indices) == 0:
            continue

        entry_price = m5_closes[m5_indices[0]]
        peak_pnl = 0.0

        for k, m5_i in enumerate(m5_indices):
            if m5_i >= m5_n:
                continue

            diff = (m5_closes[m5_i] - entry_price) if direction == 1 else (entry_price - m5_closes[m5_i])
            peak_pnl = max(peak_pnl, diff)
            bars_held = k + 1
            atr_val = m5_atr[m5_i]
            atr_safe = max(atr_val, 0.01)

            # --- Compute trade-state features ---
            pnl_atr = diff / atr_safe
            peak_atr = peak_pnl / atr_safe
            drawdown = max(0, peak_pnl - diff) / max(peak_pnl, 0.001)

            # Momentum decay: compare recent momentum to entry momentum
            if m5_i >= 5 and direction == 1:
                recent_move = m5_closes[m5_i] - m5_closes[m5_i - 5]
                entry_move = m5_closes[m5_indices[0] + min(5, len(m5_indices) - 1)] - entry_price
                momentum_decay = 1.0 - min(abs(recent_move) / max(abs(entry_move), 0.001), 2.0)
            elif m5_i >= 5 and direction == -1:
                recent_move = m5_closes[m5_i - 5] - m5_closes[m5_i]
                entry_move = entry_price - m5_closes[m5_indices[0] + min(5, len(m5_indices) - 1)]
                momentum_decay = 1.0 - min(abs(recent_move) / max(abs(entry_move), 0.001), 2.0)
            else:
                momentum_decay = 0.5

            # --- Generate label ---
            # Count counter-candles in last 5 bars
            counter_streak = 0
            for j in range(min(5, k + 1)):
                idx = m5_indices[k - j]
                if direction == 1 and m5_closes[idx] < m5_closes[idx - 1]:
                    counter_streak += 1
                elif direction == -1 and m5_closes[idx] > m5_closes[idx - 1]:
                    counter_streak += 1
                else:
                    break

            # Check H1 trend at this point
            cur_h1_idx = h1_idx[m5_i] if m5_i < len(h1_idx) else h1_start
            cur_h1_idx = np.clip(cur_h1_idx, 0, h1_n - 1)
            cur_trend = trend_arr[cur_h1_idx]

            label = 1  # TREND_ALIVE (default)

            # TREND_EXHAUSTED conditions (relaxed to generate more samples):
            # Multiple weaker conditions that trigger exhaustion
            exhausted_score = 0
            if drawdown > 0.30 and bars_held > 5:
                exhausted_score += 1
            if drawdown > 0.50:
                exhausted_score += 1
            if counter_streak >= 2:
                exhausted_score += 1
            if pnl_atr < -0.5 and bars_held > 5:
                exhausted_score += 1
            if pnl_atr < -1.0:
                exhausted_score += 1
            if momentum_decay > 0.6 and bars_held > 3:
                exhausted_score += 1
            if momentum_decay > 0.8:
                exhausted_score += 1
            # Need 2+ exhaustion signals to label EXHAUSTED
            if exhausted_score >= 2:
                label = 0  # EXHAUSTED

            # NEW_SETUP conditions (relaxed):
            # 1. Trend reversed (H1 trend changed direction)
            # 2. Counter-trend breakout (1 candle is enough)
            if cur_trend != 0 and cur_trend != direction:
                label = 2  # NEW_SETUP
            elif direction == 1 and m5_i >= 2:
                if m5_closes[m5_i] < m5_lows[m5_i - 1]:
                    label = 2  # NEW_SETUP (breakdown)
            elif direction == -1 and m5_i >= 2:
                if m5_closes[m5_i] > m5_highs[m5_i - 1]:
                    label = 2  # NEW_SETUP (breakout)

            rows.append({
                "m5_idx": m5_i,
                "bars_held": bars_held,
                "trend_pnl_atr": round(pnl_atr, 4),
                "trend_peak_atr": round(peak_atr, 4),
                "trend_drawdown_pct": round(drawdown, 4),
                "trend_momentum_decay": round(momentum_decay, 4),
                "target": label,
            })

    return rows


def load_year_df(year):
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    return m5, h1


def process_year(year):
    t0 = time.time()
    m5, h1 = load_year_df(year)
    if len(m5) == 0 or len(h1) == 0:
        return []

    m5 = m5.set_index("time")
    h1 = h1.set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    print(f"  [{year}] Loaded {len(m5)} M5, {len(h1)} H1 ({time.time()-t0:.1f}s)", flush=True)

    # Compute M5 features
    t1 = time.time()
    h1_for_feat = h1.reindex(m5.index, method="ffill")
    m5_feats = compute_features(m5, h1_for_feat)
    print(f"  [{year}] Features ({time.time()-t1:.1f}s)", flush=True)

    if m5_feats is None or len(m5_feats) == 0:
        return []

    # Detect H1 trends
    t1 = time.time()
    h1_trend = detect_trend_on_h1(h1)
    print(f"  [{year}] H1 trend detection ({time.time()-t1:.1f}s)", flush=True)

    # Simulate trades
    t1 = time.time()
    trade_rows = simulate_trades_h1(h1, m5, h1_trend)
    print(f"  [{year}] {len(trade_rows)} trade rows ({time.time()-t1:.1f}s)", flush=True)

    # Build feature+label rows
    rows = []
    for tr in trade_rows:
        m5_i = tr["m5_idx"]
        if m5_i >= len(m5_feats):
            continue
        feat_row = m5_feats.iloc[m5_i].to_dict()

        # Add trade-state features
        feat_row["bars_held"] = tr["bars_held"]
        feat_row["trend_pnl_atr"] = tr["trend_pnl_atr"]
        feat_row["trend_peak_atr"] = tr["trend_peak_atr"]
        feat_row["trend_drawdown_pct"] = tr["trend_drawdown_pct"]
        feat_row["trend_momentum_decay"] = tr["trend_momentum_decay"]
        feat_row["target"] = tr["target"]
        feat_row["year"] = year
        rows.append(feat_row)

    print(f"  [{year}] Done: {len(rows)} rows ({time.time()-t0:.1f}s)", flush=True)
    return rows


def main():
    all_rows = []
    for year in range(2007, TRAIN_END + 1):
        try:
            rows = process_year(year)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  {year}: ERROR - {e}", flush=True)
            import traceback; traceback.print_exc()

    if not all_rows:
        print("No training data!")
        return

    print(f"\nTotal: {len(all_rows)} rows")
    df = pd.DataFrame(all_rows)

    # Fill missing columns
    for c in EXIT_TREND_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    df = df.dropna(subset=EXIT_TREND_FEATURE_COLS + ["target"])
    print(f"After NaN drop: {len(df)}")
    print(f"  TREND_ALIVE(1):    {(df['target']==1).sum()}")
    print(f"  TREND_EXHAUSTED(0): {(df['target']==0).sum()}")
    print(f"  NEW_SETUP(2):      {(df['target']==2).sum()}")

    # Train/test split
    train_mask = df["year"] < 2020
    X_train = df[EXIT_TREND_FEATURE_COLS][train_mask].values
    y_train = df["target"][train_mask].values
    X_val = df[EXIT_TREND_FEATURE_COLS][~train_mask].values
    y_val = df["target"][~train_mask].values
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # Compute class weights for training samples (inverse frequency)
    class_counts = np.bincount(y_train.astype(int), minlength=3)
    total = len(y_train)
    class_weights = {0: total / (3 * max(class_counts[0], 1)),
                     1: total / (3 * max(class_counts[1], 1)),
                     2: total / (3 * max(class_counts[2], 1))}
    sample_weights = np.array([class_weights[int(y)] for y in y_train])
    print(f"Class counts: EXHAUSTED={class_counts[0]}, ALIVE={class_counts[1]}, NEW_SETUP={class_counts[2]}")
    print(f"Class weights: EXHAUSTED={class_weights[0]:.2f}, ALIVE={class_weights[1]:.2f}, NEW_SETUP={class_weights[2]:.2f}")

    # Train XGBoost with class weights
    print("\nTraining XGBoost (3-class: EXHAUSTED=0, ALIVE=1, NEW_SETUP=2)...")
    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=0.1,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=50,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_val, y_val)], verbose=True)

    # Validation metrics
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nVal accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred,
                                target_names=["EXHAUSTED", "ALIVE", "NEW_SETUP"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Feature importance
    imp = pd.DataFrame({"feature": EXIT_TREND_FEATURE_COLS,
                         "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    print(f"\nTop 15 features:\n{imp.head(15).to_string(index=False)}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    np.save(FEATURE_PATH, np.array(EXIT_TREND_FEATURE_COLS))
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
