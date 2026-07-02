"""
Train SL/TP-aware model: predicts whether a BUY or SELL trade will hit
2×ATR TP before 1×ATR SL within 48 M5 bars.
"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib

from app.dukascopy_client import DukascopyClient
from app.direction_predictor import compute_features, FEATURE_COLS

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def create_sltp_target(m5, horizon=48):
    """For each M5 bar, determine if BUY or SELL would hit 2×ATR TP before 1×ATR SL.
    Returns: buy_target (1=TP wins, 0=SL wins, NaN=no touch), sell_target (same)"""
    n = len(m5)
    closes = m5["close"].values.astype(np.float64)
    highs = m5["high"].values.astype(np.float64)
    lows = m5["low"].values.astype(np.float64)

    # ATR(14) for each bar
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1)),
        ),
    )
    atr = pd.Series(tr).rolling(14, min_periods=2).mean().values

    buy_target = np.full(n, np.nan)
    sell_target = np.full(n, np.nan)

    for i in range(n - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        ep = closes[i]
        buy_tp = ep + 2 * atr[i]
        buy_sl = ep - atr[i]
        sell_tp = ep - 2 * atr[i]
        sell_sl = ep + atr[i]

        lim = min(i + horizon, n)
        buy_done = False
        sell_done = False
        for j in range(i + 1, lim):
            h = highs[j]
            l = lows[j]
            if not buy_done:
                if h >= buy_tp:
                    buy_target[i] = 1.0
                    buy_done = True
                elif l <= buy_sl:
                    buy_target[i] = 0.0
                    buy_done = True
            if not sell_done:
                if l <= sell_tp:
                    sell_target[i] = 1.0
                    sell_done = True
                elif h >= sell_sl:
                    sell_target[i] = 0.0
                    sell_done = True
            if buy_done and sell_done:
                break
        # if neither touched within horizon, leave as NaN

    return buy_target, sell_target


def load_year_data(year):
    client = DukascopyClient()
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]
    return m5, h1


# Load training data 2007-2021
print("Loading training data (2007-2021)...")
t0 = time.time()
train_parts, h1_parts = [], []
for y in range(2007, 2022):
    m5, h1 = load_year_data(y)
    train_parts.append(m5)
    h1_parts.append(h1)
    print(f"  {y}: {len(m5)} M5 bars")

train_m5 = pd.concat(train_parts).sort_index()
train_m5 = train_m5[~train_m5.index.duplicated(keep="first")]
h1_train = pd.concat(h1_parts).sort_index()
h1_train = h1_train[~h1_train.index.duplicated(keep="first")]
print(f"Total: {len(train_m5)} M5 bars ({time.time()-t0:.0f}s)")

# Features
print("Engineering features...")
ft = compute_features(train_m5, h1_train)
print(f"  {len(ft)} rows, {len(ft.columns)} features")

# SL/TP targets
print("Computing SL/TP targets...")
t0 = time.time()
buy_tgt, sell_tgt = create_sltp_target(train_m5)
n_buy_win = np.sum(buy_tgt == 1)
n_buy_lose = np.sum(buy_tgt == 0)
n_sell_win = np.sum(sell_tgt == 1)
n_sell_lose = np.sum(sell_tgt == 0)
print(f"  BUY: {n_buy_win} wins, {n_buy_lose} losses (WR={n_buy_win/max(1,n_buy_win+n_buy_lose)*100:.1f}%)")
print(f"  SELL: {n_sell_win} wins, {n_sell_lose} losses (WR={n_sell_win/max(1,n_sell_win+n_sell_lose)*100:.1f}%)")
print(f"  {time.time()-t0:.0f}s")


def train_and_test(name, target, ft, train_m5):
    """Train a model for one target and test OOS."""
    data = ft.copy()
    data["target"] = target
    data = data.dropna(subset=["target"])

    mask = ~data.index.duplicated(keep="first")
    data = data[mask]

    X = data[FEATURE_COLS]
    y = data["target"]

    # Train/val split
    split = int(len(X) * 0.85)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    print(f"\n=== {name} ===")
    print(f"Train: {len(X_tr)} rows, Val: {len(X_val)} rows")
    print(f"  Win rate: {y.mean()*100:.1f}%")

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=0.1,
        eval_metric="logloss", early_stopping_rounds=30,
        random_state=42, n_jobs=-1,
        scale_pos_weight=(len(y) - y.sum()) / y.sum() if y.sum() > 0 else 1.0,
    )
    model.fit(X_tr.values, y_tr.values, eval_set=[(X_val.values, y_val.values)], verbose=False)

    # OOS test on 2022-2025
    for ty in [2022, 2023, 2024, 2025]:
        tm5, th1 = load_year_data(ty)
        tft = compute_features(tm5, th1)
        tgt, _ = create_sltp_target(tm5) if "BUY" in name else (None, None)
        if "SELL" in name:
            _, tgt = create_sltp_target(tm5)
        tdata = tft.copy()
        tdata["target"] = tgt
        tdata = tdata.dropna(subset=["target"])
        if len(tdata) == 0:
            print(f"  {ty}: no test data")
            continue
        X_te = tdata[FEATURE_COLS]
        y_te = tdata["target"]
        y_pred = model.predict(X_te.values)
        y_prob = model.predict_proba(X_te.values)
        y_conf = np.max(y_prob, axis=1)
        acc = accuracy_score(y_te, y_pred)

        hc = y_conf >= 0.60
        hc_acc = accuracy_score(y_te[hc], y_pred[hc]) if hc.sum() > 0 else 0

        n_win = int(y_pred.sum())
        n_lose = int(len(y_pred) - y_pred.sum())
        wr_pred = n_win / len(y_pred) * 100
        actual_wr = y_te.mean() * 100

        print(f"  {ty}: Acc={acc:.4f} HighConf={hc_acc:.4f} ({hc.mean()*100:.0f}%) "
              f"PredWR={wr_pred:.1f}% ActualWR={actual_wr:.1f}%")

    return model


print("\n--- Training BUY model ---")
buy_model = train_and_test("BUY_TP", buy_tgt, ft, train_m5)
joblib.dump(buy_model, f"{MODEL_DIR}/buy_sltp_xgb.joblib")
print(f"BUY model saved to {MODEL_DIR}/buy_sltp_xgb.joblib")

print("\n--- Training SELL model ---")
sell_model = train_and_test("SELL_TP", sell_tgt, ft, train_m5)
joblib.dump(sell_model, f"{MODEL_DIR}/sell_sltp_xgb.joblib")
print(f"SELL model saved to {MODEL_DIR}/sell_sltp_xgb.joblib")
