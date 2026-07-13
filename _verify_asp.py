"""Verify ASP model: no look-ahead bias, check signal-label alignment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib, numpy as np
from app.asp_features import compute_asp_features, ASP_FEATURE_COLS
from app.asp_labels import create_asp_labels
from app.dukascopy_client import DukascopyClient

client = DukascopyClient()
m1 = client.download_year(2025)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
m5 = client.resample_to(m1, 5)
h1 = client.resample_to(m1, 16385)
m5 = m5.set_index("time")
h1 = h1.set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]

features = compute_asp_features(m5, h1)
labels = create_asp_labels(m5)

# Load model
saved = joblib.load("models/asp_swing_xgb_m5.joblib")
model = saved["model"]
inv_map = {v: k for k, v in saved["label_map"].items()}

valid = features[ASP_FEATURE_COLS].dropna()
feat_arr = valid[ASP_FEATURE_COLS].values
raw = model.predict(feat_arr)
model_preds = np.array([inv_map[p] for p in raw])

# Ground truth labels at valid indices
valid_labels = labels.loc[valid.index].values

from collections import Counter
combos = Counter()
for l, p in zip(valid_labels, model_preds):
    combos[(l, p)] += 1

print("Label vs Prediction (ground truth vs model output):")
for (l, p), c in sorted(combos.items()):
    l_name = {-1: "SELL", 0: "NEUT", 1: "BUY"}[l]
    p_name = {-1: "SELL", 0: "NEUT", 1: "BUY"}[p]
    print(f"  Label={l_name:4s} Pred={p_name:4s}: {c:6d} ({c/len(valid_labels)*100:.1f}%)")

# Key: of BUY labels, how often does model predict BUY?
buy_mask = valid_labels == 1
sell_mask = valid_labels == -1
n_buy = buy_mask.sum()
n_sell = sell_mask.sum()

buy_pred_buy = (model_preds[buy_mask] == 1).sum()
buy_pred_sell = (model_preds[buy_mask] == -1).sum()
buy_pred_neut = (model_preds[buy_mask] == 0).sum()

sell_pred_buy = (model_preds[sell_mask] == 1).sum()
sell_pred_sell = (model_preds[sell_mask] == -1).sum()
sell_pred_neut = (model_preds[sell_mask] == 0).sum()

print(f"\nOf {n_buy} BUY labels: model predicts BUY={buy_pred_buy} ({buy_pred_buy/n_buy*100:.1f}%), SELL={buy_pred_sell}, NEUTRAL={buy_pred_neut}")
print(f"Of {n_sell} SELL labels: model predicts SELL={sell_pred_sell} ({sell_pred_sell/n_sell*100:.1f}%), BUY={sell_pred_buy}, NEUTRAL={sell_pred_neut}")

# Check: model predicts BUY but label is NEUTRAL (false signals)
pred_buy = model_preds == 1
pred_sell = model_preds == -1
n_pred_buy = pred_buy.sum()
n_pred_sell = pred_sell.sum()

false_buy = (pred_buy & (valid_labels == 0)).sum()
false_sell = (pred_sell & (valid_labels == 0)).sum()

print(f"\nModel predicts BUY {n_pred_buy} times: correct={buy_pred_buy}, false(false_positive)={false_buy} (label=NEUTRAL)")
print(f"Model predicts SELL {n_pred_sell} times: correct={sell_pred_sell}, false(false_positive)={false_sell} (label=NEUTRAL)")

# KEY QUESTION: when model predicts BUY, is the close price near a swing low?
# Check if the entry price at close is actually a good entry
# How many bars between consecutive BUY signals?
buy_indices = np.where(pred_buy)[0]
if len(buy_indices) > 1:
    gaps = np.diff(buy_indices)
    print(f"\nBUY signal gaps: min={gaps.min()}, max={gaps.max()}, mean={gaps.mean():.1f}, median={np.median(gaps):.0f}")
    print(f"  (consecutive BUY signals within 3 bars: {(gaps <= 3).sum()} of {len(gaps)})")

# Are we entering on EVERY bar the model says BUY, even consecutive ones?
print(f"\nIMPORTANT: Backtest enters on EVERY BUY signal bar. Check consecutive signals:")
consec = 0
max_consec = 0
total_consec_trades = 0
for g in gaps:
    if g <= 1:
        consec += 1
    else:
        if consec > 0:
            total_consec_trades += consec
        consec = 0
        max_consec = max(max_consec, consec)
print(f"  Consecutive BUY bars (gap=1): {(gaps == 1).sum()} pairs")
print(f"  This means backtest enters on EVERY signal bar even if already in a trade...")
print(f"  BUT: backtest only enters when NOT in_trade, so it waits for exit first")
