"""
Fast 2025 backtest with LOT_MULTIPLIER=2 to check $5 loss frequency.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features

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
    pa = c > slow
    votes[pa & (slow_slope >= 0)] += 0.5
    votes[~pa & (slow_slope <= 0)] -= 0.5
    lb = 5; n = len(c); ih = np.zeros(n, bool); il = np.zeros(n, bool)
    for i in range(lb, n - lb):
        ih[i] = all(h[i] >= h[i-j] for j in range(1, lb+1)) and all(h[i] >= h[i+j] for j in range(1, lb+1))
        il[i] = all(l[i] <= l[i-j] for j in range(1, lb+1)) and all(l[i] <= l[i+j] for j in range(1, lb+1))
    ss = np.zeros(n)
    for i in range(n):
        hi = np.where(ih[:i+1])[0]; lo = np.where(il[:i+1])[0]
        if len(hi) >= 3 and len(lo) >= 3:
            rh = hi[-3:]; rl = lo[-3:]
            hu = sum(1 for k in range(1, len(rh)) if h[rh[k]] > h[rh[k-1]])
            hd = sum(1 for k in range(1, len(rh)) if h[rh[k]] < h[rh[k-1]])
            lu = sum(1 for k in range(1, len(rl)) if l[rl[k]] > l[rl[k-1]])
            ld = sum(1 for k in range(1, len(rl)) if l[rl[k]] < l[rl[k-1]])
            ss[i] = ((hu - hd) + (lu - ld)) / max(1, (len(rh)-1) + (len(rl)-1))
    t = votes + ss; b = np.zeros(n, np.int8)
    b[t >= 0.75] = 1; b[t <= -0.75] = -1
    return b

LOT_MULT = 5
MAX_EVENT_LOSS = 20.00
print(f"LOT_MULTIPLIER={LOT_MULT}  MAX_EVENT_LOSS_USD={MAX_EVENT_LOSS}", flush=True)

pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
client = DukascopyClient()
m1 = client.download_year(2025)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
m5 = client.resample_to(m1, 5).set_index("time")
h1 = client.resample_to(m1, 16385).set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]
h1 = h1[~h1.index.duplicated(keep="first")]

ft = compute_features(m5, h1)
X = ft[ft.columns.intersection(pred._feature_cols)]
for c in pred._feature_cols:
    if c not in X.columns: X[c] = 0.0
X = X[pred._feature_cols]
Xa = X.reindex(m5.index, method="ffill")
mm = ~Xa.isna().any(axis=1)
md = np.full(len(m5), None, object)
if mm.any():
    vi = np.where(mm)[0]
    probs = pred.model.predict_proba(Xa[mm].values)
    pu = np.array([p[1] for p in probs])
    pd_ = np.array([p[0] for p in probs])
    for k, idx in enumerate(vi):
        if pu[k] >= cfg.ML_CONFIDENCE_THRESHOLD: md[idx] = "BUY"
        elif pd_[k] >= cfg.ML_CONFIDENCE_THRESHOLD: md[idx] = "SELL"

hb = compute_bias_vectorized(h1)
him = h1.index.get_indexer(m5.index, method="ffill")
him = np.clip(him, 0, len(hb) - 1)
ba = np.full(len(him), None, object)
ba[hb[him] == 1] = "BUY"
ba[hb[him] == -1] = "SELL"
m5h = np.where(him > 0, h1["high"].values[him-1], h1["high"].values[him])
m5l = np.where(him > 0, h1["low"].values[him-1], h1["low"].values[him])

all_pnls = []
for i in range(60, len(m5) - 15):
    e = ba[i]
    if e is None: continue
    if md[i] != e: continue
    p = float(m5.iloc[i]["close"])
    h1h = float(m5h[i]); h1l = float(m5l[i])
    if h1h <= h1l: continue
    if e == "BUY" and p <= h1h: continue
    if e == "SELL" and p >= h1l: continue
    ep = p
    n = max(0.01, min(1.0, 20.0 * 200 / ep / 100))
    n_lot = max(0.01, min(n * LOT_MULT, 1.0))
    atr_val = float(m5.iloc[i]["close"]) * 0.002
    if e == "BUY":
        sl = ep - atr_val; tp1 = ep + atr_val*2; tp2 = ep + atr_val*4; tp3 = ep + atr_val*6
    else:
        sl = ep + atr_val; tp1 = ep - atr_val*2; tp2 = ep - atr_val*4; tp3 = ep - atr_val*6
    for j in range(i+1, min(i+48, len(m5)-1)):
        fb = m5.iloc[j]; fp = float(fb["close"]); fh = float(fb["high"]); fl = float(fb["low"])
        prof = (fp - ep) * 100 * n_lot if e == "BUY" else (ep - fp) * 100 * n_lot
        if (e == "BUY" and (fl <= sl or fp >= tp1 or fp >= tp2 or fp >= tp3)) or \
           (e == "SELL" and (fh >= sl or fp <= tp1 or fp <= tp2 or fp <= tp3)):
            all_pnls.append(prof); break
    else:
        fp = float(m5.iloc[min(i+48, len(m5)-1)]["close"])
        all_pnls.append((fp - ep) * 100 * n_lot if e == "BUY" else (ep - fp) * 100 * n_lot)

arr = np.array(all_pnls)
wins = arr[arr > 0]; losses = arr[arr < 0]
losses_old = int((arr < -5).sum())
losses_new = int((arr < -MAX_EVENT_LOSS).sum())
losses2 = int((arr < -2).sum())
max_l = float(abs(arr[arr < 0].min())) if len(losses) > 0 else 0
wr = len(wins)/max(1,len(arr))*100
gp = wins.sum() if len(wins)>0 else 0
gl = abs(losses.sum()) if len(losses)>0 else 0

print(f"\nTrades: {len(arr)}")
print(f"WR: {wr:.1f}%  PF: {gp/max(gl,1e-9):.2f}")
print(f"Losses >= $5 (old limit): {losses_old} ({losses_old/max(1,len(arr))*100:.1f}%)")
print(f"Losses >= ${MAX_EVENT_LOSS:.0f} (new limit): {losses_new} ({losses_new/max(1,len(arr))*100:.1f}%)")
print(f"Losses >= $2: {losses2} ({losses2/max(1,len(arr))*100:.1f}%)")
print(f"Max loss: ${max_l:.2f}")
print(f"Trades exceeding ${MAX_EVENT_LOSS:.0f} that would be stopped: {losses_new}")
print(f"Trades between $5 and ${MAX_EVENT_LOSS:.0f}: {losses_old - losses_new} ({((losses_old-losses_new)/max(1,len(arr))*100):.1f}%)")
