"""Backtest: SL/TP model on 2025 only — NO SL (TP-only exit)."""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import joblib
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import compute_features, FEATURE_COLS

buy_model = joblib.load("models/buy_sltp_xgb.joblib")
sell_model = joblib.load("models/sell_sltp_xgb.joblib")

CS = 100; LEV = 200

def compute_bias_vectorized(h1):
    c = h1["close"].values.astype(np.float64)
    h = h1["high"].values.astype(np.float64)
    l = h1["low"].values.astype(np.float64)
    fast = pd.Series(c).ewm(span=20, adjust=False).mean().values
    slow = pd.Series(c).ewm(span=50, adjust=False).mean().values
    fs = np.full(len(c), 0.0); ss = np.full(len(c), 0.0)
    if len(c) >= 6:
        fs[5:] = fast[5:] - fast[:-5]; ss[5:] = slow[5:] - slow[:-5]
    votes = np.zeros(len(c))
    votes[(fast>slow)&(fs>0)] += 1.0; votes[(fast<slow)&(fs<0)] -= 1.0
    pa = c > slow
    votes[pa & (ss>=0)] += 0.5; votes[~pa & (ss<=0)] -= 0.5
    n = len(c); lb=5; ih=np.zeros(n,bool); il=np.zeros(n,bool)
    for i in range(lb, n-lb):
        ih[i] = all(h[i]>=h[i-j] for j in range(1,lb+1)) and all(h[i]>=h[i+j] for j in range(1,lb+1))
        il[i] = all(l[i]<=l[i-j] for j in range(1,lb+1)) and all(l[i]<=l[i+j] for j in range(1,lb+1))
    sw = np.zeros(n)
    for i in range(n):
        hi = np.where(ih[:i+1])[0]; lo = np.where(il[:i+1])[0]
        if len(hi)>=3 and len(lo)>=3:
            rh=hi[-3:]; rl=lo[-3:]
            hu=sum(1 for _ in range(1,len(rh)) if h[rh[_]]>h[rh[_-1]])
            hd=sum(1 for _ in range(1,len(rh)) if h[rh[_]]<h[rh[_-1]])
            lu=sum(1 for _ in range(1,len(rl)) if l[rl[_]]>l[rl[_-1]])
            ld=sum(1 for _ in range(1,len(rl)) if l[rl[_]]<l[rl[_-1]])
            sw[i]=((hu-hd)+(lu-ld))/max(1,(len(rh)-1)+(len(rl)-1))
    total = votes + sw
    bias = np.zeros(n, dtype=np.int8)
    bias[total >= 0.75] = 1; bias[total <= -0.75] = -1
    return bias

for y in [2025]:
    client = DukascopyClient()
    m1 = client.download_year(y)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]
    h1 = h1[~h1.index.duplicated(keep="first")]

    print(f"  [{y}] Features ({len(m5)} bars)...", end=" ", flush=True)
    ft = compute_features(m5, h1)
    print(f"OK ({len(ft)} rows)")

    X = ft[FEATURE_COLS].reindex(m5.index, method="ffill")
    ml_mask = ~X.isna().any(axis=1)
    buy_probs = np.full(len(m5), 0.0, dtype=np.float64)
    sell_probs = np.full(len(m5), 0.0, dtype=np.float64)
    if ml_mask.any():
        valid_idx = np.where(ml_mask)[0]
        bp = buy_model.predict_proba(X[ml_mask].values)
        sp = sell_model.predict_proba(X[ml_mask].values)
        for k, idx in enumerate(valid_idx):
            buy_probs[idx] = bp[k][1]
            sell_probs[idx] = sp[k][1]

    print(f"  [{y}] Bias compute...", end=" ", flush=True)
    h1_bias = compute_bias_vectorized(h1)
    h1_idx_map = h1.index.get_indexer(m5.index, method="ffill")
    h1_idx_map = np.clip(h1_idx_map, 0, len(h1_bias) - 1)
    m5_bias = h1_bias[h1_idx_map]
    m5_bias_arr = np.full(len(m5_bias), None, dtype=object)
    m5_bias_arr[m5_bias == 1] = "BUY"; m5_bias_arr[m5_bias == -1] = "SELL"
    m5_h1h = h1["high"].values[h1_idx_map]
    m5_h1l = h1["low"].values[h1_idx_map]
    m5_h1h = np.where(h1_idx_map > 0, h1["high"].values[h1_idx_map - 1], m5_h1h)
    m5_h1l = np.where(h1_idx_map > 0, h1["low"].values[h1_idx_map - 1], m5_h1l)
    print("OK")

    CONTRACT_SIZE = 100; LEV = 200
    bal = 20.0; total = 0; wins = 0; dd = 0; peak = 20.0; t0 = time.time()

    for i in range(60, len(m5) - 15):
        expected = m5_bias_arr[i]
        if expected is None: continue
        if expected == "BUY" and buy_probs[i] < 0.60: continue
        if expected == "SELL" and sell_probs[i] < 0.60: continue

        p = float(m5.iloc[i]["close"])
        h1h = float(m5_h1h[i]); h1l = float(m5_h1l[i])
        if h1h <= h1l: continue
        if expected == "BUY" and p <= h1h: continue
        if expected == "SELL" and p >= h1l: continue

        ep = p; total += 1
        n = max(0.01, bal * LEV / ep / CONTRACT_SIZE); n = min(n, 1.0)
        atr_val = float(m5.iloc[i]["close"]) * 0.002

        # NO STOP LOSS — only TP targets
        if expected == "BUY":
            tp1 = ep + atr_val * 2; tp2 = ep + atr_val * 4
        else:
            tp1 = ep - atr_val * 2; tp2 = ep - atr_val * 4

        for j in range(i + 1, min(i + 48, len(m5) - 1)):
            fb = m5.iloc[j]; fp = float(fb["close"]); fh = float(fb["high"]); fl = float(fb["low"])
            prof = (fp - ep) * CONTRACT_SIZE * n if expected == "BUY" else (ep - fp) * CONTRACT_SIZE * n
            if expected == "BUY":
                if fh >= tp2: bal += prof; wins += 1; break
                if fh >= tp1: bal += prof; wins += 1; break
            else:
                if fl <= tp2: bal += prof; wins += 1; break
                if fl <= tp1: bal += prof; wins += 1; break
        else:
            fb = m5.iloc[min(i + 48, len(m5) - 1)]
            fp = float(fb["close"])
            prof = (fp - ep) * CONTRACT_SIZE * n if expected == "BUY" else (ep - fp) * CONTRACT_SIZE * n
            bal += prof
            if prof > 0: wins += 1
        peak = max(peak, bal); dd = max(dd, peak - bal)

    elapsed = time.time() - t0
    wr = (wins / total * 100) if total else 0
    pf = wins / max(1, total - wins)
    print(f"{y} (NO SL): {total:>4} tr WR={wr:>5.1f}% PF={pf:>5.2f} PnL=${bal-20:>+8.2f} DD=${dd:>8.2f} Bal=${bal:>8.2f} [{elapsed:.0f}s]")
