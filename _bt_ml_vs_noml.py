"""
Compare 2025 backtest WITH vs WITHOUT ML direction filtering.
Same engine, same bias — just toggle ML agreement check.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features

def compute_bias_vectorized(h1):
    c=h1["close"].values.astype(np.float64); h=h1["high"].values.astype(np.float64); l=h1["low"].values.astype(np.float64)
    fast=pd.Series(c).ewm(span=20,adjust=False).mean().values; slow=pd.Series(c).ewm(span=50,adjust=False).mean().values
    fs=np.full(len(c),0.0); ss=np.full(len(c),0.0)
    if len(c)>=6: fs[5:]=fast[5:]-fast[:-5]; ss[5:]=slow[5:]-slow[:-5]
    v=np.zeros(len(c))
    v[(fast>slow)&(fs>0)]+=1; v[(fast<slow)&(fs<0)]-=1
    pa=c>slow; v[pa&(ss>=0)]+=0.5; v[~pa&(ss<=0)]-=0.5
    lb=5; n=len(c); ih=np.zeros(n,bool); il=np.zeros(n,bool)
    for i in range(lb,n-lb):
        ih[i]=all(h[i]>=h[i-j] for j in range(1,lb+1)) and all(h[i]>=h[i+j] for j in range(1,lb+1))
        il[i]=all(l[i]<=l[i-j] for j in range(1,lb+1)) and all(l[i]<=l[i+j] for j in range(1,lb+1))
    sw=np.zeros(n)
    for i in range(n):
        hi=np.where(ih[:i+1])[0]; lo=np.where(il[:i+1])[0]
        if len(hi)>=3 and len(lo)>=3:
            rh=hi[-3:]; rl=lo[-3:]
            hu=sum(1 for k in range(1,len(rh)) if h[rh[k]]>h[rh[k-1]]); hd=sum(1 for k in range(1,len(rh)) if h[rh[k]]<h[rh[k-1]])
            lu=sum(1 for k in range(1,len(rl)) if l[rl[k]]>l[rl[k-1]]); ld=sum(1 for k in range(1,len(rl)) if l[rl[k]]<l[rl[k-1]])
            sw[i]=((hu-hd)+(lu-ld))/max(1,(len(rh)-1)+(len(rl)-1))
    t=v+sw; b=np.zeros(n,np.int8); b[t>=0.75]=1; b[t<=-0.75]=-1
    return b

def run_backtest(use_ml=True):
    pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
    client = DukascopyClient()
    m1 = client.download_year(2025)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]; h1 = h1[~h1.index.duplicated(keep="first")]

    ft = compute_features(m5, h1)
    X = ft[ft.columns.intersection(pred._feature_cols)]
    for c in pred._feature_cols:
        if c not in X.columns: X[c] = 0.0
    X = X[pred._feature_cols]
    Xa = X.reindex(m5.index, method="ffill")
    mm = ~Xa.isna().any(axis=1)
    ml_dir = np.full(len(m5), None, object)
    ml_conf = np.full(len(m5), 0.0)
    if mm.any():
        vi = np.where(mm)[0]
        probs = pred.model.predict_proba(Xa[mm].values)
        pu = np.array([p[1] for p in probs]); pd = np.array([p[0] for p in probs])
        for k, idx in enumerate(vi):
            if pu[k] >= cfg.ML_CONFIDENCE_THRESHOLD:
                ml_dir[idx] = "BUY"; ml_conf[idx] = pu[k]
            elif pd[k] >= cfg.ML_CONFIDENCE_THRESHOLD:
                ml_dir[idx] = "SELL"; ml_conf[idx] = pd[k]
            else:
                ml_conf[idx] = max(pu[k], pd[k])

    hb = compute_bias_vectorized(h1)
    him = h1.index.get_indexer(m5.index, method="ffill"); him = np.clip(him, 0, len(hb)-1)
    ba = np.full(len(him), None, object); ba[hb[him]==1]="BUY"; ba[hb[him]==-1]="SELL"
    m5h = np.where(him>0, h1["high"].values[him-1], h1["high"].values[him])
    m5l = np.where(him>0, h1["low"].values[him-1], h1["low"].values[him])

    bal = 20.0; peak = 20.0
    all_pnls = []; ml_correct = []; ml_total = 0; bias_correct = 0

    for i in range(60, len(m5)-15):
        expected = ba[i]
        if expected is None: continue
        # ML agreement check
        if use_ml and (ml_dir[i] is None or ml_dir[i] != expected): continue

        p = float(m5.iloc[i]["close"]); h1h = float(m5h[i]); h1l = float(m5l[i])
        if h1h <= h1l: continue
        if expected == "BUY" and p <= h1h: continue
        if expected == "SELL" and p >= h1l: continue
        ep = p

        # Track ML accuracy: does the ML prediction match future outcome?
        ml_total += 1
        future_close = float(m5.iloc[min(i+3, len(m5)-1)]["close"])
        actual_dir = "BUY" if future_close > ep else "SELL"
        if ml_dir[i] == actual_dir: ml_correct.append(1)
        else: ml_correct.append(0)

        # Track bias accuracy
        if expected == actual_dir: bias_correct += 1

        n = max(0.01, min(1.0, bal*200/ep/100))
        atr_val = float(m5.iloc[i]["close"])*0.002
        if expected == "BUY": sl=ep-atr_val; tp1=ep+atr_val*2; tp2=ep+atr_val*4; tp3=ep+atr_val*6
        else: sl=ep+atr_val; tp1=ep-atr_val*2; tp2=ep-atr_val*4; tp3=ep-atr_val*6
        for j in range(i+1, min(i+48, len(m5)-1)):
            fb=m5.iloc[j]; fp=float(fb["close"]); fh=float(fb["high"]); fl=float(fb["low"])
            prof=(fp-ep)*100*n if expected=="BUY" else (ep-fp)*100*n
            if (expected=="BUY" and (fl<=sl or fp>=tp1 or fp>=tp2 or fp>=tp3)) or \
               (expected=="SELL" and (fh>=sl or fp<=tp1 or fp<=tp2 or fp<=tp3)):
                all_pnls.append(prof); break
        else:
            fp=float(m5.iloc[min(i+48, len(m5)-1)]["close"])
            all_pnls.append((fp-ep)*100*n if expected=="BUY" else (ep-fp)*100*n)

    arr=np.array(all_pnls); wins=arr[arr>0]; losses=arr[arr<0]
    wr=len(wins)/max(1,len(arr))*100; gp=wins.sum() if len(wins)>0 else 0; gl=abs(losses.sum()) if len(losses)>0 else 0
    ml_acc = np.mean(ml_correct)*100 if ml_correct else 0
    bias_acc = bias_correct/max(1,ml_total)*100
    return {
        "trades": len(arr), "wr": wr, "pf": gp/max(gl,1e-9),
        "max_loss": float(abs(arr[arr<0].min())) if len(losses)>0 else 0,
        "ml_accuracy": ml_acc, "bias_accuracy": bias_acc, "ml_total": ml_total
    }

print("Running WITH ML...", flush=True)
r_ml = run_backtest(use_ml=True)
print("Running WITHOUT ML...", flush=True)
r_no = run_backtest(use_ml=False)

print(f"\n{'='*60}")
print(f"{'Metric':<30} {'WITH ML':>12} {'WITHOUT ML':>12}")
print(f"{'='*60}")
print(f"{'Trades':<30} {r_ml['trades']:>12} {r_no['trades']:>12}")
print(f"{'Win Rate':<30} {r_ml['wr']:>11.1f}% {r_no['wr']:>11.1f}%")
print(f"{'Profit Factor':<30} {r_ml['pf']:>12.2f} {r_no['pf']:>12.2f}")
print(f"{'Max Loss':<30} ${r_ml['max_loss']:>9.2f} ${r_no['max_loss']:>9.2f}")
print(f"{'ML Direction Acc (3 bars)':<30} {r_ml['ml_accuracy']:>11.1f}% {'N/A':>12}")
print(f"{'Bias Direction Acc (3 bars)':<30} {r_ml['bias_accuracy']:>11.1f}% {r_no['bias_accuracy']:>11.1f}%")
