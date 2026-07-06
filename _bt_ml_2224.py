"""
Validate ML entry + ML exit for 2022, 2023, 2024.
Tracks: WR, PF, % hitting $20 limit, ML exit stats, $20 balance survival.
"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features

def bias_vec(h1):
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
    t=v+sw; b=np.zeros(n,np.int8); b[t>=0.75]=1; b[t<=-0.75]=-1; return b

def pnl(ep, xp, d, lot, ntr):
    delta = (xp - ep) if d == "BUY" else (ep - xp)
    return delta * 100 * lot * ntr - 25*lot*ntr*2 - 3*lot*ntr

def run_year(year, pred):
    t0 = time.time()
    client = DukascopyClient()
    m1 = client.download_year(year)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).set_index("time")
    h1 = client.resample_to(m1, 16385).set_index("time")
    m5 = m5[~m5.index.duplicated(keep="first")]; h1 = h1[~h1.index.duplicated(keep="first")]
    print(f"  [{year}] {len(m5)} M5 bars loaded [{time.time()-t0:.0f}s]", flush=True)

    ft = compute_features(m5, h1)
    X = ft[ft.columns.intersection(pred._feature_cols)]
    for c in pred._feature_cols:
        if c not in X.columns: X[c] = 0.0
    X = X[pred._feature_cols].reindex(m5.index, method="ffill")
    mm = ~X.isna().any(axis=1)
    ml_dir = np.full(len(m5), None, object)
    if mm.any():
        vi = np.where(mm)[0]; probs = pred.model.predict_proba(X[mm].values)
        pu = np.array([p[1] for p in probs]); pd_ = np.array([p[0] for p in probs])
        for k, idx in enumerate(vi):
            if pu[k] >= cfg.ML_CONFIDENCE_THRESHOLD: ml_dir[idx] = "BUY"
            elif pd_[k] >= cfg.ML_CONFIDENCE_THRESHOLD: ml_dir[idx] = "SELL"

    hb = bias_vec(h1)
    him = h1.index.get_indexer(m5.index, method="ffill"); him = np.clip(him, 0, len(hb)-1)
    bias = np.full(len(him), None, object); bias[hb[him]==1]="BUY"; bias[hb[him]==-1]="SELL"
    h1h = np.where(him>0, h1["high"].values[him-1], h1["high"].values[him])
    h1l = np.where(him>0, h1["low"].values[him-1], h1["low"].values[him])

    print(f"  [{year}] Running backtest...", flush=True)

    EVENT_LOSS = 20.0
    scen_022 = {"lot": 0.02, "ntr": 3, "label": "LOT=0.02×3 (EQUITYSCALER LOT_MULT=2)"}
    scen_05 = {"lot": 0.05, "ntr": 3, "label": "LOT=0.05×3 (EQUITYSCALER LOT_MULT=5)"}
    scens = [scen_022, scen_05]
    results = {}

    for sc in scens:
        FIXED_LOT = sc["lot"]; FIXED_NTR = sc["ntr"]
        trades = []
        for i in range(60, len(m5)-15):
            e = bias[i]
            if e is None: continue
            if ml_dir[i] is None or ml_dir[i] != e: continue

            p = float(m5.iloc[i]["close"]); hh = float(h1h[i]); ll = float(h1l[i])
            if hh <= ll: continue
            if e == "BUY" and p <= hh: continue
            if e == "SELL" and p >= ll: continue

            score = min((p - hh)/(hh-ll) if e=="BUY" else (ll-p)/(hh-ll), 1.0)
            if score < cfg.MIN_BREAKOUT_SCORE: continue

            atr_val = float(m5.iloc[i]["close"]) * 0.002
            ep = p
            if e == "BUY": sl = ep - atr_val; tp1 = ep+atr_val*2; tp2 = ep+atr_val*4; tp3 = ep+atr_val*6
            else: sl = ep + atr_val; tp1 = ep-atr_val*2; tp2 = ep-atr_val*4; tp3 = ep-atr_val*6

            event_pnl = 0.0; reason = "max_bars"
            for j in range(i+1, min(i+48, len(m5)-1)):
                bar = m5.iloc[j]; fp = float(bar["close"]); fh = float(bar["high"]); fl = float(bar["low"])
                m2m_pnl = pnl(ep, fp, e, FIXED_LOT, FIXED_NTR)

                # ML exit: opposite direction signal
                if ml_dir[j] is not None and ml_dir[j] != e:
                    event_pnl = pnl(ep, fp, e, FIXED_LOT, FIXED_NTR); reason = "ml_reversal"; break

                if m2m_pnl <= -EVENT_LOSS:
                    event_pnl = m2m_pnl; reason = "ev_loss"; break
                if e == "BUY":
                    if fl <= sl: event_pnl = pnl(ep, sl, e, FIXED_LOT, FIXED_NTR); reason = "sl"; break
                    if fp >= tp3: event_pnl = pnl(ep, tp3, e, FIXED_LOT, FIXED_NTR); reason = "tp3"; break
                    if fp >= tp2: event_pnl = pnl(ep, tp2, e, FIXED_LOT, FIXED_NTR); reason = "tp2"; break
                    if fp >= tp1: event_pnl = pnl(ep, tp1, e, FIXED_LOT, FIXED_NTR); reason = "tp1"; break
                else:
                    if fh >= sl: event_pnl = pnl(ep, sl, e, FIXED_LOT, FIXED_NTR); reason = "sl"; break
                    if fp <= tp3: event_pnl = pnl(ep, tp3, e, FIXED_LOT, FIXED_NTR); reason = "tp3"; break
                    if fp <= tp2: event_pnl = pnl(ep, tp2, e, FIXED_LOT, FIXED_NTR); reason = "tp2"; break
                    if fp <= tp1: event_pnl = pnl(ep, tp1, e, FIXED_LOT, FIXED_NTR); reason = "tp1"; break
            else:
                fp = float(m5.iloc[min(i+48, len(m5)-1)]["close"])
                event_pnl = pnl(ep, fp, e, FIXED_LOT, FIXED_NTR); reason = "expire"

            trades.append({"pnl": event_pnl, "reason": reason})

        pnls = np.array([t["pnl"] for t in trades])
        if len(pnls) == 0:
            results[sc["label"]] = None; continue

        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        wr = len(wins)/len(pnls)*100; gp = wins.sum() if len(wins)>0 else 0; gl = abs(losses.sum()) if len(losses)>0 else 0
        pf = gp/max(gl,1e-9); max_loss = float(abs(pnls.min())) if len(losses)>0 else 0

        ev_loss_count = (pnls <= -EVENT_LOSS).sum()
        sl_count = sum(1 for t in trades if t["reason"] == "sl")
        ml_rev_count = sum(1 for t in trades if t["reason"] == "ml_reversal")
        ml_rev_pnls = np.array([t["pnl"] for t in trades if t["reason"] == "ml_reversal"])
        ml_rev_wr = (ml_rev_pnls > 0).sum() / max(1, len(ml_rev_pnls)) * 100 if len(ml_rev_pnls) > 0 else 0

        # $20 balance survival (sequential events)
        bal = 20.0; min_bal = 20.0; blew_up = False
        for pnl_val in pnls:
            bal += pnl_val; min_bal = min(min_bal, bal)
            if bal < 10: blew_up = True; break

        results[sc["label"]] = {
            "total": len(pnls), "wr": wr, "pf": pf, "net": pnls.sum(),
            "max_loss": max_loss, "ev_loss": ev_loss_count,
            "ev_loss_pct": ev_loss_count/max(1,len(pnls))*100,
            "sl": sl_count, "ml_rev": ml_rev_count,
            "ml_rev_pct": ml_rev_count/max(1,len(pnls))*100,
            "ml_rev_wr": ml_rev_wr,
            "survived": not blew_up, "end_bal": bal, "min_bal": min_bal,
        }
    return results

pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
YEARS = [2022, 2023, 2024]
all_results = {}

t0 = time.time()
for y in YEARS:
    yr_t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  YEAR {y}")
    print(f"{'='*70}")
    r = run_year(y, pred)
    all_results[y] = r
    print(f"  [{y}] done in {time.time()-yr_t0:.0f}s", flush=True)

print(f"\n\n{'='*70}")
print(f"  SUMMARY: ML ENTRY + ML EXIT | EVENT LOSS LIMIT = $20")
print(f"{'='*70}")

for y in YEARS:
    print(f"\n--- {y} ---")
    for label, r in all_results[y].items():
        if r is None:
            print(f"  {label}: NO TRADES"); continue
        print(f"  {label}")
        print(f"    Events:        {r['total']}")
        print(f"    Win Rate:      {r['wr']:.1f}%")
        print(f"    Profit Factor: {r['pf']:.2f}")
        print(f"    Net PnL:       ${r['net']:+.2f}")
        print(f"    Max event loss: ${r['max_loss']:.2f}")
        print(f"    Hit $20 limit: {r['ev_loss']} ({r['ev_loss_pct']:.1f}%)")
        print(f"    Hard SL:       {r['sl']} ({r['sl']/max(1,r['total'])*100:.1f}%)")
        print(f"    ML reversal:   {r['ml_rev']} ({r['ml_rev_pct']:.1f}%)")
        print(f"    ML reversal WR:{r['ml_rev_wr']:.1f}%")
        print(f"    $20 survival:  {'SURVIVED' if r['survived'] else 'BLEW UP'}")
        if r['survived']:
            print(f"    End bal: ${r['end_bal']:.2f}  Min bal: ${r['min_bal']:.2f}")

print(f"\n{'='*70}")
print(f"  TOTAL RUNTIME: {time.time()-t0:.0f}s")
print(f"{'='*70}")
