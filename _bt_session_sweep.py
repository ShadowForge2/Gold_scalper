"""Per-symbol session sweep — fast, standalone."""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
os.chdir(base)

from app.dukascopy_client import DukascopyClient
from app.asp_features import compute_asp_features
from app.asp_predictor import ASPPredictor
from app.swing_quality_predictor import SwingQualityPredictor
from app.direction_predictor import DirectionPredictor, compute_features, compute_chop_score

SYMBOLS = ["XAUUSD", "US100"]
CFG = {
    "XAUUSD": {"cache_dir":"data/dukascopy", "asp_model":"models/asp_swing_xgb_m5.joblib",
               "asp_features":"models/asp_swing_m5_features.npy", "sq_model":"models/swing_quality_xgb.json",
               "dp_model":"models/direction_xgb_m5_XAUUSD.joblib", "min_atr":0.5, "sl_atr":2.0,
               "tp_atr":1.0, "timeout_bars":10, "spread":0.35},
    "US100": {"cache_dir":"data/dukascopy_us100", "asp_model":"models/asp_swing_xgb_m5_US100.joblib",
              "asp_features":"models/asp_swing_m5_features_US100.npy", "sq_model":"models/swing_quality_xgb_US100.json",
              "dp_model":"models/direction_xgb_m5_US100.joblib", "min_atr":5.0, "sl_atr":2.0,
              "tp_atr":1.0, "timeout_bars":10, "spread":1.5},
}

def precompute(sym, cfg, years):
    client = DukascopyClient(symbol=sym, cache_dir=cfg["cache_dir"])
    all_m5 = []
    for y in years:
        m1 = client.download_year(y)
        if m1 is None or len(m1) == 0: continue
        m1 = m1.sort_values("time").drop_duplicates(subset="time")
        m5 = client.resample_to(m1, 5)
        h1 = client.resample_to(m1, 16385)
        m5 = m5.set_index("time"); h1 = h1.set_index("time")
        m5 = m5[~m5.index.duplicated(keep="first")].sort_index()
        h1 = h1[~h1.index.duplicated(keep="first")].sort_index()
        all_m5.append((m5, h1))
    if not all_m5: return None
    m5 = pd.concat([x[0] for x in all_m5]); h1 = pd.concat([x[1] for x in all_m5])

    asp = ASPPredictor(cfg["asp_model"], cfg["asp_features"])
    sq = SwingQualityPredictor(cfg["sq_model"])
    dp = DirectionPredictor(cfg["dp_model"])

    af = compute_asp_features(m5, h1); n = len(af)
    arr = af.values.astype(np.float32)
    valid = ~(np.isnan(arr).any(axis=1) | np.isinf(arr).any(axis=1))
    clean = np.where(valid[:,None], arr, 0.0)

    sq_proba = sq.model.predict_proba(clean)[:, 1] if sq.ready and sq.model is not None else np.full(n, np.nan)
    sq_proba[~valid] = np.nan

    raw = asp.model.predict(clean)
    asp_dir = np.array([asp.label_map.get(r, 0) for r in raw], dtype=np.int8)
    asp_dir[~valid] = 0
    asp_conf = asp.model.predict_proba(clean).max(axis=1)
    asp_conf[~valid] = 0

    df = compute_features(m5, h1)
    dp_dir = np.zeros(n, dtype=np.int8)
    if df is not None and len(df) == n:
        da = df.values.astype(np.float32)
        dv = ~(np.isnan(da).any(axis=1) | np.isinf(da).any(axis=1))
        dc = np.where(dv[:,None], da, 0.0)
        dp_proba = dp.model.predict_proba(dc)
        dp_cls = list(dp.model.classes_)
        best = dp_proba.argmax(axis=1)
        up_idx = dp_cls.index(1) if 1 in dp_cls else -1
        dn_idx = dp_cls.index(0) if 0 in dp_cls else -1
        for i in range(n):
            if not dv[i]: continue
            bc = dp_cls[best[i]]
            if bc == 1 and dp_proba[i, up_idx] >= 0.6: dp_dir[i] = 1
            elif bc == 0 and dp_proba[i, dn_idx] >= 0.6: dp_dir[i] = -1

    mc = m5["close"].values.astype(np.float64)
    mh = m5["high"].values.astype(np.float64)
    ml = m5["low"].values.astype(np.float64)
    tr = np.maximum(mh-ml, np.maximum(np.abs(mh-np.concatenate([[mc[0]],mc[:-1]])), np.abs(ml-np.concatenate([[mc[0]],mc[:-1]]))))
    atr = pd.Series(tr).rolling(14, min_periods=2).mean().values
    chop_v = compute_chop_score(m5)
    chop = chop_v.values if chop_v is not None else np.full(n, np.nan)

    return {"idx":af.index, "mc":mc, "mh":mh, "ml":ml, "atr":atr,
            "asp_dir":asp_dir, "asp_conf":asp_conf, "sq_prob":sq_proba,
            "dp_dir":dp_dir, "chop":chop}

SESSIONS = {"ASIA": (0,8), "LONDON": (7,17), "NEW_YORK": (12,22)}

def run_bt(data_cache, xau_sess, us_sess):
    def allowed(ts, sym):
        dt = pd.Timestamp(ts)
        h = dt.hour + dt.minute/60.0
        s = xau_sess if sym=="XAUUSD" else us_sess
        for name in s:
            if name in SESSIONS:
                lo, hi = SESSIONS[name]
                if lo <= h < hi: return True
        return False

    bal = 20.0; peak = 20.0; trades = []
    state = {s:{"in":False} for s in SYMBOLS}
    events = []
    for k,d in data_cache.items():
        sym=k.split("_")[0]
        for i in range(19, len(d["idx"])):
            if allowed(d["idx"][i], sym):
                events.append((d["idx"][i], sym, i, k))
    events.sort(key=lambda x:x[0])

    for ts,sym,bi,key in events:
        if bal<=0: break
        d=data_cache[key]; atr=d["atr"][bi]
        atr=atr if np.isfinite(atr) and atr>0 else 1.0
        st=state[sym]
        if st["in"]:
            st["b"]+=1; c=CFG[sym]; hs=ht=False
            if st["d"]=="BUY":
                st["best"]=max(st["best"],d["mh"][bi])
                if st["best"]>=st["ep"]+atr*1.5: st["sl"]=max(st["sl"],st["best"]-atr*2.5)
                if d["ml"][bi]<=st["sl"]: hs=True
                elif d["mh"][bi]>=st["tp"]: ht=True
            else:
                st["best"]=min(st["best"],d["ml"][bi])
                if st["best"]<=st["ep"]-atr*1.5: st["sl"]=min(st["sl"],st["best"]+atr*2.5)
                if d["mh"][bi]>=st["sl"]: hs=True
                elif d["ml"][bi]<=st["tp"]: ht=True
            if hs or ht:
                ep=st["sl"] if hs else st["tp"]
                pnl=(ep-st["ep"])*st["lot"]-c["spread"]*st["lot"]
                if st["d"]=="SELL": pnl=(st["ep"]-ep)*st["lot"]-c["spread"]*st["lot"]
                bal+=pnl; peak=max(peak,bal)
                trades.append({"sym":sym,"pnl":pnl,"bal":bal}); st["in"]=False
                continue
            if st["b"]>=c["timeout_bars"]:
                ep=d["mc"][bi]
                pnl=(ep-st["ep"])*st["lot"]-c["spread"]*st["lot"]
                if st["d"]=="SELL": pnl=(st["ep"]-ep)*st["lot"]-c["spread"]*st["lot"]
                bal+=pnl; peak=max(peak,bal)
                trades.append({"sym":sym,"pnl":pnl,"bal":bal}); st["in"]=False
                continue
        else:
            if np.isfinite(d["chop"][bi]) and d["chop"][bi]>0.7: continue
            if np.isnan(d["sq_prob"][bi]) or d["sq_prob"][bi]<0.4: continue
            ad=d["asp_dir"][bi]; ac=d["asp_conf"][bi]
            if ad==0 or ac<0.65: continue
            dd=d["dp_dir"][bi]
            if dd!=0 and dd!=ad: continue
            if atr<CFG[sym]["min_atr"]: continue
            dr="BUY" if ad==1 else "SELL"
            c=CFG[sym]
            lot=0.02*(bal/20.0)
            ddp=(peak-bal)/peak*100 if peak>0 else 0
            if ddp>15: lot*=0.5
            lot=round(lot/0.01)*0.01; lot=max(0.01,min(lot,10.0))
            ep=d["mc"][bi]; sd=atr*c["sl_atr"]; td=atr*c["tp_atr"]
            st.update({"ep":ep,"sl":ep-sd if dr=="BUY" else ep+sd,"tp":ep+td if dr=="BUY" else ep-td,
                       "lot":lot,"in":True,"d":dr,"b":0,"best":ep})

    df=pd.DataFrame(trades); r={}
    for sym in SYMBOLS:
        sdf=df[df["sym"]==sym]
        if len(sdf)==0: r[sym]={"t":0,"wr":0,"pf":0,"net":0}; continue
        n=len(sdf); w=(sdf["pnl"]>0).sum(); net=sdf["pnl"].sum()
        gp=sdf.loc[sdf["pnl"]>0,"pnl"].sum(); gl=abs(sdf.loc[sdf["pnl"]<=0,"pnl"].sum())
        pf=gp/gl if gl>0 else 99.9; r[sym]={"t":n,"wr":w/n*100,"pf":pf,"net":net}
    n=len(df); w=(df["pnl"]>0).sum(); net=df["pnl"].sum()
    gp=df.loc[df["pnl"]>0,"pnl"].sum(); gl=abs(df.loc[df["pnl"]<=0,"pnl"].sum())
    pf=gp/gl if gl>0 else 99.9; pk=20.0; mdd=0
    for _,rr in df.iterrows(): pk=max(pk,rr["bal"]); mdd=max(mdd,(pk-rr["bal"])/pk*100)
    r["all"]={"t":n,"wr":w/n*100,"pf":pf,"net":net,"dd":mdd,"final":bal}
    return r

print("="*70)
print("  Pre-computing data (2025-2026)...")
DC={}
for sym in SYMBOLS:
    print(f"  {sym}...", end=" ", flush=True)
    d=precompute(sym, CFG[sym], [2025,2026])
    if d: DC[f"{sym}_2025"]=d; print(f"{len(d['idx'])} bars")

# Sweep combos
XAU = [(["LONDON","NEW_YORK"],"LONDON+NY"), (["LONDON"],"LONDON"), (["NEW_YORK"],"NY"),
       (["ASIA","LONDON","NEW_YORK"],"ALL"), (["ASIA"],"ASIA")]
US = [(["NEW_YORK"],"NY"), (["LONDON","NEW_YORK"],"LONDON+NY")]

print(f"\n{'='*70}")
print(f"  SESSION SWEEP")
print(f"{'='*70}")
rows=[]
for xl,xn in XAU:
    for ul,un in US:
        t0=time.time()
        r=run_bt(DC, xl, ul)
        a=r["all"]; x=r["XAUUSD"]; u=r["US100"]
        rows.append((xn,un,x,u,a,time.time()-t0))
        xw=x['wr']; xp=x['pf']; xn_net=x['net']
        uw=u['wr']; up=u['pf']; un_net=u['net']
        print(("  XAU=%-12s US=%-12s  XAU:%5d WR=%4.1f%% PF=%4.2f Net=$%8.0f  "
               "US:%5d WR=%4.1f%% PF=%4.2f Net=$%8.0f") % (
            xn, un, x['t'], xw, xp, xn_net, u['t'], uw, up, un_net))

print(f"\n{'='*70}")
print(f"  SUMMARY TABLE (sorted by combined PF)")
print(f"{'='*70}")
print(f"  {'XAU':<12} {'US100':<12} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Net':>11} {'DD%':>6} {'Final':>11}")
print(f"  {'-'*71}")
rows.sort(key=lambda r: r[4]["pf"], reverse=True)
for xn,un,x,u,a,sec in rows:
    print(f"  {xn:<12} {un:<12} {a['t']:>7} {a['wr']:>5.1f}% {a['pf']:>5.2f} ${a['net']:>8,.0f} {a['dd']:>5.1f}% ${a['final']:>8,.0f}")
print()
