"""
2025 ML + Meta backtest — EARLY STAGE only.
Fixed EquityScaler sizing at $20 balance (0.01 lot * mult=2 = 0.02/trade).
Validates if $20 event loss limit is survivable.
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

t0 = time.time()
print("Loading...", flush=True)
pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
client = DukascopyClient()
m1 = client.download_year(2025)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
m5 = client.resample_to(m1, 5).set_index("time")
h1 = client.resample_to(m1, 16385).set_index("time")
m5 = m5[~m5.index.duplicated(keep="first")]; h1 = h1[~h1.index.duplicated(keep="first")]

print(f"Features...", flush=True)
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

print(f"Bias...", flush=True)
hb = bias_vec(h1)
him = h1.index.get_indexer(m5.index, method="ffill"); him = np.clip(him, 0, len(hb)-1)
bias = np.full(len(him), None, object); bias[hb[him]==1]="BUY"; bias[hb[him]==-1]="SELL"
h1h = np.where(him>0, h1["high"].values[him-1], h1["high"].values[him])
h1l = np.where(him>0, h1["low"].values[him-1], h1["low"].values[him])

print(f"Backtesting... [{time.time()-t0:.0f}s]", flush=True)

# FIXED early-stage sizing: match EquityScaler at $20 balance
# lot = max(0.01, min(0.01, 20/5000)) = 0.01
# Effective: 0.01 * LOT_MULT=2 = 0.02 per trade
FIXED_LOT = 0.02
FIXED_NTR = 3  # MAX_TRADES_PER_EVENT default
EVENT_LOSS = 20.0

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

    # Simulate event: track running PnL, use ML for exit signals too
    event_pnl = 0.0; reason = "max_bars"
    # Pre-compute ML direction for exit: we need predictions for future bars
    # From entry bar (i), predict on each subsequent bar
    for j in range(i+1, min(i+48, len(m5)-1)):
        bar = m5.iloc[j]; fp = float(bar["close"]); fh = float(bar["high"]); fl = float(bar["low"])
        m2m_pnl = pnl(ep, fp, e, FIXED_LOT, FIXED_NTR)

        # ML exit check: if ML predicts opposite direction, exit
        if ml_dir[j] is not None and ml_dir[j] != e:
            # ML reversed — exit immediately
            event_pnl = pnl(ep, fp, e, FIXED_LOT, FIXED_NTR)
            reason = "ml_reversal"
            break

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
        event_pnl = pnl(ep, fp, e, FIXED_LOT, FIXED_NTR)
        reason = "expire"

    trades.append({"pnl": event_pnl, "reason": reason, "price": p, "atr": atr_val})

# Analysis
df = pd.DataFrame(trades)
pnls = df["pnl"].values
wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
wr = len(wins)/len(pnls)*100
gp = wins.sum() if len(wins)>0 else 0; gl = abs(losses.sum()) if len(losses)>0 else 0

ev_loss_count = (pnls <= -EVENT_LOSS).sum()
sl_count = (df["reason"] == "sl").sum()
ml_rev_count = (df["reason"] == "ml_reversal").sum()
ml_rev_pnls = df[df["reason"] == "ml_reversal"]["pnl"].values
ml_rev_wr = (ml_rev_pnls > 0).sum() / max(1, len(ml_rev_pnls)) * 100
# Simulate initial $20 balance, 1 event at a time with cooldown
# Start at $20, apply event PnL, track survival
bal = 20.0; min_bal = 20.0; blew_up = False; blowup_trade = None
for i, (pnl_val, reason) in enumerate(zip(pnls, df["reason"])):
    bal += pnl_val
    min_bal = min(min_bal, bal)
    if bal < 10:
        blew_up = True; blowup_trade = (i+1, bal, pnl_val, reason, bal - pnl_val); break

print(f"\n{'='*60}")
print(f"2025 EARLY STAGE | LOT=0.02×3 | EvLoss=${EVENT_LOSS}")
print(f"{'='*60}")
print(f"{'Total signals':<30} {len(df):>8}")
print(f"{'Win Rate':<30} {wr:>7.1f}%")
print(f"{'Profit Factor':<30} {gp/max(gl,1e-9):>8.2f}")
print(f"{'Max single event loss':<30} ${float(abs(pnls[pnls<0].min())) if len(losses)>0 else 0:>7.2f}")
print(f"{'Events hitting $20 limit':<30} {ev_loss_count} ({ev_loss_count/len(pnls)*100:.1f}%)")
print(f"{'Events hitting hard SL':<30} {sl_count} ({sl_count/len(pnls)*100:.1f}%)")
print(f"{'ML reversal exits':<30} {ml_rev_count} ({ml_rev_count/len(pnls)*100:.1f}%)")
print(f"{'ML reversal WR':<30} {ml_rev_wr:.1f}%")
print(f"{'Events between $5-$20':<30} {((pnls<-5)&(pnls>-20)).sum()}")

print(f"\n--- $20 BALANCE SURVIVAL ---")
print(f"Start balance: $20.00")
if blew_up:
    print(f" BLEW UP at trade #{blowup_trade[0]}")
    print(f"   Balance before: ${blowup_trade[4]:.2f}")
    print(f"   Event PnL: ${blowup_trade[2]:.2f} ({blowup_trade[3]})")
    print(f"   Balance after: ${blowup_trade[1]:.2f}")
else:
    print(f"✅ SURVIVED all {len(df)} events")
    print(f"End balance: ${bal:.2f}")
    print(f"Min balance: ${min_bal:.2f}")

print(f"\n--- $20 LIMIT HITS (first 10) ---")
hits = df[pnls <= -EVENT_LOSS].head(10)
for i, row in hits.iterrows():
    print(f"  Trade #{i+1}: ${row['pnl']:.2f} (price=${row['price']:.0f} atr=${row['atr']:.2f})")

print(f"\n--- WORST 10 EVENTS ---")
for i, l in enumerate(sorted([x for x in pnls if x < 0])[:10]):
    print(f"  {i+1}. ${l:.2f}")
