"""
2025 ML + Meta + EquityScaler backtest with realistic event loss limit.
Validates survivability with MAX_EVENT_LOSS_USD=20, mult=2.
"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import DirectionPredictor, compute_features
from app.risk_manager import EquityScaler
from app.meta_strategy import MetaStrategy

CS, LEV = 100, 200

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
    return delta * CS * lot * ntr - 25.0 * lot * ntr * 2 - 3.0 * lot * ntr

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
bal, peak = 20.0, 20.0
scaler = EquityScaler(); scaler.initialize(20.0)
meta = MetaStrategy()
trades = []
cur_dir, cur_ep, cur_lot, cur_ntr, cur_atr, cur_event_pnl = None, 0, 0, 0, 0, 0.0
consec_losses, daily_pnl = 0, 0.0
cur_day, cooldown_until = None, None
EVENT_LOSS = cfg.MAX_EVENT_LOSS_USD
DAY_LOSS = cfg.MAX_DAILY_LOSS_USD

for i in range(60, len(m5)-15):
    ts = m5.index[i]
    if cur_day != ts.date():
        cur_day = ts.date(); daily_pnl = 0.0

    # --- EXIT ---
    if cur_dir is not None:
        close_px = float(m5.iloc[i]["close"])
        running_pnl = pnl(cur_ep, close_px, cur_dir, cur_lot, cur_ntr)
        cur_event_pnl = running_pnl

        # Event loss limit check
        if cur_event_pnl <= -EVENT_LOSS:
            bal += cur_event_pnl; daily_pnl += cur_event_pnl
            peak = max(peak, bal)
            scaler.update_peak(bal)
            consec_losses += 1
            cooldown_until = ts + pd.Timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1+consec_losses))
            meta.record_trade(cur_event_pnl, 1.0)
            meta.update(bal, {"strength": 1.0, "bias": cur_dir})
            trades.append({"ts":ts, "pnl":cur_event_pnl, "reason":"ev_loss", "bal":bal})
            cur_dir = None; continue

        # SL/TP check
        bar = m5.iloc[i]; fp = float(bar["close"]); fh = float(bar["high"]); fl = float(bar["low"])
        if cur_dir == "BUY":
            hit = fl <= (cur_ep - cur_atr) or fp >= cur_ep + cur_atr*2 or fp >= cur_ep + cur_atr*4 or fp >= cur_ep + cur_atr*6
        else:
            hit = fh >= (cur_ep + cur_atr) or fp <= cur_ep - cur_atr*2 or fp <= cur_ep - cur_atr*4 or fp <= cur_ep - cur_atr*6
        if hit:
            bal += cur_event_pnl; daily_pnl += cur_event_pnl
            peak = max(peak, bal)
            scaler.update_peak(bal)
            if cur_event_pnl > 0: consec_losses = 0
            else: consec_losses += 1
            cooldown_until = ts + pd.Timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1+consec_losses))
            meta.record_trade(cur_event_pnl, 1.0)
            meta.update(bal, {"strength": 1.0, "bias": cur_dir})
            trades.append({"ts":ts, "pnl":cur_event_pnl, "reason":"sl_tp", "bal":bal})
            cur_dir = None; continue

    # --- ENTRY GATE ---
    if cur_dir is not None: continue
    if bal < cfg.MIN_BALANCE: continue
    if daily_pnl <= -DAY_LOSS: continue
    if cooldown_until is not None and ts < cooldown_until: continue
    cooldown_until = None

    e = bias[i]
    if e is None: continue
    if ml_dir[i] is None or ml_dir[i] != e: continue

    p = float(m5.iloc[i]["close"]); hh = float(h1h[i]); ll = float(h1l[i])
    if hh <= ll: continue
    if e == "BUY" and p <= hh: continue
    if e == "SELL" and p >= ll: continue

    score = min((p - hh) / (hh - ll) if e == "BUY" else (ll - p) / (hh - ll), 1.0)
    if score < cfg.MIN_BREAKOUT_SCORE: continue

    meta.update(bal, {"strength": 1.0, "bias": e})
    atr_val = float(m5.iloc[i]["close"]) * 0.002
    atr_et = atr_val / max(hh-ll, 0.01)
    if score < max(atr_et, meta.current_threshold): continue

    # Sizing
    lot = scaler.get_lot(bal)
    eff_lot = max(cfg.MIN_LOT, min(lot * meta.current_lot_mult, cfg.MAX_LOT))
    ntr = max(1, min(meta.current_trades_per_event, scaler.get_trades_per_event(bal, score)))
    if eff_lot * ntr <= 0: continue

    cur_dir, cur_ep, cur_lot, cur_ntr, cur_atr = e, p, eff_lot, ntr, atr_val
    cur_event_pnl = 0.0

# Final
if cur_dir is not None:
    fp = float(m5.iloc[-1]["close"])
    final_pnl = pnl(cur_ep, fp, cur_dir, cur_lot, cur_ntr)
    bal += final_pnl
    trades.append({"ts":m5.index[-1], "pnl":final_pnl, "reason":"end", "bal":bal})

# Analysis
df = pd.DataFrame(trades)
if len(df) == 0:
    print("NO TRADES"); sys.exit(0)

pnls = df["pnl"].values; wins = pnls[pnls>0]; losses = pnls[pnls<0]
wr = len(wins)/len(pnls)*100
gp = wins.sum() if len(wins)>0 else 0; gl = abs(losses.sum()) if len(losses)>0 else 0
df["cum"] = df["pnl"].cumsum() + 20.0
df["p_cum"] = df["cum"].cummax()
df["dd"] = df["p_cum"] - df["cum"]
min_bal = df["cum"].min()
max_dd = df["dd"].max()

# Early stage (first 50 or all if less)
n_early = min(50, len(df))
early = df.head(n_early)
early_min = early["cum"].min()
early_end = early["cum"].iloc[-1]

print(f"\n{'='*60}")
print(f"2025 ML+META | Mult=2 | EvLoss=${EVENT_LOSS:.0f} | DayLoss=${DAY_LOSS:.0f}")
print(f"{'='*60}")
print(f"{'Trades':<25} {len(df):>8}")
print(f"{'Win Rate':<25} {wr:>7.1f}%")
print(f"{'Profit Factor':<25} {gp/max(gl,1e-9):>8.2f}")
print(f"{'Net PnL':<25} ${bal-20:>+7.2f}")
print(f"{'End Balance':<25} ${bal:>7.2f}")
print(f"{'Max DD':<25} ${max_dd:>7.2f}")
print(f"{'Min Balance':<25} ${min_bal:>7.2f}")
print(f"{'Trades with $20+ loss':<25} {(pnls<-20).sum()} ({(pnls<-20).sum()/len(pnls)*100:.1f}%)")
print(f"{'Trades with $10+ loss':<25} {(pnls<-10).sum()} ({(pnls<-10).sum()/len(pnls)*100:.1f}%)")
print(f"{'Times blew up (<$5)':<25} {(df['cum']<5).sum()}")
print(f"{'Times near zero (<$10)':<25} {(df['cum']<10).sum()}")

print(f"\n--- Early Stage (first {n_early} trades) ---")
print(f"  Min balance:  ${early_min:.2f}")
print(f"  End balance:  ${early_end:.2f}")
print(f"  Recovery >$20: {'YES' if early_end >= 20 else 'NO'}")

print(f"\n--- Worst 10 losses ---")
for i, l in enumerate(sorted([x for x in pnls if x < 0])[:10]):
    print(f"  {i+1}. ${l:.2f}")
