"""
Compare 2-class vs 3-class using same feature set (36 features).
2-class: drops NO_TRADE during training, evaluates on UP/DOWN only
3-class: includes NO_TRADE as class 2
Both trained on 2007-2021, tested 2022-2026.
"""
import os, sys, time
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import (
    compute_features, create_target, FEATURE_COLS,
)

print("Loading training data (2007-2021)...")
t0 = time.time()
client = DukascopyClient()
train_m5_parts, h1_parts = [], []
for y in range(2007, 2022):
    m1 = client.download_year(y).sort_values("time").drop_duplicates(subset="time")
    train_m5_parts.append(client.resample_to(m1, 5).sort_values("time").drop_duplicates(subset="time").set_index("time"))
    h1_parts.append(client.resample_to(m1, 16385).sort_values("time").drop_duplicates(subset="time").set_index("time"))
train_m5 = pd.concat(train_m5_parts).sort_index()
h1_train = pd.concat(h1_parts).sort_index()
train_m5 = train_m5[~train_m5.index.duplicated(keep="first")]
h1_train = h1_train[~h1_train.index.duplicated(keep="first")]
print(f"Train: {len(train_m5)} M5 bars, {time.time()-t0:.1f}s")

print("Features & target...")
t0 = time.time()
train_feats = compute_features(train_m5, h1_train)
train_target = create_target(train_m5)
common = train_feats.index.intersection(train_target.dropna().index)
X_all = train_feats.loc[common, FEATURE_COLS]
y_all = train_target.loc[common]
uc = int((y_all == 1).sum()); dc = int((y_all == 0).sum()); nc = int((y_all == 2).sum())
print(f"  UP={uc} DOWN={dc} NO_TRADE={nc}, {len(X_all)} total, {time.time()-t0:.1f}s")

# Train 2-class: drop NO_TRADE
print("Training 2-class (drop NO_TRADE)...")
t0 = time.time()
mask2 = y_all != 2
X2, y2 = X_all[mask2], y_all[mask2]
sp2 = int(len(X2) * 0.85)
m2 = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
).fit(X2.iloc[:sp2], y2.iloc[:sp2],
      eval_set=[(X2.iloc[sp2:], y2.iloc[sp2:])], verbose=False)
print(f"  Done: {time.time()-t0:.1f}s")

# Train 3-class
print("Training 3-class...")
t0 = time.time()
sp3 = int(len(X_all) * 0.85)
m3 = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, objective="multi:softprob", num_class=3,
).fit(X_all.iloc[:sp3], y_all.iloc[:sp3],
      eval_set=[(X_all.iloc[sp3:], y_all.iloc[sp3:])], verbose=False)
print(f"  Done: {time.time()-t0:.1f}s")

# Test
print(f"\n{'Year':>5} {'Mdl':>6} {'N':>8} {'Acc':>8} {'DirAcc':>8} {'UPpred':>7} {'DNpred':>7} {'NTpred':>7} {'HC%':>6} {'HCAcc':>7}")
for y in range(2022, 2027):
    m1 = client.download_year(y).sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5).sort_values("time").drop_duplicates(subset="time").set_index("time")
    h1 = client.resample_to(m1, 16385).sort_values("time").drop_duplicates(subset="time").set_index("time")
    feats = compute_features(m5, h1)
    target = create_target(m5)
    common = feats.index.intersection(target.dropna().index)
    Xt = feats.loc[common, FEATURE_COLS]
    yt = target.loc[common]

    # 2-class: only UP/DOWN rows
    m2t = yt != 2
    X2t, y2t = Xt[m2t], yt[m2t]
    p2 = m2.predict(X2t)
    acc2 = (p2 == y2t.values).mean()

    # 3-class: all rows
    p3 = m3.predict(Xt)
    acc3 = (p3 == yt.values).mean()
    dm = yt != 2
    dir3 = ((p3 == yt.values) & dm).sum() / max(dm.sum(), 1) if dm.any() else 0
    probs = m3.predict_proba(Xt)
    hc = probs.max(axis=1) >= 0.60
    hca = ((p3[hc] == yt.values[hc]).sum() / max(hc.sum(), 1)) if hc.any() else 0

    u = int((p3==1).sum()); d = int((p3==0).sum()); n = int((p3==2).sum())
    print(f"{y:>5} {'2-cls':>6} {len(y2t):>8} {acc2:>8.4f} {'-':>8} {'-':>7} {'-':>7} {'-':>7} {'-':>6} {'-':>7}")
    print(f"{'':>5} {'3-cls':>6} {len(yt):>8} {acc3:>8.4f} {dir3:>8.4f} {u:>7} {d:>7} {n:>7} {hc.mean()*100:>5.1f}% {hca:>7.4f}")
