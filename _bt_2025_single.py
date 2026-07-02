import sys, os, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
import numpy as np
import pandas as pd
import time
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import compute_features, FEATURE_COLS

buy_model = joblib.load('models/buy_sltp_xgb.joblib')
sell_model = joblib.load('models/sell_sltp_xgb.joblib')
client = DukascopyClient()

y = 2025
print(f"Downloading {y} data...")
m1 = client.download_year(y)
m1 = m1.sort_values('time').drop_duplicates(subset='time')
print(f"M1 bars: {len(m1)}")

m5 = client.resample_to(m1, 5).set_index('time')
h1 = client.resample_to(m1, 16385).set_index('time')
m5 = m5[~m5.index.duplicated(keep='first')]
h1 = h1[~h1.index.duplicated(keep='first')]
print(f"M5 bars: {len(m5)}, H1 bars: {len(h1)}")

ft = compute_features(m5, h1)
X = ft[FEATURE_COLS].reindex(m5.index, method='ffill')
mask = ~X.isna().any(axis=1)
X_valid = X[mask]

bp = buy_model.predict_proba(X_valid.values)
sp = sell_model.predict_proba(X_valid.values)
buy_probs = np.full(len(m5), 0.5)
sell_probs = np.full(len(m5), 0.5)
idx_v = np.where(mask)[0]
for k, idx in enumerate(idx_v):
    buy_probs[idx] = bp[k][1]
    sell_probs[idx] = sp[k][1]

# Bias calculation
c = h1['close'].values
h_vals = h1['high'].values
l_vals = h1['low'].values
fast = pd.Series(c).ewm(span=20).mean().values
slow = pd.Series(c).ewm(span=50).mean().values
fs = np.full(len(c), 0.0)
ss = np.full(len(c), 0.0)
if len(c) >= 6:
    fs[5:] = fast[5:] - fast[:-5]
    ss[5:] = slow[5:] - slow[:-5]
votes = np.zeros(len(c))
votes[(fast > slow) & (fs > 0)] += 1
votes[(fast < slow) & (fs < 0)] -= 1
pa = c > slow
votes[pa & (ss >= 0)] += 0.5
votes[~pa & (ss <= 0)] -= 0.5

lk = 5
n = len(c)
ih = np.zeros(n, bool)
il = np.zeros(n, bool)
for i in range(lk, n - lk):
    ih[i] = all(h_vals[i] >= h_vals[i - j] for j in range(1, lk + 1)) and all(h_vals[i] >= h_vals[i + j] for j in range(1, lk + 1))
    il[i] = all(l_vals[i] <= l_vals[i - j] for j in range(1, lk + 1)) and all(l_vals[i] <= l_vals[i + j] for j in range(1, lk + 1))
sw = np.zeros(n)
for i in range(n):
    hi = np.where(ih[:i + 1])[0]
    lo = np.where(il[:i + 1])[0]
    if len(hi) >= 3 and len(lo) >= 3:
        rh = hi[-3:]
        rl = lo[-3:]
        hu = sum(1 for _ in range(1, len(rh)) if h_vals[rh[_]] > h_vals[rh[_ - 1]])
        hd = sum(1 for _ in range(1, len(rh)) if h_vals[rh[_]] < h_vals[rh[_ - 1]])
        lu = sum(1 for _ in range(1, len(rl)) if l_vals[rl[_]] > l_vals[rl[_ - 1]])
        ld = sum(1 for _ in range(1, len(rl)) if l_vals[rl[_]] < l_vals[rl[_ - 1]])
        sw[i] = ((hu - hd) + (lu - ld)) / max(1, (len(rh) - 1) + (len(rl) - 1))
tot = votes + sw
h1b = np.zeros(n, np.int8)
h1b[tot >= 0.75] = 1
h1b[tot <= -0.75] = -1

him = h1.index.get_indexer(m5.index, method='ffill')
him = np.clip(him, 0, len(h1b) - 1)
hh_v = np.where(him > 0, h_vals[him - 1], h_vals[him])
hl_v = np.where(him > 0, l_vals[him - 1], l_vals[him])

# Backtest
THRESHOLD = 0.60
ISO_TIMES = []
KLINE_TIMES = []
PCT_TIMES = []
CST_TIMES = []

for cname in ['iso', 'kline', 'pct', 'cst']:
    col = [x for x in m1.columns if cname.lower() in x.lower()]
    if col:
        cname_lower = cname
        if cname == 'cst':
            CST_TIMES = m1[col[0]].values if hasattr(m1[col[0]], 'values') else m1[col[0]]
        elif cname == 'kline':
            KLINE_TIMES = m1[col[0]].values if hasattr(m1[col[0]], 'values') else m1[col[0]]
        elif cname == 'pct':
            PCT_TIMES = m1[col[0]].values if hasattr(m1[col[0]], 'values') else m1[col[0]]
        elif cname == 'iso':
            ISO_TIMES = m1[col[0]].values if hasattr(m1[col[0]], 'values') else m1[col[0]]

bal = 20.0
total = 0
wins = 0
dd = 0
peak = 20.0
t0 = time.time()

for i in range(60, len(m5) - 15):
    bv = h1b[him[i]]
    if bv == 0:
        continue

    e = 'BUY' if bv == 1 else 'SELL'
    p = float(m5.iloc[i]['close'])
    hh = float(hh_v[i])
    hl = float(hl_v[i])
    if hh <= hl:
        continue
    if e == 'BUY' and p <= hh:
        continue
    if e == 'SELL' and p >= hl:
        continue
    if e == 'BUY' and buy_probs[i] < THRESHOLD:
        continue
    if e == 'SELL' and sell_probs[i] < THRESHOLD:
        continue

    ep = p
    total += 1
    atr_v = float(m5.iloc[i]['close']) * 0.002
    if e == 'BUY':
        sl = ep - atr_v
        tp = ep + atr_v * 2
    else:
        sl = ep + atr_v
        tp = ep - atr_v * 2

    lim = min(i + 48, len(m5) - 1)
    for j in range(i + 1, lim):
        fb = m5.iloc[j]
        fp = float(fb['close'])
        fl = float(fb['low'])
        fh = float(fb['high'])
        prof = (fp - ep) * 100 / ep * bal if e == 'BUY' else (ep - fp) * 100 / ep * bal
        sl_hit = (e == 'BUY' and fl <= sl) or (e == 'SELL' and fh >= sl)
        tp_hit = (e == 'BUY' and fh >= tp) or (e == 'SELL' and fl <= tp)
        if sl_hit:
            bal += prof
            break
        if tp_hit:
            bal += prof
            wins += 1
            break
    else:
        fp = float(m5.iloc[lim]['close'])
        prof = (fp - ep) * 100 / ep * bal if e == 'BUY' else (ep - fp) * 100 / ep * bal
        bal += prof
        if prof > 0:
            wins += 1

    peak = max(peak, bal)
    dd = max(dd, peak - bal)

wr = wins / total * 100 if total else 0
pf = wins / max(1, total - wins)
losses = total - wins
el = time.time() - t0

print(f"\n===== 2025 Backtest: SL/TP Model (threshold={THRESHOLD}) =====")
print(f"  Total trades: {total}")
print(f"  Wins: {wins} | Losses: {losses}")
print(f"  Win rate: {wr:.1f}%")
print(f"  Profit factor: {pf:.2f}")
print(f"  PnL: ${bal - 20:+.2f}")
print(f"  Final balance: ${bal:.2f}")
print(f"  Max drawdown: ${dd:.2f}")
print(f"  Time: {el:.0f}s")
