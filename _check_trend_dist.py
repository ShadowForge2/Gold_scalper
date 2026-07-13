"""Quick check of trend model prediction distribution on 2025 data."""
import sys, os
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient
from app.direction_predictor import TrendPredictor, compute_trend_features

TREND_MODEL_PATH = "models/trend_xgb_h1.joblib"
TREND_CONF_THRESHOLD = 0.55

trend_model = TrendPredictor.load(TREND_MODEL_PATH)

client = DukascopyClient()
t0 = time.time()
m1 = client.download_year(2025)
m1 = m1.sort_values("time").drop_duplicates(subset="time")
m5 = client.resample_to(m1, 5)
h1 = client.resample_to(m1, 16385)
print(f"Loaded in {time.time()-t0:.1f}s: M5={len(m5)}, H1={len(h1)}")

m5 = m5.set_index("time")
h1 = h1.set_index("time")
h1_for_feat = h1.reindex(m5.index, method="ffill")
m5_features = compute_trend_features(m5, h1_for_feat)
print(f"Features: {len(m5_features)}")

# Sample every 12th bar (= hourly)
sample_indices = list(range(0, len(m5_features), 12))
print(f"Sampling {len(sample_indices)} points (every 12th M5 = hourly)...")

# Get predictions bar by bar
prob_ups = []
prob_downs = []
prob_rangings = []
for idx in sample_indices:
    row = m5_features.iloc[[idx]]
    pd_val, pu_val, pr_val = trend_model.predict_proba(row)
    prob_downs.append(pd_val)
    prob_ups.append(pu_val)
    prob_rangings.append(pr_val)

prob_up = np.array(prob_ups)
prob_down = np.array(prob_downs)
prob_ranging = np.array(prob_rangings)

n = len(sample)
up_count = (prob_up >= TREND_CONF_THRESHOLD).sum()
down_count = (prob_down >= TREND_CONF_THRESHOLD).sum()
ranging_count = n - up_count - down_count

print(f"\nPrediction distribution (threshold={TREND_CONF_THRESHOLD}):")
print(f"  Total samples: {n}")
print(f"  UP_TREND:    {up_count:5d} ({up_count/n*100:.1f}%)")
print(f"  DOWN_TREND:  {down_count:5d} ({down_count/n*100:.1f}%)")
print(f"  RANGING:     {ranging_count:5d} ({ranging_count/n*100:.1f}%)")

# Probability distribution
print(f"\nProbability stats:")
print(f"  prob_up:    min={prob_up.min():.3f} median={np.median(prob_up):.3f} max={prob_up.max():.3f} mean={prob_up.mean():.3f}")
print(f"  prob_down:  min={prob_down.min():.3f} median={np.median(prob_down):.3f} max={prob_down.max():.3f} mean={prob_down.mean():.3f}")
print(f"  prob_range: min={prob_ranging.min():.3f} median={np.median(prob_ranging):.3f} max={prob_ranging.max():.3f} mean={prob_ranging.mean():.3f}")

# Show distribution at different thresholds
for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    up = (prob_up >= thr).sum()
    down = (prob_down >= thr).sum()
    total_signals = up + down
    print(f"  thr={thr:.2f}: UP={up:4d} DOWN={down:4d} signals={total_signals:4d} ({total_signals/n*100:.1f}%)")
