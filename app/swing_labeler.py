"""
Fast vectorized H4 swing detection + M5 window building.
All numpy, no Python loops in hot path.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d


def compute_atr(highs, lows, closes, period=14):
    prev_c = np.empty_like(closes)
    prev_c[0] = closes[0]
    prev_c[1:] = closes[:-1]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    cumsum = np.cumsum(tr)
    atr = np.empty_like(closes)
    atr[:period] = np.nan
    atr[period:] = (cumsum[period:] - cumsum[:-period]) / period
    return atr


def detect_fractals_vec(highs, lows, lookback):
    window = 2 * lookback + 1
    swing_high = highs == maximum_filter1d(highs, size=window, mode='constant', cval=-np.inf)
    swing_low = lows == minimum_filter1d(lows, size=window, mode='constant', cval=np.inf)
    swing_high[:lookback] = False
    swing_high[-lookback:] = False
    swing_low[:lookback] = False
    swing_low[-lookback:] = False
    return swing_high, swing_low


def detect_h4_swings_vec(highs, lows, closes, forward=8,
                          min_move_atr=1.5, lookbacks=(2, 3, 5)):
    n = len(closes)
    atr = compute_atr(highs, lows, closes, 14)
    atr = np.nan_to_num(atr, nan=np.nanmean(atr))
    atr = np.maximum(atr, 0.01)

    cand_high = np.zeros(n, dtype=bool)
    cand_low = np.zeros(n, dtype=bool)
    for lb in lookbacks:
        sh, sl = detect_fractals_vec(highs, lows, lb)
        cand_high |= sh
        cand_low |= sl

    fwd_count = min(forward, n - 1)
    if fwd_count <= 0:
        return cand_high, cand_low, atr

    idx2d = np.arange(fwd_count)[None, :] + np.arange(1, n - fwd_count + 1)[:, None]
    fwd_max_high = np.max(highs[idx2d], axis=1)
    fwd_min_low = np.min(lows[idx2d], axis=1)

    pad_fwd_max = np.full(n, np.nan)
    pad_fwd_min = np.full(n, np.nan)
    pad_fwd_max[1:1+len(fwd_max_high)] = fwd_max_high
    pad_fwd_min[1:1+len(fwd_min_low)] = fwd_min_low

    entry = closes
    adv_high = pad_fwd_max - entry
    fav_high = entry - pad_fwd_min
    adv_low = entry - pad_fwd_min
    fav_low = pad_fwd_max - entry

    adv_high_atr = np.nan_to_num(adv_high / atr, nan=999.0)
    fav_high_atr = np.nan_to_num(fav_high / atr, nan=0.0)
    adv_low_atr = np.nan_to_num(adv_low / atr, nan=999.0)
    fav_low_atr = np.nan_to_num(fav_low / atr, nan=0.0)

    swing_high = cand_high & (fav_high_atr >= min_move_atr) & (adv_high_atr < 1.0)
    swing_low = cand_low & (fav_low_atr >= min_move_atr) & (adv_low_atr < 1.0)

    return swing_high, swing_low, atr


def map_h4_to_m5(m5_times, h4_times, h4_swing_high, h4_swing_low):
    h4_ts = h4_times.values.astype(np.int64) // 10**9
    m5_ts = m5_times.values.astype(np.int64) // 10**9
    h4_idx = np.searchsorted(h4_ts, m5_ts, side='right') - 1
    h4_idx = np.clip(h4_idx, 0, len(h4_times) - 1)

    h4_swing = h4_swing_high | h4_swing_low
    m5_swing = h4_swing[h4_idx]
    return m5_swing


def build_windows_vec(opens, highs, lows, closes, volumes, atr, is_swing, seq_len=30):
    n = len(closes)
    valid = np.arange(seq_len - 1, n)
    swing_valid = valid[is_swing[valid]]
    non_valid = valid[~is_swing[valid]]

    n_samples = min(len(swing_valid), len(non_valid))
    if n_samples == 0:
        return np.array([]), np.array([])

    swing_chosen = np.random.choice(swing_valid, size=n_samples, replace=len(swing_valid) < n_samples)
    non_chosen = np.random.choice(non_valid, size=n_samples, replace=len(non_valid) < n_samples)

    all_idx = np.concatenate([swing_chosen, non_chosen])
    np.random.shuffle(all_idx)

    starts = all_idx - seq_len + 1
    idx_arr = starts[:, None] + np.arange(seq_len)[None, :]

    ref = closes[starts]
    ref = np.maximum(ref, 1e-8)

    X = np.zeros((len(all_idx), seq_len, 6), dtype=np.float32)
    X[:, :, 0] = opens[idx_arr] / ref[:, None]
    X[:, :, 1] = highs[idx_arr] / ref[:, None]
    X[:, :, 2] = lows[idx_arr] / ref[:, None]
    X[:, :, 3] = closes[idx_arr] / ref[:, None]

    vol_ref = np.mean(volumes[idx_arr], axis=1, keepdims=True)
    vol_ref = np.maximum(vol_ref, 1.0)
    X[:, :, 4] = volumes[idx_arr] / vol_ref

    X[:, :, 5] = atr[idx_arr] / ref[:, None]

    Y = is_swing[all_idx].astype(np.float32)

    return X, Y
