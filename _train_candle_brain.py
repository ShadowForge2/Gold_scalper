"""
Train CandleBrain Transformer on historical M1 data.

STRATEGY: Regime-Aware Asymmetric Edge (consistent profit, not high win rate).

For every M5 bar we simulate the exact trade we will take live:
  - enter at bar close
  - SL = SL_ATR * ATR below/above entry (1.0R default)
  - TP = TP_ATR * ATR above/below entry (2.0R default)
  - exit at market after MAX_HOLD_BARS if neither hit
  - subtract COST_R (spread + slippage) from the result
This yields a "realized R" in [-1, +2] for the long side and the short side.

Labels (regime-shaped automatically):
  - BUY  when long  realized R >= ENTRY_MIN_R and long  is the better side
  - SELL when short realized R >= ENTRY_MIN_R and short is the better side
  - NONE otherwise  (chop / range -> no edge -> no trade -> no losses)

The model is fed regime features (ADX, volatility regime, squeeze, trend state)
so the single brain learns to switch behavior by market condition.

Objective = profit, not accuracy:
  - per-sample loss is weighted by realized edge, and calling a trade on a bar
    where BOTH sides lose is penalised hardest (chop-trap protection)
  - model selection uses out-of-sample simulated net R minus a drawdown penalty

Usage:
  python _train_candle_brain.py --symbol XAUUSD --epochs 100
  python _train_candle_brain.py --symbol US100 --epochs 100
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg

SEQ_LEN = 25
LABEL_WINDOW = 24          # max hold: 24 M5 bars = 2 hours
ATR_PERIOD = 14
COST_R = 0.05              # spread + slippage in R units

SL_R = float(getattr(cfg, "CANDLE_BRAIN_SL_ATR", 1.0))
TP_R = float(getattr(cfg, "CANDLE_BRAIN_TP_ATR", 2.0))
ENTRY_MIN_R = float(getattr(cfg, "CANDLE_BRAIN_ENTRY_MIN_R", 0.35))
EDGE_MARGIN = float(getattr(cfg, "CANDLE_BRAIN_EDGE_MARGIN", 1.0))
VAL_CONF_THRESH = 0.40     # min softmax prob for reported val trades
DD_PENALTY = 2.0           # model-selection penalty per unit of max drawdown
MIN_VAL_TRADES = 100       # ignore degenerate all-NONE models

SYMBOL_MAP = {
    "XAUUSD": "data/dukascopy",
    "US100": "data/dukascopy_us100",
    "US500": "data/dukascopy_us500",
    "US30": "data/dukascopy_us30",
    "JP225": "data/dukascopy_jp225",
    "DE40": "data/dukascopy_de40",
}


def load_m1_data(symbol: str, start_year: int = None, end_year: int = None) -> pd.DataFrame:
    data_dir = SYMBOL_MAP.get(symbol)
    if data_dir is None:
        raise ValueError(f"Unknown symbol: {symbol}")
    path = Path(data_dir)
    frames = []
    # CRITICAL: filter by symbol prefix. data/dukascopy also contains
    # AUDUSD/GBPUSD/USDJPY parquets — globbing *.parquet polluted the XAUUSD
    # training data (2023-2024) with forex bars and destroyed model edge.
    prefix = symbol + "_"
    for f in sorted(path.glob(f"{prefix}*.parquet")):
        try:
            y = int(str(f.name).split("_")[-1].split(".")[0])
        except Exception:
            y = None
        if y is not None:
            if start_year is not None and y < start_year:
                continue
            if end_year is not None and y > end_year:
                continue
        try:
            df = pd.read_parquet(f)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "tick_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def resample_m5(m1: pd.DataFrame) -> pd.DataFrame:
    idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
    m5 = idx.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum",
    }).dropna()
    return m5


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_bb(close: pd.Series, period: int = 20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / sma.replace(0, 1e-10)
    pos = (close - lower) / (upper - lower).replace(0, 1e-10)
    return pos.clip(0, 1), width


def compute_features(m5: pd.DataFrame) -> pd.DataFrame:
    df = m5.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    rng = (h - l).replace(0, 1e-10)
    body = (c - o).abs()

    df["rsi"] = compute_rsi(c).fillna(50) / 100
    atr = compute_atr(df, ATR_PERIOD).fillna(rng.rolling(14).mean().fillna(rng))
    df["atr_norm"] = (atr / c.replace(0, 1e-10)).fillna(0)
    bb_pos, bb_width = compute_bb(c)
    df["bb_pos"] = bb_pos.fillna(0.5)
    df["bb_width"] = bb_width.fillna(0)
    df["body_ratio"] = (body / rng).fillna(0)
    df["range_ratio"] = (rng / atr.replace(0, 1e-10)).fillna(1).clip(0, 5)
    df["momentum_z"] = (c.pct_change(3) / c.pct_change(3).rolling(20).std().replace(0, 1e-10)).fillna(0).clip(-3, 3)
    df["close_pos"] = ((c - l) / rng).fillna(0.5)
    df["direction"] = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(float)
    df["sweep_high"] = ((h.rolling(12).max() - h) / atr.replace(0, 1e-10)).fillna(0).clip(0, 10)
    df["sweep_low"] = ((l - l.rolling(12).min()) / atr.replace(0, 1e-10)).fillna(0).clip(0, 10)
    # 5-bar linear-regression slope via a pure shift-weighted sum (no per-row
    # pandas apply — the old `rolling().apply` loop was the eval/train bottleneck,
    # freezing low-RAM machines for minutes). For x = [-2,-1,0,1,2] (xbar=0,
    # sxx=10) the slope is sum(x*z)/10 = (2c + c1 - c3 - 2c4)/10.
    w = 5
    slope = (2.0 * c + c.shift(1) - c.shift(3) - 2.0 * c.shift(4)) / 10.0
    df["micro_slope"] = ((slope / c.rolling(w).mean().replace(0, 1e-10)).fillna(0).clip(-0.01, 0.01) * 100)

    short_vol = atr.rolling(3).mean()
    long_vol = atr.rolling(24).mean()
    df["volatility_ratio"] = (short_vol / long_vol.replace(0, 1e-10)).fillna(1).clip(0.2, 3)

    ema20 = c.ewm(span=20).mean()
    ema50 = c.ewm(span=50).mean()
    df["trend_strength"] = ((ema20 - ema50) / atr.replace(0, 1e-10)).fillna(0).clip(-3, 3)

    # Scale-free price representation: distance from the 50-EMA, in ATR units.
    atr_safe = atr.replace(0, 1e-10)
    df["open_norm"] = ((o - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["high_norm"] = ((h - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["low_norm"] = ((l - ema50) / atr_safe).fillna(0).clip(-8, 8)
    df["close_norm"] = ((c - ema50) / atr_safe).fillna(0).clip(-8, 8)

    df["return_1"] = c.pct_change(1).fillna(0).clip(-0.05, 0.05) * 100

    df["volume"] = df["tick_volume"].fillna(0)
    df["volume"] = (df["volume"] / df["volume"].rolling(20).mean().replace(0, 1)).fillna(1).clip(0, 5)

    # ── Regime features ──────────────────────────────────────────────────
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_s = atr.replace(0, 1e-10)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr_s
    minus_di = 100 * minus_dm.rolling(14).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(14).mean()
    df["adx_norm"] = (adx / 60.0).clip(0, 1).fillna(0)

    vol_med = atr.rolling(96).median()
    df["vol_regime"] = (((atr / vol_med.replace(0, 1e-10)) - 0.5).clip(0, 1.5) / 1.5).fillna(0.5)

    bb_avg = bb_width.rolling(96).mean()
    df["squeeze"] = (bb_width < bb_avg * 0.9).astype(float).fillna(0)

    df["regime_trend"] = np.where(
        (ema20 > ema50) & (adx > 20), 1.0,
        np.where((ema20 < ema50) & (adx > 20), -1.0, 0.0),
    ).astype(float)

    return df


def add_h1_context(m5: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
    h1 = idx.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()

    h1_close = h1["close"]
    h1_ema20 = h1_close.ewm(span=20).mean()
    h1_ema50 = h1_close.ewm(span=50).mean()
    h1_trend = ((h1_ema20 - h1_ema50) / h1_close.replace(0, 1e-10)).fillna(0).clip(-3, 3)
    h1_rsi = compute_rsi(h1_close).fillna(50) / 100

    m5["h1_trend"] = h1_trend.reindex(m5.index, method="ffill").fillna(0)
    m5["h1_rsi"] = h1_rsi.reindex(m5.index, method="ffill").fillna(0.5)

    h1_hh = h1["high"].rolling(20).max().reindex(m5.index, method="ffill")
    h1_ll = h1["low"].rolling(20).min().reindex(m5.index, method="ffill")
    h1_range = (h1_hh - h1_ll).replace(0, 1e-10)

    m5["dist_h1_high"] = ((h1_hh - m5["close"]) / h1_range).fillna(0.5).clip(0, 1)
    m5["dist_h1_low"] = ((m5["close"] - h1_ll) / h1_range).fillna(0.5).clip(0, 1)

    return m5


def add_swing_features(m5: pd.DataFrame) -> pd.DataFrame:
    h = m5["high"].values
    l = m5["low"].values
    n = len(m5)

    idx = np.arange(n)
    swing_h = np.zeros(n, dtype=bool)
    swing_l = np.zeros(n, dtype=bool)
    swing_h[1:-1] = (h[1:-1] > h[:-2]) & (h[1:-1] > h[2:])
    swing_l[1:-1] = (l[1:-1] < l[:-2]) & (l[1:-1] < l[2:])

    last_h = np.maximum.accumulate(np.where(swing_h, idx, 0))
    last_l = np.maximum.accumulate(np.where(swing_l, idx, 0))
    since_h = idx - last_h
    since_l = idx - last_l

    m5["bars_since_swing_h"] = np.clip(since_h / 12, 0, 1)
    m5["bars_since_swing_l"] = np.clip(since_l / 12, 0, 1)
    return m5


def add_time_features(m5: pd.DataFrame) -> pd.DataFrame:
    if hasattr(m5.index, "hour"):
        hours = m5.index.hour
        days = m5.index.dayofweek
    else:
        hours = pd.Series(12, index=m5.index)
        days = pd.Series(0, index=m5.index)

    m5["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    m5["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    m5["day_sin"] = np.sin(2 * np.pi * days / 7)

    session = pd.Series(1.0, index=m5.index)
    session[(hours >= 0) & (hours < 8)] = 0.0
    session[(hours >= 8) & (hours < 14)] = 0.5
    session[(hours >= 14) & (hours < 22)] = 1.0
    session[(hours >= 22)] = 0.0
    m5["session_enc"] = session

    return m5


def compute_realized_r(
    m5: pd.DataFrame, atr: pd.Series
) -> tuple:
    """Simulate the SL/TP trade for both sides on every bar.

    Returns (long_r, short_r) arrays of realized R in [-1, +TP_R].
    """
    from numpy.lib.stride_tricks import sliding_window_view

    c = m5["close"].values.astype(np.float64)
    h = m5["high"].values.astype(np.float64)
    l = m5["low"].values.astype(np.float64)
    a = atr.values.astype(np.float64)
    n = len(c)
    w = LABEL_WINDOW

    long_r = np.zeros(n, dtype=np.float64)
    short_r = np.zeros(n, dtype=np.float64)
    if n <= w:
        return long_r, short_r

    win_h = sliding_window_view(h, w + 1)[:, 1:]  # future highs  (n-w, w)
    win_l = sliding_window_view(l, w + 1)[:, 1:]  # future lows   (n-w, w)
    win_c = sliding_window_view(c, w + 1)[:, 1:]  # future closes (n-w, w)

    CHUNK = 200_000
    for s in range(0, n - w, CHUNK):
        t = min(s + CHUNK, n - w)
        entry = c[s:t, None]
        at = a[s:t, None]
        fh = win_h[s:t]
        fl = win_l[s:t]
        fc = win_c[s:t]
        cost = COST_R
        good = at[:, 0] > 0

        # ── Long ──
        tp_up = entry + TP_R * at
        sl_dn = entry - SL_R * at
        hit_tp = fh >= tp_up
        hit_sl = fl <= sl_dn
        first_tp = (hit_tp.cumsum(axis=1) == 1).argmax(axis=1)
        first_sl = (hit_sl.cumsum(axis=1) == 1).argmax(axis=1)
        has_tp = hit_tp.any(axis=1)
        has_sl = hit_sl.any(axis=1)
        win_tp = has_tp & ((~has_sl) | (first_tp <= first_sl))
        win_sl = has_sl & ((~has_tp) | (first_sl < first_tp))
        drift = (fc[:, -1] - entry[:, 0]) / at[:, 0]
        lr = np.where(win_tp, TP_R, np.where(win_sl, -SL_R, drift)) - cost
        long_r[s:t] = np.where(good, lr, 0.0)

        # ── Short ──
        tp_dn = entry - TP_R * at
        sl_up = entry + SL_R * at
        hit_tp = fl <= tp_dn
        hit_sl = fh >= sl_up
        first_tp = (hit_tp.cumsum(axis=1) == 1).argmax(axis=1)
        first_sl = (hit_sl.cumsum(axis=1) == 1).argmax(axis=1)
        has_tp = hit_tp.any(axis=1)
        has_sl = hit_sl.any(axis=1)
        win_tp = has_tp & ((~has_sl) | (first_tp <= first_sl))
        win_sl = has_sl & ((~has_tp) | (first_sl < first_tp))
        drift = (entry[:, 0] - fc[:, -1]) / at[:, 0]
        sr = np.where(win_tp, TP_R, np.where(win_sl, -SL_R, drift)) - cost
        short_r[s:t] = np.where(good, sr, 0.0)

    return long_r, short_r


def generate_labels(m5: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    """Turn realized R into BUY/SELL/NONE labels + conf + mgmt targets."""
    long_r, short_r = compute_realized_r(m5, atr)
    n = len(m5)

    entry_label = np.full(n, 2, dtype=np.int64)   # NONE default
    entry_conf = np.zeros(n, dtype=np.float32)
    mgmt_label = np.zeros(n, dtype=np.int64)

    buy_ok = (long_r >= ENTRY_MIN_R) & ((long_r - short_r) >= EDGE_MARGIN)
    sell_ok = (short_r >= ENTRY_MIN_R) & ((short_r - long_r) >= EDGE_MARGIN)

    entry_label[buy_ok] = 0
    entry_label[sell_ok] = 1
    entry_conf[buy_ok] = np.clip(long_r[buy_ok] / TP_R, 0, 1)
    entry_conf[sell_ok] = np.clip(short_r[sell_ok] / TP_R, 0, 1)

    # Mgmt: for a labeled side that ends up a loser (< 0.2R), the model should
    # CLOSE fast rather than hold. Learned so live CLOSE calls are meaningful.
    mgmt_label[np.where(buy_ok & (long_r < 0.2))[0]] = 1
    mgmt_label[np.where(sell_ok & (short_r < 0.2))[0]] = 1

    m5["edge_long"] = long_r.astype(np.float32)
    m5["edge_short"] = short_r.astype(np.float32)
    m5["entry_label"] = entry_label
    m5["entry_conf"] = entry_conf
    m5["mgmt_label"] = mgmt_label
    return m5


def build_dataset(m5: pd.DataFrame, max_samples: int = 40_000, val_split: float = 0.15):
    from app.candle_brain import FEATURE_COLS, N_FEATURES

    missing = [c for c in FEATURE_COLS if c not in m5.columns]
    for c in missing:
        m5[c] = 0.0

    m5 = m5.dropna(subset=["entry_label"])
    features = m5[FEATURE_COLS].values.astype(np.float32)
    entry_labels = m5["entry_label"].values.astype(np.int64)
    mgmt_labels = m5["mgmt_label"].values.astype(np.int64)
    conf_targets = m5["entry_conf"].values.astype(np.float32)
    edges = np.column_stack([
        m5["edge_long"].values, m5["edge_short"].values
    ]).astype(np.float32)

    features = np.nan_to_num(features, nan=0.0)
    n = len(features)

    # Rows need SEQ_LEN bars of history AND a full future window.
    use = np.zeros(n, dtype=bool)
    use[SEQ_LEN:n - LABEL_WINDOW] = True

    # Time-aware split: train = older bars, val = newest bars (no leakage).
    val_start = int(n * (1 - val_split))

    rng = np.random.default_rng(42)

    def subsample(indices, count):
        if len(indices) <= count:
            return indices
        return rng.choice(indices, size=count, replace=False)

    def build(idx):
        m = len(idx)
        X = np.empty((m, SEQ_LEN, N_FEATURES), dtype=np.float32)
        Y_e = np.empty(m, dtype=np.int64)
        Y_m = np.empty(m, dtype=np.int64)
        Y_c = np.empty(m, dtype=np.float32)
        Y_r = np.empty((m, 2), dtype=np.float32)
        for j, i in enumerate(idx):
            X[j] = features[i-SEQ_LEN:i]
            Y_e[j] = entry_labels[i]
            Y_m[j] = mgmt_labels[i]
            Y_c[j] = conf_targets[i]
            Y_r[j] = edges[i]
        return X, Y_e, Y_m, Y_c, Y_r

    def split_valid(start, end):
        idx_range = np.arange(start, end)
        return idx_range[(idx_range >= SEQ_LEN) & use[idx_range]]

    def pick_class(idx, cls):
        return idx[entry_labels[idx] == cls]

    train_idx = split_valid(0, val_start)
    val_idx = split_valid(val_start, n)

    per = max_samples // 3
    tr_parts = [
        subsample(pick_class(train_idx, c), per) for c in (0, 1, 2)
    ]
    chosen_train = np.concatenate(tr_parts)
    chosen_train.sort()

    per_v = max_samples // 9
    va_parts = [
        subsample(pick_class(val_idx, c), per_v) for c in (0, 1, 2)
    ]
    chosen_val = np.concatenate(va_parts)
    chosen_val.sort()

    if len(chosen_train) < 1000 or len(chosen_val) < 500:
        raise RuntimeError("Not enough data for train/val split")

    X_train, Y_e_train, Y_m_train, Y_c_train, Y_r_train = build(chosen_train)
    X_val, Y_e_val, Y_m_val, Y_c_val, Y_r_val = build(chosen_val)
    return (
        X_train, Y_e_train, Y_m_train, Y_c_train, Y_r_train,
        X_val, Y_e_val, Y_m_val, Y_c_val, Y_r_val,
    )


def sample_weights(edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Profit-weighted loss weights.

    - Correct trade on a big-edge bar: rewarded (up to 1.4x).
    - Calling a trade where BOTH sides lose (chop-trap): penalised 1.8x.
    - Everything else: mild (0.5-0.6x) so NONE isn't over-rewarded.
    """
    w = np.ones(len(labels), dtype=np.float32)
    is_trade = labels < 2
    side_r = np.where(labels == 0, edges[:, 0], edges[:, 1])
    w[is_trade] = 0.6 + 0.8 * np.clip(side_r[is_trade] / TP_R, 0, 1)
    both_lose = (edges.max(axis=1) < 0.0) & (~is_trade)
    w[both_lose] = 1.8
    w[(~is_trade) & (~both_lose)] = 0.5
    return w


def simulate_val(
    model, X_val: np.ndarray, Y_r_val: np.ndarray, device: str,
    conf_thresh: float = VAL_CONF_THRESH,
) -> dict:
    """Simulate the exact SL/TP trades the model would take on validation.

    conf_thresh = 0.0 uses ALL raw directional calls (model-selection metric);
    conf_thresh > 0 reports only confident trades.
    """
    import torch
    import torch.nn.functional as F

    probs = []
    with torch.no_grad():
        for i in range(0, len(X_val), 2048):
            xb = torch.from_numpy(X_val[i:i+2048]).to(device)
            entry_logits, _, _ = model(xb)
            probs.append(F.softmax(entry_logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs)
    pred = probs.argmax(axis=1)
    pconf = probs.max(axis=1)

    raw_trades = int((pred != 2).sum())
    trades = []
    for i in range(len(pred)):
        if pred[i] == 2 or pconf[i] < conf_thresh:
            continue
        trades.append(float(Y_r_val[i, int(pred[i])]))

    empty = {"net_r": 0.0, "trades": 0, "raw_trades": raw_trades,
             "wr": 0.0, "exp": 0.0, "pf": 0.0, "dd": 0.0}
    if not trades:
        return empty
    rs = np.asarray(trades)
    net = float(rs.sum())
    wins = float(rs[rs > 0].sum())
    losses = float((-rs[rs < 0]).sum())
    pf = (wins / losses) if losses > 0 else float("inf")
    equity = np.cumsum(rs)
    peak = np.maximum.accumulate(equity)
    dd = float((peak - equity).max())
    return {
        "net_r": net,
        "trades": len(rs),
        "raw_trades": raw_trades,
        "wr": 100.0 * float((rs > 0).mean()),
        "exp": net / len(rs),
        "pf": pf,
        "dd": dd,
    }


def train(
    symbol: str,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 3e-4,
    patience: int = 20,
    val_split: float = 0.15,
    max_samples: int = 40_000,
    start_year: int = None,
    end_year: int = None,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from app.candle_brain import CandleBrainTransformer, N_FEATURES, SEQ_LEN

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Loading {symbol} M1 data (years {start_year or 'start'}..{end_year or 'end'})...", flush=True)
    m1 = load_m1_data(symbol, start_year=start_year, end_year=end_year)
    print(f"  {len(m1):,} M1 bars", flush=True)

    print("Resampling to M5...", flush=True)
    m5 = resample_m5(m1)
    print(f"  {len(m5):,} M5 bars", flush=True)

    print("Computing features (incl. regime: ADX/vol/squeeze/trend)...", flush=True)
    m5 = compute_features(m5)
    m5 = add_h1_context(m5, m1)
    m5 = add_swing_features(m5)
    m5 = add_time_features(m5)

    atr = compute_atr(m5, ATR_PERIOD)

    print(f"Simulating SL={SL_R}R / TP={TP_R}R trades over {LABEL_WINDOW} bars "
          f"(cost {COST_R}R, min edge {ENTRY_MIN_R}R, margin {EDGE_MARGIN}R)...", flush=True)
    m5 = generate_labels(m5, atr)

    buy_count = (m5["entry_label"] == 0).sum()
    sell_count = (m5["entry_label"] == 1).sum()
    none_count = (m5["entry_label"] == 2).sum()
    total = buy_count + sell_count + none_count
    print(f"  BUY: {buy_count} ({100*buy_count/total:.1f}%)")
    print(f"  SELL: {sell_count} ({100*sell_count/total:.1f}%)")
    print(f"  NONE: {none_count} ({100*none_count/total:.1f}%)")
    edge_vals = np.concatenate([
        m5.loc[m5["entry_label"] == 0, "edge_long"].values,
        m5.loc[m5["entry_label"] == 1, "edge_short"].values,
    ])
    if len(edge_vals):
        print(f"  Labeled-trade avg realized R: {edge_vals.mean():.3f}")

    print("Building sequence dataset...")
    (
        X_train, Y_e_train, Y_m_train, Y_c_train, Y_r_train,
        X_val, Y_e_val, Y_m_val, Y_c_val, Y_r_val,
    ) = build_dataset(m5, max_samples=max_samples, val_split=val_split)
    print(f"  train: {len(X_train):,} samples x {SEQ_LEN} bars x {N_FEATURES} features")
    print(f"  val:   {len(X_val):,} samples")

    w_train = sample_weights(Y_r_train, Y_e_train)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_e_train),
        torch.from_numpy(Y_m_train),
        torch.from_numpy(Y_c_train),
        torch.from_numpy(w_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(Y_e_val),
        torch.from_numpy(Y_m_val),
        torch.from_numpy(Y_c_val),
        torch.from_numpy(Y_r_val),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = CandleBrainTransformer(n_features=N_FEATURES, seq_len=SEQ_LEN).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing WITHOUT warmup. OneCycleLR's long warmup kept the LR
    # near zero through the first ~20 epochs, so the model never learned
    # before early stopping fired.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )

    best_score = -float("inf")
    best_metrics = None
    epochs_no_improve = 0
    best_state = None

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for X_batch, Y_e, Y_m, Y_c, W in train_loader:
            X_batch = X_batch.to(device)
            Y_e = Y_e.to(device)
            Y_m = Y_m.to(device)
            Y_c = Y_c.to(device)
            W = W.to(device)

            entry_logits, conf, mgmt_logits = model(X_batch)

            loss_e = (F.cross_entropy(entry_logits, Y_e, reduction="none") * W).mean()
            loss_c = F.binary_cross_entropy(conf.squeeze(-1), Y_c)
            loss_m = F.cross_entropy(mgmt_logits, Y_m)
            loss = loss_e + 0.4 * loss_c + 0.15 * loss_m

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            n_train += X_batch.size(0)

        train_loss /= max(n_train, 1)

        scheduler.step()

        model.eval()
        # Model selection uses ALL raw directional calls (conf_thresh=0) so the
        # model is rewarded as soon as its direction has real edge. The
        # confident-trade breakdown is reported for visibility.
        val_raw = simulate_val(model, X_val, Y_r_val, device, conf_thresh=0.0)
        val_conf = simulate_val(model, X_val, Y_r_val, device, conf_thresh=VAL_CONF_THRESH)
        val_metrics = val_raw
        score = (
            val_metrics["net_r"] - DD_PENALTY * val_metrics["dd"]
            if val_metrics["trades"] >= MIN_VAL_TRADES
            else -float("inf")
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | "
                f"raw trades={val_metrics['trades']:>5} WR={val_metrics['wr']:4.1f}% "
                f"exp={val_metrics['exp']:+.3f}R PF={val_metrics['pf']:.2f} "
                f"net={val_metrics['net_r']:+.1f}R dd={val_metrics['dd']:.1f}R "
                f"(conf@>{VAL_CONF_THRESH}: {val_conf['trades']} trades "
                f"exp={val_conf['exp']:+.3f}R)",
                flush=True,
            )

        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (no val-profit improvement)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = out_dir / f"candle_brain_{symbol}.pt"
    torch.save(model.state_dict(), str(out_path))
    print(f"\nSaved: {out_path}", flush=True)
    if best_metrics:
        print(
            f"Best val: trades={best_metrics['trades']} WR={best_metrics['wr']:.1f}% "
            f"exp={best_metrics['exp']:+.3f}R PF={best_metrics['pf']:.2f} "
            f"net={best_metrics['net_r']:+.1f}R maxDD={best_metrics['dd']:.1f}R",
            flush=True,
        )

    return best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CandleBrain (Regime-Aware Asymmetric Edge)")
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "US100", "US500", "US30"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=40_000)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()

    train(
        symbol=args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        max_samples=args.max_samples,
        start_year=args.start_year,
        end_year=args.end_year,
    )
