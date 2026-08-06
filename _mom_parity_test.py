"""Backtest-parity test for app/momentum_engine.py vs _two_engine.us100_jump_trades.

Checks on US100 M5 2025:
  1. feature parity   — engine.compute_features == _train_candle_brain.compute_features
  2. signal parity    — engine detect() per-bar rule == backtest sig array
  3. live detect()    — end-to-end detect() fires on the same candle as bt_sig
  4. exit parity      — bot's exit block (SL/retrace/timeout) == backtest exit loop

Run:  python _mom_parity_test.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t
from app.momentum_engine import MomentumEngine

ENGINE = MomentumEngine(mz_min=2.0, body_min=0.60, ts_min=0.50, ema_span=480,
                        sl_r=1.0, jump_target=1.0, retr_r=0.25, max_hold=12,
                        atr_period=14)

MAX_HOLD = 12
JUMP_TARGET = 1.0
RETR_DIST = 0.25
SL_R = 1.0
EMA = 480
MZ_MIN = 2.0


def load_m5(path):
    m1 = pd.read_parquet(path)
    m5 = ENGINE.resample_m5(m1)
    return m1, m5


def bt_features_parity(m5):
    eng = ENGINE.compute_features(m5)
    m5v = m5.copy()
    m5v["tick_volume"] = 0  # volume not used by the momentum signal columns
    back = t.compute_features(m5v)
    diffs = {}
    a_atr = eng["atr"].fillna(0).values
    b_atr = t.compute_atr(m5, t.ATR_PERIOD).fillna(0).values
    atr_diff = np.nanmax(np.abs(a_atr[14:] - b_atr[14:])) if len(a_atr) > 14 else 0.0
    diffs["atr"] = float(atr_diff)
    for col in ("body_ratio", "momentum_z", "trend_strength"):
        a = eng[col].fillna(0).values
        b = back[col].fillna(0).values
        diff = np.nanmax(np.abs(a - b)) if len(a) else 0.0
        diffs[col] = float(diff)
    return diffs


def bt_sig_array(m5, atr):
    f = ENGINE.compute_features(m5)
    c = f["close"].values
    o = f["open"].values
    br = f["body_ratio"].values
    ts = f["trend_strength"].values
    mz = f["momentum_z"].values
    n = len(f)
    regime = c > pd.Series(c).ewm(span=EMA, adjust=False).mean().values
    sig = np.zeros(n, dtype=np.int64)
    sig[(o < c) & (br >= 0.60) & (ts >= 0.50) & (mz >= MZ_MIN) & regime] = 1
    sig[(o > c) & (br >= 0.60) & (ts <= -0.50) & (mz <= -MZ_MIN) & ~regime] = -1
    sig[:max(50, EMA) + 1] = 0  # backtest scan filter: idx > max(50, ema)
    return sig


def engine_rule_array(m5):
    f = ENGINE.compute_features(m5)
    c = f["close"].values
    o = f["open"].values
    atr = f["atr"].values
    br = f["body_ratio"].values
    ts = f["trend_strength"].values
    mz = f["momentum_z"].values
    n = len(f)
    closes = pd.Series(c, index=f.index)
    ema = closes.ewm(span=EMA, adjust=False).mean().values
    regime_up = c > ema
    out = np.zeros(n, dtype=np.int64)
    for i in range(EMA + 1, n):
        if (o[i] < c[i] and br[i] >= 0.60 and ts[i] >= 0.50
                and mz[i] >= MZ_MIN and regime_up[i]):
            out[i] = 1
        elif (o[i] > c[i] and br[i] >= 0.60 and ts[i] <= -0.50
              and mz[i] <= -MZ_MIN and not regime_up[i]):
            out[i] = -1
    return out


def live_detect_checks(m1, m5, bt_sig):
    """Call detect() at the completion of a sample of candles and verify it fires
    exactly like bt_sig on that candle (end-to-end: resample + warmup + formed filter)."""
    n = len(m5)
    rng = np.random.default_rng(42)
    sig_idx = np.where(bt_sig != 0)[0]
    sig_idx = sig_idx[sig_idx > EMA]
    sample = rng.choice(sig_idx, size=min(40, len(sig_idx)), replace=False)
    sample = np.append(sample, rng.choice(np.where(bt_sig == 0)[0][np.where(bt_sig == 0)[0] > EMA],
                                          size=min(40, int(np.sum(bt_sig == 0))), replace=False))
    fails = []
    for i in sample:
        end_boundary = int((m5.index[i] + pd.Timedelta(minutes=5)).value // 10**9)
        now_ts = end_boundary + 1
        m1_up_to = m1[m1["time"] <= pd.Timestamp(end_boundary, unit="s")]
        sig = ENGINE.detect(m1_up_to, now_ts=now_ts)
        got = 0 if sig is None else (1 if sig["direction"] == "BUY" else -1)
        if got != bt_sig[i]:
            fails.append((str(m5.index[i]), bt_sig[i], got))
    return fails


def bot_exit_replay(m5, atr, i, d):
    """Replicate the bot's momentum exit block for a trade entered on candle i.

    Returns (exit_bar, reason) or (None, None) if never reached n-1. Uses the same
    completed-candle stepping as the bot: candle j is evaluated at cur_boundary =
    entry_boundary + (j - i + 1)*300 during candle j+1's window."""
    c = m5["close"].values
    h = m5["high"].values
    l = m5["low"].values
    n = len(m5)
    entry = c[i]
    atr_i = max(atr.values[i], 1e-10)
    sl = entry - d * SL_R * atr_i
    peak = entry
    jump_target = JUMP_TARGET
    retr_r = RETR_DIST
    max_hold = MAX_HOLD
    entry_boundary = int((m5.index[i] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1))
    entry_boundary = entry_boundary - (entry_boundary % 300)
    for j in range(i + 1, n):
        cur_boundary = entry_boundary + (j - i + 1) * 300
        if d > 0:
            if sl > 0 and l[j] <= sl:
                return j, "momentum_sl"
            if h[j] > peak:
                peak = h[j]
            bfe = (peak - entry) / atr_i
            if bfe >= jump_target and c[j] <= peak - retr_r * atr_i:
                return j, "momentum_retrace"
        else:
            if sl > 0 and h[j] >= sl:
                return j, "momentum_sl"
            if l[j] < peak:
                peak = l[j]
            bfe = (entry - peak) / atr_i
            if bfe >= jump_target and c[j] >= peak + retr_r * atr_i:
                return j, "momentum_retrace"
        bars_held = int((cur_boundary - entry_boundary) / 300) - 1
        if bars_held >= max_hold:
            return j, "momentum_timeout"
    return None, None


def bt_exit_replay(m5, atr, i, d):
    """Replicate _two_engine.us100_jump_trades exit loop for entry candle i."""
    c = m5["close"].values
    h = m5["high"].values
    l = m5["low"].values
    n = len(m5)
    entry = c[i]
    atr_i = max(atr.values[i], 1e-10)
    sl = entry - d * SL_R * atr_i
    peak = entry
    bfe = 0.0
    for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
        if d > 0:
            if h[j] > peak:
                peak = h[j]
                bfe = (h[j] - entry) / atr_i
            if l[j] <= sl:
                return j, "sl"
            if bfe >= JUMP_TARGET and c[j] <= peak - RETR_DIST * atr_i:
                return j, "retrace"
        else:
            if l[j] < peak:
                peak = l[j]
                bfe = (entry - l[j]) / atr_i
            if h[j] >= sl:
                return j, "sl"
            if bfe >= JUMP_TARGET and c[j] >= peak + RETR_DIST * atr_i:
                return j, "retrace"
    j = min(i + MAX_HOLD, n - 1)
    return j, "timeout"


def main():
    path = os.path.join("data", "dukascopy_us100", "US100_M1_2025.parquet")
    m1, m5 = load_m5(path)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    print(f"M5 bars: {len(m5)}  range {m5.index[0]} .. {m5.index[-1]}")

    # 1) feature parity
    diffs = bt_features_parity(m5)
    print("\n[1] Feature parity (max abs diff):")
    for col, d in diffs.items():
        status = "OK " if d < 1e-9 else "DIFF"
        print(f"    {col:16s} {d:.2e}  {status}")
    assert max(diffs.values()) < 1e-9, "feature mismatch"

    # 2) signal parity (full-array rule)
    bt_sig = bt_sig_array(m5, atr)
    eng_sig = engine_rule_array(m5)
    n = len(m5)
    mism = np.where(bt_sig != eng_sig)[0]
    print(f"\n[2] Signal parity: {n} bars, {np.count_nonzero(bt_sig)} bt signals, "
          f"{np.count_nonzero(eng_sig)} engine signals, {len(mism)} mismatches")
    if len(mism):
        for i in mism[:10]:
            print(f"    bar {i} {m5.index[i]} bt={bt_sig[i]} engine={eng_sig[i]}")
    assert len(mism) == 0, "signal mismatch"

    # 3) live detect() end-to-end on sampled candle-completion timestamps
    fails = live_detect_checks(m1, m5, bt_sig)
    print(f"[3] Live detect() checks: {len(fails)} mismatches")
    for f_ in fails[:10]:
        print(f"    bar {f_}")
    assert not fails, "live detect mismatch"

    # 4) exit parity on all bt signal bars (skip last 2 like the backtest scan)
    sig_idx = np.where(bt_sig != 0)[0]
    sig_idx = sig_idx[(sig_idx > max(50, EMA)) & (sig_idx < n - 2)]
    exit_mism = 0
    checked = 0
    for i in sig_idx:
        be, br_ = bot_exit_replay(m5, atr, i, bt_sig[i])
        e, r = bt_exit_replay(m5, atr, i, bt_sig[i])
        checked += 1
        if be != e or (br_ or "").replace("momentum_", "") != (r or ""):
            exit_mism += 1
            if exit_mism <= 10:
                print(f"    exit mismatch bar {m5.index[i]} bt={e}({r}) bot={be}({br_})")
    print(f"[4] Exit parity: {checked} trades checked, {exit_mism} mismatches")
    assert exit_mism == 0, "exit mismatch"

    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
