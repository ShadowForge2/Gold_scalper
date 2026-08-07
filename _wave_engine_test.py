import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _sweep_candle_wave import run_candle_wave
from app.wave_scalper import WaveScalper

PARAMS = dict(entry_r=0.50, cut_r=0.03, profit_r=0.05, cost_r=0.05,
              jump_break_r=1.5, jump_body_r=0.70, trail_r=0.5, reversal_r=0.5)


class FixedAtrWave(WaveScalper):
    """Override the gate so atr=1.0 and gate always open -> state machine only."""

    def _compute_gate(self, prev_idx):
        self._gate_ok = True
        h1 = self._resample_h1()
        ts = self._candle_ts
        if h1 is not None and ts is not None and ts in h1.index and prev_idx >= 0:
            self._candle_open = float(h1["open"].iloc[h1.index.get_loc(ts)])
            self._atr = 1.0
        else:
            self._atr = 0.0


def make_m1(n_candles=40, bars_per=60, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_candles * bars_per, freq="1min")
    px = 100.0 + np.cumsum(rng.normal(0, 0.4, len(idx)))
    df = pd.DataFrame({"time": idx})
    df["open"] = px - 0.1
    df["high"] = px + np.abs(rng.normal(0.5, 0.3, len(idx)))
    df["low"] = px - np.abs(rng.normal(0.5, 0.3, len(idx)))
    df["close"] = px
    df["tick_volume"] = rng.integers(10, 100, len(idx))
    return df


def run_backtest_per_candle(m1):
    groups = m1.groupby(pd.to_datetime(m1["time"]).dt.floor("1h"))
    out = {}
    for k, (_, g) in enumerate(groups):
        o = float(g["open"].iloc[0])
        sub = g[["open", "high", "low", "close"]].values
        out[k] = run_candle_wave(sub, o, 1.0, **PARAMS)
    return out


def run_live_per_candle(m1):
    eng = FixedAtrWave(symbol="TEST", model=None, **PARAMS)
    eng.set_history(m1)
    out = {}
    for _, row in m1.iterrows():
        now = pd.Timestamp(row["time"]).value / 1e9 + 61
        frame = m1[m1["time"] == row["time"]]
        action = eng.feed(frame, now_ts=now)
        if action is None:
            continue
        if action["type"] == "enter":
            eng.confirm_entry(action["entry"])
        elif action["type"] == "exit":
            st = eng.state_dict()
            candle = pd.Timestamp(st["candle"])
            hour = candle.floor("h")
            k = _hour_index(m1, hour)
            e = st["entry"]
            pos = st["pos"]
            r = (action["price"] - e) * pos - PARAMS["cost_r"]
            out.setdefault(k, []).append(r)
            eng.confirm_exit()
    return out


def _hour_index(m1, hour):
    hours = sorted(set(pd.to_datetime(m1["time"]).dt.floor("1h")))
    return hours.index(hour)


def main():
    ok = True
    for seed in range(5):
        m1 = make_m1(n_candles=40, bars_per=60, seed=seed)
        bt = run_backtest_per_candle(m1)
        lv = run_live_per_candle(m1)
        n = len(bt)
        # Compare candles 1..N-2: candle 0 has no valid prior ATR (engine skips),
        # and candle N-1's engine force-close waits for a next bar that the data
        # does not contain (live always has it).
        for k in range(1, n - 1):
            a = np.array(bt[k], dtype=float)
            b = np.array(lv.get(k, []), dtype=float)
            if len(a) != len(b) or not np.allclose(a, b, atol=1e-9):
                ok = False
                print(f"seed={seed} candle={k} MISMATCH bt={len(a)} lv={len(b)} "
                      f"sum_bt={a.sum():+.3f} sum_lv={b.sum():+.3f}")
                for i in range(min(len(a), len(b))):
                    if not np.allclose(a[i], b[i], atol=1e-9):
                        print(f"  first diff {i}: bt={a[i]:+.6f} lv={b[i]:+.6f}")
                        break
                break
        print(f"seed={seed} ok={ok}")
    print("ALL MATCH" if ok else "MISMATCHES FOUND")


if __name__ == "__main__":
    main()
