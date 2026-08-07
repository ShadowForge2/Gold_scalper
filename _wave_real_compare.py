"""Real-data engine-vs-backtest equality check for WaveScalper.

Replays actual XAUUSD M1 bars through the live WaveScalper state machine
(model=None so the chop gate is always open) and compares the engine's
per-candle realized R sequence against `_sweep_candle_wave.run_candle_wave`
computed on the same candles with the same ATR/candle-open.

Feeds one candle's worth of bars per `feed()` call (the live bot feeds every
minute, but `feed` is purely time-filtered so batching a whole candle is
semantically identical). Comparison excludes:
  - the first test candle (the sweep skips k==0)
  - the last test candle (the sweep force-closes it at the last M1 close;
    the engine only fires candle_end when the NEXT bar arrives)
  - any candle with < 2 M1 bars (the sweep skips those)
"""
import sys

import pandas as pd

from _train_candle_h1 import load_m1_data
from app.wave_scalper import WaveScalper
import _sweep_candle_wave as sweep

PARAMS = dict(
    entry_r=0.50, cut_r=0.03, profit_r=0.05, cost_r=0.05,
    jump_break_r=1.5, jump_body_r=0.70, trail_r=0.5, reversal_r=0.5,
)
ATR_PERIOD = 14
CACHE_CAP = 80000


def bucket_m1(m1, start, end):
    idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
    idx = idx[~idx.index.duplicated(keep="first")].sort_index()
    bucket = idx.index.floor("1h")
    mask = (bucket >= start) & (bucket <= end)
    sub = idx[mask]
    b = bucket[mask]
    out = {}
    for ts, grp in sub.groupby(b, sort=True):
        out[ts] = grp[["open", "high", "low", "close"]].values.astype(float)
    return out


def run_engine(m1, hist_end, test):
    eng = WaveScalper("XAUUSD", model=None, logger=None, **PARAMS)
    hist = m1[(m1["time"] < hist_end) & (m1["time"] >= hist_end - pd.Timedelta(days=180))]
    eng.set_history(hist)
    eng._warm = True
    eng._last_ts = pd.Timestamp(hist["time"].iloc[-1])
    eng._last_close = float(hist["close"].iloc[-1])
    eng._start_candle(pd.Timestamp(test["time"].iloc[0]).floor("h"))

    out = []
    test = test.sort_values("time").reset_index(drop=True)
    chunk = 4000
    for s in range(0, len(test), chunk):
        blk = test.iloc[s:s + chunk]
        eng.set_history(blk)
        if eng._cache is not None and len(eng._cache) > CACHE_CAP:
            eng._cache = eng._cache.iloc[-CACHE_CAP:]
        for cts, grp in blk.groupby(blk["time"].dt.floor("h")):
            now_ts = (pd.Timestamp(grp["time"].iloc[-1]) + pd.Timedelta(minutes=1)).timestamp()
            while True:
                action = eng.feed(None, now_ts=now_ts)
                if action is None:
                    break
                st = eng.state_dict()
                if action["type"] == "enter":
                    eng.confirm_entry(float(action["entry"]))
                elif action["type"] == "exit":
                    candle = pd.Timestamp(st["candle"]).floor("h")
                    pos = st["pos"]
                    e = st["entry"]
                    atr = st["atr"] if st["atr"] > 0 else 1e-9
                    r = (float(action["price"]) - e) * pos / atr - PARAMS["cost_r"]
                    out.append((candle, float(r)))
                    eng.confirm_exit()
                else:
                    break
    return out


def main():
    m1 = load_m1_data("XAUUSD")
    h1_all = sweep.resample_h1(m1, 60)
    atr_all = sweep.compute_atr(h1_all, ATR_PERIOD)

    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-04-01")
    end = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2024-06-30 23:59")
    test = m1[(m1["time"] >= start) & (m1["time"] <= end)]
    print(f"replaying {start.date()}..{end.date()} ({len(test)} bars)", flush=True)
    engine_trades = run_engine(m1, start, test)

    buckets = bucket_m1(m1, start, end)
    h1_idx = h1_all.index
    bt = {}
    for ts, sub in buckets.items():
        k = h1_idx.get_loc(ts)
        prev = k - 1
        if k == 0 or prev < 0 or len(sub) < 2:
            continue
        o = float(h1_all["open"].iloc[k])
        a = float(atr_all.iloc[prev]) if atr_all.iloc[prev] > 0 else 1e-9
        bt[ts] = [float(x) for x in sweep.run_candle_wave(
            sub, o, a, **PARAMS)]

    if buckets:
        bt.pop(max(buckets.keys()), None)
    # The sweep skips its k==0 candle (no prior completed candle to gate on at
    # the very start of the window). The engine's cold-start first candle has
    # no valid prior ATR either — exclude it the same way.
    if buckets:
        bt.pop(min(buckets.keys()), None)
    compare_candles = set(bt.keys())

    eng_by_candle = {}
    for c, r in engine_trades:
        if c in compare_candles:
            eng_by_candle.setdefault(c, []).append(r)

    mismatches = 0
    n_trades_bt = sum(len(bt[c]) for c in compare_candles)
    n_trades_eng = sum(len(eng_by_candle.get(c, [])) for c in compare_candles)
    for c in sorted(compare_candles):
        a = bt.get(c, [])
        b = eng_by_candle.get(c, [])
        if a != b:
            mismatches += 1
            if mismatches <= 8:
                print(f"  MISMATCH {c} bt={a} eng={b}")
    print(f"candles={len(compare_candles)} bt_trd={n_trades_bt} "
          f"eng_trd={n_trades_eng} mismatches={mismatches}")
    print("RESULT:", "ALL MATCH" if mismatches == 0 and n_trades_bt == n_trades_eng else "DIFF")


if __name__ == "__main__":
    main()
