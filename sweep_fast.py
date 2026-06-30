"""
Fast EURUSD parameter sweep — pre-computes all indicators, then runs
lightweight backtests using numpy arrays. No pandas in hot path.
"""

import sys, os, time, itertools, json
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.dukascopy_client import DukascopyClient

INITIAL_BALANCE = 20.0

# ── helpers ──

def ema(values, period):
    if len(values) < period:
        return np.full_like(values, values[-1] if len(values) > 0 else 0)
    mult = 2.0 / (period + 1)
    result = np.empty_like(values)
    result[:period] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = (values[i] - result[i-1]) * mult + result[i-1]
    return result

def atr(high, low, close, period=14):
    N = len(high)
    tr = np.zeros(N)
    tr[0] = high[0] - low[0]
    for i in range(1, N):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    result = np.zeros(N)
    result[:period] = np.mean(tr[:period])
    for i in range(period, N):
        result[i] = (result[i-1] * (period - 1) + tr[i]) / period
    return result

def rsi(closes, period=14):
    N = len(closes)
    if N < period + 1:
        return np.full(N, 50.0)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    result = np.full(N, 50.0)
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    for i in range(period, N):
        avg_g = (avg_g * (period - 1) + gains[i-1]) / period
        avg_l = (avg_l * (period - 1) + losses[i-1]) / period
        if avg_l == 0:
            result[i] = 100.0
        else:
            rs = avg_g / avg_l
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result

def resample_np(times, close, high, low, vol, rule_min=5):
    """Resample 1-min numpy arrays to M5. Returns indices mapping."""
    N = len(times)
    # Group every 5 bars
    groups = np.arange(N) // rule_min
    n_groups = groups[-1] + 1
    # For simplicity, take last of each group as the bar
    uniq, idx = np.unique(groups, return_index=True)
    last_idx = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        mask = groups == g
        last_idx[g] = np.where(mask)[0][-1]
    return last_idx

def run_fast(t, c_close, c_high, c_low, c_open, c_vol,
             ema_f_arr, ema_s_arr, atr_arr,
             m5_idx_map, m5_close, m5_high, m5_low, m5_ema21, m5_rsi,
             cfg):
    """Fast backtest using pre-computed numpy arrays."""
    N = len(c_close)
    session_ranges = []
    sess = cfg.get("session", "lon_ny")
    if sess == "lon_ny":
        session_ranges = [(7, 16), (12, 21)]
    elif sess == "london":
        session_ranges = [(7, 16)]
    elif sess == "ny":
        session_ranges = [(12, 21)]
    else:
        session_ranges = [(0, 24)]

    def session_ok(ts):
        h = ts.hour + ts.minute / 60.0
        return any(start <= h < end for start, end in session_ranges)

    score_thr = cfg.get("score_threshold", 0.65)
    sl_mult = cfg.get("sl_atr_mult", 0.8)
    tp_mult = cfg.get("tp_atr_mult", 2.0)
    rsi_buy_min = cfg.get("rsi_buy_min", 40)
    rsi_sell_max = cfg.get("rsi_sell_max", 60)
    use_cdl = cfg.get("use_candle_filter", True)
    cdl_look = cfg.get("candle_dir_lookback", 2)
    retest_range = cfg.get("retest_range_atr", 0.2)
    use_vol = cfg.get("use_volume_filter", False)
    vol_mult = cfg.get("volume_surge_mult", 1.2)
    vol_look = cfg.get("volume_lookback", 10)
    consec_limit = cfg.get("consec_loss_limit", 4)
    max_daily_tr = cfg.get("max_daily_trades", 10)
    max_daily_loss = cfg.get("max_daily_loss", 5.0)
    min_bal = cfg.get("min_balance", 10.0)
    cooldown_sec = cfg.get("cooldown_seconds", 30)
    max_bars = cfg.get("max_bars_hold", 240)

    trades = []
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    consec_losses = 0
    daily_trades = 0
    cur_day = None
    daily_pnl = 0.0
    cooldown_until = None
    in_trade = False
    entry_price = 0.0
    entry_dir = ""
    entry_signal = {}
    entry_idx = 0

    last_c1 = np.zeros(2)
    last_o1 = np.zeros(2)

    for idx in range(N):
        ts = t[idx]
        bid = c_close[idx]
        ask = bid
        mid = bid

        # Daily reset
        if cur_day != ts.date():
            cur_day = ts.date()
            daily_trades = 0
            daily_pnl = 0.0
            consec_losses = 0

        # ── Exit check ──
        if in_trade:
            mm = m5_idx_map[idx]
            if mm >= 0:
                px = c_close[idx]
                sl_p = entry_signal.get("sl", 0)
                tp_p = entry_signal.get("tp1", 0)
                bars_held = idx - entry_idx

                exit_reason = None
                if entry_dir == "BUY":
                    if px <= sl_p: exit_reason = "sl"
                    elif px >= tp_p: exit_reason = "tp"
                else:
                    if px >= sl_p: exit_reason = "sl"
                    elif px <= tp_p: exit_reason = "tp"

                if exit_reason or bars_held >= max_bars:
                    pnl = (mid - entry_price) / 0.0001 * 0.01 * 10 if entry_dir == "BUY" else (entry_price - mid) / 0.0001 * 0.01 * 10
                    balance += pnl
                    daily_pnl += pnl
                    if balance > peak: peak = balance
                    trades.append(pnl)
                    if pnl < 0: consec_losses += 1
                    else: consec_losses = 0
                    cooldown_until = ts + timedelta(seconds=cooldown_sec)
                    in_trade = False
                    continue

        # ── Entry check ──
        if not in_trade:
            if cooldown_until and ts < cooldown_until: continue
            cooldown_until = None
            if not session_ok(ts): continue
            if consec_losses >= consec_limit: continue
            if daily_trades >= max_daily_tr: continue
            if daily_pnl <= -max_daily_loss: continue
            if balance < min_bal: continue

            mm = m5_idx_map[idx]
            if mm < 0: continue

            # M5 trend
            trend = "NEUTRAL"
            m5_cl = m5_close[mm]
            if m5_cl > m5_ema21[mm]: trend = "BULLISH"
            elif m5_cl < m5_ema21[mm]: trend = "BEARISH"
            if trend == "NEUTRAL": continue

            # M1 EMA values
            e_fast = ema_f_arr[idx]
            e_slow = ema_s_arr[idx]
            atr_v = atr_arr[idx]
            if atr_v <= 0: atr_v = 0.0001

            # Volume
            if use_vol:
                if idx < vol_look + 1: continue
                recent_vol = c_vol[idx]
                avg_vol = np.mean(c_vol[idx-vol_look:idx])
                if avg_vol > 0 and recent_vol <= avg_vol * vol_mult: continue

            # RSI
            m5_r = m5_rsi[mm]

            # Candle direction
            if use_cdl and idx >= cdl_look:
                bullish = 0
                for i in range(1, cdl_look + 1):
                    if c_close[idx-i] > c_open[idx-i]: bullish += 1
                has_bullish = bullish > 0
            else:
                has_bullish = True

            entry_mode = None

            if trend == "BULLISH":
                if use_cdl and not has_bullish: continue
                crossover = e_fast > e_slow and (idx == 0 or ema_f_arr[idx-1] <= ema_s_arr[idx-1])
                retest = e_fast > e_slow and abs(c_close[idx] - e_slow) <= atr_v * retest_range
                if not (crossover or retest): continue
                if m5_r < rsi_buy_min: continue
                entry_mode = "crossover" if crossover else "retest"
                score = 0.5 + (0.15 if crossover else 0) + min(abs(e_fast - e_slow) / atr_v * 0.3, 0.2)
                if rsi_buy_min <= m5_r <= rsi_sell_max: score += 0.1
                if score < score_thr: continue
                sl_price = ask - atr_v * sl_mult
                tp_price = ask + atr_v * tp_mult
                entry_dir = "BUY"
            else:
                if use_cdl and has_bullish: continue
                crossover = e_fast < e_slow and (idx == 0 or ema_f_arr[idx-1] >= ema_s_arr[idx-1])
                retest = e_fast < e_slow and abs(c_close[idx] - e_slow) <= atr_v * retest_range
                if not (crossover or retest): continue
                if m5_r > rsi_sell_max: continue
                entry_mode = "crossover" if crossover else "retest"
                score = 0.5 + (0.15 if crossover else 0) + min(abs(e_fast - e_slow) / atr_v * 0.3, 0.2)
                if rsi_buy_min <= m5_r <= rsi_sell_max: score += 0.1
                if score < score_thr: continue
                sl_price = bid + atr_v * sl_mult
                tp_price = bid - atr_v * tp_mult
                entry_dir = "SELL"

            entry_price = ask if entry_dir == "BUY" else bid
            entry_signal = {"sl": sl_price, "tp1": tp_price}
            entry_idx = idx
            in_trade = True
            daily_trades += 1

    # Close last trade
    if in_trade:
        pnl = (c_close[-1] - entry_price) / 0.0001 * 0.01 * 10 if entry_dir == "BUY" else (entry_price - c_close[-1]) / 0.0001 * 0.01 * 10
        balance += pnl
        trades.append(pnl)

    if not trades:
        return {"trades": 0, "net_pnl": 0, "win_rate": 0, "profit_factor": 0,
                "max_dd": 0, "final_bal": balance, "profit_days": 0, "total_days": 0, "avg_daily_pnl": 0}

    arr = np.array(trades)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / len(arr) * 100
    gp = wins.sum() if len(wins) > 0 else 0
    gl = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)
    cum = np.cumsum(arr)
    peak_cum = np.maximum.accumulate(cum)
    dd = (peak_cum - cum).max()

    return {
        "trades": len(arr),
        "net_pnl": round(arr.sum(), 2),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "max_dd": round(dd, 2),
        "final_bal": round(balance, 2),
        "avg_daily_pnl": 0,  # skip detailed calc in sweep
        "profit_days": 0,
        "total_days": 0,
    }


def main():
    print("Loading EURUSD M1 data...", flush=True)
    client = DukascopyClient(symbol="EURUSD")
    df = client.download_range(2024, 2024)
    if len(df) == 0:
        print("No data")
        return
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Loaded {len(df)} bars", flush=True)

    # First 2 weeks for sweep
    start_date = df["time"].min()
    mask = df["time"] < start_date + timedelta(days=14)
    df = df[mask].reset_index(drop=True)
    print(f"Sweep window: {df['time'].min()} to {df['time'].max()} ({len(df)} bars)", flush=True)

    # Pre-compute ALL arrays
    t = pd.to_datetime(df["time"].values).to_pydatetime()
    c_close = df["close"].values.astype(float)
    c_high = df["high"].values.astype(float)
    c_low = df["low"].values.astype(float)
    c_open = df["open"].values.astype(float)
    c_vol = df["tick_volume"].values.astype(float) if "tick_volume" in df.columns else np.ones(len(df))

    # Pre-compute EMAs for all periods needed
    ema_cache = {}
    for p in [3, 5, 8, 13, 21]:
        ema_cache[p] = ema(c_close, p)

    # Pre-compute ATR(14)
    atr_arr = atr(c_high, c_low, c_close, 14)

    # M5 resampling: map each M1 bar to its M5 bar index
    m5_idx_map = np.zeros(len(df), dtype=int)
    m5_last = -1
    for i in range(len(df)):
        m5_i = i // 5
        m5_idx_map[i] = m5_i
    n_m5 = m5_idx_map[-1] + 1

    # Build M5 arrays (last bar of each 5-min group)
    m5_close = np.zeros(n_m5)
    m5_high = np.zeros(n_m5)
    m5_low = np.zeros(n_m5)
    for g in range(n_m5):
        mask = m5_idx_map == g
        m5_close[g] = c_close[mask][-1]
        m5_high[g] = c_high[mask].max()
        m5_low[g] = c_low[mask].min()

    m5_ema21 = ema(m5_close, 21)
    m5_rsi = rsi(m5_close, 14)

    print(f"M5 bars: {n_m5}", flush=True)

    # ── Grid ──
    grid = {
        "ema_fast": [3, 5],
        "ema_slow": [8, 13],
        "sl_atr_mult": [0.5, 0.8, 1.0],
        "tp_atr_mult": [1.5, 2.0, 2.5, 3.0],
        "rsi_buy_min": [35, 40],
        "rsi_sell_max": [60, 65],
        "score_threshold": [0.50, 0.60, 0.70],
        "session": ["all", "lon_ny"],
        "use_candle_filter": [True, False],
    }

    keys = list(grid.keys())
    all_combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Total combos: {len(all_combos)}", flush=True)

    results = []
    start_t = time.time()

    for i, combo in enumerate(all_combos):
        cfg = dict(zip(keys, combo))

        # Select pre-computed EMA arrays
        ef_arr = ema_cache[cfg["ema_fast"]]
        es_arr = ema_cache[cfg["ema_slow"]]

        r = run_fast(t, c_close, c_high, c_low, c_open, c_vol,
                     ef_arr, es_arr, atr_arr,
                     m5_idx_map, m5_close, m5_high, m5_low,
                     m5_ema21, m5_rsi, cfg)
        results.append({**cfg, **r})

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_t
            best = max(rr["net_pnl"] for rr in results)
            print(f"[{(i+1)/len(all_combos)*100:5.1f}%] {i+1}/{len(all_combos)} "
                  f"elapsed={elapsed:.0f}s best=${best:+.2f}", flush=True)

    # Sort
    results.sort(key=lambda x: x["net_pnl"], reverse=True)

    print(f"\n{'='*100}")
    print(f"  TOP 20 (2-week sweep)")
    print(f"{'='*100}")
    h = f"  {'Rank':>4} {'Net':>7} {'WR%':>6} {'PF':>6} {'Trades':>6} {'Config'}"
    print(h)
    print(f"  {'-'*90}")
    for rank, r in enumerate(results[:20], 1):
        c = (f"EMA{r['ema_fast']}/{r['ema_slow']} "
             f"SL{r['sl_atr_mult']} TP{r['tp_atr_mult']} "
             f"thr{r['score_threshold']} "
             f"rsi={r['rsi_buy_min']}-{r['rsi_sell_max']} "
             f"sess={r['session']} cdl={r['use_candle_filter']}")
        print(f"  {rank:>4} ${r['net_pnl']:>+6.2f} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
              f"{r['trades']:>5} {c}")

    with open("sweep_results.json", "w") as f:
        # Convert numpy types
        clean = []
        for r in results:
            clean.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()})
        json.dump(clean, f, indent=2)
    print(f"\nSaved sweep_results.json", flush=True)

    # Validate top 3 on full year
    print(f"\n{'='*100}")
    print(f"  VALIDATE TOP 3 ON FULL YEAR")
    print(f"{'='*100}")

    # Reload full year data
    df2 = client.download_range(2024, 2024)
    df2 = df2.sort_values("time").reset_index(drop=True)
    c_close2 = df2["close"].values.astype(float)
    c_high2 = df2["high"].values.astype(float)
    c_low2 = df2["low"].values.astype(float)
    c_open2 = df2["open"].values.astype(float)
    c_vol2 = df2["tick_volume"].values.astype(float) if "tick_volume" in df2.columns else np.ones(len(df2))
    t2 = pd.to_datetime(df2["time"].values).to_pydatetime()
    atr2 = atr(c_high2, c_low2, c_close2, 14)

    ema_cache2 = {}
    for p in [3, 5, 8, 13, 21]:
        ema_cache2[p] = ema(c_close2, p)

    m5_idx2 = np.zeros(len(df2), dtype=int)
    for i in range(len(df2)):
        m5_idx2[i] = i // 5
    n_m5_2 = m5_idx2[-1] + 1
    m5_close2 = np.zeros(n_m5_2)
    m5_high2 = np.zeros(n_m5_2)
    m5_low2 = np.zeros(n_m5_2)
    for g in range(n_m5_2):
        mask = m5_idx2 == g
        m5_close2[g] = c_close2[mask][-1]
        m5_high2[g] = c_high2[mask].max()
        m5_low2[g] = c_low2[mask].min()
    m5_ema21_2 = ema(m5_close2, 21)
    m5_rsi_2 = rsi(m5_close2, 14)

    for rank, r in enumerate(results[:3], 1):
        cfg = {k: r[k] for k in keys}
        ef = ema_cache2[cfg["ema_fast"]]
        es = ema_cache2[cfg["ema_slow"]]
        vr = run_fast(t2, c_close2, c_high2, c_low2, c_open2, c_vol2,
                      ef, es, atr2,
                      m5_idx2, m5_close2, m5_high2, m5_low2,
                      m5_ema21_2, m5_rsi_2, cfg)
        c = (f"EMA{r['ema_fast']}/{r['ema_slow']} "
             f"SL{r['sl_atr_mult']} TP{r['tp_atr_mult']} "
             f"thr{r['score_threshold']}")
        print(f"  [{rank}/3] {c}")
        print(f"    2-wk: ${r['net_pnl']:+.2f} {r['win_rate']}% {r['trades']}t PF={r['profit_factor']}")
        print(f"    Year: ${vr['net_pnl']:+.2f} {vr['win_rate']}% {vr['trades']}t PF={vr['profit_factor']} "
              f"bal=${vr['final_bal']}", flush=True)

    print(f"\nDONE ({time.time()-start_t:.0f}s total)", flush=True)

if __name__ == "__main__":
    main()
