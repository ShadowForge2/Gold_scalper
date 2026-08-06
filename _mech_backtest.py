"""
MECHANICAL (rule-based) scalping backtest — no ML.

Goal: verify the strategy makes CONSISTENT PROFIT in pure mechanical mode
BEFORE spending time training models. Uses the exact same trade simulation
as CandleBrain training (enter at bar close, SL=SL_ATR*ATR, TP=TP_ATR*ATR,
exit at market after MAX_HOLD_BARS, cost subtracted) so mechanical and ML
results are directly comparable in R units.

A simple rule picks BUY/SELL/NONE per completed M5 bar using only causal
data (no lookahead: H1 trend is shifted one full H1 bar). The rule's trades
are then scored against the realized-R simulation. Reported in profit terms:
expectancy (R/trade), PF, net R, maxDD, monthly consistency, per-year.

Usage:
  python _mech_backtest.py --symbol XAUUSD --start 2020 --end 2025
  python _mech_backtest.py --symbol XAUUSD --start 2020 --end 2025 --body 0.5
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t


# --------------------------------------------------------------------------
# Causal H1 context (previous COMPLETED H1 bar, shifted — no partial-bar leak)
# --------------------------------------------------------------------------
def causal_h1_context(m5, m1):
    idx = m1.set_index("time") if "time" in m1.columns else m1.copy()
    h1 = idx.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()

    h1_close = h1["close"]
    h1_high = h1["high"]
    h1_low = h1["low"]
    ema20 = h1_close.ewm(span=20).mean()
    ema50 = h1_close.ewm(span=50).mean()
    # Normalize by H1 ATR (not price) so trend is in ATR units — meaningful
    # magnitude for gold, where (ema20-ema50)/close is ~0.01 at most.
    tr = pd.concat([
        h1_high - h1_low,
        (h1_high - h1_close.shift(1)).abs(),
        (h1_low - h1_close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    h1_atr = tr.rolling(14).mean().replace(0, 1e-10)
    h1_trend = ((ema20 - ema50) / h1_atr).clip(-5, 5)
    h1_hh = h1["high"].rolling(20).max()
    h1_ll = h1["low"].rolling(20).min()

    # Shift by one H1 bar: only use the PREVIOUS completed hour.
    h1_trend = h1_trend.shift(1)
    h1_hh = h1_hh.shift(1)
    h1_ll = h1_ll.shift(1)

    out = pd.DataFrame(index=m5.index)
    out["h1_trend"] = h1_trend.reindex(m5.index, method="ffill").fillna(0.0)
    h1_range = (h1_hh - h1_ll).replace(0, 1e-10).reindex(m5.index, method="ffill")
    out["dist_h1_high"] = ((h1_hh.reindex(m5.index, method="ffill") - m5["close"]) / h1_range).clip(0, 1).fillna(0.5)
    out["dist_h1_low"] = ((m5["close"] - h1_ll.reindex(m5.index, method="ffill")) / h1_range).clip(0, 1).fillna(0.5)
    return out


# --------------------------------------------------------------------------
# Mechanical rule
# --------------------------------------------------------------------------
def mechanical_signal(m5, h1, cfg):
    """Return (sig, mask) where sig in {0=BUY,1=SELL,2=NONE} per bar."""
    o = m5["open"].values
    h = m5["high"].values
    l = m5["low"].values
    c = m5["close"].values
    n = len(m5)

    rng = (h - l)
    body = np.abs(c - o)
    body_ratio = np.where(rng > 0, body / np.where(rng > 0, rng, 1), 0.0)
    atr = t.compute_atr(m5, t.ATR_PERIOD).values
    atr_safe = np.where(atr > 0, atr, np.nan)

    # Breakout: close beyond prior-bar extreme (1-bar range break).
    up_break = c > np.roll(h, 1)
    dn_break = c < np.roll(l, 1)
    up_break[0] = False
    dn_break[0] = False

    # Momentum: candle body must be a decent fraction of its range.
    body_ok = body_ratio >= cfg.body_ratio

    # H1 trend filter: longs need up-trend, shorts need down-trend.
    h1_t = h1["h1_trend"].values
    if cfg.h1_require == "yes":
        trend_ok_up = h1_t > cfg.h1_min
        trend_ok_dn = h1_t < -cfg.h1_min
    elif cfg.h1_require == "neutral":
        trend_ok_up = h1_t >= -cfg.h1_min
        trend_ok_dn = h1_t <= cfg.h1_min
    else:  # none
        trend_ok_up = np.ones(n, dtype=bool)
        trend_ok_dn = np.ones(n, dtype=bool)

    # Volatility band: keep ATR within [low, high] multiples of its rolling median.
    vol_ok = np.ones(n, dtype=bool)
    if cfg.vol_band:
        med = pd.Series(atr_safe).rolling(96).median().values
        ratio = atr_safe / np.where(med > 0, med, np.nan)
        vol_ok = (ratio >= cfg.vol_low) & (ratio <= cfg.vol_high)
        vol_ok = np.where(np.isnan(vol_ok), False, vol_ok)

    # Session filter (UTC hours).
    if cfg.session != "all":
        hrs = m5.index.hour.values
        if cfg.session == "london_ny":
            sess_ok = (hrs >= 13) & (hrs < 20)
        elif cfg.session == "ny":
            sess_ok = (hrs >= 14) & (hrs < 21)
        elif cfg.session == "asia_london":
            sess_ok = (hrs >= 7) & (hrs < 14)
        else:
            sess_ok = np.ones(n, dtype=bool)
    else:
        sess_ok = np.ones(n, dtype=bool)

    sig = np.full(n, 2, dtype=np.int64)
    buy = up_break & body_ok & trend_ok_up & vol_ok & sess_ok
    sell = dn_break & body_ok & trend_ok_dn & vol_ok & sess_ok
    # One trade per bar; avoid both at once.
    both = buy & sell
    buy[both] = False
    sig[buy] = 0
    sig[sell] = 1
    return sig


# --------------------------------------------------------------------------
# Mechanical rule: pullback-long (buy dips in H1 uptrend when vol is normal)
# --------------------------------------------------------------------------
def pullback_long_signal(m5, h1, cfg):
    rsi = m5["rsi"].values
    vol = m5["volatility_ratio"].values
    ht = h1["h1_trend"].values
    n = len(m5)

    sig = np.full(n, 2, dtype=np.int64)
    buy = (ht > cfg.h1_min) & (rsi < cfg.rsi_max) & (vol <= cfg.vol_max)
    if cfg.session != "all":
        hrs = m5.index.hour.values
        if cfg.session == "london_ny":
            buy &= (hrs >= 13) & (hrs < 20)
        elif cfg.session == "ny":
            buy &= (hrs >= 14) & (hrs < 21)
        elif cfg.session == "asia_london":
            buy &= (hrs >= 7) & (hrs < 14)
    sig[buy] = 0
    return sig


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def stats(rs, times):
    rs = np.asarray(rs, dtype=float)
    n = len(rs)
    if n == 0:
        return None
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    dd = float((peak - eq).max())
    return {
        "trades": n,
        "wr": 100.0 * (rs > 0).mean(),
        "exp": float(rs.mean()),
        "pf": float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        "net_r": float(rs.sum()),
        "dd": dd,
    }


def main(symbol, start_year, end_year, cfg):
    print(f"=== MECHANICAL backtest {symbol} {start_year}-{end_year} [rule={cfg.rule}] ===", flush=True)
    print(f"rule: body>={cfg.body_ratio} h1={cfg.h1_require}(>={cfg.h1_min}) "
          f"rsi<={cfg.rsi_max} vol<={cfg.vol_max} session={cfg.session} | "
          f"SL={cfg.sl_r}R TP={cfg.tp_r}R hold<={cfg.max_hold} cost={cfg.cost_r}R",
          flush=True)

    m1 = t.load_m1_data(symbol, start_year=start_year, end_year=end_year)
    print(f"  {len(m1):,} M1 bars", flush=True)
    m5 = t.resample_m5(m1)
    m5 = t.compute_features(m5)
    h1 = causal_h1_context(m5, m1)
    m5 = t.add_swing_features(m5)
    m5 = t.add_time_features(m5)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    del m1

    # The realized-R simulator reads these module globals at call time.
    t.SL_R = cfg.sl_r
    t.TP_R = cfg.tp_r
    t.COST_R = cfg.cost_r
    t.LABEL_WINDOW = cfg.max_hold
    long_r, short_r = t.compute_realized_r(m5, atr)

    sig = mechanical_signal(m5, h1, cfg) if cfg.rule == "breakout" else pullback_long_signal(m5, h1, cfg)
    n = len(sig)
    use = np.ones(n, dtype=bool)
    use[:t.SEQ_LEN] = False
    use[n - t.LABEL_WINDOW:] = False

    idx = np.where((sig != 2) & use)[0]
    r_vals = np.where(sig[idx] == 0, long_r[idx], short_r[idx])
    times = pd.to_datetime(m5.index.values[idx])

    # Consecutive-loss pause overlay: after `pause_consec` straight losing
    # trades, skip new entries for the next `pause_bars` M5 bars. This avoids
    # entering into the continuation of a failing streak (OOS-validated).
    if cfg.pause_consec > 0:
        keep = np.ones(len(idx), dtype=bool)
        skip_until = -1
        loss_streak = 0
        for k, (i, r) in enumerate(zip(idx, r_vals)):
            if i < skip_until:
                keep[k] = False
                continue
            if r > 0:
                loss_streak = 0
            else:
                loss_streak += 1
                if loss_streak >= cfg.pause_consec:
                    skip_until = i + cfg.pause_bars
                    loss_streak = 0
        idx = idx[keep]
        r_vals = r_vals[keep]
        times = times[keep]

    m = stats(r_vals, times)
    if m is None:
        print("  No trades — filters too tight.")
        return

    print(f"  Trades: {m['trades']} | WR {m['wr']:.1f}% | exp {m['exp']:+.3f}R | "
          f"PF {m['pf']:.2f} | net {m['net_r']:+.1f}R | maxDD {m['dd']:.1f}R", flush=True)

    daily = {}
    for i, ts in enumerate(times):
        daily.setdefault(ts.strftime("%Y-%m-%d"), []).append(r_vals[i])
    pos_days = sum(1 for k in daily if np.array(daily[k]).sum() > 0)
    print(f"  Daily consistency: {pos_days}/{len(daily)} positive days "
          f"({100.0*pos_days/max(len(daily),1):.1f}%)", flush=True)

    yearly = {}
    for i, ts in enumerate(times):
        yearly.setdefault(str(ts.year), []).append(r_vals[i])
    print("\n  Per-year:")
    for y in sorted(yearly):
        r = np.array(yearly[y])
        wins = r[r > 0].sum()
        losses = -r[r < 0].sum()
        pf = wins / losses if losses > 0 else float("inf")
        print(f"    {y}: trades={len(r):>5} WR={100.0*(r>0).mean():4.1f}% "
              f"exp={r.mean():+.3f}R PF={pf:4.2f} net={r.sum():+.1f}R")

    monthly = {}
    for i, ts in enumerate(times):
        monthly.setdefault(ts.strftime("%Y-%m"), []).append(r_vals[i])
    pos = sum(1 for k in monthly if np.array(monthly[k]).sum() > 0)
    print(f"\n  Monthly consistency: {pos}/{len(monthly)} positive "
          f"({100.0*pos/max(len(monthly),1):.0f}%)")

    # Regime breakdown using causal trend sign.
    regime = np.sign(h1["h1_trend"].values[idx])
    for rv, name in [(1.0, "UP-TREND"), (-1.0, "DOWN-TREND"), (0.0, "RANGE")]:
        mask = regime == rv
        if mask.sum():
            r = r_vals[mask]
            w = 100.0 * (r > 0).mean()
            print(f"    {name:>10}: {mask.sum():>5} trades WR={w:4.1f}% net={r.sum():+.1f}R")

    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--rule", choices=["breakout", "pullback_long"], default="pullback_long")
    p.add_argument("--start", type=int, default=2020)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--body-ratio", type=float, default=0.40)
    p.add_argument("--h1-require", choices=["yes", "neutral", "none"], default="yes")
    p.add_argument("--h1-min", type=float, default=1.0)
    p.add_argument("--rsi-max", type=float, default=0.35)
    p.add_argument("--vol-max", type=float, default=0.9)
    p.add_argument("--vol-band", action="store_true")
    p.add_argument("--vol-low", type=float, default=0.6)
    p.add_argument("--vol-high", type=float, default=1.6)
    p.add_argument("--session", choices=["all", "london_ny", "ny", "asia_london"], default="all")
    p.add_argument("--sl-r", type=float, default=1.0)
    p.add_argument("--tp-r", type=float, default=2.0)
    p.add_argument("--max-hold", type=int, default=24)
    p.add_argument("--cost-r", type=float, default=0.05)
    p.add_argument("--pause-consec", type=int, default=0,
                   help="after N straight losses, pause entries (0=off)")
    p.add_argument("--pause-bars", type=int, default=6,
                   help="bars to skip after pause triggers")
    args = p.parse_args()

    main(args.symbol, args.start, args.end, args)
