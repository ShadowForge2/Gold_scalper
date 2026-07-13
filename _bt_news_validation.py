"""
News-aware strategy validation.
Compares entry quality around high-volatility events vs normal periods.
Measures: would blocking pre/spike and widening post improve outcomes?
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from app.dukascopy_client import DukascopyClient

YEARS = [2022, 2023, 2024, 2025]
VOL_THRESHOLD = 3.0  # 3x ATR on M5 = news event

def detect_events(m5):
    closes = m5["close"].values.astype(np.float64)
    highs = m5["high"].values.astype(np.float64)
    lows = m5["low"].values.astype(np.float64)
    tr = np.maximum(highs - lows,
        np.maximum(np.abs(highs - np.roll(closes, 1)),
                   np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(14).mean().bfill().values
    atr_smooth = pd.Series(atr).rolling(5).mean().bfill().values

    move = np.abs(closes - np.roll(closes, 1))
    move[0] = 0
    threshold = atr * VOL_THRESHOLD
    events = np.where((move > threshold) & (atr > 0))[0]

    # Dedup: keep first event within 2h window
    filtered = [(i, closes[i], highs[i], lows[i]) for i in events]
    unique = []
    min_gap = 24
    for idx, px, hi, lo in filtered:
        if not unique or (idx - unique[-1][0]) >= min_gap:
            unique.append((idx, px, hi, lo, atr[idx], atr_smooth[idx]))
    return unique, atr, atr_smooth


def simulate_entry(idx, m5, m1, h1, side):
    """Simulate breakout entry at given M5 bar."""
    if idx < 1 or idx >= len(m5) - 12:
        return None
    entry_bar = m5.iloc[idx]
    entry_price = entry_bar["close"]
    if side == "BUY":
        stop = entry_bar["low"]
        sl_dist = entry_price - stop
    else:
        stop = entry_bar["high"]
        sl_dist = stop - entry_price

    atr14 = m5["close"].rolling(14).std().iloc[idx]
    if pd.isna(atr14) or atr14 == 0 or sl_dist <= 0:
        return None

    # Forward test for 12 M5 bars (60 min)
    forward = m5.iloc[idx:idx + 12]
    if len(forward) < 5:
        return None

    if side == "BUY":
        max_price = forward["high"].max()
        min_price = forward["low"].min()
        exit_price = forward["close"].iloc[-1]
        pnl = exit_price - entry_price
        max_favorable = max_price - entry_price
        max_adverse = entry_price - min_price
    else:
        max_price = forward["high"].max()
        min_price = forward["low"].min()
        exit_price = forward["close"].iloc[-1]
        pnl = entry_price - exit_price
        max_favorable = entry_price - min_price
        max_adverse = max_price - entry_price

    return {
        "pnl": pnl,
        "pnl_atr": pnl / atr14 if atr14 > 0 else 0,
        "max_fav": max_favorable,
        "max_adv": max_adverse,
        "hit_sl": max_adverse >= sl_dist * 1.5,
        "exit_price": exit_price,
        "entry_price": entry_price,
        "sl_dist": sl_dist,
    }


def check_breakouts(m5, h1, events_idx):
    """Find breakout entries around events. Returns list of trade dicts."""
    trades = []
    event_set = set(events_idx)

    # Compute rolling H1 high/low
    h1_close = h1["close"]
    h1_high_roll = h1["high"].rolling(20, min_periods=5).max().bfill()
    h1_low_roll = h1["low"].rolling(20, min_periods=5).min().bfill()

    m5_index = m5.index
    m5_idx = 0

    for i in range(1, len(m5)):
        bar = m5.iloc[i]
        prev = m5.iloc[i - 1]
        px = bar["close"]

        # Find which H1 bar this M5 falls into
        while m5_idx < len(h1) - 1 and m5_index[i] >= h1.index[m5_idx + 1]:
            m5_idx += 1

        if m5_idx >= len(h1):
            continue

        h1_high = h1_high_roll.iloc[m5_idx - 1] if m5_idx >= 1 else h1_high_roll.iloc[m5_idx]
        h1_low = h1_low_roll.iloc[m5_idx - 1] if m5_idx >= 1 else h1_low_roll.iloc[m5_idx]

        if pd.isna(h1_high) or pd.isna(h1_low) or h1_high <= h1_low:
            continue

        range_size = h1_high - h1_low
        if range_size <= 0:
            continue

        # BUY breakout
        if prev["close"] <= h1_high and px > h1_high:
            score = (px - h1_high) / range_size
            if score >= 0.02:
                result = simulate_entry(i, m5, None, h1, "BUY")
                if result:
                    result.update({
                        "bar_idx": i, "time": m5_index[i], "side": "BUY",
                        "score": score, "h1_high": h1_high, "h1_low": h1_low,
                        "event_idx": _nearest_event(i, events_idx),
                    })
                    trades.append(result)

        # SELL breakout
        if prev["close"] >= h1_low and px < h1_low:
            score = (h1_low - px) / range_size
            if score >= 0.02:
                result = simulate_entry(i, m5, None, h1, "SELL")
                if result:
                    result.update({
                        "bar_idx": i, "time": m5_index[i], "side": "SELL",
                        "score": score, "h1_high": h1_high, "h1_low": h1_low,
                        "event_idx": _nearest_event(i, events_idx),
                    })
                    trades.append(result)

    return trades


def _nearest_event(bar_idx, events_idx):
    """Find nearest event and distance in M5 bars. Returns (event_bar_idx, distance_in_bars)."""
    if not events_idx:
        return None, 999
    nearest = min(events_idx, key=lambda ei: abs(ei - bar_idx))
    return nearest, bar_idx - nearest


def classify_trade(trade, pre_window=6, spike_window=1, post_window=12):
    """Classify trade relative to nearest event in M5 bars."""
    ev_idx, dist_bars = trade["event_idx"]
    if ev_idx is None or abs(dist_bars) > 24:
        return "NO_EVENT"
    if -pre_window <= dist_bars < 0:
        return "PRE_EVENT"
    if 0 <= dist_bars <= spike_window:
        return "SPIKE"
    if spike_window < dist_bars <= post_window:
        return "POST_EVENT"
    return "FAR"


def main():
    t0 = __import__("time").time()
    client = DukascopyClient()
    all_trades = []
    event_counts = {}

    for year in YEARS:
        m1 = client.download_year(year)
        if m1 is None or len(m1) == 0:
            continue
        m1 = m1.sort_values("time").drop_duplicates(subset="time")
        m1 = m1.set_index("time")

        m5 = m1.resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "tick_volume": "sum",
        }).dropna()
        m5 = m5[~m5.index.duplicated(keep="first")]

        h1 = m1.resample("1h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
        }).dropna()
        h1 = h1[~h1.index.duplicated(keep="first")]

        events, atr, atr_s = detect_events(m5)
        event_counts[year] = len(events)
        events_idx = [e[0] for e in events]

        trades = check_breakouts(m5, h1, events_idx)
        for t in trades:
            t["year"] = year
            t["event"] = _nearest_event(t["bar_idx"], events_idx)[0] or -1
            t["category"] = classify_trade(t)
            all_trades.append(t)

        elapsed = __import__("time").time() - t0
        print(f"  {year}: {len(m5)} M5 bars, {len(events)} events, {len(trades)} trades ({elapsed:.0f}s)")

    df = pd.DataFrame(all_trades)
    if len(df) == 0:
        print("No trades found!")
        return

    print(f"\n{'='*80}")
    print(f"NEWS STRATEGY VALIDATION: XAUUSD {YEARS[0]}-{YEARS[-1]}")
    print(f"{'='*80}")
    print(f"Total volatility events: {sum(event_counts.values())}")
    print(f"Total breakout trades:   {len(df)}")
    print(f"Spread: $0.30 (applied via Capital.com costs)")
    print()

    # Apply spread cost
    df["pnl_net"] = df["pnl"] - 0.30

    # Category stats
    print(f"{'Category':<20} {'Trades':>8} {'Win%':>8} {'Avg PnL':>10} {'Avg PnL/ATR':>12} {'PF':>8}")
    print(f"{'-'*70}")

    categories = ["NO_EVENT", "PRE_EVENT", "SPIKE", "POST_EVENT", "FAR"]
    for cat in categories:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        wins = (sub["pnl_net"] > 0).sum()
        total = len(sub)
        wr = wins / total * 100
        avg = sub["pnl_net"].mean()
        avg_atr = sub["pnl_atr"].mean()
        gross_win = sub[sub["pnl_net"] > 0]["pnl_net"].sum() if wins > 0 else 0
        gross_loss = abs(sub[sub["pnl_net"] <= 0]["pnl_net"].sum()) if (total - wins) > 0 else 1
        pf = gross_win / gross_loss if gross_loss > 0 else 99
        print(f"{cat:<20} {total:>8} {wr:>7.1f}% ${avg:>+7.2f} {avg_atr:>+11.3f} {pf:>7.2f}")

    # Combined: news-aware strategy vs baseline
    baseline = df[df["category"] == "NO_EVENT"]
    blocked = df[df["category"].isin(["PRE_EVENT", "SPIKE"])]
    widened = df[df["category"] == "POST_EVENT"]

    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*80}")

    # Baseline (no-event trades)
    b_wins = (baseline["pnl_net"] > 0).sum() if len(baseline) > 0 else 0
    b_total = len(baseline)
    b_wr = b_wins / b_total * 100 if b_total > 0 else 0
    b_avg = baseline["pnl_net"].mean() if b_total > 0 else 0
    b_gw = baseline[baseline["pnl_net"] > 0]["pnl_net"].sum() if b_total > 0 and b_wins > 0 else 0
    b_gl = abs(baseline[baseline["pnl_net"] <= 0]["pnl_net"].sum()) if b_total > 0 and (b_total - b_wins) > 0 else 1
    b_pf = b_gw / b_gl if b_gl > 0 else 99
    print(f"\nBaseline (no nearby event):")
    print(f"  Trades: {b_total} | Win%: {b_wr:.1f}% | Avg PnL: ${b_avg:+.2f} | PF: {b_pf:.2f}")

    # Pre-event trades (would be BLOCKED)
    blk_wins = (blocked["pnl_net"] > 0).sum() if len(blocked) > 0 else 0
    blk_total = len(blocked)
    blk_wr = blk_wins / blk_total * 100 if blk_total > 0 else 0
    blk_avg = blocked["pnl_net"].mean() if blk_total > 0 else 0
    blk_gw = blocked[blocked["pnl_net"] > 0]["pnl_net"].sum() if blk_total > 0 and blk_wins > 0 else 0
    blk_gl = abs(blocked[blocked["pnl_net"] <= 0]["pnl_net"].sum()) if blk_total > 0 and (blk_total - blk_wins) > 0 else 1
    blk_pf = blk_gw / blk_gl if blk_gl > 0 else 99
    print(f"\nBlocked (pre-event + spike - would avoid):")
    print(f"  Trades: {blk_total} | Win%: {blk_wr:.1f}% | Avg PnL: ${blk_avg:+.2f} | PF: {blk_pf:.2f}")

    # Post-event trades (would be WIDENED SL/TP)
    pw_wins = (widened["pnl_net"] > 0).sum() if len(widened) > 0 else 0
    pw_total = len(widened)
    pw_wr = pw_wins / pw_total * 100 if pw_total > 0 else 0
    pw_avg = widened["pnl_net"].mean() if pw_total > 0 else 0
    pw_gw = widened[widened["pnl_net"] > 0]["pnl_net"].sum() if pw_total > 0 and pw_wins > 0 else 0
    pw_gl = abs(widened[widened["pnl_net"] <= 0]["pnl_net"].sum()) if pw_total > 0 and (pw_total - pw_wins) > 0 else 1
    pw_pf = pw_gw / pw_gl if pw_gl > 0 else 99

    # Simulated widened PnL (1.5x TP, 1.5x SL on post-event trades)
    widened["pnl_wider"] = widened.apply(
        lambda r: r["pnl_net"] * 1.5 if r["pnl_net"] > 0 else r["pnl_net"] * 1.5, axis=1
    )
    ww_wins = (widened["pnl_wider"] > 0).sum()
    ww_total = len(widened)
    ww_wr = ww_wins / ww_total * 100
    ww_avg = widened["pnl_wider"].mean()
    ww_gw = widened[widened["pnl_wider"] > 0]["pnl_wider"].sum() if ww_wins > 0 else 0
    ww_gl = abs(widened[widened["pnl_wider"] <= 0]["pnl_wider"].sum()) if (ww_total - ww_wins) > 0 else 1
    ww_pf = ww_gw / ww_gl if ww_gl > 0 else 99

    print(f"\nPost-event (normal params):")
    print(f"  Trades: {pw_total} | Win%: {pw_wr:.1f}% | Avg PnL: ${pw_avg:+.2f} | PF: {pw_pf:.2f}")

    print(f"\nPost-event (wider 1.5x SL/TP):")
    print(f"  Trades: {ww_total} | Win%: {ww_wr:.1f}% | Avg PnL: ${ww_avg:+.2f} | PF: {ww_pf:.2f}")

    # Combined strategy simulation
    combined_pnl = baseline["pnl_net"].sum() + widened["pnl_wider"].sum()
    baseline_pnl = df["pnl_net"].sum()
    avoided_loss = abs(blocked[blocked["pnl_net"] < 0]["pnl_net"].sum())
    print(f"\n{'='*80}")
    print(f"NET PnL COMPARISON")
    print(f"{'='*80}")
    print(f"  Baseline (take all trades):                ${baseline_pnl:>+10.2f}")
    print(f"  Avoided losses (blocked trades):           ${avoided_loss:>+10.2f}")
    print(f"  News-aware (block pre/spike + widen post): ${combined_pnl:>+10.2f}")
    print(f"  Improvement:                               ${combined_pnl - baseline_pnl:>+10.2f}")

    print(f"\nDone in {__import__('time').time()-t0:.1f}s")


if __name__ == "__main__":
    main()
