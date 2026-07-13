"""
News-event analysis for XAUUSD.
Detects high-volatility events (proxy for news) and measures
price behavior before/after to find the optimal trading approach.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dukascopy_client import DukascopyClient

YEARS = list(range(2018, 2026))
VOLATILITY_THRESHOLD_MULT = 3.0  # M5 move > 3x ATR = news event
LOOKBACK_M5 = 6   # 30 min before
LOOKAHEAD_M5 = 12 # 60 min after

def detect_news_events(m5: pd.DataFrame, h1: pd.DataFrame):
    """Find M5 bars where price move exceeds volatility threshold."""
    closes = m5["close"].values.astype(np.float64)
    highs = m5["high"].values.astype(np.float64)
    lows = m5["low"].values.astype(np.float64)
    
    tr = np.maximum(highs - lows,
        np.maximum(np.abs(highs - np.roll(closes, 1)),
                   np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(14).mean().bfill().values
    
    hilo_range = highs - lows
    move = np.abs(closes - np.roll(closes, 1))
    move[0] = 0
    
    threshold = atr * VOLATILITY_THRESHOLD_MULT
    events = np.where((move > threshold) & (atr > 0))[0]
    
    # Filter: keep only first event within a 2-hour window
    filtered = []
    min_gap_m5 = 24  # 2 hours
    for ev in events:
        if not filtered or (ev - filtered[-1]) >= min_gap_m5:
            filtered.append(ev)
    
    return filtered, atr

def analyze_events(m5: pd.DataFrame, events: list, atr: np.ndarray):
    """Analyze price behavior around each news event."""
    n = len(m5)
    closes = m5["close"].values.astype(np.float64)
    highs = m5["high"].values.astype(np.float64)
    lows = m5["low"].values.astype(np.float64)
    
    results = []
    for ev_idx in events:
        if ev_idx < LOOKBACK_M5 + 1 or ev_idx > n - LOOKAHEAD_M5 - 1:
            continue
        
        pre_start = ev_idx - LOOKBACK_M5
        event_price = closes[ev_idx]
        
        # --- Pre-event analysis ---
        pre_high = float(np.max(highs[pre_start:ev_idx]))
        pre_low = float(np.min(lows[pre_start:ev_idx]))
        pre_range = pre_high - pre_low
        pre_vol = float(np.std(closes[pre_start:ev_idx]))
        pre_atr = float(np.mean(atr[pre_start:ev_idx]))
        
        # False breakout count in pre-event
        false_breakouts = 0
        for i in range(pre_start + 1, ev_idx):
            h1_range = pre_high - pre_low
            if h1_range > 0:
                score = max((highs[i] - pre_high) / h1_range, (pre_low - lows[i]) / h1_range)
                if score > 0.7 and score < 1.0:
                    false_breakouts += 1
        
        # --- Event spike ---
        spike_high = highs[ev_idx]
        spike_low = lows[ev_idx]
        
        # --- Post-event analysis ---
        after_start = ev_idx + 1
        after_end = min(ev_idx + LOOKAHEAD_M5 + 1, n - 1)
        
        # Measure post-event trend consistency
        post_closes = closes[after_start:after_end]
        post_highs = highs[after_start:after_end]
        post_lows = lows[after_start:after_end]
        
        if len(post_closes) < 5:
            continue
        
        # Direction at event
        event_direction = 1 if closes[ev_idx] > closes[ev_idx - 1] else -1
        
        # Post-event direction after initial 2-bar settle
        settle_idx = min(ev_idx + 3, n - 1)
        if event_direction > 0:
            post_dir = 1 if closes[-1] > closes[settle_idx] else -1
        else:
            post_dir = 1 if closes[settle_idx] < closes[-1] else -1  # reversed: post-news continuation
        
        # Simplified: post-event trend direction (same as spike or reversal?)
        spike_bar = ev_idx
        next_bar = min(spike_bar + 1, n - 1)
        # The move from pre-event close to post-settle
        pre_close = closes[max(0, spike_bar - 1)]
        settle_close = closes[settle_idx]
        final_bar = min(spike_bar + LOOKAHEAD_M5, n - 1)
        final_close = closes[final_bar]
        
        # Move magnitudes
        spike_move = abs(closes[spike_bar] - pre_close)
        settle_to_final = final_close - settle_close
        settle_to_final_abs = abs(settle_to_final)
        
        # Did the direction established after 3 bars (settle) continue?
        if settle_to_final > 0 and (closes[settle_idx] > pre_close) or \
           settle_to_final < 0 and (closes[settle_idx] < pre_close):
            # Trend continuation
            trend_continues = settle_to_final * (closes[settle_idx] - pre_close) > 0
        else:
            trend_continues = False
        
        # Max favorable excursion after event
        if event_direction > 0:
            max_favorable = float(np.max(highs[after_start:after_end])) - event_price
            max_adverse = event_price - float(np.min(lows[after_start:after_end]))
        else:
            max_favorable = event_price - float(np.min(lows[after_start:after_end]))
            max_adverse = float(np.max(highs[after_start:after_end])) - event_price
        
        # Check if initial spike reverses within 3 bars
        reversed_3bars = False
        for i in range(spike_bar + 1, min(spike_bar + 4, n - 1)):
            if event_direction > 0 and lows[i] < pre_close:
                reversed_3bars = True
                break
            elif event_direction < 0 and highs[i] > pre_close:
                reversed_3bars = True
                break
        
        results.append({
            "event_time": m5.index[ev_idx],
            "event_idx": ev_idx,
            "event_price": event_price,
            "event_direction": event_direction,
            "spike_move_atr": spike_move / pre_atr if pre_atr > 0 else 0,
            "pre_range_atr": pre_range / pre_atr if pre_atr > 0 else 0,
            "pre_false_breakouts": false_breakouts,
            "post_trend_continues": trend_continues,
            "max_favorable_atr": max_favorable / pre_atr if pre_atr > 0 else 0,
            "max_adverse_atr": max_adverse / pre_atr if pre_atr > 0 else 0,
            "settle_to_final": settle_to_final,
            "settle_to_final_abs": settle_to_final_abs,
            "reversed_3bars": reversed_3bars,
        })
    
    return pd.DataFrame(results)

def main():
    print("Loading XAUUSD M1 data...")
    t0 = time.time()
    client = DukascopyClient()
    
    all_m5 = []
    
    for year in YEARS:
        m1 = client.download_year(year)
        if m1 is None or len(m1) == 0:
            continue
        m1 = m1.sort_values("time").drop_duplicates(subset="time")
        m1_idx = m1.set_index("time")
        m5 = m1_idx.resample("5min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "tick_volume": "sum",
        }).dropna()
        h1 = m1_idx.resample("1h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
        }).dropna()
        
        m5 = m5[~m5.index.duplicated(keep="first")]
        h1 = h1[~h1.index.duplicated(keep="first")]
        
        events_idx, atr = detect_news_events(m5, h1)
        print(f"  {year}: {len(m5)} M5 bars, {len(events_idx)} events")
        
        if len(events_idx) > 0:
            df = analyze_events(m5, events_idx, atr)
            df["year"] = year
            all_m5.append(df)
    
    if not all_m5:
        print("No events found!")
        return
    
    result = pd.concat(all_m5).reset_index(drop=True)
    
    # --- Summary Statistics ---
    print(f"\n{'='*70}")
    print(f"NEWS EVENT ANALYSIS: XAUUSD {YEARS[0]}-{YEARS[-1]}")
    print(f"{'='*70}")
    print(f"Total events detected: {len(result)}")
    print(f"Volatility threshold: {VOLATILITY_THRESHOLD_MULT}x ATR")
    print(f"Lookback: {LOOKBACK_M5*5}min | Lookahead: {LOOKAHEAD_M5*5}min")
    print()
    
    # 1. Pre-event behavior
    print(f"{'-'*70}")
    print(f"PRE-EVENT BEHAVIOR (30 min before)")
    print(f"{'-'*70}")
    print(f"  Avg range (ATR):          {result['pre_range_atr'].mean():.2f}")
    print(f"  Avg false breakouts:      {result['pre_false_breakouts'].mean():.1f}")
    print(f"  Events with false BOs:    {(result['pre_false_breakouts'] > 0).mean()*100:.1f}%")
    print(f"  False BO rate:            {result['pre_false_breakouts'].sum()/len(result):.2f} per event")
    print()
    
    # 2. Event spike
    print(f"{'-'*70}")
    print(f"EVENT SPIKE")
    print(f"{'-'*70}")
    print(f"  Avg spike (ATR):          {result['spike_move_atr'].mean():.2f}")
    print(f"  Max favorable (ATR):      {result['max_favorable_atr'].mean():.2f}")
    print(f"  Max adverse (ATR):        {result['max_adverse_atr'].mean():.2f}")
    print(f"  Favorable/Adverse ratio:  {(result['max_favorable_atr']/result['max_adverse_atr'].replace(0, np.nan)).mean():.2f}")
    print()
    
    # 3. Post-event direction
    print(f"{'-'*70}")
    print(f"POST-EVENT DIRECTION")
    print(f"{'-'*70}")
    
    # Trend continuation analysis
    continues = result[result['post_trend_continues'] == True]
    reverses = result[result['post_trend_continues'] == False]
    print(f"  Trend continues:          {len(continues)} ({len(continues)/len(result)*100:.1f}%)")
    print(f"  Trend reverses:           {len(reverses)} ({len(reverses)/len(result)*100:.1f}%)")
    print()
    
    # Initial reversal within 3 bars
    early_rev = result[result['reversed_3bars'] == True]
    no_rev = result[result['reversed_3bars'] == False]
    print(f"  Reversed within 3 bars:   {len(early_rev)} ({len(early_rev)/len(result)*100:.1f}%)")
    print(f"  Did NOT reverse 3 bars:   {len(no_rev)} ({len(no_rev)/len(result)*100:.1f}%)")
    
    if len(no_rev) > 0:
        print(f"  -> Non-reversed avg move:  {no_rev['settle_to_final_abs'].mean():.2f} pts")
    if len(early_rev) > 0:
        print(f"  -> Reversed avg move:      {early_rev['settle_to_final_abs'].mean():.2f} pts")
    print()
    
    # 4. Strategy comparison
    print(f"{'-'*70}")
    print(f"STRATEGY COMPARISON")
    print(f"{'-'*70}")
    
    # Strategy 1: Post-news + trend continuation (wait 2 bars, enter direction)
    # Take entry 2 bars after event, exit at end of window
    s1_trades = []
    for _, row in result.iterrows():
        # Only trade non-reversed events
        if not row["reversed_3bars"]:
            # Direction = direction after settle
            if row["settle_to_final"] > 0:
                s1_trades.append(1)
            else:
                s1_trades.append(-1)
    
    s1_win = sum(1 for t in s1_trades if t > 0)
    s1_loss = sum(1 for t in s1_trades if t < 0)
    s1_total = len(s1_trades)
    
    # Strategy 2: Both directions (enter immediately, take both sides)
    # Entry at event price, exit at max excursion 30 min later
    s2_trades = []
    for _, row in result.iterrows():
        mf = row["max_favorable_atr"]
        ma = row["max_adverse_atr"]
        if mf > 0 and ma > 0:
            # Long: if favorable > adverse, this wins
            s2_trades.append(mf > ma)
    
    s2_win = sum(1 for t in s2_trades if t)
    s2_total = len(s2_trades)
    
    # Strategy 3: Post-news + wider params (wait + widen stop/TP)
    # Use only continuation events, wider target
    s3_trades = continues
    s3_good = len(s3_trades[s3_trades["settle_to_final_abs"] > 0])
    s3_total = len(s3_trades)
    
    print(f"\n{'Strategy':40} {'Trades':>8} {'Win%':>8} {'Avg PnL':>10}")
    print(f"{'-'*70}")
    
    if s1_total > 0:
        print(f"{'1: Post-News Trend (wait 2 bars)':40} {s1_total:>8} {s1_win/s1_total*100:>7.1f}% {s1_trades.count(1)/s1_total:>10.3f}")
    
    if s2_total > 0:
        print(f"{'2: Both Directions (immediate entry)':40} {s2_total:>8} {s2_win/s2_total*100:>7.1f}%")
    
    if s3_total > 0:
        print(f"{'3: Post-News + Wider Params':40} {s3_total:>8} {s3_good/s3_total*100:>7.1f}% {s3_trades['settle_to_final_abs'].mean():>10.2f}")
    
    # --- Time distribution ---
    print(f"\n{'-'*70}")
    print(f"EVENT TIME DISTRIBUTION")
    print(f"{'-'*70}")
    event_hours = result["event_time"].dt.hour
    for h in range(24):
        ct = (event_hours == h).sum()
        if ct > 0:
            print(f"  {h:02d}:00 UTC:  {ct} events ({ct/len(result)*100:.1f}%)")
    
    print(f"\nDone. {len(result)} events analyzed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
