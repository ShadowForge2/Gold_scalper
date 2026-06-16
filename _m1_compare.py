#!/usr/bin/env python3
"""
M1 vs M5 comparison for January 2025.
Downloads M1 data from Dukascopy, runs backtest, compares with M5.
"""

import sys, os, lzma, struct, urllib.request
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from app.signal_engine import SignalEngine
from app.bias_engine import BiasEngine
from app.risk_manager import EquityScaler

# ── Constants ──
SYMBOL = "XAUUSD"
M1_CSV = os.path.join(os.path.dirname(__file__), "_m1_jan2025.csv")
TARGET_MONTH = (2025, 1)
INITIAL_BALANCE = 20.0
CONTRACT_SIZE = 100
SPREAD_PER_LOT_USD = 25.0
LEVERAGE = 500
MARGIN_MAX_PCT = 0.80

# ── Dukascopy Downloader ──

def dl_hour(symbol, year, month, day, hour):
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        raw = resp.read()
        if len(raw) < 20:
            return None
        return lzma.decompress(raw)
    except:
        return None

def parse_ticks(binary, base_dt):
    n = len(binary) // 20
    records = []
    for i in range(n):
        off = i * 20
        delta = struct.unpack(">I", binary[off:off+4])[0]
        ask = struct.unpack(">f", binary[off+4:off+8])[0]
        bid = struct.unpack(">f", binary[off+8:off+12])[0]
        ts = base_dt + timedelta(milliseconds=delta)
        records.append((ts, (ask + bid) / 2))
    return records

def download_m1_csv():
    if os.path.exists(M1_CSV):
        print(f"Loading cached {M1_CSV}")
        return pd.read_csv(M1_CSV, parse_dates=["time"])

    year, month = TARGET_MONTH
    days_in_month = 31
    print("Downloading Jan 2025 ticks from Dukascopy (30 threads)...")

    # Build list of (day, hour) tuples
    hours = [(day, hour) for day in range(1, days_in_month + 1) for hour in range(24)]
    results = {}

    with ThreadPoolExecutor(max_workers=30) as ex:
        fut_map = {}
        for day, hour in hours:
            fut = ex.submit(dl_hour, SYMBOL, year, month, day, hour)
            fut_map[fut] = (day, hour)

        done = 0
        for fut in as_completed(fut_map):
            day, hour = fut_map[fut]
            raw = fut.result()
            if raw and len(raw) >= 20:
                results[(day, hour)] = raw
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(hours)} hours downloaded ({len(results)} with data)")

    print(f"  {len(results)}/{len(hours)} hours with data")

    # Parse in order
    all_ticks = []
    for day in range(1, days_in_month + 1):
        for hour in range(24):
            raw = results.get((day, hour))
            if raw:
                base = datetime(year, month, day, hour)
                all_ticks.extend(parse_ticks(raw, base))

    print(f"Total ticks: {len(all_ticks):,}")

    # Aggregate to M1
    print("Aggregating to M1...")
    df = pd.DataFrame(all_ticks, columns=["time", "price"])
    df["time"] = pd.to_datetime(df["time"]).dt.floor("1min")
    m1 = df.groupby("time").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        tick_volume=("price", "count"),
    ).reset_index()
    m1.columns = ["time", "open", "high", "low", "close", "tick_volume"]
    m1["spread"] = 0
    m1["real_volume"] = 0
    print(f"M1 bars: {len(m1)}")

    m1.to_csv(M1_CSV, index=False)
    print(f"Saved to {M1_CSV}")
    return m1


# ── MT5 Data Helpers ──

def load_h1_mt5(start, end):
    import MetaTrader5 as mt5
    mt5.initialize(login=int(cfg.MT5_ACCOUNT), password=cfg.MT5_PASSWORD, server=cfg.MT5_SERVER)
    h1_raw = mt5.copy_rates_range(cfg.SYMBOL, mt5.TIMEFRAME_H1, start, end)
    mt5.shutdown()
    if h1_raw is None or len(h1_raw) == 0:
        return None
    h1 = pd.DataFrame(h1_raw)
    h1["time"] = pd.to_datetime(h1["time"], unit="s")
    return h1

def load_m5_mt5(start, end):
    import MetaTrader5 as mt5
    mt5.initialize(login=int(cfg.MT5_ACCOUNT), password=cfg.MT5_PASSWORD, server=cfg.MT5_SERVER)
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=7), end)
        raw = mt5.copy_rates_range(cfg.SYMBOL, 5, cursor, chunk_end)
        if raw is not None and len(raw) > 0:
            chunks.append(pd.DataFrame(raw))
        cursor = chunk_end
    mt5.shutdown()
    if not chunks:
        return None
    sig = pd.concat(chunks, ignore_index=True)
    sig.drop_duplicates(subset="time", keep="last", inplace=True)
    sig.reset_index(drop=True, inplace=True)
    sig["time"] = pd.to_datetime(sig["time"], unit="s")
    return sig


# ── Backtest Engine ──

def _pnl(entry_px, exit_px, direction, lot, num_trades):
    delta = exit_px - entry_px if direction == "BUY" else entry_px - exit_px
    gross = delta * CONTRACT_SIZE * lot * num_trades
    spread = SPREAD_PER_LOT_USD * lot * num_trades
    return gross - spread

def _tier(balance):
    return 3 if balance >= 100 else (2 if balance >= 50 else 1)

def _tier_params(t):
    return {1: (0.0, 0.5, 1, 1), 2: (0.0, 0.5, 2, 2), 3: (0.0, 0.5, 3, 3)}[t]

def _session_allowed(ts, sessions):
    if sessions.upper() == "ALL":
        return True
    h = ts.hour
    sessions = [s.strip().upper() for s in sessions.split(",")]
    if "ASIA" in sessions and 0 <= h < 8: return True
    if "LONDON" in sessions and 7 <= h < 16: return True
    if "NEW_YORK" in sessions and 12 <= h < 21: return True
    return False

def run_bt(sig_df, h1, label):
    print(f"\n--- Running {label} ---")

    # Pre-compute bias
    N = len(sig_df)
    bias_engine = BiasEngine()
    biases = [None] * N
    h1_times = h1["time"].values.astype(np.int64)
    sig_times = sig_df["time"].values.astype(np.int64)
    h1_idx = np.searchsorted(h1_times, sig_times, side="right") - 2
    h1_idx = np.clip(h1_idx, 0, len(h1) - 1)
    h1_high = h1["high"].values[h1_idx]
    h1_low = h1["low"].values[h1_idx]

    for idx in range(N):
        ts = sig_df["time"].iloc[idx]
        current_h1_start = ts.replace(minute=0, second=0, microsecond=0)
        hs = h1[h1["time"] < current_h1_start].tail(96)
        if len(hs) >= 20:
            biases[idx] = bias_engine.update(hs)
        elif idx > 0:
            biases[idx] = biases[idx - 1]

    # Pre-compute signals
    signal_engine = SignalEngine()
    sigs = [None] * N
    scores = np.zeros(N)
    for idx in range(10, N):
        bsum = biases[idx]
        if bsum is None: continue
        if bsum.get("bias") not in ("BULLISH", "BEARISH") or bsum.get("strength", 0) < cfg.BIAS_STRENGTH_MIN:
            continue
        row = sig_df.iloc[idx]
        mid = (row["high"] + row["low"]) / 2
        window = sig_df.iloc[max(0, idx - 50):idx]
        sig = signal_engine.evaluate(window, bsum, mid,
                                     h1_high=float(h1_high[idx]),
                                     h1_low=float(h1_low[idx]))
        if sig:
            sigs[idx] = sig
            scores[idx] = sig["score"]

    signal_count = sum(1 for s in sigs if s is not None)
    print(f"  Signals: {signal_count} / {N} candles")

    # Backtest loop
    scaler = EquityScaler()
    scaler.initialize(INITIAL_BALANCE)
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    trades = []
    cur = None
    consec_losses = 0
    cooldown = None
    day_trades = 0
    event_pnl = 0
    daily_pnl = 0
    cur_day = None

    params = {
        "entry_threshold": cfg.SIGNAL_ENTRY_THRESHOLD,
        "exit_threshold": cfg.EXIT_THRESHOLD_TIGHT,
        "sessions": cfg.ALLOWED_SESSIONS,
        "lot_mult": cfg.LOT_MULTIPLIER,
        "base_trades": cfg.MAX_TRADES_PER_EVENT,
        "max_trades": cfg.MAX_TRADES_PER_EVENT,
        "consec_loss_limit": cfg.MAX_CONSECUTIVE_LOSSES,
        "max_trades_per_session": cfg.MAX_TRADES_PER_SESSION,
        "event_loss_limit": cfg.MAX_EVENT_LOSS_USD,
        "daily_loss_limit": cfg.MAX_DAILY_LOSS_USD,
    }

    for idx in range(N):
        ts = sig_df["time"].iloc[idx]
        row = sig_df.iloc[idx]
        px = row["open"]

        if cur_day != ts.date():
            if cur_day is not None:
                print(f"  {cur_day}: {day_trades:3d} tr  ${balance:7.2f}  pk=${peak:7.2f}  dd=${peak - balance:5.2f}")
            cur_day = ts.date()
            day_trades = 0
            daily_pnl = 0
            consec_losses = 0

        if idx < 10: continue
        bsum = biases[idx]
        if bsum is None: continue

        # EXIT
        if cur is not None:
            event_pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"])
            exit_window = sig_df.iloc[max(0, idx - 10):idx]
            should_exit, exit_score, reason = signal_engine.evaluate_exit(
                exit_window, cur["entry_price"], cur["direction"], cur.get("entry_score"),
                exit_threshold=params["exit_threshold"], exit_mode=1)
            if should_exit:
                pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"])
                balance += pnl
                daily_pnl += pnl
                if balance > peak: peak = balance
                scaler.update_peak(balance)
                trades.append({"entry_time": cur["entry_time"], "exit_time": ts,
                    "direction": cur["direction"], "entry_price": cur["entry_price"],
                    "exit_price": round(px, 2), "pnl": round(pnl, 2),
                    "lot": cur["lot"], "bars_held": cur.get("bars_held", 0),
                    "exit_reason": reason, "balance": round(balance, 2)})
                consec_losses = 0 if pnl > 0 else consec_losses + 1
                cooldown = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + consec_losses))
                cur = None
                continue
            if event_pnl <= -params["event_loss_limit"]:
                pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"])
                balance += pnl; daily_pnl += pnl
                if balance > peak: peak = balance
                scaler.update_peak(balance)
                trades.append({"entry_time": cur["entry_time"], "exit_time": ts,
                    "direction": cur["direction"], "entry_price": cur["entry_price"],
                    "exit_price": round(px, 2), "pnl": round(pnl, 2),
                    "lot": cur["lot"], "bars_held": cur.get("bars_held", 0),
                    "exit_reason": "event_loss", "balance": round(balance, 2)})
                consec_losses += 1
                cooldown = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + consec_losses))
                cur = None
                continue
            cur["bars_held"] = cur.get("bars_held", 0) + 1

        if cooldown is not None and ts < cooldown: continue
        cooldown = None

        # ENTRY
        if cur is None:
            if not _session_allowed(ts, params["sessions"]): continue
            if consec_losses >= params["consec_loss_limit"] or balance <= 0: continue
            if daily_pnl <= -params["daily_loss_limit"]: continue
            if day_trades >= params["max_trades_per_session"]: continue
            if bsum.get("bias") not in ("BULLISH", "BEARISH"): continue
            if bsum.get("strength", 0) < cfg.BIAS_STRENGTH_MIN: continue

            score = scores[idx]
            if score >= params["entry_threshold"] and sigs[idx] is not None:
                scaler.base_trades = params["base_trades"]
                lot = scaler.get_lot(balance) * params["lot_mult"]
                t = _tier_params(_tier(balance))
                num_tr = min(t[3], params["max_trades"], scaler.get_trades_per_event(balance, score))
                total_lot = lot * num_tr
                margin_per_lot = (row["close"] * CONTRACT_SIZE) / LEVERAGE
                if total_lot * margin_per_lot > balance * MARGIN_MAX_PCT:
                    scale = (balance * MARGIN_MAX_PCT) / (total_lot * margin_per_lot)
                    lot *= scale; total_lot = lot * num_tr
                if total_lot <= 0: continue
                cur = {"entry_time": ts, "entry_price": px,
                    "direction": sigs[idx]["direction"], "lot": lot, "num_trades": num_tr,
                    "bars_held": 0, "entry_score": float(score)}
                event_pnl = 0; day_trades += 1
                print(f"  ENTRY {ts.strftime('%m/%d %H:%M')} {cur['direction']} @ {px:.2f} "
                      f"sc={score:.3f} l={lot:.5f} bal=${balance:.2f}")

    # Final close
    if cur is not None:
        mid = float(sig_df["close"].iloc[-1])
        pnl = _pnl(cur["entry_price"], mid, cur["direction"], cur["lot"], cur["num_trades"])
        balance += pnl
        trades.append({"entry_time": cur["entry_time"], "exit_time": sig_df["time"].iloc[-1],
            "direction": cur["direction"], "entry_price": cur["entry_price"],
            "exit_price": round(mid, 2), "pnl": round(pnl, 2),
            "lot": cur["lot"], "bars_held": cur.get("bars_held", 0),
            "exit_reason": "end", "balance": round(balance, 2)})

    print(f"  {cur_day}: {day_trades:3d} tr  ${balance:7.2f}  pk=${peak:7.2f}  dd=${peak - balance:5.2f}")

    if not trades:
        return None
    df = pd.DataFrame(trades)
    total = df["pnl"].sum()
    wins = df[df["pnl"] > 0]; losses = df[df["pnl"] < 0]
    wr = len(wins) / len(df) * 100 if len(df) > 0 else 0
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = losses["pnl"].sum() if len(losses) > 0 else 0
    pf = abs(gp / gl) if gl != 0 else float("inf")
    df["cumulative"] = df["pnl"].cumsum()
    df["peak_cum"] = df["cumulative"].cummax()
    max_dd = (df["peak_cum"] - df["cumulative"]).max()

    return {
        "trades": len(df), "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1), "net_pnl": round(total, 2),
        "profit_factor": round(pf, 2), "max_dd": round(max_dd, 2),
        "ending_balance": round(balance, 2),
        "return_pct": round((balance / INITIAL_BALANCE - 1) * 100, 2),
    }


# ── Main ──

if __name__ == "__main__":
    jan_start = datetime(2025, 1, 1)
    jan_end = datetime(2025, 2, 1)  # exclusive end

    # H1 data (shared)
    print("Loading H1 data from MT5...")
    h1 = load_h1_mt5(datetime(2024, 12, 15), jan_end)

    # ── M5 baseline ──
    print("Loading M5 data from MT5...")
    m5 = load_m5_mt5(jan_start, jan_end)
    result_m5 = run_bt(m5, h1, "M5")

    # ── M1 from Dukascopy ──
    m1 = download_m1_csv()
    result_m1 = run_bt(m1, h1, "M1")

    # ── Comparison ──
    print("\n" + "=" * 65)
    print("  JANUARY 2025 COMPARISON")
    print("=" * 65)
    hdr = f"{'':<12} {'Trades':<8} {'WR%':<7} {'PF':<8} {'Net P&L':<12} {'Max DD':<10} {'End Bal':<10}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in [("M5", result_m5), ("M1", result_m1)]:
        if r:
            print(f"{label:<12} {r['trades']:<8} {r['win_rate']:<7} {r['profit_factor']:<8.2f} "
                  f"${r['net_pnl']:<8.2f} ${r['max_dd']:<7.2f} ${r['ending_balance']:<7.2f}")
        else:
            print(f"{label:<12} NO TRADES")
    print("-" * len(hdr))
    print(f"Using H1 bias from MT5. {cfg.LOT_MULTIPLIER=}")
