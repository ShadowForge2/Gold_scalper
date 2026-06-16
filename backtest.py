"""
Multi-TF directional backtest — adapted for XAUUSD.
H1 bias filter + M5 momentum entry + anti-candle exit.
Note: Uses M5 (not M1) because demo accounts lack deep M1 history.
      Live bot still uses M1 for finer granularity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from app.signal_engine import SignalEngine
from app.bias_engine import BiasEngine
from app.risk_manager import EquityScaler
from app.capital_client import CapitalClient

BACKTEST_BROKER = "MT5"       # "MT5", "MT5_H1", "CAPITAL", "YAHOO", or "YAHOO_H1"
BACKTEST_YEAR = 2025          # Full year H1 validation
BACKTEST_START = datetime(2025, 1, 1)
BACKTEST_END = datetime(2025, 12, 31, 23, 59)
BIAS_WARMUP_DAYS = 14
INITIAL_BALANCE = 20.0
BACKTEST_CASH_FLOWS = []     # e.g. [{"time": "2025-07-01 00:00", "amount": -500.0}]
CONTRACT_SIZE = 100        # XAUUSD: 1 standard lot = 100 oz
SPREAD_PER_LOT_USD = 25.0  # ~$25 spread round trip per lot on XAUUSD
LEVERAGE = 500              # high leverage to reduce margin per lot
MARGIN_MAX_PCT = 0.80       # cap margin used at 80% of equity

# Timeframe constants (same values as MT5 constants)
TIMEFRAME_H1 = 16385
TIMEFRAME_M5 = 5

# ── helpers ──

def _session_allowed(ts, sessions_str):
    if sessions_str.upper() == "ALL":
        return True
    h = ts.hour
    sessions = [s.strip().upper() for s in sessions_str.split(",")]
    if "ASIA" in sessions and 0 <= h < 8:
        return True
    if "LONDON" in sessions and 7 <= h < 16:
        return True
    if "NEW_YORK" in sessions and 12 <= h < 21:
        return True
    return False

def _pnl(entry_px, exit_px, direction, lot, num_trades):
    delta = exit_px - entry_px if direction == "BUY" else entry_px - exit_px
    gross = delta * CONTRACT_SIZE * lot * num_trades
    spread = SPREAD_PER_LOT_USD * lot * num_trades
    return gross - spread

# ── equity tiers ──

def _get_tier(balance):
    if balance >= 100.0:
        return 3
    elif balance >= 50.0:
        return 2
    return 1

def _tier_params(tier):
    tiers = {
        1: {"entry_threshold": 0.00, "exit_threshold": 0.50, "max_trades": 1, "num_trades": 1},
        2: {"entry_threshold": 0.00, "exit_threshold": 0.50, "max_trades": 2, "num_trades": 2},
        3: {"entry_threshold": 0.00, "exit_threshold": 0.50, "max_trades": 3, "num_trades": 3},
    }
    return tiers.get(tier, tiers[1])

def _normalize_cash_flows(cash_flows):
    normalized = []
    for flow in cash_flows or []:
        if isinstance(flow, dict):
            when = flow.get("time") or flow.get("date")
            amount = flow.get("amount", 0.0)
        else:
            when, amount = flow
        when = pd.to_datetime(when).to_pydatetime()
        normalized.append((when, float(amount)))
    normalized.sort(key=lambda x: x[0])
    return normalized

# ── data loading & pre-computation ──

def _load_from_capital():
    client = CapitalClient()
    if not client.initialize(api_key=cfg.CAPITAL_API_KEY,
                             identifier=cfg.CAPITAL_IDENTIFIER,
                             password=cfg.CAPITAL_PASSWORD,
                             demo=cfg.CAPITAL_DEMO):
        print("Capital.com init failed")
        return None

    h1_fr = BACKTEST_START - timedelta(days=BIAS_WARMUP_DAYS)
    print("Downloading H1 data from Capital.com...", flush=True)
    h1 = client.get_rates_range(cfg.SYMBOL, TIMEFRAME_H1, h1_fr, BACKTEST_END)
    print("Downloading M5 data from Capital.com...", flush=True)
    sig_df = client.get_rates_range(cfg.SYMBOL, TIMEFRAME_M5, BACKTEST_START, BACKTEST_END)
    client.shutdown()

    if h1 is None or sig_df is None or len(h1) == 0 or len(sig_df) == 0:
        print(f"Data download failed: H1={'OK' if h1 is not None else 'NOK'} SIG={'OK' if sig_df is not None else 'NOK'}")
        return None

    return _pre_compute(sig_df, h1)

def _load_from_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not installed")
        return None

    acct = cfg.MT5_ACCOUNT
    if not mt5.initialize(login=int(acct), password=cfg.MT5_PASSWORD, server=cfg.MT5_SERVER):
        print(f"MT5 init failed: {mt5.last_error()}")
        return None

    h1_fr = BACKTEST_START - timedelta(days=BIAS_WARMUP_DAYS)
    print("Downloading H1 data from MT5...", flush=True)
    h1_raw = mt5.copy_rates_range(cfg.SYMBOL, mt5.TIMEFRAME_H1, h1_fr, BACKTEST_END)
    print("Downloading M5 data (monthly chunks) from MT5...", flush=True)
    chunks = []
    cursor = BACKTEST_START
    while cursor < BACKTEST_END:
        chunk_end = min(cursor + timedelta(days=28), BACKTEST_END)
        raw = mt5.copy_rates_range(cfg.SYMBOL, TIMEFRAME_M5, cursor, chunk_end)
        if raw is not None and len(raw) > 0:
            chunks.append(pd.DataFrame(raw))
        else:
            print(f"  WARN: no data for {cursor.date()} to {chunk_end.date()}", flush=True)
        cursor = chunk_end
    mt5.shutdown()

    if not chunks:
        return None
    sig_df = pd.concat(chunks, ignore_index=True)
    sig_df.drop_duplicates(subset="time", keep="last", inplace=True)
    sig_df.reset_index(drop=True, inplace=True)

    if h1_raw is None or sig_df is None:
        print(f"Data download failed: H1={'OK' if h1_raw is not None else 'NOK'} SIG={'OK' if sig_df is not None else 'NOK'}")
        return None

    sig_df["time"] = pd.to_datetime(sig_df["time"], unit="s")
    h1 = pd.DataFrame(h1_raw)
    h1["time"] = pd.to_datetime(h1["time"], unit="s")
    return _pre_compute(sig_df, h1)

def _load_from_mt5_h1():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not installed")
        return None

    acct = cfg.MT5_ACCOUNT
    if not mt5.initialize(login=int(acct), password=cfg.MT5_PASSWORD, server=cfg.MT5_SERVER):
        print(f"MT5 init failed: {mt5.last_error()}")
        return None

    h1_fr = BACKTEST_START - timedelta(days=BIAS_WARMUP_DAYS)
    print("Downloading H1 data from MT5...", flush=True)
    h1_raw = mt5.copy_rates_range(cfg.SYMBOL, mt5.TIMEFRAME_H1, h1_fr, BACKTEST_END)
    mt5.shutdown()

    if h1_raw is None or len(h1_raw) == 0:
        print("No H1 data available")
        return None

    h1 = pd.DataFrame(h1_raw)
    h1["time"] = pd.to_datetime(h1["time"], unit="s")
    return _pre_compute_h1(h1)

def _pre_compute(sig_df, h1):
    N = len(sig_df)
    print(f"Loaded {N} signal candles, {len(h1)} H1 candles", flush=True)

    h1_times_ns = h1["time"].values.astype(np.int64)
    sig_times_ns = sig_df["time"].values.astype(np.int64)
    h1_idx = np.searchsorted(h1_times_ns, sig_times_ns, side="right") - 2
    h1_idx = np.clip(h1_idx, 0, len(h1) - 1)
    h1_high_arr = h1["high"].values[h1_idx]
    h1_low_arr = h1["low"].values[h1_idx]

    bias_engine = BiasEngine()
    biases = [None] * N
    last_bias_hour = -1

    for idx in range(N):
        if idx % 20000 == 0:
            print(f"  bias pass: {idx}/{N}", flush=True)
        ts = sig_df["time"].iloc[idx]
        bh = ts.hour
        if bh != last_bias_hour or biases[idx - 1] is None:
            last_bias_hour = bh
            current_h1_start = ts.replace(minute=0, second=0, microsecond=0)
            hs = h1[h1["time"] < current_h1_start].tail(96)
            if len(hs) >= 20:
                biases[idx] = bias_engine.update(hs)
            elif idx > 0:
                biases[idx] = biases[idx - 1]
        else:
            biases[idx] = biases[idx - 1] if idx > 0 else None

    print("Pre-computing entry signals...", flush=True)
    signal_engine = SignalEngine()
    sig_signal = [None] * N
    sig_score = np.zeros(N)

    for idx in range(10, N):
        if idx % 10000 == 0:
            print(f"  signal pass: {idx}/{N}", flush=True)
        bsum = biases[idx]
        if bsum is None:
            continue
        if bsum.get("bias") not in ("BULLISH", "BEARISH") or bsum.get("strength", 0) < cfg.BIAS_STRENGTH_MIN:
            continue

        row = sig_df.iloc[idx]
        mid = (row["high"] + row["low"]) / 2
        window = sig_df.iloc[max(0, idx - 50):idx]
        sig = signal_engine.evaluate(window, bsum, mid,
                                     h1_high=float(h1_high_arr[idx]),
                                     h1_low=float(h1_low_arr[idx]))
        if sig:
            sig_signal[idx] = sig
            sig_score[idx] = sig["score"]

    signal_count = sum(1 for s in sig_signal if s is not None)
    print(f"  {signal_count} signals over {N} candles", flush=True)

    return {
        "sig_df": sig_df,
        "biases": biases,
        "sig_signal": sig_signal,
        "sig_score": sig_score,
        "signal_engine": signal_engine,
    }

def _load_from_yahoo():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed")
        return None

    h1_fr = BACKTEST_START - timedelta(days=BIAS_WARMUP_DAYS)
    ticker = "GC=F"
    print(f"Downloading H1 data from Yahoo ({ticker})...", flush=True)
    h1_raw = yf.download(ticker, start=h1_fr, end=BACKTEST_END, interval="1h", progress=False)
    print(f"Downloading M5 data from Yahoo ({ticker})...", flush=True)
    sig_raw = yf.download(ticker, start=BACKTEST_START, end=BACKTEST_END, interval="5m", progress=False)

    if h1_raw.empty or sig_raw.empty:
        print(f"Data download failed: H1={'OK' if not h1_raw.empty else 'NOK'} SIG={'OK' if not sig_raw.empty else 'NOK'}")
        return None

    def _flatten_yf(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "tick_volume"})
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        df.insert(0, "time", df.index)
        df["tick_volume"] = df["tick_volume"].fillna(0).astype(int)
        df["spread"] = 0
        df["real_volume"] = 0
        df.reset_index(drop=True, inplace=True)
        return df

    h1 = _flatten_yf(h1_raw)
    sig_df = _flatten_yf(sig_raw)
    return _pre_compute(sig_df, h1)

def _load_from_yahoo_h1():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed")
        return None

    h1_fr = BACKTEST_START - timedelta(days=BIAS_WARMUP_DAYS)
    ticker = "GC=F"
    print(f"Downloading H1 data from Yahoo ({ticker})...", flush=True)
    h1_raw = yf.download(ticker, start=h1_fr, end=BACKTEST_END, interval="1h", progress=False)

    if h1_raw.empty:
        print("Data download failed")
        return None

    def _flatten_yf(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "tick_volume"})
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        df.insert(0, "time", df.index)
        df["tick_volume"] = df["tick_volume"].fillna(0).astype(int)
        df["spread"] = 0
        df["real_volume"] = 0
        df.reset_index(drop=True, inplace=True)
        return df

    h1 = _flatten_yf(h1_raw)
    return _pre_compute_h1(h1)

def _pre_compute_h1(h1):
    N = len(h1)
    print(f"Loaded {N} H1 candles (H1-only mode)", flush=True)

    h1_times_ns = h1["time"].values.astype(np.int64)
    h1_idx = np.clip(np.arange(N) - 2, 0, N - 1)
    h1_high_arr = h1["high"].values[h1_idx]
    h1_low_arr = h1["low"].values[h1_idx]

    bias_engine = BiasEngine()
    biases = [None] * N
    last_bias_hour = -1

    for idx in range(N):
        if idx % 5000 == 0:
            print(f"  bias pass: {idx}/{N}", flush=True)
        ts = h1["time"].iloc[idx]
        bh = ts.hour
        if bh != last_bias_hour or biases[idx - 1] is None:
            last_bias_hour = bh
            current_h1_start = ts.replace(minute=0, second=0, microsecond=0)
            hs = h1[h1["time"] < current_h1_start].tail(96)
            if len(hs) >= 20:
                biases[idx] = bias_engine.update(hs)
            elif idx > 0:
                biases[idx] = biases[idx - 1]
        else:
            biases[idx] = biases[idx - 1] if idx > 0 else None

    print("Pre-computing entry signals...", flush=True)
    signal_engine = SignalEngine()
    sig_signal = [None] * N
    sig_score = np.zeros(N)

    for idx in range(10, N):
        if idx % 5000 == 0:
            print(f"  signal pass: {idx}/{N}", flush=True)
        bsum = biases[idx]
        if bsum is None:
            continue
        if bsum.get("bias") not in ("BULLISH", "BEARISH") or bsum.get("strength", 0) < cfg.BIAS_STRENGTH_MIN:
            continue

        row = h1.iloc[idx]
        mid = (row["high"] + row["low"]) / 2
        window = h1.iloc[max(0, idx - 50):idx]
        sig = signal_engine.evaluate(window, bsum, mid,
                                     h1_high=float(h1_high_arr[idx]),
                                     h1_low=float(h1_low_arr[idx]))
        if sig:
            sig_signal[idx] = sig
            sig_score[idx] = sig["score"]

    signal_count = sum(1 for s in sig_signal if s is not None)
    print(f"  {signal_count} signals over {N} candles", flush=True)

    return {
        "sig_df": h1,
        "biases": biases,
        "sig_signal": sig_signal,
        "sig_score": sig_score,
        "signal_engine": signal_engine,
    }

def load_and_compute():
    if BACKTEST_BROKER == "CAPITAL":
        return _load_from_capital()
    elif BACKTEST_BROKER == "YAHOO":
        return _load_from_yahoo()
    elif BACKTEST_BROKER == "YAHOO_H1":
        return _load_from_yahoo_h1()
    elif BACKTEST_BROKER == "MT5_H1":
        return _load_from_mt5_h1()
    else:
        return _load_from_mt5()

# ── backtest runner ──

def run_backtest(data: dict, params: dict = None, verbose: bool = True):
    p = {
        "entry_threshold": cfg.SIGNAL_ENTRY_THRESHOLD,
        "exit_threshold": cfg.EXIT_THRESHOLD_TIGHT,
        "base_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
        "sessions": cfg.ALLOWED_SESSIONS,
        "bias_strength_min": cfg.BIAS_STRENGTH_MIN,
        "lot_multiplier": cfg.LOT_MULTIPLIER,
        "max_total_lot_per_event": max(1.0, INITIAL_BALANCE / 20.0),
        "max_trades_per_event": cfg.MAX_TRADES_PER_EVENT,
        "event_loss_limit": cfg.MAX_EVENT_LOSS_USD,
        "consecutive_loss_limit": cfg.MAX_CONSECUTIVE_LOSSES,
        "daily_loss_limit": cfg.MAX_DAILY_LOSS_USD,
        "max_trades_per_session": cfg.MAX_TRADES_PER_SESSION,
        "min_balance": cfg.MIN_BALANCE,
        "cash_flows": BACKTEST_CASH_FLOWS,
        "exit_mode": 4 if BACKTEST_BROKER in ("YAHOO_H1", "MT5_H1") else 1,  # ATR-based for H1, trail_sl for M5
    }
    if params:
        p.update(params)

    et = p["entry_threshold"]
    ext = p["exit_threshold"]
    base_tr = p["base_trades_per_event"]
    sessions = p["sessions"]
    bias_min = p["bias_strength_min"]
    lot_mult = p["lot_multiplier"]
    max_total_lot = p["max_total_lot_per_event"]
    max_trades = p["max_trades_per_event"]
    event_loss_limit = p["event_loss_limit"]
    consec_loss_limit = p["consecutive_loss_limit"]
    daily_loss_limit = p["daily_loss_limit"]
    max_trades_per_session = p["max_trades_per_session"]
    min_balance = p["min_balance"]
    cash_flows = _normalize_cash_flows(p["cash_flows"])
    exit_mode = p["exit_mode"]

    sig_df = data["sig_df"]
    biases = data["biases"]
    sigs = data["sig_signal"]
    scores = data["sig_score"]
    signal_engine = data["signal_engine"]
    N = len(sig_df)

    if verbose:
        print(f"entry>={et} exit<={ext:.2f} tr={base_tr} mult={lot_mult} {sessions}", flush=True)

    scaler = EquityScaler()
    scaler.initialize(INITIAL_BALANCE)

    trades = []
    cur = None
    balance = INITIAL_BALANCE
    consec_losses = 0
    daily_trades = 0
    cur_day = None
    peak = INITIAL_BALANCE
    cooldown = None
    event_pnl = 0.0
    daily_pnl = 0.0
    cash_idx = 0
    deposits = 0.0
    withdrawals = 0.0

    for idx in range(N):
        ts = sig_df["time"].iloc[idx]
        row = sig_df.iloc[idx]
        px = row["open"]
        tier = _get_tier(balance)
        tp = _tier_params(tier)

        # daily header
        if cur_day != ts.date():
            if cur_day is not None and verbose:
                print(f"  {cur_day}: {daily_trades:3d} tr  ${balance:7.2f}  pk=${peak:7.2f}  dd=${peak - balance:5.2f}", flush=True)
            cur_day = ts.date()
            daily_trades = 0
            daily_pnl = 0.0
            consec_losses = 0

        while cash_idx < len(cash_flows) and cash_flows[cash_idx][0] <= ts:
            cash_time, cash_amount = cash_flows[cash_idx]
            balance += cash_amount
            if cash_amount >= 0:
                deposits += cash_amount
            else:
                withdrawals += abs(cash_amount)
            if balance > peak:
                peak = balance
            scaler.update_peak(balance)
            if verbose:
                action = "DEPOSIT" if cash_amount >= 0 else "WITHDRAWAL"
                print(f"  {action} {cash_time.strftime('%m/%d %H:%M')} "
                      f"${abs(cash_amount):.2f} -> bal=${balance:.2f}", flush=True)
            cash_idx += 1

        if idx < 10:
            continue

        bsum = biases[idx]
        if bsum is None:
            continue

        # ── EXIT ──
        if cur is not None:
            event_pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"])
            exit_window = sig_df.iloc[max(0, idx - 10):idx]
            should_exit, exit_score_val, reason = signal_engine.evaluate_exit(
                exit_window, cur["entry_price"], cur["direction"], cur.get("entry_score"),
                exit_threshold=tp["exit_threshold"], exit_mode=exit_mode
            )

            if should_exit:
                pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"])
                balance += pnl
                daily_pnl += pnl
                if balance > peak:
                    peak = balance
                scaler.update_peak(balance)
                trades.append({
                    "entry_time": cur["entry_time"], "exit_time": ts,
                    "direction": cur["direction"], "entry_price": cur["entry_price"],
                    "exit_price": round(px, 2), "pnl": round(pnl, 2),
                    "lot": cur["lot"], "num_trades": cur["num_trades"],
                    "bars_held": cur.get("bars_held", 0), "exit_reason": reason,
                    "exit_score": exit_score_val, "balance": round(balance, 2),
                })
                consec_losses = 0 if pnl > 0 else consec_losses + 1
                cooldown = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + consec_losses))
                cur = None
                continue

            if event_pnl <= -event_loss_limit:
                pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"])
                balance += pnl
                daily_pnl += pnl
                if balance > peak:
                    peak = balance
                scaler.update_peak(balance)
                trades.append({
                    "entry_time": cur["entry_time"], "exit_time": ts,
                    "direction": cur["direction"], "entry_price": cur["entry_price"],
                    "exit_price": round(px, 2), "pnl": round(pnl, 2),
                    "lot": cur["lot"], "num_trades": cur["num_trades"],
                    "bars_held": cur.get("bars_held", 0), "exit_reason": "event_loss",
                    "exit_score": 1.0, "balance": round(balance, 2),
                })
                consec_losses += 1
                cooldown = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + consec_losses))
                cur = None
                continue

            cur["bars_held"] = cur.get("bars_held", 0) + 1

        if cooldown is not None and ts < cooldown:
            continue
        cooldown = None

        # ── ENTRY ──
        if cur is None:
            if not _session_allowed(ts, sessions):
                continue
            if consec_losses >= consec_loss_limit or balance < min_balance:
                continue
            if daily_pnl <= -daily_loss_limit:
                continue
            if daily_trades >= max_trades_per_session:
                continue
            if bsum.get("bias") not in ("BULLISH", "BEARISH"):
                continue
            if bsum.get("strength", 0) < bias_min:
                continue

            score = scores[idx]
            tier_et = max(et, tp["entry_threshold"])
            if score >= tier_et and sigs[idx] is not None:
                scaler.base_trades = base_tr
                lot = scaler.get_lot(balance) * lot_mult
                num_tr = min(tp["num_trades"], max_trades, scaler.get_trades_per_event(balance, score))
                # cap total exposure per event
                total_lot = lot * num_tr
                if total_lot > max_total_lot:
                    num_tr = max(1, int(max_total_lot / lot))
                    total_lot = lot * num_tr
                if total_lot > max_total_lot:
                    lot = max_total_lot / num_tr
                # margin check
                margin_per_lot = (row["close"] * CONTRACT_SIZE) / LEVERAGE
                margin_needed = total_lot * margin_per_lot
                if margin_needed > balance * MARGIN_MAX_PCT:
                    scale = (balance * MARGIN_MAX_PCT) / margin_needed
                    lot *= scale
                    total_lot = lot * num_tr
                if total_lot <= 0:
                    continue

                cur = {
                    "entry_time": ts, "entry_price": px,
                    "direction": sigs[idx]["direction"],
                    "lot": lot, "num_trades": num_tr,
                    "bars_held": 0, "entry_score": float(score),
                }
                event_pnl = 0.0
                daily_trades += 1
                if verbose:
                    print(f"  ENTRY {ts.strftime('%m/%d %H:%M')} {cur['direction']} "
                          f"@ {px:.2f} sc={score:.3f} l={lot:.5f} n={num_tr} "
                          f"x={lot * num_tr:.5f} bal=${balance:.2f}", flush=True)

    # final close
    if cur is not None:
        mid = float(sig_df["close"].iloc[-1])
        pnl = _pnl(cur["entry_price"], mid, cur["direction"], cur["lot"], cur["num_trades"])
        balance += pnl
        trades.append({
            "entry_time": cur["entry_time"], "exit_time": sig_df["time"].iloc[-1],
            "direction": cur["direction"], "entry_price": cur["entry_price"],
            "exit_price": round(mid, 2), "pnl": round(pnl, 2),
            "lot": cur["lot"], "num_trades": cur["num_trades"],
            "bars_held": cur.get("bars_held", 0), "exit_reason": "end",
            "exit_score": 0, "balance": round(balance, 2),
        })

    # final day line
    if cur_day is not None and verbose:
        print(f"  {cur_day}: {daily_trades:3d} tr  ${balance:7.2f}  pk=${peak:7.2f}  dd=${peak - balance:5.2f}", flush=True)

    if not trades:
        return None

    df = pd.DataFrame(trades)
    total_pnl = df["pnl"].sum()
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / len(df) * 100 if len(df) > 0 else 0
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = losses["pnl"].sum() if len(losses) > 0 else 0
    pf = abs(gp / gl) if gl != 0 else float("inf")
    df["cumulative"] = df["pnl"].cumsum()
    df["peak_cum"] = df["cumulative"].cummax()
    df["dd"] = df["peak_cum"] - df["cumulative"]
    max_dd = df["dd"].max()

    # monthly breakdown
    df["month"] = df["exit_time"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
    )
    monthly["wr"] = (monthly["wins"] / monthly["trades"] * 100).round(1)
    monthly_breakdown = {str(k): {"trades": int(v["trades"]), "pnl": round(v["pnl"], 2), "wr": v["wr"]}
                         for k, v in monthly.iterrows()}

    result = {
        "trades": len(df), "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1), "gross_profit": round(gp, 2),
        "gross_loss": round(gl, 2), "net_pnl": round(total_pnl, 2),
        "profit_factor": round(pf, 2),
        "avg_win": round(wins["pnl"].mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses["pnl"].mean(), 2) if len(losses) > 0 else 0,
        "max_dd": round(max_dd, 2),
        "ending_balance": round(balance, 2),
        "deposits": round(deposits, 2),
        "withdrawals": round(withdrawals, 2),
        "net_cash_flow": round(deposits - withdrawals, 2),
        "return_pct": round((total_pnl / INITIAL_BALANCE) * 100, 2),
        "avg_bars_held": round(df["bars_held"].mean(), 1),
        "monthly": monthly_breakdown,
    }
    if verbose:
        print(f"  => ${total_pnl:.2f} WR={wr:.1f}% PF={pf:.2f} DD=${max_dd:.2f} "
              f"Ret={result['return_pct']:.2f}%", flush=True)
        print("\nMonthly breakdown:", flush=True)
        for m, v in monthly_breakdown.items():
            print(f"  {m}: {v['trades']:3d} tr  ${v['pnl']:+.2f}  WR={v['wr']:.1f}%", flush=True)
        total_pnl_all = sum(v["pnl"] for v in monthly_breakdown.values())
        print(f"  TOTAL: ${total_pnl_all:+.2f}", flush=True)
    return result


if __name__ == "__main__":
    default_em = 4 if BACKTEST_BROKER in ("YAHOO_H1", "MT5_H1") else 1
    em = int(sys.argv[1]) if len(sys.argv) > 1 else default_em
    print(f"Broker={BACKTEST_BROKER} Year={BACKTEST_YEAR} exit_mode={em}", flush=True)
    print("Loading and pre-computing...", flush=True)
    d = load_and_compute()
    if d:
        print("\nRunning backtest...", flush=True)
        r = run_backtest(d, {"exit_mode": em}, verbose=True)
        if r:
            print("\nDone.", flush=True)
