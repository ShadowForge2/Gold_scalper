"""
Backtest with $1,000 profit cap: every time balance exceeds $1,000,
withdraw the excess and keep $1,000 working.
Shows worst losses at realistic lot sizes.
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import config as cfg
from app.dukascopy_client import DukascopyClient
from app.signal_engine import SignalEngine
from app.bias_engine import BiasEngine
from app.risk_manager import EquityScaler
from datetime import datetime, timedelta

CAP_BALANCE = 1000.0

CONTRACT_SIZE = 100
SPREAD_PER_LOT_USD = 25.0
SLIPPAGE_PER_LOT_USD = 3.0
SPREAD_ROUND_TRIP = 2
LEVERAGE = 200

cfg.EXIT_THRESHOLD_TIGHT = 0.75
cfg.SIGNAL_ENTRY_THRESHOLD = 0.50
cfg.LOT_MULTIPLIER = 2
cfg.MAX_LOT = 100.00
cfg.MIN_LOT = 0.01
cfg.LOT_SIZE = 0.01
cfg.MIN_BALANCE = 10.0
cfg.MAX_TRADES_PER_EVENT = 10
cfg.MAX_TRADES_PER_SESSION = 10
cfg.MAX_CONSECUTIVE_LOSSES = 3
cfg.MAX_DAILY_LOSS_USD = 10.00
cfg.MAX_EVENT_LOSS_USD = 5.00
cfg.RE_ENTRY_COOLDOWN_SEC = 60
cfg.ALLOWED_SESSIONS = "ASIA,LONDON,NEW_YORK"
cfg.BIAS_STRENGTH_MIN = 0.30
MARGIN_MAX_PCT = 1.0

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "dukascopy")

def _session_allowed(ts, sessions_str):
    if sessions_str.upper() == "ALL":
        return True
    h = ts.hour
    sessions = [s.strip().upper() for s in sessions_str.split(",")]
    if "ASIA" in sessions and 0 <= h < 8:
        return True
    if "LONDON" in sessions and 8 <= h < 17:
        return True
    if "NEW_YORK" in sessions and 13 <= h < 22:
        return True
    return False

def _pnl(entry_px, exit_px, direction, lot, num_trades, entry_time=None):
    delta = exit_px - entry_px if direction == "BUY" else entry_px - exit_px
    gross = delta * CONTRACT_SIZE * lot * num_trades
    spread_cost = SPREAD_PER_LOT_USD * lot * num_trades * SPREAD_ROUND_TRIP
    slippage_cost = SLIPPAGE_PER_LOT_USD * lot * num_trades
    return gross - spread_cost - slippage_cost

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

def _get_spread_multiplier(entry_time):
    return 1.0

def run_capped_backtest(data, year_label):
    sig_df = data["sig_df"]
    biases = data["biases"]
    sigs = data["sig_signal"]
    scores = data["sig_score"]
    N = len(sig_df)
    signal_engine = data["signal_engine"]

    et = cfg.SIGNAL_ENTRY_THRESHOLD
    ext = cfg.EXIT_THRESHOLD_TIGHT
    base_tr = cfg.MAX_TRADES_PER_EVENT
    sessions = cfg.ALLOWED_SESSIONS
    bias_min = cfg.BIAS_STRENGTH_MIN
    lot_mult = cfg.LOT_MULTIPLIER
    max_total_lot = cfg.MAX_LOT
    max_trades = cfg.MAX_TRADES_PER_EVENT
    event_loss_limit = cfg.MAX_EVENT_LOSS_USD
    consec_loss_limit = cfg.MAX_CONSECUTIVE_LOSSES
    daily_loss_limit = cfg.MAX_DAILY_LOSS_USD
    max_trades_per_session = cfg.MAX_TRADES_PER_SESSION
    min_balance = cfg.MIN_BALANCE

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
    daily_pnl = 0.0
    total_withdrawn = 0.0
    withdrawal_events = 0

    for idx in range(N):
        ts = sig_df["time"].iloc[idx]
        row = sig_df.iloc[idx]
        px = row["open"]
        tier = _get_tier(balance)
        tp = _tier_params(tier)

        if cur_day != ts.date():
            if cur_day is not None:
                pass
            cur_day = ts.date()
            daily_trades = 0
            daily_pnl = 0.0
            consec_losses = 0

        if idx < 10:
            continue

        bsum = biases[idx]
        if bsum is None:
            continue

        # ── EXIT ──
        if cur is not None:
            exit_window = sig_df.iloc[max(0, idx - 10):idx]
            should_exit, exit_score_val, reason = signal_engine.evaluate_exit(
                exit_window, cur["entry_price"], cur["direction"], cur.get("entry_score"),
                exit_threshold=ext, exit_mode=1
            )

            if should_exit:
                pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"], entry_time=cur["entry_time"])
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
                    "withdrawn": 0.0,
                })
                consec_losses = 0 if pnl > 0 else consec_losses + 1
                cooldown = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + consec_losses))
                cur = None

                # ── CAP CHECK ──
                if balance > CAP_BALANCE:
                    withdraw_amt = balance - CAP_BALANCE
                    total_withdrawn += withdraw_amt
                    withdrawal_events += 1
                    trades[-1]["withdrawn"] = round(withdraw_amt, 2)
                    balance = CAP_BALANCE
                    scaler.initialize(balance)
                    peak = balance
                continue

            event_pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"], entry_time=cur["entry_time"])
            if event_pnl <= -event_loss_limit:
                pnl = _pnl(cur["entry_price"], px, cur["direction"], cur["lot"], cur["num_trades"], entry_time=cur["entry_time"])
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
                    "withdrawn": 0.0,
                })
                consec_losses += 1
                cooldown = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + consec_losses))
                cur = None

                if balance > CAP_BALANCE:
                    withdraw_amt = balance - CAP_BALANCE
                    total_withdrawn += withdraw_amt
                    withdrawal_events += 1
                    trades[-1]["withdrawn"] = round(withdraw_amt, 2)
                    balance = CAP_BALANCE
                    scaler.initialize(balance)
                    peak = balance
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
                lot = min(scaler.get_lot(balance) * lot_mult, cfg.MAX_LOT)
                num_tr = min(tp["num_trades"], max_trades, scaler.get_trades_per_event(balance, score))
                total_lot = lot * num_tr
                if total_lot > max_total_lot:
                    num_tr = max(1, int(max_total_lot / lot))
                    total_lot = lot * num_tr
                if total_lot > max_total_lot:
                    lot = max_total_lot / num_tr
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
                daily_trades += 1

    # final close
    if cur is not None:
        mid = float(sig_df["close"].iloc[-1])
        pnl = _pnl(cur["entry_price"], mid, cur["direction"], cur["lot"], cur["num_trades"], entry_time=cur["entry_time"])
        balance += pnl
        trades.append({
            "entry_time": cur["entry_time"], "exit_time": sig_df["time"].iloc[-1],
            "direction": cur["direction"], "entry_price": cur["entry_price"],
            "exit_price": round(mid, 2), "pnl": round(pnl, 2),
            "lot": cur["lot"], "num_trades": cur["num_trades"],
            "bars_held": cur.get("bars_held", 0), "exit_reason": "end",
            "exit_score": 0, "balance": round(balance, 2),
            "withdrawn": 0.0,
        })

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

    return {
        "trades_df": df,
        "trades": len(df), "wins": len(wins), "losses": len(losses),
        "win_rate": round(wr, 1), "gross_profit": round(gp, 2),
        "gross_loss": round(gl, 2), "net_pnl": round(total_pnl, 2),
        "profit_factor": round(pf, 2),
        "avg_win": round(wins["pnl"].mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses["pnl"].mean(), 2) if len(losses) > 0 else 0,
        "max_dd": round(max_dd, 2),
        "ending_balance": round(balance, 2),
        "total_withdrawn": round(total_withdrawn, 2),
        "withdrawal_events": withdrawal_events,
    }


for year, y_start, y_end, d_start, d_end in [
    (2023, 2022, 2023, datetime(2023, 1, 1), datetime(2023, 12, 31, 23, 59)),
    (2025, 2024, 2025, datetime(2025, 1, 1), datetime(2025, 12, 31, 23, 59)),
]:
    INITIAL_BALANCE = 20.0
    WARMUP_DAYS = 21

    print("=" * 70)
    print(f"GOLD SCALPER - {year} Backtest ($1,000 Cap)")
    print(f"Balance: $20 -> withdraw excess > ${CAP_BALANCE:.0f}")
    print("=" * 70)

    client = DukascopyClient(cache_dir=CACHE_DIR)
    all_m1 = client.download_range(d_start.year - 1, d_start.year)
    print(f"M1 bars: {len(all_m1)}")

    warmup_start = d_start - timedelta(days=WARMUP_DAYS)
    mask = (all_m1["time"] >= warmup_start) & (all_m1["time"] < d_end)
    m1 = all_m1[mask].copy()
    h1 = client.resample_to(m1, 16385)
    m5 = client.resample_to(m1, 5)
    print(f"M5: {len(m5)}  H1: {len(h1)}")

    print("Pre-computing signals...")
    import backtest as bt
    data = bt._pre_compute(m5, h1)
    if data is None:
        print("Pre-compute failed.")
        continue

    print("Running backtest...")
    t0 = datetime.now()
    result = run_capped_backtest(data, str(year))
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Done in {elapsed:.0f}s\n")

    if result is None:
        print("No trades.\n")
        continue

    df = result["trades_df"].copy()
    df["entry_time"] = df["entry_time"].apply(
        lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, 'strftime') else str(t)
    )
    df["exit_time"] = df["exit_time"].apply(
        lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, 'strftime') else str(t)
    )

    print("=" * 80)
    print("  FULL YEAR SUMMARY")
    print("=" * 80)
    print(f"  Total trades:      {result['trades']}")
    print(f"  Wins / Losses:     {result['wins']} / {result['losses']}")
    print(f"  Win rate:          {result['win_rate']}%")
    print(f"  Gross profit:      ${result['gross_profit']:,.2f}")
    print(f"  Gross loss:        ${result['gross_loss']:,.2f}")
    print(f"  Net P&L:           ${result['net_pnl']:,.2f}")
    print(f"  Profit factor:     {result['profit_factor']}")
    print(f"  Max DD:            ${result['max_dd']:,.2f}")
    print(f"  Avg win:           ${result['avg_win']:,.2f}")
    print(f"  Avg loss:          ${result['avg_loss']:,.2f}")
    print(f"  Ending balance:    ${result['ending_balance']:,.2f}")
    print(f"  Total withdrawn:   ${result['total_withdrawn']:,.2f}")
    print(f"  Withdrawal events: {result['withdrawal_events']}")
    print(f"  Total earned:      ${result['total_withdrawn'] + result['ending_balance'] - 20:,.2f}")
    print()

    # Monthly breakdown
    df["month"] = df["exit_time"].str[:7]
    monthly = df.groupby("month").agg(
        trades=("pnl", "count"), pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
        withdrawn=("withdrawn", "sum"),
    )
    monthly["losses"] = monthly["trades"] - monthly["wins"]
    monthly["wr"] = (monthly["wins"] / monthly["trades"] * 100).round(1)

    print("=" * 125)
    print(f"  {'Month':<8} {'Tr':<5} {'W':<4} {'L':<4} {'WR%':<6} {'PnL':<14} {'Avg PnL':<11} {'Withdrawn':<13} {'Bal':<10}")
    print("=" * 125)
    cumul = 20.0
    cum_wd = 0.0
    for month, row in monthly.iterrows():
        avg_pnl = row["pnl"] / row["trades"] if row["trades"] > 0 else 0
        cumul += row["pnl"]
        cum_wd += row["withdrawn"]
        print(f"  {month:<8} {int(row['trades']):<5} {int(row['wins']):<4} {int(row['losses']):<4} "
              f"{row['wr']:<5.1f}% ${row['pnl']:<+10.2f} ${avg_pnl:<+8.2f} ${row['withdrawn']:<10.2f} ${cumul:<7.2f}")
    print("=" * 125)
    print(f"  {'TOTAL':<8} {result['trades']:<5} {result['wins']:<4} {result['losses']:<4} "
          f"{result['win_rate']:<5.1f}% ${result['net_pnl']:<+10.2f}          "
          f"${result['total_withdrawn']:<10.2f} ${result['ending_balance']:<7.2f}")
    print()

    # Loss analysis
    losers = df[df["pnl"] < 0]
    print("  LOSS ANALYSIS BY EXIT REASON")
    for r in losers["exit_reason"].unique():
        s = losers[losers["exit_reason"] == r]
        print(f"    {r}: count={len(s):3d}  avg=${s['pnl'].mean():>+8.2f}  "
              f"worst=${s['pnl'].min():>+8.2f}  max lot={s['lot'].max():.5f}")

    # Bottom 10
    print("\n  BOTTOM 10 LOSSES")
    print(f"  {'#':<4} {'Date':<16} {'Dir':<5} {'PnL':<14} {'Lot':<10} {'Reason':<20}")
    print("  " + "-" * 69)
    for i, (_, t) in enumerate(df.nsmallest(10, "pnl").iterrows()):
        print(f"  {i+1:<4} {t['exit_time']:<16} {t['direction']:<5} "
              f"${t['pnl']:<+10.2f} {t['lot']:<10.5f} {t['exit_reason']:<20}")

    # Lot distribution
    print("\n  LOT SIZE DISTRIBUTION")
    bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        c = ((df["lot"] > lo) & (df["lot"] <= hi)).sum()
        if c > 0:
            print(f"    {lo:<5}-{hi:<5}: {c} trades")
    last = (df["lot"] > 10).sum()
    if last > 0:
        print(f"    >10      : {last} trades")
    print(f"    Max lot:  {df['lot'].max():.5f}  Avg lot:  {df['lot'].mean():.5f}")

    csv_path = os.path.join(os.path.dirname(__file__), f"backtest_{year}_cap1000.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}\n")
