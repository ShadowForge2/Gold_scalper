"""Sweep ASP timeout values on January 2025 data."""
import sys, os, time as _time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.asp_features import compute_asp_features
from app.dukascopy_client import DukascopyClient
from app.asp_labels import compute_atr

ASP_MODEL_PATH = "models/asp_swing_xgb_m5.joblib"
ASP_FEATURE_PATH = "models/asp_swing_m5_features.npy"

SL_ATR_MULT = 2.0
TP_ATR_MULT = 1.0
TIMEOUTS = [4, 6, 8, 10, 12, 15, 20, 24, 30]
ATR_PERIOD = 14

STARTING_BALANCE = 20.0
BASE_LOT = 0.01
CONTRACT_SIZE = 100
SPREAD_PTS = 0.30
LOT_STEP = 0.01
MIN_LOT = 0.01
MAX_LOT = 9999.0


def calc_lot(balance, peak_balance):
    reference = 20.0
    lot = BASE_LOT * (balance / reference)
    if peak_balance and balance < peak_balance * 0.9:
        lot *= 0.5
    lot = round(lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(lot, MAX_LOT))


def run_backtest(m5, h1, model, features, inv_map, timeout_bars):
    n = len(m5)
    m5_h = m5["high"].values.astype(np.float64)
    m5_l = m5["low"].values.astype(np.float64)
    m5_c = m5["close"].values.astype(np.float64)
    atr_arr = compute_atr(m5_h, m5_l, m5_c, ATR_PERIOD)

    asp_feats = compute_asp_features(m5, h1)
    asp_feats = asp_feats.reindex(m5.index)
    valid = asp_feats[features].dropna()

    signal_series = pd.Series(0, index=m5.index, dtype=np.int8)
    if len(valid) > 0:
        feat_arr = valid[features].values
        raw_preds = model.predict(feat_arr)
        preds = np.array([inv_map[p] for p in raw_preds])
        signal_series.loc[valid.index] = preds
    signal_arr = signal_series.values

    bal = STARTING_BALANCE
    peak = STARTING_BALANCE
    trades = []
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    entry_dir = ""
    sl_price = 0.0
    tp_price = 0.0
    entry_lot = 0.01
    ticks_in_trade = 0

    START_IDX = max(ATR_PERIOD + 5, 100)

    for i in range(START_IDX, n):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        if in_trade:
            ticks_in_trade += 1

            if entry_dir == "BUY" and m5_l[i] <= sl_price:
                pnl = (sl_price - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                bal += pnl
                peak = max(peak, bal)
                trades.append({"reason": "sl_hit", "pnl": pnl, "bars": ticks_in_trade})
                in_trade = False
                ticks_in_trade = 0
                continue

            if entry_dir == "SELL" and m5_h[i] >= sl_price:
                pnl = (entry_price - sl_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                bal += pnl
                peak = max(peak, bal)
                trades.append({"reason": "sl_hit", "pnl": pnl, "bars": ticks_in_trade})
                in_trade = False
                ticks_in_trade = 0
                continue

            if entry_dir == "BUY" and m5_h[i] >= tp_price:
                pnl = (tp_price - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                bal += pnl
                peak = max(peak, bal)
                trades.append({"reason": "tp_hit", "pnl": pnl, "bars": ticks_in_trade})
                in_trade = False
                ticks_in_trade = 0
                continue

            if entry_dir == "SELL" and m5_l[i] <= tp_price:
                pnl = (entry_price - tp_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                bal += pnl
                peak = max(peak, bal)
                trades.append({"reason": "tp_hit", "pnl": pnl, "bars": ticks_in_trade})
                in_trade = False
                ticks_in_trade = 0
                continue

            if ticks_in_trade >= timeout_bars:
                if entry_dir == "BUY":
                    pnl = (m5_c[i] - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                else:
                    pnl = (entry_price - m5_c[i]) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
                bal += pnl
                peak = max(peak, bal)
                trades.append({"reason": "timeout", "pnl": pnl, "bars": ticks_in_trade})
                in_trade = False
                ticks_in_trade = 0
                continue

        else:
            d = signal_arr[i]
            if d != 0:
                atr_val = atr_arr[i]
                if np.isnan(atr_val) or atr_val <= 0:
                    continue

                direction = "BUY" if d == 1 else "SELL"
                entry_price = m5_c[i]
                sl_dist = atr_val * SL_ATR_MULT
                tp_dist = atr_val * TP_ATR_MULT

                if direction == "BUY":
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                if sl_dist < 0.50:
                    continue

                entry_lot = calc_lot(bal, peak)
                in_trade = True
                entry_idx = i
                entry_dir = direction
                ticks_in_trade = 0

    if in_trade:
        if entry_dir == "BUY":
            pnl = (m5_c[-1] - entry_price) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        else:
            pnl = (entry_price - m5_c[-1]) * entry_lot * CONTRACT_SIZE - SPREAD_PTS * entry_lot * CONTRACT_SIZE
        bal += pnl
        trades.append({"reason": "eoy_close", "pnl": pnl, "bars": ticks_in_trade})

    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    trades_count = len(trades)
    wr = wins / trades_count * 100 if trades_count > 0 else 0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    net_pnl = sum(pnls)

    reasons = {}
    for t in trades:
        r = t["reason"]
        if r not in reasons:
            reasons[r] = {"count": 0, "wins": 0, "pnls": []}
        reasons[r]["count"] += 1
        reasons[r]["pnls"].append(t["pnl"])
        if t["pnl"] > 0:
            reasons[r]["wins"] += 1

    return {
        "trades": trades_count,
        "wr": wr,
        "pf": pf,
        "net_pnl": net_pnl,
        "final_bal": bal,
        "reasons": reasons,
    }


def main():
    t0 = _time.time()
    print("ASP Timeout Sweep -- January 2025 Only")
    print(f"  SL={SL_ATR_MULT}x ATR | TP={TP_ATR_MULT}x ATR | Balance=${STARTING_BALANCE}")
    print(f"  Timeouts: {TIMEOUTS}\n")

    asp_saved = joblib.load(ASP_MODEL_PATH)
    model = asp_saved["model"]
    label_map = asp_saved["label_map"]
    inv_map = {v: k for k, v in label_map.items()}
    features = list(np.load(ASP_FEATURE_PATH, allow_pickle=True))

    client = DukascopyClient()

    t1 = _time.time()
    m1 = client.download_year(2025)
    m1 = m1.sort_values("time").drop_duplicates(subset="time")
    m5 = client.resample_to(m1, 5)
    h1 = client.resample_to(m1, 16385)
    m5 = m5.set_index("time")
    h1 = h1.set_index("time")
    m5.index = m5.index.tz_localize(None) if m5.index.tz else m5.index
    h1.index = h1.index.tz_localize(None) if h1.index.tz else h1.index
    jan_start = pd.Timestamp("2025-01-01")
    jan_end = pd.Timestamp("2025-02-01")
    m5_jan = m5[(m5.index >= jan_start) & (m5.index < jan_end)]
    print(f"  Loaded: M5={len(m5_jan)}, H1={len(h1)} ({_time.time()-t1:.1f}s)\n")

    results = []
    for timeout in TIMEOUTS:
        r = run_backtest(m5_jan, h1, model, features, inv_map, timeout)
        r["timeout"] = timeout
        results.append(r)

    elapsed = _time.time() - t0

    print(f"{'='*90}")
    print(f"  RESULTS ({elapsed:.1f}s)")
    print(f"{'='*90}")
    print(f"{'T/O':>5} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Net PnL':>14} {'Final Bal':>14}")
    print(f"{'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*14} {'-'*14}")

    best_pnl = max(results, key=lambda x: x["net_pnl"])
    best_pf = max(results, key=lambda x: x["pf"])

    for r in results:
        marker = ""
        if r is best_pnl:
            marker = " <-- BEST PNL"
        elif r is best_pf:
            marker = " <-- BEST PF"
        print(f"{r['timeout']:>5} {r['trades']:>7} {r['wr']:>6.1f}% {r['pf']:>6.2f} ${r['net_pnl']:>12,.2f} ${r['final_bal']:>12,.2f}{marker}")

    print(f"\n  Best PnL:  {best_pnl['timeout']} bars -> ${best_pnl['net_pnl']:,.2f} (WR {best_pnl['wr']:.1f}%, PF {best_pnl['pf']:.2f}, {best_pnl['trades']} trades)")
    print(f"  Best PF:   {best_pf['timeout']} bars -> PF {best_pf['pf']:.2f} (WR {best_pf['wr']:.1f}%, PnL ${best_pf['net_pnl']:,.2f}, {best_pf['trades']} trades)")

    print(f"\n  Exit breakdown for timeout={best_pnl['timeout']}:")
    for reason, stats in sorted(best_pnl["reasons"].items()):
        wr_r = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        avg_r = np.mean(stats["pnls"]) if stats["pnls"] else 0
        print(f"    {reason:>10}: {stats['count']:>4} trades, WR={wr_r:.1f}%, avg_pnl=${avg_r:.2f}")


if __name__ == "__main__":
    main()
