"""
TWO-ENGINE MECHANICAL daily-income backtest: XAUUSD pullback-long + US100
momentum-jump, with the EquityScaler aggressive lot sizing from app/risk_manager.py.

Engines (both OOS-validated, pure rules, no ML):
  GOLD  : pullback-long — H1 uptrend >1.0 ATR + M5 RSI<0.35 + vol<=0.9,
          SL 1R / TP 1R / max hold 12, cost 0.05R, pause 2 losses -> skip 6 bars.
  US100 : momentum-jump — surge (mz>=2, body>=0.6, ts>=0.5) in EMA480 regime
          (long above / short below), close after jump 1.0R + 0.25R retrace,
          SL 1R / max hold 12, cost 0.05R.

Lot sizing (mirrors EquityScaler + config.py):
  lot = base_lot * (balance / 20) * LOT_MULTIPLIER * aggr * score_mult
  drawdown >15% from peak -> lot halved
  score_mult: very-strong (score>=0.5) x2.0, strong (>=0.3) x1.5
  score: gold = 1 - rsi/0.35,  US100 = mz/2

USD conversion: per symbol, usd = R * M5-ATR * lot * contract_size.
  XAUUSD contract_size 100 (0.01 lot = 1 oz, $1 per $1 move)
  US100  contract_size 1   (1.0 lot = $1 per 1pt move; min lot 0.001)

Margin model (matches Capital.com): margin = lot * contract_size * price / leverage.
  Capital.com CFDs are micro-style (contractSize=1), so notional is small:
  0.01 lot gold at ~$4000 = $40 notional (needs ~$2 margin at 1:20), so gold
  IS tradeable from a $20 account at min lot. Lot is still capped so margin
  <= 90% of free margin; if even min lot doesn't fit, the trade is blocked.

Usage:
  python _two_engine.py --start 2025 --end 2025
  python _two_engine.py --start 2025 --end 2025 --aggr 2.0 --use-aggr-score
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _train_candle_brain as t
from _mech_backtest import causal_h1_context, pullback_long_signal

COST_R = 0.05
DRAWDOWN_HALF_PCT = 15.0
MARGIN_SAFETY = 0.9  # never let one position exceed 90% of free margin

# Real Capital.com contract specs (verified from API position payloads):
#   contractSize=1 for commodities AND indices.
#   XAUUSD 1.0 lot = 1 oz -> $1 per $1 move per 1.0 lot
#   US100  1.0 lot = $1 per index point; min traded quantity 0.001
SYMBOL_SPECS = {
    "XAUUSD": {"contract_size": 1, "min_lot": 0.01, "lot_step": 0.01},
    "US100":  {"contract_size": 1, "min_lot": 0.001, "lot_step": 0.001},
}


# --------------------------------------------------------------------------
# US100 momentum-jump engine
# --------------------------------------------------------------------------
def us100_jump_trades(m5, atr, jump_target=1.0, retr_dist=0.25, max_hold=12,
                      sl_r=1.0, mz_min=2.0, ema=480):
    c = m5["close"].values
    h = m5["high"].values
    l = m5["low"].values
    o = m5["open"].values
    br = m5["body_ratio"].values
    ts = m5["trend_strength"].values
    mz = m5["momentum_z"].values
    n = len(m5)

    regime = np.ones(n, dtype=bool)
    if ema:
        regime = (c > pd.Series(c).ewm(span=ema, adjust=False).mean().values)
    short_ok = ~regime

    sig = np.zeros(n, dtype=np.int64)
    sig[(o < c) & (br >= 0.60) & (ts >= 0.50) & (mz >= mz_min) & regime] = 1
    sig[(o > c) & (br >= 0.60) & (ts <= -0.50) & (mz <= -mz_min) & short_ok] = -1

    idx = np.where(sig != 0)[0]
    idx = idx[(idx > max(50, ema)) & (idx < n - 2)]

    trades = []
    next_ok = -1
    for i in idx:
        if i < next_ok:
            continue
        d = sig[i]
        entry = c[i]
        atr_i = max(atr.values[i], 1e-10)
        sl = entry - d * sl_r * atr_i
        peak = entry
        bfe = 0.0
        r_out = None
        exit_bar = None
        for j in range(i + 1, min(i + max_hold + 1, n)):
            if d > 0:
                if h[j] > peak:
                    peak = h[j]
                    bfe = (h[j] - entry) / atr_i
                if l[j] <= sl:
                    r_out = -sl_r
                    exit_bar = j
                    break
                if bfe >= jump_target and c[j] <= peak - retr_dist * atr_i:
                    r_out = (c[j] - entry) / atr_i
                    exit_bar = j
                    break
            else:
                if l[j] < peak:
                    peak = l[j]
                    bfe = (entry - l[j]) / atr_i
                if h[j] >= sl:
                    r_out = -sl_r
                    exit_bar = j
                    break
                if bfe >= jump_target and c[j] >= peak + retr_dist * atr_i:
                    r_out = (entry - c[j]) / atr_i
                    exit_bar = j
                    break
        else:
            j = min(i + max_hold, n - 1)
            r_out = ((c[j] - entry) / atr_i) if d > 0 else ((entry - c[j]) / atr_i)
            exit_bar = j
        trades.append({
            "time": m5.index[i],
            "symbol": "US100",
            "r": r_out - COST_R,
            "atr": atr.values[i],
            "price": entry,
            "score": abs(mz[i]) / 2.0,
        })
        next_ok = exit_bar + 1
    return trades


# --------------------------------------------------------------------------
# Gold pullback-long engine (long-only, reuses validated rule + pause overlay)
# --------------------------------------------------------------------------
def gold_pullback_trades(m5, h1, atr, rsi_max=0.35, vol_max=0.9, h1_min=1.0,
                         pause_consec=2, pause_bars=6):
    t.SL_R = 1.0
    t.TP_R = 1.0
    t.COST_R = COST_R
    t.LABEL_WINDOW = 12
    long_r, _ = t.compute_realized_r(m5, atr)

    class Cfg:
        pass
    c = Cfg()
    c.h1_min = h1_min
    c.rsi_max = rsi_max
    c.vol_max = vol_max
    c.session = "all"

    sig = pullback_long_signal(m5, h1, c)
    n = len(sig)
    use = np.ones(n, dtype=bool)
    use[: t.SEQ_LEN] = False
    use[n - t.LABEL_WINDOW:] = False
    idx = np.where((sig == 0) & use)[0]
    r_vals = long_r[idx]

    if pause_consec > 0:
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
                if loss_streak >= pause_consec:
                    skip_until = i + pause_bars
                    loss_streak = 0
        idx = idx[keep]
        r_vals = r_vals[keep]

    rsi = m5["rsi"].values
    close_vals = m5["close"].values
    trades = []
    for i, r in zip(idx, r_vals):
        trades.append({
            "time": pd.to_datetime(m5.index.values[i]),
            "symbol": "XAUUSD",
            "r": float(r),
            "atr": float(atr.values[i]),
            "price": float(close_vals[i]),
            "score": float(np.clip(1 - rsi[i] / rsi_max, 0.05, 1.0)),
        })
    return trades


# --------------------------------------------------------------------------
# Aggressive lot sizing (EquityScaler port + aggression knobs)
# --------------------------------------------------------------------------
class AggressiveScaler:
    def __init__(self, base_lots, lot_mult, aggr, use_score_mult,
                 very_thr=0.50, very_mult=2.0, strong_thr=0.30, strong_mult=1.5):
        self.base_lots = base_lots
        self.lot_mult = lot_mult
        self.aggr = aggr
        self.use_score_mult = use_score_mult
        self.very_thr = very_thr
        self.very_mult = very_mult
        self.strong_thr = strong_thr
        self.strong_mult = strong_mult
        self.reference = 20.0
        self.peak = None

    def in_drawdown(self, balance):
        if self.peak is None or self.peak <= 0:
            return False
        return (self.peak - balance) / self.peak * 100 > DRAWDOWN_HALF_PCT

    def get_lot(self, balance, symbol, score):
        if self.peak is None or balance > self.peak:
            self.peak = balance
        spec = SYMBOL_SPECS.get(symbol, {"contract_size": 1, "min_lot": 0.01, "lot_step": 0.01})
        lot = self.base_lots.get(symbol, 0.02) * (balance / self.reference) \
            * self.lot_mult * self.aggr
        if self.use_score_mult:
            if score >= self.very_thr:
                lot *= self.very_mult
            elif score >= self.strong_thr:
                lot *= self.strong_mult
        if self.in_drawdown(balance):
            lot *= 0.5
        step = spec["lot_step"]
        lot = round(lot / step) * step
        return max(spec["min_lot"], lot)


# --------------------------------------------------------------------------
# USD equity walk (with real margin constraint per Capital.com specs)
# --------------------------------------------------------------------------
def run_usd(trades, start_equity, scaler, leverage):
    trades = sorted(trades, key=lambda x: x["time"])
    balance = start_equity
    scaler.peak = start_equity
    rows = []
    equity = start_equity
    below_zero = 0
    first_below = None
    min_equity = start_equity
    blocked = 0
    for tr in trades:
        sym = tr["symbol"]
        spec = SYMBOL_SPECS.get(sym, {"contract_size": 1, "min_lot": 0.01, "lot_step": 0.01})
        lev = float(leverage.get(sym, 20.0))

        # Margin constraint: margin = lot * contract_size * price / leverage.
        # Even the minimum lot must fit within MARGIN_SAFETY of free margin,
        # else the broker would reject the order (blocked).
        min_margin = spec["min_lot"] * spec["contract_size"] * tr["price"] / lev
        if min_margin > balance * MARGIN_SAFETY:
            blocked += 1
            continue

        lot = scaler.get_lot(balance, sym, tr["score"])
        step = spec["lot_step"]
        max_lot = balance * MARGIN_SAFETY * lev / (spec["contract_size"] * tr["price"])
        if max_lot < lot:
            lot = max(spec["min_lot"], round(max_lot / step) * step)

        usd = tr["r"] * tr["atr"] * lot * spec["contract_size"]
        equity = balance + usd
        if equity < 0:
            below_zero += 1
            if first_below is None:
                first_below = tr["time"]
        min_equity = min(min_equity, equity)
        rows.append({**tr, "lot": lot, "usd": usd, "equity": equity})
        balance = equity
    return rows, balance, min_equity, below_zero, first_below, blocked


def report(tag, rows, end_balance, start_equity, min_equity, below_zero, first_below, blocked=0):
    df = pd.DataFrame(rows)
    n = len(df)
    if n == 0:
        print(f"{tag:>34} | no trades (blocked {blocked})"); return None
    wins = df[df["usd"] > 0]
    losses = df[df["usd"] < 0]
    pf = wins["usd"].sum() / -losses["usd"].sum() if len(losses) else float("inf")
    daily = df.groupby(pd.to_datetime(df["time"]).dt.date)["usd"].sum()
    d5 = float((daily >= 5).mean())
    d7 = float((daily >= 7).mean())
    dn5 = float((daily <= -5).mean())
    final = end_balance
    print(f"{tag:>34} | {n:>4} | {final:>9.2f} | {100*(final-start_equity)/start_equity:+8.1f}% | "
          f"{daily.mean():+6.2f} | {daily.median():+6.2f} | {100*(daily>0).mean():5.1f}% | "
          f"{d5:5.1f}% {d7:5.1f}% {dn5:5.1f}% | {daily.min():+7.2f} | {min_equity:8.2f} | "
          f"{below_zero:>4} | {blocked:>4}")
    return df


def load_engines(start, end, pause_consec, pause_bars, jump_target):
    all_trades = []

    m1 = t.load_m1_data("XAUUSD", start_year=start, end_year=end)
    m5 = t.resample_m5(m1); m5 = t.compute_features(m5)
    h1 = causal_h1_context(m5, m1)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    gold_tr = gold_pullback_trades(m5, h1, atr, pause_consec=pause_consec,
                                   pause_bars=pause_bars)
    print(f"  XAUUSD engine: {len(gold_tr)} trades", flush=True)
    all_trades += gold_tr
    del m1

    m1 = t.load_m1_data("US100", start_year=start, end_year=end)
    m5 = t.resample_m5(m1); m5 = t.compute_features(m5)
    atr = t.compute_atr(m5, t.ATR_PERIOD)
    us_tr = us100_jump_trades(m5, atr, jump_target=jump_target)
    print(f"  US100 engine:  {len(us_tr)} trades", flush=True)
    all_trades += us_tr
    del m1

    return all_trades


def main(args, all_trades):
    hdr = (f"{'config':>30} | {'n':>4} | {'final':>9} | {'grow':>8} | "
           f"{'avg$/d':>6} | {'med$/d':>6} | {'posD%':>6} | "
           f"{'+$5%':>5} {'+$7%':>5} {'-$5%':>5} | {'worstD':>7} | {'minEq':>8} | {'<0':>4} | {'blk':>4}")
    print(hdr, flush=True)

    leverage = {
        "XAUUSD": float(args.lev_xau),
        "US100": float(args.lev_us100),
    }
    scaler = AggressiveScaler(
        base_lots={"XAUUSD": 0.02, "US100": 0.02},
        lot_mult=args.lot_mult,
        aggr=args.aggr,
        use_score_mult=args.use_aggr_score,
    )
    tag = f"mult x{args.lot_mult} aggr x{args.aggr} score={args.use_aggr_score}"
    rows, endb, mineq, below, first, blocked = run_usd(
        list(all_trades), args.start_equity, scaler, leverage)
    report(tag, rows, endb, args.start_equity, mineq, below, first, blocked)
    if below and first is not None:
        print(f"  -> equity went <0 at {first}, {below} trades below zero", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2025)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--start-equity", type=float, default=20.0)
    p.add_argument("--lot-mult", type=float, default=2.0)
    p.add_argument("--aggr", type=float, default=1.0)
    p.add_argument("--use-aggr-score", action="store_true")
    p.add_argument("--jump-target", type=float, default=1.0)
    p.add_argument("--pause-consec", type=int, default=2)
    p.add_argument("--pause-bars", type=int, default=6)
    p.add_argument("--lev-xau", type=float, default=20.0, help="Capital.com commodities leverage for XAUUSD")
    p.add_argument("--lev-us100", type=float, default=20.0, help="Capital.com indices leverage for US100")
    args = p.parse_args()

    print("=== TWO-ENGINE: XAUUSD pullback-long + US100 momentum-jump ===", flush=True)
    print(f"period {args.start}-{args.end} | start_equity ${args.start_equity:.0f} | "
          f"jump {args.jump_target}R | gold pause {args.pause_consec}/{args.pause_bars} | "
          f"lev XAU {args.lev_xau:.0f}:1 / US100 {args.lev_us100:.0f}:1",
          flush=True)

    all_trades = load_engines(args.start, args.end, args.pause_consec,
                              args.pause_bars, args.jump_target)

    df_all = pd.DataFrame(all_trades)
    for sym in ("XAUUSD", "US100"):
        d = df_all[df_all["symbol"] == sym]
        rs = d["r"].values
        if len(rs):
            w = rs[rs > 0].sum(); ls = -rs[rs < 0].sum()
            print(f"  R-level {sym}: n={len(rs)} WR={100*(rs>0).mean():.1f}% "
                  f"exp={rs.mean():+.3f}R PF={w/ls if ls else 99:.2f} "
                  f"net={rs.sum():+.0f}R", flush=True)
    rs = df_all["r"].values
    w = rs[rs > 0].sum(); ls = -rs[rs < 0].sum()
    eq = np.cumsum(rs); dd = (np.maximum.accumulate(eq) - eq).max()
    print(f"  R-level COMBINED: n={len(rs)} WR={100*(rs>0).mean():.1f}% "
          f"exp={rs.mean():+.3f}R PF={w/ls if ls else 99:.2f} "
          f"net={rs.sum():+.0f}R maxDD={dd:.0f}R", flush=True)

    for aggr_scale in (1.0, 2.0, 3.0):
        for use_score in (False, True):
            print(f"\n--- aggression scale x{aggr_scale}, score_mult={use_score} ---", flush=True)
            main(argparse.Namespace(
                start=args.start, end=args.end, start_equity=args.start_equity,
                lot_mult=args.lot_mult, aggr=aggr_scale,
                use_aggr_score=use_score, jump_target=args.jump_target,
                pause_consec=args.pause_consec, pause_bars=args.pause_bars,
                lev_xau=args.lev_xau, lev_us100=args.lev_us100),
                all_trades)
