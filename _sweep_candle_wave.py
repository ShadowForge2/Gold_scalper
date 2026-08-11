"""
Wave-scalper inside the forming H1 candle.

Strategy (user spec):
  - The model is ONLY a chop gate: if the previous H1 candle was NONE-leaning
    and not a jump, sit out the whole candle.
  - Direction comes from the candle itself. Within the forming H1 candle we
    watch M1 bars and scalp each micro-wave:
      * enter when price makes a clear move (entry_r * ATR) past the wave base
      * cut-loss ~0 (cut_r * ATR) the instant price ticks against us
      * wave profit = the wave's max: exit when the wave pulls back
        profit_r * ATR from its peak (lock the wave)
      * re-enter on each resumption -> many entries per candle
  - Jump-rider: if the forming candle's body reaches jump_break_r * ATR, stop
    scalping, enter in the jump direction, hold and trail (trail_r), close on
    reversal. No re-entry this candle after the rider exit.

Usage:
  python _sweep_candle_wave.py --symbols XAUUSD --train-end 2022 --test-start 2023 --test-end 2025
"""

import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from _train_candle_h1 import SYMBOL_MAP, load_m1_data, resample_h1
from app.candle_engine import (
    FEATURE_COLS, compute_features, compute_atr,
)

ATR_PERIOD = 14


def train_model(X_tr, Y_tr, X_va, Y_va):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_tr, label=Y_tr)
    va_w = np.ones(len(Y_va), dtype=np.float32)
    va_w[Y_va == 2] = 0.5
    dval = xgb.DMatrix(X_va, label=Y_va, weight=va_w)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 50,
        "lambda": 2.0,
        "eval_metric": "mlogloss",
        "seed": 42,
    }
    watch = [(dtrain, "train"), (dval, "val")]
    b = xgb.train(params, dtrain, num_boost_round=800, evals=watch,
                  early_stopping_rounds=60, verbose_eval=False)
    best = b.best_iteration if b.best_iteration is not None else 800
    return xgb.train(params, dtrain, num_boost_round=best + 1, evals=watch,
                     verbose_eval=False)


def compute_jump_flags(h1, atr, break_r, body_r):
    o = h1["open"].values
    c = h1["close"].values
    h = h1["high"].values
    l = h1["low"].values
    a = atr.values
    rng = h - l
    body = np.abs(c - o)
    ok_rng = rng > 0
    ok_atr = a > 0
    res = np.zeros(len(h1), dtype=bool)
    res = (ok_atr & ok_rng & (body >= break_r * np.where(a > 0, a, 0))
           & (np.where(ok_rng, body, 0) / np.where(ok_rng, rng, 1) >= body_r))
    return res


def run_candle_wave(m1, o, atr, entry_r, cut_r, profit_r, cost_r,
                    jump_break_r, jump_body_r, trail_r, reversal_r,
                    rider_enabled=True, fill="market",
                    h1_dir_mode="none", h1_dir_prev=0, min_body_r=0.30):
    """Scalp the micro-waves of ONE forming H1 candle on its M1 bars.

    `m1`: (n,4) array of [open, high, low, close] sub-bars.
    `o`: candle open. `atr`: ATR at the candle's start.
    `fill`: "market" (default) fills entries AND exits at the trigger bar's
        close `bc` and anchors stop/lock/trail at the ACTUAL fill — this mirrors
        the live WaveScalper, which opens/closes at market on the completed
        bar. "level" is the legacy mode that fills at the exact trigger level
        (impossible live: it buys the 0.5R trigger even when the bar closed
        2R past it, and enters jump-riders at the candle open).
    `h1_dir_mode`: "none" trades both directions freely; "prev" uses the
        PREVIOUS H1 candle's color as the only tradable direction; "current"
        lets the FORMING candle decide — the running color (close vs open) must
        reach min_body_r*ATR before ANY entry, and entries are then locked to
        that color. This is "let the candle decide and follow it tightly":
        a green H1 candle only allows BUY waves (each pullback is bought), a
        red one only SELL.
    Returns list of realized R values.
    """
    trades = []
    base = o
    flat = True
    pos = 0
    entry = 0.0          # theoretical trigger level (base anchoring parity)
    entry_fill = 0.0     # actual fill: anchors stop/lock/trail and R
    peak = 0.0
    rider = False
    jump_dir = 0
    just_entered = False

    def f(level, bc):
        return bc if fill == "market" else level

    def run_dir(bc):
        # The H1 color as known at this completed bar (honest, no lookahead).
        if h1_dir_mode == "prev":
            return h1_dir_prev
        if h1_dir_mode == "current":
            body = bc - o
            if atr <= 0:
                return 0
            if body >= min_body_r * atr:
                return 1
            if body <= -min_body_r * atr:
                return -1
            return 0
        return 0

    for bo, bh, bl, bc in m1:
        d = run_dir(bc)
        if flat and not rider:
            allow_buy = h1_dir_mode == "none" or d >= 0
            allow_sell = h1_dir_mode == "none" or d <= 0
            if allow_buy and bh >= base + entry_r * atr:
                entry = base + entry_r * atr
                entry_fill = f(entry, bc)
                pos = 1
                peak = entry_fill
                flat = False
                just_entered = True
            elif allow_sell and bl <= base - entry_r * atr:
                entry = base - entry_r * atr
                entry_fill = f(entry, bc)
                pos = -1
                peak = entry_fill
                flat = False
                just_entered = True
            elif rider_enabled and (allow_buy or allow_sell):
                # Jump-rider trigger: forming candle body reached jump break.
                if allow_buy and (bh - o) >= jump_break_r * atr and (bh - bl) > 0:
                    body = bh - o
                    if body / (bh - bl) >= jump_body_r:
                        entry = o
                        entry_fill = f(o, bc)
                        pos = 1
                        peak = entry_fill
                        flat = False
                        rider = True
                        jump_dir = 1
                        just_entered = True
                elif allow_sell and (o - bl) >= jump_break_r * atr and (bh - bl) > 0:
                    body = o - bl
                    if body / (bh - bl) >= jump_body_r:
                        entry = o
                        entry_fill = f(o, bc)
                        pos = -1
                        peak = entry_fill
                        flat = False
                        rider = True
                        jump_dir = -1
                        just_entered = True
            if just_entered:
                continue
        elif pos == 1:
            # Strict: check exits against the PREVIOUS peak, then update it.
            # A bar can never sell its own high (no buy-low/sell-high same bar).
            stop = entry_fill - cut_r * atr
            lock = peak - profit_r * atr
            if bl <= stop:
                px = f(stop, bc)
                trades.append((px - entry_fill) / atr - cost_r)
                base = stop
                flat = True
                pos = 0
            elif bl <= lock:
                px = f(lock, bc)
                trades.append((px - entry_fill) / atr - cost_r)
                base = lock
                flat = True
                pos = 0
            elif rider:
                ts = peak - trail_r * atr
                if bl <= ts:
                    px = f(ts, bc)
                    trades.append((px - entry_fill) / atr - cost_r)
                    flat = True
                    pos = 0
                    rider = False
                    base = ts
            if bh > peak:
                peak = bh
        elif pos == -1:
            stop = entry_fill + cut_r * atr
            lock = peak + profit_r * atr
            if bh >= stop:
                px = f(stop, bc)
                trades.append((entry_fill - px) / atr - cost_r)
                base = stop
                flat = True
                pos = 0
            elif bh >= lock:
                px = f(lock, bc)
                trades.append((entry_fill - px) / atr - cost_r)
                base = lock
                flat = True
                pos = 0
            elif rider:
                ts = peak + trail_r * atr
                if bh >= ts:
                    px = f(ts, bc)
                    trades.append((entry_fill - px) / atr - cost_r)
                    flat = True
                    pos = 0
                    rider = False
                    base = ts
            if bl < peak:
                peak = bl

    if pos != 0:
        r = (bc - entry_fill) * pos / atr - cost_r
        trades.append(r)
    return trades


def run_symbol(symbol, train_start, train_end, test_start, test_end, combos,
               tf_min=60, rider_enabled=True, gate_enabled=True,
               start_balance=20.0,
               ref_balance=20.0, lot_exp=0.0, compound=False,
               fill="market", h1_dir_mode="none", min_body_r=0.30):
    import xgboost as xgb
    print(f"\n=== {symbol} ({tf_min}min) ===", flush=True)
    m1 = load_m1_data(symbol, start_year=train_start, end_year=test_end)
    h1_all = resample_h1(m1, tf_min)
    if len(h1_all) < 2000:
        print("  SKIP", flush=True)
        return

    feats_all = compute_features(h1_all)
    atr_all = compute_atr(h1_all, ATR_PERIOD)
    tr_mask = (h1_all.index.year >= train_start) & (h1_all.index.year <= train_end)
    te_mask = (h1_all.index.year >= test_start) & (h1_all.index.year <= test_end)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    if len(tr_idx) < 1000 or len(te_idx) < 500:
        print("  SKIP train/test", flush=True)
        return

    trade_params = dict(
        sl_r=float(getattr(cfg, "CANDLE_ENGINE_LABEL_SL_ATR", 1.0)),
        reversal_r=float(getattr(cfg, "CANDLE_ENGINE_REVERSAL_ATR", 0.5)),
        trail_r=float(getattr(cfg, "CANDLE_ENGINE_TRAIL_ATR", 0.5)),
        max_hold=int(getattr(cfg, "CANDLE_ENGINE_MAX_HOLD_BARS", 24)),
        cost_r=float(getattr(cfg, "CANDLE_ENGINE_COST_R", 0.05)),
        entry_min_r=float(getattr(cfg, "CANDLE_ENGINE_ENTRY_MIN_R", 0.90)),
        edge_margin=float(getattr(cfg, "CANDLE_ENGINE_EDGE_MARGIN", 1.75)),
    )
    from app.candle_engine import generate_labels

    X = feats_all[FEATURE_COLS].fillna(0.0).values
    labeled = generate_labels(feats_all, atr_all, **trade_params)
    Yl = labeled["entry_label"].values
    n_buy = int((Yl == 0).sum())
    n_sell = int((Yl == 1).sum())
    n_none = int((Yl == 2).sum())
    print(f"  labels BUY {n_buy} SELL {n_sell} NONE {n_none}", flush=True)
    rng = np.random.default_rng(42)

    def balanced(idx, per_class):
        parts = []
        for cls in (0, 1, 2):
            sel = idx[Yl[idx] == cls]
            if len(sel) > per_class:
                sel = rng.choice(sel, size=per_class, replace=False)
            parts.append(sel)
        out = np.concatenate(parts)
        out.sort()
        return out

    per_class = int(getattr(cfg, "CANDLE_ENGINE_TRAIN_PER_CLASS", 15000))
    if gate_enabled:
        tr_sel = balanced(tr_idx, per_class)
        va_sel = tr_idx[int(0.85 * len(tr_idx)):]
        model = train_model(X[tr_sel], Yl[tr_sel], X[va_sel], Yl[va_sel])
    else:
        model = None

    h1_test = feats_all.iloc[te_idx]
    atr_test = atr_all.iloc[te_idx]
    X_test = h1_test[FEATURE_COLS].fillna(0.0).values
    probs = None
    if gate_enabled:
        probs = model.predict(xgb.DMatrix(X_test))

    jump_cache = {}
    for combo in combos:
        key = (combo["jump_break_r"], combo["jump_body_r"])
        if key not in jump_cache:
            jump_cache[key] = compute_jump_flags(
                h1_test, atr_test, key[0], key[1])

    # Bucket M1 bars into the same H1 test candles for intra-candle scalping.
    rule = f"{tf_min}min" if tf_min != 60 else "1h"
    m1_idx = m1.set_index("time") if "time" in m1.columns else m1
    m1_idx = m1_idx[~m1_idx.index.duplicated(keep="first")].sort_index()
    m1_bucket = m1_idx.index.floor(rule)
    test_start_ts = h1_test.index[0]
    test_end_ts = h1_test.index[-1]
    mask = (m1_bucket >= test_start_ts) & (m1_bucket <= test_end_ts)
    m1_test = m1_idx[mask]
    bucket = m1_bucket[mask]
    o_arr = m1_test["open"].values
    h_arr = m1_test["high"].values
    l_arr = m1_test["low"].values
    c_arr = m1_test["close"].values
    groups = pd.Series(np.arange(len(bucket)), index=m1_test.index).groupby(
        pd.Series(bucket.values, index=m1_test.index), sort=True)
    candle_m1 = {}
    for ts, idx_ in groups.indices.items():
        idx_ = np.asarray(idx_)
        candle_m1[ts] = np.column_stack(
            [o_arr[idx_], h_arr[idx_], l_arr[idx_], c_arr[idx_]])

    results = []
    for combo in combos:
        jf = jump_cache[(combo["jump_break_r"], combo["jump_body_r"])]
        entry_r = combo["entry_r"]
        cut_r = combo["cut_r"]
        profit_r = combo["profit_r"]
        cost_r = combo["cost_r"]
        all_r = []
        n_wave = 0
        n_candle = 0
        equity = start_balance
        peak_equity = start_balance
        max_dd = 0.0
        for k, ts in enumerate(h1_test.index):
            if k == 0:
                continue
            # Chop gate uses the PREVIOUS completed candle (known at this open).
            prev = k - 1
            if gate_enabled:
                pb, ps, pn = float(probs[prev][0]), float(probs[prev][1]), float(probs[prev][2])
                if pn > max(pb, ps) and not jf[prev]:
                    continue
            n_candle += 1
            sub = candle_m1.get(ts)
            if sub is None or len(sub) < 2:
                continue
            o = float(h1_test["open"].iloc[k])
            atr = float(atr_test.iloc[prev]) if atr_test.iloc[prev] > 0 else 1e-9
            # H1-color directional gate: only the candle's color may be traded.
            prev_body = float(h1_test["close"].iloc[prev]) - float(h1_test["open"].iloc[prev])
            h1_dir_prev = 1 if prev_body > 0 else (-1 if prev_body < 0 else 0)
            # Lot-scaled cut/profit: cut grows with position size so the
            # dollar loss per cut stays a constant fraction of the wave.
            # scale = 1 at the reference balance (lot = base lot).
            lot_ratio = max(equity, cfg.MIN_LOT) / ref_balance if ref_balance > 0 else 1.0
            scale = (lot_ratio ** lot_exp) if lot_exp else 1.0
            c_r = cut_r * scale
            p_r = profit_r * scale
            rs = run_candle_wave(
                sub, o, atr, entry_r, c_r, p_r, cost_r,
                combo["jump_break_r"], combo["jump_body_r"],
                combo.get("trail_r", 0.5), combo.get("reversal_r", 0.5),
                rider_enabled=rider_enabled,
                fill=fill,
                h1_dir_mode=h1_dir_mode, h1_dir_prev=h1_dir_prev,
                min_body_r=min_body_r,
            )
            n_wave += len(rs)
            all_r.extend(rs)
            if compound:
                # Dollar PnL = R x ATR x lot. lot ~ base x (equity/ref).
                # Track equity in "R x ATR x base_lot" units so a 1.0R wave
                # at reference balance moves equity by exactly 1.0.
                w_dollars = float(np.asarray(rs).sum()) * atr
                equity += w_dollars * lot_ratio
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

        rs_arr = np.asarray(all_r, dtype=float)
        if len(rs_arr) == 0:
            results.append((combo, None))
            continue
        wins = float(rs_arr[rs_arr > 0].sum())
        losses = float((-rs_arr[rs_arr < 0]).sum())
        pf = (wins / losses) if losses > 0 else float("inf")
        eq = np.cumsum(rs_arr)
        peak_eq = np.maximum.accumulate(eq)
        dd = float((peak_eq - eq).max())
        results.append((combo, {
            "trades": len(rs_arr),
            "candles": n_candle,
            "wr": 100 * float((rs_arr > 0).mean()),
            "exp": float(rs_arr.mean()),
            "pf": pf,
            "net": float(rs_arr.sum()),
            "dd": dd,
            "eq_final": equity if compound else None,
            "eq_dd": max_dd if compound else None,
        }))

    print(f"\n  {'entry':>5} {'cut':>5} {'profit':>6} {'jb':>4} {'jbd':>4} "
          f"{'trail':>5} {'trd':>6} {'cndl':>5} {'WR%':>5} {'expR':>6} "
          f"{'PF':>5} {'netR':>7} {'dd':>5}", flush=True)
    if compound:
        print("  (compound: final equity in base-lot units, dd as % of peak)", flush=True)
    for combo, m in sorted(results, key=lambda kv: (kv[1] or {}).get("pf", 0), reverse=True):
        if m is None:
            print(f"  {combo['entry_r']:>5.2f} {combo['cut_r']:>5.2f} "
                  f"{combo['profit_r']:>6.2f} {combo['jump_break_r']:>4.1f} "
                  f"{combo['jump_body_r']:>4.2f} {combo['trail_r']:>5.2f} |  no trades", flush=True)
            continue
        line = (f"  {combo['entry_r']:>5.2f} {combo['cut_r']:>5.2f} "
                f"{combo['profit_r']:>6.2f} {combo['jump_break_r']:>4.1f} "
                f"{combo['jump_body_r']:>4.2f} {combo['trail_r']:>5.2f} "
                f"| {m['trades']:>6} {m['candles']:>5} "
                f"{m['wr']:>5.1f} {m['exp']:>+6.3f} {m['pf']:>5.2f} "
                f"{m['net']:>+7.1f} {m['dd']:>5.1f}")
        if compound and m.get("eq_final") is not None:
            mult = m["eq_final"] / start_balance
            line += f" | final {m['eq_final']:>9.1f} (x{mult:>5.1f}) dd {100*m['eq_dd']:>5.1f}%"
        print(line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="XAUUSD")
    parser.add_argument("--train-start", type=int, default=2018)
    parser.add_argument("--train-end", type=int, default=2022)
    parser.add_argument("--test-start", type=int, default=2023)
    parser.add_argument("--test-end", type=int, default=2025)
    parser.add_argument("--tf", type=int, default=60)
    parser.add_argument("--no-rider", action="store_true",
                        help="disable jump-rider mode (pure wave scalper)")
    parser.add_argument("--no-gate", action="store_true",
                        help="disable the ML chop gate entirely (trade every candle)")
    parser.add_argument("--start-balance", type=float, default=20.0,
                        help="starting equity for compounding sim (reference = 20)")
    parser.add_argument("--ref-balance", type=float, default=20.0,
                        help="balance at which scale = 1 (lot = base lot)")
    parser.add_argument("--lot-exp", type=float,
                        default=float(getattr(cfg, "CANDLE_ENGINE_WAVE_LOT_EXP", 0.0)),
                        help="cut/profit scale exponent: scale=(equity/ref)^exp "
                             "(0 = no lot scaling — LOCKED, do not change)")
    parser.add_argument("--compound", action="store_true",
                        help="track equity through the loop (dollar PnL = R*ATR*lot)")
    parser.add_argument("--sweep", action="store_true",
                        help="restore full combo grid (overrides locked config)")
    parser.add_argument("--sweep-rider", action="store_true",
                        help="sweep jump-rider trail_r (0.3/0.5/0.8/1.0)")
    parser.add_argument("--sweep-jump", action="store_true",
                        help="sweep jump triggers (break 1.0/1.5/2.0 x body 0.5/0.7/0.9)")
    parser.add_argument("--entry-r", type=float, default=None,
                        help="override the entry grid with this single value")
    parser.add_argument("--cut-r", type=float, default=None,
                        help="override the cut grid with this single value")
    parser.add_argument("--profit-r", type=float, default=None,
                        help="override the profit grid with this single value")
    parser.add_argument("--fill", default="market",
                        choices=["market", "level"],
                        help="entry/exit fill convention: 'market' fills at the "
                             "trigger bar's close like the live bot (default), "
                             "'level' fills at the exact trigger level (legacy)")
    parser.add_argument("--dir-mode", default="none",
                        choices=["none", "prev", "current"],
                        help="H1-color directional gate: 'none' trades both ways; "
                             "'prev' only trades the PREVIOUS H1 candle's color; "
                             "'current' lets the forming H1 candle decide "
                             "(running color must reach min_body_r*ATR first)")
    parser.add_argument("--min-body-r", type=float, default=0.30,
                        help="min running H1 body (xATR) before 'current' mode "
                             "locks a direction")
    args = parser.parse_args()

    lock_entry = args.entry_r if args.entry_r is not None else float(getattr(cfg, "CANDLE_ENGINE_WAVE_ENTRY_R", 0.50))
    lock_cut = args.cut_r if args.cut_r is not None else float(getattr(cfg, "CANDLE_ENGINE_WAVE_CUT_R", 0.03))
    lock_profit = args.profit_r if args.profit_r is not None else float(getattr(cfg, "CANDLE_ENGINE_WAVE_PROFIT_R", 0.05))
    lock_trail = float(getattr(cfg, "CANDLE_ENGINE_WAVE_TRAIL_R", 0.5))
    lock_reversal = float(getattr(cfg, "CANDLE_ENGINE_WAVE_REVERSAL_R", 0.5))
    combos = []
    if args.sweep:
        grid = [(e, c, p) for e, c, p in itertools.product(
            [0.10, 0.20, 0.30, 0.50], [0.01, 0.03], [0.01, 0.05, 0.10])]
    else:
        grid = [(lock_entry, lock_cut, lock_profit)]
    for entry_r, cut_r, profit_r in grid:
        if args.sweep_jump:
            for jb, jbd in itertools.product([1.0, 1.5, 2.0], [0.5, 0.7, 0.9]):
                if args.sweep_rider:
                    for tr in [0.3, 0.5, 0.8, 1.0]:
                        combos.append(dict(
                            entry_r=entry_r, cut_r=cut_r, profit_r=profit_r,
                            cost_r=float(getattr(cfg, "CANDLE_ENGINE_COST_R", 0.05)),
                            jump_break_r=jb, jump_body_r=jbd,
                            trail_r=tr, reversal_r=lock_reversal,
                        ))
                else:
                    combos.append(dict(
                        entry_r=entry_r, cut_r=cut_r, profit_r=profit_r,
                        cost_r=float(getattr(cfg, "CANDLE_ENGINE_COST_R", 0.05)),
                        jump_break_r=jb, jump_body_r=jbd,
                        trail_r=lock_trail, reversal_r=lock_reversal,
                    ))
        elif args.sweep_rider:
            for tr in [0.3, 0.5, 0.8, 1.0]:
                combos.append(dict(
                    entry_r=entry_r, cut_r=cut_r, profit_r=profit_r,
                    cost_r=float(getattr(cfg, "CANDLE_ENGINE_COST_R", 0.05)),
                    jump_break_r=float(getattr(cfg, "CANDLE_ENGINE_JUMP_BREAK_R", 1.5)),
                    jump_body_r=float(getattr(cfg, "CANDLE_ENGINE_JUMP_BODY_R", 0.70)),
                    trail_r=tr, reversal_r=lock_reversal,
                ))
        else:
            combos.append(dict(
                entry_r=entry_r, cut_r=cut_r, profit_r=profit_r,
                cost_r=float(getattr(cfg, "CANDLE_ENGINE_COST_R", 0.05)),
                jump_break_r=float(getattr(cfg, "CANDLE_ENGINE_JUMP_BREAK_R", 1.5)),
                jump_body_r=float(getattr(cfg, "CANDLE_ENGINE_JUMP_BODY_R", 0.70)),
                trail_r=lock_trail, reversal_r=lock_reversal,
            ))

    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        try:
            run_symbol(sym, args.train_start, args.train_end,
                       args.test_start, args.test_end, combos, tf_min=args.tf,
                       rider_enabled=not args.no_rider,
                       gate_enabled=not args.no_gate,
                       start_balance=args.start_balance,
                       ref_balance=args.ref_balance,
                       lot_exp=args.lot_exp,
                       compound=args.compound,
                       fill=args.fill,
                       h1_dir_mode=args.dir_mode,
                       min_body_r=args.min_body_r)
        except Exception as e:
            import traceback
            print(f"\n=== {sym} FAILED: {e} ===", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
