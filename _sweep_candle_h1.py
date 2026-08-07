"""
Sweep entry/exit params for the H1 candle engine on OOS data.
Trains the model ONCE per symbol, then simulates the engine across many
(min_conf, jump_break_r, jump_body_r, reversal_r, trail_r, sl_r, max_hold)
combinations, reporting PF / net R / trades / WR per combo.

Usage:
  python _sweep_candle_h1.py --symbols XAUUSD --train-end 2022 --test-start 2023 --test-end 2025
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
    FEATURE_COLS, compute_features, compute_atr, generate_labels,
    jump_signal,
)

TRAIL_ACTIVATE_R = 0.5


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
    """Vectorized jump-candle flags (no per-bar .iloc in the hot loop)."""
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


def simulate_engine(probs, h1, atr, jump_flags, engine_params):
    """Walk OOS bars exactly like the live engine. One trade at a time.
    `probs` is the precomputed (n, 3) model probability array — the exit/entry
    sweep only changes how those probabilities are thresholded, never the model.
    `jump_flags` is the precomputed jump-candle mask for this (break, body) combo.
    """
    sl_r = engine_params["sl_r"]
    reversal_r = engine_params["reversal_r"]
    trail_r = engine_params["trail_r"]
    max_hold = engine_params["max_hold"]
    cost_r = engine_params["cost_r"]
    min_conf = engine_params["min_conf"]
    entry_mode = engine_params.get("entry_mode", "conf")

    c = h1["close"].values
    o = h1["open"].values
    h = h1["high"].values
    l = h1["low"].values
    a = atr.values
    n = len(h1)
    times = h1.index

    trades = []
    i = 0
    while i < n - 1:
        if i < 60:
            i += 1
            continue
        pb, ps, pn = float(probs[i][0]), float(probs[i][1]), float(probs[i][2])
        conf = max(pb, ps)
        is_jump = jump_flags[i]

        if entry_mode == "candle":
            # Ride the NEW H1 candle: entry side = candle close direction.
            # Analysis gate only: skip when the model flags chop (NONE-leaning
            # and no jump confirmation). NO confidence floor.
            if o[i] == c[i]:
                i += 1
                continue
            direction = "BUY" if c[i] > o[i] else "SELL"
            if pn > max(pb, ps) and not is_jump:
                i += 1
                continue
        elif entry_mode == "candle_agree":
            # Ride the candle direction ONLY when the model's directional
            # analysis agrees (pb>=ps on an up-candle). No confidence floor,
            # but the model's BUY/SELL call must confirm the candle.
            if o[i] == c[i]:
                i += 1
                continue
            candle_up = c[i] > o[i]
            direction = "BUY" if candle_up else "SELL"
            model_buy = pb >= ps
            if candle_up != model_buy:
                i += 1
                continue
            if pn > max(pb, ps) and not is_jump:
                i += 1
                continue
        else:
            direction = "BUY" if pb >= ps else "SELL"
            eff_conf = min(conf + 0.15, 0.99) if is_jump else conf
            if eff_conf < min_conf:
                i += 1
                continue
            if pn > max(pb, ps) and not is_jump:
                i += 1
                continue

        e = float(c[i])
        ai = float(a[i]) if a[i] > 0 else 1e-9
        entry_time = times[i]
        entry_side = 1 if direction == "BUY" else -1

        exit_px = None
        exit_i = None
        peak = e
        trough = e
        trailing = False
        for j in range(i + 1, min(i + 1 + max_hold, n)):
            hj, lj, cj = h[j], l[j], c[j]
            if entry_side == 1:
                if lj <= e - sl_r * ai:
                    exit_px = e - sl_r * ai
                    exit_i = j
                    break
                if not trailing and hj >= e + TRAIL_ACTIVATE_R * ai:
                    trailing = True
                if trailing:
                    if hj > peak:
                        peak = hj
                    ts = peak - trail_r * ai
                    if lj <= ts:
                        exit_px = ts
                        exit_i = j
                        break
                if cj <= e - reversal_r * ai:
                    exit_px = cj
                    exit_i = j
                    break
            else:
                if hj >= e + sl_r * ai:
                    exit_px = e + sl_r * ai
                    exit_i = j
                    break
                if not trailing and lj <= e - TRAIL_ACTIVATE_R * ai:
                    trailing = True
                if trailing:
                    if lj < trough:
                        trough = lj
                    ts = trough + trail_r * ai
                    if hj >= ts:
                        exit_px = ts
                        exit_i = j
                        break
                if cj >= e + reversal_r * ai:
                    exit_px = cj
                    exit_i = j
                    break
        if exit_px is None:
            exit_i = min(i + max_hold, n - 1)
            exit_px = float(c[exit_i])

        r = (exit_px - e) / ai * entry_side - cost_r
        trades.append({
            "time": entry_time,
            "direction": direction,
            "r": r,
            "is_jump": is_jump,
            "conf": conf,
        })
        i = exit_i

    return pd.DataFrame(trades) if trades else pd.DataFrame(columns=["time", "direction", "r", "is_jump", "conf"])


def run_symbol(symbol, train_start, train_end, test_start, test_end, combos, tf_min=60,
               label_min_r=None, label_margin=None):
    import xgboost as xgb
    print(f"\n=== {symbol} ({tf_min}min) ===", flush=True)
    m1 = load_m1_data(symbol, start_year=train_start, end_year=test_end)
    h1_all = resample_h1(m1, tf_min)
    if len(h1_all) < 2000:
        print("  SKIP", flush=True)
        return

    feats_all = compute_features(h1_all)
    atr_all = compute_atr(h1_all, 14)
    tr_mask = (h1_all.index.year >= train_start) & (h1_all.index.year <= train_end)
    te_mask = (h1_all.index.year >= test_start) & (h1_all.index.year <= test_end)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    if len(tr_idx) < 1000 or len(te_idx) < 500:
        print("  SKIP train/test", flush=True)
        return

    trade_params = cfg_trade_params()
    label_params = dict(trade_params)
    label_params["sl_r"] = float(getattr(cfg, "CANDLE_ENGINE_LABEL_SL_ATR", 1.0))
    if label_min_r is not None:
        label_params["entry_min_r"] = label_min_r
    if label_margin is not None:
        label_params["edge_margin"] = label_margin

    X = feats_all[FEATURE_COLS].fillna(0.0).values
    labeled = generate_labels(feats_all, atr_all, **label_params)
    Yl = labeled["entry_label"].values
    n_buy = int((Yl == 0).sum())
    n_sell = int((Yl == 1).sum())
    n_none = int((Yl == 2).sum())
    print(f"  labels BUY {n_buy} SELL {n_sell} NONE {n_none} "
          f"(label_sl={label_params['sl_r']}, entry_min_r={label_params['entry_min_r']}, "
          f"edge_margin={label_params['edge_margin']})", flush=True)
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
    tr_sel = balanced(tr_idx, per_class)
    va_sel = tr_idx[int(0.85 * len(tr_idx)):]
    model = train_model(X[tr_sel], Yl[tr_sel], X[va_sel], Yl[va_sel])

    h1_test = feats_all.iloc[te_idx]
    atr_test = atr_all.iloc[te_idx]

    # Batch-predict ALL test bars ONCE — the sweep only varies how the same
    # probabilities are thresholded / how trades are exited.
    import xgboost as xgb
    X_test = h1_test[FEATURE_COLS].fillna(0.0).values
    probs = model.predict(xgb.DMatrix(X_test))

    # Precompute jump flags per (break_r, body_r) combo — vectorized.
    jump_cache = {}
    for combo in combos:
        key = (combo["jump_break_r"], combo["jump_body_r"])
        if key not in jump_cache:
            jump_cache[key] = compute_jump_flags(
                h1_test, atr_test, key[0], key[1])

    results = []
    for combo in combos:
        ep = dict(combo)
        ep["cost_r"] = cfg_trade_params()["cost_r"]
        jump_flags = jump_cache[(combo["jump_break_r"], combo["jump_body_r"])]
        trades = simulate_engine(probs, h1_test, atr_test, jump_flags, ep)
        if len(trades) == 0:
            results.append((combo, None))
            continue
        rs = trades["r"].values
        wins = float(rs[rs > 0].sum())
        losses = float((-rs[rs < 0]).sum())
        pf = (wins / losses) if losses > 0 else float("inf")
        eq = np.cumsum(rs)
        peak = np.maximum.accumulate(eq)
        dd = float((peak - eq).max())
        results.append((combo, {
            "trades": len(rs),
            "wr": 100 * float((rs > 0).mean()),
            "exp": float(rs.mean()),
            "pf": pf,
            "net": float(rs.sum()),
            "dd": dd,
            "jump": int(trades["is_jump"].sum()),
        }))

    print(f"\n  {'min_conf':>8} {'jumpBr':>6} {'jumpBd':>6} {'sl':>4} {'rev':>5} {'trail':>5} "
          f"{'hold':>5} | {'trd':>5} {'WR%':>5} {'expR':>6} {'PF':>5} {'netR':>7} {'dd':>5} {'jmp':>4}",
          flush=True)
    for combo, m in sorted(results, key=lambda kv: (kv[1] or {}).get("pf", 0), reverse=True):
        if m is None:
            print(f"  {combo['min_conf']:>8.2f} {combo['jump_break_r']:>6.1f} {combo['jump_body_r']:>6.2f} "
                  f"{combo['sl_r']:>4.1f} {combo['reversal_r']:>5.1f} {combo['trail_r']:>5.1f} "
                  f"{combo['max_hold']:>5} |  no trades", flush=True)
            continue
        print(f"  {combo['min_conf']:>8.2f} {combo['jump_break_r']:>6.1f} {combo['jump_body_r']:>6.2f} "
              f"{combo['sl_r']:>4.1f} {combo['reversal_r']:>5.1f} {combo['trail_r']:>5.1f} "
              f"{combo['max_hold']:>5} | {m['trades']:>5} {m['wr']:>5.1f} {m['exp']:>+6.3f} "
              f"{m['pf']:>5.2f} {m['net']:>+7.1f} {m['dd']:>5.1f} {m['jump']:>4}", flush=True)


def cfg_trade_params():
    return dict(
        sl_r=float(getattr(cfg, "CANDLE_ENGINE_SL_ATR", 1.0)),
        reversal_r=float(getattr(cfg, "CANDLE_ENGINE_REVERSAL_ATR", 0.5)),
        trail_r=float(getattr(cfg, "CANDLE_ENGINE_TRAIL_ATR", 0.5)),
        max_hold=int(getattr(cfg, "CANDLE_ENGINE_MAX_HOLD_BARS", 24)),
        cost_r=float(getattr(cfg, "CANDLE_ENGINE_COST_R", 0.05)),
        entry_min_r=float(getattr(cfg, "CANDLE_ENGINE_ENTRY_MIN_R", 0.90)),
        edge_margin=float(getattr(cfg, "CANDLE_ENGINE_EDGE_MARGIN", 1.75)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="XAUUSD")
    parser.add_argument("--train-start", type=int, default=2018)
    parser.add_argument("--train-end", type=int, default=2022)
    parser.add_argument("--test-start", type=int, default=2023)
    parser.add_argument("--test-end", type=int, default=2025)
    parser.add_argument("--validate-top", action="store_true",
                        help="only run the top combos found on the tiny-data sweep")
    parser.add_argument("--quick", action="store_true",
                        help="fast edge check: trail=0.5 only, fewer params")
    parser.add_argument("--conf-sweep", action="store_true",
                        help="sweep min_conf only with the validated exit params")
    parser.add_argument("--candle-mode", action="store_true",
                        help="ride the new H1 candle direction, no confidence gate")
    parser.add_argument("--confs", type=float, nargs="+", default=[0.40, 0.45, 0.50, 0.55, 0.60],
                        help="confidence thresholds for --conf-sweep")
    parser.add_argument("--tf", type=int, default=60,
                        help="bar size in minutes (60 = H1, 30 = 30m)")
    parser.add_argument("--label-min-r", type=float, default=None,
                        help="override CANDLE_ENGINE_ENTRY_MIN_R for label generation")
    parser.add_argument("--label-margin", type=float, default=None,
                        help="override CANDLE_ENGINE_EDGE_MARGIN for label generation")
    args = parser.parse_args()

    if args.validate_top:
        # Best combos from the tiny (2021) sweep, in priority order.
        combos = [
            dict(min_conf=0.80, jump_break_r=1.0, jump_body_r=0.60, sl_r=1.5, reversal_r=0.5, trail_r=0.5, max_hold=12),
            dict(min_conf=0.80, jump_break_r=2.0, jump_body_r=0.60, sl_r=1.0, reversal_r=0.5, trail_r=0.5, max_hold=12),
            dict(min_conf=0.80, jump_break_r=1.5, jump_body_r=0.60, sl_r=1.5, reversal_r=0.5, trail_r=0.5, max_hold=12),
            dict(min_conf=0.80, jump_break_r=1.0, jump_body_r=0.70, sl_r=1.5, reversal_r=0.5, trail_r=0.5, max_hold=12),
            dict(min_conf=0.80, jump_break_r=1.5, jump_body_r=0.70, sl_r=1.5, reversal_r=0.5, trail_r=0.5, max_hold=12),
            dict(min_conf=0.80, jump_break_r=1.0, jump_body_r=0.60, sl_r=1.0, reversal_r=0.5, trail_r=0.5, max_hold=12),
            dict(min_conf=0.70, jump_break_r=1.0, jump_body_r=0.70, sl_r=1.5, reversal_r=1.0, trail_r=0.5, max_hold=24),
            dict(min_conf=0.70, jump_break_r=1.5, jump_body_r=0.70, sl_r=1.5, reversal_r=1.0, trail_r=0.5, max_hold=24),
            dict(min_conf=0.70, jump_break_r=1.0, jump_body_r=0.60, sl_r=1.5, reversal_r=1.0, trail_r=0.5, max_hold=24),
        ]
    elif args.candle_mode:
        # Ride the new H1 candle direction, no confidence floor. Sweep the
        # exit params around the validated defaults.
        combos = []
        for sl_r, reversal_r, trail_r, max_hold in itertools.product(
            [1.0, 1.5],
            [0.5, 1.0],
            [0.5],
            [12, 24],
        ):
            combos.append(dict(
                min_conf=0.0, entry_mode="candle",
                jump_break_r=1.5, jump_body_r=0.70,
                sl_r=sl_r, reversal_r=reversal_r, trail_r=trail_r, max_hold=max_hold,
            ))
    elif args.conf_sweep:
        # Trade-frequency sweep: vary min_conf only, fixed winning exits.
        combos = []
        for min_conf in args.confs:
            combos.append(dict(
                min_conf=min_conf, jump_break_r=1.5, jump_body_r=0.70,
                sl_r=1.5, reversal_r=0.5, trail_r=0.5, max_hold=24,
            ))
    elif args.quick:
        # Fast edge check: only trail=0.5 (known mandatory), 30m-scaled holds.
        combos = []
        for min_conf, jump_break_r, jump_body_r, sl_r, reversal_r, max_hold in itertools.product(
            [0.60, 0.70],
            [1.0, 1.5],
            [0.60, 0.70],
            [1.0, 1.5],
            [0.5, 1.0],
            [24, 48],
        ):
            combos.append(dict(
                min_conf=min_conf, jump_break_r=jump_break_r, jump_body_r=jump_body_r,
                sl_r=sl_r, reversal_r=reversal_r, trail_r=0.5, max_hold=max_hold,
            ))
    else:
        combos = []
        for min_conf, jump_break_r, jump_body_r, sl_r, reversal_r, trail_r, max_hold in itertools.product(
            [0.60, 0.70, 0.80],
            [1.0, 1.5, 2.0],
            [0.60, 0.70, 0.80],
            [1.0, 1.5],
            [0.5, 1.0],
            [0.5, 1.0],
            [12, 24],
        ):
            combos.append(dict(
                min_conf=min_conf, jump_break_r=jump_break_r, jump_body_r=jump_body_r,
                sl_r=sl_r, reversal_r=reversal_r, trail_r=trail_r, max_hold=max_hold,
            ))

    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        try:
            run_symbol(sym, args.train_start, args.train_end,
                       args.test_start, args.test_end, combos, tf_min=args.tf,
                       label_min_r=args.label_min_r, label_margin=args.label_margin)
        except Exception as e:
            import traceback
            print(f"\n=== {sym} FAILED: {e} ===", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
