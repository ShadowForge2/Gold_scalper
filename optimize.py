"""
Fast parameter sweep — pre-computes once, then 1.7s per combo.
"""

import itertools, sys, os, csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from backtest import load_and_compute, run_backtest

GRID = {
    "entry_threshold": [0.60, 0.65, 0.70, 0.75],
    "exit_threshold": [0.60, 0.70, 0.80, 0.90],
    "base_trades_per_event": [3, 5, 7, 10],
    "sessions": ["LONDON,NEW_YORK"],
    "bias_strength_min": [0.3],
}
KEYS = list(GRID.keys())
COMBOS = list(itertools.product(*(GRID[k] for k in KEYS)))
print(f"Combinations: {len(COMBOS)}", flush=True)

print("Pre-computing signals...", flush=True)
data = load_and_compute()
if data is None:
    sys.exit(1)
print(f"Done. {len(data['sig_df'])} candles.", flush=True)

results = []
start = datetime.now()

for i, combo in enumerate(COMBOS):
    params = dict(zip(KEYS, combo))
    elapsed = (datetime.now() - start).total_seconds()
    eta = (elapsed / (i + 1)) * (len(COMBOS) - i - 1) if i > 0 else 0
    print(f"[{i+1:2d}/{len(COMBOS)}] entry={params['entry_threshold']} "
          f"exit={params['exit_threshold']} tr={params['base_trades_per_event']}  "
          f"ETA={eta:.0f}s", end="", flush=True)

    try:
        r = run_backtest(data, params, verbose=False)
        if r:
            r["params"] = params
            results.append(r)
            print(f"  -> ${r['net_pnl']:.2f} WR={r['win_rate']:.1f}% "
                  f"PF={r['profit_factor']:.2f} DD=${r['max_dd']:.2f}", flush=True)
        else:
            print("  -> None", flush=True)
    except Exception as e:
        print(f"  -> ERROR: {e}", flush=True)

total = (datetime.now() - start).total_seconds()
print(f"\nDone: {len(results)} runs in {total:.0f}s", flush=True)
if not results:
    sys.exit(0)

# Score ranking
results.sort(key=lambda r: r["score"], reverse=True)
print("\n" + "=" * 90)
print("TOP 10 — Score (PnL - 2xDD)")
print("=" * 90)
print(f"{'Rank':<5} {'PnL':>8} {'WR':>6} {'PF':>7} {'DD':>8} {'Ret':>7} "
      f"{'Tr':>4} {'Score':>8}  Params")
print("-" * 90)
for rank, r in enumerate(results[:10], 1):
    p = r["params"]
    print(f"#{rank:<3} ${r['net_pnl']:>6.2f} {r['win_rate']:>5.1f}% "
          f"{r['profit_factor']:>6.2f}x ${r['max_dd']:>6.2f} "
          f"{r['return_pct']:>6.2f}% {r['trades']:>4} "
          f"{r['score']:>7.2f}  "
          f"entry={p['entry_threshold']} exit={p['exit_threshold']} "
          f"tr={p['base_trades_per_event']}", flush=True)

# PF ranking
results.sort(key=lambda r: r["profit_factor"], reverse=True)
print("\nTOP 5 — Profit Factor")
for rank, r in enumerate(results[:5], 1):
    p = r["params"]
    print(f"#{rank:<3} ${r['net_pnl']:>6.2f} {r['win_rate']:>5.1f}% "
          f"{r['profit_factor']:>6.2f}x ${r['max_dd']:>6.2f} "
          f"entry={p['entry_threshold']} exit={p['exit_threshold']} "
          f"tr={p['base_trades_per_event']}", flush=True)

# PnL ranking
results.sort(key=lambda r: r["net_pnl"], reverse=True)
print("\nTOP 5 — Net P&L")
for rank, r in enumerate(results[:5], 1):
    p = r["params"]
    print(f"#{rank:<3} ${r['net_pnl']:>6.2f} {r['win_rate']:>5.1f}% "
          f"{r['profit_factor']:>6.2f}x ${r['max_dd']:>6.2f} "
          f"entry={p['entry_threshold']} exit={p['exit_threshold']} "
          f"tr={p['base_trades_per_event']}", flush=True)

results.sort(key=lambda r: r["score"], reverse=True)
csv_path = os.path.join(os.path.dirname(__file__), "optimize_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[k for k in results[0].keys() if k != "params"])
    w.writeheader()
    for r in results:
        w.writerow({k: v for k, v in r.items() if k != "params"})
print(f"\nSaved: optimize_results.csv", flush=True)

best = results[0]
p = best["params"]
print(f"\n{'=' * 90}")
print("RECOMMENDED PARAMETERS")
print(f"{'=' * 90}")
print(f"  SIGNAL_ENTRY_THRESHOLD = {p['entry_threshold']}")
print(f"  EXIT_THRESHOLD         = {p['exit_threshold']}")
print(f"  MAX_TRADES_PER_EVENT   = {p['base_trades_per_event']}")
print(f"  ALLOWED_SESSIONS       = '{p['sessions']}'")
print(f"  BIAS_STRENGTH_MIN      = {p['bias_strength_min']}")
print(f"  Expected: ${best['net_pnl']:.2f} profit, {best['win_rate']:.1f}% WR, "
      f"{best['profit_factor']:.2f} PF, ${best['max_dd']:.2f} max DD", flush=True)
