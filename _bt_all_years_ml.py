import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import config as cfg
from app.direction_predictor import DirectionPredictor

# Import run_bt from the 2025 ML override script
from _bt_2025_ml_override import run_bt

pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
print("Model loaded.\n")

START_YEAR = 2007
END_YEAR = 2025

results = []
t0 = time.time()

for y in range(START_YEAR, END_YEAR + 1):
    yr_t0 = time.time()
    total, wr, pf, net_pnl, dd, overrides, elapsed = run_bt(y, pred)
    yr_elapsed = time.time() - yr_t0
    results.append((y, total, wr, pf, net_pnl, dd, overrides))
    print(f"\n  Year {y}: {net_pnl:>+10.2f}  WR={wr:.1f}%  PF={pf:.2f}  "
          f"DD=${dd:.0f}  tr={total}  ovr={overrides}  "
          f"({yr_elapsed:.1f}s)\n")
    print("-" * 70)

total_elapsed = time.time() - t0

print("\n" + "=" * 70)
print(f"  ML Override Backtest: {START_YEAR}-{END_YEAR}")
print("=" * 70)
print(f"  {'Year':>6} {'Trades':>8} {'WR%':>8} {'PF':>8} "
      f"{'Net PnL':>12} {'Max DD':>10} {'Overrides':>10}")
print("-" * 70)
total_tr = 0
total_wins = 0
total_net = 0.0
total_ovr = 0
for y, tr, wr, pf, net, dd, ovr in results:
    total_tr += tr
    total_wins += int(tr * wr / 100)
    total_net += net
    total_ovr += ovr
    print(f"  {y:>6} {tr:>8} {wr:>7.1f}% {pf:>7.2f} "
          f"{net:>+11.2f} {dd:>9.0f} {ovr:>10}")
print("-" * 70)
total_wr = (total_wins / total_tr * 100) if total_tr else 0
total_pf = total_wins / max(1, total_tr - total_wins) if total_tr > 0 else 0
print(f"  {'TOTAL':>6} {total_tr:>8} {total_wr:>7.1f}% {total_pf:>7.2f} "
      f"{total_net:>+11.2f} {max(dd for _,_,_,_,_,dd,_ in results):>9.0f} {total_ovr:>10}")
print("=" * 70)
print(f"  Runtime: {total_elapsed:.1f}s")
