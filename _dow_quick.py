"""
Quick DOW analysis for small account startups (2020-2025).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from app.direction_predictor import DirectionPredictor
from _dow_analysis import run_bt_year

pred = DirectionPredictor.load("models/direction_xgb_m5.joblib")
all_entries = []
for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    r = run_bt_year(year, pred)
    if r: all_entries.append(r["entries"])

combined = pd.concat(all_entries, ignore_index=True)
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

print("=== FIRST TRADE PERFORMANCE BY DAY OF WEEK (2020-2025) ===")
print("  Measures what happens if you deposit $20 and start on each day")
print()
for dow in range(5):
    day_df = combined[combined["dow"] == dow].sort_values("entry_time")
    if len(day_df) == 0: continue
    print(f"  {day_names[dow]}:")
    for n in [1, 2, 3, 5, 10, 20]:
        first_n = day_df.head(n)
        bal = 20.0 + first_n["profit"].sum()
        wr = len(first_n[first_n["profit"] > 0]) / n * 100
        print(f"    After {n:>2} trades: bal=${bal:>+6.2f}  WR={wr:.0f}%")

print()
print("=== MEANINGFUL WIN RATE (>$1 profit) ===")
for dow in range(5):
    day_df = combined[combined["dow"] == dow]
    total = len(day_df)
    meaningful = len(day_df[day_df["profit"] > 1.0])
    small_loss = len(day_df[(day_df["profit"] <= 0) & (day_df["profit"] > -1.0)])
    big_loss = len(day_df[day_df["profit"] <= -1.0])
    print(f"  {day_names[dow]:<10}: {total:>4}t | >$1: {meaningful:>4} ({meaningful/total*100:.0f}%) | "
          f"-$1-0: {small_loss:>4} | <-$1: {big_loss:>4} ({big_loss/total*100:.0f}%)")

# Which day has best early streak?
print()
print("=== EARLY DRAWDOWN ANALYSIS ===")
print("  First 5 trades: what's the chance of being underwater?")
for dow in range(5):
    day_df = combined[combined["dow"] == dow].sort_values("entry_time")
    if len(day_df) < 5: continue
    first5 = day_df.head(5)
    cum = 20.0 + first5["profit"].cumsum()
    min_bal = cum.min()
    final_bal = cum.iloc[-1]
    underwater = min_bal < 20.0
    print(f"  {day_names[dow]:<10}: min_bal=${min_bal:>+6.2f} final=${final_bal:>+6.2f} "
          f"{'UNDERWATER' if underwater else 'GREEN'} "
          f"({'BAD START' if final_bal < 18 else 'OK' if final_bal < 20 else 'GOOD'})")

# Which day's trades have best win rate by session?
print()
print("=== BEST DAY OVERALL (weighted score) ===")
print("  Score = win_rate * 0.3 + profit_factor_rank * 0.3 + early_balance * 0.4")
print()
for dow in range(5):
    day_df = combined[combined["dow"] == dow]
    wins = day_df[day_df["profit"] > 0]
    losses = day_df[day_df["profit"] <= 0]
    wr = len(wins)/len(day_df)*100
    gw = wins["profit"].sum() if len(wins) > 0 else 0
    gl = abs(losses["profit"].sum()) if len(losses) > 0 else 0
    pf = gw/gl if gl > 0 else 999
    net = day_df["profit"].sum()

    # First trade impact
    sorted_d = day_df.sort_values("entry_time")
    first3 = sorted_d.head(3)
    early_net = first3["profit"].sum()

    print(f"  {day_names[dow]:<10}: WR={wr:.1f}%  PF={pf:.2f}  Net=${net:>+9.2f}  "
          f"First3=${early_net:>+7.2f}  Trades={len(day_df)}")
