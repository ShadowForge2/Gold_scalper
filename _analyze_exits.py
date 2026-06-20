import pandas as pd

for label, path in [("2025", "backtest_2025_live.csv"), ("2023", "backtest_2023_live.csv")]:
    trades = pd.read_csv(path)
    print(f"=== EXIT REASON DISTRIBUTION ({label}) ===")
    print(trades["exit_reason"].value_counts().to_string())
    print()
    losers = trades[trades["pnl"] < 0]
    print("=== LOSSES BY EXIT REASON ===")
    for r in losers["exit_reason"].unique():
        subset = losers[losers["exit_reason"] == r]
        avg = subset["pnl"].mean()
        mn = subset["pnl"].min()
        mx = subset["pnl"].max()
        print(f"  {r}: count={len(subset):3d}  avg=${avg:>+8.2f}  min=${mn:>+8.2f}  max=${mx:>+8.2f}")
    print()
