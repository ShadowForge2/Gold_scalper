import pandas as pd

for year, path in [(2023, "backtest_2023_cap1000.csv"), (2025, "backtest_2025_cap1000.csv")]:
    df = pd.read_csv(path)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    start = df["exit_time"].iloc[0]
    cutoff = start + pd.DateOffset(months=2)
    
    early = df[df["exit_time"] <= cutoff].copy()
    
    print(f"=== {year}: First ~2 months (trades {early.index[0]}-{early.index[-1]}) ===")
    print(f"Period: {early['exit_time'].iloc[0]} to {early['exit_time'].iloc[-1]}")
    print(f"Trades: {len(early)}  Wins: {(early['pnl']>0).sum()}  Losses: {(early['pnl']<0).sum()}  WR: {(early['pnl']>0).mean()*100:.1f}%")
    
    bal = 20.0
    print(f"\n{'#':<4} {'Date':<16} {'Dir':<5} {'Entry':<8} {'Exit':<8} {'PnL':<10} {'Lot':<8} {'Reason':<22} {'Bal':<8}")
    print("-" * 85)
    for i, (_, t) in enumerate(early.iterrows()):
        bal += t["pnl"]
        wd = t.get("withdrawn", 0)
        if wd > 0:
            bal -= wd
        print(f"{i+1:<4} {t['exit_time'].strftime('%m/%d %H:%M'):<16} {t['direction']:<5} "
              f"{t['entry_price']:<8} {t['exit_price']:<8} ${t['pnl']:<+7.2f} {t['lot']:<8.5f} "
              f"{t['exit_reason']:<22} ${bal:<6.2f}")
        if wd > 0:
            print(f"     --- WITHDRAW ${wd:<.2f} -> bal=${bal:<.2f}")
    
    print(f"\nEnding balance: ${bal:.2f}")
    print(f"Total withdrawn: ${early['withdrawn'].sum():.2f}")
    print()
