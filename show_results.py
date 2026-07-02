import json
d = json.load(open('sweep_confirmation_results.json'))
print(f"{'Mode':<16} {'Trades':>6} {'Net PnL':>8} {'WR%':>6} {'PF':>6} {'Max DD':>8} {'Balance':>9}")
print("-"*65)
for r in d:
    print(f'{r["name"]:<16} {r["trades"]:>6} {r["net_pnl"]:>+8.2f} {r["win_rate"]:>5.1f}% {r["profit_factor"]:>6.2f} {r["max_dd"]:>8.2f} {r["final_bal"]:>9.2f}')
