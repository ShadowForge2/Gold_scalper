import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from app.capital_client import CapitalClient

client = CapitalClient()
client.initialize(api_key=cfg.CAPITAL_API_KEY, identifier=cfg.CAPITAL_IDENTIFIER,
                  password=cfg.CAPITAL_PASSWORD, demo=cfg.CAPITAL_DEMO)

info = client.get_symbol_info("XAUUSD")
print(f"Price: Bid={info['bid']} Ask={info['ask']}")

h1 = client.get_rates("XAUUSD", 16385, 5)
print(f"\nH1 candles:")
print(h1[["time","high","low","close"]].to_string(index=False))

h1_high = h1.iloc[-2]["high"]
h1_low = h1.iloc[-2]["low"]
range_sz = h1_high - h1_low
ask = info["ask"]

if ask > h1_high:
    bd = ask - h1_high
    score = min(bd / range_sz, 1.0)
    print(f"\n>>> BUY possible! breakout_dist={bd:.2f} score={score:.3f} (need ≥0.50)")
    if score >= 0.50:
        print(">>> WOULD ENTER NOW!")
else:
    need = h1_high - ask
    print(f"\nNo BUY: need +{need:.2f} above H1 high {h1_high}")
    print(f"  For score 0.50, need +{need + range_sz*0.5:.2f} above current")

bid = info["bid"]
if bid < h1_low:
    bd = h1_low - bid
    score = min(bd / range_sz, 1.0)
    print(f"SELL possible! breakout_dist={bd:.2f} score={score:.3f}")
else:
    need = bid - h1_low
    print(f"No SELL: need -{need:.2f} below H1 low {h1_low}")

client.shutdown()
