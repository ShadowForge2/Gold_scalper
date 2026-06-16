import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from app.capital_client import CapitalClient

client = CapitalClient()
client.initialize(api_key=cfg.CAPITAL_API_KEY, identifier=cfg.CAPITAL_IDENTIFIER,
                  password=cfg.CAPITAL_PASSWORD, demo=cfg.CAPITAL_DEMO)

info = client.get_symbol_info("XAUUSD")
print(f"Bid={info['bid']} Ask={info['ask']} Spread={info['spread']}")

h1 = client.get_rates("XAUUSD", 16385, 5)
if h1 is not None and len(h1) >= 2:
    print(f"\nH1 -2: high={h1.iloc[-2]['high']} low={h1.iloc[-2]['low']}")
    print(f"H1 -1: high={h1.iloc[-1]['high']} low={h1.iloc[-1]['low']}")
    print(h1[["time","high","low","close"]].to_string(index=False))

    # Simulate signal check
    current_price = info["ask"]  # BUY uses ask
    h1_high = h1.iloc[-2]["high"]
    h1_low = h1.iloc[-2]["low"]
    range_sz = h1_high - h1_low
    if current_price > h1_high:
        bd = current_price - h1_high
        score = min(bd / range_sz, 1.0)
        print(f"\nBUY signal would fire: breakout_dist={bd:.2f} score={score:.3f} (need >=0.50)")
    else:
        print(f"\nNo BUY: price {current_price} <= H1 high {h1_high} (need to break above by {h1_high - current_price:.2f})")

    current_price_sell = info["bid"]
    if current_price_sell < h1_low:
        bd = h1_low - current_price_sell
        score = min(bd / range_sz, 1.0)
        print(f"SELL signal would fire: breakout_dist={bd:.2f} score={score:.3f} (need >=0.50)")
    else:
        print(f"No SELL: price {current_price_sell} >= H1 low {h1_low} (need to break below by {current_price_sell - h1_low:.2f})")

m5 = client.get_rates("XAUUSD", 5, 10)
if m5 is not None:
    print(f"\nLast 5 M5 candles:")
    print(m5[["time","high","low","close"]].tail(5).to_string(index=False))

client.shutdown()
