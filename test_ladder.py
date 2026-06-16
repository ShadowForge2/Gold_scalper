import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import config as cfg
cfg.MIN_LOT = 0.01
cfg.MAX_LOT = 100.0
cfg.LOT_SIZE = 0.01
cfg.LOT_STEP = 0.01
from app.risk_manager import EquityScaler

balances = [20, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
sc = EquityScaler()
sc.initialize(20.0)
for b in balances:
    sc.peak_balance = b
    lot = sc.get_lot(b)
    print(f"  Start ${b:>6,}: lot={lot:.4f}")
