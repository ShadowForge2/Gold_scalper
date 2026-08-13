"""Verify vectorized sweep vs _tune_pull_prevh1.run() on cached XAUUSD."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _sweep_pull_params import build_all
from _sweep_pull_params_vec import eval_vec_full, build_req
from _tune_pull_prevh1 import run

ball = build_all("XAUUSD")
t = pd.DatetimeIndex(ball["t"])
b = dict(c=ball["c"], t=t, pa=ball["pa"], h1dir=ball["h1dir"],
         cost_r=ball["cost_r"], n=len(ball["c"]))

configs = [(0.10, 0.15, 12), (0.15, 0.25, 24), (0.20, 0.35, 6), (0.30, 0.50, 48)]
for pull, trail, H in configs:
    bv = dict(b, year_mask=(t.year >= 2024) & (t.year <= 2024))
    a = run("XAUUSD", bv, pull, trail, H)
    c = eval_vec_full(ball, 2024, 2024, pull, trail, H)
    if a is None or c is None:
        print(f"pull {pull} trail {trail} H {H}: one None (a={a is not None}, c={c is not None})")
        continue
    ok = abs(a["net"] - c["net"]) < 1e-6 and a["trd"] == c["trd"] and abs(a["pf"] - c["pf"]) < 1e-6
    print(f"pull {pull} trail {trail} H {H}: "
          f"py trd={a['trd']} net={a['net']:.1f} pf={a['pf']:.3f} | "
          f"vec trd={c['trd']} net={c['net']:.1f} pf={c['pf']:.3f} | {'MATCH' if ok else 'MISMATCH'}")
