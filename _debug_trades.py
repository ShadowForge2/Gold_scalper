import sys, os, time; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from datetime import datetime, timedelta
from app.dukascopy_client import DukascopyClient
from app.signal_engine import SignalEngine
from app.bias_engine import BiasEngine
from app.meta_strategy import MetaStrategy
from app.risk_manager import EquityScaler
import config as cfg

INITIAL_BALANCE = 20.0; CONTRACT_SIZE = 100; LEVERAGE = 200; MARGIN_MAX_PCT = 1.0
SPREAD_PER_LOT_USD = 25.0; SLIPPAGE_PER_LOT_USD = 3.0; SPREAD_ROUND_TRIP = 2

def _pnl(ep, xp, d, lot, n, et=None):
    delta = xp - ep if d == 'BUY' else ep - xp
    gross = delta * CONTRACT_SIZE * lot * n
    sm = (0.9 if et and 13 <= et.hour < 17 else 1.0 if et and 8 <= et.hour < 13 else 1.6 if et and 0 <= et.hour < 8 else 1.3)
    return gross - SPREAD_PER_LOT_USD * lot * n * SPREAD_ROUND_TRIP * sm - SLIPPAGE_PER_LOT_USD * lot * n

def check_conf(br, atr, br_win, atr_win):
    if len(br_win) < 10 or len(atr_win) < 50: return True
    lo, hi = np.percentile(atr_win, [25, 75])
    if atr <= lo: return br >= np.percentile(br_win, 60)
    if lo < atr < hi: return br >= np.percentile(br_win, 40)
    return True

client = DukascopyClient(); t0 = time.time()
year = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
print(f"\n=== Debugging {year} ===", flush=True)
m1 = client.download_range(year - 1, year)
from_dt = datetime(year, 1, 1); to_dt = datetime(year, 12, 31, 23, 59)
warmup = from_dt - timedelta(days=14)
m1 = m1[(m1['time'] >= warmup) & (m1['time'] <= to_dt)]
h1 = client.resample_to(m1, 16385); m5 = client.resample_to(m1, 5)
N = len(m5)
trade_offset = np.searchsorted(m5['time'].values, np.datetime64(from_dt), side='right')
print(f"M5 bars: {N}, offset: {trade_offset}, trade: {N-trade_offset}", flush=True)

h1_tns = h1['time'].values.astype(np.int64); m5_tns = m5['time'].values.astype(np.int64)
h1_idx = np.clip(np.searchsorted(h1_tns, m5_tns, side='right') - 2, 0, len(h1) - 1)
h1h = h1['high'].values[h1_idx]; h1l = h1['low'].values[h1_idx]; h1r = np.maximum(h1h - h1l, 1e-10)
mc = m5['close'].values; mh = m5['high'].values; ml = m5['low'].values; mo = m5['open'].values
bs = np.maximum(mh - h1h, h1l - ml) / h1r; br = np.abs(mc - mo) / np.maximum(mh - ml, 1e-10)
tr = np.maximum(mh[1:] - ml[1:], np.maximum(np.abs(mh[1:] - mc[:-1]), np.abs(ml[1:] - mc[:-1])))
atrs = np.zeros(N)
for i in range(14, N): atrs[i] = np.mean(tr[i-14:i])

be = BiasEngine(); bias = [None] * N; cache = {}; last_bh = -1
for idx in range(N):
    ts = m5['time'].iloc[idx]; bh = ts.hour
    if bh == last_bh and bias[idx-1] is not None: bias[idx] = bias[idx-1]; continue
    last_bh = bh; key = ts.replace(minute=0, second=0, microsecond=0)
    if key in cache: bias[idx] = cache[key]; continue
    hs = h1[h1['time'] < key].tail(96)
    if len(hs) >= 20: bias[idx] = be.update(hs); cache[key] = bias[idx]
    elif idx > 0: bias[idx] = bias[idx-1]

se = SignalEngine(); sigs = [None] * N
for idx in range(10, N):
    b = bias[idx]
    if b is None or b.get('bias') not in ('BULLISH','BEARISH') or b.get('strength', 0) < cfg.BIAS_STRENGTH_MIN: continue
    if bs[idx] < cfg.SIGNAL_ENTRY_THRESHOLD: continue
    win = m5.iloc[max(0, idx-50):idx]
    s = se.evaluate(win, b, float(mc[idx]), h1_high=float(h1h[idx]), h1_low=float(h1l[idx]))
    if s: sigs[idx] = s
sig_count = sum(1 for s in sigs if s is not None)
print(f"Signals: {sig_count}", flush=True)

meta = MetaStrategy(); scaler = EquityScaler(); scaler.initialize(INITIAL_BALANCE)
bal = INITIAL_BALANCE; trades = []; cur = None; cl = 0; cd = None
cur_day = None; dpnl = 0.0; br_win = []; atr_win = []
reasons = {"exit_signal": 0, "event_loss": 0, "other": 0}
monthly_dir = {m: {"buys": 0, "sells": 0, "wins": 0, "losses": 0, "pnl": 0.0} for m in range(1,13)}

for idx in range(trade_offset, N):
    ts = m5['time'].iloc[idx]; row = m5.iloc[idx]; px = float(row['open'])
    if cur_day != ts.date(): cur_day = ts.date(); dg = 0; dpnl = 0.0; cl = 0
    brv = float(br[idx]); atv = float(atrs[idx]); br_win.append(brv); atr_win.append(atv)
    if len(br_win) > 100: br_win.pop(0)
    if len(atr_win) > 100: atr_win.pop(0)
    b = bias[idx]
    if b is None: continue
    if cur is not None:
        xp = float(row['close'])
        ew = m5.iloc[max(0, idx-20):idx]
        sx, sv, sr = se.evaluate_exit(ew, cur['entry_price'], cur['direction'], cur.get('entry_score'),
                                       exit_threshold=cfg.EXIT_THRESHOLD_TIGHT, exit_mode=6, signal=cur.get('entry_signal'))
        if sx:
            p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], cur['entry_time'])
            bal += p; dpnl += p; tr = {'pnl': round(p,2), 'dir': cur['direction'], 'reason': sr, 'lot': cur['lot'], 'n': cur['num_trades']}
            trades.append(tr); reasons["exit_signal"] += 1
            cl = 0 if p > 0 else cl + 1
            cd = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + cl))
            meta.record_trade(p, b.get('strength', 0)); meta.update(bal, {'direction': b.get('bias'), 'strength': b.get('strength')})
            cur = None; continue
        if _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades']) <= -cfg.MAX_EVENT_LOSS_USD:
            p = _pnl(cur['entry_price'], xp, cur['direction'], cur['lot'], cur['num_trades'], cur['entry_time'])
            bal += p; dpnl += p; trades.append({'pnl': round(p,2), 'dir': cur['direction'], 'reason': 'event_loss', 'lot': cur['lot'], 'n': cur['num_trades']})
            reasons["event_loss"] += 1; cl += 1
            cd = ts + timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC * (1 + cl))
            meta.record_trade(p, b.get('strength', 0)); meta.update(bal, {'direction': b.get('bias'), 'strength': b.get('strength')})
            cur = None; continue
    if cd and ts < cd: continue; cd = None
    if cur is None:
        if cl >= cfg.MAX_CONSECUTIVE_LOSSES or bal < cfg.MIN_BALANCE: continue
        if dpnl <= -cfg.MAX_DAILY_LOSS_USD: continue
        sig = sigs[idx]
        if sig is None: continue
        scv = bs[idx]
        if scv < meta.current_threshold: continue
        if not check_conf(brv, atv, br_win, atr_win): continue
        lot_mult = meta.current_lot_mult
        if cfg.AGGRESSIVE_SIZING_ENABLED:
            if scv >= cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD: lot_mult *= cfg.AGGRESSIVE_VERY_STRONG_LOT_MULT
            elif scv >= cfg.AGGRESSIVE_STRONG_THRESHOLD: lot_mult *= cfg.AGGRESSIVE_STRONG_LOT_MULT
        lot = min(scaler.get_lot(bal) * lot_mult, cfg.MAX_LOT)
        lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP; lot = max(cfg.MIN_LOT, lot)
        num_tr = min(meta.current_trades_per_event, scaler.get_trades_per_event(bal, scv))
        margin_per = (float(row['close']) * CONTRACT_SIZE) / LEVERAGE
        max_by_margin = (bal * MARGIN_MAX_PCT) / margin_per
        if max_by_margin < cfg.MIN_LOT: continue
        lot = min(lot, max_by_margin); lot = round(lot / cfg.LOT_STEP) * cfg.LOT_STEP; lot = max(cfg.MIN_LOT, lot)
        num_tr = max(1, min(num_tr, int(max_by_margin / max(lot, 1e-9))))
        if lot * num_tr <= 0: continue
        cur = {'entry_time': ts, 'entry_price': px, 'direction': sig['direction'],
               'lot': lot, 'num_trades': num_tr, 'entry_score': float(scv), 'entry_signal': sig}
        mo = ts.month
        monthly_dir[mo][sig['direction'].lower() + 's' if sig['direction'] in ('BUY','SELL') else 'buys'] += 1

if cur:
    p = _pnl(cur['entry_price'], float(m5['close'].iloc[-1]), cur['direction'], cur['lot'], cur['num_trades'], cur['entry_time'])
    bal += p; trades.append({'pnl': round(p,2), 'dir': cur['direction'], 'reason': 'eoy', 'lot': cur['lot'], 'n': cur['num_trades']})

if not trades:
    print("No trades"); sys.exit(0)

pnls = np.array([t['pnl'] for t in trades])
wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
wr = len(wins)/len(pnls)*100; gp = wins.sum() if len(wins) > 0 else 0; gl = abs(losses.sum()) if len(losses) > 0 else 0
pf = gp/gl if gl > 0 else (999 if gp > 0 else 0)
cum = np.cumsum(pnls); dd = (np.maximum.accumulate(cum)-cum).max()

# Signal direction accuracy
buy_trades = [t for t in trades if t['dir'] == 'BUY']
sell_trades = [t for t in trades if t['dir'] == 'SELL']
buy_wins = sum(1 for t in buy_trades if t['pnl'] > 0)
sell_wins = sum(1 for t in sell_trades if t['pnl'] > 0)

print(f"\n=== {year} Summary ===")
print(f"Trades: {len(trades)} | WR: {wr:.1f}% | PF: {pf:.2f} | PnL: ${pnls.sum():+.2f} | DD: ${dd:.2f} | Final: ${bal:.2f}")
print(f"BUYs: {len(buy_trades)} ({buy_wins} wins, {buy_wins/len(buy_trades)*100 if buy_trades else 0:.0f}%) | SELLs: {len(sell_trades)} ({sell_wins} wins, {sell_wins/len(sell_trades)*100 if sell_trades else 0:.0f}%)")
print(f"Exit via signal: {reasons['exit_signal']} | event_loss: {reasons['event_loss']}")

# First 20 trades
print(f"\nFirst 20 trades:")
for i, t in enumerate(trades[:20]):
    print(f"  {i+1}. {t['dir']} {t['lot']:.2f}×{t['n']} PnL=${t['pnl']:+.2f} ({t['reason']})")

# Last 10 trades
print(f"\nLast 10 trades:")
for i, t in enumerate(trades[-10:]):
    print(f"  {len(trades)-10+i+1}. {t['dir']} {t['lot']:.2f}×{t['n']} PnL=${t['pnl']:+.2f} ({t['reason']})")

# PnL distribution
print(f"\nTop 5 winners: ${sorted([t['pnl'] for t in trades], reverse=True)[:5]}")
print(f"Top 5 losers: ${sorted([t['pnl'] for t in trades])[:5]}")
print(f"Avg win: ${gp/len(wins):+.2f}" if len(wins) > 0 else "")
print(f"Avg loss: ${-gl/len(losses):+.2f}" if len(losses) > 0 else "")

# Check meta threshold behavior
print(f"\nMeta threshold range: {meta.base_entry_threshold} → {meta._current_threshold}")
