import sys,os,time; sys.path.insert(0,os.path.dirname(__file__))
import numpy as np
from datetime import datetime, timedelta
from app.dukascopy_client import DukascopyClient
from app.signal_engine import SignalEngine
from app.bias_engine import BiasEngine
from app.meta_strategy import MetaStrategy
from app.risk_manager import EquityScaler
import config as cfg

BAL=20.0; CS=100; LEV=200; SPR=25.0; SLIP=3.0; RT=2
def pnl(ep,xp,d,l,n,et=None):
    dlt=xp-ep if d=='BUY' else ep-xp
    g=dlt*CS*l*n
    sm=(0.9 if et and 13<=et.hour<17 else 1.0 if et and 8<=et.hour<13 else 1.6 if et and 0<=et.hour<8 else 1.3)
    return g-SPR*l*n*RT*sm-SLIP*l*n
def conf(br,atr,bw,aw):
    if len(bw)<10 or len(aw)<50: return True
    l,h=np.percentile(aw,[25,75])
    if atr<=l: return br>=np.percentile(bw,60)
    if l<atr<h: return br>=np.percentile(bw,40)
    return True

def exit_no_sl(ew, entry, direction, score, signal):
    """Exit only on TP, momentum, or wick rejection. NO HARD SL."""
    momentum=ew['close'].iloc[-5:].pct_change().mean()*100 if len(ew)>=5 else 0
    px=float(ew['close'].iloc[-1]); diff=px-entry if direction=='BUY' else entry-px
    if signal is None: return False,0.0,'no_signal'
    tp1,tp2=signal.get('tp1'),signal.get('tp2')
    if tp1 is None: return False,0.0,'no_tp'
    if direction=='BUY':
        if px>=tp2:
            trail=px-float(ew['high'].iloc[-5:].std()*2) if len(ew)>=5 else px-0.5
            if px<=trail: return True,1.0,'trail_after_tp2'
            if momentum<0 and diff>0: return True,0.8,'momentum_exhausted'
        if px>=tp1:
            lock=entry+(tp1-entry)*0.3
            if px<=lock: return True,0.7,'lock_tp1'
            if momentum<0: return True,0.8,'momentum_exhausted'
            candle=ew.iloc[-1]
            wick=(candle['high']-max(candle['close'],candle['open']))/(candle['high']-candle['low']+1e-10)
            if wick>0.6: return True,0.9,'wick_rejection'
        prog=(px-entry)/(tp1-entry+1e-10)
        if prog>=0.8:
            candle=ew.iloc[-1]
            wick=(candle['high']-max(candle['close'],candle['open']))/(candle['high']-candle['low']+1e-10)
            if wick>0.6: return True,0.9,'wick_rejection'
    else:
        if px<=tp2:
            trail=px+float(ew['low'].iloc[-5:].std()*2) if len(ew)>=5 else px+0.5
            if px>=trail: return True,1.0,'trail_after_tp2'
            if momentum>0 and diff>0: return True,0.8,'momentum_exhausted'
        if px<=tp1:
            lock=entry-(entry-tp1)*0.3
            if px>=lock: return True,0.7,'lock_tp1'
            if momentum>0: return True,0.8,'momentum_exhausted'
            candle=ew.iloc[-1]
            wick=(min(candle['close'],candle['open'])-candle['low'])/(candle['high']-candle['low']+1e-10)
            if wick>0.6: return True,0.9,'wick_rejection'
        prog=(entry-px)/(entry-tp1+1e-10)
        if prog>=0.8:
            candle=ew.iloc[-1]
            wick=(min(candle['close'],candle['open'])-candle['low'])/(candle['high']-candle['low']+1e-10)
            if wick>0.6: return True,0.9,'wick_rejection'
    # No exit yet — keep holding
    return False,0.0,'holding'

client=DukascopyClient()
for yr in [2022,2023,2024,2025]:
    t0=time.time()
    m1=client.download_range(yr-1,yr)
    fd=datetime(yr,1,1); td=datetime(yr,12,31,23,59)
    m1=m1[(m1['time']>=fd-timedelta(days=14))&(m1['time']<=td)]
    h1=client.resample_to(m1,16385); m5=client.resample_to(m1,5)
    N=len(m5); off=np.searchsorted(m5['time'].values,np.datetime64(fd),side='right')
    h1_t=h1['time'].values.astype(np.int64); m5_t=m5['time'].values.astype(np.int64)
    hi=np.clip(np.searchsorted(h1_t,m5_t,side='right')-2,0,len(h1)-1)
    h1h,h1l=h1['high'].values[hi],h1['low'].values[hi]; h1r=np.maximum(h1h-h1l,1e-10)
    mc,ml,mh,mo=m5['close'].values,m5['low'].values,m5['high'].values,m5['open'].values
    bs=np.maximum(mh-h1h,h1l-ml)/h1r; br=np.abs(mc-mo)/np.maximum(mh-ml,1e-10)
    tr=np.maximum(mh[1:]-ml[1:],np.maximum(np.abs(mh[1:]-mc[:-1]),np.abs(ml[1:]-mc[:-1])))
    atrs=np.zeros(N)
    for i in range(14,N): atrs[i]=np.mean(tr[i-14:i])
    be=BiasEngine(); bias=[None]*N; cache={}; lb=-1
    for idx in range(N):
        ts=m5['time'].iloc[idx]; bh=ts.hour
        if bh==lb and bias[idx-1] is not None: bias[idx]=bias[idx-1]; continue
        lb=bh; k=ts.replace(minute=0,second=0,microsecond=0)
        if k in cache: bias[idx]=cache[k]; continue
        hs=h1[h1['time']<k].tail(96)
        if len(hs)>=20: bias[idx]=be.update(hs); cache[k]=bias[idx]
        elif idx>0: bias[idx]=bias[idx-1]
    se=SignalEngine(); sigs=[None]*N
    for idx in range(10,N):
        b=bias[idx]
        if b is None or b.get('bias') not in ('BULLISH','BEARISH') or b.get('strength',0)<cfg.BIAS_STRENGTH_MIN: continue
        if bs[idx]<cfg.SIGNAL_ENTRY_THRESHOLD: continue
        win=m5.iloc[max(0,idx-50):idx]
        s=se.evaluate(win,b,float(mc[idx]),h1_high=float(h1h[idx]),h1_low=float(h1l[idx]))
        if s: sigs[idx]=s
    meta=MetaStrategy(); sc=EquityScaler(); sc.initialize(BAL)
    bal=BAL; trds=[]; cur=None; cl=0; cd=None; cd2=None; dpnl=0.0; bw=[]; aw=[]
    reasons={"win":0,"loss":0}
    for idx in range(off,N):
        ts=m5['time'].iloc[idx]; row=m5.iloc[idx]; px=float(row['open'])
        if cd2!=ts.date(): cd2=ts.date(); dg=0; dpnl=0.0; cl=0
        bv=float(br[idx]); av=float(atrs[idx]); bw.append(bv); aw.append(av)
        if len(bw)>100: bw.pop(0)
        if len(aw)>100: aw.pop(0)
        b=bias[idx]
        if b is None: continue
        if cur is not None:
            xp=float(row['close'])
            ew=m5.iloc[max(0,idx-20):idx]
            # Check event loss first
            unrealized=pnl(cur['entry_price'],xp,cur['direction'],cur['lot'],cur['num_trades'])
            if unrealized<=-cfg.MAX_EVENT_LOSS_USD:
                p=pnl(cur['entry_price'],xp,cur['direction'],cur['lot'],cur['num_trades'],cur['entry_time'])
                bal+=p; dpnl+=p; trds.append(round(p,2))
                if p>0: reasons["win"]+=1
                else: reasons["loss"]+=1
                cl=0 if p>0 else cl+1
                cd=ts+timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC*(1+cl))
                meta.record_trade(p,b.get('strength',0)); meta.update(bal,{'direction':b.get('bias'),'strength':b.get('strength')})
                cur=None; continue
            # Exit check — NO HARD SL
            sx,sv,sr=exit_no_sl(ew,cur['entry_price'],cur['direction'],cur.get('entry_score'),cur.get('entry_signal'))
            if sx:
                p=pnl(cur['entry_price'],xp,cur['direction'],cur['lot'],cur['num_trades'],cur['entry_time'])
                bal+=p; dpnl+=p; trds.append(round(p,2))
                if p>0: reasons["win"]+=1
                else: reasons["loss"]+=1
                cl=0 if p>0 else cl+1
                cd=ts+timedelta(seconds=cfg.RE_ENTRY_COOLDOWN_SEC*(1+cl))
                meta.record_trade(p,b.get('strength',0)); meta.update(bal,{'direction':b.get('bias'),'strength':b.get('strength')})
                cur=None; continue
        if cd and ts<cd: continue; cd=None
        if cur is None:
            if cl>=cfg.MAX_CONSECUTIVE_LOSSES or bal<cfg.MIN_BALANCE: continue
            if dpnl<=-cfg.MAX_DAILY_LOSS_USD: continue
            sig=sigs[idx]
            if sig is None: continue
            scv=bs[idx]
            if scv<meta.current_threshold: continue
            if not conf(bv,av,bw,aw): continue
            lm=meta.current_lot_mult
            if cfg.AGGRESSIVE_SIZING_ENABLED:
                if scv>=cfg.AGGRESSIVE_VERY_STRONG_THRESHOLD: lm*=cfg.AGGRESSIVE_VERY_STRONG_LOT_MULT
                elif scv>=cfg.AGGRESSIVE_STRONG_THRESHOLD: lm*=cfg.AGGRESSIVE_STRONG_LOT_MULT
            lot=min(sc.get_lot(bal)*lm,cfg.MAX_LOT)
            lot=round(lot/cfg.LOT_STEP)*cfg.LOT_STEP; lot=max(cfg.MIN_LOT,lot)
            nt=min(meta.current_trades_per_event,sc.get_trades_per_event(bal,scv))
            mg=(float(row['close'])*CS)/LEV
            mx=(bal*1.0)/mg
            if mx<cfg.MIN_LOT: continue
            lot=min(lot,mx); lot=round(lot/cfg.LOT_STEP)*cfg.LOT_STEP; lot=max(cfg.MIN_LOT,lot)
            nt=max(1,min(nt,int(mx/max(lot,1e-9))))
            if lot*nt<=0: continue
            cur={'entry_time':ts,'entry_price':px,'direction':sig['direction'],'lot':lot,'num_trades':nt,'entry_score':float(scv),'entry_signal':sig}
    if cur:
        p=pnl(cur['entry_price'],float(m5['close'].iloc[-1]),cur['direction'],cur['lot'],cur['num_trades'],cur['entry_time'])
        bal+=p; trds.append(round(p,2))
        if p>0: reasons["win"]+=1
        else: reasons["loss"]+=1
    if trds:
        pnls=np.array(trds); wins=pnls[pnls>0]; losses=pnls[pnls<0]
        wr=len(wins)/len(pnls)*100 if len(pnls)>0 else 0
        gp=wins.sum() if len(wins)>0 else 0; gl=abs(losses.sum()) if len(losses)>0 else 0
        pf=gp/gl if gl>0 else (999 if gp>0 else 0)
        cum=np.cumsum(pnls); dd=(np.maximum.accumulate(cum)-cum).max()
        print(f'{yr}: {len(pnls):>4} tr WR={wr:>5.1f}% PF={pf:>6.2f} PnL=${pnls.sum():>+10.2f} DD=${dd:>9.2f} Bal=${bal:>9.2f} [{time.time()-t0:.0f}s]')
    else:
        print(f'{yr}: no trades')
