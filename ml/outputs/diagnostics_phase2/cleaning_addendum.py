from pathlib import Path
import os,sys,re,hashlib,struct,zlib

ROOT=Path('/mnt/c/Users/Srijan Gupta/Downloads/Developement/AQUIS/ml')
OUT=ROOT/'outputs/diagnostics_phase2'
OUT.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'venv312/lib/python3.12/site-packages'))
os.chdir(ROOT)
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CUTOFF=pd.Timestamp('2025-10-01')
SIXH=pd.Timedelta(hours=6)
DAY=pd.Timedelta(days=1)
WIN=120 # 30 days on the six-hour grid; fixed analysis window, not a learned threshold.
TEST_START=pd.Timestamp('2025-10-01')
TEST_END=pd.Timestamp('2026-10-01')
KNOWN_CODES={99.0,-99.0,999.0,-999.0,9999.0,-9999.0}
GUARD_NAMES=['Mathura Chaumuha Navagaon-11','Block Office, Kurara','Hamirpur Sarila Purani']

def norm(x): return str(x).strip().casefold()
def slug(x): return re.sub(r'[^A-Za-z0-9]+','_',str(x)).strip('_')[:64]+'_'+hashlib.md5(str(x).encode()).hexdigest()[:8]
def md(df,cols=None,digits=4):
    if cols is not None: df=df[cols]
    def f(x):
        if pd.isna(x): return ''
        if isinstance(x,(float,np.floating)): return f'{x:.{digits}f}'
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(f(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def chunk(t,b): return struct.pack('>I',len(b))+t+b+struct.pack('>I',zlib.crc32(t+b)&0xffffffff)
def draw_line(pix,w,h,x0,y0,x1,y1,col):
    dx=abs(x1-x0); sx=1 if x0<x1 else -1; dy=-abs(y1-y0); sy=1 if y0<y1 else -1; err=dx+dy
    while True:
        if 0<=x0<w and 0<=y0<h:
            i=(y0*w+x0)*4; pix[i:i+4]=bytes((*col,255))
        if x0==x1 and y0==y1: break
        e2=2*err
        if e2>=dy: err+=dy; x0+=sx
        if e2<=dx: err+=dx; y0+=sy
def draw_plot(path,g,masked,flag_flat,flag_flip,break_times):
    w,h=1400,440; pix=bytearray(bytes((255,255,255,255))*(w*h))
    def dot(x,y,c,r=2):
        for xx in range(x-r,x+r+1):
            for yy in range(y-r,y+r+1):
                if 0<=xx<w and 0<=yy<h:
                    i=(yy*w+xx)*4; pix[i:i+4]=bytes((*c,255))
    for panel,(left,right) in enumerate([(58,690),(748,1380)]):
        top,bottom=22,398
        vals=g.gwl.to_numpy(float)
        ymin=float(np.nanmin(vals)); ymax=float(np.nanmax(vals)); pad=max((ymax-ymin)*.05,.25); ymin-=pad; ymax+=pad
        for k in range(6):
            y=int(top+(bottom-top)*k/5); draw_line(pix,w,h,left,y,right,y,(228,232,235))
        for k in range(7):
            x=int(left+(right-left)*k/6); draw_line(pix,w,h,x,top,x,bottom,(238,240,242))
        draw_line(pix,w,h,left,top,left,bottom,(55,62,70)); draw_line(pix,w,h,left,bottom,right,bottom,(55,62,70))
        times=pd.to_datetime(g.time).astype('int64').to_numpy(); den=max(int(times.max()-times.min()),1)
        coords=[]
        values=vals.copy()
        if panel==1: values[np.asarray(masked,dtype=bool)]=np.nan
        for t,v in zip(times,values):
            if not np.isfinite(v): coords.append(None); continue
            x=int(left+(right-left)*(int(t)-int(times.min()))/den); y=int(bottom-(bottom-top)*(v-ymin)/(ymax-ymin))
            coords.append((x,y))
        prev=None
        for xy in coords:
            if xy is None: prev=None; continue
            if prev is not None: draw_line(pix,w,h,prev[0],prev[1],xy[0],xy[1],(70,118,102) if panel else (116,130,141))
            prev=xy
        flags=np.asarray(flag_flat,dtype=bool)|np.asarray(flag_flip,dtype=bool)
        obs=pd.to_numeric(g.gwl_obs,errors='coerce').to_numpy()==1
        for (xy,seen,fl) in zip(coords,obs,flags):
            if xy is not None and seen: dot(xy[0],xy[1],(23,105,130) if not fl else (205,104,28),2)
        for bt in break_times:
            if len(times):
                bx=int(left+(right-left)*(int(pd.Timestamp(bt).value)-int(times.min()))/den)
                if left<=bx<=right: draw_line(pix,w,h,bx,top,bx,bottom,(178,58,54))
        # Panel identification without font dependencies: top rule colors differ; report supplies the station name.
    scan=b''.join(b'\x00'+pix[y*w*4:(y+1)*w*4] for y in range(h))
    data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(scan,6))+chunk(b'IEND',b'')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)

def exact_runs(g):
    z=g[['Station','time','gwl','gwl_obs']].copy()
    same_station=z.Station.eq(z.Station.shift())
    same_time=(z.time-z.time.shift())==SIXH
    same_value=z.gwl.eq(z.gwl.shift()) & z.gwl.notna() & z.gwl.shift().notna()
    z['_start']=~(same_station&same_time&same_value)
    z['_rid']=z.groupby('Station',sort=False)['_start'].cumsum()
    z['_obs']=z.gwl_obs.eq(1).astype('int8')
    runs=z.groupby(['Station','_rid'],sort=False).agg(start=('time','min'),end=('time','max'),value=('gwl','first'),
        slots=('gwl','size'),observed_slots=('_obs','sum')).reset_index()
    return runs

def observed_reversal(g,step_cut):
    obs_mask=(g.gwl_obs.to_numpy()==1)&g.gwl.notna().to_numpy()
    obs_idx=np.flatnonzero(obs_mask)
    vals=g.gwl.to_numpy(float)[obs_idx]
    times=pd.to_datetime(g.time).astype('int64').to_numpy()[obs_idx]
    d=np.diff(vals)
    dt=np.diff(times)
    valid=(dt>0)&(dt<=pd.Timedelta(days=2).value)&np.isfinite(d)
    large=valid&(np.abs(d)>=step_cut)&(np.abs(d)>0)
    rev=np.zeros(len(obs_idx),dtype=bool)
    if len(d)>1:
        rev[2:]=large[1:]&large[:-1]&(np.sign(d[1:])!=np.sign(d[:-1]))
    if len(times):
        cs=np.cumsum(rev.astype(np.int32))
        left=np.searchsorted(times,times-pd.Timedelta(days=30).value,side='left')
        counts=cs-np.where(left>0,cs[np.maximum(left-1,0)],0)
    else: counts=np.zeros(0,dtype=float)
    return obs_idx,times,counts

def invalid_observed_slots(times,counts,threshold):
    invalid=np.zeros(len(times),dtype=bool)
    for i in np.flatnonzero(np.isfinite(counts)&(counts>threshold)):
        left=np.searchsorted(times,times[i]-pd.Timedelta(days=30).value,side='left')
        invalid[left:i+1]=True
    return invalid

# Load raw aligned table once; never mutate/write this input.
raw=pd.read_parquet(ROOT/'data/aligned/table_6h.parquet',columns=['Station','time','gwl','gwl_obs'])
raw['time']=pd.to_datetime(raw.time); raw['gwl']=pd.to_numeric(raw.gwl,errors='coerce'); raw['gwl_obs']=pd.to_numeric(raw.gwl_obs,errors='coerce')
raw=raw.sort_values(['Station','time']).reset_index(drop=True)
groups={st:g.reset_index(drop=True) for st,g in raw.groupby('Station',sort=False)}
phase1=pd.read_csv(OUT/'outlier_forensics.csv')
union=set(phase1.Station.astype(str))
train=raw[(raw.time<CUTOFF)&(raw.gwl_obs==1)&raw.gwl.notna()].copy()

# Training-only exact-run length threshold.
train_runs=exact_runs(train.sort_values(['Station','time']).reset_index(drop=True))
obsrun=train_runs[train_runs.observed_slots==train_runs.slots]
run_q=float(obsrun.slots.quantile(.9999)) if len(obsrun) else 1.0
N=max(1,int(np.ceil(run_q)))

# Training-only station tail limits and exact extreme repeat threshold.
tail_limits={}; tail_frequencies=[]; exact_value_counts={}
for st,g in train.groupby('Station',sort=False):
    vals=g.gwl.to_numpy(float); lo,hi=np.quantile(vals,[.005,.995]) if len(vals) else (np.nan,np.nan)
    tail_limits[str(st)]=(float(lo),float(hi))
    vc=pd.Series(vals).value_counts()
    exact_value_counts[str(st)]=vc
    tail_frequencies.extend([int(n) for val,n in vc.items() if val<=lo or val>=hi])
repeat_q=float(np.quantile(tail_frequencies,.95)) if tail_frequencies else 2.0
REPEAT_MIN=max(2,int(np.ceil(repeat_q)))

# Training-only 30-day rolling flatness threshold.
flat_train=[]
for st,g in groups.items():
    pre_g=g[(g.time<CUTOFF)&g.gwl.notna()]
    if len(pre_g)<80: continue
    std=pre_g.set_index('time').gwl.rolling('30D',min_periods=80).std().dropna()
    if len(std): flat_train.extend(std.to_numpy(float).tolist())
flat_threshold=float(np.quantile(flat_train,.01)) if flat_train else 0.0

# Training-only p95 consecutive-observation step thresholds; rolling 30-day reversal-count and station invalid-share cutoffs.
step_cutoffs={}; pre_rev_counts=[]; pre_invalid_shares={}
pre_rec_cache={}
for st,g in groups.items():
    obs_mask=(g.gwl_obs.to_numpy()==1)&g.gwl.notna().to_numpy()
    obs_idx=np.flatnonzero(obs_mask); ot=pd.to_datetime(g.time).astype('int64').to_numpy()[obs_idx]
    ov=g.gwl.to_numpy(float)[obs_idx]; d0=np.diff(ov); dt=np.diff(ot)
    hist_mask=(dt>0)&(dt<=pd.Timedelta(days=2).value)&(ot[1:]<CUTOFF.value)
    hist=np.abs(d0[hist_mask&np.isfinite(d0)])
    cut=float(np.quantile(hist,.95)) if len(hist) else 0.0
    step_cutoffs[st]=cut
    oi,times,counts=observed_reversal(g,cut)
    pre_mask=times<CUTOFF.value
    pre_counts=counts[pre_mask]
    pre_counts=pre_counts[np.isfinite(pre_counts)]
    if len(pre_counts): pre_rev_counts.extend(pre_counts.tolist())
    pre_rec_cache[st]=(oi,times,counts,pre_mask)
REV_THRESHOLD=float(np.quantile(pre_rev_counts,.995)) if pre_rev_counts else 0.0
REV_THRESHOLD=float(np.ceil(REV_THRESHOLD))
for st,g in groups.items():
    oi,times,counts,pre_mask=pre_rec_cache[st]
    invalid=invalid_observed_slots(times,counts,REV_THRESHOLD)
    pre_obs=pre_mask
    pre_invalid_shares[st]=float((invalid&pre_obs).sum()/pre_obs.sum()) if pre_obs.sum() else 0.0
INVALID_SHARE_THRESHOLD=float(np.quantile(list(pre_invalid_shares.values()),.99)) if pre_invalid_shares else 0.0

# Training-only station-specific monthly median shift thresholds; pooled p99.5 is the fallback.
monthly=[]; cp_cutoffs={}
for st,g in train.groupby('Station',sort=False):
    m=g.assign(month=g.time.dt.to_period('M')).groupby('month').gwl.median().sort_index()
    delta=m.diff().abs().dropna()
    if len(delta):
        a=delta.to_numpy(float); monthly.extend(a.tolist()); cp_cutoffs[st]=float(np.quantile(a,.995))
CP_THRESHOLD=float(np.quantile(monthly,.995)) if monthly else float('inf')
for st in groups: cp_cutoffs.setdefault(st,CP_THRESHOLD)

thresholds=pd.DataFrame([
 dict(detector='constant_value_run',threshold_slots=N,fit='ceil(p99.99 of observed exact-value run lengths in pre-2025-10-01 training data)',window_slots=1),
 dict(detector='repeated_exact_extreme',threshold_slots=REPEAT_MIN,fit='max(2, ceil(p95 of pre-cutoff exact-value frequencies among station tail values <=p0.5 or >=p99.5))',window_slots=1),
 dict(detector='extreme_value',threshold_slots=np.nan,fit='station-specific pre-cutoff p0.5/p99.5; known sentinels 99,-99,999,-999,9999,-9999 are also candidates',window_slots=1),
 dict(detector='near_flat_30d_std',threshold_slots=flat_threshold,fit='p1 of pre-cutoff 30d rolling standard deviations; requires >=80 slots',window_slots=WIN),
 dict(detector='flipflop_step_size',threshold_slots=np.nan,fit='station-specific p95 of pre-cutoff observed consecutive-read absolute differences (6h to 48h apart)',window_slots=WIN),
 dict(detector='flipflop_reversal_count',threshold_slots=REV_THRESHOLD,fit='ceil(p99.5 of pre-cutoff rolling 30d count of consecutive large sign reversals)',window_slots=WIN),
 dict(detector='whole_station_invalid_share',threshold_slots=INVALID_SHARE_THRESHOLD,fit='p99 of per-station pre-cutoff share of observed slots covered by flipflop windows',window_slots=WIN),
 dict(detector='persistent_month_shift',threshold_slots=np.nan,fit=f'station-specific pre-cutoff p99.5 of consecutive observed monthly-median absolute changes; pooled fallback={CP_THRESHOLD:.4f}m; confirmation uses next available month',window_slots=0),
])
thresholds.to_csv(OUT/'cleaning_detector_thresholds.csv',index=False)

# Candidate repeated extremes; exact-value repetition counts are based on observed values.
repeated_values={}; pre_q_by_station={}
for st,g in groups.items():
    lo,hi=tail_limits.get(st,(np.nan,np.nan)); pre_q_by_station[st]=(lo,hi)
    v=g.gwl.to_numpy(float); obs=g.gwl_obs.to_numpy()==1
    full_vc=pd.Series(v[obs&np.isfinite(v)]).value_counts()
    extreme_counts={float(x):int(n) for x,n in full_vc.items() if (x in KNOWN_CODES) or (np.isfinite(lo) and (x<=lo or x>=hi))}
    repeated_values[st]={x:n for x,n in extreme_counts.items() if n>=REPEAT_MIN}

# Current-period event detectors and candidate mask; segment shifts are descriptive, not deleted.
runs=exact_runs(raw)
rows_by_station={}; run_by_station={st:g.copy() for st,g in runs.groupby('Station',sort=False)}
change_segments=[]; breakpoints={st:[] for st in groups}
flat_by_station={}; flip_by_station={}; spike_masks={}; placeholder_masks={}; stuck_test_masks={}
monthly_medians={}
for st,g in groups.items():
    n=len(g); dates=g.time
    r=run_by_station[st]
    lo,hi=pre_q_by_station[st]
    # Extreme initial run: starts at first series row, lasts beyond N, and equals a sentinel or training tail.
    first_time=g.time.min(); first=r[r.start==first_time]
    placeholder=np.zeros(n,dtype=bool)
    if len(first):
        fr=first.iloc[0]; val=float(fr.value) if pd.notna(fr.value) else np.nan
        extreme=(val in KNOWN_CODES) or (np.isfinite(lo) and (val<=lo or val>=hi))
        if int(fr.slots)>N and extreme:
            placeholder=((g.time>=fr.start)&(g.time<=fr.end)).to_numpy()
    placeholder_masks[st]=placeholder
    # Exact long runs are flagged, not automatically dropped. Test-window subset is reported separately.
    flat=np.zeros(n,dtype=bool); stuck=np.zeros(n,dtype=bool)
    for rr in r[r.slots>N].itertuples(index=False):
        runmask=((dates>=rr.start)&(dates<=rr.end)).to_numpy()
        flat|=runmask
        testmask=runmask&(dates.to_numpy()>=TEST_START.to_datetime64())&(dates.to_numpy()<TEST_END.to_datetime64())
        if testmask.sum()>N and int(rr.observed_slots)>=2: stuck|=testmask
    flat_by_station[st]=flat; stuck_test_masks[st]=stuck
    # Per-point rolling near-flat score using pre-fitted 30-day SD cutoff.
    s=g.set_index('time').gwl
    roll_std=s.rolling('30D',min_periods=80).std().to_numpy()
    nearflat=np.isfinite(roll_std)&(roll_std<=flat_threshold)
    flat_by_station[st] |= nearflat
    # Large sign-reversal windows at the pre-fitted station step size and global pre-fitted reversal threshold.
    oi,ot,counts,_pre=pre_rec_cache[st]
    invalid_obs=invalid_observed_slots(ot,counts,REV_THRESHOLD)
    invalid=np.zeros(n,dtype=bool); invalid[oi]=invalid_obs
    flip_by_station[st]=invalid
    # Spike removal is deliberately narrow: tail/sentinel value repeated in training/test and immediate exact reversion.
    vals=g.gwl.to_numpy(float); obs=g.gwl_obs.to_numpy()==1; times=g.time.to_numpy()
    removed=np.zeros(n,dtype=bool); vc=repeated_values.get(st,{})
    for i in range(1,n-1):
        val=float(vals[i]) if np.isfinite(vals[i]) else np.nan
        if not obs[i] or not np.isfinite(val) or val not in vc: continue
        if obs[i-1] and obs[i+1] and times[i]-times[i-1]==SIXH.to_timedelta64() and times[i+1]-times[i]==SIXH.to_timedelta64() and vals[i-1]==vals[i+1] and val!=vals[i-1]:
            removed[i]=True
    spike_masks[st]=removed
    # Persistent level shifts: monthly medians; adjacent month delta above pre threshold and confirmed by next available month.
    gm=g[(g.gwl_obs==1)&g.gwl.notna()].copy()
    gm['month']=gm.time.dt.to_period('M')
    m=gm.groupby('month').gwl.median().sort_index(); monthly_medians[st]=m
    breakpoints[st]=[]
    cp_cut=cp_cutoffs.get(st,CP_THRESHOLD)
    if len(m)>=3 and np.isfinite(cp_cut):
        mv=m.to_numpy(float); mp=list(m.index)
        for j in range(1,len(m)-1):
            delta=mv[j]-mv[j-1]
            if abs(delta)>cp_cut and abs(mv[j+1]-mv[j])<=cp_cut:
                start=pd.Timestamp(mp[j].start_time)
                breakpoints[st].append((start,mv[j-1],mv[j],float(delta),float(cp_cut)))
    # Write segment intervals; every segment kept as data pending review, flagged transitions define boundaries.
    bps=breakpoints[st]
    bounds=[g.time.min()]+[b[0] for b in bps]+[g.time.max()+SIXH]
    for sid,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
        pre_seg=b<=CUTOFF
        change=bps[sid-1] if sid>0 else None
        change_segments.append(dict(Station=st,segment_id=sid+1,start=a,end=b-SIXH,
            transition_from_previous=change is not None,transition_delta_m=change[3] if change else np.nan,
            threshold_m=change[4] if change else np.nan,
            rows=int(((g.time>=a)&(g.time<b)).sum()),segment_status='kept; segment boundary retained for separate feature history' if not change else 'kept pending review; do not join feature history across detected shift'))
    # Summary counts from detectors.
    allv=np.isfinite(vals); rep_slots=sum(int((obs&(vals==x)).sum()) for x in vc)
    rows_by_station[st]=dict(Station=st,pre_cutoff_observed_slots=int(((g.time<CUTOFF)&obs).sum()),
        exact_extreme_values_repeated=int(len(vc)),repeated_extreme_observed_slots=rep_slots,
        isolated_repeated_extreme_spikes_to_mask=int(removed.sum()),placeholder_start_slots=int(placeholder.sum()),
        flat_exact_run_slots=int(flat.sum()),nearflat_30d_slots=int(nearflat.sum()),test_stuck_slots=int(stuck.sum()),
        flipflop_invalid_slots=int(invalid.sum()),flipflop_invalid_share_all_observed=float((invalid&obs).sum()/obs.sum()) if obs.sum() else 0.,
        flipflop_invalid_share_pre_observed=float(pre_invalid_shares.get(st,0.)),
        exceeds_pre_whole_station_invalid_share=bool(pre_invalid_shares.get(st,0.)>INVALID_SHARE_THRESHOLD),
        change_points=int(len(bps)),segments=int(len(bps)+1),
        rows_in_training_station_extreme_support=int(sum(1 for val in vc if val in exact_value_counts.get(st,pd.Series(dtype=int)).index)),
        first_record=g.time.min(),last_record=g.time.max())

detector=pd.DataFrame(rows_by_station.values()).sort_values('Station')
detector.to_csv(OUT/'station_detector_review.csv',index=False)
pd.DataFrame(change_segments).to_csv(OUT/'change_point_segments.csv',index=False)

# Quality test-window stuck segments and Phase 1 XGB error around entry into / during the stuck value.
stuck_runs=[]
for st,g in groups.items():
    for rr in run_by_station[st][run_by_station[st].slots>N].itertuples(index=False):
        a=max(pd.Timestamp(rr.start),TEST_START); b=min(pd.Timestamp(rr.end),TEST_END-SIXH)
        slots=int(((g.time>=a)&(g.time<=b)).sum()) if a<=b else 0
        if slots>N and int(rr.observed_slots)>=2:
            stuck_runs.append(dict(Station=st,run_start=rr.start,run_end=rr.end,value=rr.value,all_run_slots=rr.slots,
                observed_slots=rr.observed_slots,test_overlap_slots=slots,contains_exact_ceiling_or_floor=bool(float(rr.value) in KNOWN_CODES)))
stuck_runs=pd.DataFrame(stuck_runs)
pred=pd.read_parquet(ROOT/'outputs/predictions_2026.parquet')
pred=pred[pred.horizon==30].copy(); pred['target_time']=pd.to_datetime(pred.time)+pd.to_timedelta(pred.horizon,unit='D')
pred['sqerr_xgb']=(pred.xgb-pred.target)**2
stuck_error=[]
for rr in stuck_runs.itertuples(index=False):
    g=pred[pred.Station.astype(str)==str(rr.Station)].copy()
    start=pd.Timestamp(rr.run_start); end=pd.Timestamp(rr.run_end)
    trans=g[(g.target_time>=start)&(g.target_time<start+DAY*30)]
    inside=g[(g.target_time>=start)&(g.target_time<=end)]
    all_sq=float(g.sqerr_xgb.sum())
    stuck_error.append(dict(Station=rr.Station,run_start=start,run_end=end,stuck_value=rr.value,
        transition_rows=len(trans),transition_sq_error=float(trans.sqerr_xgb.sum()),transition_share_station_sq_error=float(trans.sqerr_xgb.sum()/all_sq) if all_sq else np.nan,
        targets_inside_stuck_rows=len(inside),inside_stuck_sq_error=float(inside.sqerr_xgb.sum()),
        inside_stuck_share_station_sq_error=float(inside.sqerr_xgb.sum()/all_sq) if all_sq else np.nan,
        all_station_xgb_sq_error=all_sq))
stuck_runs.to_csv(OUT/'test_stuck_runs.csv',index=False)
pd.DataFrame(stuck_error).to_csv(OUT/'test_stuck_phase1_error.csv',index=False)

# Exact 27.5 audit for Dera Devra bujurg.
dera='Primary School, Dera Devra bujurg'
dg=groups.get(dera,pd.DataFrame(columns=raw.columns))
d275=dg[dg.gwl==27.5]
d_runs=run_by_station.get(dera,pd.DataFrame())
d_runs=d_runs[d_runs.value==27.5] if len(d_runs) else d_runs
dera_audit=pd.DataFrame([dict(Station=dera,value=27.5,occurrences_all_slots=len(d275),occurrences_observed=int((d275.gwl_obs==1).sum()),
    first_occurrence=d275.time.min() if len(d275) else pd.NaT,last_occurrence=d275.time.max() if len(d275) else pd.NaT,
    contiguous_runs=len(d_runs),longest_run_slots=int(d_runs.slots.max()) if len(d_runs) else 0,
    run_lengths_slots=';'.join(map(str,d_runs.slots.astype(int).tolist())) if len(d_runs) else '',
    max_run_months=int(max((rr.end.to_period('M').ordinal-rr.start.to_period('M').ordinal+1) for rr in d_runs.itertuples(index=False))) if len(d_runs) else 0,
    sentinel_code=27.5 in KNOWN_CODES,pre_tail_value=(27.5<=pre_q_by_station.get(dera,(np.nan,np.nan))[0] or 27.5>=pre_q_by_station.get(dera,(np.nan,np.nan))[1]) if np.isfinite(pre_q_by_station.get(dera,(np.nan,np.nan))[0]) else False)])
dera_audit.to_csv(OUT/'dera_27_5_audit.csv',index=False)
d275.assign(month=d275.time.dt.to_period('M').astype(str)).groupby('month').agg(occurrences=('gwl','size'),observed_occurrences=('gwl_obs',lambda s:int((s==1).sum())),first=('time','min'),last=('time','max')).reset_index().to_csv(OUT/'dera_27_5_month_counts.csv',index=False)

# Training-only pumping recovery shape detector. Thresholds are per-station p99.5 of pre-period observed 1d changes;
# require non-exact gradual recovery to the preceding 7d median within 30d.
pump_rows=[]
pump_scan=union|{x for x in groups if norm(x) in {norm(a) for a in GUARD_NAMES}}
for st in pump_scan:
    if st not in groups: continue
    g=groups[st]
    v=g.gwl.to_numpy(float); t=pd.to_datetime(g.time); obs=g.gwl_obs.to_numpy()==1
    one=np.full(len(g),np.nan); one[4:]=v[4:]-v[:-4]
    pre_ix=(t<CUTOFF).to_numpy()&obs&np.isfinite(one)
    jump_cut=float(np.quantile(np.abs(one[pre_ix]),.995)) if pre_ix.any() else np.inf
    # Training baseline residual tolerance: p99.5 of absolute departure from prior 7d observed median.
    ser=pd.Series(np.where(obs,v,np.nan))
    baseline=ser.shift(1).rolling(28,min_periods=4).median().to_numpy()
    dev=np.abs(v-baseline)
    tol=float(np.quantile(dev[pre_ix&np.isfinite(dev)],.995)) if (pre_ix&np.isfinite(dev)).any() else 0.0
    for i in range(len(g)-120):
        if not obs[i] or not np.isfinite(dev[i]) or dev[i]<=max(jump_cut,tol): continue
        j=min(i+120,len(g)-1)
        if t.iloc[j]-t.iloc[i]<DAY*20 or not obs[j] or not np.isfinite(baseline[i]): continue
        if abs(v[j]-baseline[i])>max(tol,.05): continue
        # Several directional increments, no one-step exact reversion, and leave-out spike rule didn't remove it.
        seg=v[i:j+1]; delta=np.diff(seg); direction=np.sign(baseline[i]-v[i])
        moves=int(((np.sign(delta)==direction)&(np.abs(delta)>0)).sum())
        if moves>=3:
            pump_rows.append(dict(Station=st,event_time=t.iloc[i],baseline_7d_m=baseline[i],excursion_value_m=v[i],
                excursion_m=dev[i],recovery_value_20_30d_m=v[j],recovery_days=(t.iloc[j]-t.iloc[i]).days,
                recovery_steps_toward_baseline=moves,precutoff_jump_threshold_m=jump_cut,
                pre_cutoff_recovery_tolerance_m=tol,exact_value_repetition_count=int(np.sum(v==v[i]))))
pump_candidates=pd.DataFrame(pump_rows).drop_duplicates(['Station','event_time']) if pump_rows else pd.DataFrame(columns=['Station','event_time'])
episodes=[]
if len(pump_candidates):
    for st,pg in pump_candidates.sort_values('event_time').groupby('Station'):
        pg=pg.sort_values('event_time').copy()
        pg['_episode']=(pg.event_time.diff().dt.total_seconds().div(86400).fillna(999)>30).cumsum()
        for eid,eg in pg.groupby('_episode'):
            best=eg.loc[eg.excursion_m.idxmax()].drop(labels=['_episode']).to_dict()
            best['episode_start']=eg.event_time.min(); best['episode_end']=eg.event_time.max(); best['candidate_points']=len(eg)
            episodes.append(best)
pumps=pd.DataFrame(episodes) if episodes else pd.DataFrame(columns=['Station','event_time'])
pumps.to_csv(OUT/'pumping_like_smooth_recovery_events.csv',index=False)

# Named clean-series guard rows; report that no observed points were masked unless a narrow repeated-spike rule matches.
guard=[]
for name in GUARD_NAMES:
    st=next((x for x in groups if norm(x)==norm(name)),None)
    if st is None:
        guard.append(dict(requested_station=name,matched_station='',found=False,spike_masked_slots=0,pumping_like_events=0)); continue
    guard.append(dict(requested_station=name,matched_station=st,found=True,spike_masked_slots=int(spike_masks[st].sum()),
        pumping_like_events=int((pumps.Station==st).sum()) if len(pumps) else 0,
        spike_rule='repeated pre-tail/sentinel exact value + observed immediate exact reversion'))
pd.DataFrame(guard).to_csv(OUT/'pumping_guard_review.csv',index=False)

# Data-gap forensics from observed timestamps: adjacent observations separated >=90d, crossing 2024-04..2025-06.
gap_rows=[]
for st,g in groups.items():
    tt=g.loc[g.gwl_obs==1,'time'].drop_duplicates().sort_values().reset_index(drop=True)
    if len(tt)<2: continue
    dif=tt.diff()
    for i in np.flatnonzero((dif>=pd.Timedelta(days=90)).to_numpy()):
        a=tt.iloc[i-1]; b=tt.iloc[i]
        if a>=pd.Timestamp('2024-04-01') and a<pd.Timestamp('2025-01-01') and b>pd.Timestamp('2025-01-01') and b<=pd.Timestamp('2025-07-01'):
            gap_rows.append(dict(Station=st,last_observed_before_gap=a,first_observed_after_gap=b,gap_days=(b-a).total_seconds()/86400))
gaps=pd.DataFrame(gap_rows)
if len(gaps):
    common_start=gaps.last_observed_before_gap.max(); common_end=gaps.first_observed_after_gap.min()
    if common_start>=common_end:
        common_start=gaps.last_observed_before_gap.quantile(.5); common_end=gaps.first_observed_after_gap.quantile(.5)
    gap_summary=dict(stations_affected=gaps.Station.nunique(),gap_station_intervals=len(gaps),
        first_last_observation=gaps.last_observed_before_gap.min(),latest_last_observation=gaps.last_observed_before_gap.max(),
        earliest_first_return=gaps.first_observed_after_gap.min(),last_first_return=gaps.first_observed_after_gap.max(),
        median_gap_days=gaps.gap_days.median(),common_gap_start=common_start,common_gap_end=common_end,
        common_interval_is_intersection=bool(gaps.last_observed_before_gap.max()<gaps.first_observed_after_gap.min()))
else:
    common_start=pd.Timestamp('2024-06-01'); common_end=pd.Timestamp('2025-03-01')
    gap_summary=dict(stations_affected=0,gap_station_intervals=0,first_last_observation=pd.NaT,latest_last_observation=pd.NaT,
        earliest_first_return=pd.NaT,last_first_return=pd.NaT,median_gap_days=np.nan,common_gap_start=common_start,common_gap_end=common_end,common_interval_is_intersection=False)
gaps.to_csv(OUT/'common_data_gap_by_station.csv',index=False)

daily_obs=raw[raw.gwl_obs==1].assign(day=lambda x:x.time.dt.floor('D')).groupby('day').Station.nunique().rename('stations_observed').reset_index()
daily_obs.to_csv(OUT/'common_data_gap_daily_coverage.csv',index=False)
om=raw[raw.gwl_obs==1].assign(month=lambda x:x.time.dt.to_period('M').astype(str))
monthly_obs=om.groupby('month').agg(stations_observed=('Station','nunique'),observed_slots=('Station','size'),median_slots_per_station=('Station',lambda s:float(s.value_counts().median()))).reset_index()
monthly_obs.to_csv(OUT/'common_data_gap_monthly_coverage.csv',index=False)
pd.DataFrame([gap_summary]).to_csv(OUT/'common_data_gap_summary.csv',index=False)

# Count feature rows with anchors/targets in or intervals overlapping the common interval, streaming Parquet row groups.
affected_stations=set(gaps.Station.astype(str)) if len(gaps) else set()
def count_features(path,scope='all stations'):
    if not path.exists(): return dict(rows=0,anchors_in_gap=0,targets_in_gap=0,target_intervals_overlapping_gap=0,targets_crossing_into_gap=0)
    pf=pq.ParquetFile(path); total=anchor=target=overlap=cross=0
    for batch in pf.iter_batches(batch_size=200000,columns=['Station','time','target_time']):
        d=batch.to_pandas()
        if scope=='affected gap stations': d=d[d.Station.astype(str).isin(affected_stations)]
        t=pd.to_datetime(d.time); q=pd.to_datetime(d.target_time)
        total+=len(d); anchor+=int(((t>=common_start)&(t<common_end)).sum()); target+=int(((q>=common_start)&(q<common_end)).sum())
        overlap+=int(((t<common_end)&(q>=common_start)).sum()); cross+=int(((t<common_start)&(q>=common_start)&(q<common_end)).sum())
    return dict(rows=total,anchors_in_gap=anchor,targets_in_gap=target,target_intervals_overlapping_gap=overlap,targets_crossing_into_gap=cross)
feature_counts=[]
for split in ['train','val']:
    for scope in ['all stations','affected gap stations']:
        c=count_features(ROOT/f'data/features/{split}.parquet',scope); feature_counts.append(dict(split=split,scope=scope,**c))
pd.DataFrame(feature_counts).to_csv(OUT/'common_data_gap_feature_rows.csv',index=False)

# Prepare the requested random audit set, plus the three named guard stations.
allstations=sorted(set(raw.Station.astype(str)))
remaining=[s for s in allstations if s not in union]
random10=sorted(np.random.default_rng(20261026).choice(remaining,size=min(10,len(remaining)),replace=False).tolist())
plot_names=set(union)|set(random10)
plot_names.update(x for x in allstations if norm(x) in {norm(a) for a in GUARD_NAMES})
plot_manifest=[]
plotdir=OUT/'before_after'
for st in sorted(plot_names):
    g=groups[st]
    masked=placeholder_masks[st]|spike_masks[st]
    flat=flat_by_station[st]; flip=flip_by_station[st]
    draw_plot(plotdir/f'{slug(st)}_before_after.png',g,masked,flat,flip,[x[0] for x in breakpoints[st]])
    plot_manifest.append(dict(Station=st,is_union=st in union,is_random_other=st in random10,is_named_guard=norm(st) in {norm(a) for a in GUARD_NAMES},
        plot=f'before_after/{slug(st)}_before_after.png',masked_placeholder_slots=int(placeholder_masks[st].sum()),
        masked_repeated_spike_slots=int(spike_masks[st].sum()),flat_slots_flagged=int(flat.sum()),flipflop_slots_flagged=int(flip.sum()),change_points=len(breakpoints[st])))
pd.DataFrame(plot_manifest).to_csv(OUT/'before_after_plot_manifest.csv',index=False)

# Strict primary labels, separate flags retained in the station detector table.
detector['pumping_like_events']=detector.Station.map(pumps.groupby('Station').size()).fillna(0).astype(int) if len(pumps) else 0
detector['strict_classification']=np.select([
    detector.placeholder_start_slots>0,
    (detector.flipflop_invalid_share_all_observed>INVALID_SHARE_THRESHOLD)&(detector.flipflop_invalid_slots>0),
    detector.isolated_repeated_extreme_spikes_to_mask>0,
    detector.pumping_like_events>0,
    detector.test_stuck_slots>0,
],['placeholder-start','flip-flop','spike-faults on clean series','pumping-like real','stuck'],default='clean')
detector.to_csv(OUT/'station_detector_review.csv',index=False)
un=detector[detector.Station.isin(union)].copy().merge(phase1[['Station','classification']].rename(columns={'classification':'phase2_classification'}),on='Station',how='left')
un=un.rename(columns={'strict_classification':'strict_classification'})
un['classification_disagrees']=np.where((un.phase2_classification=='consistent with datum/telemetry fault') != un.strict_classification.isin(['placeholder-start','stuck','flip-flop','spike-faults on clean series']),True,False)
un.to_csv(OUT/'union_strict_reclassification.csv',index=False)

# Per-event exact repeated extreme audit across the union (and all data summary).
rep_rows=[]
for st in sorted(union):
    g=groups[st]; lo,hi=pre_q_by_station[st]
    obs=g[g.gwl_obs==1].gwl.dropna(); vc=obs.value_counts()
    for val,n in vc.items():
        if val in KNOWN_CODES or (np.isfinite(lo) and (val<=lo or val>=hi)):
            rep_rows.append(dict(Station=st,value=float(val),observed_occurrences=int(n),
                qualifies_repeated=(int(n)>=REPEAT_MIN),is_known_sentinel=(val in KNOWN_CODES),
                pre_p005=lo,pre_p995=hi,month_count=int(g[(g.gwl==val)&(g.gwl_obs==1)].time.dt.to_period('M').nunique())))
pd.DataFrame(rep_rows).to_csv(OUT/'union_repeated_extreme_values.csv',index=False)

# Guard series: prove whether named episodes were masked, plus observations from detector tables.
guarddf=pd.DataFrame(guard)
guarddf=guarddf.merge(detector[['Station','strict_classification','placeholder_start_slots','test_stuck_slots','flipflop_invalid_share_all_observed','isolated_repeated_extreme_spikes_to_mask']],left_on='matched_station',right_on='Station',how='left')
guarddf.to_csv(OUT/'pumping_guard_review.csv',index=False)

# Replace any prior addendum generated by this script and append the new review to the single report.
report_path=OUT/'phase2_report.md'
old=report_path.read_text(encoding='utf-8')
marker='\n## Cleaning addendum: multi-pattern detectors and review plots\n'
if marker in old: old=old.split(marker)[0]
report=[]
report += [marker.strip(),'','### Threshold fitting and safe-treatment rules','',
    'All numeric detector cutoffs below were fitted from `gwl_obs=1` data with timestamps strictly before 2025-10-01. The 30-day/120-slot window is the module horizon. Exact repeated-spike masking additionally requires an observed isolated exact value, repeated at least the training-fitted count, with immediate observed exact reversion to the same value on both sides. Smooth-recovery events are recorded and kept. Flat, flip-flop, and change-point segments are flagged for review; they are not silently deleted.', '',
    md(thresholds,['detector','threshold_slots','fit','window_slots']), '',
    '### 39-station strict labels and disagreements','',
    md(un.sort_values('Station'),['Station','phase2_classification','strict_classification','classification_disagrees','placeholder_start_slots','isolated_repeated_extreme_spikes_to_mask','test_stuck_slots','flipflop_invalid_share_pre_observed','flipflop_invalid_share_all_observed','change_points','pumping_like_events']), '',
    'Disagreement means Phase 2 called the station a general datum/telemetry-fault candidate but this stricter primary detector label is not one of placeholder-start, stuck, flip-flop, or isolated spike-faults (or vice versa). Individual detectors can overlap; the separate flags remain visible.', '',
    '### Station-level detector counts','',
    md(detector.groupby('strict_classification').agg(stations=('Station','nunique'),placeholder_slots=('placeholder_start_slots','sum'),spike_slots=('isolated_repeated_extreme_spikes_to_mask','sum'),stuck_slots=('test_stuck_slots','sum'),flip_share_mean=('flipflop_invalid_share_all_observed','mean')).reset_index(),['strict_classification','stations','placeholder_slots','spike_slots','stuck_slots','flip_share_mean']), '',
    '### Whole-station flip-flop exclusion proposal','',
    f'Proposed train-only rule: exclude a station from one pooled fit if its pre-cutoff observed-slot invalid share (slots covered by a 30-day window with more than {REV_THRESHOLD:.0f} large sign reversals) exceeds the pre-cutoff station-share p99 = {INVALID_SHARE_THRESHOLD:.6f}. Large-step thresholds are station-specific pre-cutoff p95 6-hour absolute changes; reversal-count threshold is the pre-cutoff pooled p99.5. `station_detector_review.csv` reports Hathras and all stations. The rule is fitted without test errors; if Hathras is below the historical cutoff, report that it is detected in test but not eligible for pre-only whole-station exclusion.', '',
    '### Test-window stuck runs and Phase 1 XGBoost error contribution','',
    f'Test window is 2025-10-01 through 2026-09-30; a stuck run is an exact flat-value run longer than the pre-fitted {N}-slot cutoff. Error contribution is XGBoost squared error for 30-day targets in the first 30 days after run entry and for targets inside the stuck interval.', '',
    md(pd.DataFrame(stuck_error).sort_values('transition_sq_error',ascending=False) if stuck_error else pd.DataFrame(columns=['Station','transition_rows','transition_sq_error','transition_share_station_sq_error','targets_inside_stuck_rows','inside_stuck_sq_error']),None), '',
    '### Dera Devra bujurg exact 27.5 audit','',md(dera_audit), '',
    'Interpretation: exact 27.5 occurrence/run/month counts are above. The h18 artifact stores absolute levels: anchor 27.500 and target -7.921 both match observed raw slots, so the anomaly is a real recorded level transition in this aligned table. If 27.5 also forms repeated runs, that repetition is reported; it does not by itself prove 27.5 is a placeholder. This evidence supersedes any interpretation that treated `obs` as a change value.', '',
    '### Pumping-style recovery guard','',
    'The detector records large excursions only when they return smoothly to their prior 7-day level within 20-30 days; exact repeated-value immediate reversions are handled by the separate narrow spike mask. The three guard plots show raw values on the left and after-candidate values on the right. `spike_masked_slots=0` means the signal was left untouched.', '',md(guarddf), '',
    'Smooth-recovery candidate events (all stations):','',md(pumps.head(100)) if len(pumps) else 'No event met the training-calibrated smooth-recovery criteria.', '',
    '### Common observed-data gap and feature-row impact','',
    md(pd.DataFrame([gap_summary])), '',
    'Feature row counts use saved 30-day `time` and `target_time` columns. `target_intervals_overlapping_gap` counts forecast intervals intersecting the station-agnostic common gap; `targets_in_gap` counts endpoints in that interval.', '',md(pd.DataFrame(feature_counts)), '',
    'Affected station-specific exact observation gaps:','',md(gaps.sort_values('gap_days',ascending=False).head(100)) if len(gaps) else 'No station gap met the 90-day and calendar-window rule.', '',
    '### Before/after PNG review set','',
    f'Generated {len(union)} union station plots, 10 deterministic random non-union stations (seed 20261026), plus named guard stations if not already included. Left panel is the source series; right panel applies only placeholder-start and exact repeated-spike masks. Flat/stuck and flip-flop slots stay visible and are highlighted; red vertical markers indicate change-point segment boundaries.', '',
    md(pd.DataFrame(plot_manifest)[['Station','is_union','is_random_other','is_named_guard','plot','masked_placeholder_slots','masked_repeated_spike_slots','flat_slots_flagged','flipflop_slots_flagged','change_points']]), '',
    '### 27.5 monthly counts','',md(pd.read_csv(OUT/'dera_27_5_month_counts.csv')) if (OUT/'dera_27_5_month_counts.csv').stat().st_size else 'No 27.5 months.', '',
    '### Addendum outputs','',
    '- `cleaning_detector_thresholds.csv`; `station_detector_review.csv`; `change_point_segments.csv`; `union_strict_reclassification.csv`.',
    '- `test_stuck_runs.csv`; `test_stuck_phase1_error.csv`; `dera_27_5_audit.csv`; `dera_27_5_month_counts.csv`; `union_repeated_extreme_values.csv`.',
    '- `pumping_like_smooth_recovery_events.csv`; `pumping_guard_review.csv`.',
    '- `common_data_gap_summary.csv`; `common_data_gap_by_station.csv`; `common_data_gap_daily_coverage.csv`; `common_data_gap_monthly_coverage.csv`; `common_data_gap_feature_rows.csv`.',
    '- `before_after_plot_manifest.csv`; PNGs are under `before_after/`.','']
for r in plot_manifest:
    if r['is_union'] or r['is_named_guard']:
        report.append(f'- [{r["Station"]}]({r["plot"]})')
old=old.rstrip()+'\n\n'+'\n'.join(report)+'\n'
report_path.write_text(old,encoding='utf-8')

print(f'training run threshold N={N}; repeat count={REPEAT_MIN}; flip reversals={REV_THRESHOLD}; invalid share p99={INVALID_SHARE_THRESHOLD:.6f}; cp threshold={CP_THRESHOLD:.4f}')
print(f'plots={len(plot_manifest)}; union={len(union)}; random_other={len(random10)}; smooth recovery events={len(pumps)}')
print('Hathras detector row:')
print(detector[detector.Station.str.contains('Hathras',case=False)][['Station','flipflop_invalid_share_pre_observed','flipflop_invalid_share_all_observed','exceeds_pre_whole_station_invalid_share','strict_classification']].to_string(index=False))
print('Dera 27.5:',dera_audit.to_string(index=False))
print('gap summary:',gap_summary)
