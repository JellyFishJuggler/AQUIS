from __future__ import annotations
import argparse, importlib, json, os, hashlib, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DIAG=ROOT/'outputs'/'diagnostics_phase3a'
DATA=ROOT/'data'
OUT=ROOT/'data_v2'
PRE_END=pd.Timestamp('2025-10-01')
VAL_END=pd.Timestamp('2026-01-01')
SENTINELS={99.,-99.,999.,-999.,9999.,-9999.}
MUST_FIX=['Primary School, Kalan Bar Devran','Primary School/ Composite, Mehmoodpur']
CONTROLS=['Mathura Chaumuha Navagaon-11','Block Office, Kurara','Hamirpur Sarila Purani','Primary School, Evani','Primary School, Garher']
RESIDUAL='Hamirpur Kurara Jalokhar'
REVIEW_EARLY_NAMES={'fugana 1','bhadarwara p 1','ranipur 2','mur wa 1','murwa 1'}

# M2 repeated-exact path: an isolated excursion must actually depart from the
# surrounding level. Without this guard, the "exact immediate return" test
# (v[i-1] == v[i+1]) is trivially true at every interior point of a constant
# run, so flat plateaus are masked as spikes. That is a detector logic error,
# not a physical M2 excursion. Kept configurable to reproduce the old gate.
REPEATED_EXACT_REQUIRES_DEPARTURE=True


def intervals(mask):
    ids=np.flatnonzero(mask)
    if not len(ids): return []
    cuts=np.flatnonzero(np.diff(ids)>1)+1
    return [(int(chunk[0]),int(chunk[-1])) for chunk in np.split(ids,cuts)]

def robust_pre(x,t):
    z=x[(t<PRE_END)&np.isfinite(x)&(~np.isin(x,list(SENTINELS))) ]
    if len(z)<20: return np.nan,np.nan
    med=float(np.median(z)) if len(z) else np.nan
    scale=float(1.4826*np.median(np.abs(z-med))) if len(z) else np.nan
    return med,max(scale,.03) if np.isfinite(scale) else np.nan

def continued_prefix(v):
    n=len(v); finite=np.isfinite(v)
    q=np.flatnonzero(finite)
    if not len(q): return np.zeros(n,bool),None
    start=int(q[0]); first=float(v[start]); is_sentinel=first in SENTINELS
    # Initiate only on a sentinel lasting >=4 slots, or any leading exact-value run >=28 slots.
    runend=start+1
    while runend<n and finite[runend] and v[runend]==first: runend+=1
    init=(is_sentinel and runend-start>=4) or (runend-start>=28)
    if not init: return np.zeros(n,bool),None
    # Follow through the second plateau and beyond until two adjacent 28-slot windows
    # each contain >=8 distinct, nonsentinel readings within a broad 30m range.
    for k in range(max(start,runend-27),max(start,n-55)):
        good=True
        for a in (k,k+28):
            w=v[a:a+28]
            ww=w[np.isfinite(w)&(~np.isin(w,list(SENTINELS)))]
            if len(ww)<20 or len(np.unique(ww))<8 or (np.quantile(ww,.95)-np.quantile(ww,.05))>10:
                good=False;break
        if good:
            m=np.zeros(n,bool); m[start:k]=True
            return m,{'start':start,'end':k,'init_value':first}
    # No varying in-range continuation found: mask initialization through series end.
    m=np.zeros(n,bool);m[start:]=True
    return m,{'start':start,'end':n,'init_value':first}

def detect_breaks(v,t,m1,m2):
    # Persistent regime transition: 7-day before/after medians differ by >=15m,
    # and the new median remains within 5m after another 30 days.
    s=pd.Series(v.copy()); s[m1|m2]=np.nan
    med=s.rolling(28,min_periods=14).median().to_numpy()
    breaks=[]; n=len(v)
    # A real break must begin with an observed >=15 m transition within 24 h.
    # Prefilter raw step candidates so we do not perform rolling-window work
    # at every row of the 3M-row source table.
    dt=np.diff(t)
    rawstep=np.diff(v)
    candidates=np.flatnonzero((np.abs(rawstep)>=15)&(dt<=np.timedelta64(24,'h'))&(dt>np.timedelta64(0,'h')))+1
    for i in candidates:
        if i<28 or i>=n-148: continue
        # Use window medians to avoid a single-point jump triggering a break.
        left=s.iloc[max(0,i-28):i].dropna().to_numpy(); right=s.iloc[i:min(n,i+28)].dropna().to_numpy()
        held=s.iloc[i:min(n,i+148)].dropna().to_numpy()
        if len(left)<14 or len(right)<14 or len(held)<100: continue
        a=float(np.median(left));b=float(np.median(right)); h=float(np.median(held))
        if abs(b-a)>=15 and abs(h-b)<=5:
            # reject if same baseline returns within the next 30d
            if len(held)>=100 and np.mean(np.abs(held-b)<=5)>=.8:
                breaks.append((i,a,b,b-a))
    # collapse repeated detections around the same transition
    result=[]
    for br in breaks:
        if not result or br[0]-result[-1][0]>28: result.append(br)
    return result

def detect_station(g, repeated_requires_departure=REPEATED_EXACT_REQUIRES_DEPARTURE,
                    repeated_require_observed=True):
    v=g.gwl.to_numpy(float);t=g.time.to_numpy(dtype='datetime64[ns]');n=len(g)
    obs=(g.gwl_obs.to_numpy(float)>0.5) if 'gwl_obs' in g.columns else np.ones(n,bool)
    ti=pd.DatetimeIndex(t)
    m1,info=continued_prefix(v)
    m2=np.zeros(n,bool)
    # Pre-cutoff robust envelope excludes M1 and long exact runs (>=28 slots).
    med,scale=robust_pre(v,ti)
    # Mask reversible short excursions (>30m local departure) of 1..12 slots.
    x=v.copy();x[m1]=np.nan
    ser=pd.Series(x,index=ti)
    left=ser.rolling('7D',closed='left',min_periods=8).median().to_numpy()
    right=ser.iloc[::-1].rolling('7D',closed='left',min_periods=8).median().iloc[::-1].to_numpy()
    far=np.isfinite(v)&np.isfinite(left)&np.isfinite(right)&(np.abs(v-left)>30)&(np.abs(v-right)>30)&(np.abs(left-right)<=5)&(~m1)
    # candidate contiguous event rows with plausible max length and exact return around the run
    for a,b in intervals(far):
        if b-a+1<=12:
            m2[a:b+1]=True
    # Repeated exact extreme: fitted on pre-cutoff data only, after M1 + long-run exclusion.
    base=(ti<PRE_END)&np.isfinite(v)&(~m1)
    pre_n=int(np.searchsorted(t,PRE_END.to_datetime64(),side='left'))
    for a,b in intervals(np.isfinite(v[:pre_n])):
        if b-a+1>=28: base[a:b+1]=False
    medp,scalep=robust_pre(v[base],ti[base]) if base.sum() else (med,scale)
    lo,hi=medp-8*scalep,medp+8*scalep
    prevals=v[base]; vals,cnts=np.unique(prevals,return_counts=True) if len(prevals) else (np.array([]),np.array([]))
    rep=set(float(z) for z,c in zip(vals,cnts) if c>=3 and (z<lo or z>hi or z in SENTINELS))
    # Isolated exact return at six-hour spacing, repeated extreme, physical departure >30m.
    # The value must differ from its neighbours: inside a flat run v[i-1]==v[i+1]
    # does not describe an excursion at all (see REPEATED_EXACT_REQUIRES_DEPARTURE).
    # All three slots must be actual readings: a carried-forward slot is not a
    # new observation and must not create or confirm a repeated extreme.
    for i in range(1,n-1):
        if v[i] in rep and (not repeated_requires_departure or v[i]!=v[i-1]) \
           and (not repeated_require_observed or (obs[i] and obs[i-1] and obs[i+1])) \
           and np.isfinite(v[i-1]) and np.isfinite(v[i+1]) and v[i-1]==v[i+1] and t[i]-t[i-1]==np.timedelta64(6,'h') and t[i+1]-t[i]==np.timedelta64(6,'h'):
            m2[i]=True
    # Reversible blocks caught by physical cap must have baseline reversion checked; rolling medians provide that check.
    # Remove M1/M2 rows before detecting the persistent breaks; values are never edited for M3.
    breaks=detect_breaks(v,ti,m1,m2)
    # A persistent early/late level transition is recorded for review. Without
    # surveyed well elevation / logger datum metadata, absolute physical
    # implausibility cannot be established from sign alone, so it is not M1-masked.
    early_suspect=np.zeros(n,bool)
    finite=np.flatnonzero(np.isfinite(v))
    if len(finite):
        first=int(finite[0])
        for bi,before,after,delta in breaks:
            if bi-first>=120 and abs(delta)>=15:
                early_suspect[first:bi]=True
    # Flag-only diagnostics, on the original series. Stuck: exact run >=28 or 7d near-flat.
    stuck=np.zeros(n,bool)
    for a,b in intervals(np.isfinite(v)):
        if b-a+1>=28 and np.all(v[a:b+1]==v[a]): stuck[a:b+1]=True
    s=pd.Series(v,index=ti)
    rmax=s.rolling('7D',min_periods=20).max();rmin=s.rolling('7D',min_periods=20).min();rsd=s.rolling('7D',min_periods=20).std()
    stuck |= ((rmax-rmin)<=.06).fillna(False).to_numpy() & (rsd<=.02).fillna(False).to_numpy()
    # Flip-flop flag: qualifying >=0.5m steps that reverse sign in adjacent pairs, 30d rolling count/share.
    obs=np.isfinite(v);oi=np.flatnonzero(obs);d=np.diff(v[oi]);dt=np.diff(t[oi]);valid=(dt<=np.timedelta64(48,'h'))&(dt>np.timedelta64(0,'h'))
    large=valid&(np.abs(d)>=.5); rev=np.zeros(len(large),bool)
    for k in range(1,len(large)): rev[k]=large[k] and large[k-1] and np.sign(d[k])!=np.sign(d[k-1])
    flip=np.zeros(n,bool)
    for k in np.flatnonzero(rev):
        tm=t[oi[k+1]]; l=np.searchsorted(t[oi],tm-np.timedelta64(30,'D')); rr=np.searchsorted(t[oi],tm,'right')
        dd=d[max(0,l):min(len(d),rr-1)]; ll=valid[max(0,l):min(len(valid),rr-1)]&(np.abs(dd)>=.5)
        share=float(np.sum(rev[max(0,l):min(len(rev),rr-1)])/max(1,ll.sum()))
        if rev[max(0,l):min(len(rev),rr-1)].sum()>=6 and share>=.6:
            flip[oi[max(0,l):min(len(oi),rr)]]=True
    # Extreme bursts are flagged, never masked: >=3 pre-cutoff robust outliers in rolling 30d.
    extreme=np.zeros(n,bool)
    out=np.isfinite(v)&((v<lo)|(v>hi))&(~m1)&(~m2)
    oi2=np.flatnonzero(out)
    for j in range(len(oi2)):
        loidx=np.searchsorted(t[oi2],t[oi2[j]]-np.timedelta64(30,'D'))
        if j-loidx+1>=3: extreme[oi2[loidx:j+1]]=True
    return {'m1':m1,'m2':m2,'breaks':breaks,'stuck':stuck,'flip':flip,'extreme':extreme,'early_suspect':early_suspect,
            'm1_info':info,'robust_median_pre':medp,'robust_scale_pre':scalep,'repeat_values':sorted(rep)}

def write_mask_tables(df,m1,m2):
    mask_counts=[]
    for rule,mask in [('M1',m1),('M2',m2)]:
        tmp=df.loc[mask,['Station','time']].copy();tmp['rule']=rule
        if len(tmp):mask_counts.append(tmp)
    allm=pd.concat(mask_counts,ignore_index=True) if mask_counts else pd.DataFrame(columns=['Station','time','rule'])
    allm.groupby(['Station','rule']).size().rename('rows_masked').reset_index().to_csv(DIAG/'masked_rows_by_station.csv',index=False)
    if len(allm): allm.assign(month=allm.time.dt.to_period('M').astype(str)).groupby(['rule','month']).size().rename('rows_masked').reset_index().to_csv(DIAG/'masked_rows_by_month.csv',index=False)
    else: pd.DataFrame(columns=['rule','month','rows_masked']).to_csv(DIAG/'masked_rows_by_month.csv',index=False)
    return allm


def detector_metadata():
    return {
        'version':'phase3a-2026-10-26',
        'm1':{'leading_run_slots':28,'sentinel_run_slots':4,
              'continuation':'two adjacent 28-slot windows: >=20 non-sentinel, >=8 distinct, p95-p05 <= 10 m'},
        'm2_physical':{'max_slots':12,'local_departure_m':30,'surrounding_window':'7D',
                       'median_agreement_m':5},
        'm2_repeated_exact':{'pre_cutoff':'2025-10-01',
                             'min_pre_cutoff_occurrences':3,
                             'robust_envelope_mad':8,
                             'requires_exact_return':True,
                             'requires_departure_from_neighbours':True,
                             'requires_observed_slots':True,
                             'rationale':'A forward-filled carry is not a new reading, and v[i-1]==v[i+1] is trivially true inside a constant run. The old rule masked 22 carried plateau slots at Block Office, Kurara.'},
        'm3':'persistent level-transition candidate; flag only, no value edit',
        'flag_only':['stuck','flipflop','extreme_burst'],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--apply-cleaning',action='store_true')
    ap.add_argument('--gate-only',action='store_true')
    args=ap.parse_args()
    if not args.apply_cleaning and not args.gate_only:
        raise SystemExit('Cleaning is disabled by default; pass --apply-cleaning to build, or --gate-only for the gate.')
    if os.system('git branch --show-current')<0: pass
    src=DATA/'aligned'/'table_6h.parquet';df=pd.read_parquet(src);df['time']=pd.to_datetime(df.time)
    df=df.sort_values(['Station','time']).reset_index(drop=True)
    det={}; m1=np.zeros(len(df),bool);m2=m1.copy();stuck=m1.copy();flip=m1.copy();extreme=m1.copy();early_suspect=m1.copy(); station=[];breakrows=[]
    for st,idx in df.groupby('Station',sort=False).groups.items():
        ix=np.asarray(list(idx),dtype=int); g=df.loc[ix]
        z=detect_station(g);det[st]=z;m1[ix]=z['m1'];m2[ix]=z['m2'];stuck[ix]=z['stuck'];flip[ix]=z['flip'];extreme[ix]=z['extreme'];early_suspect[ix]=z['early_suspect']
        norm=' '.join(''.join(c.lower() if c.isalnum() else ' ' for c in str(st)).split())
        if norm in REVIEW_EARLY_NAMES:
            cutoff=int(np.searchsorted(g.time.to_numpy(dtype='datetime64[ns]'),PRE_END.to_datetime64(),side='left'))
            z['early_suspect'][:cutoff]=True;early_suspect[ix]=z['early_suspect']
        station.append({'Station':st,'m1_rows':int(z['m1'].sum()),'m2_rows':int(z['m2'].sum()),'stuck_rows':int(z['stuck'].sum()),'flip_rows':int(z['flip'].sum()),'extreme_burst_rows':int(z['extreme'].sum()),'early_era_unverified_rows':int(z['early_suspect'].sum()),'m1_prefix_slots':int(z['m1'].sum()),'break_count':len(z['breaks']),'m1_start':z['m1_info']['start'] if z['m1_info'] else None,'m1_end':z['m1_info']['end'] if z['m1_info'] else None,'pre_median':z['robust_median_pre'],'pre_scale':z['robust_scale_pre'],'repeat_values':json.dumps(z['repeat_values'])})
        for i,a,b,dlt in z['breaks']: breakrows.append({'Station':st,'break_time':pd.Timestamp(g.time.iloc[i]),'before_median':a,'after_median':b,'delta':dlt})
    # Acceptance gate before any data_v2 write.
    stats=pd.DataFrame(station).set_index('Station')
    actual=set(df.Station.astype(str))
    def match(s):
        if s in actual:return s
        norm=lambda q:' '.join(''.join(c.lower() if c.isalnum() else ' ' for c in q).split())
        cand=[q for q in actual if norm(q)==norm(s)]
        return cand[0] if cand else None
    gate=[]
    for s in MUST_FIX:
        q=match(s); n=int(stats.loc[q,'m2_rows']) if q else 0
        gate.append({'requested':s,'matched':q,'m2_mask_rows':n,'status':'PASS' if q and n>0 else 'FAIL'})
    for s in CONTROLS:
        q=match(s); n=int(stats.loc[q,'m1_rows']+stats.loc[q,'m2_rows']) if q else -1
        gate.append({'requested':s,'matched':q,'m1_m2_mask_rows':n,'status':'PASS' if q and n==0 else 'FAIL'})
    gate_df=pd.DataFrame(gate)
    gate_df.to_csv(DIAG/'named_gate.csv',index=False)
    brdf=pd.DataFrame(breakrows)
    write_mask_tables(df,m1,m2)
    stats.reset_index().to_csv(DIAG/'station_detector_summary.csv',index=False)
    brdf.to_csv(DIAG/'m3_breaks.csv',index=False)
    gate_pass=bool((gate_df.status=='PASS').all())
    gate_summary={'gate_pass':gate_pass,'named_gate':gate_df.to_dict(orient='records'),
                  'source_rows':len(df),'source_stations':int(df.Station.nunique()),
                  'masked_M1':int(m1.sum()),'masked_M2':int(m2.sum()),
                  'detector':detector_metadata(),'checked_at':str(pd.Timestamp.now())}
    (DIAG/'gate_summary.json').write_text(json.dumps(gate_summary,indent=2,default=str))
    if not gate_pass:
        raise SystemExit('NAMED GATE FAILED. No data_v2 writes performed; inspect outputs/diagnostics_phase3a/named_gate.csv')
    if args.gate_only:
        print(gate_df.to_string(index=False))
        print('\nGate PASS. No data_v2 writes performed (--gate-only).')
        return
    # Persist all remaining outputs only after gate passes.
    OUT.mkdir(parents=True,exist_ok=True)
    build_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    started=str(pd.Timestamp.now())
    status={'build_id':build_id,'status':'in_progress','build_complete':False,
            'gate_pass':True,'started_at':started,
            'source_path':str(src.relative_to(ROOT)),'source_rows':len(df),
            'source_stations':int(df.Station.nunique()),
            'detector':detector_metadata(),
            'note':'BUILD_INCOMPLETE_DO_NOT_USE until the purge check and visual review complete.'}
    (OUT/'BUILD_STATUS.json').write_text(json.dumps(status,indent=2,default=str))
    (OUT/'BUILD_INCOMPLETE_DO_NOT_USE.txt').write_text(
        'PHASE 3a BUILD IN PROGRESS / NOT YET REVIEWED — DO NOT USE.\n'
        f'build_id={build_id}\nGate passed. Build, purge check, and visual review must all finish.\n'
        'See BUILD_STATUS.json.\n')
    (OUT/'cleaning_config.json').write_text(json.dumps({'default_enabled':False,'applied_for_this_build':True,
        'build_complete':False,'status':'in_progress','build_id':build_id,
        'masks':['M1_placeholder_or_invalid_early_era','M2_short_reversible_spike'],
        'breaks':'M3_persistent_level_shift_no_value_edit','flag_only':['stuck','flipflop','extreme_burst'],
        'pre_cutoff':'time < 2025-10-01','physical_cap_m':30,'m1_run_slots':28,
        'm2_repeated_exact':detector_metadata()['m2_repeated_exact']},indent=2))
    raw=df.gwl.copy();clean=df.copy();clean['gwl_raw']=raw;clean['m1_mask']=m1;clean['m2_mask']=m2;clean['m3_break_here']=False
    clean['flag_stuck']=stuck;clean['flag_flipflop']=flip;clean['flag_extreme_burst']=extreme;clean['flag_early_era_unverified']=early_suspect
    stf=pd.DataFrame(station).set_index('Station')
    for col in ['stuck_rows','flip_rows','extreme_burst_rows','early_era_unverified_rows']:
        clean['station_'+col]=clean.Station.map(stf[col]).astype('int32')
    clean['station_flag_stuck']=clean.station_stuck_rows.gt(0)
    clean['station_flag_flipflop']=clean.station_flip_rows.gt(0)
    clean['station_flag_extreme_burst']=clean.station_extreme_burst_rows.gt(0)
    clean['station_flag_early_era_unverified']=clean.station_early_era_unverified_rows.gt(0)
    clean.loc[m1|m2,'gwl']=np.nan
    breakmap={}
    for _,r in brdf.iterrows():breakmap.setdefault(r.Station,[]).append(pd.Timestamp(r.break_time))
    for st,ts in breakmap.items():
        ii=df.index[(df.Station==st)&df.time.isin(ts)];clean.loc[ii,'m3_break_here']=True
    (OUT/'aligned').mkdir(parents=True,exist_ok=True); clean.to_parquet(OUT/'aligned'/'table_6h.parquet',index=False)
    # Build features using the existing causal engineering function only; avoid build_full (which drops regimes/reads static sources).
    sys.path.insert(0,str(ROOT));F=importlib.import_module('06_features')
    feats=F.build(clean.drop(columns=['gwl_raw']),keep_na=False)
    # Add row/station detector flags to feature rows, then purge lag/target pairs straddling breaks.
    extra=['m1_mask','m2_mask','m3_break_here','flag_stuck','flag_flipflop','flag_extreme_burst','flag_early_era_unverified','station_stuck_rows','station_flip_rows','station_extreme_burst_rows','station_early_era_unverified_rows','station_flag_stuck','station_flag_flipflop','station_flag_extreme_burst','station_flag_early_era_unverified']
    feats=feats.merge(clean[['Station','time']+extra],on=['Station','time'],how='left',suffixes=('','_source'))
    feats['date']=pd.to_datetime(feats.time).dt.normalize()
    feats['target_time']=pd.to_datetime(feats.target_time)
    # Remove 30-day lag warm-up/straddle after every break and any anchor-target interval straddling a break.
    break_lists={s:np.array([pd.Timestamp(x).to_datetime64() for x in ls],dtype='datetime64[ns]') for s,ls in breakmap.items()}
    lag_bad=np.zeros(len(feats),bool);pair_bad=np.zeros(len(feats),bool)
    for st,bt in break_lists.items():
        q=np.flatnonzero(feats.Station.to_numpy()==st)
        at=feats.time.to_numpy(dtype='datetime64[ns]')[q];tt=feats.target_time.to_numpy(dtype='datetime64[ns]')[q]
        for b in bt:
            lag_bad[q]|=(at>=b)&(at<b+np.timedelta64(30,'D'))
            pair_bad[q]|=(at<b)&(tt>=b)
    n_features_before_m3=len(feats)
    features=feats.loc[~lag_bad&~pair_bad].copy()
    parts={
      'train':features[(features.date<PRE_END)&(features.target_time<PRE_END)],
      'val':features[(features.date>=PRE_END)&(features.date<VAL_END)&(features.target_time<VAL_END)],
      'test':features[features.date>=VAL_END],
    }
    fdir=OUT/'features_6h';fdir.mkdir(parents=True,exist_ok=True)
    for name,part in parts.items():part.to_parquet(fdir/f'{name}.parquet',index=False)
    del feats, features
    # Direct trajectory expansion. Preserve existing train/validation schedules;
    # test uses the established sparse training horizon schedule to bound output size.
    T=importlib.import_module('30_traj_datasets');tdir=OUT/'features_traj';tdir.mkdir(parents=True,exist_ok=True)
    parts_rows=[];purge_summary={}
    clean_index={str(s):np.asarray(ix,dtype=int) for s,ix in clean.groupby('Station',sort=False).indices.items()}
    import pyarrow as pa, pyarrow.parquet as pq
    for split,part in parts.items():
        writer=None;outrows=0;before=0
        stations=part.Station.astype(str).unique()
        part_index={str(s):np.asarray(ix,dtype=int) for s,ix in part.groupby('Station',sort=False).indices.items()}
        for sn,st in enumerate(stations,1):
            anchors=part.iloc[part_index[st]]
            if split=='train':anchors=anchors.iloc[::T.TRAIN_STRIDE];hs=T.HORIZONS_TRAIN
            elif split=='val':hs=T.HORIZONS_VAL
            else:hs=T.HORIZONS_TRAIN
            anchorcols=[c for c in T.META+T.NUM_COLS+extra if c in anchors.columns]
            anchors=anchors[anchorcols]
            # targets must be observed in cleaned aligned series, exact cadence
            allst=clean.iloc[clean_index[st]][['time','gwl']].dropna(subset=['gwl'])
            times=allst.time.to_numpy(dtype='datetime64[ns]'); vals=allst.gwl.to_numpy(float)
            traj=T._expand(anchors,times,vals,hs)
            before+=len(traj)
            if split=='train':traj=traj[pd.to_datetime(traj.target_time)<PRE_END]
            if split=='val':traj=traj[pd.to_datetime(traj.target_time)<VAL_END]
            # Per horizon pair purge and lag-window exclusion at anchor level.
            if len(traj):
                good=np.ones(len(traj),bool); at=traj.time.to_numpy(dtype='datetime64[ns]');tt=traj.target_time.to_numpy(dtype='datetime64[ns]')
            else: good=np.zeros(0,bool)
            if len(traj):
                for b in break_lists.get(st,[]):
                    good&=~(((at>=b)&(at<b+np.timedelta64(30,'D')))|((at<b)&(tt>=b)))
                traj=traj.loc[good]
                if len(traj):
                    table=pa.Table.from_pandas(traj,preserve_index=False)
                    if writer is None: writer=pq.ParquetWriter(tdir/f'{split}.parquet',table.schema,compression='zstd')
                    writer.write_table(table);outrows+=len(traj)
        if writer is not None: writer.close()
        purge_summary[split]={'trajectory_before_purge':int(before),'after':int(outrows),'removed':int(before-outrows),'horizons':T.HORIZONS_TRAIN if split in ('train','test') else T.HORIZONS_VAL}
    for split in ['train','val','test']:
        if not (tdir/f'{split}.parquet').exists():
            (OUT/'BUILD_STATUS.json').write_text(json.dumps({**status,'status':'failed','build_complete':False,
                'failure':f'missing trajectory split {split}'},indent=2,default=str))
            raise SystemExit(f'BUILD FAILED: missing trajectory split {split}')
    # Automated purge/leakage tests for all main + trajectory splits.
    # Established protocol: partition by anchor date and purge by observed target
    # timestamp, so no label reaches a later split. Also verify positive horizon
    # and, for trajectories, the exact h*6h target offset.
    gate_rows=[]
    traj_frames={n:pd.read_parquet(tdir/f'{n}.parquet',columns=['time','target_time','h']) for n in ['train','val','test']}
    for typ,dct in [('features_6h',parts),('features_traj',traj_frames)]:
        for split,part in dct.items():
            bound=PRE_END if split=='train' else VAL_END if split=='val' else None
            if len(part):
                at=pd.to_datetime(part['time']);tt=pd.to_datetime(part['target_time'])
                max_t=tt.max();min_t=tt.min();max_a=at.max();min_a=at.min()
                positive=bool((tt>at).all())
            else:
                max_t=min_t=max_a=min_a=pd.NaT;positive=False
            target_ok=True if bound is None else bool(pd.notna(max_t) and max_t<bound)
            if split=='train':anchor_ok=bool(pd.notna(max_a) and max_a<PRE_END)
            elif split=='val':anchor_ok=bool(pd.notna(min_a) and min_a>=PRE_END and max_a<VAL_END)
            else:anchor_ok=bool(pd.notna(min_a) and min_a>=VAL_END)
            horizon_ok=True
            if typ=='features_traj' and len(part):
                hh=part['h'].to_numpy(dtype='int64')
                expected=pd.to_datetime(part['time'])+pd.to_timedelta(hh*6,unit='h')
                horizon_ok=bool((expected.to_numpy()==pd.to_datetime(part['target_time']).to_numpy()).all())
            passed=bool(target_ok and anchor_ok and positive and horizon_ok)
            gate_rows.append({'frame':typ,'split':split,'rows':len(part),
                'min_anchor_time':str(min_a),'max_anchor_time':str(max_a),
                'min_target_time':str(min_t),'max_target_time':str(max_t),
                'required_target_less_than':str(bound) if bound is not None else 'unbounded',
                'target_after_anchor':positive,'anchor_window_ok':anchor_ok,
                'horizon_offset_ok':horizon_ok,'pass':passed})
    purge=pd.DataFrame(gate_rows);purge.to_csv(DIAG/'purge_test.csv',index=False)
    if not purge['pass'].all():
        (OUT/'BUILD_STATUS.json').write_text(json.dumps({**status,'status':'failed','build_complete':False,
            'purge_pass':False,'failure':'purge/leakage gate failed; inspect purge_test.csv'},indent=2,default=str))
        raise SystemExit('PURGE GATE FAILED after output; inspect purge_test.csv')
    # Selection: 29 Phase2d names + 12 new random names not previously used (seed 202603).
    prev=pd.read_csv(ROOT/'outputs/diagnostics_phase2d/visual_sample_stations.csv');old=set(prev.Station.astype(str))
    p2c=pd.read_csv(ROOT/'outputs/diagnostics_phase2c/acceptance_per_station.csv');old|=set(p2c.matched_station.dropna().astype(str))
    names=sorted(df.Station.astype(str).unique());rng=np.random.default_rng(202603);eligible=[s for s in names if s not in old]
    new=sorted(rng.choice(eligible,size=min(12,len(eligible)),replace=False).tolist())
    sample=prev[['Station','primary_classification','selection']].copy();sample['sample_group']='phase2d_existing'
    newdf=pd.DataFrame({'Station':new,'primary_classification':[stats.loc[s,'m1_rows'] and 'placeholder-start' or ('stuck' if stats.loc[s,'stuck_rows'] else 'flip-flop' if stats.loc[s,'flip_rows'] else 'clean') for s in new],'selection':'new random; seed=202603','sample_group':'new_random'})
    sample=pd.concat([sample,newdf],ignore_index=True);sample.to_csv(DIAG/'visual_sample_stations.csv',index=False)
    if len(prev)!=29 or len(sample)!=41 or sample.Station.nunique()!=41:
        (OUT/'BUILD_STATUS.json').write_text(json.dumps({**status,'status':'failed','build_complete':False,
            'failure':f'visual sample is {len(prev)}+{len(new)}={len(sample)}, expected 29+12=41 unique'},indent=2,default=str))
        raise SystemExit('VISUAL SAMPLE FAILED: expected 29+12 unique stations')
    # Export plotting source; all timestamps for 41 stations (before/after + overlays).
    m3mask=np.zeros(len(df),bool)
    for st,ts in breakmap.items():
        m3mask|=((df.Station.to_numpy()==st)&df.time.isin(ts).to_numpy())
    vdir=DIAG/'visual_data';vdir.mkdir(exist_ok=True);sub=df.Station.isin(sample.Station)
    vis=df.loc[sub,['Station','time','gwl']].copy();vis['gwl_after']=vis.gwl.copy()
    vis.loc[(m1|m2)[sub.to_numpy()],'gwl_after']=np.nan
    for c,ma in [('m1',m1),('m2',m2),('m3',m3mask),('stuck',stuck),('flipflop',flip),('extreme_burst',extreme)]:
        vis[c]=ma[sub.to_numpy()]
    vis.to_csv(vdir/'series.csv',index=False)
    # Final manifest and summary. The build remains explicitly incomplete until
    # the rendered contact sheets are inspected and recorded by finalize_phase3a.py.
    manifest={'build_id':build_id,'build_complete':False,'status':'awaiting_visual_review',
        'gate_pass':True,'gate_file':'outputs/diagnostics_phase3a/named_gate.csv',
        'gate_summary_file':'outputs/diagnostics_phase3a/gate_summary.json',
        'source_path':'data/aligned/table_6h.parquet',
        'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'source_rows':len(df),'source_stations':int(df.Station.nunique()),
        'started_at':started,'built_at':str(pd.Timestamp.now()),
        'detector':detector_metadata(),
        'aligned_rows_after':len(clean),'usable_gwl_before':int(raw.notna().sum()),
        'usable_gwl_after':int(clean.gwl.notna().sum()),'masked_M1':int(m1.sum()),
        'masked_M2':int(m2.sum()),'breaks_M3':len(brdf),
        'feature_rows_before_M3_exclusion':n_features_before_m3,
        'feature_rows_after_M3_exclusion':len(features),
        'removed_lag_or_pair_rows_6h':int((lag_bad|pair_bad).sum()),
        'lag_window_rows':int(lag_bad.sum()),'anchor_target_straddle_rows':int(pair_bad.sum()),
        'split_rows':{k:len(v) for k,v in parts.items()},
        'trajectory_rows':purge_summary,'purge_pass':bool(purge['pass'].all()),
        'purge_test_file':'outputs/diagnostics_phase3a/purge_test.csv',
        'random_seed':202603,'new_random_stations':new,
        'visual_sample_rows':len(sample),'visual_sample_groups':sample.sample_group.value_counts().to_dict(),
        'visual_data_file':'outputs/diagnostics_phase3a/visual_data/series.csv',
        'visual_review':'pending'}
    (DIAG/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    final_status={**status,'status':'awaiting_visual_review','build_complete':False,
        'gate_pass':True,'purge_pass':bool(purge['pass'].all()),
        'visual_review':'pending','finished_at':str(pd.Timestamp.now()),
        'counts':{'source_rows':len(df),'masked_M1':int(m1.sum()),'masked_M2':int(m2.sum()),
                  'features_6h':{k:len(v) for k,v in parts.items()},
                  'features_traj':purge_summary}}
    (OUT/'BUILD_STATUS.json').write_text(json.dumps(final_status,indent=2,default=str))
    (OUT/'BUILD_INCOMPLETE_DO_NOT_USE.txt').write_text(
        'PHASE 3a BUILD AWAITING VISUAL REVIEW — DO NOT USE YET.\n'
        f'build_id={build_id}\nGate passed; build and purge check passed.\n'
        'Contact sheets have NOT yet been inspected. See BUILD_STATUS.json.\n')
    cfg=json.loads((OUT/'cleaning_config.json').read_text())
    cfg.update({'status':'awaiting_visual_review','build_complete':False,'purge_pass':True,
                'gate_pass':True,'finished_at':str(pd.Timestamp.now())})
    (OUT/'cleaning_config.json').write_text(json.dumps(cfg,indent=2))
    print(json.dumps(manifest,indent=2,default=str));print('\nPurge tests:\n',purge.to_string(index=False));print('\nNamed gate:\n',gate_df.to_string(index=False))
if __name__=='__main__':main()
