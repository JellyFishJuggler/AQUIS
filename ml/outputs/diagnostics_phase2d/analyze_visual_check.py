from pathlib import Path
import sys, os, re, hashlib
ROOT=Path('/mnt/c/Users/Srijan Gupta/Downloads/Developement/AQUIS/ml')
OUT=ROOT/'outputs/diagnostics_phase2d'; OUT.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True; sys.path.insert(0,str(ROOT/'venv312/lib/python3.12/site-packages')); os.chdir(ROOT)
import numpy as np, pandas as pd
START=pd.Timestamp('2025-10-01'); END=pd.Timestamp('2026-10-01'); SIXH=pd.Timedelta(hours=6); SENT={99.,-99.,999.,-999.,9999.,-9999.}
MIN_RUN=28; REPEAT=3; MAD_MULT=8.; FLIP_STEP=.5; FLIP_N=6; FLIP_SHARE=.60; FLIP_STATION_SHARE=.20
NAMED=['Hamirpur Kurara Jalokhar','Hamirpur Sarila Purani','Mathura Chaumuha Navagaon-11','Primary School Kalan Bar Devran','Primary School/ Composite, Mehmoodpur']
def norm(s): return re.sub(r'[^a-z0-9]+','',str(s).casefold())
def slug(s): return re.sub(r'[^A-Za-z0-9]+','_',str(s)).strip('_')[:60]+'_'+hashlib.md5(str(s).encode()).hexdigest()[:8]
def md(df):
 def c(x):
  if pd.isna(x): return ''
  return str(x).replace('|','\\|').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(c(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])
def runs(g):
 v=g.gwl.to_numpy(float); tn=g.time.astype('int64').to_numpy(); same=(np.diff(tn)==SIXH.value)&np.isfinite(v[1:])&np.isfinite(v[:-1])&(v[1:]==v[:-1]); s=np.r_[0,np.flatnonzero(~same)+1]; e=np.r_[s[1:]-1,len(g)-1]
 return pd.DataFrame({'start':g.time.to_numpy()[s],'end':g.time.to_numpy()[e],'value':v[s],'slots':e-s+1})
def masks(g, row):
 n=len(g); v=g.gwl.to_numpy(float); finite=np.isfinite(v); obs=(g.gwl_obs.to_numpy()==1)&finite; t=g.time; tt=t.to_numpy(); first_t=t.iloc[np.flatnonzero(finite)[0]] if finite.any() else t.min()
 lead=np.zeros(n,bool); early=(t>=first_t)&(t<=first_t+pd.Timedelta(days=60))&finite; vals=pd.Series(v[early]).value_counts(); sent=vals[vals.index.isin(SENT)]
 if len(sent):
  fv=float(sent.index[0]); lead|=early&(v==fv); sec=vals.drop(labels=[fv],errors='ignore'); sec=sec[sec>=4]
  if len(sec): lead|=early&(v==float(sec.index[0]))
 elif finite.any():
  fv=v[np.flatnonzero(finite)[0]]
  if int(vals.get(fv,0))>=MIN_RUN: lead|=early&(v==fv)
 rr=runs(g); long=np.zeros(n,bool)
 for r in rr[rr.slots>=MIN_RUN].itertuples(index=False): long|=((t>=r.start)&(t<=r.end)).to_numpy()
 fit=obs&(~long)&(~lead)&(~np.isin(v,list(SENT)))&(tt<START.to_datetime64()); x=v[fit]
 if len(x): med=float(np.median(x)); mad=float(np.median(np.abs(x-med))); sc=max(1.4826*mad,.03); lo=med-MAD_MULT*sc; hi=med+MAD_MULT*sc
 else: med=mad=lo=hi=np.nan
 vc=pd.Series(v[finite]).value_counts() if finite.any() else pd.Series(dtype=int)
 repeated={float(k):int(c) for k,c in vc.items() if c>=REPEAT and (k in SENT or (np.isfinite(lo) and (k<lo or k>hi)))}
 spike=np.zeros(n,bool)
 if n>2 and repeated:
  tn=t.astype('int64').to_numpy(); spike[1:-1]=finite[1:-1]&np.isin(v[1:-1],list(repeated))&finite[:-2]&finite[2:]&(v[:-2]==v[2:])&(v[1:-1]!=v[:-2])&((tn[1:-1]-tn[:-2])==SIXH.value)&((tn[2:]-tn[1:-1])==SIXH.value)
 outlier=finite&np.isfinite(lo)&((v<lo)|(v>hi))&(~lead); outix=np.flatnonzero(outlier); tv=t.astype('int64').to_numpy(); ol=np.searchsorted(tv[outix],tv[outix]-pd.Timedelta(days=30).value,side='left') if len(outix) else np.array([],int); burst=bool(len(outix) and np.any(np.arange(len(outix))-ol+1>=3))
 # Stuck flag: exact-value occupancy or near-flat seven-day rolling window during test.
 stuck=np.zeros(n,bool); testix=np.flatnonzero((tt>=START.to_datetime64())&(tt<END.to_datetime64())&finite)
 stuck_values=[]
 if len(testix):
  counts=pd.Series(v[testix]).value_counts(); tnum=t.astype('int64').to_numpy()
  for value,count in counts[counts>=MIN_RUN].items():
   ix=testix[v[testix]==float(value)]; left=np.searchsorted(tnum[ix],tnum[ix]-pd.Timedelta(days=30).value,side='left')
   if np.any(np.arange(len(ix))-left+1>=MIN_RUN): stuck_values.append(float(value)); stuck[ix]=True
 ser=pd.Series(v,index=t); rmin=ser.rolling('7D',min_periods=20).min(); rmax=ser.rolling('7D',min_periods=20).max(); rsd=ser.rolling('7D',min_periods=20).std()
 near=(rmin.notna()&((rmax-rmin)<=.06)&(rsd<=.02)).to_numpy(); near_test=near&(tt>=START.to_datetime64())&(tt<END.to_datetime64()); stuck|=near_test
 # Current flip detector: qualifying large adjacent-step reversals in a trailing 30-day window.
 oi=np.flatnonzero(finite); ov=v[oi]; ot=t.astype('int64').to_numpy()[oi]; d=np.diff(ov); dt=np.diff(ot); valid=(dt>0)&(dt<=pd.Timedelta(days=2).value); large=valid&(np.abs(d)>=FLIP_STEP)
 pair=large[1:]&large[:-1]; rev=pair&(np.sign(d[1:])!=np.sign(d[:-1])); pt=ot[2:]; left=np.searchsorted(pt,pt-pd.Timedelta(days=30).value,side='left'); pc=np.r_[0,np.cumsum(pair.astype(np.int32))]; pr=np.r_[0,np.cumsum(rev.astype(np.int32))]; ix=np.arange(len(pt)); lcnt=pc[ix+1]-pc[left]; rcnt=pr[ix+1]-pr[left]; share=rcnt/np.maximum(1,lcnt); bad=(rcnt>=FLIP_N)&(share>=FLIP_SHARE)
 flip=np.zeros(n,bool); flip[oi[2:]]=bad; invalid=float((flip&obs).sum()/obs.sum()) if obs.sum() else 0.; flip_station=invalid>=FLIP_STATION_SHARE
 return {'placeholder_prefix':lead,'exact_spike_mask':spike,'stuck_window':stuck,'flipflop_window':(flip if flip_station else np.zeros(n,bool)),'extreme_burst_point':(outlier if burst else np.zeros(n,bool)), 'all_flip_endpoints':flip,'invalid_share':invalid,'burst':burst,'lo':lo,'hi':hi,'med':med,'mad':mad,'runs':rr,'stuck_values':stuck_values,'nearflat':near_test,'nearflat_all':near,'repeated':repeated,'outlier':outlier}

classification=pd.read_csv(OUT.parent/'diagnostics_phase2c/classification_600_stations.csv')
accept=pd.read_csv(OUT.parent/'diagnostics_phase2c/acceptance_per_station.csv')
excluded=set(accept.matched_station.dropna().astype(str))-set([''])
sample=[]; rng=np.random.default_rng(202602)
for cl in ['stuck','flip-flop','placeholder-start','clean']:
 pool=classification[(classification.primary_classification==cl)&(~classification.Station.isin(excluded))].Station.astype(str).to_numpy()
 chosen=sorted(rng.choice(pool,size=min(6,len(pool)),replace=False).tolist())
 for st in chosen: sample.append(dict(Station=st,primary_classification=cl,selection='stratified random; seed=202602; excludes Phase 2c A-E'))
sampled={x['Station'] for x in sample}
for st in NAMED:
 found=classification.loc[classification.Station.map(norm)==norm(st),'Station']
 actual=found.iloc[0] if len(found) else st
 if actual not in sampled: sample.append(dict(Station=actual,primary_classification=(classification.loc[classification.Station==actual,'primary_classification'].iloc[0] if actual in set(classification.Station) else 'missing'),selection='named diagnostic; was used in Phase 2c acceptance'))
sampledf=pd.DataFrame(sample); sampledf.to_csv(OUT/'visual_sample_stations.csv',index=False)

raw=pd.read_parquet(ROOT/'data/aligned/table_6h.parquet',columns=['Station','time','gwl','gwl_obs'])
raw['time']=pd.to_datetime(raw.time); raw['gwl']=pd.to_numeric(raw.gwl,errors='coerce'); raw['gwl_obs']=pd.to_numeric(raw.gwl_obs,errors='coerce'); raw=raw.sort_values(['Station','time']).reset_index(drop=True)
groups={s:g.reset_index(drop=True) for s,g in raw.groupby('Station',sort=False)}
by_station=classification.set_index('Station')
# Export timestamp-level visual data and overlay counts for selected stations.
plot_meta=[]; visual_dir=OUT/'visual_data'; visual_dir.mkdir(exist_ok=True)
for st in sampledf.Station:
 g=groups[st]; m=masks(g,by_station.loc[st]); od=pd.DataFrame({'time':g.time,'gwl':g.gwl,'gwl_obs':g.gwl_obs})
 for key in ['placeholder_prefix','exact_spike_mask','stuck_window','flipflop_window','extreme_burst_point']: od[key]=m[key].astype(np.int8)
 od.to_csv(visual_dir/f'{slug(st)}.csv',index=False)
 plot_meta.append(dict(Station=st,primary_classification=by_station.loc[st,'primary_classification'],selection=sampledf.loc[sampledf.Station==st,'selection'].iloc[0],
  placeholder_prefix_slots=int(m['placeholder_prefix'].sum()),exact_spike_mask_slots=int(m['exact_spike_mask'].sum()),stuck_window_slots=int(m['stuck_window'].sum()),
  flipflop_window_slots=int(m['flipflop_window'].sum()),extreme_burst_points=int(m['extreme_burst_point'].sum()),flipflop_invalid_share=m['invalid_share'],
  robust_low=m['lo'],robust_high=m['hi'],plot_data=f'visual_data/{slug(st)}.csv'))
pd.DataFrame(plot_meta).to_csv(OUT/'visual_detector_counts.csv',index=False)

# Truncation audit for all 29+ stations at the Phase 2c >=236-slot prefix threshold.
prefix=classification[classification.placeholder_start_slots>=236].copy(); trunc=[]
for rr in prefix.itertuples(index=False):
 st=rr.Station; g=groups[st]; m=masks(g,by_station.loc[st]); v=g.gwl.to_numpy(float); finite=np.isfinite(v); t=g.time; ix=np.flatnonzero(finite)
 if not len(ix): continue
 first=t.iloc[ix[0]]; cap=first+pd.Timedelta(days=60); horizon=first+pd.Timedelta(days=120); post=(t>cap)&(t<=horizon)&finite
 prefix_end=t[m['placeholder_prefix']].max() if m['placeholder_prefix'].any() else first
 between=(t>prefix_end)&(t<=cap)&finite
 between_values=v[between.to_numpy()]; between_g=g.loc[between].reset_index(drop=True); between_runs=runs(between_g) if len(between_g) else pd.DataFrame(columns=['slots','value','start','end']); between_flat=int((m['nearflat_all']&between.to_numpy()).sum())
 post_values=v[post.to_numpy()]; post_times=t[post].reset_index(drop=True); post_sentinel=np.isin(post_values,list(SENT)); sent_count=int(post_sentinel.sum())
 postg=g.loc[post].reset_index(drop=True); pr=runs(postg) if len(postg) else pd.DataFrame(columns=['slots','value','start','end']); maxrun=int(pr.slots.max()) if len(pr) else 0
 # Strong evidence of continuation: sentinel repeats >=4 times or exact plateau run >=28 slots beyond cap.
 post_flat=int((m['nearflat_all']&post.to_numpy()).sum()); cont=bool(sent_count>=4 or maxrun>=28 or post_flat>0)
 fit=(g.gwl_obs.to_numpy()==1)&finite&(t.to_numpy()<START.to_datetime64())&(~m['placeholder_prefix'])&(~np.isin(v,list(SENT)))
 med=float(np.median(v[fit])) if fit.any() else np.nan
 changed_level=bool(len(post_values)>=10 and np.nanstd(post_values)>.05 and (not np.isfinite(med) or np.nanmedian(post_values)<med+1000))
 distinct_mid=int(pd.Series(between_values).nunique()) if len(between_values) else 0
 real_mid=bool(distinct_mid>=10 and np.nanstd(between_values)>.05 and int(between_runs.slots.max())<28 if len(between_runs) else distinct_mid>=10 and np.nanstd(between_values)>.05)
 mid_cont=bool(int(np.isin(between_values,list(SENT)).sum())>=4 or (len(between_runs) and int(between_runs.slots.max())>=28) or between_flat>0)
 status='placeholder/flat continues beyond 60d' if cont else ('placeholder/flat persists between prefix and cap, then varies' if mid_cont and changed_level else ('variable data starts after prefix' if real_mid else ('variable data after cap' if changed_level else 'ambiguous after prefix/cap')))
 trunc.append(dict(Station=st,placeholder_prefix_slots=int(rr.placeholder_start_slots),first_finite_time=first,search_cap_time=cap,flagged_prefix_end=prefix_end,
  post_prefix_to_cap_rows=int(between.sum()),post_prefix_distinct_values=distinct_mid,post_prefix_sentinel_slots=int(np.isin(between_values,list(SENT)).sum()),post_prefix_max_exact_run_slots=int(between_runs.slots.max()) if len(between_runs) else 0,post_prefix_nearflat_slots=between_flat,real_looking_data_after_prefix=real_mid,
  post_cap_to_120d_rows=int(post.sum()),post_cap_sentinel_slots=sent_count,max_post_cap_exact_run_slots=maxrun,post_cap_nearflat_slots=post_flat,post_cap_placeholder_or_flat_continues=cont,post_cap_series_std=float(np.nanstd(post_values)) if len(post_values) else np.nan,interpretation=status))
truncdf=pd.DataFrame(trunc).sort_values('Station'); truncdf.to_csv(OUT/'placeholder_truncation_check.csv',index=False)

# Sensitivity: current detector counts and alternative rules from raw series.
sens=[]; sarila_rows=[]
for st,g in groups.items():
 v=g.gwl.to_numpy(float); finite=np.isfinite(v); obs=(g.gwl_obs.to_numpy()==1)&finite; t=g.time; tt=t.to_numpy(); rr=runs(g)
 exact_run=False
 if len(rr): exact_run=bool((rr.slots>=28).any() and ((rr.end>=START)&(rr.start<END)&(rr.slots>=28)).any())
 # test-period strict and current near-flat windows
 ser=pd.Series(v,index=t); rmin=ser.rolling('7D',min_periods=20).min(); rmax=ser.rolling('7D',min_periods=20).max(); rsd=ser.rolling('7D',min_periods=20).std(); test=(tt>=START.to_datetime64())&(tt<END.to_datetime64())
 near_current=bool(np.any(test&rmin.notna().to_numpy()&((rmax-rmin).to_numpy()<=.06)&(rsd.to_numpy()<=.02)))
 near_strict=bool(np.any(test&rmin.notna().to_numpy()&((rmax-rmin).to_numpy()<=.01)&(rsd.to_numpy()<=.005)))
 exact_occ=False; testix=np.flatnonzero((tt>=START.to_datetime64())&(tt<END.to_datetime64())&finite)
 if len(testix):
  cnt=pd.Series(v[testix]).value_counts(); tnum=t.astype('int64').to_numpy()
  for value,nc in cnt[cnt>=28].items():
   ix=testix[v[testix]==float(value)]; left=np.searchsorted(tnum[ix],tnum[ix]-pd.Timedelta(days=30).value,side='left')
   if np.any(np.arange(len(ix))-left+1>=28): exact_occ=True; break
 row=by_station.loc[st]
 cur_stuck=bool(float(row.test_stuck_slots)>0)
 # flip current/d: calculation over all finite six-hour series, with station share threshold.
 oi=np.flatnonzero(finite); ov=v[oi]; ot=t.astype('int64').to_numpy()[oi]; d=np.diff(ov); dt=np.diff(ot); valid=(dt>0)&(dt<=pd.Timedelta(days=2).value)
 def flipcalc(step,sharecut):
  large=valid&(np.abs(d)>=step); pair=large[1:]&large[:-1]; rev=pair&(np.sign(d[1:])!=np.sign(d[:-1])); pt=ot[2:]; left=np.searchsorted(pt,pt-pd.Timedelta(days=30).value,side='left'); pc=np.r_[0,np.cumsum(pair.astype(np.int32))]; pr=np.r_[0,np.cumsum(rev.astype(np.int32))]; j=np.arange(len(pt)); lc=pc[j+1]-pc[left]; rc=pr[j+1]-pr[left]; sh=rc/np.maximum(1,lc); bad=(rc>=6)&(sh>=sharecut); f=np.zeros(len(g),bool); f[oi[2:]]=bad; invalid_obs=int((f&obs).sum()); inv=float(invalid_obs/obs.sum()) if obs.sum() else 0.; return inv>=.20,inv,invalid_obs,int(bad.sum()),int(large.sum()),int(rev.sum())
 _calc_cur,share_cur,sarila_invalid_obs,n_bad,n_large,n_rev=flipcalc(.5,.60); strictflip,share_strict,_,_,nl2,nr2=flipcalc(1.,.80)
 cur_flip=bool(row.get('flipflop_station_flag',False))
 sens.append(dict(Station=st,current_primary=row.primary_classification,current_stuck=cur_stuck,current_flipflop=cur_flip,strict_nearflat_stuck=bool(exact_occ or near_strict),exact_value_runs_only_stuck=exact_run,flipflop_step1_share08=strictflip,
  exact_occurrence_28_in_30d=exact_occ,exact_28_slot_run=exact_run,nearflat_current=near_current,nearflat_strict=near_strict,flip_share_current=share_cur,flip_share_strict=share_strict))
 if norm(st)==norm('Hamirpur Sarila Purani'):
  sarila_rows.append(dict(Station=st,finite_slots=len(oi),observed_slots=int(obs.sum()),large_steps_ge_0_5m=n_large,reversing_large_step_pairs=n_rev,qualifying_window_endpoints=n_bad,qualifying_observed_slots=sarila_invalid_obs,current_invalid_share=share_cur,strict_step1_share08_invalid_share=share_strict,primary_classification=row.primary_classification,interpretation='The 0.87 is a coverage share: qualifying endpoints among gwl_obs=1 rows divided by all observed rows. The 30-day windows overlap, so one local burst flags many neighboring endpoints; this is not a claim that 87% of the series is visibly jagged.'))
sensdf=pd.DataFrame(sens); sensdf.to_csv(OUT/'sensitivity_station_flags.csv',index=False); pd.DataFrame(sarila_rows).to_csv(OUT/'sarila_flipflop_explanation.csv',index=False)
summary=[]
for name,col in [('current thresholds','current_stuck'),('strict range<=0.01, sd<=0.005','strict_nearflat_stuck'),('exact-value contiguous runs >=28 slots only','exact_value_runs_only_stuck')]: summary.append(dict(detector='stuck',setting=name,stations_flagged=int(sensdf[col].sum())))
for name,col in [('current: step>=0.5m, reversal share>=0.60','current_flipflop'),('stricter: step>=1.0m, reversal share>=0.80','flipflop_step1_share08')]: summary.append(dict(detector='flip-flop',setting=name,stations_flagged=int(sensdf[col].sum())))
pd.DataFrame(summary).to_csv(OUT/'sensitivity_summary.csv',index=False)

report=['# Phase 2d: visual precision check','',
'Read-only audit of the Phase 2c detector output. No data or models were changed. Phase 2c A-E acceptance stations are excluded from the random sample. The five requested named stations are then added as targeted diagnostics and are not independent evidence.',
'','## Sample','',f'Fixed seed: 202602. Stratified random sample: 6 each from stuck, flip-flop, placeholder-start, and clean after excluding every matched station in the Phase 2c A-E acceptance file. Added {len(NAMED)} named diagnostics. Total unique visual stations: {len(sampledf)}. See `visual_sample_stations.csv`.',md(sampledf),'',
'Per-station PNGs are in `plots/`; contact sheets contain up to six stations each in a 3x2 layout. Raw observed series and overlay flags are exported to `visual_data/`. Legend colors distinguish placeholder, exact-spike mask, stuck, flip-flop, and extreme-burst points.',
'','## Placeholder truncation check','',f'The Phase 2c prefix threshold selects {len(truncdf)} stations (>=236 flagged slots). Of those, {int(truncdf.post_cap_placeholder_or_flat_continues.sum())} have sentinel repetition or a >=28-slot exact plateau continuing after the first 60 days, within the next 60-day audit interval.',md(truncdf),'',
'This is a direct boundary check: the existing 60-day start search can truncate initialization sequences. Extend the search only when a sentinel/flat run continues, or use run-continuation logic that follows the second plateau until varying in-range observations begin. Do not extend a fixed mask blindly.',
'','## Sensitivity','',md(pd.DataFrame(summary)),'',
'`exact-value runs only` means a contiguous exact-valued run >=28 six-hour slots overlapping the test period. The stricter near-flat setting combines that exact rule with a 7-day rolling range <=0.01m and SD <=0.005m. Flip-flop alternatives retain >=6 reversals and >=20% station invalid-share; only step and reversal-share thresholds change.',
'These are independent detector flags and overlap: Phase 2c exclusive primary labels were 230 stuck and 133 flip-flop stations. Zero stations have a contiguous exact-value test run of >=28 six-hour slots; current stuck flags arise from exact-value occupancy within 30 days and/or near-flat rolling windows.',
'','### Sarila Purani flip-flop share','',md(pd.DataFrame(sarila_rows)),'',
'The current share is the proportion of observed timestamps that end in qualifying 30-day reversal windows, not the fraction of calendar duration or the fraction of visual points with a jagged appearance. Sarila has 154 >=0.5m steps and 32 adjacent reversing large-step pairs; overlapping 30-day windows mark 3,584 of 4,108 observed timestamps (0.8724). The plot can look smooth at full-series scale because the threshold also counts sub-meter reversals alongside much larger level movement. At the stricter >=1m / >=0.80 rule Sarila has invalid-share 0. Its mutually exclusive Phase 2c primary label is `stuck`; the independent flip-flop flag is also true.',
'','## Mask-safety recommendations','',
'- Placeholder prefix: safe only as a candidate mask when the prefix is an explicit sentinel/initial flat sequence and run-continuation confirms the boundary; current 60-day truncation requires review for the stations in `placeholder_truncation_check.csv`.',
'- Exact repeated spikes: safe to mask only with exact repeated value, isolated point, and immediate exact reversion on both adjacent six-hour slots.',
'- Stuck/near-flat and flip-flop: flag-only. Do not delete or replace these points automatically.',
'- Extreme-burst points: flag-only unless they independently satisfy the narrow exact-spike mask.',
'','## Limits and files','',
'The sample is a visual precision check, not an independent performance estimate. Extreme-burst points use robust station bounds from pre-test observed values after excluding placeholder prefixes, known sentinels, and long exact runs. A flagged overlay is an algorithmic label, not verified telemetry fault attribution.','',
'Precision is not estimable from the positive-only phase2c class sample without visual labels for negatives. This pass reports overlay counts and lets a reviewer assess false positives in the 24 clean/stuck/flip-flop/placeholder random strata.',
'','Files: `visual_sample_stations.csv`, `visual_detector_counts.csv`, `placeholder_truncation_check.csv`, `sensitivity_summary.csv`, `sensitivity_station_flags.csv`, `sarila_flipflop_explanation.csv`, `visual_plot_manifest.csv`, individual PNGs in `plots/`, and contact sheets in `contact_sheets/`.','']
(OUT/'phase2d_visual_precision_report.md').write_text('\n'.join(report),encoding='utf-8')
print('visual sample',len(sampledf),'truncation',len(truncdf),'continued',int(truncdf.post_cap_placeholder_or_flat_continues.sum()))
print(pd.DataFrame(summary).to_string(index=False)); print('Sarila',pd.DataFrame(sarila_rows).to_dict('records'))
