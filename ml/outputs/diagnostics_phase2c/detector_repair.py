from pathlib import Path
import os, sys, re, hashlib, struct, zlib

ROOT=Path('/mnt/c/Users/Srijan Gupta/Downloads/Developement/AQUIS/ml')
OUT=ROOT/'outputs/diagnostics_phase2c'; OUT.mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'venv312/lib/python3.12/site-packages'))
os.chdir(ROOT)
import numpy as np
import pandas as pd

TEST0=pd.Timestamp('2025-10-01'); TEST1=pd.Timestamp('2026-10-01'); SIXH=pd.Timedelta(hours=6)
A=[
'Composite School Sagarpur','Gadhela Panchayat Bhawan','Goshala Bada Gaon','Goshala Bhadehdu','Primary School Sanda-Sanee','Primary School Vamanpura','Primary School chandsina','Prmary School Rajpura','Rajkia High School Pakhrauli','Uccha Prathmik Vidhyalaya, Lohra (Pampawan)','Upper Primary School Sarahan Bujurg','Aagan Wadi Kendra Chandni','Composite School Basantpur Saitali-1','Composite School, Ghamhapur','Swarg Dham, Hardua']
B=['Primary School, Dera Devra bujurg','Primary School chandsina','Primary School Sanda-Sanee']
B_CHECK=['Composite School Sagarpur','Rohari']
C=['Primary School/ Composite, Sidki','Primary School/ Composite, Mehmoodpur','Primary School, Silana','Primary School, Kalan Bar Devran','Chitrakoot Manikpur Agarhudha','Banda Tindwari Bhindowra','Hamirpur Kurara Jalokhar']
D=['Hathras Hathras Hathras','Varansi Cholapur Chaubepur','Varansi Harahua Gurwat','Varansi Chirai Gaon Chhahi']
E=['Block Office Kurara','Hamirpur Sarila Purani','Primary School Evani','Primary School Garher','Saint Ravidas Nagar Abholi Gangarampur','Varansi Harahua Kakalpur','Mathura Chaumuha Navagaon-11']
MIN_RUN=28; SENTINEL_RUN=4; SPIKE_REPEAT=3; SPIKE_MAD_MULT=8.; FLIP_STEP_M=0.5; FLIP_REVERSAL_SHARE=.60; FLIP_MIN_REVERSALS=6; FLIP_STATION_SHARE=.20
PUMP_STEP_CAP_M=20.; PUMP_EXCURSION_CAP_M=30.;

def norm(s): return re.sub(r'[^a-z0-9]+','',str(s).casefold())
def slug(s): return re.sub(r'[^A-Za-z0-9]+','_',str(s)).strip('_')[:60]+'_'+hashlib.md5(str(s).encode()).hexdigest()[:8]
def chunk(t,b): return struct.pack('>I',len(b))+t+b+struct.pack('>I',zlib.crc32(t+b)&0xffffffff)
def line(p,w,h,x0,y0,x1,y1,c):
 dx=abs(x1-x0); sx=1 if x0<x1 else -1; dy=-abs(y1-y0); sy=1 if y0<y1 else -1; err=dx+dy
 while True:
  if 0<=x0<w and 0<=y0<h:
   i=(y0*w+x0)*4; p[i:i+4]=bytes((*c,255))
  if x0==x1 and y0==y1: break
  e=2*err
  if e>=dy: err+=dy; x0+=sx
  if e<=dx: err+=dx; y0+=sy
def render(path,g,masked,flagged):
 w,h=1500,480; p=bytearray(bytes((255,255,255,255))*(w*h)); vals=g.gwl.to_numpy(float); times=pd.to_datetime(g.time).astype('int64').to_numpy()
 finite=vals[np.isfinite(vals)]; lo,hi=np.quantile(finite,[.01,.99]) if len(finite) else (-1,1); pad=max((hi-lo)*.08,.1); lo-=pad; hi+=pad
 for panel,(l,r) in enumerate([(55,735),(790,1470)]):
  t,b=30,425
  for k in range(6):
   y=int(t+(b-t)*k/5); line(p,w,h,l,y,r,y,(228,232,235))
  line(p,w,h,l,t,l,b,(70,76,82)); line(p,w,h,l,b,r,b,(70,76,82))
  den=max(int(times.max()-times.min()),1); vv=vals.copy()
  if panel: vv[np.asarray(masked,bool)]=np.nan
  prev=None
  for tm,v in zip(times,vv):
   if not np.isfinite(v): prev=None; continue
   x=int(l+(r-l)*(int(tm)-int(times.min()))/den); y=int(b-(b-t)*(v-lo)/(hi-lo))
   if prev is not None: line(p,w,h,*prev,x,y,(64,117,94) if panel else (107,122,134))
   prev=(x,y)
  # Header-color bar identifies source (gray) and repaired candidate (green).
  line(p,w,h,l,12,r,12,(70,145,110) if panel else (80,95,110))
 raw=b''.join(b'\x00'+p[y*w*4:(y+1)*w*4] for y in range(h))
 blob=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b'')
 path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(blob)

raw=pd.read_parquet(ROOT/'data/aligned/table_6h.parquet',columns=['Station','time','gwl','gwl_obs'])
raw['time']=pd.to_datetime(raw.time); raw['gwl']=pd.to_numeric(raw.gwl,errors='coerce'); raw['gwl_obs']=pd.to_numeric(raw.gwl_obs,errors='coerce')
raw=raw.sort_values(['Station','time']).reset_index(drop=True)
groups={s:g.reset_index(drop=True) for s,g in raw.groupby('Station',sort=False)}
lookup={norm(s):s for s in groups};
def actual(name): return lookup.get(norm(name))
PUMP_REVIEW={actual(x) for x in E+['Primary School, Kalan Bar Devran','Primary School/ Composite, Mehmoodpur','Rohari']}; PUMP_REVIEW.discard(None)
def md(df):
 def cell(x):
  if pd.isna(x): return ''
  return str(x).replace('|','\\|').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

station=[]; event_rows=[]; changed_by_station={}; thresholds=[]
thresholds += [dict(detector='leading_constant_run',threshold='28 six-hour slots (7 days); sentinel first-level >=4 slots',justification='Groundwater level telemetering should not preserve an identical 3-decimal value for a week; shorter sentinel threshold catches coded initial values.'),
dict(detector='two_step_placeholder_start',threshold='repeated sentinel >=4 slots in first 60 days; flag sentinel and subsequent level repeated >=4 times',justification='Catches a placeholder plateau followed by a second flat initialization level; the whole leading initialization prefix is flagged.'),
dict(detector='stuck_test',threshold='>=28 occurrences of one exact value in any 30-day window, plus 7-day rolling range <=0.06m and SD <=0.02m',justification='A week with <=6cm total movement and <=2cm standard deviation is a near-flat logger plateau; flagged only, never removed.'),
dict(detector='repeated_extreme_spike',threshold='same exact observed value >=3 times and >=8 robust MAD from station median (or sentinel), remove only isolated 6h point with immediate exact reversion',justification='Exact recurrence plus immediate reversal distinguishes telemetry spikes from smooth pumping-style recovery.'),
dict(detector='extreme_burst_review',threshold='>=3 robust-range outliers in a 30-day span, even when their exact values differ',justification='Flags bursty extreme events for review without broad magnitude-only removal; masking still requires the exact-repeat and immediate-reversion rule.'),
dict(detector='flipflop_30d',threshold='step magnitude >=0.5m; >=6 qualifying adjacent sign reversals; reversal share >=0.60; station invalid-share >=0.20',justification='Absolute hydrologic level can vary; repeated half-meter-or-larger alternating observed steps in a month and >=20% affected observed slots is an operational sensor-instability criterion.'),
dict(detector='pumping_like_recovery',threshold='each observed 6h step <=20m and total excursion <=30m; multi-step smooth return within 30d',justification='A single >20m step or >30m excursion is not credible pumping drawdown/recovery; such events are fault/level-shift candidates.'),
dict(detector='robust_station_range',threshold='median +/- 8 * 1.4826 * MAD, fitted after excluding long exact runs and sentinel values',justification='Robust scale is not inflated by flagged initialization or stuck segments; supports repeated-extreme screening only.')]

def run_table(g):
 tv=g.time.astype('int64').to_numpy(); v=g.gwl.to_numpy(float); seen=(g.gwl_obs.to_numpy()==1).astype(np.int32)
 same=(np.diff(tv)==SIXH.value)&np.isfinite(v[1:])&np.isfinite(v[:-1])&(v[1:]==v[:-1])
 starts=np.r_[0,np.flatnonzero(~same)+1]; ends=np.r_[starts[1:]-1,len(g)-1]
 obs_prefix=np.r_[0,np.cumsum(seen)]
 return pd.DataFrame(dict(start=g.time.to_numpy()[starts],end=g.time.to_numpy()[ends],value=v[starts],slots=ends-starts+1,obs=obs_prefix[ends+1]-obs_prefix[starts]))

for st,g in groups.items():
 v=g.gwl.to_numpy(float); obs=(g.gwl_obs.to_numpy()==1)&np.isfinite(v); tt=g.time
 runs=run_table(g); lead_mask=np.zeros(len(g),bool); test_stuck=np.zeros(len(g),bool); all_long=np.zeros(len(g),bool)
 finite_times=tt[np.isfinite(v)]; first_finite=finite_times.min() if len(finite_times) else tt.min()
 long_runs=runs[runs.slots>=MIN_RUN]
 for rr in long_runs.itertuples(index=False):
  m=((tt>=rr.start)&(tt<=rr.end)).to_numpy()
  if rr.slots>=MIN_RUN: all_long|=m
  if rr.start==first_finite and rr.slots >= (SENTINEL_RUN if rr.value in (99.,-99.,999.,-999.,9999.,-9999.) else MIN_RUN): lead_mask|=m
  if rr.slots>=MIN_RUN:
   test_stuck|=m&(tt.to_numpy()>=TEST0.to_datetime64())&(tt.to_numpy()<TEST1.to_datetime64())
 # Initialization pattern: first sentinel/long constant run, optionally followed by second flat run in first 30 days.
 lead_runs=runs[runs.start==first_finite]
 early_end=first_finite+pd.Timedelta(days=60); early=(tt>=first_finite)&(tt<=early_end)&np.isfinite(v)
 early_counts=pd.Series(v[early]).value_counts(); first_ix=np.flatnonzero(np.isfinite(v)); first_value=float(v[first_ix[0]]) if len(first_ix) else np.nan
 sentinel=(99.,-99.,999.,-999.,9999.,-9999.)
 sentinel_counts=early_counts[early_counts.index.isin(sentinel)]
 if len(sentinel_counts):
  first_value=float(sentinel_counts.index[0]); lead_mask|=early&(v==first_value)
  second_values=early_counts.drop(labels=[first_value],errors='ignore'); second_values=second_values[second_values>=SENTINEL_RUN]
  if len(second_values):
   second_value=float(second_values.index[0]); lead_mask|=early&(v==second_value)
 elif np.isfinite(first_value) and int(early_counts.get(first_value,0))>=MIN_RUN:
  lead_mask|=early&(v==first_value)
 # robust physical-ish station bounds excluding sentinel and long constant runs
 fit=obs&(~all_long)&(~lead_mask)&(~np.isin(v,[99.,-99.,999.,-999.,9999.,-9999.]))&(tt.to_numpy()<TEST0.to_datetime64())
 x=v[fit]
 if len(x): med=float(np.median(x)); mad=float(np.median(np.abs(x-med))); scale=max(1.4826*mad,.03); low=med-SPIKE_MAD_MULT*scale; high=med+SPIKE_MAD_MULT*scale
 else: med=mad=low=high=np.nan
 finite=np.isfinite(v); vc=pd.Series(v[finite]).value_counts() if finite.any() else pd.Series(dtype=int)
 repeated={float(k):int(n) for k,n in vc.items() if n>=SPIKE_REPEAT and (k in (99.,-99.,999.,-999.,9999.,-9999.) or (np.isfinite(low) and (k<low or k>high)))}
 outlier_mask=finite&np.isfinite(low)&((v<low)|(v>high))&(~lead_mask)
 out_ix=np.flatnonzero(outlier_mask); tvall=tt.astype('int64').to_numpy()
 out_left=np.searchsorted(tvall[out_ix],tvall[out_ix]-pd.Timedelta(days=30).value,side='left') if len(out_ix) else np.array([],int)
 extreme_burst=bool(len(out_ix) and np.any(np.arange(len(out_ix))-out_left+1>=3))
 # remove only exact repeated observed isolated points with immediate exact reversion at adjacent six-hour slots
 spike=np.zeros(len(g),bool)
 if len(g)>2 and repeated:
  tv=tt.astype('int64').to_numpy(); eligible=np.isin(v[1:-1],np.fromiter(repeated.keys(),dtype=float))
  spike[1:-1]=(finite[1:-1]&eligible&finite[:-2]&finite[2:]&(v[:-2]==v[2:])&(v[1:-1]!=v[:-2])&
   ((tv[1:-1]-tv[:-2])==SIXH.value)&((tv[2:]-tv[1:-1])==SIXH.value))
 # rolling reversal fraction on observed consecutive reads <=48h apart
 oi=np.flatnonzero(finite); ov=v[oi]; ot=tt.astype('int64').to_numpy()[oi]; d=np.diff(ov); dt=np.diff(ot); valid=(dt>0)&(dt<=pd.Timedelta(days=2).value)
 large=valid&(np.abs(d)>=FLIP_STEP_M); pair_large=large[1:]&large[:-1]
 pair_rev=pair_large&(np.sign(d[1:])!=np.sign(d[:-1])); pair_times=ot[2:]
 left=np.searchsorted(pair_times,pair_times-pd.Timedelta(days=30).value,side='left')
 prev_rev=np.r_[0,np.cumsum(pair_rev.astype(np.int32))]; prev_large=np.r_[0,np.cumsum(pair_large.astype(np.int32))]
 idx=np.arange(len(pair_times)); rev_count=prev_rev[idx+1]-prev_rev[left]; large_count=prev_large[idx+1]-prev_large[left]
 rev_share=rev_count/np.maximum(1,large_count)
 roll_bad=(rev_count>=FLIP_MIN_REVERSALS)&(rev_share>=FLIP_REVERSAL_SHARE)
 flip_obs=np.zeros(len(g),bool); flip_obs[oi[2:]]=roll_bad
 invalid_share=float((flip_obs&obs).sum()/obs.sum()) if obs.sum() else 0.; flip_station=invalid_share>=FLIP_STATION_SHARE
 # In the test window, a repeated exact value occupying >=28 slots within a 30-day span is a stuck plateau.
 stuck_values=[]; test_ix=np.flatnonzero((tt.to_numpy()>=TEST0.to_datetime64())&(tt.to_numpy()<TEST1.to_datetime64())&finite)
 if len(test_ix):
  test_counts=pd.Series(v[test_ix]).value_counts(); tnum=tt.astype('int64').to_numpy()
  for value,count in test_counts[test_counts>=MIN_RUN].items():
   ixv=test_ix[v[test_ix]==float(value)]; ix_times=tnum[ixv]; leftv=np.searchsorted(ix_times,ix_times-pd.Timedelta(days=30).value,side='left')
   if np.any(np.arange(len(ixv))-leftv+1>=MIN_RUN): stuck_values.append(float(value)); test_stuck[ixv]=True
 roll_min=pd.Series(v,index=tt).rolling('7D',min_periods=20).min()
 roll_max=pd.Series(v,index=tt).rolling('7D',min_periods=20).max()
 roll_std=pd.Series(v,index=tt).rolling('7D',min_periods=20).std()
 nearflat=(roll_min.notna()&((roll_max-roll_min)<=.06)&(roll_std<=.02)).to_numpy()
 nearflat_test=nearflat&(tt.to_numpy()>=TEST0.to_datetime64())&(tt.to_numpy()<TEST1.to_datetime64())
 test_stuck|=nearflat_test
 # pumping-like smooth return candidates on consecutive observed points; enforce hard physical caps
 pump_events=[]
 # Rolling baseline is computed in vectorized form; only possible moderate excursions are tested.
 if st in PUMP_REVIEW:
  obs_values=pd.Series(ov)
  baseline=obs_values.shift(1).rolling(28,min_periods=4).median().to_numpy()
  excursion=np.abs(ov-baseline)
  possible=np.isfinite(baseline)&(excursion>=.5)&(excursion<=PUMP_EXCURSION_CAP_M)
  # Treat only local excursion peaks as episode starts to avoid scoring every point in one recovery.
  prev=np.r_[np.inf,excursion[:-1]]; nxt=np.r_[excursion[1:],np.inf]
  candidate_ix=np.flatnonzero(possible&(excursion>=prev)&(excursion>=nxt))
  for j in candidate_ix:
   a=oi[j]
   if j>0 and (ot[j]-ot[j-1])<=pd.Timedelta(days=2).value and abs(ov[j]-ov[j-1])>PUMP_STEP_CAP_M: continue
   target=min(j+30,len(ov)-1); direction=np.sign(baseline[j]-ov[j]); delta=np.diff(ov[j:target+1]); moves=int(((np.sign(delta)==direction)&(np.abs(delta)>0)).sum())
   if target>j and moves>=3 and abs(ov[target]-baseline[j])<=max(.5,excursion[j]*.2):
    pump_events.append((a,oi[target],float(excursion[j]),moves))
 # Placeholder prefix/two-step and test stuck are flags. Cleaned mask only removes exact isolated repeated spikes and placeholder init prefix.
 changed=lead_mask|spike
 spike_station=bool(repeated) or extreme_burst
 status='placeholder-start' if lead_mask.any() else ('stuck' if test_stuck.any() else ('flip-flop' if flip_station else ('spike-faults' if spike_station else ('pumping-like real' if pump_events else 'clean'))))
 station.append(dict(Station=st,primary_classification=status,placeholder_start_slots=int(lead_mask.sum()),leading_run_slots=int(lead_runs.iloc[0].slots) if len(lead_runs) else 0,
  test_stuck_slots=int(test_stuck.sum()),test_stuck_runs=len(stuck_values),test_stuck_values=';'.join(map(str,stuck_values)),
  repeated_extreme_values=len(repeated),repeated_extreme_flag=spike_station,extreme_outlier_points=int(outlier_mask.sum()),extreme_burst_flag=extreme_burst,repeated_extreme_observed_slots=int(sum((obs&(v==q)).sum() for q in repeated)),isolated_spike_points=int(spike.sum()),
  robust_median=med,robust_mad=mad,robust_low=low,robust_high=high,flipflop_invalid_share=invalid_share,flipflop_station_flag=flip_station,
  pump_candidate_events=len(pump_events),pump_capped_candidates=int(sum(1 for e in pump_events if e[2]>PUMP_EXCURSION_CAP_M)),changed_slots=int(changed.sum()),first_time=tt.min(),last_time=tt.max()))
 # per detector counts/events table for stations with flagged rows
 changed_by_station[st]=changed
 for detector,mask in [('placeholder_start',lead_mask),('test_stuck',test_stuck),('repeated_extreme_spike',spike),('flipflop_burst',flip_obs),('candidate_changed',changed)]:
  if mask.any():
   ix=np.flatnonzero(mask); event_rows.append(dict(Station=st,detector=detector,flagged_rows=len(ix),first_flagged_time=tt.iloc[ix[0]],last_flagged_time=tt.iloc[ix[-1]]))

stations=pd.DataFrame(station).sort_values('Station'); stations.to_csv(OUT/'classification_600_stations.csv',index=False)
pd.DataFrame(event_rows).to_csv(OUT/'detector_flagged_rows.csv',index=False)
pd.DataFrame(thresholds).to_csv(OUT/'final_thresholds.csv',index=False)

# Evaluate exact hand-labelled acceptance sets without changing the lists.
cases=[]
for group,names,detector in [('A_placeholder',A,'placeholder_start_slots'),('B_stuck',B,'test_stuck_slots'),('B_check',B_CHECK,'test_stuck_slots'),('C_repeated_spike',C,'repeated_extreme_flag'),('D_flipflop',D[:1],'flipflop_station_flag'),('D_check',D[1:],'flipflop_station_flag'),('E_untouched',E,'changed_slots')]:
 for name in names:
  st=actual(name); row=stations[stations.Station==st].iloc[0] if st else None
  val=bool(row[detector]>0) if row is not None else False
  descriptive=group in ('B_check','D_check')
  passed=(not val) if group=='E_untouched' else ('' if descriptive else val)
  cases.append(dict(group=group,case_type=('descriptive' if descriptive else 'acceptance'),requested_station=name,matched_station=st or '',detector=detector,detector_value=(row[detector] if row is not None else ''),observed_flag=val,expected=('inspect' if descriptive else ('untouched' if group=='E_untouched' else 'flagged')),pass_acceptance=passed))
accept=pd.DataFrame(cases); accept.to_csv(OUT/'acceptance_per_station.csv',index=False)
group_rows=[]
for group,g in accept.groupby('group',sort=False):
 n=len(g); descriptive=g.case_type.iloc[0]=='descriptive'
 if descriptive: passed=failed=precision=recall=specificity=np.nan
 elif group=='E_untouched': passed=int(g.pass_acceptance.sum()); failed=0; precision=recall=np.nan; specificity=passed/n if n else np.nan
 else: passed=int(g.pass_acceptance.sum()); failed=n-passed; precision=np.nan; recall=passed/n if n else np.nan; specificity=np.nan
 group_rows.append(dict(group=group,case_type=g.case_type.iloc[0],stations=n,flagged=int(g.observed_flag.sum()),passed=passed,failed=failed,precision=precision,recall=recall,negative_control_specificity=specificity))
pd.DataFrame(group_rows).to_csv(OUT/'acceptance_group_summary.csv',index=False)

# Before/after diagnostic plots for unique union A-D and E stations; same date/scale panels.
plotstations=set()
for nm in A+B+B_CHECK+C+D+E:
 s=actual(nm)
 if s: plotstations.add(s)
manifest=[]
for st in sorted(plotstations):
 g=groups[st]; row=stations[stations.Station==st].iloc[0]
 # Only placeholder prefixes and exact spikes are masked; stuck and flip-flop flags remain visible.
 mask=changed_by_station[st]
 render(OUT/'before_after'/f'{slug(st)}_before_after.png',g,mask,mask)
 manifest.append(dict(Station=st,plot=f'before_after/{slug(st)}_before_after.png',changed_slots=int(mask.sum()),acceptance_groups=';'.join(sorted(set(accept.loc[accept.matched_station==st,'group'])))))
pd.DataFrame(manifest).to_csv(OUT/'before_after_manifest.csv',index=False)

report=['# Phase 2c: detector repair and acceptance results','',
'Read-only forensic pass. The input aligned table and model directories were not changed. No data rebuild was attempted. Hand-labelled acceptance sets A-E were preserved as supplied; disagreements are shown directly in the station table.',
'','## Detector thresholds and rationale','',md(pd.DataFrame(thresholds)),'',
'A/B flags are not all removals: stuck and flip-flop segments are flagged for review and retained in the after-series. The after panel masks only detected placeholder initialization and exact repeated spikes with immediate exact reversion. Pumping-like candidates are not removed.',
'','## Acceptance summary','',md(pd.DataFrame(group_rows)),'',
'A-E station-level pass/fail detail: `acceptance_per_station.csv`. A-D provide positive hand-labelled examples only, so precision cannot be estimated without a labelled negative sample; recall is the share of requested positives flagged. E provides negative controls, so `negative_control_specificity` is the share left unchanged. B_check and D_check are descriptive only: their `flagged` counts report detector output without imposing a hand label.',
'','### Descriptive checks: Sagarpur, Rohari, and other oscillation stations','',md(accept[accept.case_type=='descriptive'][['group','requested_station','matched_station','detector_value','observed_flag']]),'',
'','## Corrected 600-station classification','',md(stations.primary_classification.value_counts().rename_axis('classification').reset_index(name='stations')),'',
'Station-level details and detector counts are in `classification_600_stations.csv`; per-station flagged-row counts and intervals are in `detector_flagged_rows.csv`.',
'','## Named guard and physical-cap review','',md(stations[stations.Station.map(norm).isin([norm(x) for x in E+['Primary School, Kalan Bar Devran','Primary School/ Composite, Mehmoodpur','Rohari']])][['Station','primary_classification','changed_slots','isolated_spike_points','flipflop_invalid_share','pump_candidate_events']]),'',
'Pumping-like label candidates are limited to smooth multi-step returns with each 6-hour change <=20m and total departure <=30m. Excursions beyond either cap are not called pumping. An abrupt change is still not a proven datum fault without independent evidence.',
'','## Still failing','']
fails=accept[(accept.case_type=='acceptance')&(accept.pass_acceptance==False)]
if len(fails): report += [md(fails[['group','requested_station','matched_station','detector_value']]),'']
else: report += ['No acceptance cases failed.','']
report += ['## Plot set','',f'Generated {len(manifest)} before/after plots for all matched unique stations in A-D and E. Files are listed in `before_after_manifest.csv`.','',
'## Limits','',
'The series-level heuristics are screening rules, not verified sensor diagnoses. Robust station limits are fit on pre-test observations after excluding known sentinels and long exact runs. No physical well datum, logger specification, or station maintenance log was available to verify fault causes. Missing requested station-name matches, if any, appear as failed acceptance rows.','']
(OUT/'phase2c_detector_repair_report.md').write_text('\n'.join(report),encoding='utf-8')
print('stations',len(stations),'plots',len(manifest))
print(pd.DataFrame(group_rows).to_string(index=False))
print('failures:')
print(fails[['group','requested_station','matched_station','detector_value']].to_string(index=False))
print('classifications:')
print(stations.primary_classification.value_counts().to_string())
