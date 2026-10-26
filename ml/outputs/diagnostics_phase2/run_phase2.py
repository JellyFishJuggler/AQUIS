from pathlib import Path
import os, sys, re, hashlib, struct, zlib

ROOT = Path('/mnt/c/Users/Srijan Gupta/Downloads/Developement/AQUIS/ml')
OUT = ROOT / 'outputs/diagnostics_phase2'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'venv312/lib/python3.12/site-packages'))
os.chdir(ROOT)

import numpy as np
import pandas as pd

def png_chunk(kind, data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)

def raster_png(path, series):
    # Small dependency-free time-series renderer; all buffers and output stay in OUT.
    width,height=1200,420
    pix=bytearray(width*height*4)
    for i in range(0,len(pix),4): pix[i:i+4]=bytes((255,255,255,255))
    def put(x,y,color):
        if 0<=x<width and 0<=y<height:
            i=(y*width+x)*4; pix[i:i+4]=bytes((*color,255))
    def line(x0,y0,x1,y1,color):
        dx=abs(x1-x0); sx=1 if x0<x1 else -1
        dy=-abs(y1-y0); sy=1 if y0<y1 else -1; err=dx+dy
        while True:
            put(x0,y0,color)
            if x0==x1 and y0==y1: break
            e2=2*err
            if e2>=dy: err+=dy; x0+=sx
            if e2<=dx: err+=dx; y0+=sy
    x0,x1=58,width-22; y0,y1=20,height-34
    vals=series.gwl.to_numpy(float)
    good=np.isfinite(vals)
    if not good.any():
        ymin,ymax=-1.,1.
    else:
        ymin,ymax=float(np.nanmin(vals)),float(np.nanmax(vals))
        pad=max((ymax-ymin)*.05,.25); ymin-=pad; ymax+=pad
    for k in range(6):
        yy=int(y0+(y1-y0)*k/5)
        line(x0,yy,x1,yy,(224,229,233))
    for k in range(7):
        xx=int(x0+(x1-x0)*k/6)
        line(xx,y0,xx,y1,(235,238,241))
    line(x0,y0,x0,y1,(45,55,65)); line(x0,y1,x1,y1,(45,55,65))
    ts=pd.to_datetime(series.time).astype('int64').to_numpy()
    tmin,tmax=int(ts.min()),int(ts.max())
    den=max(tmax-tmin,1)
    coords=[]
    for t,v,ok in zip(ts,vals,good):
        if not ok:
            coords.append(None); continue
        x=int(x0+(x1-x0)*(int(t)-tmin)/den)
        y=int(y1-(y1-y0)*(v-ymin)/(ymax-ymin))
        coords.append((x,y))
    prev=None
    for xy in coords:
        if xy is None: prev=None; continue
        if prev is not None: line(prev[0],prev[1],xy[0],xy[1],(112,128,140))
        prev=xy
    observed=(pd.to_numeric(series.gwl_obs,errors='coerce').to_numpy()==1)
    for (xy,obs) in zip(coords,observed):
        if xy is not None and obs:
            x,y=xy
            for dx in (-1,0,1):
                for dy in (-1,0,1): put(x+dx,y+dy,(18,103,130))
    scan=b''.join(b'\x00'+pix[y*width*4:(y+1)*width*4] for y in range(height))
    png=b'\x89PNG\r\n\x1a\n'+png_chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,6,0,0,0))+png_chunk(b'IDAT',zlib.compress(scan,6))+png_chunk(b'IEND',b'')
    path.write_bytes(png)

CUTOFF = pd.Timestamp('2025-10-01')
DAILY = pd.Timedelta(days=1)
THIRTY = pd.Timedelta(days=30)
JUMP5 = 5.0
JUMP15 = 15.0
P99_RULE_THRESHOLD = 15.0

def norm(s):
    return str(s).strip().casefold()

def station_slug(s):
    base = re.sub(r'[^A-Za-z0-9]+','_',str(s)).strip('_')[:70] or 'station'
    return base + '_' + hashlib.md5(str(s).encode('utf-8')).hexdigest()[:8]

def rmse(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float('nan')

def mae(v):
    return float(np.mean(np.abs(v))) if len(v) else float('nan')

def score(df, model_col, obs_col, base_col, label, subset):
    d=df[[model_col,obs_col,base_col,'Station']].dropna().copy()
    em=(d[model_col].to_numpy(float)-d[obs_col].to_numpy(float))
    eb=(d[base_col].to_numpy(float)-d[obs_col].to_numpy(float))
    cap=float(np.quantile(np.r_[np.abs(em),np.abs(eb)],.99)) if len(d) else float('nan')
    station=[]
    for st,g in d.groupby('Station'):
        a=g[model_col].to_numpy(float)-g[obs_col].to_numpy(float)
        b=g[base_col].to_numpy(float)-g[obs_col].to_numpy(float)
        station.append(1-rmse(a)/rmse(b) if rmse(b)>0 else np.nan)
    return dict(artifact=label,subset=subset,n=len(d),n_stations=d.Station.nunique(),
        rmse_model=rmse(em),rmse_persistence=rmse(eb),skill=1-rmse(em)/rmse(eb) if rmse(eb)>0 else np.nan,
        mae_model=mae(em),mae_persistence=mae(eb),mae_skill=1-mae(em)/mae(eb) if mae(eb)>0 else np.nan,
        median_station_skill=float(np.nanmedian(station)) if station else np.nan,
        winsor_cap_abs_error_p99=cap,
        winsor_rmse_model=rmse(np.clip(em,-cap,cap)) if len(d) else np.nan,
        winsor_rmse_persistence=rmse(np.clip(eb,-cap,cap)) if len(d) else np.nan,
        winsor_rmse_skill=(1-rmse(np.clip(em,-cap,cap))/rmse(np.clip(eb,-cap,cap))) if len(d) and rmse(np.clip(eb,-cap,cap))>0 else np.nan)

def station_metrics(df, model_col, obs_col, base_col):
    rows=[]
    for st,g in df.groupby('Station'):
        a=g[model_col].to_numpy(float)-g[obs_col].to_numpy(float)
        b=g[base_col].to_numpy(float)-g[obs_col].to_numpy(float)
        rows.append(dict(Station=st,n=len(g),rmse_model=rmse(a),rmse_persistence=rmse(b),
                         skill=1-rmse(a)/rmse(b) if rmse(b)>0 else np.nan,
                         mae_model=mae(a),mae_persistence=mae(b),
                         mae_skill=1-mae(a)/mae(b) if mae(b)>0 else np.nan,
                         sqerr=float(np.square(a).sum())))
    return pd.DataFrame(rows)

# Required read-only saved inputs.
raw=pd.read_parquet(ROOT/'data/aligned/table_6h.parquet', columns=['Station','time','gwl','gwl_obs'])
raw['time']=pd.to_datetime(raw['time'])
raw=raw[(raw.time>=pd.Timestamp('2021-01-01'))&(raw.time<pd.Timestamp('2027-01-01'))].copy()
raw=raw.sort_values(['Station','time']).reset_index(drop=True)
raw['gwl_obs']=pd.to_numeric(raw.gwl_obs,errors='coerce')
raw['gwl']=pd.to_numeric(raw.gwl,errors='coerce')
grp=raw.groupby('Station',sort=False)
raw['_prev1d']=grp.gwl.shift(4)
raw['_tprev1d']=grp.time.shift(4)
raw['_prev30d']=grp.gwl.shift(120)
raw['_tprev30d']=grp.time.shift(120)
raw['_obs_prev1d']=grp.gwl_obs.shift(4)
raw['_obs_prev30d']=grp.gwl_obs.shift(120)
raw['_delta1d']=raw.gwl-raw._prev1d
raw['_delta30d']=raw.gwl-raw._prev30d
raw['_exact1d']=(raw.time-raw._tprev1d)==DAILY
raw['_exact30d']=(raw.time-raw._tprev30d)==THIRTY
raw['_observed1d']=(raw.gwl_obs==1)&(raw._obs_prev1d==1)
raw['_observed30d']=(raw.gwl_obs==1)&(raw._obs_prev30d==1)

# Reproduce the existing flag_regime_shift logic from ml/06_features.py (max_delta=15m).
raw['date']=raw.time.dt.normalize()
annual=raw.dropna(subset=['gwl']).assign(year=lambda x:x.date.dt.year).groupby(['Station','year']).gwl.median().unstack()
base_years=[y for y in range(2021,2025) if y in annual.columns]
base=annual[base_years].median(axis=1) if base_years else pd.Series(dtype=float)
dev=annual.sub(base,axis=0).abs().dropna(axis=1,how='all') if len(base) else pd.DataFrame(index=annual.index)
max_annual_deviation=dev.max(axis=1) if not dev.empty else pd.Series(index=annual.index,dtype=float)
regime_bad=set(max_annual_deviation[max_annual_deviation>15].index.astype(str))

# Outlier union: top 20 XGBoost 30d contribution + top 20 trajectory DL h720 contribution.
phase1_top=pd.read_csv(OUT.parent/'diagnostics_phase1/top20_squared_error_contribution.csv')
xgb_top=phase1_top[(phase1_top.source=='predictions_2026')&(phase1_top.model=='XGBoost')].sort_values('squared_error',ascending=False).head(20)
traj_rows=pd.read_parquet(ROOT/'outputs/traj_backtest_rows_dl.parquet')
traj_rows['anchor_t']=pd.to_datetime(traj_rows.anchor_t)
traj720=traj_rows[traj_rows.horizon_h==720].copy()
traj720['Station']=traj720.station
traj720['sqerr_p']=np.square(traj720.obs-traj720.persist)
traj720['sqerr_m']=np.square(traj720.traj_q50-traj720.obs)
traj_top=traj720.groupby('Station').sqerr_m.sum().sort_values(ascending=False).head(20).rename('squared_error').reset_index()
top_xgb=set(xgb_top.station.astype(str)); top_traj=set(traj_top.Station.astype(str)); union=sorted(top_xgb|top_traj)

outlier_rows=[]
for st in union:
    g=raw[raw.Station.astype(str)==st].copy()
    j=g[g._exact1d & g._observed1d & g._delta1d.notna()]
    d30=g[g._exact30d & g._observed30d & g._delta30d.notna()]
    md=float(max_annual_deviation.get(st,np.nan))
    max_jump=float(j._delta1d.abs().max()) if len(j) else np.nan
    max30_idx=d30._delta30d.abs().idxmax() if len(d30) else None
    max30=float(d30.loc[max30_idx,'_delta30d']) if max30_idx is not None else np.nan
    is_bad=st in regime_bad
    fault_consistent=is_bad or (np.isfinite(max_jump) and max_jump>15)
    category='consistent with datum/telemetry fault' if fault_consistent else 'plausible hydrology'
    if is_bad:
        evidence=f'existing annual-median regime filter fails: maximum deviation from 2021-24 station baseline = {md:.2f} m (>15 m); consistent with a persistent offset, though cause is unverified.'
    elif fault_consistent:
        evidence=f'filter passes, but an observed-endpoint 1-day jump reaches {max_jump:.2f} m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.'
    else:
        evidence=f'existing annual-median regime filter passes (max annual-median deviation {md:.2f} m; threshold 15 m); max observed-endpoint 1-day jump is {max_jump:.2f} m.'
    outlier_rows.append(dict(Station=st,top_xgb_30d=st in top_xgb,top_traj_dl_h720=st in top_traj,
        range_min=float(g.gwl.min()) if g.gwl.notna().any() else np.nan,range_max=float(g.gwl.max()) if g.gwl.notna().any() else np.nan,
        n_1day_jumps_gt5=int((j._delta1d.abs()>JUMP5).sum()),n_1day_jumps_gt15=int((j._delta1d.abs()>JUMP15).sum()),
        max_abs_1day_jump=float(max_jump),observed_slot_share=float((g.gwl_obs==1).sum()/len(g)) if len(g) else np.nan,
        max_abs_30day_change=float(d30._delta30d.abs().max()) if len(d30) else np.nan,
        signed_change_at_max_abs_30day=max30,existing_regime_shift_filter_passed=not is_bad,
        max_annual_median_deviation_from_2021_24_m=md,classification=category,evidence=evidence))
    raster_png(OUT/f'{station_slug(st)}.png',g)
outliers=pd.DataFrame(outlier_rows)
outliers.to_csv(OUT/'outlier_forensics.csv',index=False)

# Pre-cutoff station-only rule: p99 absolute observed-to-observed exact 30-day change >15m.
pre=raw[(raw.time<CUTOFF)&raw._exact30d&raw._observed30d&raw._delta30d.notna()].copy()
p99=pre.groupby('Station')._delta30d.apply(lambda x:float(x.abs().quantile(.99))).rename('pre_cutoff_p99_abs_30day_change_m').reset_index()
p99['quality_filter_removed']=p99.pre_cutoff_p99_abs_30day_change_m>P99_RULE_THRESHOLD
removed=set(p99.loc[p99.quality_filter_removed,'Station'].astype(str))
p99['rule_threshold_m']=P99_RULE_THRESHOLD
p99['rule']='p99(|GWL(t)-GWL(t-30d)|), exact 30d and both endpoints gwl_obs=1, using t<2025-10-01; remove if >15m'
p99.to_csv(OUT/'precutoff_quality_rule_by_station.csv',index=False)
removed_rows=p99[p99.quality_filter_removed].sort_values('pre_cutoff_p99_abs_30day_change_m',ascending=False)
removed_rows.to_csv(OUT/'quality_filter_removed_stations.csv',index=False)

pred=pd.read_parquet(ROOT/'outputs/predictions_2026.parquet')
pred30=pred[pred.horizon==30].copy(); pred30['Station']=pred30.Station.astype(str)
evals=[]
for label,d,col,obs,basecol in [('XGBoost 30d',pred30,'xgb','target','persist'),('Trajectory DL h720',traj720,'traj_q50','obs','persist')]:
    evals.append(score(d,col,obs,basecol,label,'all stations'))
    kept=d[~d.Station.isin(removed)]
    evals.append(score(kept,col,obs,basecol,label,'retained after pre-cutoff rule'))
pd.DataFrame(evals).to_csv(OUT/'quality_filter_skill_comparison.csv',index=False)

# Nearest rain-gauge association by GWL station; <150km thresholds as requested.
assoc=pd.read_csv(ROOT/'data/meta/assoc_rainfall.csv')
assoc['dist_km']=pd.to_numeric(assoc.dist_km,errors='coerce')
assoc['_station_key']=assoc.gwl_station.map(norm)
nearest=assoc.dropna(subset=['dist_km']).groupby('_station_key').dist_km.min().to_dict()
def distance_bucket(st):
    dist=nearest.get(norm(st),np.nan)
    if not np.isfinite(dist) or dist>150: return 'none'
    if dist<25: return '<25 km'
    if dist<75: return '25-75 km'
    return '75-150 km'
dist_rows=[]
for label,d,col,obs,basecol in [('XGBoost 30d',pred30,'xgb','target','persist'),('Trajectory DL h720',traj720,'traj_q50','obs','persist')]:
    d=d.copy(); d['distance_bucket']=d.Station.map(distance_bucket)
    for bucket,g in d.groupby('distance_bucket',sort=False):
        metrics=score(g,col,obs,basecol,label,bucket)
        dist_rows.append(dict(distance_bucket=bucket,**metrics))
pd.DataFrame(dist_rows).to_csv(OUT/'rain_gauge_distance_skill.csv',index=False)
station_distance=pd.DataFrame([dict(Station=s,nearest_rain_gauge_km=nearest.get(norm(s),np.nan),distance_bucket=distance_bucket(s)) for s in sorted(set(pred30.Station)|set(traj720.Station))])
station_distance.to_csv(OUT/'rain_gauge_distance_by_station.csv',index=False)

# h18 analysis: match anchor and target times to the approved aligned table for observation flags.
raw_flag=raw[['Station','time','gwl','gwl_obs']].drop_duplicates(['Station','time']).copy()
raw_flag['Station']=raw_flag.Station.astype(str)
traj_rows['Station']=traj_rows.station.astype(str)
traj_rows['target_t']=traj_rows.anchor_t+pd.to_timedelta(traj_rows.horizon_h,unit='h')
anchor_flags=raw_flag.rename(columns={'time':'anchor_t','gwl':'raw_anchor_gwl','gwl_obs':'anchor_gwl_obs'})
target_flags=raw_flag.rename(columns={'time':'target_t','gwl':'raw_target_gwl','gwl_obs':'target_gwl_obs'})
hdf=traj_rows.merge(anchor_flags,on=['Station','anchor_t'],how='left').merge(target_flags,on=['Station','target_t'],how='left')
hdf['persist_err']=hdf.persist-hdf.obs
hdf['persist_sqerr']=hdf.persist_err.pow(2)
hdf['anchor_hour']=hdf.anchor_t.dt.hour
hdf['flags_group']=hdf.anchor_gwl_obs.fillna(-1).astype(int).astype(str)+'->'+hdf.target_gwl_obs.fillna(-1).astype(int).astype(str)
hs=hdf[hdf.horizon_h.isin([6,12,18,24])].copy()
hs.groupby('horizon_h').apply(lambda g:pd.Series(dict(n=len(g),n_stations=g.Station.nunique(),
    persistence_rmse=rmse(g.persist_err.to_numpy()),persistence_mae=mae(g.persist_err.to_numpy()),
    share_anchor_observed=(g.anchor_gwl_obs==1).mean(),share_target_observed=(g.target_gwl_obs==1).mean(),
    share_both_observed=((g.anchor_gwl_obs==1)&(g.target_gwl_obs==1)).mean(),
    artifact_obs_matches_raw_target_level=((g.obs-g.raw_target_gwl).abs()<1e-5).mean()))).reset_index().to_csv(OUT/'h18_horizon_observation_summary.csv',index=False)
hs.groupby(['horizon_h','anchor_gwl_obs','target_gwl_obs'],dropna=False).agg(n=('persist_err','size'),
    persistence_rmse=('persist_err',lambda x:rmse(x.to_numpy())),persistence_mae=('persist_err',lambda x:mae(x.to_numpy())),
    n_stations=('Station','nunique')).reset_index().to_csv(OUT/'h18_observation_flag_groups.csv',index=False)
hs.groupby(['horizon_h','anchor_hour']).agg(n=('persist_err','size'),n_stations=('Station','nunique'),
    persistence_rmse=('persist_err',lambda x:rmse(x.to_numpy())),persistence_mae=('persist_err',lambda x:mae(x.to_numpy())),
    squared_error=('persist_sqerr','sum')).reset_index().to_csv(OUT/'h18_anchor_hour_summary.csv',index=False)
station_h=hs.groupby(['horizon_h','Station']).agg(n=('persist_err','size'),persistence_rmse=('persist_err',lambda x:rmse(x.to_numpy())),
    persistence_mae=('persist_err',lambda x:mae(x.to_numpy())),squared_error=('persist_sqerr','sum'),
    share_anchor_observed=('anchor_gwl_obs',lambda x:(x==1).mean()),share_target_observed=('target_gwl_obs',lambda x:(x==1).mean())).reset_index()
station_h['share_horizon_sqerr']=station_h.squared_error/station_h.groupby('horizon_h').squared_error.transform('sum')
station_h.sort_values(['horizon_h','squared_error'],ascending=[True,False]).groupby('horizon_h').head(20).to_csv(OUT/'h18_top_station_contributions.csv',index=False)
h18=hs[hs.horizon_h==18].sort_values('persist_sqerr',ascending=False)
h18.head(30)[['Station','anchor_t','target_t','anchor_hour','obs','persist','persist_err','persist_sqerr','anchor_gwl_obs','target_gwl_obs','raw_anchor_gwl','raw_target_gwl']].to_csv(OUT/'h18_top_anchor_errors.csv',index=False)

# Summarize h18 driver stations' contribution and overlap across horizons.
top_by_h={int(h):set(g.sort_values('squared_error',ascending=False).head(10).Station) for h,g in station_h.groupby('horizon_h')}
top_overlap=[]
for h in [6,12,18,24]:
    top_overlap.append(dict(horizon_h=h,top10_station_overlap_with_h18=len(top_by_h[h]&top_by_h[18]),
        n_top10=10,stations_top10='; '.join(sorted(top_by_h[h]))))
pd.DataFrame(top_overlap).to_csv(OUT/'h18_top10_station_overlap.csv',index=False)

# Report tables.
def md(df,cols=None,digits=4):
    if cols: df=df[cols]
    def c(v):
        if pd.isna(v): return ''
        if isinstance(v,(float,np.floating)): return f'{v:.{digits}f}'
        if isinstance(v,(int,np.integer)): return str(v)
        return str(v).replace('|','\\|')
    hs='| '+' | '.join(map(str,df.columns))+' |'
    sep='| '+' | '.join(['---']*len(df.columns))+' |'
    return '\n'.join([hs,sep]+['| '+' | '.join(c(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

outlier_md=outliers.sort_values(['top_xgb_30d','top_traj_dl_h720','Station'],ascending=[False,False,True])
quality=pd.DataFrame(evals)
dist=pd.DataFrame(dist_rows)
horiz_summary=pd.read_csv(OUT/'h18_horizon_observation_summary.csv')
hours=pd.read_csv(OUT/'h18_anchor_hour_summary.csv')
flag_groups=pd.read_csv(OUT/'h18_observation_flag_groups.csv')
topst=pd.read_csv(OUT/'h18_top_station_contributions.csv')
report=['# Phase 2 read-only forensics','',
    '## Scope and assumptions','',
    '- Read only `data/aligned/table_6h.parquet`, `data/meta/assoc_rainfall.csv`, saved prediction/backtest artifacts, and Phase 1 diagnostics. No training, refresh, or network fetch was run. New files are confined to `outputs/diagnostics_phase2/`. Existing changes in the source Parquet/metadata files were left untouched.',
    '- For outlier one-day jumps, both endpoints are exactly 24 hours apart and flagged `gwl_obs=1`; for 30-day change metrics both endpoints are exactly 30 days apart and observed. The observed-slot share denominator is every table slot for that station during 2021–2026.',
    '- The regime-filter result reproduces `06_features.flag_regime_shift`: median per station/year; baseline median of 2021–2024 yearly medians; fail if any available year median differs by more than 15m. A station is labeled “consistent with datum/telemetry fault” if it fails that filter or has an observed-endpoint 1-day jump >15m; other union stations are labeled “plausible hydrology”. These are evidence-based screening labels, not proof of cause.',
    '- Quality proposal uses only exact 30-day pairs ending before 2025-10-01 and observed at both endpoints. Remove if station p99 absolute 30-day change exceeds 15m. The cutoff was selected as a physical-scale screening threshold and was not tuned using any 2026 model errors.',
    '- Winsorized-RMSE skill caps both model and persistence absolute errors at the 99th percentile of their pooled absolute errors within each reported sample. This is a robustness summary, not the quality filter.',
    '- Nearest rain gauge is the minimum `dist_km` for each GWL station in `assoc_rainfall.csv`; `none` means no matched rain association within 150km. Buckets use `<25`, `[25,75)`, `[75,150]` km.',
    '- For h6/h12/h18/h24, `obs` and `persist` are saved absolute GWL levels (persistence equals the anchor level). Raw flags are matched to exact anchor and target timestamps in `table_6h.parquet`. Missing timestamp matches are reported as unknown, not observed. The regime-filter audit reproduces the existing function over all years present; it is descriptive and can use post-cutoff data, unlike the proposed quality rule.', '',
    '## 1. Outlier forensics','',
    f'Union size: {len(union)} stations ({len(top_xgb)} XGBoost 30d top stations; {len(top_traj)} trajectory DL h720 top stations).', '',
    md(outlier_md,['Station','top_xgb_30d','top_traj_dl_h720','range_min','range_max','n_1day_jumps_gt5','n_1day_jumps_gt15','observed_slot_share','max_abs_30day_change','existing_regime_shift_filter_passed','classification','evidence']), '',
    'Each station has a 2021–2026 time-series PNG. These plots show recorded values and `gwl_obs=1` slots; they can reveal abrupt offsets or sustained steps, but alone do not establish a telemetry fault or hydrological cause.', '',
    '## 2. Pre-cutoff quality-filter proposal','',
    f'Rule: station p99 of observed, exact 30-day absolute GWL change before 2025-10-01 > {P99_RULE_THRESHOLD:.0f}m. Removed {len(removed)} of {len(p99)} stations with at least one qualifying pre-cutoff pair; stations without qualifying pairs are not removed by this rule.', '',
    (md(removed_rows,['Station','pre_cutoff_p99_abs_30day_change_m','rule_threshold_m']) if len(removed_rows) else 'No stations exceeded the threshold.'), '',
    'Model skill comparison (paired rows; all stations vs retained stations):','',md(quality,['artifact','subset','n','n_stations','rmse_model','rmse_persistence','skill','mae_model','mae_persistence','mae_skill','median_station_skill','winsor_cap_abs_error_p99','winsor_rmse_model','winsor_rmse_persistence','winsor_rmse_skill']), '',
    'The station removal rule uses only pre-cutoff raw GWL history. Model errors were used only for the requested post-filter evaluation, not to choose the rule or threshold.', '',
    '## 3. Skill by distance to rain gauge','',
    md(dist,['artifact','distance_bucket','n','n_stations','rmse_model','rmse_persistence','skill','mae_model','mae_persistence','mae_skill','median_station_skill']), '',
    '## 4. h18 persistence-RMSE anomaly','',
    'The saved `obs` is an absolute target level, not a change. The h18 jump is dominated by `Primary School, Dera Devra bujurg`: its 2026-03-04 00:00 anchor is 27.500m and the observed level 18 hours later is -7.921m (35.421m change); both endpoint flags are `gwl_obs=1`. Its h18 persistence squared-error contribution is 1,255.06 of about 1,697.56 total (73.9%). Of 2,207 h18 rows, 2,148 (97.3%) have a midnight anchor; these contribute 1,690.94 squared error (99.6%). The 0.877-to-0.819 RMSE drop at h24 does not remove the anomaly: Dera Devra remains the leading h24 station with 1,232.46 squared error (83.3% of h24 total), while other stations’ combined squared error drops from about 442.5 at h18 to 247.5 at h24. Dera Devra is not a top-10 station at h12; top-10 station overlap with h18 is 3/10 at h12 and 4/10 at h24. This is a recorded abrupt level discontinuity; the file does not prove whether it was a telemetry fault or real site event.', '',
    'The horizon summary compares h6/h12/h18/h24, observed-flag rates, and whether saved absolute `obs` matches the exact raw target level. See the station contribution, anchor-hour, flag-group and top-anchor CSVs for attribution.', '',
    md(horiz_summary,['horizon_h','n','n_stations','persistence_rmse','persistence_mae','share_anchor_observed','share_target_observed','share_both_observed','artifact_obs_matches_raw_target_level']), '',
    '### Persistence error by anchor hour','',md(hours,['horizon_h','anchor_hour','n','n_stations','persistence_rmse','persistence_mae','squared_error']), '',
    '### Persistence error by exact observed flags','',md(flag_groups,['horizon_h','anchor_gwl_obs','target_gwl_obs','n','persistence_rmse','persistence_mae','n_stations']), '',
    '### Largest station contributors by horizon','',md(topst.sort_values(['horizon_h','squared_error'],ascending=[True,False]),['horizon_h','Station','n','persistence_rmse','squared_error','share_horizon_sqerr','share_anchor_observed','share_target_observed']), '',
    '## Output files','',
    '- `outlier_forensics.csv`: union station metrics, classification, and evidence.',
    '- `precutoff_quality_rule_by_station.csv`, `quality_filter_removed_stations.csv`, `quality_filter_skill_comparison.csv`.',
    '- `rain_gauge_distance_skill.csv`, `rain_gauge_distance_by_station.csv`.',
    '- `h18_horizon_observation_summary.csv`, `h18_observation_flag_groups.csv`, `h18_anchor_hour_summary.csv`, `h18_top_station_contributions.csv`, `h18_top_anchor_errors.csv`, `h18_top10_station_overlap.csv`.',
    '- PNG filenames use a readable station slug and a stable short hash.','',
    '### Station plots','']
for r in outlier_rows:
    png=f'{station_slug(r["Station"])}.png'
    report.append(f'- [{r["Station"]}]({png}) — {r["classification"]}. {r["evidence"]}')
(OUT/'phase2_report.md').write_text('\n'.join(report),encoding='utf-8')

print(f'outliers={len(union)}; quality_removed={len(removed)}; table_6h_rows={len(raw)}')
print('Wrote report and CSV/PNG outputs under',OUT)
print('quality rule scores:')
print(pd.DataFrame(evals)[['artifact','subset','n','n_stations','skill','mae_skill','median_station_skill','winsor_rmse_skill']].to_string(index=False))
print('horizons:')
print(horiz_summary.to_string(index=False))
