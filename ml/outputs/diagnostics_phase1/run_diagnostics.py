from pathlib import Path
import json
import math
import os
import sys

ROOT = Path('/mnt/c/Users/Srijan Gupta/Downloads/Developement/AQUIS/ml')
sys.path.insert(0, str(ROOT / 'venv312/lib/python3.12/site-packages'))
os.chdir(ROOT)
import numpy as np
import pandas as pd

OUT = ROOT / 'outputs/diagnostics_phase1'
OUT.mkdir(parents=True, exist_ok=True)

pred = pd.read_parquet(ROOT / 'outputs/predictions_2026.parquet')
traj = pd.read_parquet(ROOT / 'outputs/traj_backtest_rows_dl.parquet')
pool = json.loads((ROOT / 'outputs/pooled_dl_eval.json').read_text())
features = pd.read_parquet(ROOT / 'data/features/test.parquet', columns=['Station','District','rain_obs7','rain_obs22','rain_obs60'])
driver = features.groupby('Station', dropna=False).agg(
    rain_obs7_max=('rain_obs7','max'), rain_obs22_max=('rain_obs22','max'), rain_obs60_max=('rain_obs60','max')
).reset_index()
driver['linked_driver'] = driver[['rain_obs7_max','rain_obs22_max','rain_obs60_max']].fillna(0).gt(0).any(axis=1)
station_district = features[['Station','District']].drop_duplicates('Station')

def season_of_month(month):
    if month in (3,4,5): return 'pre-monsoon'
    if month in (6,7,8,9): return 'monsoon'
    return 'post-monsoon'

def metric_table(df, model_col, base_col, keys, source, label):
    rows=[]
    for group_key, g in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple): group_key=(group_key,)
        good=g[[model_col,base_col]].notna().all(axis=1)
        g=g.loc[good]
        if len(g)==0: continue
        em=(g[model_col]-g['_obs']).astype(float)
        eb=(g[base_col]-g['_obs']).astype(float)
        rmse=float(np.sqrt(np.mean(em**2))); brmse=float(np.sqrt(np.mean(eb**2)))
        row=dict(zip(keys,group_key)); row.update(source=source,model=label,n=len(g),n_stations=g['_station'].nunique(),
            rmse=rmse,persistence_rmse=brmse,skill_score=(1-rmse/brmse if brmse else np.nan),
            mae=float(np.mean(np.abs(em))),median_abs_error=float(np.median(np.abs(em))))
        rows.append(row)
    return rows

def prepare_pred():
    d=pred.copy(); d['_obs']=d['target']; d['_station']=d['Station']
    d['_target_time']=pd.to_datetime(d['time'])+pd.to_timedelta(d['horizon'],unit='D')
    d['season']=d['_target_time'].dt.month.map(season_of_month)
    d=d.merge(driver[['Station','linked_driver']],on='Station',how='left')
    d['linked_driver']=d['linked_driver'].fillna(False)
    d['model_xgb']=d['xgb']; d['model_ridge']=d['ridge']; d['base']=d['persist']
    return d

def prepare_traj():
    d=traj.copy(); d['_obs']=d['obs']; d['_station']=d['station']
    d['target_time']=pd.to_datetime(d['anchor_t'])+pd.to_timedelta(d['horizon_h'],unit='h')
    d['season']=d['target_time'].dt.month.map(season_of_month)
    d=d.merge(station_district.rename(columns={'Station':'station','District':'district'}),on='station',how='left')
    d=d.merge(driver[['Station','linked_driver']].rename(columns={'Station':'station'}),on='station',how='left')
    d['linked_driver']=d['linked_driver'].fillna(False)
    d['model_traj_dl']=d['traj_q50']; d['base']=d['persist']
    return d

def station_summary(df, model_col, base_col, source, model, group_keys):
    out=[]
    keys=['_station']+group_keys
    for group_key,g in df.groupby(keys,dropna=False):
        if not isinstance(group_key,tuple): group_key=(group_key,)
        station=group_key[0]
        g=g[[model_col,base_col,'_obs']].dropna()
        if not len(g): continue
        e=(g[model_col]-g['_obs']).astype(float); b=(g[base_col]-g['_obs']).astype(float)
        rmse=float(np.sqrt(np.mean(e**2))); brmse=float(np.sqrt(np.mean(b**2)))
        row=dict(source=source,model=model,station=station,n=len(g),rmse=rmse,mae=float(np.mean(np.abs(e))),
            median_abs_error=float(np.median(np.abs(e))),persistence_rmse=brmse,skill_score=1-rmse/brmse if brmse else np.nan,
            squared_error=float(np.sum(e**2)))
        row.update(dict(zip(group_keys,group_key[1:])))
        out.append(row)
    return out

all_metrics=[]; all_stations=[]; contributions=[]
pdframe=prepare_pred()
for col,name in [('model_xgb','XGBoost'),('model_ridge','Ridge')]:
    all_metrics += metric_table(pdframe,col,'base',['horizon'],'predictions_2026',name)
    all_metrics += metric_table(pdframe,col,'base',['season'],'predictions_2026',name)
    all_metrics += metric_table(pdframe,col,'base',['District'],'predictions_2026',name)
    all_metrics += metric_table(pdframe,col,'base',['linked_driver'],'predictions_2026',name)
    ss=station_summary(pdframe,col,'base','predictions_2026',name,['horizon']); all_stations += ss
    # Top 20 per model by sum of squared model errors across all prediction rows.
    contributions += sorted([dict(source='predictions_2026',model=name,station=r['station'],n=r['n'],
        squared_error=r['squared_error'],share_of_model_squared_error=np.nan) for r in ss],key=lambda x:x['squared_error'],reverse=True)[:20]

tdf=prepare_traj()
for col,name in [('model_traj_dl','Trajectory DL q50')]:
    all_metrics += metric_table(tdf,col,'base',['horizon_h'],'traj_backtest_rows_dl',name)
    all_metrics += metric_table(tdf,col,'base',['season'],'traj_backtest_rows_dl',name)
    all_metrics += metric_table(tdf,col,'base',['district'],'traj_backtest_rows_dl',name)
    all_metrics += metric_table(tdf,col,'base',['linked_driver'],'traj_backtest_rows_dl',name)
    ss=station_summary(tdf,col,'base','traj_backtest_rows_dl',name,['horizon_h']); all_stations += ss
    ss_all=station_summary(tdf,col,'base','traj_backtest_rows_dl',name,[])
    contributions += sorted([dict(source='traj_backtest_rows_dl',model=name,station=r['station'],n=r['n'],
        squared_error=r['squared_error'],share_of_model_squared_error=np.nan) for r in ss_all],key=lambda x:x['squared_error'],reverse=True)[:20]

# The pooled DL JSON stores aggregate metrics only; row-level predictions are not present in the requested artifacts.
pp=pool['persistence']; dl=pool['DL']
pooled_skill=1-dl['rmse']/pp['rmse']
aggregate=pd.DataFrame([dict(source='pooled_dl_eval.json',model='Pooled daily DL',n=dl['n'],n_stations=dl['n_stations'],
    rmse=dl['rmse'],persistence_rmse=pp['rmse'],skill_score=pooled_skill,mae=dl['mae'],median_abs_error=np.nan,
    note='Aggregate values copied from saved JSON; row level outcomes are absent.')])
pd.DataFrame(all_metrics).to_csv(OUT/'group_metrics.csv',index=False)
pd.DataFrame(all_stations).to_csv(OUT/'station_metrics.csv',index=False)
pd.DataFrame(contributions).to_csv(OUT/'top20_squared_error_contribution.csv',index=False)
aggregate.to_csv(OUT/'pooled_dl_aggregate.csv',index=False)

def md_table(df, cols=None, digits=4):
    if cols: df=df[cols]
    def cell(v):
        if pd.isna(v): return ''
        if isinstance(v,(float,np.floating)): return f'{v:.{digits}f}'
        return str(v).replace('|','\\|')
    headers=list(df.columns)
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines += ['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)

station_df=pd.DataFrame(all_stations)
dist=[]
dimension_cols=[c for c in ['horizon','horizon_h'] if c in station_df.columns]
for group_key,g in station_df.groupby(['source','model']+dimension_cols,dropna=False):
    if not isinstance(group_key,tuple): group_key=(group_key,)
    src,model,*dimension_values=group_key
    row=dict(source=src,model=model,stations=len(g),median_station_skill=g.skill_score.median(),
        p10_station_skill=g.skill_score.quantile(.1),p90_station_skill=g.skill_score.quantile(.9),
        median_station_rmse=g.rmse.median(),p10_station_rmse=g.rmse.quantile(.1),p90_station_rmse=g.rmse.quantile(.9),
        median_station_mae=g.mae.median(),median_station_median_abs_error=g.median_abs_error.median())
    row.update(dict(zip(dimension_cols,dimension_values)))
    dist.append(row)
pd.DataFrame(dist).to_csv(OUT/'station_skill_distribution.csv',index=False)
pd.DataFrame(dist).to_csv(OUT/'station_error_distribution.csv',index=False)

# Add each model's share of total squared error to its top 20.
tops=pd.DataFrame(contributions)
for (src,model),idx in tops.groupby(['source','model']).groups.items():
    complete=station_df[(station_df.source==src)&(station_df.model==model)]
    total=complete.squared_error.sum()
    tops.loc[idx,'share_of_model_squared_error']=tops.loc[idx,'squared_error']/total if total else np.nan
tops.to_csv(OUT/'top20_squared_error_contribution.csv',index=False)

# Quantify concentration among top 20 stations per model.
concentration=tops.groupby(['source','model']).agg(top20_squared_error=('squared_error','sum')).reset_index()
concentration['total_squared_error']=concentration.apply(lambda r: station_df[(station_df.source==r.source)&(station_df.model==r.model)].squared_error.sum(),axis=1)
concentration['top20_share']=concentration.top20_squared_error/concentration.total_squared_error
concentration.to_csv(OUT/'error_concentration.csv',index=False)

metrics=pd.DataFrame(all_metrics)
distdf=pd.DataFrame(dist)
report=[]
report += ['# ML diagnostics: Phase 1','',
    'Scope: read-only analysis of the saved artifacts named in the request. No model was retrained and no files under `models/`, `outputs/` outside this directory, or `data/refresh/` were written. New tables are in this directory.','',
    '## Assumptions and limits','',
    '- Error is prediction minus observation; RMSE, MAE and median absolute error use paired non-null rows. Skill is `1 - RMSE(model)/RMSE(persistence)` on the same rows. Positive skill means lower RMSE than persistence.',
    '- Seasons are assigned by forecast target time: `time + horizon days` for `predictions_2026.parquet` and `anchor_t + horizon_h` for the trajectory artifact. Pre-monsoon = March-May, monsoon = June-September, post-monsoon = October-February.',
    '- Driver linkage is station-level: a station has a linked driver when any saved 2026 test feature has `rain_obs7`, `rain_obs22`, or `rain_obs60` greater than zero. These columns represent observed rain-gauge coverage. This is an empirical test-frame proxy, not an independent station-to-gauge registry.',
    '- `pooled_dl_eval.json` contains aggregate metrics only. The requested artifact set has no row-level pooled DL predictions, so its season, district, station distribution, median-error distribution, driver split, and top contributors cannot be computed from that JSON.','',
    '## Aggregate results','',md_table(aggregate,['source','model','n','n_stations','rmse','persistence_rmse','skill_score','mae','median_abs_error']), '',
    'The `group_metrics.csv` table contains per-horizon, per-season, per-district and driver-group results available from row artifacts.','',
    '### Horizon breakdown: 30-day level predictions (days)','',md_table(metrics[metrics.horizon.notna()],['source','model','horizon','n','rmse','persistence_rmse','skill_score','mae','median_abs_error']), '',
    '### Horizon breakdown: trajectory backtest (hours)','',md_table(metrics[metrics.horizon_h.notna()],['source','model','horizon_h','n','rmse','persistence_rmse','skill_score','mae','median_abs_error']), '',
    '### Seasonal results','',md_table(metrics[metrics.season.notna()],['source','model','season','n','rmse','persistence_rmse','skill_score','mae','median_abs_error']), '',
    '### Per-station skill and error distribution','',md_table(distdf,['source','model','horizon','horizon_h','stations','median_station_skill','p10_station_skill','p90_station_skill','median_station_rmse','p10_station_rmse','p90_station_rmse','median_station_mae','median_station_median_abs_error']), '',
    'The station-level CSV gives RMSE, MAE and median absolute error for each station. The distribution table reports medians and skill percentiles across stations.', '',
    '### Per-district results','',
    md_table(metrics.loc[metrics.District.notna(),['source','model','District','n','n_stations','rmse','persistence_rmse','skill_score','mae','median_abs_error']]) if metrics.District.notna().any() else 'No district rows in the available artifacts.', '',
    md_table(metrics.loc[metrics.district.notna(),['source','model','district','n','n_stations','rmse','persistence_rmse','skill_score','mae','median_abs_error']]) if metrics.district.notna().any() else '', '',
    '### Driver-group comparison','',md_table(metrics[metrics.linked_driver.notna()],['source','model','linked_driver','n','n_stations','rmse','persistence_rmse','skill_score','mae','median_abs_error']), '',
    'Interpretation: XGBoost has small positive skill in both groups (2.7% with no linked rain driver; 2.6% with one). Trajectory DL is also positive in both groups (4.6% and 3.9%). The gain is therefore not confined to stations with observed rain-gauge coverage. Ridge gains only in the no-driver group and is worse than persistence in the linked-driver group. Pooled daily DL cannot be split from the aggregate JSON.', '',
    '### Top stations by squared error','',md_table(tops.sort_values(['source','model','squared_error'],ascending=[True,True,False]),['source','model','station','n','squared_error','share_of_model_squared_error']), '',
    '### RMSE concentration','',md_table(concentration,['source','model','top20_squared_error','total_squared_error','top20_share']), '',
    'Interpretation: a small number of stations dominate squared error: the top 20 contribute 76.0% for XGBoost, 61.6% for Ridge and 88.0% for trajectory DL. `Primary School chandsina` alone contributes 33.5% of XGBoost squared error (25.8% for Ridge); `Primary School, Dera Devra bujurg` contributes 67.4% for trajectory DL. This concentration is compatible with station-specific outliers, but the saved evaluation rows do not establish whether datum errors or heavy pumping caused them.', '',
    '## Artifact purge audit','',
    '| Artifact | Artifact date | Effective input lineage | Purged? | Evidence |','|---|---|---|---|---|',
    '| `outputs/predictions_2026.parquet` | 2026-10-07 21:16 | `data/features/test.parquet`; `prep.json` built 2026-10-07, before purge commit `84935e8` (2026-10-08) | No | Both prediction artifact and feature prep predate commit. |',
    '| `outputs/pooled_dl_eval.json` | 2026-10-11 20:31 | pooled model trained 2026-10-11, but `data/features` prep is 2026-10-07 | No (effective input) | Output is post-commit; upstream saved feature frame is pre-commit. File timestamp alone is insufficient. |',
    '| `outputs/traj_backtest_rows_dl.parquet` | 2026-10-11 19:37 | `data/features_traj` prep built 2026-10-07 | No (effective input) | Output/model is post-commit; saved trajectory feature prep predates commit. |','',
    '## Persistence RMSE values found in the module','',
    '| RMSE | Protocol / sample | Interpretation |','|---:|---|---|',
    '| 1.907 | 30-day (`h720`) trajectory backtest; 19,406 anchors / 150 stations overall, 2,023 paired h720 rows | Reported in README trajectory results/model comparison and ML_MODULE_GUIDE; saved XGB and DL trajectory summaries use this baseline. Current `traj_backtest_rows_dl.parquet` reproduces 1.9071. |',
    '| 2.187 | Pooled daily GRU evaluation, 4,355 rows, 511 stations | Reported in `pooled_dl_eval.json`, README algorithm comparison and ML_MODULE_GUIDE. It is the saved JSON aggregate, not the 6-hourly model metric. |',
    '| 2.290 | Historical 2026 test on the 6-hour grid; README pairs it with XGBoost RMSE 2.242, Ridge 2.456 and climatology 3.572 | Repeated in README’s 2026 test and algorithm comparison. README says the 6-hour grid lowered persistence from 2.395 to 2.290. Pre-purge; current `predictions_2026.parquet` recomputes to 2.3209. |',
    '| 2.382 | Historical stride evaluation: one 30-day endpoint per station every 120 six-hour steps; README pairs it with model RMSE 2.339 | Reported in README stride-evaluation paragraph; values predate target-time purge. The requested artifacts do not include its row-level evaluation file. |','',
    '## `st_id` / `dist_id` feature audit','',
    '| Script | Uses station/district ID as model feature? | Detail |','|---|---|---|',
    '| `06_features.py` | Creates and carries both | `st_id` and `dist_id` are ordinal metadata columns; not part of `NUM_COLS`. |',
    '| `06_train.py` | XGBoost: yes; Ridge: no | XGBoost feature columns append both IDs. Ridge one-hot encodes original `Station`/`District` strings instead. |',
    '| `31_train_traj.py` | No | Feature list is numeric predictors plus horizon encodings. |',
    '| `34_train_traj_dl.py` | No | Model input is `NUM_COLS`; station identity is used for sequence grouping only. |',
    '| `36_train_pooled_dl.py` | No | Dense/seasonal columns are model inputs; `st_id` is used for run boundaries/grouping, not as an input. `dist_id` is not used. |','',
    '## Files','',
    '- `group_metrics.csv`: available metrics by horizon, season, district and linked-driver group.',
    '- `station_metrics.csv`: paired station-level RMSE, MAE, median absolute error and skill.',
    '- `station_skill_distribution.csv`: median/p10/p90 station skill and RMSE.',
    '- `top20_squared_error_contribution.csv`: top 20 stations per row-level model/artifact, including contribution share.',
    '- `error_concentration.csv`: total squared error and top-20 share.',
    '- `pooled_dl_aggregate.csv`: the JSON-only pooled-DL aggregate result.','']
(OUT/'phase1_report.md').write_text('\n'.join(report),encoding='utf-8')
print('Wrote diagnostics to',OUT)
print('prediction rows',len(pred),'trajectory rows',len(traj),'driver stations',int(driver.linked_driver.sum()),'/',len(driver))
print(aggregate.to_string(index=False))
