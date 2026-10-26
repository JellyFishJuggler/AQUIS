"""Compare M2 repeated-exact detector variants across all stations."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

DIAG = Path(__file__).resolve().parent
ROOT = DIAG.parents[1]
sys.path.insert(0, str(ROOT))
import build_phase3a as B

SRC = ROOT / 'data' / 'aligned' / 'table_6h.parquet'
df = pd.read_parquet(SRC)
df['time'] = pd.to_datetime(df.time)
df = df.sort_values(['Station', 'time']).reset_index(drop=True)

records = []
diff_rows = []
named_rows = []
for st, idx in df.groupby('Station', sort=False).groups.items():
    g = df.loc[list(idx)].reset_index(drop=True)
    z_old = B.detect_station(g, repeated_requires_departure=False, repeated_require_observed=False)
    z_dep = B.detect_station(g, repeated_requires_departure=True, repeated_require_observed=False)
    z_obs = B.detect_station(g, repeated_requires_departure=True, repeated_require_observed=True)
    rec = {
        'Station': st,
        'm1_rows': int(z_obs['m1'].sum()),
        'm2_old': int(z_old['m2'].sum()),
        'm2_departure_only': int(z_dep['m2'].sum()),
        'm2_departure_observed': int(z_obs['m2'].sum()),
        'removed_by_departure': int((z_old['m2'] & ~z_dep['m2']).sum()),
        'removed_by_observed': int((z_dep['m2'] & ~z_obs['m2']).sum()),
        'stuck_rows': int(z_obs['stuck'].sum()),
        'flip_rows': int(z_obs['flip'].sum()),
        'extreme_burst_rows': int(z_obs['extreme'].sum()),
        'early_era_unverified_rows': int(z_obs['early_suspect'].sum()),
        'break_count': len(z_obs['breaks']),
        'pre_median': z_obs['robust_median_pre'],
        'pre_scale': z_obs['robust_scale_pre'],
        'repeat_values': json.dumps(z_obs['repeat_values']),
    }
    records.append(rec)

    for i in np.flatnonzero(z_old['m2'] & ~z_obs['m2']):
        lo, hi = max(0, i - 6), min(len(g), i + 7)
        diff_rows.append({
            'Station': st, 'time': g.time.iloc[i], 'value': g.gwl.iloc[i],
            'gwl_obs': g.gwl_obs.iloc[i] if 'gwl_obs' in g else np.nan,
            'prev_obs': g.gwl_obs.iloc[i - 1] if i > 0 and 'gwl_obs' in g else np.nan,
            'next_obs': g.gwl_obs.iloc[i + 1] if i + 1 < len(g) and 'gwl_obs' in g else np.nan,
            'prev': g.gwl.iloc[i - 1] if i > 0 else np.nan,
            'next': g.gwl.iloc[i + 1] if i + 1 < len(g) else np.nan,
            'run_context': '|'.join(f'{x:.3f}' for x in g.gwl.iloc[lo:hi].to_numpy()),
        })

    if st in set(B.MUST_FIX + B.CONTROLS + [B.RESIDUAL]):
        for i in np.flatnonzero(z_obs['m2']):
            named_rows.append({
                'Station': st, 'time': g.time.iloc[i], 'gwl': g.gwl.iloc[i],
                'gwl_obs': g.gwl_obs.iloc[i] if 'gwl_obs' in g else np.nan,
                'prev': g.gwl.iloc[i - 1] if i > 0 else np.nan,
                'next': g.gwl.iloc[i + 1] if i + 1 < len(g) else np.nan,
            })

summary = pd.DataFrame(records).set_index('Station')
summary.to_csv(DIAG / 'detector_old_vs_new_summary.csv')

diff = pd.DataFrame(diff_rows)
diff.to_csv(DIAG / 'm2_removed_by_detector_fix.csv', index=False)
pd.DataFrame(named_rows).to_csv(DIAG / 'named_station_m2_rows_corrected.csv', index=False)

print('=== totals ===')
for c in ['m2_old', 'm2_departure_only', 'm2_departure_observed']:
    print(f'{c:28s} {int(summary[c].sum()):6d}')
print('removed_by_departure', int(summary.removed_by_departure.sum()))
print('removed_by_observed ', int(summary.removed_by_observed.sum()))
print('\n=== removed rows by gwl_obs status ===')
if len(diff):
    print(diff.groupby(['gwl_obs', 'prev_obs', 'next_obs'], dropna=False).size().rename('rows').to_string())
    print('\nremoved by station (top 20):')
    print(diff.groupby('Station').size().sort_values(ascending=False).head(20).to_string())
else:
    print('none')

print('\n=== named gate under corrected detector (departure + observed) ===')
for s in B.MUST_FIX + B.CONTROLS + [B.RESIDUAL]:
    if s in summary.index:
        r = summary.loc[s]
        print(f"{s!r:55s} m1={int(r.m1_rows):4d} m2_old={int(r.m2_old):4d} "
              f"m2_dep={int(r.m2_departure_only):4d} m2_final={int(r.m2_departure_observed):4d}")
    else:
        print('MISSING', s)

print('\n=== retained M2 rows at named stations ===')
nr = pd.DataFrame(named_rows)
if len(nr):
    print(nr.groupby('Station').size().to_string())
    print('\nobs status of retained rows:')
    print(nr.groupby(['Station', 'gwl_obs']).size().to_string())
else:
    print('none')
