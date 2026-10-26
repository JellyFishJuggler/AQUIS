"""Inspect Block Office, Kurara M2 rows and detector evidence (Phase 3a).

Read-only. Does not write data_v2.
"""
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
STATION = 'Block Office, Kurara'

df = pd.read_parquet(SRC)
df['time'] = pd.to_datetime(df.time)
df = df.sort_values(['Station', 'time']).reset_index(drop=True)
g = df[df.Station == STATION].reset_index(drop=True)
print('station rows', len(g), 'range', g.time.min(), g.time.max())
print('gwl finite', int(np.isfinite(g.gwl).sum()), 'unique', g.gwl.nunique())

z = B.detect_station(g)
v = g.gwl.to_numpy(float)
t = g.time.to_numpy(dtype='datetime64[ns]')
m1, m2 = z['m1'], z['m2']
print('\nm1', int(m1.sum()), 'm2', int(m2.sum()))
print('m1_info', z['m1_info'])
print('pre_median', z['robust_median_pre'], 'pre_scale', z['robust_scale_pre'])
print('repeat_values', z['repeat_values'])

# Reproduce repeated-exact path components
ti = pd.DatetimeIndex(t)
base = (ti < B.PRE_END) & np.isfinite(v) & (~m1)
pre_n = int(np.searchsorted(t, B.PRE_END.to_datetime64(), side='left'))
for a, b in B.intervals(np.isfinite(v[:pre_n])):
    if b - a + 1 >= 28:
        base[a:b + 1] = False
medp, scalep = B.robust_pre(v[base], ti[base]) if base.sum() else (np.nan, np.nan)
lo, hi = medp - 8 * scalep, medp + 8 * scalep
prevals = v[base]
vals, cnts = np.unique(prevals, return_counts=True)
rep = set(float(x) for x, c in zip(vals, cnts) if c >= 3 and (x < lo or x > hi or x in B.SENTINELS))
print('base n', int(base.sum()), 'medp/scalep', medp, scalep, 'envelope', lo, hi)
print('rep n', len(rep))
print('rep values', sorted(rep))

rows = []
for i in range(1, len(v) - 1):
    if v[i] in rep and np.isfinite(v[i-1]) and np.isfinite(v[i+1]) and v[i-1] == v[i+1] \
       and t[i]-t[i-1] == np.timedelta64(6, 'h') and t[i+1]-t[i] == np.timedelta64(6, 'h'):
        rows.append(i)

print('\nrepeated-exact detections', len(rows))
outrows = []
for i in rows:
    outrows.append({
        'time': pd.Timestamp(t[i]), 'value': v[i], 'prev': v[i-1], 'next': v[i+1],
        'pre_count_of_value': int((prevals == v[i]).sum()),
        'in_m2': bool(m2[i]), 'in_m1': bool(m1[i]),
    })
out = pd.DataFrame(outrows)
if len(out):
    print(out.to_string(index=False))
    print('\ncount by year', out.time.dt.year.value_counts().sort_index().to_dict())
    print('value counts', out.value.value_counts().to_dict())

# All m2 rows with context +/-3 slots
print('\nAll M2 rows with context:')
for i in np.flatnonzero(m2):
    a, b = max(0, i - 3), min(len(v), i + 4)
    print('\n---', pd.Timestamp(t[i]), 'value', v[i])
    print(pd.DataFrame({'time': pd.DatetimeIndex(t[a:b]), 'gwl': v[a:b],
                        'm1': m1[a:b], 'm2': m2[a:b]}).to_string(index=False))

# Distribution / repeated values across whole series
print('\nWhole-series value counts for detected values:')
for val in sorted(set(v[i] for i in rows)):
    ii = np.flatnonzero(v == val)
    print(val, 'n=', len(ii), 'first', pd.Timestamp(t[ii[0]]), 'last', pd.Timestamp(t[ii[-1]]),
          'years', sorted(set(pd.DatetimeIndex(t[ii]).year)))

# Largest gaps / cadence
print('\nlast 30 rows:')
print(g.tail(30).to_string(index=False))
