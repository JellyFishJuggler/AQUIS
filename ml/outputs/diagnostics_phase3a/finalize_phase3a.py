"""Validate the Phase 3a data_v2 build and mark it complete only after review.

Run without --complete first to print checks. With --complete, all checks must
pass and a reviewed visual_review.json must exist for the 29+12 contact sheets.

This script never writes ml/data, models, or refresh state.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / 'outputs' / 'diagnostics_phase3a'
OUT = ROOT / 'data_v2'
SRC = ROOT / 'data' / 'aligned' / 'table_6h.parquet'
SHEETS = DIAG / 'contact_sheets'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--complete', action='store_true')
    ap.add_argument('--review-file', default=str(DIAG / 'visual_review.json'))
    args = ap.parse_args()
    checks = []

    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
        return bool(ok)

    status = json.loads((OUT / 'BUILD_STATUS.json').read_text())
    manifest = json.loads((OUT / 'manifest.json').read_text())
    cfg = json.loads((OUT / 'cleaning_config.json').read_text())
    gate = json.loads((DIAG / 'gate_summary.json').read_text())
    purge = pd.read_csv(DIAG / 'purge_test.csv')

    check('build status is awaiting_visual_review', status.get('status') == 'awaiting_visual_review', status.get('status'))
    check('build not already marked complete', status.get('build_complete') is False)
    check('manifest not already marked complete', manifest.get('build_complete') is False)
    check('gate passed', bool(gate.get('gate_pass')) and all(r['status'] == 'PASS' for r in gate['named_gate']))
    check('purge passed', bool((purge['pass'] == True).all()), f"{int((purge['pass'] != True).sum())} failures")
    check('manifest purge flag', manifest.get('purge_pass') is True)

    expected = [
        OUT / 'aligned' / 'table_6h.parquet',
        OUT / 'features_6h' / 'train.parquet', OUT / 'features_6h' / 'val.parquet',
        OUT / 'features_6h' / 'test.parquet',
        OUT / 'features_traj' / 'train.parquet', OUT / 'features_traj' / 'val.parquet',
        OUT / 'features_traj' / 'test.parquet', OUT / 'manifest.json',
        OUT / 'cleaning_config.json', OUT / 'BUILD_STATUS.json',
    ]
    for p in expected:
        check(f'exists: {p.relative_to(ROOT)}', p.exists() and p.stat().st_size > 0,
              p.stat().st_size if p.exists() else 'missing')

    check('source hash unchanged', sha256(SRC) == manifest.get('source_sha256'),
          manifest.get('source_sha256', ''))

    aligned = pd.read_parquet(OUT / 'aligned' / 'table_6h.parquet')
    src = pd.read_parquet(SRC)
    src['time'] = pd.to_datetime(src.time); aligned['time'] = pd.to_datetime(aligned.time)
    src = src.sort_values(['Station', 'time']).reset_index(drop=True)
    aligned = aligned.sort_values(['Station', 'time']).reset_index(drop=True)
    check('aligned row count', len(aligned) == len(src) == manifest.get('source_rows'),
          f'{len(aligned)} vs {len(src)} vs {manifest.get("source_rows")}')
    check('raw source preserved in gwl_raw', bool(np.allclose(aligned.gwl_raw.to_numpy(float), src.gwl.to_numpy(float), equal_nan=True)))
    m1 = aligned.m1_mask.to_numpy(bool); m2 = aligned.m2_mask.to_numpy(bool)
    check('M1/M2 counts match manifest', int(m1.sum()) == manifest['masked_M1'] and int(m2.sum()) == manifest['masked_M2'],
          f'{int(m1.sum())}/{int(m2.sum())}')
    check('masked rows are null in cleaned gwl', bool(aligned.loc[m1 | m2, 'gwl'].isna().all()))
    unmasked = ~(m1 | m2)
    check('unmasked cleaned gwl equals source', bool(np.allclose(aligned.loc[unmasked, 'gwl'].to_numpy(float), src.loc[unmasked, 'gwl'].to_numpy(float), equal_nan=True)))

    f6 = {s: pd.read_parquet(OUT / 'features_6h' / f'{s}.parquet', columns=['target_time', 'time']) for s in ['train', 'val', 'test']}
    check('features_6h rows match manifest', {s: len(v) for s, v in f6.items()} == manifest['split_rows'],
          f"{ {s: len(v) for s, v in f6.items()} } vs {manifest['split_rows']}")
    for s, v in f6.items():
        check(f'features_6h {s} targets', bool((pd.to_datetime(v.target_time) > pd.to_datetime(v.time)).all()))
    ft = {s: pd.read_parquet(OUT / 'features_traj' / f'{s}.parquet', columns=['target_time', 'time']) for s in ['train', 'val', 'test']}
    check('features_traj rows match manifest', {s: len(v) for s, v in ft.items()} ==
          {s: manifest['trajectory_rows'][s]['after'] for s in ['train', 'val', 'test']},
          f"{ {s: len(v) for s, v in ft.items()} }")
    for s, v in ft.items():
        check(f'features_traj {s} targets', bool((pd.to_datetime(v.target_time) > pd.to_datetime(v.time)).all()))

    sheets = sorted(SHEETS.glob('phase3a_contact_sheet_*.png')) if SHEETS.exists() else []
    check('contact sheet count is 7 (29+12 stations, 6 per sheet)', len(sheets) == 7, len(sheets))
    check('contact sheets non-empty', all(p.stat().st_size > 100_000 for p in sheets),
          [p.stat().st_size for p in sheets])
    sample = pd.read_csv(DIAG / 'visual_sample_stations.csv')
    check('visual sample is 29+12=41 unique', len(sample) == 41 and sample.Station.nunique() == 41,
          sample.sample_group.value_counts().to_dict() if 'sample_group' in sample else len(sample))

    review_ok = False
    review = {}
    rp = Path(args.review_file)
    if rp.exists():
        review = json.loads(rp.read_text())
        review_ok = (review.get('decision') == 'accept' and len(review.get('sheets', [])) == 7
                     and all((SHEETS / s).exists() for s in review.get('sheets', [])))
    check('visual review file accepted for all 7 sheets', review_ok, str(rp))

    out = {'checked_at': datetime.now(timezone.utc).isoformat(), 'checks': checks,
           'all_pass': all(c['pass'] for c in checks)}
    (DIAG / 'finalize_checks.json').write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(checks).to_string(index=False))
    print(f"\nALL CHECKS PASS: {out['all_pass']}")
    if not out['all_pass']:
        return 1

    if not args.complete:
        print('Dry run only. Re-run with --complete after the visual review file is accepted.')
        return 0

    now = datetime.now(timezone.utc).isoformat()
    manifest.update({'build_complete': True, 'status': 'complete', 'visual_review': 'complete',
                     'finalized_at': now, 'visual_review_file': str(rp.relative_to(ROOT)),
                     'contact_sheets': [str(p.relative_to(ROOT)) for p in sheets]})
    status.update({'build_complete': True, 'status': 'complete', 'visual_review': 'complete',
                   'finalized_at': now})
    cfg.update({'build_complete': True, 'status': 'complete', 'visual_review': 'complete',
                'finalized_at': now})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    (DIAG / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    (OUT / 'BUILD_STATUS.json').write_text(json.dumps(status, indent=2, default=str))
    (OUT / 'cleaning_config.json').write_text(json.dumps(cfg, indent=2))
    (OUT / 'BUILD_COMPLETE.txt').write_text(
        'PHASE 3a BUILD COMPLETE — VALID.\n'
        f"build_id={manifest['build_id']}\n"
        f"gate_pass=true purge_pass=true visual_review=complete\n"
        f"contact_sheets={len(sheets)}\n"
        f"finalized_at={now}\n"
        'See manifest.json and outputs/diagnostics_phase3a/phase3a_gate_report.md.\n')
    old = OUT / 'BUILD_INCOMPLETE_DO_NOT_USE.txt'
    if old.exists():
        dest = DIAG / 'superseded_build_markers'
        dest.mkdir(exist_ok=True)
        shutil.move(str(old), str(dest / f'BUILD_INCOMPLETE_DO_NOT_USE_{manifest["build_id"]}.txt'))
    print('BUILD MARKED COMPLETE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
