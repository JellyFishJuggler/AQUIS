"""Archive the stale Phase 3a data_v2 artifacts without deleting them.

Writes an exact inventory (path, bytes, sha256, mtime) and moves the files to
outputs/diagnostics_phase3a/stale_data_v2_20261026/, preserving relative paths.
"""
from __future__ import annotations
import csv, hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data_v2'
DST = Path(__file__).resolve().parent / 'stale_data_v2_20261026'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


files = sorted(p for p in SRC.rglob('*') if p.is_file())
rows = []
for p in files:
    st = p.stat()
    rows.append({
        'relative_path': str(p.relative_to(SRC)).replace('\\', '/'),
        'bytes': st.st_size,
        'sha256': sha256(p),
        'mtime_utc': datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    })

(DST).mkdir(parents=True, exist_ok=True)
with (DST / 'inventory.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['relative_path', 'bytes', 'sha256', 'mtime_utc'])
    w.writeheader()
    w.writerows(rows)

for p in files:
    rel = p.relative_to(SRC)
    dest = DST / 'files' / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(dest))

# remove now-empty directories, deepest first (not a data deletion)
for d in sorted((p for p in SRC.rglob('*') if p.is_dir()), key=lambda x: len(x.parts), reverse=True):
    try:
        d.rmdir()
    except OSError:
        pass

(DST / 'README.md').write_text(
    '# Stale data_v2 archive (Phase 3a)\n\n'
    'These files were the stale/partial `ml/data_v2` artifacts explicitly marked\n'
    'DO NOT USE before the 2026-10-26 Phase 3a rebuild. They were moved here\n'
    '(not deleted) so the old state is auditable. `inventory.csv` records the exact\n'
    'relative path, byte size, SHA-256, and mtime of every file before the move.\n\n'
    'Do not consume these files. The rebuilt dataset lives in `ml/data_v2` and is\n'
    'valid only when its `BUILD_STATUS.json` / `manifest.json` say build_complete.\n',
    encoding='utf-8')

print(f'archived {len(rows)} files')
print(f'inventory: {DST / "inventory.csv"}')
for r in rows:
    print(f"  {r['relative_path']:55s} {r['bytes']:>10d} {r['sha256'][:16]}")
