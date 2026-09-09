"""13_refresh_nwic — incremental refresh of the GWL telemetry archive + optional
retrain, plus deployed-data sync (coverage check / staged parquet install).

The model's reference archive (``config.GWL_PARQUET``, i.e.
``data/processed/common.parquet``) is built from the NWIC archive (2021-2025).
The live 2026 resource exposes the *latest* observed groundwater telemetry.
This script pulls the newest records for EVERY AQUIS district (generalized — no
hard-coded district), merges them (deduped, chronologically sorted) into the
reference parquet, and optionally retrains the pooled model so it consumes the
refreshed data.

It is *incremental*: it only fetches rows newer than the current per-district
max timestamp already present in the parquet, so repeated runs are cheap.

Modes:
    --dry-run      report how many new rows would be added, write nothing.
    --retrain      after merging, re-run the downstream chain
                   (gwl extract -> normalize -> align -> features -> train).
    --deploy-check only report coverage of the loaded parquet (no writes).
    --deploy-install PATH
                   backup current parquet and swap in a staged candidate.

Usage (from the ml/ directory):
    venv/bin/python -u 13_refresh_nwic.py [--dry-run] [--retrain] [--workers N]
    venv/bin/python -u 13_refresh_nwic.py --deploy-check
    venv/bin/python -u 13_refresh_nwic.py --deploy-install /path/to/common.parquet
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from ml.data.raw import nwic  # noqa: E402 - needs sys.path set above

import config  # noqa: E402

LIVE_RESOURCE = nwic.RES_GROUNDWATER_LIVE
DEFAULT_PQ = config.GWL_PARQUET
RETRAIN_CHAIN = [
    "02_fetch_selected.py",   # regenerates gwl_selected.csv from refreshed parquet
    "03_merge_normalize.py",
    "04_align.py",
    "04b_align_6h.py",
    "06_features.py",
    "06_train.py",
]


def _coerce_to_ref(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Cast incoming columns to the reference parquet's dtypes.

    The live NWIC API returns numeric-looking strings for e.g. ``SlNo`` while the
    archive stores integers; without this, ``pd.concat`` produces an object column
    and ``to_parquet`` raises ArrowInvalid. Non-numeric leftovers stay as strings.
    """
    frame = frame.copy()
    for col, ref_dtype in ref.dtypes.items():
        if col not in frame.columns:
            continue
        s = str(ref_dtype).lower()
        if "int" in s and "datetime" not in s:
            num = pd.to_numeric(frame[col].astype(str).str.strip(), errors="coerce")
            if not num.isna().any():
                frame[col] = num.astype(ref_dtype)
        elif "float" in s or "timedelta" in s:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        elif "datetime" in s:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def _per_district_max(df: pd.DataFrame) -> dict:
    """Latest timestamp already present per district."""
    if df.empty or nwic.TIME_FIELD not in df.columns:
        return {}
    df = df.copy()
    df[nwic.TIME_FIELD] = pd.to_datetime(df[nwic.TIME_FIELD], errors="coerce")
    g = df.groupby(nwic.DISTRICT_FIELD)[nwic.TIME_FIELD].max()
    return {k: v for k, v in g.items()}


def refresh(districts: list[str], parquet_path: Path, dry_run: bool = False,
            sleep: float = 0.2) -> tuple[list[pd.DataFrame], int]:
    """Fetch + merge the latest per-district 2026 rows.

    Returns (new_frames, total_new_rows). If dry_run, returns frames but does
    not write to disk.
    """
    cur = pd.read_parquet(parquet_path)
    cur_max = _per_district_max(cur)
    new_frames: list[pd.DataFrame] = []
    total_new = 0

    for d in districts:
        start_ts = cur_max.get(d)
        # Fetch strictly-new rows: start a microsecond after the current max.
        start = (start_ts + pd.Timedelta(microseconds=1)).strftime("%Y-%m-%d") if start_ts else "2021-01-01"
        try:
            records = nwic.fetch_district(LIVE_RESOURCE, d, start=start, end=None,
                                          limit=5000, sleep=sleep)
        except RuntimeError as e:
            print(f"  [FAIL] {d}: {e}")
            continue
        if not records:
            print(f"  [ok]  {d}: no new rows (up to {start_ts})")
            continue
        frame = nwic.normalize_records(records)
        if start_ts is not None and nwic.TIME_FIELD in frame.columns:
            frame = frame[pd.to_datetime(frame[nwic.TIME_FIELD]) > start_ts]
        if frame.empty:
            print(f"  [ok]  {d}: no new rows")
            continue
        new_frames.append(frame)
        total_new += len(frame)
        print(f"  [new] {d}: +{len(frame)} rows")
        time.sleep(sleep)

    if dry_run:
        return new_frames, total_new

    if new_frames:
        new = pd.concat(new_frames, ignore_index=True)
        # Relabel incoming District to the canonical value already in the archive,
        # keyed by Station (live returns 'KASGANJ' but AQUIS stores 'Kansiram Nagar').
        station_district = cur.dropna(subset=[nwic.STATION_FIELD, nwic.DISTRICT_FIELD]) \
                                .groupby(nwic.STATION_FIELD)[nwic.DISTRICT_FIELD] \
                                .agg(lambda s: s.value_counts().index[0]).to_dict()
        if nwic.DISTRICT_FIELD in new.columns:
            new[nwic.DISTRICT_FIELD] = new[nwic.STATION_FIELD].map(station_district) \
                .fillna(new[nwic.DISTRICT_FIELD])
        new = _coerce_to_ref(new, cur)
        merged = pd.concat([cur, new], ignore_index=True)
        keys = [nwic.DISTRICT_FIELD, nwic.STATION_FIELD, nwic.TIME_FIELD]
        merged = merged.drop_duplicates(subset=[k for k in keys if k in merged.columns])
        merged = merged.sort_values([nwic.STATION_FIELD, nwic.TIME_FIELD]).reset_index(drop=True)
        shutil.copy2(parquet_path, str(parquet_path) + ".bak.refresh")
        new.to_parquet(str(parquet_path.parent / (parquet_path.stem + "_new_rows.parquet")), index=False)
        merged.to_parquet(parquet_path, index=False)
        print(f"\n  Merged {total_new} new rows -> {parquet_path} ({len(merged)} total)")
    else:
        print("\n  No new rows to merge; dataset is current.")
    return new_frames, total_new


def deploy_check(parquet_path: Path) -> dict:
    """Report rows / min / max / has_2026 / by_year for the loaded archive."""
    import pyarrow.parquet as pq

    ts = pd.to_datetime(pq.ParquetFile(parquet_path)
                        .read(columns=[nwic.TIME_FIELD]).to_pandas()[nwic.TIME_FIELD])
    if ts.empty:
        return {"rows": 0, "has_2026": False}
    yr = ts.dt.year.value_counts().sort_index()
    return {
        "rows": int(len(ts)),
        "min": ts.min().normalize(),
        "max": ts.max().normalize(),
        "has_2026": bool((ts >= pd.Timestamp("2026-01-01")).any()),
        "by_year": {int(k): int(v) for k, v in yr.items()},
    }


def deploy_install(candidate: Path, dest: Path) -> None:
    """Backup current archive, validate the candidate, swap it in."""
    today = pd.Timestamp.now().normalize()
    new = deploy_check(candidate)
    print(f"Candidate     : {candidate}")
    print(f"  rows={new.get('rows')}  {new.get('min')} .. {new.get('max')}  has_2026={new.get('has_2026')}")
    if not new.get("has_2026"):
        print(f"[!] Candidate ends {new['max'].date()} — also stale. Refusing to install.")
        sys.exit(1)
    print(f"Backing up current build to {dest}.bak.deploy ...")
    shutil.copy2(dest, str(dest) + ".bak.deploy")
    print(f"Installing {candidate} -> {dest} ...")
    shutil.copy2(candidate, dest)
    print(f"Done. {new.get('rows'):,} rows, coverage through {new['max'].date()} "
          f"({(today - new['max']).days} days behind today).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PQ)
    ap.add_argument("--deploy-check", action="store_true")
    ap.add_argument("--deploy-install", type=Path, default=None, metavar="PATH")
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"[!] No dataset at {args.parquet}")
        sys.exit(1)

    if args.deploy_check:
        info = deploy_check(args.parquet)
        print(f"Loaded build : {args.parquet}")
        print(f"  rows={info.get('rows')}  {info.get('min')} .. {info.get('max')}")
        print(f"  has_2026    : {info.get('has_2026')}")
        print(f"  by_year     : {info.get('by_year')}")
        sys.exit(0 if info.get("has_2026") else 1)

    if args.deploy_install is not None:
        deploy_install(args.deploy_install, args.parquet)
        return

    districts = nwic.get_aquis_districts(args.parquet)
    print(f"Districts to refresh ({len(districts)}): {', '.join(districts[:8])} ...")

    new_frames, total_new = refresh(districts, args.parquet, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDRY RUN: {total_new} new rows would be added.")
        return

    if args.retrain and total_new > 0:
        print("\nRetraining pooled model against refreshed data ...")
        # 02_fetch_selected only re-extracts GWL when the CSV is missing; the
        # refreshed archive must be re-extracted, so remove the stale file first.
        stale_gwl = config.SELECTED / "gwl_selected.csv"
        if stale_gwl.exists():
            stale_gwl.unlink()
            print(f"  removed stale {stale_gwl.name}")
        for step in RETRAIN_CHAIN:
            cmd = [sys.executable, "-u", str(ROOT / step)]
            print(f"  == {step} ==", flush=True)
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                print(f"  [FAIL] {step} exit code {rc}")
                sys.exit(rc)
        print("Retrain chain complete.")


if __name__ == "__main__":
    main()