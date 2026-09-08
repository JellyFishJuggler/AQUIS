"""Refresh AQUIS groundwater telemetry from the LIVE 2026 NWIC resource.

The deployed/local ``common.parquet`` is built from the NWIC archive
(2021-2025). The live 2026 resource
(``31c66a49-e110-405d-bcf0-a0ea0d9c8b0c``) exposes the *latest* observed
groundwater telemetry. This script pulls the newest records for EVERY AQUIS
district (generalized — no hard-coded district), merges them (deduped,
chronologically sorted) into the model's ``common.parquet``, and optionally
triggers a retrain so the model actually consumes the refreshed data.

It is *incremental*: it only fetches rows newer than the current per-district
max timestamp already present in ``common.parquet``, so repeated runs are cheap.

Usage:
    python -m ml.scripts.refresh_nwic_data [--dry-run] [--retrain] [--workers N]

    --dry-run   report how many new rows would be added, write nothing.
    --retrain   after merging, refresh all station models against the new data.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.data.raw import nwic  # noqa: E402 - needs sys.path set above

LIVE_RESOURCE = nwic.RES_GROUNDWATER_LIVE
DEFAULT_PQ = _ML_ROOT / "data" / "processed" / "common.parquet"


def _per_district_max(df: pd.DataFrame) -> dict:
    """Latest timestamp already present per district."""
    if df.empty or "Data Acquisition Time" not in df.columns:
        return {}
    df = df.copy()
    df["Data Acquisition Time"] = pd.to_datetime(df["Data Acquisition Time"], errors="coerce")
    g = df.groupby(nwic.DISTRICT_FIELD)["Data Acquisition Time"].max()
    return {k: v for k, v in g.items()}


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
        # Keep only rows strictly newer than the current max.
        if start_ts is not None and "Data Acquisition Time" in frame.columns:
            frame = frame[pd.to_datetime(frame["Data Acquisition Time"]) > start_ts]
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
        # Relabel incoming District to the canonical value already in common.parquet,
        # keyed by Station (e.g. live returns 'KASGANJ' but AQUIS stores 'Kansiram Nagar').
        station_district = cur.dropna(subset=[nwic.STATION_FIELD, nwic.DISTRICT_FIELD]) \
                                .groupby(nwic.STATION_FIELD)[nwic.DISTRICT_FIELD] \
                                .agg(lambda s: s.value_counts().index[0]).to_dict()
        if nwic.DISTRICT_FIELD in new.columns:
            new[nwic.DISTRICT_FIELD] = new[nwic.STATION_FIELD].map(station_district) \
                .fillna(new[nwic.DISTRICT_FIELD])
        # Cast incoming columns to the archive parquet's dtypes (the live API returns
        # e.g. numeric-looking strings for SlNo), so concat+to_parquet cannot fail.
        new = _coerce_to_ref(new, cur)
        merged = pd.concat([cur, new], ignore_index=True)
        # Dedup on district+station+time, then chronological sort.
        keys = [nwic.DISTRICT_FIELD, nwic.STATION_FIELD, "Data Acquisition Time"]
        merged = merged.drop_duplicates(subset=[k for k in keys if k in merged.columns])
        merged = merged.sort_values([nwic.STATION_FIELD, "Data Acquisition Time"]).reset_index(drop=True)
        # Back up current, then install.
        shutil.copy2(parquet_path, str(parquet_path) + ".bak.refresh")
        new.to_parquet(str(parquet_path.parent / (parquet_path.stem + "_new_rows.parquet")), index=False)
        merged.to_parquet(parquet_path, index=False)
        print(f"\n  Merged {total_new} new rows -> {parquet_path} ({len(merged)} total)")
    else:
        print("\n  No new rows to merge; dataset is current.")
    return new_frames, total_new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PQ)
    args = ap.parse_args()

    if not args.parquet.exists():
        print(f"[!] No dataset at {args.parquet}")
        sys.exit(1)

    districts = nwic.get_aquis_districts(args.parquet)
    print(f"Districts to refresh ({len(districts)}): {', '.join(districts[:8])} ...")

    new_frames, total_new = refresh(districts, args.parquet, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDRY RUN: {total_new} new rows would be added.")
        return

    if args.retrain and total_new > 0:
        print("\nRetraining all station models against refreshed data ...")
        cmd = [sys.executable, "-m", "ml.training.train_all_forecast",
               "--workers", str(args.workers),
               "--parquet", str(args.parquet)]
        rc = subprocess.call(cmd)
        print(f"Retrain exit code: {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
