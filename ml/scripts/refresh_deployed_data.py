"""Diagnose and fix a stale (non-2026) deployed ``common.parquet``.

The 2026-observed-data bug on the deployed host is a DATA STAGING problem, not a
forecasting bug:

  * The app has NO date cutoff — it plots every observed row of ``common.parquet``
    as blue "Observed", and only draws the green "Estimated Catch-up" between the
    last observed row and TODAY.
  * Locally, ``ml/data/processed/common.parquet`` is the 2026-bearing build
    (readings through ~Aug 2026). The stale "pre-ext" build ends 2025-12-31.
  * On a deployed host running the stale build, "today" (2026) is after every
    station's last observation, so the green catch-up fills 2026 even though real
    2026 readings exist in the newer build — the symptom the operator saw.

This script verifies which build is loaded and offers to swap the current one in.
It does NOT retrain models or alter forecasting; it only stages the correct data
file, then the app can be restarted with 2026 observations intact.

Usage:
    python -m ml.scripts.refresh_deployed_data [--path PATH] [--into PATH]
        --path   candidate 2026-bearing parquet (default: the local current build)
        --into   destination parquet (default: ml/data/processed/common.parquet)
        --check  only report the loaded build's coverage, do not copy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

DEFAULT_DEST = _ML_ROOT / "data" / "processed" / "common.parquet"
KNOWN_NEW_BUILDS = [
    _ML_ROOT / "data" / "processed" / "common_extended.parquet",
    _ML_ROOT / "data" / "processed" / "common.parquet",
]


def _coverage(path: Path) -> dict:
    """Rows, min/max date, row-counts by year for a parquet's Time column."""
    import pyarrow.parquet as pq

    ts = pd.to_datetime(pq.ParquetFile(path).read(columns=["Data Acquisition Time"]).to_pandas()["Data Acquisition Time"])
    if ts.empty:
        return {"rows": 0}
    yr = ts.dt.year.value_counts().sort_index()
    return {
        "rows": int(len(ts)),
        "min": ts.min().normalize(),
        "max": ts.max().normalize(),
        "has_2026": bool((ts >= pd.Timestamp("2026-01-01")).any()),
        "by_year": {int(k): int(v) for k, v in yr.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, do not copy")
    ap.add_argument("--path", type=Path, default=None, help="candidate 2026-bearing parquet to install")
    ap.add_argument("--into", type=Path, default=DEFAULT_DEST, help="destination parquet")
    args = ap.parse_args()

    dest = args.into
    today = pd.Timestamp.now().normalize()

    if not dest.exists():
        print(f"[!] No loaded dataset at {dest}")
        sys.exit(1)

    cur = _coverage(dest)
    print(f"Loaded build : {dest}")
    print(f"  rows={cur.get('rows')}  {cur.get('min')} .. {cur.get('max')}")
    print(f"  has_2026    : {cur.get('has_2026')}")
    print(f"  by_year     : {cur.get('by_year')}")
    cur_behind = (today - pd.Timestamp(cur["max"])).days if "max" in cur else None
    if not cur.get("has_2026"):
        print(f"[!!!] STALE BUILD — data ends {cur['max'].date()}, {cur_behind} days behind "
              f"today ({today.date()}). This is the cause of the green catch-up covering 2026.")

    if args.check:
        sys.exit(0 if cur.get("has_2026") else 1)

    # Choose candidate: explicit --path, else the newest of the known 2026 builds.
    cand = args.path
    if cand is None:
        has = [p for p in KNOWN_NEW_BUILDS if p.exists() and p != dest]
        if not has:
            print("[!] No candidate 2026-bearing build found locally. Pass --path to point at one.")
            sys.exit(1)
        cand = max(has, key=lambda p: pd.Timestamp(_coverage(p)["max"]))

    new = _coverage(cand)
    print(f"\nCandidate     : {cand}")
    print(f"  rows={new.get('rows')}  {new.get('min')} .. {new.get('max')}  has_2026={new.get('has_2026')}")
    if not new.get("has_2026"):
        print(f"[!] Candidate ends {new['max'].date()} — also stale. Refusing to install.")
        sys.exit(1)

    print(f"\nBacking up current build to {dest}.bak.pre_ext ...")
    shutil.copy2(dest, str(dest) + ".bak.pre_ext")
    print(f"Installing {cand} -> {dest} ...")
    shutil.copy2(cand, dest)
    print("Done. Restart the app — 2026 observed readings will now render as blue "
          "'Observed' instead of green catch-up.")


if __name__ == "__main__":
    main()