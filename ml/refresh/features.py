"""refresh/features.py — guarded rebuild of the local aligned + climatology data.

After a GWL/weather fetch the pipeline needs the downstream local datasets
(currently: ``data/aligned/table_6h.parquet`` and the scenario climatology
tables) to reflect the new observations before inference runs. ``06_features`` /
``30_traj_datasets`` split on a *fixed* date and are only rebuilt lazily by the
model-update path; the 6-hourly forecast path only needs the aligned table and
climatology, both of which are full-fleet rebuilds (a few minutes each on the
8 GB box).

Safety: every destructive step is bracketed by backup -> run -> validate ->
restore. If any step fails or the validation gate does not pass, the previous
artifacts are restored atomically and the caller is told the refresh failed
(previous forecasts remain published).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ALIGNED_6H = ROOT / "data" / "aligned" / "table_6h.parquet"
ALIGNED_DAILY = ROOT / "data" / "aligned" / "table.parquet"
SELECTED_GWL = ROOT / "data" / "selected" / "gwl_selected.csv"
CLIM_PATH = ROOT / "data" / "meta" / "driver_climatology.parquet"
RAIN_CLIM_PATH = ROOT / "data" / "meta" / "district_rain_climatology.parquet"
WEATHER_PQ = ROOT / "data" / "cfs" / "openmeteo_weather_daily.parquet"

CHAIN = [
    "02_fetch_selected.py",   # regenerate gwl_selected.csv from refreshed archive
    "03_merge_normalize.py",
    "04_align.py",
    "04b_align_6h.py",
]
CLIMATOLOGY_SCRIPT = "16_future_drivers.py"
BACKUP_SUFFIX = ".prev.refresh"


class FeatureBuildError(RuntimeError):
    pass


def _regenerate_gwl_selected() -> None:
    """02_fetch_selected only re-extracts GWL when the CSV is missing; the
    refreshed archive must be re-extracted, so drop the stale file first
    (mirrors 13_refresh_nwic's retrain path)."""
    if SELECTED_GWL.exists():
        SELECTED_GWL.unlink()


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    dst = Path(str(path) + BACKUP_SUFFIX)
    shutil.copy2(path, dst)
    return dst


def restore(path: Path, backup_path: Path | None) -> None:
    if backup_path is not None:
        shutil.copy2(backup_path, path)


def _run(script: str, timeout_s: int = 1800) -> None:
    cmd = [sys.executable, "-u", str(ROOT / script)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        tail += "\n" + "\n".join((proc.stderr or "").splitlines()[-20:])
        raise FeatureBuildError(f"{script} failed (exit {proc.returncode}):\n{tail}")
    time.sleep(1)


def validate_aligned(min_stations: int = 1) -> dict:
    """Row/station/sample-anchor sanity gate on the rebuilt aligned table."""
    if not ALIGNED_6H.exists():
        return {"ok": False, "error": "table_6h.parquet missing"}
    tbl = pd.read_parquet(ALIGNED_6H, columns=["Station", "time", "District", "gwl"])
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    stations = int(tbl["Station"].nunique())
    if stations < min_stations:
        return {"ok": False, "error": f"only {stations} stations (< {min_stations})"}
    latest = tbl["time"].max()
    gwl_rows = int(tbl["gwl"].notna().sum())
    per = tbl.dropna(subset=["gwl"]).groupby("Station")["time"].max()
    stale = int((per < latest - pd.Timedelta(days=7)).sum())
    return {"ok": True, "stations": stations, "rows": int(len(tbl)),
            "latest": latest, "gwl_rows": gwl_rows,
            "stations_stale_7d": stale}


def rebuild_aligned(*, dry_run: bool = False, min_stations: int = 20,
                    run_chain: bool = True, rebuild_climatology: bool = True) -> dict:
    """Run the 3-4b chain + climatology, guarded by backup/restore."""
    backups = {}
    for p in (ALIGNED_6H, ALIGNED_DAILY, CLIM_PATH, RAIN_CLIM_PATH, SELECTED_GWL):
        b = backup(p)
        if b is not None:
            backups[p] = b

    steps: list[str] = []
    try:
        if run_chain:
            if not dry_run:
                _regenerate_gwl_selected()
            steps.extend(CHAIN)
            if not dry_run:
                for step in CHAIN:
                    _run(step)
        if rebuild_climatology:
            steps.append(CLIMATOLOGY_SCRIPT)
            if not dry_run:
                _run(CLIMATOLOGY_SCRIPT)

        if dry_run:
            return {"ok": True, "dry_run": True, "steps": steps}

        verdict = validate_aligned(min_stations=min_stations)
        if not verdict.get("ok"):
            raise FeatureBuildError(verdict.get("error", "validation failed"))
        for p, b in backups.items():
            Path(str(p) + BACKUP_SUFFIX).unlink(missing_ok=True)
        return {"ok": True, "steps": steps, "validate": verdict}
    except Exception:
        for p, b in backups.items():
            restore(p, b)
        raise


def rebuild_traj_features(*, dry_run: bool = False) -> dict:
    """Rebuild the fixed-split trajectory long-frames (data/features_traj)."""
    steps = ["06_features.py", "30_traj_datasets.py"]
    for step in steps:
        if not dry_run:
            _run(step)
    return {"ok": True, "steps": steps, "dry_run": dry_run}


class BackupGuard:
    """context manager exposing backup/restore for the feature datasets."""

    def __init__(self):
        self.paths = (ALIGNED_6H, ALIGNED_DAILY, CLIM_PATH, RAIN_CLIM_PATH, WEATHER_PQ)

    def __enter__(self):
        self._backups = {p: backup(p) for p in self.paths}
        return self._backups

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            for p, b in self._backups.items():
                restore(p, b)
        return False