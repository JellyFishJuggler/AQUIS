"""
Reusable time-series preprocessing shared across model families
(XGBoost, Random Forest, ...).

Keeps a common set of helpers (station lookup, slugging, cleaning,
chronological splits) so every model consumes the same prepared data.
"""

import json
import re
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

GWL_COL = "Groundwater Level Telemetry 6 Hourly (meter)"
TIME_COL = "Data Acquisition Time"
TRAIN_RATIO = 0.8

MAX_ABS_GWL = 100.0
SENTINEL_VALUES = {-9999, -1000, -999, -99, 99, 999, 1000, 9999}


def station_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return slug.strip("_")


def filter_corrupt_gwl(df: pd.DataFrame) -> pd.DataFrame:
    """Drop physically impossible GWL readings (telemetry sentinels + spikes).

    Telemetry failures surface as sentinel codes (e.g. -1000, 999, 99) or as
    large spikes (hundreds of meters) that are impossible for groundwater
    aquifers measured every 6 hours.  Readings with |value| > ``MAX_ABS_GWL``
    or matching ``SENTINEL_VALUES`` are dropped before feature construction.
    """
    vals = pd.to_numeric(df[GWL_COL], errors="coerce")
    ok = vals.abs() <= MAX_ABS_GWL
    not_sentinel = ~vals.isin(SENTINEL_VALUES)
    return df[(ok.to_numpy() & not_sentinel.to_numpy())]


def resolve_slug_dir(root: Path, station: str) -> Path | None:
    """Return the artifact dir storing ``station``, tolerating slug collisions.

    ``station_slug`` is not injective, so we first try the naive slug dir when
    it provably stores ``station``, then scan every dir's ``features.json``
    station field.  Returns ``None`` when no dir stores the station (callers
    must NOT fall back to a colliding sibling's dir).
    """
    slug = station_slug(station)
    direct = root / slug
    if (direct / "features.json").is_file():
        try:
            with open(direct / "features.json") as f:
                if json.load(f).get("station") == station:
                    return direct
        except json.JSONDecodeError:
            pass
    if root.exists():
        for d in sorted(root.iterdir()):
            fp = d / "features.json"
            if fp.is_file():
                try:
                    with open(fp) as f:
                        if json.load(f).get("station") == station:
                            return d
                except json.JSONDecodeError:
                    continue
    return None


def unique_station_dir(root: Path, station: str) -> Path:
    """Pick a non-clobbering artifact dir for ``station``.

    ``station_slug`` is not injective (e.g. 'Aanganwadi Kendra' and
    'Aanganwadi kendra' both slug to 'aanganwadi_kendra'); when the naive
    slug dir already stores a *different* station, fall back to appending
    _2, _3, ... until a free slot is found (or the same station is reused).
    """
    base = station_slug(station)
    candidate = root / base
    if candidate.exists() and (candidate / "features.json").is_file():
        try:
            with open(candidate / "features.json") as f:
                if json.load(f).get("station") == station:
                    return candidate
        except json.JSONDecodeError:
            pass
        i = 2
        while (root / f"{base}_{i}").exists():
            i += 1
        return root / f"{base}_{i}"
    return candidate


def build_time_index(station_df: pd.DataFrame) -> pd.DataFrame:
    df = station_df.sort_values(TIME_COL).reset_index(drop=True)
    t0 = df[TIME_COL].min()
    df["time_hours"] = (df[TIME_COL] - t0).dt.total_seconds() / 3600
    return df


def get_station_series(
    parquet_path: str | Path,
    station: str,
) -> pd.DataFrame:
    """Load one station's sorted series with a ``time_hours`` index.

    Corrupt telemetry readings (sentinels / physically impossible spikes)
    are dropped before feature construction so training and prediction both
    operate on the same clean series.
    """
    df = pd.read_parquet(parquet_path)
    sdf = df[df["Station"] == station].copy()
    return build_time_index(filter_corrupt_gwl(sdf))


def detect_gaps(station_df: pd.DataFrame, threshold_hours: float = 9.0) -> pd.DataFrame:
    diffs = station_df[TIME_COL].diff()
    gap_mask = diffs.dt.total_seconds() / 3600 > threshold_hours
    gaps = station_df.loc[gap_mask].copy()
    gaps["gap_hours"] = diffs[gap_mask].dt.total_seconds() / 3600
    return gaps