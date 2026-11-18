"""Optional static loaders: soil texture, CGWB extraction and ISRO LULC.

All sources are optional: a missing CSV/JSON yields an empty frame/dict so the
feature builder and serving paths can skip the feature instead of failing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SOIL_DIR = BASE / "data" / "soil"
SOIL_CSV = SOIL_DIR / "soil_stations.csv"
SOIL_META = SOIL_DIR / "soil_meta.json"
EXTRACTION_CSV = SOIL_DIR / "cgwb_extraction_district.csv"
LULC_CSV = SOIL_DIR / "lulc_district.csv"
LULC_META = BASE / "data" / "meta" / "lulc_meta.json"

SOIL_COLS = [
    "sand_0_30", "silt_0_30", "clay_0_30", "bdod_0_30",
    "sand_60_100", "silt_60_100", "clay_60_100", "bdod_60_100",
]
EXTRACTION_COLS = ["ann_gw_draft_mcm", "gw_stage_dev_pct"]
LULC_COLS = [
    "lulc_builtup_pct", "lulc_cropland_pct", "lulc_plantation_pct",
    "lulc_fallow_pct", "lulc_shifting_cult_pct", "lulc_forest_pct",
    "lulc_grass_pct", "lulc_wasteland_pct", "lulc_water_pct",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_soil() -> pd.DataFrame:
    """Per-station texture profile; empty frame if the fetch never ran."""
    return _read_csv(SOIL_CSV)


def load_soil_meta() -> dict:
    return _read_json(SOIL_META)


def load_extraction() -> pd.DataFrame:
    """District x year CGWB assessment (district | year | draft | stage)."""
    df = _read_csv(EXTRACTION_CSV)
    if not df.empty:
        df["district"] = df["district"].astype(str).str.upper().str.strip()
        df["year"] = df["year"].astype(int)
    return df


def load_lulc() -> pd.DataFrame:
    """Per-district static LULC percentages; empty frame if never fetched."""
    df = _read_csv(LULC_CSV)
    if not df.empty:
        df["District"] = df["District"].astype(str).str.upper().str.strip()
    return df


def load_lulc_meta() -> dict:
    return _read_json(LULC_META)


def extraction_year_grid(assess: pd.DataFrame, years: pd.Series) -> pd.DataFrame:
    """Carry each assessment forward until the next one for its district."""
    if assess.empty:
        return pd.DataFrame()
    all_years = sorted(set(years.astype(int).tolist()))
    grid = pd.DataFrame(
        [(d, y) for d in assess["district"].unique() for y in all_years],
        columns=["district", "row_year"],
    )
    merged = grid.merge(assess, on="district", how="left")
    merged = merged[merged["year"] <= merged["row_year"]]
    keep = (merged.sort_values("row_year")
            .groupby(["district", "row_year"], as_index=False).last())
    return keep.rename(columns={"row_year": "year"})
