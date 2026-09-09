"""Shared loaders for static per-station soil texture and district-year
groundwater extraction (CGWB annual assessment, user-maintained CSV).

Used by the feature pipeline (06) and the Streamlit app. All optional:
extraction data may be absent; callers must handle empty frames.
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

SOIL_COLS = [
    "sand_0_30", "silt_0_30", "clay_0_30", "bdod_0_30",
    "sand_60_100", "silt_60_100", "clay_60_100", "bdod_60_100",
]

EXTRACTION_COLS = ["ann_gw_draft_mcm", "gw_stage_dev_pct"]


def load_soil() -> pd.DataFrame:
    """Per-station texture profile; empty frame if the fetch never ran."""
    if not SOIL_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SOIL_CSV)


def load_soil_meta() -> dict:
    if not SOIL_META.exists():
        return {}
    return json.loads(SOIL_META.read_text())


def load_extraction() -> pd.DataFrame:
    """District x year CGWB assessment. Expected schema:
        district | year | net_annual_gw_draft_mcm | stage_of_development_pct
    Returns empty frame if the CSV is absent (feature is skipped).
    """
    if not EXTRACTION_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(EXTRACTION_CSV)
    df["district"] = df["district"].astype(str).str.upper().str.strip()
    df["year"] = df["year"].astype(int)
    return df


def extraction_year_grid(assess: pd.DataFrame, years: pd.Series) -> pd.DataFrame:
    """Carry each assessment forward until the next one for its district.

    Returns a frame of (district, row_year, ann_gw_draft_mcm, gw_stage_dev_pct)
    such that each row_year uses the most recent assessment <= row_year.
    """
    if assess.empty:
        return pd.DataFrame()
    all_years = sorted(set(years.astype(int).tolist()))
    grid = pd.DataFrame(
        [(d, y) for d in assess["district"].unique() for y in all_years],
        columns=["district", "row_year"],
    )
    merged = grid.merge(assess, left_on="district", right_on="district", how="left")
    merged = merged[merged["year"] <= merged["row_year"]]
    keep = (
        merged.sort_values("row_year")
        .groupby(["district", "row_year"], as_index=False)
        .last()
    )
    return keep.rename(columns={"row_year": "year"})