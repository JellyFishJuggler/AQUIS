"""Shared loader for static per-district ISRO Bhuvan LULC features.

Fetched by 08_lulc.py (LULC 50K Thematic Statistics, 2011-12 cycle) -> wide CSV
keyed by the repository's canonical District names. Optional like soil/extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _soil import EXTRACTION_COLS  # noqa: F401  (re-export for build_full)

BASE = Path(__file__).resolve().parent
LULC_CSV = BASE / "data" / "soil" / "lulc_district.csv"
LULC_META = BASE / "data" / "meta" / "lulc_meta.json"

LULC_COLS = [
    "lulc_builtup_pct", "lulc_cropland_pct", "lulc_plantation_pct",
    "lulc_fallow_pct", "lulc_shifting_cult_pct", "lulc_forest_pct",
    "lulc_grass_pct", "lulc_wasteland_pct", "lulc_water_pct",
]


def load_lulc() -> pd.DataFrame:
    """Per-district static LULC percentages; empty frame if never fetched."""
    if not LULC_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(LULC_CSV)
    df["District"] = df["District"].astype(str).str.upper().str.strip()
    return df


def load_lulc_meta() -> dict:
    if not LULC_META.exists():
        return {}
    return json.loads(LULC_META.read_text())