"""Shared data-loading helpers for the ml explorer app.

Two module-level paths are read directly:
  * corpus_manifest  -> ml/data/meta/manifest.json    (per-source quality summary)
  * selected_gwl     -> ml/data/meta/selected_gwl_stations.csv
  * district_set     -> ml/data/meta/selected_districts.json
  * table            -> ml/data/aligned/table.parquet (daily station x day)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
META = DATA / "meta"
ALIGNED = DATA / "aligned"

DRIVER_LABELS = {
    "rain_1d": "Rain, 1 day",
    "rain_7d": "Rain, 7 days",
    "rain_30d": "Rain, 30 days",
    "rain": "Rain (daily)",
    "temp": "Air temperature",
    "humidity": "Relative humidity",
    "solar": "Solar radiation",
    "wind_speed": "Wind speed",
    "wind_dir": "Wind direction",
    "pressure": "Atmospheric pressure",
    "river_level": "River water level",
    "canal_level": "Canal water level",
    # engineered model features (6-hourly grid)
    "gwl": "GWL (current)",
    "lag1": "GWL lag 6h",
    "lag4": "GWL lag 1d",
    "lag8": "GWL lag 2d",
    "lag28": "GWL lag 7d",
    "lag120": "GWL lag 30d",
    "gwl_roll7_mean": "GWL 7d mean",
    "gwl_roll7_std": "GWL 7d spread",
    "gwl_roll30_mean": "GWL 30d mean",
    "gwl_roll30_std": "GWL 30d spread",
    "temp_7d": "Temp 7d mean",
    "humidity_7d": "Humidity 7d mean",
    "year": "Year",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
    "doy_sin": "Day-of-year (sin)",
    "doy_cos": "Day-of-year (cos)",
    "hour_sin": "Hour (sin)",
    "hour_cos": "Hour (cos)",
    "monsoon": "Monsoon season",
    # ordinal encodings (model inputs, not hydrologically meaningful)
    "st_id": "Station (ordinal)",
    "dist_id": "District (ordinal)",
}

SOURCE_LABELS = {
    "gwl": "Groundwater level",
    "rainfall": "Rainfall",
    "temperature": "Temperature",
    "river_level": "River water level",
    "humidity": "Humidity",
    "solar": "Solar radiation",
    "wind_speed": "Wind speed",
    "wind_direction": "Wind direction",
    "pressure": "Pressure",
    "canal_level": "Canal water level",
    "canal_discharge": "Canal discharge",
}


@st.cache_data(ttl="10m", max_entries=4)
def load_manifest() -> dict:
    return json.loads((META / "manifest.json").read_text())


@st.cache_data(ttl="10m", max_entries=4)
def load_selected_gwl() -> pd.DataFrame:
    return pd.read_csv(META / "selected_gwl_stations.csv")


@st.cache_data(ttl="10m", max_entries=4)
def load_district_set() -> list[str]:
    return sorted({s for s in json.loads((META / "selected_districts.json").read_text())})


@st.cache_data(ttl="10m", max_entries=4)
def load_report() -> pd.DataFrame:
    return pd.read_csv(BASE / "outputs" / "correlation_report.csv")


@st.cache_data(ttl="10m", max_entries=4)
def load_lag_curves() -> pd.DataFrame:
    return pd.read_csv(BASE / "outputs" / "lag_curves.csv")


@st.cache_data(ttl="10m", max_entries=2)
def load_table() -> pd.DataFrame:
    df = pd.read_parquet(ALIGNED / "table.parquet")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values(["Station", "date"]).reset_index(drop=True)


@st.cache_data(ttl="10m", max_entries=2)
def load_table_6h() -> pd.DataFrame:
    df = pd.read_parquet(ALIGNED / "table_6h.parquet")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.sort_values(["Station", "time"]).reset_index(drop=True)


@st.cache_data(ttl="10m", max_entries=2)
def station_recency() -> pd.Series:
    """Most-recent reading per station (6h table), already cached on top of
    load_table_6h so the 3M-row groupby only runs once per TTL window."""
    return load_table_6h().groupby("Station")["time"].max()


def nice(name: str) -> str:
    return DRIVER_LABELS.get(name, name.replace("_", " ").capitalize())


def source_nice(name: str) -> str:
    return SOURCE_LABELS.get(name, name.replace("_", " "))
