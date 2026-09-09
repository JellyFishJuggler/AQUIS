"""ml — pooled GWL forecasting + correlation config.

ALL NWIC/NWDP resource IDs come from back-end/db/api/api.txt (authoritative source
map). Every source is fetched through the CKAN datastore_search API (never the CSV
download links) exactly like the production GWL pipeline.

Sources:  GWL + rainfall + temperature + river water level + humidity + solar
          radiation + wind speed + wind direction + atmospheric pressure
          + canal water level + canal discharge + reservoir discharge (barrages).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ML = Path(__file__).resolve().parent
DATA = ML / "data"
RAW = DATA / "raw"
SELECTED = DATA / "selected"
ALIGNED = DATA / "aligned"
META = DATA / "meta"
OUT = ML / "outputs"

for _p in (RAW, SELECTED, ALIGNED, META, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# Reference GWL archive (production parquet — API-originated, cleaned, large).
GWL_PARQUET = DATA / "processed" / "common.parquet"

# Time window for the experiment (matches local parquet coverage).
START = "2021-01-01"
END = "2026-09-05"  # inclusive

GWL_FIELD = "Groundwater Level Telemetry 6 Hourly (meter)"
TIME_FIELD = "Data Acquisition Time"

# Association radii (km). Rainfall can be blended over multiple gauges (IDW);
# river/canal join is nearest-gauge WITH a distance flag (hydrological relevance
# to be judged later, brief §9).
RAIN_RADIUS_KM = 150.0
RAIN_IDW_NEIGHBOURS = 3
WEATHER_RADIUS_KM = 150.0
RIVER_RADIUS_KM = 100.0
CANAL_RADIUS_KM = 100.0

# Rainfall hourly cap (mm/hr) — corpus protection from corrupt AWS gauges
# (previous lesson: a single gauge reported 2.24e6 mm/hr).
MAX_RAIN_MM_H = 150.0

# Full-2026 coverage rule for "live" stations:
#   first 2026 record <= FULL26_FIRST_MAX, at least FULL26_MIN_MONTHS of the
#   9 months (Jan..Sep 2026) with >= 1 record, last record >= FULL26_LAST_MIN.
FULL26_FIRST_MAX = "2026-01-15"
FULL26_MIN_MONTHS = 8
FULL26_LAST_MIN = "2026-08-01"

# ---------------------------------------------------------------------------
# Sources: archive(2021-25) + live(2026+) resource pairs, value-column hint.
# `agg`: how to collapse hourly -> daily. `kind`: semantic group used later.
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "gwl": {
        "archive": "84bfda45-8ead-436d-9c8e-f7a93ee57522",
        "live": "31c66a49-e110-405d-bcf0-a0ea0d9c8b0c",
        "value_hint": GWL_FIELD,
        "agg": "mean",
        "kind": "gwl",
        "enabled": True,
    },
    "rainfall": {
        "archive": "2c0805b7-bc7c-4174-9c5f-cf7bc1ecd148",
        "live": "46e9afa0-7deb-4fac-8fab-49e0b5b1b6d3",
        "value_hint": "Telemetry Hourly Rainfall (mm)",
        "agg": "sum",
        "kind": "rain",
        "enabled": True,
    },
    "temperature": {
        "archive": "feb54802-f6cc-4061-b873-8311ae04df0e",
        "live": "c2c7b5fa-615d-4b95-8a2d-18eb5297f7b0",
        "value_hint": "Air Temperature Telemetry Hourly (AoC)",
        "agg": "mean",
        "kind": "weather",
        "enabled": True,
    },
    "river_level": {
        "archive": "9672a7e9-0bfe-43ac-9577-a4128792d070",
        "live": "94b5345e-b0d2-45b8-94a4-d93cdede2025",
        "value_hint": "River Water Level Telemetry Hourly (meter)",
        "agg": "mean",
        "kind": "river",
        "enabled": True,
    },
    "humidity": {
        "archive": "14a3cfa8-3eb7-4e6e-8f94-c64693725779",
        "live": "0fcb6700-0ca1-43ad-8550-41397f417c8c",
        "value_hint": "Telemetry Hourly Relative Humidity (%)",
        "agg": "mean",
        "kind": "weather",
        "enabled": True,
    },
    "solar": {
        "archive": "6128c3ff-6d86-4058-b55f-5b2ef492e700",
        "live": "c1666f7a-81e8-4227-b856-39ec192ad004",
        "value_hint": None,  # auto-detect
        "agg": "mean",
        "kind": "weather",
        "enabled": True,
    },
    "wind_speed": {
        "archive": "fc1314ef-a47c-4ebe-b148-5b1ec7a90037",
        "live": "beb454fb-8400-44c1-aa92-bf8b4b282efa",
        "value_hint": None,
        "agg": "mean",
        "kind": "weather",
        "enabled": True,
    },
    "wind_direction": {
        "archive": "b9acc862-92fd-40e4-b8e0-57d3580a75b6",
        "live": "9e8132bc-97ff-480a-af05-adbac43f7b24",
        "value_hint": None,
        "agg": "mean",
        "kind": "weather",
        "enabled": True,
    },
    "pressure": {
        "archive": "32a8fd55-767a-48f6-9220-ac24c447f28a",
        "live": "3525c93c-6122-4174-96b4-3033fd21a645",
        "value_hint": None,
        "agg": "mean",
        "kind": "weather",
        "enabled": True,
    },
    "canal_level": {
        "archive": "ea65ac74-1f6b-41d6-bf85-f927633def4d",
        "live": "8075a3b8-7973-4e60-ba7a-94b193f3936b",
        "value_hint": None,
        "agg": "mean",
        "kind": "canal",
        "enabled": True,
    },
    "canal_discharge": {
        "archive": "a6158de8-089a-4908-9442-6cf5296f161a",
        "live": "14a80ee5-da51-4a97-8a57-1f5857b843ff",
        "value_hint": None,
        "agg": "mean",
        "kind": "canal",
        "enabled": True,
    },
    "res_okhla": {
        "archive": "07e7248d-bedd-47b7-a16d-34eab4c489ec",
        "live": "76b5c177-2ab0-4909-a6fe-feb0b25eabf1",
        "value_hint": None,
        "agg": "mean",
        "kind": "reservoir",
        "enabled": True,
    },
    "res_okhla_agra_canal": {
        "archive": "29aac408-555d-4d3a-87c8-935e3ea3f57a",
        "live": "43a2ba27-8be6-488d-8cdb-7c4d23599696",
        "value_hint": None,
        "agg": "mean",
        "kind": "reservoir",
        "enabled": True,
    },
    "res_matatila": {
        "archive": "26b88468-256b-4a30-bca5-553600d09e30",
        "live": "cd5160e6-7131-4776-adea-9565786f3870",
        "value_hint": None,
        "agg": "mean",
        "kind": "reservoir",
        "enabled": True,
    },
    "res_ganga_rmc": {
        "archive": "e8301a3d-1ced-4435-a805-f572d6b824db",
        "live": "3eee06ec-522a-4d71-9495-9268629e852b",
        "value_hint": None,
        "agg": "mean",
        "kind": "reservoir",
        "enabled": True,
    },
    "res_ganga_1": {
        "archive": "ddc0ece2-a025-4d41-8a1d-5b67cf3a3696",
        "live": "1939859c-acb0-4210-a4ab-02480e4e1d5c",
        "value_hint": None,
        "agg": "mean",
        "kind": "reservoir",
        "enabled": True,
    },
}

# Columns to EXCLUDE when auto-detecting the value column of a resource.
META_COLS = {
    "_id", "Agency", "Basin", "Block", "Data Acquisition Time", "Date",
    "District", "District LGD Code", "State", "State LGD Code", "Station",
    "Tehsil", "Village", "River", "Local River", "Tributary", "Subtributary",
    "SubSubtributary", "Latitude", "Longitude", "SlNo", "RL_MSL",
    "RL_of_zeroGauge", "MeanSeaLevel", "Is_DischargeDataAvailable",
}