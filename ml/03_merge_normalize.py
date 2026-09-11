"""03_merge_normalize — cleaned, standardised long table for every completed source.

Applies per-source plausibility caps (reporting what is dropped) and writes
ml/data/raw/<source>_norm.parquet + a manifest of per-source quality stats.
The spatial/temporal alignment (04) consumes this directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import config  # noqa: E402

CAPS: dict[str, tuple[float | None, float | None]] = {
    "gwl": (None, None),            # datum-referenced elevation (can be <0); per-station trim
    "rainfall": (0.0, config.MAX_RAIN_MM_H),  # mm/hour (hourly gauge)
    "temperature": (-10.0, 55.0),   # deg C
    "river_level": (-5.0, 200.0),   # metres
    "humidity": (0.0, 100.0),       # %
    "solar": (0.0, 1600.0),         # W/m2
    "wind_speed": (0.0, 200.0),     # km/h
    "wind_direction": (0.0, 360.0),  # degrees (treated as categorical in analysis)
    "pressure": (900.0, 1100.0),    # mb
    "canal_level": (-5.0, 200.0),   # m
    "canal_discharge": (0.0, None),  # cusec
    "res_okhla": (0.0, None),
    "res_okhla_agra_canal": (0.0, None),
    "res_matatila": (0.0, None),
    "res_ganga_rmc": (0.0, None),
    "res_ganga_1": (0.0, None),
}

# Per-station robust quantile trim (kills telemetry spikes/dry-well sentinels
# without imposing a global datum). Low side trims dry-well sentinels; the high
# side is relaxed to a physically-implausible bound so genuine monsoon-recharge
# (shallow) readings survive instead of silently vanishing from the aligned set.
STATION_TRIM_SOURCES = {"gwl"}
TRIM_LOW, TRIM_HIGH = 0.005, 0.995
# Mirror inverse-water-level datum: values shallower than this (i.e. water above
# the well datum) are reading errors, not real recovery.
GWL_HIGH_PLAUSIBLE_M = -0.1

# Discharge resources carry gate-wise columns; we keep the single auto-detected
# series (Gate-1). This is a known approximation (documented in README).
DISCHARGE_NOTES = {"canal_discharge", "res_okhla", "res_okhla_agra_canal",
                   "res_matatila", "res_ganga_rmc", "res_ganga_1"}


def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(config.SELECTED / f"{name}_selected.csv")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["District"] = df["District"].astype(str).str.strip().str.upper()
    df["Station"] = df["Station"].astype(str).str.strip()
    if "lat" not in df.columns and "Latitude" in df.columns:
        df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    if "Tehsil" in df.columns:
        df["Tehsil"] = df["Tehsil"].astype(str).str.strip()
    return df


def clean(df: pd.DataFrame, lo: float | None, hi: float | None, name: str) -> tuple[pd.DataFrame, int]:
    before = len(df)
    if name in STATION_TRIM_SOURCES and len(df):
        lo_q = df.groupby("Station")["value"].quantile(TRIM_LOW).rename("lo")
        hi_q = df.groupby("Station")["value"].quantile(TRIM_HIGH).rename("hi")
        hi_q = hi_q.clip(lower=GWL_HIGH_PLAUSIBLE_M)
        df = df.join(lo_q, on="Station").join(hi_q, on="Station")
        df = df[df["value"].between(df["lo"], df["hi"])]
        df = df.drop(columns=["lo", "hi"])
    else:
        if lo is not None:
            df = df[df["value"] >= lo]
        if hi is not None:
            df = df[df["value"] <= hi]
    df = df[df["value"].notna() & df["time"].notna()]
    return df, before - len(df)


def main() -> None:
    names = [n for n in config.SOURCES if config.SOURCES[n].get("enabled")]
    manifest = []
    for name in names:
        path = config.SELECTED / f"{name}_selected.csv"
        if not path.exists():
            print(f"[skip] {name} (no CSV yet)", flush=True)
            continue
        df = load_csv(name)
        lo, hi = CAPS.get(name, (None, None))
        df, dropped = clean(df, lo, hi, name)
        df = df.drop_duplicates(subset=["Station", "time"])
        df["source"] = name
        core = ["source", "Station", "District", "lat", "lon", "time", "value"]
        if "Tehsil" in df.columns:
            core.insert(2, "Tehsil")
        df = df[core]
        df = df.sort_values(["Station", "time"])
        out = config.RAW / f"{name}_norm.parquet"
        df.to_parquet(out, index=False)
        stats = {
            "source": name,
            "rows": len(df),
            "stations": int(df["Station"].nunique()),
            "districts": int(df["District"].nunique()),
            "dropped_by_caps": int(dropped),
            "min": float(df["value"].min()) if len(df) else None,
            "max": float(df["value"].max()) if len(df) else None,
            "span": f"{df['time'].min():%Y-%m-%d}..{df['time'].max():%Y-%m-%d}"
                    if len(df) else None,
            "discharge_gate1_only": name in DISCHARGE_NOTES,
        }
        manifest.append(stats)
        print(f"[ok] {name}: rows={stats['rows']:,} st={stats['stations']} "
              f"dist={stats['districts']} dropped={dropped} span={stats['span']}", flush=True)

    (config.META / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nmanifest -> {config.META / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()