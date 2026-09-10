"""16_future_drivers — future-driver forecasting / scenario layer.

The recursive GWL engine (17_recursive) needs rain / temp / humidity / solar /
wind / pressure / river / canal at EVERY future 6-hourly step. Real inference
must never see observed future values, so this module supplies them as:

  * scenario = "climatology" (default): apply the TRAINING-period (time <
    2025-10-01) seasonal profile for the target timestamp —
      - per (District, doy, hour) 6h-mean rain  (rainfall seasonality)
      - per (Station, doy)        mean temp / humidity / solar / wind / pressure
      - per (Station, doy)        mean river_level / canal_level (they help
        slightly in ablation; kept when present)
    Nothing from 2026 (the test/backtest period) is used, so there is no
    future-data leakage by construction.
  * scenario = "persistence": carry that station's latest observed driver value
    forward unchanged for the whole horizon (documented scenario).
  * scenario = "dry": climatology but with rain forced to 0 (drought scenario).

Climatology table is built once and cached to data/meta/driver_climatology.parquet
so the backtest and the app use byte-identical scenario values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
META = ROOT / "data" / "meta"
CLIM_PATH = META / "driver_climatology.parquet"

RAIN_COLS = ["rain"]
SEASONAL_WEATHER = ["temp", "humidity", "solar", "wind_speed", "pressure"]
RIVER_CANAL = ["river_level", "canal_level"]
DRIVERS = RAIN_COLS + SEASONAL_WEATHER + RIVER_CANAL

TRAIN_CUT = pd.Timestamp("2025-10-01")   # shares the pipeline train split


def build_climatology(tbl: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Mean seasonal driver profile from TRAINING-year data only.

    Returns (weather_frame[Station, doy, temp...canal_level, trained_from],
             daily[district_rain: District, doy, hour, rain]).
    """
    if tbl is None:
        tbl = pd.read_parquet(ALIGNED)
        tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tr = tbl[tbl["time"] < TRAIN_CUT].copy()
    tr["date"] = tr["time"].dt.normalize()
    tr["hour"] = tr["time"].dt.hour
    tr["doy"] = tr["date"].dt.dayofyear
    if tr.empty or "District" not in tr.columns:
        return pd.DataFrame(), {}

    daily: dict[pd.DataFrame] = {}
    if "District" in tr.columns and "rain" in tr.columns:
        rain = (tr.groupby(["District", "date", "hour"])["rain"]
                  .mean().reset_index(name="rain"))
        rain["doy"] = rain["date"].dt.dayofyear
        daily["district_rain"] = rain.drop(columns=["date"])

    weather = (tr.dropna(subset=SEASONAL_WEATHER)
                 .groupby(["Station", "doy"])[SEASONAL_WEATHER].mean().reset_index())
    rc = (tr.dropna(subset=RIVER_CANAL)
            .groupby(["Station", "doy"])[RIVER_CANAL].mean().reset_index())
    out = weather if not weather.empty else pd.DataFrame()
    if not rc.empty:
        out = out.merge(rc, on=["Station", "doy"], how="left")
    if not out.empty:
        out["trained_from"] = str(TRAIN_CUT.date())
    return out, daily


def _load_clim() -> tuple[pd.DataFrame, dict]:
    if not CLIM_PATH.exists():
        return pd.DataFrame(), {}
    return (pd.read_parquet(CLIM_PATH),
            json.loads(CLIM_PATH.with_name("driver_climatology_meta.json").read_text())
            if CLIM_PATH.with_name("driver_climatology_meta.json").exists()
            else {"scenarios": ["climatology", "persistence", "dry"]})


def scenario_frame(station: str, district: str,
                   hist: pd.DataFrame,
                   horizon_steps: int = 120,
                   scenario: str = "climatology") -> tuple[pd.DataFrame, dict]:
    """Future driver table (one row per +6h step) for the recursive engine.

    ``hist`` = this station's observed 6h table (Station/time/district/rain/...).
    Returns (frame[time, rain, temp, humidity, solar, wind_speed, pressure,
    river_level, canal_level], scenario_meta).
    """
    station = str(station)
    district = str(district).strip().upper()
    clim, daily = _load_clim()
    last_time = hist["time"].max()
    times = [last_time + pd.Timedelta(hours=6 * (k + 1)) for k in range(horizon_steps)]
    frame = pd.DataFrame({"time": times})
    frame["doy"] = frame["time"].dt.dayofyear
    frame["hour"] = frame["time"].dt.hour
    frame["Station"] = station
    frame["District"] = district

    last_obs = hist.sort_values("time").iloc[-1]

    for c in DRIVERS:                      # default column -> NaN
        frame[c] = np.nan

    # ---- rain: district day/hour climatology or zero under "dry" -----------
    if scenario == "dry":
        frame["rain"] = 0.0
    elif "district_rain" in daily and not daily["district_rain"].empty:
        dr = daily["district_rain"]
        merged = frame.merge(
            dr[dr["District"] == district][["doy", "hour", "rain"]],
            on=["doy", "hour"], how="left")
        frame["rain"] = merged["rain_y"] if "rain_y" in merged else merged["rain"]
        frame["rain"] = frame["rain"].fillna(0.0)
    elif scenario == "persistence":
        frame["rain"] = last_obs.get("rain", np.nan)
    else:
        frame["rain"] = 0.0

    # ---- weather + river/canal by scenario ---------------------------------
    if scenario == "persistence":
        for c in DRIVERS:
            if c == "rain":
                continue
            v = last_obs.get(c, np.nan)
            frame[c] = float(v) if v is not None and np.isfinite(v) else np.nan
    elif not clim.empty:
        sub = clim[clim["Station"] == station]
        if not sub.empty:
            frame = frame.merge(sub.drop(columns=["trained_from"]),
                                on=["Station", "doy"], how="left",
                                suffixes=("", "_c"))
            for c in SEASONAL_WEATHER + RIVER_CANAL:
                if f"{c}_c" in frame:
                    frame[c] = frame[f"{c}_c"]
                    frame = frame.drop(columns=[f"{c}_c"])
        # station has no seasonal profile -> fall back to persistence (documented)
        for c in SEASONAL_WEATHER + RIVER_CANAL:
            if frame[c].isna().all():
                frame[c] = last_obs.get(c, np.nan)
    else:
        for c in SEASONAL_WEATHER + RIVER_CANAL:
            frame[c] = last_obs.get(c, np.nan)

    meta = {
        "scenario": scenario,
        "rain_source": "district_doy_hour_climatology" if scenario != "dry"
                       else "forced_zero",
        "weather_source": "station_doy_climatology" if not clim.empty
                          else "persistence_fallback",
        "river_canal_source": "station_doy_climatology" if not clim.empty
                              else "persistence_fallback",
        "training_period": str(TRAIN_CUT.date()),
        "no_future_observed_values": True,
    }
    return frame[["time", *DRIVERS]], meta


def main() -> None:
    tbl = pd.read_parquet(ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    clim, daily = build_climatology(tbl)
    META.mkdir(parents=True, exist_ok=True)
    if not clim.empty:
        clim.to_parquet(CLIM_PATH, index=False)
    rain = daily.get("district_rain")
    if rain is not None:
        rain.to_parquet(META / "district_rain_climatology.parquet", index=False)
    meta = {
        "stations": sorted(clim["Station"].astype(str).unique()) if not clim.empty else [],
        "districts": sorted(rain["District"].astype(str).unique()) if rain is not None else [],
        "weather_cols": SEASONAL_WEATHER + RIVER_CANAL,
        "rail_cols": RAIN_COLS,
        "train_cut": str(TRAIN_CUT.date()),
        "scenarios": ["climatology", "persistence", "dry"],
        "built_at": str(pd.Timestamp.now())[:19],
    }
    (META / "driver_climatology_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"climatology: {len(clim):,} station-day rows | "
          f"{len(meta['stations']):,} stations | {len(meta['districts']):,} districts")


if __name__ == "__main__":
    main()