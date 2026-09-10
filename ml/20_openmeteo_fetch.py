"""Fetch UP district-level daily weather via Open-Meteo (replaces blocked CFS).

Sources (both free, no API key):
  * Archive API  https://archive-api.open-meteo.com/v1/archive   (ERA5-downscaled, 1940+)
  * Forecast API https://api.open-meteo.com/v1/forecast          (GFS/ICON/CFS, up to 16 days)

Hourly -> daily per district centroid: rain_mm (sum), temp_c / humidity_pct /
pressure_hpa / wind_kmh / solar_wm2 (mean). Mirrors AQUIS NUM_COLS semantics so
rain_1d/7d/30d and other rolling features can be rebuilt downstream.

Output: data/cfs/openmeteo_weather_daily.parquet  (one row per date x District)
        data/meta/openmeteo_meta.json            (source/model/run metadata)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "cfs"
META = ROOT / "data" / "meta"
OUT.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HOURLY = ("precipitation,temperature_2m,relative_humidity_2m,"
          "surface_pressure,wind_speed_10m,shortwave_radiation")
RETRIES = 2
TIMEOUT = 30
MODEL_ARCHIVE = "era5_seamless"
WORKERS = 8

DAILY_VARS = {
    "rain_mm": ("precipitation", "sum"),
    "temp_c": ("temperature_2m", "mean"),
    "humidity_pct": ("relative_humidity_2m", "mean"),
    "pressure_hpa": ("surface_pressure", "mean"),
    "wind_kmh": ("wind_speed_10m", "mean"),
    "solar_wm2": ("shortwave_radiation", "mean"),
}


def district_centroids() -> pd.DataFrame:
    sel = pd.read_csv(ROOT / "data" / "meta" / "selected_gwl_stations.csv")
    c = sel.groupby("District")[["Latitude", "Longitude"]].mean()
    c["lat"] = c["Latitude"]
    c["lon"] = c["Longitude"]
    return c.reset_index()[["District", "lat", "lon"]]


def _get(url: str) -> dict:
    last_err = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"Open-Meteo reach after {RETRIES} tries: {last_err}")


def _series_to_df(payload: dict, prefix: str) -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"]),
                       "rain_mm": h["precipitation"], "temp_c": h["temperature_2m"],
                       "humidity_pct": h["relative_humidity_2m"],
                       "pressure_hpa": h["surface_pressure"],
                       "wind_kmh": h["wind_speed_10m"], "solar_wm2": h["shortwave_radiation"]})
    df[prefix] = pd.to_datetime(payload["utc_offset_seconds"], unit="s").tz_localize("UTC")
    return df


def fetch_district(lat: float, lon: float, start: str, end: str,
                   fwd_days: int) -> pd.DataFrame:
    """Return one date x 6-var daily frame for a point, history + forecast."""
    hist = _get(f"{ARCHIVE}?latitude={lat}&longitude={lon}"
                f"&start_date={start}&end_date={end}&hourly={HOURLY}&models={MODEL_ARCHIVE}")
    src = _get(f"{FORECAST}?latitude={lat}&longitude={lon}"
               f"&hourly={HOURLY}&forecast_days={fwd_days}&timezone=UTC")
    df = pd.concat([_series_to_df(hist, "src_hist"), _series_to_df(src, "src_fwd")])
    df = df.dropna(subset=["rain_mm"]).copy()
    df["date"] = df["time"].dt.normalize()
    d = df.groupby("date")[["rain_mm", "temp_c", "humidity_pct",
                            "pressure_hpa", "wind_kmh", "solar_wm2"]].agg(
        {"rain_mm": "sum", "temp_c": "mean", "humidity_pct": "mean",
         "pressure_hpa": "mean", "wind_kmh": "mean", "solar_wm2": "mean"})
    d = d.reset_index().drop_duplicates("date")
    d["District"] = None
    d["source"] = None
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=(dt.date.today() - dt.timedelta(days=365)).isoformat())
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--forecast-days", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all districts")
    args = ap.parse_args()

    cents = district_centroids()
    if args.limit:
        cents = cents.head(args.limit)

    frames = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_one_district, c, args.start, args.end,
                          args.forecast_days): c["District"]
                for _, c in cents.iterrows()}
        done = 0
        for fut in cf.as_completed(futs):
            dist = futs[fut]
            try:
                d = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {dist}: {e}", flush=True)
                continue
            frames.append(d)
            done += 1
            print(f"  ok {dist} ({done}/{len(cents)}) -> {len(d)} days", flush=True)

    out = pd.concat(frames, ignore_index=True).sort_values(["District", "date"])
    if out.empty:
        raise SystemExit("no data fetched")
    out.to_parquet(OUT / "openmeteo_weather_daily.parquet", index=False)

    meta = {
        "source": "open-meteo",
        "hist_api": ARCHIVE + f"?models={MODEL_ARCHIVE}",
        "fwd_api": FORECAST,
        "start": args.start, "end": args.end, "forecast_days": args.forecast_days,
        "districts": int(out["District"].nunique()), "rows": int(len(out)),
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (META / "openmeteo_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {OUT / 'openmeteo_weather_daily.parquet'} ({meta['rows']} rows, "
          f"{meta['districts']} districts)\n{out.groupby('District').size().head(20)}", flush=True)


def _one_district(c, start: str, end: str, fwd_days: int) -> pd.DataFrame:
    """Fetch one-district daily frame (concurrency friendly)."""
    d = fetch_district(c["lat"], c["lon"], start, end, fwd_days)
    d["District"] = c["District"]
    d["source"] = "open-meteo"
    return d


if __name__ == "__main__":
    main()