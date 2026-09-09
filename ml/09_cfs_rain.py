"""
Fetch CFSv2 (NCEI archive) expected-rainfall series for UP districts.

Source: NCEI THREDDS fileServer, model-cfs_v2_for_6h_flxf (6-hourly surface flux).
Per-step GRIB2 file (~4 MB) hosts ~40 vars; we parse only `prate`.

Design (seasonal-lite):
  * 20 runs: init 00z on Jan 1 / Apr 1 / Jul 1 / Oct 1, 2021..2025 (quarterly).
  * Per run, 30 forecast days; use the 12Z valid-time step as the daily sample.
  * Daily proxy mm = prate(kg/m2/s) * 21600 s * 4  (systematic diurnal scale
    bias is absorbed later by bias-correction against NWIC local rain).
  * Extract UP bbox grid; assign each of the 29 districts the nearest cell.
  * Output: data/cfs/cfs_rain_daily.parquet  (tiny; one row per run/district/day)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "cfs"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://www.ncei.noaa.gov/thredds/fileServer/model-cfs_v2_for_6h_flxf"
BBOX = dict(lat_min=23.9, lat_max=30.1, lon_min=76.9, lon_max=84.1)
RUN_INITS = [
    dt.date(y, m, 1) for y in range(2021, 2026) for m in (1, 4, 7, 10)
]
VALID_HOUR = 12          # single daily sample at 12Z
FORECAST_DAYS = 30       # valid dates init+1 .. init+30
DRY_THRESHOLD = 0.10
RETRIES = 4
TIMEOUT = 60
WORKERS = 8


def run_tag(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def district_centroids() -> pd.DataFrame:
    sel = pd.read_csv(ROOT / "data" / "meta" / "selected_gwl_stations.csv")
    c = sel.groupby("District")[["Latitude", "Longitude"]].mean()
    c["lat"] = c["Latitude"]
    c["lon"] = c["Longitude"]
    return c.reset_index()[["District", "lat", "lon"]]


def nearest_indices(centroids: pd.DataFrame, lat: np.ndarray, lon: np.ndarray):
    """Nearest grid index (xi, yi) for each district centroid."""
    out = []
    for _, row in centroids.iterrows():
        i = int(np.argmin((lat - row["lat"]) ** 2 + (lon - row["lon"]) ** 2))
        out.append(np.unravel_index(i, lat.shape))
    return out


def fetch_step(url: str, dest: Path) -> bool:
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return True
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1) + np.random.random())
    with open(ROOT / "fetch.log", "a") as fh:
        fh.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} CF error {url} -> {last_err}\n")
    print(f"  FETCH-FAIL {url.split('/')[-1]} ({last_err})")
    return False


def load_priate(grib: Path) -> tuple:
    ds = xr.open_dataset(
        grib, engine="cfgrib",
        filter_by_keys={"typeOfLevel": "surface"},
        backend_kwargs={"errors": "ignore", "indexpath": ""},
    )
    p = ds["prate"]
    lat = p["latitude"].values
    lon = p["longitude"].values
    m = (lat >= BBOX["lat_min"]) & (lat <= BBOX["lat_max"])
    n = (lon >= BBOX["lon_min"]) & (lon <= BBOX["lon_max"])
    return lat[m], lon[n], np.asarray(p.isel(latitude=m, longitude=n))


def process_run(init_date: dt.date, src_tmp: Path, centroids: pd.DataFrame) -> pd.DataFrame | None:
    tag = run_tag(init_date)
    rows = []
    lat = lon = None
    for d in range(1, FORECAST_DAYS + 1):
        valid = init_date + dt.timedelta(days=d)
        vtag = valid.strftime("%Y%m%d%H")
        itag = init_date.strftime("%Y%m%d%H")
        url = f"{BASE}/{init_date:%Y/%Y%m}/{init_date:%Y%m%d}/{itag}/flxf{vtag}.01.{itag}.grb2"
        grib = src_tmp / f"{vtag}.01.{itag}.grb2"
        if not fetch_step(url, grib):
            print(f"  FAIL {tag} day{d}")
            return None
        try:
            lat, lon, pr = load_priate(grib)
        except Exception as e:
            print(f"  PARSE-FAIL {tag} day{d}: {e}")
            return None
        finally:
            try:
                grib.unlink()
            except OSError:
                pass
        mm = float(np.nanmean(pr)) * 21600.0 * 4.0
        rows.append((valid, mm, pr))
    L = len(rows)
    if L == 0:
        return None
    # build per-district assignment via copied pr arrays
    df = pd.DataFrame({"run_date": [init_date] * L, "valid_date": [r[0] for r in rows],
                       "day_since_init": list(range(1, L + 1)),
                       "up_mean_mm": [r[1] for r in rows]})
    # district-level from nearest grid cell
    for i, (_, c) in enumerate(centroids.iterrows()):
        vals = []
        for _, pr in rows:
            j = int(np.argmin((lat - c["lat"]) ** 2 + (lon - c["lon"]) ** 2))
            vals.append(float(pr.ravel()[j]) * 21600.0 * 4.0)
        df[f"d_{i}"] = vals
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None, help="init date >= this (YYYYMMDD)")
    ap.add_argument("--dry", action="store_true",
                    help="list pending runs + first URL and exit (no network)")
    args = ap.parse_args()

    centroids = district_centroids()
    tmp = OUT / "tmp"
    tmp.mkdir(exist_ok=True)
    existing = set()
    state = OUT / "cfs_download.log"
    if state.exists():
        existing = {ln.strip() for ln in state.read_text().splitlines() if ln.strip()}

    inits = [d for d in RUN_INITS if d >= dt.date(2021, 1, 1)]
    if args.start:
        s = dt.datetime.strptime(args.start, "%Y%m%d").date()
        inits = [d for d in inits if d >= s]
    inits = [d for d in inits if run_tag(d) not in existing]
    print(f"{dt.datetime.now():%H:%M:%S} runs to fetch: {[run_tag(d) for d in inits]} ({len(inits)})")

    if args.dry:
        if inits:
            itag = inits[0].strftime("%Y%m%d%H")
            vtag = (inits[0] + dt.timedelta(days=1)).strftime("%Y%m%d%H")
            print("first URL:", f"{BASE}/{inits[0]:%Y/%Y%m}/{inits[0]:%Y%m%d}/{itag}/flxf{vtag}.01.{itag}.grb2")
        print(f"dry: {len(inits)} runs pending, {len(existing)} already done, "
              f"{len(centroids)} districts, workers={WORKERS}. No network touched.")
        sys.exit(0)

    dfs = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_run, d, tmp, centroids): d for d in inits}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            d = futs[fut]
            try:
                df = fut.result()
            except Exception as e:
                df = None
                print(f"[{dt.datetime.now():%H:%M:%S}] run {run_tag(d)} ERROR: {e}")
            if df is not None:
                dfs.append(df)
                with open(state, "a") as fh:
                    fh.write(run_tag(d) + "\n")
                print(f"[{dt.datetime.now():%H:%M:%S}] {i}/{len(inits)} run {run_tag(d)} OK "
                      f"({time.time()-t0:.0f}s, {len(df)} rows)")
            else:
                print(f"[{dt.datetime.now():%H:%M:%S}] {i}/{len(inits)} run {run_tag(d)} FAILED")

    if not dfs:
        print("no runs fetched")
        sys.exit(1)

    big = pd.concat(dfs, ignore_index=True)
    cols = ["run_date", "valid_date", "day_since_init", "up_mean_mm"]
    big = big.melt(id_vars=cols[:3], value_vars=[c for c in big.columns if c.startswith("d_")],
                   var_name="_i", value_name="district_mm")
    big["_i"] = big["_i"].str[2:].astype(int)
    big = big.merge(centroids.reset_index().reset_index().rename(columns={"index": "_i"})[["_i", "district"]], on="_i")
    big = big.drop(columns="_i")
    big["district"] = big["district"].astype(str).str.upper()
    big.to_parquet(OUT / "cfs_rain_daily.parquet", index=False)
    print(f"saved {OUT/'cfs_rain_daily.parquet'}  rows={len(big)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()