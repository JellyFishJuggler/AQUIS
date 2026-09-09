"""07_soil — fetch per-station soil texture from ISRIC SoilGrids v2.

For each GWL station (lat,lon) query sand/silt/clay/bdod at 0-5, 5-15, 15-30
and 60-100cm, then aggregate the top three into a thickness-weighted 0-30cm
profile. Saved to ml/data/soil/soil_stations.csv (cached; re-running
resumes from the cache).

Fetches via `curl` subprocess (the Python `requests` stack is ~30x slower
against this host) with a small worker pool.

Units: sand/silt/clay in %, bulk density in g/cm3.
Source: ISRIC SoilGrids 2.0 (CC-BY), rest.isric.org
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOIL_DIR = DATA / "soil"
SOIL_CSV = SOIL_DIR / "soil_stations.csv"
SOIL_META = SOIL_DIR / "soil_meta.json"

GWL_NORM = DATA / "raw" / "gwl_norm.parquet"
API = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = ["sand", "silt", "clay", "bdod"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "60-100cm"]
THICK = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}
WORKERS = 3
MAX_ATTEMPTS = 3

_lock = threading.Lock()


def fetch(lon: float, lat: float) -> dict | None:
    url = "{}?lon={}&lat={}&{}&value=mean".format(
        API, lon, lat,
        "&".join([f"property={p}" for p in PROPERTIES] + [f"depth={d}" for d in DEPTHS]),
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "60", "-H", "Accept: application/json", url],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            delay = 15 * attempt
            time.sleep(delay)
            continue
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            time.sleep(15 * attempt)
            continue
        if not payload.get("properties", {}).get("layers"):
            time.sleep(15 * attempt)
            continue
        return payload
    return None


def mean_at_depth(payload: dict, prop: str, depth: str) -> float | None:
    for layer in payload.get("properties", {}).get("layers", []):
        if layer["name"] != prop:
            continue
        for d in layer.get("depths", []):
            if d["label"] == depth:
                v = d.get("values", {}).get("mean")
                if v is None:
                    return None
                return float(v) / 10.0 if prop in ("sand", "silt", "clay") else float(v) / 100.0
    return None


def agg_0_30(payload: dict, prop: str) -> float | None:
    vals = [(mean_at_depth(payload, prop, d), w) for d, w in THICK.items()]
    vals = [(v, w) for v, w in vals if v is not None]
    if not vals:
        return None
    return sum(v * w for v, w in vals) / sum(w for _, w in vals)


def make_row(payload: dict, srow: pd.Series) -> dict:
    return {
        "Station": srow["Station"],
        "District": srow["District"],
        "lat": srow["lat"],
        "lon": srow["lon"],
        "sand_0_30": agg_0_30(payload, "sand"),
        "silt_0_30": agg_0_30(payload, "silt"),
        "clay_0_30": agg_0_30(payload, "clay"),
        "bdod_0_30": agg_0_30(payload, "bdod"),
        "sand_60_100": mean_at_depth(payload, "sand", "60-100cm"),
        "silt_60_100": mean_at_depth(payload, "silt", "60-100cm"),
        "clay_60_100": mean_at_depth(payload, "clay", "60-100cm"),
        "bdod_60_100": mean_at_depth(payload, "bdod", "60-100cm"),
    }


def persist(row: dict) -> None:
    with _lock:
        pd.DataFrame([row]).to_csv(
            SOIL_CSV, mode="a", header=not SOIL_CSV.exists(), index=False)


def main() -> None:
    gwl = pd.read_parquet(GWL_NORM, columns=["Station", "District", "Tehsil", "lat", "lon"])
    stations = gwl.drop_duplicates("Station").copy()
    SOIL_DIR.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if SOIL_CSV.exists():
        done = set(pd.read_csv(SOIL_CSV)["Station"])

    todo = stations[~stations["Station"].isin(done)]
    total = len(todo)
    print(f"{len(done)} cached, {total} to fetch", flush=True)

    n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, float(r["lon"]), float(r["lat"])): (i, r)
                for i, (_, r) in enumerate(todo.iterrows())}
        for fut in as_completed(futs):
            i, srow = futs[fut]
            payload = fut.result()
            if payload is None:
                print(f"[{i+1}/{total}] SKIP {srow['Station']}: fetch failed", flush=True)
                continue
            row = make_row(payload, srow)
            CORE = ("sand_0_30", "clay_0_30", "silt_0_30", "sand_60_100", "clay_60_100", "silt_60_100")
            if any(row[c] is None for c in CORE):
                print(f"[{i+1}/{total}] SKIP {srow['Station']}: incomplete profile (NullResponse)",
                      flush=True)
                time.sleep(0.15)
                continue
            persist(row)
            n += 1
            print(f"[{n}] {srow['Station']}: sand30={row['sand_0_30']:.1f}% clay30={row['clay_0_30']:.1f}%",
                  flush=True)
            time.sleep(0.15)

    df = pd.read_csv(SOIL_CSV).drop_duplicates("Station")
    meta = {
        "source": "ISRIC SoilGrids v2.0",
        "url": API,
        "queried_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "properties": PROPERTIES,
        "depths_raw": DEPTHS,
        "depths_agg": ["0-30cm (thickness-weighted)", "60-100cm"],
        "units": "sand/silt/clay in %, bdod in g/cm3",
        "stations": int(df["Station"].nunique()),
    }
    SOIL_META.write_text(json.dumps(meta, indent=2))
    print(f"saved {len(df):,} stations ({n} new) -> {SOIL_CSV.name}", flush=True)


if __name__ == "__main__":
    main()