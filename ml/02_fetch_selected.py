"""02_fetch_selected — pull every configured source 2020-11→2026-09 and store the
*selected-district* subset as temp CSVs exactly where the model step will read them.

District-aware: for each source we pull per selected district directly from the
API (datastore_search District filter). Districts with zero rows cost exactly one
request. Resume-safe via per-source .done markers.

Run: ML_TEMP=[ml] python -u 02_fetch_selected.py > fetch.log 2>&1 &

Outputs: ml/data/selected/<source>_selected.csv
         ml/data/selected/gwl_selected.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data.raw.nwic import BASE_URL, _get, _parse_ts, normalize_records  # noqa: E402

import config  # noqa: E402

FETCH_START = (pd.to_datetime(config.START) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
FETCH_END = (pd.to_datetime(config.END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_district(resource_id: str, district: str,
                   *, limit: int = 5000, sleep: float = 0.25) -> list[dict]:
    session = requests.Session()
    filters = {"District": district}
    params: dict = {"resource_id": resource_id, "filters": json.dumps(filters), "limit": limit}
    rows: list[dict] = []
    offset, total = 0, None
    while True:
        params["offset"] = offset
        result = _get(session, params, max_retries=6)
        if result is None:
            raise RuntimeError(f"{resource_id[:8]} district={district} failed at offset {offset}")
        if total is None:
            total = result.get("total", 0)
        batch = result.get("records", [])
        if not batch:
            break
        rows.extend(r for r in batch if _in_window(r))
        offset += len(batch)
        if offset >= total:
            break
        time.sleep(sleep)
    return rows


def _in_window(r: dict) -> bool:
    ts = _parse_ts(r.get("Data Acquisition Time") or r.get("Date"))
    return bool(ts is not None and pd.to_datetime(FETCH_START) <= ts <= pd.to_datetime(FETCH_END))


def extract_gwl() -> pd.DataFrame:
    sel = pd.read_csv(config.META / "selected_gwl_stations.csv")
    stations = sel["Station"].tolist()
    cols = ["Station", "District", "Tehsil", "Block", "Latitude", "Longitude",
            "RL_MSL", "Data Acquisition Time", config.GWL_FIELD]
    df = pd.read_parquet(config.GWL_PARQUET, columns=cols)
    df = df[df["Station"].isin(stations)].copy()
    df["time"] = pd.to_datetime(df["Data Acquisition Time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[(df["time"] >= config.START) & (df["time"] <= config.END)]
    df = df.rename(columns={config.GWL_FIELD: "value"})
    return df.sort_values(["Station", "time"])


def main() -> None:
    probe = json.loads((config.META / "probe.json").read_text())
    sel_dist = sorted({s.strip().upper() for s in
                       json.loads((config.META / "selected_districts.json").read_text())})
    print(f"selected districts ({len(sel_dist)}): {', '.join(sel_dist)}", flush=True)

    for name, src in config.SOURCES.items():
        if not src.get("enabled") or name == "gwl":
            continue
        marker = config.SELECTED / f"{name}.done"
        if marker.exists():
            print(f"[skip] {name} (done)", flush=True)
            continue
        col = probe[name]["live"]["value_col"] or probe[name]["archive"]["value_col"]
        print(f"[fetch] {name}  (a={src['archive'][:8]} l={src['live'][:8]})", flush=True)

        t0 = time.time()
        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        for rid_label, rid in (("archive", src["archive"]), ("live", src["live"])):
            hits = 0
            for d in sel_dist:
                try:
                    rows = fetch_district(rid, d)
                except Exception as e:  # noqa: BLE001 - never abort the whole source
                    failures.append(f"{rid_label}/{d}: {e}")
                    print(f"    [warn] {rid_label} {d}: {e}", flush=True)
                    rows = []
                if not rows:
                    continue
                df = normalize_records(rows)
                df["time"] = pd.to_datetime(df["Data Acquisition Time"], errors="coerce")
                df["_src"] = rid_label
                frames.append(df)
                hits += 1
                print(f"    {rid_label} {d}: {len(rows):,} rows", flush=True)
            print(f"    {rid_label}: {hits}/{len(sel_dist)} districts with data", flush=True)

        if not frames:
            print(f"    ! nothing within selected districts ({time.time()-t0:.0f}s)", flush=True)
            continue

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["Station", "time"])
        if col and col in df.columns:
            df["value"] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["value", "time"])
        if "Latitude" in df.columns and "Longitude" in df.columns:
            df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
            df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
        keep = ["Station", "District", "lat", "lon", "time", "value"]
        if "Tehsil" in df.columns:
            df["Tehsil"] = df["Tehsil"].astype(str).str.strip()
            keep = ["Station", "District", "Tehsil", "lat", "lon", "time", "value"]
        df = df[keep].sort_values(["Station", "time"])
        path = config.SELECTED / f"{name}_selected.csv"
        df.to_csv(path, index=False)
        marker.touch()
        print(f"    -> {len(df):,} rows / {df['Station'].nunique():,} stations / "
              f"{df['District'].nunique()} dist  ({time.time()-t0:.0f}s)", flush=True)

    gwl_out = config.SELECTED / "gwl_selected.csv"
    if not gwl_out.exists():
        gwl = extract_gwl()
        gwl[["Station", "District", "Tehsil", "Block", "Latitude", "Longitude",
             "RL_MSL", "Data Acquisition Time", "value", "time"]].to_csv(gwl_out, index=False)
        print(f"[gwl] extract -> {len(gwl):,} rows / {gwl['Station'].nunique():,} stations "
              f"({gwl_out})", flush=True)
    else:
        print("[gwl] already extracted", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()