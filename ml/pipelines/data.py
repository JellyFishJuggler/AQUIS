"""Ingest, alignment, static-source fetchers and the batch CLI dispatcher.

Shared geo helpers (haversine, gauge->well association, IDW/nearest joins) live
here once; model training/evaluation lives in ``pipelines.model``.

Run a stage with:  python -m pipelines.data <stage>
Run --list to see all stage names and legacy aliases. The ``pipeline_data``
root shim forwards to this module for backward compatibility.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (ROOT, REPO):          # config + ml.data.raw.nwic when run as a script
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ml.data.raw import nwic as nwic_mod  # noqa: E402
from ml.data.raw.nwic import _get, _parse_ts, normalize_records  # noqa: E402
import config  # noqa: E402

R = 6371.0
STEP = pd.Timedelta("6h")
MAX_FFILL_STEPS = 12                    # carry a reading across gaps <= 3 days
FETCH_START = (pd.to_datetime(config.START) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
FETCH_END = (pd.to_datetime(config.END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

STATION_COLS = ["Station", "District", "Tehsil", "Block", "Latitude", "Longitude",
                "Data Acquisition Time"]

# Per-source plausibility caps. GWL is datum-referenced (can be negative), so it
# uses a per-station robust quantile trim instead of a global bound.
CAPS: dict[str, tuple[float | None, float | None]] = {
    "gwl": (None, None),
    "rainfall": (0.0, config.MAX_RAIN_MM_H), "temperature": (-10.0, 55.0),
    "river_level": (-5.0, 200.0), "humidity": (0.0, 100.0),
    "solar": (0.0, 1600.0), "wind_speed": (0.0, 200.0),
    "wind_direction": (0.0, 360.0), "pressure": (900.0, 1100.0),
    "canal_level": (-5.0, 200.0), "canal_discharge": (0.0, None),
    "res_okhla": (0.0, None), "res_okhla_agra_canal": (0.0, None),
    "res_matatila": (0.0, None), "res_ganga_rmc": (0.0, None),
    "res_ganga_1": (0.0, None),
}
STATION_TRIM_SOURCES = {"gwl"}
TRIM_LOW, TRIM_HIGH = 0.005, 0.995
GWL_HIGH_PLAUSIBLE_M = -0.1            # shallower than this = reading error
DISCHARGE_NOTES = {"canal_discharge", "res_okhla", "res_okhla_agra_canal",
                   "res_matatila", "res_ganga_rmc", "res_ganga_1"}

NEAREST_SOURCES = {
    "temperature": ("temp", None), "river_level": ("river_level", config.RIVER_RADIUS_KM),
    "humidity": ("humidity", None), "solar": ("solar", None),
    "wind_speed": ("wind_speed", None), "wind_direction": ("wind_dir", None),
    "pressure": ("pressure", None), "canal_level": ("canal_level", config.CANAL_RADIUS_KM),
    "canal_discharge": ("canal_discharge", config.CANAL_RADIUS_KM),
    **{n: (n, config.CANAL_RADIUS_KM) for n in (
        "res_okhla", "res_okhla_agra_canal", "res_matatila",
        "res_ganga_rmc", "res_ganga_1")},
}
NEAREST_WITH_DIST = {"river_level", "canal_level", "canal_discharge"} | {
    n for n in NEAREST_SOURCES if n.startswith("res_")}


# ---------------------------------------------------------------- stage: probe
def probe_resource(session: requests.Session, resource_id: str) -> dict:
    result = _get(session, {"resource_id": resource_id, "limit": 1}, max_retries=5)
    if result is None:
        return {"ok": False, "total": None, "fields": [], "sample": None}
    records = result.get("records", [])
    fields = [f.get("id") for f in result.get("fields", [])] or (
        list(records[0]) if records else [])
    return {"ok": True, "total": result.get("total"), "fields": fields,
            "sample": records[0] if records else None}


def detect_value_col(fields: list[str], sample: dict | None, hint: str | None) -> str | None:
    """Numeric measurement column, honouring the config hint when present."""
    if hint is not None and hint in fields:
        return hint
    cols = [f for f in fields if f not in config.META_COLS and "id" not in f.lower()
            and not f.lower().startswith("_")]
    if sample:
        for c in cols:
            v = sample.get(c)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return c
    return cols[0] if cols else None


def probe_main() -> None:
    session = requests.Session()
    probe = {}
    for name, src in config.SOURCES.items():
        if not src.get("enabled"):
            probe[name] = {"enabled": False}
            continue
        print(f"[probe] {name}  archive={src['archive'][:8]} live={src['live'][:8]}")
        arch = probe_resource(session, src["archive"])
        time.sleep(0.3)
        live = probe_resource(session, src["live"])
        time.sleep(0.3)
        ac = detect_value_col(arch["fields"], arch["sample"], src.get("value_hint"))
        lc = detect_value_col(live["fields"], live["sample"], src.get("value_hint"))
        probe[name] = {
            "enabled": True,
            "archive": {"id": src["archive"], "ok": arch["ok"],
                        "total": arch["total"], "value_col": ac},
            "live": {"id": src["live"], "ok": live["ok"],
                     "total": live["total"], "value_col": lc},
            "agg": src["agg"], "kind": src["kind"],
        }
        print(f"    archive total={arch['total']} col={ac!r} | "
              f"live total={live['total']} col={lc!r}")
    out = config.META / "probe.json"
    out.write_text(json.dumps(probe, indent=2))
    print(f"\nprobe.json saved -> {out}")


# ---------------------------------------------------------------- stage: select
def load_gwl() -> pd.DataFrame:
    df = pd.read_parquet(config.GWL_PARQUET, columns=STATION_COLS)
    df["Data Acquisition Time"] = pd.to_datetime(df["Data Acquisition Time"], errors="coerce")
    return df.dropna(subset=["Data Acquisition Time"])


def full26_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Full-2026 live-station rule (config.FULL26_*)."""
    t = df["Data Acquisition Time"]
    per = df.assign(year=t.dt.year, month=t.dt.month, d2026=t.dt.year == 2026)
    idx = pd.Index(per["Station"].drop_duplicates())
    months = (per[per["d2026"]].groupby("Station")["month"]
              .apply(lambda s: s.drop_duplicates().nunique()))
    first = per[per["d2026"]].groupby("Station")["Data Acquisition Time"].min()
    last = per.groupby("Station")["Data Acquisition Time"].max()
    out = pd.DataFrame(index=idx)
    out["n_2026_months"] = months.reindex(idx).fillna(0).astype(int)
    out["first_2026"] = first.reindex(idx)
    out["last_record"] = last.reindex(idx)
    out["full26"] = (
        (out["n_2026_months"] >= config.FULL26_MIN_MONTHS)
        & (per.groupby("Station")["d2026"].sum().reindex(idx) > 0)
        & (pd.to_datetime(out["first_2026"]).dt.date.astype(str) <= config.FULL26_FIRST_MAX)
        & (pd.to_datetime(out["last_record"]).dt.date.astype(str) >= config.FULL26_LAST_MIN)
    )
    out["n_2026_records"] = (per[per["d2026"]].groupby("Station").size()
                             .reindex(idx).fillna(0).astype(int))
    return out


def select_main() -> None:
    print("loading GWL parquet ...")
    df = load_gwl()
    print(f"  raw rows={len(df):,} stations={df['Station'].nunique():,} "
          f"districts={df['District'].nunique()}")
    flags = full26_flags(df)
    st = df.drop_duplicates("Station").set_index("Station")[
        ["District", "Tehsil", "Block", "Latitude", "Longitude"]]
    sel = st.join(flags).reset_index()
    sel = sel[["Station", "District", "Tehsil", "Block", "Latitude", "Longitude",
               "n_2026_months", "n_2026_records", "first_2026", "last_record", "full26"]]
    sel["first_2026"] = pd.to_datetime(sel["first_2026"])
    sel["last_record"] = pd.to_datetime(sel["last_record"])
    keep = sel[sel["full26"]].copy()
    districts = sorted(keep["District"].dropna().unique().tolist())
    sel.to_csv(config.META / "gwl_full_flags.csv", index=False)
    keep.to_csv(config.META / "selected_gwl_stations.csv", index=False)
    (config.META / "selected_districts.json").write_text(json.dumps(districts, indent=2))

    # Stale-coverage watchlist: 2026 telemetry that failed the live gate.
    watch = sel[(sel["full26"] == False) & (sel["n_2026_months"] >= 1)].copy()  # noqa: E712
    if not watch.empty:
        wbreak = pd.to_datetime(config.FULL26_LAST_MIN)
        wstart = pd.to_datetime(config.FULL26_FIRST_MAX)
        watch["reason"] = "other"
        watch.loc[watch["last_record"] < wbreak, "reason"] = "telemetry_stale"
        watch.loc[(watch["reason"] == "other")
                  & (watch["n_2026_months"] < config.FULL26_MIN_MONTHS),
                  "reason"] = "sparse_2026"
        watch.loc[(watch["reason"] == "other") & (watch["first_2026"] > wstart),
                  "reason"] = "late_start"
    watch.to_csv(config.META / "stale_watchlist.csv", index=False)
    print(f"\nstale-coverage watchlist: {len(watch):,} stations -> "
          f"{config.META / 'stale_watchlist.csv'}")
    if not watch.empty:
        print(watch["reason"].value_counts().to_string())
    print(f"\nfull-2026 qualifying stations: {len(keep):,} of {len(sel):,}")
    print(f"selected districts ({len(districts)}): {', '.join(districts)}")
    print(f"saved -> {config.META / 'selected_gwl_stations.csv'}")


# ---------------------------------------------------------------- stage: fetch
def _get_effective_end() -> str:
    """Newest timestamp in the refreshed archive, so config.END never lops off
    freshly fetched readings when this stage is re-run during a refresh."""
    try:
        if config.GWL_PARQUET.exists():
            import pyarrow.parquet as pq
            names = [n.lower() for n in pq.ParquetFile(config.GWL_PARQUET).schema.names]
            tc = "Data Acquisition Time" if "data acquisition time" in names else "time"
            ts = pd.to_datetime(
                pq.ParquetFile(config.GWL_PARQUET).read(columns=[tc]).to_pandas()[tc],
                errors="coerce")
            if ts.max() is not None and pd.notna(ts.max()):
                return str(ts.max())
    except Exception:  # noqa: BLE001 - never block extraction on a probe
        pass
    return config.END


def fetch_district(resource_id: str, district: str,
                   *, limit: int = 5000, sleep: float = 0.25) -> list[dict]:
    session = requests.Session()
    params = {"resource_id": resource_id,
              "filters": json.dumps({"District": district}), "limit": limit}
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
    cols = ["Station", "District", "Tehsil", "Block", "Latitude", "Longitude",
            "RL_MSL", "Data Acquisition Time", config.GWL_FIELD]
    df = pd.read_parquet(config.GWL_PARQUET, columns=cols)
    df = df[df["Station"].isin(sel["Station"].tolist())].copy()
    df["time"] = pd.to_datetime(df["Data Acquisition Time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[(df["time"] >= config.START) & (df["time"] <= pd.to_datetime(_get_effective_end()))]
    df = df.rename(columns={config.GWL_FIELD: "value"})
    return df.sort_values(["Station", "time"])


def fetch_main() -> None:
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
        t0, frames, failures = time.time(), [], []
        for label, rid in (("archive", src["archive"]), ("live", src["live"])):
            hits = 0
            for d in sel_dist:
                try:
                    rows = fetch_district(rid, d)
                except Exception as e:  # noqa: BLE001 - one bad district != whole source
                    failures.append(f"{label}/{d}: {e}")
                    print(f"    [warn] {label} {d}: {e}", flush=True)
                    rows = []
                if not rows:
                    continue
                df = normalize_records(rows)
                df["time"] = pd.to_datetime(df["Data Acquisition Time"], errors="coerce")
                df["_src"] = label
                frames.append(df)
                hits += 1
                print(f"    {label} {d}: {len(rows):,} rows", flush=True)
            print(f"    {label}: {hits}/{len(sel_dist)} districts with data", flush=True)
        if not frames:
            print(f"    ! nothing within selected districts ({time.time()-t0:.0f}s)", flush=True)
            continue
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Station", "time"])
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
        path = config.SELECTED / f"{name}_selected.csv"
        df[keep].sort_values(["Station", "time"]).to_csv(path, index=False)
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


# ---------------------------------------------------------------- stage: normalize
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


def clean(df: pd.DataFrame, lo: float | None, hi: float | None,
          name: str) -> tuple[pd.DataFrame, int]:
    before = len(df)
    if name in STATION_TRIM_SOURCES and len(df):
        low = df.groupby("Station")["value"].quantile(TRIM_LOW).rename("lo")
        high = df.groupby("Station")["value"].quantile(TRIM_HIGH).rename("hi")
        df = df.join(low, on="Station").join(high.clip(lower=GWL_HIGH_PLAUSIBLE_M), on="Station")
        df = df[df["value"].between(df["lo"], df["hi"])].drop(columns=["lo", "hi"])
    else:
        if lo is not None:
            df = df[df["value"] >= lo]
        if hi is not None:
            df = df[df["value"] <= hi]
    return df[df["value"].notna() & df["time"].notna()], before - len(df)


def normalize_main() -> None:
    manifest = []
    for name, src in config.SOURCES.items():
        if not src.get("enabled"):
            continue
        path = config.SELECTED / f"{name}_selected.csv"
        if not path.exists():
            print(f"[skip] {name} (no CSV yet)", flush=True)
            continue
        df, dropped = clean(load_csv(name), *CAPS.get(name, (None, None)), name)
        df = df.drop_duplicates(subset=["Station", "time"])
        df["source"] = name
        core = ["source", "Station", "District", "lat", "lon", "time", "value"]
        if "Tehsil" in df.columns:
            core.insert(2, "Tehsil")
        df = df[core].sort_values(["Station", "time"])
        out = config.RAW / f"{name}_norm.parquet"
        df.to_parquet(out, index=False)
        stats = {
            "source": name, "rows": len(df),
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


# ----------------------------------------------------------- shared align helpers
def haversine(lat1, lon1, lat2, lon2) -> pd.Series:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def associate(gwl: pd.DataFrame, src: pd.DataFrame, radius_km: float | None,
              name: str, *, k: int = 1) -> pd.DataFrame:
    """Nearest gauge(s) per well within radius; writes assoc_<name>.csv.

    k=1 is the old ``nearest``; k>1 is the old ``top_k`` (rainfall IDW).
    """
    rows = []
    for _, s in gwl.iterrows():
        if pd.isna(s["lat"]) or pd.isna(s["lon"]):
            continue
        d = haversine(s["lat"], s["lon"], src["lat"], src["lon"])
        near = d <= radius_km
        if not near.any():
            continue
        sub = src[near].copy()
        sub["dist_km"] = d[near]
        sub["gwl_station"] = s["Station"]
        sub["src_station"] = sub["Station"]
        cols = ["gwl_station", "src_station", "dist_km"]
        picked = sub[cols].sort_values("dist_km").head(k)
        # k=1 mirrors the old ``nearest`` (one gauge per well); k>1 keeps all
        # neighbours — dropping by well would break rainfall IDW blending.
        rows.append(picked.drop_duplicates("gwl_station") if k == 1 else picked)
    out = (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols))
    out.to_csv(config.META / f"assoc_{name}.csv", index=False)
    return out


def tehsil_assoc(gwl: pd.DataFrame, src: pd.DataFrame, name: str) -> pd.DataFrame:
    """Exact (District, Tehsil) match, falling back to district-level proxy when
    well tehsils are unpublished; both networks may lack coordinates."""
    g = gwl.dropna(subset=["District"])[
        ["Station", "District", "Tehsil"]].rename(columns={"Station": "gwl_station"})
    s = src.dropna(subset=["District"])[
        ["Station", "District", "Tehsil"]].rename(columns={"Station": "src_station"})
    cols = ["gwl_station", "src_station", "dist_km"]
    g_ok, s_ok = g.dropna(subset=["Tehsil"]), s.dropna(subset=["Tehsil"])
    out = (g_ok.merge(s_ok, on=["District", "Tehsil"])[cols[:2]]
           if not g_ok.empty and not s_ok.empty else pd.DataFrame(columns=cols[:2]))
    mode = "tehsil"
    if out.empty:
        out = g[["gwl_station", "District"]].merge(
            s[["src_station", "District"]], on="District")[cols[:2]]
        mode = "district"
    out["dist_km"] = np.nan
    out = out.drop_duplicates(cols[:2])
    out.to_csv(config.META / f"assoc_{name}.csv", index=False)
    if mode == "district":
        print("      (tehsils unpublished -> district-level proxy used)", flush=True)
    return out


def idw_join(assoc: pd.DataFrame, vals: pd.DataFrame, value_col: str,
             tcol: str) -> pd.DataFrame:
    """1/d^2 weighted gauge value per well and time bin."""
    cols = ["Station", tcol, value_col, f"{value_col}_n"]
    if assoc.empty or vals.empty:
        return pd.DataFrame(columns=cols)
    m = assoc.merge(vals, left_on="src_station", right_on="Station", how="left")
    m = m.dropna(subset=[tcol, "value"])
    m["w"] = 1.0 / np.maximum(m["dist_km"], 0.05) ** 2
    g = m.groupby(["gwl_station", tcol]).apply(
        lambda x: pd.Series({
            value_col: float(np.average(x["value"], weights=x["w"])),
            f"{value_col}_n": int(x["src_station"].nunique()),
        }), include_groups=False)
    return g.reset_index().rename(columns={"gwl_station": "Station"})


def nearest_join(assoc: pd.DataFrame, vals: pd.DataFrame, value_col: str,
                 tcol: str, *, with_dist: bool = False) -> pd.DataFrame:
    if assoc.empty or vals.empty:
        out = pd.DataFrame(columns=["Station", tcol, value_col])
        if with_dist:
            out[f"{value_col}_dist_km"] = []
        return out
    m = assoc.merge(vals, left_on="src_station", right_on="Station", how="left")
    m = m.dropna(subset=[tcol, "value"])
    agg = {value_col: ("value", "mean")}
    if with_dist:
        agg[f"{value_col}_dist_km"] = ("dist_km", "mean")
    g = m.groupby(["gwl_station", tcol], as_index=False).agg(**agg)
    return g.rename(columns={"gwl_station": "Station"})


def load_norm(name: str) -> pd.DataFrame:
    p = config.RAW / f"{name}_norm.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["date"] = df["time"].dt.normalize()
    return df.dropna(subset=["date"])


def aggregate(df: pd.DataFrame, tcol: str, agg: str) -> pd.DataFrame:
    """Group raw values into (Station, tcol); agg in {sum, median, mean}."""
    how = agg if agg in ("sum", "median") else "mean"
    return df.groupby(["Station", tcol], as_index=False)["value"].agg(how)


def load_assoc(name: str) -> pd.DataFrame:
    p = config.META / f"assoc_{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def bin6(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["bin"] = df["time"].dt.floor(STEP)
    return df.dropna(subset=["bin"])


# ---------------------------------------------------------------- stage: align
def align_main() -> None:
    gwl = load_norm("gwl")
    if gwl.empty:
        raise SystemExit("gwl missing — run the normalize stage first")
    # NWIC uses -999 as dry/offline sentinel; >500 m is unphysical in UP aquifers.
    bad = gwl["value"].abs() > 500
    if bad.sum():
        print(f"gwl sentinel/telemetry rows dropped: {int(bad.sum()):,}", flush=True)
    gwl = gwl[~bad]
    anchor = (gwl.drop_duplicates("Station").set_index("Station")
              [["District", "Tehsil", "lat", "lon"]].reset_index())
    print(f"GWL anchors: {len(anchor):,} stations / {anchor['District'].nunique()} districts",
          flush=True)
    tbl = aggregate(gwl, "date", "median").rename(columns={"value": "gwl"}).merge(
        anchor[["Station", "District"]], on="Station", how="left")

    rain = load_norm("rainfall")
    if not rain.empty:
        ra = rain.dropna(subset=["lat", "lon"])
        assoc = associate(anchor, ra.drop_duplicates("Station")[["Station", "lat", "lon"]],
                          config.RAIN_RADIUS_KM, "rainfall", k=config.RAIN_IDW_NEIGHBOURS)
        rj = idw_join(assoc, aggregate(ra, "date", "sum"), "rain", "date")
        tbl = tbl.merge(rj, on=["Station", "date"], how="left")
        print(f"  rainfall: {len(assoc):,} well-gauge links", flush=True)

    for name, (col, radius) in NEAREST_SOURCES.items():
        src = load_norm(name)
        if src.empty:
            print(f"  [skip] {name}: no normalized data", flush=True)
            continue
        s_coord = src.dropna(subset=["lat", "lon"])
        if s_coord.empty:
            assoc = tehsil_assoc(anchor, src, name)
            print(f"  {name}: tehsil-fallback ({len(assoc):,} well-gauge links)", flush=True)
        else:
            rad = radius or (config.CANAL_RADIUS_KM if name.startswith("res_")
                             else config.WEATHER_RADIUS_KM)
            assoc = associate(anchor, s_coord.drop_duplicates("Station")
                              [["Station", "lat", "lon"]], rad, name)
            print(f"  {name}: {len(assoc):,} well-gauge links", flush=True)
        j = nearest_join(assoc, aggregate(src, "date", "mean"), col,
                         "date", with_dist=name in NEAREST_WITH_DIST)
        if not j.empty:
            tbl = tbl.merge(j, on=["Station", "date"], how="left")

    tbl = tbl.sort_values(["Station", "date"]).reset_index(drop=True)
    if "rain" in tbl.columns:
        for w in (7, 30):
            tbl[f"rain_{w}d"] = tbl.groupby("Station")["rain"].transform(
                lambda s: s.rolling(w).sum())
        tbl["rain_1d"] = tbl["rain"]
    out_p, out_c = config.ALIGNED / "table.parquet", config.ALIGNED / "table.csv"
    tbl.to_parquet(out_p, index=False)
    tbl.to_csv(out_c, index=False)
    print(f"\ntable: {len(tbl):,} rows / {tbl['Station'].nunique():,} stations "
          f"/ span {tbl['date'].min():%Y-%m-%d}..{tbl['date'].max():%Y-%m-%d}", flush=True)
    print(f"saved -> {out_p.name} / {out_c.name}", flush=True)


# -------------------------------------------------------------- stage: align6
def gwl_track(gwl: pd.DataFrame) -> pd.DataFrame:
    """Regular 6h grid per station; median per slot, ffill up to 3 days.

    ``gwl_obs``: 1 = actual reading, 0 = carried forward, NaN = gap beyond the
    carry limit. Feature builders use it so trends never sit on a filled plateau.
    """
    gwl = gwl[gwl["value"].abs() <= 500]
    frames = []
    for st, sub in gwl.groupby("Station"):
        sub = sub.sort_values("time")
        idx = pd.date_range(sub["time"].min().floor(STEP),
                            sub["time"].max().floor(STEP), freq=STEP)
        slot = (pd.Series(sub["value"].values,
                          index=pd.to_datetime(sub["time"]).values)
                .groupby(lambda t: pd.Timestamp(t).floor(STEP)).median())
        s = slot.reindex(idx)
        obs = s.notna()
        s = s.ffill(limit=MAX_FFILL_STEPS)
        frames.append(pd.DataFrame({
            "Station": st, "time": idx, "gwl": s.values,
            "gwl_obs": np.where(s.notna(), obs.astype(float), np.nan),
        }))
    return pd.concat(frames, ignore_index=True)


def align6_main() -> None:
    gwl = load_norm("gwl")
    if gwl.empty:
        raise SystemExit("gwl missing")
    anchor = (gwl.drop_duplicates("Station").set_index("Station")
              [["District", "Tehsil", "lat", "lon"]].reset_index())
    print(f"GWL anchors: {len(anchor):,} stations", flush=True)
    track = gwl_track(gwl).merge(anchor[["Station", "District"]], on="Station", how="left")
    print(f"gwl track: {len(track):,} rows / {track['Station'].nunique():,} stations", flush=True)

    rain = load_norm("rainfall").dropna(subset=["lat", "lon"])
    if not rain.empty:
        ra = bin6(rain)
        assoc = load_assoc("rainfall")
        if assoc.empty:
            assoc = associate(anchor, ra.drop_duplicates("Station")[["Station", "lat", "lon"]],
                              config.RAIN_RADIUS_KM, "rainfall", k=config.RAIN_IDW_NEIGHBOURS)
        rj = idw_join(assoc, aggregate(ra, "bin", "sum"), "rain", "bin")
        track = track.merge(rj, left_on=["Station", "time"],
                            right_on=["Station", "bin"], how="left").drop(columns=["bin"])
        print(f"  rainfall: {len(rj):,} well-6h rows", flush=True)

    for name, (col, radius) in NEAREST_SOURCES.items():
        src = load_norm(name)
        if src.empty:
            print(f"  [skip] {name}: no normalized data", flush=True)
            continue
        s = bin6(src)
        assoc = load_assoc(name)
        if assoc.empty:
            s_coord = s.dropna(subset=["lat", "lon"])
            if s_coord.empty:
                assoc = tehsil_assoc(anchor, s, name)
            else:
                assoc = associate(anchor, s_coord.drop_duplicates("Station")
                                  [["Station", "lat", "lon"]],
                                  radius or config.WEATHER_RADIUS_KM, name)
        j = nearest_join(assoc, aggregate(s, "bin", "mean"), col, "bin")
        if not j.empty:
            track = track.merge(j, left_on=["Station", "time"],
                                right_on=["Station", "bin"], how="left").drop(columns=["bin"])
            print(f"  {name}: {len(j):,} well-6h rows", flush=True)

    track = track.sort_values(["Station", "time"]).reset_index(drop=True)
    out = config.ALIGNED / "table_6h.parquet"
    track.to_parquet(out, index=False)
    print(f"\ntable_6h: {len(track):,} rows / {track['Station'].nunique():,} stations / "
          f"span {track['time'].min():%Y-%m-%d}..{track['time'].max():%Y-%m-%d}", flush=True)
    print(f"saved -> {out.name}", flush=True)


# ======================================================= CWC river-level forecast
CWC_URL = "http://117.250.2.226:84/FMISCFloodDashboard/up_flood_dashboard.html"
CWC_PARQUET = ROOT / "data" / "cfs" / "river_forecast_cwc.parquet"
CWC_META = ROOT / "data" / "meta" / "river_forecast_cwc_meta.json"
CWC_DAYS = 7
CWC_DAY_BASE = 11                 # Day-1 (dt, cond, wl) starts here in every row
CWC_SNAPSHOT_COLS = {"observed_dt": 8, "observed_cond": 9, "observed_wl": 10}
CWC_RIVERS = {"BETWA", "GANDAK", "GANGA", "GHAGRA", "GOMTI", "KEN",
              "RAMGANGA", "RAPTI", "SAI", "YAMUNA"}


def _cwc_norm(s: object) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def _cwc_num(s: object) -> float:
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return float("nan")


def _cwc_parse_dt(s: str) -> pd.Timestamp:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%y %H:%M"):
        try:
            return pd.to_datetime(str(s).strip(), format=fmt)
        except ValueError:
            continue
    return pd.NaT


class _CwcParser(HTMLParser):
    """First dashboard table as raw cell rows."""

    def __init__(self) -> None:
        super().__init__()
        self._in_table = self._in_tr = False
        self.cells: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:  # noqa: D102
        if tag == "table":
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._in_tr, self.cells = True, []
        elif tag in ("th", "td") and self._in_tr:
            self.cells.append("")

    def handle_data(self, data: str) -> None:  # noqa: D102
        if self._in_tr and self.cells:
            self.cells[-1] = (self.cells[-1] + " " + data).strip()

    def handle_endtag(self, tag: str) -> None:  # noqa: D102
        if tag in ("th", "td") and self._in_tr and self.cells:
            self.cells[-1] = self.cells[-1].strip()
        elif tag == "tr" and self._in_tr:
            self.rows.append(list(self.cells))
            self.cells, self._in_tr = [], False
        elif tag == "table":
            self._in_table = False


def _cwc_align_row(row: list[str], header: list[str]) -> list[str] | None:
    """Slide a stray leading empty cell back into alignment with the header."""
    hr_sr = next((i for i, x in enumerate(header) if "Sr.No" in str(x)), None)
    nums = [k for k, x in enumerate(row) if re.fullmatch(r"\d+", str(x).strip() or "")]
    if not nums:
        return None
    sr_pos = hr_sr if hr_sr is not None else 0
    offset = nums[0] - sr_pos
    if offset > 0:
        row = row[offset:] if offset < len(row) else []
        row = list(row) + [""] * (sr_pos if offset - sr_pos > 0 else 0)
    elif offset < 0:
        row = [""] * (-offset) + list(row)
    return (row + [""] * 33)[:33]


def _cwc_row_to_record(row: list[str], header: list[str]) -> dict | None:
    aligned = _cwc_align_row(row, header)
    if aligned is None or not re.fullmatch(r"\d+", str(aligned[0]).strip() or ""):
        return None
    rec: dict = {
        "sr_no": int(aligned[0]), "site": aligned[1].strip(),
        "river": _cwc_norm(aligned[2]), "district": _cwc_norm(aligned[3]),
        "state": _cwc_norm(aligned[4]),
        "wl_c": _cwc_num(aligned[5]), "dl_c": _cwc_num(aligned[6]),
        "hfl_c": _cwc_num(aligned[7]),
    }
    for name, idx in CWC_SNAPSHOT_COLS.items():
        rec[name] = (_cwc_parse_dt(aligned[idx]) if name.endswith("_dt")
                     else _cwc_num(aligned[idx]) if name.endswith("_wl")
                     else str(aligned[idx]).strip())
    for day in range(1, CWC_DAYS + 1):
        base = CWC_DAY_BASE + 3 * (day - 1)
        rec[f"d{day}_dt"] = _cwc_parse_dt(aligned[base])
        rec[f"d{day}_cond"] = str(aligned[base + 1]).strip()
        rec[f"d{day}_wl"] = _cwc_num(aligned[base + 2])
    return rec


def cwc_parse_html(html: str) -> pd.DataFrame:
    """Dashboard HTML -> one row per CWC site (observed + 7 forecast levels)."""
    parser = _CwcParser()
    parser.feed(html)
    header = next((r for r in parser.rows if len(r) >= 9 and "Sr.No" in str(r[0])),
                  parser.rows[1] if len(parser.rows) > 1 else [])
    recs = [r for r in (_cwc_row_to_record(row, header) for row in parser.rows) if r]
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    df["river"] = df["river"].where(df["river"].isin(CWC_RIVERS), df["river"])
    return df


def _cwc_atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _cwc_atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def cwc_fetch_forecast(url: str | None = None, timeout: int = 30,
                       out_parquet: Path | None = None, out_meta: Path | None = None,
                       keep_last: int = 40, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    """Fetch, parse and persist a rolling CWC snapshot. Returns (df, meta)."""
    url = url or CWC_URL
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="replace")
    df = cwc_parse_html(html)
    if df.empty:
        raise RuntimeError(f"cwc: no river rows parsed from {url}")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df.insert(0, "fetched_at", fetched_at)

    if out_parquet is not None and not dry_run:
        merged = df
        if out_parquet.exists():
            merged = pd.concat([pd.read_parquet(out_parquet), df], ignore_index=True)
            snaps = sorted(merged["fetched_at"].unique())
            if len(snaps) > keep_last:
                merged = merged[merged["fetched_at"].isin(snaps[-keep_last:])]
        _cwc_atomic_parquet(out_parquet, merged.reset_index(drop=True))
        meta = {
            "source": "cwc-fmisc-up-flood-forecast", "forecast_days": CWC_DAYS,
            "sites": int(df["site"].nunique()),
            "rivers": sorted(df["river"].unique().tolist()),
            "districts": sorted(df["district"].unique().tolist()),
            "fetched_at": fetched_at, "cached_rows": int(len(merged)),
        }
        if out_meta is not None:
            _cwc_atomic_json(out_meta, meta)
        return df, meta
    return df, {
        "source": "cwc-fmisc-up-flood-forecast", "forecast_days": CWC_DAYS,
        "sites": int(df["site"].nunique()), "fetched_at": fetched_at,
    }


def cwc_main() -> None:
    df, meta = cwc_fetch_forecast(out_parquet=CWC_PARQUET, out_meta=CWC_META)
    print(f"cwc river forecast: {meta['sites']} sites "
          f"({len(meta['rivers'])} rivers, {len(meta['districts'])} districts) "
          f"@ {meta['fetched_at']}")


# ==================================================== future-driver climatology
CLIM_ALIGNED = config.ALIGNED / "table_6h.parquet"
CLIM_DIR = ROOT / "data" / "meta"
CLIM_PATH = CLIM_DIR / "driver_climatology.parquet"
CLIM_RAIN = ["rain"]
CLIM_WEATHER = ["temp", "humidity", "solar", "wind_speed", "pressure"]
CLIM_RIVER_CANAL = ["river_level", "canal_level"]
CLIM_TRAIN_CUT = pd.Timestamp("2025-10-01")   # shares the model train split


def clim_build(tbl: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Mean seasonal driver profile from TRAINING-period rows only."""
    if tbl is None:
        tbl = pd.read_parquet(CLIM_ALIGNED)
        tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    tr = tbl[tbl["time"] < CLIM_TRAIN_CUT].copy()
    if tr.empty or "District" not in tr.columns:
        return pd.DataFrame(), {}
    tr["date"] = tr["time"].dt.normalize()
    tr["hour"], tr["doy"] = tr["time"].dt.hour, tr["date"].dt.dayofyear

    daily: dict = {}
    if "rain" in tr.columns:
        rain = (tr.groupby(["District", "date", "hour"])["rain"]
                  .mean().reset_index(name="rain"))
        rain["doy"] = rain["date"].dt.dayofyear
        daily["district_rain"] = rain.drop(columns=["date"])

    weather = (tr.dropna(subset=CLIM_WEATHER)
                 .groupby(["Station", "doy"])[CLIM_WEATHER].mean().reset_index())
    rc = (tr.dropna(subset=CLIM_RIVER_CANAL, how="all")
            .groupby(["Station", "doy"])[CLIM_RIVER_CANAL].mean().reset_index())
    out = weather if not weather.empty else pd.DataFrame()
    if not rc.empty:
        out = rc if out.empty else out.merge(rc, on=["Station", "doy"], how="outer")
    if not out.empty:
        present = [c for c in CLIM_WEATHER + CLIM_RIVER_CANAL if c in out.columns]
        out = out.dropna(subset=present, how="all").reset_index(drop=True)
        out["trained_from"] = str(CLIM_TRAIN_CUT.date())
    return out, daily


def climatology_main() -> None:
    tbl = pd.read_parquet(CLIM_ALIGNED)
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    clim, daily = clim_build(tbl)
    CLIM_DIR.mkdir(parents=True, exist_ok=True)
    if not clim.empty:
        clim.to_parquet(CLIM_PATH, index=False)
    rain = daily.get("district_rain")
    if rain is not None:
        rain.to_parquet(CLIM_DIR / "district_rain_climatology.parquet", index=False)
    meta = {
        "stations": sorted(clim["Station"].astype(str).unique()) if not clim.empty else [],
        "districts": sorted(rain["District"].astype(str).unique()) if rain is not None else [],
        "weather_cols": CLIM_WEATHER + CLIM_RIVER_CANAL, "rail_cols": CLIM_RAIN,
        "train_cut": str(CLIM_TRAIN_CUT.date()),
        "scenarios": ["climatology", "persistence", "dry"],
        "built_at": str(pd.Timestamp.now())[:19],
    }
    (CLIM_DIR / "driver_climatology_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"climatology: {len(clim):,} station-day rows | "
          f"{len(meta['stations']):,} stations | {len(meta['districts']):,} districts")


# ============================================================== soil + LULC fetch
SOIL_DIR = ROOT / "data" / "soil"
SOIL_CSV = SOIL_DIR / "soil_stations.csv"
SOIL_META_FILE = SOIL_DIR / "soil_meta.json"
SOIL_GWL_NORM = config.RAW / "gwl_norm.parquet"
SOIL_API = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPERTIES = ["sand", "silt", "clay", "bdod"]
SOIL_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "60-100cm"]
SOIL_THICK = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}
SOIL_WORKERS, SOIL_ATTEMPTS = 3, 3
_soil_lock = threading.Lock()


def soil_fetch(lon: float, lat: float) -> dict | None:
    url = "{}?lon={}&lat={}&{}&value=mean".format(
        SOIL_API, lon, lat,
        "&".join([f"property={p}" for p in SOIL_PROPERTIES]
                 + [f"depth={d}" for d in SOIL_DEPTHS]))
    for attempt in range(1, SOIL_ATTEMPTS + 1):
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "60", "-H", "Accept: application/json", url],
            capture_output=True, text=True)
        if proc.returncode != 0:
            time.sleep(15 * attempt)
            continue
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            time.sleep(15 * attempt)
            continue
        if payload.get("properties", {}).get("layers"):
            return payload
        time.sleep(15 * attempt)
    return None


def soil_mean_at_depth(payload: dict, prop: str, depth: str) -> float | None:
    for layer in payload.get("properties", {}).get("layers", []):
        if layer["name"] != prop:
            continue
        for d in layer.get("depths", []):
            if d["label"] == depth:
                v = d.get("values", {}).get("mean")
                if v is None:
                    return None
                return float(v) / (10.0 if prop in ("sand", "silt", "clay") else 100.0)
    return None


def soil_agg_0_30(payload: dict, prop: str) -> float | None:
    vals = [(soil_mean_at_depth(payload, prop, d), w) for d, w in SOIL_THICK.items()]
    vals = [(v, w) for v, w in vals if v is not None]
    return sum(v * w for v, w in vals) / sum(w for _, w in vals) if vals else None


def soil_make_row(payload: dict, srow: pd.Series) -> dict:
    row = {"Station": srow["Station"], "District": srow["District"],
           "lat": srow["lat"], "lon": srow["lon"]}
    for prop in SOIL_PROPERTIES:
        row[f"{prop}_0_30"] = soil_agg_0_30(payload, prop)
        row[f"{prop}_60_100"] = soil_mean_at_depth(payload, prop, "60-100cm")
    return row


def soil_persist(row: dict) -> None:
    with _soil_lock:
        pd.DataFrame([row]).to_csv(SOIL_CSV, mode="a", header=not SOIL_CSV.exists(), index=False)


def soil_main() -> None:
    gwl = pd.read_parquet(SOIL_GWL_NORM, columns=["Station", "District", "Tehsil", "lat", "lon"])
    stations = gwl.drop_duplicates("Station").copy()
    SOIL_DIR.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(SOIL_CSV)["Station"]) if SOIL_CSV.exists() else set()
    todo = stations[~stations["Station"].isin(done)]
    total = len(todo)
    print(f"{len(done)} cached, {total} to fetch", flush=True)

    n = 0
    with ThreadPoolExecutor(max_workers=SOIL_WORKERS) as ex:
        futs = {ex.submit(soil_fetch, float(r["lon"]), float(r["lat"])): (i, r)
                for i, (_, r) in enumerate(todo.iterrows())}
        for fut in as_completed(futs):
            i, srow = futs[fut]
            payload = fut.result()
            if payload is None:
                print(f"[{i+1}/{total}] SKIP {srow['Station']}: fetch failed", flush=True)
                continue
            row = soil_make_row(payload, srow)
            core = ("sand_0_30", "clay_0_30", "silt_0_30",
                    "sand_60_100", "clay_60_100", "silt_60_100")
            if any(row[c] is None for c in core):
                continue
            soil_persist(row)
            n += 1
            print(f"[{n}] {srow['Station']}: sand30={row['sand_0_30']:.1f}% "
                  f"clay30={row['clay_0_30']:.1f}%", flush=True)

    df = pd.read_csv(SOIL_CSV).drop_duplicates("Station")
    SOIL_META_FILE.write_text(json.dumps({
        "source": "ISRIC SoilGrids v2.0", "url": SOIL_API,
        "queried_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "properties": SOIL_PROPERTIES, "depths_raw": SOIL_DEPTHS,
        "depths_agg": ["0-30cm (thickness-weighted)", "60-100cm"],
        "units": "sand/silt/clay in %, bdod in g/cm3",
        "stations": int(df["Station"].nunique()),
    }, indent=2))
    print(f"saved {len(df):,} stations ({n} new) -> {SOIL_CSV.name}", flush=True)


LULC_API = "https://bhuvan-app1.nrsc.gov.in/api/lulc/curljson.php"
LULC_YEARS = {"1112": "2011-12", "0506": "2005-06"}
LULC_DISTRICT_CODES = {
    "Saharanpur": "0901", "Muzaffarnagar": "0902", "Bijnor": "0903", "Moradabad": "0904",
    "Rampur": "0905", "Jyotiba Phule Nagar": "0906", "Meerut": "0907", "Baghpat": "0908",
    "Ghaziabad": "0909", "Gautam Buddha Nagar": "0910", "Bulandshahr": "0911",
    "Aligarh": "0912", "Hathras": "0913", "Mathura": "0914", "Agra": "0915",
    "Firozabad": "0916", "Etah": "0917", "Mainpuri": "0918", "Badaun": "0919",
    "Bareilly": "0920", "Pilibhit": "0921", "Shahjahanpur": "0922", "Lakhimpur Kheri": "0923",
    "Sitapur": "0924", "Hardoi": "0925", "Unnao": "0926", "Lucknow": "0927", "Rae Bareli": "0928",
    "Farrukhabad": "0929", "Kannauj": "0930", "Etawah": "0931", "Auraiya": "0932",
    "Kanpur Dehat": "0933", "Kanpur": "0934", "Jalaun": "0935", "Jhansi": "0936",
    "Lalitpur": "0937", "Hamirpur": "0938", "Mahoba": "0939", "Banda": "0940",
    "Chitrakoot": "0941", "Fatehpur": "0942", "Pratapgarh": "0943", "Kaushambi": "0944",
    "Allahabad": "0945", "Bara Banki": "0946", "Faizabad": "0947", "Ambedkar Nagar": "0948",
    "Sultanpur": "0949", "Bahraich": "0950", "Shravasti": "0951", "Balrampur": "0952",
    "Gonda": "0953", "Siddharth Nagar": "0954", "Basti": "0955", "Sant Kabir Nagar": "0956",
    "Maharajganj": "0957", "Gorakhpur": "0958", "Kushinagar": "0959", "Deoria": "0960",
    "Azamgarh": "0971", "Mau": "0962", "Ballia": "0963", "Jaunpur": "0964",
    "Ghazipur": "0965", "Chandauli": "0966", "Varanasi": "0967", "Sant Ravi Das Nagar": "0968",
    "Mirzapur": "0969", "Sonbhadra": "0970", "Kansiramnagar": "0972",
}
LULC_CLASSES = {
    "l01": "Builtup,Urban", "l02": "Builtup,Rural", "l03": "Builtup,Mining",
    "l04": "Agriculture,Crop land", "l05": "Agriculture,Plantation",
    "l06": "Agriculture,Fallow", "l07": "Agriculture, Shifting Cultivation",
    "l08": "Forest,Evergreen/Semi evergreen", "l09": "Forest,Deciduous",
    "l10": "Forest,Forest Plantation", "l11": "Forest,Scrub Forest",
    "l12": "Forest,Swamp/Mangroves", "l13": "Grass/Grazing",
    "l14": "Barren/Wastelands, Salt Affected", "l15": "Barren/Wastelands, Gullied/Ravinus",
    "l16": "Barren/Wastelands, Scrub land", "l17": "Barren/Wastelands, Sandy area",
    "l18": "Barren/Wastelands, Barren rocky", "l19": "Barren/Wastelands, Rann",
    "l20": "Wetlands/Water, Inland Wetland", "l21": "Wetlands/Water, Coastal Wetland",
    "l22": "Wetlands/Water, River/Stream/Canal", "l23": "Wetlands/Water, Reservoir/Lake/Pond",
    "l24": "Snow and Glacier",
}
LULC_GROUPS = {
    "lulc_builtup_pct": ["l01", "l02", "l03"], "lulc_cropland_pct": ["l04"],
    "lulc_plantation_pct": ["l05"], "lulc_fallow_pct": ["l06"],
    "lulc_shifting_cult_pct": ["l07"],
    "lulc_forest_pct": ["l08", "l09", "l10", "l11", "l12"],
    "lulc_grass_pct": ["l13"],
    "lulc_wasteland_pct": ["l14", "l15", "l16", "l17", "l18", "l19"],
    "lulc_water_pct": ["l20", "l21", "l22", "l23"],
}
LULC_CODE_MAP = {name.upper().replace(" ", ""): code
                 for name, code in LULC_DISTRICT_CODES.items()}
LULC_CODE_MAP.update({"BUDAUN": "0919", "SHRAWASTI": "0951"})


def lulc_read_env(repo: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    f = repo / ".env"
    if not f.exists():
        return env
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip("'\"")
    return env


def lulc_norm(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def lulc_fetch_district(name: str, code: str, year: str,
                        token: str) -> dict[str, float | str]:
    params = urllib.parse.urlencode({"year": year, "distcode": code, "token": token})
    req = urllib.request.Request(
        f"{LULC_API}?{params}", headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    total = float(data.get("totalarea") or data.get("1"))
    out: dict[str, float | str] = {"District": name, "total_area_sqkm": total}
    for i in range(1, 25):
        key = f"l{i:02d}"
        v = data.get(key) or data.get(str(i + 1))
        out[key + "_sqkm"] = float(v) if v is not None else np.nan
    return out


def lulc_main() -> None:
    token = lulc_read_env(REPO).get("LULC_STATISTICS_API_KEY", "").strip()
    if not token:
        print("MISSING LULC_STATISTICS_API_KEY in <repo>/.env", flush=True)
        return
    districts = sorted(json.loads((config.META / "selected_districts.json").read_text()))
    code_of = {lulc_norm(d): LULC_CODE_MAP.get(lulc_norm(d)) for d in districts}
    if missing := {d: c for d, c in code_of.items() if not c}:
        print(f"no Bhuvan code for: {list(missing)}", flush=True)
        return

    long_rows: list[dict] = []
    wide_rows: list[dict] = []
    n = 0
    for d in districts:
        code = code_of[lulc_norm(d)]
        for year, label in LULC_YEARS.items():
            row = lulc_fetch_district(d, code, year, token)
            total = float(row["total_area_sqkm"])
            for code_l, classname in LULC_CLASSES.items():
                a = float(row[f"{code_l}_sqkm"])
                long_rows.append({"district": d, "year": label, "class": classname,
                                  "area_sqkm": a, "pct": 100.0 * a / total if total else np.nan})
            if year == "1112":
                wide = {"District": d}
                tot = max(total, 1e-9)
                for group, codes in LULC_GROUPS.items():
                    wide[group] = 100.0 * sum(float(row[f"{c}_sqkm"]) for c in codes) / tot
                wide_rows.append(wide)
            n += 1
            print(f"[{n}] {d}: total {total:.0f} km2 ({label})", flush=True)
            time.sleep(0.3)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(config.META / "lulc_long.csv", index=False)
    wide_df = pd.DataFrame(wide_rows)
    SOIL_DIR.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(SOIL_DIR / "lulc_district.csv", index=False)
    print(f"lulc_long: {len(long_df):,} rows; lulc_district: {len(wide_df)} districts", flush=True)

    tbl = pd.read_parquet(config.ALIGNED / "table_6h.parquet")
    tbl["date"] = pd.to_datetime(tbl["time"])
    g = tbl.dropna(subset=["gwl"]).groupby("District")["gwl"]
    summ = pd.DataFrame({
        "District": [k for k, _ in g], "gwl_mean_m": [v.mean() for _, v in g],
        "gwl_median_m": [v.median() for _, v in g], "gwl_std_m": [v.std() for _, v in g],
        "gwl_amp85_m": [np.nanpercentile(v, 95) - np.nanpercentile(v, 5) for _, v in g],
    })
    corr = wide_df.merge(summ, on="District", how="inner")
    rows = []
    for c in LULC_GROUPS:
        for m in ("gwl_mean_m", "gwl_median_m", "gwl_std_m", "gwl_amp85_m"):
            x, y = corr[c].astype(float), corr[m].astype(float)
            mask = x.notna() & y.notna()
            r = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() > 8 else np.nan
            rows.append({"lulc_feature": c, "gwl_metric": m,
                         "districts": int(mask.sum()), "pearson_r": r})
    out = pd.DataFrame(rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    out.to_csv(OUT_LULC_CORR, index=False)
    print(f"correlation_lulc -> {OUT_LULC_CORR}", flush=True)
    (config.META / "lulc_meta.json").write_text(json.dumps({
        "source": "ISRO Bhuvan LULC 50K Thematic Statistics", "api": LULC_API,
        "cycles": list(LULC_YEARS.values()), "classes": len(LULC_CLASSES),
        "features_used": "2011-12 cycle (latest 50K); percent of district total area",
        "districts": len(wide_df), "key_used": "LULC_STATISTICS_API_KEY (<repo>/.env)",
        "queried_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print("done", flush=True)


OUT_LULC_CORR = ROOT / "outputs" / "correlation_lulc.csv"


# ==================================================== incremental NWIC refresh
NWIC_LIVE_RESOURCE = nwic_mod.RES_GROUNDWATER_LIVE
NWIC_DEFAULT_PQ = config.GWL_PARQUET
NWIC_RETRAIN_CHAIN = ["fetch", "normalize", "align", "align6", "features", "train"]


def nwic_coerce_to_ref(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Cast API string columns to the reference parquet dtypes before concat."""
    frame = frame.copy()
    for col, dtype in ref.dtypes.items():
        if col not in frame.columns:
            continue
        s = str(dtype).lower()
        if "int" in s and "datetime" not in s:
            num = pd.to_numeric(frame[col].astype(str).str.strip(), errors="coerce")
            if not num.isna().any():
                frame[col] = num.astype(dtype)
        elif "float" in s or "timedelta" in s:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        elif "datetime" in s:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def nwic_per_district_max(df: pd.DataFrame) -> dict:
    if df.empty or nwic_mod.TIME_FIELD not in df.columns:
        return {}
    df = df.copy()
    df[nwic_mod.TIME_FIELD] = pd.to_datetime(df[nwic_mod.TIME_FIELD], errors="coerce")
    return df.groupby(nwic_mod.DISTRICT_FIELD)[nwic_mod.TIME_FIELD].max().to_dict()


def nwic_per_station_max(df: pd.DataFrame) -> dict:
    """Per-station max: district-wide max alone starves infrequent reporters."""
    if df.empty or nwic_mod.TIME_FIELD not in df.columns:
        return {}
    df = df.copy()
    df[nwic_mod.TIME_FIELD] = pd.to_datetime(df[nwic_mod.TIME_FIELD], errors="coerce")
    return df.groupby(nwic_mod.STATION_FIELD)[nwic_mod.TIME_FIELD].max().to_dict()


def nwic_district_min_station_max(df: pd.DataFrame) -> dict:
    """Earliest station-max per district = safe scan start for every reporter."""
    if df.empty or nwic_mod.TIME_FIELD not in df.columns:
        return {}
    df = df.copy()
    df[nwic_mod.TIME_FIELD] = pd.to_datetime(df[nwic_mod.TIME_FIELD], errors="coerce")
    df = df.dropna(subset=[nwic_mod.TIME_FIELD])
    if df.empty:
        return {}
    st_max = df.groupby(nwic_mod.STATION_FIELD)[nwic_mod.TIME_FIELD].max()
    st_dist = df.groupby(nwic_mod.STATION_FIELD)[nwic_mod.DISTRICT_FIELD].agg(
        lambda s: s.value_counts().index[0])
    per_d: dict = {}
    for station, district in st_dist.items():
        t = st_max[station]
        per_d[district] = min(per_d[district], t) if district in per_d else t
    return per_d


def nwic_refresh(districts: list[str], parquet_path: Path, dry_run: bool = False,
                 sleep: float = 0.2, lookback_days: int = 45):
    """Fetch + merge the newest per-station rows; return (frames, n_new, source_max)."""
    cur = pd.read_parquet(parquet_path)
    cur_max_dist = nwic_per_district_max(cur)
    cur_max_st = nwic_per_station_max(cur)
    cur_min_st = nwic_district_min_station_max(cur)
    new_frames: list[pd.DataFrame] = []
    total_new, source_max = 0, None

    for d in districts:
        dist_max, slowest = cur_max_dist.get(d), cur_min_st.get(d)
        if slowest is not None:
            scan_from = (slowest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        elif dist_max is not None:
            scan_from = (dist_max - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        else:
            scan_from = "2021-01-01"
        try:
            records = nwic_mod.fetch_district(NWIC_LIVE_RESOURCE, d, start=scan_from,
                                              end=None, limit=5000, sleep=sleep)
        except RuntimeError as e:
            print(f"  [FAIL] {d}: {e}")
            continue
        if not records:
            print(f"  [ok]  {d}: no rows in scan window (up to {dist_max})")
            continue
        frame = nwic_mod.normalize_records(records)
        if frame.empty:
            print(f"  [ok]  {d}: no rows")
            continue
        if nwic_mod.TIME_FIELD in frame.columns:
            smax = pd.to_datetime(frame[nwic_mod.TIME_FIELD], errors="coerce").max()
            if pd.notna(smax):
                source_max = smax if source_max is None else max(source_max, smax)
        if nwic_mod.TIME_FIELD in frame.columns and nwic_mod.STATION_FIELD in frame.columns:
            frame[nwic_mod.TIME_FIELD] = pd.to_datetime(frame[nwic_mod.TIME_FIELD],
                                                        errors="coerce")
            st_max = frame[nwic_mod.STATION_FIELD].map(cur_max_st)
            frame = frame[st_max.isna() | (frame[nwic_mod.TIME_FIELD] > st_max)]
        if frame.empty:
            print(f"  [ok]  {d}: no new rows (per-station up to date, scan since {scan_from})")
            continue
        new_frames.append(frame)
        total_new += len(frame)
        print(f"  [new] {d}: +{len(frame)} rows")
        time.sleep(sleep)

    if dry_run or not new_frames:
        if not dry_run:
            print("\n  No new rows to merge; dataset is current.")
        return new_frames, total_new, source_max

    new = pd.concat(new_frames, ignore_index=True)
    station_district = (cur.dropna(subset=[nwic_mod.STATION_FIELD, nwic_mod.DISTRICT_FIELD])
                        .groupby(nwic_mod.STATION_FIELD)[nwic_mod.DISTRICT_FIELD]
                        .agg(lambda s: s.value_counts().index[0]).to_dict())
    if nwic_mod.DISTRICT_FIELD in new.columns:
        new[nwic_mod.DISTRICT_FIELD] = (new[nwic_mod.STATION_FIELD].map(station_district)
                                        .fillna(new[nwic_mod.DISTRICT_FIELD]))
    new = nwic_coerce_to_ref(new, cur)
    merged = pd.concat([cur, new], ignore_index=True)
    keys = [nwic_mod.DISTRICT_FIELD, nwic_mod.STATION_FIELD, nwic_mod.TIME_FIELD]
    merged = merged.drop_duplicates(subset=[k for k in keys if k in merged.columns])
    merged = merged.sort_values([nwic_mod.STATION_FIELD, nwic_mod.TIME_FIELD]).reset_index(drop=True)
    shutil.copy2(parquet_path, str(parquet_path) + ".bak.refresh")
    new.to_parquet(str(parquet_path.parent / (parquet_path.stem + "_new_rows.parquet")), index=False)
    merged.to_parquet(parquet_path, index=False)
    print(f"\n  Merged {total_new} new rows -> {parquet_path} ({len(merged)} total)")
    return new_frames, total_new, source_max


def nwic_deploy_check(parquet_path: Path) -> dict:
    import pyarrow.parquet as pq

    ts = pd.to_datetime(pq.ParquetFile(parquet_path)
                        .read(columns=[nwic_mod.TIME_FIELD]).to_pandas()[nwic_mod.TIME_FIELD])
    if ts.empty:
        return {"rows": 0, "has_2026": False}
    yr = ts.dt.year.value_counts().sort_index()
    return {"rows": int(len(ts)), "min": ts.min().normalize(), "max": ts.max().normalize(),
            "has_2026": bool((ts >= pd.Timestamp("2026-01-01")).any()),
            "by_year": {int(k): int(v) for k, v in yr.items()}}


def nwic_deploy_install(candidate: Path, dest: Path) -> None:
    today = pd.Timestamp.now().normalize()
    new = nwic_deploy_check(candidate)
    print(f"Candidate     : {candidate}")
    print(f"  rows={new.get('rows')}  {new.get('min')} .. {new.get('max')}  "
          f"has_2026={new.get('has_2026')}")
    if not new.get("has_2026"):
        print(f"[!] Candidate ends {new['max'].date()} — also stale. Refusing to install.")
        sys.exit(1)
    shutil.copy2(dest, str(dest) + ".bak.deploy")
    shutil.copy2(candidate, dest)
    print(f"Done. {new.get('rows'):,} rows, coverage through {new['max'].date()} "
          f"({(today - new['max']).days} days behind today).")


def nwic_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipelines.data nwic", description="NWIC archive refresh")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parquet", type=Path, default=NWIC_DEFAULT_PQ)
    ap.add_argument("--deploy-check", action="store_true")
    ap.add_argument("--deploy-install", type=Path, default=None, metavar="PATH")
    args = ap.parse_args(list(sys.argv[2:] if argv is None else argv))

    if not args.parquet.exists():
        print(f"[!] No dataset at {args.parquet}")
        return 1
    if args.deploy_check:
        info = nwic_deploy_check(args.parquet)
        print(f"Loaded build : {args.parquet}")
        print(f"  rows={info.get('rows')}  {info.get('min')} .. {info.get('max')}")
        print(f"  has_2026    : {info.get('has_2026')}")
        print(f"  by_year     : {info.get('by_year')}")
        return 0 if info.get("has_2026") else 1
    if args.deploy_install is not None:
        nwic_deploy_install(args.deploy_install, args.parquet)
        return 0

    districts = nwic_mod.get_aquis_districts(args.parquet)
    print(f"Districts to refresh ({len(districts)}): {', '.join(districts[:8])} ...")
    _, total_new, _ = nwic_refresh(districts, args.parquet, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\nDRY RUN: {total_new} new rows would be added.")
        return 0
    if args.retrain and total_new > 0:
        print("\nRetraining pooled model against refreshed data ...")
        stale_gwl = config.SELECTED / "gwl_selected.csv"
        if stale_gwl.exists():
            stale_gwl.unlink()
            print(f"  removed stale {stale_gwl.name}")
        for step in NWIC_RETRAIN_CHAIN:
            print(f"  == {step} ==", flush=True)
            rc = subprocess.call([sys.executable, "-u", "-m", "pipelines.data", step], cwd=ROOT)
            if rc != 0:
                print(f"  [FAIL] {step} exit code {rc}")
                return rc
        print("Retrain chain complete.")
    return 0


def nwic_refresh_stage() -> None:
    """Stage-form entry point (no argparse flags)."""
    nwic_main([])


# ================================================================ batch CLI
STAGES = {
    "probe": ("pipelines.data", "probe_main"),
    "select": ("pipelines.data", "select_main"),
    "fetch": ("pipelines.data", "fetch_main"),
    "normalize": ("pipelines.data", "normalize_main"),
    "align": ("pipelines.data", "align_main"),
    "align6": ("pipelines.data", "align6_main"),
    "climatology": ("pipelines.data", "climatology_main"),
    "soil": ("pipelines.data", "soil_main"),
    "lulc": ("pipelines.data", "lulc_main"),
    "nwic": ("pipelines.data", "nwic_refresh_stage"),
    "cwc_forecast": ("pipelines.data", "cwc_main"),
    "features": ("pipelines.model", "features_main"),
    "correlate": ("pipelines.model", "correlate_main"),
    "train": ("pipelines.model", "train_main"),
    "evaluate": ("pipelines.model", "evaluate_main"),
    "features6h": ("pipelines.model", "features6h_main"),
    "ablate": ("pipelines.model", "ablate_main"),
    "quantile": ("pipelines.model", "quantile_main"),
    "diagnostics": ("pipelines.model", "diagnostics_main"),
    "fleet": ("pipelines.model", "fleet_main"),
}
ALIASES = {
    "00_probe": "probe", "01_select": "select", "02_fetch_selected": "fetch",
    "03_merge_normalize": "normalize", "04_align": "align", "04b_align_6h": "align6",
    "05_correlate": "correlate", "06_features": "features", "06_train": "train",
    "07_evaluate": "evaluate", "07_soil": "soil", "08_lulc": "lulc",
    "10_ablate": "ablate", "11_fleet": "fleet", "11_quantile": "quantile",
    "12_diagnostics": "diagnostics", "13_refresh_nwic": "nwic",
    "14_6h_features": "features6h", "16_future_drivers": "climatology",
    "18_cwc_river_forecast": "cwc_forecast",
}


def run_stage(stage: str) -> None:
    """Import lazily so a single stage does not load the whole pipeline."""
    from importlib import import_module

    key = ALIASES.get(stage, stage)
    if key not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from: {', '.join(STAGES)}")
    module, func = STAGES[key]
    # argparse-based stages (e.g. fleet) read sys.argv directly. Hide the stage
    # token so they see only their own flags, exactly as when run standalone.
    saved_argv = sys.argv
    if len(saved_argv) > 1 and saved_argv[1] == stage:
        sys.argv = [saved_argv[0]] + saved_argv[2:]
    try:
        getattr(import_module(module), func)()
    finally:
        sys.argv = saved_argv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "nwic":
        return nwic_main(argv[1:])
    if not argv or argv[0] in ("-h", "--help", "--list"):
        print("stages: " + ", ".join(STAGES))
        print("aliases: " + ", ".join(ALIASES))
        return 0
    run_stage(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
