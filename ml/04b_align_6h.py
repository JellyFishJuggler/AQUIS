"""04b_align_6h — 6-hourly station×time track (production-consistent cadence).

The production pipeline models on a 6-hourly telemetry grid (4 steps/day); the
GWL resource is natively 6-hourly and the drivers are finer (1-min rain,
15-min weather), so everything re-bins onto a 00/06/12/18 UTC grid.

Uses the SAME gauge->well associations as 04 (data/meta/assoc_*.csv):
  * rainfall   : existing IDW (1/d^2) links -> 6h gauge sums blended per well
  * weather    : nearest-gauge links -> 6h mean of that gauge
  * river/canal: district-proxy / nearest links -> 6h mean

Outputs:
  ml/data/aligned/table_6h.parquet   (Station, District, time, gwl, drivers)
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

import config  # noqa: E402

_align = importlib.import_module("04_align")
haversine = _align.haversine
top_k = _align.top_k
nearest = _align.nearest
tehsil_assoc = _align.tehsil_assoc
load_norm = _align.load_norm
NEAREST_SOURCES = _align.NEAREST_SOURCES
NEAREST_WITH_DIST = _align.NEAREST_WITH_DIST

STEP = pd.Timedelta("6h")
MAX_FFILL_STEPS = 12  # carry a reading forward across gaps <= 3 days


def bin6(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["bin"] = df["time"].dt.floor(STEP)
    return df.dropna(subset=["bin"])


def load_assoc(name: str) -> pd.DataFrame:
    p = Path(config.META) / f"assoc_{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def idw_join_bin(assoc: pd.DataFrame, binned: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if assoc.empty or binned.empty:
        return pd.DataFrame(columns=["Station", "bin", value_col])
    m = assoc.merge(binned, left_on="src_station", right_on="Station", how="left")
    m = m.dropna(subset=["bin", "value"])
    m["w"] = 1.0 / np.maximum(m["dist_km"], 0.05) ** 2
    g = m.groupby(["gwl_station", "bin"]).apply(
        lambda x: pd.Series({
            value_col: float(np.average(x["value"], weights=x["w"])),
            f"{value_col}_n": int(x["src_station"].nunique()),
        }), include_groups=False)
    return g.reset_index().rename(columns={"gwl_station": "Station"})


def nearest_join_bin(assoc: pd.DataFrame, binned: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if assoc.empty or binned.empty:
        return pd.DataFrame(columns=["Station", "bin", value_col])
    m = assoc.merge(binned, left_on="src_station", right_on="Station", how="left")
    m = m.dropna(subset=["bin", "value"])
    g = m.groupby(["gwl_station", "bin"], as_index=False).agg(**{value_col: ("value", "mean")})
    return g.rename(columns={"gwl_station": "Station"})


def gwl_track(gwl: pd.DataFrame) -> pd.DataFrame:
    """Regular 6h grid per station; sentinels dropped; median per slot; a missing
    slot is carried from the last reading only for gaps <= MAX_FFILL_STEPS."""
    gwl = gwl[gwl["value"].abs() <= 500]
    frames = []
    for st, sub in gwl.groupby("Station"):
        sub = sub.sort_values("time")
        lo = sub["time"].min().floor(STEP)
        hi = sub["time"].max().floor(STEP)
        idx = pd.date_range(lo, hi, freq=STEP)
        slot = (pd.Series(sub["value"].values, index=pd.to_datetime(sub["time"]).values)
                .groupby(lambda t: pd.Timestamp(t).floor(STEP)).median())
        s = slot.reindex(idx)
        s = s.ffill(limit=MAX_FFILL_STEPS)
        frames.append(pd.DataFrame({"Station": st, "time": idx, "gwl": s.values}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    gwl = load_norm("gwl")
    if gwl.empty:
        raise SystemExit("gwl missing")

    anchor = (gwl.drop_duplicates("Station").set_index("Station")
              [["District", "Tehsil", "lat", "lon"]].reset_index())
    print(f"GWL anchors: {len(anchor):,} stations", flush=True)

    track = gwl_track(gwl)
    track = track.merge(anchor[["Station", "District"]], on="Station", how="left")
    print(f"gwl track: {len(track):,} rows / {track['Station'].nunique():,} stations", flush=True)

    # Rainfall via the same IDW links; gauge 6h sums blended per well.
    rain = load_norm("rainfall").dropna(subset=["lat", "lon"])
    if not rain.empty:
        ra = bin6(rain)
        r6 = ra.groupby(["Station", "bin"], as_index=False)["value"].sum()
        assoc = load_assoc("rainfall")
        if assoc.empty:
            assoc = top_k(anchor, ra.drop_duplicates("Station")[["Station", "lat", "lon"]],
                          config.RAIN_RADIUS_KM, config.RAIN_IDW_NEIGHBOURS, "rainfall")
        rj = idw_join_bin(assoc, r6, "rain")
        track = track.merge(rj, left_on=["Station", "time"], right_on=["Station", "bin"],
                            how="left").drop(columns=["bin"])
        print(f"  rainfall: {len(rj):,} well-6h rows", flush=True)

    for name, (col, radius) in NEAREST_SOURCES.items():
        src = load_norm(name)
        if src.empty:
            print(f"  [skip] {name}: no normalized data", flush=True)
            continue
        s = bin6(src)
        s6 = s.groupby(["Station", "bin"], as_index=False)["value"].mean()
        assoc = load_assoc(name)
        if assoc.empty:
            s_coord = s.dropna(subset=["lat", "lon"])
            if s_coord.empty:
                assoc = tehsil_assoc(anchor, s, name)
            else:
                rad = radius if radius is not None else config.WEATHER_RADIUS_KM
                assoc = nearest(anchor, s_coord.drop_duplicates("Station")
                                [["Station", "lat", "lon"]], rad, name)
        j = nearest_join_bin(assoc, s6, col)
        if not j.empty:
            track = track.merge(j, left_on=["Station", "time"], right_on=["Station", "bin"],
                                how="left").drop(columns=["bin"])
            print(f"  {name}: {len(j):,} well-6h rows", flush=True)

    track = track.sort_values(["Station", "time"]).reset_index(drop=True)
    out = config.ALIGNED / "table_6h.parquet"
    track.to_parquet(out, index=False)
    span = track["time"].max() - track["time"].min()
    print(f"\ntable_6h: {len(track):,} rows / {track['Station'].nunique():,} stations / "
          f"span {track['time'].min():%Y-%m-%d}..{track['time'].max():%Y-%m-%d} ({span.days}d)",
          flush=True)
    print(f"saved -> {out.name}", flush=True)


if __name__ == "__main__":
    main()