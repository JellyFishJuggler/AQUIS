"""04_align — build the daily station×day table with spatial association.

For every selected GWL station:
  * rainfall  : IDW (1/d^2) blend of up to 3 nearest gauges within RAIN_RADIUS_KM
                (daily gauge sums -> weighted daily value at the well)
  * weather/river/canal/reservoir: nearest gauge within its radius; river/canal
                keep distance_km for the user to judge hydrological relevance.
  * gwl       : daily mean of the 6-hourly groundwater level (metres)

Outputs:
  ml/data/aligned/table.parquet / table.csv   (Station, District, date, ...)
  ml/data/meta/assoc_<source>.csv             (which gauge fed each well)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config  # noqa: E402

R = 6371.0


def haversine(lat1, lon1, lat2, lon2) -> pd.Series:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest(gwl: pd.DataFrame, src: pd.DataFrame, radius_km: float,
            name: str) -> pd.DataFrame:
    """For each GWL station, the nearest source gauge(s) within radius."""
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
        sub = sub[["gwl_station", "src_station", "dist_km"]].sort_values("dist_km")
        sub = sub.drop_duplicates("gwl_station")
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["gwl_station", "src_station", "dist_km"])
    out.to_csv(config.META / f"assoc_{name}.csv", index=False)
    return out


# Gauge networks that do not publish coordinates (e.g. river/canal SW gauges):
# prefer exact (District, Tehsil) match; when GWL tehsils are unpublished ("-"),
# fall back to District-level (mean across the district's gauges) — a flagged proxy.
def tehsil_assoc(gwl: pd.DataFrame, src: pd.DataFrame, name: str) -> pd.DataFrame:
    g = gwl.dropna(subset=["District"])[
        ["Station", "District", "Tehsil"]].rename(columns={"Station": "gwl_station"})
    s = src.dropna(subset=["District"])[
        ["Station", "District", "Tehsil"]].rename(columns={"Station": "src_station"})

    g_ok = g.dropna(subset=["Tehsil"])
    s_ok = s.dropna(subset=["Tehsil"])
    out = (g_ok.merge(s_ok, on=["District", "Tehsil"])
           [["gwl_station", "src_station"]] if not g_ok.empty and not s_ok.empty else
           pd.DataFrame(columns=["gwl_station", "src_station"]))

    mode = "tehsil"
    if out.empty:
        out = g[["gwl_station", "District"]].merge(
            s[["src_station", "District"]], on="District")[["gwl_station", "src_station"]]
        mode = "district"
    out["dist_km"] = np.nan
    out = out.drop_duplicates(["gwl_station", "src_station"])
    out.to_csv(config.META / f"assoc_{name}.csv", index=False)
    if mode == "district":
        print(f"      (tehsils unpublished -> district-level proxy used)", flush=True)
    return out


def top_k(gwl: pd.DataFrame, src: pd.DataFrame, radius_km: float, k: int,
          name: str) -> pd.DataFrame:
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
        sub = sub[["gwl_station", "src_station", "dist_km"]].sort_values("dist_km").head(k)
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["gwl_station", "src_station", "dist_km"])
    out.to_csv(config.META / f"assoc_{name}.csv", index=False)
    return out


def idw_join(assoc: pd.DataFrame, daily: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Weighted (1/d^2) daily value per GWL station from gauge daily values."""
    if assoc.empty or daily.empty:
        return pd.DataFrame(columns=["Station", "date", value_col, f"{value_col}_n"])
    m = assoc.merge(daily, left_on="src_station", right_on="Station", how="left")
    m = m.dropna(subset=["date", "value"])
    m["w"] = 1.0 / np.maximum(m["dist_km"], 0.05) ** 2
    g = m.groupby(["gwl_station", "date"]).apply(
        lambda x: pd.Series({
            value_col: float(np.average(x["value"], weights=x["w"])),
            f"{value_col}_n": int(x["src_station"].nunique()),
        }), include_groups=False)
    g = g.reset_index().rename(columns={"gwl_station": "Station"})
    return g


def nearest_join(assoc: pd.DataFrame, daily: pd.DataFrame, value_col: str,
                 with_dist: bool = False) -> pd.DataFrame:
    if assoc.empty or daily.empty:
        out = pd.DataFrame(columns=["Station", "date", value_col])
        if with_dist:
            out[f"{value_col}_dist_km"] = []
        return out
    m = assoc.merge(daily, left_on="src_station", right_on="Station", how="left")
    m = m.dropna(subset=["date", "value"])
    g = m.groupby(["gwl_station", "date"]).agg(
        **{value_col: ("value", "mean")},
        **({f"{value_col}_dist_km": ("dist_km", "mean")} if with_dist else {}),
    ).reset_index().rename(columns={"gwl_station": "Station"})
    return g


NEAREST_SOURCES = {
    "temperature": ("temp", None), "river_level": ("river_level", config.RIVER_RADIUS_KM),
    "humidity": ("humidity", None), "solar": ("solar", None),
    "wind_speed": ("wind_speed", None), "wind_direction": ("wind_dir", None),
    "pressure": ("pressure", None), "canal_level": ("canal_level", config.CANAL_RADIUS_KM),
    "canal_discharge": ("canal_discharge", config.CANAL_RADIUS_KM),
    "res_okhla": ("res_okhla", config.CANAL_RADIUS_KM),
    "res_okhla_agra_canal": ("res_okhla_agra_canal", config.CANAL_RADIUS_KM),
    "res_matatila": ("res_matatila", config.CANAL_RADIUS_KM),
    "res_ganga_rmc": ("res_ganga_rmc", config.CANAL_RADIUS_KM),
    "res_ganga_1": ("res_ganga_1", config.CANAL_RADIUS_KM),
}

NEAREST_WITH_DIST = {"river_level", "canal_level", "canal_discharge",
                     "res_okhla", "res_okhla_agra_canal", "res_matatila",
                     "res_ganga_rmc", "res_ganga_1"}


def load_norm(name: str) -> pd.DataFrame:
    p = config.RAW / f"{name}_norm.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["date"] = df["time"].dt.normalize()
    return df.dropna(subset=["date"])


def daily(source: str, df: pd.DataFrame, agg: str) -> pd.DataFrame:
    if agg == "sum":
        return df.groupby(["Station", "date"], as_index=False)["value"].sum()
    if agg == "median":
        return df.groupby(["Station", "date"], as_index=False)["value"].median()
    return df.groupby(["Station", "date"], as_index=False)["value"].mean()


def main() -> None:
    gwl = load_norm("gwl")
    if gwl.empty:
        raise SystemExit("gwl missing — run 03 first")

    # NWIC groundwater telemetry uses -999/-999.999 as "dry/offline" sentinels and
    # occasionally a wildly wrong transducer value. |-value|>500 m is unphysical
    # for any UP aquifer (well head or depth), so drop before daily aggregation.
    sentinels = gwl["value"].abs() > 500
    if sentinels.sum():
        print(f"gwl sentinel/telemetry rows dropped: {int(sentinels.sum()):,}", flush=True)
    gwl = gwl[~sentinels]

    anchor = (
        gwl.drop_duplicates("Station")
        .set_index("Station")[["District", "Tehsil", "lat", "lon"]]
    )
    anchor = anchor.reset_index()
    print(f"GWL anchors: {len(anchor):,} stations / {anchor['District'].nunique()} districts", flush=True)

    gwl_daily = daily("gwl", gwl, "median").rename(columns={"Station": "Station", "value": "gwl"})

    tbl = gwl_daily.merge(
        anchor.rename(columns={"Station": "Station"})[["Station", "District"]],
        on="Station", how="left")

    # Rainfall via IDW
    rain = load_norm("rainfall")
    if not rain.empty:
        ra = rain.dropna(subset=["lat", "lon"])
        r_daily = daily("rainfall", ra, "sum").rename(columns={"value": "value"})
        assoc = top_k(anchor, ra.drop_duplicates("Station")[["Station", "lat", "lon"]],
                      config.RAIN_RADIUS_KM, config.RAIN_IDW_NEIGHBOURS, "rainfall")
        rj = idw_join(assoc, r_daily, "rain")
        tbl = tbl.merge(rj, on=["Station", "date"], how="left")
        print(f"  rainfall: {len(assoc):,} well-gauge links", flush=True)

    # Nearest-station sources
    for name, (col, radius) in NEAREST_SOURCES.items():
        src = load_norm(name)
        if src.empty:
            print(f"  [skip] {name}: no normalized data", flush=True)
            continue
        s = src.copy()
        s_coord = s.dropna(subset=["lat", "lon"])
        if s_coord.empty:
            assoc = tehsil_assoc(anchor, s, name)
            print(f"  {name}: tehsil-fallback ({len(assoc):,} well-gauge links)", flush=True)
        else:
            rad = radius if radius is not None else (
                config.CANAL_RADIUS_KM if name.startswith("res_") else config.WEATHER_RADIUS_KM)
            assoc = nearest(anchor, s_coord.drop_duplicates("Station")
                            [["Station", "lat", "lon"]], rad, name)
            print(f"  {name}: {len(assoc):,} well-gauge links", flush=True)
        s_daily = daily(name, s, "mean").rename(columns={"value": "value"})
        with_dist = name in NEAREST_WITH_DIST
        j = nearest_join(assoc, s_daily, col, with_dist=with_dist)
        if not j.empty:
            tbl = tbl.merge(j, on=["Station", "date"], how="left")

    tbl = tbl.sort_values(["Station", "date"]).reset_index(drop=True)

    # Persist rolling rainfall windows (1/7/30d) so analysis AND the app see the
    # same accumulated-rain features that drive correlation/forecasts.
    if "rain" in tbl.columns:
        for w in (7, 30):
            tbl[f"rain_{w}d"] = tbl.groupby("Station")["rain"].transform(
                lambda s: s.rolling(w).sum())
        tbl["rain_1d"] = tbl["rain"]

    out_p = config.ALIGNED / "table.parquet"
    out_c = config.ALIGNED / "table.csv"
    tbl.to_parquet(out_p, index=False)
    tbl.to_csv(out_c, index=False)
    print(f"\ntable: {len(tbl):,} rows / {tbl['Station'].nunique():,} stations "
          f"/ span {tbl['date'].min():%Y-%m-%d}..{tbl['date'].max():%Y-%m-%d}", flush=True)
    print(f"saved -> {out_p.name} / {out_c.name}", flush=True)


if __name__ == "__main__":
    main()