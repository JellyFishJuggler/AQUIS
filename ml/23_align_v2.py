"""23_align_v2 — prototype 6 h alignment using the new `_association` layer.

**Non-destructive by design.** It does NOT touch `04_align` / `04b_align_6h` or
`data/aligned/table_6h.parquet`, and it does NOT overwrite the persisted
`assoc_*.csv` links. It writes to `data/aligned_v2/` and `data/meta/assoc_v2_*.csv`
so the canonical pipeline keeps working while the new association is evaluated.

Numbering note: this lives in the free 21-29 utility block deliberately. It was
briefly named `31_align_v2.py`, which collided with the trajectory chain
(`30_traj_datasets.py`, `31_train_traj.py`, `32_backtest_traj.py`) and made the
ten-step sequence ambiguous to read.

What it does differently from `04b_align_6h`:

  * k-nearest + Gaussian distance weighting, **no radius cliff** — a well whose
    gauges are all far away gets a small weight and an explicit coverage column
    instead of silently getting nothing;
  * `weighted_delta` for datum-relative river/canal levels — the batch path
    currently transplants absolute levels across gauges that are on different
    local datums, which is physically meaningless and also disagrees with the
    serving path (`refresh/future_drivers.py`);
  * per-parameter coverage columns (`_n`, `_dmin_km`, `_wsum`) on every driver,
    not just rainfall;
  * training-era-only normalisation statistics, written to a sidecar JSON so the
    serving path can reuse exactly the same numbers.

Outputs:
    data/aligned_v2/table_6h_v2.parquet        station x 6h, new driver columns
    data/meta/assoc_v2_<param>.csv             the links actually used
    data/meta/assoc_v2_report.json             per-parameter coverage + norms
    data/meta/norm_stats_v2.json               training-era normalisation stats

Run from ml/:
    venv/bin/python 23_align_v2.py                 # all parameters
    venv/bin/python 23_align_v2.py --only rainfall river_level
    venv/bin/python 23_align_v2.py --dry-run       # links + coverage, no table
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _association as A

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "meta"
OUT = ROOT / "data" / "aligned_v2"

STEP_H = 6
TRAIN_END = pd.Timestamp("2025-10-01")      # mirrors 06_features.VAL_START

# Parameters to associate. `gwl` itself is the observation, not a driver.
DRIVERS = ["rainfall", "temperature", "humidity", "solar", "wind_speed",
           "pressure", "river_level", "canal_level"]

# Output column names, kept aligned with the existing feature schema so a
# switch-over is a column substitution rather than a schema change.
OUT_COL = {
    "rainfall": "rain", "temperature": "temp", "humidity": "humidity",
    "solar": "solar", "wind_speed": "wind_speed", "pressure": "pressure",
    "river_level": "river_level", "canal_level": "canal_level",
}


def load_anchors() -> pd.DataFrame:
    p = META / "selected_gwl_stations.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}; run 01_select.py first")
    df = pd.read_csv(p)
    ren = {}
    for c in df.columns:
        if c.lower() == "latitude":
            ren[c] = "lat"
        elif c.lower() == "longitude":
            ren[c] = "lon"
    df = df.rename(columns=ren)
    if "lat" not in df or "lon" not in df:
        raise SystemExit("selected_gwl_stations.csv has no Latitude/Longitude columns")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    keep = [c for c in ("Station", "District", "lat", "lon") if c in df.columns]
    return df[keep].reset_index(drop=True)


def load_norm(key: str) -> pd.DataFrame:
    """Normalised long driver table for a parameter (Station, time, value, lat, lon)."""
    p = RAW / f"{key}_norm.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if not {"Station", "time", "value"} <= set(df.columns):
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value", "Station"])
    # 6h bins, matching 04b_align_6h
    df["time"] = df["time"].dt.floor(pd.Timedelta(hours=STEP_H))
    return df


def gauge_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Unique gauge coordinates.

    Readings arrive in per-district normalised files, so one gauge appears many
    times — dedupe before the nearest-neighbour search. Missing, out-of-range and
    `(0, 0)` "null island" coordinates are dropped: the last would otherwise pass
    `dropna` and produce a ~8,000 km link with a near-zero weight.
    """
    if not {"lat", "lon"} <= set(df.columns):
        return pd.DataFrame()
    out = df[["Station", "lat", "lon"]].copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    out = out[~((out["lat"].abs() < 1e-6) & (out["lon"].abs() < 1e-6))]
    out = out[out["lat"].between(-90, 90) & out["lon"].between(-180, 180)]
    return out.drop_duplicates("Station").reset_index(drop=True)


def bin_gauge(agg: str, df: pd.DataFrame) -> pd.DataFrame:
    """Collapse readings to one value per (gauge, 6 h bin)."""
    how = {"sum": "sum", "mean": "mean", "median": "median"}.get(agg, "mean")
    g = df.groupby(["Station", "time"], as_index=False)["value"]
    return g.agg(how)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build links + coverage only; do not write the table")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    anc = load_anchors()
    anchors_xy = anc.dropna(subset=["lat", "lon"])
    print(f"anchors: {len(anc):,} ({len(anchors_xy):,} with coordinates)")

    keys = args.only or DRIVERS
    report: list[dict] = []
    norm_stats: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []

    for key in keys:
        s = A.spec(key)
        src = load_norm(key)
        if src.empty:
            print(f"  [skip] {key}: no normalized table")
            report.append({"parameter": key, "status": "no normalized table"})
            continue
        coords = gauge_coords(src)
        if coords.empty:
            print(f"  [skip] {key}: no gauge coordinates "
                  f"(needs the tehsil/district proxy path instead)")
            report.append({"parameter": key, "status": "no gauge coordinates",
                           "method": s.method})
            continue

        links = A.k_nearest(anchors_xy, coords, key)
        cov = A.coverage_summary(links, len(anchors_xy), key)
        links.rename(columns={"anchor_station": "gwl_station"}).to_csv(
            META / f"assoc_v2_{key}.csv", index=False)

        # aggregate per gauge into 6 h bins, using the parameter's natural agg
        agg = "sum" if key in ("rainfall", "eto") else "mean"
        binned = bin_gauge(agg, src)
        combined = A.combine(links, binned, key, window_steps=120)
        print(f"  {key:14s} {s.method:15s} gauges={len(coords):3d} "
              f"linked={cov['linked']:4d}/{len(anchors_xy)} "
              f"in_radius={cov['anchors_with_gauge_in_radius']:4d} "
              f"wsum={cov['median_wsum']:.3f} -> {len(combined):,} rows")

        if not combined.empty:
            col = OUT_COL.get(key, key)
            part = combined.rename(columns={
                "value": col, "n": f"{col}_n", "dmin_km": f"{col}_dmin_km",
                "wsum": f"{col}_wsum"})
            part = part[["Station", "time", col, f"{col}_n", f"{col}_dmin_km",
                         f"{col}_wsum"]]
            frames.append(part)

            # training-era normalisation statistics, computed ONCE here so the
            # serving path can reuse identical numbers (no per-call refitting).
            ser = pd.Series(part[col].to_numpy(), index=part["time"])
            stats = A.training_stats(ser, cutoff=TRAIN_END)
            stats["method"] = A.normalisation_for(key)
            norm_stats[key] = stats

        report.append({"parameter": key, **cov, "agg": agg,
                       "status": "ok", "rows": int(len(combined))})

    (META / "assoc_v2_report.json").write_text(json.dumps(report, indent=2))
    (META / "norm_stats_v2.json").write_text(json.dumps(norm_stats, indent=2))
    print(f"\nwrote {META / 'assoc_v2_report.json'} and norm_stats_v2.json")

    if args.dry_run:
        print("dry run — table not written")
        print(f"elapsed {time.time() - t0:.0f}s")
        return
    if not frames:
        print("no driver frames were built — nothing to write")
        return

    tbl = frames[0]
    for f in frames[1:]:
        tbl = tbl.merge(f, on=["Station", "time"], how="outer")
    tbl = tbl.sort_values(["Station", "time"]).reset_index(drop=True)
    out = OUT / "table_6h_v2.parquet"
    tbl.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(tbl):,} rows x {tbl.shape[1]} cols)")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
