"""22_association_audit — what the current joins do, vs what the new design would.

READ-ONLY. Writes nothing to `data/aligned/`, `data/features*` or `models/`; its
only output is `outputs/association_audit.json`. It rebuilds the gauge-to-well
links under the new `_association` rules in memory and compares them with the links
already persisted in `data/meta/assoc_*.csv`, so the coverage gain is visible
BEFORE committing to an alignment re-run.

Numbering note: this lives in the free 21-29 utility block deliberately. It was
briefly named `30_association_audit.py`, which collided with the trajectory chain
(`30_traj_datasets.py`, `31_train_traj.py`) and sorted immediately before it —
exactly the ambiguity that gets the wrong script run in a ten-step sequence.

Prints, per parameter:
    currently linked anchors   (from the existing assoc_*.csv)
    new linked anchors         (k-nearest, no radius cliff)
    anchors with NO gauge in radius now, and whether the new rule rescues them
    median weight sum          (how much information a well actually gets)
    median nearest distance

Run from ml/:
    venv/bin/python 22_association_audit.py
    venv/bin/python 22_association_audit.py --only rainfall river_level temperature
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _association as A

ROOT = Path(__file__).resolve().parent
META = ROOT / "data" / "meta"
RAW = ROOT / "data" / "raw"

GAUGE_SOURCES = [
    "rainfall", "temperature", "humidity", "solar", "wind_speed",
    "wind_direction", "pressure", "canal_level", "river_level",
]


def anchors() -> pd.DataFrame:
    """GWL stations with usable coordinates (lat/lon), as `04_align` builds them."""
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
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def gauges_for(key: str) -> pd.DataFrame:
    """Gauge coordinates for a parameter, from its normalised raw table.

    Readings live in per-district files, so the same gauge appears many times —
    dedupe before building a graph or the nearest-neighbour search sees phantom
    duplicates. A `(0, 0)` coordinate is a null that survives `dropna` and would
    otherwise produce an ~8,000 km link with a near-zero weight, so it is dropped
    explicitly.
    """
    p = RAW / f"{key}_norm.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p, columns=["Station", "lat", "lon"])
    except Exception:  # noqa: BLE001 — column set may differ
        try:
            df = pd.read_parquet(p)
        except Exception:
            return pd.DataFrame()
        if not {"Station", "lat", "lon"} <= set(df.columns):
            return pd.DataFrame()
        df = df[["Station", "lat", "lon"]]
    return _clean_coords(df)


def _clean_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing / out-of-range / null-island coordinates, then dedupe."""
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    out = df.assign(lat=lat, lon=lon).dropna(subset=["lat", "lon"])
    out = out[~((out["lat"].abs() < 1e-6) & (out["lon"].abs() < 1e-6))]
    out = out[(out["lat"].between(-90, 90)) & (out["lon"].between(-180, 180))]
    return out.drop_duplicates("Station").reset_index(drop=True)


def existing_links(key: str) -> pd.DataFrame:
    """Links persisted by the current pipeline (`assoc_<key>.csv`)."""
    p = META / f"assoc_{key}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of parameter keys")
    ap.add_argument("--out", default="outputs/association_audit.json")
    args = ap.parse_args()

    anc = anchors()
    keys = args.only or list(A.SPECS)
    print(f"anchors with coordinates: {len(anc):,}")
    print(f"{'parameter':16s} {'method':15s} {'gauges':>7s} {'old':>6s} "
          f"{'new':>6s} {'beyond-radius':>13s} "
          f"{'medSum':>8s} {'medDmin':>8s}")
    print("-" * 96)

    report = []
    for key in keys:
        g = gauges_for(key)
        old = existing_links(key)
        old_n = (old["gwl_station"].nunique()
                 if not old.empty and "gwl_station" in old.columns else 0)

        if g.empty:
            why = ("no *_norm.parquet" if not (RAW / f"{key}_norm.parquet").exists()
                   else "gauge table has no usable coordinates")
            print(f"{key:16s} {A.spec(key).method:15s} {'—':>7s} {old_n:>6d} "
                  f"{'n/a':>6s} {'—':>13s} {'—':>8s} {'—':>8s}"
                  f"   ({why})")
            report.append({"parameter": key, "gauges": 0, "old_linked": old_n,
                           "new_linked": None, "reason": why})
            continue

        links = A.k_nearest(anc, g, key)
        cov = A.coverage_summary(links, len(anc), key)
        beyond = int(((links["rank"] == 1) & (links["n_in_radius"] == 0)).sum())
        dmin = cov["median_dmin_km"]
        dmin_s = f"{dmin:>8.1f}" if dmin is not None else f"{'—':>8s}"

        print(f"{key:16s} {A.spec(key).method:15s} {len(g):>7d} {old_n:>6d} "
              f"{cov['linked']:>6d} {beyond:>13d} "
              f"{cov['median_wsum']:>8.3f} {dmin_s}")
        report.append({"parameter": key, "gauges": int(len(g)),
                       "old_linked": int(old_n), "new_linked": int(cov["linked"]),
                       "new_linked_pct": cov["linked_pct"],
                       "no_gauge_in_radius_now": beyond,
                       "median_wsum": cov["median_wsum"],
                       "median_dmin_km": dmin,
                       "method": cov["method"]})

    print("-" * 96)
    print("old  = anchors with a persisted assoc_*.csv link (current pipeline)")
    print("new  = anchors the k-nearest rule links (no radius cliff)")
    print("beyond-radius = anchors whose k nearest all sit BEYOND the radius;")
    print("                the current pipeline silently gives these nothing.")
    print("medSum = median over anchors of the RAW kernel sum exp(-(d/L)^2):")
    print("         ~k for k close gauges, <<1 when only distant ones exist.")
    print("         (The sum of the normalised `weight` column is always 1.0")
    print("         by construction and would hide the shortfall entirely.)")

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()