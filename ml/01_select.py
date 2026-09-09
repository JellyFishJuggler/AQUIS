"""01_select — choose the "live" (full-2026) GWL station set.

Rule (config.FULL26_*):
  * has >= 1 record in >= 8 of the 9 months Jan..Sep 2026
  * first 2026 record <= FULL26_FIRST_MAX (mid-Jan)
  * last 2026(overall) record >= FULL26_LAST_MIN (>= Aug 2026)

Outputs:
  ml/data/meta/selected_gwl_stations.csv   — per-station record + coords
  ml/data/meta/selected_districts.json     — districts of the selected set
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config

COLS = ["Station", "District", "Tehsil", "Block", "Latitude", "Longitude",
        "Data Acquisition Time"]


def load_gwl() -> pd.DataFrame:
    df = pd.read_parquet(config.GWL_PARQUET, columns=COLS)
    df["Data Acquisition Time"] = pd.to_datetime(df["Data Acquisition Time"], errors="coerce")
    df = df.dropna(subset=["Data Acquisition Time"])
    return df


def full26_flags(df: pd.DataFrame) -> pd.DataFrame:
    t = df["Data Acquisition Time"]
    g = df.groupby("Station")

    per = df.assign(
        year=t.dt.year,
        month=t.dt.month,
        d2026=t.dt.year == 2026,
    )

    months = (
        per[per["d2026"]]
        .groupby("Station")["month"]
        .apply(lambda s: s.drop_duplicates().nunique())
    )
    first2026 = per[per["d2026"]].groupby("Station")["Data Acquisition Time"].min()
    last_all = per.groupby("Station")["Data Acquisition Time"].max()

    idx = pd.Index(per["Station"].drop_duplicates())
    out = pd.DataFrame(index=idx)
    out["n_2026_months"] = months.reindex(idx)
    out["first_2026"] = first2026.reindex(idx)
    out["last_record"] = last_all.reindex(idx)
    out["n_2026_months"] = out["n_2026_months"].fillna(0).astype(int)
    out["full26"] = (
        (out["n_2026_months"] >= config.FULL26_MIN_MONTHS)
        & (per.groupby("Station")["d2026"].sum().reindex(idx) > 0)
        & (pd.to_datetime(out["first_2026"]).dt.date.astype(str) <= config.FULL26_FIRST_MAX)
        & (pd.to_datetime(out["last_record"]).dt.date.astype(str) >= config.FULL26_LAST_MIN)
    )
    out["n_2026_records"] = per[per["d2026"]].groupby("Station").size().reindex(idx).fillna(0).astype(int)
    return out


def main() -> None:
    print("loading GWL parquet ...")
    df = load_gwl()
    print(f"  raw rows={len(df):,} stations={df['Station'].nunique():,} districts={df['District'].nunique()}")

    flags = full26_flags(df)

    st_meta = (
        df.drop_duplicates("Station")
        .set_index("Station")[["District", "Tehsil", "Block", "Latitude", "Longitude"]]
    )
    sel = st_meta.join(flags).reset_index()
    sel = sel[["Station", "District", "Tehsil", "Block", "Latitude", "Longitude",
               "n_2026_months", "n_2026_records", "first_2026", "last_record", "full26"]]
    sel["first_2026"] = pd.to_datetime(sel["first_2026"])
    sel["last_record"] = pd.to_datetime(sel["last_record"])

    keep = sel[sel["full26"]].copy()
    districts = sorted(keep["District"].dropna().unique().tolist())

    sel.to_csv(config.META / "gwl_full_flags.csv", index=False)
    keep.to_csv(config.META / "selected_gwl_stations.csv", index=False)
    (config.META / "selected_districts.json").write_text(json.dumps(districts, indent=2))

    # stale-coverage watchlist: stations with 2026 telemetry that failed the live
    # gate, bucketed by why (stale vs sparse vs late start). Coarse gaps like
    # telemetry going quiet mid-year show up here even though the station had a
    # healthy 2025 archive.
    watch = sel[(sel["full26"] == False) & (sel["n_2026_months"] >= 1)].copy()  # noqa: E712
    if not watch.empty:
        wbreak = pd.to_datetime(config.FULL26_LAST_MIN)
        wstart = pd.to_datetime(config.FULL26_FIRST_MAX)
        watch["reason"] = "other"
        watch.loc[watch["last_record"] < wbreak, "reason"] = "telemetry_stale"
        watch.loc[(watch["reason"] == "other") & (watch["n_2026_months"] < config.FULL26_MIN_MONTHS),
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


if __name__ == "__main__":
    main()