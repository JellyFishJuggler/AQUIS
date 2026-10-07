#!/usr/bin/env python
"""Diagnostic 4c follow-up — READ-ONLY on production.

(i)   Top-20 stations by flagged points under the SAME fixed rule as run.py.
(ii)  For the bucket (ii) PERMANENT LEVEL SHIFT events: check whether the
      nearest 3 neighbouring GWL stations (by great-circle distance) show a
      con-directional 6-hourly jump of >= 50% of the event's peak jump within
      +/- 6 days of the event start. A neighbour co-event => a real regional
      level shift; silence across all 3 => datum/sensor shift at this station.

Reuses run.py's load/flag_points/build_events/classify verbatim so the flagged
point set and event buckets are identical to the published 4c diagnostics.

Writes only into data/scratch_assoc/diag_4c_glitches/.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
META_CSV = ROOT / "data" / "meta" / "selected_gwl_stations.csv"

import run as C4

CO_EVENT_WIN = pd.Timedelta(days=6)
MAG_FRAC = 0.5


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def main() -> None:
    print("loading + flagging (same rule as run.py)...", flush=True)
    df = C4.load()
    df = C4.flag_points(df)
    n_flagged = int(df["flagged"].sum())
    n_stations = df["Station"].nunique()

    # ---- (i) top-20 stations by flagged points ---------------------------
    top = (
        df.loc[df["flagged"]]
        .groupby("Station", sort=False)
        .size()
        .sort_values(ascending=False)
        .head(20)
    )
    top20 = pd.DataFrame({"n_flagged_points": top.values}, index=top.index)
    top20.index.name = "station"

    # ---- events (rebuild so direction is available per row) --------------
    ev = C4.build_events(df)
    fl = df.loc[df["flagged"], ["Station", "time"]].copy()
    gap = fl.groupby("Station", sort=False)["time"].diff()
    fl["event_id"] = (gap.isna() | (gap > C4.EVENT_GAP)).groupby(
        fl["Station"], sort=False).cumsum()
    df.loc[df["flagged"], "event_id"] = fl["event_id"].to_numpy()
    ev = C4.classify(df, ev)

    # event direction / end level from the raw series
    stations_ts = {}
    for s, idx in df.groupby("Station", sort=False):
        stations_ts[s] = (idx["time"].to_numpy(dtype="datetime64[us]"),
                          idx["gwl"].to_numpy(dtype=float),
                          idx["prev_gwl"].to_numpy(dtype=float),
                          idx["jump"].to_numpy(dtype=float),
                          idx["flagged"].to_numpy(dtype=bool),
                          idx["threshold"].to_numpy(dtype=float))

    ev = ev[ev["bucket"] == C4.B_II].copy()
    print(f"bucket (ii) permanent shifts: {len(ev):,}", flush=True)

    # ---- coordinates (canonical meta list) -------------------------------
    meta = pd.read_csv(META_CSV)
    coords = {}
    for r in meta.itertuples(index=False):
        latv, lonv = r[4], r[5]
        if np.isfinite(latv) and np.isfinite(lonv):
            coords[str(r[0])] = (float(latv), float(lonv))
    names = sorted(set(ev["Station"]) | set(df["Station"].unique()))
    have = {s: coords.get(s) for s in names}
    missing = sorted(s for s in names if have.get(s) is None)
    print(f"stations with coords: {len(names) - len(missing)}/{len(names)} "
          f"(missing {len(missing)})", flush=True)
    if missing:
        print("   coords-missing stations:", missing[:10], flush=True)

    # ---- (ii) neighbour co-event check ----------------------------------
    rows = []
    cache = {}
    for k, row in enumerate(ev.itertuples(index=False)):
        s = row.Station
        base = have.get(s)
        if base is None:
            rows.append({"station": s, "n_neighbours": 0, "n_co_events": 0,
                         "co_share": np.nan, "missing_coords": True})
            continue
        t0 = np.datetime64(row.first_time.to_datetime64())
        times, gwl, prev, jump, flag, thr = stations_ts[s]
        i0 = int(np.searchsorted(times, t0, side="left"))
        i_end = min(int(np.searchsorted(times, row.last_time.to_datetime64(),
                                        side="left")), len(times) - 1)
        direction = float(np.sign(gwl[i_end] if np.isfinite(gwl[i_end]) else 0.0
                                  - prev[i0] if np.isfinite(prev[i0]) else 0.0))
        if direction == 0.0:
            direction = float(np.sign(row.peak_jump_m))
        peak = float(row.peak_jump_m)
        need = MAG_FRAC * peak

        # nearest 3 = all stations except self, by great-circle km
        others = sorted(
            (round(haversine_km(base[0], base[1], c[0], c[1]), 2), n)
            for n, c in have.items() if n != s and c is not None)
        nbrs = others[:3]

        co = 0
        for dkm, nb in nbrs:
            key = nb
            if key not in cache:
                cache[key] = stations_ts[key]
            nt, ng, np_, nj, nf, nth = cache[key]
            lo = int(np.searchsorted(nt, t0 - np.timedelta64(6 * 24, "h"), side="left"))
            hi = int(np.searchsorted(nt, t0 + np.timedelta64(6 * 24, "h"), side="right"))
            wj = nj[lo:hi]
            wd = np_.copy(); wd[lo:hi] = ng[lo:hi] - np_[lo:hi]
            wf = nf[lo:hi]
            if len(wd):
                hits = (wf & (np.sign(wd[lo:hi]) == direction) & (wj >= need)).sum()
            else:
                hits = 0
            co += int(hits)
        rows.append({
            "station": s,
            "first_time": str(row.first_time),
            "peak_jump_m": round(peak, 2),
            "direction": "+" if direction > 0 else "-",
            "n_neighbours": len(nbrs),
            "n_co_events": co,
            "co_share": (float(co) / len(nbrs)) if nbrs else np.nan,
            "nearest_km": [d for d, _ in nbrs],
            "neighbour_names": [n for _, n in nbrs],
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["co_group"] = out["n_neighbours"].where(
            out["n_neighbours"] > 0).apply(
            lambda nn: f"{out.loc[out.index, 'n_co_events']}/{nn}"
            if nn else np.nan)

    summary = {
        "diagnostic": "4c follow-up (read-only) — top stations + permanent-shift neighbour check",
        "fixed_rule": C4.RULE,
        "co_event_definition": (
            "neighbour has a flagged 6-hourly jump within +/- 6 d of the event "
            "start, same direction, |jump| >= 0.5 * event peak jump"),
        "fleet": {
            "n_rows": int(len(df)),
            "n_stations": n_stations,
            "n_flagged_points": n_flagged,
            "n_bucket_ii_permanent_shifts": int(len(ev)),
        },
        "top20_by_flagged_points": top20.reset_index().to_dict("records"),
        "permanent_shift_neighbour_check": {
            "n_events_with_coords": int((~out["n_co_events"].isna()).sum())
            if len(out) else 0,
            "group_shares": {
                f"{k}/{out.loc[k, 'n_neighbours']}": float(v / len(out))
                for k, v in out.groupby(
                    [out["n_co_events"].astype(str),
                     out["n_neighbours"].astype(str)]).size().to_dict().items()
            } if len(out) else {},
            "share_ge_one_neighbour_coevent": (
                float((out["n_co_events"] >= 1).mean()) if len(out) else np.nan),
            "share_all_three_coevent": (
                float((out["n_co_events"] >= 3).mean()) if len(out) else np.nan),
            "max_co_events_in_window": int(out["n_co_events"].max()) if len(out) else 0,
        },
        "station_listing": out.to_dict("records"),
        "varansi_note": (
            "station spelling in the aligned/meta data is 'Varansi' (one a); "
            "the diagnostic-4b/4c 'Varanasi' prefix search therefore found 0. "
            "'Varansi Harahua Gurwat' EXISTS and is included here."),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    top20.reset_index().to_csv(HERE / "followup_top20.csv", index=False)
    out.to_csv(HERE / "followup_shift_neighbours.csv", index=False)
    (HERE / "followup_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nTOP-20 stations by flagged points:")
    for r in top20.reset_index().itertuples(index=False):
        print(f"  {getattr(r, 'station'):<32s} {getattr(r, 'n_flagged_points'):>5,}")
    print(f"\npermanent shifts: {len(ev):,}; group shares of co-event neighbours:")
    if len(out):
        grp = (out["n_co_events"].astype(str) + "/" + out["n_neighbours"].astype(str))
        sh = grp.value_counts()
        for g in ("0/3", "1/3", "2/3", "3/3"):
            print(f"  co-group {g}: {sh.get(g, 0):>4}  "
                  f"{100 * float(sh.get(g, 0)) / len(out):5.1f}%")
        print("  >=1 neighbour co-event:", round(100 * float((out["n_co_events"] >= 1).mean()), 1), "%",
              "| all 3 co-event:", round(100 * float((out["n_co_events"] >= 3).mean()), 1), "%")
    print("wrote followup_top20.csv, followup_shift_neighbours.csv, followup_summary.json")


if __name__ == "__main__":
    main()