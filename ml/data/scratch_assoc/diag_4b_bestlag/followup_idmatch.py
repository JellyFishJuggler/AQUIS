#!/usr/bin/env python
"""Diagnostic 4b follow-up — six-dashboard-station rows matched by STATION ID.

Station identity = sha1(Station)[:16], the same key the dashboard archive and
app_pages/_data.py use (``_station_hash``). The six groups are seeded from the
CANONICAL meta spellings (data/meta/selected_gwl_stations.csv) so the
"Varansi" vs "Varanasi" spelling variant cannot drop a station; rows are then
selected/joined BY ID, never by name prefix.

Default algorithm identical to run.py: lags 0..45 d from the ISSUE (stride
anchor) date, issued = xgb, actual = aligned gwl, eligibility >=3 stride rows /
>=10 matched lags, best lag = argmin RMSE (ties -> smallest lag).

READ-ONLY on production; writes only into data/scratch_assoc/diag_4b_bestlag/.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREDS = ROOT / "outputs" / "predictions_2026.parquet"
ALIGNED = ROOT / "data" / "aligned" / "table_6h.parquet"
META = ROOT / "data" / "meta" / "selected_gwl_stations.csv"

LAGS = np.arange(0, 46)
MIN_ROWS_PER_STATION = 3
MIN_MATCHED_LAGS = 10

# canonical (meta-file spelling) labels and their prefix seeds
DASH_GROUPS = {
    "Azamgarh": ("Azamgarh",),
    "Deoria": ("Deoria",),
    "Banda": ("Banda",),
    "Meerut": ("Meerut",),
    "Varansi (Harahua)": ("Varansi",),
    "ASHADHA": ("ASHADHA",),
}


def station_id(s) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    # ---- canonical name -> id for the six groups -------------------------
    meta = pd.read_csv(META)
    canon = {str(r[0]): station_id(str(r[0])) for r in meta.itertuples(index=False)}

    group_ids: dict[str, list[str]] = {}
    all_ids: set[str] = set()
    for label, pfs in DASH_GROUPS.items():
        ids = [
            canon[n] for n in canon
            if any(n.lower().startswith(p.lower()) for p in pfs)
        ]
        group_ids[label] = sorted(ids)
        all_ids.update(ids)

    # ---- 1. stride rows --------------------------------------------------
    p = pd.read_parquet(PREDS, columns=["Station", "time", "xgb", "target", "persist", "stride"])
    p = p[p["stride"] == True].copy()  # noqa: E712
    p["time"] = pd.to_datetime(p["time"])
    p = p[np.isfinite(p["xgb"]) & np.isfinite(p["target"])]
    counts = p.groupby("Station").size()
    keep = counts[counts >= MIN_ROWS_PER_STATION].index
    p = p[p["Station"].isin(keep)].reset_index(drop=True)
    p["row_id"] = np.arange(len(p))

    # ---- 2. aligned actuals ---------------------------------------------
    a = pd.read_parquet(ALIGNED, columns=["Station", "time", "gwl"])
    a["time"] = pd.to_datetime(a["time"])
    a = a.dropna(subset=["gwl"]).sort_values(["Station", "time"])

    # ---- 3. best-lag mesh ------------------------------------------------
    n, nl = len(p), len(LAGS)
    mesh = pd.DataFrame({
        "row_id": np.repeat(p["row_id"].values, nl),
        "Station": np.repeat(p["Station"].values, nl),
        "lag": np.tile(LAGS, n),
        "time": (p["time"].values.astype("datetime64[s]")[:, None]
                 + np.array(LAGS, dtype="timedelta64[D]").astype("timedelta64[s]")
                 .reshape(1, nl)).ravel(),
        "issued": np.repeat(p["xgb"].values, nl),
    })
    m = mesh.merge(a.rename(columns={"time": "time", "gwl": "actual"}),
                   on=["Station", "time"], how="inner")
    m["err2"] = (m["issued"] - m["actual"]) ** 2
    g = (m.groupby(["Station", "lag"], sort=True)
           .agg(n_match=("err2", "size"), sse=("err2", "sum"))
           .reset_index())
    g["rmse"] = np.sqrt(g["sse"] / g["n_match"])

    wide_rmse = g.pivot(index="Station", columns="lag", values="rmse").reindex(columns=LAGS)
    wide_n = g.pivot(index="Station", columns="lag", values="n_match").reindex(columns=LAGS).fillna(0)
    n_valid = (wide_n > 0).sum(axis=1)
    eligible = n_valid[n_valid >= MIN_MATCHED_LAGS].index
    masked = wide_rmse.where(wide_n > 0)
    best_lag = masked.idxmin(axis=1)
    rmse_at_best = masked.min(axis=1)

    rows = []
    for st in eligible:
        sub = p[p["Station"] == st]
        e_x = sub["xgb"] - sub["target"]
        e_p = sub["persist"] - sub["target"]
        rows.append({
            "station_id": station_id(st),
            "station": st,
            "n_stride_rows": int(len(sub)),
            "best_lag_days": int(best_lag[st]),
            "rmse_at_best_lag": float(rmse_at_best[st]),
            "rmse_at_lag30": float(wide_rmse.loc[st, 30]) if wide_n.loc[st, 30] > 0 else None,
            "rmse_xgb_30d": float(np.sqrt(np.mean(e_x ** 2))),
            "rmse_persist_30d": float(np.sqrt(np.mean(e_p ** 2))),
        })
    out = pd.DataFrame(rows)

    six = out[out["station_id"].isin(all_ids)].copy()
    six["group"] = six["station_id"].map(
        {i: lab for lab, ids in group_ids.items() for i in ids})
    six = six.sort_values(["group", "station"]).reset_index(drop=True)

    summary = {
        "diagnostic": "4b follow-up — six dashboard groups by STATION ID "
                      "(sha1(station)[:16], dashboard archive key)",
        "matching": "groups seeded from data/meta/selected_gwl_stations.csv "
                    "canonical spellings; rows joined/selected by station_id, "
                    "never by name prefix",
        "group_seeds": DASH_GROUPS,
        "n_members_per_group_by_id": {lab: len(ids) for lab, ids in group_ids.items()},
        "group_member_count_in_aligned_eligible_bestlag": {
            lab: int(six[six["group"] == lab].shape[0]) for lab in group_ids},
        "eligible_group_stations": six.to_dict("records"),
        "note_on_any_group_absent": {
            lab: [i for i in ids if i not in set(six["station_id"])]
            for lab, ids in group_ids.items()},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    six.to_csv(HERE / "followup_six_by_id.csv", index=False)
    (HERE / "followup_six_by_id.json").write_text(json.dumps(summary, indent=2))

    print("group member counts (from meta, by id) / eligible in best-lag:")
    for lab, ids in group_ids.items():
        n_in = int(six[six["group"] == lab].shape[0])
        print(f"  {lab:<20} meta={len(ids):>2}  eligible_rows={n_in:>2}")
    print("\nsix-station rows (best lag, 30d rmse xgb vs persist):")
    for r in six.itertuples(index=False):
        print(f"  {r.group:<20} {r.station_id} {r.station:<32s} "
              f"lag={r.best_lag_days:>2}d rmse@lag={r.rmse_at_best_lag:.3f} "
              f"xgb30={r.rmse_xgb_30d:.3f} persist30={r.rmse_persist_30d:.3f}")
    print("wrote followup_six_by_id.csv / .json")


if __name__ == "__main__":
    main()