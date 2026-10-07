#!/usr/bin/env python
"""Diagnostic 4c follow-up — six-dashboard-station events by STATION ID.

Events/flags identical to run.py (same fixed rule, same event definition,
same buckets). Station identity = sha1(Station)[:16] (dashboard archive key);
the six groups are seeded from data/meta/selected_gwl_stations.csv canonical
spellings (so the "Varansi" vs "Varanasi" variant cannot drop stations) and all
selection/grouping below is by station_id, never by name prefix.

READ-ONLY on production; writes only into data/scratch_assoc/diag_4c_glitches/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as C4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
META = ROOT / "data" / "meta" / "selected_gwl_stations.csv"

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
    meta = pd.read_csv(META)
    canon = {str(r[0]): station_id(str(r[0])) for r in meta.itertuples(index=False)}
    group_ids: dict[str, list[str]] = {}
    all_ids: set[str] = set()
    for label, pfs in DASH_GROUPS.items():
        ids = [canon[n] for n in canon
               if any(n.lower().startswith(p.lower()) for p in pfs)]
        group_ids[label] = sorted(ids)
        all_ids.update(ids)

    df = C4.load()
    df = C4.flag_points(df)
    ev = C4.build_events(df)
    fl = df.loc[df["flagged"], ["Station", "time"]].copy()
    gap = fl.groupby("Station", sort=False)["time"].diff()
    fl["event_id"] = (gap.isna() | (gap > C4.EVENT_GAP)).groupby(
        fl["Station"], sort=False).cumsum()
    df.loc[df["flagged"], "event_id"] = fl["event_id"].to_numpy()
    ev = C4.classify(df, ev)

    ev = ev.copy()
    ev["station_id"] = ev["Station"].map(canon)
    ev["group"] = ev["station_id"].map(
        {i: lab for lab, ids in group_ids.items() for i in ids})

    flagged = df.loc[df["flagged"]].copy()
    flagged["station_id"] = flagged["Station"].map(canon)
    flagged_counts = (
        flagged.groupby("station_id").size().rename("n_flagged_points"))

    six = ev[ev["group"].notna()].copy()
    six = six.sort_values(["group", "Station", "first_time"]).reset_index(drop=True)

    six_recs = six.drop(columns=["Station"]).rename(
        columns={"first_time": "event_start", "last_time": "event_end"})
    for c in ("event_start", "event_end"):
        six_recs[c] = six_recs[c].astype(str)

    summary = {
        "diagnostic": "4c follow-up — six dashboard groups by STATION ID "
                      "(sha1(station)[:16])",
        "fixed_rule": C4.RULE,
        "event_definition": C4.EVENT_GAP.__str__() + " grouping (12 h)",
        "matching": "groups seeded from data/meta/selected_gwl_stations.csv; "
                    "events/flagged joined and grouped by station_id only",
        "per_group": {
            lab: {
                "n_members_by_id": len(ids),
                "n_members_with_events": int(six.loc[six["group"] == lab, "station_id"].nunique()),
                "n_events": int((six["group"] == lab).sum()),
                "events_by_bucket": {
                    b.split("_")[0]: int(((six["group"] == lab) & (six["bucket"] == b)).sum())
                    for b in (C4.B_I, C4.B_II, C4.B_III)
                },
                "n_flagged_points": int(flagged_counts.reindex(ids).sum()),
            }
            for lab, ids in group_ids.items()
        },
        "six_station_events": six_recs.to_dict("records"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    six.to_csv(HERE / "followup_six_by_id_events.csv", index=False)
    (HERE / "followup_six_by_id_events.json").write_text(json.dumps(summary, indent=2))

    print("six groups by ID — events and buckets:")
    for lab, s_ in summary["per_group"].items():
        print(f"  {lab:<20} stations={s_['n_members_with_events']:>2}/{s_['n_members_by_id']:>2} "
              f"events={s_['n_events']:>3} buckets={s_['events_by_bucket']} "
              f"flagged={s_['n_flagged_points']:>5}")
    print("wrote followup_six_by_id_events.csv / .json")


if __name__ == "__main__":
    main()