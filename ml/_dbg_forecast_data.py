"""_dbg_forecast_data — why 2026 validation shows nothing in the Forecast explorer.

Reads only; prints. Two independent checks:

  `ledger`  — what is actually in `data/refresh/verification.parquet`, and whether
              any row exists for a given station at horizon 720 with a 2026 target.
              This is the artifact the Forecast explorer matches on.
  `archive` — `data/refresh/forecast_archive/` holds thousands of issued-forecast
              JSONs, which ARE a point-in-time record of what was published. The
              app does not currently use them for validation. This counts them and
              reports the anchor-date span, so we can tell whether they could
              substitute or supplement the empty ledger.

Run:  venv/bin/python _dbg_forecast_data.py [station]
      venv/bin/python _dbg_forecast_data.py --archive-limit 400
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ML = Path(__file__).resolve().parent
REFRESH = ML / "data" / "refresh"
LEDGER = REFRESH / "verification.parquet"
ARCHIVE = REFRESH / "forecast_archive"


def section(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def ledger_report(station: str) -> None:
    section("verification.parquet  (what the Forecast explorer matches on)")
    if not LEDGER.exists():
        print("MISSING:", LEDGER)
        return
    df = pd.read_parquet(LEDGER)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    if df.empty:
        print("\n>>> The ledger is EMPTY. Every station will show 'no matches'.")
        return
    print("\nhead:\n", df.head(3).to_string())

    for col in ("station", "model_version"):
        if col in df.columns:
            print(f"\n{col}: {df[col].nunique()} distinct")
            print(df[col].value_counts().head(8).to_string())

    if "horizon_h" in df.columns:
        hh = pd.to_numeric(df["horizon_h"], errors="coerce")
        print("\nhorizon_h counts:\n", hh.value_counts().head(12).to_string())
        print("horizon_h == 720 present:", bool((hh == 720).any()))

    if "target_time" in df.columns:
        tt = pd.to_datetime(df["target_time"], errors="coerce")
        print("\ntarget_time year counts:\n",
              tt.dt.year.value_counts().sort_index().to_string())

    if {"station", "horizon_h"} <= set(df.columns):
        m = df[df["station"].astype(str) == station]
        print(f"\nrows for {station!r}: {len(m)}")
        if len(m):
            print(m["horizon_h"].value_counts().head(12).to_string())
            if "target_time" in m.columns:
                tt = pd.to_datetime(m["target_time"], errors="coerce")
                print("  target years:", tt.dt.year.value_counts().sort_index().to_dict())
            m720 = m[pd.to_numeric(m["horizon_h"], errors="coerce") == 720]
            print(f"  ...of which horizon 720: {len(m720)}")
        else:
            print(">>> No rows at all for this station. Compare the station-name "
                  "spelling above with the distinct values listed.")


def archive_report(station: str, limit: int) -> None:
    section("forecast_archive/  (issued forecasts — a point-in-time record)")
    if not ARCHIVE.is_dir():
        print("MISSING:", ARCHIVE)
        return
    files = sorted(ARCHIVE.glob("*.json"))
    print("total JSON files:", len(files))
    if not files:
        return

    step = max(1, len(files) // max(1, limit))
    sample = files[::step]
    print(f"sampling every {step}th file -> {len(sample)} read")

    stations: Counter = Counter()
    anchors: list[str] = []
    traj_lens: Counter = Counter()
    for f in sample:
        try:
            o = json.loads(f.read_text())
        except Exception:
            continue
        s = o.get("station")
        if s:
            stations[s] += 1
        a = o.get("anchor_time")
        if a:
            anchors.append(str(a))
        traj_lens[len(o.get("trajectory") or [])] += 1

    print("distinct stations (in sample):", len(stations))
    print("top stations:", stations.most_common(5))
    if anchors:
        print("anchor_time span:", min(anchors), "->", max(anchors))
    print("trajectory length counts:", traj_lens.most_common(6))

    hits = [f for f in sample
            if _station_of(f) == station]
    print(f"\nsampled files whose station == {station!r}: {len(hits)}")
    if hits:
        o = json.loads(sorted(hits)[-1].read_text())
        tr = o.get("trajectory") or []
        print("newest sampled entry keys:", sorted(o.keys()))
        print("trajectory points:", len(tr))
        if len(tr) >= 120:
            p = tr[119]
            print("h=120 point:", {k: p.get(k) for k in
                                   ("time", "q05", "q50", "q95", "confidence_level")})
        print("\nNOTE: to validate, match each archived `time` against the observed "
              "level at that same timestamp. The archive is a genuine record of what "
              "was issued, so it is usable for honest backtesting — unlike re-scoring "
              "today's model on today's data.")


def _station_of(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("station")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("station", nargs="?", default="ASHADHA PRATHMIK VIDYALAYA")
    ap.add_argument("--archive-limit", type=int, default=300,
                    help="max archive files to read (sampled evenly)")
    args = ap.parse_args()
    ledger_report(args.station)
    archive_report(args.station, args.archive_limit)


if __name__ == "__main__":
    main()
