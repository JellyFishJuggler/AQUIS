"""11_fleet — fleet 30-day forecast scan + decline / recovery ranking.

Scores every data-rich station (>= 2 y span, >= 2000 observations) with the same
pooled 30-day forward path as the app (06_features.build_full + pooled XGBoost +
calibrated q05/q95), then buckets stations as declining / stable / recovering and
ranks them. Snapshot is written incrementally so an interrupt keeps progress.

Outputs -> outputs/fleet_forecast_snapshot.json (+ .csv, fleet_district.csv)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SNAP = OUT / "fleet_forecast_snapshot.json"
CSV = OUT / "fleet_forecast.csv"
DIST_CSV = OUT / "fleet_district.csv"

MIN_OBS = 2000
MIN_SPAN_DAYS = 730
DECLINE_M = 0.3          # >= 0.3 m drop over 30 days = flagged decline
HIGH_DECLINE_M = 0.6
HORIZON = 30


def category(change: float | None, anchor_valid: bool = True,
             plausible: bool = True, verbose: bool = True) -> str:
    if not anchor_valid or not plausible:
        return "unreliable"
    if change is None:
        return "unknown"
    if change <= -HIGH_DECLINE_M:
        return "decline (high)"
    if change <= -DECLINE_M:
        return "decline"
    if change >= DECLINE_M:
        return "recovering"
    return "stable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max stations to scan (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="skip first N stations")
    ap.add_argument("--force", action="store_true", help="ignore existing snapshot")
    args = ap.parse_args()

    import numpy as np
    import _model
    from _utils import load_table_6h

    table6 = load_table_6h()
    obs_range = table6.groupby("Station")["gwl"].agg(omin="min", omax="max")
    g = table6.groupby("Station").agg(
        n=("gwl", "count"), first=("time", "min"), last=("time", "max")).reset_index()
    g["span_days"] = (g["last"] - g["first"]).dt.days
    eligible = g[(g["n"] >= MIN_OBS) & (g["span_days"] >= MIN_SPAN_DAYS)]
    stations = eligible.sort_values("n", ascending=False)["Station"].tolist()

    done: dict[str, dict] = {}
    if not args.force and SNAP.exists():
        try:
            done = {s["station"]: s for s in json.loads(SNAP.read_text())["stations"]}
            print(f"resume: {len(done)} stations already in snapshot")
        except Exception as e:  # noqa: BLE001
            print(f"snapshot unreadable, starting fresh: {e}")

    todo = [s for s in stations[args.start:args.start + (args.limit or len(stations))]
            if s not in done]
    print(f"eligible {len(stations)} stations — scanning {len(todo)}")
    t0 = time.time()
    now = pd.Timestamp.now()
    for i, station in enumerate(todo, 1):
        fc = _model.forward_forecast(station)
        if not fc:
            done[station] = {"station": station, "district": None,
                             "error": "no feature frame"}
            continue
        chg_pooled = fc["pred_xgb"]
        anchor = fc["anchor"]
        xgb_level = fc["xgb_level"]
        q50_level = fc.get("q50_level")
        chg = (q50_level - anchor) if (q50_level is not None and np.isfinite(q50_level)
                                      and np.isfinite(anchor)) else chg_pooled
        anchor_valid = bool(np.isfinite(anchor) and np.isfinite(xgb_level))
        age_days = int((now - fc["date_from"]).days) if fc["date_from"] is not None else None
        ore = obs_range.loc[station] if station in obs_range.index else (np.nan, np.nan)
        olo, ohi = float(ore["omin"]), float(ore["omax"])
        plausible = anchor_valid and abs(chg) <= 12.0 \
            and (not np.isfinite(olo) or (olo - 25.0 <= xgb_level <= ohi + 25.0)) \
            and (age_days is None or age_days <= 45)
        done[station] = {
            "station": station,
            "district": str(table6.loc[table6["Station"] == station, "District"].iloc[0])
            if len(table6.loc[table6["Station"] == station, "District"]) else None,
            "anchor": round(float(anchor), 3) if np.isfinite(anchor) else None,
            "date_from": str(fc["date_from"]),
            "age_days": age_days,
            "anchor_valid": anchor_valid,
            "xgb_level": round(float(xgb_level), 3) if np.isfinite(xgb_level) else None,
            "q05_level": round(float(fc["q05_level"]), 3) if fc.get("q05_level") is not None else None,
            "q50_level": round(float(fc["q50_level"]), 3) if fc.get("q50_level") is not None else None,
            "q95_level": round(float(fc["q95_level"]), 3) if fc.get("q95_level") is not None else None,
            "change_30d_m": round(float(chg), 3) if np.isfinite(chg) else None,
            "change_pooled_m": round(float(chg_pooled), 3) if np.isfinite(chg_pooled) else None,
            "band_half_m": round(float(fc["band_half"]), 3) if fc.get("band_half") is not None else None,
            "plausible": bool(plausible),
            "category": category(chg, anchor_valid=anchor_valid, plausible=plausible),
        }
        if i % 25 == 0 or i == len(todo):
            _write_snapshot(done, stations)
            print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)

    _write_snapshot(done, stations)
    _write_csvs(done)
    print(f"scan complete: {len(done)} stations in {time.time()-t0:.0f}s")


def _write_snapshot(done: dict[str, dict], stations: list[str]) -> None:
    SNAP.write_text(json.dumps({
        "horizon_days": HORIZON,
        "decline_threshold_m": DECLINE_M,
        "eligible_stations": len(stations),
        "scanned_stations": sum(1 for s in done.values() if "error" not in s),
        "generated_at": str(pd.Timestamp.now())[:19],
        "stations": [done[s] for s in stations if s in done],
    }, indent=1))


def _write_csvs(done: dict[str, dict]) -> None:
    rows = [d for s in done for d in [done[s]] if "error" not in d]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    good = df[df["category"] != "unreliable"]
    dist = (good.groupby("district")
              .agg(n=("station", "count"),
                   n_decline=("change_30d_m", lambda x: int((x <= -DECLINE_M).sum())),
                   n_recover=("change_30d_m", lambda x: int((x >= DECLINE_M).sum())),
                   median_change_m=("change_30d_m", "median"))
              .sort_values("n_decline", ascending=False))
    dist["decline_share"] = (dist["n_decline"] / dist["n"]).round(3)
    dist.to_csv(DIST_CSV)
    n_decline = int((good["change_30d_m"] <= -DECLINE_M).sum())
    n_recover = int((good["change_30d_m"] >= DECLINE_M).sum())
    n_unreliable = int((df["category"] == "unreliable").sum())
    print(f"\nfleet: {len(good)} scored (excl {n_unreliable} unreliable) | "
          f"decline {n_decline} | recovering {n_recover} | stable "
          f"{len(good)-n_decline-n_recover}")
    print("worst 8 (30d change):")
    print(good.nsmallest(8, "change_30d_m")[["station", "district", "change_30d_m", "category"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()