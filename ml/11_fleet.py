"""11_fleet — fleet 30-day forecast scan + decline / recovery ranking.

Scores every data-rich station (>= 2 y span, >= 2000 observations) with the
runtime-selected trajectory backend (_traj_dispatch -> quantile GRU by default,
XGBoost when pinned) and takes the +30d point of its 120-step trajectory. That
keeps /fleet consistent with /forecast/<slug>. The legacy ``xgb_level`` column
keeps its name for schema compatibility but now holds that trajectory point;
``change_pooled_m`` still carries the pooled-30d delta from
``_model.forward_forecast`` when that engine has a value.

Snapshot is written incrementally so an interrupt keeps progress.

Outputs -> outputs/fleet_forecast_snapshot.json (+ .csv, fleet_district.csv)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from _thresholds import DECLINE_M, category  # noqa: E402 - single source of truth

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SNAP = OUT / "fleet_forecast_snapshot.json"
CSV = OUT / "fleet_forecast.csv"
DIST_CSV = OUT / "fleet_district.csv"

MIN_OBS = 2000
MIN_SPAN_DAYS = 730
HORIZON = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max stations to scan (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="skip first N stations")
    ap.add_argument("--force", action="store_true", help="ignore existing snapshot")
    args = ap.parse_args()

    import numpy as np
    import _model
    from _traj_dispatch import active_trajectory_forecast
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
    pooled_failures = 0
    for i, station in enumerate(todo, 1):
        traj_out = active_trajectory_forecast(station)
        points = (traj_out or {}).get("trajectory") or []
        if not points:
            done[station] = {"station": station, "district": None,
                             "error": (traj_out or {}).get("error", "no trajectory")}
            if i % 25 == 0 or i == len(todo):
                _write_snapshot(done, stations)
                print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
            continue

        end = points[-1]
        anchor = float(traj_out["anchor_gwl"])
        point_level = float(end["q50"])      # legacy column name: xgb_level
        q05_level = float(end["q05"])
        q50_level = point_level
        q95_level = float(end["q95"])
        band_half = (q95_level - q05_level) / 2.0

        # Legacy change_pooled_m: pooled-30d delta from the forward backend
        # (a different engine). Optional — a missing value must not drop a row.
        chg_pooled = None
        try:
            pooled = _model.forward_forecast(station)
            v = pooled.get("pred_xgb") if pooled else None
            if v is not None and np.isfinite(v):
                chg_pooled = float(v)
        except Exception as e:  # noqa: BLE001 - pooled view is informational
            pooled_failures += 1
            print(f"  pooled view unavailable for {station}: "
                  f"{type(e).__name__}: {e}", flush=True)

        chg = q50_level - anchor
        anchor_valid = bool(np.isfinite(anchor) and np.isfinite(point_level))
        anchor_time = pd.Timestamp(traj_out["anchor_time"])
        age_days = int((now - anchor_time).days) if pd.notna(anchor_time) else None
        ore = obs_range.loc[station] if station in obs_range.index else (np.nan, np.nan)
        olo, ohi = float(ore["omin"]), float(ore["omax"])
        plausible = anchor_valid and abs(chg) <= 12.0 \
            and (not np.isfinite(olo) or (olo - 25.0 <= point_level <= ohi + 25.0)) \
            and (age_days is None or age_days <= 45)
        done[station] = {
            "station": station,
            "district": str(table6.loc[table6["Station"] == station, "District"].iloc[0])
            if len(table6.loc[table6["Station"] == station, "District"]) else None,
            "anchor": round(anchor, 3) if np.isfinite(anchor) else None,
            "date_from": str(traj_out["anchor_time"]),
            "age_days": age_days,
            "anchor_valid": anchor_valid,
            "xgb_level": round(point_level, 3) if np.isfinite(point_level) else None,
            "q05_level": round(q05_level, 3) if np.isfinite(q05_level) else None,
            "q50_level": round(q50_level, 3) if np.isfinite(q50_level) else None,
            "q95_level": round(q95_level, 3) if np.isfinite(q95_level) else None,
            "change_30d_m": round(chg, 3) if np.isfinite(chg) else None,
            "change_pooled_m": round(chg_pooled, 3) if chg_pooled is not None else None,
            "band_half_m": round(band_half, 3) if np.isfinite(band_half) else None,
            "plausible": bool(plausible),
            "category": category(chg, anchor_valid=anchor_valid, plausible=plausible),
        }
        if i % 25 == 0 or i == len(todo):
            _write_snapshot(done, stations)
            print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)

    _write_snapshot(done, stations)
    _write_csvs(done)
    print(f"scan complete: {len(done)} stations in {time.time()-t0:.0f}s "
          f"| pooled view missing for {pooled_failures}")


def _backends() -> dict:
    """Backend flags recorded in the snapshot so a scan is auditable."""
    try:
        from _traj_dispatch import traj_backend
        traj = traj_backend()
    except Exception:  # noqa: BLE001 - metadata only
        traj = None
    try:
        from _model import forward_backend
        fwd = forward_backend()
    except Exception:  # noqa: BLE001 - metadata only
        fwd = None
    return {"traj_backend": traj, "forward_backend": fwd}


def _write_snapshot(done: dict[str, dict], stations: list[str]) -> None:
    SNAP.write_text(json.dumps({
        "horizon_days": HORIZON,
        "decline_threshold_m": DECLINE_M,
        "eligible_stations": len(stations),
        "scanned_stations": sum(1 for s in done.values() if "error" not in s),
        "generated_at": str(pd.Timestamp.now())[:19],
        **_backends(),
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