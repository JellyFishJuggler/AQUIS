"""refresh/inference.py — per-station anchored forecast refresh + atomic publish.

Runs the production trajectory engine for every (or a subset of) station(s),
each anchored at that station's LATEST observed GWL reading (never a forecast /
bucket / fleet-global anchor), then publishes:
  * data/refresh/forecasts/{station_sha}.json        per-station full forecast
  * data/refresh/forecasts.parquet                    fleet summary table
  * data/refresh/meta.json                            commit manifest / freshness
  * data/refresh/last_anchors.parquet                 per-station anchor map
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import _trajectory as traj
from refresh import publish
from refresh.config import REFRESH_DIR
from refresh.state import RefreshState, refresh_lock

LAST_ANCHORS = REFRESH_DIR / "last_anchors.parquet"


def _load_engine():
    traj._CACHE.clear()   # the scheduler is long-running: never reuse stale tables/models
    return traj._load()


def current_anchors() -> pd.Series:
    """Per-station latest OBSERVED gwl time from the spike-dropped aligned table."""
    c = _load_engine()
    t = c["tbl"]
    s = t.dropna(subset=["gwl"])
    g = s.groupby(s["Station"].astype(str))["time"].max()
    return g.rename_axis("Station")


def previous_anchors() -> pd.Series:
    if not LAST_ANCHORS.exists():
        return pd.Series(dtype="datetime64[ns]")
    df = pd.read_parquet(LAST_ANCHORS)
    if df.empty or "Station" not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return df.set_index(df["Station"].astype(str))["anchor_time"].astype("datetime64[ns]")


def refresh_forecasts(*, cfg, state: RefreshState, stations=None, max_stations: int = 0,
                      skip_unchanged: bool = True, publish: bool = True,
                      dry_run: bool = False, source_report: dict | None = None,
                      engine_extra: dict | None = None) -> dict:
    """Compute + publish the per-station trajectory forecasts for one cycle."""
    c = _load_engine()
    fnames_model = c["fnames"]
    tbl = c["tbl"]

    all_stations = sorted(tbl["Station"].astype(str).unique())
    if stations is None:
        stations = all_stations
    else:
        stations = [s for s in stations if s in set(all_stations)]
    if max_stations:
        stations = stations[:max_stations]

    cur = current_anchors()
    prev = previous_anchors()

    forecasts, errors = [], {}
    unchanged = 0
    for station in stations:
        anchor = cur.get(station)
        if skip_unchanged and publish and not dry_run:
            if station in prev.index and prev.loc[station] == anchor:
                unchanged += 1
                continue
        out = traj.trajectory_forecast(station)
        if "error" in out:
            errors[station] = out["error"]
        else:
            out["model_features"] = fnames_model
            forecasts.append(out)

    if not dry_run and publish:
        if len(forecasts) < cfg.min_fresh_stations:
            raise RuntimeError(
                f"publish aborted: only {len(forecasts)} fresh forecasts "
                f"(< min_fresh_stations={cfg.min_fresh_stations})")
        tcfg = traj.TRAJ_CFG
        import json
        cfgj = json.loads(tcfg.read_text()) if tcfg.exists() else {}
        model_version = cfgj.get("version") or str(cfgj.get("trained_at", ""))
        meta = publish.publish_forecasts(
            forecasts, engine="trajectory-v2", model_version=model_version,
            model_trained_at=cfgj.get("trained_at"), source_report=source_report,
            keep_staging=True)
        publish.update_freshness(meta, cfg.stale_after_hours,
                                 state.get("last_forecast_refresh_ts"))
        # persist per-station anchor map (for skip-unchanged next cycle)
        anchors_df = pd.DataFrame({"Station": list(cur.index), "anchor_time": list(cur.values)})
        publish.atomic_write_parquet(LAST_ANCHORS, anchors_df)

    state.update(last_forecast_refresh_ts=state.get("last_forecast_refresh_ts") or "now")

    return {
        "dry_run": dry_run,
        "published": bool(publish and not dry_run),
        "stations": len(stations),
        "forecasted": len(forecasts),
        "unchanged_skipped": unchanged,
        "failed": errors,
        "model_features": fnames_model,
        "engine_extra": engine_extra,
    }


def status() -> dict:
    """Convenience summary for the CLI/UI without running anything."""
    meta = publish.read_meta()
    anchors = current_anchors()
    return {
        "forecast_generation_ts": meta.get("forecast_generation_ts"),
        "model_version": meta.get("model_version"),
        "model_trained_at": meta.get("model_trained_at"),
        "data_status": meta.get("data_status"),
        "stale_reason": meta.get("stale_reason"),
        "stations_forecasted": meta.get("stations_forecasted"),
        "stations_in_table": int(len(anchors)),
        "latest_observed_ts": (str(anchors.max()) if len(anchors) else None),
    }