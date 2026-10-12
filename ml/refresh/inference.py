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

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import _trajectory as traj
from _traj_dispatch import active_trajectory_forecast, traj_backend
from refresh import publish as pubmod
from refresh.config import REFRESH_DIR
from refresh.state import RefreshState, refresh_lock

LAST_ANCHORS = REFRESH_DIR / "last_anchors.parquet"
DL_CFG = Path(__file__).resolve().parents[1] / "models" / "traj_dl_config.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_engine():
    traj._CACHE.clear()   # the scheduler is long-running: never reuse stale tables/models
    try:                  # the DL engine caches its torch model + table the same way
        import _dl
        _dl.clear_caches()
    except Exception:     # noqa: BLE001 - DL optional / not installed
        pass
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
                      allow_empty: bool = False,
                      engine_extra: dict | None = None) -> dict:
    """Compute + publish the per-station trajectory forecasts for one cycle.

    ``allow_empty`` is set by the pipeline only when the GWL fetch succeeded but
    returned zero new rows: in that case every anchor is unchanged, *zero* fresh
    forecasts is the expected result, and it is reported as a benign
    ``no_new_source`` no-op instead of raising. Any other reason for too few
    fresh forecasts (a real fetch/inference failure) still raises.
    """
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
    backend = traj_backend()
    if backend == "dl" and DL_CFG.exists():
        try:
            fnames_model = json.loads(DL_CFG.read_text()).get("feature_cols", fnames_model)
        except Exception:  # noqa: BLE001
            pass
    for station in stations:
        anchor = cur.get(station)
        if skip_unchanged and publish and not dry_run:
            if station in prev.index and prev.loc[station] == anchor:
                unchanged += 1
                continue
        out = active_trajectory_forecast(station)
        if "error" in out:
            errors[station] = out["error"]
        else:
            out["model_features"] = fnames_model
            forecasts.append(out)

    no_new_source = False
    if not dry_run and publish:
        if len(forecasts) < cfg.min_fresh_stations:
            # only an *empty* result gated on a clean zero-row fetch is the
            # expected no-op; anything else (a few fresh, or an error path) is a
            # real anomaly and must still abort the publish.
            if allow_empty and not forecasts:
                no_new_source = True
            else:
                raise RuntimeError(
                    f"publish aborted: only {len(forecasts)} fresh forecasts "
                    f"(< min_fresh_stations={cfg.min_fresh_stations})")
        if not no_new_source:
            if backend == "dl":
                cfgj = json.loads(DL_CFG.read_text()) if DL_CFG.exists() else {}
                engine = "trajectory-v2-dl"
            else:
                tcfg = traj.TRAJ_CFG
                cfgj = json.loads(tcfg.read_text()) if tcfg.exists() else {}
                engine = "trajectory-v2"
            model_version = cfgj.get("version") or str(cfgj.get("trained_at", ""))
            meta = pubmod.publish_forecasts(
                forecasts, engine=engine, model_version=model_version,
                model_trained_at=cfgj.get("trained_at"), source_report=source_report,
                keep_staging=True)
            pubmod.update_freshness(meta, cfg.stale_after_hours,
                                    state.get("last_forecast_refresh_ts"))
            # persist per-station anchor map (for skip-unchanged next cycle)
            anchors_df = pd.DataFrame({"Station": list(cur.index), "anchor_time": list(cur.values)})
            pubmod.atomic_write_parquet(LAST_ANCHORS, anchors_df)

    state.update(last_forecast_refresh_ts=_now_iso() if not state.get("last_forecast_refresh_ts") or state.get("last_forecast_refresh_ts") == "now" else state.get("last_forecast_refresh_ts"))

    return {
        "dry_run": dry_run,
        "published": bool(publish and not dry_run and not no_new_source),
        "no_new_source": bool(not dry_run and publish and no_new_source),
        "stations": len(stations),
        "forecasted": len(forecasts),
        "unchanged_skipped": unchanged,
        "table_latest": (str(cur.max()) if len(cur) else None),
        "failed": errors,
        "model_features": fnames_model,
        "engine_extra": engine_extra,
    }


def status() -> dict:
    """Convenience summary for the CLI/UI without running anything."""
    meta = pubmod.read_meta()
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