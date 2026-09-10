"""refresh/pipeline.py — orchestration of one refresh cycle.

Order for a forecast cycle (each step journaled; failure at any step aborts
before inference/publish, preserving previous forecasts):

    1. fetch       NWIC GWL + Open-Meteo weather (retries, per-source status)
    2. features    guarded rebuild of the aligned table + climatology
    3. inference   per-station anchored forecast -> atomic publish + freshness
    4. (optional)  retrain  scheduled full refit + gated promotion

The whole cycle holds the terminal flock (see refresh.state), so overlapping
runs from cron/daemon/manual are impossible.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from refresh import features, inference, model_update, sources
from refresh.config import load_config
from refresh.state import RefreshState, refresh_lock


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def run_forecast_cycle(*, cfg=None, state=None, dry_run: bool = False,
                       do_fetch: bool = True, do_features: bool = True,
                       do_inference: bool = True, stations=None,
                       max_stations: int = 0, skip_unchanged: bool = True,
                       lock: bool = True, source_report: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ctx = refresh_lock() if lock else _nullctx()
    with ctx:
        state = state or RefreshState()
        now = _utcnow()
        report: dict = {"cycle_ts": _iso(now), "dry_run": dry_run}

        if do_fetch:
            report["fetch"] = sources.fetch_all(cfg, dry_run=dry_run)
            state.record("fetch", ok=bool(report["fetch"].get("all_ok")),
                         msg="", districts=len(report["fetch"].get("gwl", {}).get("per_district", {})))

        if do_features and not dry_run:
            report["features"] = features.rebuild_aligned(
                dry_run=False, min_stations=cfg.min_fresh_stations)
            state.record("features", ok=True, msg="aligned table + climatology rebuilt")
        elif do_features:
            report["features"] = {"ok": True, "dry_run": True}

        if do_inference:
            inf = inference.refresh_forecasts(
                cfg=cfg, state=state, stations=stations, max_stations=max_stations,
                skip_unchanged=skip_unchanged, publish=not dry_run, dry_run=dry_run,
                source_report=(report.get("fetch") if do_fetch else source_report),
                engine_extra={"features": report.get("features", {}).get("validate")})
            report["inference"] = inf
            state.record("forecast", ok=(not dry_run and inf["published"]),
                         msg=f"{inf.get('forecasted', 0)} stations forecast",
                         forecasted=inf.get("forecasted"), skipped=inf.get("unchanged_skipped"))

        if not dry_run:
            state.update(next_refresh_due=_iso(now + timedelta(hours=cfg.refresh_interval_hours)),
                         last_forecast_refresh_ts=_iso(now))
        return report


def run_retrain_cycle(*, cfg=None, state=None, smoke: bool = False,
                      dry_run: bool = False, feature_mode: str = "flat",
                      lock: bool = True) -> dict:
    cfg = cfg or load_config()
    if feature_mode not in ("flat", "future"):
        raise ValueError(f"feature_mode must be flat|future, got {feature_mode!r}")
    ctx = refresh_lock() if lock else _nullctx()
    with ctx:
        state = state or RefreshState()
        now = _utcnow()
        future_data = None
        if feature_mode == "future":
            future_data = _load_future_data()
        if dry_run:
            return {"dry_run": True}

        state.record("retrain.start", ok=True, msg="candidate long-frame build")
        prep = model_update.build_candidate_longframe(cfg, smoke=smoke)
        cfgout = model_update.train_candidate(cfg, smoke=smoke)
        state.record("retrain.train", ok=True,
                     msg=f"fit_rows={cfgout['fit_rows']} val={cfgout['val_rows_scored']}")

        gate = model_update.run_gate_evaluation(cfg, mode=feature_mode, future_data=future_data)
        decision = gate["decision"]
        if decision == "promote":
            out = model_update.promote(cfg, state, gate)
            state.record("retrain.promote", ok=True,
                         msg=f"candidate {out['candidate_30d_rmse']} vs incumbent "
                             f"{out['incumbent_30d_rmse']} (30d, same anchors)")
        else:
            out = model_update.reject(state, gate)
            state.record("retrain.reject", ok=False, msg=str(gate["reasons"]))
        out["gate"] = gate
        out["prep"] = prep
        out["dry_run"] = False
        if not dry_run:
            state.update(next_retrain_due=_iso(now + timedelta(hours=cfg.retrain_cadence_hours)))
        return out


def _load_future_data() -> tuple:
    from _trajectory import OM, CLIM, RAIN_CLIM
    import pandas as pd
    om = pd.read_parquet(OM) if OM.exists() else None
    clim = pd.read_parquet(CLIM) if CLIM.exists() else None
    rc = pd.read_parquet(RAIN_CLIM) if RAIN_CLIM.exists() else None
    return om, clim, rc


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False