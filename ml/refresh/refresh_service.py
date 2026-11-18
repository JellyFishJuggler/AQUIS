"""refresh_service — cycle orchestration, daemon loop and CLI command handlers.

Merged from pipeline/scheduler/cli. The forecast cycle holds one non-blocking
lock; a failed fetch/feature/inference step aborts before publish so previously
published forecasts remain intact. Cadences for forecast and retrain are
independent and restart-safe via persisted due-times.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from refresh import refresh_core as core
from refresh import refresh_data as data
from refresh import refresh_engines as engines
from refresh.refresh_core import (ConfigError, RefreshState, load_config,
                                  models_frozen, next_grid_due, refresh_lock)

ROOT = Path(__file__).resolve().parent.parent
sources = features = data
inference = verification = model_update = backtest = engines
publish = scheduler = pipeline = None  # set to this module at the end


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _is_no_new_source(fetch_report: dict | None) -> bool:
    """True only when GWL fetched cleanly AND returned zero new rows."""
    gwl = ((fetch_report or {}).get("gwl") or {})
    if not gwl.get("ok"):
        return False
    try:
        new_rows = int(gwl.get("new_rows", 1))
    except (TypeError, ValueError):
        return False
    return new_rows == 0


# --- cycle orchestration -----------------------------------------------------
def run_forecast_cycle(*, cfg=None, state=None, dry_run: bool = False,
                       do_fetch: bool = True, do_features: bool = True,
                       do_inference: bool = True, stations=None,
                       max_stations: int = 0, skip_unchanged: bool = True,
                       lock: bool = True, source_report: dict | None = None) -> dict:
    """Fetch -> guarded features -> anchored inference/publish -> verification."""
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
        no_new_source = _is_no_new_source(report.get("fetch"))
        inf = None
        if do_inference:
            inf = inference.refresh_forecasts(
                cfg=cfg, state=state, stations=stations, max_stations=max_stations,
                skip_unchanged=skip_unchanged, publish=not dry_run, dry_run=dry_run,
                source_report=(report.get("fetch") if do_fetch else source_report),
                allow_empty=no_new_source,
                engine_extra={"features": report.get("features", {}).get("validate")})
            report["inference"] = inf
            if inf.get("no_new_source"):
                gwl = ((report.get("fetch") or {}).get("gwl") or {})
                state.record(
                    "no_new_source", ok=True,
                    msg="no new source data (fetch ok, 0 new rows); forecasts unchanged",
                    source_latest=gwl.get("source_latest"),
                    table_latest=inf.get("table_latest"),
                    unchanged_skipped=inf.get("unchanged_skipped"))
            else:
                state.record("forecast", ok=(not dry_run and inf["published"]),
                             msg=f"{inf.get('forecasted', 0)} stations forecast",
                             forecasted=inf.get("forecasted"),
                             skipped=inf.get("unchanged_skipped"))
        noop = bool(inf and inf.get("no_new_source"))
        if not dry_run and do_inference and inf.get("published"):
            report["verification"] = verification.verify_forecasts()
        if not dry_run:
            if noop:
                # minimum interval so an empty source can never hot-loop the daemon
                floor = now + timedelta(minutes=cfg.no_new_data_backoff_min)
                due = max(next_grid_due(cfg, now), floor)
                report["rearmed_next_refresh_due"] = _iso(due)
                state.update(next_refresh_due=_iso(due))
            else:
                state.update(next_refresh_due=_iso(next_grid_due(cfg, now)))
        return report


def _load_future_data() -> tuple:
    from serving.engines import OM, CLIM, RAIN_CLIM
    om = pd.read_parquet(OM) if OM.exists() else None
    clim = pd.read_parquet(CLIM) if CLIM.exists() else None
    rc = pd.read_parquet(RAIN_CLIM) if RAIN_CLIM.exists() else None
    return om, clim, rc


def run_retrain_cycle(*, cfg=None, state=None, smoke: bool = False,
                      dry_run: bool = False, feature_mode: str = "flat",
                      lock: bool = True) -> dict:
    """Build candidate, backtest on same anchors, promote only if gates pass."""
    cfg = cfg or load_config()
    if feature_mode not in ("flat", "future"):
        raise ValueError(f"feature_mode must be flat|future, got {feature_mode!r}")
    if models_frozen(cfg):
        state = state or RefreshState()
        state.record("retrain.skip", ok=True,
                     msg=f"models frozen until {cfg.models_frozen_until}")
        return {"skipped": True, "reason": "models_frozen",
                "frozen_until": cfg.models_frozen_until, "promoted": False}
    ctx = refresh_lock() if lock else _nullctx()
    with ctx:
        state = state or RefreshState()
        now = _utcnow()
        future_data = _load_future_data() if feature_mode == "future" else None
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
        out["gate"], out["prep"], out["dry_run"] = gate, prep, False
        state.update(next_retrain_due=_iso(now + timedelta(hours=cfg.retrain_cadence_hours)))
        return out


# --- scheduler daemon --------------------------------------------------------
def _parse(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def refresh_due(cfg, state: RefreshState, now: datetime | None = None) -> bool:
    now = now or _utcnow()
    due = _parse(state.get("next_refresh_due"))
    return due is None or now >= due


def retrain_due(cfg, state: RefreshState, now: datetime | None = None) -> bool:
    now = now or _utcnow()
    due = _parse(state.get("next_retrain_due"))
    if due is not None and now < due:
        return False
    if cfg.retrain_at_hours and now.hour not in cfg.retrain_at_hours:
        return False
    cursor = _parse(state.get("next_retrain_due")) or (
        now - timedelta(hours=cfg.retrain_cadence_hours))
    return now >= cursor


def drift_retrain_due(cfg, state: RefreshState, now: datetime | None = None) -> bool:
    """Drift override: streak of DRIFT_CONSECUTIVE_CYCLES tripped summaries, counted once each."""
    now = now or _utcnow()
    vd = verification.drift_status(None)
    summary_ts = vd.get("updated_ts")
    counted_ts = state.get("drift_counted_ts")
    tripped = bool(vd.get("tripped"))
    streak = int(state.get("drift_streak", 0))
    if summary_ts and summary_ts != counted_ts:
        if tripped:
            streak += 1
            state.update(drift_streak=streak, drift_counted_ts=summary_ts,
                         drift_reason=vd.get("reason"))
        else:
            if streak or counted_ts:
                state.update(drift_streak=0, drift_counted_ts=summary_ts,
                             drift_reason=None)
            streak = 0
    return streak >= verification.DRIFT_CONSECUTIVE_CYCLES


def scheduler_loop(*, cfg=None, state=None, poll_s: float = 60.0, once: bool = False,
                   do_fetch: bool = True, do_features: bool = True,
                   do_inference: bool = True, smoke_retrain: bool = False) -> None:
    cfg = cfg or load_config()
    state = state or RefreshState()
    print(f"[refresh] scheduler started cwd={__file__} poll={poll_s}s "
          f"refresh_every={cfg.refresh_interval_hours}h retrain_every={cfg.retrain_cadence_hours}h "
          f"models_frozen_until={cfg.models_frozen_until or '-'}", flush=True)
    while True:
        now = _utcnow()
        if refresh_due(cfg, state, now):
            try:
                with refresh_lock():
                    if refresh_due(cfg, state, _utcnow()):
                        print(f"[refresh] forecast cycle @ {_utcnow():%Y-%m-%d %H:%M:%S}", flush=True)
                        rep = run_forecast_cycle(
                            cfg=cfg, state=state, do_fetch=do_fetch,
                            do_features=do_features, do_inference=do_inference, lock=False)
                        print(f"[refresh] cycle done: {rep.get('inference', {})}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep daemon alive
                state.record("scheduler.error", ok=False, msg=f"forecast cycle: {exc}")
                print(f"[refresh] forecast cycle failed: {exc}", flush=True)
        retrain_forced = drift_retrain_due(cfg, state, _utcnow())
        if not models_frozen(cfg) and (retrain_due(cfg, state, _utcnow()) or retrain_forced):
            try:
                with refresh_lock():
                    if retrain_due(cfg, state, _utcnow()) or retrain_forced:
                        print(f"[refresh] retrain cycle @ {_utcnow():%Y-%m-%d %H:%M:%S}"
                              f"{' (drift-forced)' if retrain_forced else ''}", flush=True)
                        rep = run_retrain_cycle(cfg=cfg, state=state,
                                                smoke=smoke_retrain, lock=False)
                        print(f"[refresh] retrain done: decision={rep.get('gate', {}).get('decision')} "
                              f"{rep.get('candidate_30d_rmse')} vs incumbent "
                              f"{rep.get('incumbent_30d_rmse')}", flush=True)
                        if retrain_forced:
                            state.update(drift_streak=0,
                                         drift_forced_at=_utcnow().isoformat(timespec="seconds"))
            except Exception as exc:  # noqa: BLE001
                state.record("scheduler.error", ok=False, msg=f"retrain cycle: {exc}")
                print(f"[refresh] retrain cycle failed: {exc}", flush=True)
        if once:
            return
        time.sleep(poll_s)


def next_due_human(cfg, state: RefreshState) -> dict:
    now = _utcnow()
    return {
        "next_refresh_due": state.get("next_refresh_due"),
        "next_retrain_due": state.get("next_retrain_due"),
        "refresh_due_now": refresh_due(cfg, state, now),
        "retrain_due_now": retrain_due(cfg, state, now),
        "models_frozen": models_frozen(cfg, now) if cfg is not None else False,
        "models_frozen_until": getattr(cfg, "models_frozen_until", "") if cfg is not None else "",
    }


# --- CLI handlers (kept importable via refresh.cli shim) ---------------------
def _config(args):
    return load_config(path=Path(args.config) if getattr(args, "config", None)
                       else core.config_path())


def _cmd_validate(args) -> int:
    try:
        cfg = _config(args)
        cfg.validate()
        print(f"config OK @ {core.config_path()}")
        print(json.dumps({k: v for k, v in cfg.__dict__.items()}, indent=2, default=str))
        return 0
    except ConfigError as exc:
        print(f"[error] {exc}")
        return 2


def _status(cfg, state):
    out = engines.status()
    out["schedule"] = next_due_human(cfg, state)
    out["feature_mode"] = (core.read_runtime() or {}).get("feature_mode")
    out["last_retrain"] = state.get("last_retrain_ts")
    out["last_retrain_ok"] = state.get("last_retrain_ok")
    out["last_retrain_error"] = state.get("last_retrain_error")
    return out


def _cmd_status(args) -> int:
    print(json.dumps(_status(_config(args), RefreshState()), indent=2, default=str))
    return 0


def _cmd_dry_run(args) -> int:
    rep = run_forecast_cycle(
        cfg=_config(args), dry_run=True, do_fetch=not args.no_fetch,
        do_features=not args.no_features, do_inference=not args.no_inference,
        max_stations=args.stations or 0)
    print(json.dumps(rep, indent=2, default=str))
    return 0


def _cmd_forecast(args) -> int:
    rep = run_forecast_cycle(
        cfg=_config(args), dry_run=False, do_fetch=not args.no_fetch,
        do_features=not args.no_features, do_inference=not args.no_inference,
        max_stations=args.stations or 0, skip_unchanged=not args.force)
    ok = bool(rep.get("inference", {}).get("published"))
    print(json.dumps(rep, indent=2, default=str))
    return 0 if ok else 1


def _cmd_retrain(args) -> int:
    rep = run_retrain_cycle(cfg=_config(args), smoke=args.smoke,
                            feature_mode=args.feature_mode or "flat")
    print(json.dumps(rep, indent=2, default=str))
    return 0 if rep.get("promoted") else 1


def _cmd_schedule(args) -> int:
    scheduler_loop(cfg=_config(args), once=args.once, poll_s=args.poll,
                   do_fetch=not args.no_fetch, do_features=not args.no_features,
                   do_inference=True, smoke_retrain=args.smoke)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m refresh.cli")
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate-config")
    p.set_defaults(fn=_cmd_validate)
    p = sub.add_parser("status")
    p.set_defaults(fn=_cmd_status)
    p = sub.add_parser("dry-run")
    for flag in ("--no-fetch", "--no-features", "--no-inference"):
        p.add_argument(flag, action="store_true")
    p.add_argument("--stations", type=int, default=0)
    p.set_defaults(fn=_cmd_dry_run)
    p = sub.add_parser("forecast")
    for flag in ("--no-fetch", "--no-features", "--no-inference"):
        p.add_argument(flag, action="store_true")
    p.add_argument("--stations", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="recompute even stations whose anchor has not changed")
    p.set_defaults(fn=_cmd_forecast)
    p = sub.add_parser("retrain")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--feature-mode", choices=["flat", "future"], default="flat")
    p.set_defaults(fn=_cmd_retrain)
    p = sub.add_parser("schedule")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-features", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(fn=_cmd_schedule)

    args = ap.parse_args(argv)
    return args.fn(args)


# Backwards-compatible module aliases (old refresh.pipeline/scheduler share this module).
pipeline = scheduler = sys.modules[__name__]
publish = core

if __name__ == "__main__":
    sys.exit(main())
