"""refresh/scheduler.py — restart-safe long-running scheduler daemon.

Polls every ``poll_s`` seconds, persists ``next_refresh_due`` / ``next_retrain_due``
in data/refresh/state.json, and re-arms fresh due-times after every action, so a
crash/restart mid-cycle simply resumes from the persisted schedule. The whole
cycle holds the flock, and ``scheduler_loop`` takes the flock itself before
deciding, so a manual CLI run and the daemon can never double-fire.

Cadences are independent (config.refresh_interval_hours vs
config.retrain_cadence_hours) and the retrain additionally only fires inside its
allowed local-hour windows (config.retrain_at_hours).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from refresh import pipeline
from refresh.config import load_config
from refresh.state import RefreshState, refresh_lock


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    cursor = _parse(state.get("next_retrain_due")) or (now - timedelta(hours=cfg.retrain_cadence_hours))
    return now >= cursor


def scheduler_loop(*, cfg=None, state=None, poll_s: float = 60.0, once: bool = False,
                   do_fetch: bool = True, do_features: bool = True,
                   do_inference: bool = True, smoke_retrain: bool = False) -> None:
    cfg = cfg or load_config()
    state = state or RefreshState()
    print(f"[refresh] scheduler started cwd={__file__} poll={poll_s}s "
          f"refresh_every={cfg.refresh_interval_hours}h retrain_every={cfg.retrain_cadence_hours}h",
          flush=True)
    while True:
        now = _utcnow()
        if refresh_due(cfg, state, now):
            try:
                with refresh_lock():
                    if refresh_due(cfg, state, _utcnow()):
                        print(f"[refresh] forecast cycle @ {_utcnow():%Y-%m-%d %H:%M:%S}", flush=True)
                        rep = pipeline.run_forecast_cycle(
                            cfg=cfg, state=state, do_fetch=do_fetch,
                            do_features=do_features, do_inference=do_inference, lock=False)
                        print(f"[refresh] cycle done: {rep.get('inference', {})}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep daemon alive
                state.record("scheduler.error", ok=False, msg=f"forecast cycle: {exc}")
                print(f"[refresh] forecast cycle failed: {exc}", flush=True)

        if retrain_due(cfg, state, _utcnow()):
            try:
                with refresh_lock():
                    if retrain_due(cfg, state, _utcnow()):
                        print(f"[refresh] retrain cycle @ {_utcnow():%Y-%m-%d %H:%M:%S}", flush=True)
                        rep = pipeline.run_retrain_cycle(cfg=cfg, state=state,
                                                         smoke=smoke_retrain, lock=False)
                        print(f"[refresh] retrain done: decision={rep.get('gate', {}).get('decision')} "
                              f"{rep.get('candidate_30d_rmse')} vs incumbent "
                              f"{rep.get('incumbent_30d_rmse')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                state.record("scheduler.error", ok=False, msg=f"retrain cycle: {exc}")
                print(f"[refresh] retrain cycle failed: {exc}", flush=True)

        if once:
            return
        time.sleep(poll_s)


def next_due_human(cfg, state: RefreshState) -> dict:
    now = _utcnow()
    dr = _parse(state.get("next_refresh_due")) or now
    du = _parse(state.get("next_retrain_due"))
    return {
        "next_refresh_due": state.get("next_refresh_due"),
        "next_retrain_due": state.get("next_retrain_due"),
        "refresh_due_now": refresh_due(cfg, state, now),
        "retrain_due_now": retrain_due(cfg, state, now),
    }