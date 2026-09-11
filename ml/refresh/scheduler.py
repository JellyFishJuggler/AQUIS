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

from refresh import pipeline, verification
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


def drift_retrain_due(cfg, state: RefreshState, now: datetime | None = None) -> bool:
    """Drift-triggered retrain override.

    When a completed verification summary trips the drift rule for
    DRIFT_CONSECUTIVE_CYCLES in a row, the daemon refits immediately, ignoring
    the retrain_at_hours window — a degrading model costs accuracy every cycle
    it keeps serving. The streak is persisted in state.json so daemon restarts
    don't forget it, and is keyed on the summary's updated_ts so one summary is
    counted exactly once (not once per daemon poll).
    """
    now = now or _utcnow()
    vd = verification.drift_status(None)
    summary_ts = vd.get("updated_ts")
    counted_ts = state.get("drift_counted_ts")
    tripped = bool(vd.get("tripped"))
    streak = int(state.get("drift_streak", 0))

    if summary_ts and summary_ts != counted_ts:
        if tripped:
            streak += 1
            state.update(drift_streak=streak,
                         drift_counted_ts=summary_ts,
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

        retrain_forced = drift_retrain_due(cfg, state, _utcnow())
        if retrain_due(cfg, state, _utcnow()) or retrain_forced:
            try:
                with refresh_lock():
                    if retrain_due(cfg, state, _utcnow()) or retrain_forced:
                        print(f"[refresh] retrain cycle @ {_utcnow():%Y-%m-%d %H:%M:%S}"
                              f"{' (drift-forced)' if retrain_forced else ''}", flush=True)
                        rep = pipeline.run_retrain_cycle(cfg=cfg, state=state,
                                                         smoke=smoke_retrain, lock=False)
                        print(f"[refresh] retrain done: decision={rep.get('gate', {}).get('decision')} "
                              f"{rep.get('candidate_30d_rmse')} vs incumbent "
                              f"{rep.get('incumbent_30d_rmse')}", flush=True)
                        # any completed refit resets the drift streak: the model
                        # gets a fresh validation period either way
                        if retrain_forced:
                            state.update(drift_streak=0, drift_forced_at=_utcnow().isoformat(timespec="seconds"))
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