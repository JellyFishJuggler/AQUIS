"""refresh/state.py — restart-safe pipeline state, journal and process lock.

The scheduler and the manual CLI both take a non-blocking ``flock`` on
``data/refresh/refresh.lock`` before touching anything: two refreshes can never
overlap. ``state.json`` persists the last-run / next-due timestamps so the
scheduler survives application restarts, and ``journal.jsonl`` keeps an audit
trail of every fetch / retrain / publish attempt.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from pathlib import Path

from refresh.config import REFRESH_DIR

DEFAULT_STATE: dict = {
    "schema": 1,
    "model_version": None,
    "model_trained_at": None,
    "feature_mode": "flat",
    "feature_mode_validated_at": None,
    "last_forecast_refresh_ts": None,
    "last_forecast_refresh_ok": None,
    "last_forecast_refresh_error": None,
    "last_retrain_ts": None,
    "last_retrain_ok": None,
    "last_retrain_error": None,
    "forecast_generation_ts": None,
    "forecast_generation_model": None,
    "next_refresh_due": None,
    "next_retrain_due": None,
    "per_source_last_success": {},
    "stale_reason": None,
    "run_history": [],
}

JOURNAL_MAX = 500


class StateError(RuntimeError):
    pass


class LockBusy(StateError):
    pass


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state_path() -> Path:
    return REFRESH_DIR / "state.json"


def journal_path() -> Path:
    return REFRESH_DIR / "journal.jsonl"


def lock_path() -> Path:
    return REFRESH_DIR / "refresh.lock"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


@contextlib.contextmanager
def refresh_lock(timeout_s: float = 0.0, poll_s: float = 0.5):
    """Exclusive flock guard. With timeout_s=0 it is non-blocking; a busy lock
    raises LockBusy (so a second cron/daemon start never double-runs)."""
    REFRESH_DIR.mkdir(parents=True, exist_ok=True)
    fd = os_open(lock_path())
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not timeout_s or time.monotonic() >= deadline:
                os.close(fd)
                raise LockBusy(f"refresh already running (lock {lock_path()})") from None
            time.sleep(poll_s)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def os_open(path: Path) -> int:
    import os
    return os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)


class RefreshState:
    """Thin, atomic, journaled key/value store backed by state.json."""

    def __init__(self, path: Path | None = None):
        self._path = path or state_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = dict(DEFAULT_STATE)
        if self._path.exists():
            try:
                self._data.update(json.loads(self._path.read_text()))
            except Exception:  # noqa: BLE001 - a corrupt state must not kill the run
                self.record("state.corrupt", ok=False, msg="state.json unreadable; resetting")
        self._guard = None

    # -- access ---------------------------------------------------------------
    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def snapshot(self) -> dict:
        return json.loads(json.dumps(self._data))

    # -- persistence ----------------------------------------------------------
    def update(self, **fields) -> None:
        for k, v in fields.items():
            if v is not None:
                self._data[k] = v
        _atomic_write(self._path, json.dumps(self._data, indent=2, default=str))

    def record(self, kind: str, *, ok: bool, msg: str = "", **extra) -> None:
        """Append an audit event and a bounded run-history entry."""
        ev = {"ts": _utcnow_iso(), "kind": kind, "ok": ok, "msg": str(msg), **extra}
        good = json.dumps(ev)
        with open(journal_path(), "a") as fh:
            fh.write(good + "\n")
        self._data.setdefault("run_history", []).append(ev)
        del self._data["run_history"][-JOURNAL_MAX:]
        _atomic_write(self._path, json.dumps(self._data, indent=2, default=str))
        return None