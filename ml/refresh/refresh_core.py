"""refresh_core — config, restart-safe state/lock, wall-clock schedule, atomic publish.

Merged from the former config/state/schedule/publish modules. All filesystem
commit points stay here: state.json/journal.jsonl, refresh.lock, staged forecast
files, forecasts.parquet, meta.json and runtime.json.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFRESH_DIR = ROOT / "data" / "refresh"
FEATURE_MODES = ("flat", "future")

# --- config -----------------------------------------------------------------
class ConfigError(ValueError):
    pass


@dataclass
class RefreshConfig:
    refresh_interval_hours: float = 6.0
    refresh_at_hours: list[int] = field(default_factory=lambda: [1, 7, 13, 19])
    retrain_cadence_hours: float = 24.0
    retrain_at_hours: list[int] = field(default_factory=lambda: [2, 14])
    stale_after_hours: float = 30.0
    retry_max: int = 3
    retry_backoff_s: float = 10.0
    fetch_workers: int = 8
    openmeteo_forecast_days: int = 16
    # "fetch ok + 0 new rows" is a benign no-op; re-arm at least this far out so
    # an empty source cannot hot-loop the daemon. Real failures still raise.
    no_new_data_backoff_min: float = 90.0
    promote_tolerance: float = 0.0
    models_frozen_until: str = ""
    candidate_holdout_days: int = 90
    candidate_backtest_window_days: int = 90
    backtest_stations: int = 60
    anchor_spacing_days: int = 14
    coverage_target: float = 0.90
    # Row caps bound the long-frame's peak RAM on the shared box.
    candidate_train_row_cap: int = 1_200_000
    candidate_val_row_cap: int = 300_000
    feature_mode: str = "flat"               # "future" only after backtest validation
    min_fresh_stations: int = 20
    max_stations: int = 0                    # 0 = all stations in the aligned table

    def validate(self) -> None:
        if self.refresh_interval_hours <= 0:
            raise ConfigError("refresh_interval_hours must be > 0")
        if self.retrain_cadence_hours <= 0:
            raise ConfigError("retrain_cadence_hours must be > 0")
        if self.stale_after_hours <= 0:
            raise ConfigError("stale_after_hours must be > 0")
        if self.no_new_data_backoff_min <= 0:
            raise ConfigError("no_new_data_backoff_min must be > 0")
        if self.feature_mode not in FEATURE_MODES:
            raise ConfigError(f"feature_mode must be one of {FEATURE_MODES}")
        if not (0.0 < self.coverage_target <= 1.0):
            raise ConfigError("coverage_target must be in (0, 1]")
        if self.candidate_holdout_days <= 0 or self.candidate_backtest_window_days <= 0:
            raise ConfigError("holdout / backtest windows must be > 0 days")
        if self.models_frozen_until:
            try:
                _parse_instant(self.models_frozen_until)
            except ValueError as exc:
                raise ConfigError(f"models_frozen_until not ISO-8601: {exc}") from exc

    def ensure_dirs(self) -> None:
        REFRESH_DIR.mkdir(parents=True, exist_ok=True)


def config_path() -> Path:
    return REFRESH_DIR / "refresh_config.json"


def load_config(path: Path | None = None, write_if_missing: bool = True) -> RefreshConfig:
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"corrupt refresh config {p}: {exc}") from exc
    cfg = RefreshConfig(**{k: v for k, v in data.items()
                           if k in RefreshConfig.__dataclass_fields__})
    _apply_env(cfg)
    # env override is a debug-only escape hatch for feature_mode
    _mode = os.environ.get("AQUIS_FEATURE_MODE")
    if _mode:
        cfg.feature_mode = _mode
    cfg.validate()
    if write_if_missing and not p.exists():
        p.write_text(json.dumps(asdict(cfg), indent=2))
    return cfg


def _parse_instant(raw: str):
    """Parse ISO-8601, tolerating a trailing Z and naive strings."""
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def models_frozen(cfg: RefreshConfig, now=None) -> bool:
    """True while models_frozen_until is set and still in the future (fail-safe on malformed)."""
    raw = getattr(cfg, "models_frozen_until", "") or ""
    if not raw:
        return False
    now = now or datetime.now(timezone.utc)
    try:
        until = _parse_instant(raw)
    except ValueError:
        return True
    return now < until


def _apply_env(cfg: RefreshConfig) -> None:
    env = {
        "refresh_interval_hours": "AQUIS_REFRESH_INTERVAL_HOURS",
        "retrain_cadence_hours": "AQUIS_RETRAIN_CADENCE_HOURS",
        "stale_after_hours": "AQUIS_STALE_AFTER_HOURS",
        "feature_mode": "AQUIS_FEATURE_MODE",
    }
    for attr, key in env.items():
        v = os.environ.get(key)
        if not v:
            continue
        current = getattr(cfg, attr)
        setattr(cfg, attr, v if isinstance(current, str) else float(v))


# --- state / journal / lock --------------------------------------------------
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

try:  # POSIX
    import fcntl
except ImportError:  # Windows msvcrt byte-range locking
    fcntl = None
    import msvcrt


class StateError(RuntimeError):
    pass


class LockBusy(StateError):
    pass


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(iso: str):
    try:
        return datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return datetime.now()


def _as_iso(dt) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def state_path() -> Path:
    return REFRESH_DIR / "state.json"


def journal_path() -> Path:
    return REFRESH_DIR / "journal.jsonl"


def lock_path() -> Path:
    return REFRESH_DIR / "refresh.lock"


def _atomic_write(path: Path, text: str) -> None:
    # PID-suffixed tmp: daemon + manual CLI can never trash each other's buffer.
    tmp = path.with_suffix(f"{path.suffix}.tmp{os.getpid()}")
    tmp.write_text(text)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _lock_fd(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)


def _unlock_fd(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    else:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


def os_open(path: Path) -> int:
    return os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)


@contextlib.contextmanager
def refresh_lock(timeout_s: float = 0.0, poll_s: float = 0.5):
    """Non-blocking exclusive lock; a busy lock raises LockBusy."""
    REFRESH_DIR.mkdir(parents=True, exist_ok=True)
    fd = os_open(lock_path())
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _lock_fd(fd)
            break
        except OSError as exc:
            if fcntl is not None and not isinstance(exc, BlockingIOError):
                os.close(fd)
                raise
            if not timeout_s or time.monotonic() >= deadline:
                os.close(fd)
                raise LockBusy(f"refresh already running (lock {lock_path()})") from None
            time.sleep(poll_s)
    try:
        yield
    finally:
        _unlock_fd(fd)
        os.close(fd)


class RefreshState:
    """Thin, atomic, journaled key/value store backed by state.json."""
    def __init__(self, path: Path | None = None):
        self._path = path or state_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = dict(DEFAULT_STATE)
        if self._path.exists():
            try:
                self._data.update(json.loads(self._path.read_text()))
            except Exception as exc:  # noqa: BLE001 - a corrupt state must not kill the run
                # Re-arm instead of leaving due=None (that caused a rapid-cycle
                # storm), and preserve the broken bytes for forensics.
                try:
                    raw = self._path.read_bytes()
                    evidence = self._path.with_suffix(
                        ".corrupt.{}.{}".format(
                            self._path.suffix.lstrip("."),
                            _utcnow_iso().replace(":", "").replace("+00:00", "z")))
                    evidence.write_bytes(raw)
                    trunc_hint = "truncated<{}B".format(len(raw)) if len(raw) < 512 else "{}B".format(len(raw))
                except Exception:  # noqa: BLE001
                    trunc_hint = "unreadable"
                now_iso = _utcnow_iso()
                if self._data.get("next_refresh_due") is None:
                    self._data["next_refresh_due"] = _as_iso(
                        _parse_iso(now_iso) + timedelta(minutes=90))
                if self._data.get("next_retrain_due") is None:
                    self._data["next_retrain_due"] = _as_iso(
                        _parse_iso(now_iso) + timedelta(hours=6))
                self.record("state.corrupt", ok=False, msg=f"state.json unreadable ({trunc_hint}); resetting")
        self._guard = None

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def snapshot(self) -> dict:
        return json.loads(json.dumps(self._data))

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


# --- schedule ---------------------------------------------------------------
def next_grid_due(cfg: RefreshConfig, now: datetime,
                  hours: Iterable[int] | None = None) -> datetime:
    """Next fixed wall-clock slot (local timezone) strictly after ``now``, as aware UTC."""
    slots = sorted({int(h) % 24 for h in (hours if hours is not None else cfg.refresh_at_hours)})
    if not slots:
        return now + timedelta(hours=cfg.refresh_interval_hours)
    local = now.astimezone()
    for hour in slots:
        cand = local.replace(hour=hour, minute=0, second=0, microsecond=0)
        if cand > local:
            return cand.astimezone(timezone.utc)
    cand = (local + timedelta(days=1)).replace(
        hour=slots[0], minute=0, second=0, microsecond=0)
    return cand.astimezone(timezone.utc)


# --- publish ----------------------------------------------------------------
FORECASTS_DIR = REFRESH_DIR / "forecasts"
FORECAST_ARCHIVE = REFRESH_DIR / "forecast_archive"
STAGING_DIR = REFRESH_DIR / "staging"
META_PATH = REFRESH_DIR / "meta.json"
RUNTIME_PATH = REFRESH_DIR / "runtime.json"
PARCEL_PATH = REFRESH_DIR / "forecasts.parquet"


class PublishError(RuntimeError):
    pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def station_key(station: str) -> str:
    return hashlib.sha1(str(station).encode("utf-8")).hexdigest()[:16]


def station_forecast_path(station: str) -> Path:
    return FORECASTS_DIR / f"{station_key(station)}.json"


def read_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def read_runtime() -> dict:
    if not RUNTIME_PATH.exists():
        return {"feature_mode": "flat", "validated": False}
    try:
        return json.loads(RUNTIME_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {"feature_mode": "flat", "validated": False}


def set_runtime(feature_mode: str, validated: bool,
                model_version: str | None = None) -> None:
    cur = read_runtime()  # preserve traj_backend / forward_backend etc.
    cur.update({
        "feature_mode": feature_mode, "validated": validated,
        "model_version": model_version, "written_at": _utcnow_iso(),
    })
    atomic_write_json(RUNTIME_PATH, cur)


def publish_forecasts(forecasts: list[dict], *, engine: str, model_version: str | None,
                      model_trained_at: str | None, source_report: dict | None = None,
                      keep_staging: bool = True) -> dict:
    """Publish per-station forecasts + fleet parquet + the commit manifest (meta.json last)."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    clean, bad = [], []
    for fc in forecasts:
        if isinstance(fc, dict) and "error" in fc:
            bad.append({"station": fc.get("station"), "error": fc["error"]})
            continue
        atomic_write_json(STAGING_DIR / f"{station_key(fc['station'])}.json", fc)
        clean.append(fc)
    if not clean and not bad:
        raise PublishError("no forecasts to publish")

    rows = []
    for fc in clean:
        t30 = fc.get("trajectory_30d", {})
        rows.append({
            "station": fc.get("station", None),
            "anchor_time": fc.get("anchor_time"),
            "anchor_gwl": fc.get("anchor_gwl"),
            "q50_720": t30.get("level"),
            "q05_720": t30.get("q05"),
            "q95_720": t30.get("q95"),
            "change_720": t30.get("change"),
            "direction": (fc.get("direction") or {}).get("label"),
            "confidence_30d": (fc.get("overall_confidence") or {}).get("level"),
            "forecast_generated": _utcnow_iso(),
        })
    atomic_write_parquet(STAGING_DIR / "forecasts.parquet", pd.DataFrame(rows))
    meta = {
        "schema": 1,
        "forecast_generation_ts": _utcnow_iso(),
        "engine": engine,
        "model_version": model_version,
        "model_trained_at": model_trained_at,
        "stations_forecasted": int(len(clean)),
        "stations_failed": bad,
        "source_report": source_report or {},
        "horizon": {"max_h": 120, "step_hours": 6, "days": 30},
        "stale_after_hours": None,  # filled by update_freshness
    }
    FORECAST_ARCHIVE.mkdir(parents=True, exist_ok=True)
    for fc in clean:
        dst = station_forecast_path(fc["station"])
        if dst.exists():
            key = station_key(fc["station"])
            anchor = str(fc.get("anchor_time", "unknown")).replace(":", "").replace(" ", "_")
            archived = FORECAST_ARCHIVE / f"{key}-{anchor}.json"
            if not archived.exists():
                prev = json.loads(dst.read_text())
                prev.setdefault("forecast_generated", meta.get("forecast_generation_ts"))
                atomic_write_json(archived, prev)
        src = STAGING_DIR / f"{station_key(fc['station'])}.json"
        os.replace(src, dst)
    os.replace(STAGING_DIR / "forecasts.parquet", PARCEL_PATH)
    atomic_write_json(META_PATH, meta)
    if not keep_staging:
        for p in STAGING_DIR.glob("*"):
            p.unlink(missing_ok=True)
    return meta


def update_freshness(meta: dict, stale_after_hours: float,
                     last_refresh_ts: str | None) -> dict:
    """Augment the committed meta with a data-status verdict + per-source lag."""
    meta["stale_after_hours"] = float(stale_after_hours)
    now = pd.Timestamp.now(tz="UTC")
    last = None
    if last_refresh_ts and last_refresh_ts != "now":
        ts = pd.Timestamp(last_refresh_ts)
        last = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    lag_h = float((now - last).total_seconds() / 3600) if last is not None else None
    if last is None or lag_h is None or lag_h > float(stale_after_hours):
        status, reason = "stale", ("no refresh within the configured window"
                                   if last is None else f"last refresh {lag_h:.1f}h ago")
    else:
        status, reason = "fresh", f"last refresh {lag_h:.1f}h ago"
    meta["data_status"] = status
    meta["stale_reason"] = reason if status == "stale" else None
    meta["last_refresh_ts"] = last_refresh_ts
    meta["data_lag_hours"] = round(lag_h, 2) if lag_h is not None else None
    atomic_write_json(META_PATH, {**meta, "_touched": _utcnow_iso()})
    return meta


def freshness_for(station: str) -> dict:
    """Per-station freshness: latest observed, generated, model, data status."""
    meta = read_meta()
    fc_path = station_forecast_path(station)
    out = {
        "forecast_generated": meta.get("forecast_generation_ts"),
        "model_version": meta.get("model_version"),
        "model_trained_at": meta.get("model_trained_at"),
        "engine": meta.get("engine"),
        "data_status": meta.get("data_status", "unknown"),
        "stale_reason": meta.get("stale_reason"),
        "data_lag_hours": meta.get("data_lag_hours"),
        "published": fc_path.exists(),
        "source_report": meta.get("source_report", {}),
    }
    if fc_path.exists():
        try:
            fc = json.loads(fc_path.read_text())
            out["latest_observed"] = fc.get("anchor_time")
            out["anchor_gwl"] = fc.get("anchor_gwl")
            out["stations_forecasted"] = meta.get("stations_forecasted")
        except Exception:  # noqa: BLE001
            pass
    return out


# Backwards-compatible module aliases (old refresh.config/state/... now share this module).
config = state = schedule = publish = sys.modules[__name__]

if __name__ == "__main__":
    print(json.dumps(read_meta(), indent=2, default=str))
