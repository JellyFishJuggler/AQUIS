"""refresh/publish.py — atomic forecast publication + freshness metadata.

The Forecast page falls back to on-the-fly computation, but production prefers
the artifacts published here. Everything is staged under ``data/refresh/staging/``
and swapped into ``data/refresh/forecasts/`` with ``os.replace``; the JSON
manifest ``data/refresh/meta.json`` is written LAST and acts as the commit point.
A reader that sees the manifest sees a fully consistent forecast set; a crash
before the manifest leaves the previous manifest in place (previous valid
forecast is retained) with the failure recorded by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from refresh.config import REFRESH_DIR

FORECASTS_DIR = REFRESH_DIR / "forecasts"
FORECAST_ARCHIVE = REFRESH_DIR / "forecast_archive"
STAGING_DIR = REFRESH_DIR / "staging"
META_PATH = REFRESH_DIR / "meta.json"
RUNTIME_PATH = REFRESH_DIR / "runtime.json"
PARCEL_PATH = REFRESH_DIR / "forecasts.parquet"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    atomic_write_json(RUNTIME_PATH, {
        "feature_mode": feature_mode, "validated": validated,
        "model_version": model_version,
        "written_at": _utcnow_iso(),
    })


def publish_forecasts(forecasts: list[dict], *, engine: str, model_version: str | None,
                      model_trained_at: str | None, source_report: dict | None = None,
                      keep_staging: bool = True) -> dict:
    """Publish per-station forecasts + a fleet parquet + the commit manifest.

    Returns the global meta dict that was committed. Raises on validation of the
    commit manifest; individual bad stations are recorded but do not abort the set.
    """
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    clean = []
    bad = []
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
        st = fc.get("station", None)
        rows.append({
            "station": st,
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
    fleet = pd.DataFrame(rows)
    atomic_write_parquet(STAGING_DIR / "forecasts.parquet", fleet)

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
        "stale_after_hours": None,  # filled by caller via update_freshness
    }

    # commit: swap staged per-station files, then parquet, manifest last.
    FORECAST_ARCHIVE.mkdir(parents=True, exist_ok=True)
    for fc in clean:
        dst = station_forecast_path(fc["station"])
        if dst.exists():
            # keep the previous generation in the archive so verification can
            # score realised values against it later (current forecast anchors
            # at the latest reading, so realisations land only afterwards)
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


class PublishError(RuntimeError):
    pass


if __name__ == "__main__":
    # trivial CLI: python refresh/publish.py status
    print(json.dumps(read_meta(), indent=2, default=str))