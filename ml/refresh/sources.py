"""refresh/sources.py — data fetching layer for the 6-hour refresh.

Every source fetch is wrapped with a retry/backoff policy, never mutates live
artifacts on failure, and returns a structured report (fetched_at, new rows,
per-district status) that the pipeline folds into the freshness metadata.

Sources refreshed every cycle:
  * Groundwater observations  — NWIC/NWDP datastore_search (live 2026+ resource),
    incremental per district, merged into data/processed/common.parquet exactly
    as 13_refresh_nwic does (deduped, chronological, dtype-coerced).
  * Weather drivers          — Open-Meteo archive + forecast (16 d) per district
    centroid, merged into data/cfs/openmeteo_weather_daily.parquet. This feeds
    the Open-Meteo->climatology fallback used by the trajectory engine.
  * Driver family coverage   — last-success timestamp per source lands in state so
    the UI can show which family is driving the forecast vs climatology.

Never fabricated: a failed source keeps the previous parquet and is reported as
``ok=False`` for that source; the pipeline then decides keep-previous/stale.
"""

from __future__ import annotations

import importlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat(timespec="seconds")


def _with_retries(fn, retry_max: int, backoff_s: float, what: str) -> tuple[bool, object, str]:
    last_err = ""
    for attempt in range(1, retry_max + 1):
        try:
            return True, fn(), ""
        except Exception as exc:  # noqa: BLE001 - robust against any source error
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retry_max:
                time.sleep(backoff_s * attempt)
    return False, None, f"{what} failed after {retry_max} tries ({last_err})"


def _import(name: str):
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# Groundwater observations
# ---------------------------------------------------------------------------
def fetch_gwl(*, parquet: Path, districts: list[str] | None = None,
              dry_run: bool = False, retry_max: int = 3, backoff_s: float = 10.0,
              sleep: float = 0.2, fetch_fn=None) -> dict:
    """Incremental per-district NWIC GWL fetch -> merge into ``parquet``.

    ``fetch_fn`` is injectable for tests (signature identical to
    ``refresh.sources._fetch_gwl_impl``); it defaults to the real 13_refresh_nwic
    logic. Returns a report; on success it mutates ``parquet`` (unless dry_run).
    """
    if fetch_fn is None:
        fetch_fn = _fetch_gwl_impl
    ok, result, err = _with_retries(
        lambda: fetch_fn(parquet=parquet, districts=districts, dry_run=dry_run, sleep=sleep),
        retry_max, backoff_s, "gwl (nwic)")
    report = result or {}
    report["fetched_at"] = _utcnow_iso()
    report["ok"] = bool(ok)
    report["error"] = err or report.get("error")
    report["source"] = "nwic-gwl-live"
    return report


def _fetch_gwl_impl(*, parquet: Path, districts: list[str] | None,
                    dry_run: bool, sleep: float) -> dict:
    nwic = _import("ml.data.raw.nwic")
    refresh_src = _import("13_refresh_nwic")
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    dists = districts or nwic.get_aquis_districts(parquet)
    new_frames, total_new = refresh_src.refresh(
        dists, parquet, dry_run=dry_run, sleep=sleep)
    per = {}
    for d, frame in zip(dists, new_frames):
        per[d] = {"new_rows": int(len(frame))}
    failures = {}
    report = {
        "districts": len(dists), "new_rows": int(total_new),
        "per_district": per, "per_district_failures": failures,
    }
    return report


# ---------------------------------------------------------------------------
# Weather drivers (Open-Meteo)
# ---------------------------------------------------------------------------
def fetch_weather(*, out_parquet: Path, out_meta: Path, districts_csv: Path | None = None,
                  start_days: int = 400, forecast_days: int = 16,
                  dry_run: bool = False, retry_max: int = 3, backoff_s: float = 10.0,
                  workers: int = 8, fetch_fn=None) -> dict:
    """Fetch Open-Meteo history+forecast per district, merge atomically."""
    if fetch_fn is None:
        fetch_fn = _fetch_weather_impl
    ok, result, err = _with_retries(
        lambda: fetch_fn(out_parquet=out_parquet, out_meta=out_meta,
                         districts_csv=districts_csv, start_days=start_days,
                         forecast_days=forecast_days, dry_run=dry_run, workers=workers),
        retry_max, backoff_s, "open-meteo")
    report = result or {}
    report["fetched_at"] = _utcnow_iso()
    report["ok"] = bool(ok)
    report["error"] = err or report.get("error")
    report["source"] = "open-meteo-archive+forecast"
    report["forecast_days"] = forecast_days
    return report


def _fetch_weather_impl(*, out_parquet: Path, out_meta: Path, districts_csv: Path | None,
                        start_days: int, forecast_days: int, dry_run: bool,
                        workers: int) -> dict:
    om = _import("20_openmeteo_fetch")
    import concurrent.futures as cf

    csv = districts_csv or (ROOT / "data" / "meta" / "selected_gwl_stations.csv")
    cents = om.district_centroids() if not csv else _centroids_from_csv(csv)
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=start_days)).strftime("%Y-%m-%d")
    end = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")

    frames, failures = [], {}
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_one_district, om, row, start, end, forecast_days): str(row["District"])
                for _, row in cents.iterrows()}
        for fut in cf.as_completed(futs):
            dist = futs[fut]
            try:
                frames.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                failures[dist] = f"{type(exc).__name__}: {exc}"

    if not frames:
        raise RuntimeError(f"open-meteo: no district data fetched ({len(failures)} failed)")

    new = pd.concat(frames, ignore_index=True)
    if out_parquet.exists() and not dry_run:
        prev = pd.read_parquet(out_parquet)
        merged = pd.concat([prev, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["District", "date"], keep="last")
        merged = merged.sort_values(["District", "date"]).reset_index(drop=True)
    else:
        merged = new

    if not dry_run:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(out_parquet, merged)
        meta = {
            "source": "open-meteo", "forecast_days": forecast_days,
            "districts": int(merged["District"].nunique()), "rows": int(len(merged)),
            "fetched_at": _utcnow_iso(), "district_failures": failures,
        }
        atomic_write_json(out_meta, meta)
    return {"districts": int(new["District"].nunique()), "rows": int(len(new)),
            "merged_rows": int(len(merged)) if not dry_run else None,
            "district_failures": failures}


def _centroids_from_csv(csv: Path) -> pd.DataFrame:
    sel = pd.read_csv(csv)
    c = sel.groupby("District")[["Latitude", "Longitude"]].mean()
    c["lat"], c["lon"] = c["Latitude"], c["Longitude"]
    return c.reset_index()[["District", "lat", "lon"]]


def _one_district(om, row, start: str, end: str, fwd_days: int) -> pd.DataFrame:
    d = om.fetch_district(float(row["lat"]), float(row["lon"]), start, end, fwd_days)
    d["District"] = str(row["District"])
    d["source"] = "open-meteo"
    return d


from refresh.publish import atomic_write_json, atomic_write_parquet  # noqa: E402  (re-export for tests)


# ---------------------------------------------------------------------------
# River-level forecast (CWC / FMISC UP flood dashboard, 7 days)
# ---------------------------------------------------------------------------
def fetch_river_forecast(*, out_parquet: Path, out_meta: Path,
                         dry_run: bool = False, retry_max: int = 3,
                         backoff_s: float = 10.0, fetch_fn=None) -> dict:
    """Fetch the CWC 7-day river water-level forecast -> rolling parquet cache.

    ``fetch_fn`` is injectable for tests (same signature as
    ``refresh.sources._fetch_river_forecast_impl``). A failed fetch keeps the
    previous cache and is reported ``ok=False`` (the pipeline decides whether
    that makes the cycle stale).
    """
    if fetch_fn is None:
        fetch_fn = _fetch_river_forecast_impl
    ok, result, err = _with_retries(
        lambda: fetch_fn(out_parquet=out_parquet, out_meta=out_meta, dry_run=dry_run),
        retry_max, backoff_s, "river-forecast (cwc)")
    report = result or {}
    report["fetched_at"] = _utcnow_iso()
    report["ok"] = bool(ok)
    report["error"] = err or report.get("error")
    report["source"] = "cwc-fmisc-up-flood-forecast"
    report["forecast_days"] = 7
    return report


def _fetch_river_forecast_impl(*, out_parquet: Path, out_meta: Path,
                               dry_run: bool) -> dict:
    cwc = _import("18_cwc_river_forecast")
    df, meta = cwc.fetch_forecast(out_parquet=out_parquet, out_meta=out_meta,
                                  dry_run=dry_run)
    return {
        "sites": int(df["site"].nunique()),
        "rivers": sorted(df["river"].unique().tolist()),
        "districts": sorted(df["district"].unique().tolist()),
        "rows": int(len(df)),
        "days": meta.get("forecast_days"),
    }


# ---------------------------------------------------------------------------
# Aggregate report for one refresh cycle
# ---------------------------------------------------------------------------
def fetch_all(cfg, *, dry_run: bool = False, districts: list[str] | None = None,
              parquet: Path | None = None, fetch_fn_g=None, fetch_fn_w=None) -> dict:
    p = parquet or (ROOT / "data" / "processed" / "common.parquet")
    wq = ROOT / "data" / "cfs" / "openmeteo_weather_daily.parquet"
    wm = ROOT / "data" / "meta" / "openmeteo_meta.json"
    rfp = ROOT / "data" / "cfs" / "river_forecast_cwc.parquet"
    rfm = ROOT / "data" / "meta" / "river_forecast_cwc_meta.json"

    gwl = fetch_gwl(parquet=p, districts=districts, dry_run=dry_run,
                    retry_max=cfg.retry_max, backoff_s=cfg.retry_backoff_s,
                    fetch_fn=fetch_fn_g)
    weather = fetch_weather(out_parquet=wq, out_meta=wm,
                            forecast_days=cfg.openmeteo_forecast_days,
                            dry_run=dry_run, retry_max=cfg.retry_max,
                            backoff_s=cfg.retry_backoff_s, workers=cfg.fetch_workers,
                            fetch_fn=fetch_fn_w)
    river = fetch_river_forecast(out_parquet=rfp, out_meta=rfm,
                                 dry_run=dry_run, retry_max=cfg.retry_max,
                                 backoff_s=cfg.retry_backoff_s)
    return {
        "fetched_at": _utcnow_iso(),
        "gwl": gwl,
        "weather": weather,
        "river_forecast": river,
        "all_ok": bool(gwl.get("ok") and weather.get("ok")),
        "river_forecast_ok": bool(river.get("ok")),
    }