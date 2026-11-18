"""refresh_data — source fetches, guarded local rebuilds and future-step features.

Merged from sources/features/future_drivers. Source failures never mutate live
artifacts; rebuilt aligned/climatology tables are backup/restore guarded; future
features keep GWL-memory pinned at the anchor and take drivers/calendar from each
future timestamp. No fabricated values: every step records its source.
"""
from __future__ import annotations

import bisect
import importlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _hydrology import (API_TAUS, DEV_WINS, SLOPE_WINS, gwl_imputed_flag,
                        gwl_observed, gwl_trend_block, rain_memory_block)
from pipeline_data import STAGES
from refresh.refresh_core import atomic_write_json, atomic_write_parquet

ROOT = Path(__file__).resolve().parent.parent
# shared paths (former sources/features/future_drivers ROOT + climatology paths)
CLIM_PATH = ROOT / "data" / "meta" / "driver_climatology.parquet"
RAIN_CLIM_PATH = ROOT / "data" / "meta" / "district_rain_climatology.parquet"
OM_PATH = ROOT / "data" / "cfs" / "openmeteo_weather_daily.parquet"
RFC_PATH = ROOT / "data" / "cfs" / "river_forecast_cwc.parquet"
ALIGNED_6H = ROOT / "data" / "aligned" / "table_6h.parquet"
ALIGNED_DAILY = ROOT / "data" / "aligned" / "table.parquet"
SELECTED_GWL = ROOT / "data" / "selected" / "gwl_selected.csv"
WEATHER_PQ = OM_PATH

# --- fetch layer -------------------------------------------------------------
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


def _fetch_gwl_impl(*, parquet: Path, districts: list[str] | None,
                    dry_run: bool, sleep: float) -> dict:
    nwic = _import("ml.data.raw.nwic")
    pipeline = _import("pipeline_data")
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    dists = districts or nwic.get_aquis_districts(parquet)
    new_frames, total_new, source_max = pipeline.nwic_refresh(
        dists, parquet, dry_run=dry_run, sleep=sleep)
    per = {d: {"new_rows": int(len(frame))} for d, frame in zip(dists, new_frames)}
    return {
        "districts": len(dists), "new_rows": int(total_new),
        "per_district": per, "per_district_failures": {},
        "source_latest": (str(source_max) if source_max is not None else None),
    }


def fetch_gwl(*, parquet: Path, districts: list[str] | None = None,
              dry_run: bool = False, retry_max: int = 3, backoff_s: float = 10.0,
              sleep: float = 0.2, fetch_fn=None) -> dict:
    """Incremental NWIC GWL fetch -> merge into ``parquet``; injectable for tests."""
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


def _fetch_weather_impl(*, out_parquet: Path, out_meta: Path, districts_csv: Path | None,
                        start_days: int, forecast_days: int, dry_run: bool,
                        workers: int) -> dict:
    om = _import("openmeteo_weather")
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


def fetch_weather(*, out_parquet: Path, out_meta: Path, districts_csv: Path | None = None,
                  start_days: int = 400, forecast_days: int = 16,
                  dry_run: bool = False, retry_max: int = 3, backoff_s: float = 10.0,
                  workers: int = 8, fetch_fn=None) -> dict:
    """Open-Meteo history+forecast per district, merged atomically."""
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


def _fetch_river_forecast_impl(*, out_parquet: Path, out_meta: Path,
                               dry_run: bool) -> dict:
    cwc = _import("pipeline_data")
    df, meta = cwc.cwc_fetch_forecast(out_parquet=out_parquet, out_meta=out_meta,
                                      dry_run=dry_run)
    return {
        "sites": int(df["site"].nunique()),
        "rivers": sorted(df["river"].unique().tolist()),
        "districts": sorted(df["district"].unique().tolist()),
        "rows": int(len(df)),
        "days": meta.get("forecast_days"),
    }


def fetch_river_forecast(*, out_parquet: Path, out_meta: Path,
                         dry_run: bool = False, retry_max: int = 3,
                         backoff_s: float = 10.0, fetch_fn=None) -> dict:
    """CWC 7-day river level forecast -> rolling parquet cache (injectable)."""
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


# --- guarded local rebuild ---------------------------------------------------
CHAIN = ["fetch", "normalize", "align", "align6"]
CLIMATOLOGY_STAGE = "climatology"
BACKUP_SUFFIX = ".prev.refresh"


class FeatureBuildError(RuntimeError):
    pass


def _regenerate_gwl_selected() -> None:
    """The fetch stage only re-extracts GWL when the CSV is missing; drop it first."""
    if SELECTED_GWL.exists():
        SELECTED_GWL.unlink()


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    dst = Path(str(path) + BACKUP_SUFFIX)
    shutil.copy2(path, dst)
    return dst


def restore(path: Path, backup_path: Path | None) -> None:
    if backup_path is not None:
        shutil.copy2(backup_path, path)


def _run(cmd_label: str, timeout_s: int = 1800, *, stage: bool = False,
         module: str = "pipeline_data") -> None:
    cmd = ([sys.executable, "-u", "-m", module, cmd_label] if stage
           else [sys.executable, "-u", str(ROOT / cmd_label)])
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        tail += "\n" + "\n".join((proc.stderr or "").splitlines()[-20:])
        raise FeatureBuildError(f"{cmd_label} failed (exit {proc.returncode}):\n{tail}")
    time.sleep(1)


def validate_aligned(min_stations: int = 1) -> dict:
    """Row/station/sample-anchor sanity gate on the rebuilt aligned table."""
    if not ALIGNED_6H.exists():
        return {"ok": False, "error": "table_6h.parquet missing"}
    tbl = pd.read_parquet(ALIGNED_6H, columns=["Station", "time", "District", "gwl"])
    tbl["time"] = pd.to_datetime(tbl["time"], errors="coerce")
    stations = int(tbl["Station"].nunique())
    if stations < min_stations:
        return {"ok": False, "error": f"only {stations} stations (< {min_stations})"}
    latest = tbl["time"].max()
    gwl_rows = int(tbl["gwl"].notna().sum())
    per = tbl.dropna(subset=["gwl"]).groupby("Station")["time"].max()
    stale = int((per < latest - pd.Timedelta(days=7)).sum())
    return {"ok": True, "stations": stations, "rows": int(len(tbl)),
            "latest": latest, "gwl_rows": gwl_rows,
            "stations_stale_7d": stale}


def rebuild_aligned(*, dry_run: bool = False, min_stations: int = 20,
                    run_chain: bool = True, rebuild_climatology: bool = True) -> dict:
    """Run the align chain + climatology, guarded by backup/restore."""
    backups = {}
    for p in (ALIGNED_6H, ALIGNED_DAILY, CLIM_PATH, RAIN_CLIM_PATH, SELECTED_GWL):
        b = backup(p)
        if b is not None:
            backups[p] = b
    steps: list[str] = []
    try:
        if run_chain:
            if not dry_run:
                _regenerate_gwl_selected()
            steps.extend(CHAIN)
            if not dry_run:
                for step in CHAIN:
                    _run(step, stage=True)
        if rebuild_climatology:
            steps.append(CLIMATOLOGY_STAGE)
            if not dry_run:
                _run(CLIMATOLOGY_STAGE, stage=True)
        if dry_run:
            return {"ok": True, "dry_run": True, "steps": steps}
        verdict = validate_aligned(min_stations=min_stations)
        if not verdict.get("ok"):
            raise FeatureBuildError(verdict.get("error", "validation failed"))
        for p, b in backups.items():
            Path(str(p) + BACKUP_SUFFIX).unlink(missing_ok=True)
        return {"ok": True, "steps": steps, "validate": verdict}
    except Exception:
        for p, b in backups.items():
            restore(p, b)
        raise


def rebuild_traj_features(*, dry_run: bool = False) -> dict:
    """Rebuild fixed-split trajectory long-frames (data/features_traj)."""
    steps = ["features", "datasets"]
    for step in steps:
        if not dry_run:
            _run(step, stage=True,
                 module="pipeline_traj" if step == "datasets" else "pipeline_data")
    return {"ok": True, "steps": steps, "dry_run": dry_run}


class BackupGuard:
    """Context manager exposing backup/restore for the feature datasets."""
    def __init__(self):
        self.paths = (ALIGNED_6H, ALIGNED_DAILY, CLIM_PATH, RAIN_CLIM_PATH, WEATHER_PQ)

    def __enter__(self):
        self._backups = {p: backup(p) for p in self.paths}
        return self._backups

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            for p, b in self._backups.items():
                restore(p, b)
        return False


# --- future-step features ----------------------------------------------------
RFC_WINDOW_DAYS = 7
STEP_H = 6
FORECAST_STEPS = 64          # 16 d on the 6 h grid
DEFAULT_STEPS = 120          # 30 d horizon
H_COLS = ("h", "h_sin", "h_cos")
OM_COL_MAP = {
    "rain": "rain_mm", "temp": "temp_c", "humidity": "humidity_pct",
    "solar": "solar_wm2", "wind_speed": "wind_kmh", "pressure": "pressure_hpa",
}
WEATHER_COLS = ["temp", "humidity", "solar", "wind_speed", "pressure"]
RAIN_COLS = ["rain"]
RC_COLS = ["river_level", "canal_level"]
DRIVER_COLS = ["rain", "temp", "humidity", "solar", "wind_speed", "pressure",
               "river_level", "canal_level"]
ROLL_MEAN_PAIRS = (("temp_7d", "temp", 28), ("humidity_7d", "humidity", 28))
RAIN_ROLL_PAIRS = (("rain_1d", 4), ("rain_7d", 28), ("rain_30d", 120))
GWL_MEAN_ROLLS = (("gwl_roll7_mean", 28), ("gwl_roll30_mean", 120))
GWL_STD_ROLLS = (("gwl_roll7_std", 28), ("gwl_roll30_std", 120))
GWL_LAGS = (("lag1", 1), ("lag4", 4), ("lag8", 8), ("lag28", 28), ("lag120", 120))
TAIL_BACK = 128


def _horizon_feats(h: int, max_h: int = DEFAULT_STEPS) -> dict:
    return {"h": float(h), "h_sin": float(np.sin(2 * np.pi * h / max_h)),
            "h_cos": float(np.cos(2 * np.pi * h / max_h))}


def _doy_hour_features(t: pd.Timestamp) -> dict:
    mo, doy = t.month, t.dayofyear
    hr = t.hour + t.minute / 60.0
    return {
        "year": float(t.year),
        "month_sin": float(np.sin(2 * np.pi * mo / 12)),
        "month_cos": float(np.cos(2 * np.pi * mo / 12)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "hour_sin": float(np.sin(2 * np.pi * hr / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hr / 24)),
        "monsoon": float(doy - 152),
    }


def _sixh_tail(hist: pd.DataFrame, col: str, anchor_t: pd.Timestamp,
               n_back: int = TAIL_BACK) -> np.ndarray:
    """Contiguous 6h driver tail ending at ``anchor_t`` (ffill; rain->0)."""
    times = pd.date_range(anchor_t - pd.Timedelta(hours=n_back * STEP_H),
                          anchor_t, freq="6h")
    if col not in hist.columns:
        return np.full(len(times), np.nan)
    sub = hist[["time", col]].sort_values("time").drop_duplicates("time")
    merged = pd.DataFrame({"time": times}).merge(sub, on="time", how="left")
    merged[col] = merged[col].ffill().fillna(0.0) if col == "rain" else merged[col].ffill()
    return merged[col].to_numpy(dtype=np.float64)


class ClimatologyIndex:
    """Keyed lookups for station/doy weather and district/doy/hour rain."""
    def __init__(self, weather: pd.DataFrame | None, rain: pd.DataFrame | None):
        self._w, self._r = weather, rain
        self._wo, self._ro = None, None
        if weather is not None and not weather.empty:
            w = weather.copy().drop_duplicates(subset=["Station", "doy"])
            idx = pd.MultiIndex.from_tuples(
                [(str(s), int(d)) for s, d in zip(w["Station"], w["doy"])],
                names=("Station", "doy"))
            self._wo = w.set_index(idx)
        if rain is not None and not rain.empty:
            r = rain.copy()
            r["District"] = r["District"].astype(str).str.upper()
            r = r.groupby(["District", "doy", "hour"], as_index=False)["rain"].mean()
            idx = pd.MultiIndex.from_tuples(
                [(d, int(o), int(h)) for d, o, h in zip(r["District"], r["doy"], r["hour"])],
                names=("District", "doy", "hour"))
            self._ro = r.set_index(idx)

    def weather(self, station: str, doy: int) -> pd.Series | None:
        if self._wo is None:
            return None
        try:
            i = self._wo.index.get_loc((station, int(doy)))
            return self._wo.iloc[i]
        except KeyError:
            return None

    def weather_present(self) -> bool:
        return self._wo is not None

    def rain(self, district: str, doy: int, hour: int) -> float | None:
        if self._ro is None:
            return None
        try:
            return float(self._ro.loc[(district.upper(), int(doy), int(hour)), "rain"])
        except KeyError:
            return None


class RiverForecastIndex:
    """District-level CWC day-1..7 river forecast as delta vs observed gauge level."""
    def __init__(self, rfc: pd.DataFrame | None):
        self._dates: dict[str, dict[object, float]] = {}
        self._bounds: dict[str, tuple[object, object]] = {}
        self._keys = {}
        if rfc is None or rfc.empty:
            return
        if "d1_wl" in rfc.columns:
            g = rfc.dropna(subset=["observed_wl", "d1_wl"], how="all")
        else:
            g = rfc.dropna(subset=["observed_wl"])
        if g.empty:
            return
        for district, sub in g.groupby("district"):
            by_date: dict[object, list[float]] = {}
            for _, row in sub.iterrows():
                obs_dt, obs_wl = row.get("observed_dt"), row.get("observed_wl")
                if pd.isna(obs_dt) or not np.isfinite(float(obs_wl)):
                    obs_dt, obs_wl = row.get("d1_dt"), row.get("d1_wl")
                    if pd.isna(obs_dt) or not np.isfinite(float(obs_wl)):
                        continue
                base = float(obs_wl)
                pairs = [(pd.Timestamp(obs_dt).normalize(), base)]
                for day in range(1, RFC_WINDOW_DAYS + 1):
                    dt, wl = row.get(f"d{day}_dt"), row.get(f"d{day}_wl")
                    if pd.isna(dt) or pd.isna(wl):
                        continue
                    pairs.append((pd.Timestamp(dt).normalize(), float(wl)))
                for date, wl in pairs:
                    by_date.setdefault(date, []).append(wl - base)
            self._dates[district] = {d: float(np.mean(v)) for d, v in by_date.items()}
            if by_date:
                self._bounds[district] = (min(by_date), max(by_date))
                self._keys[district] = sorted(by_date)

    def covers(self, district: str, t) -> bool:
        try:
            d = pd.Timestamp(t).normalize()
        except (TypeError, ValueError):
            return False
        if district not in self._bounds:
            return False
        lo, hi = self._bounds[district]
        return lo <= d <= hi

    def delta(self, district: str, t) -> float | None:
        try:
            d = pd.Timestamp(t).normalize()
        except (TypeError, ValueError):
            return None
        keys = self._keys.get(district)
        if not keys:
            return None
        i = bisect.bisect_right(keys, d) - 1
        if i < 0:
            return None
        v = self._dates[district][keys[i]]
        return float(v) if np.isfinite(v) else None


def _load_river_forecast() -> pd.DataFrame | None:
    """Latest CWC snapshot from the rolling cache (None when absent/empty)."""
    if not RFC_PATH.exists():
        return None
    df = pd.read_parquet(RFC_PATH)
    if df.empty or "fetched_at" not in df.columns:
        return None
    last = df["fetched_at"].max()
    out = df[df["fetched_at"] == last].reset_index(drop=True)
    return out if not out.empty else None


def _last_obs(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else None


def _series_get(cw, key: str) -> float:
    if cw is None:
        return float("nan")
    try:
        v = cw.get(key)
    except (KeyError, AttributeError):
        return float("nan")
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def _gwl_hydro_pinned(hist: pd.DataFrame, anchor_gwl: float, memory: dict) -> dict:
    """Anchor-pinned GWL state/trend (constant over every horizon step).

    ``hist`` is restricted to rows <= anchor_t. Mirrors pipeline_model.build.
    """
    gwl = hist["gwl"]
    obs_flags = hist["gwl_obs"] if "gwl_obs" in hist.columns else None
    station = (hist["Station"].astype(str) if "Station" in hist.columns
               else pd.Series("_single", index=hist.index))
    block: dict = {}
    block["gwl_imputed"] = float(gwl_imputed_flag(gwl, obs_flags).iloc[-1])
    for w in DEV_WINS:
        block[f"gwl_dev{w}"] = anchor_gwl - float(memory[f"gwl_roll{w}_mean"])
    trend = gwl_trend_block(hist["time"], gwl, gwl_observed(gwl, obs_flags), station)
    last = trend.iloc[-1]
    for w in SLOPE_WINS:
        block[f"gwl_slope{w}"] = float(last[f"gwl_slope{w}"])
        block[f"gwl_obs{w}"] = float(last[f"gwl_obs{w}"])
    return block


def _rain_memory_steps(hist: pd.DataFrame, fw: pd.DataFrame,
                       anchor_t: pd.Timestamp, steps: int) -> tuple[dict, int, int]:
    """Per-step rain-memory over full observed history + forecast forcing.

    Returns (columns, n_hist, n_hist_observed). Only finite HISTORY rain counts
    as observed; forecast/climatology drives rain_api* but never rain_obs*.
    """
    names = ([f"rain_api{int(t)}" for t in API_TAUS]
             + [f"rain_obs{int(t)}" for t in API_TAUS])
    if "rain" not in hist.columns:
        return {c: np.full(steps, np.nan) for c in names}, 0, 0
    hist_rain = hist["rain"].to_numpy(dtype=np.float64)
    n_hist = int(hist_rain.size)
    fwd_rain = fw["rain"].to_numpy(dtype=np.float64)
    series = np.concatenate([hist_rain, fwd_rain])
    mask = np.concatenate([np.isfinite(hist_rain), np.zeros(fwd_rain.size, dtype=bool)])
    n_hist_obs = int(np.isfinite(hist_rain).sum())
    blk = rain_memory_block(series, mask)
    # Step k ends its trailing window at anchor + k*6h (same convention as the
    # rain_1d/7d/30d windows below), so the step's own forecast slot is excluded.
    start = n_hist - 1
    cols = {c: np.asarray(blk[c][start: start + steps], dtype=np.float64)
            for c in names}
    return cols, n_hist, n_hist_obs


def _rc_values(cw, rfi: RiverForecastIndex | None, district: str, t,
               river_last: float | None, canal_last: float | None
               ) -> tuple[float, float, str, str]:
    """River: CWC delta -> climatology -> persistence -> NaN. Canal: clim -> persistence -> NaN."""
    clim_r, clim_c = _series_get(cw, "river_level"), _series_get(cw, "canal_level")
    river, river_src = float("nan"), "river_canal:unavailable"
    if rfi is not None and rfi.covers(district, t):
        delta = rfi.delta(district, t)
        baseline = clim_r if np.isfinite(clim_r) else (
            river_last if river_last is not None else np.nan)
        if delta is not None and np.isfinite(baseline):
            river, river_src = float(baseline + delta), "river_canal:river_cwc_forecast"
    if not np.isfinite(river) and np.isfinite(clim_r):
        river, river_src = float(clim_r), "river_canal:river_climatology"
    if not np.isfinite(river) and river_last is not None and np.isfinite(river_last):
        river, river_src = float(river_last), "river_canal:river_persistence"
    canal, canal_src = float("nan"), "river_canal:unavailable"
    if np.isfinite(clim_c):
        canal, canal_src = float(clim_c), "river_canal:canal_climatology"
    elif canal_last is not None and np.isfinite(canal_last):
        canal, canal_src = float(canal_last), "river_canal:canal_persistence"
    return river, canal, river_src, canal_src


def _om_lookup(om_frame: pd.DataFrame, district: str) -> dict:
    if om_frame is None or om_frame.empty:
        return {}
    g = om_frame.copy()
    dists = g["District"].astype(str).str.upper().to_list()
    dates = pd.to_datetime(g["date"]).dt.normalize().to_list()
    return {tup: i for i, tup in enumerate(zip(dists, dates))}


def future_weather_matrix(*, station: str, district: str, anchor_t: pd.Timestamp,
                          steps: int = DEFAULT_STEPS, om_frame: pd.DataFrame | None = None,
                          climatology: pd.DataFrame | None = None,
                          rain_clim: pd.DataFrame | None = None,
                          river_forecast: pd.DataFrame | None = None,
                          river_last: float | None = None, canal_last: float | None = None,
                          forecast_days: int = int(FORECAST_STEPS / 4)
                          ) -> tuple[pd.DataFrame, list[dict]]:
    """Per-step driver values + explicit per-step source resolution."""
    hs = np.arange(1, steps + 1, dtype=np.int64)
    ts = anchor_t + pd.to_timedelta(hs * STEP_H, unit="h")
    om_idx = _om_lookup(om_frame, district)
    clim = ClimatologyIndex(climatology, rain_clim)
    rfi = RiverForecastIndex(river_forecast) if river_forecast is not None else None
    dkey = str(district).upper()
    driver_cols = {c: np.full(steps, np.nan) for c in DRIVER_COLS}
    srcs = []
    for k, t in enumerate(ts):
        rec = {"weather": "unavailable", "rain": "unavailable", "river_canal": "unavailable"}
        cw = None
        # One OM daily row covers the day's 4 six-hour slots; bucket 00:00 under
        # the prior date (06/12/18 and the overnight 00:00 of the next day).
        dt = (t - pd.Timedelta(hours=STEP_H)).normalize()
        om_row = om_idx.get((dkey, dt))
        if om_row is not None:
            row = om_frame.iloc[om_row]
            for out_col, om_col in OM_COL_MAP.items():
                driver_cols[out_col][k] = float(row[om_col])
            driver_cols["rain"][k] = float(row["rain_mm"]) / 4.0
            rec["weather"], rec["rain"] = "open-meteo", "open-meteo"
        else:
            cw = clim.weather(station, int(t.dayofyear)) if clim.weather_present() else None
            if cw is not None:
                for c in WEATHER_COLS:
                    driver_cols[c][k] = float(cw[c])
                rec["weather"] = "climatology"
            rv = clim.rain(dkey, int(t.dayofyear), int(t.hour))
            if rv is not None and np.isfinite(rv):
                driver_cols["rain"][k] = float(rv)
                rec["rain"] = "climatology"
        rv, cv, river_src, canal_src = _rc_values(cw, rfi, dkey, t, river_last, canal_last)
        driver_cols["river_level"][k] = rv
        driver_cols["canal_level"][k] = cv
        rec["river_canal"] = f"{river_src}|{canal_src}"
        srcs.append(rec)
    return pd.DataFrame(driver_cols), srcs


def build_step_features(*, station: str, district: str, anchor_t: pd.Timestamp,
                        hist: pd.DataFrame, fnames: list[str],
                        om_frame: pd.DataFrame | None = None,
                        climatology: pd.DataFrame | None = None,
                        rain_clim: pd.DataFrame | None = None,
                        river_forecast: pd.DataFrame | None = None,
                        forecast_days: int = int(FORECAST_STEPS / 4),
                        steps: int = DEFAULT_STEPS, mode: str = "future",
                        anchor_gwl: float | None = None,
                        max_h: int = DEFAULT_STEPS) -> tuple[np.ndarray, list[dict]]:
    """Build (steps x len(fnames)) float32 future-mode matrix + source records."""
    if mode != "future":
        raise ValueError(f"build_step_features only supports mode='future', got {mode!r}")
    base_cols = [c for c in fnames if c not in H_COLS]
    h_cols = [c for c in fnames if c in H_COLS]
    _lc = hist.sort_values("time")
    river_last = _last_obs(_lc, "river_level")
    canal_last = _last_obs(_lc, "canal_level")
    fw, srcs = future_weather_matrix(
        station=station, district=district, anchor_t=anchor_t, steps=steps,
        om_frame=om_frame, climatology=climatology, rain_clim=rain_clim,
        river_forecast=river_forecast, river_last=river_last, canal_last=canal_last,
        forecast_days=forecast_days)
    hs = np.arange(1, steps + 1, dtype=np.int64)
    ts = anchor_t + pd.to_timedelta(hs * STEP_H, unit="h")
    tail_rain = _sixh_tail(hist, "rain", anchor_t)
    tail_temp = _sixh_tail(hist, "temp", anchor_t)
    tail_hum = _sixh_tail(hist, "humidity", anchor_t)
    i0 = len(tail_rain) - 1
    rain_ser = np.concatenate([tail_rain[: i0 + 1], fw["rain"].to_numpy()])
    temp_ser = np.concatenate([tail_temp[: i0 + 1], fw["temp"].to_numpy()])
    hum_ser = np.concatenate([tail_hum[: i0 + 1], fw["humidity"].to_numpy()])
    g = hist["gwl"].to_numpy(dtype=np.float64)
    anchor_gwl = float(g[-1]) if anchor_gwl is None else float(anchor_gwl)
    memory: dict = {"gwl": anchor_gwl}
    for c, lag in GWL_LAGS:
        memory[c] = float(g[-1 - lag]) if len(g) > lag else np.nan
    for c, win in GWL_MEAN_ROLLS:
        memory[c] = float(pd.Series(g).rolling(win, min_periods=1).mean().iloc[-1])
    for c, win in GWL_STD_ROLLS:
        memory[c] = float(pd.Series(g).rolling(win, min_periods=1).std().iloc[-1])
    hydro_pin = _gwl_hydro_pinned(_lc, anchor_gwl, memory)
    rain_cols, _, _ = _rain_memory_steps(_lc, fw, anchor_t, steps)
    rows = []
    for k in range(steps):
        t = ts[k]
        p = i0 + (k + 1)
        row = dict(memory)
        row.update(hydro_pin)
        for c in ("temp", "humidity", "solar", "wind_speed", "pressure"):
            row[c] = float(fw[c].iloc[k])
        row["river_level"] = fw["river_level"].iloc[k]
        row["canal_level"] = fw["canal_level"].iloc[k]
        for out, win in RAIN_ROLL_PAIRS:
            lo = max(0, p - win + 1)
            row[out] = float(np.nansum(rain_ser[lo:p + 1])) if p >= lo else np.nan
        for out, src, win in ROLL_MEAN_PAIRS:
            ser = temp_ser if src == "temp" else hum_ser
            lo = max(0, p - win + 1)
            w = ser[lo:p + 1]
            row[out] = float(np.nanmean(w)) if np.isfinite(w).sum() > 0 else np.nan
        for c in rain_cols:
            row[c] = float(rain_cols[c][k])
        row.update(_doy_hour_features(t))
        row.update(_horizon_feats(hs[k], max_h=max_h))
        rows.append(row)
    df = pd.DataFrame(rows)
    X = df[list(fnames)].to_numpy(dtype=np.float32)
    return X, srcs


# Backwards-compatible module aliases (old refresh.sources/features/... share this module).
sources = features = future_drivers = sys.modules[__name__]
