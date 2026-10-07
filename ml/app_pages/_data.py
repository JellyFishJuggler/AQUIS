"""_data — artifact loaders + safe capability probes for the ML showcase app.

READ-ONLY BY CONSTRUCTION. Nothing in this module trains, retrains, fetches,
writes, or generates a forecast. It only:

  * reads artifacts that already exist on disk (``models/``, ``outputs/``,
    ``data/features*``, ``data/aligned/``, ``data/refresh/``), and
  * optionally *calls* the existing inference engine
    (``_trajectory.trajectory_forecast``) when the caller explicitly asks for it
    via :func:`trajectory_forecast_safe`, which is a pure read-and-score path.

Every loader returns ``None`` (or an empty frame) when its artifact is missing,
and :func:`artifact_report` gives the page-level code a single place to render a
"data/artifact unavailable" state instead of raising.

Kept free of Streamlit imports so it can be imported from a plain Python shell
or a test; the pages wrap the expensive ones in ``st.cache_*``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parent.parent          # .../ml
REPO_ROOT = ML_ROOT.parent                                # .../AQUIS
MODELS = ML_ROOT / "models"
OUT = ML_ROOT / "outputs"
DATA = ML_ROOT / "data"
ALIGNED = DATA / "aligned"
FEATURES = DATA / "features"
FEATURES_6H = DATA / "features_6h"
FEATURES_TRAJ = DATA / "features_traj"
META = DATA / "meta"
REFRESH = DATA / "refresh"
FORECASTS = REFRESH / "forecasts"
FORECAST_ARCHIVE = REFRESH / "forecast_archive"

# Make the flat module names (06_features, _hydrology, ...) importable no matter
# where streamlit is launched from.
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


# ---------------------------------------------------------------------------
# artifact registry
# ---------------------------------------------------------------------------
# (label, path, kind) — the single source of truth for what this app can show.
ARTIFACTS: list[tuple[str, Path, str]] = [
    # configs / model cards
    ("feature_config.json (pooled model schema)", MODELS / "feature_config.json", "json"),
    ("model_metadata.json (pooled, train-time facts)", MODELS / "model_metadata.json", "json"),
    ("traj_config.json (trajectory v2 schema)", MODELS / "traj_config.json", "json"),
    ("traj_calibration.json (per-horizon widening)", MODELS / "traj_calibration.json", "json"),
    ("traj_reliability.json (confidence tables)", MODELS / "traj_reliability.json", "json"),
    ("quantile_calibration.json (pooled interval)", MODELS / "quantile_calibration.json", "json"),
    # weights (existence only — the app does not load these unless it scores)
    ("xgb_multihorizon.joblib", MODELS / "xgb_multihorizon.joblib", "bin"),
    ("linear_multihorizon.joblib", MODELS / "linear_multihorizon.joblib", "bin"),
    ("traj_xgb_q50.json", MODELS / "traj_xgb_q50.json", "bin"),
    # pooled evaluation outputs
    ("model_metrics.csv (4-way, 30 d)", OUT / "model_metrics.csv", "table"),
    ("predictions_2026.parquet (paired actuals and forecasts)",
     OUT / "predictions_2026.parquet", "bin"),
    ("verification.parquet (published forecasts matched to realised GWL)",
     REFRESH / "verification.parquet", "bin"),
    ("honest_metrics.csv (overlap-aware)", OUT / "honest_metrics.csv", "table"),
    ("station_summary.csv (per-station)", OUT / "station_summary.csv", "table"),
    ("eval_by_station.csv", OUT / "eval_by_station.csv", "table"),
    ("eval_by_district.csv", OUT / "eval_by_district.csv", "table"),
    ("ablation.csv (drop-group)", OUT / "ablation.csv", "table"),
    ("feature_importance.csv", OUT / "feature_importance.csv", "table"),
    ("feature_coefficients.csv", OUT / "feature_coefficients.csv", "table"),
    ("overlap.json", OUT / "overlap.json", "json"),
    ("residual_acf.json", OUT / "residual_acf.json", "json"),
    ("residual_spatial.json", OUT / "residual_spatial.json", "json"),
    ("spatial_cv_metrics.csv", OUT / "spatial_cv_metrics.csv", "table"),
    ("diagnostics.json", OUT / "diagnostics.json", "json"),
    ("diagnostics_permutation.csv", OUT / "diagnostics_permutation.csv", "table"),
    ("correlation_report.csv", OUT / "correlation_report.csv", "table"),
    ("lag_curves.csv (recharge lag)", OUT / "lag_curves.csv", "table"),
    ("fleet_forecast.csv", OUT / "fleet_forecast.csv", "table"),
    # trajectory v2 evaluation outputs
    ("traj_backtest_metrics.csv (per-horizon)", OUT / "traj_backtest_metrics.csv", "table"),
    ("traj_backtest_summary.json", OUT / "traj_backtest_summary.json", "json"),
    # external baseline
    ("arima_vs_xgb.csv", OUT / "arima_vs_xgb.csv", "table"),
    # feature tables (existence / schema only — not loaded in full)
    ("data/features/{train,val,test}.parquet", FEATURES / "train.parquet", "bin"),
    ("data/features/prep.json", FEATURES / "prep.json", "json"),
    ("data/features_traj/train.parquet (long-frame)", FEATURES_TRAJ / "train.parquet", "bin"),
    ("data/features_traj/prep_traj.json", FEATURES_TRAJ / "prep_traj.json", "json"),
    ("data/aligned/table_6h.parquet (6 h substrate)", ALIGNED / "table_6h.parquet", "bin"),
    ("data/meta/manifest.json (source coverage)", META / "manifest.json", "json"),
    ("data/meta/driver_climatology_meta.json", META / "driver_climatology_meta.json", "json"),
    # serving state
    ("data/refresh/forecasts/ (per-station)", FORECASTS, "dir"),
    ("data/refresh/forecast_archive/ (archived gens)", FORECAST_ARCHIVE, "dir"),
    ("data/refresh/runtime.json", REFRESH / "runtime.json", "json"),
    ("data/refresh/meta.json (publish manifest)", REFRESH / "meta.json", "json"),
]


def artifact_report() -> pd.DataFrame:
    """One row per known artifact: present/absent + size. Drives the UI banner."""
    rows = []
    for label, path, kind in ARTIFACTS:
        if kind == "dir":
            exists = path.is_dir()
            size = sum(f.stat().st_size for f in path.glob("*")) if exists else 0
            n = len(list(path.glob("*"))) if exists else 0
        else:
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            n = 1 if exists else 0
        rows.append({"artifact": label, "path": str(path.relative_to(REPO_ROOT)),
                     "present": exists, "files": n, "bytes": size})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# generic readers
# ---------------------------------------------------------------------------
def read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — a missing/corrupt artifact is a UI state
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------
def feature_config() -> dict | None:
    return read_json(MODELS / "feature_config.json")


def model_metadata() -> dict | None:
    return read_json(MODELS / "model_metadata.json")


def traj_config() -> dict | None:
    return read_json(MODELS / "traj_config.json")


def traj_calibration() -> dict | None:
    return read_json(MODELS / "traj_calibration.json")


def traj_reliability() -> dict | None:
    return read_json(MODELS / "traj_reliability.json")


def quantile_calibration() -> dict | None:
    return read_json(MODELS / "quantile_calibration.json")


def prep_json() -> dict | None:
    return read_json(FEATURES / "prep.json")


def prep_traj_json() -> dict | None:
    return read_json(FEATURES_TRAJ / "prep_traj.json")


# ---------------------------------------------------------------------------
# metrics tables
# ---------------------------------------------------------------------------
def model_metrics() -> pd.DataFrame | None:
    return read_csv(OUT / "model_metrics.csv")


def predictions_2026() -> pd.DataFrame | None:
    """Paired 2026 target readings and frozen-model forecasts, if saved."""
    path = OUT / "predictions_2026.parquet"
    columns = ["Station", "District", "time", "horizon", "target", "xgb",
               "ridge", "persist", "clim"]
    base = read_parquet_safe(path, columns=columns)
    if base is None:
        return None
    # Prediction bands are optional in older output artifacts.
    bands = read_parquet_safe(path, columns=["q05_lvl", "q95_lvl"])
    if bands is not None and len(bands) == len(base):
        base = pd.concat([base.reset_index(drop=True),
                          bands.reset_index(drop=True)], axis=1)
    return base


def verification_ledger() -> pd.DataFrame | None:
    """Published trajectory forecasts joined to later realised observations."""
    columns = ["station", "anchor_time", "horizon_h", "target_time", "q05",
               "q50", "q95", "actual", "model_version", "generated_ts"]
    return read_parquet_safe(REFRESH / "verification.parquet", columns=columns)


def honest_metrics() -> pd.DataFrame | None:
    return read_csv(OUT / "honest_metrics.csv")


def station_summary() -> pd.DataFrame | None:
    return read_csv(OUT / "station_summary.csv")


def eval_by_station() -> pd.DataFrame | None:
    return read_csv(OUT / "eval_by_station.csv")


def eval_by_district() -> pd.DataFrame | None:
    return read_csv(OUT / "eval_by_district.csv")


def ablation() -> pd.DataFrame | None:
    return read_csv(OUT / "ablation.csv")


def feature_importance() -> pd.DataFrame | None:
    return read_csv(OUT / "feature_importance.csv")


def traj_backtest_metrics() -> pd.DataFrame | None:
    return read_csv(OUT / "traj_backtest_metrics.csv")


def traj_backtest_summary() -> dict | None:
    return read_json(OUT / "traj_backtest_summary.json")


def spatial_cv_metrics() -> pd.DataFrame | None:
    return read_csv(OUT / "spatial_cv_metrics.csv")


def diagnostics() -> dict | None:
    return read_json(OUT / "diagnostics.json")


def arima_vs_xgb() -> pd.DataFrame | None:
    return read_csv(OUT / "arima_vs_xgb.csv")


def lag_curves() -> pd.DataFrame | None:
    return read_csv(OUT / "lag_curves.csv")


def correlation_report() -> pd.DataFrame | None:
    return read_csv(OUT / "correlation_report.csv")


def fleet_forecast() -> pd.DataFrame | None:
    return read_csv(OUT / "fleet_forecast.csv")


def manifest() -> list | None:
    return read_json(META / "manifest.json")


# ---------------------------------------------------------------------------
# hydrology feature blocks (schema-level; no model training involved)
# ---------------------------------------------------------------------------
# Imported lazily so a missing scipy (or any import error) surfaces as a UI
# state rather than an app crash.
def hydro_columns() -> tuple[list[str], list[str]]:
    """(NUM_COLS, HYDRO_COLS) — what the frozen models use vs what is carried."""
    try:
        import importlib
        F = importlib.import_module("06_features")
        return list(F.NUM_COLS), list(F.HYDRO_COLS)
    except Exception:  # noqa: BLE001
        return [], []


def feature_group_table() -> pd.DataFrame:
    """Grouped feature inventory, flagging trained vs carried-not-trained.

    Groups mirror the modelling intent (not the code order): groundwater
    state/trend, rainfall/weather, seasonal, spatial/static, candidate hydrology.
    """
    num, hydro = hydro_columns()
    groups = {
        "groundwater state / trend": [
            "gwl", "lag1", "lag4", "lag8", "lag28", "lag120",
            "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
        ],
        "rainfall / weather": [
            "rain_1d", "rain_7d", "rain_30d", "temp", "temp_7d", "humidity",
            "humidity_7d", "solar", "wind_speed", "pressure",
            "river_level", "canal_level",
        ],
        "seasonal / calendar": [
            "year", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "hour_sin", "hour_cos", "monsoon",
        ],
        "spatial / static": ["st_id", "dist_id"],
        "candidate hydrology (carried, untrained)": list(hydro),
    }
    rows = []
    for group, cols in groups.items():
        for c in cols:
            rows.append({
                "group": group,
                "feature": c,
                # HYDRO_COLS are NOT in the frozen num_cols; ordinals are added
                # by the trainer rather than prep.json, so mark them explicitly.
                "in_frozen_num_cols": c in num,
                "status": ("trained" if c in num
                           else "carried, untrained" if c in hydro
                           else "model input (not in prep.num_cols)"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# serving state
# ---------------------------------------------------------------------------
def runtime_json() -> dict | None:
    return read_json(REFRESH / "runtime.json")


def refresh_meta() -> dict | None:
    return read_json(REFRESH / "meta.json")


def forecasts_parquet() -> pd.DataFrame | None:
    return read_parquet_safe(REFRESH / "forecasts.parquet")


def read_parquet_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:  # noqa: BLE001
        return None


def station_names() -> list[str]:
    """Canonical station list, most-authoritative source first."""
    cfg = feature_config()
    if cfg and cfg.get("stations"):
        return sorted(str(s) for s in cfg["stations"])
    p = ALIGNED / "table_6h.parquet"
    df = read_parquet_safe(p, columns=["Station"])
    if df is not None and "Station" in df.columns:
        return sorted(df["Station"].astype(str).unique())
    return []


def station_history(station: str, tail_days: int = 400) -> pd.DataFrame | None:
    """Observed GWL history for one station (thin slice of the 6 h table)."""
    sub = station_frame(station)
    if sub is None:
        return None
    if tail_days:
        sub = sub[sub["time"] >= sub["time"].max() - pd.Timedelta(days=tail_days)]
    keep = [c for c in ("Station", "time", "gwl") if c in sub.columns]
    return sub[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# hydrologist-facing reads
# ---------------------------------------------------------------------------
# Columns a hydrograph / water-balance view wants, kept explicit so a single-well
# view never materialises the full 3 M-row table.
HYDRO_COLUMNS = ["Station", "District", "time", "gwl", "gwl_obs", "rain",
                 "temp", "humidity", "river_level", "canal_level"]


def _table_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:  # noqa: BLE001
        return []


def station_frame(station: str) -> pd.DataFrame | None:
    """One station's full 6 h record, with row-group pushdown when possible.

    Uses a pyarrow filter so only this station's rows are read. Falls back to a
    full read plus a boolean mask when pyarrow is unavailable (slower, same
    result). Returns None when the table or the station is missing.
    """
    path = ALIGNED / "table_6h.parquet"
    if not path.exists():
        return None
    available = _table_columns(path)
    cols = [c for c in HYDRO_COLUMNS if not available or c in available]
    df = None
    if available:
        try:
            import pyarrow.parquet as pq
            df = pq.read_table(path, columns=cols,
                               filters=[("Station", "==", str(station))]).to_pandas()
        except Exception:  # noqa: BLE001 — fall through to the full read
            df = None
    if df is None:
        df = read_parquet_safe(path, columns=cols)
        if df is None:
            return None
        df = df[df["Station"].astype(str) == str(station)].copy()
    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def station_coords() -> pd.DataFrame:
    """Station → District + Latitude/Longitude, from the selection metadata.

    `data/meta/selected_gwl_stations.csv` is written by `01_select` and is the
    authoritative station list *with coordinates*; the aligned table has no
    coordinates. Rows with no coordinate are dropped (they cannot be mapped) but
    the caller can compare counts to see how many were lost.
    """
    df = read_csv(META / "selected_gwl_stations.csv")
    if df is None or df.empty:
        return pd.DataFrame(columns=["Station", "District", "latitude", "longitude"])
    rename = {}
    for c in df.columns:
        if c.lower() == "latitude":
            rename[c] = "latitude"
        elif c.lower() == "longitude":
            rename[c] = "longitude"
    df = df.rename(columns=rename)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame(columns=["Station", "District", "latitude", "longitude"])
    df["Station"] = df["Station"].astype(str)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    keep = [c for c in ("Station", "District", "latitude", "longitude")
            if c in df.columns]
    df = df[keep].dropna(subset=["latitude", "longitude"])
    return df.reset_index(drop=True)


def station_universe() -> pd.DataFrame:
    """Station -> District map. Two columns, one pass, ~600 rows out."""
    df = read_parquet_safe(ALIGNED / "table_6h.parquet", columns=["Station", "District"])
    if df is None:
        return pd.DataFrame(columns=["Station", "District"])
    if "District" not in df.columns:
        df["District"] = ""
    return (df.dropna(subset=["Station"]).astype({"Station": str})
              .drop_duplicates("Station").sort_values("Station")
              .reset_index(drop=True))


def recent_window(days: int = 120) -> pd.DataFrame | None:
    """All stations, recent window only, thin columns — for the fleet screen.

    Row-group pushdown on `time` keeps this to a fraction of the full table
    instead of ~3 M rows.
    """
    path = ALIGNED / "table_6h.parquet"
    if not path.exists():
        return None
    available = _table_columns(path)
    wanted = ["Station", "District", "time", "gwl", "gwl_obs", "rain"]
    cols = [c for c in wanted if not available or c in available]
    df = None
    last = None
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        md = pf.metadata
        if md is not None and md.num_row_groups:
            # cheap upper bound on time from the footer statistics
            for rg in range(md.num_row_groups - 1, -1, -1):
                st = md.row_group(rg).column(
                    md.schema.names.index("time")).statistics
                if st is not None and st.has_min_max:
                    last = pd.Timestamp(st.max)
                    break
    except Exception:  # noqa: BLE001
        last = None
    if last is not None:
        try:
            import pyarrow.parquet as pq
            cutoff = last - pd.Timedelta(days=days)
            df = pq.read_table(path, columns=cols,
                               filters=[("time", ">=", cutoff)]).to_pandas()
        except Exception:  # noqa: BLE001
            df = None
    if df is None:
        df = read_parquet_safe(path, columns=cols)
        if df is None:
            return None
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df[df["time"] >= df["time"].max() - pd.Timedelta(days=days)]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).reset_index(drop=True)


def archived_forecast(station: str) -> dict | None:
    """Newest archived (previous-generation) forecast for a station.

    Uses the filename index (`archive_station_index`) so this is a handful of
    reads rather than ~7,000; the `station` field is still verified before return.
    """
    best: tuple[str, dict] | None = None
    for _, payload in archive_station_files(station):
        anchor = str(payload.get("anchor_time", ""))
        if best is None or anchor > best[0]:
            best = (anchor, payload)
    return best[1] if best else None


@lru_cache(maxsize=4)
def archive_station_index() -> dict[str, list[str]]:
    """station_key -> sorted [archive filenames].

    Built from the **filename** only: `publish.py` writes
    ``<station_key>-<anchor>.json`` and `station_key` is a stable hash of the
    station name, so grouping by the key groups by station. That avoids reading
    ~7,000 JSON files just to enumerate which stations the archive covers.
    """
    idx: dict[str, list[str]] = {}
    if not FORECAST_ARCHIVE.is_dir():
        return idx
    for f in FORECAST_ARCHIVE.glob("*.json"):
        key = f.stem.split("-", 1)[0]
        idx.setdefault(key, []).append(f.name)
    for k in idx:
        idx[k].sort()
    return idx


def station_key(station: str) -> str:
    """Must match `refresh.publish.station_key` exactly, or the archive index
    will not find a station's files."""
    return hashlib.sha1(str(station).encode("utf-8")).hexdigest()[:16]


def archive_station_files(station: str):
    """Yield (name, payload) for every archived generation of `station`."""
    key = station_key(station)
    for name in archive_station_index().get(key, []):
        payload = read_json(FORECAST_ARCHIVE / name)
        if isinstance(payload, dict) and payload.get("station") == station:
            yield name, payload


def archive_validation(station: str, horizon_h: int = 720,
                       tail_days: int = 400) -> pd.DataFrame:
    """Issued-vs-realised pairs at one horizon, from the forecast archive.

    The archive is a genuine point-in-time record: each file is a forecast as
    published at that anchor. Matching its target timestamp against the reading
    actually observed then is therefore honest backtesting, and — unlike the
    verification ledger — it is not limited to the horizons that happened to have
    elapsed when the forecast was first scored.

    Returns columns: anchor_time, target_time, q05, q50, q95, actual, error.
    Empty frame when the archive, the station, or matching observations are absent.
    """
    path = ALIGNED / "table_6h.parquet"
    if not FORECAST_ARCHIVE.is_dir() or not path.exists():
        return pd.DataFrame(columns=["anchor_time", "target_time", "q05", "q50",
                                     "q95", "actual", "error"])
    obs = station_frame(station)
    if obs is None or obs.empty:
        return pd.DataFrame(columns=["anchor_time", "target_time", "q05", "q50",
                                     "q95", "actual", "error"])
    times = pd.to_datetime(obs["time"], errors="coerce", utc=True).dt.tz_localize(None)
    vals = pd.to_numeric(obs["gwl"], errors="coerce")
    if tail_days:
        cutoff = times.max() - pd.Timedelta(days=tail_days)
    else:
        cutoff = times.min()

    k = max(1, int(horizon_h // 6))
    rows = []
    for _, pod in archive_station_files(station):
        try:
            anchor = pd.Timestamp(pod.get("anchor_time"))
            if anchor.tzinfo is not None:
                anchor = anchor.tz_localize(None)
            steps = pod.get("trajectory") or []
            if len(steps) < k:
                continue
            step = steps[k - 1]
            target = pd.Timestamp(step["time"])
            if target.tzinfo is not None:
                target = target.tz_localize(None)
        except Exception:  # noqa: BLE001 — a malformed generation is skipped
            continue
        if target < cutoff:
            continue
        pos = int(np.searchsorted(times.to_numpy(), np.datetime64(target)))
        if pos >= len(times) or times.iloc[pos] != target:
            continue
        actual = float(vals.iloc[pos])
        q50 = float(step.get("q50", step.get("gwl", np.nan)))
        if not (np.isfinite(actual) and np.isfinite(q50)):
            continue
        rows.append({
            "anchor_time": anchor,
            "target_time": target,
            "q05": float(step.get("q05", np.nan)),
            "q50": q50,
            "q95": float(step.get("q95", np.nan)),
            "actual": actual,
            "error": q50 - actual,
            "generated_ts": pod.get("forecast_generated")
                            or pod.get("forecast_generation_ts"),
        })
    if not rows:
        return pd.DataFrame(columns=["anchor_time", "target_time", "q05", "q50",
                                     "q95", "actual", "error", "generated_ts"])
    return pd.DataFrame(rows).sort_values("target_time").reset_index(drop=True)


def trajectory_forecast_safe(station: str) -> dict:
    """Call the EXISTING inference engine — a read-and-score path, no retraining.

    ``_trajectory.trajectory_forecast`` builds features from the aligned table and
    scores the frozen quantile models; it writes nothing. Returns a dict with an
    ``error`` key instead of raising when artifacts or data are missing.
    """
    if not station:
        return {"error": "no station selected", "reason": "no_station"}
    try:
        import importlib
        traj = importlib.import_module("_trajectory")
        return traj.trajectory_forecast(str(station))
    except Exception as e:  # noqa: BLE001 — surface as a UI state
        return {"error": f"{type(e).__name__}: {e}", "reason": "inference_failed"}


def inference_available() -> tuple[bool, str]:
    """Cheap pre-flight so the UI can explain *why* a forecast cannot be shown."""
    needed = {
        "models/traj_config.json": MODELS / "traj_config.json",
        "models/traj_xgb_q05.json": MODELS / "traj_xgb_q05.json",
        "models/traj_xgb_q50.json": MODELS / "traj_xgb_q50.json",
        "models/traj_xgb_q95.json": MODELS / "traj_xgb_q95.json",
        "models/traj_calibration.json": MODELS / "traj_calibration.json",
        "models/traj_reliability.json": MODELS / "traj_reliability.json",
        "data/aligned/table_6h.parquet": ALIGNED / "table_6h.parquet",
    }
    missing = [name for name, p in needed.items() if not p.exists()]
    if missing:
        return False, "missing: " + ", ".join(missing)
    try:
        import importlib
        importlib.import_module("xgboost")
        importlib.import_module("_trajectory")
    except Exception as e:  # noqa: BLE001
        return False, f"import failed: {type(e).__name__}: {e}"
    return True, "ready"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.0f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"
