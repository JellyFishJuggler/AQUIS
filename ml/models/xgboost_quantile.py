"""
XGBoost + quantile-regression forecasting for groundwater level time series.

Point prediction:  ``XGBRegressor(objective="reg:squarederror")``.
Uncertainty:       three quantile models at q = 0.05 / 0.5 / 0.95 via the
                   native ``reg:quantileerror`` objective -> 90% prediction
                   interval.  (Approach follows Alkon et al. 2024, Environ.
                   Res. Lett. and subsequent GWPZ-modelling literature.)

Features (tree models are scale-invariant, so no scaling is needed):
    * ``time_hours``                 - hours since first reading (always)
    * ``sin_doy`` / ``cos_doy`` / year  - seasonal encoding (optional)
    * ``lag_{L}`` + ``roll_{W}``     - causal autoregressive features
      (L in {1,2,3,4,28,120} 6-hour steps = 1-24h, 7d, 30d; W in {28,120}
      = trailing 7d / 30d means).  These are the primary drivers of short-
      horizon accuracy and interval calibration on the chronological holdout.

Artifacts per station (``artifacts/<slug>/``):
    xgb_point.joblib, xgb_q05.joblib, xgb_q50.joblib, xgb_q95.joblib
    features.json, xgboost_metadata.json

Public API:
    train_xgb_quantile_for_station()
    predict_xgb_quantile()      (recursive forecasting; works for any t)
    get_test_predictions()      (test-period actuals vs predictions)
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

_ML_ROOT = Path(__file__).resolve().parent.parent

from ml.preprocessing.timeseries import (  # noqa: E402
    ARTIFACTS_DIR,
    GWL_COL,
    TIME_COL,
    TRAIN_RATIO,
    build_time_index,
    get_station_series,
    resolve_slug_dir,
    unique_station_dir,
)
from ml.utils import format_duration  # noqa: E402

DEFAULT_PARQUET = _ML_ROOT / "data" / "processed" / "common.parquet"

SAMPLING_HOURS = 6  # data cadence: 6-hourly telemetry
LAG_STEPS = [1, 2, 3, 4, 28, 120]
ROLL_WINDOWS = [28, 120]

DEFAULT_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
QUANTILES = [0.05, 0.5, 0.95]
POINT_FILES = {
    0.05: "xgb_q05.joblib",
    0.5: "xgb_q50.joblib",
    0.95: "xgb_q95.joblib",
}
POINT_MODEL_FILE = "xgb_point.joblib"


def build_xgb_point(params: dict | None = None) -> XGBRegressor:
    """Point-prediction model (conditional mean)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    p.pop("quantile_alpha", None)
    return XGBRegressor(objective="reg:squarederror", **p)


def build_xgb_quantile(quantile: float, params: dict | None = None) -> XGBRegressor:
    """Quantile-regression model using XGBoost's native quantile error loss."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    p.pop("quantile_alpha", None)
    return XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=quantile, **p
    )


def build_feature_dataframe(
    station_df: pd.DataFrame,
    use_seasonal: bool = True,
    use_lags: bool = True,
) -> pd.DataFrame:
    """Build the model feature matrix for a station's series.

    ``time_hours`` is intentionally unscaled: tree ensembles are invariant
    to monotone feature transforms, so no StandardScaler is needed.

    Lag / rolling features are causal (computed from earlier observations
    only), so the matrix remains valid for genuine forecasting.
    """
    if "time_hours" not in station_df.columns:
        station_df = build_time_index(station_df)

    t = pd.to_datetime(station_df[TIME_COL].values)
    data = {"time_hours": station_df["time_hours"].values}

    if use_seasonal:
        doy = t.dayofyear.to_numpy()
        data["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        data["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        data["year"] = t.year.astype(float).to_numpy()

    if use_lags:
        y = station_df[GWL_COL]
        for step in LAG_STEPS:
            data[f"lag_{step}"] = y.shift(step)
        for win in ROLL_WINDOWS:
            data[f"roll_{win}"] = y.rolling(win).mean()

    return pd.DataFrame(data)


def split_series(
    station_df: pd.DataFrame,
    use_seasonal: bool = True,
    use_lags: bool = True,
    train_ratio: float = TRAIN_RATIO,
):
    """Align features with valid targets and split chronologically.

    Rows are dropped when the target OR any feature is NaN (warm-up lags),
    then split at ``train_ratio``.  Returns
    ``(X_train, X_test, y_train, y_test, t_train, t_test)``.
    """
    X = build_feature_dataframe(station_df, use_seasonal, use_lags)
    y = station_df[GWL_COL].values.astype(float)
    valid = ~np.isnan(y) & X.notna().all(axis=1).to_numpy()
    timestamps = pd.to_datetime(station_df[TIME_COL].values[valid])
    X = X[valid].reset_index(drop=True)
    y = y[valid]
    split = int(len(y) * train_ratio)
    return (
        X.iloc[:split],
        X.iloc[split:],
        y[:split],
        y[split:],
        timestamps[:split],
        timestamps[split:],
    )


def _round_metrics(metrics: dict) -> dict:
    return {k: round(float(v), 4) for k, v in metrics.items()}


def _load_config(out_dir: Path) -> dict:
    with open(out_dir / "features.json") as f:
        return json.load(f)


def _load_models(out_dir: Path) -> dict:
    return {
        "point": joblib.load(out_dir / POINT_MODEL_FILE),
        **{q: joblib.load(out_dir / POINT_FILES[q]) for q in QUANTILES},
    }


def _resolve_station_dir(station: str, artifacts_root: Path | None = None) -> Path:
    root = artifacts_root or ARTIFACTS_DIR
    resolved = resolve_slug_dir(root, station)
    if resolved is not None and (resolved / POINT_MODEL_FILE).is_file():
        return resolved
    as_slug = root / station
    if (as_slug / POINT_MODEL_FILE).is_file():
        return as_slug
    raise FileNotFoundError(
        f"No XGBoost model for '{station}' (looked in {root}). "
        "Train it first: python -m ml.training.train_forecast --station '...'"
    )


def _observed_buffer(station_df: pd.DataFrame) -> dict:
    """Map {sample_index: observed_value} for all non-NaN observations."""
    vals = station_df[GWL_COL].values.astype(float)
    return {i: float(v) for i, v in enumerate(vals) if not np.isnan(v)}


def _position(time_hours: float) -> int:
    """Sample position for a time (hours since first reading)."""
    return int(np.floor(time_hours / SAMPLING_HOURS))


def _features_at(cfg: dict, buffer: dict, pos: int, time_hours: float) -> pd.DataFrame:
    """Build one feature row at grid-``pos`` (target ``time_hours``)."""
    min_known = min(buffer)
    pad = buffer[min_known]
    t0 = pd.to_datetime(cfg["t0"])
    ts = t0 + timedelta(hours=float(time_hours))

    data: dict = {"time_hours": float(time_hours)}
    if cfg["use_seasonal"]:
        doy = ts.timetuple().tm_yday
        data["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        data["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        data["year"] = float(ts.year)

    if cfg["use_lags"]:
        for step in cfg["lag_steps"]:
            data[f"lag_{step}"] = buffer.get(pos - step, pad)
        for win in cfg["roll_windows"]:
            recent = []
            j = pos - 1
            while j >= min_known and len(recent) < win:
                if j in buffer:
                    recent.insert(0, buffer[j])
                j -= 1
            while len(recent) < win:
                recent.insert(0, pad)
            data[f"roll_{win}"] = float(np.mean(recent))

    return pd.DataFrame([data])[cfg["features"]]


def _fill_buffer_until(cfg: dict, buffer: dict, target_pos: int, point_model) -> None:
    """Recursively forecast observations up to ``target_pos - 1``."""
    max_known = max(buffer)
    while max_known < target_pos - 1:
        nxt = max_known + 1
        X = _features_at(cfg, buffer, nxt, nxt * SAMPLING_HOURS)
        buffer[nxt] = float(point_model.predict(X)[0])
        max_known = nxt


def _predict_for_times(cfg: dict, buffer: dict, time_hours, models) -> dict:
    th = np.asarray(time_hours, dtype=float)
    rows: list[pd.DataFrame] = []
    for t in th:
        pos = _position(float(t))
        if pos > max(buffer) + 1:
            _fill_buffer_until(cfg, buffer, pos, models["point"])
        rows.append(_features_at(cfg, buffer, pos, float(t)))
    X = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        [], columns=cfg["features"]
    )
    return {
        "time_hours": th,
        "point": models["point"].predict(X),
        "lower": models[0.05].predict(X),
        "upper": models[0.95].predict(X),
    }


def train_xgb_quantile_for_station(
    station: str,
    params: dict | None = None,
    use_seasonal: bool = True,
    use_lags: bool = True,
    parquet_path: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train point + {0.05, 0.5, 0.95} quantile XGBoost models for a station.

    Saves the four models plus ``features.json`` and ``xgboost_metadata.json``
    into ``artifacts/<station_slug>/``.

    Returns metrics, coverage, training duration, and artifact dir.
    """
    parquet_path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET
    root = Path(artifacts_root) if artifacts_root else ARTIFACTS_DIR
    out_dir = unique_station_dir(root, station)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = out_dir.name

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    station_df = get_station_series(parquet_path, station)
    X_train, X_test, y_train, y_test, _, t_test = split_series(
        station_df, use_seasonal=use_seasonal, use_lags=use_lags
    )

    start = time.monotonic()
    point = build_xgb_point(params).fit(X_train, y_train)
    quantile_models = {
        q: build_xgb_quantile(q, params).fit(X_train, y_train)
        for q in QUANTILES
    }
    train_seconds = time.monotonic() - start

    yte = y_test
    pred_point = point.predict(X_test)
    pred_q = {q: m.predict(X_test) for q, m in quantile_models.items()}

    point_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(yte, pred_point))),
        "mae": float(mean_absolute_error(yte, pred_point)),
        "r2": float(r2_score(yte, pred_point)),
    }
    q50_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(yte, pred_q[0.5]))),
        "mae": float(mean_absolute_error(yte, pred_q[0.5])),
        "r2": float(r2_score(yte, pred_q[0.5])),
    }
    inside = (yte >= pred_q[0.05]) & (yte <= pred_q[0.95])
    coverage = float(np.mean(inside))
    interval_width = float(np.mean(pred_q[0.95] - pred_q[0.05]))

    joblib.dump(point, out_dir / POINT_MODEL_FILE)
    for q in QUANTILES:
        joblib.dump(quantile_models[q], out_dir / POINT_FILES[q])

    cfg = {
        "station": station,
        "use_seasonal": bool(use_seasonal),
        "use_lags": bool(use_lags),
        "lag_steps": LAG_STEPS,
        "roll_windows": ROLL_WINDOWS,
        "features": list(X_train.columns),
        "t0": pd.to_datetime(station_df[TIME_COL].min()).isoformat(),
        "train_ratio": TRAIN_RATIO,
    }
    with open(out_dir / "features.json", "w") as f:
        json.dump(cfg, f, indent=2)

    metadata = {
        "station": station,
        "slug": slug,
        "model_type": "xgboost_quantile",
        "quantiles": QUANTILES,
        "use_seasonal": cfg["use_seasonal"],
        "use_lags": cfg["use_lags"],
        "features": cfg["features"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": int(len(y_train)),
        "test_size": int(len(yte)),
        "point_metrics": _round_metrics(point_metrics),
        "quantile_metrics": {
            "coverage_90": round(coverage, 4),
            "mean_interval_width": round(interval_width, 4),
            "q50_metrics": _round_metrics(q50_metrics),
        },
        "train_seconds": round(train_seconds, 2),
        "duration": format_duration(train_seconds),
    }
    with open(out_dir / "xgboost_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _log(f"  Train: {len(y_train):,}  |  Test: {len(yte):,}")
    _log(
        f"  Point  RMSE: {point_metrics['rmse']:.4f}  "
        f"MAE: {point_metrics['mae']:.4f}  R²: {point_metrics['r2']:.4f}"
    )
    _log(
        f"  90% PI     coverage: {coverage:.1%}   "
        f"mean width: {interval_width:.4f} m"
    )
    _log(f"  Training time: {format_duration(train_seconds)}")

    return {
        "station": station,
        "slug": slug,
        "point_metrics": point_metrics,
        "quantile_metrics": {
            "coverage_90": coverage,
            "mean_interval_width": interval_width,
            "q50_metrics": q50_metrics,
        },
        "duration": format_duration(train_seconds),
        "train_seconds": train_seconds,
        "artifact_dir": str(out_dir),
        "test_times": t_test,
        "test_predictions": pred_point,
        "test_lower": pred_q[0.05],
        "test_upper": pred_q[0.95],
        "test_actual": yte,
    }


def predict_xgb_quantile(
    time_hours,
    station: str,
    artifacts_root: str | Path | None = None,
) -> dict:
    """Point prediction + 90% interval (q05..q95) for given time-hours.

    ``time_hours`` is hours elapsed since the station's first reading.
    Works for historical points, missing-data gaps, and future horizons
    (recursive forecasting).  Returns arrays ``point``, ``lower``, ``upper``.
    """
    out_dir = _resolve_station_dir(station, artifacts_root)
    cfg = _load_config(out_dir)
    models = _load_models(out_dir)

    station_df = get_station_series(DEFAULT_PARQUET, cfg.get("station", station))
    buffer = _observed_buffer(station_df)
    if not buffer:
        raise ValueError(f"No valid observations for station '{station}'.")

    return _predict_for_times(cfg, buffer, time_hours, models)


def get_test_predictions(
    station: str,
    artifacts_root: str | Path | None = None,
) -> dict:
    """Actual vs predicted in the held-out test period (for plotting)."""
    out_dir = _resolve_station_dir(station, artifacts_root)
    cfg = _load_config(out_dir)
    models = _load_models(out_dir)

    station_df = get_station_series(DEFAULT_PARQUET, cfg.get("station", station))
    _, X_test, _, y_test, _, t_test = split_series(
        station_df, use_seasonal=cfg["use_seasonal"], use_lags=cfg["use_lags"]
    )

    return {
        "time": t_test,
        "actual": y_test,
        "point": models["point"].predict(X_test),
        "lower": models[0.05].predict(X_test),
        "upper": models[0.95].predict(X_test),
    }