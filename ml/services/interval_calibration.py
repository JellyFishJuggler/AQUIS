"""Interval calibration for honest multi-step prediction intervals."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    DIRECT_HORIZONS,
    MAX_HORIZON,
    RECURSIVE_HORIZON_START,
    load_models,
    predict_direct,
    predict_recursive,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    GWL_COL,
    TIME_COL,
    full_pipeline,
    prepare_feature_matrix,
)


@dataclass
class IntervalCalibration:
    """Holds per-depth/horizon half-widths for calibrated intervals."""
    half_widths: dict[int | str, float]
    alpha: float
    is_direct: dict[int | str, bool]

    def half_width_at(self, depth_or_horizon: int | str) -> float:
        """Get calibrated half-width for a depth (recursive) or horizon (direct)."""
        if depth_or_horizon in self.half_widths:
            return self.half_widths[depth_or_horizon]
        if isinstance(depth_or_horizon, int):
            keys = [k for k in self.half_widths.keys() if isinstance(k, int) and k <= depth_or_horizon]
            if keys:
                return self.half_widths[max(keys)]
            return max(self.half_widths.values()) if self.half_widths else 0.0
        return 0.0

    def half_width_at_horizon(self, horizon: int) -> float:
        """Get calibrated half-width for a specific horizon (direct)."""
        if horizon <= 7:
            return self.half_width_at(horizon)
        elif horizon <= 14:
            return self.half_width_at("8_14")
        elif horizon <= 21:
            return self.half_width_at("15_21")
        elif horizon <= 30:
            return self.half_width_at("22_30")
        return self.half_width_at(horizon)


def _smooth_half_widths(raw_widths: dict[int, float], smooth_span: int = 7) -> dict[int, float]:
    """Forward-fill gaps, rolling-median smooth, enforce monotonic non-decreasing."""
    if not raw_widths:
        return {}

    max_depth = max(raw_widths.keys())
    arr = np.full(max_depth + 1, np.nan)
    for d, w in raw_widths.items():
        arr[d] = w

    arr = pd.Series(arr).ffill().bfill().values
    arr = pd.Series(arr).rolling(smooth_span, min_periods=1, center=True).median().values
    arr = np.maximum.accumulate(arr)

    return {d: float(arr[d]) for d in range(1, max_depth + 1)}


def estimate_calibration_direct(
    cfg: dict,
    models: dict,
    station_df: pd.DataFrame,
    feature_cols: list[str],
    alpha: float = 0.90,
) -> dict[int | str, float]:
    """Calibration for direct horizons using held-out test set."""
    half_widths = {}

    for h in DIRECT_HORIZONS:
        h_str = str(h)
        if h_str not in models["direct"]:
            continue

        errors = []
        for station, grp in station_df.groupby("Station"):
            grp = grp.sort_values(TIME_COL)
            if len(grp) < 200:
                continue

            X_all, y_all, _ = prepare_feature_matrix(grp[feature_cols + [GWL_COL]])
            if len(X_all) < 50:
                continue

            split = int(len(X_all) * 0.8)
            X_test = X_all[split:]
            y_test = y_all[split:]

            if isinstance(h, int):
                target_shift = h * 4
            else:
                start = int(h.split("_")[0])
                target_shift = start * 4

            if target_shift >= len(y_test):
                continue

            y_shifted = np.roll(y_test, -target_shift)
            y_shifted[-target_shift:] = np.nan
            valid = ~np.isnan(y_shifted)
            if valid.sum() < 10:
                continue

            preds = predict_direct(models, X_test[valid], h)
            point_preds = preds["point"]
            actual = y_shifted[valid]

            abs_errors = np.abs(actual - point_preds)
            if len(abs_errors) > 0:
                errors.extend(abs_errors)

        if errors:
            half_widths[h] = float(np.quantile(errors, alpha))

    return half_widths


def estimate_calibration_recursive(
    cfg: dict,
    models: dict,
    station_df: pd.DataFrame,
    feature_cols: list[str],
    alpha: float = 0.90,
    smooth_span: int = 7,
) -> dict[int, float]:
    """Calibration for recursive horizons using held-out test set."""
    point_model = models["recursive"]["point"]
    ec_model = models.get("error_correction")

    all_depth_errors: dict[int, list[float]] = {}

    for station, grp in station_df.groupby("Station"):
        grp = grp.sort_values(TIME_COL)
        if len(grp) < 300:
            continue

        X_all, y_all, _ = prepare_feature_matrix(grp[feature_cols + [GWL_COL]])
        if len(X_all) < 100:
            continue

        split = int(len(X_all) * 0.8)
        X_test = X_all[split:]
        y_test = y_all[split:]

        for anchor_idx in range(0, len(X_test) - MAX_HORIZON, MAX_HORIZON // 2):
            last_feats = X_test[anchor_idx]
            true_future = y_test[anchor_idx + 1:anchor_idx + MAX_HORIZON + 1]
            if len(true_future) < MAX_HORIZON:
                continue

            preds = predict_recursive(models, last_feats, MAX_HORIZON, feature_cols, ec_model)
            point_preds = np.array(preds["point"])

            for depth, (actual, pred) in enumerate(zip(true_future, point_preds, strict=False), 1):
                if depth not in all_depth_errors:
                    all_depth_errors[depth] = []
                all_depth_errors[depth].append(abs(actual - pred))

    raw_widths = {}
    for depth, errors in all_depth_errors.items():
        if len(errors) >= 10:
            raw_widths[depth] = float(np.quantile(errors, alpha))

    return _smooth_half_widths(raw_widths, smooth_span)


def estimate_calibration(
    cfg: dict,
    models: dict,
    station_df: pd.DataFrame,
    feature_cols: list[str],
    alpha: float = 0.90,
    smooth_span: int = 7,
) -> IntervalCalibration:
    """Estimate calibration for both direct and recursive horizons."""
    direct_widths = estimate_calibration_direct(cfg, models, station_df, feature_cols, alpha)
    recursive_widths = estimate_calibration_recursive(cfg, models, station_df, feature_cols, alpha, smooth_span)

    half_widths = {**direct_widths, **recursive_widths}
    is_direct = {k: k in direct_widths for k in half_widths}

    return IntervalCalibration(half_widths=half_widths, alpha=alpha, is_direct=is_direct)


def widen(
    calibration: IntervalCalibration,
    time_hours: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    anchor_pos: int,
    is_direct: bool = False,
    horizon: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Union of raw interval and symmetric calibrated band around POINT estimate."""
    point = np.asarray(point)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    new_lower = lower.copy()
    new_upper = upper.copy()

    for i in range(len(point)):
        if is_direct and horizon is not None:
            cal_half = calibration.half_width_at_horizon(horizon)
        else:
            depth = i + 1
            cal_half = calibration.half_width_at(depth)

        cal_lower = point[i] - cal_half
        cal_upper = point[i] + cal_half

        new_lower[i] = min(lower[i], cal_lower)
        new_upper[i] = max(upper[i], cal_upper)

    return new_lower, new_upper


def calibrate_and_widen(
    station_display_name: str,
    time_hours: np.ndarray,
    artifacts_root: Path | None = None,
    alpha: float = 0.90,
    parquet_path: Path | str | None = None,
    backend_csv: Path | str | None = None,
) -> dict[str, Any]:
    """One-stop: load models, calibrate, predict, widen."""
    if artifacts_root is None:
        artifacts_root = ARTIFACTS_DIR
    if parquet_path is None:
        parquet_path = Path(__file__).resolve().parent.parent.parent / "ml" / "data" / "processed" / "common.parquet"
    if backend_csv is None:
        backend_csv = Path(__file__).resolve().parent.parent.parent / "back-end" / "db" / "data.csv"

    pipe = full_pipeline(parquet_path, backend_csv)
    feature_cols = pipe["feature_cols"]
    station_df = pipe["train"]

    slug = station_display_name.replace("/", "_").replace("\\", "_")
    artifact_dir = Path(artifacts_root) / slug
    if not artifact_dir.exists():
        for d in artifact_dir.parent.iterdir():
            if d.is_dir() and station_display_name in d.name:
                artifact_dir = d
                break

    models = load_models(artifact_dir)

    calibration = estimate_calibration(cfg={}, models=models, station_df=station_df, feature_cols=feature_cols, alpha=alpha)

    X_last, _, _ = prepare_feature_matrix(station_df[feature_cols + [GWL_COL]].tail(1))
    if len(X_last) == 0:
        raise ValueError("No valid features for prediction")

    last_feats = X_last[0]
    n_steps = len(time_hours)

    direct_preds = {"point": [], "lower": [], "upper": []}
    recursive_preds = {"point": [], "lower": [], "upper": []}

    for i, h in enumerate(time_hours):
        horizon_days = int(h / 24)
        if horizon_days <= 30 and horizon_days in DIRECT_HORIZONS:
            pred = predict_direct(models, last_feats.reshape(1, -1), horizon_days)
            direct_preds["point"].append(pred["point"])
            direct_preds["lower"].append(pred["q05"])
            direct_preds["upper"].append(pred["q95"])
        else:
            break

    rec_start = len(direct_preds["point"])
    if rec_start < n_steps:
        rec_steps = n_steps - rec_start
        rec = predict_recursive(models, last_feats, rec_steps, feature_cols, models.get("error_correction"))
        recursive_preds["point"].extend(rec["point"])
        recursive_preds["lower"].extend(rec["q05"])
        recursive_preds["upper"].extend(rec["q95"])

    point_arr = np.array(direct_preds["point"] + recursive_preds["point"])
    lower_arr = np.array(direct_preds["lower"] + recursive_preds["lower"])
    upper_arr = np.array(direct_preds["upper"] + recursive_preds["upper"])

    anchor_pos = 0
    new_lower, new_upper = widen(calibration, time_hours, point_arr, lower_arr, upper_arr, anchor_pos)

    return {
        "time_hours": time_hours,
        "point": point_arr,
        "lower": new_lower,
        "upper": new_upper,
        "calibration": calibration,
    }


def diagnose_station(
    cfg: dict,
    models: dict,
    station_df: pd.DataFrame,
    feature_cols: list[str],
    alpha: float = 0.90,
    coverage_floor: float = 0.75,
    horizon_days: int = 90,
    test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Classify station as reliable / directional / weak.

    Evaluates on the held-out test split (``test_df``) when provided so the
    reported metrics agree with the dashboard's Test Period panel. Falls back
    to an in-house 90/10 split of ``station_df`` when ``test_df`` is None.
    """
    calibration = estimate_calibration(cfg, models, station_df, feature_cols, alpha)

    point_model = models["recursive"]["point"]
    q05_model = models["recursive"]["q05"]
    q95_model = models["recursive"]["q95"]
    ec_model = models.get("error_correction")

    if test_df is not None and len(test_df) > 0:
        X_test, y_test, _ = prepare_feature_matrix(test_df[feature_cols + [GWL_COL]])
    else:
        X_all, y_all, _ = prepare_feature_matrix(station_df[feature_cols + [GWL_COL]])
        split = int(len(X_all) * 0.9)
        X_test = X_all[split:]
        y_test = y_all[split:]

    if len(X_test) == 0:
        return {
            "station": station_df["Station"].iloc[0],
            "slug": station_df["slug"].iloc[0] if "slug" in station_df.columns else station_df["Station"].iloc[0],
            "label": "weak",
            "reason": "no evaluation data",
            "coverage": 0.0,
            "one_step_coverage": 0.0,
            "one_step_rmse": 0.0,
            "one_step_mae": 0.0,
            "one_step_r2": 0.0,
            "one_step_nrmse": 1.0,
            "multi_step_rmse": 0.0,
            "multi_step_r2": 0.0,
            "multi_step_nrmse": 1.0,
            "r2_meaningful": False,
            "metric_note": "no evaluation data",
            "shallow_error": 1.0,
            "gwl_span": 0.0,
            "n_obs": int(len(station_df)),
            "tail_half_width": 0.0,
            "half_width_at_horizon": 0.0,
            "horizon_days": horizon_days,
            "max_depth": min(horizon_days * 4, MAX_HORIZON),
            "calibration": {"alpha": calibration.alpha, "half_widths": {}},
        }

    one_step_point = point_model.predict(X_test)

    def _recon(raw: np.ndarray) -> np.ndarray:
        if models.get("delta_mode") and models.get("lag1_index") is not None and len(X_test) > 0:
            return X_test[:, models["lag1_index"]] + raw
        return raw

    one_step_point = _recon(one_step_point)
    one_step_lower = _recon(q05_model.predict(X_test))
    one_step_upper = _recon(q95_model.predict(X_test))

    one_step_cov = np.mean((y_test >= one_step_lower) & (y_test <= one_step_upper))
    one_step_rmse = float(np.sqrt(np.mean((y_test - one_step_point) ** 2)))
    one_step_mae = float(np.mean(np.abs(y_test - one_step_point)))
    one_step_r2 = float(1 - np.sum((y_test - one_step_point) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    gwl_span = float(np.max(y_test) - np.min(y_test))

    shallow_rmse = one_step_rmse
    shallow_error = shallow_rmse / gwl_span if gwl_span > 0 else 1.0

    max_depth = min(horizon_days * 4, MAX_HORIZON)
    multi_covs = []
    multi_errors = []

    for anchor_idx in range(0, len(X_test) - max_depth, max_depth // 4):
        last_feats = X_test[anchor_idx]
        true_future = y_test[anchor_idx + 1:anchor_idx + max_depth + 1]
        if len(true_future) < max_depth:
            continue

        preds = predict_recursive(models, last_feats, max_depth, feature_cols, ec_model)
        point_preds = np.array(preds["point"])
        lower_preds = np.array(preds["q05"])
        upper_preds = np.array(preds["q95"])

        for i, (actual, pt, lo, hi) in enumerate(zip(true_future, point_preds, lower_preds, upper_preds, strict=False)):
            depth = i + 1
            cal_half = calibration.half_width_at(depth)
            cal_lo = pt - cal_half
            cal_hi = pt + cal_half
            covered = cal_lo <= actual <= cal_hi
            multi_covs.append(covered)
            multi_errors.append(abs(actual - pt))

    calibrated_coverage = float(np.mean(multi_covs)) if multi_covs else 0.0
    multi_rmse = float(np.sqrt(np.mean(np.array(multi_errors) ** 2))) if multi_errors else 0.0
    multi_r2 = float(1 - np.sum(np.array(multi_errors) ** 2) / np.sum((y_test[-len(multi_errors):] - np.mean(y_test[-len(multi_errors):])) ** 2)) if multi_errors else -1.0

    tail_half_width = calibration.half_width_at(max_depth)
    horizon_half_width = calibration.half_width_at_horizon(horizon_days)

    # Rule 5.1: on near-flat wells (gwl_span < 0.75 m) R2 is dominated by a tiny SS_tot
    # denominator and is NOT meaningful. NRMSE = RMSE / gwl_span is the honest scale metric.
    r2_meaningful = bool(gwl_span >= 0.75)
    one_nrmse = float(one_step_rmse / gwl_span) if gwl_span > 0 else 1.0
    multi_nrmse = float(multi_rmse / gwl_span) if gwl_span > 0 else 1.0
    if not r2_meaningful:
        metric_note = "low target variance (gwl_span<0.75 m): R2 unreliable; use RMSE/MAE/NRMSE"
    else:
        metric_note = "R2 meaningful (gwl_span>=0.75 m)"

    if calibrated_coverage >= coverage_floor and shallow_error < 0.15 and horizon_half_width / gwl_span < 0.5:
        label = "reliable"
        reason = f"coverage={calibrated_coverage:.2f}, shallow_err/GWL={shallow_error:.2f}, horizon_width/GWL={horizon_half_width/gwl_span:.2f}"
    elif calibrated_coverage >= coverage_floor:
        label = "directional"
        reason = f"coverage OK ({calibrated_coverage:.2f}) but wide intervals or high shallow error ({shallow_error:.2f})"
    else:
        label = "weak"
        reason = f"coverage={calibrated_coverage:.2f} < floor ({coverage_floor})"

    return {
        "station": station_df["Station"].iloc[0],
        "slug": station_df["slug"].iloc[0] if "slug" in station_df.columns else station_df["Station"].iloc[0],
        "label": label,
        "reason": reason,
        "coverage": calibrated_coverage,
        "one_step_coverage": float(one_step_cov),
        "one_step_rmse": one_step_rmse,
        "one_step_mae": one_step_mae,
        "one_step_r2": one_step_r2,
        "one_step_nrmse": one_nrmse,
        "multi_step_rmse": multi_rmse,
        "multi_step_r2": multi_r2,
        "multi_step_nrmse": multi_nrmse,
        "r2_meaningful": r2_meaningful,
        "metric_note": metric_note,
        "shallow_error": shallow_error,
        "gwl_span": gwl_span,
        "n_obs": int(len(station_df)),
        "tail_half_width": tail_half_width,
        "half_width_at_horizon": horizon_half_width,
        "horizon_days": horizon_days,
        "max_depth": max_depth,
        "calibration": {
            "alpha": calibration.alpha,
            "half_widths": {str(k): v for k, v in calibration.half_widths.items()},
        },
    }