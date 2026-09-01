"""Honest prediction-interval calibration for recursive (multi-step) forecasts.

Why this exists
---------------
The raw quantile XGBoost interval (q05..q95) is fit one-step-ahead.  When the
model is asked to forecast many steps out it runs *recursively* — each step
feeds its own previous prediction back as a lag input — and uncertainty grows
with forecast distance.  The raw interval does not widen with distance, so on
long horizons it is far too narrow: real multi-step 90% coverage is ~19% on
the fleet, far below the nominal 90%.

This module re-calibrates the interval using the model's own held-out test
period (data it never saw while training).  It measures, as a function of
recursion depth, how large the recursive absolute error actually is and widens
the interval so that a chosen fraction (default 90%) of held-out points fall
inside.  The future forecast is also a recursive run, so the same per-depth
widths transfer to it.

Only the *interval* is changed; the point (median) forecast is left untouched.
This is deliberately conservative and honest: we widen rather than pretend the
narrow one-step interval is valid far into the future.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.models.xgboost_quantile import (
    DEFAULT_PARQUET,
    SAMPLING_HOURS,
    _load_config,
    _load_models,
    _observed_buffer,
    _position,
    _predict_for_times,
    split_series,
)
from ml.preprocessing.timeseries import GWL_COL, TIME_COL, get_station_series

DEFAULT_ALPHA = 0.90


def _station_artifact_dir(station: str, root) -> Path:
    for d in root.iterdir():
        if (d / "features.json").is_file() and (d / "xgb_point.joblib").is_file():
            meta_file = d / "xgboost_metadata.json"
            if meta_file.is_file():
                import json

                meta = json.load(open(meta_file))
                if meta.get("station") == station:
                    return d
    raise FileNotFoundError(f"No artifact dir for station {station!r}")


class IntervalCalibration:
    """Per-depth symmetric half-width covering ~``alpha`` of recursive errors."""

    def __init__(self, half_widths: np.ndarray, alpha: float):
        # half_widths[0] applies to the first forecast step (depth 1).
        self.half_widths = np.asarray(half_widths, dtype=float)
        self.max_depth = len(self.half_widths)
        self.alpha = alpha

    def half_width_at(self, depth: int) -> float:
        """Symmetric half-width multiplier/margin for a given recursion depth."""
        d = int(depth)
        if d <= 0:
            return self.half_widths[0]
        idx = min(d - 1, self.max_depth - 1)
        return float(self.half_widths[idx])

    def report(self, station: str) -> str:
        import numpy as _np

        return (
            f"{station}: alpha={self.alpha:.0%}, calibration depths={self.max_depth}, "
            f"half-width at d=1 {self.half_widths[0]:.3f} m, "
            f"at max depth {self.half_widths[-1]:.3f} m"
        )


def estimate_calibration(
    cfg: dict,
    models: dict,
    station_df: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    smooth_span: int = 7,
) -> IntervalCalibration:
    """Estimate per-depth interval half-widths from the held-out recursive test.

    Runs the recursive forecast over the entire held-out test period, computes
    the absolute error at each recursion depth (position in the test window),
    then takes the symmetric ``alpha``-quantile per depth.  A light rolling
    median smooths depth-to-depth noise so widths grow monotonically-ish.

    ``smooth_span`` is in points (each depth == one 6-hour step).
    """
    y = station_df[GWL_COL].to_numpy(dtype=float)
    t = pd.to_datetime(station_df[TIME_COL].to_numpy())
    hours = station_df["time_hours"].to_numpy()
    idx_by_ts = {ts: i for i, ts in enumerate(t)}

    _, X_test, _, y_test, _, t_test = split_series(
        station_df, use_seasonal=cfg["use_seasonal"], use_lags=cfg["use_lags"]
    )  # noqa: F401
    max_train_t = t_test[0] - pd.Timedelta(hours=SAMPLING_HOURS)

    buffer: dict = {}
    for i, (rt, ry) in enumerate(zip(t, y)):
        if rt <= max_train_t and not np.isnan(ry):
            buffer[_position(hours[i])] = float(ry)

    test_hours = [hours[idx_by_ts[ts]] for ts in t_test]
    if not test_hours:
        raise ValueError("no held-out test points for calibration")

    # Capture the recursion anchor BEFORE predicting: _predict_for_times fills
    # the buffer forward (mutating it), so max(buffer) afterwards would be the
    # last test position and every depth would collapse to 1.
    anchor_pos = max(buffer)

    res = _predict_for_times(cfg, buffer, test_hours, models)
    err = np.abs(np.asarray(res["point"]) - np.asarray(y_test, dtype=float))

    # The recursion walks the 6-hour grid forward from the last training
    # observation, so the "depth" of a test point is its grid position minus
    # the recursion anchor — NOT its index in the (gappy) test array.  A test
    # point two real 6-hour ticks past a telemetry gap is depth 2, regardless
    # of its array index.
    depths = np.maximum(1, np.floor(np.asarray(test_hours) / SAMPLING_HOURS).astype(int) - anchor_pos)
    max_depth = int(depths.max())

    half = np.full(max_depth, np.nan, dtype=float)
    # group test points that share the same grid depth and take the quantile
    for d in range(1, max_depth + 1):
        vals = err[depths == d]
        if len(vals):
            half[d - 1] = float(np.quantile(vals, alpha))
    # forward-fill any grid depths with no test point (e.g. from gaps)
    running = None
    for i in range(len(half)):
        if np.isnan(half[i]):
            half[i] = running if running is not None else 0.0
        else:
            running = half[i]

    # smooth to reduce per-depth noise and encourage monotonic non-decrease-ish
    if len(half) > smooth_span:
        s = pd.Series(half).rolling(smooth_span, min_periods=1, center=True).median()
        half = s.bfill().fillna(0.0).to_numpy(dtype=float)
        # enforce non-decreasing running max so width never shrinks with depth
        running = np.maximum.accumulate(half)
        half = running

    return IntervalCalibration(half, alpha)


def widen(
    calibration: IntervalCalibration,
    time_hours,
    point,
    lower,
    upper,
    anchor_pos: int | None = None,
):
    """Widen a recursive interval so it honestly covers ``alpha`` of outcomes.

    ``time_hours`` are the requested forecast times (hours since first reading).
    ``anchor_pos`` is the last known real grid position (recursion start); if
    None it is inferred as ``_position(min(time_hours)) - 1``.
    """
    th = np.asarray(time_hours, dtype=float)
    point = np.asarray(point, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if anchor_pos is None:
        # fallback: assume recursion starts 1 step before the earliest request
        anchor_pos = _position(float(th.min())) - 1
    anchor_pos = int(anchor_pos)

    depths = np.maximum(1, (np.floor(th / SAMPLING_HOURS).astype(int) - anchor_pos))
    cal_half = np.array([calibration.half_width_at(int(d)) for d in depths])

    # Never shrink the raw interval; expand it to at least cover an honest
    # symmetric band around the point estimate (recursion drifts, so the band
    # belongs around the point, not the raw q05/q95 midpoint).
    lower = np.minimum(lower, point - cal_half)
    upper = np.maximum(upper, point + cal_half)

    return lower, upper


def diagnose_station(
    cfg: dict,
    models: dict,
    station_df: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    coverage_floor: float = 0.75,
    horizon_days: float = 90.0,
) -> dict:
    """Classify how much trust a station's multi-step forecast deserves.

    Runs the recursive test forecast once, measures (a) the calibrated
    multi-step coverage, (b) the error magnitude early in the recursion, and (c)
    the GWL dynamic range.  From these it returns an honest label:

    * ``weak``  — data quality problem (nearly zero GWL range, or the shallow
      recursive error is large relative to the range; e.g. a stuck/noisy
      sensor).  Its numbers are not trustworthy; show it as data-quality issue.
    * ``directional`` — recursion drifts too much to pin exact levels far out:
      calibrated coverage stays below ``coverage_floor`` even after honest
      widening, or the honest interval at the ``horizon_days`` scale is as wide
      as the whole observed GWL range.  Show it as direction-only, not absolute
      levels.
    * ``reliable`` — held-out calibrated coverage is at/above the floor and the
      near-horizon interval stays within the observed GWL range.
    """
    y = station_df[GWL_COL].to_numpy(dtype=float)
    t = pd.to_datetime(station_df[TIME_COL].to_numpy())
    hours = station_df["time_hours"].to_numpy()
    idx_by_ts = {ts: i for i, ts in enumerate(t)}

    _, _, _, y_test, _, t_test = split_series(
        station_df, use_seasonal=cfg["use_seasonal"], use_lags=cfg["use_lags"]
    )  # noqa: F401
    max_train_t = t_test[0] - pd.Timedelta(hours=SAMPLING_HOURS)

    buffer: dict = {}
    for i, (rt, ry) in enumerate(zip(t, y)):
        if rt <= max_train_t and not np.isnan(ry):
            buffer[_position(hours[i])] = float(ry)
    anchor_pos = max(buffer)

    test_hours = [hours[idx_by_ts[ts]] for ts in t_test]
    res = _predict_for_times(cfg, buffer, test_hours, models)
    actual = np.asarray(y_test, dtype=float)
    err = np.abs(np.asarray(res["point"]) - actual)

    cal = estimate_calibration(cfg, models, station_df, alpha=alpha)
    lo, hi = widen(cal, test_hours, res["point"], res["lower"], res["upper"], anchor_pos)
    coverage = float(np.mean((actual >= lo) & (actual <= hi)))
    shallow = err[0 : max(1, min(24, len(err)))] if len(err) else err
    shallowerr = float(np.median(shallow)) if len(shallow) else 0.0

    good = ~np.isnan(y)
    span = float(np.nanmax(y[good]) - np.nanmin(y[good])) if good.sum() else 0.0
    n = int(good.sum())
    # Honest interval width at the display horizon (4 six-hour steps/day).
    horizon_depth = max(1, int(round(horizon_days * 24 / SAMPLING_HOURS)))
    hw_at_horizon = float(cal.half_width_at(horizon_depth))

    label = "reliable"
    reason = "held-out calibrated coverage at/above floor"
    if span < 1e-6 or n < 30:
        label, reason = "weak", "insufficient/constant data"
    elif span > 1e-6 and shallowerr > 0.35 * span:
        label, reason = "weak", "data-quality: error dominates GWL range"
    elif coverage < coverage_floor:
        label, reason = "directional", f"coverage {coverage:.0%} below {coverage_floor:.0%} floor"
    elif span > 1e-6 and hw_at_horizon > 0.5 * span:
        label, reason = "directional", (
            f"{horizon_days:.0f}-day interval width {hw_at_horizon:.2f}m exceeds half "
            f"the observed range; direction-only beyond mid-range")

    return {
        "label": label,
        "reason": reason,
        "coverage": coverage,
        "shallow_error": shallowerr,
        "gwl_span": span,
        "n_obs": n,
        "tail_half_width": float(cal.half_widths[-1]) if cal.max_depth else 0.0,
        "half_width_at_horizon": hw_at_horizon,
        "horizon_days": horizon_days,
        "max_depth": cal.max_depth,
        "calibration": cal,
    }


def calibrate_and_widen(
    station: str,
    time_hours,
    artifacts_root: Path | str | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """One-stop: load models, prepare buffer, forecast, and widen the interval.

    Returns a dict identical in shape to ``predict_xgb_quantile`` but with the
    calibrated (wider, honest) ``lower``/``upper``.
    """
    from ml.models.xgboost_quantile import ARTIFACTS_DIR, predict_xgb_quantile

    root = Path(artifacts_root) if artifacts_root else ARTIFACTS_DIR
    out_dir = _station_artifact_dir(station, root)
    cfg = _load_config(out_dir)
    models = _load_models(out_dir)
    station_df = get_station_series(DEFAULT_PARQUET, cfg.get("station", station))
    buffer = _observed_buffer(station_df)
    if not buffer:
        raise ValueError(f"No valid observations for station {station!r}")

    cal = estimate_calibration(cfg, models, station_df, alpha=alpha)
    anchor_pos = max(buffer)  # BEFORE predict mutates the buffer
    res = _predict_for_times(cfg, buffer, time_hours, models)
    lo, hi = widen(cal, time_hours, res["point"], res["lower"], res["upper"], anchor_pos)
    return {
        "time_hours": np.asarray(time_hours, dtype=float),
        "point": np.asarray(res["point"]),
        "lower": lo,
        "upper": hi,
        "calibration": cal,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args()

    from ml.models.xgboost_quantile import ARTIFACTS_DIR

    root = ARTIFACTS_DIR
    out_dir = _station_artifact_dir(args.station, root)
    cfg = _load_config(out_dir)
    models = _load_models(out_dir)
    sdf = get_station_series(DEFAULT_PARQUET, cfg.get("station", args.station))
    cal = estimate_calibration(cfg, models, sdf, alpha=args.alpha)
    print(cal.report(args.station))
