"""
Inference for the XGBoost + quantile-regression forecast model.

Returns the point prediction plus the 90% prediction interval
(5th / 95th percentiles) in original GWL (meters) units.

Usage (from repo root):
    python -m ml.inference.predict_forecast --station asafpur_up_077 --time 1000 2000 3000
    python -m ml.inference.predict_forecast --time 1000 2000 3000   (uses first trained station)
"""

import sys
from pathlib import Path

import numpy as np

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    POINT_MODEL_FILE,
    predict_xgb_quantile,
)


def _find_default_station() -> str | None:
    root = _ML_ROOT / "artifacts"
    if not root.exists():
        return None
    for d in sorted(root.iterdir()):
        if (d / POINT_MODEL_FILE).is_file():
            return d.name
    return None


def predict_forecast(
    time_hours,
    station: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (point, lower_q05, upper_q95) predictions in meters."""
    station_name = station if station else _find_default_station()
    if station_name is None:
        raise FileNotFoundError(
            "No trained XGBoost station found. Train first: "
            "python -m ml.training.train_forecast --station '...'"
        )
    result = predict_xgb_quantile(time_hours, station_name)
    return (
        result["point"],
        result["lower"],
        result["upper"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost forecast prediction")
    parser.add_argument(
        "--station", type=str, default=None,
        help="Station slug or full name (default: first trained)",
    )
    parser.add_argument(
        "--time",
        nargs="+",
        type=float,
        required=True,
        help="Time(s) in hours since first reading",
    )
    args = parser.parse_args()

    point, lower, upper = predict_forecast(args.time, station=args.station)
    for t, p, lo, hi in zip(args.time, point, lower, upper):
        print(f"t={t:.1f}h  ->  GWL={p:.4f} m  90% CI = [{lo:.4f}, {hi:.4f}] m")