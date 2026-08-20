"""
Load a trained GPR model and make predictions.

All artifact paths resolve relative to *this file* via Path(__file__),
so the script works regardless of the working directory used to invoke it.

Usage:
    python -m ml.inference.predict_gpr --station Asafpur --time 1000 2000 3000
    python -m ml.inference.predict_gpr --time 1000 2000 3000  (uses first trained station)
"""

import sys
from pathlib import Path

import joblib
import numpy as np

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

ARTIFACTS = _ML_ROOT / "artifacts"


def _find_default_station() -> str | None:
    """Find the first station with trained artifacts."""
    if not ARTIFACTS.exists():
        return None
    for d in sorted(ARTIFACTS.iterdir()):
        if (d / "metadata.json").is_file():
            return d.name
    return None


def predict_gpr(
    time_hours,
    station: str | None = None,
    model_path: Path | None = None,
    x_scaler_path: Path | None = None,
    y_scaler_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict groundwater level + uncertainty for given time-hours input(s).

    Parameters
    ----------
    time_hours : float or array-like
        Hours elapsed since the station's first reading.
    station : str, optional
        Station slug (directory name under artifacts/). If not provided,
        uses the first found station or flat artifacts/ for backwards compat.
    model_path, x_scaler_path, y_scaler_path : Path, optional
        Override artifact locations.

    Returns
    -------
    predictions_meters : np.ndarray
    uncertainty_meters : np.ndarray
    """
    if model_path is None:
        if station:
            station_dir = ARTIFACTS / station
        else:
            default = _find_default_station()
            station_dir = ARTIFACTS / default if default else ARTIFACTS

        model_path = station_dir / "gpr_model.joblib"
        x_scaler_path = x_scaler_path or station_dir / "x_scaler.pkl"
        y_scaler_path = y_scaler_path or station_dir / "y_scaler.pkl"
    else:
        x_scaler_path = x_scaler_path or model_path.parent / "x_scaler.pkl"
        y_scaler_path = y_scaler_path or model_path.parent / "y_scaler.pkl"

    gpr = joblib.load(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    X = np.asarray(time_hours, dtype=float).reshape(-1, 1)
    X_sc = x_scaler.transform(X)
    y_pred_sc, std_sc = gpr.predict(X_sc, return_std=True)

    y_pred = y_scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    uncertainty = std_sc * y_scaler.scale_[0]

    return y_pred, uncertainty


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPR prediction")
    parser.add_argument("--station", type=str, default=None, help="Station slug")
    parser.add_argument(
        "--time",
        nargs="+",
        type=float,
        required=True,
        help="Time(s) in hours since first reading",
    )
    args = parser.parse_args()

    preds, unc = predict_gpr(args.time, station=args.station)
    for t, p, u in zip(args.time, preds, unc):
        print(f"t={t:.1f}h  ->  GWL={p:.4f}m +/- {1.96 * u:.4f}m  (95% CI)")
