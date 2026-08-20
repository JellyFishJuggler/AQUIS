"""
Train and evaluate the GPR model.

Run from the repo root:
    python -m ml.training.train_gpr
    python -m ml.training.train_gpr --station Asafpur
    python -m ml.training.train_gpr --station Jairampur
"""

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.gaussian_process import build_gpr
from ml.preprocessing.gpr import full_pipeline, station_slug
from ml.utils import format_duration

ARTIFACTS = _ML_ROOT / "artifacts"
PARQUET = _ML_ROOT / "data" / "processed" / "common.parquet"


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_for_station(
    station: str,
    parquet_path: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    n_restarts: int = 5,
    verbose: bool = True,
) -> dict:
    """Train a GPR model for a single station.

    Saves model, scalers, and metadata.json to
    ``artifacts/<station_slug>/``.

    Returns a dict with metrics, duration, and the artifact directory path.
    """
    parquet_path = Path(parquet_path) if parquet_path else PARQUET
    artifacts_root = Path(artifacts_root) if artifacts_root else ARTIFACTS
    slug = station_slug(station)
    out_dir = artifacts_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Preprocess
    data = full_pipeline(parquet_path, station=station)
    _log(f"[{datetime.now():%H:%M:%S}] Training station: {station}")
    _log(
        f"  Train: {len(data['X_train_scaled']):,}  |  "
        f"Test: {len(data['X_test_scaled']):,}"
    )

    # 2. Build model
    gpr = build_gpr(data["x_scaler"], n_restarts=n_restarts)

    # 3. Fit with heartbeat
    _log("Fitting Gaussian Process...")
    fit_error = [None]

    def _fit():
        try:
            gpr.fit(data["X_train_scaled"], data["y_train_scaled"])
        except Exception as e:
            fit_error[0] = e

    start_time = time.monotonic()
    thread = threading.Thread(target=_fit, daemon=True)
    thread.start()

    while thread.is_alive():
        thread.join(timeout=10)
        elapsed = time.monotonic() - start_time
        if thread.is_alive():
            _log(
                f"  [{datetime.now():%H:%M:%S}] Still fitting... "
                f"elapsed: {format_duration(elapsed)}"
            )

    if fit_error[0] is not None:
        raise fit_error[0]

    fit_elapsed = time.monotonic() - start_time
    _log(f"  Training completed in {format_duration(fit_elapsed)}")

    lml = float(gpr.log_marginal_likelihood_value_)
    _log(f"  Log-marginal-likelihood: {lml:.2f}")

    # 4. Predict + evaluate
    y_scaler = data["y_scaler"]

    y_train_pred_sc, _ = gpr.predict(data["X_train_scaled"], return_std=True)
    y_test_pred_sc, std_test_sc = gpr.predict(
        data["X_test_scaled"], return_std=True
    )

    y_train_pred = y_scaler.inverse_transform(
        y_train_pred_sc.reshape(-1, 1)
    ).ravel()
    y_test_pred = y_scaler.inverse_transform(
        y_test_pred_sc.reshape(-1, 1)
    ).ravel()

    train_metrics = evaluate(data["y_train"], y_train_pred)
    test_metrics = evaluate(data["y_test"], y_test_pred)

    _log(f"  Train RMSE: {train_metrics['rmse']:.4f}  MAE: {train_metrics['mae']:.4f}  R²: {train_metrics['r2']:.4f}")
    _log(f"  Test  RMSE: {test_metrics['rmse']:.4f}  MAE: {test_metrics['mae']:.4f}  R²: {test_metrics['r2']:.4f}")

    # 5. Save artifacts
    joblib.dump(gpr, out_dir / "gpr_model.joblib")
    joblib.dump(data["x_scaler"], out_dir / "x_scaler.pkl")
    joblib.dump(data["y_scaler"], out_dir / "y_scaler.pkl")

    metadata = {
        "station": station,
        "slug": slug,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(data["X_train_scaled"]),
        "test_size": len(data["X_test_scaled"]),
        "total_points": len(data["X_train_scaled"]) + len(data["X_test_scaled"]),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "log_marginal_likelihood": lml,
        "kernel": str(gpr.kernel_),
        "fit_duration": format_duration(fit_elapsed),
        "n_restarts": n_restarts,
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _log(f"  Saved -> {out_dir}")

    return {
        "metrics": {"train": train_metrics, "test": test_metrics},
        "duration": format_duration(fit_elapsed),
        "log_marginal_likelihood": lml,
        "artifact_dir": str(out_dir),
    }


def list_trained_stations(artifacts_root: Path | None = None) -> list[dict]:
    """Return metadata for all trained stations."""
    root = artifacts_root or ARTIFACTS
    stations = []
    if not root.exists():
        return stations
    for d in sorted(root.iterdir()):
        meta_path = d / "metadata.json"
        if meta_path.is_file():
            with open(meta_path) as f:
                stations.append(json.load(f))
    return stations


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPR for a station")
    parser.add_argument("--station", type=str, default=None, help="Station name")
    parser.add_argument(
        "--list", action="store_true", help="List trained stations and exit"
    )
    args = parser.parse_args()

    if args.list:
        for m in list_trained_stations():
            rmse = m["test_metrics"]["rmse"]
            r2 = m["test_metrics"]["r2"]
            print(f"  {m['station']:20s}  RMSE={rmse:.4f}  R²={r2:.4f}  ({m['trained_at']})")
        return

    station = args.station
    if not station:
        import pandas as pd

        df = pd.read_parquet(PARQUET)
        counts = df.groupby("Station").size().sort_values(ascending=False)
        station = counts.index[0]
        print(f"No station specified; using largest: {station}")

    result = train_for_station(station)
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
