"""
Model comparison: XGBoost-quantile vs Random Forest.

XGBoost models and metrics are reused from ``artifacts/<slug>/`` when
already trained (fast), otherwise trained on the spot.  The Random Forest
runs on the identical feature matrix / chronological split for a fair
apples-to-apples comparison.

Usage:
    python -m ml.training.compare_models
    python -m ml.training.compare_models --stations "Asafpur (UP-077)"
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from ml.models.xgboost_quantile import (  # noqa: E402
    DEFAULT_PARQUET,
    get_station_series,
    split_series,
    train_xgb_quantile_for_station,
)
from ml.preprocessing.timeseries import station_slug  # noqa: E402
from ml.utils import format_duration  # noqa: E402

ARTIFACTS = _ML_ROOT / "artifacts"

DEFAULT_STATIONS = [
    "Asafpur (UP-077)",
    "Amgaon (UP-067)",
    "Agra City (UP-017)",
]


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _load_json(path: Path) -> dict | None:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return None


def _xgb_metrics(station: str) -> dict:
    slug = station_slug(station)
    meta = _load_json(ARTIFACTS / slug / "xgboost_metadata.json")
    if meta is not None:
        return meta["point_metrics"]
    result = train_xgb_quantile_for_station(station, verbose=False)
    return result["point_metrics"]


def _rf_metrics(station: str) -> dict:
    """RandomForest on the identical XGBoost feature matrix (same split)."""
    station_df = get_station_series(DEFAULT_PARQUET, station)
    X_train, X_test, y_train, y_test, *_ = split_series(station_df)
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return _metrics(y_test, rf.predict(X_test))


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost vs RF comparison")
    parser.add_argument(
        "--stations", nargs="+", default=DEFAULT_STATIONS,
        help="Station names (default: Asafpur, Amgaon, Agra City)",
    )
    args = parser.parse_args()

    rows = []
    start_all = time.monotonic()
    for station in args.stations:
        print(f"\n=== {station} ===")

        t0 = time.monotonic()
        xgb = _xgb_metrics(station)
        print(f"  XGBoost  RMSE={xgb['rmse']:.4f}  MAE={xgb['mae']:.4f}  R²={xgb['r2']:.4f}  ({format_duration(time.monotonic()-t0)})")

        t0 = time.monotonic()
        rf = _rf_metrics(station)
        print(f"  RF       RMSE={rf['rmse']:.4f}  MAE={rf['mae']:.4f}  R²={rf['r2']:.4f}  ({format_duration(time.monotonic()-t0)})")

        rows.append({"station": station, "model": "xgboost_quantile", **xgb})
        rows.append({"station": station, "model": "random_forest", **rf})

    table = pd.DataFrame(rows)
    out_csv = ARTIFACTS / "model_comparison.csv"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    print(table.to_string(index=False))
    print(f"\nSaved -> {out_csv}")
    print(f"Total time: {format_duration(time.monotonic() - start_all)}")


if __name__ == "__main__":
    main()