"""
Train and evaluate the XGBoost + quantile-regression forecast model.

Usage (from repo root):
    python -m ml.training.train_forecast
    python -m ml.training.train_forecast --station "Asafpur (UP-077)"
    python -m ml.training.train_forecast --station "Asafpur (UP-077)" --no-seasonal
"""

import argparse
import json
import sys
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

import pandas as pd  # noqa: E402

from ml.models.xgboost_quantile import (  # noqa: E402
    DEFAULT_PARQUET,
    train_xgb_quantile_for_station,
)

MODEL_NAMES = {
    0.05: "xgb_q05.joblib",
    0.5: "xgb_q50.joblib",
    0.95: "xgb_q95.joblib",
}


def list_trained_stations(artifacts_root: Path | None = None) -> list[dict]:
    root = artifacts_root or (_ML_ROOT / "artifacts")
    stations = []
    if not root.exists():
        return stations
    for d in sorted(root.iterdir()):
        meta_path = d / "xgboost_metadata.json"
        if meta_path.is_file():
            with open(meta_path) as f:
                stations.append(json.load(f))
    return stations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost + quantile regression for a station"
    )
    parser.add_argument("--station", type=str, default=None, help="Station name")
    parser.add_argument(
        "--use-seasonal", dest="use_seasonal", action="store_true",
        default=True,
        help="Include sin/cos(day-of-year) + year features (default)",
    )
    parser.add_argument(
        "--no-seasonal", dest="use_seasonal", action="store_false",
        help="Use time_hours only",
    )
    parser.add_argument(
        "--use-lags", dest="use_lags", action="store_true", default=True,
        help="Include causal lag/rolling features (default)",
    )
    parser.add_argument(
        "--no-lags", dest="use_lags", action="store_false",
        help="Omit lag/rolling features (spec-minimal feature set)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List XGBoost-trained stations and exit"
    )
    args = parser.parse_args()

    if args.list:
        for m in list_trained_stations():
            pm = m["point_metrics"]
            qm = m["quantile_metrics"]
            print(
                f"  {m['station']:30s}  RMSE={pm['rmse']:.4f}  "
                f"R²={pm['r2']:.4f}  cover={qm['coverage_90']:.1%}  "
                f"({m['duration']})"
            )
        return

    station = args.station
    if not station:
        df = pd.read_parquet(DEFAULT_PARQUET)
        counts = df.groupby("Station").size().sort_values(ascending=False)
        station = counts.index[0]
        print(f"No station specified; using largest: {station}")

    result = train_xgb_quantile_for_station(
        station, use_seasonal=args.use_seasonal, use_lags=args.use_lags
    )
    print()
    summary = {
        "station": result["station"],
        "duration": result["duration"],
        "point_metrics": result["point_metrics"],
        "quantile_metrics": result["quantile_metrics"],
        "artifact_dir": result["artifact_dir"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()