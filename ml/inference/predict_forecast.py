"""Inference CLI: calibrated point + 90% PI for a station at given horizon(s)."""

import argparse
import sys
from pathlib import Path

import numpy as np

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    DIRECT_HORIZONS,
    MAX_HORIZON,
    load_models,
    predict_direct,
    predict_recursive,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    full_pipeline,
    prepare_feature_matrix,
)
from ml.services.interval_calibration import (  # noqa: E402
    IntervalCalibration,
    calibrate_and_widen,
    estimate_calibration,
    widen,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict groundwater level with calibrated 90% PI")
    parser.add_argument("--station", required=True, help="Station slug")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 7, 14, 30, 60, 90], help="Horizon(s) in days")
    parser.add_argument("--artifacts", type=Path, default=None, help="Artifacts root")
    parser.add_argument("--parquet", type=Path, default=None, help="Path to common.parquet")
    parser.add_argument("--backend", type=Path, default=None, help="Path to back-end data.csv")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    artifacts_root = args.artifacts or ARTIFACTS_DIR
    parquet_path = args.parquet or _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet"
    backend_csv = args.backend or _ML_ROOT.parent / "back-end" / "db" / "data.csv"

    pipe = full_pipeline(parquet_path, backend_csv, station_slug_filter=args.station)
    train_df = pipe["train"]
    feature_cols = pipe["feature_cols"]

    artifact_dir = artifacts_root / args.station
    if not artifact_dir.exists():
        for d in artifact_dir.parent.iterdir():
            if d.is_dir() and args.station in d.name:
                artifact_dir = d
                break

    models = load_models(artifact_dir)

    cfg = {}
    calibration = estimate_calibration(cfg, models, train_df, feature_cols)

    X_last, _, _ = prepare_feature_matrix(train_df[feature_cols + ["Groundwater Level Telemetry 6 Hourly (meter)"]].tail(1))
    if len(X_last) == 0:
        print("ERROR: No valid features for prediction", file=sys.stderr)
        sys.exit(1)

    last_feats = X_last[0]
    last_gwl = train_df["Groundwater Level Telemetry 6 Hourly (meter)"].iloc[-1]
    last_date = train_df["Data Acquisition Time"].iloc[-1]

    results = []
    for h in args.horizon:
        if h <= 30 and h in DIRECT_HORIZONS:
            pred = predict_direct(models, last_feats.reshape(1, -1), h)
            point = pred["point"]
            lower = pred["q05"]
            upper = pred["q95"]
            cal_half = calibration.half_width_at_horizon(h)
            cal_lower = point - cal_half
            cal_upper = point + cal_half
            final_lower = min(lower, cal_lower)
            final_upper = max(upper, cal_upper)
            model_type = "direct"
        else:
            n_steps = h * 4
            rec = predict_recursive(models, last_feats, n_steps, feature_cols, models.get("error_correction"))
            point = rec["point"][-1]
            lower = rec["q05"][-1]
            upper = rec["q95"][-1]
            cal_half = calibration.half_width_at(n_steps)
            cal_lower = point - cal_half
            cal_upper = point + cal_half
            final_lower = min(lower, cal_lower)
            final_upper = max(upper, cal_upper)
            model_type = "recursive"

        pred_date = last_date + pd.Timedelta(days=h)
        results.append({
            "station": args.station,
            "horizon_days": h,
            "prediction_date": str(pred_date),
            "point": round(float(point), 4),
            "lower_90": round(float(final_lower), 4),
            "upper_90": round(float(final_upper), 4),
            "raw_lower_90": round(float(lower), 4),
            "raw_upper_90": round(float(upper), 4),
            "calibrated_half_width": round(float(cal_half), 4),
            "model_type": model_type,
            "last_observed_gwl": round(float(last_gwl), 4),
            "last_observed_date": str(last_date),
        })

    import json
    output = {
        "station": args.station,
        "predictions": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    import pandas as pd
    main()