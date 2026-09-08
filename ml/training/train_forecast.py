"""Train forecast models for a single station."""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_PARAMS,
    METADATA_FILE,
    train_models_for_station,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    full_pipeline,
)
from ml.services.interval_calibration import (  # noqa: E402
    diagnose_station,
    estimate_calibration,
)
from ml.services.interpretability import (  # noqa: E402
    permutation_importance_report,
    vif_report,
)
from ml.preprocessing.timeseries import prepare_feature_matrix  # noqa: E402


def build_interpretability(models, train_df, test_df, feature_cols):
    """Compute and return the interpretive diagnostics for a trained station.

    Uses the trained recursive point model. Returns a serialisable dict or None
    when the required models/data are unavailable. Additive only — never affects
    the forecasts themselves.
    """
    try:
        pmod = models["recursive"].get("point")
        if pmod is None or not feature_cols:
            return None
        Xtr, _, _ = prepare_feature_matrix(train_df[feature_cols + ["Groundwater Level Telemetry 6 Hourly (meter)"]])
        Xte, yte, _ = prepare_feature_matrix(test_df[feature_cols + ["Groundwater Level Telemetry 6 Hourly (meter)"]])
        if Xtr.shape[0] == 0 or len(feature_cols) == 0 or Xtr.shape[0] < 2:
            return None
        perm = permutation_importance_report(pmod, Xte, yte, feature_cols,
                                             n_repeats=5, seed=42)
        vif = vif_report(Xtr, feature_cols)
        return {"permutation_importance": perm, "vif": vif}
    except Exception as e:  # noqa: BLE001 - diagnostics must not break training
        return {"error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forecast models for a single station")
    parser.add_argument("--station", required=True, help="Station slug (e.g., 'Jairampur_PMS_CGWB_1')")
    parser.add_argument("--no-seasonal", action="store_true", help="Disable seasonal features")
    parser.add_argument("--no-lags", action="store_true", help="Disable lag features")
    parser.add_argument("--no-exogenous", action="store_true", help="Disable exogenous features")
    parser.add_argument("--no-direct", action="store_true", help="Disable direct models (1-30d)")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive model (31-90d)")
    parser.add_argument("--no-error-correction", action="store_true", help="Disable error-correction head")
    parser.add_argument("--artifacts", type=Path, default=None, help="Artifacts root directory")
    parser.add_argument("--parquet", type=Path, default=None, help="Path to common.parquet")
    parser.add_argument("--backend", type=Path, default=None, help="Path to back-end data.csv")
    args = parser.parse_args()

    artifacts_root = args.artifacts or ARTIFACTS_DIR
    parquet_path = args.parquet or _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet"
    backend_csv = args.backend or _ML_ROOT.parent / "back-end" / "db" / "data.csv"

    print(f"Training station: {args.station}")
    print(f"Artifacts: {artifacts_root}")
    print(f"Parquet: {parquet_path}")
    print(f"Backend: {backend_csv}")

    pipe = full_pipeline(parquet_path, backend_csv, station_slug_filter=args.station)
    train_df = pipe["train"]
    feature_cols = pipe["feature_cols"]

    if args.no_seasonal:
        feature_cols = [c for c in feature_cols if not c.startswith(("sin_", "cos_", "year"))]
    if args.no_lags:
        feature_cols = [c for c in feature_cols if not c.startswith("lag_")]
    if args.no_exogenous:
        feature_cols = [c for c in feature_cols if c.startswith(("district_"))]

    print(f"Using {len(feature_cols)} features")
    print(f"Train samples: {len(train_df)}")

    artifact_dir = artifacts_root / args.station
    artifact_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    models = train_models_for_station(
        train_df,
        feature_cols,
        args.station,
        artifact_dir,
        use_direct=not args.no_direct,
        use_recursive=not args.no_recursive,
        use_error_correction=not args.no_error_correction,
    )
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    cfg = {}
    calibration = estimate_calibration(cfg, models, train_df, feature_cols)
    diag = diagnose_station(cfg, models, train_df, feature_cols, test_df=pipe["test"])

    # ---- Interpretability (paper's feature-importance / VIF suite) ---------
    interpretability = build_interpretability(models, train_df, pipe["test"], feature_cols)

    metadata = {
        "station": train_df["Station"].iloc[0],
        "slug": args.station,
        "n_train": int(len(train_df)),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "direct_horizons": list(models["direct"].keys()),
        "has_recursive": "point" in models["recursive"],
        "has_error_correction": models["error_correction"] is not None,
        "one_step_rmse": diag["one_step_rmse"],
        "one_step_mae": diag["one_step_mae"],
        "one_step_r2": diag["one_step_r2"],
        "linear_one_step_rmse": diag.get("linear_one_step_rmse", 0.0),
        "linear_one_step_mae": diag.get("linear_one_step_mae", 0.0),
        "linear_one_step_r2": diag.get("linear_one_step_r2", 0.0),
        "multi_step_rmse": diag["multi_step_rmse"],
        "multi_step_r2": diag["multi_step_r2"],
        "calibrated_coverage": diag["coverage"],
        "reliability_label": diag["label"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_sec": elapsed,
        "params": DEFAULT_PARAMS,
    }

    with open(artifact_dir / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    if interpretability is not None:
        with open(artifact_dir / "interpretability.json", "w") as f:
            json.dump(interpretability, f, indent=2, default=float)

    print(f"Metadata saved to {artifact_dir / METADATA_FILE}")
    print(f"Label: {diag['label'].upper()} (coverage={diag['coverage']:.3f})")


if __name__ == "__main__":
    import json
    main()