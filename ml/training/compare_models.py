"""Compare XGBoost (hybrid) vs Random Forest vs Persistence vs Seasonal Naive."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

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
    GWL_COL,
    TIME_COL,
    STATION_COL,
    full_pipeline,
    prepare_feature_matrix,
)
from ml.services.interval_calibration import (  # noqa: E402
    calibrate_and_widen,
    estimate_calibration,
    widen,
)


def persistence_forecast(last_value: float, n_steps: int) -> np.ndarray:
    """Flat-line persistence forecast."""
    return np.full(n_steps, last_value)


def seasonal_naive_forecast(series: np.ndarray, period: int, n_steps: int) -> np.ndarray:
    """Seasonal naive: repeat last observed seasonal cycle."""
    if len(series) < period:
        return np.full(n_steps, series[-1])
    last_cycle = series[-period:]
    reps = (n_steps + period - 1) // period
    return np.tile(last_cycle, reps)[:n_steps]


def train_rf_recursive(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """Train Random Forest for recursive forecasting."""
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R2."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else -1.0
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}


def compare_station(station_slug: str, feature_cols: list[str], train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict]:
    """Compare all models on a single station."""
    results = []

    X_train, y_train, _ = prepare_feature_matrix(train_df[feature_cols + [GWL_COL]])
    X_test, y_test, _ = prepare_feature_matrix(test_df[feature_cols + [GWL_COL]])

    artifact_dir = ARTIFACTS_DIR / station_slug
    if not artifact_dir.exists():
        return []

    xgb_models = load_models(artifact_dir)

    last_train_value = train_df[GWL_COL].iloc[-1]
    test_dates = test_df[TIME_COL].values

    for h in [1, 7, 14, 30, 60, 90]:
        target_shift = h * 4
        if target_shift >= len(y_test):
            continue

        y_shifted = np.roll(y_test, -target_shift)
        y_shifted[-target_shift:] = np.nan
        valid = ~np.isnan(y_shifted)
        if valid.sum() < 10:
            continue

        X_val = X_test[valid]
        y_val = y_shifted[valid]

        xgb_pred = predict_direct(xgb_models, X_val, h) if h <= 30 else None
        if xgb_pred:
            results.append(evaluate_model(f"XGB_direct_h{h}", y_val, np.full(len(y_val), xgb_pred["point"])))
            results[-1]["horizon"] = h
            results[-1]["type"] = "direct"

        if h > 30:
            pass

        rf_model = train_rf_recursive(X_train, y_train)
        rf_recursive = []
        current_feats = X_test[0].copy()
        for step in range(max(1, target_shift // 4)):
            pred = rf_model.predict(current_feats.reshape(1, -1))[0]
            rf_recursive.append(pred)
            current_feats = update_features_recursive(current_feats, pred, feature_cols)
        rf_final = rf_recursive[-1] if rf_recursive else last_train_value
        results.append(evaluate_model(f"RF_recursive_h{h}", y_val, np.full(len(y_val), rf_final)))
        results[-1]["horizon"] = h
        results[-1]["type"] = "recursive"

        pers = persistence_forecast(last_train_value, len(y_val))
        results.append(evaluate_model(f"Persistence_h{h}", y_val, pers))
        results[-1]["horizon"] = h
        results[-1]["type"] = "baseline"

        seasonal = seasonal_naive_forecast(train_df[GWL_COL].values, 365 * 4, len(y_val))
        results.append(evaluate_model(f"SeasonalNaive_h{h}", y_val, seasonal))
        results[-1]["horizon"] = h
        results[-1]["type"] = "baseline"

    return results


def update_features_recursive(features: np.ndarray, new_pred: float, feature_cols: list[str]) -> np.ndarray:
    new_feats = features.copy()
    lag_cols = [c for c in feature_cols if c.startswith("lag_")]
    lag_indices = {c: i for i, c in enumerate(feature_cols) if c.startswith("lag_")}
    sorted_lags = sorted(lag_indices.items(), key=lambda x: int(x[0].split("_")[1]))
    for i, (col, idx) in enumerate(sorted_lags):
        if i == 0:
            new_feats[idx] = new_pred
        else:
            prev_col, prev_idx = sorted_lags[i - 1]
            new_feats[idx] = features[prev_idx]
    return new_feats


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compare models on representative stations")
    parser.add_argument("--stations", nargs="+", default=None, help="Station slugs (default: 3 diverse)")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()

    if args.stations:
        station_slugs = args.stations
    else:
        station_slugs = [
            "390713- Jairampur PMS -Shallow_UPGW_dae67f3f",
        ]

    pipe = full_pipeline(
        _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet",
        _ML_ROOT.parent / "back-end" / "db" / "data.csv",
    )
    feature_cols = pipe["feature_cols"]

    all_results = []
    for slug in station_slugs:
        print(f"Comparing models for {slug}...")
        train_df = pipe["train"][pipe["train"]["slug"] == slug]
        test_df = pipe["test"][pipe["test"]["slug"] == slug]
        if train_df.empty or test_df.empty:
            print(f"  No data for {slug}, skipping")
            continue
        results = compare_station(slug, feature_cols, train_df, test_df)
        for r in results:
            r["station"] = slug
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    out_path = args.output or (_ML_ROOT / "artifacts" / "model_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()