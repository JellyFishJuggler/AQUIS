"""XGBoost Quantile Regression models with hybrid direct/recursive multi-step forecasting."""

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

from ml.preprocessing.timeseries import (  # noqa: E402
    GWL_COL,
    TIME_COL,
    STATION_COL,
    prepare_feature_matrix,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
POINT_MODEL_FILE = "xgb_point.joblib"
QUANTILE_MODEL_FILES = {
    0.05: "xgb_q05.joblib",
    0.50: "xgb_q50.joblib",
    0.95: "xgb_q95.joblib",
}
FEATURES_FILE = "features.json"
METADATA_FILE = "xgboost_metadata.json"


def station_dirs(artifacts_root: Path | None = None) -> list[Path]:
    """Trained station directories (have a point model + metadata)."""
    root = Path(artifacts_root) if artifacts_root else ARTIFACTS_DIR
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "recursive" / POINT_MODEL_FILE).is_file() and (p / METADATA_FILE).is_file()
    )

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.06,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

DIRECT_HORIZONS = list(range(1, 8)) + ["8_14", "15_21", "22_30"]
RECURSIVE_HORIZON_START = 31
MAX_HORIZON = 90


def _make_point_model(**kwargs) -> XGBRegressor:
    params = {**DEFAULT_PARAMS, **kwargs, "objective": "reg:squarederror"}
    return XGBRegressor(**params)


def _make_quantile_model(alpha: float, **kwargs) -> XGBRegressor:
    params = {**DEFAULT_PARAMS, **kwargs, "objective": "reg:quantileerror", "quantile_alpha": alpha}
    return XGBRegressor(**params)


def train_models_for_station(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    station_slug: str,
    artifact_dir: Path,
    use_direct: bool = True,
    use_recursive: bool = True,
    use_error_correction: bool = True,
) -> dict[str, Any]:
    """Train all models for a station: direct (1-30d), recursive (31-90d), error-correction."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_data = train_df[feature_cols + [GWL_COL]].copy()
    valid_mask = train_data[feature_cols].notna().all(axis=1) & train_data[GWL_COL].notna()
    train_data = train_data[valid_mask]
    
    X_train, y_train, _ = prepare_feature_matrix(train_data)

    models = {"direct": {}, "recursive": {}, "error_correction": None}

    if use_direct:
        for h in DIRECT_HORIZONS:
            if isinstance(h, int):
                h_str = str(h)
                target_shift = h * 4
            else:
                h_str = h
                start, end = map(int, h.split("_"))
                target_shift = start * 4

            y_shifted = np.roll(y_train, -target_shift)
            y_shifted[-target_shift:] = np.nan
            valid = ~np.isnan(y_shifted)

            if valid.sum() < 50:
                continue

            X_h = X_train[valid]
            y_h = y_shifted[valid]

            point_model = _make_point_model()
            point_model.fit(X_h, y_h)
            models["direct"][h_str] = {"point": point_model}

            for alpha in [0.05, 0.50, 0.95]:
                q_model = _make_quantile_model(alpha)
                q_model.fit(X_h, y_h)
                models["direct"][h_str][f"q{int(alpha*100):02d}"] = q_model

    if use_recursive:
        point_model = _make_point_model()
        point_model.fit(X_train, y_train)
        models["recursive"]["point"] = point_model

        for alpha in [0.05, 0.50, 0.95]:
            q_model = _make_quantile_model(alpha)
            q_model.fit(X_train, y_train)
            models["recursive"][f"q{int(alpha*100):02d}"] = q_model

    if use_error_correction and use_recursive:
        models["error_correction"] = train_error_correction_head(train_df, feature_cols, models["recursive"]["point"])

    save_models(models, artifact_dir, station_slug)
    return models


def train_error_correction_head(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    recursive_point_model: XGBRegressor,
    max_horizon: int = MAX_HORIZON,
) -> Any:
    """Train a lightweight model to correct recursive drift residuals vs horizon depth."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    station_residuals = []
    station_features = []

    for station, grp in train_df.groupby(STATION_COL):
        grp = grp.sort_values(TIME_COL)
        gwl = grp[GWL_COL].values
        feats = grp[feature_cols].values
        n = len(gwl)

        if n < 200:
            continue

        for start_idx in range(100, n - max_horizon, 50):
            last_known = feats[start_idx].copy()
            true_future = gwl[start_idx + 1:start_idx + max_horizon + 1]

            pred_future = []
            current_feats = last_known.copy()
            for step in range(max_horizon):
                pred = recursive_point_model.predict(current_feats.reshape(1, -1))[0]
                pred_future.append(pred)
                current_feats = update_features_recursive(current_feats, pred, feature_cols)

            pred_future = np.array(pred_future)
            residuals = true_future - pred_future
            depths = np.arange(1, len(residuals) + 1)

            for d, r in zip(depths, residuals, strict=False):
                correction_feats = np.concatenate([
                    [d / max_horizon],
                    [pred_future[min(d-1, len(pred_future)-1)]],
                    last_known[-10:],
                ])
                station_features.append(correction_feats)
                station_residuals.append(r)

    if len(station_residuals) < 100:
        return None

    X_corr = np.array(station_features)
    y_corr = np.array(station_residuals)

    corr_model = Ridge(alpha=1.0, random_state=42)
    corr_model.fit(X_corr, y_corr)

    return corr_model


def update_features_recursive(features: np.ndarray, new_pred: float, feature_cols: list[str]) -> np.ndarray:
    """Update feature vector for recursive step: shift lags, insert new prediction."""
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

    roll_cols = [c for c in feature_cols if c.startswith("roll_") and c.endswith("_mean")]
    for col in roll_cols:
        idx = feature_cols.index(col)
        window = int(col.split("_")[1])
        available_lags = [features[lag_indices[f"lag_{i}"]] for i in range(1, window + 1) if f"lag_{i}" in lag_indices]
        if available_lags:
            new_feats[idx] = np.mean(available_lags)

    return new_feats


def predict_direct(models: dict, X: np.ndarray, horizon: int | str) -> dict[str, float]:
    """Predict using direct model for a specific horizon."""
    h_str = str(horizon)
    if h_str not in models["direct"]:
        if isinstance(horizon, int) and horizon <= 7:
            h_str = str(horizon)
        elif isinstance(horizon, int) and 8 <= horizon <= 14:
            h_str = "8_14"
        elif isinstance(horizon, int) and 15 <= horizon <= 21:
            h_str = "15_21"
        elif isinstance(horizon, int) and 22 <= horizon <= 30:
            h_str = "22_30"
        else:
            raise ValueError(f"No direct model for horizon {horizon}")

    m = models["direct"][h_str]
    return {
        "point": float(m["point"].predict(X)[0]),
        "q05": float(m["q05"].predict(X)[0]),
        "q50": float(m["q50"].predict(X)[0]),
        "q95": float(m["q95"].predict(X)[0]),
    }


def predict_recursive(
    models: dict,
    last_features: np.ndarray,
    n_steps: int,
    feature_cols: list[str],
    error_correction_model: Any = None,
) -> dict[str, list[float]]:
    """Recursive multi-step forecast with optional error correction."""
    point_model = models["recursive"]["point"]
    q05_model = models["recursive"]["q05"]
    q50_model = models["recursive"]["q50"]
    q95_model = models["recursive"]["q95"]

    points = []
    q05s = []
    q50s = []
    q95s = []

    current_feats = last_features.copy()

    for step in range(n_steps):
        pt = point_model.predict(current_feats.reshape(1, -1))[0]
        q05 = q05_model.predict(current_feats.reshape(1, -1))[0]
        q50 = q50_model.predict(current_feats.reshape(1, -1))[0]
        q95 = q95_model.predict(current_feats.reshape(1, -1))[0]

        if error_correction_model is not None:
            corr_feats = np.concatenate([
                [(step + 1) / MAX_HORIZON],
                [pt],
                current_feats[-10:],
            ])
            correction = error_correction_model.predict(corr_feats.reshape(1, -1))[0]
            pt = pt + correction

        points.append(float(pt))
        q05s.append(float(q05))
        q50s.append(float(q50))
        q95s.append(float(q95))

        current_feats = update_features_recursive(current_feats, pt, feature_cols)

    return {"point": points, "q05": q05s, "q50": q50s, "q95": q95s}


def save_models(models: dict, artifact_dir: Path, station_slug: str) -> None:
    """Save all models and metadata to artifact directory."""
    direct_dir = artifact_dir / "direct"
    recursive_dir = artifact_dir / "recursive"
    direct_dir.mkdir(parents=True, exist_ok=True)
    recursive_dir.mkdir(parents=True, exist_ok=True)

    for h_str, m in models["direct"].items():
        joblib.dump(m["point"], direct_dir / f"h{h_str}_point.joblib")
        for alpha in [0.05, 0.50, 0.95]:
            joblib.dump(m[f"q{int(alpha*100):02d}"], direct_dir / f"h{h_str}_q{int(alpha*100):02d}.joblib")

    joblib.dump(models["recursive"]["point"], recursive_dir / "xgb_point.joblib")
    for alpha in [0.05, 0.50, 0.95]:
        joblib.dump(models["recursive"][f"q{int(alpha*100):02d}"], recursive_dir / f"xgb_q{int(alpha*100):02d}.joblib")

    if models["error_correction"] is not None:
        joblib.dump(models["error_correction"], recursive_dir / "error_correction_head.joblib")

    feature_cols = [c for c in models.get("feature_cols", []) if not c.startswith("lag_") and not c.startswith("roll_") and c not in ["trend_28", "trend_60"]]
    with open(artifact_dir / FEATURES_FILE, "w") as f:
        json.dump({"feature_cols": feature_cols, "direct_horizons": list(models["direct"].keys())}, f)


def load_models(artifact_dir: Path) -> dict[str, Any]:
    """Load all models for a station."""
    direct_dir = artifact_dir / "direct"
    recursive_dir = artifact_dir / "recursive"

    models = {"direct": {}, "recursive": {}, "error_correction": None}

    if direct_dir.exists():
        for point_file in direct_dir.glob("h*_point.joblib"):
            h_str = point_file.stem.replace("h", "").replace("_point", "")
            models["direct"][h_str] = {"point": joblib.load(point_file)}
            for alpha in [0.05, 0.50, 0.95]:
                q_file = direct_dir / f"h{h_str}_q{int(alpha*100):02d}.joblib"
                if q_file.exists():
                    models["direct"][h_str][f"q{int(alpha*100):02d}"] = joblib.load(q_file)

    if (recursive_dir / "xgb_point.joblib").exists():
        models["recursive"]["point"] = joblib.load(recursive_dir / "xgb_point.joblib")
        for alpha in [0.05, 0.50, 0.95]:
            q_file = recursive_dir / f"xgb_q{int(alpha*100):02d}.joblib"
            if q_file.exists():
                models["recursive"][f"q{int(alpha*100):02d}"] = joblib.load(q_file)

    ec_file = recursive_dir / "error_correction_head.joblib"
    if ec_file.exists():
        models["error_correction"] = joblib.load(ec_file)

    return models


def _load_config(artifact_dir: Path) -> dict:
    with open(artifact_dir / METADATA_FILE) as f:
        return json.load(f)


def get_station_series(parquet_path: Path | str, station_display_name: str) -> pd.DataFrame:
    """Get full series for a station by display name."""
    from ml.preprocessing.timeseries import load_and_clean, station_slug
    df = load_and_clean(parquet_path)
    df["slug"] = df.apply(lambda r: station_slug(r[STATION_COL], r["Agency"], r["SlNo"]), axis=1)
    return df[df[STATION_COL] == station_display_name].sort_values(TIME_COL).reset_index(drop=True)


def get_default_paths() -> tuple[Path, Path]:
    """Get default parquet and backend paths."""
    root = Path(__file__).resolve().parent.parent.parent
    return root / "ml" / "data" / "processed" / "common.parquet", root / "ml" / "back-end" / "db" / "data.csv"


def get_test_predictions(station_slug: str, parquet_path: Path | str = None) -> dict[str, np.ndarray]:
    """One-step predictions on test set for dashboard snapshot."""
    from ml.preprocessing.timeseries import full_pipeline
    from ml.models.xgboost_quantile import load_models

    if parquet_path is None:
        parquet_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "common.parquet"

    pipe = full_pipeline(parquet_path, station_slug_filter=station_slug)
    test_df = pipe["test"]
    feature_cols = pipe["feature_cols"]

    artifact_dir = ARTIFACTS_DIR / station_slug
    models = load_models(artifact_dir)

    X_test, y_test, _ = prepare_feature_matrix(test_df[feature_cols + [GWL_COL]])

    point_model = models["recursive"]["point"]
    q05_model = models["recursive"]["q05"]
    q50_model = models["recursive"]["q50"]
    q95_model = models["recursive"]["q95"]

    return {
        "time": test_df.loc[~test_df[feature_cols].isna().any(axis=1), TIME_COL].values,
        "actual": y_test,
        "point": point_model.predict(X_test),
        "lower": q05_model.predict(X_test),
        "median": q50_model.predict(X_test),
        "upper": q95_model.predict(X_test),
    }