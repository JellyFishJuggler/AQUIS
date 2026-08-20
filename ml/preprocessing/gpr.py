"""
Reusable GPR preprocessing functions.
Extracted from notebooks/gpr_preprocessing.ipynb for modular use.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import re

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

GWL_COL = "Groundwater Level Telemetry 6 Hourly (meter)"
TIME_COL = "Data Acquisition Time"

def station_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return slug.strip("_")


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_station_summary(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Station").size().sort_values(ascending=False)


def build_time_index(station_df: pd.DataFrame) -> pd.DataFrame:
    df = station_df.sort_values(TIME_COL).reset_index(drop=True)
    t0 = df[TIME_COL].min()
    df["time_hours"] = (df[TIME_COL] - t0).dt.total_seconds() / 3600
    return df


def detect_gaps(station_df: pd.DataFrame, threshold_hours: float = 9.0) -> pd.DataFrame:
    diffs = station_df[TIME_COL].diff()
    gap_mask = diffs.dt.total_seconds() / 3600 > threshold_hours
    gaps = station_df.loc[gap_mask].copy()
    gaps["gap_hours"] = diffs[gap_mask].dt.total_seconds() / 3600
    return gaps


def prepare_features(station_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = station_df[["time_hours"]].values
    y = station_df[GWL_COL].values
    valid = ~np.isnan(y)
    return X[valid], y[valid]


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    station_df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> dict:
    split_idx = int(len(X) * train_ratio)
    return {
        "X_train": X[:split_idx],
        "X_test": X[split_idx:],
        "y_train": y[:split_idx],
        "y_test": y[split_idx:],
        "split_time": station_df[TIME_COL].values[split_idx],
    }


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_sc = x_scaler.fit_transform(X_train)
    X_test_sc = x_scaler.transform(X_test)
    y_train_sc = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_sc = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return {
        "X_train_scaled": X_train_sc,
        "X_test_scaled": X_test_sc,
        "y_train_scaled": y_train_sc,
        "y_test_scaled": y_test_sc,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def save_scalers(
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    path: Path | None = None,
) -> None:
    path = path or ARTIFACTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, path / "x_scaler.pkl")
    joblib.dump(y_scaler, path / "y_scaler.pkl")


def load_scalers(
    path: Path | None = None,
    station: str | None = None,
) -> tuple[StandardScaler, StandardScaler]:
    if path is None:
        path = ARTIFACTS_DIR / (station_slug(station) if station else "")
    return (
        joblib.load(path / "x_scaler.pkl"),
        joblib.load(path / "y_scaler.pkl"),
    )


def full_pipeline(parquet_path: str | Path, station: str | None = None) -> dict:
    df = load_parquet(parquet_path)
    counts = get_station_summary(df)
    station = station or counts.index[0]
    sdf = df[df["Station"] == station].copy()
    sdf = build_time_index(sdf)
    X, y = prepare_features(sdf)
    split = time_split(X, y, sdf)
    scaled = scale_features(
        split["X_train"], split["X_test"], split["y_train"], split["y_test"]
    )
    slug = station_slug(station)
    save_scalers(scaled["x_scaler"], scaled["y_scaler"], ARTIFACTS_DIR / slug)
    return {**split, **scaled, "station": station, "station_df": sdf}
