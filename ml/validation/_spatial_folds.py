"""Spatial block assignment + leave-one-block-out folds (port of ml/preprocessing/spatial_cv.py).

Assigned stations by coordinate k-means into ``n_blocks`` geographic clusters;
each fold holds out one block so no training station shares the region of a test
station (spatial leakage guard).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "meta"
OUT = ROOT / "outputs"

STATION_CSV = META / "selected_gwl_stations.csv"


def load_station_coords() -> pd.DataFrame:
    meta = pd.read_csv(STATION_CSV, usecols=["Station", "Latitude", "Longitude"])
    meta = meta.dropna(subset=["Latitude", "Longitude"])
    return meta


def spatial_block_labels(coords: np.ndarray, n_blocks: int = 5,
                         random_state: int = 42) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < n_blocks:
        raise ValueError("not enough stations for %d spatial blocks" % n_blocks)
    km = KMeans(n_clusters=n_blocks, n_init=10, random_state=random_state)
    return km.fit_predict(coords[:, :2])


def spatial_folds(coords: np.ndarray, n_blocks: int = 5,
                  random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = spatial_block_labels(coords, n_blocks=n_blocks, random_state=random_state)
    folds = []
    for b in range(int(labels.max()) + 1):
        test_idx = np.where(labels == b)[0]
        train_idx = np.where(labels != b)[0]
        if len(test_idx) and len(train_idx):
            folds.append((train_idx, test_idx))
    return folds, labels


def write_fold_assignment(df: pd.DataFrame, labels: np.ndarray,
                          out: Path = OUT / "spatial_folds.csv") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = df[["Station"]].copy()
    frame["block"] = labels
    frame.to_csv(out, index=False)
    return out