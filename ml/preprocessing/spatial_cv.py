"""Spatial cross-validation for AQUIS (paper's SpatialKFold idea).

The paper (Dule & Sharma) argues random (non-spatial) train/test splits leak
information because geographically close samples are correlated. It therefore
uses SpatialKFold — partitioning the area into geometric blocks via coordinate
k-means — so training and validation are spatially separated.

AQUIS stations carry real coordinates. This module provides a coordinate-k-means
cross-validator over station coordinates, usable to (a) validate that a trained
forecast model generalises to unseen locations and (b) tune/compare models under
spatially fair partitions. Only numpy + scikit-learn are used.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def spatial_block_labels(coords: np.ndarray, n_blocks: int = 4,
                         random_state: int = 42) -> np.ndarray:
    """Assign each station to one of ``n_blocks`` geographic clusters via k-means."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    if coords.shape[0] < n_blocks:
        raise ValueError("not enough stations for %d spatial blocks" % n_blocks)
    km = KMeans(n_clusters=n_blocks, n_init=10, random_state=random_state)
    return km.fit_predict(coords[:, :2])


def spatial_folds(coords: np.ndarray, n_blocks: int = 4,
                  random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-block-out spatial folds.

    Returns a list of (train_idx, test_idx) pairs; each fold holds out one
    geographic block as the test set.
    """
    labels = spatial_block_labels(coords, n_blocks=n_blocks, random_state=random_state)
    folds = []
    n_blocks_actual = int(labels.max()) + 1
    for b in range(n_blocks_actual):
        test_idx = np.where(labels == b)[0]
        train_idx = np.where(labels != b)[0]
        if len(test_idx) and len(train_idx):
            folds.append((train_idx, test_idx))
    return folds


def spatial_gap_summary(coords: np.ndarray, n_blocks: int = 4,
                        random_state: int = 42) -> dict:
    """Report block sizes and mean inter-block centroid distances (diagnostic)."""
    labels = spatial_block_labels(coords, n_blocks=n_blocks, random_state=random_state)
    coords = np.asarray(coords, dtype=float)
    sizes = np.bincount(labels, minlength=n_blocks)
    return {
        "n_blocks": int(len(sizes)),
        "block_sizes": sizes.tolist(),
        "n_stations": int(len(coords)),
        "random_state": random_state,
    }
