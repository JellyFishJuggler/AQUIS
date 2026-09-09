"""Spatial autocorrelation of 30-day forecast residuals (Moran's I / Geary's C).

Per-station mean XGBoost test residual vs station coordinates. Significant
positive Moran's I means error clusters spatially — insufficient spatial
structure in the model. Port of ml/services/interpretability.residual_autocorrelation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
META = ROOT / "data" / "meta"
PREDS = OUT / "predictions_2026.parquet"
RESID_JSON = OUT / "residual_spatial.json"

MIN_STATIONS = 10


def _knn_edges(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(coords)
    k = max(1, min(k, n - 1))
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    idx = np.atleast_2d(idx)[:, 1:]
    ei, ej = [], []
    for i in range(n):
        for j in idx[i]:
            j = int(j)
            if j != i:
                ei.append(i)
                ej.append(j)
    return np.asarray(ei, int), np.asarray(ej, int)


def residual_autocorrelation(residuals: np.ndarray, coords: np.ndarray,
                             k: int = 5, n_sim: int = 999, seed: int = 42) -> dict:
    residuals = np.asarray(residuals, float)
    coords = np.asarray(coords, float)
    ei, ej = _knn_edges(coords, k=k)

    def moran(z):
        dz = z - z.mean()
        num = np.dot(dz[ei], dz[ej]) / len(ei)
        den = (dz ** 2).mean()
        return float(num / den) if den else 1.0

    def geary(z):
        dz = z - z.mean()
        varz = (dz ** 2).mean()
        if varz == 0:
            return 1.0
        return float(((z[ei] - z[ej]) ** 2).sum() / (2 * len(ei)) / varz)

    moran_obs, geary_obs = moran(residuals), geary(residuals)
    rng = np.random.RandomState(seed)
    vals = residuals.copy()
    moran_p = geary_p = 0
    for _ in range(n_sim):
        rng.shuffle(vals)
        if abs(moran(vals)) >= abs(moran_obs):
            moran_p += 1
        if abs(geary(vals)) >= abs(geary_obs):
            geary_p += 1
    return {"moran_I": moran_obs, "moran_p": (1 + moran_p) / (1 + n_sim),
            "geary_C": geary_obs, "geary_p": (1 + geary_p) / (1 + n_sim),
            "k": k, "n_sim": n_sim}


def main() -> None:
    df = pd.read_parquet(PREDS)
    meta = pd.read_csv(META / "selected_gwl_stations.csv",
                       usecols=["Station", "Latitude", "Longitude"])
    resid = (df.groupby("Station")["err_xgb"].mean().rename("resid").reset_index())
    both = resid.merge(meta, on="Station", how="inner").dropna(
        subset=["resid", "Latitude", "Longitude"])

    if len(both) < MIN_STATIONS:
        print(f"too few stations ({len(both)}) for spatial test; skipping")
        return

    report = residual_autocorrelation(both["resid"].to_numpy(),
                                      both[["Longitude", "Latitude"]].to_numpy())
    report["n_stations"] = int(len(both))
    report["metric"] = "per-station mean XGBoost test residual (err_xgb)"
    RESID_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()