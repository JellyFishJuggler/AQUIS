"""Model interpretability + sensitivity for AQUIS (paper's interpretability suite).

Adds to the single AQUIS system the paper's model-diagnostic techniques:

  * permutation feature importance (scikit-learn built-in),
  * VIF (Variance Inflation Factor) for multicollinearity,
  * residual global spatial autocorrelation (Moran's I / Geary's C) across the
    fleet, computed on one-step test residuals using each station's coordinates,
  * a first-order / total-order sensitivity ranking (Sobol-style, pure numpy).

All are additive diagnostics; none alter how forecasts are produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from scipy.spatial import cKDTree
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------
def permutation_importance_report(
    model, X: np.ndarray, y_test: np.ndarray, feature_cols: list[str],
    n_repeats: int = 10, seed: int = 42,
) -> list[dict]:
    """Per-feature permutation importance (mean/std), ranked descending."""
    X = np.asarray(X, float)
    res = permutation_importance(
        model, X, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )
    order = np.argsort(res.importances_mean)[::-1]
    rows = []
    for rank, i in enumerate(order):
        rows.append({
            "rank": rank + 1,
            "feature": feature_cols[i],
            "importance_mean": float(res.importances_mean[i]),
            "importance_std": float(res.importances_std[i]),
        })
    return rows


# ---------------------------------------------------------------------------
# VIF (collinearity)
# ---------------------------------------------------------------------------
def vif_report(X: np.ndarray, feature_cols: list[str]) -> dict:
    """Variance Inflation Factor per feature."""
    X = np.asarray(X, float)
    X = X - X.mean(axis=0)
    p = X.shape[1]
    corr = np.corrcoef(X.T) if p > 1 else np.array([[1.0]])
    corr = corr + np.eye(p) * 1e-9  # ridge for stability
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)
    vifs = np.diag(inv)
    flagged = [feature_cols[i] for i in range(p) if vifs[i] > 10]
    return {
        "feature": feature_cols,
        "vif": [float(v) for v in vifs],
        "highly_correlated_gt10": flagged,
    }


# ---------------------------------------------------------------------------
# Residual spatial autocorrelation (Moran's I / Geary's C), fleet-level
# ---------------------------------------------------------------------------
def _knn_edges(coords: np.ndarray, k: int = 5):
    n = coords.shape[0]
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


def residual_autocorrelation(
    residuals: np.ndarray, coords: np.ndarray, k: int = 5, n_sim: int = 999,
    seed: int = 42,
) -> dict:
    """Global Moran's I and Geary's C on residuals + permutation p-values."""
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

    moran_obs = moran(residuals)
    geary_obs = geary(residuals)
    rng = np.random.RandomState(seed)
    vals = residuals.copy()
    moran_p, geary_p = 0, 0
    for _ in range(n_sim):
        rng.shuffle(vals)
        if abs(moran(vals)) >= abs(moran_obs):
            moran_p += 1
        if abs(geary(vals)) >= abs(geary_obs):
            geary_p += 1
    return {
        "moran_I": moran_obs,
        "moran_p": (1 + moran_p) / (1 + n_sim),
        "geary_C": geary_obs,
        "geary_p": (1 + geary_p) / (1 + n_sim),
        "k": k, "n_sim": n_sim,
    }


# ---------------------------------------------------------------------------
# Sensitivity (Sobol-style, pure numpy)
# ---------------------------------------------------------------------------
def sensitivity_report(
    model, X: np.ndarray, feature_cols: list[str], n_base: int = 256, seed: int = 42,
    max_features: int = 8,
) -> dict:
    """First-/total-order sensitivity ranking via a Saltelli scheme.

    To stay tractable, only the top ``max_features`` by permutation/variance are
    used. Returns S1/ST per considered feature.
    """
    X = np.asarray(X, float)
    d_all = len(feature_cols)
    # Select the most influential features using the model's own importance when
    # available (tree-based learners expose feature_importances_), else variance.
    fi = getattr(model, "feature_importances_", None)
    if fi is not None and len(fi) == d_all:
        order = np.argsort(np.asarray(fi, float))[::-1][: min(max_features, d_all)]
    else:
        order = np.argsort(X.var(axis=0))[::-1][: min(max_features, d_all)]
    sel = list(order)
    feats = [feature_cols[i] for i in sel]
    d = len(sel)

    # Predictions always use the FULL feature width; non-selected features are held
    # at their column mean so only the selected inputs vary.
    base = X.mean(axis=0).copy()
    lo = np.percentile(X[:, sel], 1, axis=0)
    hi = np.percentile(X[:, sel], 99, axis=0)

    def to_full(sub: np.ndarray) -> np.ndarray:
        out = np.tile(base, (len(sub), 1))
        out[:, sel] = sub
        return out

    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = sampler.random(n_base)
    A = lo + m * (hi - lo)
    B = lo + qmc.Sobol(d=d, scramble=True, seed=seed + 1).random(n_base) * (hi - lo)

    yA = model.predict(to_full(A))
    yB = model.predict(to_full(B))
    var = np.var(np.concatenate([yA, yB]))
    if var <= 1e-12:
        return {"features": feats, "S1": [0.0] * d, "ST": [0.0] * d,
                "n_base": n_base, "warning": "zero-variance target"}

    f0 = np.mean(yA)
    f0sq = f0 * f0
    den = np.mean(yA ** 2) - f0sq
    S1, ST = [], []
    for j in range(d):
        AB = A.copy()
        AB[:, j] = B[:, j]
        yAB = model.predict(to_full(AB))
        S1.append((np.mean(yA * yAB) - f0sq) / den if den != 0 else 0.0)
        ST.append(1.0 - (np.mean(yB * yAB) - f0sq) / den if den != 0 else 0.0)
    return {"features": feats, "S1": [float(x) for x in S1],
            "ST": [float(x) for x in ST], "n_base": n_base}
