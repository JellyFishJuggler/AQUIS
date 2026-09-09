"""12_diagnostics — collinearity (VIF), permutation importance & Sobol sensitivity.

Runs on the pooled 30-day model using the non-overlap (stride) test subset so the
feature contributions aren't inflated by overlapping 30-day windows.

Outputs -> outputs/diagnostics_permutation.csv, outputs/diagnostics.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

N_REPEATS = 5
SEED = 42


def vif_report(X: np.ndarray, feature_cols: list[str]) -> dict:
    """Variance Inflation Factor per feature (correlation-matrix invert).

    Constant columns (zero variance, e.g. `year` in a single-year test set) are
    excluded from the inversion and reported separately — they cannot have a VIF.
    """
    X = np.asarray(X, float)
    var = X.var(axis=0)
    const_ix = [i for i in range(X.shape[1]) if var[i] <= 1e-12]
    keep = [i for i in range(X.shape[1]) if i not in const_ix]
    Xc = X[:, keep] - X[:, keep].mean(axis=0)
    p = len(keep)
    corr = np.corrcoef(Xc.T) if p > 1 else np.array([[1.0]])
    corr = corr + np.eye(p) * 1e-9
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)
    vifs = np.diag(inv)
    out = {feature_cols[i]: float("nan") if i in const_ix else float(vifs[keep.index(i)])
           for i in range(X.shape[1])}
    return {
        "feature": list(out.keys()),
        "vif": [out[f] for f in out],
        "constant_excluded": [feature_cols[i] for i in const_ix],
        "highly_correlated_gt10": [f for f, v in out.items() if v == v and v > 10],
    }


def permutation_report(model, X: np.ndarray, y: np.ndarray, feature_cols: list[str]) -> list[dict]:
    from sklearn.inspection import permutation_importance

    res = permutation_importance(
        model, X, y, scoring="neg_root_mean_squared_error",
        n_repeats=N_REPEATS, random_state=SEED, n_jobs=1,
    )
    order = np.argsort(res.importances_mean)[::-1]
    return [
        {"rank": r + 1, "feature": feature_cols[i],
         "importance_mean": float(res.importances_mean[i]),
         "importance_std": float(res.importances_std[i])}
        for r, i in enumerate(order)
    ]


def sensitivity_report(model, feature_cols: list[str], X: np.ndarray,
                       n_top: int = 8) -> dict:
    """One-at-a-time sensitivity: swing between the feature's 2.5th and 97.5th
    percentiles (each row keeping its own covariates), reported as the mean move
    of the forecast 30-day change (m)."""
    imp = np.asarray(model.feature_importances_)
    top = np.argsort(imp)[::-1][: n_top]
    feats = [feature_cols[i] for i in top]
    lo = np.nanpercentile(X[:, top], 2.5, axis=0)
    hi = np.nanpercentile(X[:, top], 97.5, axis=0)
    base = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    swing = []
    for i, j in enumerate(top):
        xl = base.copy(); xl[:, j] = lo[i]
        xh = base.copy(); xh[:, j] = hi[i]
        swing.append(float(np.mean(np.abs(model.predict(xh) - model.predict(xl)))))
    return {"features": feats, "swing_mean_m": swing, "n_rows": int(len(base))}


def main() -> None:
    import joblib

    from _model import load_feature_config

    t0 = time.time()
    cfg = load_feature_config()
    xcols = cfg["num_cols"] + ["st_id", "dist_id"]
    test = pd.read_parquet(ROOT / "data/features/test.parquet")
    test["step_idx"] = test.groupby("Station").cumcount()
    stride = test["step_idx"] % 120 == 0
    X = test.loc[stride, xcols].to_numpy(float)
    y = test.loc[stride, "target"].to_numpy(float)
    print(f"stride-subset for diagnostics: {len(y):,} rows ({len(test):,} raw)")

    model = joblib.load(ROOT / "models" / "xgb_multihorizon.joblib")
    fnames = list(getattr(model, "feature_names_in_", np.asarray(xcols)))
    col_ix = [fnames.index(c) for c in fnames]
    X = X[:, col_ix]

    perm = permutation_report(model, X, y, fnames)
    pd.DataFrame(perm).to_csv(OUT / "diagnostics_permutation.csv", index=False)
    print("permutation importance (stride RMSE, mean patience per feature):")
    for p in perm[:10]:
        print(f"  {p['rank']:>2}. {p['feature']:<20} {p['importance_mean']:+.4f} "
              f"(+/-{p['importance_std']:.4f})")

    # VIF on a random complete-subset-of-imputed sample from ALL test rows. The
    # stride subset is all hour-0 rows (constant hour_sin/cos) and most test rows
    # carry NaN in the (mostly-missing) river/weather drivers, so we mean-impute
    # the sample for this collinearity diagnostic.
    full = test.loc[:, xcols].to_numpy(float)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(full), size=min(50_000, len(full)), replace=False)
    samp = full[idx].astype(float)
    nan = np.isnan(samp)
    nan_cnt = nan.sum(axis=0)
    colmean = np.nanmean(samp, axis=0)
    samp[nan] = np.take(colmean, np.nonzero(nan)[1])
    vif = vif_report(samp[:, col_ix], fnames)
    print(f"\nVIF (mean-imputed sample, {len(samp):,} rows) > 10 (collinear): "
          f"{vif['highly_correlated_gt10'] or 'none'}")
    if vif["constant_excluded"]:
        print("VIF constant-excluded:", vif["constant_excluded"])

    sens = sensitivity_report(model, fnames, X)
    print("\nOAT sensitivity (mean |Δ 30-day change|, 2.5→97.5 pct):")
    for f, s in zip(sens["features"], sens["swing_mean_m"]):
        print(f"  {f:<20} {s:.3f} m")

    OUT.joinpath("diagnostics.json").write_text(json.dumps({
        "model": "xgb_multihorizon (pooled, delta 30d)",
        "basis": "stride (non-overlap) test subset",
        "n_stride_rows": int(len(y)),
        "permutation": perm,
        "vif": vif,
        "sensitivity": sens,
        "generated_at": str(pd.Timestamp.now())[:19],
        "seconds": round(time.time() - t0, 1),
    }, indent=1))
    print(f"\ndone in {time.time()-t0:.1f}s -> outputs/diagnostics.json + diagnostics_permutation.csv")


if __name__ == "__main__":
    main()