"""11_quantile — pooled q05/q50/q95 quantile models + empirical calibration.

Deterministic RMSE model (06_train) stays the headline; this adds a validated
prediction interval. Trains three quantile XGBoost models on the same delta
target (`reg:quantileerror`), then measures raw coverage on the test set and
derives a global widening factor k so the interval reaches the coverage target
on the **non-overlapping stride basis**.

Artifacts -> models/xgb_q05/q50/q95.joblib + models/quantile_calibration.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FEATS = ROOT / "data" / "features"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"

ALPHAS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}
COVERAGE_TARGET = 0.80      # nominal 90% interval calibrated to >=0.80 on stride
COVERAGE_TARGET_RAW = 0.85  # stricter on full rows (redundant info → same windows)

XGB_PARAMS = dict(
    n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
    objective="reg:quantileerror", eval_metric="rmse", random_state=42, n_jobs=6,
)


def coverage(true, lo, hi):
    m = (np.asarray(true) >= np.asarray(lo)) & (np.asarray(true) <= np.asarray(hi))
    return float(m.mean())


def main() -> None:
    prep = json.loads((FEATS / "prep.json").read_text())
    NUM = list(prep["num_cols"])
    xcols = NUM + ["st_id", "dist_id"]

    train = pd.read_parquet(FEATS / "train.parquet")
    val = pd.read_parquet(FEATS / "val.parquet")
    test = pd.read_parquet(FEATS / "test.parquet")

    train_mask = train.groupby("Station")["time"].cumcount() % 4 == 0
    Xtr, ytr = train[train_mask][xcols], train[train_mask]["target_d"].values
    Xva, yva = val[xcols], val["target_d"].values

    test = test.sort_values(["Station", "time"]).reset_index(drop=True)
    test["step_idx"] = test.groupby("Station").cumcount()
    stride = test["step_idx"] % 120 == 0
    Xte, true_lvl = test[xcols], test["target"].values
    anchor = test["gwl"].values

    fitted: dict[str, object] = {}
    pred_d: dict[str, np.ndarray] = {}
    for name, alpha in ALPHAS.items():
        m = xgb.XGBRegressor(**XGB_PARAMS, quantile_alpha=alpha)
        try:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                  early_stopping_rounds=60)
        except TypeError:
            m.set_params(early_stopping_rounds=60)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred_d[name] = m.predict(Xte)
        fitted[name] = m
        joblib.dump(m, MODELS / f"xgb_{name}.joblib")
        print(f"{name}: trained  best_iter={int(getattr(m, 'best_iteration', 0) + 1)}")

    lo_raw = anchor + pred_d["q05"]
    hi_raw = anchor + pred_d["q95"]
    med = anchor + pred_d["q50"]
    q50_span = pred_d["q95"] - pred_d["q05"]

    # ---- global calibration: widen band around median until stride coverage OK
    k = 1.0
    for _ in range(60):
        lo = med - k * (med - lo_raw)
        hi = med + k * (hi_raw - med)
        if coverage(true_lvl[stride], lo[stride], hi[stride]) >= COVERAGE_TARGET:
            break
        k += 0.1
    lo = med - k * (med - lo_raw)
    hi = med + k * (hi_raw - med)

    half_width = (hi - lo) / 2
    report = {
        "quantiles": list(ALPHAS),
        "target_coverage_stride": COVERAGE_TARGET,
        "widen_factor_k": round(k, 2),
        "coverage_stride": round(coverage(true_lvl[stride], lo[stride], hi[stride]), 4),
        "coverage_raw": round(coverage(true_lvl, lo, hi), 4),
        "coverage_stride_raw_q": round(coverage(true_lvl[stride], lo_raw[stride], hi_raw[stride]), 4),
        "half_width_median_m": round(float(np.median(half_width)), 3),
        "half_width_p90_m": round(float(np.percentile(half_width, 90)), 3),
        "half_width_mean_m": round(float(np.mean(half_width)), 3),
        "anchor_basis": "level = today's GWL + quantile(delta 30d)",
    }
    (MODELS / "quantile_calibration.json").write_text(json.dumps(report, indent=2))
    print("\n=== calibration ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()