"""15_train_6h — one-step (+6h) recursive trajectory models.

Trains on the 6-hourly grid's one-step delta target GWL(t+6h) - GWL(t):
  * point model  -> XGBoost (reg:squarederror), early-stop on val
  * quantiles    -> XGBoost q05/q50/q95 (reg:quantileerror) for per-step
                    trajectory uncertainty (widened per-horizon in backtest)

All features are strictly causal (observable at time t). The point model is
rolled forward recursively by 17_recursive.py.

Artifacts -> models/6h_*.joblib + models/6h_feature_config.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FEATS = ROOT / "data" / "features_6h"
MODELS = ROOT / "models"

# full 6h grid is used for the one-step model (no 1/day downsampling — the
# recursive engine predicts at every 6h slot, so all hours must be in-sample).
XGB_PARAMS = dict(
    n_estimators=2000, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
    objective="reg:squarederror", eval_metric="rmse", random_state=42, n_jobs=6,
)
Q_PARAMS = dict(XGB_PARAMS, objective="reg:quantileerror")
ALPHAS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main() -> None:
    prep = json.loads((FEATS / "prep_6h.json").read_text())
    NUM = list(prep["num_cols"])
    xcols = NUM + ["st_id", "dist_id"]

    train = pd.read_parquet(FEATS / "train.parquet").sort_values(["Station", "time"])
    val = pd.read_parquet(FEATS / "val.parquet").sort_values(["Station", "time"])
    test = pd.read_parquet(FEATS / "test.parquet").sort_values(["Station", "time"])

    Xtr, ytr = train[xcols], train["target_d"].values
    Xva, yva = val[xcols], val["target_d"].values
    yva_lvl = val["target"].values

    # ---- point model (recursive one-step mean path) ------------------------
    dxgb = xgb.XGBRegressor(**XGB_PARAMS)
    try:
        dxgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                 early_stopping_rounds=60)
    except TypeError:
        dxgb.set_params(early_stopping_rounds=60)
        dxgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best_it = getattr(dxgb, "best_iteration", None)
    n_trees = best_it + 1 if best_it is not None else dxgb.n_estimators
    xgb_val_d = rmse(yva, dxgb.predict(Xva))
    xgb_val_lvl = rmse(yva_lvl, val["gwl"].values + dxgb.predict(Xva))
    print(f"+6h XGBoost point: best_iteration={n_trees} "
          f"val_rmse(delta)={xgb_val_d:.4f} val_rmse(level)={xgb_val_lvl:.4f} m")

    # ---- quantile models (same causal features) -----------------------------
    fitted: dict[str, object] = {}
    for name, alpha in ALPHAS.items():
        m = xgb.XGBRegressor(**Q_PARAMS, quantile_alpha=alpha)
        try:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                  early_stopping_rounds=60)
        except TypeError:
            m.set_params(early_stopping_rounds=60)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        fitted[name] = m
        joblib.dump(m, MODELS / f"6h_xgb_{name}.joblib")
        print(f"+6h {name}: trained  best_iter={int(getattr(m, 'best_iteration', 0) + 1)}")

    # raw quantile coverage on ONE-STEP holdout (pre-calibration, for ref)
    lo = val["gwl"].values + fitted["q05"].predict(Xva)
    hi = val["gwl"].values + fitted["q95"].predict(Xva)
    raw_cov = float(((yva_lvl >= lo) & (yva_lvl <= hi)).mean())
    raw_hw = float(np.median((hi - lo) / 2.0))
    print(f"+6h raw q05-q95 coverage (val, 1-step): {raw_cov:.3f}  "
          f"median half-width {raw_hw:.3f} m")

    # point model test sanity + persistence comparison (single step)
    Xte, yte_d = test[xcols], test["target_d"].values
    yte_lvl = test["target"].values
    xgb_te_d = rmse(yte_d, dxgb.predict(Xte))
    pers_te_d = rmse(yte_d, np.zeros_like(yte_d))
    print(f"+6h test: xgb_rmse_delta={xgb_te_d:.4f}  persistence={pers_te_d:.4f} m  "
          f"({100*(pers_te_d-xgb_te_d)/pers_te_d:+.1f}% vs persistence)")

    # ---- artifacts -----------------------------------------------------------
    MODELS.mkdir(exist_ok=True)
    joblib.dump(dxgb, MODELS / "6h_xgb_point.joblib")
    config = {
        "model_type": "pooled XGBoost, one-step +6h delta target for recursive "
                      "6-hourly trajectory (120 steps -> 30 days)",
        "target": "delta (GWL_{t+1} - GWL_t) on the 6h grid; level = current + delta",
        "horizons": prep["horizons"],
        "horizon_steps": prep["horizon_steps"],
        "steps_per_day": prep["steps_per_day"],
        "num_cols": NUM,
        "stations": prep["stations"],
        "districts": prep["districts"],
        "train_rows": prep["train_rows"],
        "val_rows": prep["val_rows"],
        "test_rows": prep["test_rows"],
        "point": {"n_trees": int(n_trees),
                  "val_rmse_delta": round(xgb_val_d, 4),
                  "val_rmse_level": round(xgb_val_lvl, 4),
                  "test_rmse_delta": round(xgb_te_d, 4)},
        "quantiles": {
            "alphas": ALPHAS,
            "raw_val_coverage_1step": round(raw_cov, 4),
            "raw_val_half_width_median_m": round(raw_hw, 4),
        },
        "trained_at": str(pd.Timestamp.now())[:19],
    }
    (MODELS / "6h_feature_config.json").write_text(json.dumps(config, indent=2))
    print(f"saved -> models/6h_xgb_point.joblib, 6h_xgb_q05/q50/q95.joblib, "
          f"6h_feature_config.json")


if __name__ == "__main__":
    main()