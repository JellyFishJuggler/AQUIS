"""06_train — pooled multi-horizon GWL forecast: XGBoost + Linear (Ridge).

Shared time split (train < 2025-10-01, val = Oct-Dec 2025, test = 2026).
Artifacts -> ml/models/  (joblib + feature_config.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
FEATS = ROOT / "data" / "features"
MODELS = ROOT / "models"


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main() -> None:
    prep = json.loads((FEATS / "prep.json").read_text())
    NUM = list(prep["num_cols"])
    train = pd.read_parquet(FEATS / "train.parquet")
    val = pd.read_parquet(FEATS / "val.parquet")

    # delta target: model the change GWL_{t+120} - GWL_t (persistence = 0 change)
    xcols = NUM + ["st_id", "dist_id"]
    # Train on the 6h grid but sample 1 slot/day per station: features (lags/
    # rolling/target) are grid-derived so values are identical; this keeps the
    # Ridge dense solve and DMatrix within memory on the pooled train set.
    train = train[train.groupby("Station")["time"].cumcount() % 4 == 0]
    Xtr = train[xcols]
    ytr_d = train["target_d"].values
    ytr_lvl = train["target"].values
    Xva = val[xcols]
    yva_d = val["target_d"].values
    yva_lvl = val["target"].values

    # ---- XGBoost (delta target, early stopping on val) -----------------------
    def make_xgb(**kw):
        base = dict(
            n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
            objective="reg:squarederror", eval_metric="rmse", random_state=42, n_jobs=6,
        )
        base.update(kw)
        return xgb.XGBRegressor(**base)

    try:
        dxgb = make_xgb()
        dxgb.fit(Xtr, ytr_d, eval_set=[(Xva, yva_d)], verbose=False, early_stopping_rounds=60)
    except TypeError:
        dxgb = make_xgb(early_stopping_rounds=60)
        dxgb.fit(Xtr, ytr_d, eval_set=[(Xva, yva_d)], verbose=False)
    best_it = getattr(dxgb, "best_iteration", None)
    n_trees = best_it + 1 if best_it is not None else dxgb.n_estimators
    xgb_val_d = rmse(yva_d, dxgb.predict(Xva))
    # level RMSE = anchor + predicted change
    xgb_val_lvl = rmse(yva_lvl, (val["gwl"].values + dxgb.predict(Xva)))
    print(f"XGBoost: best_iteration={n_trees} val_rmse(delta)={xgb_val_d:.4f} val_rmse(level)={xgb_val_lvl:.4f} m")

    # ---- Linear (Ridge, delta target) ----------------------------------------
    colt = ColumnTransformer(
        [
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
             ["Station", "District"]),
        ],
        remainder="drop",
    )
    ridge = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
    pipe = Pipeline([("colt", colt), ("ridge", ridge)])
    pipe.fit(train[NUM + ["Station", "District"]], ytr_d)
    ridge_val_d = rmse(yva_d, pipe.predict(val[NUM + ["Station", "District"]]))
    ridge_val_lvl = rmse(yva_lvl, (val["gwl"].values + pipe.predict(val[NUM + ["Station", "District"]])))
    print(f"Ridge: alpha={pipe.named_steps['ridge'].alpha_} val_rmse(delta)={ridge_val_d:.4f} val_rmse(level)={ridge_val_lvl:.4f} m")

    # ---- persist artifacts ---------------------------------------------------
    MODELS.mkdir(exist_ok=True)
    joblib.dump(dxgb, MODELS / "xgb_multihorizon.joblib")
    joblib.dump(pipe, MODELS / "linear_multihorizon.joblib")

    config = {
        "model_type": "pooled global model, single 30-day horizon (6h grid, 120 steps)",
        "target": "delta (GWL_{t+120} - GWL_t); level = anchor + delta; anchor never predicted",
        "horizons": prep["horizons"],
        "num_cols": NUM,
        "stations": prep["stations"],
        "districts": prep["districts"],
        "train_rows": prep["train_rows"],
        "val_rows": prep["val_rows"],
        "test_rows": prep["test_rows"],
        "xgb": {"n_trees": int(n_trees), "val_rmse_delta": round(xgb_val_d, 4), "val_rmse_level": round(xgb_val_lvl, 4)},
        "ridge": {"alpha": float(pipe.named_steps["ridge"].alpha_), "val_rmse_delta": round(ridge_val_d, 4), "val_rmse_level": round(ridge_val_lvl, 4)},
        "trained_at": str(pd.Timestamp.now())[:19],
    }
    (MODELS / "feature_config.json").write_text(json.dumps(config, indent=2))
    print(f"saved -> {MODELS}/xgb_multihorizon.joblib, linear_multihorizon.joblib, feature_config.json")

    _write_model_metadata(config, prep, xgb_val_lvl, ridge_val_lvl)


def _write_model_metadata(config: dict, prep: dict, xgb_val_lvl: float, ridge_val_lvl: float) -> None:
    """Write models/model_metadata.json — train-time facts + any existing validation
    artifacts (honest metrics, spatial CV, quantile calibration), so a deployed host
    can check which build is loaded without re-deriving anything."""
    import datetime as _dt

    meta: dict = {
        "generated_by": "06_train.py",
        "architecture": config["model_type"],
        "target": config["target"],
        "horizons": config["horizons"],
        "n_stations": len(prep.get("stations", [])),
        "n_districts": len(prep.get("districts", [])),
        "train_val_rows": config["train_rows"] + config["val_rows"],
        "test_rows": config["test_rows"],
        "val_rmse_level_m": {
            "xgb": float(xgb_val_lvl),
            "ridge": float(ridge_val_lvl),
        },
        "split": prep.get("split", {}),
        "trained_at": config["trained_at"],
        "generated_at": str(pd.Timestamp.now())[:19],
        "source_archive": "data/processed/common.parquet",
        "validated": {},
    }

    def _try(path: Path, section: str) -> None:
        if not path.exists():
            return
        try:
            if path.suffix == ".csv":
                df = pd.read_csv(path)
                meta["validated"][section] = df.head(20).to_dict(orient="records")
            elif path.suffix == ".json":
                meta["validated"][section] = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001 - metadata is best-effort
            meta["validated"][section] = {"error": str(e)}

    _try(MODELS / "quantile_calibration.json", "quantile_calibration")
    _try(ROOT / "outputs" / "honest_metrics.csv", "honest_metrics")
    _try(ROOT / "outputs" / "spatial_cv_metrics.csv", "spatial_cv")
    _try(ROOT / "outputs" / "spatial_cv_summary.json", "spatial_cv_summary")

    (MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {MODELS}/model_metadata.json")


if __name__ == "__main__":
    main()