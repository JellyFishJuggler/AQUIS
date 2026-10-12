"""Spatial CV for the pooled 30-day forecast: leave-one-spatial-block-out.

Detects spatial leakage by holding out a whole geographic block of stations from
training and scoring the pooled XGBoost on that block's 2026 test rows. A model
that only memorises district/monsoon patterns performs worse here than on the
temporal split, exposing how much of the number is real out-of-domain accuracy.

Method faithfully mirrors 06_train: fit on `target_d` (delta), anchor gwl is
never predicted, features are rebuilt once and only the station split changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "data" / "features"
OUT = ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _spatial_folds import (  # noqa: E402
    load_station_coords,
    spatial_folds,
    write_fold_assignment,
)

FOLDS_CSV = OUT / "spatial_folds.csv"
METRICS_CSV = OUT / "spatial_cv_metrics.csv"
SUMMARY_JSON = OUT / "spatial_cv_summary.json"

XGB_PARAMS = dict(
    n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
    objective="reg:squarederror", eval_metric="rmse", random_state=42, n_jobs=6,
)
EARLY_STOP = 60


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_blocks", type=int, default=5)
    ap.add_argument("--limit_folds", type=int, default=0, help="only run first N folds")
    args = ap.parse_args()

    prep = json.loads((FEATS / "prep.json").read_text())
    NUM = list(prep["num_cols"])
    xcols = NUM + ["st_id", "dist_id"]

    coords_df = load_station_coords()
    train = pd.read_parquet(FEATS / "train.parquet")
    val = pd.read_parquet(FEATS / "val.parquet")
    test = pd.read_parquet(FEATS / "test.parquet")

    # fold assignment from ALL candidate stations (not only test stations)
    coord_arr = coords_df[["Latitude", "Longitude"]].to_numpy()
    folds, labels = spatial_folds(coord_arr, n_blocks=args.n_blocks)
    write_fold_assignment(coords_df, labels)
    print(f"assigned {len(coords_df)} stations to {int(labels.max()) + 1} spatial blocks "
          f"(sizes {np.bincount(labels).tolist()})")

    # replicate 06_train's 1-slot/day train sampling (frame order preserved)
    train_mask = train.groupby("Station")["time"].cumcount() % 4 == 0

    done_blocks = set()
    rows: list[dict] = []
    if METRICS_CSV.exists():
        prior = pd.read_csv(METRICS_CSV)
        if "block" in prior:
            rows = prior.to_dict("records")
            done_blocks = {int(b) for b in prior["block"].tolist()}
            print(f"resuming: {len(done_blocks)} blocks already done")

    for bi, (tr_idx, te_idx) in enumerate(folds):
        if args.limit_folds and bi >= args.limit_folds:
            break
        if bi in done_blocks:
            continue
        tr_names = set(coords_df["Station"].iloc[tr_idx])
        te_names = set(coords_df["Station"].iloc[te_idx])

        # exclusivity assert: no station in both train & test of the fold
        assert not (tr_names & te_names), f"fold {bi}: spatial leak (shared station)"
        n_tr, n_te = len(tr_names), len(te_names)

        if n_tr < 30 or n_te == 0:
            print(f"[fold {bi}] skipped ({n_tr} train / {n_te} test stations)")
            continue
        print(f"[fold {bi}] train {n_tr} stn, test {n_te} stn", flush=True)
        t0 = time.time()

        trf = train[train_mask & train["Station"].isin(tr_names)]
        Xtr, ytr = trf[xcols], trf["target_d"].values
        vf = val[val["Station"].isin(tr_names)]
        Xva, yva = (vf[xcols], vf["target_d"].values) if len(vf) else (Xtr, ytr)

        dxgb = xgb.XGBRegressor(**XGB_PARAMS)
        try:
            dxgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                     early_stopping_rounds=EARLY_STOP)
        except TypeError:
            dxgb.set_params(early_stopping_rounds=EARLY_STOP)
            dxgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        tef = test[test["Station"].isin(te_names)]
        pred = tef["gwl"].values + dxgb.predict(tef[xcols])
        true = tef["target"].values
        err = pred - true
        row = {
            "block": bi, "n_train_stations": n_tr, "n_test_stations": n_te,
            "test_rows": int(len(true)), "n_trees": int(getattr(dxgb, "best_iteration", 0) + 1),
            "rmse": round(rmse(true, pred), 4), "mae": round(float(np.mean(np.abs(err))), 4),
            "r2": round(float(r2_score(true, pred)), 4),
            "train_seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"  fold {bi}: rmse={row['rmse']:.3f} m  mae={row['mae']:.3f}  r2={row['r2']:.3f} "
              f"({row['test_rows']} rows, {row['train_seconds']}s)", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(METRICS_CSV, index=False)

    # ---- summary -----------------------------------------------------------
    if METRICS_CSV.exists():
        mdf = pd.read_csv(METRICS_CSV)
        summary = {
            "n_blocks": int(labels.max()) + 1,
            "n_folds_evaluated": len(mdf),
            "pooled_test_rows": int(mdf["test_rows"].sum()),
            "fold_rmse_mean": round(float(mdf["rmse"].mean()), 4),
            "fold_rmse_median": round(float(mdf["rmse"].median()), 4),
            "fold_rmse_min": round(float(mdf["rmse"].min()), 4),
            "fold_rmse_max": round(float(mdf["rmse"].max()), 4),
            "fold_r2_mean": round(float(mdf["r2"].mean()), 4),
            "block_sizes": np.bincount(labels).tolist(),
        }
        (SUMMARY_JSON).write_text(json.dumps(summary, indent=2))
        print("\n=== Spatial CV summary (unseen-region RMSE) ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()