"""10_ablate — drop-group feature ablation for the pooled 30-day forecast.

Measures the marginal predictive value of each driver family by re-training the
pooled XGBoost on the already-materialised feature frames with one group of
columns removed, and re-scoring the 2026 test set on raw + non-overlapping
(window-stride) bases. No feature rebuild needed — groups exist in prep num_cols.

Group semantics: "rain" = rainfall buckets, "weather" = temp/humidity/solar/wind/
pressure, "river_canal" = river + canal level, "calendar" = date/hour/monsoon
encodings, "gwl_lag" = the groundwater trajectory memory (lags/rolls), PLUS
cross configs ("no_drivers", "no_gwl_history").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FEATS = ROOT / "data" / "features"
OUT = ROOT / "outputs"

GROUPS = {
    "rain": ["rain_1d", "rain_7d", "rain_30d"],
    "weather": ["temp", "temp_7d", "humidity", "humidity_7d", "solar",
                "wind_speed", "pressure"],
    "river_canal": ["river_level", "canal_level"],
    "calendar": ["year", "month_sin", "month_cos", "doy_sin", "doy_cos",
                 "hour_sin", "hour_cos", "monsoon"],
    "gwl_lag": ["lag1", "lag4", "lag8", "lag28", "lag120",
                "gwl_roll7_mean", "gwl_roll7_std",
                "gwl_roll30_mean", "gwl_roll30_std"],
}

XGB_PARAMS = dict(
    n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
    objective="reg:squarederror", eval_metric="rmse", random_state=42, n_jobs=6,
)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def stride_rmse(df, pred):
    d = df.sort_values(["Station", "time"]).reset_index(drop=True)
    d["step_idx"] = d.groupby("Station").cumcount()
    stride = d["step_idx"] % 120 == 0
    return rmse(d.loc[stride, "target"].values, np.asarray(pred)[stride.values])


def main() -> None:
    prep = json.loads((FEATS / "prep.json").read_text())
    NUM = list(prep["num_cols"])
    train = pd.read_parquet(FEATS / "train.parquet")
    val = pd.read_parquet(FEATS / "val.parquet")
    test = pd.read_parquet(FEATS / "test.parquet")

    train_mask = train.groupby("Station")["time"].cumcount() % 4 == 0
    in_groups = {k: [c for c in cols if c in NUM] for k, cols in GROUPS.items()}
    all_group_cols = sorted({c for cols in in_groups.values() for c in cols})

    configs: list[tuple[str, list[str]]] = [("baseline", NUM)]
    for name, cols in in_groups.items():
        if cols:
            configs.append((f"drop_{name}", [c for c in NUM if c not in cols]))
    no_drivers = [c for c in NUM if c not in {c for grp in
                  ("rain", "weather", "river_canal") for c in in_groups[grp]}]
    configs.append(("no_drivers", no_drivers))
    no_gwl = [c for c in NUM if c not in in_groups["gwl_lag"]]
    configs.append(("no_gwl_history", no_gwl))

    rows: list[dict] = []
    for name, feats in configs:
        xcols = feats + ["st_id", "dist_id"]
        trf = train[train_mask]
        vf = val
        Xtr, ytr = trf[xcols], trf["target_d"].values
        Xva, yva = (vf[xcols], vf["target_d"].values) if len(vf) \
            else (Xtr, ytr)
        dxgb = xgb.XGBRegressor(**XGB_PARAMS)
        try:
            dxgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False,
                     early_stopping_rounds=60)
        except TypeError:
            dxgb.set_params(early_stopping_rounds=60)
            dxgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred = test["gwl"].values + dxgb.predict(test[xcols])
        err = pred - test["target"].values
        row = {
            "config": name,
            "num_feats": len(feats),
            "n_trees": int(getattr(dxgb, "best_iteration", 0) + 1),
            "val_rmse_delta": round(rmse(yva, dxgb.predict(Xva)), 4),
            "raw_rmse": round(rmse(test["target"].values, pred), 4),
            "raw_mae": round(float(np.mean(np.abs(err))), 4),
            "raw_r2": round(1 - np.sum(err ** 2) / float(
                np.sum((test["target"].values - test["target"].values.mean()) ** 2)), 4),
            "stride_rmse": round(stride_rmse(test, pred), 4),
        }
        rows.append(row)
        print(f"{name:16s} feats={row['num_feats']:2d}  trees={row['n_trees']:3d}  "
              f"raw_rmse={row['raw_rmse']:.3f}  stride_rmse={row['stride_rmse']:.3f}  "
              f"val_rmse={row['val_rmse_delta']:.3f}  r2={row['raw_r2']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation.csv", index=False)

    base = df.set_index("config").loc["baseline"]
    print("\n=== delta vs baseline (raw RMSE) ===")
    for _, r in df.iterrows():
        if r["config"] != "baseline":
            d = (r["raw_rmse"] - base["raw_rmse"])
            print(f"  {r['config']:16s}  {d:+.4f}  m   ({base['raw_rmse']:.3f} -> {r['raw_rmse']:.3f})")


if __name__ == "__main__":
    main()