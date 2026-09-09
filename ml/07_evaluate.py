"""07_evaluate — 4-way comparison on the held-out 2026 test set.

Models: XGBoost, Ridge, persistence (gwl_{t+h} = gwl_t), seasonal climatology
(per-station day-of-year mean from training years).

Outputs -> ml/outputs/*.csv + predictions_2026.parquet
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATS = ROOT / "data" / "features"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"

DRIVER_COLS = ["rain_1d", "rain_7d", "rain_30d", "temp", "humidity", "solar",
               "wind_speed", "pressure", "river_level", "canal_level"]

TOLERANCES = {"within_025": 0.25, "within_05": 0.5, "within_10": 1.0}


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    true = np.asarray(true, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    ok = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[ok], pred[ok]
    if true.size == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "resid_std": np.nan,
                **{k: np.nan for k in TOLERANCES},
                **{f"p{q}": np.nan for q in (5, 50, 95)}}
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) or np.nan
    r2 = 1.0 - ss_res / ss_tot
    out = {"rmse": rmse, "mae": mae, "r2": r2, "resid_std": float(err.std())}
    for k, tol in TOLERANCES.items():
        out[k] = float(np.mean(np.abs(err) <= tol))
    for q in (5, 50, 95):
        out[f"p{q}"] = float(np.percentile(err, q))
    return out


def add_metrics(rows: list[dict], label: str, h: int, true, pred, n_feat_days) -> None:
    m = metrics(true, pred)
    rows.append({"model": label, "horizon": h, "rows": int(len(true)),
                 "feat_days": int(n_feat_days), **m})


def main() -> None:
    prep = json.loads((FEATS / "prep.json").read_text())
    NUM = list(prep["num_cols"])
    config = json.loads((MODELS / "feature_config.json").read_text())
    dxgb = joblib.load(MODELS / "xgb_multihorizon.joblib")
    pipe = joblib.load(MODELS / "linear_multihorizon.joblib")

    train = pd.read_parquet(FEATS / "train.parquet")
    test = pd.read_parquet(FEATS / "test.parquet")

    # ---- climate baseline: per-station mean of target by day-of-year (train only)
    tr = train.copy()
    tr["doy"] = (tr["date"] + pd.to_timedelta(tr["horizon"], unit="D")).dt.dayofyear
    clim = tr.groupby(["Station", "doy"])["target"].mean().rename("clim")
    stmean = tr.groupby("Station")["target"].mean().rename("stmean")

    te = test.copy()
    te["doy"] = (te["date"] + pd.to_timedelta(te["horizon"], unit="D")).dt.dayofyear
    te = te.merge(clim.reset_index(), on=["Station", "doy"], how="left")
    te = te.merge(stmean.reset_index(), on="Station", how="left")
    pred_seas = te["clim"].fillna(te["stmean"]).values

    true = te["target"].values
    true_d = te["target_d"].values
    xcols = NUM + ["st_id", "dist_id"]
    # models train on the change (delta) target; report on the level = anchor + delta
    pred_xgb = te["gwl"].values + dxgb.predict(te[xcols])
    pred_ridge = te["gwl"].values + pipe.predict(te[NUM + ["Station", "District"]])
    pred_persist = te["gwl"].values
    pred_xgb_d = pred_xgb - te["gwl"].values
    pred_ridge_d = pred_ridge - te["gwl"].values
    pred_persist_d = np.zeros_like(pred_xgb_d)
    pred_seas_d = pred_seas - te["gwl"].values

    # driver-richness per station (for honest coverage reporting)
    te["feat_days"] = te[DRIVER_COLS].notna().any(axis=1).astype(np.int8)

    # ---- headline metrics (only horizon = 30 d) ------------------------------
    rows: list[dict] = []
    for h in (30,):
        m = te["horizon"] == h
        nfd = int(te.loc[m, "feat_days"].sum())
        add_metrics(rows, "xgboost", h, true[m], pred_xgb[m], nfd)
        add_metrics(rows, "ridge", h, true[m], pred_ridge[m], nfd)
        add_metrics(rows, "persistence", h, true[m], pred_persist[m], nfd)
        add_metrics(rows, "climatology", h, true[m], pred_seas[m], nfd)
        for label, pd_in, pd_d in (("xgboost", pred_xgb, pred_xgb_d),
                                   ("ridge", pred_ridge, pred_ridge_d),
                                   ("persistence", pred_persist, pred_persist_d),
                                   ("climatology", pred_seas, pred_seas_d)):
            dm = metrics(true_d[m], pd_d[m])
            rows.append({"model": label + " (delta)", "horizon": h, "rows": int(m.sum()),
                         "feat_days": nfd, "rmse": dm["rmse"], "mae": dm["mae"],
                         "r2": dm["r2"], "resid_std": dm["resid_std"],
                         "within_025": dm["within_025"], "within_05": dm["within_05"],
                         "within_10": dm["within_10"], "p5": dm["p5"], "p50": dm["p50"],
                         "p95": dm["p95"]})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "model_metrics.csv", index=False)
    print(metrics_df.round(3).to_string(index=False), flush=True)
    print()

    # ---- by district / by station -------------------------------------------
    dist_rows = []
    for (d, h), sub in te.groupby(["District", "horizon"]):
        m = sub["target"].values
        for label, p in (("xgboost", pred_xgb), ("ridge", pred_ridge),
                         ("persistence", pred_persist), ("climatology", pred_seas)):
            pm = p[sub.index]
            r = metrics(m, pm)
            dist_rows.append({"district": d, "horizon": h, "model": label,
                              "rows": len(m), "rmse": r["rmse"], "mae": r["mae"],
                              "within_05": r["within_05"]})
    pd.DataFrame(dist_rows).to_csv(OUT / "eval_by_district.csv", index=False)

    st_rows = []
    for (s_, h), sub in te.groupby(["Station", "horizon"]):
        m = sub["target"].values
        row = {"station": s_, "horizon": h, "district": sub["District"].iloc[0],
               "rows": len(m), "feat_days": int(sub["feat_days"].sum())}
        for label, p in (("xgboost", pred_xgb), ("ridge", pred_ridge),
                         ("persistence", pred_persist), ("climatology", pred_seas)):
            row[label + "_rmse"] = np.sqrt(np.mean((m - p[sub.index]) ** 2))
        st_rows.append(row)
    st_df = pd.DataFrame(st_rows)
    st_df.to_csv(OUT / "eval_by_station.csv", index=False)
    n_driver_rich = int((st_df.groupby("station")["feat_days"].sum() >= 90).sum())
    print(f"driver-rich test stations (>=90 feature-days in 2026): {n_driver_rich} / {st_df['station'].nunique()}")

    # ---- predictions (for the app + honest validation) -----------------------
    preds = te[["Station", "District", "date", "time", "horizon", "target", "gwl",
                "feat_days"]].copy()
    preds["xgb"] = pred_xgb
    preds["ridge"] = pred_ridge
    preds["persist"] = pred_persist
    preds["clim"] = pred_seas
    preds["err_xgb"] = pred_xgb - true
    preds["err_ridge"] = pred_ridge - true
    # calibrated quantile levels on the test set (band for the backtest chart)
    try:
        qmods = {q: joblib.load(MODELS / f"xgb_{q}.joblib") for q in ("q05", "q50", "q95")}
        qcal = json.loads((MODELS / "quantile_calibration.json").read_text())
        qlev = {q: te["gwl"].values + qmods[q].predict(te[xcols]) for q in qmods}
        kq = float(qcal.get("widen_factor_k", 1.0))
        # Quantile models trained independently can cross at extreme telemetry;
        # enforce monotone ordering (lo <= med <= hi) before widening.
        med = qlev["q50"]
        lo = np.minimum(qlev["q05"], med)
        hi = np.maximum(qlev["q95"], med)
        preds["q05_lvl"] = med - kq * (med - lo)
        preds["q95_lvl"] = med + kq * (hi - med)
        print("calibrated quantile test band added (q05_lvl/q95_lvl)")
    except Exception as e:  # noqa: BLE001 — quantile models are optional
        print(f"quantile band skipped: {e}")
    # non-overlapping 30-day evaluation windows: each station's 6h test rows are
    # binned into 120-step (~30-day) windows; `stride` keeps 1 row/window.
    preds = preds.sort_values(["Station", "time"])
    preds["step_idx"] = preds.groupby("Station").cumcount()
    preds["window_id"] = preds["step_idx"] // 120
    preds["stride"] = preds["step_idx"] % 120 == 0
    preds.to_parquet(OUT / "predictions_2026.parquet", index=False)
    print(f"predictions -> outputs/predictions_2026.parquet  ({len(preds):,} rows)")

    # ---- feature importance / coefficients -----------------------------------
    gain = dxgb.get_booster().get_score(importance_type="gain")
    freq = dxgb.get_booster().get_score(importance_type="weight")
    imp = pd.DataFrame({"feature": list(dxgb.feature_names_in_)})
    imp["gain"] = imp["feature"].map(lambda f: gain.get(f, 0.0))
    imp["freq"] = imp["feature"].map(lambda f: freq.get(f, 0))
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    names = pipe.named_steps["colt"].get_feature_names_out()
    coef = pipe.named_steps["ridge"].coef_
    coefs = pd.DataFrame({"feature": names, "coef": np.asarray(coef)})
    coefs["abs_coef"] = coefs["coef"].abs()
    coefs.to_csv(OUT / "feature_coefficients.csv", index=False)
    print(f"xgb feature names had no gain: {(imp['gain'] == 0).sum()}")

    # ---- improvement over persistence ----------------------------------------
    print("\nRMSE improvement vs persistence (30 d):")
    mm = metrics_df.pivot(index="horizon", columns="model", values="rmse")
    for h in (30,):
        for lbl in ("xgboost", "ridge"):
            imp_p = 100 * (mm.loc[h, "persistence"] - mm.loc[h, lbl]) / mm.loc[h, "persistence"]
            print(f"  h={h}: {lbl:8s} {mm.loc[h, lbl]:.3f} m  vs persist {mm.loc[h, 'persistence']:.3f} m  ({imp_p:+.1f}%)")

    # ---- honest validation suite (overlap windows, residual ACF, spatial
    #      autocorrelation) — regenerates outputs/honest_metrics.csv, /overlap.json,
    #      /residual_acf.json, /residual_spatial.json -------------------------
    import sys
    vtools = ["overlap", "residual_acf", "spatial_residual"]
    for tool in vtools:
        script = ROOT / "validation" / f"{tool}.py"
        if script.exists():
            r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            if r.returncode == 0:
                print(f"validation {tool}: ok")
            else:
                print(f"validation {tool}: FAILED\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


if __name__ == "__main__":
    main()