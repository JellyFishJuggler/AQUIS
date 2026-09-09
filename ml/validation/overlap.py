"""Overlap-aware evaluation of the 30-day forecasts.

The 6h grid means consecutive test rows forecast overlapping ~30-day windows
(119/120 of the target window is shared), so the 378k-row test set carries far
fewer independent forecasts than the raw count suggests. This module:

  * assigns each station's rows to 120-step (30-day) windows,
  * keeps 1 row per window (stride) as a non-overlapping unbiased sample,
  * aggregates per-window RMSE so every independent window weighs equally,
  * reports the effective sample size vs the raw row count.

Reads outputs/predictions_2026.parquet (produced by 07_evaluate). Writes
outputs/honest_metrics.csv, outputs/station_summary.csv, outputs/overlap.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

PREDS = OUT / "predictions_2026.parquet"
HONEST_CSV = OUT / "honest_metrics.csv"
STATION_CSV = OUT / "station_summary.csv"
OVERLAP_JSON = OUT / "overlap.json"

MODELS = ["xgb", "ridge", "persist", "clim"]
WINDOW_STEPS = 120  # 120 six-hour steps = 30 days

TOLERANCES = {"within_025": 0.25, "within_05": 0.5, "within_10": 1.0}


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[ok], pred[ok]
    if true.size == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) or np.nan
    out = {"rmse": float(np.sqrt(np.mean(err ** 2))),
           "mae": float(np.mean(np.abs(err))),
           "r2": 1.0 - ss_res / ss_tot}
    for k, tol in TOLERANCES.items():
        out[k] = float(np.mean(np.abs(err) <= tol))
    for q in (5, 50, 95):
        out[f"p{q}"] = float(np.percentile(err, q))
    return out


def main() -> None:
    df = pd.read_parquet(PREDS)
    df = df[df["horizon"] == 30]
    df = df.sort_values(["Station", "time"]).reset_index(drop=True)

    n_raw = int(len(df))
    n_windows = int(df[df["stride"]].shape[0])
    n_stations = int(df["Station"].nunique())
    stations_with_window = int(df[df["stride"]].groupby("Station").ngroups)

    # per-station stride sample (1 independent window per ~30 days)
    stride = df[df["stride"]]

    rows: list[dict] = []
    for model in MODELS:
        m_raw = _metrics(df["target"].values, df[model].values)
        m_stride = _metrics(stride["target"].values, stride[model].values)
        # per-window RMSE, then aggregate over windows
        gw = df.groupby(["Station", "window_id"])
        wrmse = gw.apply(
            lambda g: float(np.sqrt(np.mean((g[model] - g["target"]) ** 2))),
            include_groups=False)
        rows.append({
            "model": model, "horizon": 30,
            "raw_rows": n_raw, "stride_rows": n_windows,
            "n_windows": int(wrmse.shape[0]),
            "basis_raw_rmse": round(m_raw["rmse"], 4),
            "basis_raw_mae": round(m_raw["mae"], 4),
            "basis_raw_r2": round(m_raw["r2"], 4),
            "basis_raw_within_05": round(m_raw["within_05"], 4),
            "basis_raw_within_10": round(m_raw["within_10"], 4),
            "basis_raw_p5": round(m_raw["p5"], 4),
            "basis_raw_p50": round(m_raw["p50"], 4),
            "basis_raw_p95": round(m_raw["p95"], 4),
            "basis_stride_rmse": round(m_stride["rmse"], 4),
            "basis_stride_mae": round(m_stride["mae"], 4),
            "basis_stride_r2": round(m_stride["r2"], 4),
            "basis_stride_within_05": round(m_stride["within_05"], 4),
            "basis_stride_within_10": round(m_stride["within_10"], 4),
            "window_rmse_mean": round(float(wrmse.mean()), 4),
            "window_rmse_median": round(float(wrmse.median()), 4),
            "window_rmse_p90": round(float(np.percentile(wrmse, 90)), 4),
        })
    honest = pd.DataFrame(rows)
    honest.to_csv(HONEST_CSV, index=False)
    print(honest[["model", "basis_raw_rmse", "basis_stride_rmse", "window_rmse_mean",
                  "window_rmse_median", "stride_rows", "n_windows"]].round(3).to_string(index=False))

    # ---- station-wise summary (window-resolved) ---------------------------
    st_rows = []
    for (s_,), sub in df.groupby(["Station"]):
        w = sub[sub["stride"]]
        row = {"station": s_, "district": sub["District"].iloc[0],
               "n_windows": int(sub["window_id"].nunique()),
               "stride_rows": int(w.shape[0]), "raw_rows": int(len(sub))}
        for model in MODELS:
            row[model + "_raw_rmse"] = _metrics(sub["target"].values, sub[model].values)["rmse"]
            if len(w):
                row[model + "_stride_rmse"] = _metrics(w["target"].values, w[model].values)["rmse"]
            else:
                row[model + "_stride_rmse"] = np.nan
        st_rows.append(row)
    st_df = pd.DataFrame(st_rows)
    st_df.to_csv(STATION_CSV, index=False)
    n_beat = int((st_df["xgb_stride_rmse"] < st_df["persist_stride_rmse"]).sum())
    n_valid = int(st_df["persist_stride_rmse"].notna().sum())
    print(f"stations with >=1 independent window: {n_valid} / {n_stations}")
    print(f"stations where xgb stride-RMSE < persistence: {n_beat} / {n_valid}")

    summary = {
        "raw_rows": n_raw, "independent_windows": n_windows,
        "stations": n_stations, "stations_with_window": stations_with_window,
        "effective_N_ratio": round(n_windows / n_raw, 5),
        "window_length_steps": WINDOW_STEPS,
        "window_length_days": 30,
        "generated_cols": ["step_idx", "window_id", "stride"],
    }
    OVERLAP_JSON.write_text(json.dumps(summary, indent=2))
    print("\noverlap summary:", json.dumps(summary))


if __name__ == "__main__":
    main()