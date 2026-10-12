"""Temporal autocorrelation of 30-day forecast residuals (overlap honesty).

Adjacent 6-hourly test rows share ~119/120 of the same 30-day target window, so
their forecast errors are highly correlated; this inflates confidence and makes
the raw 378k-row RMSE look more precise than it is. Computes per-station ACF of
the XGBoost test residual (err_xgb) and an effective sample size per station.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PREDS = OUT / "predictions_2026.parquet"
ACF_JSON = OUT / "residual_acf.json"

LAGS = [1, 2, 3, 6, 12, 24, 48, 120, 239]          # 6h steps (~0.25..60 days)
K_EFF = 120                                         # 30 days of lags for n_eff
MIN_OBS = 30


def acf(x: np.ndarray, lags) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    x = x - x.mean() if len(x) else x
    den = float(np.dot(x, x))
    if den <= 0 or len(x) < 2:
        return np.full(len(lags), np.nan), np.full(len(lags), np.nan)
    out = np.full(len(lags), np.nan)
    for i, k in enumerate(lags):
        if k >= len(x):
            break
        out[i] = float(np.dot(x[: len(x) - k], x[k:]) / den)
    return out, np.asarray(lags, float)


def effective_n(x: np.ndarray, max_lag: int) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n < MIN_OBS:
        return float(n)
    xc = x - x.mean()
    den = float(np.dot(xc, xc))
    if den <= 0:
        return float(n)
    rho_sum = 0.0
    kmax = min(max_lag, n - 1)
    for k in range(1, kmax + 1):
        rho = float(np.dot(xc[: n - k], xc[k:]) / den)
        if rho > 0:
            rho_sum += rho
    return float(n / (1 + 2 * rho_sum))


def main() -> None:
    df = pd.read_parquet(PREDS)
    df = df.sort_values(["Station", "time"])
    df = df.dropna(subset=["err_xgb"])

    per_station = {}
    acf_rows: list[dict] = []
    mean_n_eff, n_total, n_dom = 0.0, 0, 0
    for station, g in df.groupby("Station"):
        x = g["err_xgb"].astype("float64").to_numpy()
        if len(x) < MIN_OBS:
            continue
        a, lags_used = acf(x, LAGS)
        n_eff = effective_n(x, K_EFF)
        n_total += len(x)
        n_dom += 1
        mean_n_eff += n_eff
        per_station[station] = {"n": int(len(x)), "n_eff": round(n_eff, 1),
                                **{f"lag{int(k)}": (round(float(a[i]), 4) if not np.isnan(a[i]) else None)
                                   for i, k in enumerate(lags_used) if not np.isnan(a[i])}}
        acf_rows.append({"station": station, "n": len(x), "n_eff": round(n_eff, 1),
                         **{f"lag{int(k)}": (round(float(a[i]), 4) if not np.isnan(a[i]) else None)
                            for i, k in enumerate(lags_used)}})

    acf_df = pd.DataFrame(acf_rows)
    pooled = {}
    for k in LAGS:
        vals = acf_df[f"lag{int(k)}"].dropna()
        pooled[f"lag{k}"] = round(float(vals.mean()), 4) if len(vals) else None
    ratio = (mean_n_eff / n_total if n_total else 0.0)
    report = {
        "stations": int(n_dom),
        "raw_test_rows": int(n_total),
        "effective_sample_size_est": round(mean_n_eff, 0),
        "effective_N_ratio": round(ratio, 4),
        "pooled_acf": pooled,
        "lags_explanation": "lags in 6h steps: 1=6h, 120=30d (full target overlap)",
        "n_eff_method": "n / (1 + 2*sum_{k=1..120} rho_k^+) (positive autocorrelation)",
    }
    ACF_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\naverage effective N per station: {mean_n_eff / n_dom:.0f}")
    print(f"effective test sample size:   {mean_n_eff:.0f}  (raw {n_total:,} rows)")


if __name__ == "__main__":
    main()