"""Verification 1: one-step vs multi-step (recursive) R²/RMSE.

One-step evaluation (stored in ``xgboost_metadata.json``) gives each test row
its *true* lagged observations as features.  Multi-step evaluation here seeds
the recursive buffer with only the training observations, then forecasts the
whole held-out test period step-by-step (each step uses the model's own
previous predictions for lags) - i.e. what actually happens when the model is
asked to forecast many timesteps ahead.

Usage (from repo root):
    python -m ml.scripts.verify_stepwise [--limit N] [--station "Name"]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_PARQUET,
    SAMPLING_HOURS,
    _load_config,
    _load_models,
    _position,
    _predict_for_times,
    split_series,
)
from ml.preprocessing.timeseries import GWL_COL, TIME_COL, get_station_series  # noqa: E402

OUT_CSV = _ML_ROOT / "artifacts" / "stepwise_comparison.csv"


def recursive_test_metrics(
    station: str,
    parquet_path: Path = DEFAULT_PARQUET,
    artifacts_root: Path | None = None,
) -> dict:
    out_dir = artifacts_root or ARTIFACTS_DIR
    for slug_dir in out_dir.iterdir():
        if (slug_dir / "features.json").is_file() and (
            slug_dir / "xgb_point.joblib"
        ).is_file():
            meta = json.load(open(slug_dir / "xgboost_metadata.json"))
            if meta.get("station") == station:
                break
    else:
        raise FileNotFoundError(f"No artifact dir for station {station!r}")
    cfg = _load_config(slug_dir)
    models = _load_models(slug_dir)

    station_df = get_station_series(parquet_path, cfg.get("station", station))
    _, X_test, _, y_test, _, t_test = split_series(
        station_df, use_seasonal=cfg["use_seasonal"], use_lags=cfg["use_lags"]
    )

    one_step = meta.get("point_metrics", {})
    one_q = meta.get("quantile_metrics", {}).get("q50_metrics", {})
    if not one_step:
        raise ValueError(f"no stored metrics for {station!r}")

    y = station_df[GWL_COL].to_numpy(dtype=float)
    t = pd.to_datetime(station_df[TIME_COL].to_numpy())
    hours = station_df["time_hours"].to_numpy()
    idx_by_ts = {ts: i for i, ts in enumerate(t)}
    max_train_t = t_test[0] - pd.Timedelta(hours=SAMPLING_HOURS)

    buffer: dict = {}
    for i, (row_t, row_y) in enumerate(zip(t, y)):
        if row_t <= max_train_t and not np.isnan(row_y):
            buffer[_position(hours[i])] = float(row_y)

    t_test_hours = [hours[idx_by_ts[ts]] for ts in t_test]

    res = _predict_for_times(cfg, buffer, t_test_hours, models)
    pred_point = res["point"]
    pred_lo = res["lower"]
    pred_hi = res["upper"]

    multi = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_point))),
        "mae": float(mean_absolute_error(y_test, pred_point)),
        "r2": float(r2_score(y_test, pred_point)),
        "q50_rmse": float(np.sqrt(mean_squared_error(y_test, pred_point))),
        "q50_mae": float(mean_absolute_error(y_test, pred_point)),
        "q50_r2": float(r2_score(y_test, pred_point)),
        "coverage_90": float(np.mean((y_test >= pred_lo) & (y_test <= pred_hi))),
        "mean_interval_width": float(np.mean(pred_hi - pred_lo)),
    }

    return {
        "station": station,
        "n_test": int(len(y_test)),
        "one_step_rmse": one_step["rmse"],
        "one_step_mae": one_step["mae"],
        "one_step_r2": one_step["r2"],
        "multi_step_rmse": multi["rmse"],
        "multi_step_mae": multi["mae"],
        "multi_step_r2": multi["r2"],
        "multi_coverage_90": multi["coverage_90"],
        "multi_interval_width": multi["mean_interval_width"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--station", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args()

    df = pd.read_parquet(DEFAULT_PARQUET)
    stations = sorted(df["Station"].unique().tolist())
    if args.station:
        stations = [s for s in stations if args.station.lower() in s.lower()]
    if args.limit:
        stations = stations[: args.limit]
    print(f"computing recursive metrics for {len(stations)} stations...")

    rows = []
    for i, s in enumerate(stations, 1):
        try:
            rows.append(recursive_test_metrics(s))
            print(f"  [{i}/{len(stations)}] {s}")
        except Exception as e:
            print(f"  [{i}/{len(stations)}] {s}  ERROR: {e}")
            rows.append({**{"station": s}, "error": str(e)})
        if i % 10 == 0:
            _write_csv(rows, args.out)

    _write_csv(rows, args.out)
    print(f"\nwrote {args.out}")


def _write_csv(rows: list[dict], out: str) -> None:
    with open(out, "w", newline="") as f:
        keys = [
            "station", "n_test",
            "one_step_rmse", "one_step_mae", "one_step_r2",
            "multi_step_rmse", "multi_step_mae", "multi_step_r2",
            "multi_coverage_90", "multi_interval_width", "error",
        ]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if k not in r else r[k]) for k in keys})


if __name__ == "__main__":
    main()