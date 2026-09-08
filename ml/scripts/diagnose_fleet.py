"""Fleet-wide diagnosis: calibrate and classify all 93 stations."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    load_models,
    station_dirs,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    full_pipeline,
)
from ml.services.interval_calibration import (  # noqa: E402
    diagnose_station,
    estimate_calibration,
)
from ml.services.interpretability import (  # noqa: E402
    residual_autocorrelation,
)

DIAGNOSIS_FILE = ARTIFACTS_DIR / "multistep_diagnosis.csv"
PROGRESS_FILE = ARTIFACTS_DIR / "diagnose_progress.json"
FAILURES_FILE = ARTIFACTS_DIR / "diagnose_failures.json"


def save_progress(done: list[str], failed: list[dict]) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done": done, "failed": failed, "timestamp": time.time()}, f)


def load_progress() -> tuple[list[str], list[dict]]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        return data.get("done", []), data.get("failed", [])
    return [], []


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    done, failed = load_progress()
    print(f"Resuming: {len(done)} done, {len(failed)} failed")

    pipe = full_pipeline(
        _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet",
        _ML_ROOT.parent / "back-end" / "db" / "data.csv",
    )
    feature_cols = pipe["feature_cols"]
    full_train_df = pipe["train"]
    full_test_df = pipe["test"]

    all_station_dirs = station_dirs()
    total = len(all_station_dirs)
    print(f"Total stations: {total}")

    results = []

    if DIAGNOSIS_FILE.exists():
        existing = pd.read_csv(DIAGNOSIS_FILE)
        results = existing.to_dict("records")

    for i, out_dir in enumerate(all_station_dirs):
        slug = out_dir.name
        if slug in done:
            continue

        print(f"[{i+1}/{total}] Diagnosing {slug}...", flush=True)
        start = time.time()

        try:
            cfg = {}
            models = load_models(out_dir)

            station_df = full_train_df[full_train_df["slug"] == slug].copy()
            if station_df.empty:
                print(f"  No training data for {slug}, skipping")
                failed.append({"slug": slug, "reason": "no training data"})
                save_progress(done, failed)
                continue

            diag = diagnose_station(cfg, models, station_df, feature_cols, test_df=full_test_df[full_test_df["slug"] == slug])
            diag["slug"] = slug
            results.append(diag)
            done.append(slug)

            elapsed = time.time() - start
            print(f"  -> {diag['label'].upper()} (coverage={diag['coverage']:.3f}, 1-step R2={diag['one_step_r2']:.3f}, multi R2={diag['multi_step_r2']:.3f}) in {elapsed:.1f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append({"slug": slug, "reason": str(e)})

        save_progress(done, failed)

    df = pd.DataFrame(results)
    df.to_csv(DIAGNOSIS_FILE, index=False)
    print(f"\nSaved diagnosis for {len(results)} stations to {DIAGNOSIS_FILE}")

    if results:
        labels = pd.Series([r["label"] for r in results]).value_counts()
        print("\nLabel distribution:")
        for lbl, cnt in labels.items():
            print(f"  {lbl}: {cnt}")
        print(f"\nMedian calibrated coverage: {df['coverage'].median():.4f}")
        print(f"Median one-step R2: {df['one_step_r2'].median():.4f}")
        print(f"Median multi-step R2: {df['multi_step_r2'].median():.4f}")

        if "gwl_span" in df.columns:
            n_flat = int((df.get("gwl_span", pd.Series([0.0])) < 0.75).sum())
            print(f"Flat wells (gwl_span<0.75 m, R2 NOT meaningful): {n_flat}/{len(df)}")
            flat = df.loc[~df.get("r2_meaningful", pd.Series([False] * len(df))), "one_step_nrmse"]
            if len(flat):
                print(f"Median one-step NRMSE on flat wells: {flat.median():.4f}")

    if failed:
        with open(FAILURES_FILE, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\nFailures ({len(failed)}): saved to {FAILURES_FILE}")

    # ---- Fleet spatial-error autocorrelation (paper's residual Moran's I) ----
    write_fleet_spatial_error(df)

    # ---- Fleet model comparison: XGBoost vs Linear baseline ----
    write_fleet_model_comparison(df)


def write_fleet_model_comparison(df: pd.DataFrame) -> None:
    """Persist + summarise the XGBoost-vs-Linear one-step comparison fleet-wide."""
    if df.empty or "linear_one_step_r2" not in df.columns:
        return
    cols = ["slug", "one_step_rmse", "one_step_mae", "one_step_r2",
            "linear_one_step_rmse", "linear_one_step_mae", "linear_one_step_r2"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["xgb_wins_rmse"] = out["one_step_rmse"] < out["linear_one_step_rmse"]
    out_path = ARTIFACTS_DIR / "model_comparison.csv"
    out.to_csv(out_path, index=False)
    n = len(out)
    xgb_win = int(out["xgb_wins_rmse"].sum()) if "xgb_wins_rmse" in out else 0
    print("\nFleet model comparison (one-step, held-out test):")
    print(f"  Median XGB R² : {out['one_step_r2'].median():.4f}   (RMSE {out['one_step_rmse'].median():.4f})")
    print(f"  Median Lin R² : {out['linear_one_step_r2'].median():.4f}   (RMSE {out['linear_one_step_rmse'].median():.4f})")
    print(f"  XGBoost beats Linear RMSE on {xgb_win}/{n} stations ({100*xgb_win/n:.0f}%)")
    print(f"  Saved per-station comparison to {out_path}")


def write_fleet_spatial_error(df: pd.DataFrame) -> None:
    """Compute and persist spatial autocorrelation of per-station model error.

    Uses each station's one-step error magnitude (one_step_nrmse) mapped to its
    (Longitude, Latitude). A significant positive Moran's I means model error
    clusters in space — a signal that spatial structure is under-modelled.
    """
    import pyarrow.parquet as pq
    try:
        if df.empty or "slug" not in df.columns or "one_step_nrmse" not in df.columns:
            return
        meta = pq.ParquetFile(
            _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet"
        ).read(columns=["Station", "Agency", "Latitude", "Longitude"]).to_pandas()
        from ml.preprocessing.timeseries import station_slug
        meta["slug"] = meta.apply(
            lambda r: station_slug(r["Station"], r["Agency"], 0), axis=1
        )
        m = meta.dropna(subset=["Latitude", "Longitude"]).drop_duplicates("slug")
        joined = df[["slug", "one_step_nrmse"]].merge(
            m[["slug", "Longitude", "Latitude"]], on="slug", how="inner"
        )
        if len(joined) < 8:
            return
        coords = joined[["Longitude", "Latitude"]].to_numpy(dtype=float)
        err = joined["one_step_nrmse"].to_numpy(dtype=float)
        report = residual_autocorrelation(err, coords, k=5, n_sim=999)
        report["n_stations"] = int(len(joined))
        with open(ARTIFACTS_DIR / "residual_autocorrelation.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\nFleet residual spatial autocorrelation (one_step_nrmse):")
        print(f"  Moran's I = {report['moran_I']:.4f} (p={report['moran_p']:.4f})")
        print(f"  Geary's C = {report['geary_C']:.4f} (p={report['geary_p']:.4f})")
    except Exception as e:  # noqa: BLE001 - additive diagnostic only
        print(f"\nFleet residual spatial autocorrelation skipped: {e}")


if __name__ == "__main__":
    main()