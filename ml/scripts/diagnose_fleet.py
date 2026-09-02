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


if __name__ == "__main__":
    main()