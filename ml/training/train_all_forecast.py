"""
Batch-train XGBoost + quantile models for every station (fast).

Resume-safe via ``artifacts/xgb_training_progress.json``; failures logged to
``artifacts/xgb_failed_stations.json``.

Usage:
    python -m ml.training.train_all_forecast
    python -m ml.training.train_all_forecast --force
    python -m ml.training.train_all_forecast --limit 20
    python -m ml.training.train_all_forecast --station "Asafpur (UP-077)"
    python -m ml.training.train_all_forecast --no-seasonal
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

import pandas as pd  # noqa: E402

from ml.models.xgboost_quantile import train_xgb_quantile_for_station  # noqa: E402
from ml.preprocessing.timeseries import station_slug  # noqa: E402
from ml.scripts.export_xgboost_models import export_bundle  # noqa: E402
from ml.utils import format_duration  # noqa: E402

ARTIFACTS = _ML_ROOT / "artifacts"
PARQUET = _ML_ROOT / "data" / "processed" / "common.parquet"
PROGRESS_PATH = ARTIFACTS / "xgb_training_progress.json"
FAILED_PATH = ARTIFACTS / "xgb_failed_stations.json"
BUNDLE_PATH = ARTIFACTS / "xgboost_bundle.joblib"


def load_progress() -> dict:
    if PROGRESS_PATH.is_file():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed": []}


def save_progress(progress: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def load_failures() -> list[dict]:
    if FAILED_PATH.is_file():
        with open(FAILED_PATH) as f:
            return json.load(f)
    return []


def save_failures(failures: list[dict]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(FAILED_PATH, "w") as f:
        json.dump(failures, f, indent=2)


def discover_stations() -> list[str]:
    df = pd.read_parquet(PARQUET)
    counts = df.groupby("Station").size().sort_values(ascending=False)
    return list(counts.index)


def eta_str(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "calculating..."
    remaining = total - done
    avg = elapsed / done
    return format_duration(avg * remaining)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-train XGBoost quantile models for all stations"
    )
    parser.add_argument(
        "--force", action="store_true", help="Retrain all stations, ignoring progress"
    )
    parser.add_argument(
        "--station", type=str, default=None, help="Train only this station"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N total completed stations (including pre-existing)",
    )
    parser.add_argument(
        "--use-seasonal", dest="use_seasonal", action="store_true", default=True,
        help="Include seasonal features (default)",
    )
    parser.add_argument(
        "--no-seasonal", dest="use_seasonal", action="store_false",
        help="Use time_hours only",
    )
    parser.add_argument(
        "--use-lags", dest="use_lags", action="store_true", default=True,
        help="Include causal lag/rolling features (default)",
    )
    parser.add_argument(
        "--no-lags", dest="use_lags", action="store_false",
        help="Omit lag/rolling features (spec-minimal feature set)",
    )
    args = parser.parse_args()

    stations = discover_stations()
    total = len(stations)

    progress = load_progress() if not args.force else {"completed": []}
    failures = [] if args.force else load_failures()
    completed_set = set(progress["completed"])

    if args.station:
        stations = [s for s in stations if s == args.station]
        if not stations:
            print(f"Station '{args.station}' not found in dataset.")
            sys.exit(1)
        total = 1

    remaining = [s for s in stations if s not in completed_set]

    if args.limit and not args.station:
        slots_left = args.limit - len(completed_set)
        if slots_left <= 0:
            print(f"Already at/above limit ({len(completed_set)}/{args.limit}). Use --force to retrain.")
            return
        remaining = remaining[:slots_left]

    print("=" * 60)
    print("  XGBoost + Quantile Batch Trainer")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    print(f"  Total stations:  {total}")
    print(f"  Already done:    {len(completed_set)}")
    print(f"  To train:        {len(remaining)}")
    print(f"  Features:        {'seasonal' if args.use_seasonal else 'time_hours only'}"
          f"{' + lags/rolling' if args.use_lags else ''}")
    if args.limit:
        print(f"  Limit:           {args.limit} total")
    if args.force:
        print("  Mode:            FORCE (retraining all)")
    print("=" * 60)
    print()

    if not remaining:
        print("Nothing to do. Use --force to retrain all.")
        if not BUNDLE_PATH.is_file():
            print("Bundle missing; exporting from existing artifacts...")
            export_bundle()
        return

    batch_start = time.monotonic()
    batch_done = 0
    batch_failed = 0
    graceful_stop = False

    def _handle_signal(sig, frame):
        nonlocal graceful_stop
        graceful_stop = True
        print(f"\n\nReceived signal {sig}. Finishing current station then exiting...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    for i, station in enumerate(remaining, 1):
        if graceful_stop:
            print("Stopping before next station (graceful exit).")
            break

        if args.limit and len(completed_set) >= args.limit:
            print(f"Reached limit of {args.limit} stations.")
            break

        idx = len(completed_set) + batch_done + 1
        header = f"[{idx}/{total}] {station}"
        print(header)
        print("-" * len(header))

        station_start = time.monotonic()
        try:
            result = train_xgb_quantile_for_station(
                station,
                use_seasonal=args.use_seasonal,
                use_lags=args.use_lags,
                verbose=False,
            )
        except Exception as exc:
            elapsed = time.monotonic() - station_start
            batch_failed += 1
            failure_record = {
                "station": station,
                "slug": station_slug(station),
                "error": str(exc),
                "traceback": str(exc.__class__.__name__),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed": format_duration(elapsed),
            }
            failures.append(failure_record)
            save_failures(failures)
            print(f"  FAILED in {format_duration(elapsed)}: {exc}\n")
            continue

        elapsed = time.monotonic() - station_start
        batch_done += 1

        pm = result["point_metrics"]
        qm = result["quantile_metrics"]
        print(f"  Done in {format_duration(elapsed)}")
        print(
            f"  RMSE: {pm['rmse']:.4f}  MAE: {pm['mae']:.4f}  R²: {pm['r2']:.4f}  "
            f"cover: {qm['coverage_90']:.1%}"
        )
        print()

        completed_set.add(station)
        progress["completed"] = list(completed_set)
        save_progress(progress)

        elapsed_total = time.monotonic() - batch_start
        done_total = batch_done + batch_failed
        print(
            f"  Progress: {done_total}/{len(remaining)} done"
            f" ({batch_done} ok, {batch_failed} failed)"
            f"  |  Elapsed: {format_duration(elapsed_total)}"
            f"  |  ETA: {eta_str(elapsed_total, done_total, len(remaining))}\n"
        )

    elapsed_total = time.monotonic() - batch_start
    print("=" * 60)
    print("  Batch complete")
    print(f"  Trained:  {batch_done}")
    print(f"  Failed:   {batch_failed}")
    print(f"  Skipped:  {len(remaining) - batch_done - batch_failed}")
    print(f"  Time:     {format_duration(elapsed_total)}")
    print("=" * 60)

    if failures:
        print(f"\nFailed stations ({len(failures)}):")
        for f in failures:
            print(f"  {f['station']}: {f['error']}")

    if batch_done or not BUNDLE_PATH.is_file():
        print("\nRefreshing model bundle...")
        try:
            export_bundle()
        except Exception as exc:
            print(f"  Bundle export failed: {exc}")
    else:
        print("\nNo stations trained; bundle left as-is.")


if __name__ == "__main__":
    main()