"""
Batch-train GPR models for every station in the dataset.

Resume-safe, failure-tolerant, progress-logged.
Designed to run overnight.

Usage:
    python -m ml.training.train_all_gpr
    python -m ml.training.train_all_gpr --force
    python -m ml.training.train_all_gpr --station "Asafpur (UP-077)"
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

import pandas as pd

from ml.training.train_gpr import train_for_station
from ml.preprocessing.gpr import station_slug
from ml.utils import format_duration

ARTIFACTS = _ML_ROOT / "artifacts"
PARQUET = _ML_ROOT / "data" / "processed" / "common.parquet"
PROGRESS_PATH = ARTIFACTS / "training_progress.json"
FAILED_PATH = ARTIFACTS / "failed_stations.json"


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
    secs = avg * remaining
    return format_duration(secs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-train GPR for all stations")
    parser.add_argument(
        "--force", action="store_true", help="Retrain all stations, ignoring progress"
    )
    parser.add_argument(
        "--station", type=str, default=None, help="Train only this station"
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

    remaining = [s for s in stations if station_slug(s) not in completed_set]

    print("=" * 60)
    print(f"  GPR Batch Trainer")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    print(f"  Total stations:  {total}")
    print(f"  Already done:    {len(completed_set)}")
    print(f"  To train:        {len(remaining)}")
    if args.force:
        print(f"  Mode:            FORCE (retraining all)")
    print("=" * 60)
    print()

    if not remaining:
        print("Nothing to do. Use --force to retrain all.")
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

        slug = station_slug(station)
        idx = len(completed_set) + batch_done + 1
        header = f"[{idx}/{total}] {station}"

        print(header)
        print("-" * len(header))
        print(f"  Preprocessing...")

        station_start = time.monotonic()

        try:
            result = train_for_station(
                station,
                n_restarts=5,
                verbose=False,
            )
        except Exception as exc:
            elapsed = time.monotonic() - station_start
            batch_failed += 1
            failure_record = {
                "station": station,
                "slug": slug,
                "error": str(exc),
                "traceback": str(exc.__class__.__name__),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed": format_duration(elapsed),
            }
            failures.append(failure_record)
            save_failures(failures)

            print(f"  FAILED in {format_duration(elapsed)}: {exc}")
            print()
            continue

        elapsed = time.monotonic() - station_start
        batch_done += 1

        test_rmse = result["metrics"]["test"]["rmse"]
        test_r2 = result["metrics"]["test"]["r2"]
        test_mae = result["metrics"]["test"]["mae"]
        lml = result["log_marginal_likelihood"]

        print(f"  Done in {format_duration(elapsed)}")
        print(f"  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  R²: {test_r2:.4f}  LML: {lml:.2f}")
        print(f"  Artifacts: {result['artifact_dir']}")
        print()

        completed_set.add(slug)
        progress["completed"] = list(completed_set)
        save_progress(progress)

        elapsed_total = time.monotonic() - batch_start
        done_total = batch_done + batch_failed
        remaining_count = len(remaining) - done_total
        print(
            f"  Progress: {done_total}/{len(remaining)} done"
            f" ({batch_done} ok, {batch_failed} failed)"
            f"  |  Elapsed: {format_duration(elapsed_total)}"
            f"  |  ETA: {eta_str(elapsed_total, done_total, len(remaining))}"
        )
        print()

    # Summary
    elapsed_total = time.monotonic() - batch_start
    print("=" * 60)
    print(f"  Batch complete")
    print(f"  Trained:  {batch_done}")
    print(f"  Failed:   {batch_failed}")
    print(f"  Skipped:  {len(remaining) - batch_done - batch_failed}")
    print(f"  Time:     {format_duration(elapsed_total)}")
    if batch_done + batch_failed > 0:
        avg = elapsed_total / (batch_done + batch_failed)
        print(f"  Avg/station: {format_duration(avg)}")
    print("=" * 60)

    if failures:
        print(f"\nFailed stations ({len(failures)}):")
        for f in failures:
            print(f"  {f['station']}: {f['error']}")


if __name__ == "__main__":
    main()
