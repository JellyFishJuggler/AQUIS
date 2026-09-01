"""Batch train all 93 stations with resume support."""

import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    station_dirs,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    full_pipeline,
)
from ml.training.train_forecast import main as train_station  # noqa: E402


PROGRESS_FILE = ARTIFACTS_DIR / "train_all_progress.json"
FAILURES_FILE = ARTIFACTS_DIR / "train_all_failures.json"


def save_progress(done: list[str], failed: list[dict]) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done": done, "failed": failed, "timestamp": time.time()}, f)


def load_progress() -> tuple[list[str], list[dict]]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        return data.get("done", []), data.get("failed", [])
    return [], []


def train_one_station(args: tuple) -> tuple[str, str | None]:
    """Wrapper for parallel training."""
    slug, parquet_path, backend_csv, artifacts_root = args
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "ml.training.train_forecast",
        "--station", slug,
        "--parquet", str(parquet_path),
        "--backend", str(backend_csv),
        "--artifacts", str(artifacts_root),
    ], capture_output=True, text=True, cwd=_ML_ROOT.parent)
    if result.returncode == 0:
        return slug, None
    return slug, result.stderr[:500]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Batch train all stations")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--backend", type=Path, default=None)
    parser.add_argument("--station", nargs="+", help="Specific stations to train")
    args = parser.parse_args()

    artifacts_root = args.artifacts or ARTIFACTS_DIR
    parquet_path = args.parquet or _ML_ROOT.parent / "ml" / "data" / "processed" / "common.parquet"
    backend_csv = args.backend or _ML_ROOT.parent / "back-end" / "db" / "data.csv"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    all_dirs = station_dirs(artifacts_root)
    if args.station:
        all_dirs = [d for d in all_dirs if d.name in args.station]

    done, failed = load_progress()
    print(f"Resuming: {len(done)} done, {len(failed)} failed")

    remaining = [d for d in all_dirs if d.name not in done]
    total = len(all_dirs)
    print(f"Total: {total}, Remaining: {len(remaining)}")

    if args.workers > 1 and len(remaining) > 1:
        print(f"Using {args.workers} parallel workers...")
        train_args = [(d.name, parquet_path, backend_csv, artifacts_root) for d in remaining]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(train_one_station, a): a[0] for a in train_args}
            for i, future in enumerate(as_completed(futures)):
                slug = futures[future]
                try:
                    result_slug, error = future.result()
                    if error:
                        failed.append({"slug": result_slug, "reason": error})
                        print(f"[{i+1}/{len(remaining)}] {result_slug} FAILED: {error}")
                    else:
                        done.append(result_slug)
                        print(f"[{i+1}/{len(remaining)}] {result_slug} OK")
                except Exception as e:
                    failed.append({"slug": slug, "reason": str(e)})
                    print(f"[{i+1}/{len(remaining)}] {slug} EXCEPTION: {e}")

                save_progress(done, failed)
    else:
        print("Running sequentially...")
        for i, out_dir in enumerate(remaining):
            slug = out_dir.name
            print(f"[{i+1}/{len(remaining)}] Training {slug}...", flush=True)
            start = time.time()
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, "-m", "ml.training.train_forecast",
                    "--station", slug,
                    "--parquet", str(parquet_path),
                    "--backend", str(backend_csv),
                    "--artifacts", str(artifacts_root),
                ], capture_output=True, text=True, cwd=_ML_ROOT.parent)
                elapsed = time.time() - start
                if result.returncode == 0:
                    done.append(slug)
                    print(f"  OK ({elapsed:.1f}s)")
                else:
                    failed.append({"slug": slug, "reason": result.stderr[:500]})
                    print(f"  FAILED: {result.stderr[:200]}")
            except Exception as e:
                failed.append({"slug": slug, "reason": str(e)})
                print(f"  EXCEPTION: {e}")

            save_progress(done, failed)

    print(f"\n=== SUMMARY ===")
    print(f"Total: {total}")
    print(f"Successful: {len(done)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failures saved to {FAILURES_FILE}")
        with open(FAILURES_FILE, "w") as f:
            json.dump(failed, f, indent=2)


if __name__ == "__main__":
    main()