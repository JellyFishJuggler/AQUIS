"""
Package the trained per-station XGBoost models into one portable bundle.

The forecasting pipeline saves four model files plus ``features.json`` and
``xgboost_metadata.json`` per station under ``artifacts/<slug>/``.  This
script bundles all of them into a single ``xgboost_bundle.joblib`` so a
deployment (the Flask service, or a packaged artifact for the app backend)
can ship one file and predict for every station without touching the loose
per-station files.

Bundled predictions are bit-identical to the per-directory path because the
exact same prediction internals are reused (``_predict_for_times`` with the
per-station ``features.json`` config and observed buffer).

Usage:
    python -m ml.scripts.export_xgboost_models
    python -m ml.scripts.export_xgboost_models --verify
    python -m ml.scripts.export_xgboost_models --all --path artifacts/xgboost_bundle.joblib
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from ml.models.xgboost_quantile import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_PARQUET,
    POINT_FILES,
    POINT_MODEL_FILE,
    QUANTILES,
    SAMPLING_HOURS,
    _load_config,
    _observed_buffer,
    _predict_for_times,
    get_station_series,
    predict_xgb_quantile,
)

SCHEMA = 1
BUNDLE_FILE = "xgboost_bundle.joblib"
COMPRESS = 3


def _station_entry(out_dir: Path) -> dict:
    cfg = _load_config(out_dir)
    slug = out_dir.name
    with open(out_dir / "xgboost_metadata.json") as f:
        metadata = json.load(f)
    models = {"point": joblib.load(out_dir / POINT_MODEL_FILE)}
    models.update(
        {q: joblib.load(out_dir / POINT_FILES[q]) for q in QUANTILES}
    )
    return {
        "name": cfg.get("station", slug),
        "slug": slug,
        "features": cfg,
        "metadata": metadata,
        "models": models,
    }


def build_bundle(
    artifacts_root: Path | None = None,
    stations: list[str] | None = None,
) -> dict:
    """Collect every trained XGBoost station into a single dict."""
    root = Path(artifacts_root) if artifacts_root else ARTIFACTS_DIR
    entries = {}
    for dirpath in sorted(root.iterdir()):
        if not (dirpath / POINT_MODEL_FILE).is_file():
            continue
        if stations is not None and dirpath.name not in stations:
            continue
        entries[dirpath.name] = _station_entry(dirpath)

    if not entries:
        raise SystemExit(f"No XGBoost models found under {root}.")

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema": SCHEMA,
        "created_utc": created,
        "engine": "xgboost-quantile",
        "quantiles": QUANTILES,
        "sampling_hours": SAMPLING_HOURS,
        "count": len(entries),
        "stations": entries,
    }


def export_bundle(
    path: Path | None = None,
    artifacts_root: Path | None = None,
    stations: list[str] | None = None,
    verbose: bool = True,
) -> Path:
    """Build and dump the bundle; returns the bundle path."""
    root = Path(artifacts_root) if artifacts_root else ARTIFACTS_DIR
    bundle_path = Path(path) if path else root / BUNDLE_FILE
    bundle = build_bundle(root, stations=stations)
    joblib.dump(bundle, bundle_path, compress=COMPRESS)
    if verbose:
        size_mb = bundle_path.stat().st_size / 1024 / 1024
        print(
            f"Exported {bundle['count']} stations -> {bundle_path} "
            f"({size_mb:.1f} MB, schema {bundle['schema']})"
        )
    return bundle_path


def load_bundle(path: Path | None = None) -> dict:
    bundle_path = Path(path) if path else ARTIFACTS_DIR / BUNDLE_FILE
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"No bundle at {bundle_path}. Build it first with "
            "python -m ml.scripts.export_xgboost_models"
        )
    return joblib.load(bundle_path)


@lru_cache(maxsize=8)
def _series_for(parquet_path: str, station: str) -> pd.DataFrame:
    return get_station_series(parquet_path, station)


def _observed_buffer_for(parquet_path: str, station: str) -> dict:
    """Fresh observation buffer for a station.

    The recursive forecast fills this dict in place (``_fill_buffer_until``
    appends projected rows), so a brand-new copy is returned on every call to
    keep the cached series pristine and keep ``max(buffer)`` pinned to the
    last *real* reading.
    """
    station_df = _series_for(parquet_path, station)
    buffer = _observed_buffer(station_df)
    if not buffer:
        raise ValueError(f"No valid observations for station '{station}'.")
    return dict(buffer)


def bundle_predict(
    bundle: dict,
    slug: str,
    time_hours,
    parquet_path: str | Path | None = None,
) -> dict:
    """Point + 90% interval (q05..q95) for ``time_hours`` from the bundle.

    Mirrors ``predict_xgb_quantile`` but uses the in-memory models/config
    embedded in the bundle (cached observed buffer per station).
    """
    entry = bundle["stations"][slug]
    cfg = entry["features"]
    parquet = str(parquet_path) if parquet_path else str(DEFAULT_PARQUET)
    buffer = _observed_buffer_for(parquet, entry["name"])
    return _predict_for_times(cfg, buffer, time_hours, entry["models"])


def _first_times(buffer: dict, steps: int, points: int) -> list[float]:
    last = max(buffer)
    return [float((last + n) * SAMPLING_HOURS) for n in range(1, points + 1)] + [
        float((last + steps) * SAMPLING_HOURS)
    ]


def verify_bundle(
    path: Path | None = None,
    sample: int | None = 3,
    all_stations: bool = False,
    forward_steps: int = 60,
    check_points: int = 4,
) -> int:
    """Cross-check bundle predictions against the per-directory path."""
    bundle = load_bundle(path)
    slugs = sorted(bundle["stations"])
    if not all_stations and sample is not None:
        slugs = slugs[:sample]

    failures = 0
    t0 = time.monotonic()
    for slug in slugs:
        buffer = _observed_buffer_for(str(DEFAULT_PARQUET), bundle["stations"][slug]["name"])
        times = _first_times(buffer, forward_steps, check_points)
        got = bundle_predict(bundle, slug, times)
        want = predict_xgb_quantile(times, slug)
        ok = all(
            all(abs(g - w) < 1e-6 for g, w in zip(ga, wa))
            for ga, wa in [(got["point"], want["point"]), (got["lower"], want["lower"]), (got["upper"], want["upper"])]
        )
        if not ok:
            failures += 1
            print(f"  MISMATCH: {slug}")
        else:
            print(f"  ok: {slug} ({len(times)} horizons, incl. +{forward_steps * SAMPLING_HOURS}h)")
    elapsed = time.monotonic() - t0
    print(
        f"Verified {len(slugs)}/{bundle['count']} stations "
        f"({failures} mismatches) in {elapsed:.1f}s"
    )
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export / verify the XGBoost bundle")
    parser.add_argument(
        "--path", default=None, help="Bundle path (default artifacts/xgboost_bundle.joblib)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Include every station in the bundle"
    )
    parser.add_argument(
        "--station", action="append", default=None,
        help="Only bundle these slugs (repeatable). Default: all trained.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", action="store_true", help="Verify bundle vs per-dir predictions")
    group.add_argument("--verify-all", action="store_true", help="Verify every station in the bundle")
    args = parser.parse_args(argv)

    if args.verify or args.verify_all:
        failed = verify_bundle(args.path, all_stations=args.verify_all)
        return 1 if failed else 0

    bundle_path = export_bundle(args.path, stations=args.station)
    print(f"Produced {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())