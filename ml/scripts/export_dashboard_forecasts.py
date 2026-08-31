"""
Export the dashboard forecast snapshot CSV from the trained XGBoost models.

The Streamlit dashboard renders two model-driven forecast panels per station;
on Streamlit Community Cloud the per-station weights may be unavailable, so
this script bakes every station's test-period curve and +90d recursive
projection into ONE committed CSV the dashboard can fall back to.

Usage:
    python -m ml.scripts.export_dashboard_forecasts
    python -m ml.scripts.export_dashboard_forecasts --verify
    python -m ml.scripts.export_dashboard_forecasts --station SLUG_1 SLUG_2
    python -m ml.scripts.export_dashboard_forecasts --path /tmp/x.csv
"""

import argparse
import sys
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT.parent))

from ml.services.forecast_snapshots import (  # noqa: E402
    SNAPSHOT_FILE,
    station_dirs,
    verify_snapshot,
    write_dashboard_forecasts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every trained station (default).",
    )
    parser.add_argument(
        "--station",
        nargs="+",
        metavar="SLUG",
        help="Export only the given station slugs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check snapshot-vs-live equality (sample of 5 stations).",
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="Check snapshot-vs-live equality for every station (slow).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip the export; only verify the existing snapshot CSV.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=f"Output CSV path (default: artifacts/{SNAPSHOT_FILE}).",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifacts root (default: ml/artifacts).",
    )
    args = parser.parse_args()

    csv_path = args.path or (_ML_ROOT / "artifacts" / SNAPSHOT_FILE)
    if not args.verify_only:
        write_dashboard_forecasts(csv_path, artifacts_root=args.artifacts, stations=args.station)

    do_verify = args.verify or args.verify_all or args.verify_only
    verify_stations = None
    if do_verify:
        if args.verify_all and args.station is None:
            verify_stations = None  # full sweep
        elif args.station is not None:
            verify_stations = args.station
        else:
            verify_stations = [d.name for d in station_dirs(args.artifacts)][:5]
            print(
                f"(sample verify on {len(verify_stations)} stations; "
                "--verify-all for the full sweep)"
            )

    if do_verify:
        failures = verify_snapshot(
            csv_path, artifacts_root=args.artifacts, stations=verify_stations
        )
        if failures:
            sys.exit(1)
        print("verify: OK")


if __name__ == "__main__":
    main()