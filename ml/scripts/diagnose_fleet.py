"""Fleet-wide honest-forecast diagnosis: classifies every station's multi-step
forecast as reliable / directional-only / weak, using the calibrated recursive
held-out coverage.

Writes ``ml/artifacts/multistep_diagnosis.csv`` with one row per station.
"""

import json
from pathlib import Path

import numpy as np

from ml.models.xgboost_quantile import (
    ARTIFACTS_DIR,
    DEFAULT_PARQUET,
    _load_config,
    _load_models,
    get_station_series,
)
from ml.services.interval_calibration import _station_artifact_dir, diagnose_station

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "multistep_diagnosis.csv"


def station_dirs(root) -> list[Path]:
    dirs = []
    for d in root.iterdir():
        if (d / "features.json").is_file() and (d / "xgb_point.joblib").is_file():
            meta_file = d / "xgboost_metadata.json"
            if meta_file.is_file():
                dirs.append(d)
    return dirs


def main() -> None:
    rows = []
    for d in station_dirs(ARTIFACTS_DIR):
        meta = json.load(open(d / "xgboost_metadata.json"))
        station = meta.get("station")
        try:
            cfg = _load_config(d)
            models = _load_models(d)
            sdf = get_station_series(DEFAULT_PARQUET, cfg.get("station", station))
            diag = diagnose_station(cfg, models, sdf)
            rows.append(
                {
                    "station": station,
                    "label": diag["label"],
                    "reason": diag["reason"],
                    "coverage": round(diag["coverage"], 4),
                    "shallow_error": round(diag["shallow_error"], 4),
                    "gwl_span": round(diag["gwl_span"], 4),
                    "n_obs": diag["n_obs"],
                    "tail_half_width": round(diag["tail_half_width"], 4),
                    "half_width_at_horizon": round(diag["half_width_at_horizon"], 4),
                    "horizon_days": diag["horizon_days"],
                    "max_depth": diag["max_depth"],
                }
            )
            print(
                f"{station[:44]:46s} {diag['label']:12s} cov={diag['coverage']:.0%} "
                f"shallow={diag['shallow_error']:.2f} hw@{diag['horizon_days']:.0f}d={diag['half_width_at_horizon']:.2f}"
            )
        except Exception as e:  # noqa: BLE001
            rows.append({"station": station, "label": "error", "reason": str(e)[:120],
                         "coverage": "", "shallow_error": "", "gwl_span": "",
                         "n_obs": "", "tail_half_width": "", "half_width_at_horizon": "",
                         "horizon_days": "", "max_depth": ""})
            print(f"{station[:44]:46s} ERROR {e}")

    import pandas as pd

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(rows)} stations)")


if __name__ == "__main__":
    main()
