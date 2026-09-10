"""33_traj_reliability - derive the per-horizon confidence/reliability tables.

The confidence LEVEL is evidence-based (NOT interval width alone). Evidence used
and persisted from the honest backtest + calibration + driver metadata:

  * interval_quality  : calibrated coverage at the horizon bucket (>= 0.80)
  * perf_vs_persist   : RMSE(traj) / RMSE(persistence) at the bucket
  * directional       : sign/directional accuracy at the bucket
  * width_scaled      : calibrated half-width vs horizon error budget (2*RMSE)

Station integrity, driver availability, anchor OOD and trajectory stability are
evaluated at inference in _trajectory.py using the rules/tables persisted here.

Output -> models/traj_reliability.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "outputs" / "traj_backtest_metrics.csv"
CALIB = ROOT / "models" / "traj_calibration.json"
OUT = ROOT / "models" / "traj_reliability.json"

COVERAGE_MIN = 0.80
DIR_GOOD = 0.62
DIR_MARGINAL = 0.55
PERF_TOL = 1.0


def main() -> None:
    cv = pd.read_csv(METRICS)
    cal = json.loads(CALIB.read_text())
    buckets = [int(x) for x in cal["horizons_h"]]
    cal_map = {b: c for b, c in zip(cal["horizons_h"], cal["coverage_calibrated"])}
    s_map = {b: s for b, s in zip(cal["horizons_h"], cal["widening_s"])}

    rules = {}
    for _, r in cv.iterrows():
        b = int(r["horizon_h"])
        perf_ratio = float(r["rmse_persist"]) / float(r["rmse"]) if r["rmse"] > 0 else np.nan
        cover = cal_map.get(b, np.nan)
        half = float(r["halfwidth_median"])
        width_ok = None
        try:
            err = float(r["rmse"])
            width_ok = bool(half <= 2.0 * err)
        except Exception:
            pass
        sa = float(r["sign_acc"]) if pd.notna(r["sign_acc"]) else np.nan
        rules[str(b)] = {
            "horizon_h": b,
            "n": int(r["n"]),
            "calibrated_coverage": round(cover, 4) if pd.notna(cover) else None,
            "widening_s": s_map.get(b),
            "rmse": round(float(r["rmse"]), 4),
            "rmse_persist": round(float(r["rmse_persist"]), 4),
            "perf_vs_persist_ratio": round(perf_ratio, 4) if np.isfinite(perf_ratio) else None,
            "sign_acc": round(sa, 4) if np.isfinite(sa) else None,
            "interval_quality": "ok" if (pd.notna(cover) and cover >= COVERAGE_MIN) else "poor",
            "width_scaled_ok": width_ok,
            "beats_persistence": bool(np.isfinite(perf_ratio) and perf_ratio >= PERF_TOL),
            "direction_supported": (
                bool(np.isfinite(sa) and sa >= DIR_GOOD) if pd.notna(sa) else False),
            "direction_marginal": (
                bool(np.isfinite(sa) and DIR_MARGINAL <= sa < DIR_GOOD)
                if pd.notna(sa) else False),
        }

    out = {
        "framework": "evidence-based confidence (not interval width alone)",
        "levels": ["HIGH", "DIRECTIONAL", "LOW"],
        "thresholds": {
            "coverage_min": COVERAGE_MIN,
            "direction_good": DIR_GOOD,
            "direction_marginal": DIR_MARGINAL,
            "perf_vs_persist_min": PERF_TOL,
            "horizon_row_score_high": 0.70,
            "horizon_row_score_directional": 0.45,
        },
        "bucket_rules": rules,
        "weights": {
            "interval_quality": 0.15, "perf_vs_persist": 0.20, "direction": 0.20,
            "width_scaled": 0.10, "station_integrity": 0.15, "driver_availability": 0.10,
            "anchor_ood": 0.05, "trajectory_stability": 0.05,
        },
        "runtime_note": ("station integrity, driver source (Open-Meteo<=16d forecast / "
                         "climatology beyond / unavailable), anchor OOD and trajectory "
                         "stability are evaluated at inference; a missing driver or "
                         "uncovered requirement must downgrade the horizon to "
                         "DIRECTIONAL/LOW."),
    }
    OUT.write_text(json.dumps(out, indent=2))

    print(f"wrote {OUT}")
    for b in buckets:
        r = rules[str(b)]
        beats = "Y" if r["beats_persistence"] else "N"
        print(f"  {b:>4}h  cov={r['calibrated_coverage']}  "
              f"ratio={r['perf_vs_persist_ratio']}  sign={r['sign_acc']}  "
              f"int={r['interval_quality']}  beats={beats}")


if __name__ == "__main__":
    main()