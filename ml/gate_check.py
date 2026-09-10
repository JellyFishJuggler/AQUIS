"""Gate check — verify forecasting regression baselines survive the merge.

Run from the ML root (ml pre-rename / ml post-rename):
    venv/bin/python gate_check.py
Exits 0 when every baseline matches, 1 otherwise.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"{'OK ' if ok else 'FAIL'} {name}  {detail}")


def main() -> None:
    honest = pd.read_csv(OUT / "honest_metrics.csv")
    xg = honest[honest["model"] == "xgb"].iloc[0]
    check("honest 30d stride RMSE = 2.338", round(xg["basis_stride_rmse"], 3) == 2.338,
          f"got {xg['basis_stride_rmse']:.4f}")

    scv = pd.read_csv(OUT / "spatial_cv_metrics.csv")
    check("spatial CV mean = 1.973", round(scv["rmse"].mean(), 3) == 1.973,
          f"got {scv['rmse'].mean():.4f}")
    check("spatial CV median = 1.819", round(scv["rmse"].median(), 3) == 1.819,
          f"got {scv['rmse'].median():.4f}")

    q = json.loads((MODELS / "quantile_calibration.json").read_text())
    check("calibrated coverage = 0.908", round(q["coverage_stride"], 3) == 0.908,
          f"got {q['coverage_stride']:.4f}")
    check("half-width median = 1.025", round(q["half_width_median_m"], 3) == 1.025,
          f"got {q['half_width_median_m']:.4f}")
    check("half-width p90 = 2.575", round(q["half_width_p90_m"], 3) == 2.575,
          f"got {q['half_width_p90_m']:.4f}")

    acf = json.loads((OUT / "residual_acf.json").read_text())
    check("eff-N = 6635", round(acf["effective_sample_size_est"]) == 6635,
          f"got {acf['effective_sample_size_est']:.0f}")
    check("lag1 ACF = 0.858", round(acf["pooled_acf"]["lag1"], 3) == 0.858,
          f"got {acf['pooled_acf']['lag1']:.4f}")

    bt = Path(OUT / "backtest_6h_summary.json")
    if bt.exists():
        s = json.loads(bt.read_text())
        check("recursive verified + recorded",
              s.get("no_overlap_windows") and s.get("no_future_observed_values")
              and "success_vs_direct" in s,
              f"rec 30d {s.get('best_30d_rmse_rec')} vs direct "
              f"{s.get('best_30d_rmse_direct_baseline')}")
        check("direct-30d stays production",
              s.get("best_30d_rmse_direct_baseline") is not None
              and s.get("best_30d_rmse_direct_baseline")
              < s.get("best_30d_rmse_rec"),
              "recursion did not beat direct at 30d")

    ok = all(o for _, o, _ in CHECKS)
    print(f"\nGATE {'PASS' if ok else 'FAIL'} — {sum(o for _, o, _ in CHECKS)}/{len(CHECKS)} checks")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()