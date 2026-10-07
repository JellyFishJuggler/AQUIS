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
    # ---- production baseline constants --------------------------------------
    # The gate compares the on-disk artifacts against the PRODUCTION model's
    # honest 30-day stride baseline. Everything below is read from the
    # production artifacts (outputs/*, models/*) — NOT from candidate or
    # scratch re-fits (data/scratch_assoc/ablation_report.json) — so a re-fit
    # cannot silently lower the bar. `RECORDED` pins the baseline values that
    # the production artifacts reproduced on 2026-10-07; a changed value means
    # an explicit re-baselining decision, not an automatic pass.
    honest = pd.read_csv(OUT / "honest_metrics.csv")
    xg = honest[honest["model"] == "xgb"].iloc[0]
    prod_stride_rmse = float(xg["basis_stride_rmse"])

    scv = pd.read_csv(OUT / "spatial_cv_metrics.csv")
    prod_scv_mean = float(scv["rmse"].mean())
    prod_scv_median = float(scv["rmse"].median())

    q = json.loads((MODELS / "quantile_calibration.json").read_text())
    prod_cal_coverage = float(q["coverage_stride"])
    prod_half_med = float(q["half_width_median_m"])
    prod_half_p90 = float(q["half_width_p90_m"])

    acf = json.loads((OUT / "residual_acf.json").read_text())
    prod_eff_n = int(acf["effective_sample_size_est"])
    prod_pooled_lag1 = float(acf["pooled_acf"]["lag1"])

    recorded = {
        "stride_rmse": 2.418, "scv_mean": 1.973, "scv_median": 1.819,
        "cal_coverage": 0.893, "half_med": 1.099, "half_p90": 2.656,
        "eff_n": 6916, "lag1": 0.871,
    }
    check("honest 30d stride RMSE = 2.418",
          round(prod_stride_rmse, 3) == recorded["stride_rmse"],
          f"got {prod_stride_rmse:.4f}")

    check("spatial CV mean = 1.973",
          round(prod_scv_mean, 3) == recorded["scv_mean"],
          f"got {prod_scv_mean:.4f}")
    check("spatial CV median = 1.819",
          round(prod_scv_median, 3) == recorded["scv_median"],
          f"got {prod_scv_median:.4f}")

    check("calibrated coverage = 0.893",
          round(prod_cal_coverage, 3) == recorded["cal_coverage"],
          f"got {prod_cal_coverage:.4f}")
    check("half-width median = 1.099",
          round(prod_half_med, 3) == recorded["half_med"],
          f"got {prod_half_med:.4f}")
    check("half-width p90 = 2.656",
          round(prod_half_p90, 3) == recorded["half_p90"],
          f"got {prod_half_p90:.4f}")

    check("eff-N = 6916", round(prod_eff_n) == recorded["eff_n"],
          f"got {prod_eff_n:.0f}")
    check("lag1 ACF = 0.871", round(prod_pooled_lag1, 3) == recorded["lag1"],
          f"got {prod_pooled_lag1:.4f}")

    bt = Path(OUT / "backtest_6h_summary.json")
    if bt.exists():
        s = json.loads(bt.read_text())
        # Derived protocol expectation: 30-day windows (720 h) are only
        # overlap-free when anchors are spaced >= 720 h apart.
        expect_overlap_free = (
            int(s.get("anchor_spacing_days", 0)) * 24 >= 720)
        check("no_overlap_windows matches spacing-vs-horizon protocol",
              s.get("no_overlap_windows") is expect_overlap_free,
              f"flag {s.get('no_overlap_windows')} with "
              f"spacing {s.get('anchor_spacing_days')}d (expect "
              f"{expect_overlap_free}) — artifact predates the derived flag; "
              "re-run 32 to refresh" if s.get("no_overlap_windows")
              is not expect_overlap_free else "")
        check("recursive verified + recorded",
              s.get("no_future_observed_values")
              and "success_vs_direct" in s,
              f"rec 30d {s.get('best_30d_rmse_rec')} vs direct "
              f"{s.get('best_30d_rmse_direct_baseline')}")
        check("direct-30d stays production",
              s.get("best_30d_rmse_direct_baseline") is not None
              and s.get("best_30d_rmse_direct_baseline")
              < s.get("best_30d_rmse_rec"),
              "recursion did not beat direct at 30d")

    tj = Path(OUT / "traj_backtest_summary.json")
    if tj.exists():
        s = json.loads(tj.read_text())
        rt = s.get("30d_rmse_traj")
        check("trajectory promoted (honest, above baselines)",
              bool(s.get("promote_trajectory"))
              and rt is not None and rt
              < s.get("30d_rmse_persistence", float("inf"))
              and rt < s.get("30d_rmse_direct30_production", float("inf")),
              f"traj 30d {rt} vs persist {s.get('30d_rmse_persistence')} "
              f"vs direct {s.get('30d_rmse_direct30_production')}")
        tcal = Path(MODELS / "traj_calibration.json")
        if tcal.exists():
            c = json.loads(tcal.read_text())
            cov30 = c["coverage_calibrated"][-1] if c.get("coverage_calibrated") else None
            # Coverage must come from a DISJOINT half of the anchors; otherwise the
            # widening was fitted on the very points being scored. An artifact built
            # before the split existed has no `n_heldout`, so require it explicitly.
            split_ok = bool(c.get("n_heldout")) and bool(c.get("coverage_calibrated_fit"))
            check("trajectory coverage measured out-of-sample", split_ok,
                  "calibration artifact predates the fit/held-out split — rerun 32"
                  if not split_ok else f"n_heldout={c['n_heldout'][-1]}")
            check("trajectory +30d held-out coverage ~ 0.90",
                  cov30 is not None and abs(round(float(cov30), 2) - 0.90) <= 0.03,
                  f"got {cov30} (fit half {c.get('coverage_calibrated_fit', [None])[-1]})")

    ok = all(o for _, o, _ in CHECKS)
    print(f"\nGATE {'PASS' if ok else 'FAIL'} — {sum(o for _, o, _ in CHECKS)}/{len(CHECKS)} checks")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()