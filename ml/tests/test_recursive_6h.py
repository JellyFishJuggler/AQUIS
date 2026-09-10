"""Recursive 6-hourly trajectory experiment (14-18) contract tests.

Data-gated on the experiment artifacts: 6h feature config, the +6h models,
the driver climatology and the honest backtest outputs. The engine must keep
its causal contract (no future observed values, ordered quantile paths, 6h
spacing) and the backtest summary must record the decision transparently.
"""

import unittest

import pandas as pd
import numpy as np

from tests._helpers import ROOT, has

REQ = ("models/6h_feature_config.json",
       "models/6h_xgb_point.joblib",
       "models/6h_xgb_q05.joblib",
       "models/6h_xgb_q50.joblib",
       "models/6h_xgb_q95.joblib",
       "models/backtest_6h_calibration.json",
       "outputs/backtest_6h_summary.json",
       "outputs/backtest_6h_metrics.csv")


@unittest.skipUnless(all(has(p) for p in REQ), "recursive 6h experiment artifacts missing")
class TestBacktestDecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.summary = json.loads(
            (ROOT / "outputs" / "backtest_6h_summary.json").read_text())
        cls.metrics = pd.read_csv(ROOT / "outputs" / "backtest_6h_metrics.csv")
        cls.calib = json.loads(
            (ROOT / "models" / "backtest_6h_calibration.json").read_text())

    def test_honest_backtest_flags(self):
        self.assertTrue(self.summary["no_overlap_windows"])
        self.assertTrue(self.summary["no_future_observed_values"])

    def test_decision_recorded_and_recursion_not_promoted(self):
        # direct-30d production model stays the headline: recursion must not
        # have beaten it at the 30 d horizon under the same anchors.
        self.assertIsNotNone(self.summary.get("best_30d_rmse_direct_baseline"))
        self.assertFalse(self.summary["success_vs_direct"])
        self.assertLess(self.summary["best_30d_rmse_direct_baseline"],
                        self.summary["best_30d_rmse_rec"])

    def test_all_seven_horizons_reported(self):
        self.assertEqual(sorted(self.metrics["horizon_h"].tolist()),
                         [6, 12, 24, 72, 168, 360, 720])
        self.assertTrue((self.metrics["n"] > 100).all(), "each horizon needs anchors")

    def test_recursive_beats_persistence_at_short_horizon(self):
        row = self.metrics[self.metrics["horizon_h"] == 6].iloc[0]
        self.assertLess(row["rmse_rec"], row["rmse_persist"])

    def test_calibration_hits_target_coverage(self):
        for h, s, cov in zip(self.calib["horizons_h"], self.calib["widening_s"],
                             self.calib["coverage_calibrated"]):
            self.assertGreaterEqual(cov, 0.88, f"horizon {h}h calibration short")


@unittest.skipUnless(all(has(p) for p in REQ[:-2]), "recursive 6h model artifacts missing")
class TestRecursiveTrajectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib
        R = importlib.import_module("17_recursive")
        cls.engine = R
        cls.out = R.recursive_trajectory("ASHADHA PRATHMIK VIDYALAYA",
                                         scenario="climatology", steps=120)
        assert "error" not in cls.out, cls.out.get("error")

    def test_horizon_is_30_days(self):
        self.assertEqual(len(self.out["gwl_q50"]), 120)
        self.assertEqual(self.out["horizon_hours"][-1], 720)

    def test_six_hourly_spacing(self):
        t = [pd.Timestamp(x) for x in self.out["times"]]
        for a, b in zip(t[:-1], t[1:]):
            self.assertEqual(b - a, pd.Timedelta(hours=6))

    def test_quantile_ordering(self):
        for lo, med, hi in zip(self.out["gwl_q05"], self.out["gwl_q50"],
                               self.out["gwl_q95"]):
            self.assertLessEqual(lo, med)
            self.assertLessEqual(med, hi)

    def test_anchor_is_observed_and_trajectory_starts_there(self):
        from _utils import load_table_6h
        t = load_table_6h()
        g = t[t["Station"] == "ASHADHA PRATHMIK VIDYALAYA"].sort_values("time")
        last = g.dropna(subset=["gwl"])["gwl"].iloc[-1]
        self.assertAlmostEqual(self.out["anchor"], float(last), places=3)
        # one-step median delta is small: trajectory must start at the anchor
        self.assertLess(abs(self.out["gwl_q50"][0] - self.out["anchor"]), 1.5)

    def test_reliability_flag_present(self):
        self.assertIn("reliable", self.out)
        self.assertIn("scenario_meta", self.out)

    def test_unknown_station_returns_error(self):
        r = self.engine.recursive_trajectory("NOT A REAL STATION")
        self.assertIn("error", r)


if __name__ == "__main__":
    unittest.main()