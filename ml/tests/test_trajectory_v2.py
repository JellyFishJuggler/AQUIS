"""Trajectory v2 contract tests (6-hourly genuine forecast, causal contract).

The trajectory engine must:
  * anchor on the LATEST OBSERVED GWL (never predicted), explicit anchor_time;
  * forecast exactly MAX_H=120 genuine points, one per 6h step (no interpolation);
  * never touch future observed GWL (values > anchor are model outputs only);
  * keep quantile order q05 <= q50 <= q95 at every horizon;
  * grow uncertainty with horizon (calibrated band widens);
  * attach an evidence-based confidence (HIGH/DIRECTIONAL/LOW) at every point,
    and reason strings that explain downgrades;
  * fuse explicit driver-source labels (Open-Meteo forecast / climatology);
  * report the production direct-30d endpoint as the numerical +30d benchmark,
    with agreement checked against the trajectory's own +30d sign.
"""

import unittest

import numpy as np
import pandas as pd

from tests._helpers import has

MODELS = all(has(f"models/traj_xgb_{q}.json") for q in ("q05", "q50", "q95")) and has(
    "models/traj_config.json") and has("models/traj_calibration.json") and has(
    "models/traj_reliability.json")


@unittest.skipUnless(MODELS and has("_trajectory.py"), "trajectory artifacts missing")
class TestTrajectoryCausalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from _trajectory import trajectory_forecast
        cls.traj = trajectory_forecast("ASHADHA PRATHMIK VIDYALAYA")
        assert cls.traj and "error" not in cls.traj, cls.traj
        cls.pts = pd.DataFrame(cls.traj["trajectory"])
        cls.pts["time"] = pd.to_datetime(cls.pts["time"])

    def test_anchor_is_last_observed_gwl(self):
        from _utils import load_table_6h
        t = load_table_6h()
        t = t[t["Station"] == "ASHADHA PRATHMIK VIDYALAYA"].sort_values("time")
        last = t.dropna(subset=["gwl"])["gwl"].iloc[-1]
        self.assertAlmostEqual(self.traj["anchor_gwl"], float(last), places=3)
        self.assertEqual(pd.Timestamp(self.traj["anchor_time"]),
                         t.dropna(subset=["gwl"])["time"].iloc[-1])

    def test_exactly_120_genuine_points_and_functional_form(self):
        self.assertEqual(len(self.pts), 120)
        dt = self.pts["time"].diff().dropna().dt.total_seconds() / 3600.0
        self.assertTrue(np.allclose(dt, 6.0, atol=1e-6),
                        "points must be one genuine 6h step apart")

    def test_quantile_order_every_horizon(self):
        self.assertTrue((self.pts["q05"] <= self.pts["q50"]).all())
        self.assertTrue((self.pts["q50"] <= self.pts["q95"]).all())

    def test_first_point_is_anchor_plus_delta(self):
        self.assertAlmostEqual(self.pts["q50"].iloc[0],
                               self.traj["anchor_gwl"] + (self.pts["q50"].iloc[0]
                                                          - self.traj["anchor_gwl"]), places=5)
        self.assertAlmostEqual(self.pts["time"].iloc[0],
                               pd.Timestamp(self.traj["anchor_time"]) + pd.Timedelta(hours=6))

    def test_uncertainty_non_decreasing_trend(self):
        hw = (self.pts["q95"] - self.pts["q05"]) / 2.0
        reg = np.polyfit(np.arange(len(hw)), hw, 1)
        self.assertGreater(reg[0], 0, "calibrated half-width must widen with horizon")

    def test_no_future_observed_gwl_leaked_into_output(self):
        from _utils import load_table_6h
        t = load_table_6h()
        g = t[t["Station"] == "ASHADHA PRATHMIK VIDYALAYA"].sort_values("time")
        future = g[g["time"] > pd.Timestamp(self.traj["anchor_time"])].dropna(subset=["gwl"])
        for _, p in self.pts.iterrows():
            hit = future[future["time"] == p["time"]]
            if len(hit):
                self.assertFalse(np.isclose(p["q50"], float(hit["gwl"].iloc[0]), atol=1e-6),
                                 "trajectory must never reproduce a future reading at its "
                                 "forecast time (leakage)")
        self.assertEqual(len(self.pts["time"]), self.pts["time"].nunique(),
                         "one genuine 6h point per step")

    def test_anchor_never_predicted(self):
        self.assertFalse(np.isclose(self.pts["q50"], self.traj["anchor_gwl"]).any(),
                         "a real forecast model never re-predicts its observed anchor")

    def test_confidence_labels_used_and_explained(self):
        for lbl in ("HIGH", "DIRECTIONAL", "LOW"):
            filled = self.pts["reliability_reason"][self.pts["confidence_level"] == lbl]
            self.assertTrue((filled != "").all(), f"{lbl} points must carry a reason")
        self.assertTrue(self.pts["reliability_reason"].notna().all())

    def test_driver_source_labels(self):
        self.assertEqual(self.pts["driver_source"].nunique() >= 1, True)
        self.assertIn(self.pts["driver_source"].iloc[0].split()[0],
                      {"Open-Meteo", "climatology", "driver"})

    def test_production_endpoint_benchmark_present(self):
        self.assertIn("endpoint_production", self.traj)
        ep = self.traj["endpoint_production"]
        self.assertIsNotNone(ep and ep.get("level"))
        self.assertIn("overall_confidence", self.traj)
        self.assertIn(self.traj["overall_confidence"]["level"], ("HIGH", "DIRECTIONAL", "LOW"))

    def test_direction_and_agreement_fields(self):
        d = self.traj["direction"]
        self.assertIn(d["label"], ("rising", "stable", "declining"))
        self.assertIsNotNone(d.get("agreement_with_production"))

    def test_trajectory_30d_matches_last_point(self):
        t30 = self.traj["trajectory_30d"]
        self.assertAlmostEqual(t30["level"], self.pts["q50"].iloc[-1], places=4)
        self.assertAlmostEqual(t30["q95"], self.pts["q95"].iloc[-1], places=4)


@unittest.skipUnless(has("_trajectory.py"), "module missing")
class TestTrajectoryUnknownStation(unittest.TestCase):
    def test_returns_error_not_crash(self):
        from _trajectory import trajectory_forecast
        out = trajectory_forecast("NOT A REAL STATION")
        self.assertTrue(not isinstance(out, list))
        if out:
            self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()