"""Backend-parity contract test: the DL trajectory engine must be a drop-in
sibling of the incumbent XGBoost engine.

Both backends must return the identical output schema and anchor on the same
observed row; only the numbers and the ``model`` label may differ. This guards
the runtime flag (data/refresh/runtime.json -> traj_backend) from ever serving
a shape the API/frontend cannot parse.

Skips cleanly when the DL artifacts are absent (fresh clone / xgb-only deploy).
"""

import unittest

import numpy as np

from tests._helpers import has

STATION = "ASHADHA PRATHMIK VIDYALAYA"
DL_READY = all(has(p) for p in (
    "models/traj_dl.pt", "models/traj_dl_config.json",
    "models/traj_dl_calibration.json", "models/traj_dl_reliability.json",
    "_dl.py", "_trajectory.py",
    "models/traj_xgb_q50.json", "models/traj_calibration.json",
    "models/traj_reliability.json",
))

_TOP_KEYS = {"station", "anchor_time", "anchor_gwl", "model", "trajectory",
             "trajectory_30d", "endpoint_production", "direction",
             "overall_confidence", "evidence"}
_POINT_KEYS = {"time", "gwl", "q05", "q50", "q95", "confidence_level",
               "reliability_reason", "driver_source"}


@unittest.skipUnless(DL_READY, "DL trajectory artifacts missing")
class TestTrajectoryBackendParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from _trajectory import trajectory_forecast as xgb_fc
        from _dl import trajectory_forecast as dl_fc
        cls.xgb = xgb_fc(STATION)
        cls.dl = dl_fc(STATION)
        assert "error" not in cls.xgb, cls.xgb
        assert "error" not in cls.dl, cls.dl

    def test_top_level_keys_match(self):
        self.assertEqual(set(self.xgb.keys()), set(self.dl.keys()), _TOP_KEYS)

    def test_anchor_identical(self):
        self.assertEqual(self.dl["anchor_time"], self.xgb["anchor_time"])
        self.assertAlmostEqual(self.dl["anchor_gwl"], self.xgb["anchor_gwl"], places=6)

    def test_trajectory_length_and_times_match(self):
        xa = self.xgb["trajectory"]
        da = self.dl["trajectory"]
        self.assertEqual(len(xa), 120)
        self.assertEqual(len(da), 120)
        self.assertEqual([p["time"] for p in xa], [p["time"] for p in da])

    def test_point_schema_matches(self):
        for p in self.dl["trajectory"]:
            self.assertEqual(set(p.keys()), _POINT_KEYS)
            self.assertLessEqual(p["q05"], p["q50"])
            self.assertLessEqual(p["q50"], p["q95"])
            self.assertIn(p["confidence_level"], ("HIGH", "DIRECTIONAL", "LOW"))
            self.assertTrue(p["reliability_reason"])

    def test_models_differ(self):
        self.assertNotEqual(self.xgb["model"], self.dl["model"])
        self.assertIn("GRU", self.dl["model"])

    def test_endpoint_schema_matches(self):
        xe, de = self.xgb["endpoint_production"], self.dl["endpoint_production"]
        if xe is None or de is None:
            self.assertEqual(xe, de)
        else:
            self.assertEqual(set(xe.keys()), set(de.keys()))

    def test_30d_block_schema_and_causal(self):
        for out in (self.xgb, self.dl):
            t30 = out["trajectory_30d"]
            self.assertEqual(set(t30.keys()),
                             {"times", "level", "q05", "q95", "change"})
            self.assertEqual(len(t30["times"]), 120)
            pts = out["trajectory"]
            self.assertAlmostEqual(t30["level"], pts[-1]["q50"], places=5)
            self.assertFalse(np.isclose(pts[-1]["q50"], out["anchor_gwl"]),
                             "trajectory must not re-predict its anchor")

    def test_confidence_fields_match_shape(self):
        for out in (self.xgb, self.dl):
            self.assertIn(out["overall_confidence"]["level"],
                          ("HIGH", "DIRECTIONAL", "LOW"))
            self.assertIn(out["direction"]["label"],
                          ("rising", "stable", "declining"))
            self.assertIn("evidence", out)


@unittest.skipUnless(DL_READY, "DL trajectory artifacts missing")
class TestTrajectoryDLUnknownStation(unittest.TestCase):
    def test_returns_error_not_crash(self):
        from _dl import trajectory_forecast
        out = trajectory_forecast("NOT A REAL STATION")
        self.assertTrue(not isinstance(out, list))
        if out:
            self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
