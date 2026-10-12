"""Runtime backend-dispatch test (data/refresh/runtime.json -> traj_backend).

Verifies that flipping ``traj_backend`` routes /forecast to the DL engine, and
that the default/fallback stays on the incumbent XGBoost engine. The DL routing
assertions skip when torch is not installed (xgb-only deploy).
"""

import importlib.util
import json
import unittest
from pathlib import Path

from tests._helpers import has

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "data" / "refresh" / "runtime.json"
STATION = "ASHADHA PRATHMIK VIDYALAYA"
HAS_TORCH = importlib.util.find_spec("torch") is not None
DL_READY = HAS_TORCH and all(has(p) for p in (
    "models/traj_dl.pt", "models/traj_dl_config.json",
    "models/traj_dl_calibration.json", "models/traj_dl_reliability.json",
    "_dl.py", "_traj_dispatch.py"))


@unittest.skipUnless(has("_traj_dispatch.py") and RUNTIME.exists(), "dispatch/module missing")
class TestTrajectoryDispatch(unittest.TestCase):
    def setUp(self):
        self._orig = RUNTIME.read_text()

    def tearDown(self):
        RUNTIME.write_text(self._orig)

    def _set(self, backend):
        data = json.loads(self._orig)
        data["traj_backend"] = backend
        RUNTIME.write_text(json.dumps(data))

    def test_default_backend_is_xgb(self):
        from _traj_dispatch import traj_backend, active_trajectory_forecast
        self._set("xgb")
        self.assertEqual(traj_backend(), "xgb")
        out = active_trajectory_forecast(STATION)
        self.assertNotIn("error", out)
        self.assertIn("XGBoost", out["model"])

    @unittest.skipUnless(DL_READY, "torch/DL artifacts missing")
    def test_dl_flag_routes_to_gru(self):
        from _traj_dispatch import traj_backend, active_trajectory_forecast
        self._set("dl")
        self.assertEqual(traj_backend(), "dl")
        out = active_trajectory_forecast(STATION)
        self.assertNotIn("error", out)
        self.assertIn("GRU", out["model"])

    def test_unknown_backend_falls_back_to_xgb(self):
        from _traj_dispatch import traj_backend, active_trajectory_forecast
        self._set("banana")
        self.assertEqual(traj_backend(), "banana")
        out = active_trajectory_forecast(STATION)
        self.assertNotIn("error", out)
        self.assertIn("XGBoost", out["model"])


if __name__ == "__main__":
    unittest.main()
