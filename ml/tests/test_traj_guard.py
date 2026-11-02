"""DL guard tests (_traj_dispatch.py).

With ``traj_backend=dl`` a GRU load/scoring failure must surface as a degraded
error payload instead of silently serving XGBoost. The XGB fallback stays
available only when it is explicitly requested or opted into per call.
"""

import json
import unittest

from tests._helpers import has

ROOT_RUNTIME = "data/refresh/runtime.json"


@unittest.skipUnless(has("_traj_dispatch.py") and has(ROOT_RUNTIME),
                     "dispatch module / runtime.json missing")
class TestDlGuard(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        self.runtime = Path(__file__).resolve().parent.parent / ROOT_RUNTIME
        self._orig = self.runtime.read_text()
        self._set("dl")

    def tearDown(self):
        self.runtime.write_text(self._orig)

    def _set(self, backend):
        data = json.loads(self._orig)
        data["traj_backend"] = backend
        self.runtime.write_text(json.dumps(data))

    def test_dl_failure_returns_degraded_error(self):
        import _dl
        import _traj_dispatch
        import _trajectory

        def boom(station):
            raise RuntimeError("torch exploded")

        called = {"xgb": 0}

        def xgb_stub(station):
            called["xgb"] += 1
            return {"model": "XGBoost", "trajectory": []}

        orig_dl, orig_xgb = _dl.trajectory_forecast, _trajectory.trajectory_forecast
        _dl.trajectory_forecast = boom
        _trajectory.trajectory_forecast = xgb_stub
        try:
            out = _traj_dispatch.active_trajectory_forecast("ANY STATION")
        finally:
            _dl.trajectory_forecast = orig_dl
            _trajectory.trajectory_forecast = orig_xgb

        self.assertIn("error", out)
        self.assertEqual(out["reason"], "backend_unavailable")
        self.assertTrue(out["degraded"])
        self.assertEqual(out["backend"], "dl")
        self.assertIn("torch exploded", out["error"])
        self.assertEqual(called["xgb"], 0, "guard must not fall back to xgb")

    def test_explicit_opt_in_falls_back(self):
        import _dl
        import _traj_dispatch
        import _trajectory

        def boom(station):
            raise RuntimeError("torch exploded")

        def xgb_stub(station):
            return {"model": "XGBoost", "trajectory": []}

        orig_dl, orig_xgb = _dl.trajectory_forecast, _trajectory.trajectory_forecast
        _dl.trajectory_forecast = boom
        _trajectory.trajectory_forecast = xgb_stub
        try:
            out = _traj_dispatch.active_trajectory_forecast("ANY STATION",
                                                            allow_fallback=True)
        finally:
            _dl.trajectory_forecast = orig_dl
            _trajectory.trajectory_forecast = orig_xgb

        self.assertNotIn("error", out)
        self.assertIn("XGBoost", out["model"])

    def test_backend_status_marks_degraded(self):
        import _traj_dispatch
        orig = _traj_dispatch.dl_probe
        _traj_dispatch.dl_probe = lambda: {"ok": False, "error": "no torch"}
        try:
            st = _traj_dispatch.backend_status()
        finally:
            _traj_dispatch.dl_probe = orig
        self.assertEqual(st["requested"], "dl")
        self.assertEqual(st["backend"], "dl")
        self.assertTrue(st["degraded"])
        self.assertEqual(st["error"], "no torch")

    def test_backend_status_healthy(self):
        import _traj_dispatch
        orig = _traj_dispatch.dl_probe
        _traj_dispatch.dl_probe = lambda: {"ok": True, "error": None}
        try:
            st = _traj_dispatch.backend_status()
        finally:
            _traj_dispatch.dl_probe = orig
        self.assertFalse(st["degraded"])
        self.assertIsNone(st["error"])


@unittest.skipUnless(has("api.py"), "api module missing")
class TestStatusEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()

    def test_health_reports_active_backend(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn(body["forecast_backend"], ("dl", "xgb"))
        self.assertIn("forecast_backend_requested", body)
        self.assertIn("forecast_degraded", body)
        self.assertIn("forecast_backend_error", body)

    @unittest.skipUnless(has("outputs/fleet_forecast.csv"), "fleet scan missing")
    def test_scan_reports_active_backend(self):
        body = self.client.get("/fleet/scan").get_json()
        self.assertIn(body.get("traj_backend"), ("dl", "xgb"))
        self.assertIn("traj_backend_requested", body)
        self.assertIn("degraded", body)
        self.assertIn("backend_error", body)


if __name__ == "__main__":
    unittest.main()
