"""Forward-forecast contract tests (same path as the Forecast page / assistant)."""

import unittest

import pandas as pd
import numpy as np

from tests._helpers import ROOT, has


@unittest.skipUnless(has("_model.py") and has("models/xgb_multihorizon.joblib"),
                     "model artifacts missing")
class TestForwardForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import _model
        cls._model = _model
        cls.fc = _model.forward_forecast("ASHADHA PRATHMIK VIDYALAYA")
        assert cls.fc, "forward_forecast returned empty for a known station"

    def test_contract_keys(self):
        for k in ("station", "date_from", "date_to", "anchor", "pred_xgb",
                  "pred_ridge", "xgb_level", "ridge_level"):
            self.assertIn(k, self.fc)

    def test_horizon_is_30_days(self):
        self.assertEqual(pd.Timedelta(self.fc["date_to"] - self.fc["date_from"]),
                         pd.Timedelta(days=30))

    def test_level_is_anchor_plus_delta(self):
        self.assertAlmostEqual(self.fc["xgb_level"], self.fc["anchor"] + self.fc["pred_xgb"], places=5)

    def test_anchor_is_last_observed_gwl(self):
        from _utils import load_table_6h
        t = load_table_6h()
        g = t[t["Station"] == "ASHADHA PRATHMIK VIDYALAYA"].sort_values("time")
        last = g.dropna(subset=["gwl"])["gwl"].iloc[-1]
        self.assertAlmostEqual(self.fc["anchor"], float(last), places=3)

    def test_plausible_range(self):
        self.assertLess(abs(self.fc["pred_xgb"]), 12.0)

    def test_quantile_ordering_when_present(self):
        q = (self.fc.get("q05_level"), self.fc.get("q50_level"), self.fc.get("q95_level"))
        if all(v is not None for v in q):
            self.assertTrue(q[0] < q[1] < q[2], f"q05/q50/q95 ordering broken: {q}")

    def test_band_half_consistent(self):
        bh = self.fc.get("band_half")
        if bh is not None:
            lo, hi = self.fc["q05_level"], self.fc["q95_level"]
            self.assertAlmostEqual(bh, (hi - lo) / 2, places=5)


@unittest.skipUnless(has("_model.py"), "module missing")
class TestForwardForecastUnknownStation(unittest.TestCase):
    def test_returns_empty_for_unknown(self):
        import _model
        self.assertEqual(_model.forward_forecast("NOT A REAL STATION"), {})


if __name__ == "__main__":
    unittest.main()