"""Pooled-30d DL backbone parity (_dl.forward_forecast).

Requires torch + the pooled DL artifacts + the aligned 6h table. Skipped when
any are missing. Runs in the torch-only env (venv312) - no streamlit needed.
"""
from __future__ import annotations

import importlib
import math
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _ready() -> bool:
    try:
        importlib.import_module("torch")
    except Exception:
        return False
    for n in ("models/pooled_dl.pt", "models/pooled_dl_config.json",
              "data/aligned/table_6h.parquet"):
        if not (BASE / n).exists():
            return False
    return True


SCHEMA = {
    "station", "date_from", "date_to", "anchor", "pred_xgb", "pred_ridge",
    "xgb_level", "ridge_level", "q05_level", "q50_level", "q95_level",
    "widen_k", "band_half", "station_stride_rmse",
}


@unittest.skipUnless(_ready(), "torch / pooled DL artifacts / 6h table missing")
class TestModelDlParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.import_module("_dl")
        cls.fc = cls.m.forward_forecast("ASHADHA PRATHMIK VIDYALAYA")

    def test_schema_and_finiteness(self):
        self.assertTrue(self.fc, "empty forecast for known station")
        self.assertEqual(set(self.fc), SCHEMA)
        for k in ("anchor", "pred_xgb", "xgb_level", "q05_level", "q50_level",
                  "q95_level", "widen_k", "band_half"):
            self.assertTrue(math.isfinite(float(self.fc[k])), k)
        self.assertLess(abs(self.fc["pred_xgb"]), 12.0)

    def test_levels_consistent(self):
        a = float(self.fc["anchor"])
        self.assertAlmostEqual(float(self.fc["xgb_level"]), a + float(self.fc["pred_xgb"]), places=6)

    def test_quantiles_ordered_and_band(self):
        lo, mid, hi = (float(self.fc[k]) for k in ("q05_level", "q50_level", "q95_level"))
        self.assertLessEqual(lo, mid)
        self.assertLessEqual(mid, hi)
        self.assertAlmostEqual(float(self.fc["band_half"]), (hi - lo) / 2, places=9)

    def test_ridge_fields_present_but_none(self):
        self.assertIsNone(self.fc["pred_ridge"])
        self.assertIsNone(self.fc["ridge_level"])

    def test_unknown_station_empty(self):
        self.assertEqual(self.m.forward_forecast("__NO_SUCH_STATION__"), {})


if __name__ == "__main__":
    unittest.main()
