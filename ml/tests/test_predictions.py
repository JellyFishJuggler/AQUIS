"""Evaluation artifacts integrity tests (predictions + honest metrics + config)."""

import unittest

import pandas as pd

from tests._helpers import ROOT, has


@unittest.skipUnless(has("outputs/predictions_2026.parquet"), "predictions missing")
class TestPredictionsIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = pd.read_parquet(ROOT / "outputs" / "predictions_2026.parquet")

    def test_required_columns(self):
        for c in ("Station", "time", "target", "gwl", "xgb", "ridge",
                  "step_idx", "window_id", "stride", "q05_lvl", "q95_lvl"):
            self.assertIn(c, self.p.columns, f"missing column {c}")

    def test_stride_rows_one_per_window(self):
        self.assertTrue((self.p["step_idx"] // 120 == self.p["window_id"]).all())
        per = self.p.groupby(["Station", "window_id"])["stride"].sum()
        self.assertTrue((per <= 1).all(), "stride picks more than 1 row in a window")
        self.assertEqual(int(per.sum()), int(self.p["stride"].sum()))

    def test_stride_count_matches_honest_metrics(self):
        honest = pd.read_csv(ROOT / "outputs" / "honest_metrics.csv")
        hx = honest[honest["model"] == "xgb"].iloc[0]
        self.assertEqual(int(self.p["stride"].sum()), int(hx["stride_rows"]))

    def test_quantile_levels_ordered(self):
        q = self.p.dropna(subset=["q05_lvl", "q95_lvl"])
        self.assertGreater(len(q), 0)
        self.assertTrue((q["q05_lvl"] < q["q95_lvl"]).all())

    def test_quantile_interval_width_positive(self):
        self.assertGreater((self.p["q95_lvl"] - self.p["q05_lvl"]).median(), 0)


@unittest.skipUnless(has("models/feature_config.json"), "config missing")
class TestConfigModelConsistency(unittest.TestCase):
    def test_xgb_feature_names_match_config(self):
        import json
        import joblib

        cfg = json.loads((ROOT / "models" / "feature_config.json").read_text())
        model = joblib.load(ROOT / "models" / "xgb_multihorizon.joblib")
        fnames = list(getattr(model, "feature_names_in_", []))
        self.assertEqual(len(fnames), len(cfg["num_cols"]) + 2)  # + st_id/dist_id
        self.assertEqual(fnames[: len(cfg["num_cols"])], cfg["num_cols"])

    def test_fleet_categories_valid(self):
        f = ROOT / "outputs" / "fleet_forecast.csv"
        if not f.exists():
            self.skipTest("fleet snapshot missing")
        df = pd.read_csv(f)
        valid = {"decline (high)", "decline", "recovering", "stable", "unreliable", "unknown"}
        self.assertTrue(set(df["category"]).issubset(valid), set(df["category"]))


if __name__ == "__main__":
    unittest.main()