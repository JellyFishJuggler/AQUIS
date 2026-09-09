"""Calibration & diagnostics artifact sanity (quantile calibration, diagnostics.json)."""

import unittest

from tests._helpers import ROOT, has


@unittest.skipUnless(has("models/quantile_calibration.json"), "calibration missing")
class TestQuantileCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.q = json.loads((ROOT / "models" / "quantile_calibration.json").read_text())

    def test_coverage_meets_target(self):
        self.assertGreaterEqual(self.q["coverage_stride_raw_q"], self.q["target_coverage_stride"])

    def test_widen_factor_possible_bounds(self):
        k = self.q["widen_factor_k"]
        self.assertGreaterEqual(k, 0.5)
        self.assertLessEqual(k, 6.0)

    def test_half_width_sane(self):
        self.assertGreater(self.q["half_width_median_m"], 0)
        self.assertLess(self.q["half_width_mean_m"], 15.0)


@unittest.skipUnless(has("outputs/diagnostics.json"), "diagnostics missing")
class TestDiagnosticsArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.d = json.loads((ROOT / "outputs" / "diagnostics.json").read_text())

    def test_permutation_has_all_features(self):
        perm = cls_d_perm(self.d)
        self.assertEqual(len(perm), 32)

    def test_permutation_ranks_unique(self):
        ranks = [p["rank"] for p in cls_d_perm(self.d)]
        self.assertEqual(sorted(ranks), ranks)

    def test_vif_features_align(self):
        vif = self.d["vif"]
        self.assertEqual(len(vif["feature"]), len(vif["vif"]))


def cls_d_perm(d):
    return d.get("permutation", [])


if __name__ == "__main__":
    unittest.main()