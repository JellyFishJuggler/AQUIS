"""Multi-fold aggregation math (no training, no data).

``folds.aggregate`` is the part that produces the published numbers, so it is
tested on synthetic runs with known per-station errors: row-pooled ratios, the
per-fold mean Â± std, the nested station bootstrap, and the incomplete-fold
guard.
"""

from __future__ import annotations

import unittest

import numpy as np

from research.gnn import folds as F


def _run(fold: int, seed: int, arch: str, g_scale: float,
         stations=("A", "B", "C"), n_per: int = 4, pinball: float = 0.1) -> dict:
    """Synthetic run: every station has ``n_per`` rows with unit error.

    ``g_scale`` multiplies the squared error, so the row-pooled RMSE ratio is
    ``sqrt(g_scale)`` exactly.
    """
    hz = {}
    for hh in F.HORIZONS:
        hz[str(hh)] = {
            "n": n_per * len(stations),
            "rmse": float(np.sqrt(g_scale)),
            "mae": 1.0,
            "station_sse": {s: float(g_scale) * n_per for s in stations},
            "station_n": {s: n_per for s in stations},
        }
    return {"fold": fold, "seed": seed, "arch": arch, "n_unseen": len(stations),
            "pinball": pinball, "horizons": hz}


class TestRatioFromSse(unittest.TestCase):
    def test_point_estimate_is_ratio_of_totals(self):
        g_sse = np.array([4.0, 6.0])
        g_n = np.array([1.0, 1.0])
        u_sse = np.array([1.0, 1.0])
        u_n = np.array([1.0, 1.0])
        # row-pooled: sqrt(10/2) / sqrt(2/2) = sqrt(5)
        self.assertAlmostEqual(
            float(F._ratio_from_sse(g_sse, g_n, u_sse, u_n)), np.sqrt(5.0), places=12)

    def test_bootstrap_samples_are_axis_wise(self):
        rng = np.random.default_rng(0)
        idx = rng.integers(0, 2, size=(5, 2))
        g_sse = np.array([4.0, 9.0])
        u_sse = np.array([1.0, 1.0])
        ones = np.ones(2)
        out = F._ratio_from_sse(g_sse[idx], ones[idx], u_sse[idx], ones[idx])
        self.assertEqual(out.shape, (5,))
        # identical models -> every resample is exactly 1
        same = F._ratio_from_sse(g_sse[idx], ones[idx], g_sse[idx], ones[idx])
        np.testing.assert_allclose(same, 1.0)

    def test_zero_counts_return_nan_not_crash(self):
        out = F._ratio_from_sse(np.array([0.0]), np.array([0.0]),
                                np.array([0.0]), np.array([0.0]))
        self.assertTrue(np.isnan(out).all())     # no data -> undefined, never raises


class TestAggregate(unittest.TestCase):
    def _runs(self, folds=2, seeds=(42, 43), g_scale=4.0, u_scale=1.0):
        runs = []
        for f in range(folds):
            for s in seeds:
                runs.append(_run(f, s, "gnn", g_scale))
                runs.append(_run(f, s, "gru", u_scale))
        return runs

    def test_ratio_and_per_fold_means(self):
        res = F.aggregate(self._runs(), folds=2, seeds=[42, 43], n_boot=200)
        for hh in F.HORIZONS:
            b = res["per_horizon"][str(hh)]
            self.assertAlmostEqual(b["ratio_mean"], 2.0, places=9)
            self.assertAlmostEqual(b["ratio_std"], 0.0, places=9)
            self.assertAlmostEqual(b["ratio_fold_mean_std"], 0.0, places=9)
            self.assertEqual(sorted(b["per_fold"]), [0, 1])
            for v in b["per_fold"].values():
                self.assertAlmostEqual(v["ratio_mean"], 2.0, places=9)
                self.assertEqual(v["n_unseen"], 3)
                self.assertEqual(v["n_rows"], 12)
                self.assertAlmostEqual(v["p_gnn_better"], 0.0, places=12)
        self.assertAlmostEqual(res["pinball"]["ratio_mean"], 1.0, places=9)

    def test_ci_brackets_the_estimate(self):
        runs = self._runs()
        # station A is worse in the gnn runs, and by a seed-dependent amount, so
        # the run-to-run ratios differ (as they do with real training seeds)
        for r in runs:
            if r["arch"] == "gnn":
                factor = 4.0 + (r["seed"] - 42) * 5.0
                for hz in r["horizons"].values():
                    hz["station_sse"]["A"] *= factor
                    hz["rmse"] = float(np.sqrt(
                        sum(hz["station_sse"].values()) / hz["n"]))
        res = F.aggregate(runs, folds=2, seeds=[42, 43], n_boot=2000, boot_seed=7)
        for hh in F.HORIZONS:
            b = res["per_horizon"][str(hh)]
            lo, hi = b["ci95"]
            self.assertLessEqual(lo, b["ratio_mean"])
            self.assertGreaterEqual(hi, b["ratio_mean"])
            self.assertLess(lo, hi)
            self.assertGreater(b["ratio_max"], b["ratio_min"])
            self.assertGreater(b["ratio_std"], 0.0)
            for v in b["per_fold"].values():
                flo, fhi = v["ci95"]
                self.assertLess(flo, fhi)
                self.assertGreaterEqual(v["p_gnn_better"], 0.0)
                self.assertLessEqual(v["p_gnn_better"], 1.0)

    def test_nested_bootstrap_is_not_a_single_fold(self):
        """Regression: the nested CI must resample per fold, not collapse to one."""
        runs = self._runs()
        for r in runs:
            for hz in r["horizons"].values():
                if r["arch"] == "gnn":
                    hz["station_sse"]["A"] *= 4.0
                    hz["rmse"] = float(np.sqrt(
                        sum(hz["station_sse"].values()) / hz["n"]))
        res = F.aggregate(runs, folds=2, seeds=[42, 43], n_boot=500, boot_seed=1)
        for hh in F.HORIZONS:
            lo, hi = res["per_horizon"][str(hh)]["ci95"]
            self.assertGreater(hi - lo, 1e-6, f"degenerate CI at h={hh}")

    def test_incomplete_fold_is_skipped(self):
        runs = self._runs(folds=2)
        runs = [r for r in runs if not (r["fold"] == 1 and r["arch"] == "gru")]
        res = F.aggregate(runs, folds=2, seeds=[42, 43], n_boot=100)
        for hh in F.HORIZONS:
            self.assertEqual(sorted(res["per_horizon"][str(hh)]["per_fold"]), [0])

    def test_missing_horizon_is_skipped(self):
        runs = self._runs(folds=1)
        for r in runs:
            r["horizons"].pop("60")
        res = F.aggregate(runs, folds=1, seeds=[42, 43], n_boot=100)
        self.assertNotIn("60", res["per_horizon"])
        self.assertIn("4", res["per_horizon"])

    def test_render_md_mentions_every_horizon(self):
        res = F.aggregate(self._runs(), folds=2, seeds=[42, 43], n_boot=100)
        md = F.render_md(res, {"generated_at": "2026-01-01 00:00:00",
                               "device": "cpu", "block_seed": 42,
                               "epochs": 20, "patience": 4})
        for hh in F.HORIZONS:
            self.assertIn(f"{hh / 4:.0f}d", md)
        self.assertIn("gnn/gru", md)
        # per-fold rows must be filled in (regression: int vs str key lookup)
        self.assertIn("| 0 | 3 |", md)


if __name__ == "__main__":
    unittest.main()

