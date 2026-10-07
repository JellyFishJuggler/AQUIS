"""_association contract tests — no data required, stdlib + numpy/pandas only.

These pin the three properties that motivated the redesign, so a future edit
can't silently reintroduce them:

  1. NO RADIUS CLIFF — a well whose k nearest all sit beyond the spec radius
     still gets links (with small weights) rather than nothing.
  2. WEIGHT DECAY — a closer gauge gets strictly more weight, and the weights of
     a set sum to 1.
  3. DATUM SAFETY — `combine_weighted_delta` returns an *anomaly*, so two gauges
     on different datums with identical variation produce the same feature,
     while `combine_idw` would not.

Plus the normalisation contract: `training_stats` must ignore post-cutoff data.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import _association as A


def _gauges(coords):
    return pd.DataFrame({
        "Station": [f"G{i}" for i in range(len(coords))],
        "lat": [c[0] for c in coords],
        "lon": [c[1] for c in coords],
    })


def _anchors(coords):
    return pd.DataFrame({
        "Station": [f"W{i}" for i in range(len(coords))],
        "lat": [c[0] for c in coords],
        "lon": [c[1] for c in coords],
    })


def _long(station_vals: dict[str, list[float]], start="2026-01-01", freq="6h"):
    rows = []
    for st, vals in station_vals.items():
        times = pd.date_range(start, periods=len(vals), freq=freq)
        rows.append(pd.DataFrame({"Station": st, "time": times, "value": vals}))
    return pd.concat(rows, ignore_index=True)


class TestNoRadiusCliff(unittest.TestCase):
    def test_links_exist_when_all_gauges_are_beyond_radius(self):
        """The old code did `if not near.any(): continue` — no link at all."""
        # ~5.4 degrees of latitude ≈ 600 km, far outside every spec radius
        anc = _anchors([(25.0, 80.0)])
        gau = _gauges([(30.0, 80.0)])
        links = A.k_nearest(anc, gau, "rainfall")
        self.assertEqual(len(links), 1, "must still link a distant gauge")
        self.assertFalse(bool(links["in_radius"].iloc[0]))
        self.assertEqual(int(links["n_in_radius"].iloc[0]), 0)
        self.assertGreater(float(links["dist_km"].iloc[0]), 500.0)

    def test_weight_is_small_for_a_distant_gauge(self):
        anc = _anchors([(25.0, 80.0)])
        near = A.k_nearest(anc, _gauges([(25.05, 80.05)]), "rainfall")
        far = A.k_nearest(anc, _gauges([(27.0, 80.0)]), "rainfall")
        # single-gauge sets renormalise to 1.0, so compare the raw decay instead
        self.assertGreater(float(A.spec("rainfall").weight(near["dist_km"].to_numpy())[0]),
                           float(A.spec("rainfall").weight(far["dist_km"].to_numpy())[0]))

    def test_unmappable_anchor_is_skipped(self):
        anc = pd.DataFrame({"Station": ["W0"], "lat": [np.nan], "lon": [80.0]})
        self.assertTrue(A.k_nearest(anc, _gauges([(25.0, 80.0)]), "rainfall").empty)


class TestWeights(unittest.TestCase):
    def test_k_limits_and_weights_sum_to_one(self):
        anc = _anchors([(25.0, 80.0)])
        gau = _gauges([(25.0, 80.0), (25.5, 80.0), (26.0, 80.0), (27.0, 80.0)])
        links = A.k_nearest(anc, gau, "rainfall")
        self.assertEqual(len(links), A.spec("rainfall").k)
        self.assertAlmostEqual(float(links["weight"].sum()), 1.0, places=9)

    def test_closer_gauge_gets_more_weight(self):
        anc = _anchors([(25.0, 80.0)])
        gau = _gauges([(25.2, 80.0), (25.9, 80.0)])
        links = A.k_nearest(anc, gau, "rainfall").sort_values("rank")
        self.assertGreater(float(links["weight"].iloc[0]), float(links["weight"].iloc[1]))
        self.assertEqual(int(links["rank"].iloc[0]), 1)

    def test_kernel_monotone_and_bounded(self):
        s = A.spec("temperature")
        d = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])
        w = s.weight(d)
        self.assertTrue(np.all(np.diff(w) < 0), "weight must fall with distance")
        self.assertTrue(np.all((w > 0) & (w <= 1.0)), "bounded in (0, 1]")
        self.assertAlmostEqual(float(w[0]), 1.0, places=6)


class TestCombination(unittest.TestCase):
    def test_idw_weighted_mean(self):
        anc = _anchors([(25.0, 80.0)])
        gau = _gauges([(25.0, 80.0), (25.9, 80.0)])
        links = A.k_nearest(anc, gau, "temperature")
        vals = _long({"G0": [10.0] * 4, "G1": [20.0] * 4})
        out = A.combine(links, vals, "temperature")
        self.assertEqual(len(out), 4)
        # blended value sits between the two gauges, closer to the near one
        v = float(out["value"].iloc[0])
        self.assertTrue(10.0 < v < 20.0, v)
        self.assertLess(v, 15.0, "nearer gauge should dominate")
        self.assertEqual(int(out["n"].iloc[0]), 2)

    def test_weighted_delta_ignores_datum_offset(self):
        """Two gauges, identical variation, wildly different datums.

        The delta feature must reflect only the common variation — proving the
        absolute level is not transplanted.
        """
        anc = _anchors([(25.0, 80.0)])
        gau = _gauges([(25.0, 80.0), (25.2, 80.0)])
        n = 300
        shape = np.sin(np.arange(n) / 20.0)
        vals = _long({
            "G0": list(1000.0 + shape),      # datum ~1000
            "G1": list(93.0 + shape),        # datum ~93
        })
        links = A.k_nearest(anc, gau, "river_level")
        out = A.combine(links, vals, "river_level")
        self.assertFalse(out.empty)
        v = out["value"].dropna()
        self.assertLess(float(v.abs().max()), 5.0,
                        "an anomaly must not carry a ~1000 m datum")
        # and it should reproduce the shared shape closely
        self.assertGreater(float(np.corrcoef(v.to_numpy()[-100:],
                                             shape[-100:])[0, 1]), 0.9)

    def test_weighted_delta_baseline_is_causal(self):
        """The baseline must exclude the current step.

        With a value of 100 arriving after 49 steps at 1.0, a *causal* trailing
        mean is 1.0 so the anomaly is ~99. A baseline that included the current
        value would give ~2.0, so the anomaly would be only ~98 — and, more
        importantly, would let the row's own value move its own baseline.
        """
        anc = _anchors([(25.0, 80.0)])
        gau = _gauges([(25.0, 80.0)])
        times = pd.date_range("2026-01-01", periods=100, freq="6h")
        vals = pd.DataFrame({"Station": "G0", "time": times,
                             "value": [1.0] * 49 + [100.0] + [1.0] * 50})
        links = A.k_nearest(anc, gau, "river_level")
        # combine_idw drops the first row (NaN anomaly, no history), so index
        # by the spike's timestamp rather than by position in the result.
        out = A.combine(links, vals, "river_level").set_index("time")
        self.assertAlmostEqual(float(out.loc[times[49], "value"]), 99.0, delta=1e-6)

    def test_weighted_delta_first_step_is_nan(self):
        """No history -> no anomaly, rather than a fabricated zero."""
        anc = _anchors([(25.0, 80.0)])
        gau = _gauges([(25.0, 80.0)])
        vals = _long({"G0": [5.0] * 10})
        links = A.k_nearest(anc, gau, "river_level")
        anom = A.gauge_anomaly(vals, window_steps=120)
        first = anom.sort_values("time").iloc[0]
        self.assertTrue(np.isnan(float(first["value"])))

    def test_method_is_declared_per_parameter(self):
        self.assertEqual(A.spec("rainfall").method, "idw")
        self.assertEqual(A.spec("river_level").method, "weighted_delta")
        self.assertEqual(A.spec("canal_level").method, "weighted_delta")
        self.assertIn("river_level", A.DATUM_RELATIVE)

    def test_wind_direction_excluded(self):
        self.assertIn("wind_direction", A.EXCLUDED_FROM_MODEL)

    def test_empty_inputs_return_empty_not_crash(self):
        self.assertTrue(A.combine(pd.DataFrame(), pd.DataFrame(), "rainfall").empty)
        self.assertTrue(A.combine_idw(pd.DataFrame(), pd.DataFrame()).empty)
        self.assertTrue(A.combine_weighted_delta(pd.DataFrame(), pd.DataFrame()).empty)


class TestNormalisation(unittest.TestCase):
    def test_asinh_keeps_small_value_resolution(self):
        x = np.array([0.0, 0.5, 1.0, 100.0])
        y = A.asinh_scale(x)
        self.assertTrue(np.all(np.diff(y) > 0))
        # ~identity at small values, and strictly closer to x than log1p is
        # (log1p(0.5) = 0.405, asinh(0.5) = 0.481, x = 0.5)
        self.assertAlmostEqual(float(y[1]), 0.5, places=1)
        self.assertGreater(float(y[1]), float(np.log1p(0.5)))
        self.assertLess(float(y[3]), 100.0)

    def test_training_stats_ignore_post_cutoff(self):
        idx = pd.date_range("2025-01-01", periods=100, freq="D")
        s = pd.Series([1.0] * 50 + [1000.0] * 50, index=idx)
        st = A.training_stats(s, cutoff=pd.Timestamp("2025-02-20"))
        self.assertAlmostEqual(st["mean"], 1.0, places=6)
        self.assertLess(st["std"], 0.01)

    def test_zscore_uses_supplied_stats(self):
        s = pd.Series([0.0, 10.0])
        z = A.normalise(s, {"mean": 5.0, "std": 5.0})
        self.assertAlmostEqual(float(z.iloc[0]), -1.0)
        self.assertAlmostEqual(float(z.iloc[1]), 1.0)

    def test_nan_is_preserved(self):
        s = pd.Series([1.0, np.nan, 3.0])
        z = A.normalise(s, {"mean": 2.0, "std": 1.0})
        self.assertTrue(np.isnan(float(z.iloc[1])))

    def test_normalisation_declared_per_parameter(self):
        self.assertEqual(A.normalisation_for("rainfall"), "asinh")
        self.assertEqual(A.normalisation_for("temperature"), "zscore")
        self.assertEqual(A.normalisation_for("pressure"), "zscore")


class TestCoverageSummary(unittest.TestCase):
    def test_reports_shortfall_explicitly(self):
        anc = _anchors([(25.0, 80.0), (28.0, 80.0)])
        gau = _gauges([(25.1, 80.0)])
        links = A.k_nearest(anc, gau, "rainfall")
        cov = A.coverage_summary(links, len(anc), "rainfall")
        self.assertEqual(cov["anchors"], 2)
        self.assertEqual(cov["linked"], 2, "both anchors are linked now")
        self.assertEqual(cov["anchors_with_gauge_in_radius"], 1,
                         "only one anchor has a gauge inside the radius")

    def test_empty_links(self):
        cov = A.coverage_summary(pd.DataFrame(), 10, "rainfall")
        self.assertEqual(cov["linked"], 0)
        self.assertEqual(cov["linked_pct"], 0.0)


class TestWsumSemantics(unittest.TestCase):
    """`wsum` must be the RAW kernel sum (effective information), not 1.0.

    k_nearest normalises `weight` per anchor, so the sum of `weight` is 1.0
    for every anchor — reporting it as coverage would claim "full information"
    for a well whose nearest gauge is 300 km away.
    """

    def test_k_nearest_keeps_raw_kernel_weight(self):
        anc = _anchors([(25.0, 80.0)])
        gau = _gauges([(25.01, 80.0), (25.02, 80.0), (25.03, 80.0)])
        links = A.k_nearest(anc, gau, "rainfall")
        self.assertAlmostEqual(float(links["weight"].sum()), 1.0, places=9,
                               msg="combining weights stay normalised")
        # raw sum for three co-located gauges is ~k, not 1
        self.assertAlmostEqual(float(links["w_raw"].sum()), 3.0, delta=0.05)

    def test_coverage_wsum_separates_near_from_distant(self):
        anc = _anchors([(25.0, 80.0)])
        near = A.coverage_summary(
            A.k_nearest(anc, _gauges([(25.01, 80.0), (25.02, 80.0),
                                      (25.03, 80.0)]), "rainfall"),
            1, "rainfall")
        far = A.coverage_summary(
            A.k_nearest(anc, _gauges([(30.0, 80.0), (30.1, 80.0),
                                      (30.2, 80.0)]), "rainfall"),
            1, "rainfall")
        self.assertGreater(near["median_wsum"], 2.0)
        self.assertLess(far["median_wsum"], 0.1)
        self.assertGreater(near["median_wsum"], far["median_wsum"] + 1.0)

    def test_combine_emits_effective_information(self):
        anc = _anchors([(25.0, 80.0)])
        vals = _long({"G0": [10.0] * 4, "G1": [10.0] * 4,
                      "G2": [10.0] * 4})
        near = A.combine(A.k_nearest(anc, _gauges([(25.01, 80.0),
                                                   (25.02, 80.0),
                                                   (25.03, 80.0)]),
                                     "rainfall"), vals, "rainfall")
        far = A.combine(A.k_nearest(anc, _gauges([(30.0, 80.0),
                                                  (30.1, 80.0),
                                                  (30.2, 80.0)]),
                                    "rainfall"), vals, "rainfall")
        self.assertGreater(float(near["wsum"].iloc[0]), 2.0)
        self.assertLess(float(far["wsum"].iloc[0]), 0.1)

    def test_combine_without_raw_column_falls_back(self):
        """Callers may supply normalised links only — must not crash."""
        links = pd.DataFrame({
            "anchor_station": ["W0"], "src_station": ["G0"],
            "dist_km": [5.0], "weight": [1.0], "rank": [1],
        })
        out = A.combine_idw(links, _long({"G0": [7.0] * 3}))
        self.assertEqual(float(out["wsum"].iloc[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
