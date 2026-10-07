"""_spatial contract tests — no data required.

Pins what the ST-GNN gate must do, so a future edit can't quietly turn a "no" into
a "yes" (or build a graph on raw levels, which would average datums):

  1. the graph excludes self and is deterministic;
  2. neighbour features are built on ANOMALIES, so station datum offsets cancel;
  3. the anomaly baseline is CAUSAL — a row's own value must not move its own
     baseline;
  4. the gate says NO on pure noise, and YES when a genuine neighbour signal is
     planted;
  5. the gate controls for the well's own anomaly, so a neighbour that merely
     mirrors the well is not counted as independent evidence.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import _spatial as S


def _anchors(n=12, spacing=0.15):
    return pd.DataFrame({
        "Station": [f"W{i:02d}" for i in range(n)],
        "lat": [25.0 + spacing * i for i in range(n)],
        "lon": [80.0] * n,
    })


def _obs(stations, n=400, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n, freq="6h")
    rows = []
    for i, st in enumerate(stations):
        base = -20.0 - 5.0 * i                      # per-station datum offset
        vals = base + 0.5 * np.sin(np.arange(n) / 50.0) + rng.normal(0, noise, n)
        rows.append(pd.DataFrame({"Station": st, "time": times, "gwl": vals}))
    return pd.concat(rows, ignore_index=True)


class TestGraph(unittest.TestCase):
    def test_deterministic_and_no_self_links(self):
        a = _anchors()
        g1 = S.neighbour_graph(a, k=3)
        g2 = S.neighbour_graph(a, k=3)
        pd.testing.assert_frame_equal(g1, g2)
        self.assertFalse((g1["station"] == g1["neighbour"]).any())

    def test_weights_normalised_per_station(self):
        g = S.neighbour_graph(_anchors(), k=3)
        wsum = g.groupby("station")["weight"].sum()
        self.assertTrue(np.allclose(wsum.to_numpy(), 1.0, atol=1e-9))

    def test_nearest_neighbour_is_rank_one(self):
        a = _anchors()
        g = S.neighbour_graph(a, k=3)
        r1 = g[g["rank"] == 1].set_index("station")
        # W00's closest station must be W01 (adjacent in the grid)
        self.assertEqual(r1.loc["W00", "neighbour"], "W01")
        self.assertAlmostEqual(float(r1.loc["W00", "dist_km"]),
                               float(S.haversine_km(25.0, 80.0, 25.15, 80.0)), places=6)

    def test_missing_coordinates_raises(self):
        with self.assertRaises(KeyError):
            S.neighbour_graph(pd.DataFrame({"Station": ["A"]}))


class TestAnomaly(unittest.TestCase):
    def test_datum_offset_cancels(self):
        """Two wells with identical variation but different datums must have
        near-identical anomalies."""
        n = 200
        times = pd.date_range("2025-01-01", periods=n, freq="6h")
        shape = np.sin(np.arange(n) / 20.0)
        obs = pd.concat([
            pd.DataFrame({"Station": "A", "time": times, "gwl": -5.0 + shape}),
            pd.DataFrame({"Station": "B", "time": times, "gwl": -95.0 + shape}),
        ], ignore_index=True)
        a = S.station_anomaly(obs, window_steps=120).dropna(subset=["anomaly"])
        piv = a.pivot_table(index="time", columns="Station", values="anomaly")
        self.assertGreater(float(np.corrcoef(piv["A"], piv["B"])[0, 1]), 0.99)
        self.assertLess(float((piv["A"] - piv["B"]).abs().max()), 1e-9)

    def test_baseline_is_causal(self):
        """A spike at step t must not already be inside its own baseline."""
        n = 100
        times = pd.date_range("2025-01-01", periods=n, freq="6h")
        vals = np.ones(n)
        vals[50] = 101.0
        obs = pd.DataFrame({"Station": "A", "time": times, "gwl": vals})
        a = S.station_anomaly(obs, window_steps=200).set_index("time")
        # baseline at t=50 is the mean of steps 0..49 == 1.0, so anomaly == 100
        self.assertAlmostEqual(float(a["anomaly"].iloc[50]), 100.0, places=6)

    def test_first_step_has_no_anomaly(self):
        obs = _obs(["A"], n=10)
        a = S.station_anomaly(obs, window_steps=120)
        self.assertTrue(np.isnan(float(a.sort_values("time")["anomaly"].iloc[0])))


class TestNeighbourFeatures(unittest.TestCase):
    def test_shapes_and_coverage_columns(self):
        a = _anchors()
        g = S.neighbour_graph(a, k=10)
        an = S.station_anomaly(_obs(a["Station"].tolist(), n=120))
        nf = S.neighbour_features(an, g, ks=(3, 5))
        for c in ("knn3_mean", "knn3_std", "knn3_dmin_km", "knn3_n",
                  "knn5_mean", "knn5_n"):
            self.assertIn(c, nf.columns, c)
        self.assertTrue((nf["knn3_n"] <= 3).all())
        self.assertTrue((nf["knn5_n"] <= 5).all())

    def test_std_nan_for_single_neighbour(self):
        a = _anchors(n=2)
        g = S.neighbour_graph(a, k=1)
        an = S.station_anomaly(_obs(a["Station"].tolist(), n=80))
        nf = S.neighbour_features(an, g, ks=(1,))
        self.assertTrue(nf["knn1_std"].isna().all(),
                        "std over one neighbour is undefined, not zero")

    def test_empty_inputs(self):
        self.assertTrue(S.neighbour_features(pd.DataFrame(), pd.DataFrame()).empty)


class TestGate(unittest.TestCase):
    def _frames(self, plant_signal: bool, n=500, seed=1):
        a = _anchors(n=10)
        g = S.neighbour_graph(a, k=5)
        obs = _obs(a["Station"].tolist(), n=n, seed=seed, noise=0.02)
        an = S.station_anomaly(obs, window_steps=120)
        own = an[["Station", "time", "anomaly"]].copy()

        rng = np.random.default_rng(seed + 7)
        tgt = own.copy()
        if plant_signal:
            # target driven by the NEIGHBOUR mean anomaly, independent of own
            nf = S.neighbour_features(an, g, ks=(5,))
            tgt = tgt.merge(nf[["Station", "time", "knn5_mean"]],
                            on=["Station", "time"], how="left")
            tgt["target_d"] = (2.0 * tgt["knn5_mean"].fillna(0.0)
                               + rng.normal(0, 0.01, len(tgt)))
        else:
            tgt["target_d"] = rng.normal(0, 1.0, len(tgt))
        return tgt, own, g

    def test_gate_rejects_pure_noise(self):
        tgt, own, g = self._frames(plant_signal=False)
        res = S.spatial_gate(tgt, own, g, k=5)
        self.assertFalse(res["passes"], res)
        self.assertIn("verdict", res)

    def test_gate_accepts_planted_neighbour_signal(self):
        tgt, own, g = self._frames(plant_signal=True)
        res = S.spatial_gate(tgt, own, g, k=5)
        self.assertTrue(res["passes"], res)
        self.assertGreater(abs(res["partial_neighbour_r"]), 0.10)

    def test_gate_reports_when_insufficient_data(self):
        a = _anchors(n=4)
        g = S.neighbour_graph(a, k=2)
        obs = _obs(a["Station"].tolist(), n=6)
        an = S.station_anomaly(obs, window_steps=120)
        tgt = an.copy()
        tgt["target_d"] = tgt["anomaly"]
        res = S.spatial_gate(tgt, an, g, k=2)
        self.assertFalse(res["passes"])
        self.assertIn("reason", res)

    def test_graph_inputs_preconditions(self):
        a = _anchors(n=5)
        a.loc[0, "lat"] = np.nan
        rep = S.graph_inputs_available(a)
        self.assertEqual(rep["unmappable"], 1)
        self.assertFalse(rep["coordinates_ok"])
        self.assertTrue(rep["warns"], "must warn about st_id and forward-fill")


class TestPartialCorr(unittest.TestCase):
    def test_partial_corr_removes_common_cause(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, 2000)
        # x and y both driven ONLY by z -> partial corr given z should be ~0
        x = 2.0 * z + rng.normal(0, 0.1, 2000)
        y = 3.0 * z + rng.normal(0, 0.1, 2000)
        self.assertGreater(abs(float(np.corrcoef(x, y)[0, 1])), 0.9)
        self.assertLess(abs(S._partial_corr(x, y, z)), 0.1)

    def test_partial_corr_keeps_direct_effect(self):
        rng = np.random.default_rng(1)
        z = rng.normal(0, 1, 2000)
        x = rng.normal(0, 1, 2000)
        y = 2.0 * x + rng.normal(0, 0.1, 2000)
        self.assertGreater(S._partial_corr(x, y, z), 0.9)


if __name__ == "__main__":
    unittest.main()
