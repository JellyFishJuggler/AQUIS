"""Consistent derived-feature perturbation: math and audit checks.

The influence perturbation moves GWL(t) on the anchor row. The rolling GWL
columns contain that row, so a consistent perturbation must move them too.
These tests pin the exact update (single-element mean/variance) against a
brute-force recompute, and assert the feature audit's classification of which
derived columns the model actually consumes.
"""

from __future__ import annotations

import unittest

import numpy as np

from research.gnn import data as D
from research.gnn.influence import (LAG_COLS, ROLL_SPEC, feature_audit,
                                    roll_delta, window_features)


def _zeros():
    return np.zeros(len(D.NUM_COLS), np.float32), np.ones(len(D.NUM_COLS), np.float32)


class TestRollDelta(unittest.TestCase):
    def setUp(self):
        self.fmean, self.fstd = _zeros()

    def _row(self, win, fmean, fstd):
        """Standardised anchor row with the true rolling stats filled in."""
        row = np.zeros(len(D.NUM_COLS), np.float64)
        row[D.NUM_COLS.index("gwl")] = win[-1]
        for name, n in (("gwl_roll7", min(len(win), 28)),
                        ("gwl_roll30", min(len(win), 120))):
            v = win[-n:]
            m = float(np.mean(v))
            s = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            for kind, stat in (("mean", m), ("std", s)):
                j = D.NUM_COLS.index(f"{name}_{kind}")
                row[j] = (stat - fmean[j]) / fstd[j]
        return row

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 3, 6, 14, 28, 120):
            win = rng.normal(-5.0, 2.0, n)
            for delta in (0.05, -0.25, 1.0):
                row = self._row(win, self.fmean, self.fstd)
                n7, n30 = min(n, 28), min(n, 120)
                out = roll_delta(row, win[-1], delta, n7, n30,
                                 self.fmean, self.fstd)
                w2 = win.copy()
                w2[-1] += delta
                for k, (name, _, kind) in enumerate(ROLL_SPEC):
                    nn = n7 if name.startswith("gwl_roll7") else n30
                    v = w2[-nn:]
                    stat = (float(np.mean(v)) if kind == "mean" else
                            (float(np.std(v, ddof=1)) if nn > 1 else 0.0))
                    j = D.NUM_COLS.index(name)
                    expected = (stat - self.fmean[j]) / self.fstd[j]
                    self.assertAlmostEqual(out[k], expected, places=6,
                                           msg=f"{name} n={n} delta={delta}")

    def test_single_observation(self):
        row = self._row(np.array([-4.0]), self.fmean, self.fstd)
        out = roll_delta(row, -4.0, 0.10, 1, 1, self.fmean, self.fstd)
        for k, (name, _, kind) in enumerate(ROLL_SPEC):
            expected = -3.9 if kind == "mean" else 0.0
            self.assertAlmostEqual(out[k], expected, places=6, msg=name)

    def test_zero_delta_is_identity(self):
        rng = np.random.default_rng(1)
        win = rng.normal(-5.0, 1.0, 12)
        row = self._row(win, self.fmean, self.fstd)
        out = roll_delta(row, win[-1], 0.0, 12, 12, self.fmean, self.fstd)
        for k, (name, _, _) in enumerate(ROLL_SPEC):
            self.assertAlmostEqual(out[k], row[D.NUM_COLS.index(name)],
                                   places=6, msg=name)


class TestWindowFeatures(unittest.TestCase):
    def _axis(self, days: int = 40):
        rows = days * 4
        times = (np.datetime64("2025-01-01T00:00")
                 + np.arange(rows) * np.timedelta64(6, "h"))
        X = np.zeros((rows, len(D.NUM_COLS)), np.float32)
        X[:, D.NUM_COLS.index("gwl")] = np.linspace(-5.0, -3.0, rows)
        return {"X": X, "finite": np.ones_like(X), "st": np.array(["S"] * rows),
                "times": times, "starts": np.array([0]), "ends": np.array([rows])}

    def test_counts_match_time_windows(self):
        axis = self._axis()
        fmean, fstd = _zeros()
        for idx in (5, 27, 28, 119, 120, 159):
            sub = {"A": 1, "a_station": np.array(["S"]), "a_axis": np.array([idx]),
                   "a_runstart": np.array([0]), "a_time": axis["times"][idx:idx + 1]}
            win = window_features(sub, axis, fmean, fstd)
            self.assertEqual(int(win["n7"][0]), min(idx + 1, 28))
            self.assertEqual(int(win["n30"][0]), min(idx + 1, 120))

    def test_counts_reset_across_long_gap(self):
        axis = self._axis(40)
        # drop 10 days of rows (indices 40..79), leaving a gap before index 80
        keep = np.r_[np.arange(0, 40), np.arange(80, len(axis["times"]))]
        axis = {"X": axis["X"][keep], "finite": axis["finite"][keep],
                "st": axis["st"][keep], "times": axis["times"][keep],
                "starts": np.array([0]), "ends": np.array([len(keep)])}
        fmean, fstd = _zeros()
        idx = 40                                # first row after the gap
        sub = {"A": 1, "a_station": np.array(["S"]), "a_axis": np.array([idx]),
               "a_runstart": np.array([0]), "a_time": axis["times"][idx:idx + 1]}
        win = window_features(sub, axis, fmean, fstd)
        t = axis["times"][idx]
        exp7 = int(np.sum((axis["times"] > t - np.timedelta64(7, "D"))
                          & (axis["times"] <= t)))
        self.assertEqual(int(win["n7"][0]), exp7)
        self.assertEqual(exp7, 1)               # 10-day gap, so 7d window is just t
        exp30 = int(np.sum((axis["times"] > t - np.timedelta64(30, "D"))
                           & (axis["times"] <= t)))
        self.assertEqual(int(win["n30"][0]), exp30)


class TestFeatureAudit(unittest.TestCase):
    def test_audit_classifies_columns(self):
        audit = feature_audit()
        self.assertEqual(audit["gwl_lag_columns_in_model"], LAG_COLS)
        self.assertFalse(audit["lag_columns_changed_by_anchor_perturbation"])
        self.assertTrue(audit["rolling_columns_changed_by_anchor_perturbation"])
        for name, _, _ in ROLL_SPEC:
            self.assertIn(name, D.NUM_COLS)
            self.assertIn(name, audit["gwl_rolling_columns_in_model"])
        for name in ("gwl_dev7", "gwl_slope30", "gwl_imputed"):
            self.assertNotIn(name, D.NUM_COLS)
            self.assertIn(name, audit["derived_columns_not_in_model"])


if __name__ == "__main__":
    unittest.main()
