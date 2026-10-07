"""rain_obs* provenance tests — forecast/climatology must never count as observed.

The rainfall-memory block emits two families:

  * ``rain_api7/22/60`` — the exponential antecedent precipitation index, built
    from the NUMERIC rainfall forcing available at the step (observations where
    present, forecast/climatology values for future steps);
  * ``rain_obs7/22/60`` — a COVERAGE column that must count only slots backed by
    an actual rainfall observation, over a fixed denominator of tau*4 slots.

Regression guard for the bug where the serving path concatenated history with
future Open-Meteo/climatology rain and then treated every finite value as
observed, so ``rain_obs*`` read ~1.0 across the whole forecast horizon on data
that had not been observed at all.

Stdlib + numpy/pandas only (no pytest), consistent with the rest of this suite.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from _hydrology import API_TAUS, STEP_H, api_alpha, rain_memory_block
from tests._helpers import has

N_HIST = 200
ANCHOR = pd.Timestamp("2026-09-05 00:00:00")


def _ref_api(rain: np.ndarray, tau: float) -> np.ndarray:
    """Textbook recurrence reference: y_t = lam*y_{t-1} + x_t, NaN->0, NaN until
    the first observed value. Used to prove the lfilter result directly."""
    lam = api_alpha(tau)
    out = np.full(rain.shape, np.nan, dtype=float)
    y = 0.0
    seen = False
    for i, v in enumerate(rain):
        if np.isfinite(v):
            seen = True
        y = lam * y + (float(v) if np.isfinite(v) else 0.0)
        if seen:
            out[i] = y
    return out


def _driver_fixture(n: int = N_HIST):
    """Single-station history slice: no Station column, no gwl_obs column —
    exactly the legacy/fixture shape the serving path must tolerate."""
    rng = np.random.default_rng(7)
    times = pd.date_range(ANCHOR - pd.Timedelta(hours=(n - 1) * STEP_H), ANCHOR,
                          freq="6h")
    i = np.arange(n, dtype=float)
    return pd.DataFrame({
        "time": times,
        "gwl": -10.0 + 0.001 * i + rng.normal(0, 0.05, n),
        "rain": 1.0 + 0.1 * i,
        "temp": 30 + 0.01 * i,
        "humidity": 60 + 0.01 * i,
        "solar": 200.0, "wind_speed": 10.0, "pressure": 990.0,
        "river_level": 200.0, "canal_level": 300.0,
    })


FNAMES = ["gwl", "lag1", "lag4", "lag8", "lag28", "lag120",
          "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
          "rain_1d", "rain_7d", "rain_30d", "temp", "temp_7d",
          "humidity", "humidity_7d", "solar", "wind_speed", "pressure",
          "river_level", "canal_level", "year", "month_sin", "month_cos",
          "doy_sin", "doy_cos", "hour_sin", "hour_cos", "monsoon",
          "h", "h_sin", "h_cos"]


class TestRainMemoryBlockMask(unittest.TestCase):
    """Unit-level semantics of ``rain_memory_block``."""

    def test_mask_counts_only_observed_slots(self):
        rain = np.full(50, 2.0)                       # every slot numerically present
        mask = np.zeros(50, dtype=bool)
        mask[-28:] = True                             # only the last 7 d "observed"
        blk = rain_memory_block(rain, mask)
        # fixed denominator tau*4 => 28/28 for tau=7, 28/88 for tau=22
        self.assertEqual(blk["rain_obs7"][-1], 1.0)
        self.assertAlmostEqual(blk["rain_obs22"][-1], 28 / 88, places=12)
        self.assertAlmostEqual(blk["rain_obs60"][-1], 28 / 240, places=12)
        # masking is coverage-only: the API still uses the numeric forcing
        np.testing.assert_allclose(blk["rain_api22"], _ref_api(rain, 22.0),
                                   rtol=1e-12, atol=1e-12)

    def test_unobserved_future_does_not_raise_coverage(self):
        """The regression: historical/forecast concatenated, all finite."""
        hist = np.full(100, 1.0)
        fut = np.full(40, 8.0)
        series = np.concatenate([hist, fut])
        mask = np.concatenate([np.ones(100, bool), np.zeros(40, bool)])
        blk = rain_memory_block(series, mask)
        # mu = 100: obs60 window is fully observed, so 100/240
        self.assertAlmostEqual(blk["rain_obs60"][100], 100 / 240, places=12)
        # and it does not increase as the forecast window rolls in
        tail = blk["rain_obs60"][100:140]
        self.assertTrue(np.all(np.diff(tail) <= 1e-12), "coverage must not increase")
        # API memory still accumulates the forecast rain
        self.assertGreater(blk["rain_api60"][120], blk["rain_api60"][100])

    def test_missing_historical_slot_is_not_observed(self):
        """A NaN inside the trailing 28-slot window must reduce coverage.

        Built at the END of the series so the slot sits inside the window; a gap
        earlier in the array would fall outside it and the assertion would pass
        vacuously.
        """
        rain = np.ones(40)
        rain[-2] = np.nan                      # inside the last 28 slots
        mask = np.isfinite(rain)
        blk = rain_memory_block(rain, mask)
        self.assertAlmostEqual(blk["rain_obs7"][-1], 27 / 28, places=12)
        # and the API still consumes the numeric forcing (the NaN contributes 0)
        np.testing.assert_allclose(blk["rain_api7"], _ref_api(rain, 7.0),
                                   rtol=1e-12, atol=1e-12)

    def test_finite_mask_beats_finite_value_when_they_disagree(self):
        """All values finite, mask empty -> coverage 0, API unchanged."""
        rain = np.full(30, 5.0)
        blk = rain_memory_block(rain, np.zeros(30, dtype=bool))
        for t in API_TAUS:
            self.assertEqual(blk[f"rain_obs{int(t)}"][-1], 0.0)
        np.testing.assert_allclose(blk["rain_api7"], _ref_api(rain, 7.0),
                                   rtol=1e-12, atol=1e-12)

    def test_legacy_none_mask_unchanged(self):
        """Callers that pass no mask keep the old finite-value behaviour."""
        rain = np.array([1.0, np.nan, 3.0, 4.0])
        blk = rain_memory_block(rain)
        self.assertAlmostEqual(blk["rain_obs7"][-1], 3 / 28, places=12)
        blk2 = rain_memory_block(rain, np.isfinite(rain))
        np.testing.assert_allclose(blk["rain_obs7"], blk2["rain_obs7"])

    def test_mask_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rain_memory_block(np.ones(10), np.ones(9, dtype=bool))


@unittest.skipUnless(has("refresh/future_drivers.py"), "refresh missing")
class TestServingRainObsProvenance(unittest.TestCase):
    """End-to-end: the serving matrix must not call forecast rain 'observed'."""

    @classmethod
    def setUpClass(cls):
        from refresh.future_drivers import (build_step_features,
                                            future_weather_matrix,
                                            _rain_memory_steps)
        cls.hist = _driver_fixture()
        cls.cols = ["rain_api7", "rain_obs7", "rain_obs22", "rain_obs60"]
        # No Open-Meteo frame -> every future slot is climatology-forced.
        cls.X, _ = build_step_features(
            station="S1", district="ALIGARH", anchor_t=ANCHOR,
            hist=cls.hist, fnames=FNAMES + cls.cols,
            om_frame=None, climatology=None, rain_clim=None,
            steps=120, mode="future")
        cls.df = pd.DataFrame(cls.X, columns=FNAMES + cls.cols)
        cls.fw, _ = future_weather_matrix(
            station="S1", district="ALIGARH", anchor_t=ANCHOR, steps=120,
            om_frame=None, climatology=None, rain_clim=None)
        cls.cols_ref, cls.n_hist, cls.n_hist_obs = _rain_memory_steps(
            cls.hist.sort_values("time"), cls.fw, ANCHOR, 120)

    def test_rain_obs7_is_fully_observed_at_step_zero(self):
        """Row k=0 ends its tau=7 window at the ANCHOR, so all 28 slots are
        observed history and coverage is 1.0.

        The trailing window ends at `anchor + k*6h`, the same convention as the
        `rain_1d/7d/30d` windows built from the same series, so the forecast slot
        for the step itself is not yet inside the window and cannot inflate
        coverage.
        """
        self.assertAlmostEqual(float(self.df["rain_obs7"].iloc[0]), 1.0, places=5)
        # the mask really did see every history slot as observed
        self.assertEqual(self.n_hist_obs, N_HIST)

    def test_later_steps_drop_the_forecast_tail(self):
        """Step k admits exactly k unobserved forecast slots into the tau=7 window."""
        for k in (0, 1, 4, 10, 27, 28, 40):
            expected = max(28 - k, 0) / 28
            # X is float32 (model dtype); allow float32 rounding in the ratio
            self.assertAlmostEqual(float(self.df["rain_obs7"].iloc[k]), expected,
                                   delta=1e-5, msg=f"step {k}")
        # beyond the window the coverage is fully forecast-backed
        self.assertEqual(float(self.df["rain_obs7"].iloc[-1]), 0.0)

    def test_coverage_is_monotone_and_bounded(self):
        for c in ("rain_obs7", "rain_obs22", "rain_obs60"):
            v = self.df[c].to_numpy()
            self.assertTrue(np.all(np.diff(v) <= 1e-12), f"{c} increased")
            self.assertTrue(np.all(v >= 0.0) and np.all(v <= 1.0), c)

    def test_api_uses_numeric_forcing_including_forecast(self):
        """rain_api must follow the numeric series (incl. climatology forcing)
        while rain_obs ignores it — the two are decoupled."""
        n = self.n_hist
        for c in ("rain_api7", "rain_obs7", "rain_obs22", "rain_obs60"):
            np.testing.assert_allclose(
                self.df[c].to_numpy(), self.cols_ref[c][:120],
                rtol=1e-5, atol=1e-6, err_msg=c)
        # fixture has no future rain forcing (rain_clim=None), so the API memory
        # strictly decays from its anchor state; it must never be counted as an
        # observation by the coverage columns
        self.assertLess(float(self.df["rain_api7"].iloc[-1]),
                        float(self.df["rain_api7"].iloc[0]))
        self.assertEqual(float(self.df["rain_obs7"].iloc[-1]), 0.0)

    def test_untouched_columns_still_present(self):
        """The fix must not disturb the frozen model's schema."""
        self.assertEqual(self.df.shape[1], len(FNAMES) + 4)
        self.assertTrue(np.all(np.isfinite(self.df["rain_1d"].to_numpy())))


class TestBatchRainObsProvenance(unittest.TestCase):
    """Batch path: a missing rain slot must not count as observed."""

    def _frame(self):
        n = 60
        t0 = pd.Timestamp("2025-01-01")
        times = pd.date_range(t0, periods=n, freq="6h")
        rain = np.full(n, 4.0)
        # Put the unobserved slot INSIDE the trailing 28-slot (7 d) window, so the
        # assertion genuinely tests reduced coverage rather than passing on a
        # window that never saw the gap.
        rain[n - 15] = np.nan
        # `06_features.build` emits and selects the driver columns, so a fixture
        # without them cannot exercise it (KeyError on the column selection).
        return pd.DataFrame({
            "Station": "S1", "District": "ALIGARH", "time": times,
            "gwl": -8.0 + 0.01 * np.arange(n), "rain": rain,
            "temp": 25.0, "humidity": 55.0, "solar": 200.0,
            "wind_speed": 8.0, "pressure": 995.0,
            "river_level": 150.0, "canal_level": 300.0,
        })

    def test_missing_slot_reduces_coverage_and_api_uses_forcing(self):
        import importlib
        F = importlib.import_module("06_features")
        frame = self._frame()
        # guard: the gap must actually sit inside the trailing 28 slots, else the
        # coverage assertion below would be vacuous
        last28 = frame["rain"].to_numpy()[-28:]
        self.assertEqual(int(np.isnan(last28).sum()), 1)
        out = F.build(frame, keep_na=True)
        self.assertIn("rain_obs7", out.columns)
        # 27 of the last 28 slots observed (the NaN sits at index n-15)
        self.assertAlmostEqual(float(out["rain_obs7"].iloc[-1]), 27 / 28, places=6)
        # float32 columns -> compare with a tolerance, not to 1e-6 absolute
        self.assertAlmostEqual(float(out["rain_api7"].iloc[-1]),
                               float(_ref_api(frame["rain"].to_numpy(), 7.0)[-1]),
                               delta=1e-4)

    def test_build_keeps_computed_columns(self):
        """`build` must not drop the columns it computes rather than reads.

        Regression guard: returning `df[KEEP]` silently removed `rain_api*` /
        `rain_obs*` / `gwl_imputed`, which broke `_model.forward_forecast` and
        bare-fixture callers while leaving the batch `build_full` path working.
        """
        import importlib
        F = importlib.import_module("06_features")
        out = F.build(self._frame(), keep_na=True)
        for c in list(F.HYDRO_COLS) + ["target_time"]:
            self.assertIn(c, out.columns, f"{c} was dropped by build()")
        # KEEP members stay first and in order
        ordered = [c for c in F.KEEP if c in out.columns]
        self.assertEqual(list(out.columns)[:len(ordered)], ordered)


if __name__ == "__main__":
    unittest.main()
