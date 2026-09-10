"""refresh future-driver tests (no fabrication, pinning, horizon math).

Fixture-driven, no real data needed. Asserts:
  * matrix shape (120 x 33) float32 in traj_config column order;
  * h/h_sin/h_cos horizon features match the 120-step formula;
  * GWL-memory features (gwl, lag*, gwl_roll*) pinned to the anchor row and
    never leak future observed GWL;
  * river_level: CWC 7-day forecast (delta over station baseline) inside the
    forecast window, climatology/persistence/NaN outside — never raw gauge
    metres, never future observations;
  * canal_level: climatology -> persistence -> NaN (nothing to forecast);
  * Open-Meteo used inside the forecast window, climatology after, rain split
    uniformly over the four 6h slots, windows composed of observed + future;
  * per-step source labels are explicit;
  * the CWC dashboard parser handles both table layouts (with/without a stray
    leading cell) and rows with a missing observed (short-range) reading.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from refresh.future_drivers import (RiverForecastIndex, build_step_features,
                                    future_weather_matrix)
from tests._helpers import has

NUM_COLS = [
    "gwl", "lag1", "lag4", "lag8", "lag28", "lag120",
    "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
    "rain_1d", "rain_7d", "rain_30d", "temp", "temp_7d",
    "humidity", "humidity_7d", "solar", "wind_speed", "pressure",
    "river_level", "canal_level", "year", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "hour_sin", "hour_cos", "monsoon",
]
H_COLS = ["h", "h_sin", "h_cos"]
FNAMES = NUM_COLS + H_COLS

STATION = "S1"
DISTRICT = "ALIGARH"
ANCHOR = pd.Timestamp("2026-09-05 00:00:00")
N_HIST = 200


def _fixture():
    rng = np.random.default_rng(7)
    times = pd.date_range(ANCHOR - pd.Timedelta(hours=(N_HIST - 1) * 6),
                          ANCHOR, freq="6h")
    i = np.arange(N_HIST, dtype=float)
    df = pd.DataFrame({
        "time": times,
        "gwl": -10.0 + 0.001 * i + rng.normal(0, 0.05, N_HIST),
        "rain": 1.0 + 0.1 * i,
        "temp": 30 + 0.01 * i,
        "humidity": 60 + 0.01 * i,
        "solar": 200.0, "wind_speed": 10.0, "pressure": 990.0,
        "river_level": 200.0, "canal_level": 300.0,
    })
    return df


def _om():
    dates = pd.date_range(ANCHOR.normalize(), periods=16, freq="1D")
    return pd.DataFrame({
        "date": dates, "District": DISTRICT,
        "rain_mm": 8.0, "temp_c": 33.0, "humidity_pct": 70.0,
        "solar_wm2": 500.0, "wind_kmh": 12.0, "pressure_hpa": 992.0,
    })


def _clim():
    return pd.DataFrame({
        "Station": STATION, "doy": np.arange(1, 367),
        "temp": 20.0 + np.arange(1, 367) * 0.01,
        "humidity": 50.0, "solar": 100.0, "wind_speed": 5.0, "pressure": 980.0,
    })


def _rain_clim():
    doys = np.arange(1, 367)
    rows = pd.DataFrame([(DISTRICT, int(d), h, 3.0 + d * 0.001 + h * 0.01)
                         for d in doys for h in (0, 6, 12, 18)],
                        columns=["District", "doy", "hour", "rain"])
    return rows


def _rfc():
    """Synthetic CWC snapshot: one gauge in DISTRICT, observed at anchor,
    day-1..7 forecast levels rising 0.5 m/day from 150.0."""
    levels = [150.0 + 0.5 * d for d in range(0, 8)]
    rec = {"site": "G1", "river": "GANGA", "district": DISTRICT, "state": "U P",
           "observed_dt": ANCHOR, "observed_wl": levels[0]}
    for d in range(1, 8):
        rec[f"d{d}_dt"] = ANCHOR + pd.Timedelta(days=d)
        rec[f"d{d}_wl"] = levels[d]
    return pd.DataFrame([rec])


@unittest.skipUnless(has("refresh/future_drivers.py"), "refresh missing")
class TestFutureFeatureMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hist = _fixture()
        cls.om = _om()
        cls.clim = _clim()
        cls.rclim = _rain_clim()
        cls.X, cls.src = build_step_features(
            station=STATION, district=DISTRICT, anchor_t=ANCHOR,
            hist=cls.hist, fnames=FNAMES, om_frame=cls.om,
            climatology=cls.clim, rain_clim=cls.rclim,
            forecast_days=16, steps=120, mode="future")
        cls.df = pd.DataFrame(cls.X, columns=FNAMES)

    def test_shape_dtype_and_column_order(self):
        self.assertEqual(self.X.shape, (120, 33))
        self.assertEqual(self.X.dtype, np.float32)
        self.assertEqual(list(self.df.columns), FNAMES)

    def test_horizon_features_formula(self):
        hs = np.arange(1, 121, dtype=float)
        self.assertTrue(np.allclose(self.df["h"], hs))
        self.assertTrue(np.allclose(self.df["h_sin"], np.sin(2 * np.pi * hs / 120)))
        self.assertTrue(np.allclose(self.df["h_cos"], np.cos(2 * np.pi * hs / 120)))

    def test_gwl_memory_pinned_to_anchor(self):
        anchor_gwl = float(self.hist["gwl"].iloc[-1])
        self.assertTrue(np.allclose(self.df["gwl"], anchor_gwl, atol=1e-6))
        self.assertTrue(np.allclose(self.df["lag1"],
                                    float(self.hist["gwl"].iloc[-2]), atol=1e-6))
        self.assertTrue(np.allclose(self.df["lag8"],
                                    float(self.hist["gwl"].iloc[-9]), atol=1e-6))
        for c in ("gwl_roll7_mean", "gwl_roll7_std",
                  "gwl_roll30_mean", "gwl_roll30_std"):
            self.assertEqual(self.df[c].nunique(), 1, f"{c} must be horizon-invariant")

    def test_no_future_observed_gwl_leak(self):
        fut = self.hist["gwl"].iloc[-1] + 50.0
        self.assertNotIn(round(float(fut), 4),
                         set(np.round(self.df["gwl"].to_numpy(), 4)))

    def test_river_canal_persistence_when_no_forecast(self):
        # no CWC forecast + no RC climatology columns -> last observed level
        self.assertTrue(np.allclose(self.df["river_level"], 200.0, atol=1e-5))
        self.assertTrue(np.allclose(self.df["canal_level"], 300.0, atol=1e-5))
        for rec in self.src:
            self.assertIn("river_canal:river_persistence",
                          rec["river_canal"])
            self.assertIn("river_canal:canal_persistence",
                          rec["river_canal"])

    def test_openmeteo_then_climatology_sources(self):
        for rec in self.src[:64]:
            self.assertEqual(rec["weather"], "open-meteo")
            self.assertEqual(rec["rain"], "open-meteo")
        for rec in self.src[64:]:
            self.assertEqual(rec["weather"], "climatology")
            self.assertEqual(rec["rain"], "climatology")

    def test_openmeteo_value_in_window(self):
        self.assertTrue(np.allclose(self.df["temp"].iloc[:64], 33.0, atol=1e-5))
        self.assertTrue(np.allclose(self.df["rain_1d"].iloc[0],
                                    self.hist["rain"].iloc[-3] + self.hist["rain"].iloc[-2]
                                    + self.hist["rain"].iloc[-1] + 8.0 / 4.0, atol=1e-3))

    def test_climatology_beyond_window(self):
        for k in (64, 90, 100):
            doy = int((ANCHOR + pd.Timedelta(hours=(k + 1) * 6)).dayofyear)
            expected = 20.0 + doy * 0.01
            self.assertAlmostEqual(float(self.df["temp"].iloc[k]), expected, places=3)
            self.assertFalse(np.isnan(float(self.df["temp"].iloc[k])))

    def test_monsoon_calendar_recomputed(self):
        self.assertGreater(float(self.df["monsoon"].iloc[-1]),
                           float(self.df["monsoon"].iloc[0]))


@unittest.skipUnless(has("refresh/future_drivers.py"), "refresh missing")
class TestFutureWeatherMatrixSources(unittest.TestCase):
    def test_no_om_frame_all_climatology(self):
        fw, src = future_weather_matrix(
            station=STATION, district=DISTRICT, anchor_t=ANCHOR,
            steps=6, om_frame=None, climatology=_clim(), rain_clim=_rain_clim())
        self.assertEqual(len(src), 6)
        self.assertTrue(all(r["weather"] == "climatology" for r in src))
        self.assertTrue(all(pd.notna(fw["temp"].iloc[i]) for i in range(6)))

    def test_missing_doy_is_nan_not_fabricated(self):
        clim = pd.DataFrame({  # only doy 1 (won't match Sept doys)
            "Station": STATION, "doy": [1], "temp": [10.0], "humidity": [50.0],
            "solar": [1.0], "wind_speed": [1.0], "pressure": [1000.0]})
        fw, src = future_weather_matrix(
            station=STATION, district=DISTRICT, anchor_t=ANCHOR, steps=6,
            om_frame=None, climatology=clim, rain_clim=None)
        self.assertTrue(all(r["weather"] == "unavailable" for r in src))
        self.assertTrue(fw["temp"].isna().all())

    def test_protects_non_future_mode(self):
        with self.assertRaises(ValueError):
            build_step_features(station=STATION, district=DISTRICT,
                                anchor_t=ANCHOR, hist=_fixture(), fnames=FNAMES,
                                mode="flat")


@unittest.skipUnless(has("refresh/future_drivers.py"), "refresh missing")
class TestRiverCanalCwcBridge(unittest.TestCase):
    """River forecast is a *delta over the station's own baseline*: in-window
    steps use station baseline + CWC district delta, out-of-window steps fall
    to persistence, and raw gauge metres never leak through."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.src = build_step_features(
            station=STATION, district=DISTRICT, anchor_t=ANCHOR,
            hist=_fixture(), fnames=FNAMES, om_frame=_om(),
            climatology=_clim(), rain_clim=_rain_clim(),
            river_forecast=_rfc(), steps=120, mode="future")
        cls.df = pd.DataFrame(cls.X, columns=FNAMES)

    def _date(self, k):
        return (ANCHOR + pd.Timedelta(hours=(k + 1) * 6)).normalize()

    def test_in_window_cwc_delta_over_baseline(self):
        baseline = 200.0   # river_last in _fixture (no RC climatology columns)
        for day in range(0, 8):
            date = (ANCHOR + pd.Timedelta(days=day)).normalize()
            ks = [k for k in range(120) if self._date(k) == date]
            self.assertTrue(ks, f"no steps on {date}")
            for k in ks:
                self.assertAlmostEqual(float(self.df["river_level"].iloc[k]),
                                       baseline + 0.5 * day, places=4)
                self.assertIn("river_cwc_forecast", self.src[k]["river_canal"])

    def test_out_of_window_persistence(self):
        for k in (32, 60, 119):
            self.assertAlmostEqual(float(self.df["river_level"].iloc[k]),
                                   200.0, places=4)
            self.assertIn("river_persistence", self.src[k]["river_canal"])

    def test_never_raw_cwc_metres(self):
        vals = self.df["river_level"].to_numpy()
        self.assertNotIn(150.0, set(np.round(vals, 3)))
        self.assertTrue((vals >= 199.0).all())

    def test_canal_unaffected_by_river_forecast(self):
        self.assertTrue(np.allclose(self.df["canal_level"], 300.0, atol=1e-5))
        for rec in self.src:
            self.assertIn("river_canal:canal_persistence", rec["river_canal"])


@unittest.skipUnless(has("refresh/future_drivers.py"), "refresh missing")
class TestRiverForecastIndex(unittest.TestCase):
    def test_covers_window(self):
        rfi = RiverForecastIndex(_rfc())
        self.assertTrue(rfi.covers("ALIGARH", ANCHOR))
        self.assertTrue(rfi.covers("ALIGARH", ANCHOR + pd.Timedelta(days=7)))
        self.assertFalse(rfi.covers("ALIGARH", ANCHOR + pd.Timedelta(days=8)))
        self.assertFalse(rfi.covers("ALIGARH", ANCHOR - pd.Timedelta(days=1)))
        self.assertFalse(rfi.covers("MEERUT", ANCHOR))
        self.assertFalse(rfi.covers("ALIGARH", "not-a-date"))

    def test_delta_by_forecast_day(self):
        rfi = RiverForecastIndex(_rfc())
        for day in range(0, 8):
            d = ANCHOR + pd.Timedelta(days=day)
            self.assertAlmostEqual(rfi.delta("ALIGARH", d), 0.5 * day, places=6)
        self.assertAlmostEqual(rfi.delta("ALIGARH", ANCHOR + pd.Timedelta(days=3)
                                         + pd.Timedelta(hours=6)), 1.5, places=6)

    def test_gap_date_forward_fills_prior_day(self):
        rfc = _rfc().copy()
        rfc = rfc.drop(columns=["d3_dt", "d3_wl"])
        rfi = RiverForecastIndex(rfc)
        d3 = ANCHOR + pd.Timedelta(days=3)
        self.assertTrue(rfi.covers("ALIGARH", d3))
        self.assertAlmostEqual(rfi.delta("ALIGARH", d3), 1.0, places=6)

    def test_missing_observed_falls_back_to_day1_base(self):
        rfc = _rfc().copy()
        rfc["observed_dt"] = pd.NaT
        rfc["observed_wl"] = np.nan
        rfi = RiverForecastIndex(rfc)
        self.assertAlmostEqual(rfi.delta("ALIGARH", ANCHOR + pd.Timedelta(days=1)),
                               0.0, places=6)
        self.assertAlmostEqual(rfi.delta("ALIGARH", ANCHOR + pd.Timedelta(days=2)),
                               0.5, places=6)

    def test_delta_means_across_sites(self):
        b = _rfc()
        b["site"] = "G2"
        b["observed_wl"] = 150.2
        b["d2_wl"] = 151.0   # site-G2 day-2 delta 0.8 vs site-G1's 1.0
        rfi = RiverForecastIndex(pd.concat([_rfc(), b], ignore_index=True))
        d2 = ANCHOR + pd.Timedelta(days=2)
        self.assertAlmostEqual(rfi.delta("ALIGARH", d2), 0.9, places=6)

    def test_none_and_empty_frames(self):
        self.assertEqual(RiverForecastIndex(None)._dates, {})
        self.assertEqual(RiverForecastIndex(pd.DataFrame())._dates, {})


def _cwc_header():
    h = ["Sr.No", "Station", "River", "District", "State",
         "WL (m)", "DL (m)", "HFL (m)", "Date", "Condition", "WL (m)"]
    for d in range(1, 8):
        h += [f"D{d} Date", f"D{d} Cond", f"D{d} WL"]
    return h


def _cwc_body_row():
    base = ["7", "AYODHYA", "GHAGRA", "AYODHYA", "UTTAR PRADESH",
            "91.7", "92.7", "94.0", "2026-09-11 09:00:00", "Severe", "93.0"]
    for d in range(1, 8):
        base += [f"2026-09-{11 + d} 09:00:00", "Normal",
                 str(round(93.0 + 0.1 * d, 2))]
    return base


def _cwc_html(*rows):
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f"<html><table>{body}</table></html>"


@unittest.skipUnless(has("18_cwc_river_forecast.py"), "cwc module missing")
class TestCwcDashboardParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "cwc18", Path(__file__).resolve().parent.parent / "18_cwc_river_forecast.py")
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_normal_layout(self):
        df = self.m.parse_html(_cwc_html(_cwc_header(), _cwc_body_row()))
        self.assertEqual(len(df), 1)
        r = df.iloc[0]
        self.assertEqual(int(r["sr_no"]), 7)
        self.assertEqual(r["site"], "AYODHYA")
        self.assertEqual(r["district"], "AYODHYA")
        self.assertAlmostEqual(float(r["observed_wl"]), 93.0, places=6)
        self.assertAlmostEqual(float(r["d1_wl"]), 93.1, places=6)
        self.assertAlmostEqual(float(r["d7_wl"]), 93.7, places=6)
        self.assertEqual(r["observed_cond"], "Severe")

    def test_stray_leading_cell_is_aligned(self):
        a = self.m.parse_html(_cwc_html(_cwc_header(), _cwc_body_row()))
        b = self.m.parse_html(_cwc_html(_cwc_header(), [""] + _cwc_body_row()))
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        for col in ("sr_no", "site", "district", "observed_wl", "d4_wl"):
            self.assertEqual(a.iloc[0][col], b.iloc[0][col], col)

    def test_missing_observed_short_range(self):
        row = _cwc_body_row()
        row[8] = ""   # observed date empty
        row[9] = ""   # observed condition empty
        row[10] = ""  # observed level empty
        df = self.m.parse_html(_cwc_html(_cwc_header(), row))
        self.assertEqual(len(df), 1)
        r = df.iloc[0]
        self.assertTrue(pd.isna(r["observed_dt"]))
        self.assertTrue(np.isnan(float(r["observed_wl"])))
        self.assertFalse(pd.isna(r["d1_dt"]))
        self.assertAlmostEqual(float(r["d1_wl"]), 93.1, places=6)


if __name__ == "__main__":
    unittest.main()