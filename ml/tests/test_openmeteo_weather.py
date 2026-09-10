"""Open-Meteo daily weather fetcher (20) contract tests.

Data-gated on the fetched parquet + meta. The output must keep the documented
schema (date x District, six daily weather columns, future-dated forecast rows
past today) so downstream climatology/recursive paths never receive malformed
drivers silently.
"""

import unittest

import json

import pandas as pd

from tests._helpers import ROOT, has

REQ = ("data/cfs/openmeteo_weather_daily.parquet",
       "data/meta/openmeteo_meta.json")

COLS = ["date", "District", "rain_mm", "temp_c", "humidity_pct",
        "pressure_hpa", "wind_kmh", "solar_wm2"]


@unittest.skipUnless(all(has(p) for p in REQ), "open-meteo weather artifacts missing")
class TestOpenMeteoWeather(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_parquet(ROOT / "data" / "cfs" / "openmeteo_weather_daily.parquet")
        cls.meta = json.loads((ROOT / "data" / "meta" / "openmeteo_meta.json").read_text())

    def test_schema(self):
        missing = [c for c in COLS if c not in self.df.columns]
        self.assertEqual(missing, [], "missing columns")

    def test_district_cover(self):
        stations = pd.read_csv(ROOT / "data" / "meta" / "selected_gwl_stations.csv")
        have = set(self.df["District"].str.upper())
        want = set(stations["District"].astype(str).str.upper())
        self.assertTrue(want <= have, f"missing districts: {want - have}")

    def test_forecast_rows_past_today(self):
        future = (self.df["date"] > pd.Timestamp.now().normalize())
        self.assertGreater(int(self.df.loc[future, "date"].nunique()), 0,
                           "no forecast days beyond today")

    def test_rain_nonnegative(self):
        self.assertGreaterEqual(float(self.df["rain_mm"].min()), 0.0)

    def test_physical_ranges(self):
        d = self.df
        self.assertGreater(d["temp_c"].median(), 0.0)
        self.assertGreaterEqual(d["humidity_pct"].min(), 0.0)
        self.assertLessEqual(d["humidity_pct"].max(), 100.0)
        self.assertGreater(d["pressure_hpa"].min(), 800.0)
        self.assertGreaterEqual(d["solar_wm2"].min(), 0.0)