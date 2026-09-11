"""Unit tests for the assistant's deterministic fact builders."""

import unittest

import numpy as np

from tests._helpers import has


@unittest.skipUnless(has("_assistant.py") and has("data/aligned/table_6h.parquet"),
                     "assistant module / aligned table missing")
class TestAssistantFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import _assistant
        cls.a = _assistant
        cls.df = _assistant.get_df()
        cls.station = "ASHADHA PRATHMIK VIDYALAYA"

    def test_series_stats_core_keys(self):
        g = self.df[self.df["Station"] == self.station]
        s = self.a._series_stats(g["gwl"], g["time"])
        for k in ("last", "last_date", "min", "max", "span", "change_30d",
                  "change_180d", "outliers"):
            self.assertIn(k, s)
        self.assertGreaterEqual(s["max"], s["min"])

    def test_driver_correlations_shape(self):
        rows = self.a._driver_correlations(self.station)
        for r in rows:
            self.assertIn("driver", r)
            self.assertIn("corr", r)
            self.assertTrue(np.isfinite(r["corr"]))
            self.assertGreaterEqual(r["n"], 30)
        if rows:
            corrs = [abs(r["corr"]) for r in rows]
            self.assertEqual(corrs, sorted(corrs, reverse=True))

    def test_annual_facts_year_ordered(self):
        rows = self.a._annual_facts(self.station)
        years = [r["year"] for r in rows]
        self.assertEqual(years, sorted(years))
        # deltas computed between consecutive years where present
        for r in rows:
            if r.get("mean_delta_prev") is not None:
                self.assertTrue(np.isfinite(r["mean_delta_prev"]))

    def test_rain_recent_keys(self):
        rain = self.a._rain_recent(self.station)
        self.assertTrue(any(rain.get(f"rain_{d}d") is not None for d in (7, 30, 90)))
        self.assertGreater(rain.get("rain_90d", 0) or 0, 0)

    def test_precautions_deterministic(self):
        g = self.df[self.df["Station"] == self.station]
        s = self.a._series_stats(g["gwl"], g["time"])
        facts = dict(s)
        facts["district_median"] = None
        facts["forecast"] = {}
        facts["rain_recent"] = {"rain_30d": 0.0}
        facts["annual"] = []
        prec = self.a._precautions(facts)
        for p in prec:
            self.assertIn(p["level"], ("info", "watch", "action"))
            self.assertTrue(p["title"])
            self.assertTrue(p["why"])

    def test_facts_full_contract(self):
        facts = self.a.StationAssistant().facts(self.station)
        for k in ("drivers", "annual", "rain_recent", "district_context",
                  "precautions", "district_median", "forecast", "fleet_recency"):
            self.assertIn(k, facts)

    def test_fleet_recency_counts_consistent(self):
        fr = self.a.StationAssistant().facts(self.station)["fleet_recency"]
        self.assertTrue(fr.get("recent_dates"))
        counts = list(fr["recent_dates"].values())
        self.assertLessEqual(sum(counts), fr["stations_with_data"])
        self.assertGreaterEqual(fr["stations_with_data"], 1)

    def test_fleet_recency_prompt_contains_counts(self):
        facts = self.a.StationAssistant().facts(self.station)
        prompt = self.a._build_prompt(
            "how many stations have their latest reading on 11 september?", facts)
        self.assertIn("FLEET RECENT-UPDATE", prompt)
        self.assertIn("stations whose latest reading is on", prompt)
        top = list(facts["fleet_recency"]["recent_dates"].items())[0][0]
        self.assertIn(top, prompt)

    def test_mention_routing_finds_explicit_station(self):
        other = next(s for s in self.a.station_names() if s != self.station)
        found = self.a.StationAssistant().facts_for_mentions(
            f"how is {other} doing?", base_station=self.station)
        self.assertTrue(any(m["station"] == other for m in found))

    def test_mention_routing_returns_empty_when_no_match(self):
        found = self.a.StationAssistant().facts_for_mentions(
            "zxqjp nonexistent station 42", base_station=self.station)
        self.assertEqual(found, [])

    def test_mention_routing_excludes_pinned_station(self):
        found = self.a.StationAssistant().facts_for_mentions(
            f"tell me about {self.station}", base_station=self.station)
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()