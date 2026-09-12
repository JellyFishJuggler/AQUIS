"""Unit tests for the Flask ML API (ml/api.py) — no server needed."""

import unittest

from tests._helpers import has


@unittest.skipUnless(has("api.py") and has("data/aligned/table_6h.parquet"),
                     "api module / aligned table missing")
class TestStationsList(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()
        r = cls.client.get("/stations?limit=5")
        assert r.status_code == 200, r.get_data(as_text=True)
        cls.items = r.get_json()["stations"]

    def test_list_items_have_coords(self):
        self.assertTrue(self.items)
        for s in self.items:
            for k in ("slug", "station", "district", "last_ts",
                      "latitude", "longitude"):
                self.assertIn(k, s)
            self.assertIsInstance(s["latitude"], float)
            self.assertIsInstance(s["longitude"], float)

    def test_known_station_coords(self):
        # selected_gwl_stations.csv: ASHADHA PRATHMIK VIDYALAYA, KAUSHAMBI
        import api
        r = self.client.get("/stations?q=ASHADHA%20PRATHMIK%20VIDYALAYA")
        rows = [s for s in r.get_json()["stations"]
                if s["station"] == "ASHADHA PRATHMIK VIDYALAYA"]
        self.assertTrue(rows)
        self.assertAlmostEqual(rows[0]["latitude"], 25.441184, places=4)
        self.assertAlmostEqual(rows[0]["longitude"], 81.392245, places=4)
        self.assertEqual(api.slugify("ASHADHA PRATHMIK VIDYALAYA"),
                         rows[0]["slug"])


@unittest.skipUnless(has("api.py") and has("data/aligned/table_6h.parquet"),
                     "api module / aligned table missing")
class TestStationDetail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.api = api
        cls.client = api.app.test_client()
        cls.slug = api.slugify("ASHADHA PRATHMIK VIDYALAYA")

    def test_detail_returns_facts(self):
        r = self.client.get(f"/stations/{self.slug}")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:300])
        body = r.get_json()
        # envelope
        self.assertEqual(body["slug"], self.slug)
        self.assertEqual(body["station"], "ASHADHA PRATHMIK VIDYALAYA")
        self.assertAlmostEqual(body["latitude"], 25.441184, places=4)
        # same rich facts object the chat endpoint returns under "facts"
        for k in ("last", "last_date", "district", "drivers",
                  "precautions", "forecast"):
            self.assertIn(k, body)

    def test_detail_matches_chat_facts(self):
        import _assistant
        chat_facts = _assistant.StationAssistant().facts(
            "ASHADHA PRATHMIK VIDYALAYA")
        body = self.client.get(f"/stations/{self.slug}").get_json()
        for k in ("last", "last_date", "min", "max", "district_median"):
            self.assertEqual(body[k], chat_facts[k])

    def test_detail_unknown_slug_404(self):
        r = self.client.get("/stations/no-such-station-xyz")
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.get_json())

    def test_stations_list_route_not_shadowed(self):
        # /stations/<slug> must not swallow the exact /stations route
        r = self.client.get("/stations?limit=1")
        self.assertEqual(r.status_code, 200)
        self.assertIn("stations", r.get_json())


@unittest.skipUnless(has("api.py") and has("data/aligned/table_6h.parquet"),
                     "api module / aligned table missing")
class TestStationSeries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.api = api
        cls.client = api.app.test_client()
        cls.slug = api.slugify("ASHADHA PRATHMIK VIDYALAYA")

    def test_series_shape(self):
        r = self.client.get(f"/stations/{self.slug}/series?drivers=rain,temp&limit=10")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:300])
        body = r.get_json()
        self.assertEqual(body["slug"], self.slug)
        self.assertEqual(body["granularity"], "6h")
        self.assertEqual(body["drivers"], ["rain", "temp"])
        self.assertTrue(body["points"])
        for p in body["points"]:
            self.assertIn("time", p)
            self.assertIn("gwl", p)  # anchor always present
            self.assertIn("rain", p)
            self.assertNotIn("pressure", p)  # not requested
        times = [p["time"] for p in body["points"]]
        self.assertEqual(times, sorted(times))

    def test_series_default_drivers(self):
        body = self.client.get(f"/stations/{self.slug}/series?limit=3").get_json()
        self.assertEqual(body["drivers"], list(self.api.SERIES_DRIVERS))
        self.assertEqual(set(body["driver_labels"]), set(body["drivers"]))

    def test_series_date_filter(self):
        body = self.client.get(
            f"/stations/{self.slug}/series?from=2026-01-01&to=2026-01-31&limit=6000").get_json()
        self.assertTrue(body["points"])
        self.assertEqual(body["from"], "2026-01-01")
        self.assertEqual(body["to"], "2026-01-31")
        for p in body["points"]:
            self.assertTrue(p["time"] >= "2026-01-01")
            self.assertTrue(p["time"] < "2026-02-01")

    def test_series_downsample_flag(self):
        body = self.client.get(f"/stations/{self.slug}/series?limit=10").get_json()
        self.assertTrue(body["downsampled"])
        self.assertLessEqual(body["count"], 10)

    def test_series_bad_driver_400(self):
        r = self.client.get(f"/stations/{self.slug}/series?drivers=bogus")
        self.assertEqual(r.status_code, 400)
        self.assertIn("available", r.get_json())

    def test_series_unknown_slug_404(self):
        r = self.client.get("/stations/no-such-station-xyz/series")
        self.assertEqual(r.status_code, 404)

    def test_series_bad_date_400(self):
        r = self.client.get(f"/stations/{self.slug}/series?from=not-a-date")
        self.assertEqual(r.status_code, 400)


@unittest.skipUnless(has("api.py") and has("data/aligned/table_6h.parquet"),
                     "api module / aligned table missing")
class TestStationAlerts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.api = api
        cls.client = api.app.test_client()
        cls.slug = api.slugify("ASHADHA PRATHMIK VIDYALAYA")

    def test_alerts_shape(self):
        body = self.client.get(f"/stations/{self.slug}/alerts").get_json()
        self.assertIn(body["zone"], ("safe", "alert", "danger", "unknown"))
        self.assertIsInstance(body["reasons"], list)
        for r in body["reasons"]:
            for k in ("level", "title", "why"):
                self.assertIn(k, r)
        self.assertLessEqual(len(body["top_drivers"]), 3)
        for d in body["top_drivers"]:
            for k in ("driver", "label", "corr", "p", "n"):
                self.assertIn(k, d)
        self.assertIn("direction", body["forecast"])

    def test_alerts_zone_consistent_with_reasons(self):
        body = self.client.get(f"/stations/{self.slug}/alerts").get_json()
        levels = {r["level"] for r in body["reasons"]}
        if body["zone"] == "danger":
            self.assertIn("action", levels)
        elif body["zone"] == "alert":
            self.assertIn("watch", levels)
        elif body["zone"] == "safe":
            self.assertNotIn("action", levels)
            self.assertNotIn("watch", levels)

    def test_alerts_n_param(self):
        body = self.client.get(f"/stations/{self.slug}/alerts?n=1").get_json()
        self.assertLessEqual(len(body["top_drivers"]), 1)

    def test_alerts_unknown_slug_404(self):
        r = self.client.get("/stations/no-such-station-xyz/alerts")
        self.assertEqual(r.status_code, 404)


@unittest.skipUnless(has("api.py") and has("outputs/fleet_forecast.csv"),
                     "api module / fleet scan missing")
class TestFleetAlerts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()

    def test_fleet_shape_and_counts(self):
        body = self.client.get("/fleet/alerts?limit=5").get_json()
        self.assertEqual(len(body["alerts"]), 5)
        for a in body["alerts"]:
            self.assertIn(a["zone"], ("safe", "alert", "danger", "unknown"))
            for k in ("slug", "station", "district", "zone", "reason",
                      "change_30d_m", "band_half_m"):
                self.assertIn(k, a)
        full = self.client.get("/fleet/alerts?limit=2000").get_json()
        self.assertEqual(full["count"], len(full["alerts"]))
        self.assertEqual(sum(full["counts"].values()), full["count"])

    def test_fleet_zone_filter(self):
        body = self.client.get("/fleet/alerts?zone=safe&limit=2000").get_json()
        self.assertTrue(body["alerts"])
        for a in body["alerts"]:
            self.assertEqual(a["zone"], "safe")
        narrow = self.client.get("/fleet/alerts?zone=danger,alert&limit=2000").get_json()
        for a in narrow["alerts"]:
            self.assertIn(a["zone"], ("danger", "alert"))

    def test_fleet_bad_zone_400(self):
        r = self.client.get("/fleet/alerts?zone=bogus")
        self.assertEqual(r.status_code, 400)


@unittest.skipUnless(has("api.py") and has("outputs/fleet_district.csv"),
                     "api module / district scan missing")
class TestDistricts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()

    def test_list_shape_and_counts(self):
        body = self.client.get("/districts?limit=200").get_json()
        self.assertTrue(body["count"] >= 30)
        self.assertEqual(sum(body["counts"].values()), body["count"])
        for d in body["districts"]:
            self.assertIn(d["scarcity"], ("scarce", "watch", "healthy", "unknown"))
            for k in ("district", "n_stations", "median_change_m", "n_decline",
                      "n_recover", "decline_share", "scarcity_basis",
                      "worst_180d_drop", "n_stressed", "n_deep_180d"):
                self.assertIn(k, d)

    def test_list_sort_name(self):
        body = self.client.get("/districts?sort=name&limit=200").get_json()
        names = [d["district"] for d in body["districts"]]
        self.assertEqual(names, sorted(names))

    def test_list_bad_sort_400(self):
        r = self.client.get("/districts?sort=bogus")
        self.assertEqual(r.status_code, 400)

    def test_detail_shape(self):
        body = self.client.get("/districts/KAUSHAMBI").get_json()
        self.assertEqual(body["district"], "KAUSHAMBI")
        self.assertIn(body["scarcity"], ("scarce", "watch", "healthy", "unknown"))
        for k in ("levels", "most_stressed", "top_driver", "model_accuracy",
                  "stations", "scarcity_basis"):
            self.assertIn(k, body)
        self.assertTrue(body["stations"])
        for s in body["stations"]:
            self.assertIn(s["zone"], ("safe", "alert", "danger", "unknown"))

    def test_detail_unknown_404(self):
        r = self.client.get("/districts/BOGUSXYZ")
        self.assertEqual(r.status_code, 404)
        self.assertIn("available", r.get_json())


@unittest.skipUnless(has("api.py") and has("outputs/fleet_forecast.csv"),
                     "api module / fleet scan missing")
class TestFleetRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()

    def test_recovery_sorted_desc(self):
        body = self.client.get("/fleet/recovery?limit=50").get_json()
        self.assertTrue(body["items"])
        chgs = [i["change_30d_m"] for i in body["items"] if i["change_30d_m"] is not None]
        self.assertEqual(chgs, sorted(chgs, reverse=True))
        for i in body["items"]:
            for k in ("slug", "station", "district", "change_30d_m",
                      "direction", "band_half_m", "category"):
                self.assertIn(k, i)

    def test_decline_sorted_asc(self):
        body = self.client.get("/fleet/recovery?sort=decline&limit=50").get_json()
        chgs = [i["change_30d_m"] for i in body["items"] if i["change_30d_m"] is not None]
        self.assertEqual(chgs, sorted(chgs))

    def test_recovery_district_filter(self):
        body = self.client.get("/fleet/recovery?district=KAUSHAMBI&limit=200").get_json()
        self.assertTrue(body["items"])
        for i in body["items"]:
            self.assertEqual(i["district"], "KAUSHAMBI")

    def test_recovery_bad_sort_400(self):
        r = self.client.get("/fleet/recovery?sort=bogus")
        self.assertEqual(r.status_code, 400)


@unittest.skipUnless(has("api.py") and has("data/aligned/table_6h.parquet"),
                     "api module / aligned table missing")
class TestAssistantChatGet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()

    def test_chat_get_works(self):
        r = self.client.get("/assistant/chat?question=latest+level%3F&station=ASHADHA+PRATHMIK+VIDYALAYA")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:300])
        body = r.get_json()
        self.assertIn("answer", body)
        self.assertEqual(body["station"], "ASHADHA PRATHMIK VIDYALAYA")

    def test_chat_get_needs_question(self):
        r = self.client.get("/assistant/chat")
        self.assertEqual(r.status_code, 400)

    def test_chat_alias_get(self):
        r = self.client.get("/assistant_chat?question=latest+level%3F")
        self.assertEqual(r.status_code, 200)
        self.assertIn("answer", r.get_json())


@unittest.skipUnless(has("api.py") and has("data/aligned/table_6h.parquet"),
                     "api module / aligned table missing")
class TestUnderscoreAliases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()
        cls.slug = api.slugify("ASHADHA PRATHMIK VIDYALAYA")

    def test_assistant_chat_alias(self):
        r = self.client.post("/assistant_chat", json={
            "question": "latest level?",
            "station": "ASHADHA PRATHMIK VIDYALAYA"})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:300])
        self.assertIn("answer", r.get_json())

    def test_series_alias(self):
        r = self.client.get(f"/stations_{self.slug}_series?drivers=rain&limit=2")
        self.assertEqual(r.status_code, 200)
        self.assertIn("points", r.get_json())

    def test_health_check_alias(self):
        self.assertEqual(self.client.get("/health_check").status_code, 200)

    def test_unknown_still_404_with_hint(self):
        r = self.client.get("/totally_bogus_xyz")
        self.assertEqual(r.status_code, 404)
        self.assertIn("available", r.get_json())
        # near-miss paths get a suggestion
        r2 = self.client.get("/healht")
        self.assertEqual(r2.status_code, 404)
        body = r2.get_json()
        self.assertIn("available", body)
        self.assertEqual(body.get("did_you_mean"), "/health")


if __name__ == "__main__":
    unittest.main()
