"""Read-only influence endpoints: availability contract + artifact handling.

The feature must be *invisible* by default (``available:false``) and must never
raise, even with missing or corrupt JSON. These tests drive ``ml/api.py`` with a
temp artifact directory and a temp ``runtime.json`` so the production runtime
file and the real ``outputs/`` tree are never touched.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._helpers import has

CAVEAT = ("Experimental. Shows model sensitivity between stations, not proven "
          "causation. Pumping and local geology are not observed.")


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _station_payload() -> dict:
    return {
        "meta": {"generated_at": "2026-01-01 00:00:00", "caveat": CAVEAT},
        "stations": {
            "28": {
                "Alpha": {
                    "node": 0, "district": "D1",
                    "strength_in": 0.4, "strength_out": 0.3,
                    "influenced_by": [
                        {"station": f"S{i}", "district": "D1", "sensitivity": 0.1 * i}
                        for i in range(1, 9)
                    ],
                    "influences": [
                        {"station": f"T{i}", "district": "D2", "sensitivity": -0.05 * i}
                        for i in range(1, 9)
                    ],
                }
            }
        },
    }


def _district_payload() -> dict:
    return {
        "meta": {"caveat": CAVEAT},
        "28": {
            "districts": ["D1", "D2"],
            "matrix": [[0.0, 0.25], [0.5, 0.0]],
            "top_pairs": [{"from": "D2", "to": "D1", "sensitivity": 0.5}],
        },
    }


def _graph_payload() -> dict:
    return {
        "meta": {"n_nodes": 2, "n_edges": 1},
        "nodes": [{"i": 0, "station": "Alpha", "district": "D1",
                   "lat": 26.0, "lon": 80.0},
                  {"i": 1, "station": "Beta", "district": "D2",
                   "lat": 26.1, "lon": 80.1}],
        "edges": [{"i": 0, "j": 1, "km": 15.0}],
    }


def _consistency_payload() -> dict:
    return {
        "meta": {"generated_at": "2026-01-01 00:00:00", "snapshots": 4,
                 "horizons_h": [28, 120]},
        "feature_audit": {"lag_columns_changed_by_anchor_perturbation": False,
                          "rolling_columns_changed_by_anchor_perturbation": True},
        "roll_reproducibility": {"max_abs_diff_m": 2.09e-06, "rows": 64},
        "example_deltas": {"step_m": 0.05},
        "lag_audit": {"lags_are_past_values": True, "per_column": []},
        "summary": {"28": {"snapshots": 4, "diff_rel_frobenius_mean": 24.0,
                           "corr_offdiag_min": -0.61, "top10_source_overlap_min": 3},
                    "120": {"snapshots": 4, "diff_rel_frobenius_mean": 9.16,
                            "corr_offdiag_min": 0.78, "top10_source_overlap_min": 10}},
        "snapshots": [{"kind": "val", "time": "2025-10-31", "nodes": 64,
                       "horizons": {"28": {"diff_max_abs": 0.02}}}],
    }


@unittest.skipUnless(has("api.py"), "api module missing")
class TestInfluenceDefaultOff(unittest.TestCase):
    """With the production runtime file (no flag) everything is hidden."""

    @classmethod
    def setUpClass(cls):
        import api
        cls.client = api.app.test_client()

    def _check(self, url: str):
        r = self.client.get(url)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:300])
        body = r.get_json()
        self.assertIs(body.get("available"), False)
        self.assertIn("reason", body)

    def test_meta_hidden(self):
        self._check("/influence/meta")

    def test_graph_hidden(self):
        self._check("/influence/graph")

    def test_station_hidden(self):
        self._check("/influence/station/anything")

    def test_district_hidden(self):
        self._check("/influence/district/anything")

    def test_consistency_hidden(self):
        self._check("/influence/consistency")

    def test_aliases_hidden(self):
        for u in ("/api/influence/meta", "/api/influence/graph",
                  "/api/influence/station/x", "/api/influence/district/x",
                  "/api/influence/consistency"):
            with self.subTest(url=u):
                self._check(u)

    def test_unknown_route_lists_influence_endpoints(self):
        r = self.client.get("/definitely-not-a-route")
        self.assertEqual(r.status_code, 404)
        avail = r.get_json().get("available", [])
        self.assertIn("/influence/meta", avail)
        self.assertIn("/influence/graph", avail)
        self.assertIn("/influence/station/<id>", avail)
        self.assertIn("/influence/district/<name>", avail)
        self.assertIn("/influence/consistency", avail)


@unittest.skipUnless(has("api.py"), "api module missing")
class TestInfluenceWithArtifacts(unittest.TestCase):
    """Flag on + temp artifacts: payload shape, clamping, casefold, corruption."""

    @classmethod
    def setUpClass(cls):
        import api
        from serving import influence as I
        cls.api, cls.I = api, I
        cls.client = api.app.test_client()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.runtime = root / "runtime.json"
        _write(cls.runtime, {"influence_enabled": True})
        cls.files = {
            "station": root / "station_influence.json",
            "district": root / "district_influence.json",
            "graph": root / "graph.json",
            "meta": root / "influence_meta.json",
            "consistency": root / "influence_consistency.json",
        }
        _write(cls.files["station"], _station_payload())
        _write(cls.files["district"], _district_payload())
        _write(cls.files["graph"], _graph_payload())
        _write(cls.files["meta"], {"caveat_text": CAVEAT, "n_stations": 2})
        _write(cls.files["consistency"], _consistency_payload())
        cls._orig = (dict(I.FILES), I.RUNTIME)
        I.FILES = dict(cls.files)
        I.RUNTIME = cls.runtime
        I.clear_cache()

    @classmethod
    def tearDownClass(cls):
        cls.I.FILES, cls.I.RUNTIME = cls._orig
        cls.I.clear_cache()
        cls.tmp.cleanup()

    def test_meta_available_with_caveat(self):
        body = self.client.get("/influence/meta").get_json()
        self.assertTrue(body["available"])
        self.assertTrue(body["experimental"])
        self.assertEqual(body["caveat"], CAVEAT)

    def test_graph_available(self):
        body = self.client.get("/influence/graph").get_json()
        self.assertTrue(body["available"])
        self.assertEqual(len(body["nodes"]), 2)
        self.assertEqual(len(body["edges"]), 1)
        self.assertEqual(body["meta"]["n_edges"], 1)

    def test_station_top_clamped(self):
        body = self.client.get("/influence/station/Alpha?top=999").get_json()
        self.assertTrue(body["available"])
        self.assertEqual(body["station"], "Alpha")
        self.assertEqual(body["district"], "D1")
        self.assertIn("28", body["horizons"])
        self.assertLessEqual(len(body["horizons"]["28"]["influenced_by"]), 25)
        self.assertLessEqual(len(body["horizons"]["28"]["influences"]), 25)

    def test_station_top_default_five(self):
        body = self.client.get("/influence/station/Alpha").get_json()
        self.assertEqual(len(body["horizons"]["28"]["influenced_by"]), 5)
        self.assertEqual(len(body["horizons"]["28"]["influences"]), 5)

    def test_station_bad_top_falls_back(self):
        body = self.client.get("/influence/station/Alpha?top=abc").get_json()
        self.assertTrue(body["available"])
        self.assertEqual(len(body["horizons"]["28"]["influenced_by"]), 5)

    def test_station_unknown_is_hidden_not_500(self):
        r = self.client.get("/influence/station/NoSuchStation")
        self.assertEqual(r.status_code, 200)
        self.assertIs(r.get_json()["available"], False)

    def test_district_casefold(self):
        body = self.client.get("/influence/district/d1").get_json()
        self.assertTrue(body["available"])
        self.assertEqual(body["district"], "D1")
        inc = body["horizons"]["28"]["influenced_by"]
        self.assertEqual(inc[0]["district"], "D2")
        self.assertAlmostEqual(inc[0]["sensitivity"], 0.5)

    def test_consistency_available_summary_only(self):
        body = self.client.get("/influence/consistency").get_json()
        self.assertTrue(body["available"])
        self.assertTrue(body["experimental"])
        self.assertIn("28", body["summary"])
        self.assertIn("120", body["summary"])
        self.assertFalse(body["feature_audit"]
                         ["lag_columns_changed_by_anchor_perturbation"])
        self.assertTrue(body["feature_audit"]
                        ["rolling_columns_changed_by_anchor_perturbation"])
        self.assertNotIn("snapshots", body)          # matrices stay offline
        self.assertEqual(body["caveat"], CAVEAT)

    def test_consistency_alias(self):
        body = self.client.get("/api/influence/consistency").get_json()
        self.assertTrue(body["available"])

    def test_missing_consistency_file_hides_feature(self):
        path = self.files["consistency"]
        original = path.read_text(encoding="utf-8")
        try:
            path.unlink()
            self.I.clear_cache()
            body = self.client.get("/influence/consistency").get_json()
            self.assertIs(body["available"], False)
            self.assertEqual(body["reason"], "consistency_missing")
        finally:
            path.write_text(original, encoding="utf-8")
            self.I.clear_cache()

    def test_corrupt_consistency_file_hides_feature(self):
        path = self.files["consistency"]
        original = path.read_text(encoding="utf-8")
        try:
            path.write_text("{not json", encoding="utf-8")
            self.I.clear_cache()
            r = self.client.get("/influence/consistency")
            self.assertEqual(r.status_code, 200)
            self.assertIs(r.get_json()["available"], False)
        finally:
            path.write_text(original, encoding="utf-8")
            self.I.clear_cache()

    def test_corrupt_station_file_hides_feature(self):
        path = self.files["station"]
        original = path.read_text(encoding="utf-8")
        try:
            path.write_text("{not json", encoding="utf-8")
            self.I.clear_cache()
            r = self.client.get("/influence/station/Alpha")
            self.assertEqual(r.status_code, 200)
            self.assertIs(r.get_json()["available"], False)
        finally:
            path.write_text(original, encoding="utf-8")
            self.I.clear_cache()

    def test_missing_graph_file_hides_feature(self):
        path = self.files["graph"]
        original = path.read_text(encoding="utf-8")
        try:
            path.unlink()
            self.I.clear_cache()
            body = self.client.get("/influence/graph").get_json()
            self.assertIs(body["available"], False)
            self.assertEqual(body["reason"], "graph_missing")
        finally:
            path.write_text(original, encoding="utf-8")
            self.I.clear_cache()

    def test_flag_flip_is_picked_up(self):
        try:
            _write(self.runtime, {"influence_enabled": False})
            self.I.clear_cache()
            body = self.client.get("/influence/meta").get_json()
            self.assertIs(body["available"], False)
            self.assertEqual(body["reason"], "influence_disabled")
        finally:
            _write(self.runtime, {"influence_enabled": True})
            self.I.clear_cache()

    def test_cache_returns_same_object_until_mtime_changes(self):
        from serving import influence as I
        a = I._load("graph")
        b = I._load("graph")
        self.assertIs(a, b)
        path = self.files["graph"]
        original = path.read_text(encoding="utf-8")
        try:
            payload = _graph_payload()
            payload["meta"]["n_edges"] = 99
            _write(path, payload)
            c = I._load("graph")
            self.assertEqual(c["meta"]["n_edges"], 99)
        finally:
            path.write_text(original, encoding="utf-8")
            I.clear_cache()


if __name__ == "__main__":
    unittest.main()
