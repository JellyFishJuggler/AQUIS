"""refresh tests — config, state/lock/journal, atomic publish, freshness.

Everything here is network-free and uses tiny temp-dir fixtures so the suite
runs anywhere (stdlib unittest only).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from refresh import config, publish, state
from tests._helpers import has

REFRESH_DIR = config.REFRESH_DIR
STATE_FILE = state.state_path()


@unittest.skipUnless(has("refresh/config.py"), "refresh package missing")
class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def tearDown(self):
        os.environ.pop("AQUIS_REFRESH_INTERVAL_HOURS", None)
        os.environ.pop("AQUIS_FEATURE_MODE", None)

    def test_defaults(self):
        cfg = config.RefreshConfig()
        cfg.validate()
        self.assertEqual(cfg.refresh_interval_hours, 6.0)
        self.assertEqual(cfg.retrain_cadence_hours, 24.0)
        self.assertEqual(cfg.feature_mode, "flat")
        self.assertEqual(cfg.max_stations, 0)  # 0 = all stations
        self.assertGreater(cfg.stale_after_hours, 0)

    def test_bad_cadence_and_mode_rejected(self):
        cfg = config.RefreshConfig(refresh_interval_hours=0)
        with self.assertRaises(config.ConfigError):
            cfg.validate()
        cfg = config.RefreshConfig(feature_mode="bogus")
        with self.assertRaises(config.ConfigError):
            cfg.validate()

    def test_file_save_roundtrip(self):
        p = self.tmp / "cfg.json"
        cfg = config.RefreshConfig(refresh_interval_hours=3)
        p.write_text(json.dumps(asdict(cfg), indent=2))
        loaded = config.load_config(p)
        self.assertEqual(loaded.refresh_interval_hours, 3)

    def test_env_override_cadence_and_mode(self):
        os.environ["AQUIS_REFRESH_INTERVAL_HOURS"] = "12"
        os.environ["AQUIS_FEATURE_MODE"] = "future"
        cfg = config.load_config(self.tmp / "none.json", write_if_missing=False)
        cfg.validate()
        self.assertEqual(cfg.refresh_interval_hours, 12.0)
        self.assertEqual(cfg.feature_mode, "future")


@unittest.skipUnless(has("refresh/state.py"), "refresh package missing")
class TestStateAndLock(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_state_atomic_roundtrip(self):
        s = state.RefreshState(self.tmp / "state.json")
        s.update(last_forecast_refresh_ts="2026-09-10T00:00:00+00:00")
        s2 = state.RefreshState(self.tmp / "state.json")
        self.assertEqual(s2.get("last_forecast_refresh_ts"),
                         "2026-09-10T00:00:00+00:00")

    def test_corrupt_state_resets_but_records(self):
        p = self.tmp / "state.json"
        p.write_text("{not json!!")
        s = state.RefreshState(p)
        self.assertIn("state.corrupt", json.loads(self._last_journal_line())["kind"])
        s.update(next_refresh_due="x")
        self.assertEqual(state.RefreshState(p).get("next_refresh_due"), "x")

    def _last_journal_line(self):
        return state.journal_path().read_text().splitlines()[-1]

    def test_journal_append(self):
        s = state.RefreshState(self.tmp / "state.json")
        before = len(state.journal_path().read_text().splitlines() or [""])
        s.record("fetch", ok=True, districts=5)
        s.record("forecast", ok=True, forecasted=100)
        lines = state.journal_path().read_text().splitlines()
        self.assertEqual(len(lines), before + 2)
        self.assertEqual(json.loads(lines[-1])["kind"], "forecast")
        self.assertEqual(json.loads(lines[-2])["kind"], "fetch")

    def test_lock_exclusion(self):
        state.REFRESH_DIR = Path(self.tmp)  # keep lock file in temp
        try:
            with state.refresh_lock():
                with self.assertRaises(state.LockBusy):
                    with state.refresh_lock():
                        pass
        finally:
            state.REFRESH_DIR = REFRESH_DIR

    def test_lock_acquired_when_free(self):
        state.REFRESH_DIR = Path(self.tmp)
        try:
            with state.refresh_lock():
                pass  # must not raise
        finally:
            state.REFRESH_DIR = REFRESH_DIR


@unittest.skipUnless(has("refresh/publish.py"), "refresh package missing")
class TestPublish(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._prev = (publish.FORECASTS_DIR, publish.STAGING_DIR,
                      publish.META_PATH, publish.PARCEL_PATH)
        publish.FORECASTS_DIR = self.tmp / "forecasts"
        publish.STAGING_DIR = self.tmp / "staging"
        publish.META_PATH = self.tmp / "meta.json"
        publish.PARCEL_PATH = self.tmp / "forecasts.parquet"

    def tearDown(self):
        (publish.FORECASTS_DIR, publish.STAGING_DIR,
         publish.META_PATH, publish.PARCEL_PATH) = self._prev

    def _fc(self, i, anchor="2026-09-05T00:00:00+00:00"):
        return {
            "station": f"ST{str(i).zfill(3)}", "anchor_time": anchor,
            "anchor_gwl": -10.0 + i * 0.1,
            "trajectory_30d": {"level": -9.0 + i * 0.1, "q05": -9.5, "q95": -8.5,
                               "change": 1.0},
            "direction": {"label": "rising"},
            "overall_confidence": {"level": "HIGH", "reason": "ok"},
            "trajectory": [{"time": anchor, "gwl": -9.0, "q05": -9.5,
                            "q50": -9.0, "q95": -8.5, "confidence_level": "HIGH",
                            "driver_source": "Open-Meteo forecast (<=16d)"}],
        }

    def test_publish_creates_per_station_and_manifest_last(self):
        meta = publish.publish_forecasts([self._fc(1), self._fc(2)],
                                         engine="trajectory-v2",
                                         model_version="v1", model_trained_at="t1")
        self.assertEqual(meta["stations_forecasted"], 2)
        self.assertTrue(publish.META_PATH.exists())
        self.assertTrue(publish.PARCEL_PATH.exists())
        for s in ("ST001", "ST002"):
            self.assertTrue(publish.station_forecast_path(s).exists())
        fleet = pd.read_parquet(publish.PARCEL_PATH)
        self.assertEqual(set(fleet["station"]), {"ST001", "ST002"})

    def test_manifest_is_commit_point(self):
        keep = self.tmp / "keep.json"
        keep.write_text(json.dumps({"old": True}))
        # simulate crash mid-publish: files staged but no manifest written
        fc = self._fc(3)
        publish.atomic_write_json(publish.STAGING_DIR / "x.json", fc)
        # previous manifest must be the one we read (we never overwrite it here):
        self.assertFalse(publish.META_PATH.exists())
        publish.publish_forecasts([fc], engine="e", model_version=None,
                                  model_trained_at=None)
        self.assertIsNotNone(publish.read_meta().get("forecast_generation_ts"))

    def test_bad_station_reported_not_fatal(self):
        bad = {"station": "BAD", "error": "no data for station BAD"}
        ok = self._fc(4)
        meta = publish.publish_forecasts([bad, ok], engine="e",
                                         model_version=None, model_trained_at=None)
        self.assertEqual(meta["stations_forecasted"], 1)
        self.assertEqual(meta["stations_failed"][0]["station"], "BAD")

    def test_freshness_stale_and_fresh(self):
        meta = publish.publish_forecasts([self._fc(7)], engine="e",
                                         model_version="v9", model_trained_at=None)
        f = publish.freshness_for("ST007")
        self.assertEqual(f["model_version"], "v9")
        self.assertTrue(f["published"])
        publish.update_freshness(meta, stale_after_hours=6, last_refresh_ts=None)
        self.assertEqual(publish.read_meta()["data_status"], "stale")
        self.assertEqual(publish.freshness_for("ST007")["data_status"], "stale")

    def test_station_key_deterministic(self):
        self.assertEqual(publish.station_key("ABC"), publish.station_key("ABC"))


if __name__ == "__main__":
    unittest.main()