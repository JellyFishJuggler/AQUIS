"""verification contract — archive-before-overwrite, scoring, ledger, drift.

Network-free: monkeypatches the alignment engine with a tiny realised table,
and points publish/verification at temp dirs. Exercises the exact flow a real
forecast cycle runs: publish (archiving the previous generation), then
verify_forecasts scoring realised readings against the *older* anchor.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from refresh import config, publish, verification
from tests._helpers import has

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pod(station: str, anchor: str, q50: float = -1.0,
         generated: str | None = None) -> dict:
    steps = []
    for i in range(2, 7):
        t = (pd.Timestamp(anchor) + pd.Timedelta(hours=6 * i)).isoformat()
        steps.append({"time": t, "q05": q50 - 0.4, "q50": q50, "q95": q50 + 0.4,
                      "gwl": q50})
    pod = {"station": station, "anchor_time": anchor, "model": "test-v1",
           "trajectory": steps}
    if generated:
        pod["forecast_generated"] = generated
    return pod


@unittest.skipUnless(has("refresh/publish.py"), "refresh package missing")
class TestVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.real_cfg = verification.cfg_orig = getattr(
            verification, "cfg_orig", None)
        # isolate publish/verification output dirs
        self.fdir = self.tmp / "forecasts"
        self.adir = self.tmp / "forecast_archive"
        self.vdir = self.tmp / "verification"
        self.sdir = self.tmp / "staging"
        self.vdir.mkdir(parents=True)
        self.old_attrs = {
            "FORECASTS_DIR": publish.FORECASTS_DIR,
            "FORECAST_ARCHIVE": publish.FORECAST_ARCHIVE,
            "STAGING_DIR": publish.STAGING_DIR,
            "PARCEL_PATH": publish.PARCEL_PATH,
            "META_PATH": publish.META_PATH,
            "VERIF_LEDGER": verification.VERIF_LEDGER,
            "VERIF_SUMMARY": verification.VERIF_SUMMARY,
        }
        publish.FORECASTS_DIR = self.fdir
        publish.FORECAST_ARCHIVE = self.adir
        publish.STAGING_DIR = self.sdir
        publish.PARCEL_PATH = self.tmp / "forecasts.parquet"
        publish.META_PATH = self.tmp / "meta.json"
        verification.VERIF_LEDGER = self.vdir / "verification.parquet"
        verification.VERIF_SUMMARY = self.vdir / "verification_summary.json"

        # fake realised readings: station A advances past its anchor, B does not
        self.actuals = pd.DataFrame({
            "Station": ["A"] * 4,
            "time": [pd.Timestamp("2026-09-01 12:00:00") + pd.Timedelta(hours=6 * i)
                     for i in range(4)],
            "gwl": [-1.0, -1.1, -1.2, -1.3],
        })
        self.old_iter = verification._read_aligned_actuals
        verification._read_aligned_actuals = lambda: self.actuals.copy()

    def tearDown(self):
        publish.FORECASTS_DIR = self.old_attrs["FORECASTS_DIR"]
        publish.FORECAST_ARCHIVE = self.old_attrs["FORECAST_ARCHIVE"]
        publish.STAGING_DIR = self.old_attrs["STAGING_DIR"]
        publish.PARCEL_PATH = self.old_attrs["PARCEL_PATH"]
        publish.META_PATH = self.old_attrs["META_PATH"]
        verification.VERIF_LEDGER = self.old_attrs["VERIF_LEDGER"]
        verification.VERIF_SUMMARY = self.old_attrs["VERIF_SUMMARY"]
        self.fdir.glob("*.json")
        for d in (self.fdir, self.adir, self.vdir):
            shutil.rmtree(d, ignore_errors=True)
        verification._read_aligned_actuals = self.old_iter

    def test_publish_archives_previous_generation(self):
        old = _pod("A", "2026-08-01T00:00:00", q50=-0.8)
        publish.station_forecast_path("A").parent.mkdir(parents=True, exist_ok=True)
        publish.atomic_write_json(publish.station_forecast_path("A"), old)

        new = _pod("A", "2026-09-01T12:00:00", q50=-1.0)
        publish.publish_forecasts([new], engine="e", model_version="v2",
                                  model_trained_at=None, keep_staging=False)

        self.assertTrue(publish.station_forecast_path("A").exists())
        archived = list(publish.FORECAST_ARCHIVE.glob("*.json"))
        self.assertEqual(len(archived), 1)
        saved = json.loads(archived[0].read_text())
        self.assertEqual(saved["anchor_time"], "2026-08-01T00:00:00")
        self.assertTrue(saved["trajectory"])

    def test_verify_scores_archived_anchor_against_realised(self):
        # cycle 1: publish forecast anchored at the last reading BEFORE the update
        old = _pod("A", "2026-09-01T06:00:00", q50=-1.0,
                   generated="2026-09-01T06:00:00")
        publish.publish_forecasts([old], engine="e", model_version="v1",
                                  model_trained_at=None, keep_staging=False)
        # the archive now holds the "previous" generation; a second publish
        # overwrites the live file (as a real cycle would when data advances)
        fresh = _pod("A", "2026-09-01T12:00:00", q50=-1.05,
                     generated="2026-09-01T12:30:00")
        publish.publish_forecasts([fresh], engine="e", model_version="v2",
                                  model_trained_at=None, keep_staging=False)

        results = verification.verify_forecasts(min_generation_age_s=0)
        self.assertGreater(results["scored"], 0)
        self.assertGreater(results["appended"], 0)
        self.assertGreater(results["ledger_rows"], 0)

        # each scored row tests q50 against the realised value at that step
        ledger = verification._load_ledger()
        self.assertFalse(ledger.empty)
        nonnull = ledger[ledger["actual"].notna()]
        self.assertEqual(len(nonnull), len(ledger))
        self.assertTrue((ledger["horizon_h"] >= 6).all())

    def test_summary_and_drift_defaults(self):
        summary = verification.build_summary(pd.DataFrame())
        self.assertEqual(summary["drift"]["tripped"], False)
        self.assertEqual(summary["windows"], {})
        self.assertIsNone(summary["baseline_30d_rmse"])

    def test_summary_windows_after_scoring(self):
        old = _pod("A", "2026-09-01T06:00:00", q50=-1.0,
                   generated="2026-09-01T06:00:00")
        fresh = _pod("A", "2026-09-01T12:00:00", q50=-1.05,
                     generated="2026-09-01T12:30:00")
        publish.publish_forecasts([old], engine="e", model_version="v1",
                                  model_trained_at=None, keep_staging=False)
        publish.publish_forecasts([fresh], engine="e", model_version="v2",
                                  model_trained_at=None, keep_staging=False)
        verification.verify_forecasts(min_generation_age_s=0)
        from refresh.verification import VERIF_LEDGER
        ledger = verification._load_ledger()
        summary = verification.build_summary(ledger)
        self.assertIn("30d", summary["windows"])
        self.assertGreaterEqual(summary["windows"]["30d"]["n"], 1)

    def test_dup_rows_not_appended_twice(self):
        old = _pod("A", "2026-09-01T06:00:00", q50=-1.0,
                   generated="2026-09-01T06:00:00")
        publish.publish_forecasts([old], engine="e", model_version="v1",
                                  model_trained_at=None, keep_staging=False)
        fresh = _pod("A", "2026-09-01T12:00:00", q50=-1.05,
                     generated="2026-09-01T12:30:00")
        publish.publish_forecasts([fresh], engine="e", model_version="v2",
                                  model_trained_at=None, keep_staging=False)

        verification.verify_forecasts(min_generation_age_s=0)
        n1 = verification._load_ledger()
        verification.verify_forecasts(min_generation_age_s=0)
        n2 = verification._load_ledger()
        self.assertEqual(len(n2), len(n1))


@unittest.skipUnless(has("refresh/scheduler.py"), "refresh package missing")
class TestSchedulerDrift(unittest.TestCase):
    """drift_retrain_due: per-summary streak, armed only once per summary."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        from refresh import scheduler, state
        self.sched = scheduler
        self.sup = self.tmp / "verification_summary.json"
        self.old_path = verification.VERIF_SUMMARY
        verification.VERIF_SUMMARY = self.sup
        self.st = state.RefreshState(path=self.tmp / "state.json")

    def tearDown(self):
        verification.VERIF_SUMMARY = self.old_path

    def _write_summary(self, tripped: bool, ts: str, reason: str | None = None):
        self.sup.parent.mkdir(parents=True, exist_ok=True)
        self.sup.write_text(json.dumps({"drift": {
            "tripped": tripped, "reason": reason, "updated_ts": ts}}))

    def test_no_summary_not_due(self):
        self.assertFalse(self.sched.drift_retrain_due(None, self.st))

    def test_streak_arms_only_after_consecutive_summaries(self):
        self._write_summary(True, "2026-09-11T10:00:00", reason="rmse trip")
        self.assertFalse(self.sched.drift_retrain_due(None, self.st))
        self.assertEqual(self.st.get("drift_streak"), 1)

        self._write_summary(True, "2026-09-11T10:30:00", reason="rmse trip")
        self.assertTrue(self.sched.drift_retrain_due(None, self.st))
        self.assertEqual(self.st.get("drift_streak"), 2)

        # same summary again must not double-count
        self.assertTrue(self.sched.drift_retrain_due(None, self.st))
        self.assertEqual(self.st.get("drift_streak"), 2)

    def test_cleared_when_healthy(self):
        self._write_summary(True, "2026-09-11T10:00:00")
        self.sched.drift_retrain_due(None, self.st)
        self.assertEqual(self.st.get("drift_streak"), 1)
        self._write_summary(False, "2026-09-11T11:00:00")
        self.sched.drift_retrain_due(None, self.st)
        self.assertEqual(self.st.get("drift_streak"), 0)


@unittest.skipUnless(os.environ.get("AQUIS_APPTEST"), "set AQUIS_APPTEST=1 to run")
class TestVerificationAppTest(unittest.TestCase):
    def test_verification_page_renders(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app_pages" / "verification.py"),
                               default_timeout=60)
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0, at.exception)


@unittest.skipUnless(has("03_merge_normalize.py"), "merge/normalize missing")
class TestTrimHighSide(unittest.TestCase):
    """High-side apex-relax: monsoon-recharge readings survive the trim.

    The 0.995 per-station quantile historically sat above every kept GWL,
    silently dropping genuine shallow-monsoon recovery. The bound is now
    floored at GWL_HIGH_PLAUSIBLE_M so only physically-implausible values
    (water above the well datum) are trimmed on the high side.
    """

    def _df(self, values):
        n = len(values)
        return pd.DataFrame({
            "Station": ["S"] * n,
            "time": [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(n)],
            "value": values,
        })

    def test_high_visibly_recent_reading_survives(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "merge_norm", ROOT / "03_merge_normalize.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # 500 dry readings at -12..-6 plus a genuinely shallow monsoon reading
        values = list(np.linspace(-12.0, -6.0, 500)) + [-1.2, 0.05]
        df, dropped = m.clean(self._df(values), None, None, "gwl")
        self.assertIn(-1.2, df["value"].values)
        self.assertNotIn(0.05, df["value"].values)  # water above the well datum
        self.assertGreater(len(df), 490)

    def test_low_sentinel_still_trimmed(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "merge_norm", ROOT / "03_merge_normalize.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        values = list(np.linspace(-120.0, -10.0, 400))
        df, _ = m.clean(self._df(values), None, None, "gwl")
        self.assertLess(df["value"].min(), -10.0)


if __name__ == "__main__":
    unittest.main()