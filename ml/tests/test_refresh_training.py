"""refresh retrain/gate/scheduler tests (no model training, no network).

Assertions:
  * moving cutoffs: train_cut = now-(holdout+backtest), val_end = now-backtest
    and ES val exactly ``holdout`` days wide;
  * gate promote/keep logic against the 5 quality gates (candidate<incumbent,
    <persistence, calibrated coverage, short-horizon, min n);
  * schedule decisions: refresh fires when due, retrain only inside
    retrain_at_hours windows.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from refresh import model_update, pipeline, scheduler
from refresh.config import RefreshConfig, models_frozen
from refresh.state import RefreshState
from tests._helpers import has

NOW = datetime(2026, 9, 10, 6, 0, tzinfo=timezone.utc)


def _per_h_df(cand30: float, inc30: float, persist30: float,
              n: int = 40, short_cand: bool = True) -> dict:
    rows_c = [{"horizon_h": 24, "rmse": 0.05 if short_cand else 9.9,
               "rmse_persist": 0.1, "n": n, "coverage_raw": 0.9},
              {"horizon_h": 720, "rmse": cand30, "rmse_persist": persist30,
               "n": n, "coverage_raw": 0.9}]
    rows_i = [{"horizon_h": 24, "rmse": 0.06, "rmse_persist": 0.1,
               "n": n, "coverage_raw": 0.9},
              {"horizon_h": 720, "rmse": inc30, "rmse_persist": persist30,
               "n": n, "coverage_raw": 0.9}]
    return {
        "per_h": {"candidate": pd.DataFrame(rows_c),
                  "incumbent": pd.DataFrame(rows_i)},
        "calib": {"horizons_h": [720],
                  "coverage_calibrated": [0.90]},
    }


@unittest.skipUnless(has("refresh/model_update.py"), "refresh missing")
class TestCutoffs(unittest.TestCase):
    def test_moving_window_geometry(self):
        cfg = RefreshConfig(candidate_holdout_days=90,
                            candidate_backtest_window_days=90)
        train_cut, val_end, now = model_update._cutoffs(cfg, pd.Timestamp("2026-09-10 00:00"))
        self.assertEqual((val_end - train_cut).days, 90)
        self.assertEqual((now - val_end).days, 90)


@unittest.skipUnless(has("refresh/model_update.py"), "refresh missing")
class TestGate(unittest.TestCase):
    def test_promote_when_candidate_wins(self):
        cfg = RefreshConfig()
        gate = model_update._gate(cfg, _per_h_df(1.0, 1.2, 1.4))
        self.assertEqual(gate["decision"], "promote")
        self.assertTrue(all(gate["reasons"].values()))

    def test_keep_when_loses_to_incumbent(self):
        cfg = RefreshConfig()
        gate = model_update._gate(cfg, _per_h_df(1.5, 1.2, 1.4))
        self.assertEqual(gate["decision"], "keep")
        self.assertFalse(gate["reasons"]["beats_incumbent"])

    def test_keep_when_loses_to_persistence(self):
        cfg = RefreshConfig()
        gate = model_update._gate(cfg, _per_h_df(1.0, 1.2, 0.8))
        self.assertEqual(gate["decision"], "keep")
        self.assertFalse(gate["reasons"]["beats_persistence"])

    def test_keep_on_low_min_scores(self):
        cfg = RefreshConfig()
        gate = model_update._gate(cfg, _per_h_df(1.0, 1.2, 1.4, n=5))
        self.assertEqual(gate["decision"], "keep")
        self.assertFalse(gate["reasons"]["min_scores"])

    def test_keep_on_bad_coverage(self):
        cfg = RefreshConfig(coverage_target=0.9)
        res = _per_h_df(1.0, 1.2, 1.4)
        res["calib"]["coverage_calibrated"] = [0.5]
        gate = model_update._gate(cfg, res)
        self.assertEqual(gate["decision"], "keep")
        self.assertFalse(gate["reasons"]["coverage_ok"])

    def test_keep_on_short_horizon_failure(self):
        cfg = RefreshConfig()
        gate = model_update._gate(cfg, _per_h_df(1.0, 1.2, 1.4,
                                                 short_cand=False))
        self.assertEqual(gate["decision"], "keep")
        self.assertFalse(gate["reasons"]["short_horizons_ok"])

    def test_promote_respects_tolerance(self):
        # candidate barely worse is accepted when tolerance allows it
        cfg = RefreshConfig(promote_tolerance=0.1)
        gate = model_update._gate(cfg, _per_h_df(1.25, 1.2, 1.4))
        self.assertEqual(gate["decision"], "promote")


@unittest.skipUnless(has("refresh/scheduler.py"), "refresh missing")
class TestSchedulerDecisions(unittest.TestCase):
    def setUp(self):
        self.cfg = RefreshConfig()
        self.state = RefreshState.__new__(RefreshState)   # no file I/O
        self.state._data = {}

    def test_refresh_due_when_unset_or_past(self):
        self.assertTrue(scheduler.refresh_due(self.cfg, self.state, NOW))
        self.state._data["next_refresh_due"] = (
            NOW - timedelta(minutes=1)).isoformat()
        self.assertTrue(scheduler.refresh_due(self.cfg, self.state, NOW))

    def test_refresh_not_due_in_future(self):
        self.state._data["next_refresh_due"] = (
            NOW + timedelta(minutes=1)).isoformat()
        self.assertFalse(scheduler.refresh_due(self.cfg, self.state, NOW))

    def test_retrain_only_inside_window(self):
        in_hour = NOW.replace(hour=2)
        out_cfg = RefreshConfig(retrain_at_hours=[2, 14])
        self.state._data["next_retrain_due"] = (
            in_hour - timedelta(hours=1)).isoformat()
        self.assertTrue(scheduler.retrain_due(out_cfg, self.state, in_hour))
        self.assertFalse(scheduler.retrain_due(out_cfg, self.state,
                                               in_hour.replace(hour=7)))

    def test_retrain_blocked_by_future_due(self):
        self.state._data["next_retrain_due"] = (NOW + timedelta(hours=2)).isoformat()
        self.assertFalse(scheduler.retrain_due(self.cfg, self.state,
                                               NOW.replace(hour=2)))


@unittest.skipUnless(has("refresh/scheduler.py"), "refresh missing")
class TestModelFreeze(unittest.TestCase):
    """models_frozen_until must block retrain + promotion until the instant."""

    FUTURE = "2030-01-01T00:00:00+00:00"
    NOW = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)

    def test_models_frozen_truth_table(self):
        self.assertFalse(models_frozen(RefreshConfig(), self.NOW))  # empty = off
        frozen = RefreshConfig(models_frozen_until=self.FUTURE)
        self.assertTrue(models_frozen(frozen, self.NOW))
        past = RefreshConfig(models_frozen_until="2026-01-01T00:00:00+00:00")
        self.assertFalse(models_frozen(past, self.NOW))
        # fail safe: a malformed instant keeps the freeze ON
        self.assertTrue(models_frozen(RefreshConfig(models_frozen_until="not-a-date"),
                                      self.NOW))

    def test_validate_rejects_malformed_freeze(self):
        from refresh.config import ConfigError
        with self.assertRaises(ConfigError):
            RefreshConfig(models_frozen_until="not-a-date").validate()

    def test_promote_refuses_while_frozen(self):
        cfg = RefreshConfig(models_frozen_until=self.FUTURE)
        with self.assertRaises(model_update.RetrainError):
            model_update.promote(cfg, object(), {"reasons": {}},
                                 cand_dir=Path("/nonexistent"))

    def test_scheduler_skips_retrain_while_frozen(self):
        import contextlib
        calls = []
        orig = scheduler.pipeline.run_retrain_cycle
        orig_lock = scheduler.refresh_lock
        def _rec(**kw):
            calls.append(kw)
            return {"gate": {}}
        scheduler.pipeline.run_retrain_cycle = _rec
        scheduler.refresh_lock = contextlib.nullcontext
        try:
            state = RefreshState.__new__(RefreshState)
            state._data = {}
            sched = {  # force the retrain branch to be "wanted"
                "refresh_due": lambda *a, **k: False,
                "retrain_due": lambda *a, **k: True,
                "drift_retrain_due": lambda *a, **k: False,
            }
            saved = {n: getattr(scheduler, n) for n in sched}
            for n, fn in sched.items():
                setattr(scheduler, n, fn)
            try:
                frozen = RefreshConfig(models_frozen_until=self.FUTURE)
                scheduler.scheduler_loop(cfg=frozen, state=state, once=True)
                self.assertEqual(calls, [])
                scheduler.scheduler_loop(cfg=RefreshConfig(), state=state, once=True)
                self.assertEqual(len(calls), 1)
            finally:
                for n, fn in saved.items():
                    setattr(scheduler, n, fn)
        finally:
            scheduler.pipeline.run_retrain_cycle = orig
            scheduler.refresh_lock = orig_lock


if __name__ == "__main__":
    unittest.main()