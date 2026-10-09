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

from refresh import inference, model_update, pipeline, scheduler
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


class _FakeState:
    """Minimal RefreshState stand-in: no file I/O, records every event."""

    def __init__(self):
        self._data = {}
        self.events = []

    def get(self, key, default=None):
        return self._data.get(key, default)

    def update(self, **fields):
        for k, v in fields.items():
            if v is not None:
                self._data[k] = v

    def record(self, kind, *, ok, msg="", **extra):
        self.events.append({"kind": kind, "ok": ok, "msg": msg, **extra})


@unittest.skipUnless(has("refresh/pipeline.py"), "refresh missing")
class TestNoNewSourceData(unittest.TestCase):
    """A clean zero-new-row fetch is a benign no-op; everything else raises."""

    def test_is_no_new_source_truth_table(self):
        f = pipeline._is_no_new_source
        self.assertTrue(f({"gwl": {"ok": True, "new_rows": 0}}))
        self.assertFalse(f({"gwl": {"ok": True, "new_rows": 5}}))
        self.assertFalse(f({"gwl": {"ok": False, "new_rows": 0}}))
        self.assertFalse(f({"gwl": {"ok": True}}))       # missing count -> not benign
        self.assertFalse(f({"gwl": {"ok": True, "new_rows": None}}))
        self.assertFalse(f({"gwl": {}}))
        self.assertFalse(f({}))
        self.assertFalse(f(None))

    def test_inference_empty_is_noop_only_when_allowed(self):
        tbl = pd.DataFrame({"Station": ["ST1", "ST2"],
                            "time": pd.to_datetime(["2026-09-01", "2026-09-01"]),
                            "gwl": [-5.0, -6.0]})
        eng = {"tbl": tbl, "fnames": []}
        prev = pd.Series(pd.to_datetime(["2026-09-01", "2026-09-01"]),
                         index=["ST1", "ST2"])
        saved_engine, saved_prev = inference._load_engine, inference.previous_anchors
        inference._load_engine = lambda: eng
        inference.previous_anchors = lambda: prev
        try:
            cfg = RefreshConfig(min_fresh_stations=20)
            st = _FakeState()
            r = inference.refresh_forecasts(cfg=cfg, state=st, allow_empty=True)
            self.assertTrue(r["no_new_source"])
            self.assertFalse(r["published"])
            self.assertEqual(r["forecasted"], 0)
            self.assertEqual(r["unchanged_skipped"], 2)
            self.assertTrue(r["table_latest"])
            with self.assertRaises(RuntimeError):
                inference.refresh_forecasts(cfg=cfg, state=st, allow_empty=False)
        finally:
            inference._load_engine = saved_engine
            inference.previous_anchors = saved_prev

    def _patch(self, **attrs):
        saved = {}
        for n, fn in attrs.items():
            saved[n] = getattr(pipeline, n)
            setattr(pipeline, n, fn)
        self.addCleanup(lambda: [setattr(pipeline, n, fn) for n, fn in saved.items()])

    def test_pipeline_noop_journals_and_rearms_with_floor(self):
        self._patch(
            sources=type("S", (), {"fetch_all": staticmethod(
                lambda cfg, dry_run=False: {"all_ok": True,
                                            "gwl": {"ok": True, "new_rows": 0,
                                                    "source_latest": "2026-09-01T00:00:00"}})})(),
            inference=type("I", (), {"refresh_forecasts": staticmethod(
                lambda **kw: {"no_new_source": True, "published": False,
                              "forecasted": 0, "unchanged_skipped": 3,
                              "table_latest": "2026-09-01 00:00:00"})})(),
            # force the grid slot to be *sooner* than the backoff floor
            next_grid_due=lambda cfg, now: now + timedelta(minutes=10),
        )
        cfg = RefreshConfig(no_new_data_backoff_min=90)
        st = _FakeState()
        rep = pipeline.run_forecast_cycle(cfg=cfg, state=st, do_fetch=True,
                                          do_features=False, do_inference=True,
                                          lock=False)
        self.assertEqual([e["kind"] for e in st.events],
                         ["fetch", "no_new_source"])
        ev = st.events[1]
        self.assertEqual(ev["source_latest"], "2026-09-01T00:00:00")
        self.assertEqual(ev["table_latest"], "2026-09-01 00:00:00")
        cycle = datetime.fromisoformat(rep["cycle_ts"])
        due = datetime.fromisoformat(st._data["next_refresh_due"])
        self.assertAlmostEqual((due - cycle).total_seconds(), 90 * 60, delta=5)
        self.assertNotIn("verification", rep)

    def test_pipeline_fetch_failure_still_raises(self):
        def _boom(cfg, dry_run=False):
            raise RuntimeError("network down")
        self._patch(sources=type("S", (), {"fetch_all": staticmethod(_boom)})())
        st = _FakeState()
        with self.assertRaises(RuntimeError):
            pipeline.run_forecast_cycle(cfg=RefreshConfig(), state=st, do_fetch=True,
                                        do_features=False, do_inference=True, lock=False)

    def test_pipeline_inference_error_raises_even_with_zero_rows(self):
        def _boom(**kw):
            raise RuntimeError("engine exploded")
        self._patch(
            sources=type("S", (), {"fetch_all": staticmethod(
                lambda cfg, dry_run=False: {"all_ok": True,
                                            "gwl": {"ok": True, "new_rows": 0}})})(),
            inference=type("I", (), {"refresh_forecasts": staticmethod(_boom)})(),
        )
        st = _FakeState()
        with self.assertRaises(RuntimeError):
            pipeline.run_forecast_cycle(cfg=RefreshConfig(), state=st, do_fetch=True,
                                        do_features=False, do_inference=True, lock=False)


if __name__ == "__main__":
    unittest.main()