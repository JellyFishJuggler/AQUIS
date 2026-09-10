"""Forecast UI tests — chart spec, snapshot PNG and (optionally) AppTest.

Pure unit tests run without Streamlit and verify the Altair spec produced by
``app_charts.trajectory_chart``, the Snapshot PNG payload, data-integrity of the
trajectory frame (120 genuine 6 h points, q05<=q50<=q95, no bridged/interpolated
rows), the dark-theme presentation and the "no other forecast model anywhere in
the Forecast UI" rule via a source-level grep.  AppTest headless checks run the
real page and are gated behind ``AQUIS_APPTEST=1`` because the trajectory engine
is slow on first invocation.

Run:  cd ml && venv/bin/python -m unittest tests.test_forecast_ui -v
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_charts import _walk  # noqa: E402

# ---------------------------------------------------------------------------
# synthetic trajectory dict (matches _trajectory.trajectory_forecast contract)
# ---------------------------------------------------------------------------

ANCHOR = pd.Timestamp("2026-09-05 00:00:00")
_ANCHOR_GWL = -12.0
_N = 120
_RNG = np.random.default_rng(42)
_h = np.arange(1, _N + 1, dtype=float)
_ts = ANCHOR + pd.to_timedelta(_h * 6, unit="h")
_q50 = _ANCHOR_GWL + 0.02 * _h + _RNG.normal(0, 0.15, len(_h))
_q05 = _q50 - (0.20 + 0.04 * _h)
_q95 = _q50 + (0.20 + 0.05 * _h)
_conf = np.where(_h < 40, "HIGH", np.where(_h < 80, "DIRECTIONAL", "LOW"))
_src = np.where(_h <= 66, "Open-Meteo forecast (<=16d)", "climatology")

_PTS_DF = pd.DataFrame({
    "time": _ts,
    "gwl": _q50,
    "q05": _q05,
    "q50": _q50,
    "q95": _q95,
    "confidence_level": _conf,
    "driver_source": _src,
})

_TAIL = pd.DataFrame({
    "date": pd.date_range(ANCHOR - pd.Timedelta(days=15), ANCHOR, freq="1D"),
    "gwl": _ANCHOR_GWL + _RNG.normal(0, 0.05, 16).cumsum() * 0.01,
})


def _fake_traj():
    """trajectory_forecast()-shaped dict with synthetic data."""
    return {
        "station": "SYN",
        "anchor_time": ANCHOR.isoformat(),
        "anchor_gwl": _ANCHOR_GWL,
        "trajectory": _PTS_DF.to_dict("records"),
        "trajectory_30d": {"level": float(_q50[-1]),
                           "change": float(_q50[-1] - _ANCHOR_GWL)},
        "direction": {"label": "declining",
                      "change_q50_30d": float(_q50[-1] - _ANCHOR_GWL),
                      "sign_accuracy_30d": 0.82},
        "overall_confidence": {"level": "DIRECTIONAL",
                               "reason": "synthetic test"},
        "evidence": {"station_integrity": 1.0, "anchor_ood": False,
                     "stability_oscillation": 0.0, "recency_days": 0.0,
                     "recent90_coverage": 0.95},
    }


# ===========================================================================
# 1) Altair chart-spec tests (fast, no Streamlit)
# ===========================================================================

class TestTrajectoryChartSpec(unittest.TestCase):
    """Verify the dark trajectory-chart spec produced by trajectory_chart."""

    @classmethod
    def setUpClass(cls):
        from app_charts import trajectory_chart
        ch = trajectory_chart(
            pts=_PTS_DF.copy(), tail=_TAIL.copy(), anchor_t=ANCHOR,
            height=480, interactive=False,
        )
        cls.spec = ch.to_dict()

    def _datasets(self):
        return [v for v in (self.spec.get("datasets") or {}).values()
                if isinstance(v, list)]

    def _first_120(self):
        return next(v for v in self._datasets() if len(v) == 120)

    # -- data integrity ------------------------------------------------------

    def test_single_120_row_dataset(self):
        n120 = sum(1 for v in self._datasets() if len(v) == 120 and bool(v)
                   and "q50" in v[0])
        self.assertEqual(n120, 1, "exactly one genuine 120-row forecast frame")

    def test_120_points_6h_spacing(self):
        rows = self._first_120()
        ts = [pd.Timestamp(r["time"]) for r in rows]
        self.assertEqual(ts[0], ANCHOR + pd.Timedelta(hours=6))
        self.assertEqual(ts[-1], ANCHOR + pd.Timedelta(days=30))
        diffs = pd.Series(ts).diff().dropna().dt.total_seconds() / 3600.0
        self.assertTrue(np.allclose(diffs, 6.0, atol=1e-6))

    def test_q05_le_q50_le_q95(self):
        for row in self._first_120():
            self.assertLessEqual(row["q05"], row["q50"] + 1e-9)
            self.assertLessEqual(row["q50"], row["q95"] + 1e-9)

    def test_no_bridge_or_interpolated_dataset(self):
        sizes = {len(v) for v in self._datasets()}
        self.assertNotIn(2, sizes, "2-point bridge row must not exist")
        self.assertNotIn(221, sizes, "interpolated 221-row frame must not exist")

    # -- tooltip fields ------------------------------------------------------

    def test_tooltip_fields_on_trajectory_line(self):
        tooltip_fields = set()
        for node in _walk(self.spec):
            tips = (node.get("encoding") or {}).get("tooltip")
            if isinstance(tips, list):
                for t in tips:
                    if isinstance(t, dict) and "field" in t:
                        tooltip_fields.add(t["field"])
        for fld in ("kind", "confidence_level", "driver_source", "q05", "q95"):
            self.assertIn(fld, tooltip_fields)

    def test_time_tooltip_format(self):
        from app_charts import TIME_TIP, AXIS_TIP
        self.assertIn("%d %b %Y", TIME_TIP)
        self.assertIn("%H:%M", TIME_TIP)
        self.assertEqual(AXIS_TIP, "%b %d")

    # -- visible forecast-start boundary (round-3 style) ---------------------

    def test_forecast_start_rule_and_label_present(self):
        rules = 0
        labels = []
        for node in _walk(self.spec):
            mk = node.get("mark")
            if isinstance(mk, dict) and mk.get("type") == "rule":
                rules += 1
            if isinstance(mk, dict) and mk.get("type") == "text":
                val = (node.get("encoding") or {}).get("text")
                if isinstance(val, dict) and "value" in val:
                    labels.append(val["value"])
        self.assertEqual(rules, 1, "one vertical 'Forecast starts' rule")
        self.assertIn("Forecast starts", labels)

    def test_no_diamond_or_other_model_marker(self):
        for node in _walk(self.spec):
            mk = node.get("mark")
            if isinstance(mk, dict) and mk.get("shape") == "diamond":
                self.fail("unexpected diamond marker")
            if isinstance(mk, dict) and mk.get("type") == "text":
                val = (node.get("encoding") or {}).get("text")
                if isinstance(val, dict) and "value" in val:
                    self.assertNotIn("30d", val["value"].lower())
                    self.assertNotIn("Direct", val["value"])

    # -- observed tail -------------------------------------------------------

    def test_observed_tail_dashed_layer(self):
        dashed = [n for n in _walk(self.spec)
                  if isinstance(n.get("mark"), dict)
                  and n["mark"].get("type") == "line"
                  and n["mark"].get("strokeDash") is not None]
        self.assertTrue(dashed, "observed tail is a dashed line layer")

    # -- confidence dots --- --------------------------------------------------

    def test_confidence_color_scale_domain(self):
        for node in _walk(self.spec):
            col = (node.get("encoding") or {}).get("color")
            if isinstance(col, dict):
                sc = col.get("scale") or {}
                if sc.get("domain") == ["HIGH", "DIRECTIONAL", "LOW"]:
                    return
        self.fail("confidence color scale not found")

    def test_no_altair_legend(self):
        """Confidence marks must not render a detached Altair legend."""
        for node in _walk(self.spec):
            col = (node.get("encoding") or {}).get("color")
            if isinstance(col, dict) and col.get("field") in ("confidence_level",):
                self.assertIsNone(col.get("legend"), "confidence legend must be None")

    # -- dark presentation ---------------------------------------------------

    def test_dark_background_in_spec(self):
        cfg = self.spec.get("config") or {}
        self.assertEqual(cfg.get("background"), "#0a0a0a")

    def test_horizontal_only_gridlines(self):
        for node in _walk(self.spec):
            axis = (node.get("encoding") or {}).get("x", {}).get("axis")
            if isinstance(axis, dict):
                self.assertNotEqual(axis.get("grid"), True,
                                    "x axis must not draw vertical gridlines")


# ===========================================================================
# 2) Source-level rules: the Forecast UI carries no other forecast model
# ===========================================================================

class TestNoOtherForecastInForecastUI(unittest.TestCase):
    """Grep the Forecast page's assets — no reference to a second forecast."""

    TOKENS = (r"\bdirect\b", "endpoint", "production")

    def _files(self):
        return [ROOT / "app_pages" / "forecast.py",
                ROOT / "app_charts.py",
                ROOT / "snapshot.py"]

    def test_forbidden_tokens_absent(self):
        for p in self._files():
            src = p.read_text()
            for tok in self.TOKENS:
                self.assertFalse(
                    re.search(tok, src, re.I),
                    f"{p.name} contains forbidden token {tok!r}")

    def test_snapshot_action_present(self):
        src = (ROOT / "app_pages" / "forecast.py").read_text()
        self.assertIn("snapshot_png", src)
        self.assertIn("download_button", src)


# ===========================================================================
# 3) Snapshot PNG tests (fast, no Streamlit)
# ===========================================================================

class TestSnapshotPNG(unittest.TestCase):
    """Verify the dark forecast-card PNG export."""

    def _traj(self, conf="HIGH"):
        pts = _PTS_DF.copy()
        pts["confidence_level"] = conf
        t = _fake_traj()
        t["trajectory"] = pts.to_dict("records")
        return t

    def test_png_magic_bytes(self):
        from snapshot import trajectory_snapshot_png
        png = trajectory_snapshot_png(traj=self._traj(), tail=_TAIL.copy(),
                                      station="SYN")
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(png), 10_000)

    def test_all_low_confidence_renders(self):
        from snapshot import trajectory_snapshot_png
        png = trajectory_snapshot_png(traj=self._traj(conf="LOW"),
                                      tail=_TAIL.copy(), station="SYN")
        self.assertEqual(png[:4], b"\x89PNG")

    def test_empty_tail_renders(self):
        from snapshot import trajectory_snapshot_png
        png = trajectory_snapshot_png(traj=self._traj(), tail=None, station="SYN")
        self.assertEqual(png[:4], b"\x89PNG")

    def test_empty_trajectory_raises(self):
        from snapshot import trajectory_snapshot_png
        t = self._traj()
        t["trajectory"] = []
        with self.assertRaises(ValueError):
            trajectory_snapshot_png(traj=t, station="SYN")


# ===========================================================================
# 4) AppTest headless page checks (slow — gated behind AQUIS_APPTEST=1)
# ===========================================================================

@unittest.skipUnless(os.environ.get("AQUIS_APPTEST"), "set AQUIS_APPTEST=1 to run")
class TestForecastAppTest(unittest.TestCase):
    """Headless Streamlit AppTest check — single session (one at.run for the
    default station, one for a station switch) because the trajectory engine is
    slow; every assertion shares that one AppTest instance."""

    def test_forecast_card_renders_full(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app_pages" / "forecast.py"),
                               default_timeout=420)
        at.run(timeout=420)
        self.assertEqual(len(at.exception), 0, at.exception)

        # single card, no tabs, chart covered by spec tests, Snapshot present
        self.assertEqual(len(at.tabs), 0)
        self.assertEqual(len(at.get("download_button")), 1)

        # exactly the 6 forecast-card metrics
        self.assertEqual(len(at.metric), 6)
        labels = [m.label for m in at.metric]
        for want in ("Anchor GWL (observed)", "+24 h", "+7 d", "+30 d",
                     "30 d change", "Confidence"):
            self.assertIn(want, labels)

        # 120-point detail table columns
        frames = at.get("dataframe")
        self.assertEqual(len(frames), 1)
        cols = set(frames[0].value.columns)
        for c in ("timestamp", "q05 (m)", "q50 (m)", "q95 (m)", "confidence", "driver"):
            self.assertIn(c, cols)
        self.assertEqual(len(frames[0].value), 120)

        # station switch re-renders in the same session
        at.selectbox[0].set_value("BADHANI PRATHMIK VIDYALAYA")
        at.run(timeout=420)
        self.assertEqual(len(at.exception), 0, at.exception)
        self.assertEqual(len(at.metric), 6)


if __name__ == "__main__":
    unittest.main()