"""Category + zone threshold tests (_thresholds.py).

These pin the single source of truth shared by pipeline_data fleet (scan categories) and
api.py (alert zones) so a threshold edit cannot silently change one side.
"""

import unittest

from tests._helpers import has


@unittest.skipUnless(has("_thresholds.py"), "_thresholds module missing")
class TestCategory(unittest.TestCase):
    def test_boundaries(self):
        from _thresholds import category
        self.assertEqual(category(-0.6), "decline (high)")
        self.assertEqual(category(-1.5), "decline (high)")
        self.assertEqual(category(-0.599), "decline")
        self.assertEqual(category(-0.3), "decline")
        self.assertEqual(category(-0.299), "stable")
        self.assertEqual(category(0.0), "stable")
        self.assertEqual(category(0.3), "recovering")

    def test_missing_and_invalid(self):
        from _thresholds import category
        self.assertEqual(category(None), "unknown")
        self.assertEqual(category(-0.9, anchor_valid=False), "unreliable")
        self.assertEqual(category(-0.9, plausible=False), "unreliable")


@unittest.skipUnless(has("_thresholds.py"), "_thresholds module missing")
class TestZoneMapping(unittest.TestCase):
    def test_category_to_zone(self):
        from _thresholds import zone_for_category
        cases = {
            "unreliable": "unknown",
            "unknown": "unknown",
            "stable": "safe",
            "recovering": "safe",
            "decline": "watch",
            "decline (high)": "watch (high)",
        }
        for cat, zone in cases.items():
            got, reason = zone_for_category(cat, -0.7)
            self.assertEqual(got, zone, f"{cat} -> {got}")
            self.assertTrue(reason)

    def test_stale_anchor_forces_unknown(self):
        from _thresholds import zone_for_category
        zone, reason = zone_for_category("decline (high)", -0.9, anchor_ok=False)
        self.assertEqual(zone, "unknown")
        self.assertIn("anchor", reason)

    def test_unmapped_category_is_unknown(self):
        from _thresholds import zone_for_category
        self.assertEqual(zone_for_category("banana")[0], "unknown")
        self.assertEqual(zone_for_category(None)[0], "unknown")

    def test_zone_order_and_worst(self):
        from _thresholds import ZONE_ORDER, worst_zone
        self.assertEqual(ZONE_ORDER[0], "danger")
        self.assertIn("watch", ZONE_ORDER)
        self.assertIn("watch (high)", ZONE_ORDER)
        self.assertEqual(worst_zone("safe", "watch"), "watch")
        self.assertEqual(worst_zone("safe", "watch (high)", "watch"), "watch (high)")
        self.assertEqual(worst_zone("alert", "watch (high)"), "alert")
        self.assertEqual(worst_zone("unknown", "safe"), "unknown")
        self.assertEqual(worst_zone(None), "unknown")


if __name__ == "__main__":
    unittest.main()
