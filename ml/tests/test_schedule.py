"""Tests for fixed wall-clock grid scheduling (refresh/schedule.py)."""

import unittest
from datetime import datetime, timezone, timedelta

from refresh.config import RefreshConfig
from refresh.schedule import next_grid_due


class TestGridDue(unittest.TestCase):
    def setUp(self):
        self.cfg = RefreshConfig(refresh_at_hours=[1, 7, 13, 19])

    def utc(self, s):
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

    def test_slot_later_same_day(self):
        # 14:25 IST (08:55 UTC) -> next slot 19:00 IST = 13:30 UTC
        now = self.utc("2026-09-11T08:55:00")
        self.assertEqual(next_grid_due(self.cfg, now),
                         self.utc("2026-09-11T13:30:00"))

    def test_just_after_slot_wraps_to_tomorrow(self):
        # 19:00:01 IST (13:30:01 UTC) -> next slot 01:00 IST next day = 19:30 UTC
        now = self.utc("2026-09-11T13:30:01")
        self.assertEqual(next_grid_due(self.cfg, now),
                         self.utc("2026-09-11T19:30:00"))

    def test_late_night_wraps_to_first_slot(self):
        # 01:01 IST (19:31 UTC prev day) -> next 07:00 IST = 01:30 UTC
        now = self.utc("2026-09-11T19:31:00")
        self.assertEqual(next_grid_due(self.cfg, now),
                         self.utc("2026-09-12T01:30:00"))

    def test_midnight_boundary(self):
        # 19:00:00 IST exactly -> 01:00 IST next day (strictly after)
        now = self.utc("2026-09-11T13:30:00")
        self.assertEqual(next_grid_due(self.cfg, now),
                         self.utc("2026-09-11T19:30:00"))

    def test_empty_grid_falls_back_to_interval(self):
        cfg = RefreshConfig(refresh_at_hours=[], refresh_interval_hours=6.0)
        now = self.utc("2026-09-11T08:55:00")
        self.assertEqual(next_grid_due(cfg, now),
                         now + timedelta(hours=6.0))

    def test_out_of_range_hours_are_normalized(self):
        cfg = RefreshConfig(refresh_at_hours=[25, -1, 7])
        # 25 -> 1, -1 -> 23, so slots {1, 7, 23}; 23:00 IST = 17:30 UTC
        now = self.utc("2026-09-11T08:55:00")  # 14:25 IST
        self.assertEqual(next_grid_due(cfg, now),
                         self.utc("2026-09-11T17:30:00"))


if __name__ == "__main__":
    unittest.main()