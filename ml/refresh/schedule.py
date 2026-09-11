"""Wall-clock grid scheduling helpers for the refresh daemon.

The daemon used to re-arm ``next_refresh_due`` as "now + refresh_interval_hours",
a rolling interval that drifts with cycle completion time. The operator instead
wants fixed local-time slots (e.g. fetch/forecast at 01:00 / 07:00 / 13:00 /
19:00 IST), so a cycle lands at the same wall-clock times every day regardless
of how long the previous cycle ran.

``next_grid_due`` computes the next slot strictly after ``now`` in the machine's
local timezone and returns an aware UTC datetime (the scheduler stores UTC ISO
strings in state.json). It lives in its own module because both ``scheduler.py``
and ``pipeline.py`` need it and importing each other would be circular.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from refresh.config import RefreshConfig


def next_grid_due(
    cfg: RefreshConfig,
    now: datetime,
    hours: Iterable[int] | None = None,
) -> datetime:
    """Next fixed wall-clock slot (local timezone) strictly after ``now``.

    ``hours`` defaults to ``cfg.refresh_at_hours``. Slots are evaluated in the
    process local timezone (the machine runs IST), then normalized to UTC so the
    daemon can compare timezone-aware instants safely.
    """
    slots = sorted({int(h) % 24 for h in (hours if hours is not None else cfg.refresh_at_hours)})
    if not slots:
        # No grid configured: fall back to the rolling interval.
        return now + timedelta(hours=cfg.refresh_interval_hours)

    local = now.astimezone()  # process local tz (IST on this box)
    for hour in slots:
        cand = local.replace(hour=hour, minute=0, second=0, microsecond=0)
        if cand > local:
            return cand.astimezone(timezone.utc)
    # All slots today already passed -> first slot tomorrow.
    cand = (local + timedelta(days=1)).replace(
        hour=slots[0], minute=0, second=0, microsecond=0)
    return cand.astimezone(timezone.utc)