"""_thresholds.py - single source of truth for fleet categories and alert zones.

Every place that turns a 30-day model change into a *category* (the fleet scan,
``pipeline_data fleet``) or a category into an *alert zone* (``api.py`` fleet/alerts,
district detail and the per-station zone rule) imports from here, so the two
cannot drift apart.

Categories (``category()``) are computed from the signed 30-day change:

    change <= -0.60 m          -> "decline (high)"
    change <= -0.30 m          -> "decline"
    change >= +0.30 m          -> "recovering"
    otherwise                  -> "stable"
    stale / NaN anchor         -> "unreliable"

Zones (``zone_for_category()``) are the notification-level buckets:

    unreliable                 -> "unknown"   (no trustworthy signal)
    stable / recovering        -> "safe"
    decline                    -> "watch"
    decline (high)             -> "watch (high)"

Danger/alert stay reachable only through the per-station precaution rules
(``api._zone_from_precautions``); the scan-level categories above never imply
them on their own.
"""

from __future__ import annotations

# ---- category thresholds (metres of 30-day change) -----------------------
DECLINE_M = 0.3            # >= 0.3 m drop over 30 days = flagged decline
HIGH_DECLINE_M = 0.6       # >= 0.6 m drop over 30 days = high decline

CATEGORIES = ("unreliable", "unknown", "stable", "recovering",
              "decline", "decline (high)")

# ---- category -> alert zone ---------------------------------------------
CATEGORY_TO_ZONE = {
    "unreliable": "unknown",
    "unknown": "unknown",
    "stable": "safe",
    "recovering": "safe",
    "decline": "watch",
    "decline (high)": "watch (high)",
}

# Severity order, most severe first. "unknown" ranks below the watch tiers: it
# means "no trustworthy signal", not "getting worse".
ZONE_ORDER = ("danger", "alert", "watch (high)", "watch", "unknown", "safe")
ZONE_SEVERITY = {z: i for i, z in enumerate(ZONE_ORDER)}


def category(change: float | None, anchor_valid: bool = True,
             plausible: bool = True) -> str:
    """Fleet category for one station's 30-day change (see module docstring)."""
    if not anchor_valid or not plausible:
        return "unreliable"
    if change is None:
        return "unknown"
    if change <= -HIGH_DECLINE_M:
        return "decline (high)"
    if change <= -DECLINE_M:
        return "decline"
    if change >= DECLINE_M:
        return "recovering"
    return "stable"


def zone_for_category(cat: str | None, change_m: float | None = None,
                      anchor_ok: bool = True) -> tuple[str, str]:
    """Map a fleet-scan category to ``(zone, reason)``.

    ``anchor_ok=False`` (stale or invalid anchor) always degrades to "unknown"
    regardless of the category the scan recorded.
    """
    cat_s = str(cat) if cat is not None else "unknown"
    if not anchor_ok:
        return "unknown", f"anchor invalid / stale (scan: {cat_s})"
    zone = CATEGORY_TO_ZONE.get(cat_s, "unknown")
    chg = "" if change_m is None else f"{change_m:+.2f} m"
    if cat_s == "decline (high)":
        return zone, (f"30-day model change {chg} (drop >= {HIGH_DECLINE_M:.1f} m)"
                      if chg else "high decline over 30 days")
    if cat_s == "decline":
        return zone, (f"30-day model change {chg} (drop >= {DECLINE_M:.1f} m)"
                      if chg else "decline over 30 days")
    if cat_s == "recovering":
        return zone, (f"30-day model change {chg} (recovery >= {DECLINE_M:.1f} m)"
                      if chg else "recovering over 30 days")
    if cat_s == "stable":
        return zone, "no scan-level warnings"
    return zone, f"insufficient reliable telemetry (scan: {cat_s})"


def worst_zone(*zones: str | None) -> str:
    """Most severe zone among ``zones`` (ignores None)."""
    present = [z for z in zones if z]
    if not present:
        return "unknown"
    return min(present, key=lambda z: ZONE_SEVERITY.get(z, len(ZONE_ORDER)))
