"""_up_boundary — a coarse outline of Uttar Pradesh, for map framing only.

**This is a hand-built approximation, not a survey boundary.** It has roughly
35 vertices across a ~750 km state, so expect tens of kilometres of error along
the borders, and the Chambal/Yamuna ravines and the Nepal foothill line in
particular are smoothed. It is good enough to say "this is Uttar Pradesh and
nothing else is in frame" and to anchor a 3D view; it is **not** good enough for
area measurement, jurisdictional reporting, or anything a hydrologist would quote.

To replace it with a real boundary, drop a GeoJSON into
`ml/data/meta/up_boundary.geojson` (a Feature or FeatureCollection with a Polygon
or MultiPolygon). `load_outline()` prefers that file and falls back to the
coordinate list below, so nothing here needs editing.

Used by `app_pages/map_view.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

META = Path(__file__).resolve().parent / "data" / "meta"
GEOJSON_PATH = META / "up_boundary.geojson"

# (longitude, latitude) — closed ring, counter-clockwise from the NW corner.
UP_OUTLINE_COORDS: list[list[float]] = [
    [77.55, 30.42], [78.10, 30.60], [78.75, 30.72], [79.30, 30.90], [79.95, 30.95],
    [80.55, 30.55], [81.05, 30.28], [81.55, 30.15], [82.10, 29.85], [82.70, 29.55],
    [83.20, 29.25], [83.70, 29.05], [84.10, 28.60], [84.35, 28.10], [84.05, 27.60],
    [83.55, 27.30], [83.20, 27.00], [83.60, 26.60], [83.95, 26.30], [84.00, 25.75],
    [83.60, 25.40], [83.20, 25.15], [82.85, 24.85], [82.45, 24.45], [82.10, 24.20],
    [81.60, 24.10], [81.15, 24.35], [80.65, 24.60], [80.15, 24.80], [79.70, 24.95],
    [79.25, 25.15], [78.85, 25.35], [78.45, 25.60], [78.05, 25.95], [77.75, 26.35],
    [77.45, 26.75], [77.20, 27.15], [77.05, 27.60], [77.15, 28.05], [77.35, 28.50],
    [77.45, 28.95], [77.40, 29.40], [77.45, 29.90], [77.55, 30.42],
]

# Centroid-ish anchor used to centre 2D and 3D views.
UP_CENTER = (80.95, 26.85)          # (longitude, latitude)
# [[min_lon, min_lat], [max_lon, max_lat]] — a little padding included.
UP_BBOX = [[76.8, 23.8], [84.9, 31.2]]


def _feature_outline(geom: dict) -> list[list[list[float]]]:
    """Normalise a Polygon / MultiPolygon into a list of rings."""
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Polygon" and coords:
        return [list(map(list, coords[0]))]
    if gtype == "MultiPolygon" and coords:
        return [list(map(list, poly[0])) for poly in coords if poly]
    return []


def load_outline() -> list[list[list[float]]]:
    """Return exterior rings as [[ [lon, lat], ... ], ...].

    Prefers `data/meta/up_boundary.geojson` when present and readable; otherwise
    returns the built-in approximation as a single ring.
    """
    try:
        if GEOJSON_PATH.exists():
            obj = json.loads(GEOJSON_PATH.read_text())
            feats = (obj.get("features") if obj.get("type") == "FeatureCollection"
                     else [obj])
            rings: list[list[list[float]]] = []
            for f in feats or []:
                geom = f.get("geometry") or f
                rings.extend(_feature_outline(geom))
            rings = [r for r in rings if len(r) >= 3]
            if rings:
                return rings
    except Exception:  # noqa: BLE001 — a bad file must not break the map
        pass
    return [UP_OUTLINE_COORDS]


def outline_source() -> str:
    return ("data/meta/up_boundary.geojson" if GEOJSON_PATH.exists()
            else "built-in approximation (≈35 vertices — framing only)")
