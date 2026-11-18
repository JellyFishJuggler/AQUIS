"""Read-only loader for the research GNN influence artifacts.

This module deliberately imports **no torch and no torch_geometric** — it only
reads JSON files produced offline by ``python -m research.gnn.influence`` and
caches them in memory with an mtime check. Every public function returns a dict
and never raises, so a missing/corrupt artifact can only ever produce
``{"available": false}`` and can never affect the forecast routes.

The feature is off unless ``data/refresh/runtime.json`` contains
``"influence_enabled": true`` (absent key => off). The production runtime file is
never written here.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # ml/
OUT = ROOT / "outputs" / "gnn_influence"
RUNTIME = ROOT / "data" / "refresh" / "runtime.json"

FILES = {
    "station": OUT / "station_influence.json",
    "district": OUT / "district_influence.json",
    "graph": OUT / "graph.json",
    "meta": OUT / "influence_meta.json",
    "consistency": OUT / "influence_consistency.json",
}

CACHE: dict[str, tuple[int, object]] = {}
_RUNTIME_CACHE: tuple[int, bool] | None = None

CAVEAT_FALLBACK = ("Experimental. Shows model sensitivity between stations, not "
                   "proven causation. Pumping and local geology are not observed.")


# --------------------------------------------------------------------------- #
# flag + file cache
# --------------------------------------------------------------------------- #
def clear_cache() -> None:
    """Drop every cached file (tests / after regenerating artifacts)."""
    global _RUNTIME_CACHE
    CACHE.clear()
    _RUNTIME_CACHE = None


def enabled() -> bool:
    """``runtime.json: influence_enabled`` — default False when absent/broken."""
    global _RUNTIME_CACHE
    try:
        m = RUNTIME.stat().st_mtime_ns
    except OSError:
        return False
    if _RUNTIME_CACHE is not None and _RUNTIME_CACHE[0] == m:
        return _RUNTIME_CACHE[1]
    try:
        raw = json.loads(RUNTIME.read_text(encoding="utf-8"))
        val = bool(raw.get("influence_enabled", False)) if isinstance(raw, dict) else False
    except (OSError, ValueError):
        val = False
    _RUNTIME_CACHE = (m, val)
    return val


def _load(key: str):
    """Parsed JSON for ``key`` with an mtime cache; ``None`` when unusable."""
    path = FILES[key]
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    hit = CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, (dict, list)):
        return None
    CACHE[key] = (mtime, data)
    return data


def _off(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _caveat() -> str:
    meta = _load("meta")
    if isinstance(meta, dict):
        txt = meta.get("caveat_text") or meta.get("caveat")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    return CAVEAT_FALLBACK


def _guard():
    """Return ``(reason, None)`` when the feature must stay hidden."""
    if not enabled():
        return "influence_disabled"
    return None


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
def meta() -> dict:
    reason = _guard()
    if reason:
        return _off(reason)
    data = _load("meta")
    if not isinstance(data, dict):
        return _off("meta_missing")
    return {"available": True, "experimental": True, "meta": data,
            "caveat": _caveat()}


def graph() -> dict:
    reason = _guard()
    if reason:
        return _off(reason)
    data = _load("graph")
    if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
        return _off("graph_missing")
    return {"available": True, "experimental": True,
            "meta": data.get("meta", {}), "nodes": data["nodes"],
            "edges": data["edges"], "caveat": _caveat()}


def station(name: str, top: int = 5) -> dict:
    reason = _guard()
    if reason:
        return _off(reason)
    data = _load("station")
    if not isinstance(data, dict):
        return _off("station_missing")
    tables = data.get("stations")
    if not isinstance(tables, dict):
        return _off("station_missing")
    name = str(name)
    horizons: dict[str, dict] = {}
    district = ""
    for h, table in tables.items():
        if not isinstance(table, dict):
            continue
        rec = table.get(name)
        if not isinstance(rec, dict):
            continue
        district = district or str(rec.get("district", ""))
        horizons[str(h)] = {
            "influenced_by": list(rec.get("influenced_by", []))[:max(1, top)],
            "influences": list(rec.get("influences", []))[:max(1, top)],
            "strength_in": rec.get("strength_in"),
            "strength_out": rec.get("strength_out"),
        }
    if not horizons:
        return _off("station_not_found")
    return {"available": True, "experimental": True, "station": name,
            "district": district, "horizons": horizons, "caveat": _caveat()}


def district(name: str, top: int = 10) -> dict:
    reason = _guard()
    if reason:
        return _off(reason)
    data = _load("district")
    if not isinstance(data, dict):
        return _off("district_missing")
    wanted = str(name).casefold().strip()
    horizons: dict[str, dict] = {}
    canonical = ""
    for h, block in data.items():
        if not isinstance(block, dict) or "districts" not in block:
            continue
        names = [str(d) for d in block.get("districts", [])]
        lookup = {d.casefold().strip(): i for i, d in enumerate(names)}
        if wanted not in lookup:
            continue
        j = lookup[wanted]
        canonical = canonical or names[j]
        mat = block.get("matrix") or []
        incoming, outgoing = [], []
        for i, dn in enumerate(names):
            if i == j:
                continue
            try:
                v_in = mat[i][j]
                v_out = mat[j][i]
            except (IndexError, TypeError):
                continue
            if v_in is not None:
                incoming.append({"district": dn, "sensitivity": float(v_in)})
            if v_out is not None:
                outgoing.append({"district": dn, "sensitivity": float(v_out)})
        incoming.sort(key=lambda r: abs(r["sensitivity"]), reverse=True)
        outgoing.sort(key=lambda r: abs(r["sensitivity"]), reverse=True)
        horizons[str(h)] = {
            "influenced_by": incoming[:max(1, top)],
            "influences": outgoing[:max(1, top)],
            "top_pairs": list(block.get("top_pairs", []))[:max(1, top)],
        }
    if not horizons:
        return _off("district_not_found")
    return {"available": True, "experimental": True, "district": canonical,
            "horizons": horizons, "caveat": _caveat()}


def consistency() -> dict:
    """Derived-feature consistency report (raw vs consistent perturbation).

    Returns only the compact summary + audits: the per-snapshot matrices are not
    stored, so this stays small enough for a UI panel.
    """
    reason = _guard()
    if reason:
        return _off(reason)
    data = _load("consistency")
    if not isinstance(data, dict):
        return _off("consistency_missing")
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return _off("consistency_missing")
    return {"available": True, "experimental": True,
            "meta": data.get("meta", {}),
            "summary": summary,
            "feature_audit": data.get("feature_audit", {}),
            "roll_reproducibility": data.get("roll_reproducibility", {}),
            "example_deltas": data.get("example_deltas", {}),
            "lag_audit": data.get("lag_audit", {}),
            "caveat": _caveat()}
