"""AQUIS ML HTTP API — Flask service over the live forecast + assistant stack.

Endpoints (all JSON):

    GET  /health               service + ollama status + dataset recency
    GET  /stations             slug list (recency-sorted; filter by district/q)
    GET  /forecast/<slug>      120-point 6-hourly trajectory (q05/q50/q95,
                               confidence_level, direction, evidence)
    POST /assistant/chat       station-locked LLM answer (Ollama llama3.2:3b)

Run:  python ml/api.py            (default port 5000, override with PORT)

The forecast endpoint is wired to ``_trajectory.trajectory_forecast`` — the
genuine 120-step 6-hourly trajectory, NOT the scalar ``_model.forward_forecast``
outlook. Mobile app clients expecting ``trajectory[].{time,gwl,q05,q50,q95,
confidence_level,...}`` will get that exact shape.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _assistant import StationAssistant, get_df, ollama_status  # noqa: E402
from _trajectory import trajectory_forecast  # noqa: E402

VERSION = "3.0.0"

app = Flask(__name__)
CORS(app)

_df: pd.DataFrame | None = None
_slug_info: dict[str, dict] | None = None


def _data() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = get_df()
    return _df


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).casefold()).strip("-")
    return s or "station"


def _index() -> dict[str, dict]:
    """slug -> {slug, station, district, last_ts}; recency already deterministic.

    Collisions (same hyphenated name) get a numeric suffix.
    """
    global _slug_info
    if _slug_info is not None:
        return _slug_info
    df = _data()
    recency: dict[str, pd.Timestamp] = {}
    for k, v in df.groupby("Station")["time"].max().items():
        recency[str(k)] = pd.Timestamp(v)
    idx: dict[str, dict] = {}
    used: dict[str, int] = {}
    for st_raw, g in df.groupby("Station"):
        st = str(st_raw)
        base = slugify(st)
        n = used.get(base, 0)
        slug = base if n == 0 else f"{base}-{n}"
        used[base] = n + 1
        last = recency.get(st)
        idx[slug] = {
            "slug": slug,
            "station": st,
            "district": str(g["District"].iloc[0]) if "District" in g else "",
            "last_ts": str(last) if last is not None else "",
        }
    _slug_info = idx
    return _slug_info


def _resolve(arg: str) -> tuple[str, dict] | None:
    arg = (arg or "").strip()
    idx = _index()
    info = idx.get(arg)
    if info:
        return arg, info
    for slug, inf in idx.items():
        if inf["station"] == arg:
            return slug, inf
    return None


def _most_recent_station() -> str | None:
    items = list(_index().values())
    items.sort(key=lambda s: s["last_ts"], reverse=True)
    return items[0]["station"] if items else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """Root index — API name, version and the available routes."""
    return jsonify({
        "service": "aquis-ml",
        "version": VERSION,
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "stations": "/stations",
            "forecast": "/forecast/<slug>",
            "assistant_chat": "/assistant/chat",
        },
        "usage": {
            "health": "GET /health — service, ollama, dataset recency",
            "stations": "GET /stations?district=&q=&limit=",
            "forecast": "GET /forecast/<slug> — 120x6h q05/q50/q95 trajectory",
            "assistant_chat": "POST /assistant/chat — {question, station?, model?}",
        },
    })


@app.get("/health")
def health():
    try:
        st = ollama_status()
        ollama = {"server": st["server"], "model": st["model"],
                  "models": sorted(st["models"]), "error": st.get("error")}
    except Exception:  # noqa: BLE001 - probe must never fail /health
        ollama = {"server": False, "error": "probe failed"}
    df = _data()
    last = pd.Timestamp(df["time"].max()) if "time" in df else None
    return jsonify({
        "status": "ok",
        "service": "aquis-ml",
        "version": VERSION,
        "stations": int(df["Station"].nunique()),
        "dataset_last": last.isoformat() if last is not None else None,
        "ollama": ollama,
        "forecast_model": "trajectory (120x6h q05/q50/q95)",
    })


@app.get("/stations")
def list_stations():
    items = list(_index().values())
    district = (request.args.get("district") or "").strip().casefold()
    q = (request.args.get("q") or "").strip().casefold()
    if district:
        items = [s for s in items if s["district"].casefold() == district]
    if q:
        items = [s for s in items
                 if q in s["station"].casefold() or q in s["district"].casefold()]
    items.sort(key=lambda s: (bool(s["last_ts"]), s["last_ts"]), reverse=True)
    limit = min(int(request.args.get("limit") or 200), 2000)
    return jsonify({"count": len(items), "stations": items[:limit]})


@app.get("/forecast/<slug>")
def forecast(slug: str):
    hit = _resolve(slug)
    if not hit:
        return jsonify({"error": "station not found",
                        "detail": f"unknown slug: {slug} (see /stations)"}), 404
    slug, info = hit
    try:
        out = trajectory_forecast(info["station"])
    except Exception as e:  # noqa: BLE001 - JSON-safe for the API client
        return jsonify({"error": "forecast failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500
    if not isinstance(out, dict) or "error" in out:
        return jsonify({"error": "forecast unavailable",
                        "detail": out.get("error", "unknown"),
                        "slug": slug, "station": info["station"]}), 422
    out["slug"] = slug
    return jsonify(out)


@app.post("/assistant/chat")
def assistant_chat():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or body.get("message") or "").strip()
    station_raw = (body.get("station") or "").strip()
    if not question:
        return jsonify({"error": "bad request",
                        "detail": "`question` is required"}), 400
    try:
        station = None
        if station_raw:
            hit = _resolve(station_raw)
            if hit:
                station = hit[1]["station"]
            else:
                return jsonify({"error": "station not found",
                                "detail": f"unknown station: {station_raw}"}), 404
        else:
            station = _most_recent_station()
            if not station:
                return jsonify({"error": "no data",
                                "detail": "no stations in the dataset"}), 404
        result = StationAssistant(model=body.get("model")).answer(question, station)
        result["station"] = station
        return jsonify(result)
    except Exception as e:  # noqa: BLE001 - surface a clean 503 for chat failures
        return jsonify({"error": "assistant failed",
                        "detail": f"{type(e).__name__}: {e}",
                        "hint": "is Ollama running? (ollama serve)"}), 503


@app.errorhandler(HTTPException)
def _http_error(e: HTTPException):
    """Unknown routes / bad methods -> clean JSON instead of HTML."""
    return jsonify({
        "error": "http error",
        "status": e.code,
        "detail": e.description if isinstance(e.description, str) else str(e),
        "path": request.path,
        "available": ["/health", "/stations", "/forecast/<slug>", "/assistant/chat"],
    }), e.code or 500


@app.errorhandler(Exception)
def _handle_error(e):
    return jsonify({"error": "internal error", "detail": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"AQUIS ML API → http://localhost:{port} (version {VERSION})")
    app.run(host="0.0.0.0", port=port, threaded=True)