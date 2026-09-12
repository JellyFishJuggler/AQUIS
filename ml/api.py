"""AQUIS ML HTTP API — Flask service over the live forecast + assistant stack.

Endpoints (all JSON):

    GET  /health               service + ollama status + dataset recency
    GET  /stations             slug list (recency-sorted; filter by district/q)
    GET  /stations/<slug>      rich per-station facts (same object as chat "facts")
    GET  /stations/<slug>/series  6-hourly GWL + driver series for relation charts
    GET  /forecast/<slug>      120-point 6-hourly trajectory (q05/q50/q95,
                               confidence_level, direction, evidence)
    POST /assistant/chat       station-locked LLM answer (Ollama llama3.2:3b)

Run:  python ml/api.py            (default port 5000, override with PORT)

The forecast endpoint is wired to ``_trajectory.trajectory_forecast`` — the
genuine 120-step 6-hourly trajectory, NOT the scalar ``_model.forward_forecast``
outlook. Mobile app clients expecting ``trajectory[].{time,gwl,q05,q50,q95,
confidence_level,...}`` will get that exact shape.

The series endpoint serves the aligned 6-hourly table (same source as the
Streamlit Correlation page and the old Drivers overlays): every point carries
the observed ``gwl`` plus the requested driver columns, so frontend clients
can render per-feature relation charts against GWL without touching parquet.
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

VERSION = "3.1.0"

# Driver columns servable by /stations/<slug>/series (aligned 6h table).
# Helper columns (rain_n, *_dist_km) and the wind_dir circular column are
# excluded — the former are counts/distances, not physical drivers.
SERIES_DRIVERS = (
    "rain", "temp", "river_level", "humidity",
    "solar", "wind_speed", "pressure", "canal_level",
)
SERIES_LABELS = {
    "rain": "Rain (daily)",
    "temp": "Air temperature",
    "river_level": "River water level",
    "humidity": "Relative humidity",
    "solar": "Solar radiation",
    "wind_speed": "Wind speed",
    "pressure": "Atmospheric pressure",
    "canal_level": "Canal water level",
}
SERIES_LIMIT_DEFAULT = 2000
SERIES_LIMIT_MAX = 6000

app = Flask(__name__)
CORS(app)

_df: pd.DataFrame | None = None
_slug_info: dict[str, dict] | None = None
_coords_map: dict[str, dict] | None = None


def _data() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = get_df()
    return _df


def _coords() -> dict[str, dict]:
    """station name -> {"latitude": float|None, "longitude": float|None}.

    Coordinates live in ``data/meta/selected_gwl_stations.csv`` (the aligned
    6h parquet carries no lat/lon). Loaded once, cached; a station missing
    from the CSV yields None instead of failing the request.
    """
    global _coords_map
    if _coords_map is not None:
        return _coords_map
    out: dict[str, dict] = {}
    try:
        meta = pd.read_csv(_ROOT / "data" / "meta" / "selected_gwl_stations.csv",
                           usecols=["Station", "Latitude", "Longitude"])
        for _, r in meta.iterrows():
            try:
                lat = float(r["Latitude"])
            except (TypeError, ValueError):
                lat = None
            try:
                lon = float(r["Longitude"])
            except (TypeError, ValueError):
                lon = None
            if lat is not None and pd.isna(lat):
                lat = None
            if lon is not None and pd.isna(lon):
                lon = None
            out[str(r["Station"])] = {"latitude": lat, "longitude": lon}
    except Exception:  # noqa: BLE001 - coords are decorative, never fatal
        pass
    _coords_map = out
    return _coords_map


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).casefold()).strip("-")
    return s or "station"


def _index() -> dict[str, dict]:
    """slug -> {slug, station, district, last_ts, latitude, longitude}.

    Recency already deterministic. Collisions (same hyphenated name) get a
    numeric suffix. Coordinates come from ``selected_gwl_stations.csv``.
    """
    global _slug_info
    if _slug_info is not None:
        return _slug_info
    df = _data()
    coords = _coords()
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
        c = coords.get(st, {})
        idx[slug] = {
            "slug": slug,
            "station": st,
            "district": str(g["District"].iloc[0]) if "District" in g else "",
            "last_ts": str(last) if last is not None else "",
            "latitude": c.get("latitude"),
            "longitude": c.get("longitude"),
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
            "station_detail": "/stations/<slug>",
            "series": "/stations/<slug>/series",
            "forecast": "/forecast/<slug>",
            "assistant_chat": "/assistant/chat",
        },
        "usage": {
            "health": "GET /health — service, ollama, dataset recency",
            "stations": "GET /stations?district=&q=&limit=",
            "station_detail": "GET /stations/<slug> — per-station facts (no LLM)",
            "series": "GET /stations/<slug>/series?drivers=&from=&to=&limit= — 6h gwl+driver points",
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


@app.get("/stations/<slug>")
def station_detail(slug: str):
    """Rich per-station facts — the same object ``/assistant/chat`` returns
    under ``"facts"``, without the LLM call (fast, works with Ollama down)."""
    hit = _resolve(slug)
    if not hit:
        return jsonify({"error": "station not found",
                        "detail": f"unknown slug: {slug} (see /stations)"}), 404
    slug, info = hit
    try:
        facts = StationAssistant().facts(info["station"])
    except Exception as e:  # noqa: BLE001 - JSON-safe for the API client
        return jsonify({"error": "facts failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500
    if not facts:
        return jsonify({"error": "no data",
                        "detail": f"no telemetry for station: {info['station']}"}), 404
    c = _coords().get(info["station"], {})
    body = {"slug": slug, "station": info["station"],
            "district": info.get("district", ""),
            "latitude": c.get("latitude"), "longitude": c.get("longitude")}
    body.update(facts)
    return jsonify(body)


@app.get("/stations/<slug>/series")
def station_series(slug: str):
    """6-hourly GWL + driver series for per-feature relation charts.

    Query params: ``drivers`` (comma list, default all of SERIES_DRIVERS),
    ``from``/``to`` (ISO dates, inclusive), ``limit`` (default 2000, max
    6000 — over-limit frames are evenly downsampled, flagged via
    ``downsampled``). Missing values are ``null`` (JSON-safe).
    """
    hit = _resolve(slug)
    if not hit:
        return jsonify({"error": "station not found",
                        "detail": f"unknown slug: {slug} (see /stations)"}), 404
    slug, info = hit
    raw_drivers = (request.args.get("drivers") or "").strip()
    if raw_drivers:
        drivers = [d.strip() for d in raw_drivers.split(",") if d.strip()]
        bad = [d for d in drivers if d not in SERIES_DRIVERS]
        if bad:
            return jsonify({"error": "bad request",
                            "detail": f"unknown drivers: {bad}",
                            "available": list(SERIES_DRIVERS)}), 400
    else:
        drivers = list(SERIES_DRIVERS)
    try:
        from_ts = (pd.Timestamp(request.args.get("from")).tz_localize(None)
                   if request.args.get("from") else None)
        to_ts = (pd.Timestamp(request.args.get("to")).tz_localize(None)
                 if request.args.get("to") else None)
    except (ValueError, TypeError):
        return jsonify({"error": "bad request",
                        "detail": "`from`/`to` must be ISO dates (YYYY-MM-DD)"}), 400
    try:
        limit = int(request.args.get("limit") or SERIES_LIMIT_DEFAULT)
    except (ValueError, TypeError):
        return jsonify({"error": "bad request",
                        "detail": "`limit` must be an integer"}), 400
    if limit < 1:
        return jsonify({"error": "bad request",
                        "detail": "`limit` must be >= 1"}), 400
    limit = min(limit, SERIES_LIMIT_MAX)
    try:
        df = _data()
        sub = df[df["Station"] == info["station"]].copy()
        sub["time"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.dropna(subset=["time"]).sort_values("time")
        if from_ts is not None:
            sub = sub[sub["time"] >= from_ts]
        if to_ts is not None:
            sub = sub[sub["time"] < to_ts + pd.Timedelta(days=1)]
        cols = ["time", "gwl", *[c for c in drivers if c in sub.columns]]
        sub = sub[cols]
    except Exception as e:  # noqa: BLE001 - JSON-safe for the API client
        return jsonify({"error": "series failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500
    downsampled = False
    if len(sub) > limit:
        step = int(len(sub) / limit) + 1
        sub = sub.iloc[::step]
        downsampled = True
    points = []
    for row in sub.itertuples(index=False):
        pt = {"time": row.time.isoformat() if pd.notna(row.time) else None}
        for c in cols[1:]:
            v = getattr(row, c)
            pt[c] = None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else float(v)
        points.append(pt)
    return jsonify({
        "slug": slug,
        "station": info["station"],
        "district": info.get("district", ""),
        "granularity": "6h",
        "drivers": drivers,
        "driver_labels": {d: SERIES_LABELS[d] for d in drivers},
        "from": from_ts.date().isoformat() if from_ts is not None else None,
        "to": to_ts.date().isoformat() if to_ts is not None else None,
        "downsampled": downsampled,
        "count": len(points),
        "points": points,
    })


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
        "available": ["/health", "/stations", "/stations/<slug>",
                      "/stations/<slug>/series",
                      "/forecast/<slug>", "/assistant/chat"],
    }), e.code or 500


@app.errorhandler(Exception)
def _handle_error(e):
    return jsonify({"error": "internal error", "detail": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"AQUIS ML API → http://localhost:{port} (version {VERSION})")
    app.run(host="0.0.0.0", port=port, threaded=True)