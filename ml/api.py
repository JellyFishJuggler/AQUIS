"""AQUIS ML HTTP API — Flask service over the live forecast + assistant stack.

Endpoints (all JSON):

    GET  /health               service + ollama status + dataset recency
    GET  /stations             slug list (recency-sorted; filter by district/q)
    GET  /stations/<slug>      rich per-station facts (same object as chat "facts")
    GET  /stations/<slug>/series  6-hourly GWL + driver series for relation charts
    GET  /stations/<slug>/alerts  notification-ready zone (safe/alert/danger/
                               unknown) + reasons + top drivers, no LLM call
    GET  /fleet/alerts         fleet-wide zones in one call (?zone=&district=&
                               sort=&limit=) for bell-icon / notification center
    GET  /districts            district list with water-scarcity status
                               (?sort=&limit=) — scarcity page source
    GET  /districts/<name>     district detail: levels, most-stressed stations,
                               top driver, model accuracy + full station list
    GET  /fleet/recovery       recovery/decline ranking (?district=&sort=&limit=)
    GET  /fleet/forecasts      full fleet forecast table (?district=&category=&
                               limit=) — map/table source
    GET  /fleet/scan           scan metadata: eligible/scanned, horizon,
                               generated_at + status counts
    GET  /models               trained-model index: pooled + trajectory specs,
                               calibration, trained_at
    GET  /forecast/<slug>?days=N  trajectory trimmed to the first N days
                               (default 30, max 30; 4 points/day)
    GET  /forecast/<slug>      120-point 6-hourly trajectory (q05/q50/q95,
                               confidence_level, direction, evidence)
    POST /assistant/chat       station-locked LLM answer (Ollama llama3.2:3b)
                               (also GET with ?question=&station=&model= for browsers)

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

import json
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

VERSION = "3.4.0"

_MODEL_META = _ROOT / "models" / "model_metadata.json"
_TRAJ_CONFIG = _ROOT / "models" / "traj_config.json"
_TRAJ_CAL = _ROOT / "models" / "traj_calibration.json"
_model_cache: dict | None = None


def _models_info() -> dict:
    """Trained-model index from committed metadata JSONs (cached).

    Never loads weight files — specs, calibration and timestamps only.
    Returns {"pooled": dict|None, "trajectory": dict|None}.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    def _read(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001 - missing artifact, never fatal
            return None

    pooled = _read(_MODEL_META)
    traj = _read(_TRAJ_CONFIG)
    qcal = _read(_ROOT / "models" / "quantile_calibration.json")
    tcal = _read(_TRAJ_CAL)
    out: dict = {"pooled": None, "trajectory": None}
    if pooled:
        out["pooled"] = {
            "id": "pooled-30d-xgb",
            "architecture": pooled.get("architecture"),
            "target": pooled.get("target"),
            "horizons": pooled.get("horizons"),
            "n_stations": pooled.get("n_stations"),
            "n_districts": pooled.get("n_districts"),
            "train_val_rows": pooled.get("train_val_rows"),
            "test_rows": pooled.get("test_rows"),
            "val_rmse_level_m": pooled.get("val_rmse_level_m"),
            "trained_at": pooled.get("trained_at"),
            "coverage_stride": (qcal or {}).get("coverage_stride"),
            "widen_factor_k": (qcal or {}).get("widen_factor_k"),
        }
    if traj:
        horizons = (tcal or {}).get("horizons_h", [])
        cal_cov = (tcal or {}).get("coverage_calibrated", [])
        out["trajectory"] = {
            "id": "trajectory-v2",
            "paradigm": traj.get("paradigm"),
            "target": traj.get("target"),
            "max_h": traj.get("max_h"),
            "days": (traj.get("max_h") or 120) // 4,
            "quantiles": list((traj.get("quantiles") or {}).keys()),
            "n_features": len(traj.get("feature_cols") or []),
            "fit_rows": traj.get("fit_rows"),
            "trained_at": traj.get("trained_at"),
            "coverage_calibrated": cal_cov[-1] if cal_cov else None,
            "horizons_h": horizons,
        }
    _model_cache = out
    return _model_cache

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

# Trailing slashes must never 404 (mobile clients / tunnels often append
# them): /health/ == /health, /stations/<slug>/series/ == .../series.
# Set before any route is registered so every Rule inherits it.
app.url_map.strict_slashes = False

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


_FLEET_CSV = _ROOT / "outputs" / "fleet_forecast.csv"
_FLEET_SNAPSHOT = _ROOT / "outputs" / "fleet_forecast_snapshot.json"
_fleet_cache: dict | None = None


def _fleet() -> dict:
    """Fleet scan artifacts (11_fleet.py): forecast table + generation stamp.

    Cached in memory; the scan is regenerated by the batch pipeline, not per
    request. Returns {"df": DataFrame|None, "generated_at": str|None}.
    """
    global _fleet_cache
    if _fleet_cache is not None:
        return _fleet_cache
    try:
        df = pd.read_csv(_FLEET_CSV)
    except Exception:  # noqa: BLE001 - scan may not exist on a fresh clone
        df = None
    gen = None
    try:
        with open(_FLEET_SNAPSHOT) as f:
            gen = json.load(f).get("generated_at")
    except Exception:  # noqa: BLE001 - stamp is decorative, never fatal
        pass
    _fleet_cache = {"df": df, "generated_at": gen}
    return _fleet_cache


def _fleet_category(station: str) -> str | None:
    """Fleet-scan category for a station ('stable' / 'unreliable' / None)."""
    df = _fleet()["df"]
    if df is None or "station" not in df.columns or "category" not in df.columns:
        return None
    rows = df[df["station"] == station]
    if not len(rows):
        return None
    return str(rows.iloc[0]["category"])


# Zone thresholds mirror the assistant's rule-based precautions
# (_assistant._precautions): any "action" -> danger, any "watch" -> alert.
ZONE_ORDER = ("danger", "alert", "unknown", "safe")

_FLEET_DISTRICT_CSV = _ROOT / "outputs" / "fleet_district.csv"
_EVAL_DISTRICT_CSV = _ROOT / "outputs" / "eval_by_district.csv"
_CORR_DISTRICT_CSV = _ROOT / "outputs" / "correlation_by_district.csv"
_district_cache: dict | None = None


def _districts_data() -> dict:
    """District-level frames (fleet roll-up, model accuracy, driver link).

    Cached in memory; all three are batch-pipeline products, not per-request.
    Returns {"fleet": DataFrame|None, "eval": DataFrame|None,
    "corr": DataFrame|None}.
    """
    global _district_cache
    if _district_cache is not None:
        return _district_cache

    def _read(path):
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001 - missing artifact, never fatal
            return None

    _district_cache = {
        "fleet": _read(_FLEET_DISTRICT_CSV),
        "eval": _read(_EVAL_DISTRICT_CSV),
        "corr": _read(_CORR_DISTRICT_CSV),
    }
    return _district_cache


# District water-scarcity statuses. Thresholds confirmed with the user:
# history-based signals (180d drops) count alongside the 30-day scan, and the
# scan is post-monsoon (declines ~0 fleet-wide) so history carries the scarcity
# call — documented per response in `scarcity_basis`.
SCARCITY_ORDER = ("scarce", "watch", "healthy", "unknown")


def _scarcity_for(n: int, unreliable_frac: float, decline_share: float,
                  median_change: float | None, worst_180d: float | None,
                  n_deep: int = 0) -> tuple[str, str]:
    """(status, basis) for one district. See SCARCITY_ORDER.

    District scarcity needs a BROAD signal (many wells falling / median
    dropping) or a DEEP + corroborated one (>=2 wells down >=2 m each).
    One lone deep well is that well's alert (served by per-station /alerts
    and shown here as most_stressed) — the district stays 'watch' on that
    alone.
    """
    if n < 3 or unreliable_frac > 0.5:
        return "unknown", "too few reliable stations to assess"
    if (decline_share >= 0.25
            or (median_change is not None and median_change <= -0.5)
            or n_deep >= 2):
        return "scarce", "broad or deep+corroborated decline"
    if ((median_change is not None and median_change < 0)
            or decline_share > 0
            or worst_180d is not None):
        return "watch", "mild negative change or stressed wells present"
    return "healthy", "stable or recovering levels, no stressed wells"


def _unreliable_frac(district: str) -> float:
    """Share of a district's scan rows that are not 'stable'."""
    df = _fleet()["df"]
    if df is None or "district" not in df.columns:
        return 0.0
    sub = df[df["district"].astype(str).str.casefold() == district.casefold()]
    if not len(sub):
        return 1.0
    cat = sub["category"].astype(str) if "category" in sub.columns else pd.Series(["stable"] * len(sub))
    return float((cat != "stable").mean())


def _zone_from_precautions(precautions: list[dict],
                           fleet_category: str | None = None) -> str:
    """safe / alert / danger / unknown from rule-based precaution levels.

    A fleet-scan 'unreliable' station (stale/NaN anchor) is 'unknown' — there
    is no trustworthy signal to zone on.
    """
    if fleet_category is not None and fleet_category != "stable":
        return "unknown"
    levels = {str(p.get("level", "")) for p in (precautions or [])}
    if "action" in levels:
        return "danger"
    if "watch" in levels:
        return "alert"
    return "safe"


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
            "alerts": "/stations/<slug>/alerts",
            "fleet_alerts": "/fleet/alerts",
            "districts": "/districts",
            "district_detail": "/districts/<name>",
            "fleet_recovery": "/fleet/recovery",
            "fleet_forecasts": "/fleet/forecasts",
            "fleet_scan": "/fleet/scan",
            "models": "/models",
            "forecast": "/forecast/<slug>",
            "assistant_chat": "/assistant/chat",
        },
        "usage": {
            "health": "GET /health — service, ollama, dataset recency",
            "stations": "GET /stations?district=&q=&limit=",
            "station_detail": "GET /stations/<slug> — per-station facts (no LLM)",
            "series": "GET /stations/<slug>/series?drivers=&from=&to=&limit= — 6h gwl+driver points",
            "alerts": "GET /stations/<slug>/alerts?n= — zone + reasons + top drivers (no LLM)",
            "fleet_alerts": "GET /fleet/alerts?zone=&district=&sort=&limit= — fleet zones",
            "districts": "GET /districts?sort=&limit= — district list with scarcity status",
            "district_detail": "GET /districts/<name> — levels, most-stressed, top driver, stations",
            "fleet_recovery": "GET /fleet/recovery?district=&sort=&limit= — recovery/decline ranking",
            "fleet_forecasts": "GET /fleet/forecasts?district=&category=&limit= — full forecast table",
            "fleet_scan": "GET /fleet/scan — scan metadata + tallies",
            "models": "GET /models — trained-model index (specs, no weights)",
            "forecast": "GET /forecast/<slug> — 120x6h q05/q50/q95 trajectory",
            "assistant_chat": "GET|POST /assistant/chat — ?question=&station=&model= or JSON body",
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


@app.get("/stations/<slug>/alerts")
def station_alerts(slug: str):
    """Notification-ready zone for one station — no LLM call.

    Zone comes from the same rule-based precautions the assistant phrases
    (any "action" -> danger, any "watch" -> alert, else safe; a fleet-scan
    "unreliable" station -> unknown). ``top_drivers`` are the station's own
    drivers ranked by |Spearman| vs GWL. Query param ``n`` (default 3, max 8)
    controls how many drivers are returned.
    """
    hit = _resolve(slug)
    if not hit:
        return jsonify({"error": "station not found",
                        "detail": f"unknown slug: {slug} (see /stations)"}), 404
    slug, info = hit
    try:
        n = int(request.args.get("n") or 3)
    except (ValueError, TypeError):
        return jsonify({"error": "bad request",
                        "detail": "`n` must be an integer"}), 400
    n = max(1, min(n, 8))
    try:
        facts = StationAssistant().facts(info["station"])
    except Exception as e:  # noqa: BLE001 - JSON-safe for the API client
        return jsonify({"error": "alerts failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500
    if not facts:
        return jsonify({"error": "no data",
                        "detail": f"no telemetry for station: {info['station']}"}), 404
    precs = facts.get("precautions") or []
    zone = _zone_from_precautions(precs, _fleet_category(info["station"]))
    drivers = []
    for d in (facts.get("drivers") or [])[:n]:
        drivers.append({
            "driver": d.get("driver"),
            "label": SERIES_LABELS.get(d.get("driver"), str(d.get("driver"))),
            "corr": d.get("corr"),
            "p": d.get("p"),
            "n": d.get("n"),
        })
    fc = facts.get("forecast") or {}
    return jsonify({
        "slug": slug,
        "station": info["station"],
        "district": info.get("district", ""),
        "zone": zone,
        "reasons": [{"level": p.get("level"), "title": p.get("title"),
                     "why": p.get("why")} for p in precs],
        "top_drivers": drivers,
        "level_now": facts.get("last"),
        "change_30d": facts.get("change_30d"),
        "change_180d": facts.get("change_180d"),
        "forecast": {
            "direction": fc.get("direction"),
            "change_30d_pred": fc.get("change_30d_pred"),
            "day30_pred": fc.get("day30_pred"),
            "q05_level": fc.get("q05_level"),
            "q95_level": fc.get("q95_level"),
            "plausible": fc.get("plausible"),
            "high_uncertainty": fc.get("high_uncertainty"),
        },
    })


@app.get("/fleet/alerts")
def fleet_alerts():
    """Fleet-wide zones in one call — bell-icon / notification-center source.

    Zones here come from the latest published fleet scan (11_fleet.py), NOT
    the per-station history rules: 'unreliable' scan rows (stale/NaN anchor)
    -> unknown; 30-day model change <= -1.0 m -> danger; <= -0.5 m -> alert;
    else safe. The per-station /alerts endpoint (full precaution rules) is
    authoritative for detail; this one is the fast fleet overview.

    Query params: ``zone`` (comma list, e.g. danger,alert), ``district``
    (exact, case-insensitive), ``sort`` (severity|change, default severity),
    ``limit`` (default 500, max 2000).
    """
    fl = _fleet()
    df = fl["df"]
    if df is None or not len(df):
        return jsonify({"error": "fleet scan unavailable",
                        "detail": "outputs/fleet_forecast.csv missing — run 11_fleet.py"}), 503
    raw_zones = (request.args.get("zone") or "").strip()
    zones = [z.strip().lower() for z in raw_zones.split(",") if z.strip()] or None
    if zones:
        bad = [z for z in zones if z not in ZONE_ORDER]
        if bad:
            return jsonify({"error": "bad request",
                            "detail": f"unknown zones: {bad}",
                            "available": list(ZONE_ORDER)}), 400
    district = (request.args.get("district") or "").strip().casefold()
    sort = (request.args.get("sort") or "severity").strip().lower()
    if sort not in ("severity", "change"):
        return jsonify({"error": "bad request",
                        "detail": "`sort` must be severity|change"}), 400
    try:
        limit = int(request.args.get("limit") or 500)
    except (ValueError, TypeError):
        return jsonify({"error": "bad request",
                        "detail": "`limit` must be an integer"}), 400
    limit = max(1, min(limit, 2000))

    idx = _index()
    station_to_slug = {inf["station"]: s for s, inf in idx.items()}
    items = []
    for _, r in df.iterrows():
        st = str(r.get("station", ""))
        chg = r.get("change_30d_m")
        chg = None if chg is None or (isinstance(chg, float) and np.isnan(chg)) else float(chg)
        cat = str(r.get("category", "stable"))
        anchor_ok = bool(r.get("anchor_valid", True)) if not pd.isna(r.get("anchor_valid", True)) else False
        if cat != "stable" or not anchor_ok:
            zone, reason = "unknown", f"insufficient reliable telemetry (scan: {cat})"
        elif chg is not None and chg <= -1.0:
            zone, reason = "danger", f"30-day model change {chg:+.2f} m (drop >= 1.0 m)"
        elif chg is not None and chg <= -0.5:
            zone, reason = "alert", f"30-day model change {chg:+.2f} m (drop >= 0.5 m)"
        else:
            zone, reason = "safe", "no scan-level warnings"
        if zones and zone not in zones:
            continue
        if district and str(r.get("district", "")).casefold() != district:
            continue
        band = r.get("band_half_m")
        band = None if band is None or (isinstance(band, float) and np.isnan(band)) else float(band)
        items.append({
            "slug": station_to_slug.get(st),
            "station": st,
            "district": str(r.get("district", "")),
            "zone": zone,
            "reason": reason,
            "change_30d_m": chg,
            "band_half_m": band,
        })
    rank = {z: i for i, z in enumerate(ZONE_ORDER)}
    if sort == "change":
        items.sort(key=lambda s: (s["change_30d_m"] is None,
                                  s["change_30d_m"] if s["change_30d_m"] is not None else 0.0))
    else:
        items.sort(key=lambda s: (rank.get(s["zone"], 9), s["change_30d_m"]
                                  if s["change_30d_m"] is not None else 0.0))
    counts = {z: sum(1 for s in items if s["zone"] == z) for z in ZONE_ORDER}
    return jsonify({
        "generated_at": fl["generated_at"],
        "count": len(items),
        "counts": counts,
        "alerts": items[:limit],
    })


@app.get("/districts")
def list_districts():
    """District list with water-scarcity status — the scarcity-page source.

    Scarcity blends the 30-day fleet scan with history-based signals (deep
    180d falls), because the scan itself is post-monsoon (declines ~0
    fleet-wide). Query params: ``sort`` (scarcity|name|decline, default
    scarcity), ``limit`` (default 100, max 200).
    """
    dd = _districts_data()
    fdf = dd["fleet"]
    if fdf is None or not len(fdf):
        return jsonify({"error": "district scan unavailable",
                        "detail": "outputs/fleet_district.csv missing — run 11_fleet.py"}), 503
    sort = (request.args.get("sort") or "scarcity").strip().lower()
    if sort not in ("scarcity", "name", "decline"):
        return jsonify({"error": "bad request",
                        "detail": "`sort` must be scarcity|name|decline"}), 400
    try:
        limit = int(request.args.get("limit") or 100)
    except (ValueError, TypeError):
        return jsonify({"error": "bad request",
                        "detail": "`limit` must be an integer"}), 400
    limit = max(1, min(limit, 200))

    from _assistant import _district_context
    table_districts = {str(d).casefold(): str(d)
                       for d in _data()["District"].astype(str).unique()}
    items = []
    for _, r in fdf.iterrows():
        raw = str(r.get("district", ""))
        canon = table_districts.get(raw.casefold(), raw)
        try:
            ctx = _district_context(canon)
        except Exception:  # noqa: BLE001 - one bad district never fails the list
            ctx = {}
        stressed = ctx.get("most_stressed") or []
        worst = min((s.get("change_180d") for s in stressed
                     if s.get("change_180d") is not None), default=None)
        n = int(r.get("n", 0) or 0)
        dshare = r.get("decline_share")
        dshare = None if dshare is None or (isinstance(dshare, float) and np.isnan(dshare)) else float(dshare)
        med = r.get("median_change_m")
        med = None if med is None or (isinstance(med, float) and np.isnan(med)) else float(med)
        n_deep = int(ctx.get("n_deep_180d", 0) or 0)
        status, basis = _scarcity_for(n, _unreliable_frac(raw),
                                      dshare if dshare is not None else 0.0,
                                      med, worst, n_deep)
        items.append({
            "district": canon,
            "n_stations": n,
            "median_change_m": med,
            "n_decline": int(r.get("n_decline", 0) or 0),
            "n_recover": int(r.get("n_recover", 0) or 0),
            "decline_share": dshare,
            "scarcity": status,
            "scarcity_basis": basis,
            "worst_180d_drop": worst,
            "n_stressed": len(stressed),
            "n_deep_180d": int(ctx.get("n_deep_180d", 0) or 0),
        })
    rank = {s: i for i, s in enumerate(SCARCITY_ORDER)}
    if sort == "name":
        items.sort(key=lambda d: d["district"])
    elif sort == "decline":
        items.sort(key=lambda d: -(d["decline_share"] or 0.0))
    else:
        items.sort(key=lambda d: (rank.get(d["scarcity"], 9),
                                   d["worst_180d_drop"]
                                   if d["worst_180d_drop"] is not None else 0.0))
    counts = {s: sum(1 for d in items if d["scarcity"] == s) for s in SCARCITY_ORDER}
    return jsonify({"count": len(items), "counts": counts,
                    "districts": items[:limit]})


@app.get("/districts/<name>")
def district_detail(name: str):
    """One district in full: levels, most-stressed stations, top driver,
    model accuracy, and the complete station list with alert zones.

    The station list reuses the fast scan signals (same rules as
    /fleet/alerts); per-station /alerts stays authoritative for detail.
    """
    dd = _districts_data()
    table_districts = {str(d).casefold(): str(d)
                       for d in _data()["District"].astype(str).unique()}
    canon = table_districts.get((name or "").strip().casefold())
    if not canon:
        return jsonify({"error": "district not found",
                        "detail": f"unknown district: {name}",
                        "available": sorted(table_districts.values())}), 404
    from _assistant import _district_context
    try:
        ctx = _district_context(canon)
    except Exception as e:  # noqa: BLE001 - JSON-safe for the API client
        return jsonify({"error": "district failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500
    if not ctx:
        return jsonify({"error": "no data",
                        "detail": f"no telemetry for district: {canon}"}), 404

    idx = _index()
    station_to_slug = {inf["station"]: s for s, inf in idx.items()}
    stressed = []
    for s in (ctx.get("most_stressed") or []):
        stressed.append({"slug": station_to_slug.get(s["station"]),
                         "station": s["station"], "level": s.get("level"),
                         "change_180d": s.get("change_180d")})

    top_driver = None
    cdf = dd["corr"]
    if cdf is not None and "District" in cdf.columns:
        sub = cdf[cdf["District"].astype(str).str.casefold() == canon.casefold()]
        if len(sub) and "spearman" in sub.columns:
            best = sub.iloc[sub["spearman"].abs().argsort()[::-1]].iloc[0]
            top_driver = {"driver": str(best.get("driver")),
                          "spearman": float(best["spearman"]),
                          "pearson": (None if pd.isna(best.get("pearson"))
                                      else float(best["pearson"])),
                          "n": int(best["n"]) if "n" in best and pd.notna(best["n"]) else None}

    accuracy = None
    edf = dd["eval"]
    if edf is not None and "district" in edf.columns:
        sub = edf[(edf["district"].astype(str).str.casefold() == canon.casefold())
                  & (edf.get("horizon", 30) == 30)
                  & (edf.get("model", "xgboost") == "xgboost")]
        if len(sub):
            r0 = sub.iloc[0]
            accuracy = {"rmse": float(r0["rmse"]), "mae": float(r0["mae"]),
                        "within_05": float(r0["within_05"]) if "within_05" in r0 and pd.notna(r0["within_05"]) else None,
                        "rows": int(r0["rows"]) if "rows" in r0 else None}

    stations = []
    fdf = _fleet()["df"]
    if fdf is not None and "district" in fdf.columns:
        for _, r in fdf[fdf["district"].astype(str).str.casefold() == canon.casefold()].iterrows():
            st = str(r.get("station", ""))
            chg = r.get("change_30d_m")
            chg = None if chg is None or (isinstance(chg, float) and np.isnan(chg)) else float(chg)
            cat = str(r.get("category", "stable"))
            ok = bool(r.get("anchor_valid", True)) if not pd.isna(r.get("anchor_valid", True)) else False
            if cat != "stable" or not ok:
                zone, reason = "unknown", f"insufficient reliable telemetry (scan: {cat})"
            elif chg is not None and chg <= -1.0:
                zone, reason = "danger", f"30-day model change {chg:+.2f} m (drop >= 1.0 m)"
            elif chg is not None and chg <= -0.5:
                zone, reason = "alert", f"30-day model change {chg:+.2f} m (drop >= 0.5 m)"
            else:
                zone, reason = "safe", "no scan-level warnings"
            band = r.get("band_half_m")
            band = None if band is None or (isinstance(band, float) and np.isnan(band)) else float(band)
            stations.append({"slug": station_to_slug.get(st), "station": st,
                             "zone": zone, "reason": reason,
                             "change_30d_m": chg, "band_half_m": band})
    rank = {z: i for i, z in enumerate(ZONE_ORDER)}
    stations.sort(key=lambda s: (rank.get(s["zone"], 9),
                                 s["change_30d_m"] if s["change_30d_m"] is not None else 0.0))
    worst = min((s.get("change_180d") for s in (ctx.get("most_stressed") or [])
                 if s.get("change_180d") is not None), default=None)
    frow = None
    if dd["fleet"] is not None and "district" in dd["fleet"].columns:
        m = dd["fleet"][dd["fleet"]["district"].astype(str).str.casefold() == canon.casefold()]
        frow = m.iloc[0] if len(m) else None
    if frow is not None:
        dshare = frow.get("decline_share")
        dshare = None if dshare is None or (isinstance(dshare, float) and np.isnan(dshare)) else float(dshare)
        med = frow.get("median_change_m")
        med = None if med is None or (isinstance(med, float) and np.isnan(med)) else float(med)
        nn = int(frow.get("n", 0) or 0)
        n_deep = int(ctx.get("n_deep_180d", 0) or 0)
        status, basis = _scarcity_for(nn,
                                      _unreliable_frac(canon),
                                      dshare if dshare is not None else 0.0,
                                      med, worst, n_deep)
    else:
        nn = ctx.get("n_stations", 0) or 0
        n_deep = int(ctx.get("n_deep_180d", 0) or 0)
        status, basis = _scarcity_for(nn, 0.0, 0.0, None, worst, n_deep)
    return jsonify({
        "district": canon,
        "scarcity": status,
        "scarcity_basis": basis,
        "levels": {"median": ctx.get("median"), "mean": ctx.get("mean"),
                   "min": ctx.get("min_level"), "max": ctx.get("max_level"),
                   "n_stations": ctx.get("n_stations"),
                   "n_analysed": ctx.get("n_analysed")},
        "most_stressed": stressed,
        "top_driver": top_driver,
        "model_accuracy": accuracy,
        "stations": stations,
    })


@app.get("/fleet/recovery")
def fleet_recovery():
    """Recovery/decline ranking across the fleet — who is bouncing back,
    who is still falling (30-day model change).

    Query params: ``district`` (exact, case-insensitive), ``sort``
    (recovery|decline, default recovery = biggest rise first), ``limit``
    (default 100, max 2000). Direction uses the ±0.15 m trajectory threshold.
    """
    fl = _fleet()
    df = fl["df"]
    if df is None or not len(df):
        return jsonify({"error": "fleet scan unavailable",
                        "detail": "outputs/fleet_forecast.csv missing — run 11_fleet.py"}), 503
    district = (request.args.get("district") or "").strip().casefold()
    sort = (request.args.get("sort") or "recovery").strip().lower()
    if sort not in ("recovery", "decline"):
        return jsonify({"error": "bad request",
                        "detail": "`sort` must be recovery|decline"}), 400
    try:
        limit = int(request.args.get("limit") or 100)
    except (ValueError, TypeError):
        return jsonify({"error": "bad request",
                        "detail": "`limit` must be an integer"}), 400
    limit = max(1, min(limit, 2000))
    idx = _index()
    station_to_slug = {inf["station"]: s for s, inf in idx.items()}
    items = []
    for _, r in df.iterrows():
        if district and str(r.get("district", "")).casefold() != district:
            continue
        st = str(r.get("station", ""))
        chg = r.get("change_30d_m")
        chg = None if chg is None or (isinstance(chg, float) and np.isnan(chg)) else float(chg)
        direction = ("rising" if chg is not None and chg >= 0.15
                     else "declining" if chg is not None and chg <= -0.15
                     else "stable" if chg is not None else None)
        band = r.get("band_half_m")
        band = None if band is None or (isinstance(band, float) and np.isnan(band)) else float(band)
        items.append({"slug": station_to_slug.get(st), "station": st,
                      "district": str(r.get("district", "")),
                      "change_30d_m": chg, "direction": direction,
                      "band_half_m": band,
                      "category": str(r.get("category", "stable"))})
    items.sort(key=lambda s: (s["change_30d_m"] is None,
                              -(s["change_30d_m"] or 0.0) if sort == "recovery"
                              else (s["change_30d_m"] or 0.0)))
    return jsonify({"generated_at": fl["generated_at"], "sort": sort,
                    "count": len(items), "items": items[:limit]})


@app.get("/fleet/forecasts")
def fleet_forecasts():
    """Full fleet forecast table — map/table source in one call.

    Every scanned station with anchor, q50/q05/q95 levels, 30d change, band
    and scan category. Query params: ``district`` (exact, case-insensitive),
    ``category`` (stable|unreliable), ``limit`` (default 500, max 2000).
    """
    fl = _fleet()
    df = fl["df"]
    if df is None or not len(df):
        return jsonify({"error": "fleet scan unavailable",
                        "detail": "outputs/fleet_forecast.csv missing — run 11_fleet.py"}), 503
    district = (request.args.get("district") or "").strip().casefold()
    category = (request.args.get("category") or "").strip().lower() or None
    if category and category not in ("stable", "unreliable"):
        return jsonify({"error": "bad request",
                        "detail": "`category` must be stable|unreliable"}), 400
    try:
        limit = int(request.args.get("limit") or 500)
    except (ValueError, TypeError):
        return jsonify({"error": "bad request",
                        "detail": "`limit` must be an integer"}), 400
    limit = max(1, min(limit, 2000))
    idx = _index()
    station_to_slug = {inf["station"]: s for s, inf in idx.items()}

    def _f(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else float(v)

    items = []
    for _, r in df.iterrows():
        if district and str(r.get("district", "")).casefold() != district:
            continue
        cat = str(r.get("category", "stable"))
        if category and cat != category:
            continue
        st = str(r.get("station", ""))
        ts = r.get("date_from")
        items.append({
            "slug": station_to_slug.get(st),
            "station": st,
            "district": str(r.get("district", "")),
            "anchor": _f(r.get("anchor")),
            "anchor_time": str(ts) if ts is not None and not pd.isna(ts) else None,
            "anchor_age_days": _f(r.get("age_days")),
            "anchor_valid": bool(r.get("anchor_valid", True)) if not pd.isna(r.get("anchor_valid", True)) else False,
            "q50_level": _f(r.get("q50_level")),
            "q05_level": _f(r.get("q05_level")),
            "q95_level": _f(r.get("q95_level")),
            "change_30d_m": _f(r.get("change_30d_m")),
            "band_half_m": _f(r.get("band_half_m")),
            "plausible": bool(r.get("plausible", True)) if not pd.isna(r.get("plausible", True)) else False,
            "category": cat,
        })
    return jsonify({"generated_at": fl["generated_at"], "count": len(items),
                    "forecasts": items[:limit]})


@app.get("/fleet/scan")
def fleet_scan():
    """Scan metadata — when the fleet was scanned, horizon, eligible vs
    scanned counts, and live stable/unreliable tallies."""
    fl = _fleet()
    meta: dict = {"generated_at": fl["generated_at"], "horizon_days": 30,
                  "decline_threshold_m": 0.3, "eligible_stations": None,
                  "scanned_stations": None}
    try:
        with open(_FLEET_SNAPSHOT) as f:
            snap = json.load(f)
        for k in ("horizon_days", "decline_threshold_m", "eligible_stations",
                  "scanned_stations", "generated_at"):
            if snap.get(k) is not None:
                meta[k] = snap[k]
    except Exception:  # noqa: BLE001 - stamp file optional
        pass
    df = fl["df"]
    if df is not None and len(df):
        meta["scanned_stations"] = len(df)
        if "category" in df.columns:
            meta["n_stable"] = int((df["category"].astype(str) == "stable").sum())
            meta["n_unreliable"] = int((df["category"].astype(str) != "stable").sum())
    return jsonify(meta)


@app.get("/models")
def list_models():
    """Trained-model index — specs, calibration and timestamps (no weights)."""
    info = _models_info()
    if not info["pooled"] and not info["trajectory"]:
        return jsonify({"error": "model metadata unavailable",
                        "detail": "models/*.json metadata missing"}), 503
    return jsonify({"models": [m for m in (info["pooled"], info["trajectory"]) if m],
                    "serving": {"forecast": "trajectory-v2",
                                "benchmark_30d": "pooled-30d-xgb"}})


@app.get("/forecast/<slug>")
def forecast(slug: str):
    """:param days: first N days of the 120-point trajectory (default 30,
    clamped 1..30 — 4 six-hourly points per day)."""
    hit = _resolve(slug)
    if not hit:
        return jsonify({"error": "station not found",
                        "detail": f"unknown slug: {slug} (see /stations)"}), 404
    slug, info = hit
    try:
        days = int(request.args.get("days") or 30)
    except (ValueError, TypeError):
        return jsonify({"error": "bad request",
                        "detail": "`days` must be an integer"}), 400
    days = max(1, min(days, 30))
    try:
        out = trajectory_forecast(info["station"])
    except Exception as e:  # noqa: BLE001 - JSON-safe for the API client
        return jsonify({"error": "forecast failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500
    if not isinstance(out, dict) or "error" in out:
        return jsonify({"error": "forecast unavailable",
                        "detail": out.get("error", "unknown"),
                        "slug": slug, "station": info["station"]}), 422
    traj = out.get("trajectory") or []
    out["trajectory"] = traj[:days * 4]
    out["days"] = days
    out["points"] = len(out["trajectory"])
    out["slug"] = slug
    return jsonify(out)


@app.route("/assistant/chat", methods=["GET", "POST"])
def assistant_chat():
    """Station-locked LLM answer.

    POST (preferred): JSON ``{question, station?, model?}``.
    GET (browser-friendly): ``?question=...&station=...&model=...``.
    """
    if request.method == "GET":
        args = request.args
        question = (args.get("question") or args.get("message") or "").strip()
        station_raw = (args.get("station") or "").strip()
        model = args.get("model")
    else:
        body = request.get_json(silent=True) or {}
        question = (body.get("question") or body.get("message") or "").strip()
        station_raw = (body.get("station") or "").strip()
        model = body.get("model")
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
        result = StationAssistant(model=model).answer(question, station)
        result["station"] = station
        return jsonify(result)
    except Exception as e:  # noqa: BLE001 - surface a clean 503 for chat failures
        return jsonify({"error": "assistant failed",
                        "detail": f"{type(e).__name__}: {e}",
                        "hint": "is Ollama running? (ollama serve)"}), 503


def _install_underscore_aliases(flask_app):
    """Accept underscore variants of multi-segment routes.

    Mobile/frontend clients sometimes send ``/assistant_chat`` instead of
    ``/assistant/chat`` (or ``/stations_<slug>_series`` etc.). Slugs produced
    by :func:`slugify` never contain underscores, so when the raw path matches
    no route but its underscore→slash form does, rewrite it. This must run as
    WSGI middleware — ``before_request`` is too late (routing already done).
    """
    from werkzeug import exceptions as _wz_exc
    from werkzeug.routing import RequestRedirect as _Redirect

    real_wsgi = flask_app.wsgi_app

    def _routes(path: str, method: str) -> bool:
        try:
            flask_app.url_map.bind("").match(path, method=method)
            return True
        except (_wz_exc.MethodNotAllowed, _Redirect):
            return True  # path is valid — only the method/redirect differs
        except Exception:  # noqa: BLE001 - NotFound etc: genuinely no route
            return False

    def _wsgi_with_aliases(environ, start_response):
        p = environ.get("PATH_INFO", "")
        method = environ.get("REQUEST_METHOD", "GET")
        if "_" in p and not _routes(p, method):
            # candidate 1: every "_" is really a "/" ("/assistant_chat")
            # candidate 2: trailing "_..." suffix ("health_check" -> "/health")
            cands = [p.replace("_", "/"), p.split("_")[0] or "/"]
            for q in cands:
                if q != p and _routes(q, method):
                    environ["PATH_INFO"] = q
                    break
        return real_wsgi(environ, start_response)

    flask_app.wsgi_app = _wsgi_with_aliases


_install_underscore_aliases(app)


@app.errorhandler(HTTPException)
def _http_error(e: HTTPException):
    """Unknown routes / bad methods -> clean JSON instead of HTML."""
    import difflib as _difflib
    available = ["/health", "/stations", "/stations/<slug>",
                 "/stations/<slug>/series", "/stations/<slug>/alerts",
                 "/fleet/alerts", "/districts", "/districts/<name>",
                 "/fleet/recovery",
                 "/forecast/<slug>", "/assistant/chat"]
    guess = _difflib.get_close_matches(
        request.path.rstrip("/") or "/", available, n=1, cutoff=0.5)
    # also try matching the first segment ("/assistant_chat" -> "/assistant/chat")
    if not guess:
        first = "/" + (request.path.strip("/").split("/")[0] if request.path.strip("/") else "")
        guess = [a for a in available
                 if a.split("/")[1:2] == [first.strip("/").split("_")[0]]][:1]
    body = {
        "error": "http error",
        "status": e.code,
        "detail": e.description if isinstance(e.description, str) else str(e),
        "path": request.path,
        "available": available,
    }
    if guess and e.code == 404:
        body["did_you_mean"] = guess[0]
    return jsonify(body), e.code or 500


@app.errorhandler(Exception)
def _handle_error(e):
    return jsonify({"error": "internal error", "detail": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"AQUIS ML API → http://localhost:{port} (version {VERSION})")
    app.run(host="0.0.0.0", port=port, threaded=True)