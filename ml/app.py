"""AQUIS ML HTTP API — headless REST service over the trained ML stack.

Serves the reusable engines (observations, trends, fleet facts, and the
station-locked Data Assistant) as JSON so the Node.js backend and the Next.js
front-end can integrate without the Streamlit dashboard.

Endpoints (all JSON):

    GET  /health                 service + model/ollama status
    GET  /stations               recency-sorted station list
    GET  /stations/<slug>        station facts + latest trend
    GET  /districts              recency-sorted district list
    GET  /forecast/<slug>?days=  30-day (default) XGBoost forecast series
    GET  /fleet/forecasts        precomputed fleet forecast snapshot
    GET  /fleet/recovery         district recovery (median level change) rank
    GET  /fleet/scan             forecast-decline scan of the snapshot
    GET  /models                 trained artifact slugs
    POST /assistant/chat         station-locked LLM answer (Ollama llama3.2:3b)

Run:  python ml/app.py            (default port 5000, override with PORT)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# The ML module only ships its own requirements inside the repo venv; running
# this file directly needs the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.agent.data_assistant import (  # noqa: E402
    ARTIFACTS_DIR,
    BACKEND_CSV,
    GWL_COL,
    PARQUET_PATH,
    TIME_COL,
    STATION_COL,
    DISTRICT_COL,
    AGENCY_COL,
    SLNO_COL,
    DataAssistant,
    _clean_series,
    _district_recovery_rank,
    _fleet_forecast_scan,
    _get_df,
)
from ml.models.xgboost_quantile import (  # noqa: E402
    DIRECT_HORIZONS,
    load_models,
    predict_direct,
    predict_recursive,
    station_dirs,
)
from ml.preprocessing.timeseries import (  # noqa: E402
    full_pipeline,
    prepare_feature_matrix,
    station_slug,
)
from ml.services.interval_calibration import (  # noqa: E402
    estimate_calibration,
    widen,
)

app = Flask(__name__)
CORS(app)

VERSION = "2.0.0"
DEFAULT_CHAT_MODEL = "llama3.2:3b"

# ---------------------------------------------------------------------------
# Cached resources (computed once per process)
# ---------------------------------------------------------------------------
_pipe_cache: dict[str, dict] = {}
_cal_cache: dict[str, object] = {}
_station_index: dict[str, dict] | None = None
_slug_to_station: dict[str, dict] = {}
_train_count: int | None = len(station_dirs())
_chat_models: dict[str, DataAssistant] = {}


def _stations() -> dict[str, dict]:
    """index: slug -> {display, district, agency, state, last_ts, last_epoch, has_model}."""
    global _station_index, _slug_to_station, _train_count
    if _station_index is not None:
        return _station_index
    df = _get_df()
    trained = {d.name for d in station_dirs()}
    _train_count = len(trained)

    # Newest-last-read first; stations with no readable timestamps sink to the
    # bottom with an alphabetical tie-break (same rule as the dashboard picker).
    def _key(item: tuple[str, dict]) -> tuple:
        eps = item[1]["last_epoch"]
        return (-(eps if eps is not None else -1e18), item[1]["display"])

    index: dict[str, dict] = {}
    for (st_name, ag), g in df.groupby([STATION_COL, AGENCY_COL]):
        st_name = str(st_name)
        ag = str(ag)
        row = g.iloc[0]
        slug = station_slug(st_name, ag, getattr(row, SLNO_COL, 0))
        ts = g[TIME_COL].max()
        try:
            epoch = float(pd.Timestamp(ts).timestamp()) if pd.notna(ts) else None
            last = str(pd.Timestamp(ts)) if epoch is not None else ""
        except Exception:  # noqa: BLE001 - optional recency field
            epoch, last = None, ""
        index[slug] = {
            "slug": slug,
            "display": st_name,
            "district": str(row[DISTRICT_COL]) if pd.notna(getattr(row, DISTRICT_COL, None)) else "",
            "agency": ag,
            "state": str(row["State"]) if pd.notna(getattr(row, "State", None)) else "",
            "has_model": slug in trained,
            "last_ts": last,
            "last_epoch": epoch,
        }
    _slug_to_station = dict(index)
    _station_index = dict(sorted(index.items(), key=_key))
    return _station_index


def _station_by_slug(slug: str) -> dict | None:
    return _slug_to_station.get(slug) or _stations().get(slug)


def _pipeline(slug: str) -> dict | None:
    if slug not in _pipe_cache:
        _pipe_cache[slug] = full_pipeline(PARQUET_PATH, BACKEND_CSV, station_slug_filter=slug)
    return _pipe_cache.get(slug)


def _calibration(slug: str, models: dict, train_df: pd.DataFrame, feature_cols: list[str]):
    if slug not in _cal_cache:
        _cal_cache[slug] = estimate_calibration({}, models, train_df, feature_cols)
    return _cal_cache[slug]


def _assistant(model: str | None = None) -> DataAssistant:
    key = model or DEFAULT_CHAT_MODEL
    if key not in _chat_models:
        _chat_models[key] = DataAssistant(model=key)
    return _chat_models[key]


def _ollama_up() -> bool:
    try:
        import requests

        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:  # noqa: BLE001 - status probe must never fail the endpoint
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    snapshot = _REPO_ROOT / "ml" / "agent" / "fleet_forecast_snapshot.json"
    snap_age = None
    if snapshot.exists():
        snap_age = time.time() - snapshot.stat().st_mtime
    return jsonify({
        "status": "ok",
        "service": "aquis-ml",
        "version": VERSION,
        "trained_stations": _train_count,
        "snapshot_age_seconds": snap_age,
        "ollama": _ollama_up(),
        "dataset": str(_get_df()[TIME_COL].max().date()) if _get_df().get(TIME_COL) is not None else None,
    })


@app.get("/stations")
def list_stations():
    sts = list(_stations().values())
    district = (request.args.get("district") or "").strip()
    q = (request.args.get("q") or "").strip().lower()
    if district:
        sts = [s for s in sts if s["district"] == district]
    if q:
        sts = [s for s in sts if q in s["display"].lower() or q in s["district"].lower()]
    limit = min(int(request.args.get("limit") or 200), 2000)
    sts = sts[:limit]
    return jsonify({"count": len(sts), "stations": sts})


@app.get("/stations/<slug>")
def station_detail(slug: str):
    info = _station_by_slug(slug)
    if not info:
        return jsonify({"error": "station not found", "detail": f"unknown slug: {slug}"}), 404
    facts = _assistant().fast_facts(station=info["display"])
    if not facts:
        return jsonify({"error": "no data", "detail": f"no telemetry for {slug}"}), 404
    facts["slug"] = slug
    facts["has_model"] = info["has_model"]
    return jsonify({"data": facts})


@app.get("/districts")
def list_districts():
    df = _get_df()
    rows = []
    for d, g in df.groupby(DISTRICT_COL):
        last = pd.Timestamp(g[TIME_COL].max())
        try:
            epoch = float(last.timestamp())
        except Exception:  # noqa: BLE001
            epoch = None
        rows.append({
            "district": str(d),
            "n_stations": int(g[STATION_COL].nunique()),
            "last_ts": str(last),
            "last_epoch": epoch,
        })
    rows.sort(key=lambda r: (not bool(r["last_epoch"]), -float(r.get("last_epoch") or 0)))
    return jsonify({"count": len(rows), "districts": rows})


@app.get("/forecast/<slug>")
def forecast(slug: str):
    info = _station_by_slug(slug)
    if not info:
        return jsonify({"error": "station not found", "detail": f"unknown slug: {slug}"}), 404

    art = ARTIFACTS_DIR / slug
    if not (art / "recursive" / "xgb_point.joblib").exists():
        return jsonify({
            "error": "no model",
            "detail": f"no trained forecast model for {slug} (see /models)",
            "trained": False,
        }), 404

    try:
        days = min(max(int(request.args.get("days", 30)), 7), 90)
        pipe = _pipeline(slug)
        if not pipe:
            return jsonify({"error": "pipeline failed", "detail": f"could not build features for {slug}"}), 502
        fc = pipe["feature_cols"]
        train_df = pipe["train"]
        full_df = pipe.get("full", train_df)
        models = load_models(art)
        calib = _calibration(slug, models, train_df, fc)

        tail = full_df.dropna(subset=[TIME_COL, GWL_COL]).tail(1)
        X, _, _ = prepare_feature_matrix(tail[fc + [GWL_COL]])
        if len(X) == 0:
            return jsonify({"error": "no valid features", "detail": f"station {slug} is not forecastable"}), 502
        last_feats = X[0]

        hours = np.arange(24, days * 24 + 1, 24, dtype=float)
        direct: dict[str, list[float]] = {"point": [], "lower": [], "upper": []}
        i = 0
        while i < len(hours):
            hd = int(hours[i] / 24)
            if hd <= 30 and hd in DIRECT_HORIZONS:
                pred = predict_direct(models, last_feats.reshape(1, -1), hd)
                direct["point"].append(pred["point"])
                direct["lower"].append(pred["q05"])
                direct["upper"].append(pred["q95"])
                i += 1
            else:
                break

        rec = {"point": [], "lower": [], "upper": [], "q50": []}
        if i < len(hours):
            r = predict_recursive(models, last_feats, len(hours) - i, fc, models.get("error_correction"))
            rec["point"].extend(r["point"])
            rec["lower"].extend(r["q05"])
            rec["q50"].extend(r["q50"])
            rec["upper"].extend(r["q95"])

        pt = np.array(direct["point"] + rec["point"])
        lo = np.array(direct["lower"] + rec["lower"])
        up = np.array(direct["upper"] + rec["upper"])
        lo, up = widen(calib, hours, pt, lo, up, anchor_pos=0)
        q50 = direct["point"] + rec["q50"] if rec["q50"] else pt

        anchor = pd.Timestamp(tail[TIME_COL].iloc[0]).normalize()
        dates = [(anchor + pd.Timedelta(days=int(h / 24))).isoformat() for h in hours]

        vals, _, _ = _clean_series(full_df[GWL_COL], full_df[TIME_COL])
        obs_min = float(vals.min()) if vals.count() else None
        obs_max = float(vals.max()) if vals.count() else None
        chg = float(pt[-1] - pt[0]) if len(pt) > 1 else None
        plausible = (chg is not None and abs(chg) <= 12.0 and
                     (obs_min is None or obs_max is None or (obs_min - 25.0 <= float(pt[-1]) <= obs_max + 25.0)))

        return jsonify({
            "trained": True,
            "slug": slug,
            "station": info["display"],
            "district": info["district"],
            "as_of": str(anchor.date()),
            "last_obs": {"date": str(anchor.date()),
                          "value": float(full_df.dropna(subset=[GWL_COL])[GWL_COL].iloc[-1])},
            "horizon_days": days,
            "dates": dates,
            "point": [float(x) for x in pt],
            "q50": [float(x) for x in q50],
            "lower": [float(x) for x in lo],
            "upper": [float(x) for x in up],
            "change_7d_pred": float(pt[6] - pt[0]) if len(pt) > 6 else None,
            "change_30d_pred": float(pt[29] - pt[0]) if len(pt) > 29 else None,
            "change_60d_pred": float(pt[59] - pt[0]) if len(pt) > 59 else None,
            "plausible": bool(plausible),
            "obs_min": obs_min,
            "obs_max": obs_max,
        })
    except Exception as e:  # noqa: BLE001 - JSON-safe for the API client
        return jsonify({"error": "forecast failed", "detail": str(e)}), 500


@app.get("/fleet/forecasts")
def fleet_forecasts():
    snapshot = _REPO_ROOT / "ml" / "agent" / "fleet_forecast_snapshot.json"
    if not snapshot.exists():
        return jsonify({"error": "snapshot missing", "detail": "run ml/agent/build_fleet_forecast.py", "available": False}), 404
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    return jsonify(data)


@app.get("/fleet/recovery")
def fleet_recovery():
    try:
        w = int(request.args.get("window_days", 60))
        top = int(request.args.get("top", 5))
        min_st = int(request.args.get("min_stations", 3))
        w = max(7, min(w, 365))
        top = max(1, min(top, 20))
        return jsonify(_district_recovery_rank(window_days=w, top=top, min_stations=min_st))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "recovery rank failed", "detail": str(e)}), 500


@app.get("/fleet/scan")
def fleet_scan():
    try:
        return jsonify(_fleet_forecast_scan(
            district=(request.args.get("district") or None),
            threshold=float(request.args.get("threshold", 0.3)),
            horizon=int(request.args.get("horizon", 60)),
        ))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "forecast scan failed", "detail": str(e)}), 500


@app.get("/models")
def list_models():
    trained = []
    for d in station_dirs():
        meta = d / "xgboost_metadata.json"
        ts = None
        if meta.exists():
            try:
                ts = json.loads(meta.read_text(encoding="utf-8")).get("trained_at")
            except Exception:  # noqa: BLE001
                ts = None
        trained.append({"slug": d.name, "trained_at": ts})
    return jsonify({"count": len(trained), "models": trained})


@app.post("/assistant/chat")
def assistant_chat():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "bad request", "detail": "`question` is required"}), 400
    try:
        result = _assistant(body.get("model")).answer(
            question,
            station=body.get("station") or None,
            station_slug=body.get("station_slug") or None,
        )
        return jsonify(result)
    except Exception as e:  # noqa: BLE001 - surface a clean 503 for chat failures
        return jsonify({"error": "assistant failed", "detail": f"{type(e).__name__}: {e}",
                        "hint": "is Ollama running? (ollama serve)"}), 503


@app.errorhandler(Exception)
def _handle_error(e):
    return jsonify({"error": "internal error", "detail": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"AQUIS ML API → http://localhost:{port} (version {VERSION})")
    app.run(host="0.0.0.0", port=port, threaded=True)