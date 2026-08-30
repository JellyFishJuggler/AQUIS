import os
import json
import logging
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")

_bundle_cache = None


def _get_bundle() -> dict:
    """Load the XGBoost model bundle once (lazy; stays cached across calls)."""
    global _bundle_cache
    if _bundle_cache is None:
        from ml.scripts.export_xgboost_models import load_bundle

        _bundle_cache = load_bundle()
        logger.info(
            f"Loaded model bundle: {_bundle_cache['count']} stations, "
            f"schema {_bundle_cache['schema']}, built {_bundle_cache['created_utc']}"
        )
    return _bundle_cache


@app.route("/health", methods=["GET"])
def health():
    response = {"status": "ok", "service": "aquis-ml", "version": "1.0.0"}
    if _bundle_cache is not None:
        response["models"] = {
            "loaded": True,
            "engine": _bundle_cache["engine"],
            "stations": _bundle_cache["count"],
        }
    return jsonify(response)


@app.route("/forecast/xgb/<path:identifier>", methods=["GET"])
def forecast_xgb(identifier):
    """XGBoost forecast served from the model bundle.

    ``identifier`` is a station slug or a substring of the station name.
    Returns a daily projection from the station's last stored reading out
    to ``horizon_days`` (default 90, max 180) as point + 90% interval.
    """
    try:
        from ml.models.xgboost_quantile import DEFAULT_PARQUET
        from ml.preprocessing.timeseries import TIME_COL
        from ml.scripts.export_xgboost_models import (
            _observed_buffer_for,
            _series_for,
            bundle_predict,
        )

        horizon_days = request.args.get("horizon_days", default=90, type=int)
        horizon_days = max(1, min(horizon_days, 180))

        bundle = _get_bundle()
        stations = bundle["stations"]

        query = identifier.strip().lower()
        slug = query if query in stations else None
        if slug is None:
            matches = sorted(
                s for s, e in stations.items()
                if query in s or query in e["name"].lower()
            )
            if len(matches) == 1:
                slug = matches[0]
            elif len(matches) > 1:
                return jsonify({"error": "ambiguous identifier", "matches": matches}), 400
            else:
                return jsonify(
                    {"error": f"unknown station '{identifier}'", "station_count": bundle["count"]}
                ), 404

        entry = stations[slug]
        buffer = _observed_buffer_for(str(DEFAULT_PARQUET), entry["name"])
        series = _series_for(str(DEFAULT_PARQUET), entry["name"])
        last_pos = max(buffer)
        last_observed_value = float(buffer[last_pos])
        anchor = series[TIME_COL].iloc[last_pos].to_pydatetime()

        steps = bundle["sampling_hours"]  # grid step hours (6)
        day_steps = 24 // steps
        times = [float((last_pos + k * day_steps) * steps) for k in range(1, horizon_days + 1)]
        pred = bundle_predict(bundle, slug, times)

        dates = [
            (anchor + timedelta(days=k)).isoformat(timespec="minutes")
            for k in range(1, horizon_days + 1)
        ]
        return jsonify({
            "engine": bundle["engine"],
            "schema": bundle["schema"],
            "station": entry["name"],
            "slug": slug,
            "anchor": anchor.isoformat(timespec="seconds"),
            "last_observed_value": last_observed_value,
            "horizon_days": horizon_days,
            "quantiles": bundle["quantiles"],
            "dates": dates,
            "point": [float(v) for v in pred["point"]],
            "lower": [float(v) for v in pred["lower"]],
            "upper": [float(v) for v in pred["upper"]],
        })
    except Exception as e:
        logger.error(f"XGB forecast error for '{identifier}': {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/forecast/<int:station_id>", methods=["POST"])
def forecast(station_id):
    try:
        import requests as req
        resp = req.get(f"{BACKEND_URL}/ml-data/telemetry/{station_id}", timeout=30)
        if resp.status_code != 200:
            return jsonify({"error": "Failed to fetch telemetry data", "status": resp.status_code}), 502
        data = resp.json()
        observations = data.get("data", [])
        if len(observations) < 10:
            return jsonify({
                "status": "insufficient_data",
                "message": f"Need at least 10 observations, got {len(observations)}",
                "station_id": station_id,
            })
        from services.forecast import run_forecast
        result = run_forecast(observations)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Forecast error for station {station_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/anomalies/<int:station_id>", methods=["POST"])
def anomalies(station_id):
    try:
        import requests as req
        resp = req.get(f"{BACKEND_URL}/ml-data/telemetry/{station_id}", timeout=30)
        if resp.status_code != 200:
            return jsonify({"error": "Failed to fetch telemetry data"}), 502
        data = resp.json()
        observations = data.get("data", [])
        if len(observations) < 20:
            return jsonify({
                "status": "insufficient_data",
                "message": f"Need at least 20 observations, got {len(observations)}",
                "station_id": station_id,
            })
        from services.anomaly import run_anomaly_detection
        result = run_anomaly_detection(observations)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Anomaly detection error for station {station_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/risk/<int:unit_id>", methods=["POST"])
def risk(unit_id):
    try:
        import requests as req
        resp = req.get(f"{BACKEND_URL}/ml-data/assessment/{unit_id}", timeout=30)
        if resp.status_code != 200:
            return jsonify({"error": "Failed to fetch assessment data"}), 502
        data = resp.json()
        records = data.get("data", [])
        if len(records) < 2:
            return jsonify({
                "status": "insufficient_data",
                "message": f"Need at least 2 years of assessment data, got {len(records)}",
                "unit_id": unit_id,
            })
        from services.risk import run_risk_assessment
        result = run_risk_assessment(records)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Risk assessment error for unit {unit_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train():
    try:
        body = request.get_json() or {}
        task = body.get("task")
        if not task:
            return jsonify({"error": "task is required"}), 400

        import requests as req
        if task == "forecast":
            from services.forecast import train_models
            result = req.post(f"{BACKEND_URL}/ml-data/assessment", timeout=60)
        elif task == "anomaly":
            return jsonify({"error": "Anomaly detection uses unsupervised methods, no training needed"}), 400
        elif task == "risk":
            from services.risk import train_risk_model
            result_data = req.get(f"{BACKEND_URL}/ml-data/assessment?limit=5000", timeout=60)
            if result_data.status_code != 200:
                return jsonify({"error": "Failed to fetch assessment data"}), 502
            model_result = train_risk_model(result_data.json().get("data", []))
            return jsonify(model_result)
        else:
            return jsonify({"error": f"Unknown task: {task}"}), 400

        return jsonify({"status": "training_initiated", "task": task})
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/models/compare", methods=["POST"])
def compare_models():
    try:
        body = request.get_json() or {}
        task = body.get("task")
        if not task:
            return jsonify({"error": "task is required"}), 400
        from services.model_comparison import get_comparison
        result = get_comparison(task)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", 5000))
    logger.info(f"Starting AQUIS ML Service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
