import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "aquis-ml", "version": "1.0.0"})


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
