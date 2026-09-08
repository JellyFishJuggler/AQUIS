# AQUIS ML Contract

Contract between the **Node.js backend** and the **Python ML service** (`ml/app.py`, Flask, headless — no dashboard/UI).

## Communication

- **Default URL:** `http://localhost:5000`
- **Configurable via:** `ML_SERVICE_URL` environment variable (Node gateway, `back-end/services/mlGateway.js`)
- **Timeout:** `ML_TIMEOUT_MS` (default 60000)
- The gateway forwards both path **and** query string (`pathname + search`).

## Service surface

```
GET  /health                     status, version, trained_stations, snapshot_age_seconds, ollama, dataset
GET  /stations                   ?district=&q=&limit=        recency-sorted station list (slugs)
GET  /stations/:slug             latest telemetry facts for one station
GET  /districts                  recency-sorted districts
GET  /models                     trained artifact slugs
GET  /forecast/:slug             ?days=7..90 (default 30)    XGBoost daily forecast + calibrated 90% interval
GET  /fleet/forecasts            precomputed fleet snapshot (build_fleet_forecast.py)
GET  /fleet/recovery             ?window_days=&top=&min_stations=   district recovery ranking
GET  /fleet/scan                 ?district=&threshold=&horizon=     forecast-decline scan
POST /assistant/chat             {question, station?, station_slug?, model?}   station-locked LLM answer
```

Full frontend-facing documentation with example payloads: **`docs/api.md`**.

## Error contract

Every error is JSON: `{ "error": string, "detail": string }`.

| Code | Meaning |
|---|---|
| 400 | malformed request (chat without `question`) |
| 404 | unknown slug / untrained station / missing snapshot |
| 500 | internal forecasting/preprocessing failure |
| 502 | feature pipeline produced nothing usable |
| 503 | Ollama down (assistant only) |

## Behavior notes for integrators

- **Slugs** are `Station_Agency_shortHash` and contain spaces — URL-encode. Resolve the current slug from `/stations`.
- **First-call latency:** `/stations`, `/districts`, `/stations/:slug` ≈ seconds; `/forecast/:slug` ≈ 7–10 s (per-slug feature pipeline + calibration); `/fleet/recovery` ≈ 8 s per window. All are cached in process after first hit.
- **Forecast output:** `point` = best estimate; `lower`/`upper` = calibration-widened 90% interval; `q50` = raw recursive median; `plausible=false` ⇒ directional-only (level left the observed range). Always render with a caveat when `plausible` is false.
- **Assistant:** requires Ollama (`ollama serve`) with `llama3.2:3b`. Station is pinned — users cannot switch stations mid-conversation. Model may be overridden via the `model` field.
- **No on-demand training:** the API never trains at request time. Run `ml.training.train_all_forecast` (or the dashboard's *Train* button) beforehand; `/models` lists what is available.

## Service not running?

Gateway returns `{ "available": false, "error": "ML service unavailable" }` (or `502` on proxy routes).
Start it with:

```bash
cd <repo>/ml
ml/venv/bin/python ml/app.py
```

## Legacy (deprecated)

The previous DB-driven endpoints — `POST /forecast/:stationId`, `POST /anomalies/:stationId`, `POST /risk/:unitId`, `POST /train`, `POST /models/compare` — are no longer served by the Python service. The Node gateway retains lightweight handlers reading the `model_outputs` table for backward compatibility, but new development must use the slug-based surface above.