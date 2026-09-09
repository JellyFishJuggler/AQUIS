# AQUIS API Reference

Two HTTP services power the platform:

| Service | URL | Status |
|---|---|---|
| **Node.js API** (Express) | `http://localhost:3000` | **Active** — frontend/mobile clients |
| **Python ML service** (Flask, headless) | `http://localhost:5000` | **Planned / deferred** (`merge_later`) — not yet merged |

The **frontend talks to the Node API on `:3000`**. ML intelligence (recency‑sorted station/district lists, per‑station facts, XGBoost forecasts, fleet recovery ranking, the LLM data assistant) is designed to be served by a headless Python service and proxied by the Node backend under `/ml/*`. **That Flask service is not part of the current `ml/` module** — it was deliberately deferred during the merge (see `merge_later` in the merge spec). Until it lands:

- `back-end/services/mlGateway.js` still resolves `ML_SERVICE_URL` (default `http://localhost:5000`).
- `/ml/*` gateway routes return `{ "available": false, "error": "ML service unavailable" }` because nothing listens on `:5000` yet.
- The **live analysis surface today is the Streamlit dashboard** — `cd ml && venv/bin/streamlit run app.py` (port 8501) — plus the artifacts/`JSON` outputs it renders (`ml/outputs/*.csv, ml/models/*.json`).

> Legacy numeric endpoints (`GET /ml/forecast/:stationId`, `/ml/risk/:unitId`, `/ml/anomalies/...`) are **kept for backward compatibility but deprecated** — they read from the DB model registry, not the live ML stack.

The remainder of this page documents the **target contract** for the deferred Flask service so the gateway can drop onto it cleanly, plus the currently-live Node endpoints.

---

## Conventions

- **Encoding:** always JSON (`Content-Type: application/json`). Query parameters for GETs.
- **Auth:** none currently — services run on localhost/LAN. CORS open on both services.
- **Errors:** every failure returns `{ "error": "...", "detail": "..." }` with an appropriate status code.
- **Slugs:** planned ML endpoints identify stations by **slug** — `Station_Agency_shortHash`, e.g. `ASHADHA PRATHMIK VIDYALAYA_UPGW_5f5b3671`. Slugs contain spaces; **URL‑encode them** (`encodeURIComponent` in JS): `ASHADHA%20PRATHMIK%20VIDYALAYA_UPGW_5f5b3671`. Always fetch the current slug from `/ml/stations` — never hand‑type one.

### Status codes (planned ML service)

| Code | Meaning |
|---|---|
| 200 | OK |
| 400 | Missing/invalid request (e.g. no `question` on chat) |
| 404 | Unknown slug / no model trained / snapshot missing |
| 500 | Internal Python error |
| 502 | ML service down or unreachable |
| 503 | Assistant unavailable (Ollama not running) |

---

## Frontend endpoints (Node `:3000`) — live

### 1. `GET /ml/health`
Service status + model/Ollama health. Fast, call on app boot.

```json
{
  "available": false,
  "error": "ML service unavailable"
}
```
`available:true` appears once the deferred Flask service is running on `:5000`; see [Python service contract](#python-service-contract---planned) below.

### 2. Stations / telemetry / assessments / trends / ml-data / data-quality / ingestion

All of these are served by the Express backend against PostgreSQL and are **live today**:

```
GET  /stations                          List all stations (paginated)
GET  /stations/:stationId               Station details
GET  /stations/nearby?lat=&lon=         Nearby stations
GET  /stations/state-summary            Station count by state
GET  /stations/district-summary         Station count by district
GET  /telemetry                         All observations (filtered)
GET  /telemetry/latest?stationId=       Latest observation for station
GET  /telemetry/summary                 State/district summary
GET  /telemetry/:stationId              History for station
GET  /assessments                       Assessment records (filtered)
GET  /assessments/:id                   Single record
GET  /assessments/:unitId/history       Multi-year history for unit
GET  /assessments/summary               State/year summary
GET  /assessments/years                 Available assessment years
GET  /trends/:stationId                 Mann-Kendall + Sen's slope for station
GET  /trends/summary                    Trend summary across all stations
GET  /ml-data/telemetry                 Clean telemetry dataset
GET  /ml-data/telemetry/:stationId      Station-specific telemetry
GET  /ml-data/assessment                Clean assessment dataset
GET  /ml-data/assessment/:unitId        Unit-specific assessment history
GET  /data-quality/:stationId           Station quality issues
GET  /data-quality/telemetry            Telemetry quality summary
GET  /data-quality/assessment           Assessment quality summary
GET  /ingestion                         List ingestion runs
GET  /ingestion/:id                     Run details
POST /ingestion/telemetry               Trigger telemetry ingestion
POST /ingestion/assessment              Trigger assessment ingestion
```

---

## Python service contract — planned

The deferred Flask service will expose the surface below; **it does not exist yet in the repo**. The Node gateway (`back-end/services/mlGateway.js`) already forwards path + query string to `ML_SERVICE_URL`.

### Service surface

```
GET  /health                     status, version, trained_stations, snapshot_age_seconds, ollama, dataset
GET  /stations                   ?district=&q=&limit=        recency-sorted station list (slugs)
GET  /stations/:slug             latest telemetry facts for one station
GET  /districts                  recency-sorted districts
GET  /models                     trained artifact slugs
GET  /forecast/:slug             ?days=7..90 (default 30)    forecast + calibrated 90% interval
GET  /fleet/forecasts            precomputed fleet snapshot
GET  /fleet/recovery             ?window_days=&top=&min_stations=   district recovery ranking
GET  /fleet/scan                 ?district=&threshold=&horizon=     forecast-decline scan
POST /assistant/chat             {question, station?, station_slug?, model?}   station-locked LLM answer
```

How the current `ml/` module maps to this contract:

| Contract endpoint | Data today (in `ml/`) |
|---|---|
| `/health` | `models/model_metadata.json` (trained_at, dataset, metrics), `outputs/residual_acf.json` |
| `/stations`, `/stations/:slug`, `/districts` | `data/meta/selected_gwl_stations.csv`, `station_summary.csv`, `fleet_forecast_snapshot.json` |
| `/models` | `models/feature_config.json`, `quantile_calibration.json` |
| `/forecast/:slug` | `_model.forward_forecast()` (pooled 30-day delta + calibrated band) |
| `/fleet/forecasts` | `outputs/fleet_forecast.csv`, `fleet_forecast_snapshot.json` |
| `/fleet/recovery` | `fleet_district.csv` |
| `/fleet/scan` | `11_fleet.py` significant-movers logic (quality-gated) |
| `/assistant/chat` | `_assistant.py` (station-pinned facts + Ollama) |

---

## Quickstarts

**Backend (live):**
```bash
curl "http://localhost:3000/stations?limit=5"
curl "http://localhost:3000/telemetry/latest?stationId=<id>"
```

**ML dashboard (live today):**
```bash
cd ml && venv/bin/streamlit run app.py        # http://localhost:8501
```

**Typical frontend flow once the ML service lands:**
1. Boot → `GET /ml/health` (EMPTY banner if `available:false`).
2. Station picker → `GET /ml/stations?q=...` (sort by recency; render `last_ts`).
3. Detail view → `GET /ml/stations/:slug` for KPIs.
4. Forecast tab → `GET /ml/live/forecast/:slug?days=30`; chart `dates` vs `point` with `lower`/`upper` band.
5. Fleet view → `GET /ml/live/fleet/recovery?window_days=60` and/or `GET /ml/live/fleet/scan`.
6. Assistant → `POST /ml/assistant/chat`.