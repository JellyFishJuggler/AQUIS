# AQUIS API Reference

Two HTTP services power the platform:

| Service | URL | Status |
|---|---|---|
| **Node.js API** (Express) | `http://localhost:3000` | **Active** — frontend/mobile clients |
| **Python ML service** (Flask, headless) | `http://localhost:5000` | **Live** (`ml/api.py`) — trajectory v2 forecast, fleet, assistant |

The **frontend talks to the Node API on `:3000`**. ML intelligence (recency‑sorted station/district lists, per‑station facts, XGBoost forecasts, fleet recovery ranking, the LLM data assistant) is served by a headless Python Flask service (`ml/api.py`) and proxied by the Node backend under `/ml/*`.

- `back-end/services/mlGateway.js` resolves `ML_SERVICE_URL` (default `http://localhost:5000`).
- When the Flask service is down, `/ml/*` gateway routes return `{ "available": false, "error": "ML service unavailable" }`.
- The **live analysis surfaces** are the headless Flask API (`ml/api.py`, port 5000) and the **Streamlit dashboard** (`cd ml && venv/bin/streamlit run app.py`, port 8501). Both read the same artifacts under `ml/outputs/` and `ml/models/`.

> Legacy numeric endpoints (`GET /ml/forecast/:stationId`, `/ml/risk/:unitId`, `/ml/anomalies/...`) are **kept for backward compatibility but deprecated** — they read from the DB model registry, not the live ML stack.

The remainder of this page documents the **live Flask contract** plus the Node endpoints.

---

## Conventions

- **Encoding:** always JSON (`Content-Type: application/json`). Query parameters for GETs.
- **Auth:** none currently — services run on localhost/LAN. CORS open on both services.
- **Errors:** every failure returns `{ "error": "...", "detail": "..." }` with an appropriate status code.
- **Slugs:** ML endpoints identify stations by **slug** — lowercase hyphenated station name, e.g. `ashadha-prathmik-vidyalaya`. Always fetch the current slug from `GET /stations` — never hand-type one. (Fallback: the exact station name also resolves when URL-encoded, e.g. `ASHADHA%20PRATHMIK%20VIDYALAYA` — use `encodeURIComponent` in JS.)

### Status codes (ML service)

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
  "status": "ok",
  "service": "aquis-ml",
  "version": "3.0.0",
  "stations": 549,
  "dataset_last": "2026-09-10T19:00:00+05:30",
  "ollama": { "server": true, "model": "llama3.2:3b" },
  "forecast_model": "trajectory (120x6h q05/q50/q95)"
}
```
`available:true` in the gateway's `/ml/health` response when Flask is running on `:5000`; see [Python service contract](#python-service-contract) below.

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

## Python service contract — live (`ml/api.py`)

The Flask service exposes the surface below; the Node gateway (`back-end/services/mlGateway.js`) forwards path + query string to `ML_SERVICE_URL`.

### Service surface (implemented)

```
GET  /                                 index: service, version, endpoints, usage
GET  /health                           status, version, stations, dataset_last, ollama, forecast_model
GET  /stations                         ?district=&q=&limit=        recency-sorted station list (slugs + lat/lon)
GET  /stations/<slug>                  per-station facts — same rich object as chat "facts", no LLM call
GET  /stations/<slug>/series           6-hourly gwl + driver points for relation charts (?drivers=&from=&to=&limit=)
GET  /forecast/<slug>                  trajectory v2 — 120×6h q05/q50/q95 (slug URL-encoded)
POST /assistant/chat                   {question, station?, model?}   station-locked LLM answer
```

> **Planned (not yet implemented):** `/districts`, `/models`, `/fleet/forecasts`, `/fleet/recovery`, `/fleet/scan`.

### Series endpoint (relation charts)

`GET /stations/<slug>/series?drivers=rain,temp&from=2026-01-01&to=2026-09-10&limit=2000`
returns the aligned 6-hourly table slice for one station — every point carries
the observed `gwl` plus the requested drivers, so frontend clients can render
per-feature relation charts against GWL (the old Drivers-page overlays) without
touching parquet:

- `drivers` — comma list from `rain,temp,river_level,humidity,solar,wind_speed,pressure,canal_level`
  (default: all); unknown key → `400` with the `available` list.
- `from`/`to` — ISO dates, inclusive; `limit` — default 2000, max 6000, over-limit
  frames are evenly downsampled (`downsampled: true`). Missing values are `null`.
- Unknown slug → `404`. Example point:
  `{"time": "2026-09-10T18:00:00", "gwl": -2.84, "rain": 0.0, "temp": 31.5}`.

How the current `ml/` module maps to this contract:

| Contract endpoint | Data today (in `ml/`) |
|---|---|
| `/` | hardcoded service index (service, version, endpoints, usage) |
| `/health` | `_data()` (aligned 6h table), `ollama_status()`, `_index()` → stations, `forecast_model` |
| `/stations` | `data/meta/selected_gwl_stations.csv` (549 stations, recency-sorted; `latitude`/`longitude` included per item) |
| `/stations/<slug>` | `_assistant.StationAssistant().facts()` (same object as chat `"facts"`, no LLM) + `slug`/`latitude`/`longitude` envelope |
| `/stations/<slug>/series` | aligned 6h table slice (`_data()` filtered by station + date, downsampled to `limit`) — chart-ready gwl+driver points |
| `/forecast/<slug>` | `_trajectory.trajectory_forecast()` (trajectory v2, 120×6h q05/q50/q95) |
| `/assistant/chat` | `_assistant.py` (station-pinned facts + Ollama) |

---

## Quickstarts

**Backend (live):**
```bash
curl "http://localhost:3000/stations?limit=5"
curl "http://localhost:3000/telemetry/latest?stationId=<id>"
```

**ML Flask API (live):**
```bash
cd ml && venv/bin/python -m api              # http://localhost:5000
curl "http://localhost:5000/"                # service index
curl "http://localhost:5000/health"
curl "http://localhost:5000/stations?district=KANPUR"
curl "http://localhost:5000/stations/ashadha-prathmik-vidyalaya/series?drivers=rain,temp&from=2026-01-01&limit=500"
```

**ML dashboard (live):**
```bash
cd ml && venv/bin/streamlit run app.py        # http://localhost:8501
```

**Frontend flow via the Node gateway (`/ml/*` proxy):**
1. Boot → `GET /ml/health` (EMPTY banner if `available:false`).
2. Station picker → `GET /ml/stations?q=...` (sort by recency; render `last_ts`).
3. Detail view → station KPIs (mirrors `GET /ml/stations/:slug` once implemented).
4. Forecast tab → `GET /ml/forecast/:slug`; chart 120×6h `time` vs `q50` with a `q05`–`q95` band.
5. Fleet view → fleet recovery/scan endpoints once implemented.
6. Assistant → `POST /ml/assistant/chat`.