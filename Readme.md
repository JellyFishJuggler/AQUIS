# AQUIS — Aquifer Query and User Information System

A role-based groundwater monitoring platform built on NWDP telemetry and CGWB assessment data, with an XGBoost + quantile-regression forecasting engine, a station-locked AI assistant, and interactive dashboards.

**Authors:** Srijan Anand Gupta, Utkarsh Srivastava — BPIT, Rohini, New Delhi

---

## Architecture

```
back-end/          Node.js + Express API server
front-end/         Next.js 16 frontend
ml/                Python ML module: XGBoost pipeline + Flask API + Streamlit dashboard
docs/              Documentation
```

**Data flow:**
NWDP Telemetry API + Assessment Excel files → PostgreSQL → Statistical Analysis + ML → Express APIs → Frontend

The **ML module (`ml/`)** is the forecasting/analytics engine. It consumes a cleaned 6-hourly telemetry archive (`common.parquet`), trains per-station XGBoost point + quantile models, and exposes two surfaces:
- a headless **Flask HTTP API** (`ml/app.py`, port 5000) that the Node backend proxies under `/ml/*` — recency-sorted stations/districts, per-station facts, XGBoost forecasts, fleet recovery ranking, and the **station-locked LLM assistant** (Ollama);
- a **Streamlit dashboard** (`ml/app/streamlit_app.py`) for analysis.

> The Node gateway talks to the Flask service at `ML_SERVICE_URL` (default `http://localhost:5000`). See `docs/api.md` for the complete endpoint reference.

---

## Setup

### 1. Prerequisites
- Node.js 18+
- PostgreSQL 14+
- Python 3.11+ (venv/packages below tested on 3.14)

### 2. Backend
```bash
cd back-end
cp .env.example .env    # Edit with your database credentials
npm install
npm run migrate         # Apply schema
npm run import-assessments   # Import assessment Excel files
npm run import-telemetry     # Import NWDP telemetry data (large dataset)
npm start              # Start API at http://localhost:3000
```

### 3. ML Service
The `ml/` folder has two runnable surfaces — a **headless Flask API** (the backend gateway target, primary for app integration) and the **Streamlit analysis dashboard**.

```bash
cd ml
python -m venv venv
source venv/bin/activate             # Windows: venv\Scripts\activate
pip install -r requirements.txt

# A) Headless Flask API (backend ML gateway, required for live app integration)
python ml/app.py                      # http://localhost:5000   (PORT env to change)

# B) Streamlit dashboard (analysis UI)
streamlit run ml/app/streamlit_app.py  # e.g. http://localhost:8690
```

> The optional LLM assistant needs **Ollama**: install `ollama`, then `ollama pull llama3.2:3b` and `ollama serve`. Health checks report `ollama: true/false`.

> The dashboard solves station lookups by slug/partial-name and renders everything (KPIs, health, real-time series, forecasts, diagnostics, assistant) for whichever station is selected.

### 4. Frontend
```bash
cd front-end
npm install
npm run dev            # Start Next.js at http://localhost:3000
```

---

## ML module (`ml/`) in detail

### Layout

```
ml/
├─ app.py                        # Headless Flask HTTP API (backend /ml/* gateway)
├─ app/streamlit_app.py          # Streamlit dashboard (all UI layers)
├─ agent/
│  ├─ data_assistant.py          # Station-locked LLM assistant (facts + Ollama)
│  ├─ build_fleet_forecast.py    # Builds fleet_forecast_snapshot.json
│  └─ fleet_forecast_snapshot.json  # 92-station forecast snapshot
├─ models/xgboost_quantile.py    # XGBoost point + q05/q50/q95 quantile pipeline
├─ preprocessing/timeseries.py   # station series, corruption filter, splits, slugs
├─ training/                     # train_forecast, train_all_forecast, compare_models
├─ inference/predict_forecast.py # forecasting CLI
├─ scripts/                      # refresh_nwic_data, refresh_deployed_data, diagnose_fleet
├─ services/                     # interval_calibration, interpretability
├─ data/processed/common.parquet # cleaned 6-hourly telemetry 2021→2026-09-05 (5.3M rows)
└─ artifacts/
   ├─ model_comparison.csv       # XGBoost vs Random Forest (test-period)
   ├─ multistep_diagnosis.csv    # stepwise-lag robustness study
   └─ <station_slug>/            # per-station trained artifacts (direct/, recursive/, features.json)
```

### Data
- **Ingestion:** `ml/scripts/refresh_nwic_data.py` refreshes the telemetry archive from the NWIC feed; `refresh_deployed_data.py` syncs the deployed bundle. `common.parquet` is the cleaned snapshot: ~**5.3M records / 1,353 stations (UP), 6-hourly, 2021-01-01 → 2026-09-05**.
- **Cleaning:** physically impossible readings are dropped — telemetry sentinels (e.g. `-1000, 99, 999`) and `|GWL| > 100 m` spikes. All features are built causally (lags only), so nothing leaks the future.

### Model
Per station, `models/xgboost_quantile.py` trains:
- **Point model** — `XGBRegressor(objective="reg:squarederror")`
- **Quantile models** — q = 0.05 / 0.5 / 0.95 via native `reg:quantileerror` → **90% prediction interval** (approach per Alkon et al. 2024)

Features (scale-invariant, tree models need no scaling):
- `time_hours` (hours since first reading), `sin_doy`/`cos_doy` + `year` (seasonality)
- `lag_{1,2,3,4,28,120}` (1–24 h, 7 d, 30 d) and `roll_{28,120}` (7 d / 30 d means) — the primary short-horizon drivers

Evaluation is a **chronological 80/20 split** (`TRAIN_RATIO = 0.8`): the held-out tail is the "test period". `predict_xgb_quantile()` is **recursive** — it forecasts into any future horizon (or missing-data gaps); `get_test_predictions()` returns the test-period actuals vs predictions.

### Run the pipeline
```bash
# Train one station            python -m ml.training.train_forecast --station "Asafpur (UP-077)"
# Batch-train all (resume-safe) python -m ml.training.train_all_forecast [--force] [--limit 20]
# Compare XGBoost vs RF         python -m ml.training.compare_models
# Fleet diagnostics            python -m ml.scripts.diagnose_fleet
# Refresh telemetry archive    python -m ml.scripts.refresh_nwic_data
# Build fleet forecast snapshot python -m ml.agent.build_fleet_forecast
# Inference CLI                python -m ml.inference.predict_forecast --station <slug> --time 1000 2000 3000
```
All commands run from the **repository root** with the `ml/venv` Python (or the activated venv), e.g. `ml/venv/bin/python -m ml.training.train_forecast`.

### Headless ML API (`ml/app.py`)

The backend gateway and any integrator-facing app use the Flask service on port 5000:

```
GET  /health                     status, version, trained_stations, snapshot_age, ollama, dataset
GET  /stations                   ?district=&q=&limit=        recency-sorted station list
GET  /stations/:slug             latest telemetry facts for a station
GET  /districts                  recency-sorted districts
GET  /models                     trained artifact slugs
GET  /forecast/:slug             ?days=7..90 (default 30)  XGBoost forecast + calibrated 90% interval
GET  /fleet/forecasts            precomputed fleet snapshot
GET  /fleet/recovery             ?window_days=&top=&min_stations=   district recovery ranking
GET  /fleet/scan                 ?district=&threshold=&horizon=     forecast-decline scan
POST /assistant/chat             {question, station?, station_slug?, model?}
```

Stations are identified by **slug** (`Station_Agency_shortHash`), *not* DB IDs, and every GET supports CORS + returns JSON errors. **Full documentation, payloads and examples: [`docs/api.md`](docs/api.md).** The Node backend proxies these under `/ml/*` (see the API Reference below). Forecasts are **never** trained at request time — only stations in `/models` respond (404 otherwise).

### Refreshing data (`scripts/refresh_nwic_data.py`)
Fetches live NWIC telemetry and merges it into `common.parquet` (resume-safe). `refresh_deployed_data.py` republishes artifacts/models touched by a refresh. The archived series runs **2021-01-01 → 2026-09-05** and is the source of every forecast and fact in the API and dashboard.

### Dashboard (`app/streamlit_app.py`)
Runs with `streamlit run ml/app/streamlit_app.py`; a single global station selector drives every layer:

1. Header + station controls
2. Overview KPIs (current level, 90-day forecast, trend, status)
3. Station health (condition, thresholds, forecast, interpretation)
4. Real-time groundwater level time series
5. **Forecast (two panels)**
   - **LEFT — "Forecast — Test Period"**: historical backtest on the held-out tail (RMSE / MAE / R² / PI coverage). *Unchanged by design* — it is the validation view.
   - **RIGHT — "Forecast — Next 2–3 Months"**: continuous **model projection** from the station's **last stored reading → today → +90 d**. Everything after the archived reading is labeled **"Projection"** (not observed), with markers at "Stored data ends · Projection starts" and "Today".
6. Model & prediction information (train stats/technical details)
7. Diagnostics & validation (fleet summary + collapsible details)
8. Observations (recent readings + telemetry gap details)
9. **Assistant** — station-locked chat (same engine the `/ml/assistant/chat` API exposes)
10. Notes / interpretation
11. All-stations data explorer (collapsed by default)

The dashboard's forecast, facts, and assistant logic all live in reusable, non-UI modules (`ml/agent/data_assistant.py`, `ml/services/interval_calibration.py`) — the same code the headless API serves.

### How a mobile/frontend app consumes the model
The model is Python (`joblib` — not loadable by Node), so clients consume it **through the APIs**, not the artifacts:
1. Broadcast station list: `GET /ml/stations` (recency-sorted, slug + `has_model`).
2. Live/fleet views: `GET /ml/live/fleet/recovery`, `GET /ml/live/fleet/scan`, `GET /ml/live/fleet/forecasts`.
3. Per-station trends: `GET /ml/stations/:slug` (observed `change_7d/30d/60d/180d`).
4. Projections: `GET /ml/live/forecast/:slug?days=30` (`point` + calibrated `lower`/`upper` band, `plausible` guard).
5. Assistant: `POST /ml/assistant/chat`.

Live *current* readings come from the NWIC telemetry feed (the model is trained offline, it does not stream); the headless API also exposes `dataset` (last archive date) via `/health`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | — | PostgreSQL connection string |
| PORT | 3000 | Backend port |
| NODE_ENV | development | Environment |
| ML_SERVICE_URL | http://localhost:5000 | Python ML service URL (backend gateway → ml app.py) |
| ML_TIMEOUT_MS | 60000 | ML request timeout |
| BACKEND_URL | http://localhost:3000 | Data/CSV paths used by some ML helper scripts |

---

## API Reference (backend → app)

### Health
```
GET /health
```

### Stations
```
GET    /stations                          List all stations (paginated)
GET    /stations/:stationId               Station details
GET    /stations/nearby?lat=&lon=         Nearby stations
GET    /stations/state-summary            Station count by state
GET    /stations/district-summary         Station count by district
```

### Telemetry
```
GET    /telemetry                         All observations (filtered)
GET    /telemetry/latest?stationId=       Latest observation for station
GET    /telemetry/summary                 State/district summary
GET    /telemetry/:stationId              History for station
```

### Assessments
```
GET    /assessments                       Assessment records (filtered)
GET    /assessments/:id                   Single record
GET    /assessments/:unitId/history       Multi-year history for unit
GET    /assessments/summary               State/year summary
GET    /assessments/years                 Available assessment years
```

### Trends (Statistical Analysis)
```
GET    /trends/:stationId                 Mann-Kendall + Sen's slope for station
GET    /trends/summary                    Trend summary across all stations
```

### ML - Live (headless Python service, proxied) — **preferred**
```
GET    /ml/health                         Service status (+ ollama, trained_stations)
GET    /ml/stations                       Recency-sorted stations (?district=&q=&limit=)
GET    /ml/stations/:slug                 Station facts
GET    /ml/districts                      Recency-sorted districts
GET    /ml/live/models                    Trained model slugs
GET    /ml/live/forecast/:slug?days=30    XGBoost forecast + calibrated 90% interval
GET    /ml/live/fleet/forecasts           Precomputed fleet snapshot
GET    /ml/live/fleet/recovery            District recovery ranking
GET    /ml/live/fleet/scan                Forecast-decline scan
POST   /ml/assistant/chat                 Station-locked LLM assistant
```
> Full documented payloads in [`docs/api.md`](docs/api.md).

### ML - Legacy (DB registry, deprecated)
```
GET    /ml/forecast/:stationId            Get forecast (legacy numeric station ID)
GET    /ml/anomalies                      All anomalies
GET    /ml/anomalies/:stationId           Station anomalies
GET    /ml/anomalies/summary              Anomaly summary
GET    /ml/risk/:unitId                   Risk assessment
GET    /ml/risk/summary                   Risk summary
GET    /ml/risk/priority-areas            High-risk areas
GET    /ml/models                         List models
GET    /ml/models/compare?task=           Model comparison
GET    /ml/models/:name                   Model details
GET    /ml/models/:name/metrics           Model metrics
```

### ML-Ready Data
```
GET    /ml-data/telemetry                 Clean telemetry dataset
GET    /ml-data/telemetry/:stationId      Station-specific telemetry
GET    /ml-data/assessment                Clean assessment dataset
GET    /ml-data/assessment/:unitId        Unit-specific assessment history
```

### Data Quality
```
GET    /data-quality/:stationId           Station quality issues
GET    /data-quality/telemetry            Telemetry quality summary
GET    /data-quality/assessment           Assessment quality summary
```

### Ingestion
```
GET    /ingestion                         List ingestion runs
GET    /ingestion/:id                     Run details
POST   /ingestion/telemetry               Trigger telemetry ingestion
POST   /ingestion/assessment              Trigger assessment ingestion
```

### Legacy
```
GET    /data                              Legacy groundwater_data records
GET    /analytics/state-summary           State analytics
GET    /analytics/hotspots                Top over-exploited districts
GET    /analytics/status-distribution     National status breakdown
GET    /alerts                            Critical + over-exploited alerts
GET    /alerts/summary                    Alert counts by severity
```

---

## Testing
```bash
cd back-end
npm test
```
42 tests covering classification, statistics, telemetry utilities, ML gateway, and app configuration.

ML dashboard smoke tests run through `streamlit.testing.v1.AppTest` (render the app per station, assert no exceptions, section headings, and that per-station metrics/forecasts diverge as expected).

---

## Key Data Facts

- NWDP telemetry dataset: ~7.4 million records across India (backend scope); ML archive = **5.3M records / 1,353 Uttar Pradesh stations, 6-hourly, 2021-01-01 → 2026-09-05** (92 stations with trained models).
- Assessment years available: 2016-2017 through 2025-2026
- 154-column CentralReport Excel format with 3-level merged headers
- CGWB classification: Safe (<70%), Semi-Critical (70-90%), Critical (90-100%), Over-Exploited (>100%)
- Forecast uncertainty: 90% prediction interval (calibration-widened q05/q95 quantiles); long recursive horizons are **directional**, not exact
- Assistant: Ollama `llama3.2:3b`, station-locked (grounded in the same facts the API exposes)

---

## Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | Node.js, Express.js, PostgreSQL |
| ML Engine | Python, XGBoost, scikit-learn, joblib, scipy |
| ML API | Python, Flask (headless, `ml/app.py`) |
| LLM Assistant | Ollama, llama3.2:3b |
| ML Dashboard | Streamlit, Plotly |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS v4 |
| Charts | Chart.js |
| Data Sources | NWIC Telemetry API (2021→2026 archive), CGWB Assessment Excel files |