# AQUIS — Aquifer Query and User Information System

A role-based groundwater monitoring platform built on NWDP telemetry and CGWB assessment data, with an XGBoost + quantile-regression forecasting engine and an interactive analysis dashboard.

**Authors:** Srijan Anand Gupta, Utkarsh Srivastava — BPIT, Rohini, New Delhi

---

## Architecture

```
back-end/          Node.js + Express API server
front-end/         Next.js 16 frontend
ml/                Python ML module: XGBoost pipeline + Streamlit dashboard
docs/              Documentation
```

**Data flow:**
NWDP Telemetry API + Assessment Excel files → PostgreSQL → Statistical Analysis + ML → Express APIs → Frontend

The **ML module (`ml/`)** is the forecasting/analytics engine. It consumes a downloaded 2021–2025 NWIC telemetry archive, trains per-station XGBoost point + quantile models, and exposes a full **Streamlit dashboard** (analysis UI) plus Python APIs that the backend gateway can wrap.

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
The `ml/` folder has two runnable surfaces — the **Streamlit analysis dashboard** (primary) and a **Flask gateway** (`app.py`) that the backend `/ml/*` endpoints call.

```bash
cd ml
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r ../requirements.txt   # or: pip install -r requirements.txt

# A) Streamlit dashboard (analysis UI)
streamlit run ml/app/streamlit_app.py     # http://localhost:8501

# B) Flask service (backend ML gateway, optional for live app integration)
python app.py                             # http://localhost:5000
```

> The dashboard solves station lookups by slug/partial-name and renders everything (KPIs, health, real-time series, forecasts, diagnostics) for whichever station is selected.

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
├─ app/streamlit_app.py        # Streamlit dashboard (all UI layers)
├─ app.py                      # Flask gateway (bundle-backed forecast endpoint)
├─ models/
│  └─ xgboost_quantile.py      # XGBoost point + q05/q50/q95 quantile pipeline
├─ preprocessing/timeseries.py # station series, corruption filter, splits, slugs
├─ training/                   # train_forecast, train_all_forecast, compare_models
├─ inference/                  # predict_forecast (CLI)
├─ scripts/                    # export_xgboost_models, decision_support, validate_against_2026, verify_stepwise
├─ services/                   # Flask app.py helpers (forecast/anomaly/risk)
├─ data/
│  ├─ processed/common.parquet   # cleaned 6-hourly telemetry (training source)
│  └─ raw/                       # groundwater.jsonl + ingestion.py (NWIC downloader)
└─ artifacts/
   ├─ xgboost_bundle.joblib      # ALL 93 stations in one portable file
   ├─ <station_slug>/            # per-station trained artifacts
   ├─ decision_support.csv       # fleet priorities / projections
   ├─ model_comparison.csv       # XGBoost vs Random Forest (test-period)
   ├─ xgboost_summary.csv        # fleet validation metrics
   ├─ stepwise_comparison.csv    # stepwise-lag robustness study
   └─ 2026_station_presence.csv  # live-2026 reconnaissance from NWIC API
```

> The former GPR experiment (baseline models, scalers, `gpr_ready.parquet`) has
> been retired — the comparison is now **XGBoost vs Random Forest** only.

### Data
- **Ingestion:** `python ml/data/raw/ingestion.py` streams the **2021–2025 NWIC resource** (`84bfda45-…`) into `groundwater.jsonl`, resume-safe with a checkpoint. `common.parquet` is the cleaned snapshot: ~**336k records / 93 Uttar Pradesh stations**, 6-hourly.
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
# Decision support (priorities) python -m ml.scripts.decision_support   # writes decision_support.csv
# Live-2026 reconnaissance      python -m ml.scripts.validate_against_2026 --presence
# Live-2026 forecast validation python -m ml.scripts.validate_against_2026 --validate
# Export ALL models -> bundle   python -m ml.scripts.export_xgboost_models --all
# Verify bundle = per-dir preds python -m ml.scripts.export_xgboost_models --verify-all
# Inference CLI                 python -m ml.inference.predict_forecast --station asafpur_up_077 --time 1000 2000 3000
```
All commands run from the **repository root** (`/home/.../AQUIS`), e.g. `python -m ml.training.train_forecast`.
`train_all_forecast` re-exports `xgboost_bundle.joblib` automatically whenever it trains (or if no bundle
exists on a no-op run).

### Decision support (`scripts/decision_support.py`)
For every trained station, writes `artifacts/decision_support.csv`:
- **Trend** — Theil-Sen slope on the last ~2 years (m/yr) with a direction label (more-negative GWL = deeper; negative slope = declining).
- **Thresholds** — data-driven: `critical` = deepest 10% of the station's own history, `caution` = deepest 30%.
- **Projection** — recursive XGBoost at **+90 / +180 days** with the 90% PI.
- **Priority** — 2×2 decision grid (Declining×Caution → **PRIORITY**, etc.), escalated to PRIORITY if projected into the critical zone in 180 d, plus a plain-language narrative.

### Model bundle & the app forecast endpoint

`python -m ml.scripts.export_xgboost_models` packs **all 93 stations** (point + q05/q50/q95 models, `features.json`, metadata) into one compressed **`artifacts/xgboost_bundle.joblib`** (~60 MB). Bundled predictions are **bit-identical** to the per-directory path (same prediction internals), which `--verify-all` proves for every station.

The Flask gateway (`ml/app.py`) loads the bundle once and serves:

```
GET /forecast/xgb/<slug-or-name-substring>?horizon_days=90
```

Response — a daily projection from the station's **last stored reading** out to `horizon_days` (max 180) with the 90% interval (true last-reading anchor, not the series end):

```json
{
  "engine": "xgboost-quantile",
  "station": "Asafpur (UP-077)",
  "slug": "asafpur_up_077",
  "anchor": "2025-12-31T12:00:00+05:30",
  "last_observed_value": -16.188,
  "horizon_days": 90,
  "quantiles": [0.05, 0.5, 0.95],
  "dates":   ["2026-01-01T12:00", "2026-01-02T12:00", "..."],
  "point":   [-16.211, -16.224, "..."],
  "lower":   [-16.620, -16.640, "..."],
  "upper":   [-15.801, -15.803, "..."]
}
```

Notes for the app backend:
- The whole bundle lives in one file — ship it (and `model_comparison.csv`, `decision_support.csv`) to the ML service, no per-station artifacts needed.
- `/health` reports the loaded bundle count once loaded.
- The legacy `POST /forecast/<int:station_id>` (sklearn RF/linear on live-fetched observations) is **kept** for backward compat; migrate the backend gateway to `/forecast/xgb/...` to get the real XGBoost + interval forecasts.
- Long recursive horizons are directional, not exact — the Weak-coverage warning in Section 5 applies here too.

### Live-2026 validation (`scripts/validate_against_2026.py`)
Probes the **2026–2030 NWIC resource** (`31c66a49-…`) by binary search (~21 API calls/station) to produce `2026_station_presence.csv` (found/missing, record count, per-station timestamp range — **timestamps only, no values**). `--validate` then fetches each station's real 2026 block and scores the recursive forecast's 90% PI coverage — the honest long-horizon check (weak coverage at ≥90 d is documented and surfaced in the dashboard).

### Dashboard (`app/streamlit_app.py`)
Runs with `streamlit run ml/app/streamlit_app.py`; a single global station selector drives every layer:

1. Header + station controls
2. Overview KPIs (current level, 90-day forecast, trend, status)
3. Station health (condition, thresholds, forecast, interpretation)
4. Real-time groundwater level time series
5. **Forecast (two panels)**
   - **LEFT — "Forecast — Test Period"**: historical backtest on the held-out tail (RMSE / MAE / R² / PI coverage). *Unchanged by design* — it is the validation view.
   - **RIGHT — "Forecast — Next 2–3 Months"**: continuous **model projection** from the station's **last stored reading → today → +90 d**. Everything after the stored archive is labeled **"Projection"** (not observed): the stored data ends at each station's own last 2025 telemetry reading, while real 2026 readings exist only in the live NWIC feed and aren't stored locally — so the gap is filled by the model, with markers at "Stored data ends · Projection starts" and "Today", and captions explaining stored-vs-live data and per-station 2026 feed status.
6. Model & prediction information (train stats/technical details)
7. Diagnostics & validation (fleet summary + collapsible details, incl. 2026 NWIC coverage)
8. Observations (recent readings + telemetry gap details)
9. Notes / interpretation
10. All-stations data explorer (collapsed by default)

### How the mobile app consumes the model
The model is Python (`joblib` — not loadable by Node). For the React Native + Node stack, use the **model output bundle** the pipeline already produces (`decision_support.csv`, `xgboost_summary.csv`, `2026_station_presence.csv`) as static data — fleet tiers, trends, +90/+180 d projections, and validation metrics per station — and optionally a **Python FastAPI/Flask wrapper** (models are already exposed via `ml/app.py`): e.g. `GET /ml/forecast/:stationId` → `{station, anchor, horizon, dates[], point[], lower[], upper[]}`, or on-demand `predict_xgb_quantile(station, hours)` for arbitrary horizons. Live *current* readings should come from the NWIC telemetry feed (the model is trained offline, it does not stream).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | — | PostgreSQL connection string |
| PORT | 3000 | Backend port |
| NODE_ENV | development | Environment |
| ML_SERVICE_URL | http://localhost:5000 | Python ML service URL (backend gateway → ml app.py) |
| ML_TIMEOUT_MS | 60000 | ML request timeout |
| BACKEND_URL | http://localhost:3000 | Used by `ml/app.py` for callbacks |

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

### ML - Forecast, Anomaly, Risk
```
GET    /ml/health                         ML service health
GET    /ml/forecast/:stationId            Get forecast
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

- NWDP telemetry dataset: ~7.4 million records across India (backend scope); ML subset = **93 Uttar Pradesh stations, 6-hourly, 2021–2025 (~336k records)**.
- Assessment years available: 2016-2017 through 2025-2026
- 154-column CentralReport Excel format with 3-level merged headers
- CGWB classification: Safe (<70%), Semi-Critical (70-90%), Critical (90-100%), Over-Exploited (>100%)
- Forecast uncertainty: 90% prediction interval from q05/q95 XGBoost quantile models; long recursive horizons are **directional**, not exact (weak coverage is measured and surfaced, not hidden)

---

## Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | Node.js, Express.js, PostgreSQL |
| ML Engine | Python, XGBoost, scikit-learn, joblib, scipy |
| ML Dashboard | Streamlit, Plotly |
| ML Gateway | Python, Flask |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS v4 |
| Charts | Chart.js |
| Data Sources | NWDP Telemetry API (2021–2025 archive + 2026–2030 live feed), CGWB Assessment Excel files |