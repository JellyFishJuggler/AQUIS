# AQUIS — Aquifer Query and User Information System

A role-based groundwater monitoring platform built on NWDP telemetry and CGWB assessment data, with a pooled XGBoost machine-learning forecasting engine, quantile uncertainty calibration, a station-locked AI assistant, and interactive dashboards.

**Authors:** Srijan Anand Gupta, Utkarsh Srivastava — BPIT, Rohini, New Delhi

---

## Architecture

```
back-end/          Node.js + Express API server
front-end/         Next.js 16 frontend
ml/                Python ML module: pooled XGBoost pipeline + Streamlit analysis dashboard
docs/              Documentation
```

**Data flow:**
NWDP Telemetry API + Assessment Excel files → PostgreSQL → Statistical Analysis + ML → Express APIs → Frontend

The **ML module (`ml/`)** is the forecasting/analytics engine. It consumes a cleaned 6-hourly telemetry archive, trains one **pooled (global) XGBoost** model over 600 Uttar Pradesh groundwater stations, calibrates q05/q50/q95 quantile forecasts into 90% prediction intervals, scans the whole fleet for 30-day moves, and exposes an 8-page **Streamlit dashboard** (`ml/app.py`, port 8501).

> A headless **Flask HTTP API** that the Node gateway (`ML_SERVICE_URL` → `http://localhost:5000`) would proxy under `/ml/*` is **planned but not yet merged** — see `merge_later` and `docs/api.md`. Currently the backend's `/ml/*` routes have no live Python service backing them; run the Streamlit app below for the current analysis surface.

---

## Scope & data coverage

**Uttar Pradesh groundwater monitoring is mostly *manual*, not real-time.** This is the single most important constraint on what AQUIS can forecast.

- UP has ~**75 districts**, and groundwater there is monitored through two separate networks on the National Water Data Portal (NWDP/NWIC):
  - **Manual** — measured a few times a year by field staff (CGWB ~1,000 UP wells, 4×/yr; UPGW publishes its own *"Ground Water Level (Manual - Quarterly)"* dataset, 1991–2020).
  - **Telemetry** — automated 6-hourly Digital Water Level Recorders, a *narrow, recently-deployed slice* of the network.
- **AQUIS only ingests the telemetry feed** (`GWL Telemetry 6-Hourly, UPGW`). That feed wires just **1,353 stations across 34 of UP's ~75 districts**. Districts without telemetry gauges never appear in the archive — verified: NWIC probes return `total=0` for e.g. Allahabad/Prayagraj, Amethi, Amroha, Sambhal, G.B. Nagar even with rename aliases (`ml/data/raw/nwic.py`).
- Therefore, in this repo: **1,353 stations / 34 districts = "everything the telemetry archive has"**, and **600 stations / 29 districts = "the ones still alive in 2026"** (selected by `01_select.py`, ≥8 of 9 Jan–Sep 2026 months with a reading). Neither number is a scope preference — the data decides.
- **Project vs ML scope:** the backend ingests NWDP telemetry countrywide (~7.4M records) + CGWB assessment Excels; the **forecasting** scope is **UP-GWL telemetry only**. A station without a live 2026 telemetry stream is simply not forecast-able by this pipeline.
- More detail + the source links (UPGW portal, CGWB monitoring-station page, CGWB high-frequency-data guidelines, UP GW Year Book) live in **`ml/MODEL_CARD.md`**.

---

## Setup

### 1. Prerequisites
- Node.js 18+
- PostgreSQL 14+
- Python 3.11+ (venv/packages below tested on 3.14)

### 2. Backend
```bash
cd back-end
cp .env.example .env        # Edit with your database credentials
npm install
npm run migrate             # Apply schema
npm run import-assessments  # Import assessment Excel files
npm run import-telemetry    # Import NWDP telemetry data (large dataset)
npm start                   # Start API at http://localhost:3000
```

### 3. ML dashboard
```bash
cd ml
venv/bin/streamlit run app.py     # http://localhost:8501
```
The ML venv lives at `ml/venv` (Python 3.14). Use `ml/venv/bin/python` for all pipelines below.

> The optional **AI assistant** page needs **Ollama**: install `ollama`, then `ollama pull llama3.2:3b` and ensure `ollama serve` is running. The instant-facts panel works even when Ollama is down.

### 4. Frontend
```bash
cd front-end
npm install
npm run dev                 # Start Next.js at http://localhost:3000
```
> Note: Express runs on `:3000`; if both run, pick a different port for one of them.

---

## ML module (`ml/`) in detail

### Layout

```
ml/
├─ app.py                        # Streamlit dashboard (8 pages)
├─ app_pages/                    # page scripts (overview, correlation, drivers, stations,
│                                #   model, forecast, assistant, sources/fleet)
├─ config.py                     # sources, resource IDs, radii, coverage rules, paths
├─ 00_probe.py … 13_refresh_nwic.py   # ordered pipeline (see below)
├─ _model.py _assistant.py _utils.py _soil.py _lulc.py   # shared libs
├─ validation/                   # P0 honesty suite (spatial CV, overlap, residual ACF)
├─ data/
│  ├─ processed/common.parquet   # cleaned 6-hourly GWL archive 2021→2026 (source of truth)
│  ├─ selected/ gwl_selected.csv # per-source selected-district pulls
│  ├─ raw/          *_norm.parquet (cleaned standardised), nwic.py (NWIC client)
│  ├─ aligned/      table.parquet (daily) + table_6h.parquet (6h grid, 2.99M rows)
│  └─ meta/         associations, manifest, probe, station flags
├─ models/                      # joblib + feature_config.json + quantile_calibration.json
│                               #   + model_metadata.json
├─ outputs/                     # correlation, benchmark, honest metrics, fleet, diagnostics
├─ tests/                       # stdlib unittest suite (no pytest dependency)
├─ MODEL_CARD.md                # lifecycle card for the pooled model
└─ gate_check.py                # regression gate: verifies forecast baselines survive edits
```

### Pipeline (run in order, each step from `ml/`)
```
00_probe.py           probe every NWIC resource (archive + live) → meta/probe.json
01_select.py          select GWL anchor stations with full-2026 coverage  → 600 of 1,353
02_fetch_selected.py  fetch each driver for the selected districts (resume-safe)
03_merge_normalize.py plausibility caps → raw/<source>_norm.parquet + manifest
04_align.py           daily station×day table with spatial association (IDW rain, nearest weather)
04b_align_6h.py       6-hourly station×slot table (production-consistent cadence)
05_correlate.py       driver correlation report, recharge-lag curve, charts
06_features.py        feature/target engineering on the 6h grid
06_train.py           pooled XGBoost + Ridge (delta target), writes models/ + model_metadata.json
07_evaluate.py        4-way benchmark (XGB / Ridge / persistence / climatology) + honest re-score
07_soil.py            ISRIC SoilGrids per-station texture (background fetch)
08_lulc.py            ISRO Bhuvan LULC 50K district stats
09_cfs_rain.py        CFSv2 seasonal-rain fetch (optional, staged)
10_ablate.py          drop-group ablation sweep
11_quantile.py        pooled q05/q50/q95 + empirical coverage calibration
11_fleet.py           whole-fleet 30-day forecast scan + recovery snapshot
12_diagnostics.py     VIF + permutation importance + OAT sensitivity
13_refresh_nwic.py    incremental NWIC refresh (+ optional retrain / deploy sync)
```

### Data
- **Archive:** `data/processed/common.parquet` — the cleaned 6-hourly telemetry source of truth: **~5.3M records / 1,353 UP stations, 2021-01-01 → 2026-09-05**. Every pipeline step reads it read-only.
- **Refresh:** `13_refresh_nwic.py` pulls *newer-than-current-max* rows per district from the live NWIC 2026 resource, merges (deduped, chronological, dtype-coerced) back into the archive, and can retrain (`--retrain`) or run the staging check / swap (`--deploy-check`, `--deploy-install`). Run it **manually**:
  ```bash
  cd ml
  venv/bin/python -u 13_refresh_nwic.py --dry-run       # report new rows
  venv/bin/python -u 13_refresh_nwic.py --retrain       # merge + rerun pipeline to the trained model
  venv/bin/python 13_refresh_nwic.py --deploy-check     # is the loaded build 2026-current?
  ```
- **Cleaning:** telemetry sentinels and `|GWL|>100 m` spikes dropped; per-station 0.5–99.5% quantile trim; stations with a >15 m annual-median regime shift (datum errors) excluded; features are causal (lags only).

### Model
One **pooled** model, not one per station:
- **Grid + horizon:** 6-hourly track, single **30-day** horizon (`GWL(t+120) − GWL(t)`, delta target). Today's level is the anchor feature — never predicted.
- **XGBoost:** `reg:squarederror`, early stopping on the Oct–Dec 2025 validation split; train sampled 1 slot/day for memory.
- **Ridge** (linear baseline), **persistence** (0-change), and per-station **day-of-year climatology** are benchmarked in `07_evaluate.py`.
- **Quantile uncertainty** (`11_quantile.py`): native `reg:quantileerror` q05/q50/q95 pooled forecasters, empirically calibrated on non-overlapping windows → `quantile_calibration.json` (stride coverage 0.911 ≥ 0.80 target, so `k = 1.0`; median band ±~1.03 m).

### Head-to-head (30-day, 2026 held-out, 6h grid)
| Model | Level RMSE (m) | Notes |
|---|---|---|
| **XGBoost (pooled)** | **2.242** | beats persistence +2.1% raw |
| persistence | 2.290 | |
| Ridge | 2.456 | |
| climatology | 3.572 | |

**Honest scale of evidence** (overlap-aware, non-trivial): because 6-hourly 30-day windows overlap ~99%, the trustworthy campaign is the **stride** set — **2.339 m vs persistence 2.382 m (+1.8%)** on 3,371 independent windows; effective sample ≈ 6,600 rows (lag-1 ACF 0.86). **Spatial CV** (leave-block-out station retraining) = 1.97 m mean / 1.82 m median — the 2026 temporal holdout, not spatial leakage, is the binding constraint. Feature ablation: rain/weather/calendar add ~+0.03–0.05 m when dropped; river/canal and GWL lags are neutral at 30 d given the anchor.

### Verify after editing
```bash
cd ml
venv/bin/python -m unittest discover -s tests    # 21 data-gated tests
venv/bin/python gate_check.py                    # 8 baseline checks (RMSE/CV/coverage/eff-N/ACF)
```

### Dashboard (`app.py`, Streamlit, 8 pages)
Overview · Correlation · Drivers · Stations · Model (benchmark + honest validation + diagnostics) · Forecast (2026 backtest + forward outlook with calibrated band) · Assistant (station-pinned LLM) · Sources (manifest quality, association method, soil/LULC status). Fleet scan + significant-movers alerts surface on the Sources/Fleet view.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | — | PostgreSQL connection string |
| PORT | 3000 | Backend port |
| NODE_ENV | development | Environment |
| ML_SERVICE_URL | http://localhost:5000 | Python ML service URL — **not active** until the Flask layer is merged |
| ML_TIMEOUT_MS | 60000 | ML request timeout |
| BACKEND_URL | http://localhost:3000 | Data/CSV paths used by some ML helper scripts |
| AQUIS_OLLAMA_MODEL | llama3.2:3b | Assistant model override |
| LULC_STATISTICS_API_KEY | — | ISRO Bhuvan LULC API key (Sources page) |

---

## API Reference (backend → app)

Full backend/ML endpoint reference lives in **[`docs/api.md`](docs/api.md)**. Summary:

- **Node `:3000`** endpoints: stations, telemetry, assessments (CGWB), trends (Mann-Kendall + Sen's slope), ml-data, data-quality, ingestion.
- **ML `/ml/*` (live, Python-backed)** — the Flask service that would serve recency-sorted stations, per-station facts, forecasts, fleet recovery/scan, and the assistant **is planned (`merge_later`), so these routes currently return "ML service unavailable"** until it lands.
- **ML legacy** (DB-registry) endpoints remain for backward compatibility.

---

## Testing
```bash
cd back-end
npm test
```
Backend tests cover classification, statistics, telemetry utilities, ML gateway, and app configuration.

ML validation: stdlib `unittest` suite in `ml/tests/` (data-gated, no pytest) + `ml/gate_check.py` regression gate + `streamlit.testing.v1.AppTest` smoke over the 8 dashboard pages.

---

## Key Data Facts

- NWDP telemetry: ~7.4M records across India (backend scope); ML archive = **5.3M records / 1,353 UP stations, 6-hourly, 2021-01-01 → 2026-09-05**; **600 stations / 29 districts** quality in as pooled-model anchors. *(Coverage is telemetry-limited — UP GW monitoring is mostly manual/quarterly; see [Scope & data coverage](#scope--data-coverage).)*
- Assessment years: 2016-2017 … 2025-2026 · 154-column CentralReport Excel (3-level merged headers).
- CGWB classification: Safe (<70%) / Semi-Critical (70–90%) / Critical (90–100%) / Over-Exploited (>100%).
- Forecast uncertainty: **calibrated 90% interval** (q05/q95, `k=1.0`), median half-width ≈1.03 m; 30-day outlooks are **directional**, not exact.
- Assistant: Ollama `llama3.2:3b`, station-pinned, grounded in the same data the model consumes.
- Fleet (Sep 2026 scan): **502 stations scored** — median Δ **+0.32 m**, decline 0, recovering 259 at ±0.3 m; no station clears its 90% band (post-monsoon recovery).

---

## Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | Node.js, Express.js, PostgreSQL |
| ML Engine | Python, XGBoost, scikit-learn, joblib, scipy |
| LLM Assistant | Ollama, llama3.2:3b |
| ML Dashboard | Streamlit, Plotly |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS v4 |
| Charts | Chart.js |
| Data Sources | NWIC Telemetry API (2021→2026 archive), CGWB Assessment Excel files |