# AQUIS — Aquifer Query and User Information System

A role-based groundwater monitoring platform built on NWDP telemetry and CGWB assessment data, with pooled XGBoost **and quantile-GRU (deep-learning)** forecasting engines replacing each other in production, quantile uncertainty calibration, a station-locked AI assistant, and interactive dashboards.

**Authors:** Srijan Anand Gupta, Utkarsh Srivastava — BPIT, Rohini, New Delhi

---

## Architecture

```
back-end/          Node.js + Express API server
front-end/         Next.js 16 frontend
ml/                Python ML module: pooled XGBoost + quantile-GRU pipelines + Streamlit analysis dashboard
docs/              Documentation
```

**Data flow:**
NWDP Telemetry API + Assessment Excel files → PostgreSQL → Statistical Analysis + ML → Express APIs → Frontend

The **ML module (`ml/`)** is the forecasting/analytics engine. It consumes a cleaned 6-hourly telemetry archive and serves **two forecast surfaces** over 600 Uttar Pradesh groundwater stations — a **pooled (global) XGBoost** 30-day model and a **quantile GRU** deep-learning model. Both surfaces are now served by the **GRU** (deep-learning) engine, with XGBoost retained as an automatic fallback and benchmark. Forecasts are calibrated q05/q50/q95 into 90% prediction intervals, the whole fleet is scanned for 30-day moves, and a **4-page Streamlit dashboard** (`ml/app.py`, port 8501) exposes it.

The forecast surface is a dedicated **trajectory v2** engine (`ml/serving_engines.py`: XGBoost + GRU, dispatched by `ml/serving_core.py`): a genuine **120-step, 6-hourly, 30-day** forecast per station (no interpolation — one real model output per 6 h step), with per-horizon q05/q50/q95 and an **evidence-based confidence label** (HIGH / DIRECTIONAL / LOW). Drivers ahead of the anchor are legitimate forecasts — **Open-Meteo** weather (days 1–16) with climatology beyond, and **CWC 3-day river forecasts** where the district is covered. The Forecast page renders this as a single dark-theme dashboard card (bright q05/q95 band, no dot markers).

A **near-real-time refresh daemon** (`ml/refresh/`) keeps forecast surfaces current: it runs fetch → align → inference → publish on a fixed 6-hourly wall-clock grid (01:00 / 07:00 / 13:00 / 19:00 IST), writes per-station forecasts + `meta.json`, and scores realised readings (verification + drift gating).

> A headless **Flask HTTP API** (`ml/api.py`) runs alongside the Streamlit dashboard —
> root `/` lists endpoints, `/health`, `/stations`, `/forecast/<slug>`, `/assistant/chat`.
> The Node gateway can proxy it via `ML_SERVICE_URL` (`http://localhost:5000`).

---

## Scope & data coverage

**Uttar Pradesh groundwater monitoring is mostly *manual*, not real-time.** This is the single most important constraint on what AQUIS can forecast.

- UP has ~**75 districts**, and groundwater there is monitored through two separate networks on the National Water Data Portal (NWDP/NWIC):
  - **Manual** — measured a few times a year by field staff (CGWB ~1,000 UP wells, 4×/yr; UPGW publishes its own *"Ground Water Level (Manual - Quarterly)"* dataset, 1991–2020).
  - **Telemetry** — automated 6-hourly Digital Water Level Recorders, a *narrow, recently-deployed slice* of the network.
- **AQUIS only ingests the telemetry feed** (`GWL Telemetry 6-Hourly, UPGW`). That feed wires just **1,353 stations across 34 of UP's ~75 districts**. Districts without telemetry gauges never appear in the archive — verified: NWIC probes return `total=0` for e.g. Allahabad/Prayagraj, Amethi, Amroha, Sambhal, G.B. Nagar even with rename aliases (`ml/data/raw/nwic.py`).
- Therefore, in this repo: **1,353 stations / 34 districts = "everything the telemetry archive has"**, and **600 stations / 29 districts = "the ones still alive in 2026"** (selected by `pipeline_data select`, ≥8 of 9 Jan–Sep 2026 months with a reading). Neither number is a scope preference — the data decides.
- **Project vs ML scope:** the backend ingests NWDP telemetry countrywide (~7.4M records) + CGWB assessment Excels; the **forecasting** scope is **UP-GWL telemetry only**. A station without a live 2026 telemetry stream is simply not forecast-able by this pipeline.
- More detail + the source links (UPGW portal, CGWB monitoring-station page, CGWB high-frequency-data guidelines, UP GW Year Book) live in **`docs/ml.md` §5.1.1**.

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
├─ app.py                        # Streamlit dashboard (4 pages: Assistant, Correlation,
│                                #   Forecast, Sources; Verification page exists, unwired)
├─ app_pages/                    # page scripts (assistant, correlation, forecast,
│                                #   data_sources + verification; legacy Overview/Drivers/
│                                #   Stations/Model/Fleet pages removed)
├─ app_charts.py                 # dark-theme Altair helpers for the forecast chart
├─ config.py                     # sources, resource IDs, radii, coverage rules, paths
├─ pipeline_data.py              # ingest/align + static/source fetchers + CLI
├─ pipeline_model.py             # features/target/train/evaluate + experiments + CLI
├─ pipeline_traj.py              # trajectory datasets/XGB backtest/reliability/CI + CLI
├─ pipeline_traj_dl.py           # trajectory + pooled quantile GRU trainers
├─ openmeteo_weather.py         # Open-Meteo daily weather (365d history + 16d forecast)
├─ benchmark_2026.py             # rolling 2026 benchmark snapshot
├─ gate_check.py                 # regression gate
├─ serving_core.py              # artifact/data loaders + runtime backend dispatch
├─ serving_engines.py           # XGB + quantile-GRU trajectory/forward engines
├─ static_loaders.py            # soil / extraction / LULC static loaders
├─ _assistant.py _hydrology.py _thresholds.py  # assistant + shared pure helpers
├─ api.py                        # Flask HTTP API (JSON; /health, /stations, /forecast/<slug>, /assistant/chat)
├─ snapshot.py                   # matplotlib PNG export (used by Snapshot tests only)
├─ refresh/                      # refresh pipeline: config, features, inference,
│                                #   model_update, publish, schedule, scheduler, state, verification, CLI
├─ data/
│  ├─ processed/common.parquet   # cleaned 6-hourly GWL archive 2021→2026 (source of truth)
│  ├─ aligned/      table.parquet (daily) + table_6h.parquet (6h grid, ~3M rows)
│  ├─ cfs/          openmeteo_weather_daily.parquet + river_forecast_cwc.parquet
│  ├─ meta/         associations, manifest, probe, station flags, climatology
│  ├─ features_6h/  train/val/test parquet (causal 6h feature frames; built by `pipeline_data features6h`)
│  └─ soil/         ISRIC SoilGrids + ISRO LULC (one-off fetch)
├─ experiments/                 # baselines (arima_compare/combine), validation, diagnosis
├─ models/                      # joblib (pooled XGB) + traj_*.json (traj XGB) + *_dl.pt/*_dl_config.json (GRU)
├─ outputs/                     # correlation, benchmark, honest metrics, fleet, diagnostics, DL eval
├─ tests/                       # stdlib unittest suite (no pytest dependency; incl. DL parity/dispatch)
└─ gate_check.py                # regression gate (frozen baselines + trajectory)
```

### Pipeline (stages, each from `ml/`)
```
python -m pipeline_data probe        # probe NWIC resources → meta/probe.json
python -m pipeline_data select       # full-2026 GWL anchor stations
python -m pipeline_data fetch        # fetch selected drivers (resume-safe)
python -m pipeline_data normalize    # plausibility caps → raw/<source>_norm.parquet
python -m pipeline_data align        # daily station×day table + spatial association
python -m pipeline_data align6       # 6-hourly station×slot table (production cadence)
python -m pipeline_data correlate    # driver correlation + recharge-lag charts
python -m pipeline_data features     # feature/target engineering on the 6h grid
python -m pipeline_model train       # pooled XGBoost + Ridge (delta target)
python -m pipeline_model evaluate    # 4-way benchmark + honest re-score
python -m pipeline_data soil         # ISRIC SoilGrids texture (optional)
python -m pipeline_data lulc         # ISRO Bhuvan LULC 50K stats (optional)
python -m pipeline_model ablate      # drop-group ablation sweep
python -m pipeline_model quantile    # pooled q05/q50/q95 + coverage calibration
python -m pipeline_model fleet       # whole-fleet 30-day scan + recovery snapshot
python -m pipeline_model diagnostics # VIF + permutation importance + OAT
python -m pipeline_data nwic         # incremental NWIC refresh
python -m pipeline_data climatology # driver-climatology refresh
python -m pipeline_data cwc_forecast # CWC 3-day river forecast fetch
python openmeteo_weather.py        # Open-Meteo daily weather

# trajectory chain
python -m pipeline_traj datasets          # train/val long frames (exact-time targets)
python -m pipeline_traj train             # shared multi-horizon XGBoost q05/q50/q95
python -m pipeline_traj backtest          # honest non-overlap backtest + calibration
python -m pipeline_traj reliability       # reliability buckets + evidence weights
python -m pipeline_traj train_dl          # trajectory quantile GRU
python -m pipeline_traj paired_ci         # DL-vs-XGB paired bootstrap CI
python -m pipeline_traj train_pooled_dl   # pooled daily quantile GRU
python -m pipeline_traj backtest_pooled_dl  # honest 2026-stride pooled-DL eval
```

### Data
- **Archive:** `data/processed/common.parquet` — the cleaned 6-hourly telemetry source of truth: **~5.3M records / 1,353 UP stations, 2021-01-01 → 2026-09-05**. Every pipeline step reads it read-only.
- **Refresh:** `pipeline_data nwic` pulls *newer-than-current-max* rows per district from the live NWIC 2026 resource, merges (deduped, chronological, dtype-coerced) back into the archive, and can retrain (`--retrain`) or run the staging check / swap (`--deploy-check`, `--deploy-install`). Run it **manually**:
  ```bash
  cd ml
  venv/bin/python -u -m pipeline_data nwic --dry-run       # report new rows
  venv/bin/python -u -m pipeline_data nwic --retrain       # merge + rerun pipeline to the trained model
  venv/bin/python -m pipeline_data nwic --deploy-check     # is the loaded build 2026-current?
  ```
- **Cleaning:** telemetry sentinels and `|GWL|>100 m` spikes dropped; per-station 0.5–99.5% quantile trim; stations with a >15 m annual-median regime shift (datum errors) excluded; features are causal (lags only).

### Models — XGBoost + quantile GRU on two surfaces

Production serves **two families on two surfaces**, and each surface is now served by the **GRU** with XGBoost kept as an automatic fallback/benchmark (`serving_engines.py`, dispatched by `serving_core.py`; backend flags in `data/refresh/runtime.json`):

- **Surface (a) — trajectory v2** (Forecast page + `/forecast/<slug>`): a genuine **120-step, 6-hourly, 30-day** per-horizon forecast.
  - **XGBoost** — direct multi-horizon shared booster, horizon `h ∈ 1..120` an input feature, q05/q50/q95 heads (`pipeline_traj train`).
  - **GRU (served)** — sequence encoder over the last 120 observed 6-h rows, horizon-conditioned quantile heads, pinball loss (`pipeline_traj train_dl` → `models/traj_dl.pt`).
- **Surface (b) — pooled 30-day** (assistant, fleet scan, +30 d benchmark): the single delta `GWL(t+120) − GWL(t)`.
  - **XGBoost** — `reg:squarederror`, early stopping on the Oct–Dec 2025 split (`pipeline_model train` → `models/xgb_multihorizon.joblib`).
  - **GRU (served)** — daily GRU over a 60-day window + 3 pinball quantile heads (q05/q50/q95), 3-seed ensemble (`pipeline_traj train_pooled_dl` → `models/pooled_dl.pt`).
- **Grid + horizon:** 6-hourly track. Today's level is the anchor feature — never predicted.
- **Baselines:** Ridge (linear), persistence (0-change) and per-station day-of-year climatology (`pipeline_model evaluate`).
- **Quantile uncertainty:** native quantile heads calibrated into 90% intervals (XGB `pipeline_model quantile`; GRU per-horizon `*_dl_calibration.json`).

### Trajectory v2 — the Forecast page model

- **Forward drivers are forecasts, not observations:** Open-Meteo weather (days 1–16) with climatology beyond, and CWC 3-day river forecasts where the district is covered. A missing driver **downgrades** confidence at those horizons.
- **Confidence framework:** per-point HIGH/DIRECTIONAL/LOW from **weighted evidence** (interval quality, vs-persistence, direction, width, station integrity, driver availability, anchor OOD, stability) — not interval width alone. Anchor = latest observed reading, never predicted; `q05 ≤ q50 ≤ q95` at every step.
- **Forecast page:** single dark-theme card — one "Forecast starts" boundary, observed tail, q50 + a bright q05/q95 band (no dot markers), 6 metric cards (+24h/+7d/+30d/change/confidence), direction banner. The card intentionally references only the trajectory forecast.

### Head-to-head (30-day, 2026 held-out, 6h grid)
| Model | Level RMSE (m) | Notes |
|---|---|---|
| **XGBoost (pooled)** | **2.242** | beats persistence +2.1% raw |
| persistence | 2.290 | |
| Ridge | 2.456 | |
| climatology | 3.572 | |

> XGBoost-level table above; the GRU's surface (b) result uses the non-overlapping 2026 stride (RMSE 2.082 m) and is detailed in **Consolidated algorithm findings** below.

**Honest scale of evidence** (overlap-aware, non-trivial): because 6-hourly 30-day windows overlap ~99%, the trustworthy campaign is the **stride** set — **2.339 m vs persistence 2.382 m (+1.8%)** on 3,371 independent windows; effective sample ≈ 6,600 rows (lag-1 ACF 0.86). **Spatial CV** (leave-block-out station retraining) = 1.97 m mean / 1.82 m median — the 2026 temporal holdout, not spatial leakage, is the binding constraint. Feature ablation: rain/weather/calendar add ~+0.03–0.05 m when dropped; river/canal and GWL lags are neutral at 30 d given the anchor.

### Consolidated algorithm findings (XGBoost · GRU · SARIMA · GPR)

Beyond the ridge/persistence/climatology benchmarks, four algorithm families were
compared on the forecast surfaces. Two are shipped (XGBoost + GRU), two rejected.

| Family | Where tested | Result | Verdict |
|---|---|---|---|
| **XGBoost** | surfaces (a)+(b) | trajectory v2 30-d RMSE **2.106 m** (< direct-30d 2.132 < persistence 2.151); pooled 2026 test **2.242 m** (+2.1% vs persistence) | incumbent / retained as fallback |
| **Quantile GRU** | surfaces (a)+(b) | surface (b): 2026 RMSE **2.082 m**, beats E0-XGB (paired ratio **0.944 [0.879, 0.972]**) and persistence (0.952 [0.906, 0.972]); surface (a): beats XGB at long horizons (h360 0.985, **h720 0.990** [0.972, 0.996]) but ties/slightly loses at h≤12 | **now serving both surfaces** |
| **SARIMA** | daily matched-pair (10 st × 5 origins) | RMSE 0.637 vs XGB 0.617 (worse), MAE 0.335 vs 0.402 (better), PICP90 **0.726 vs 0.968** (under-covers); ARIMA+XGB blend beat the better single in 0/27 folds | rejected / dropped |
| **GPR** | per-station, time-only feature | frozen pre-2026 → 2026: Tindwara RMSE **7.82 / R² −401**, Banda 2.13 / −7.6; PICP 1.00 (uninformative); O(n³) fit | rejected (interpolation only) |

**Detailed verdicts:**
- **XGBoost** — the benchmark both surfaces must beat; still shipped as the fallback path when the DL artifacts are unavailable.
- **Quantile GRU** — the two DL models replace the incumbents **on the served path**. Evidence: (b) is a clean win over E0-XGB and persistence on the non-overlapping 2026 stride (`ml/outputs/pooled_dl_eval.json`, n=4,355, 511 stations; DL bias −0.030, direction-acc 0.732, coverage_90 0.925); (a) wins the 30-day endpoint (`traj_backtest_summary_dl.json`: 30-d RMSE 1.802 vs persistence 1.907, dir 0.725, calibrated coverage 0.894) and the per-horizon bootstrap (`traj_paired_ci_dl_vs_xgb.json`) shows it **better or equal from ~2 days out**, losing only marginally at 6–12 h. Promotion gates stay conservative (`promote_trajectory=false` until short-horizon gates clear and the win generalises beyond 2026), so DL runs under the same conservative gating as the XGB trajectory.
- **SARIMA** (`experiments/baselines/arima_compare.py`, protocol: daily means, 10 stations × 5 rolling-2026 origins — not comparable to the 6-hourly tables): point-competitive on stable stations (Tindwara RMSE 0.028) but collapses on volatile ones (Banda 5.91) and under-covers (PICP90 0.726 vs XGB 0.968); the ARIMA+XGB **blend gave no out-of-sample gain** (leave-one-origin-out: mean blend RMSE 0.501 vs best single 0.401, beat the better single in 0/27 folds). Removed from the pipeline; kept as `ml/outputs/arima.md`.
- **GPR** (Gaussian Process Regression, commit `cd07868`, harness once in repo-root `ml-gpr/`): per-station GP with a single `time_hours` feature. Fits the training span (train R² ≈ 0.95) but a time-only GP reverts to its training mean outside it, so it **extrapolates catastrophically** — frozen pre-2026 → 2026 daily means: Tindwara **RMSE 7.82 / R² −401**, Banda **2.13 / R² −7.6** — with bands inflating until the 90% interval covers everything (PICP 1.00, uninformative). Fit is also O(n³) (~6 min/station at 3k rows), no path to 600 pooled stations. Rejected.


### Verify after editing
```bash
cd ml
venv/bin/python -m unittest discover -s tests    # 237-test suite, 3 pre-existing Ollama-chat failures (AppTest gated via AQUIS_APPTEST=1)
venv/bin/python gate_check.py                    # baseline checks (frozen RMSE/CV/coverage/eff-N/ACF + trajectory; latest: GATE PASS 13/13)
```

### Dashboard (`app.py`, Streamlit, 4 active pages)
**Assistant** (station-pinned LLM, default page) · Correlation (mode/metric pickers + recharge-lag curve) ·
**Forecast (single dark-theme trajectory v2 card: 120 genuine 6-hourly points, bright q05/q95 band, confidence, direction)** ·
Sources (manifest quality, association method, soil/LULC status). Legacy pages (Overview, Drivers, Stations, Model, Fleet) were removed from `app_pages/`; Verification (`app_pages/verification.py`) can be wired in when wanted.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | — | PostgreSQL connection string |
| PORT | 3000 | Backend port |
| NODE_ENV | development | Environment |
| ML_SERVICE_URL | http://localhost:5000 | Python ML service URL (Flask api.py; proxy via Node gateway) |
| ML_TIMEOUT_MS | 60000 | ML request timeout |
| BACKEND_URL | http://localhost:3000 | Data/CSV paths used by some ML helper scripts |
| AQUIS_OLLAMA_MODEL | llama3.2:3b | Assistant model override |
| LULC_STATISTICS_API_KEY | — | ISRO Bhuvan LULC API key (Sources page) |

---

## API Reference (backend → app)

Endpoint reference: see [`ml/README.md`](ml/README.md) for the ML service (`ml/api.py`) and the backend gateway. Summary:

- **Node `:3000`** endpoints: stations, telemetry, assessments (CGWB), trends (Mann-Kendall + Sen's slope), ml-data, data-quality, ingestion.
- **ML Flask `:5000` (`ml/api.py`, live, v3.4.0)** — JSON endpoints: root `/` (lists available routes), `/health`, `/stations` (with district/q filtering, lat/lon per station), `/stations/<slug>` (per-station facts, no LLM), `/stations/<slug>/series` (6-hourly GWL + driver points for relation charts), `/stations/<slug>/alerts` (notification-ready zone + reasons + top drivers, no LLM), `/fleet/alerts` (fleet-wide zones for bell-icon/notification center), `/districts` (district list with water-scarcity status), `/districts/<name>` (district detail + station list), `/fleet/recovery` (recovery/decline ranking), `/forecast/<slug>` (trajectory v2), `/assistant/chat` (Ollama-backed LLM). CORS enabled; the Node gateway can proxy via `ML_SERVICE_URL`.

---

## Testing
```bash
cd back-end
npm test
```
Backend tests cover classification, statistics, telemetry utilities, ML gateway, and app configuration.

ML validation: stdlib `unittest` suite in `ml/tests/` (177 data-gated tests, no pytest; the full Forecast-page AppTest is gated behind `AQUIS_APPTEST=1` because the trajectory engine is slow) + `ml/gate_check.py` regression gate.

---

## Key Data Facts

- NWDP telemetry: ~7.4M records across India (backend scope); ML archive = **5.3M records / 1,353 UP stations, 6-hourly, 2021-01-01 → 2026-09-05**; **600 stations / 29 districts** quality in as pooled-model anchors. *(Coverage is telemetry-limited — UP GW monitoring is mostly manual/quarterly; see [Scope & data coverage](#scope--data-coverage).)*
- Assessment years: 2016-2017 … 2025-2026 · 154-column CentralReport Excel (3-level merged headers).
- CGWB classification: Safe (<70%) / Semi-Critical (70–90%) / Critical (90–100%) / Over-Exploited (>100%).
- Forecast uncertainty: **calibrated 90% interval** (q05/q95, `k=1.0`), median half-width ≈1.03 m; 30-day outlooks are **directional**, not exact.
- Forecast surface: the Forecast page shows the **trajectory v2** forecast — **120 genuine 6-hourly steps to +30 d** per station (30-day RMSE 2.106 m, beats both the direct benchmark 2.132 m and persistence 2.151 m on the honest non-overlap 2026 set; calibrated to 90% coverage per horizon). Every point carries an evidence-based confidence label; drivers are Open-Meteo (days 1–16) + CWC river forecasts where available.
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