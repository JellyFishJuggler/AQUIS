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

The forecast surface is a dedicated **trajectory v2** engine (`ml/_trajectory.py`, XGBoost; `ml/_dl.py`, GRU): a genuine **120-step, 6-hourly, 30-day** forecast per station (no interpolation — one real model output per 6 h step), with per-horizon q05/q50/q95 and an **evidence-based confidence label** (HIGH / DIRECTIONAL / LOW). Drivers ahead of the anchor are legitimate forecasts — **Open-Meteo** weather (days 1–16) with climatology beyond, and **CWC 3-day river forecasts** where the district is covered. The Forecast page renders this as a single dark-theme dashboard card (bright q05/q95 band, no dot markers).

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
- Therefore, in this repo: **1,353 stations / 34 districts = "everything the telemetry archive has"**, and **600 stations / 29 districts = "the ones still alive in 2026"** (selected by `01_select.py`, ≥8 of 9 Jan–Sep 2026 months with a reading). Neither number is a scope preference — the data decides.
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
├─ 00_probe.py … 13_refresh_nwic.py   # classic pooled pipeline
├─ 14_6h_features.py             # causal 6h feature frames (used by 30_traj_datasets)
├─ 16_future_drivers.py          # driver-climatology refresh + future-driver bridge
├─ 18_cwc_river_forecast.py      # CWC 3-day river-forecast fetch (per district)
├─ 20_openmeteo_fetch.py         # Open-Meteo daily weather (365d history + 16d forecast)
├─ 30_traj_datasets.py … 33_traj_reliability.py   # trajectory v2 (XGBoost) train/backtest/reliability
├─ 34_train_traj_dl.py           # trajectory v2 DL: sequence GRU + horizon-conditioned quantile heads
├─ 35_traj_paired_ci.py          # DL-vs-XGB paired station-bootstrap CI (per horizon)
├─ 36_train_pooled_dl.py         # pooled 30-day DL: daily GRU + 3 pinball heads (3-seed ensemble)
├─ 37_backtest_pooled_dl.py      # honest 2026-stride backtest for the pooled DL
├─ _model.py _trajectory.py _dl.py _traj_dispatch.py  # forward/trajectory serving (XGB + GRU dispatch)
├─ _assistant.py _utils.py _soil.py _lulc.py _hydrology.py _association.py _spatial.py  # shared libs
├─ api.py                        # Flask HTTP API (JSON; /health, /stations, /forecast/<slug>, /assistant/chat)
├─ snapshot.py                   # matplotlib PNG export (used by Snapshot tests only)
├─ refresh/                      # refresh pipeline: config, features, inference,
│                                #   model_update, publish, schedule, scheduler, state, verification, CLI
├─ data/
│  ├─ processed/common.parquet   # cleaned 6-hourly GWL archive 2021→2026 (source of truth)
│  ├─ aligned/      table.parquet (daily) + table_6h.parquet (6h grid, ~3M rows)
│  ├─ cfs/          openmeteo_weather_daily.parquet + river_forecast_cwc.parquet
│  ├─ meta/         associations, manifest, probe, station flags, climatology
│  ├─ features_6h/  train/val/test parquet (causal 6h feature frames; built by 14_6h_features)
│  └─ soil/         ISRIC SoilGrids + ISRO LULC (one-off fetch)
├─ experiments/                 # baselines (arima_compare/combine), validation, diagnosis
├─ models/                      # joblib (pooled XGB) + traj_*.json (traj XGB) + *_dl.pt/*_dl_config.json (GRU)
├─ outputs/                     # correlation, benchmark, honest metrics, fleet, diagnostics, DL eval
├─ tests/                       # stdlib unittest suite (no pytest dependency; incl. DL parity/dispatch)
└─ gate_check.py                # regression gate (frozen baselines + trajectory)
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
10_ablate.py          drop-group ablation sweep
11_quantile.py        pooled q05/q50/q95 + empirical coverage calibration
11_fleet.py           whole-fleet 30-day forecast scan + recovery snapshot
12_diagnostics.py     VIF + permutation importance + OAT sensitivity
13_refresh_nwic.py    incremental NWIC refresh (+ optional retrain / deploy sync)
14_6h_features.py     causal 6h feature frames (used by 30_traj_datasets for traj v2)
16_future_drivers.py  driver-climatology refresh + future-driver bridge
18_cwc_river_forecast.py  CWC 3-day river forecast fetch (per district, resume-safe)
20_openmeteo_fetch.py     Open-Meteo daily weather: 365d history + 16d forecast
30_traj_datasets.py       trajectory v2 feature frames (train/val, exact-time targets)
31_train_traj.py          trajectory shared multi-horizon XGBoost (q05/q50/q95)
32_backtest_traj.py       honest non-overlap trajectory backtest + per-horizon calibration
33_traj_reliability.py    reliability bucket rules + evidence weights
34_train_traj_dl.py       trajectory v2 DL (sequence GRU + horizon-conditioned quantile heads)
35_traj_paired_ci.py      DL-vs-XGB paired bootstrap CI, per horizon (reads 32_'s rows)
36_train_pooled_dl.py     pooled 30-day DL (daily GRU + 3 pinball heads, 3-seed ensemble)
37_backtest_pooled_dl.py  honest 2026-stride pooled-DL eval vs incumbent XGB
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

### Models — XGBoost + quantile GRU on two surfaces

Production serves **two families on two surfaces**, and each surface is now served by the **GRU** with XGBoost kept as an automatic fallback/benchmark (`_dl.py` / `_model.py`, dispatched by `_traj_dispatch.py`; backend flags in `data/refresh/runtime.json`):

- **Surface (a) — trajectory v2** (Forecast page + `/forecast/<slug>`): a genuine **120-step, 6-hourly, 30-day** per-horizon forecast.
  - **XGBoost** — direct multi-horizon shared booster, horizon `h ∈ 1..120` an input feature, q05/q50/q95 heads (`31_train_traj.py`).
  - **GRU (served)** — sequence encoder over the last 120 observed 6-h rows, horizon-conditioned quantile heads, pinball loss (`34_train_traj_dl.py` → `models/traj_dl.pt`).
- **Surface (b) — pooled 30-day** (assistant, fleet scan, +30 d benchmark): the single delta `GWL(t+120) − GWL(t)`.
  - **XGBoost** — `reg:squarederror`, early stopping on the Oct–Dec 2025 split (`06_train.py` → `models/xgb_multihorizon.joblib`).
  - **GRU (served)** — daily GRU over a 60-day window + 3 pinball quantile heads (q05/q50/q95), 3-seed ensemble (`36_train_pooled_dl.py` → `models/pooled_dl.pt`).
- **Grid + horizon:** 6-hourly track. Today's level is the anchor feature — never predicted.
- **Baselines:** Ridge (linear), persistence (0-change) and per-station day-of-year climatology (`07_evaluate.py`).
- **Quantile uncertainty:** native quantile heads calibrated into 90% intervals (XGB `11_quantile.py`; GRU per-horizon `*_dl_calibration.json`).

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