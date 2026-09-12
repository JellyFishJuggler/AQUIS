# AQUIS ML / System Specification

**Purpose:** Single source of truth for building the complete AQUIS application
(frontend + backend + mobile) on top of the existing Python ML module (`ml/`).
Every name, schema, JSON shape and function signature below is taken from the
actual implemented code and produced artifacts — not a generic design.

**Scope:** Uttar Pradesh groundwater telemetry forecasting (pooled XGBoost,
30-day horizon, 6-hourly grid) + correlation/impact analysis + fleet scan +
station-locked LLM assistant.

**Live today:**
- Streamlit dashboard — `ml/app.py` (port 8501) — 4 pages in nav: Assistant (default) · Correlation · Forecast · Sources
- Headless Flask ML API — `ml/api.py` (port 5000) — `/`, `/health`, `/stations`, `/forecast/<slug>`, `/assistant/chat`
- Node.js Express API — `back-end/` (port 3000), proxying `/ml/*` to the Flask service via `mlGateway.js`
- Refresh daemon — `ml/refresh/` — 6-hourly fetch→inference→publish grid on `{01,07,13,19}:00 IST` (see `ml/README.md` "Refresh pipeline")

---
---

## 1. Complete end-to-end application flow

```
┌──────────────────────── NWIC / NWDP (CKAN datastore_search) ─────────────────────────┐
│  GWL telemetry 6-hourly (UPGW) · Rainfall AWS · Temp/AWMS · Humidity · Solar ·        │
│  Wind speed · Wind direction · Pressure · River level · Canal level · Discharges     │
└──────────────┬───────────────────────────────────────────────────────────────────────┘
               │ 00_probe.py → 01_select.py → 02_fetch_selected.py (resume-safe pulls)
               ▼
        common.parquet ── cleaned 6-hourly GWL archive 2021-01-01 → 2026-09-05
        (5.3M+ rows / 1,353 UP stations)                    │  [source of truth]
               │ 04_align.py / 04b_align_6h.py (IDW rain, nearest weather)
               ▼
        table.parquet (daily station×day) · table_6h.parquet (6h grid, 2.99M rows / 600 stations)
               │ 06_features.py (drop spikes → regime-shift drop → feature/target build)
               ▼
        data/features/{train,val,test}.parquet
               │ 06_train.py (pooled XGBoost + Ridge) · 11_quantile.py (q05/q50/q95)
               ▼
        models/{xgb_multihorizon, linear_multihorizon, xgb_q05/q50/q95}.joblib
        models/{feature_config, quantile_calibration, model_metadata}.json
               │ 07_evaluate.py · validation/* · 11_fleet.py · 12_diagnostics.py
               ▼
        outputs/{model_metrics, honest_metrics, feature_importance, correlation_report,
                 predictions_2026.parquet, fleet_forecast.csv, ...}
               │
     ┌─────────┴───────────────────────────────────────────────────────────────┐
     ▼                                                                          ▼
Streamlit app (ml/app.py, :8501)                                        Express API (:3000)
4 pages: Assistant (default) · Correlation · Forecast · Sources          /stations /telemetry /assessments
(Overview/Drivers/Stations/Model/Fleet deactivated;                    /trends /ml-data /data-quality
 Verification page exists but unwired)                                      
     │                                                                  │
     └────────── ML client (React/Next or mobile) ◄──────────────────────┘
                   ↵ ml/api.py (:5000) Flask: /health /stations /forecast/<slug> /assistant/chat
```

**Production data flow** (repo README): NWDP Telemetry API + CGWB Assessment
Excel → PostgreSQL → Statistical Analysis + ML → Express APIs → Frontend.

**Key constraint:** UP groundwater monitoring is *mostly manual* (quarterly),
AQUIS ingests only the **telemetry** feed → 1,353 stations / 34 districts known
to the archive → 600 stations / 29 districts "still alive in 2026" qualify as
model anchors. A station without a live 2026 telemetry stream is not
forecast-able.

---

## 2. ML features

One feature row per **(station, 6h-slot)**. `gwl` is the **anchor** — used to
build the 30-day change but never predicted. Target = `GWL(t+120) − GWL(t)`
(30 days × 4 slots/day = 120 six-hour steps).

Full feature list = `ml/models/feature_config.json` → `num_cols` (30) +
2 ordinal columns (`st_id`, `dist_id`) = **32 model inputs** for XGBoost.

| # | Feature | Group | Meaning / formula |
|---|---|---|---|
| 1 | `gwl` | anchor | Current reading (day-0 anchor, never target) |
| 2 | `lag1` | GWL lags | GWL 1 slot earlier (=6 h) |
| 3 | `lag4` | GWL lags | GWL 4 slots earlier (=1 day) |
| 4 | `lag8` | GWL lags | GWL 8 slots earlier (=2 days) |
| 5 | `lag28` | GWL lags | GWL 28 slots earlier (=7 days) |
| 6 | `lag120` | GWL lags | GWL 120 slots earlier (=30 days) |
| 7 | `gwl_roll7_mean` | GWL rolling | Rolling mean over 28 slots (=7 d), min_periods=1 |
| 8 | `gwl_roll7_std` | GWL rolling | Rolling std over 28 slots (=7 d) |
| 9 | `gwl_roll30_mean` | GWL rolling | Rolling mean over 120 slots (=30 d) |
| 10 | `gwl_roll30_std` | GWL rolling | Rolling std over 120 slots (=30 d) |
| 11 | `rain_1d` | driver | 6h-rain sum over 4 slots (=1 d), mm |
| 12 | `rain_7d` | driver | 6h-rain sum over 28 slots (=7 d), mm |
| 13 | `rain_30d` | driver | 6h-rain sum over 120 slots (=30 d), mm |
| 14 | `temp` | driver | Air temperature, °C |
| 15 | `temp_7d` | driver | 28-slot rolling mean of `temp` |
| 16 | `humidity` | driver | Relative humidity, % |
| 17 | `humidity_7d` | driver | 28-slot rolling mean of `humidity` |
| 18 | `solar` | driver | Solar radiation (auto-detected unit) |
| 19 | `wind_speed` | driver | Wind speed (auto-detected unit) |
| 20 | `pressure` | driver | Atmospheric pressure (auto-detected unit) |
| 21 | `river_level` | driver | River water level, m |
| 22 | `canal_level` | driver | Canal water level, m |
| 23 | `year` | calendar | Calendar year (2012→2026) |
| 24 | `month_sin` | calendar | `sin(2π·month/12)` |
| 25 | `month_cos` | calendar | `cos(2π·month/12)` |
| 26 | `doy_sin` | calendar | `sin(2π·doy/365.25)` |
| 27 | `doy_cos` | calendar | `cos(2π·doy/365.25)` |
| 28 | `hour_sin` | calendar | `sin(2π·hour/24)` (fractional hour) |
| 29 | `hour_cos` | calendar | `cos(2π·hour/24)` |
| 30 | `monsoon` | calendar | `doy − 152` (positive during monsoon ≈ Jun 1 onward) |
| 31 | `st_id` | ordinal | Stable integer per station (sorted by name, train/val/test shared) |
| 32 | `dist_id` | ordinal | Stable integer per district (alphabetical) |

> **Gated/experimental features** (fetched, shown in app, but disabled in the
> shipped model — see §6 note and README items 6–9): `sand_0_30`, `silt_0_30`,
> `clay_0_30`, `bdod_0_30`, `sand_60_100`, `silt_60_100`, `clay_60_100`,
> `bdod_60_100` (ISRIC SoilGrids v2, `ENABLE_SOIL_FEATURES = False`),
> `lulc_*_pct` (9 class shares, ISRO Bhuvan 50K, kept out via comment),
> `rain_exp_30d` (`ENABLE_RAIN_EXP = False`), CGWB extraction
> `ann_gw_draft_mcm` / `stage_of_development_pct` (merged when available).

**Feature ranking by XGBoost gain** (`outputs/feature_importance.csv`):

| Rank | Feature | gain | freq |
|---|---|---|---|
| 1 | monsoon | 25381.8 | 149 |
| 2 | gwl_roll7_std | 21670.7 | 172 |
| 3 | doy_sin | 14347.3 | 265 |
| 4 | gwl_roll30_std | 10718.3 | 369 |
| 5 | year | 10004.1 | 103 |
| 6 | doy_cos | 7990.7 | 226 |
| 7 | lag1 | 7534.9 | 85 |
| 8 | gwl | 7330.6 | 333 |
| 9 | month_sin | 7019.1 | 50 |
| 10 | hour_sin | 6426.8 | 72 |
| … | gwl_roll30_mean | 6329.9 | 121 |
| … | lag120 | 5723.3 | 184 |
| … | lag28 | 5447.7 | 160 |
| … | lag4 | 5311.8 | 81 |
| … | rain_30d | 2589.3 | 201 |
| … | st_id | 2239.0 | 243 |
| … | dist_id | 1748.2 | 167 |

---

## 3. Input parameters — name, meaning, unit, datatype, range, status

### 3.1 Model features (per 6h row, dtype float32 in feature tables)

| Param | Meaning | Unit | dtype | Typical range | Required |
|---|---|---|---|---|---|
| `gwl` | Groundwater level (depth below ground, negative convention) | m | float32 | −120 … +70 (per-station trimmed) | yes |
| `lag1/4/8/28/120` | past GWL at 6h/1d/2d/7d/30d lag | m | float32 | same as `gwl` | yes (NaN allowed in-lag) |
| `gwl_roll7_mean/std` | 7-d rolling mean/std of GWL | m | float32 | −120 … +70 | yes |
| `gwl_roll30_mean/std` | 30-d rolling mean/std of GWL | m | float32 | −120 … +70 | yes |
| `rain_1d/7d/30d` | cumulative rainfall | mm | float32 | 0 … ~2000 (30d) | yes |
| `temp`, `temp_7d` | air temperature | °C | float32 | −10 … 55 | yes |
| `humidity`, `humidity_7d` | relative humidity | % | float32 | 0 … 100 | yes |
| `solar` | solar radiation | auto | float32 | varies | yes |
| `wind_speed` | wind speed | auto (km/h or m/s) | float32 | varies | yes |
| `pressure` | atmospheric pressure | hPa | float32 | varies | yes |
| `river_level` | river water level | m | float32 | varies | yes (sparse) |
| `canal_level` | canal water level | m | float32 | varies | yes (sparse) |
| `year` | calendar year | yr | int (float32) | 2012–2026 | yes |
| `month_sin/cos`, `doy_sin/cos`, `hour_sin/cos` | cyclic encodings | dimensionless | float32 | −1 … 1 | yes |
| `monsoon` | day-of-year − 152 | day | float32 | −152 … 213 | yes |
| `st_id` | station ordinal | dim-less | int32 | 0 … 548 (549 stations) | yes (XGBoost only) |
| `dist_id` | district ordinal | dim-less | int32 | 0 … 28 (29 districts) | yes (XGBoost only) |
| `Station` | station name (raw string) | — | str | see `feature_config.stations` | yes (Ridge OHE) |
| `District` | district name (raw string) | — | str | see `feature_config.districts` | yes (Ridge OHE) |

> **Missing-data policy:** XGBoost natively handles NaN (default via `tree_method
> hist`); Ridge uses sklearn `SimpleImputer(strategy="median")` inside the numeric
> branch of a `ColumnTransformer`. Features are causal (lags only) — never future.

### 3.2 Runtime request inputs (API / app)

| Param | Meaning | dtype | Range / example | Required |
|---|---|---|---|---|
| `station` (or slug) | station identifier | str | `"Ramchhitoni Sahawar (UP-011)"` | yes (forecast/chat) |
| `district` | district filter | str | `"KAUSHAMBI"`, `"GORAKHPUR"` | optional |
| `model` | model choice for the page | str | `"xgboost"` \| `"ridge"` | optional (default xgboost) |
| `question` | free-text for chatbot | str | non-empty | yes (chat) |
| `history` | prior chat turns | list[dict] | `[{"role":"user","content":…}]` | optional |
| `model` (chat) | LLM model override | str | `"llama3.2:3b"` | optional |
| `days` | forecast horizon (API, fixed at 30) | int | 7–90, default 30 | optional |

---

## 4. Dataset & schema

### 4.1 `data/processed/common.parquet` — raw cleaned GWL archive (source of truth)
~5.3M rows / 1,353 UP stations / 2021-01-01 → 2026-09-05. Key columns:
`Station, District, Tehsil, Block, Latitude, Longitude, RL_MSL,
Data Acquisition Time, Groundwater Level Telemetry 6 Hourly (meter)`.
Refreshed incrementally by `13_refresh_nwic.py`.

### 4.2 `data/aligned/table_6h.parquet` — 6-hourly station×slot (model input)
2,990,000+ rows / 600 stations. Columns: `Station, District, time, gwl, rain,
temp, humidity, solar, wind_speed, wind_direction, pressure, river_level,
canal_level` (+ `dist_km` variants where relevant). `time` is 6-hourly
(00/06/12/18); drivers re-binned to the same 6h steps.

### 4.3 `data/aligned/table.parquet` — daily station×day (correlation/overview)
`Station, District, date, gwl, <drivers>` collapsed daily.

### 4.4 `data/features/{train,val,test}.parquet` — feature/target tables
Column order (from `KEEP` in `06_features.py`):
`Station, District, time, date, horizon, target, target_d, st_id, dist_id` +
30 `NUM_COLS` (all float32). `target = GWL(t+120)`, `target_d = target − gwl`.

### 4.5 `outputs/predictions_2026.parquet` — 2026 test predictions (app chart)
Schema (verified live):

| Col | dtype | Purpose |
|---|---|---|
| `Station` | str | station name |
| `District` | str | district |
| `date` | datetime64 | prediction date |
| `time` | datetime64 | 6h slot |
| `horizon` | int64 | 30 |
| `target` | float64 | actual GWL(t+120) |
| `gwl` | float32 | anchor |
| `feat_days` | int8 | feature-history days used |
| `xgb` | float32 | XGBoost level prediction |
| `ridge` | float64 | Ridge level prediction |
| `persist` | float32 | persistence (anchor) |
| `clim` | float64 | climatology |
| `err_xgb`, `err_ridge` | float64 | level errors |
| `q05_lvl`, `q95_lvl` | float32 | calibrated 90% interval (level) |
| `step_idx` | int64 | index within 120-step window |
| `window_id` | int64 | non-overlap window id |
| `stride` | bool | True = independent (non-overlap) row |

### 4.6 `data/meta/selected_gwl_stations.csv` — anchor registry (§11)
`Station, District, Tehsil, Block, Latitude, Longitude, n_2026_months,
n_2026_records, first_2026, last_record, full26`.

### 4.7 `data/meta/selected_districts.json` — the 29 live districts
`["AGRA","AZAMGARH","BAGHPAT","BANDA","BAREILLY","BUDAUN","CHITRAKOOT",
"DEORIA","ETAH","FATEHPUR","GHAZIABAD","GORAKHPUR","HAMIRPUR","JALAUN",
"JHANSI","KAUSHAMBI","KANSIRAM NAGAR","LALITPUR","MAHOBA","MEERUT",
"MIRZAPUR","MUZAFFARNAGAR","PRATAPGARH","SAHARANPUR","SANT RAVIDAS NAGAR",
"SHRAWASTI","SIDDHARTHNAGAR","SONBHADRA","VARANASI"]`

---

## 5. Data sources & preprocessing

### 5.1 Sources (all via NWIC/NDWP CKAN `datastore_search`)
IDs live in `ml/config.py` (`SOURCES`); authoritative source map is
`back-end/db/api/api.txt`. Fetch done per-district with resume markers.

| Source | archive res id (uuid) | live res id (uuid) | value hint | agg |
|---|---|---|---|---|
| gwl | `84bfda45-…-f7a93ee57522` | `31c66a49-…-a0ea0d9c8b0c` | Groundwater Level Telemetry 6 Hourly (meter) | mean |
| rainfall | `2c0805b7-…-cf7bc1ecd148` | `46e9afa0-…-49e0b5b1b6d3` | Telemetry Hourly Rainfall (mm) | sum |
| temperature | `feb54802-…-8311ae04df0e` | `c2c7b5fa-…-18eb5297f7b0` | Air Temperature Telemetry Hourly (AoC) | mean |
| river_level | `9672a7e9-…-a4128792d070` | `94b5345e-…-d93cdede2025` | River Water Level Telemetry Hourly (meter) | mean |
| humidity | `14a3cfa8-…-c64693725779` | `0fcb6700-…-41397f417c8c` | Telemetry Hourly Relative Humidity (%) | mean |
| solar | `6128c3ff-…-5b2ef492e700` | `c1666f7a-…-39ec192ad004` | auto-detect | mean |
| wind_speed | `fc1314ef-…-5b1ec7a90037` | `beb454fb-…-bf8b4b282efa` | auto-detect | mean |
| wind_direction | `b9acc862-…-57d3580a75b6` | `9e8132bc-…-af05-adbac43f7b24` | auto-detect | mean |
| pressure | `32a8fd55-…-ac24c447f28a` | `3525c93c-…-3033fd21a645` | auto-detect | mean |
| canal_level | `ea65ac74-…-f927633def4d` | `8075a3b8-…-ba7a-94b193f3936b` | auto-detect | mean |
| canal_discharge | `a6158de8-…-6cf5296f161a` | `14a80ee5-…-1f5857b843ff` | — | mean |
| 5× reservoir discharge | listed in config | listed in config | — | mean |

> `canal_discharge` and reservoir discharges: **0 rows within selected districts**
> (barrages outside them) — confirmed-empty, not re-probed per rerun.

Covariates in the unioned selected districts (from `data/meta/manifest.json`):
gwl 2,279,048 rows/600 st · rainfall 42,745/50 st/8 dist · temperature
107,950/5 · humidity 118,947/6 · solar 109,555/6 · wind_speed 122,527/6 ·
pressure 110,949/4 · river_level 295,302/6/2 · canal_level 88,089/17/1.

### 5.2 Spatial association
- **Rainfall:** IDW (`1/d²`) over 3 nearest gauges within `RAIN_RADIUS_KM=150`.
- **Weather/river/canal:** nearest gauge within radius (`WEATHER_RADIUS_KM=150`,
  `RIVER_RADIUS_KM=100`, `CANAL_RADIUS_KM=100`); river/canal add `dist_km`.
- **River (no published coords):** district-level proxy (mean across district
  gauges), flagged `dist_km=NaN`.

### 5.3 Preprocessing / hygiene steps
1. **Sentinel & spike drop** (`03`): `|GWL|>500 m` and implausible records
   dropped; per-station **0.5–99.5 % quantile trim**. Manifest counts e.g. gwl
   dropped 20,273 rows by caps; rainfall max 149.5 mm/h (cap `MAX_RAIN_MM_H`
   = 150 — a real gauge once reported 2.24e6 mm/h).
2. **Daily/6h median:** daily-median GWL for correlation; 6h-median per slot
   (`04b`) for modelling.
3. **Spike mask** (`06_features.drop_gwl_spikes`): per-station robust mask,
   drop readings `> 30 × (1.4826 × MAD)` from the median → set to NaN.
4. **Regime-shift drop** (`06_features.flag_regime_shift`): stations whose
   per-year median jumps `> 15 m` from the 2021–2024 baseline are dropped
   wholesale (telemetry datum errors, e.g. −6 → −97 m overnight wells; 51
   dropped on the 6h grid).
5. **Coverage selection** (`01_select`): full-2026 rule —
   first 2026 record ≤ `2026-01-15`, ≥ 8 of 9 months (Jan–Sep 2026) have a
   record, last record ≥ `2026-08-01` → **600 of 1,353 stations / 29 districts**.
6. **Feature/target build** (`06_features`): lags/rollings always causal,
   per-sample 6h bins, cyclic calendar encodings.
7. Static ISRIC soil per-station texture (`07_soil`) and Bhuvan LULC district
   stats (`08_lulc`) are fetched/app-visible but **gated out** of the model.
   CFSv2 seasonal rain (`09_cfs_rain.py`) was **removed** (NCEI gridded services
   returned S3-403; Open-Meteo replaced the driver feed).

---

## 6. ML models — detail

### 6.1 Task
Pooled **global** model (one model, not per-station). Grid 6-hourly, horizon
**30 days = 120 steps**. Target **delta**: `y = GWL(t+120) − GWL(t)`. Level =
`anchor + delta`. Persistence baseline = 0-change. `gwl` at t is anchor — never a
target.

### 6.2 Models & hyperparameters (from `06_train.py` + `11_quantile.py`)

**XGBoost regressor** (`xgb_multihorizon.joblib`, `reg:squarederror`):
- `n_estimators=1500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=20`, `tree_method="hist"`,
  `objective=reg:squarederror`, `eval_metric=rmse`, `random_state=42`, `n_jobs=6`
- Early stopping on validation split, `early_stopping_rounds=60`
- Trained: **best_iteration n_trees = 15** (per `feature_config.json`)
- Input matrix: `NUM_COLS(30) + st_id + dist_id`
- Train rows *sampled 1 slot/day per station* (`cumcount % 4 == 0`) for memory
  (Ridge dense solve 7.5 GB→1.9 GB)

**Ridge** (`linear_multihorizon.joblib`, ridge-as-model):
- `RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))` → chosen `alpha = 0.1`
- Pipeline: `ColumnTransformer` = numeric branch
  (`SimpleImputer(median)` → `StandardScaler`) over `NUM_COLS` + categorical
  `OneHotEncoder(handle_unknown="ignore")` over `["Station","District"]`

**Quantile forecasters** (`xgb_q05/q50/q95.joblib`, `reg:quantileerror`):
- Same pooled XGBoost recipe at α = 0.05 / 0.50 / 0.95, output delta quantiles.
- Calibrated empirically on non-overlap windows →
  `quantile_calibration.json` (below).

### 6.3 Split & sizes (`feature_config.json` / `model_metadata.json`)
- Train: `< 2025-10-01` — val: `2025-10-01 .. 2025-12-31` — test: `>= 2026-01-01`
- `train_rows` 1,629,574 · `val_rows` 11,647 · `test_rows` 378,554
  (model_metadata: train_val_rows 1,641,221, n_stations 549, n_districts 29)
- `trained_at: 2026-09-09 17:03:49`

### 6.4 Validation metrics (all 30-day horizon)

**Head-to-head on 2026 held-out — `outputs/model_metrics.csv` (level basis):**

| Model | RMSE | MAE | R² | ±0.5 m | ±1.0 m |
|---|---|---|---|---|---|
| **xgboost** | **2.242** | 0.704 | 0.901 | 67.2% | 85.1% |
| persistence | 2.290 | 0.744 | 0.897 | 63.9% | 83.2% |
| ridge | 2.456 | 1.260 | 0.881 | 33.3% | 57.9% |
| climatology | 3.572 | 1.657 | 0.748 | 32.7% | 54.6% |

**Honest (overlap-aware) re-score — `outputs/honest_metrics.csv` & `overlap.json`:**
378,554 raw test rows are ~99% overlapping (120-step windows on a 6h grid →
`effective_N_ratio = 0.0089` → **3,371 independent stride windows**; effective
sample ≈ 6,635 from lag-1 ACF 0.858).

| Model | raw RMSE | **stride RMSE** | window med RMSE |
|---|---|---|---|
| xgb | 2.2418 | **2.3389** | 0.4548 |
| ridge | 2.4559 | 2.5123 | 0.9315 |
| persist | 2.2896 | 2.3824 | 0.4932 |
| clim | 3.5716 | 3.6225 | 1.0291 |

> XGBoost beats persistence by **+1.8%** on independent windows (raw +2.1%).
> Per-row RMSE deltas below ~±0.05 m are within noise.

**Spatial CV — `outputs/spatial_cv_metrics.csv` + `spatial_cv_summary.json`:**
leave-one-block-out station retraining, 5 folds: fold RMSE mean **1.9733** /
median **1.8191** (blocks: 140/129/163/106/62 stations). Temporal 2026 holdout,
not spatial leakage, is the binding constraint. Residual: Moran's I = 0.042
(p=0.051), Geary's C = 1.083 (ns) — no strong spatial clustering.

**Quantile calibration — `quantile_calibration.json`:**
`target_coverage_stride 0.8`, `widen_factor_k 1.0`, `coverage_stride 0.908`
(raw 0.9003), `half_width_median_m 1.025`, `half_width_p90_m 2.575`,
`half_width_mean_m 1.376`. Anchor basis: `level = today's GWL + quantile(delta 30d)`.

**Regression gate (`ml/gate_check.py`, 12 checks; latest run GATE PASS 10/10):**
honest stride RMSE 2.339 · spatial CV mean 1.973 / median 1.819 ·
coverage 0.908 · half-width median 1.025 / p90 2.575 · eff-N 6,635 · lag1 ACF 0.858 ·
trajectory 30d 2.106 (promoted) · +30d calibrated coverage 0.90.
(The 2 recursive-model checks skip — the `backtest_6h_*` artifacts were removed.)

---

## 7. Prediction input/output schema

### 7.1 Forward forecast — `_model.forward_forecast(station: str) -> dict`
Rebuilds the feature frame for the station (via `06_features.build_full`),
takes the **last row** (latest time), scores with XGBoost (32 cols), Ridge
(30 num + Station + District), and the q05/q50/q95 boosters; quantile
interval calibrated with `k = widen_factor_k`.

Real example (live output):

```json
{
  "station": "Ramchhitoni Sahawar (UP-011)",
  "date_from": "2026-09-05 00:00:00",
  "date_to": "2026-10-05 00:00:00",
  "anchor": -2.841,
  "pred_xgb": 0.121,
  "pred_ridge": -0.58,
  "xgb_level": -2.72,
  "ridge_level": -3.421,
  "q05_level": -4.266,
  "q50_level": -2.691,
  "q95_level": -1.039,
  "widen_k": 1.0,
  "band_half": 1.613,
  "station_stride_rmse": 0.314
}
```

| Field | Meaning |
|---|---|
| `date_from` | anchor timestamp (latest 6h slot) |
| `date_to` | `date_from + 30 days` |
| `anchor` | latest observed GWL (KPI, never predicted) |
| `pred_xgb` / `pred_ridge` | predicted 30-day *change* (delta) |
| `xgb_level` / `ridge_level` | level = anchor + change |
| `q05_level / q50_level / q95_level` | calibrated 90% interval on level |
| `band_half` | `(q95 − q05) / 2` |
| `station_stride_rmse` | honest non-overlap RMSE for this station (if in test set) |

Returns `{}` if the station has no usable feature frame.

### 7.2 2026 backtest loader — `_model.load_predictions()` (returns DataFrame)
Reads only 10 columns from `predictions_2026.parquet` (TTL 24 h):
`Station, District, date, time, target, gwl, xgb, ridge, q05_lvl, q95_lvl`.

### 7.3 Feature config — `load_feature_config()` (feature_config.json) keys
`model_type, target, horizons[30], num_cols[30], stations[549],
districts[29], train_rows, val_rows, test_rows,
xgb:{n_trees,val_rmse_delta,val_rmse_level}, ridge:{alpha,val_rmse_delta,...},
trained_at`.

### 7.4 Model metadata — `model_metadata.json` keys
`generated_by, architecture, target, horizons, n_stations, n_districts,
train_val_rows, test_rows, val_rmse_level_m:{xgb,ridge}, split, trained_at,
generated_at, source_archive, validated:{quantile_calibration, honest_metrics,
spatial_cv, spatial_cv_summary}`.

---

## 8. API-style specification (per ML functionality)

Two services:
- **Node `:3000`** (live): `/stations`, `/telemetry`, `/assessments`,
  `/trends`, `/ml-data`, `/data-quality`, `/ingestion` — full list in §14.
- **Python `:5000` Flask** (`ml/api.py`, live) — surface below. The Node
  gateway (`back-end/services/mlGateway.js`) forwards path+query to
  `ML_SERVICE_URL`. Status codes: 200 / 400 / 404 / 422 / 500 / 502 / 503
  (`{ "error": …, "detail": … }`).

| Endpoint | Method | Input | Response (shape) | Backed by |
|---|---|---|---|---|
| `/` | GET | — | `{service, version, status, endpoints, usage}` | hardcoded index |
| `/health` | GET | — | `{status, service, version, stations, dataset_last, ollama:{server,models,model}, forecast_model}` | `_data()`, `_index()`, `_assistant.ollama_status()` |
| `/stations` | GET | `?district=&q=&limit=` | `{count, stations:[{station, district, slug, latitude, longitude, last_ts,…}]}` | `selected_gwl_stations.csv`, `_index()` |
| `/forecast/<slug>` | GET | slug (URL-encoded) | 120×6h trajectory `time/q05/q50/q95` + confidence + drivers | `_trajectory.trajectory_forecast()` |
| `/assistant/chat` | POST | `{question, station?, model?}` | `{answer, facts, station}` | `_assistant.StationAssistant.answer()` |

**Planned additions (not yet implemented):** `/stations/:slug` (per-station
facts), `/districts`, `/models`, `/fleet/forecasts`, `/fleet/recovery`,
`/fleet/scan`.

**Slugs vs names:** API uses lowercase hyphenated slugs
(e.g. `ashadha-prathmik-vidyalaya`); always fetch the slug
from `GET /stations` — never hand-type it. The exact station name also
resolves as a fallback when URL-encoded. Internally the model uses the raw
station name string (e.g. `"Ramchhitoni Sahawar (UP-011)"`).

**Errors:**
- 400 — e.g. chat without `question`, or `days` out of 7–90
- 404 — unknown slug / no model trained / fleet snapshot missing
- 500 — internal Python error
- 502 — ML service down/`ML_SERVICE_URL` unreachable
- 503 — assistant unavailable (Ollama not running)
- Legacy `/ml/forecast/:stationId`, `/ml/risk/:unitId`, `/ml/anomalies` — **deprecated**, DB registry only.

---

## 9. Forecasting complete logic

1. **Anchor** = latest observed GWL for the station (`date_from` = last 6h slot;
   never predicted). If no feature frame → empty response.
2. **Feature rebuild** — same path as training: filter `table_6h` by station →
   `06_features.drop_gwl_spikes` → `build_full(keep_na=True)` → last row.
3. **Score**:
   - `pred_xgb = xgb_b.predict(last[feature_names])` — delta
   - `pred_ridge = linear_b.predict(last[NUM_COLS + [Station, District]])`
   - quantiles: `med = anchor + q50`, `lo = anchor + q05`, `hi = anchor + q95`;
     calibrated `q05 = med − k(med−lo)`, `q95 = med + k(hi−med)`, `k=1.0`
   - `band_half = (q95 − q05)/2`
4. **Interval semantics:** 90% calibrated interval (`coverage_stride = 0.911 ≥
   0.80`). Median half-width ≈ **1.03 m** (p90 ≈ 2.58 m).
5. **Level outputs** = `anchor + delta`. **Persistence** = unchanged anchor.
6. **Fleet scan (`11_fleet.py`)**: eligible = ≥ 2000 obs and ≥ 730-day span
   (596 stations). Per station `forward_forecast()`; headline = `q50 − anchor`;
   keep `change_pooled_m` as secondary (pooled delta saturates to identical
   extremes on stations with empty recent lag/driver columns). Thresholds:
   decline ≤ −0.3 m, high decline ≤ −0.6 m, recovery ≥ +0.3 m. Categories:
   `decline (high) / decline / stable / recovering / unknown / unreliable`.
   **Quality gate → `unreliable`** if: NaN anchor, `|change| > 12 m`,
   predicted level outside `[obs_min − 25, obs_max + 25]`, or anchor older than
   45 days. Snapshot result (2026-09): **502 scored**, median Δ **+0.32 m**,
   decline **0** / recovering **259** at ±0.3 m — nearly all changes inside
   their 90% interval (post-monsoon recovery).

---

## 10. Groundwater impact analysis

Available in `ml/outputs/`:

1. **Feature importance (XGBoost gain)** — `feature_importance.csv`
   (`feature, gain, freq`). Top: `monsoon`, `gwl_roll7_std`, `doy_sin`,
   `gwl_roll30_std`, `year`, `doy_cos`, `lag1`, `gwl`. (§2 table.)
2. **Ridge coefficients** — `feature_coefficients.csv` (`feature, coef,
   abs_coef`). Top |coef|: `num__gwl −9.73`, `num__lag4 −0.35`,
   `num__lag8 −0.30`.
3. **Correlation report** — `correlation_report.csv`
   (`driver, metric(spearman/pearson), mode(raw/deseason/diff), corr,
   stations_ok`). Headlines (|spearman| raw): `river_level +0.409`
   (deseason 0.168) · `pressure +0.199` · `rain_30d −0.199` ·
   `canal_level −0.146` · `humidity +0.085` · `solar −0.055` ·
   `temp −0.041` · `wind_speed −0.095`. **Recharge lag:** best
   GWL-vs-rainfall lag ≈ **22 days** (Pearson median +0.136, min 180 valid
   days) — `lag_curves.csv`.
4. **Ablation** — `ablation.csv` (drop-one-group retrain): rain (+0.03),
   weather (+0.05), calendar (+0.05) add marginal RMSE when dropped; river/canal
   (−0.005) and GWL lags (≈0) neutral given anchor + calendar; all within
   overlap noise ~±0.05 m.
5. **Diagnostics** — `diagnostics.json` + `diagnostics_permutation.csv`
   (stride subset, n=3,371):
   - **VIF:** nothing > 10; `constant_excluded: [year]`; no collinearity.
   - **Permutation importance (Δ RMSE):** ~all ≈ 0 (top `gwl_roll30_std`
     +0.0026, `temp_7d` +0.0016; most below zero-noise).
   - **Sensitivity (OAT, 2.5→97.5 pct):** max swing `doy_cos` ≈ **0.385 m**,
     then `gwl` 0.148, `gwl_roll30_std` 0.146, `monsoon` 0.132 — season/level/
     monsoon carry the largest single-driver influence.
6. **SHAP/explainability:** **not implemented** in this codebase (no `shap`
   dependency). Grounded explanation today = XGBoost gain + Ridge coef +
   permutation importance + OAT sensitivity + Spearman correlation. Adding SHAP
   is recommended as a follow-up on the stride subset (32 features →
   `shap.TreeExplainer`).

---

## 11. Geography hierarchy & IDs

**India → Uttar Pradesh → District → Tehsil → Block → Station.**

- The archive has 1,353 UP stations across 34 of UP's ~75 districts; only
  **600 stations / 29 districts** pass the full-2026 rule and enter the model.
- `selected_gwl_stations.csv` gives per station: `Station, District, Tehsil,
  Block, Latitude, Longitude, n_2026_months, n_2026_records, first_2026,
  last_record, full26`. Note: Tehsil/Block are `-` (not published) for most
  telemetry stations.
- **Model ordinals** — `st_id` (sorted station names, 0..548) and `dist_id`
  (sorted district names, 0..28) are fit on the combined train/val/test station
  set and reused in the app so forward-forecast shares byte-identical columns.
- Backend stations table keeps the same identity columns (`state,
  district, rehsil… tehsil, block, village, river, basin, latitude, longitude,
  rl_msl`, LGD codes).
- Two endpoint-visible geo keys: station **slug** (planned API) and raw
  **Station name** (models/fleet).

---

## 12. Streamlit app — live features & purpose

`ml/app.py` (port 8501), 4 pages via `st.navigation` (+ Verification, unwired):

| Page | File | Purpose / content |
|---|---|---|
| **Assistant** | `app_pages/assistant.py` | Station-pinned chat (default page); instant-facts panel works even if Ollama is down; Ollama status probe |
| **Correlation** | `app_pages/correlation.py` | Driver×metric×mode pickers, recharge-lag curve (GWL vs rainfall), static-feature (soil/LULC) vs GWL summary |
| **Forecast** | `app_pages/forecast.py` | Single dark-theme trajectory v2 card: 120 genuine 6-hourly points, observed tail, "Forecast starts" boundary, bright q05/q95 band (no dot markers), direction banner, 6 metric cards, freshness caption |
| **Sources** | `app_pages/data_sources.py` | Per-source manifest quality (rows/stations/districts/dropped), empty-source table, static hydrogeology (soil/LULC status), association method + radius |
| **Verification** *(not in nav)* | `app_pages/verification.py` | Realised-forecast quality from `data/refresh/verification_summary.json` + sign-accuracy ledger; wire into `app.py` to enable |

> The legacy explorer pages (Overview, Drivers, Stations, Model, Fleet) were
> **removed** from `app_pages/`; their pipeline outputs (`fleet_forecast.csv`,
> `model_metrics.csv`, honest/ablation/diagnostics) are still produced by the
> numbered scripts and served through `_model.py` loaders, the Flask API and
> the assistant facts.

---

## 13. Chatbot — flow, inputs, outputs, prompt

Implemented in `ml/_assistant.py` (class `StationAssistant`), page
`app_pages/assistant.py`.

**Design principle:** station-locked, deterministic facts only — the LLM phrases
precomputed numbers and never invents values.

**Flow:**
1. `facts(station)` → builds a fact dict (§13.1).
2. If no telemetry → `{"answer": "No telemetry found for station 'X'.", "facts": {}, "station": X}`.
3. Else `_build_prompt(question, facts, history)` →
4. `_invoke_llm` → `ollama.Client(timeout=None).chat(model, messages=[{role:"user", content:prompt}], options={"temperature": 0.2})` →
5. returns `{"answer", "facts", "station"}`.

**13.1 Facts dict (station level):**
`level:"station", station, district, last, last_date,
min, max, span, n_obs, outliers, change_7d, change_30d, change_60d,
change_180d, district_median, district_n_stations,
forecast:{anchor, day30_pred, change_30d_pred, direction("expected rise"|"expected decline"),
plausible, obs_min, obs_max, band_half, q05_level, q95_level,
station_stride_rmse, high_uncertainty}`.

**13.2 Prompt template (verbatim structure):**
```
You are the AQUIS groundwater assistant. Answer the user's question using
ONLY the given observed facts. No speculation. Be concise (max ~6 lines).
Use metres (m) for levels/changes. State dates where known.
Never refuse or say data is unavailable…
[CONVERSATION history, "always trust the LATEST FACTS" if numbers differ]
FACTS:
station: …  district: …
latest level: <last> m on <last_date>
range: <min> to <max> m (span <span> m, <n_obs> observations)
[note: N outlier(s) excluded]
level change: 7d=… | 30d=… | 60d=… | 180d=… m
model forecast (+30 d): level=<day30_pred> m, change 30d=<…> m (<direction>); 90% band +/- <band_half> m (anchor <anchor> m); q05/q95 interval …m..…m
[note: model UNSTABLE … or high-uncertainty station … where flagged]
district median (<district>): <median> m across <n> stations
QUESTION: <question>
ANSWER:
```

**13.3 Quality notes fed to prompt:**
- `plausible = |change| ≤ 12 m AND day30 ∈ [obs_min−25, obs_max+25]`; else a
  "runaway" note tells the LLM to trust only observed facts.
- `high_uncertainty` when `station_stride_rmse > 2 × fleet median
  (xgb_stride_rmse`.

**13.4 LLM setup:** local Ollama, default model `llama3.2:3b`
(`ollama list` probe first); override env `AQUIS_OLLAMA_MODEL` (from repo
`.env` or environment). Server down → facts panel still renders; page shows an
Ollama status warning.

---

## 14. Database design (PostgreSQL) + suggested tables

### 14.1 Existing backend schema (`back-end/db/schema.sql`, 9 tables)
- `ingestion_runs` — `id, source, source_ref, status(enum running/completed/
  failed/partial), started_at, finished_at, records_seen/inserted/updated/
  rejected/duplicates, error_count, error_summary JSONB, metadata JSONB`
- `stations` — `id, external_station_id UNIQUE, station_name, agency, state,
  state_lgd_code, district, district_lgd_code, tehsil, block, village, river,
  basin, tributary, subtributary, sub_subtributary, local_river, latitude,
  longitude, rl_msl, first_observed_at, last_observed_at, observation_count,
  timestamps` + indexes on state/district/agency/coords/external
- `telemetry_observations` — `id, station_id FK→stations, observed_at,
  groundwater_level, source(default 'nwdp_api'), source_record_id,
  ingestion_run_id FK, created_at`; unique `(station_id, observed_at)`
- `assessment_units` / `assessment_records` — CGWB assessment geography +
  multi-year records
- `data_quality` — per-station/telemetry/assessment quality issues
- `model_metadata` / `model_outputs` — DB model registry (legacy; the live ML
  stack uses `ml/models/*.json` instead)
- `groundwater_data` — legacy GWL table

### 14.2 Suggested additions for the ML-backed app
| Table/collection | Key fields | Relation |
|---|---|---|
| `ml_stations` (mirror of `selected_gwl_stations.csv`) | station(PK name), district, tehsil, block, lat, lon, n_2026_months, n_2026_records, first_2026, last_record, full26 | 1:N telemetry |
| `ml_features` (optional, parquet mirrors) | station, time(6h), 30 features, st_id, dist_id | FK station |
| `ml_predictions` (from `predictions_2026.parquet`) | station, time, horizon, target, gwl, xgb, ridge, persist, clim, err_*, q05_lvl, q95_lvl, step_idx, window_id, stride | FK station |
| `ml_fleet_snapshot` (from `fleet_forecast.csv`) | station, district, anchor, date_from, age_days, anchor_valid, xgb_level, q05/q50/q95_level, change_30d_m, change_pooled_m, band_half_m, plausible, category | FK station |
| `ml_quantile_calibration` / `ml_model_meta` | mirrors `quantile_calibration.json` / `model_metadata.json` | singleton |
| `chat_logs` | id, station, question, answer, facts JSONB, model, created_at | optional FK station |
| `alert_rules` / `alerts` (future) | station, threshold_m, band_ref, status, created_at, triggered_at | FK station |

---

## 15. ML project folder / file structure (with purpose)

```
ml/
├─ app.py                        Streamlit entry — st.navigation of 4 pages, :8501
├─ app_pages/{assistant,correlation,forecast,data_sources}.py   live pages (§12)
│             + verification.py (refresh verification UI, not in nav)
├─ config.py                     SOURCES (archive/live resource ids), radii, coverage
│                                rules (FULL26_*), caps (MAX_RAIN_MM_H), field names, paths
├─ 00_probe.py                   probe every NWIC resource (archive + live) → meta/probe.json
├─ 01_select.py                  full-2026 coverage selection → 600 st / 29 dist
├─ 02_fetch_selected.py          per-district driver fetch, resume-safe
├─ 03_merge_normalize.py         plausibility caps → raw/<source>_norm.parquet + manifest
├─ 04_align.py                   daily station×day, spatial association (IDW rain, nearest weather)
├─ 04b_align_6h.py               6-hourly station×slot table (model track)
├─ 05_correlate.py               correlation report + recharge-lag curves/charts
├─ 06_features.py                spike mask, regime-shift drop, feature/target build → features/*.parquet
├─ 06_train.py                   pooled XGBoost + Ridge, deltas, joblib + feature_config/metadata
├─ 07_evaluate.py                4-way benchmark (XGB/Ridge/persistence/climatology) + honest re-score
├─ 07_soil.py                    ISRIC SoilGrids v2 per-station texture fetch
├─ 08_lulc.py                    ISRO Bhuvan LULC 50K district stats
├─ 10_ablate.py                  drop-group ablation sweep → outputs/ablation.csv
├─ 11_quantile.py                pooled q05/q50/q95 + empirical coverage calibration
├─ 11_fleet.py                   whole-fleet 30-day scan + recovery ranking
├─ 12_diagnostics.py             VIF + permutation importance + OAT sensitivity
├─ 13_refresh_nwic.py            incremental NWIC refresh (+ --retrain / --deploy-check)
├─ _model.py                     model/output loaders + forward_forecast() (§7)
├─ _utils.py                     shared loaders: manifest, selected CSV, table/table_6h,
│                                station_recency() (cached groupby), label maps
├─ _assistant.py                 StationAssistant facts + prompt + Ollama client (§13)
├─ _soil.py / _lulc.py           static soil / LULC loaders
├─ validation/                   P0 honesty suite: spatial_cv.py, overlap.py,
│                                residual_acf.py, spatial_residual.py, _spatial_folds.py
├─ gate_check.py                 12-check regression gate (run after edits; latest GATE PASS 10/10)
├─ tests/                        stdlib unittest (data-gated, no pytest)
├─ MODEL_CARD.md                 model lifecycle card
├─ README.md                     module doc + decisions log
├─ data/processed/common.parquet cleaned 6-hourly GWL archive (source of truth)
├─ data/selected|raw|aligned|meta|features|soil   intermediate + meta dirs
├─ models/*.joblib + *.json      xgb/linear/q05/q50/q95 + feature_config/quantile_calibration/model_metadata
└─ outputs/*.csv|json|parquet    evaluation, fleet, diagnostics, predictions
```

---

## 16. Dependencies, env vars, external APIs

### 16.1 requirements.txt (repo root — used by Streamlit Cloud)
```
pandas  numpy  matplotlib  scikit-learn  requests  flask  flask-cors
scipy  xgboost  joblib  streamlit  plotly
```
App additionally imports at runtime (present in the ml venv): `altair`
(Streamlit bundles it), `ollama` (Python SDK, via `/opt/venv` — **not in root
requirements; add `ollama` if deploying the assistant**). Python version:
**3.14** (ml venv, pipelines + app tested); backend needs Node 18+/PostgreSQL 14+.

### 16.2 Environment variables
| Variable | Default | Used by |
|---|---|---|
| `AQUIS_OLLAMA_MODEL` | `llama3.2:3b` | assistant model override (`_assistant.py`) |
| `LULC_STATISTICS_API_KEY` | — | Bhuvan LULC district stats (`08_lulc.py`) |
| `LULC_AOI_WISE_API_KEY` | (in example) | Bhuvan AOI endpoint (unused) |
| `DATABASE_URL` | — | backend PostgreSQL |
| `PORT` | 3000 | backend |
| `ML_SERVICE_URL` | `http://localhost:5000` | backend ML gateway (planned) |
| `ML_TIMEOUT_MS` | 60000 | backend ML timeout |
| `BACKEND_URL` | `http://localhost:3000` | some ML helper scripts |

Env file pattern: plain `KEY=VALUE` lines in repo `.env`, parsed by
`_assistant.read_env(repo)` / `08_lulc` (same pattern).

### 16.3 External APIs
- **NWIC/NWDP CKAN `datastore_search`** — all telemetry (GWL + drivers); ids in
  `config.py`; authoritative map `back-end/db/api/api.txt`.
- **ISRO Bhuvan LULC statistics** (`bhuvan-app1.nrsc.gov.in/api`) — district
  class shares; key in `.env`.
- **ISRIC SoilGrids v2 REST** — per-station sand/silt/clay/bdod; background fetch.
- **Ollama (local)** — `llama3.2:3b` ~2 GB, `ollama serve`; not an external SaaS.
- **CGWB assessments** — Excel files (imported via backend), manual/quarterly
  UPGW dataset documented in `MODEL_CARD.md` (not ingested into ML).
- NCEI CFSv2 seasonal rain was **removed** (`09_cfs_rain.py` deleted — NCEI
  gridded services returned S3-403); Open-Meteo is the driver feed.

---

## 17. Validation rules & edge cases

### 17.1 Data-quality gates
- Coverage rule: first 2026 ≤ 2026-01-15, ≥ 8/9 months (Jan–Sep) present, last
  ≥ 2026-08-01 → else station excluded (can't forecast).
- Sentinels: `|GWL| > 500 m` dropped; per-station 0.5–99.5% trim.
- Rainfall hourly cap 150 mm/h; caps per-source via manifest.
- `drop_gwl_spikes`: > 30×(1.4826·MAD) deviation → NaN.
- Regime shift: per-year median jump > 15 m from 2021-24 baseline → station dropped.
- Fleet quality gate → `unreliable`: NaN anchor, `|Δ|>12 m`, level outside
  `[obs_min−25, obs_max+25]`, or anchor age > 45 days.

### 17.2 Edge cases / errors (exact behaviours)
- `forward_forecast(station)` with no feature frame → returns `{}` (app shows
  "No feature frame for this station.").
- Assistant with unknown station → `answer: "No telemetry found for station 'X'."`.
- Fleet with no snapshot → page shows warning "run `ml/11_fleet.py` first" + `st.stop()`.
- Ollama down → chat raises; page surfaces status; facts panel still renders.
- Missing driver columns → feature builder skips those windows (conditionals),
  XGBoost handles NaN, Ridge median-imputes.
- Static soil/lulc incomplete → skipped with a printed note; app shows "no
  profile mapped" caption.
- Ridge on new station not in OneHotEncoder → `handle_unknown="ignore"` (all-zero
  category vector).
- Stale model vs live NWIC → `13_refresh_nwic.py --deploy-check` verifies the
  loaded build is 2026-current.

### 17.3 Honest-validation rules (don't overclaim)
- Never quote per-row RMSE deltas < ~±0.05 m as signal (overlap noise).
- Use stride (non-overlap) metrics for claims: XGBoost 2.339 vs persistence
  2.382 m, coverage 0.908 on 3,371 windows, eff-N 6,635.
- Significance of a fleet move: |Δ| must exceed the 90% band half-width.

---

## 18. Sample request–response examples

### 18.1 Forecast (app/API)
```json
// GET /ml/forecast/Ramchhitoni%20Sahawar%20(UP-011)?days=30
{
  "station": "Ramchhitoni Sahawar (UP-011)",
  "date_from": "2026-09-05T00:00:00",
  "date_to": "2026-10-05T00:00:00",
  "anchor": -2.841,
  "pred_xgb": 0.121,
  "pred_ridge": -0.58,
  "xgb_level": -2.72,
  "ridge_level": -3.421,
  "q05_level": -4.266,
  "q50_level": -2.691,
  "q95_level": -1.039,
  "widen_k": 1.0,
  "band_half": 1.613,
  "station_stride_rmse": 0.314
}
```

### 18.2 Fleet scan
```json
// GET /ml/fleet/forecasts
{
  "horizon_days": 30,
  "decline_threshold_m": 0.3,
  "eligible_stations": 596,
  "scanned_stations": 502,
  "generated_at": "2026-09-09 …",
  "stations": [
    {"station": "Mahgaon (UP-031)", "district": "KAUSHAMBI", "anchor": -17.171,
     "date_from": "2026-09-03 18:00:00", "age_days": 6, "anchor_valid": true,
     "xgb_level": -16.881, "q05_level": -18.637, "q50_level": -16.689,
     "q95_level": -11.989, "change_30d_m": 0.482, "change_pooled_m": 0.29,
     "band_half_m": 3.324, "plausible": true, "category": "recovering"}
  ]
}
```

### 18.3 Impact analysis
```json
// GET /ml/models  (subset)
{
  "n_stations": 549, "n_districts": 29,
  "val_rmse_level_m": {"xgb": 1.2748, "ridge": 1.708},
  "validated": {
    "honest_metrics": [{"model": "xgb", "basis_stride_rmse": 2.3389, "n_windows": 3371}],
    "spatial_cv_summary": {"fold_rmse_mean": 1.9733, "fold_rmse_median": 1.8191},
    "quantile_calibration": {"coverage_stride": 0.908, "half_width_median_m": 1.025}
  }
}
```

### 18.4 Chatbot
```json
// POST /ml/assistant/chat
{ "question": "Is level rising?", "station": "Ramchhitoni Sahawar (UP-011)" }

// 200
{
  "answer": "Latest level is -2.84 m on 2026-09-05. Over 180 days the level changed -0.06 m. The model expects a rise of about +0.12 m (to -2.72 m) over the next 30 days, inside a 90% band of +/- 1.61 m.",
  "facts": {
    "station": "Ramchhitoni Sahawar (UP-011)", "district": "KANSIRAM NAGAR",
    "last": -2.841, "last_date": "2026-09-05", "min": …, "max": …,
    "change_7d": …, "change_30d": …, "change_60d": …, "change_180d": …,
    "district_median": …, "district_n_stations": …,
    "forecast": { "anchor": -2.841, "day30_pred": -2.72, "change_30d_pred": 0.121,
      "direction": "expected rise", "plausible": true, "band_half": 1.613,
      "q05_level": -4.266, "q95_level": -1.039, "station_stride_rmse": 0.314,
      "high_uncertainty": false }
  },
  "station": "Ramchhitoni Sahawar (UP-011)"
}
```

---

## 19. Sample dataset (real rows)

**`outputs/predictions_2026.parquet`** (station ASHADHA PRATHMIK VIDYALAYA, KAUSHAMBI):

| date | time | target | gwl | xgb | ridge | q05_lvl | q95_lvl |
|---|---|---|---|---|---|---|---|
| 2026-01-01 | 2026-01-01 00:00:00 | -6.267 | -6.411 | -6.624 | -7.896 | -7.835 | -4.558 |
| 2026-01-01 | 2026-01-01 06:00:00 | -6.414 | -6.165 | -6.378 | -7.720 | -7.589 | -4.673 |
| 2026-01-01 | 2026-01-01 12:00:00 | -6.156 | -6.284 | -6.497 | -7.721 | -7.708 | -4.856 |

**`outputs/fleet_forecast.csv`** (subset):

| station | district | anchor | date_from | age_days | anchor_valid | xgb_level | q05_level | q50_level | q95_level | change_30d_m | change_pooled_m | band_half_m | plausible | category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ramchhitoni Sahawar (UP-011) | KANSIRAM NAGAR | -2.841 | 2026-09-05 00:00:00 | 4 | True | -2.72 | -4.266 | -2.691 | -1.039 | 0.15 | 0.121 | 1.613 | True | stable |
| Mahgaon (UP-031) | KAUSHAMBI | -17.171 | 2026-09-03 18:00:00 | 6 | True | -16.881 | -18.637 | -16.689 | -11.989 | 0.482 | 0.29 | 3.324 | True | recovering |

**`data/meta/selected_gwl_stations.csv`** (subset):

| Station | District | Tehsil | Block | Latitude | Longitude | n_2026_months | n_2026_records | first_2026 | last_record | full26 |
|---|---|---|---|---|---|---|---|---|---|---|
| ASHADHA PRATHMIK VIDYALAYA | KAUSHAMBI | - | - | 25.441184 | 81.392245 | 9 | 822 | 2026-01-01 00:00:00 | 2026-09-05 18:00:00 | True |
| Aagan Wadi Kendra Chandni | JALAUN | - | - | 25.929918 | 79.141493 | 9 | 900 | 2026-01-02 00:00:00 | 2026-09-05 18:00:00 | True |

**`outputs/station_summary.csv`** (per-station honest RMSE):

| station | district | n_windows | stride_rows | raw_rows | xgb_raw_rmse | xgb_stride_rmse | ridge_raw_rmse | ridge_stride_rmse | persist_raw_rmse | persist_stride_rmse | clim_raw_rmse | clim_stride_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ASHADHA PRATHMIK VIDYALAYA | KAUSHAMBI | 7 | 7 | 754 | 0.457 | 0.597 | 1.786 | 1.698 | 0.437 | 0.556 | 2.786 | 2.853 |

---

## 20. Feature-to-API mapping (build plan)

| UI / applet | Model feature(s) | Endpoint(s) (planned) | Data/artifact today |
|---|---|---|---|
| Health banner / boot check | — | `GET /health` | `model_metadata.json`, `quantile_calibration.json`, `_assistant.ollama_status()` |
| Station picker / watchlist (recency-sorted) | `gwl` anchor | `GET /stations?q=&district=` | `selected_gwl_stations.csv` + `station_recency()` (table_6h groupby) |
| Station detail KPIs (level, min/max, changes 7/30/60/180 d) | `gwl`, lags | `GET /stations/:slug`, `GET /telemetry/:stationId` | `table_6h.parquet`, backend `telemetry_observations` |
| Price-style GWL chart + 90% band (Bollinger-like) | `q05_level/q95_level` | `GET /forecast/:slug?days=30` | `predictions_2026.parquet` (q05_lvl/q95_lvl) + `forward_forecast()` |
| Forecast outlook card / alerts (±band) | `anchor + delta`, `band_half` | `GET /forecast/:slug` | `forward_forecast()` |
| Indicator overlays (7d/30d MA, rolling-std vol band, rain volume) | `gwl_roll7/30_mean/std`, `rain_1d/7d/30d` | `GET /stations/:slug` (extended) | `table_6h.parquet` |
| Seasonal climatology overlay + monsoon shading | `doy_sin/cos`, `month_sin/cos`, `monsoon` | `GET /telemetry/:stationId?years=` | `table.parquet` + seasonal median |
| Weather fundamentals panel | `temp(+7d)`, `humidity(+7d)`, `solar`, `wind_speed`, `pressure`, `river_level`, `canal_level` | `GET /stations/:slug?drivers=` | `table_6h.parquet`, `correlation_report.csv` |
| District / UP heatmap | `dist_id`, fleet Δ | `GET /districts`, `GET /fleet/recovery` | `fleet_district.csv`, `selected_districts.json` |
| District station comparison (sidebar) | q50 change, band | `GET /fleet/scan?district=` | `fleet_forecast.csv` |
| Fleet movers (top gainers/losers) | `change_30d_m`, `category` | `GET /fleet/forecasts` | `fleet_forecast.csv` / `fleet_forecast_snapshot.json` |
| Model / metrics screen | — | `GET /models` | `model_metrics.csv`, `honest_metrics.csv`, `feature_importance.csv`, `feature_coefficients.csv`, `spatial_cv_summary.json`, `diagnostics.json` |
| Impact analysis tab | gain/coef/permutation/OAT | `GET /models` (+`?analysis=importance`) | `feature_importance.csv`, `feature_coefficients.csv`, `diagnostics.json`, `correlation_report.csv` |
| Assistant chat | station facts + forecast | `POST /assistant/chat` | `_assistant.StationAssistant` |
| Sources / provenance page | — | `GET /models` | `manifest.json`, `config.py` SOURCES |

---

*Generated from actual code (`ml/06_features.py`, `ml/06_train.py`,
`ml/_model.py`, `ml/_assistant.py`, `ml/11_fleet.py`, `ml/config.py`), trained
artifacts (`ml/models/*.json`), and produced outputs (`ml/outputs/*`).*