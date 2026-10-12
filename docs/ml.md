# AQUIS ML — consolidated specification

Consolidated from the former `docs/ml-*.md` files. Sections: 1. System specification, 2. Full-stack build specification, 3. Module ↔ gateway contract, 4. Trajectory v2 spec, 5. rain_api train/serve mismatch.


---

# Part 1 — System specification

<!-- source: docs/ml-system-spec.md -->

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

> The frames also carry a **causal hydrology-informed block** (`HYDRO_COLS` in
> `06_features.py`, built by `_hydrology.py`) and `target_time`. Both are
> **carried, not yet trained**: they are present in
> `data/features/{train,val,test}.parquet` and in the serving row dicts, but the
> frozen artifacts' `num_cols` deliberately exclude them, so the shipped models
> are unchanged. Wiring them in requires a deliberate retrain (the refresh gate
> refuses any candidate whose `feature_cols` differ from production). See §2.1.

### 2.1 Hydrology-informed block (`HYDRO_COLS` — carried, not yet trained)

Built once in `_hydrology.py` and shared verbatim by the batch builder
(`06_features.build`) and the serving path
(`refresh.future_drivers.build_step_features`), so training and inference cannot
drift. Every column is **causal** (trailing windows only, per-row station
history) and every windowed value is paired with an explicit coverage column —
never a silently shortened window.

| # | Feature | Group | Meaning / formula |
|---|---|---|---|
| H1 | `gwl_dev7` | GWL state | `gwl − gwl_roll7_mean` (m) |
| H2 | `gwl_dev30` | GWL state | `gwl − gwl_roll30_mean` (m) |
| H3 | `gwl_slope7` | GWL trend | Theil-Sen slope of the per-day median observed GWL over a trailing 7 d window (m/day) |
| H4 | `gwl_slope30` | GWL trend | Same, trailing 30 d |
| H5 | `gwl_slope90` | GWL trend | Same, trailing 90 d |
| H6 | `gwl_obs7` | GWL coverage | Fraction of the 7 d window whose days held ≥1 **observed** (not carried-forward) reading |
| H7 | `gwl_obs30` | GWL coverage | Same, 30 d |
| H8 | `gwl_obs90` | GWL coverage | Same, 90 d |
| H9 | `gwl_imputed` | GWL provenance | 1 = the row's GWL is a forwarded value, 0 = observed, NaN = no reading |
| H10 | `rain_api7` | rain memory | Exponential antecedent precipitation index, τ = 7 d (mm) |
| H11 | `rain_api22` | rain memory | API, τ = 22 d — the corpus-measured recharge lag |
| H12 | `rain_api60` | rain memory | API, τ = 60 d |
| H13 | `rain_obs7` | rain coverage | Fraction of the τ = 7 d window **backed by an actual rain observation** (fixed denominator; missing slots and all future forecast/climatology slots count as unobserved) |
| H14 | `rain_obs22` | rain coverage | Same, τ = 22 d |
| H15 | `rain_obs60` | rain coverage | Same, τ = 60 d |

Definitions and the causality contract are in the `_hydrology.py` module
docstring. Key points:

- **API recursion**: `API_t = λ·API_{t−1} + rain_t` on the 6-hour grid with
  `λ = exp(−(6/24)/τ)`; zero state before the station's first observed rain, NaN
  before it (no fabricated warm-up). A missing rain slot contributes zero
  accumulation while the decay continues.
- **Rain provenance — `rain_api*` and `rain_obs*` mean different things.**
  `rain_api*` is built from the **numeric rainfall forcing** available at the
  step: observations where they exist, and forecast/climatology values for future
  steps, because that is the best rainfall knowledge at that step. `rain_obs*` is
  a **coverage** column and counts only slots backed by an **actual observation**
  — a missing historical slot and every future forecast/climatology slot count as
  unobserved. The two are therefore deliberately decoupled: masking changes
  coverage only, never the API value. Coverage uses the fixed denominator `τ·4`
  slots (7 d = 28, 22 d = 88, 60 d = 240), so a short or sparse history reads < 1
  rather than pretending to be complete. In the serving path this means
  `rain_obs*` *decays* across the forecast horizon instead of reading ~1.0 on
  rain that has not been observed. Rainfall has **no** forward-fill or
  interpolation anywhere in the pipeline (`03_merge_normalize` drops NaN rows;
  `04_align`/`04b_align_6h` sum-or-mean over gauges and left-join, leaving
  unobserved slots as NaN), so in batch frames a finite value *is* an observation
  — `06_features.build` builds this mask explicitly rather than inferring it, so
  the meaning survives any imputation step added upstream later.
- **Slopes** are computed on per-day medians of *observed* readings first, then
  as the median of all pairwise day-to-day slopes, so one telemetry spike cannot
  tilt the trend. A slope needs ≥3 observed days and ≥50% window coverage, else
  NaN.
- **GWL provenance** comes from the aligned table's `gwl_obs` column
  (`04b_align_6h.gwl_track` records 1 = reading, 0 = forward-filled, NaN = gap
  beyond the 3-day carry limit). On legacy aligned tables without that column,
  every present reading is treated as observed and `gwl_imputed` is 0.
- Nothing here fits a parameter on validation/test data — the only
  station-specific reference is the observed history at or before the row.

**Serving parity.** `build_step_features` computes all 15 columns for every
horizon step. The `gwl_dev*` / `gwl_slope*` / `gwl_obs*` / `gwl_imputed` values
are **pinned at the anchor** (constant across the 120 steps) because they
describe information available at time *t*; the `rain_api*` / `rain_obs*` values
**vary per step**, because the API recursion is advanced across the forecast
window (observed history + Open-Meteo/climatology rain) while `rain_obs*` coverage
falls as unobserved forecast slots enter its window. The rain recursion runs
over the **full observed history**, not the 128-step driver tail, so a τ = 60 d
kernel is warmed up before the anchor. A caller whose `fnames` omit these columns
— including the frozen production model — is unaffected: the final
`df[base_cols + h_cols]` selection keeps unrequested keys invisible.

> **`rain_api*` has a train/serve definition mismatch — read
> Part 5 (rain_api train/serve mismatch) before training on it.**
> In batch, `rain_api*` is built from rainfall known **at the anchor** and is
> horizon-invariant. In serving, the recursion is **advanced with future
> forecast/climatology rainfall**, so the value varies with *h*. A model fitted on
> one definition would be served the other. Reproducing the serving definition in
> training would require point-in-time forecast rainfall, which **is not
> archived** (the archived forecast JSONs keep GWL quantiles and a `driver_source`
> label, not the numeric driver matrix), and substituting the Open-Meteo *archive*
> (ERA5) would train on realized future rainfall — disallowed. The mismatch is
> **latent today** because `feature_mode = "flat"` never calls the future path.
> `rain_obs*` is unaffected: its semantics are correct in both paths.

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

### 5.1.1 Coverage scope & authoritative sources (model card)

**Scope is determined by the telemetry archive, not by choice.** The model consumes
*only* the NWIC/NWDP real-time feed — `GWL Telemetry (Six Hourly), Uttar Pradesh
Ground Water Department` (UPGW). That feed wires **1,353 stations across 34 of UP's
~75 districts**. UP's remaining groundwater monitoring is **manual/quarterly** and
surfaces as *separate* datasets this pipeline cannot forecast from in near-real-time.

Coverage facts (all verified against the local archive):

- **1,353 stations / 34 districts** = the full extent of the 6-hourly telemetry
  network (2021-01-01 → 2026-09-05, ~5.3M raw records in
  `data/processed/common.parquet`).
- **600 stations / 29 districts** = the subset still "live" in 2026 (`01_select.py`:
  ≥8 of 9 months Jan–Sep 2026 with a reading, first 2026 record ≤ Jan 15, last record
  ≥ Aug 1). 5 districts' stations all went stale.
- **34 ≠ scope target**: districts absent from the archive were probed and return
  `total=0` — e.g. Allahabad/Prayagraj, Amethi, Amroha, Sambhal, G.B. Nagar — even
  under rename aliases (`ml/data/raw/nwic.py`).

Source material (why most UP groundwater monitoring is manual):

| Topic | Link |
|---|---|
| UPGW datasets on NWDP portal (Manual-Quarterly vs Telemetry-Hourly vs Telemetry-Six-Hourly) | https://www.nwdp.nwic.gov.in/organization/upgw |
| UPGW "Ground Water Level (Manual - Quarterly)" dataset page (1991–2020) | https://nwdp.nwic.gov.in/dataset/ground-water-level-manual-quarterly-upgw |
| CGWB monitoring-network page (4×/year manual nationwide; DWLR telemetry is the exception) | https://cgwb.gov.in/en/ground-water-level-monitoring |
| CGWB Guidelines on High-Frequency GW Data (2026): ~84k wells nation-wide, ~26k CGWB (62% dug wells, manual mode), ~39k state manual stations | https://cgwb.gov.in/sites/default/files/2026-02/final_guidelines_on_high_frequency_gw_data.pdf |
| CGWB UP Ground Water Year Book 2022–23 (1,007 UP monitoring wells, 4×/yr) | https://cgwb.gov.in/cgwbpnm/public/uploads/documents/17032384351875462524file.pdf |

**Implication:** a station without a live 2026 telemetry stream is not forecast-able
here; forecasting scope is UP-GWL *telemetry* only, while the backend's ingestion
scope is NWDP-telemetry countrywide + CGWB assessment Excels.

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
- **Target-time purge.** These boundaries apply to the **anchor** timestamp, and
  `split()` additionally removes any row whose **target** is observed inside a
  later split: a train row is kept only if `target_time < 2025-10-01`, and a val
  row only if `target_time < 2026-01-01`. Without this, a 30-day label attached
  to a late-September anchor would be observed inside the validation window (and
  a late-December val label inside the test window), so early stopping would be
  selecting on test-period labels. `06_features.build` writes `target_time` as
  the **actual observed** timestamp of the label row (the same shift position as
  `target`), not `anchor + horizon`, so the purge is correct on a gappy axis.
  The same rule is applied to the trajectory long-frame
  (`30_traj_datasets.main`: train `< VAL_START`, val `< TEST_START`) and to the
  refresh candidate build (`refresh/model_update.build_candidate_longframe`:
  train `< train_cut`, val `< val_end`).
- `train_rows` 1,629,574 · `val_rows` 11,647 · `test_rows` 378,554
  (pre-purge values; the purge removes roughly the last 30 days of anchors from
  train and val, so counts drop by ≈120 rows × n_stations)
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

**Regression gate (`ml/gate_check.py`, 13 checks; latest run GATE PASS 13/13):**
honest stride RMSE 2.339 · spatial CV mean 1.973 / median 1.819 ·
coverage 0.908 · half-width median 1.025 / p90 2.575 · eff-N 6,635 · lag1 ACF 0.858 ·
trajectory 30d 2.106 (promoted) · +30d calibrated coverage 0.90.
(the recursive-model checks skip — the `backtest_6h_*` artifacts were removed.)

### 6.5 Deep-learning models (quantile GRU) — the served engines

Both surfaces are now **served by pooled PyTorch quantile-GRU models** (`_dl.py`,
dispatched via `_model.forward_forecast` / `_traj_dispatch.active_trajectory_forecast`;
backend flags in `data/refresh/runtime.json`). XGBoost stays as an automatic
fallback/benchmark. Architecture = sequence GRU encoder + N pinball quantile heads
(q05/q50/q95), inputs z-scored, outputs sorted so `q05 ≤ q50 ≤ q95`.

**Surface (a) — trajectory DL** (`34_train_traj_dl.py` → `models/traj_dl.pt`,
`traj_dl_config.json`):
- Anchor-centric: one history window per `(station, t)` anchor is encoded **once**,
  then each observed `(h, delta)` pair is scored by horizon-conditioned heads (no
  `5.9M × 120` sequence matrix materialised).
- Sequence = last **120** observed 6-h rows on the station's own axis, left-zero-padded
  + masked when history < 120. Features = the 30 `NUM_COLS`.
- `nn.GRU(hidden=64)` + LayerNorm + horizon-conditioned MLP heads; AdamW lr 2e-3,
  weight-decay 1e-4, ReduceLROnPlateau (factor 0.5, patience 2), up to 40 epochs.
- Same frames/target/split as `31_train_traj.py` (inherits the per-anchor+horizon
  target-time purge).

**Surface (b) — pooled 30-day DL** (`36_train_pooled_dl.py` → `models/pooled_dl.pt`,
`pooled_dl_config.json`):
- Daily GRU (last 6-h slot per station-day), `window_w=60`, `hidden=32`, 39-col
  dense+seasonal frame (GWL slopes/devs, `rain_api*`, `*_obs*`), 3 quantile heads,
  `MAX_EPOCHS 60 / patience 8 / batch 2048 / lr 1e-3`, **3-seed ensemble**
  `{42, 840, 2056}` (seed-mean), `widen_k = 1.1`.
- Split identical to `06_train` (train < 2025-10-01, val Oct–Dec 2025, test 2026).

**Honest evidence:**

| Surface | Artifact | Result |
|---|---|---|
| (b) pooled | `outputs/pooled_dl_eval.json` | 2026 non-overlap stride, n=4,355/511 st: persistence 2.187 · E0-XGB 2.206 · **DL 2.082**; paired ratio vs E0 **0.944 [0.879, 0.972]**, vs persistence 0.952 [0.906, 0.972]; bias −0.030, dir-acc 0.732, coverage_90 0.925 (raw 0.908) |
| (a) trajectory | `outputs/traj_backtest_summary_dl.json` | 19,406 anchors / 150 st, 2025-10 → 2026-08: 30-d RMSE **1.802** vs persistence 1.907 (direct-30d 1.198); dir 0.725; coverage_calibrated 0.894 (fit-half 0.901); `promote_trajectory = false` |
| (a) per-horizon | `outputs/traj_paired_ci_dl_vs_xgb.json` | 2,000 station-bootstrap resamples: DL not better at h6 (1.012) / h12 (1.003); beats XGB at h48 (0.962), h360 (0.985), **h720 (0.990 [0.972, 0.996])**; `dl_beats_xgb_all_horizons = false`, `dl_beats_xgb_30d = true` |

**Caveat:** the win is a 2026-only signal and short horizons remain XGB's edge, so
the promotion gate stays conservative. Serving parity is guarded by
`tests/test_model_dl_parity.py`, `test_trajectory_backend_parity.py`,
`test_trajectory_dispatch.py`.

### 6.6 Other algorithms evaluated (SARIMA, GPR) — rejected

- **SARIMA** (`experiments/baselines/arima_compare.py`; protocol: daily means,
  10 stations × 5 rolling-2026 origins — **not** comparable to the 6-h tables):
  `SARIMAX(1,1,1)×(1,0,1,7)`, RMSE 0.637 vs XGB 0.617, MAE 0.335 vs 0.402,
  PICP90 **0.726 vs 0.968** (under-covers). The ARIMA+XGB **blend gave no
  out-of-sample gain** (leave-one-origin-out: blend 0.501 vs best single 0.401,
  beat the better single in 0/27 folds). Module removed; kept as `outputs/arima.md`.
- **GPR** (commit `cd07868`, harness once in `ml-gpr/`): per-station GP with a single
  `time_hours` feature. Fits the training span (train R² ≈ 0.95) but **extrapolates
  catastrophically** (pre-2026 → 2026: Tindwara RMSE 7.82 / R² −401, Banda 2.13 /
  −7.6, PICP90 1.00 uninformative) and is O(n³). Rejected — interpolation only.

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
├─ experiments/validation/        P0 honesty suite: spatial_cv.py, overlap.py,
│                                residual_acf.py, spatial_residual.py, _spatial_folds.py
├─ gate_check.py                 13-check regression gate (run after edits; latest GATE PASS 13/13)
├─ tests/                        stdlib unittest (data-gated, no pytest)
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
  UPGW dataset documented in §5.1.1 (not ingested into ML).
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


---

# Part 2 — Full-stack build specification (ML-anchored)

<!-- source: docs/ml-build-spec.md -->

# AQUIS Full-Stack Build Specification (ML-Anchored)

> **Purpose:** Single source of truth to build the complete production application
> (frontend + backend + database + ML integration + chatbot) on top of the
> **existing** `ml/` module. Every name, JSON shape, function and metric below is
> taken from the actual code (`ml/*.py`, `back-end/*`, trained artifacts and
> produced outputs). Where something is **not yet implemented** it is explicitly
> labelled `[PLANNED]` vs `[LIVE]`.
>
> Related docs: `Part 3` (module↔gateway
> contract), `Part 1` (ML/deep-dive spec, incl. the model card in §5.1.1/§6.5).

---

## 1. Project Overview

### 1.1 Purpose
AQUIS (**A**quifer **Q**uery and **U**ser **I**nformation **S**ystem) is a
groundwater monitoring platform for Uttar Pradesh. It ingests **6-hourly Digital
Water Level Recorder telemetry** from NWDP/NWIC, cleans it into an archive,
trains a **pooled XGBoost 30-day forecasting model** over 600 stations, evaluates
drivers (rain, weather, river/canal), runs whole-fleet recovery scans, and
explains everything through a **station-locked LLM assistant**.

### 1.2 Problem statement
UP's groundwater is monitored **mostly manually (quarterly)**. The only real-time
slice is a narrow telemetry network (1,353 stations / 34 districts). Officials
and researchers have no single tool that (a) serves live levels, (b) forecasts
30-day change with honest uncertainty, (c) ranks declines/recoveries, and (d)
answers plain-language questions — without inventing numbers.

### 1.3 Target users
| User | Need |
|---|---|
| **Officer (UPGW/CGWB)** | district-level decline alerts, fleet movers, forecast with confidence bands |
| **Researcher** | driver correlation, feature importance, honest validation numbers |
| **Public/student** | plain-language assistant, station detail, trend charts |
| **Developer/integration** | robust REST/JSON contract to build apps on top of |

### 1.4 End-to-end system flow
```
NWIC/NWDP CKAN (datastore_search)                      ← external, free, no auth
   │  00_probe → 01_select → 02_fetch → 03_normalize
   ▼
ml/data/processed/common.parquet  (5.3M rows / 1,353 st / 2021→2026-09-05)
   │  04/04b align (IDW rain, nearest weather, 6h grid)
   ▼
ml/data/aligned/table_6h.parquet  (2.99M rows / 600 st)
   │  06_features (spike-mask, regime-shift, features+target)
   ▼
ml/data/features/{train,val,test}.parquet
   │  06_train (XGBoost+Ridge) · 11_quantile (q05/q50/q95)
   ▼
ml/models/*.joblib + feature_config.json + quantile_calibration.json + model_metadata.json
   │  07_evaluate → validation/* → 11_fleet → 12_diagnostics
   ▼
ml/outputs/{predictions_2026.parquet, fleet_forecast.csv, model_metrics.csv, …}

   ┌──────────────┬──────────────────────────────┐
   ▼              ▼                              ▼
Streamlit app   Express API (:3000) [LIVE]    Flask ML API (:5000) [LIVE v3.4.0]
(ml/app.py,     /stations /telemetry           /stations /forecast
:8501)          /assessments /trends            /assistant/chat …
4 pages         /ml*/ml-data /data-quality      proxied by back-end/mlGateway.js
                /ingestion                       under /ml/live/*
   ▼
Next.js frontend (front-end/) / mobile app
```

### 1.5 ML module's role
One Python module (`ml/`) that is: the **data pipeline** (00–13), the
**forecasting engine** (pooled XGBoost + quantile calibration), the
**analytics** (correlation, importance, diagnostics), the **fleet scanner**
(`11_fleet.py`), the **assistant** (`_assistant.py`), and a **Streamlit
dashboard** (`app.py`, the current live analysis surface).

### 1.6 Streamlit app features (currently live) — `ml/app.py`
4 pages (`st.navigation`) + Verification (exists, not in nav):

| Page | What it shows |
|---|---|
| **Assistant** | station-locked chat (Ollama llama3.2:3b); instant facts panel works without LLM |
| **Correlation** | mode `raw/deseason/diff` × metric `spearman/pearson` table; recharge-lag curve (GWL vs rainfall); static features (soil/LULC) vs GWL |
| **Forecast** | single dark-theme trajectory v2 card: 120 genuine 6-hourly points, observed tail, "Forecast starts" boundary, bright q05/q95 band, direction banner, 6 metric cards, freshness caption |
| **Sources** | manifest quality, empty sources, static hydrogeology, LULC/soil status, association method |
| **Verification** *(not in nav)* | realised-forecast quality from the refresh pipeline's verification summary + sign-accuracy ledger |

> The legacy explorer pages (Overview, Drivers, Stations, Model, Fleet) were
> removed from `app_pages/`; their pipeline outputs (fleet scan, benchmark,
> honest validation, ablation, diagnostics) are still produced by the numbered
> scripts and served via `_model.py` loaders, the Flask API and the assistant.

---

## 2. Complete Feature List

| # | Feature | Purpose | User input | System process | Output | ML needed | Frontend display |
|---|---|---|---|---|---|---|---|
| F1 | Station picker / watchlist | find a station to inspect | text/select (station or district) | `station_recency()` sort (table_6h groupby max) | recency-ordered `[station, district, last_ts]` | no (data only) | dropdown + sortable table + last-updated badge |
| F2 | Station detail KPIs | latest level & history stats | station | `_assistant._series_stats()` (clean, then last/min/max/span/n_obs/outliers/change_7/30/60/180d) | facts dict (§10.4) | no | 5 KPI cards + caption |
| F3 | GWL trend chart | see level over time | station, interval | `load_table_6h()` slice | date×gwl series | no | line/area chart (price-style) |
| F4 | 2026 backtest curve | validate model vs actual | station, model (xgb/ridge) | `load_predictions()` fiter + melt to long | date×{target, xgb/ridge, band} | yes (inference on saved preds) | line chart + shaded q05/q95 error-band |
| F5 | 30-day forward forecast | predict future level | station | `forward_forecast(station)` (rebuild features, score xgb/ridge/q05/q50/q95) | forecast dict (§7.3) | **yes** | KPI table + anchor→+30d segment chart |
| F6 | Confidence interval | show uncertainty | (auto) | `forward_forecast` q05/q95 → `band_half=(q95−q05)/2` | band_half ±, interval | **yes** | shaded band on chart; "±x.xx m" KPI |
| F7 | Fleet score scan | rank all stations | none (snapshot) | `11_fleet.py` → fleet_forecast.csv | stations×Δ/category | **yes** | 6 KPI cards + table + histogram |
| F8 | Significant movers | stations moving faster than noise | none | `band_half.notna() & \|Δ\|>band_half` | movers table | yes | warning banner + table |
| F9 | District rollup | district-level comparison | district | `fleet_district.csv` aggregation | n, n_decline, n_recover, median Δ | yes (from snapshot) | table + heat tiles |
| F10 | Station scan filters | drill into fleet | category multi-select, district, movers-only | dataframe filters | filtered fleet table | no | filter widgets + table |
| F11 | Driver correlation | rank environmental drivers | mode, metric | `correlation_report.csv` | driver×corr×stations_ok | no (stats) | bar chart + table |
| F12 | Recharge-lag analysis | how many days before rain recharges GWL | (auto) | `lag_curves.csv` | lag(0–30d)×pearson median | no | line chart (peak ≈22 d) |
| F13 | Driver overlay | compare one driver vs GWL | station + driver | `load_table()` slice | date×{gwl, driver} | no | dual-axis chart |
| F14 | Model benchmark | compare model quality | metric basis (level/delta) | `model_metrics.csv` | 4-model table | no (eval data) | RMSE bars + table |
| F15 | Honest validation | see trustworthy numbers | (auto) | honest_metrics.csv, overlap.json, spatial_cv, residual_acf, residual_spatial | stride RMSE, eff-N, Moran's I … | no (eval data) | expander + metrics |
| F16 | Diagnostics | collinearity/importance/sensitivity | (auto) | `diagnostics.json` + permutation csv | VIF, permutation, OAT | no (eval data) | charts + metrics |
| F17 | Feature importance | which inputs matter | gain vs \|coef\| | `feature_importance.csv` / `feature_coefficients.csv` | ranked features | no (eval data) | horizontal bar chart |
| F18 | Station assistant chat | plain-language Q&A | station + question (+history) | `StationAssistant.answer()` (facts→prompt→Ollama) | `{answer, facts, station}` | **yes (LLM + model)** | chat UI + facts panel |
| F19 | Sources/provenance | data quality per source | (auto) | `manifest.json`, config SOURCES | rows/st/dropped per source | no | tables + captions |
| F20 | Soil/LULC context | static hydrogeology | station | `_soil.load_soil()`, `_lulc.load_lulc()` | soil texture %, LULC class top-4 % | no (gated) | expander caption |

---

## 3. Input Parameters

### 3.1 Model feature inputs (per 6h row — produced by `06_features.build`)
Columns in `KEEP` in `06_features.py`:
`Station, District, time, date, horizon, target, target_d, target_time, st_id, dist_id`
+ 30 floats (`NUM_COLS`) + 15 floats (`HYDRO_COLS`, carried — see 3.1.1).

| Param | Meaning | dtype | Unit | Range | Req | Example | Source |
|---|---|---|---|---|---|---|---|
| `gwl` | current GWL (anchor) | float32 | m | per-station trimmed (−120…70) | yes | −6.411 | telemetry |
| `lag1` | GWL 1 slot ago | float32 | m | same | yes* | −6.5 | calc (shift1) |
| `lag4` | GWL 1 day ago | float32 | m | same | yes* | −6.2 | calc (shift4) |
| `lag8` | GWL 2 days ago | float32 | m | same | yes* | −6.3 | calc |
| `lag28` | GWL 7 days ago | float32 | m | same | yes* | −6.0 | calc |
| `lag120` | GWL 30 days ago | float32 | m | same | yes* | −5.8 | calc |
| `gwl_roll7_mean` | 7d rolling mean | float32 | m | same | yes* | −6.2 | calc |
| `gwl_roll7_std` | 7d rolling std | float32 | m | ≥0 | yes* | 0.35 | calc |
| `gwl_roll30_mean` | 30d rolling mean | float32 | m | same | yes* | −6.0 | calc |
| `gwl_roll30_std` | 30d rolling std | float32 | m | ≥0 | yes* | 0.6 | calc |
| `rain_1d` | rain sum 1d | float32 | mm | 0…150×4 | yes* | 0.0 | IDW assoc |
| `rain_7d` | rain sum 7d | float32 | mm | 0… | yes* | 42.0 | calc |
| `rain_30d` | rain sum 30d | float32 | mm | 0… | yes* | 180.0 | calc |
| `temp` | air temp | float32 | °C | −10…55 | yes* | 31.5 | AWMS |
| `temp_7d` | 7d mean temp | float32 | °C | −10…55 | yes* | 30.0 | calc |
| `humidity` | rel humidity | float32 | % | 0…100 | yes* | 68.0 | AWMS |
| `humidity_7d` | 7d mean humidity | float32 | % | 0…100 | yes* | 70.0 | calc |
| `solar` | solar radiation | float32 | auto | auto | yes* | 350.0 | AWMS |
| `wind_speed` | wind speed | float32 | auto | auto | yes* | 4.2 | AWMS |
| `pressure` | atm pressure | float32 | hPa | auto | yes* | 1005.0 | AWMS |
| `river_level` | river water level | float32 | m | auto; sparse (2 dist) | yes* | NaN | district proxy |
| `canal_level` | canal water level | float32 | m | auto; 17 st/1 dist | yes* | NaN | nearest gauge |
| `year` | calendar year | float32 | yr | 2021…2026 | yes | 2026 | time |
| `month_sin/cos` | cyclic month | float32 | — | −1…1 | yes | −0.5 | calc |
| `doy_sin/cos` | cyclic day-of-year | float32 | — | −1…1 | yes | −0.9 | calc |
| `hour_sin/cos` | cyclic hour | float32 | — | −1…1 | yes | 0.0 | calc |
| `monsoon` | doy − 152 | float32 | day | −152…213 | yes | 90.0 | calc |
| `st_id` | station ordinal | int32 | — | 0…548 | yes (XGB only) | 0 | ordinal encode |
| `dist_id` | district ordinal | int32 | — | 0…28 | yes (XGB only) | 6 | ordinal encode |
| `Station` | station name | str | — | feature_config.stations | yes (Ridge) | "Agra City (UP-017)" | meta |
| `District` | district name | str | — | feature_config.districts | yes (Ridge) | "KAUSHAMBI" | meta |
| `target_time` | observed timestamp of the label row | datetime64 | — | — | yes (split only) | 2026-01-15 00:00 | calc (same shift as `target`) |

`*` = may be NaN (XGBoost handles; Ridge median-imputes). Missing driver columns
are simply absent → feature builder skips that window group.

#### 3.1.1 Hydrology-informed columns (`HYDRO_COLS` — carried, **not yet trained**)

Built in `_hydrology.py`, shared verbatim by `06_features.build` and
`refresh.future_drivers.build_step_features`. **These are present in the feature
frames and in the serving row dicts but are deliberately excluded from the frozen
models' `num_cols`**, so the shipped artifacts are unaffected; wiring them in
requires a deliberate retrain (the refresh gate refuses a candidate whose
`feature_cols` differ from production). Every column is causal (trailing windows
only) and every windowed value has an explicit coverage companion.

| Param | Meaning | dtype | Unit | Range | Req | Example | Source |
|---|---|---|---|---|---|---|---|
| `gwl_dev7` | `gwl − gwl_roll7_mean` | float32 | m | auto | no (carried) | 0.12 | calc |
| `gwl_dev30` | `gwl − gwl_roll30_mean` | float32 | m | auto | no (carried) | −0.30 | calc |
| `gwl_slope7` | Theil-Sen slope, trailing 7 d | float32 | m/day | auto | no (carried) | −0.01 | calc |
| `gwl_slope30` | Theil-Sen slope, trailing 30 d | float32 | m/day | auto | no (carried) | −0.03 | calc |
| `gwl_slope90` | Theil-Sen slope, trailing 90 d | float32 | m/day | auto | no (carried) | −0.02 | calc |
| `gwl_obs7` | observed-day coverage of the 7 d slope window | float32 | 0…1 | 0…1 | no (carried) | 0.86 | calc |
| `gwl_obs30` | same, 30 d | float32 | 0…1 | 0…1 | no (carried) | 0.91 | calc |
| `gwl_obs90` | same, 90 d | float32 | 0…1 | 0…1 | no (carried) | 0.88 | calc |
| `gwl_imputed` | 1 = forwarded reading, 0 = observed, NaN = no reading | float32 | flag | {0,1,NaN} | no (carried) | 0.0 | `04b_align_6h.gwl_obs` |
| `rain_api7` | API, τ = 7 d | float32 | mm | ≥0 | no (carried) | 4.2 | IDW rain |
| `rain_api22` | API, τ = 22 d (corpus-measured recharge lag) | float32 | mm | ≥0 | no (carried) | 9.8 | IDW rain |
| `rain_api60` | API, τ = 60 d | float32 | mm | ≥0 | no (carried) | 21.5 | IDW rain |
| `rain_obs7` | fraction of the τ = 7 d window backed by an **actual raingauge observation** (fixed denominator; future forecast/climatology and missing slots are unobserved) | float32 | 0…1 | 0…1 | no (carried) | 0.94 | calc |
| `rain_obs22` | same, τ = 22 d | float32 | 0…1 | 0…1 | no (carried) | 0.90 | calc |
| `rain_obs60` | same, τ = 60 d | float32 | 0…1 | 0…1 | no (carried) | 0.83 | calc |

Causality contract and formulas: `_hydrology.py` module docstring. In the serving
path the `gwl_dev*`/`gwl_slope*`/`gwl_obs*`/`gwl_imputed` values are **pinned at
the anchor** (constant across all 120 steps) while `rain_api*`/`rain_obs*`
**advance per step** using the observed history plus forecast/climatology rain;
the rain recursion runs over the full observed history so a τ = 60 d kernel is
warmed up.

`rain_api*` and `rain_obs*` deliberately mean different things and must not be
conflated:

* `rain_api*` — the numeric memory term. Built from the rainfall **forcing**
  available at the step, including Open-Meteo forecast and climatology values for
  future steps, because that is the best rainfall knowledge at that step.
* `rain_obs*` — a **coverage** column. Counts only slots backed by an actual
  raingauge observation; a missing historical slot and every future
  forecast/climatology slot are unobserved. It therefore **decays** across the
  forecast horizon rather than reading ~1.0 on rain that has not been observed.
  Coverage uses the fixed denominator τ·4 slots (28 / 88 / 240).

Rainfall has no forward-fill or interpolation anywhere in the pipeline
(`03_merge_normalize` drops NaN rows; `04_align`/`04b_align_6h` sum-or-mean over
gauges and left-join, leaving unobserved slots as NaN), so in a batch frame a
finite value *is* an observation — `06_features.build` constructs that mask
explicitly rather than inferring it, so the meaning survives an imputation step
added upstream later. `build_step_features` computes these unconditionally but
returns only the columns named in `fnames`, so a caller that omits them (including
the frozen production model) sees an unchanged matrix.

> **Before training on `rain_api*`, read
> Part 5 (rain_api train/serve mismatch).** The batch definition uses
> rainfall known at the anchor (horizon-invariant); the serving definition advances
> the recursion with future forecast/climatology rainfall (varies with `h`).
> Reproducing serving in training needs point-in-time forecast rainfall, which is
> not archived. The block therefore stays **experimental/untrained**.
> `rain_obs*` semantics are correct in both paths.

### 3.2 Runtime / API input params
| Param | Meaning | dtype | Valid | Req | Example | Source |
|---|---|---|---|---|---|---|
| `station` | station identity | str | any station in `feature_config.stations` | yes (F5/F18) | "Ramchhitoni Sahawar (UP-011)" | user/DB |
| `station_slug` | slug form (§12.2) | str | from /stations | alt for chat | "ashadha-prathmik-vidyalaya" | API |
| `district` | filter | str | 29 districts | opt | "KAUSHAMBI" | user |
| `model` (page) | model choice | str | `xgboost`\|`ridge` | opt (def xgb) | "xgboost" | user |
| `question` | chat text | str | non-empty | yes (chat) | "Is level rising?" | user |
| `history` | last chat turns | list[dict] | role|content | opt | `[{"role":"user","content":…}]` | session |
| `days` | forecast horizon | int | 7–90 (default 30) | opt | 30 | user |
| `limit/offset` | pagination | int | ≥0 | opt | 50/0 | UI |
| `q` | search text | str | — | opt | "Agra" | UI |

---

## 4. Dataset & Data Schema

### 4.1 Sources
All via NWIC/NWDP CKAN `datastore_search` (free, no auth, no key). Resource IDs
and value hints are in `ml/config.py` `SOURCES`. Full list in §6.5 of
`Part 1`.

### 4.2 `common.parquet` (source of truth) — `[LIVE]`
`data/processed/common.parquet`, ~5.3M rows / 1,353 UP stations /
2021-01-01 → 2026-09-05.
Columns: `Station (str), District (str), Tehsil (str, often "-"), Block (str),
Latitude (float), Longitude (float), RL_MSL (float, 72–400 m),
Data Acquisition Time (timestamp 6h), Groundwater Level Telemetry 6 Hourly
(meter) (float, negative = depth below ground; 0 m = ground level;
water table amsl = RL_MSL + gwl)`.

### 4.3 `table_6h.parquet` (model-ready aligned) — `[LIVE]`
`data/aligned/table_6h.parquet`, 2,990,000+ rows / 600 stations.
Columns: `Station, District, time (6h slot), gwl, rain, temp, humidity, solar,
wind_speed, wind_direction, pressure, river_level, canal_level` (+ `dist_km`
where spatial association applied).

### 4.4 `table.parquet` (daily) — `[LIVE]`
`data/aligned/table.parquet`, daily station×day for correlation/overview.

### 4.5 Feature tables — `[LIVE]`
`data/features/{train,val,test}.parquet` — columns per §3.1; `target = GWL(t+120)`,
`target_d = target − gwl`, `horizon = 30`, `st_id/dist_id` int32.

### 4.6 `predictions_2026.parquet` — `[LIVE]` (19 cols)
| Col | dtype | Meaning |
|---|---|---|
| `Station` | str | station |
| `District` | str | district |
| `date` | datetime64[us] | day |
| `time` | datetime64[us] | 6h slot |
| `horizon` | int64 | 30 |
| `target` | float64 | actual GWL(t+120) |
| `gwl` | float32 | anchor |
| `feat_days` | int8 | feature-history days used |
| `xgb` | float32 | XGBoost level pred |
| `ridge` | float64 | Ridge level pred |
| `persist` | float32 | persistence |
| `clim` | float64 | climatology |
| `err_xgb` | float64 | xgb level error |
| `err_ridge` | float64 | ridge level error |
| `q05_lvl` | float32 | calibrated lower |
| `q95_lvl` | float32 | calibrated upper |
| `step_idx` | int64 | index in 120-step window |
| `window_id` | int64 | non-overlap window id |
| `stride` | bool | True = independent row |

### 4.7 Column-by-column handling rules
| Concern | Policy (actual code) |
|---|---|
| Missing values | kept as NaN in features; XGBoost tolerates, Ridge `SimpleImputer(strategy="median")` |
| Duplicates | `13_refresh_nwic.py` merges deduped/chronological/dtype-coerced; telemetry unique `(station_id, observed_at)` in DB |
| Outliers | sentinel `|GWL|>500` dropped; per-station 0.5–99.5% quantile trim; `drop_gwl_spikes` robust MAD z=30; rainfall cap 150 mm/h |
| Regime shifts | `flag_regime_shift`: per-year median jump >15 m from 2021-24 → station dropped wholesale |
| Cleaning | daily + 6h medians; water table amsl = RL_MSL + gwl for sanity (corr 0.965) |
| Split | train <2025-10-01, val Oct–Dec 2025, test ≥2026-01-01 **by anchor time, then purged by target time** — a train row survives only if `target_time < 2025-10-01` and a val row only if `target_time < 2026-01-01`, so no 30-day label is observed inside a later split. Counts shown (train 1,629,574 / val 11,647 / test 378,554) are pre-purge; the purge removes ≈120 anchors × n_stations from train and val. |
| Sample rows | see §19 |

---

## 5. Location Structure

```
India → Uttar Pradesh (UP)
          └── 34 districts in telemetry archive (1,353 stations)
                └── 29 districts "live 2026" → 600 anchor stations
                      ├── Tehsil / Block (published as "-" for most telemetry stations)
                      ├── station: name, lat, lon, RL_MSL
```
- **Registry:** `data/meta/selected_gwl_stations.csv` columns:
  `Station, District, Tehsil, Block, Latitude, Longitude, n_2026_months,
  n_2026_records, first_2026, last_record, full26`.
- **Districts:** `data/meta/selected_districts.json` (29, alphabetical; see §4.7 of
  `Part 1`).
- **Model IDs:** `st_id` (sorted station names, 0..548) and `dist_id` (sorted
  districts, 0..28) — shared across train/val/test + app.
- **API IDs:** lowercase hyphenated slugs (`ashadha-prathmik-vidyalaya`), URL-encoded.
- **Backend:** `stations` table holds same geo fields + `state_lgd_code`,
  `district_lgd_code` + `rl_msl`.
- **Geo requirements:** lat/lon are **optional for ML** (drivers use spatial
  association; a station without coords still forecasts — anchored on its own
  GWL series). Required only for driver association/district proxies.
- **Coverage:** only the 29 districts have data; others never appear (probes
  return total=0, verified for e.g. Prayagraj, Amethi, Sambhal, G.B. Nagar).

---

## 6. ML Models

### 6.1 XGBoost pooled 30-day forecaster — `[LIVE]`
| Field | Value |
|---|---|
| Model name | `xgb_multihorizon.joblib` (XGBRegressor via `joblib`) |
| Algorithm | Gradient boosting, `reg:squarederror` |
| Purpose | 30-day delta `GWL(t+120)−GWL(t)` for any of 549 stations |
| Input features | 30 `num_cols` + `st_id` + `dist_id` (32 cols) |
| Target | `target_d` (delta); level = anchor + delta |
| Training data | `data/features/train.parquet`, sampled 1 slot/day per station |
| Preprocessing | SHIFT lags, rolling mean/std (min_periods=1), rain sums, cyclic sin/cos, monsoon |
| Scaling/encoding | none for trees; ordinals `st_id/dist_id` |
| Hyperparameters | `n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, tree_method=hist, eval_metric=rmse, random_state=42, n_jobs=6, early_stopping_rounds=60` → best_iteration **15** |
| Training process | `06_train.py` (early stop on val Oct–Dec 2025) |
| Evaluation | val delta RMSE **1.2748** m; 2026 test level RMSE **2.242** m; stride RMSE **2.339** m |
| Performance | beats persistence raw +2.1%, stride +1.8%; M±0.5 m 67.2%, ±1.0 m 85.1% |
| Limitations | near-agnostic to single drivers at 30d; region/2026 holdout is binding; do not trust per-row deltas < ±0.05 m |
| Saved file | `ml/models/xgb_multihorizon.joblib` |
| Load function | `_model.load_xgb_model()` |
| Predict function | `_model.forward_forecast(station)` (dict) |

### 6.2 Ridge linear baseline — `[LIVE]`
| Field | Value |
|---|---|
| Model name | `linear_multihorizon.joblib` (sklearn `Pipeline`) |
| Algorithm | Ridge regression (delta target) |
| Purpose | linear baseline + interpretable coefficients |
| Input features | 30 `num_cols` + `Station` + `District` (categorical OHE) |
| Preprocessing | numeric: `SimpleImputer(median)` → `StandardScaler`; cat: `OneHotEncoder(handle_unknown="ignore")`; ridge `RidgeCV(alphas=(0.1,1,10,100))` → alpha **0.1** |
| Evaluation | val delta RMSE 1.708 m; test level RMSE 2.456 m |
| Extra artifact | `feature_coefficients.csv` (top |coef|: `num__gwl -9.73`) |
| Load | `_model.load_linear_model()` |

### 6.3 Quantile models (uncertainty) — `[LIVE]`
| Field | Value |
|---|---|
| Files | `xgb_q05.joblib`, `xgb_q50.joblib`, `xgb_q95.joblib` |
| Algorithm | pooled XGBoost `reg:quantileerror` at α=0.05/0.50/0.95 |
| Purpose | calibrated 90% prediction interval on level |
| Calibration | `quantile_calibration.json`: `target_coverage_stride=0.8`, `widen_factor_k=1.0`, `coverage_stride=0.908`, `half_width_median_m=1.025`, `half_width_p90_m=2.575` |
| Anchor basis | `level = today's GWL + quantile(delta 30d)` |
| Load | `_model.load_quantile_models()` → dict `{q05,q50,q95}` |

### 6.4 Baselines (not trained, computed) — `[LIVE]`
- **Persistence:** 0-change (carry anchor). Test level RMSE 2.290 m.
- **Climatology:** per-station day-of-year mean. Test RMSE 3.572 m.
- Benchmarked in `07_evaluate.py` → `model_metrics.csv`, `honest_metrics.csv`.

Note: the `model_metadata`/`model_outputs` tables in the backend DB are a
**legacy DB registry** — the live stack uses `ml/models/*.json`. All `/ml/*`
legacy endpoints (`forecast/:stationId`, `risk/:unitId`, `anomalies`) are
**deprecated**.

---

## 7. Prediction System

### 7.1 Trigger
`[LIVE]` Streamlit: page load → `forward_forecast(station)`.
`[LIVE]` API: `GET /ml/live/forecast/:slug` → `mlGateway.getLiveForecast` → Flask `GET /forecast/:slug` → `_trajectory.trajectory_forecast(station)` (120×6h trajectory v2).

### 7.2 Required inputs
`station` name (or slug). Everything else is **derived** from `table_6h.parquet`
(no user-side feature building).

### 7.3 Processing pipeline (actual code — `_model.forward_forecast`)
1. `t = load_table_6h()[Station==station]`
2. `t = 06_features.drop_gwl_spikes(t)`
3. `feats = 06_features.build_full(t, keep_na=True)`; `last = feats.sort_values("time").tail(1)`
4. `pred_xgb = xgb_b.predict(last[fnames])` where `fnames = model.feature_names_in_ or cfg["num_cols"]`
5. `pred_ridge = linear_b.predict(last[cfg["num_cols"]+["Station","District"]])`
6. `anchor = last["gwl"]`; `qlev = anchor + quantile_predictions`; calibration `k`
7. `band_half = (q95−q05)/2`; `station_stride_rmse` from `station_summary.csv`
8. Returns dict or `{}` if no feature frame.

### 7.4 Exact output
```json
{
  "station": "Ramchhitoni Sahawar (UP-011)",
  "date_from": "2026-09-05 00:00:00",
  "date_to": "2026-10-05 00:00:00",
  "anchor": -2.841,
  "pred_xgb": 0.121, "pred_ridge": -0.58,
  "xgb_level": -2.72, "ridge_level": -3.421,
  "q05_level": -4.266, "q50_level": -2.691, "q95_level": -1.039,
  "widen_k": 1.0, "band_half": 1.613, "station_stride_rmse": 0.314
}
```
All levels in **metres** (float). `anchor/pred_*` in m, `band_half` in m.

### 7.5 Categories / confidence
No hard categories in the prediction itself. Uncertainty = calibrated 90%
interval (`coverage_stride=0.908`). Fleet scan adds categories
(`decline (high)/decline/stable/recovering/unknown/unreliable`) — §9.

### 7.6 Example input → output
`forward_forecast("Ramchhitoni Sahawar (UP-011)")` → §7.4 JSON (real live output).

---

## 8. Forecasting

### 8.1 What is forecast
30-day change `GWL(t+120) − GWL(t)` on the 6-hourly grid; level = anchor + change.
Not an event forecast — a **directional water-level outlook** with interval.

### 8.2 Historical data requirement
For a forward forecast: the station's own series inside `table_6h.parquet`
(any length ≥1 row works; more history = better). Fleet eligibility: ≥2000 obs
and ≥730-day span. For campaign-level claims, use the 2026 test / stride subset.

### 8.3 Forecasting algorithm/model
Pooled XGBoost (§6.1) + quantile (§6.3). Persistence = anchor (benchmark).

### 8.4 Required parameters
`station`. Optional (planned API): `days` 7–90 (single-horizon model — the
"days" param selects the interval built from the anchor; the model itself is
fixed at 30 d).

### 8.5 Time granularity
Input grid 6-hourly (00/06/12/18 UTC-ish native). Forecast horizon **30 days
(120 steps)** — that is the min/max for a true model prediction.

### 8.6 Output format
Dict (§7.4). Chart-ready series in the app: anchor date → +30 d segments for
XGBoost/Ridge/Persistence.

### 8.7 Chart data (verified real)
```
[{"date":"2026-09-05 00:00:00","XGBoost":-2.841,"Ridge":-2.841,"Persistence":-2.841},
 {"date":"2026-10-05 00:00:00","XGBoost":-2.72,"Ridge":-3.421,"Persistence":-2.841}]
```

### 8.8 Example forecast response
§7.4 (that IS the forecast). Fleet-level: `fleet_forecast.csv` row:
```
{"station":"Mahgaon (UP-031)","district":"KAUSHAMBI","anchor":-17.171,
 "date_from":"2026-09-03 18:00:00","age_days":6,"anchor_valid":true,
 "xgb_level":-16.881,"q05_level":-18.637,"q50_level":-16.689,"q95_level":-11.989,
 "change_30d_m":0.482,"change_pooled_m":0.29,"band_half_m":3.324,"plausible":true,"category":"recovering"}
```

---

## 9. Groundwater Impact / Analysis

### 9.1 Parameters affecting GWL (+ sign for shallow-rising)
| Driver | raw \|Spearman\| | sign | mode best | meaning |
|---|---|---|---|---|
| river_level | 0.409 | + | raw 0.409 / desean 0.168 | rising river ↔ rising GWL |
| pressure | 0.199 | + | desean 0.205 | high pressure → recharge |
| rain_30d | 0.199 | − (raw −0.199) | diff +0.013 | recent rain raises level (lag ≈22 d) |
| canal_level | 0.146 | − (raw −0.146) | desean −0.104 | canal-fed influence |
| humidity | 0.085/0.158 | +raw/−des | desean −0.158 | mixed |
| wind_speed | 0.095 | − | raw −0.095 | dry conditions |
| temp | 0.041 | − | desean +0.092 | evap demand |
| solar | 0.055 | − | desean −0.073 | evap |

### 9.2 Feature importance (XGBoost gain) — `[LIVE]`
Top: `monsoon 25381`, `gwl_roll7_std 21670`, `doy_sin 14347`, `gwl_roll30_std
10718`, `year 10004`, `doy_cos 7990`, `lag1 7534`, `gwl 7330`. (Full table in
`Part 1` §2.)

### 9.3 Ridge coefficients — `[LIVE]`
`feature_coefficients.csv`: `num__gwl -9.73`, `num__lag4 -0.35`,
`num__lag8 -0.30`, `num__lag1 +0.26`.

### 9.4 Recharge lag — `[LIVE]`
Best GWL↔rainfall lag ≈ **22 days** (Pearson median +0.136, ≥180 valid days).

### 9.5 Ablation & diagnostics — `[LIVE]`
Dropping rain/weather/calendar hurts +0.03–0.05 m; river/canal & GWL lags
neutral at 30d. No VIF>10. Permutation ≈0 for all. OAT max swing: `doy_cos`
0.385 m, `gwl` 0.148, `gwl_roll30_std` 0.146, `monsoon` 0.132.

### 9.6 SHAP/explainability
**Not implemented** (`no shap dependency`). Recommended follow-up:
`shap.TreeExplainer` on the stride subset. Today's explainability = gain + coef +
permutation + OAT + Spearman.

### 9.7 Exposure to frontend `[PLANNED API]`
`GET /ml/models` (+`?analysis=`) returns importance/coef/diagnostics; station
page embeds driver correlation; assistant phrases these facts.

---

## 10. Chatbot

### 10.1 Purpose
Station-locked plain-language Q&A that **phrases precomputed facts only**
(no invented numbers).

### 10.2 Architecture `[LIVE]`
```
User question + station
   → StationAssistant.answer(question, station, history)
       → facts = StationAssistant.facts(station)     (deterministic, from table_6h + model)
       → prompt = _build_prompt(question, facts, history)
       → answer = _invoke_llm(prompt)               (ollama.Client chat, temperature 0.2)
       → return {"answer", "facts", "station"}
```

### 10.3 Supported questions (examples)
"latest level", "trend over 30 days", "what's the 30-day outlook?", "how many
observations?", "district median?", "is it declining?"

### 10.4 Forecast data how it reaches chatbot
`facts() → _forecast_summary(station)` calls `_model.forward_forecast(station)`
and computes `day30_pred, change_30d_pred, direction, plausible, band_half,
q05/q95, station_stride_rmse, high_uncertainty`.

### 10.5 Context/data passed
facts dict (§10.4 of this doc = `_assistant._series_stats` + district median +
forecast), plus last 6 chat turns as conversation context.

### 10.6 LLM / model
`Ollama`, model default `llama3.2:3b`, override `AQUIS_OLLAMA_MODEL`
(repo `.env` or env). `ollama_status()` probes server + pulled models.

### 10.7 Prompt / system prompt
Verbatim template in `_assistant._build_prompt`:
```
You are the AQUIS groundwater assistant. Answer the user's question using
ONLY the given observed facts. No speculation. Be concise (max ~6 lines).
Use metres (m) for levels/changes. State dates where known.
Never refuse or say data is unavailable — use the station/district names …
[CONVERSATION (context only) …, always trust the LATEST FACTS]
FACTS:
station: … district: …
latest level: <last> m on <last_date>
range: <min> to <max> m (span <span> m, <n_obs> observations)
[note: N outliers excluded]
level change: 7d=… | 30d=… | 60d=… | 180d=… m
model forecast (+30 d): level=<day30> m, change 30d=<chg> m (<direction>); 90% band +/- <band_half> m (anchor <anchor> m); [q05/q95 interval …]
[unstable / high-uncertainty note where flagged]
district median (<dist>): <med> m across <n> stations
QUESTION: <question>
ANSWER:
```

### 10.8 API `[LIVE logic; planned HTTP]`
`POST /ml/assistant/chat` body `{question, station?, station_slug?, model?}` →
`{answer, facts, station}`.

### 10.9 Request/response format
```json
{"question": "Is level rising?", "station": "Ramchhitoni Sahawar (UP-011)"}
→
{"answer": "Latest level is -2.84 m on 2026-09-05 … expects a rise of about +0.12 m
 (to -2.72 m) over 30 days, 90% band +/- 1.61 m.",
 "facts": {"last": -2.841, "last_date": "2026-09-05", "change_30d": …, "forecast": {…}},
 "station": "Ramchhitoni Sahawar (UP-011)"}
```

### 10.10 Error handling
- No telemetry → `answer: "No telemetry found for station '<X>'."`, facts `{}`.
- Ollama down/pull-missing (page) → warning + facts panel still renders; chat
  returns error message with `ollama list` / `ollama pull {model}` instructions.
- Controller: missing `question` → `400 {error:"bad request", detail:"\`question\` is required"}`;
  gateway failure → `502`; assistant error → `503`.

### 10.11 Chat history requirement
`[PLANNED]` persist to `chat_logs` table (station, question, answer, facts JSONB,
model, created_at). Streamlit keeps `<6` turns in `st.session_state.as_messages`.

### 10.12 Forecast → summary/recommendation
`_forecast_summary` turns model numbers into text:
`direction` = "expected rise" (change ≥0) / "expected decline" + `band_half`
+ `plausible` flag (stops the LLM over-trusting runaway predictions) +
`high_uncertainty` (station stride RMSE > 2× fleet median).

---

## 11. Existing Code Structure (`ml/` + backend)

### 11.1 `ml/` files & functions
| File | Purpose | Key functions |
|---|---|---|
| `app.py` | Streamlit entry (4 pages + Verification, unwired) | `st.navigation(…)` |
| `app_pages/*.py` | assistant, correlation, forecast, data_sources + verification (§1.6) | page scripts |
| `config.py` | sources/radii/coverage/caps/paths | `SOURCES`, `FULL26_*`, `RAIN/WEATHER/RIVER/CANAL_RADIUS_KM`, `MAX_RAIN_MM_H` |
| `00_probe.py` | probe NWIC resources | → `probe.json` |
| `01_select.py` | full-2026 selection | → 600/1353, `selected_gwl_stations.csv` |
| `02_fetch_selected.py` | fetch drivers | resume-safe, per-district |
| `03_merge_normalize.py` | caps+clean | → `raw/*_norm.parquet`, `manifest.json` |
| `04_align.py` | daily alignment | IDW rain, nearest weather |
| `04b_align_6h.py` | 6h grid | → `table_6h.parquet` |
| `05_correlate.py` | correlation + lag | → `correlation_report.csv`, `lag_curves.csv` |
| `06_features.py` | features/target | `drop_gwl_spikes`, `flag_regime_shift`, `build`, `build_full`, `rain_exp_30d`, `split` |
| `06_train.py` | training | `main`, `_write_model_metadata` |
| `07_evaluate.py` | benchmark | → `model_metrics.csv`, `predictions_2026.parquet` |
| `07_soil.py` | SoilGrids fetch | → `data/soil/` |
| `08_lulc.py` | Bhuvan LULC | → `data/soil/lulc_district.csv` |
| `10_ablate.py` | ablation | → `ablation.csv` |
| `11_quantile.py` | quantile models | → `xgb_q*.joblib`, `quantile_calibration.json` |
| `11_fleet.py` | fleet scan | `category`, `main`, `_write_snapshot`, `_write_csvs` |
| `12_diagnostics.py` | VIF/permutation/OAT | → `diagnostics.json` |
| `13_refresh_nwic.py` | incremental refresh + retrain/deploy | `--dry-run`, `--retrain`, `--deploy-check` |
| `14_6h_features.py` | causal 6h feature frames (feeds `30_traj_datasets`) | → `data/features_6h/` |
| `16_future_drivers.py` | driver climatology + future-driver bridge | → `data/meta/driver_climatology.parquet` |
| `18_cwc_river_forecast.py` | CWC 3-day river forecast fetch | → `data/cfs/river_forecast_cwc.parquet` |
| `20_openmeteo_fetch.py` | Open-Meteo daily weather (365d history + 16d forecast) | → `data/cfs/openmeteo_weather_daily.parquet` |
| `30_traj_datasets.py` | trajectory v2 feature frames | → `data/features_traj/` |
| `31_train_traj.py` | trajectory multi-horizon quantile XGBoost | → `models/traj_xgb_*.json`, `traj_config.json` |
| `32_backtest_traj.py` | honest trajectory backtest + calibration | → `traj_backtest_*.json/csv`, `traj_calibration.json` |
| `33_traj_reliability.py` | reliability buckets + evidence weights | → `models/traj_reliability.json` |
| `_trajectory.py` | trajectory v2 engine (Forecast page + `/forecast/<slug>`) | `trajectory_forecast` |
| `api.py` | Flask HTTP API v3.4.0 | `/health`, `/stations`, `/series`, `/forecast/<slug>`, `/assistant/chat`, `/districts`, `/fleet/*`, `/alerts` |
| `app_charts.py` / `snapshot.py` | forecast chart helpers / PNG export | `trajectory_chart`, `trajectory_snapshot_png` |
| `refresh/` | fetch → features → inference → publish daemon + retrain gate | `cli.py`, `pipeline.py`, `scheduler.py`, `sources.py`, `features.py`, `inference.py`, `model_update.py`, `publish.py`, `verification.py` |
| `_model.py` | model loaders + forward forecast | `load_xgb_model`, `load_linear_model`, `load_quantile_models`, `load_predictions`, `forward_forecast`, `load_fleet_table`, `load_diagnostics`, … |
| `_utils.py` | shared loaders | `load_table_6h`, `station_recency`, `load_manifest`, `load_report`, `load_lag_curves`, `DRIVER_LABELS`, `SOURCE_LABELS` |
| `_assistant.py` | chatbot | `StationAssistant.facts/answer`, `_build_prompt`, `_forecast_summary`, `ollama_status`, `ollama_model`, `_clean_series`, `_series_stats` |
| `_soil.py` | soil loader | `load_soil`, `SOIL_COLS` |
| `_lulc.py` | lulc loader | `load_lulc`, `LULC_COLS` |
| `validation/*.py` | honest suite | `spatial_cv.py`, `overlap.py`, `residual_acf.py`, `spatial_residual.py` |
| `gate_check.py` | 13-check regression gate (latest GATE PASS 13/13) | `check(name, ok, detail)` |
| `tests/` | stdlib unittest, 177 tests (2 skipped) | `test_trajectory_v2.py`, `test_refresh_*.py`, `test_verification.py`, `test_api.py`, … |
| `README.md` | docs | — |

### 11.2 `back-end/` (Express) `[LIVE]`
- `app.js` — middleware (helmet, cors, json, morgan) + route mounts + 404 +
  `errorHandler`.
- `routes/` — `station, telemetry, assessment, trend, ml, mlData, dataQuality,
  ingestion, data, analytics, alert`.
- `controllers/` — one per route family; `ml.controller.js` implements the
  gateway passthroughs (`getLiveStations`, `getLiveForecast`, `postAssistantChat`, …).
- `services/` — `mlGateway.js` (HTTP proxy→:5000, `makeRequest`),
  `stationService`, `telemetryService`, `assessmentService`, `modelService`,
  `statisticsService` (Mann-Kendall, Sen's slope), `groundwaterClassification`,
  `dataQualityService`, ingestion services.
- `middleware/errorHandler.js`, `middleware/validators.js` (express-validator rules).
- `db/schema.sql`, `db/pool.js`.

### 11.3 `front-end/` (Next.js 16, React 19, TS, Tailwind v4, Chart.js) `[LIVE scaffold]`
- `app/page.tsx` dashboard home (AlertBanner, StatGrid, Charts, QuickActions).
- `components/dashboard/` (`AlertBanner, Charts, QuickActions, StatGrid,
  StationList`), `components/ui/` (`Button, Card, Modal, StatCard`),
  `components/layout/`, `components/profile/`.
- No API wiring yet — components use hard-coded props.

---

## 12. Backend Integration Requirements — API contracts

> **Status:** Node routes + gateway exist and are live (they proxy to `:5000`),
> and the **Flask ML service is live** (`ml/api.py` v3.4.0: `/health`,
> `/stations`, `/stations/<slug>`, `/stations/<slug>/series`, `/forecast/<slug>`,
> `/assistant/chat`, `/districts`, `/fleet/*`, `/alerts`).
> When Flask is down, `/ml/live/*` returns
> `502 {"error":"ML service error","detail":"ML service unavailable"}`.

### 12.1 Node routes currently mounted `[LIVE]` (see §14 list)
`/health`, `/stations*`, `/telemetry*`, `/assessments*`, `/trends*`, `/ml*`,
`/ml-data*`, `/data-quality*`, `/ingestion*`, `/data`, `/analytics`, `/alerts`.

### 12.2 Slug convention `[LIVE]`
Lowercase hyphenated station name (e.g. `ashadha-prathmik-vidyalaya`): use
`encodeURIComponent(slug)`. Always fetch
slugs from `GET /stations` — never hand-type. The exact station name also
resolves as a fallback.

### 12.3 Per-ML-feature contracts (target Flask `:5000`)

#### 12.3.1 Forecast
```
Endpoint: GET /ml/live/forecast/:slug
Method:   GET            (proxied to GET /forecast/:slug?days=)
Request:  path=slug (URL-encoded), query=days (7..90, default 30)
Response: 200 forecast dict (§7.4)     404 unknown slug     502 gateway     500 internal
```
```json
// GET /forecast/ashadha-prathmik-vidyalaya
{"station":"ASHADHA PRATHMIK VIDYALAYA","date_from":"2026-09-05T00:00:00",
 "date_to":"2026-10-05T00:00:00","anchor":-6.411,"pred_xgb":-0.213,
 "xgb_level":-6.624,"ridge_level":-7.896,"q05_level":-7.835,"q50_level":-6.12,
 "q95_level":-4.558,"widen_k":1.0,"band_half":1.64,"station_stride_rmse":0.597}
```

#### 12.3.2 Stations / location
```
GET /ml/live/stations            ?district=&q=&limit=   → [{station,district,slug,lat,lon,last_ts,full26}]
GET /ml/live/stations/:slug                             → station facts (§10.4) KPIs
GET /ml/live/districts                                  → [{district, n_stations, last_ts}]
GET /ml/live/models                                     → model/artifact index + metrics
```

#### 12.3.3 Fleet
```
GET /ml/live/fleet/forecasts                            → {horizon_days, generated_at, stations:[…]}
GET /ml/live/fleet/recovery   ?window_days=&top=&min_stations=  → district recovery ranking
GET /ml/live/fleet/scan       ?district=&threshold=&horizon=    → decline scan (quality-gated)
```

#### 12.3.4 Assistant
```
POST /ml/assistant/chat
Body:   {"question": "...", "station": "..."} | optional station_slug, model
400     missing question           {"error":"bad request","detail":"`question` is required"}
503     assistant down             {"error":"assistant unavailable", ...}
200     {"answer": "...", "facts": {...}, "station": "..."}
```

#### 12.3.5 Backend 4xx/5xx conventions
- 404 route: `{"error":"Route not found"}`
- validators: `{error: "…"}` / array of `{param, msg}` via errorHandler.
- 502 ML service unreachable: `{"error":"ML service error","detail":"ML service unavailable"}`.

---

## 13. Database Requirements

### 13.1 Existing schema (`back-end/db/schema.sql`) `[LIVE]`
Tables: `ingestion_runs`, `stations`, `telemetry_observations`,
`assessment_units`, `assessment_records`, `data_quality`, `model_metadata`,
`model_outputs`, `groundwater_data`.
Key relations:
- `telemetry_observations.station_id → stations.id` (FK, cascade), unique
  `(station_id, observed_at)`, index on observed_at.
- `ingestion_run_id → ingestion_runs.id`.
- `assessment_records → assessment_units.id`.
- `model_outputs → model_metadata.id`.
- `stations` has `external_station_id UNIQUE`, `first/last_observed_at`,
  `observation_count`, indexes on state/district/agency/coords.

### 13.2 Suggested additions `[PLANNED]`
| Table | Fields (types) | Keys | Notes |
|---|---|---|---|
| `ml_stations` | `station PK text`, district, tehsil, block, lat/lon numeric(12,8), rl_msl numeric, n_2026_months int, n_2026_records int, first_2026 ts, last_record ts, full26 bool | PK station | mirror of selected_gwl_stations.csv |
| `ml_predictions` | `id PK bigserial`, station FK→ml_stations, time ts, horizon int, target/gwl/xgb/ridge/persist/clim/q05_lvl/q95_lvl numeric, err_xgb/err_ridge numeric, stride bool, window_id int | unique(station,time) | bulk-load predictions_2026.parquet (or serve from parquet) |
| `ml_fleet_snapshot` | `station PK`, district, anchor/date_from, age_days int, anchor_valid bool, xgb_level, q05/q50/q95_level, change_30d_m, change_pooled_m, band_half_m, plausible bool, category text, generated_at ts | PK station | fleet_forecast.csv |
| `quantile_calibration` (singleton) | coverage_stride, widen_factor_k, half_width_median_m, half_width_p90_m | PK id=1 | quantile_calibration.json |
| `model_meta` (singleton) | trained_at, n_stations, n_districts, val_rmse_xgb, val_rmse_ridge, source_archive | PK id=1 | model_metadata.json |
| `chat_logs` | `id PK bigserial`, station text, question text, answer text, facts jsonb, model text, created_at ts | idx station, created_at | assistant history |
| `alert_rules` (future) | `id`, station, threshold_m, band_ref text, active bool, created_at | FK station | for push alerts |
| `fleet_alerts` (future) | `id`, station, change_30d_m, band_half_m, breached bool, snapshot_ts | FK station | significant movers |

### 13.3 Requirements in prose
Store: stations, telemetry (or keep parquet source), predictions (historical),
fleet snapshots, calibration + model meta (single-row), chat history, alert
rules/alerts, user/profile (if roles added), ingestion runs, assessment data,
data-quality records. Indexes on `station`, `observed_at/date`, `district`,
`category`, `created_at`. Foreign keys: telemetry→stations, predictions→stations,
fleet→stations, outputs→model_meta.

---

## 14. Frontend Requirements (per screen)

> `[LIVE]` = scaffold exists; `[PLANNED]` = build new.

| Page | Purpose | Components | Inputs | Charts | Cards/KPIs | Tables | Needs API |
|---|---|---|---|---|---|---|---|
| **Home/Dashboard** `[LIVE scaffold]` | state/district headline | AlertBanner, StatGrid, Charts, QuickActions | district/state select | total/current extraction donut, recharge bar | extraction %, rainfall mm, recharge MCM | — | `GET /ml/live/stations?district=`, `/assessment/summary` |
| **Station Watchlist** `[PLANNED]` | pick + search stations | search, virtual list, recency badge | q, district | sparkline per row | — | station×last_ts | `GET /ml/live/stations` |
| **Station Detail** `[PLANNED]` | KPIs + GWL + forecast | KPI row, line chart w/ band, tabs | station | GWL series + q05/q95 band + segment | anchor, ±band, RMSE | facts table | `GET /ml/live/stations/:slug`, `/ml/live/forecast/:slug`, `/telemetry/:stationId` |
| **Forecast** `[PLANNED]` | backtest + outlook | selectors (station, model), error-band chart, outlook table | station, model | backtest curve + band, outlook segment, rain volume overlay | RMSE, interval ±, anchored reads | metrics | `GET /ml/live/forecast/:slug`, `GET /ml/live/models` |
| **Fleet/Signals** `[PLANNED]` | whole-fleet movers | 6 KPI row, histogram, filters | category, district, movers-only | Δ histogram ±0.3 m rules | 6 KPIs | movers, district rollup, station scan | `GET /ml/live/fleet/forecasts`, `/fleet/recovery`, `/fleet/scan` |
| **Impact/Analysis** `[PLANNED]` | driver importance | segmented mode/metric, bars, table | mode, metric | importance & |coef| bars, correlation bars | strongest/weakest driver | ranked table | `GET /ml/live/models` (+analysis) |
| **Assistant** `[PLANNED]` | chat + facts | chat list, input, facts panel, model-status | question, station | — | fact KPIs | facts | `POST /ml/assistant/chat`, `GET /ml/live/stations` |
| **Sources** `[PLANNED]` | data provenance | manifest tables | source | coverage bars | — | per-source quality, empty sources, soil/LULC status | `GET /ml/live/models`, `/ml-data/*` |
| **Mobile (design target)** | trading-terminal UI | dark theme, watchlist, price-style charts, heatmap, sidebar comparison | station/district | GWL+band charts, district heatmap, indicator overlays, rain volume | watchlist sparkline | district comparison, movers | same API set |

**Loading/error/empty states to handle**: spinner while `makeRequest` (60s
timeout) runs; `available:false` banner on boot; "No feature frame for this
station." when forecast `{}`; "run `ml/11_fleet.py` first" when fleet empty;
Ollama-down warning in chat; empty charts when station has no data.

---

## 15. Frontend ↔ Backend ↔ ML data flow

### 15.1 Forecast flow
```
User ── select station ──► Frontend (Forecast page)
  │ GET /ml/live/forecast/:slug?days=30
  ▼
Backend ml.controller.getLiveForecast → mlGateway.getLiveForecast(slug, days)
  │ HTTP GET http://localhost:5000/forecast/:slug  (makeRequest, 60s timeout)
  ▼
Flask route [LIVE] → trajectory_forecast(station)
  │ anchor row from table_6h.parquet → shared direct multi-horizon quantile
  │ models (h = 1..120) → 120 genuine q05/q50/q95 deltas; level = anchor + delta
  ▼
Backend ← JSON (station, slug, anchor, trajectory[120] with time/gwl/q05/q50/q95/confidence_level, …)
  ▼
Frontend renders the trajectory card + anchor→+30d segment chart + interval caption.
```

### 15.2 Assistant flow
```
User question ─► Assistant page ─► POST /ml/assistant/chat {question, station}
  ▼ Backend postAssistantChat -> mlGateway.assistantChat
  ▼ Flask /assistant/chat [LIVE]
StationAssistant.facts(station) → _forecast_summary(station) → forward_forecast
_build_prompt(question, facts, history) → ollama.Client.chat(temperature 0.2)
  ▼ {answer, facts, station} ─► Frontend chat bubble + facts panel + (optional) DB chat_logs
```

### 15.3 Fleet flow
```
Frontend Fleet page → GET /ml/live/fleet/forecasts (+ /fleet/recovery, /fleet/scan)
  ▼ Backend passthrough + mlGateway
  ▼ [PLANNED] Flask reads outputs/fleet_forecast.csv (precomputed by 11_fleet.py --force; the offline scan exists, the HTTP endpoint does not yet)
  ▼ JSON snapshot ─► 6 KPIs, histogram, district rollup, station table
```

### 15.4 Static/backend data flow (live Node only)
```
Frontend → GET /stations, /telemetry/:stationId, /assessments, /trends/:stationId
  ▼ station/telemetry/assessment service ⇄ PostgreSQL (schema §13)
  ▼ JSON → tables/charts
```

---

## 16. External APIs & Services

| Service | Purpose | Auth | Env vars | Rate/free | Notes |
|---|---|---|---|---|---|
| NWIC/NWDP CKAN `datastore_search` | all telemetry (GWL + drivers) | none | — | free | resource ids in `ml/config.py`; source map `back-end/db/api/api.txt` |
| ISRO Bhuvan LULC statistics | district land-cover shares | key | `LULC_STATISTICS_API_KEY` | free/patronised | `bhuvan-app1.nrsc.gov.in/api` |
| ISRIC SoilGrids v2 REST | per-station texture | none | — | free | `07_soil.py` background fetch |
| Ollama (local) | LLM `llama3.2:3b` (~2 GB) | none | `AQUIS_OLLAMA_MODEL` | free/local | no SaaS; `ollama serve` |
| NCEI CFSv2 6h-FLX | seasonal rain (staged) | none | — | free | paused; S3-403 workaround via per-step GRIB |
| CGWB assessments | Excel files via backend import | none | — | public | not in ML features |

> No API keys/secrets appear in this document. `.env` holds
> `LULC_STATISTICS_API_KEY` (example values in `.env.example`).

---

## 17. Dependencies & Setup

### 17.1 Python (ml venv — verified versions)
```
pandas==3.0.5  numpy==2.5.2  matplotlib==3.11.1  scikit-learn==1.9.0
requests==2.34.2  flask==3.1.3  flask-cors==6.0.5  scipy==1.18.0
xgboost==3.4.1  joblib==1.5.3  streamlit==1.62.0  plotly==6.9.0
altair==6.2.2  ollama==0.6.2
```
Root `requirements.txt` (Streamlit Cloud): `pandas, numpy, matplotlib,
scikit-learn, requests, flask, flask-cors, scipy, xgboost, joblib, streamlit,
plotly`.
> **Add `ollama` to requirements** if the assistant is part of the deploy.

### 17.2 Node (backend)
Node **v22.23.1** in use (runtime requires Node 18+), Express, PostgreSQL 14+.
`cd back-end && npm install && npm run migrate`.

### 17.3 Frontend
Next.js 16.2.9, React 19.2.4, TypeScript 5, Tailwind CSS v4, Chart.js ^4.5.1.
`cd front-end && npm install && npm run dev`.

### 17.4 Environment variables
`DATABASE_URL, PORT, NODE_ENV, ML_SERVICE_URL, ML_TIMEOUT_MS, BACKEND_URL,
AQUIS_OLLAMA_MODEL, LULC_STATISTICS_API_KEY`.

### 17.5 Model setup / local dev
```bash
# reproduce models from archive (optional; models already committed for deploy)
cd ml
venv/bin/python 06_features.py
venv/bin/python 06_train.py
venv/bin/python 11_quantile.py
venv/bin/python 11_fleet.py --force       # ~4-6 min
venv/bin/python 30_traj_datasets.py      # trajectory v2 frames (~7M rows)
venv/bin/python 31_train_traj.py         # shared multi-horizon q05/q50/q95
venv/bin/python 32_backtest_traj.py      # honest backtest + calibration
venv/bin/python 33_traj_reliability.py   # reliability buckets + weights
venv/bin/python 13_refresh_nwic.py --retrain   # or run app directly on committed artifacts

# run the live analysis dashboard
venv/bin/streamlit run app.py              # http://localhost:8501

# assistant
ollama serve &  ollama pull llama3.2:3b

# verify after edits
venv/bin/python -m unittest discover -s tests -v
venv/bin/python gate_check.py
```

### 17.6 Deployment (Streamlit Cloud)
Main file **`ml/app.py`**, root `requirements.txt`, models force-committed
(`.gitignore` keeps them; ignore `ml/data/features/`, `ml/data/aligned/*.csv`,
`ml/*.log`). For full-stack: deploy Flask ML service (`:5000`) + Point
`ML_SERVICE_URL` at it.

---

## 18. Validation & Error Handling

| Case | Where | Expected response |
|---|---|---|
| invalid/unsupported station | Flask forecast | `404 {"error": "unknown slug"}` / forecast `{}` → app "No feature frame for this station." |
| missing `question` | Node assistant | `400 {"error":"bad request","detail":"\`question\` is required"}` |
| wrong datatype (`days`="abc") | planned Flask | `400 {"error":"invalid \`days\`; must be int 7..90"}` |
| out-of-range `days` | planned Flask | `400` (7–90) |
| `limit/offset` bad | Node validators | `400 {error:…}` |
| location not in 29 districts | `/districts`/`/stations` | returns whatever exists; district filter yields empty list (not error) |
| missing data (empty table) | app loaders | page `st.stop()` with "run `ml/11_fleet.py` first" (fleet) / "No data for that selection." (assistant) |
| insufficient history | forward_forecast | `{}` → warning |
| model failure | Flask | `500 {error, detail}` |
| ML service down | Node gateway | `502 {"error":"ML service error","detail":"ML service unavailable"}` |
| ML timeout (>60s) | gateway | `502` "ML service timeout" |
| assistant/Ollama down | Node/{Flask} | `503`; app shows facts panel + instruction message |
| DB failure | Node services | `500` via errorHandler |
| unknown route | Express | `404 {"error":"Route not found"}` |

---

## 19. Testing

### 19.1 Existing ML tests `[LIVE]` — `ml/tests/`
177 tests (2 skipped), stdlib `unittest`, data-gated, no pytest: trajectory v2,
refresh pipeline (core/future/training/schedule), verification, Flask API,
assistant facts, calibration/diagnostics, forecast UI. Run:
`venv/bin/python -m unittest discover -s tests -v` (the full Forecast-page
AppTest is gated behind `AQUIS_APPTEST=1`).
Gate: `venv/bin/python gate_check.py` → 13 checks, latest **GATE PASS 13/13**
(stride RMSE 2.339, spatial CV 1.973/1.819, coverage 0.908, half-width
1.025/2.575, eff-N 6635, lag1 ACF 0.858, trajectory 30d 2.106 promoted, +30d
coverage 0.90; the 2 recursive-model checks skip).

### 19.2 Sample inputs → expected outputs `[PLANNED automated]`
| Test | Input | Expected |
|---|---|---|
| forecast | station "Ramchhitoni Sahawar (UP-011)" | dict §7.4; `anchor≈-2.84`, `band_half≈1.61`, all keys present |
| forecast empty | "Nonexistent Station XYZ" | `{}` |
| fleet | `fleet_forecast.csv` | 502 scored; median Δ +0.32 m; `category` ∈ enum |
| correlation | driver=river_level, metric=spearman, mode=raw | corr ≈ 0.409 |
| importance | `feature_importance.csv` | row 0 = `monsoon` |
| qcal | `quantile_calibration.json` | `widen_factor_k==1.0`, `coverage_stride≥0.8` |
| assistant | q="Is level rising?", station set | answer contains level/change; `facts.forecast.plausible` in {True,False} |
| Node forecast | GET /ml/live/forecast/:slug (Flask running) | 200 + keys |
| Node no-Flask | GET /ml/live/forecast/:slug | 502 + detail |
| chat missing question | POST /ml/assistant/chat {} | 400 |
| validators | /stations?limit=bad | 400 |

### 19.3 Edge cases
Stale anchor (>45 d) → fleet `unreliable`; runaway pred (`|Δ|>12` or outside
obs±25) → `plausible=false`; NaN quantiles → `band_half` null; station-missing
in `station_summary` → `station_stride_rmse` null; driver columns absent →
skipped windows.

---

## 20. Performance & Deployment

| Concern | Value (measured/expected) |
|---|---|
| Inference (single station) | `forward_forecast` ≈ 0.1–0.2 s (after table cached); `trajectory_forecast` scores 120 horizons in one pass (XGBoost predict on 120 rows negligible) |
| Table load | `load_table_6h` ≈ 1.7 s (cached 10 m, 2.99M rows, ~58 MB mem) |
| Predictions read | `load_predictions` 10-col ≈ 0.07 s (18 MB parquet) after 24 h TTL |
| Groupby recency | `station_recency()` 3M-row groupby ≈ 0.09 s (cached) |
| Fleet scan (fresh `--force`) | ≈ 4–6 min (502 stations) |
| Full eval/honest rerun | minutes (06/07/11) |
| Memory | ~2–3 GB RSS for app (table6 + preds + models); Ridge fit switched to 1 slot/day sampling to fit 16 GB |
| CPU/GPU | CPU only (XGBoost `hist`, n_jobs=6 during train); no GPU required |
| API response (plan) | forecast <500 ms after warm cache; fleet from snapshot <100 ms; chat 15–60 s (LLM) |
| Concurrency | Streamlit is single-session-per-browser; for production use the flask service + persistent table cache, cache `forward_forecast` (TTL) |
| Caching | `@st.cache_data` TTLs: 10 m (tables/evals), 24 h (predictions), 15 m (fleet/diag); `@st.cache_resource` for joblib models |
| Model loading | joblib once per process (`@st.cache_resource`); warm lazy |
| Deploy | Streamlit Cloud (`app.py`, root requirements) for dashboard; Flask on app host for API; models committed; parquet dirs git-ignored |
| Production limits | Trajectory v2 outputs 120 genuine 6-hourly steps (direct multi-horizon, no recursion); the pooled direct model is fixed at the 30-day endpoint; UP telemetry coverage limited; chat needs local Ollama (2 GB) or an external host |

---

## 21. Security

- **Auth:** none today (localhost/LAN). `[PLANNED]` token/JWT for the API + roles
  (Officer/Researcher/Public/Admin) if deployed publicly; mobile needs auth at
  Node layer.
- **Authorization:** role-based pages (Officer: fleet/alerts; Public: read-only
  explorer; Researcher: impact/analysis).
- **API security:** Express already uses `helmet` (CSP off), `cors`, JSON body
  limit; add rate limiting (`express-rate-limit`) + input validation via
  `middleware/validators.js` (extend for new routes).
- **Secrets:** only in `.env` (git-ignored); `.env.example` has placeholders;
  no keys in repo.
- **Input sanitisation:** express-validator chains; LLM prompt build is a strict
  template — no injection surface beyond the fee-text `question` (phrased into
  the prompt as-is; keep it that way, never add system-level tool calls).
- **Rate limiting:** apply to `/ml/assistant/chat` (LLM is local but slow) and
  `/ml/live/*`; e.g. 30/min per IP, 300/h.
- **User-data privacy:** telemetry is public government data; chat logs are
  PII-light but store facts/answers; honour minimal-collection + retention.

---

## 22. Important ML Limitations (don't over-claim)

1. **Coverage:** only 29 districts/600 stations have live 2026 telemetry → 45 UP
   districts are NOT forecastable. Manual (quarterly) GWL is not ingested.
2. **Driver scarcity:** rainfall 8 districts, weather ~3, river 2, canal 1 →
   most stations have NaN drivers; correlation sample is small; river uses a
   **district-level proxy** (no coords published).
3. **Time:** archive starts 2021 (5.6 y) — seasonality beyond 5 y is
   under-sampled; regimes pre-2021 ignored.
4. **Horizon single-step:** 30-day is the only trained horizon; "7–90 days" API
   param is not a multi-step model.
5. **Overlap honesty:** 378k test rows ≈ 6.6k effective; use **stride** numbers
   (RMSE 2.339 vs persistence 2.382, +1.8%). Deltas < ±0.05 m are noise.
6. **When not to trust:** NaN/stale anchor (>45 d), `plausible=false`
   (runaway Δ>12 m or outside obs±25 m), `high_uncertainty` (stride RMSE > 2×
   fleet median), any fleet `category="unreliable"`.
7. **Assumptions:** level = anchor + delta where persistence = 0-change; GWL
   sign convention negative = depth below ground (0 = ground level); static
   soil/LULC gated out (redundant with ordinals); no extraction data available
   (CGWB PDF/Excel only, 1–2 y lag).

---

## 23. Exact Feature → API → ML Mapping

| Feature | Frontend screen | API (planned) | Backend fn | ML model | Input | Output |
|---|---|---|---|---|---|---|
| F1 watchlist | Watchlist | `GET /ml/live/stations` | `getLiveStations`→`mlGateway.getStations` | none | district/q/limit | station list |
| F2 KPIs | Station Detail | `GET /ml/live/stations/:slug` | `getLiveStation`→`getStation` | none | slug | facts |
| F3 trend | Station Detail | `GET /telemetry/:stationId` | telemetryService | none | stationId | series |
| F4 backtest | Forecast | (parquet via `load_predictions`) | `_model.load_predictions` | xgb/ridge | station | date×target/gwl/xgb/qband |
| F5 forward forecast | Forecast | `GET /ml/live/forecast/:slug` [LIVE] | `getLiveForecast`→`trajectory_forecast` | trajectory v2 (120×6h q05/q50/q95) | station | trajectory dict (§7.4 shape + trajectory/confidence) |
| F6 confidence | Forecast/Detail | same as F5 | `trajectory_forecast` | evidence-weighted (interval + vs-persistence + direction + …) | station | per-point confidence_level + reason |
| F7 fleet score | Fleet | `GET /ml/live/fleet/forecasts` | `getFleetForecasts` | pooled fwd | none | csv rows |
| F8 movers | Fleet | `GET /ml/live/fleet/scan` | getFleetScan | pooled fwd | dist/threshold/horizon | movers |
| F9 district rollup | Fleet | `GET /ml/live/fleet/recovery` | getFleetRecovery | pooled fwd | window/top/min | district stats |
| F10 scan filters | Fleet | (client-side) | — | — | filters | filtered rows |
| F11 correlation | Impact | `GET /ml/live/models`(+analysis) | load_report | none (stats) | mode/metric | corr table |
| F12 recharge lag | Impact | ditto | load_lag_curves | none | — | lag curve |
| F13 driver overlay | Station Detail | `GET /ml-data/telemetry/:stationId` | mlData controller | none | driver | dual series |
| F14 benchmark | Impact | `GET /ml/live/models` | load_model_metrics | all | basis | 4-model table |
| F15 honest | Impact | ditto | honest/metric loaders | — | — | stride metrics |
| F16 diagnostics | Impact | ditto | load_diagnostics | — | — | VIF/perm/OAT |
| F17 importance | Impact | ditto | load_importance/coefs | xgb/ridge | kind | ranked |
| F18 assistant | Assistant | `POST /ml/assistant/chat` | `postAssistantChat`→`assistantChat` | LLM+pooled | question/station | answer+facts |
| F19 sources | Sources | `GET /ml/live/models`, `/ml-data/*` | manifest/mlData | none | — | per-source quality |
| F20 soil/LULC | Sources/Detail | (artifacts) | load_soil/load_lulc | gated | station/district | texture/classes |

---

## 24. Final Integration Checklist — Streamlit ML → Full-Stack

**Phase 0 — freeze artifacts:** `.joblib` + `models/*.json` committed; parquet
dirs git-ignored; `13_refresh_nwic.py --retrain` reproducible.

**Phase 1 — ML API (the big gap):**
- [ ] Ship a **Flask app** exposing the contract in §12 (`/health`,
  `/stations`, `/stations/:slug`, `/districts`, `/models`, `/forecast/:slug`,
  `/fleet/forecasts`, `/fleet/recovery`, `/fleet/scan`, `/assistant/chat`).
- [ ] Map each route to the existing fn (`forward_forecast`, `StationAssistant`,
  `load_fleet_table`, manifests) — no new ML logic needed.
- [ ] Slug↔name resolver built on `selected_gwl_stations.csv`.
- [ ] Persistent cache (in-process dict / tinykv) for table6 + forward forecasts.
- [ ] Error mapping per §18 (400/404/500).

**Phase 2 — Backend:**
- [ ] Verify `back-end/services/mlGateway.js` against the Flask surface (already
  has all callers + `_qs` + timeout).
- [ ] Point `ML_SERVICE_URL` at Flask in production `.env`.
- [ ] Add tables §13.2 (ml_stations, ml_predictions opt, ml_fleet_snapshot,
  chat_logs, alerts); migrate script.
- [ ] Extend `validators.js` for new live routes (slug, days, question).

**Phase 3 — Frontend (Next.js):**
- [ ] Wire `components/dashboard/*` to API (remove hardcoded props).
- [ ] Build pages §14: Watchlist → Detail → Forecast → Fleet → Impact →
  Assistant → Sources, each with loading/error/empty states (§18).
- [ ] API client module (`lib/api.ts`) hitting Node `/ml/live/*`.
- [ ] Chart components (Chart.js) for GWL+band, backtest, histogram, heatmap,
  driver overlays.

**Phase 4 — Mobile (trading-app-style):**
- [ ] Watchlist (recency-sorted) · station detail with GWL "price chart" + 90%
  band (Bollinger-like) · indicator overlays (7d/30d MA, rolling-std band, rain
  volumes) · seasonal/monsoon overlay · district/UP heatmap · district sidebar
  comparison · fleet movers · alerts (Δ > band push) — all fed by the same APIs.

**Phase 5 — Hardening:**
- [ ] Auth/JWT + roles; rate-limit chat; input sanitisation; CSRF not applicable
  (API practices) / proper CORS origins.
- [ ] `gate_check.py` + `unittest` in CI; AppTest smoke for the 4 live pages; API tests
  (§19.2).
- [ ] Monitoring: `setuptools` read of `model_metadata.trained_at` vs NWIC max
  ts (`13_refresh_nwic.py --deploy-check`).
- [ ] Chat history persistence + retention policy.

---

*Everything above is extracted from the shipping code and artifacts. Numbers
(metrics, hyperparameters, JSON examples) were read live from the repo on
2026-09-09/10.*


---

# Part 3 — Module ↔ gateway contract

<!-- source: docs/ml-contract.md -->

# AQUIS ML Contract

Contract between the **Node.js backend** and the **Python ML module** (`ml/`, trajectory v2).

## Status

The current `ml/` module includes a **trajectory-v2 model + Streamlit dashboard + headless Flask API** (`ml/api.py` on `:5000`). The Node backend proxies `/ml/*` requests to the Flask service. This document defines:

1. the **data / artifact contract** the current `ml/` produces (live today), and
2. the **HTTP contract** for the Flask API (live).

## 1. Artifact contract (live)

Everything below lives under `ml/` and is regenerated by the numbered pipeline; `ml/models/` and `ml/outputs/` are git-ignored (reproducible from the archive + pipeline).

### `data/processed/common.parquet`
Cleaned 6-hourly NWIC GWL archive — the read-only source of truth: **~5.3M rows / 1,353 UP stations / 2021-01-01 → 2026-09-05**. Columns include `Station`, `District`, `Tehsil`, `Block`, `Latitude`, `Longitude`, `RL_MSL`, `Data Acquisition Time`, `Groundwater Level Telemetry 6 Hourly (meter)`. Refreshed incrementally by `13_refresh_nwic.py`.

### `models/` (trained artifacts)
| File | Produced by | Contents |
|---|---|---|
| `traj_xgb_{q05,q50,q95}.json`, `traj_config.json` | `31_train_traj.py` (offline) / `refresh/model_update.py` (scheduled refit) | trajectory v2 direct multi-horizon quantile XGBoost (120×6h) |
| `traj_calibration.json` | `32_backtest_traj.py` | per-horizon widening to 90% coverage |
| `traj_reliability.json` | `33_traj_reliability.py` | bucket rules + evidence weights for confidence |
| `xgb_multihorizon.joblib`, `linear_multihorizon.joblib` | `06_train.py` | pooled 30-day XGBoost + Ridge (delta target) |
| `xgb_q{05,50,95}.joblib` | `11_quantile.py` | pooled quantile forecasters |
| `feature_config.json` | `06_train.py` | model_type, target, horizons, num_cols, stations, districts, split rows, val RMSE, trained_at |
| `quantile_calibration.json` | `11_quantile.py` | `coverage_stride`, `widen_factor_k`, `half_width_median_m`, `half_width_p90_m` |
| `model_metadata.json` | `06_train.py` | architecture, split, val RMSE, trained_at + embedded validated metrics (calibration / honest / spatial CV) |

### `outputs/` (evaluation + fleet)
| File | Produced by | Contents |
|---|---|---|
| `model_metrics.csv`, `eval_by_station.csv`, `eval_by_district.csv` | `07_evaluate.py` | 4-way benchmark on 2026 held-out |
| `honest_metrics.csv`, `overlap.json`, `station_summary.csv` | `validation/overlap.py` | overlap-aware (stride-window) re-score |
| `spatial_cv_metrics.csv`, `spatial_cv_summary.json` | `validation/spatial_cv.py` | leave-block-out CV |
| `residual_acf.json`, `residual_spatial.json` | `validation/` | effective sample size, lag-1 ACF, Moran's I / Geary's C |
| `fleet_forecast.csv`, `fleet_district.csv`, `fleet_forecast_snapshot.json` | `11_fleet.py` | whole-fleet 30-day scan + recovery |
| `diagnostics.json`, `diagnostics_permutation.csv` | `12_diagnostics.py` | VIF / permutation importance / OAT sensitivity |
| `correlation_report.csv`, `correlation_by_district.csv`, `lag_curves.csv` | `05_correlate.py` | driver correlation + recharge lag |
| `ablation.csv` | `10_ablate.py` | drop-group robustness sweep |

### Baseline numbers (regression gate — `gate_check.py`, latest run GATE PASS 13/13)
- Honest 30-day stride RMSE: **2.339 m** (persistence 2.382 m, +1.8%)
- Spatial CV: mean **1.973 m** / median **1.819 m**
- Effective sample size ≈ **6,635 rows** (lag-1 ACF **0.858**)
- Calibrated q05–q95 coverage: **0.908** (target 0.80) → `k = 1.0`; median half-width **1.025 m**, p90 **2.575 m**
- Trajectory v2 (honest 2026 backtest, 73,881 non-overlap windows): 30-day RMSE **trajectory 2.106 m < direct-30d 2.132 m < persistence 2.151 m** → `promote_trajectory = True`; calibrated to **0.90 coverage at every horizon**
- Fleet scan (Sep 2026): 502 stations scored, median Δ **+0.32 m**, decline 0, recovering 259.

### Verification
```bash
cd ml
venv/bin/python -m unittest discover -s tests   # 177 tests (2 skipped)
venv/bin/python gate_check.py                   # baseline regression checks (GATE PASS 13/13)
```

## 2. HTTP contract (live — `ml/api.py`, version 3.4.0)

The Flask service runs on `:5000` and exposes:

```
GET  /                                 index: service, version, endpoints, usage
GET  /health                           status, version, stations, dataset_last, ollama, forecast_model
GET  /stations                         ?district=&q=&limit=        recency-sorted station list (slugs + lat/lon)
GET  /stations/<slug>                  per-station facts — same rich object as chat "facts", no LLM call
GET  /stations/<slug>/series           6-hourly gwl + driver points for relation charts (?drivers=&from=&to=&limit=)
GET  /stations/<slug>/alerts           notification-ready zone + reasons + top drivers (?n=), no LLM call
GET  /fleet/alerts                     fleet-wide zones (?zone=&district=&sort=&limit=) — bell-icon source
GET  /districts                        district list with water-scarcity status (?sort=&limit=)
GET  /districts/<name>                 district detail + station list
GET  /fleet/recovery                   recovery/decline ranking (?district=&sort=&limit=)
GET  /forecast/<slug>                  trajectory v2 — 120×6h q05/q50/q95 (slug URL-encoded)
GET|POST /assistant/chat              station-locked LLM answer: JSON {question, station?, model?} or GET ?question=&station=&model?
Voice input is client-side (browser Web Speech API / on-device mobile STT) → text goes to `/assistant/chat` above; no audio endpoint.
```

- **URL:** `http://localhost:5000`, configurable via `ML_SERVICE_URL`; gateway forwards path + query string (see `back-end/services/mlGateway.js`).
- **Timeout:** `ML_TIMEOUT_MS` (default 60000).
- **Errors:** JSON `{ "error", "detail" }`; 400 / 404 / 500 / 502 / 503 (see `ml/README.md`).
- **Slugs:** lowercase hyphenated station names (e.g. `ashadha-prathmik-vidyalaya`) — always URL-encode; resolve current slugs from `/stations`. The exact station name also resolves as a fallback.

### Planned additions (not yet implemented)
`/models`, `/fleet/forecasts`, `/fleet/scan`.

### Behavior notes
- Data sources for each endpoint are already produced by the current artifact contract (table above).
- **No on-demand training:** the Flask service never trains at request time; models are pre-trained artifacts in `ml/models/`.
- **Forecasts:** trajectory v2 serves 120 genuine 6h steps (`/forecast/<slug>`);
  the pooled direct-30d dict (`_model.forward_forecast`) stays the +30d benchmark.
  `point` = best estimate; `lower`/`upper` = calibrated 90% interval
  (`quantile_calibration.json`); only stations in the 600-station anchor set have feature rows.
- **Series:** `/stations/<slug>/series` serves observed history (gwl + drivers)
  from the aligned 6h table — the chart source for per-feature relation views;
  never a prediction.
- Assistant requires Ollama (`ollama serve`, `llama3.2:3b`); the instant-facts panel (no LLM) always works.

## Service not running?

```bash
cd ml && venv/bin/streamlit run app.py    # Streamlit dashboard: http://localhost:8501
cd ml && venv/bin/python -m api           # Flask API: http://localhost:5000
```

## Legacy (deprecated)

The previous DB-driven Python endpoints — `POST /forecast/:stationId`, `POST /anomalies/:stationId`, `POST /risk/:unitId`, `POST /train`, `POST /models/compare` — are no longer served by the ML module. The Node gateway retains lightweight handlers reading the `model_outputs` table for backward compatibility only.


---

# Part 4 — Trajectory v2 (6-hourly forecast) spec

<!-- source: docs/ml-trajectory-v2-spec.md -->

# Trajectory v2 — genuine 6-hourly groundwater forecast (ML-only spec)

Status: **implemented, validated and promoted** (2026-09-10). The production Forecast page +
`refresh` daemon + `/forecast/<slug>` API all serve the trajectory engine (see
`ml/README.md` and the model card in §5.1.1/§6.5).

## Objective (as commissioned)

Build a *genuine* 6-hourly groundwater-level forecast trajectory for the next 30 days
(120 steps) per station, expanding the single direct-30d level forecast, with an
**explicit, evidence-based confidence/reliability label at every horizon**. Non-negotiables:

- The latest **observed** GWL is the anchor; it is never a prediction.
- No future observed GWL or drivers are used at inference or in backtest.
- 120 genuine model outputs on the 6h grid — no interpolation.
- q05 ≤ q50 ≤ q95 at every horizon; quantile band calibrated to ≥ 90% coverage per horizon.
- Confidence reflects **data/model reliability**, not interval width alone; it may decline
  with horizon; a missing driver must downgrade it.
- The production direct-30d model stays the numerical +30d benchmark; the trajectory is
  promoted only on honest backtest evidence.

## Decisions locked with the user

| Decision | Choice |
|---|---|
| Architecture | Direct multi-horizon **shared** XGBoost; horizon `h` (1..120) is an input feature |
| Recursive one-step engine | Rejected (31d-validated: RMSE 1.856 vs persistence 1.840 vs direct 1.083) |
| Open-Meteo weather | **Past-window driver features only**; also feeds driver-source reliability (≤16d FORECAST / 17–30d CLIMATOLOGY / UNAVAILABLE) |
| Backtest scope | 150 stations (SEED 42) first, then full fleet (549) in the background |
| UI | Single dark-theme card on the Forecast page (observed tail, "Forecast starts" boundary, q50 + bright q05/q95 band) |
| Freeze | all round-1 baselines frozen; gate extended to 13 checks |

## Pipeline (`ml/`)

| Step | Script | Output |
|---|---|---|
| Datasets | `30_traj_datasets.py` | `data/features_traj/{train,val}.parquet` + `prep_traj.json` (train 6.53M rows, EMA stride 8; targets set by exact-time lookup) |
| Models | `31_train_traj.py` | `models/traj_xgb_{q05,q50,q95}.json` + `traj_config.json` (33 features incl. h,h_sin,h_cos) |
| Honest backtest | `32_backtest_traj.py [n_stations]` | `outputs/traj_backtest_metrics.csv`, `traj_backtest_summary.json`, `models/traj_calibration.json` |
| Reliability tables | `33_traj_reliability.py` | `models/traj_reliability.json` (bucket rules + weights) |
| Runtime engine | `_trajectory.py` | Forecast page card + `/forecast/<slug>`; unit-tested causal contract |
| Weather | `20_openmeteo_fetch.py` | `data/cfs/openmeteo_weather_daily.parquet` (37 districts, 365d history + 16d forecast) |

## Backtest design (honest)

- 150 stations (SEED 42), anchors every 14 days, 2025-10-01 → 2026-08-20 → **non-overlap**
  forecast windows only.
- Baselines at every horizon: persistence, per-doy 6h-drift climatology, production
  direct-30d (at +30d; previous recursive 1.856 kept as reference).
- Calibration: bisection widening `s` per horizon to 90% coverage.

## Results

**Full fleet (549/600 stations, 73,881 scored windows; 150-station sample in parens)**

| horizon | traj RMSE | persistence | climatology | calibrated cov. | sign-acc |
|---|---|---|---|---|---|
| 6h | 1.147 (0.303) | 1.147 (0.303) | 1.180 (0.437) | 0.90 | 0.50 (0.47) |
| 12h | 0.992 (0.352) | 0.993 (0.354) | 1.097 (0.712) | 0.90 | 0.64 (0.62) |
| 24h | 1.133 (0.348) | 1.135 (0.348) | 1.468 (1.291) | 0.90 | 0.60 (0.59) |
| 7d | 1.157 (0.528) | 1.165 (0.538) | 6.575 (8.840) | 0.90 | 0.62 (0.63) |
| 15d | 1.602 (0.787) | 1.631 (0.839) | 13.860 (19.102) | 0.90 | 0.71 (0.72) |
| **30d** | **2.106 (1.028)** | **2.151 (1.120)** | 28.723 (39.332) | **0.90** | **0.75 (0.76)** |

- 30d reference compares on the same scope (full fleet): trajectory **2.106** < production
  direct-30d **2.132** < persistence 2.151. (Previous recursive 1.856 was measured on the
  150-station scope only — not directly comparable to fleet-wide numbers.)
  `promote_trajectory := True`.
- widening `s` ∈ [0.80, 1.28] across horizons; coverage 0.90 bang-on at every horizon.
- Note: 6h–24h sign-acc is ~0.5–0.6 — the band is informative but sub-daily *direction* reads
  are only DIRECTIONAL by design (dense/noisy stations dominate these horizons).

## Confidence framework

Per-point confidence = weighted evidence, NOT interval width alone:

```
score = 0.15·interval_quality + 0.20·perf_vs_persist + 0.20·direction
      + 0.10·width_scaled   + 0.15·station_integrity + 0.10·driver_availability
      + 0.05·anchor_ood     + 0.05·trajectory_stability
```

- HIGH: score ≥ 0.70 ∧ coverage ok ∧ station fresh ∧ driver ≥ forecast ∧ direction supported.
- DIRECTIONAL: score ≥ 0.45 ∧ station ≥ 0.4 ∧ direction supported/marginal.
- LOW: otherwise.
- Inference-time downgrades: stale reading / low recent coverage, missing or climatology-only
  drivers, anchor outside training range, oscillating trajectory — each carries a reason string.

## Gate (12/12 PASS)

New checks: trajectory promoted above persistence+direct30 (`promote_trajectory`), trajectory
`+30d` calibrated coverage = 0.90. All round-1 frozen baselines unchanged (2.338 / 1.973 /
1.819 / 0.908 / 1.025 / 6635 / 0.858).

## Provenance / leakage safeguards

- Training targets: exact-time lookup (`searchsorted`) because the aligned 6h grid has gaps —
  never positional shift across gaps.
- Anchor = last observed eligible reading; features rebuilt by `06_features.build_full`
  (`keep_na=True`, 30d horizon label) exactly like training, with no future observations.
- Backtest uses only windows whose target exists at exact 6h offsets; non-overlap by 14-day
  anchor spacing > forecast length.

## Outstanding

- Sub-daily DIRECTIONAL reads: acceptable, documented; revisit only if a denser objective
  (e.g., pumpage) ever arrives.


---

# Part 5 — `rain_api` train/serve mismatch

<!-- source: docs/ml-hydro-train-serve.md -->

# `rain_api` train/serve mismatch — definition, evidence, and what must change

**Status:** the hydrology block (`HYDRO_COLS`) is **carried but untrained**, and this
document explains one concrete reason it cannot be promoted fairly yet. Nothing
described here has been changed in the pipeline; this is an analysis of the code
as it stands.

Scope: `rain_api7/22/60` and `rain_obs7/22/60` as built by `_hydrology.rain_memory_block`.

---

## 1. Where rainfall comes from

| Layer | File | What it does to rainfall |
|---|---|---|
| Normalise | `03_merge_normalize.py` | numeric coercion; drops NaN rows. **No fill.** |
| Daily align | `04_align.py:111-123` | IDW (1/d²) over up to 3 nearest gauges within 150 km, merged onto wells with `how="left"` (`:212`) |
| 6 h align | `04b_align_6h.py:116-127` | 6 h **sum** per gauge, then IDW to the well; `how="left"` (`:125`) → an unobserved slot stays **NaN** |
| Feature build | `06_features.py:166-180` | mask = `np.isfinite(rain)` |

**Consequence:** rainfall is never forward-filled or interpolated anywhere in the
pipeline. In a batch frame a finite value *is* a real observation, and a missing
slot is NaN. That is why the batch `rain_obs*` mask can be `isfinite` — and why
the code builds it explicitly rather than assuming it, so the meaning survives any
imputation step added upstream later.

---

## 2. Definition used for **training** (batch)

`_hydrology.rain_memory_block(rain_values, mask)` called once per station over the
station's **observed history only** — `06_features.py:166-180`:

```
rain_api{τ}[t] = λ·rain_api{τ}[t-1] + rain[t],   λ = exp(-(6/24)/τ)
rain_obs{τ}[t] = (# of finite rain slots in [t-τ·4+1 .. t]) / (τ·4)
```

* **`rain_api{τ}`** — exponential memory of rainfall **known at or before the row's
  own timestamp**. For a training row anchored at `t`, that is rainfall observed
  up to `t`. No future value can enter: the row's `rain` series simply has no
  entries after `t`.
* **`rain_obs{τ}`** — observed-rain fraction of the trailing τ·4-slot window
  (28 / 88 / 240 slots for τ = 7 / 22 / 60 d).

Causality is therefore satisfied in the batch path.

---

## 3. Definition used by **serving** (future mode)

`refresh.future_drivers._rain_memory_steps:329-370`, reached only when
`feature_mode == "future"` (currently **not** active — see §5):

```
series = concat(observed_history_rain, forecast_rain[0..steps-1])
mask   = concat(isfinite(observed_history_rain), zeros(steps))   # future = UNOBSERVED
blk    = rain_memory_block(series, mask)
step k -> blk[...][n_hist + k]
```

* **`rain_api{τ}`** — the recursion is **advanced across the forecast window**. For
  forecast step *k*, the value depends on precipitation at
  `anchor + 6h, anchor + 12h, … anchor + (k+1)·6h`, i.e. rainfall **after the
  anchor**. The values come from
  `future_weather_matrix` (`future_drivers.py:437-459`): Open-Meteo forecast
  within 16 days, climatology beyond, NaN otherwise.
* **`rain_obs{τ}`** — correctly unaffected: future slots are masked unobserved, so
  coverage counts observed history only and **decays** across the horizon.

So the two families are correctly separated, but `rain_api{τ}` at serving time is
**not the same function of the anchor** as `rain_api{τ}` at training time.

---

## 4. The mismatch, precisely

For a row anchored at `t`, horizon step `k` (forecast time `t + (k+1)·6h`):

| | Training feature | Serving feature |
|---|---|---|
| Information used | rainfall ≤ `t` | rainfall ≤ `t` **plus forecast rainfall in `(t, t+(k+1)·6h]`** |
| Depends on `k`? | **No** — constant across horizons | **Yes** — advances per step |
| `rain_obs` | observed fraction ≤ `t` | identical semantics, decays per step |

**Two distinct problems, and they must not be conflated:**

1. **Schema inconsistency.** In training the column is horizon-invariant; in
   serving it varies with `k`. A model fitted on the training definition is being
   served a different function. This alone makes the feature unfair.
2. **Leakage risk if "fixed" naively.** The obvious repair — build training
   `rain_api` from what *would have been* forecast at `t` — requires
   point-in-time forecast rainfall. Using the Open-Meteo **archive** API for that
   is the trap the code already contains: `refresh/future_drivers.py:14` labels it
   "Open-Meteo (archive for backtest / forecast <=16 d for live)", and
   `ml/20_openmeteo_fetch.py:85-86` calls that archive for *past* dates. For a
   backtest that is defensible, because ERA5 at `t` is a fair mirror of an
   operational forecast. For **training** it is not: ERA5 is *realized* rainfall,
   so a training row would be fitted on rainfall that had not yet happened.
   **Using realized future rainfall as a training feature is disallowed.**

**Verdict:** the two definitions are not interchangeable, and the safe repair is
not "make training look like serving" — it is to pick a definition that is
legitimately available at both times.

---

## 5. Current exposure

`data/refresh/runtime.json` reports `"feature_mode": "flat"`, which means the
serving path replicates the anchor row across all 120 horizons
(`_trajectory.py:201-205`) and **never calls** `_rain_memory_steps`. The mismatch
is therefore **latent, not live**: the `future` path is gated behind a backtest
precisely so it cannot ship unvalidated.

It becomes live the moment `feature_mode` is flipped to `"future"`, or the moment
`HYDRO_COLS` is promoted into a trained `num_cols` and the serving feature vector
is assembled from `build_step_features`.

---

## 6. Does a suitable archive exist?

**Partly — and not sufficiently.**

* **Archived forecast JSONs exist.** `ml/data/refresh/forecast_archive/` holds
  ~7,187 files of the form `<station_key>-<anchor>.json`, written each publish
  cycle by `refresh/publish.py:145-158`.
* **But they do not contain driver values.** Each archived record stores only the
  forecast GWL quantiles per horizon, the confidence labels, and a
  `driver_source` **label** string (`"Open-Meteo forecast (<=16d)"`,
  `"climatology (beyond Open-Meteo forecast)"`). Example:
  `forecast_archive/29bd98b3d1bf483f-2026-09-06T120000.json:6-16`.
  **The numeric rainfall that produced the forecast is not persisted.**
* **No point-in-time driver store exists.** `ml/data/cfs/openmeteo_weather_daily.parquet`
  is a rolling overwrite (the fetch re-queries archive + forecast and rewrites the
  file), so past *forecast* rainfall is not recoverable from it.
* **Coverage would be thin even if it were.** The archive spans roughly Aug–Sep
  2026 in the sampled filenames, and rainfall observations exist for only ~8
  districts / ~50 gauges, so most archived forecasts have no ground truth to be
  scored against.

**Conclusion: there is no archive suitable for training `rain_api` in its
serving definition.**

---

## 7. What must change before these features can be trained and promoted fairly

Ordered; (A) is sufficient to train the block, (B) is needed to keep the
forecast-rain signal.

### Option A — pin the definition to anchor-only information *(recommended first)*

Make serving compute `rain_api{τ}` from **observed rainfall up to the anchor
only**, identical to batch.

* Change: in `_rain_memory_steps`, stop concatenating `fw["rain"]` into the API
  series (or keep the concatenation for a *separate, explicitly named* column, e.g.
  `rain_api_fwd{τ}`, and never let it carry the trained name).
* Cost: forfeits the forecast-rain signal, which was the point of "future" mode.
  The `flat` production path already accepts that loss, so this is not a
  regression against what ships today.
* Benefit: training and serving become the *same function of the anchor*;
  `rain_api` becomes horizon-invariant and the schema is consistent.

### Option B — persist a point-in-time driver archive *(enables the forecast signal)*

* Change: extend `refresh/publish.py` (and the `future_drivers` build step) to
  persist, per anchor, the **numeric per-step driver matrix actually used**
  (rain at minimum, ideally all `DRIVER_COLS`) alongside the existing forecast
  JSON.
* Then training can consume archived forecast rainfall from the same vintage the
  model saw — legitimately, with no realized-future contamination.
* Cost: a data-engineering task plus months of accumulation before there is enough
  history to train distance-to-anchor features. **Until it exists, state plainly
  that the forecast-rain form of `rain_api` is not trainable.**

### Whichever option is taken

1. **Implement the chosen definition in the shared helper**, not per-caller, so
   batch and serving cannot drift again (`_hydrology.rain_memory_block` is already
   the shared choke point).
2. **Add a parity test** asserting a serving row at the anchor equals the batch row
   at the same timestamp for every `rain_api*` / `rain_obs*` column. The existing
   `ml/tests/test_hydro_rain_obs.py` covers `rain_obs` semantics and the
   `rain_api`-uses-forcing rule; a train/serve parity assertion does not yet exist.
3. **Re-run the backtest** on the corrected definition before any promotion, on the
   honest anchor set, with the calibration fitted and reported on disjoint anchors.
4. **Only then** append `HYDRO_COLS` to `num_cols` and retrain — the refresh gate
   refuses a candidate whose `feature_cols` differ from production
   (`refresh/model_update.py:282`), so this is a deliberate, coordinated change and
   not something that can happen by accident.

---

## 8. What is *not* in question

* `rain_obs7/22/60` semantics are correct and consistent in both paths: only actual
  gauge observations count, over a fixed τ·4-slot denominator, and
  forecast/climatology values never count as observed.
* The other hydrology columns do not have an analogous train/serve problem:
  `gwl_dev*`, `gwl_slope*`, `gwl_obs*` and `gwl_imputed` are computed from
  history ≤ the anchor in **both** paths and are pinned at the anchor in serving.

---

## 9. Summary table

| Feature | Training definition | Serving definition | Fair to train? |
|---|---|---|---|
| `rain_api{τ}` | rainfall ≤ anchor, constant across h | rainfall ≤ anchor **+ forecast in-window**, varies with h | **No** — mismatch; no point-in-time forecast archive |
| `rain_obs{τ}` | observed fraction ≤ anchor | observed fraction ≤ anchor, decays with h | **Semantics correct**; still untrained |
| `gwl_dev{7,30}` | history ≤ anchor | history ≤ anchor, pinned | Yes |
| `gwl_slope{7,30,90}` | history ≤ anchor | history ≤ anchor, pinned | Yes |
| `gwl_obs{7,30,90}` | history ≤ anchor | history ≤ anchor, pinned | Yes |
| `gwl_imputed` | anchor row provenance | anchor row provenance | Yes |
