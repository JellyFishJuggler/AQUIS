# AQUIS Full-Stack Build Specification (ML-Anchored)

> **Purpose:** Single source of truth to build the complete production application
> (frontend + backend + database + ML integration + chatbot) on top of the
> **existing** `ml/` module. Every name, JSON shape, function and metric below is
> taken from the actual code (`ml/*.py`, `back-end/*`, trained artifacts and
> produced outputs). Where something is **not yet implemented** it is explicitly
> labelled `[PLANNED]` vs `[LIVE]`.
>
> Related docs: `docs/api.md` (API reference), `docs/ml-contract.md` (module↔gateway
> contract), `docs/ml-system-spec.md` (ML/deep-dive spec), `ml/MODEL_CARD.md`.

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
Streamlit app   Express API (:3000) [LIVE]    Flask ML API (:5000) [LIVE v3.0.0]
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
`Station, District, time, date, horizon, target, target_d, st_id, dist_id` + 30 floats.

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

`*` = may be NaN (XGBoost handles; Ridge median-imputes). Missing driver columns
are simply absent → feature builder skips that window group.

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
`docs/ml-system-spec.md`.

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
| Split | train <2025-10-01, val Oct–Dec 2025, test ≥2026-01-01; train 1,629,574 / val 11,647 / test 378,554 |
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
  `docs/ml-system-spec.md`).
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
`docs/ml-system-spec.md` §2.)

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
| `api.py` | Flask HTTP API v3.0.0 | `/health`, `/stations`, `/forecast/<slug>`, `/assistant/chat` |
| `app_charts.py` / `snapshot.py` | forecast chart helpers / PNG export | `trajectory_chart`, `trajectory_snapshot_png` |
| `refresh/` | fetch → features → inference → publish daemon + retrain gate | `cli.py`, `pipeline.py`, `scheduler.py`, `sources.py`, `features.py`, `inference.py`, `model_update.py`, `publish.py`, `verification.py` |
| `_model.py` | model loaders + forward forecast | `load_xgb_model`, `load_linear_model`, `load_quantile_models`, `load_predictions`, `forward_forecast`, `load_fleet_table`, `load_diagnostics`, … |
| `_utils.py` | shared loaders | `load_table_6h`, `station_recency`, `load_manifest`, `load_report`, `load_lag_curves`, `DRIVER_LABELS`, `SOURCE_LABELS` |
| `_assistant.py` | chatbot | `StationAssistant.facts/answer`, `_build_prompt`, `_forecast_summary`, `ollama_status`, `ollama_model`, `_clean_series`, `_series_stats` |
| `_soil.py` | soil loader | `load_soil`, `SOIL_COLS` |
| `_lulc.py` | lulc loader | `load_lulc`, `LULC_COLS` |
| `validation/*.py` | honest suite | `spatial_cv.py`, `overlap.py`, `residual_acf.py`, `spatial_residual.py` |
| `gate_check.py` | 12-check regression gate (latest GATE PASS 10/10) | `check(name, ok, detail)` |
| `tests/` | stdlib unittest, 151 tests (2 skipped) | `test_trajectory_v2.py`, `test_refresh_*.py`, `test_verification.py`, `test_api.py`, … |
| `MODEL_CARD.md`, `README.md` | docs | — |

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
> and the **Flask ML service is live** (`ml/api.py` v3.0.0: `/health`,
> `/stations`, `/stations/<slug>`, `/forecast/<slug>`, `/assistant/chat`).
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
151 tests (2 skipped), stdlib `unittest`, data-gated, no pytest: trajectory v2,
refresh pipeline (core/future/training/schedule), verification, Flask API,
assistant facts, calibration/diagnostics, forecast UI. Run:
`venv/bin/python -m unittest discover -s tests -v` (the full Forecast-page
AppTest is gated behind `AQUIS_APPTEST=1`).
Gate: `venv/bin/python gate_check.py` → 12 checks, latest **GATE PASS 10/10**
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