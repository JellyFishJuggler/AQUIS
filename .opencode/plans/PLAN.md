# AQUIS ML Module — Build Plan from Scratch (Multi-Step Accuracy Focus)

## Current State
- `ml/data/processed/common.parquet` exists (336k rows, 93 stations, 2021-2025)
- `ml/requirements.txt` has all dependencies
- `ml/venv/` exists
- Back-end data: district-level annual rainfall, recharge, extraction (static features)
- All other ML code deleted

---

## Phase 1: Preprocessing (`ml/preprocessing/timeseries.py`)

### 1.1 Data Loading & Cleaning
- `load_and_clean(parquet_path)`: read parquet, parse "Data Acquisition Time", drop duplicates, sort chronologically per station
- Save cleaned data as `ml/data/processed/cleaned.parquet`

### 1.2 Station Slug Generation
- `station_slug(station_name, agency, sl_no)`: return `f"{station_name}_{agency}_{sl_no}"` (case-sensitive, includes unique SlNo)
- Apply to all 93 stations — verify no collisions

### 1.3 Time Index
- `build_time_index(df)`: add `time_hours` = hours since station's first reading

### 1.4 Gap Detection
- `detect_gaps(df, threshold_hours=72)`: flag gaps >72h, return gap metadata — DO NOT interpolate

### 1.5 Sentinel Value Detection (IQR-based)
- `detect_sentinel_values(df)`: for each station, compute diff of GWL, flag values where |diff| > Q3 + 3*IQR or < Q1 - 3*IQR
- Also flag repeated identical values (10+ times in a row)
- Return mask of valid rows per station

### 1.6 Static Exogenous Features (from back-end)
- `load_exogenous_features(district_mapping)`: join district-level annual data (rainfall, recharge, extraction stage) from `back-end/db/data.csv` as static station features

### 1.7 Time Split
- `time_split(df, train_frac=0.8)`: chronological 80/20 split per station, NO random shuffle

### 1.8 Full Pipeline
- `full_pipeline(parquet_path, station_slug)`: returns dict with train/test DataFrames, feature arrays, metadata, exogenous features

---

## Phase 2: Model — Hybrid Direct + Recursive (`ml/models/xgboost_quantile.py`)

### 2.1 Feature Engineering (Enhanced for Multi-Step)
- **Base**: `time_hours` + seasonal (sin_doy, cos_doy, sin_doy_2, cos_doy_2, year) + causal lags (lag_1, lag_2, lag_3, lag_4, lag_7, lag_14, lag_28, lag_60, lag_120) + rolling (roll_7, roll_28, roll_60, roll_120)
- **Trend**: linear trend coefficient over last 28/60 days
- **Static exogenous**: district rainfall, recharge_total, extraction_stage, irrigation_extraction (from back-end)
- All lag/rolling features causal only (past values)

### 2.2 Hybrid Modeling Strategy
| Horizon | Approach | Models |
|---------|----------|--------|
| 1-7 days | **Direct** (separate model per day) | 7 × (point + q05/q50/q95) |
| 8-30 days | **Direct** (bucketed: 8-14, 15-21, 22-30) | 3 × (point + q05/q50/q95) |
| 31-90 days | **Recursive** (single model, feeds back) | 1 × (point + q05/q50/q95) + **error-correction head** |

### 2.3 Direct Multi-Step Models
- `train_direct_models(train_df, horizons)`: trains independent XGBoost for each horizon/bucket
- Features include horizon-specific lags (e.g., for h=14, lag_14 becomes lag_0 equivalent)
- Quantile models per horizon: `objective='reg:quantileerror'`, `quantile_alpha=[0.05, 0.5, 0.95]`

### 2.4 Recursive Model + Error-Correction Head
- Base recursive: standard XGBoost quantile (as before)
- **Error-correction head**: Lightweight model (Ridge/RandomForest) trained on recursive residuals vs. horizon depth
  - Input: `[horizon_depth, recursive_point_pred, recent_lags, seasonal]`
  - Target: `actual - recursive_point_pred` (residual)
  - Applied at inference: `corrected_point = recursive_point + correction_head.predict(...)`

### 2.5 Model Definitions (Shared Params)
- `n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.5, random_state=42, n_jobs=-1`
- Quantile: `objective='reg:quantileerror'`, `quantile_alpha=[0.05, 0.5, 0.95]`

### 2.6 Artifact Structure
```
ml/artifacts/<station_slug>/
├── direct/
│   ├── h1_point.joblib, h1_q05.joblib, h1_q50.joblib, h1_q95.joblib
│   ├── h2_point.joblib, ...
│   ├── h3_point.joblib, ...
│   ├── h4_point.joblib, ...
│   ├── h5_point.joblib, ...
│   ├── h6_point.joblib, ...
│   ├── h7_point.joblib, ...
│   ├── h8_14_point.joblib, ...
│   ├── h15_21_point.joblib, ...
│   └── h22_30_point.joblib, ...
├── recursive/
│   ├── xgb_point.joblib, xgb_q05.joblib, xgb_q50.joblib, xgb_q95.joblib
│   └── error_correction_head.joblib
├── features.json
└── xgboost_metadata.json
```

---

## Phase 3: Interval Calibration (`ml/services/interval_calibration.py`)

### 3.1 Calibration Estimation (Per Approach)
- `estimate_calibration_direct(cfg, models, station_df, horizons, alpha=0.90)`: held-out test per direct horizon
- `estimate_calibration_recursive(cfg, models, station_df, alpha=0.90, smooth_span=7)`: recursive held-out test
  - Compute per-depth absolute error (depth = step index from anchor)
  - Alpha-quantile per depth → forward-fill → rolling-median smooth (span=7) → monotonic non-decreasing (running max)
  - Return `IntervalCalibration(half_widths, alpha)` with `half_width_at(depth)`

### 3.2 Critical Bug Prevention
- Capture `anchor_pos = len(buffer)` BEFORE any `predict()` call that mutates shared buffer

### 3.3 Widening (Unified)
- `widen(calibration, time_hours, point, lower, upper, anchor_pos, is_direct=False, horizon=None)`:
  - Union of raw interval + symmetric band around **POINT estimate**
  - `cal_half = calibration.half_width_at(depth)` (recursive) or `calibration.half_width_at_horizon(horizon)` (direct)
  - `lower = min(raw_lower, point - cal_half)`
  - `upper = max(raw_upper, point + cal_half)`

### 3.4 One-stop Loader
- `calibrate_and_widen(station_slug, time_hours, artifacts_root, alpha=0.90)`: routes to direct/recursive based on horizon, returns calibrated prediction

---

## Phase 4: Fleet Diagnosis (`ml/scripts/diagnose_fleet.py`)

### 4.1 Per-Station Metrics (Both Approaches)
- One-step (h=1): RMSE, MAE, R²
- Direct multi-step (1-30d): RMSE, MAE, R² per horizon/bucket
- Recursive multi-step (31-90d): RMSE, MAE, R² + **error-corrected** metrics
- Calibrated coverage: direct (per horizon) + recursive (per depth)

### 4.2 Reliability Classification
- **reliable**: coverage ≥ 75% at all horizons, shallow_error/GWL_span < 0.15, corrected R² > 0 at 90d
- **directional**: coverage OK but corrected R² ≤ 0 at 90d, or wider intervals
- **weak**: fails coverage floor at any horizon

### 4.3 Output
- `ml/artifacts/multistep_diagnosis.csv`: station, slug, label, reason, coverage_1d, coverage_7d, coverage_30d, coverage_90d, rmse_1d, rmse_7d, rmse_30d, rmse_90d, rmse_90d_corrected, r2_1d, r2_7d, r2_30d, r2_90d, r2_90d_corrected, shallow_error, gwl_span, n_obs, ...

### 4.4 Background Execution
- `setsid PYTHONPATH=/home/srijan/Downloads/Developement/AQUIS ml/venv/bin/python -u -m ml.scripts.diagnose_fleet > /tmp/diagnose_fleet.log 2>&1 & disown`

---

## Phase 5: Training CLIs

### 5.1 Single Station (`ml/training/train_forecast.py`)
- `--station SLUG`, `--no-seasonal`, `--no-lags`, `--no-exogenous`, `--no-direct`, `--no-error-correction` flags
- Trains direct (1-30d) + recursive (31-90d) + error-correction head

### 5.2 Batch Train All (`ml/training/train_all_forecast.py`)
- Iterate all 93 stations, resume support (progress/failure tracking)
- Parallelizable per station (use joblib/ProcessPoolExecutor)
- Report total time, failure count, per-station timing

### 5.3 Model Comparison (`ml/training/compare_models.py`)
- XGBoost (hybrid) vs Random Forest (recursive) vs Persistence vs Naive Seasonal
- 3 representative stations (different hydrogeology)
- Output CSV with one-step & multi-step metrics

---

## Phase 6: Inference CLI (`ml/inference/predict_forecast.py`)

- `--station SLUG --horizon DAYS` (1-90)
- Auto-routes: direct (1-30d) or recursive+correction (31-90d)
- Returns calibrated point + 90% PI in original GWL units

---

## Phase 7: Streamlit App (`ml/app/streamlit_app.py`)

### 7.1 Layout
- Station dropdown with stats (points, date range, gaps, sentinels, exogenous features)
- **Panel 1**: "Forecast — Test Period" (one-step backtest + 90% band)
- **Panel 2**: "Forecast — Next 2-3 Months" 
  - 1-30d: direct predictions (higher accuracy)
  - 31-90d: recursive + error-correction (calibrated bands)
  - Visual separator at 30d mark

### 7.2 Plot Handling
- Break lines at gaps >72h (no misleading interpolation)

### 7.3 Reliability Badge
- Green/Yellow/Red — based on diagnosis.csv label

### 7.4 KPIs & Labels
- One-step RMSE/MAE/R²: "reliable short-range (1-14 day) accuracy"
- 7-day / 30-day direct R²: "validated multi-step accuracy"
- 90-day recursive: "directional trend only — levels uncertain" + corrected R² if available
- Show error-correction status (active/inactive)

### 7.5 Train Button
- For stations without artifacts; shows elapsed time per component

### 7.6 Model Info Panel
- Build timestamp, feature list, calibration status, modeling approach per horizon

---

## Phase 8: Verification Checklist

1. `python -m py_compile` all files — no errors
2. `streamlit.testing.v1.AppTest` — 0 exceptions (select, retrain, predict)
3. **Multi-step accuracy targets**:
   - 1-day R² > 0.5 (median)
   - 7-day direct R² > 0.3 (median)
   - 30-day direct R² > 0.1 (median)
   - 90-day recursive+correction R² > -0.5 (median) — improvement over -1.14
   - Calibrated coverage ~90% at all horizons
4. Slug uniqueness verified
5. Sentinel exclusion works (Agra City metrics reasonable)
6. Exogenous features loaded and used

---

## Phase 9: Git

- `.gitignore`: `ml/venv/`, `ml/data/staging/`, `__pycache__/`, `*.pyc`, `/tmp/`, `back-end/db/centralreport_ingres.zip`
- Commit all new files
- Push to `github.com/JellyFishJuggler/AQUIS`

---

## Execution Order

| Phase | Files | Est. Time |
|-------|-------|-----------|
| 1 | preprocessing/timeseries.py | 45 min |
| 2 | models/xgboost_quantile.py | 90 min |
| 3 | services/interval_calibration.py | 60 min |
| 4 | scripts/diagnose_fleet.py | 45 min + 30-120 min background |
| 5 | training/train_forecast.py, train_all_forecast.py, compare_models.py | 90 min |
| 6 | inference/predict_forecast.py | 30 min |
| 7 | app/streamlit_app.py | 90 min |
| 8 | Verification | 45 min |
| 9 | Git | 10 min |

**Total active coding: ~7-8 hours** + background diagnosis run

---

## Key Architecture Decisions (Multi-Step Accuracy)

| Decision | Rationale |
|----------|-----------|
| **Hybrid direct/recursive** | Direct avoids error accumulation for 1-30d; recursive only for 31-90d where data is sparse |
| **Error-correction head** | Learns systematic drift pattern from recursive residuals; low compute, meaningful R² gain |
| **Enhanced features** | Longer lags (60, 120), 2nd harmonic seasonal, trend coeff, static exogenous (district rainfall/recharge/extraction) |
| **Bucketed direct horizons** | 7 daily + 3 weekly buckets = 10 direct models vs 90 — manageable compute |
| **Calibration per approach** | Direct horizons need per-horizon calibration; recursive needs per-depth |
| **Static exogenous from back-end** | District-level annual rainfall/recharge/extraction as station features — no time-series exogenous available |

---

## Data Mapping: Station → District (for exogenous features)

Need to create a mapping from station (in NWIC data) → district (in back-end/db/data.csv). Stations have `District` column; back-end has `DISTRICT`. Join on normalized district name.

If no match: use state-level (UP) averages as fallback.