# Model Card — AQUIS pooled 30-day groundwater-level forecast (ml)

| Field | Value |
|---|---|
| Model | Pooled **XGBoost** (delta) trained once across all stations + Ridge baseline |
| Task | Forecast the **30-day change** in groundwater level, GWL(t+120) − GWL(t), on a 6-hourly grid |
| Package | `ml`: `06_train.py` → `models/xgb_multihorizon.joblib`, `linear_multihorizon.joblib` |
| Intended use | 30-day outlook per monitored station / district; whole-fleet scan (`11_fleet.py`) |
| Not for | Sub-day decisions, wells outside the 511 test stations, attribution/causal claims |

## Data coverage & sources

**Scope is determined by the telemetry archive, not by choice.** The model consumes *only* the NWIC/NWDP real-time feed — `GWL Telemetry (Six Hourly), Uttar Pradesh Ground Water Department` (UPGW). That feed wires **1,353 stations across 34 of UP's ~75 districts**. UP's remaining groundwater monitoring is **manual/quarterly** and surfaces as *separate* datasets, which this pipeline cannot forecast from in near-real-time.

Coverage facts (all verified against the local archive):

- **1,353 stations / 34 districts** = the full extent of the 6-hourly telemetry network (2021-01-01 → 2026-09-05, ~5.3M raw records in `data/processed/common.parquet`).
- **600 stations / 29 districts** = the subset still "live" in 2026 (`01_select.py`: ≥8 of 9 months Jan–Sep 2026 with a reading, first 2026 record ≤ Jan 15, last record ≥ Aug 1). 5 districts' stations all went stale.
- **34 ≠ scope target**: districts absent from the archive were probed and return `total=0` — e.g. Allahabad/Prayagraj, Amethi, Amroha, Sambhal, G.B. Nagar — even under rename aliases (`ml/data/raw/nwic.py:54`). They have no telemetry gauges at all.

Source material (why most UP groundwater monitoring is manual):

| Topic | Link |
|---|---|
| UPGW datasets on NWDP portal (Manual-Quarterly vs Telemetry-Hourly vs Telemetry-Six-Hourly) | https://www.nwdp.nwic.gov.in/organization/upgw |
| UPGW "Ground Water Level (Manual - Quarterly)" dataset page (1991–2020) | https://nwdp.nwic.gov.in/dataset/ground-water-level-manual-quarterly-upgw |
| CGWB monitoring-network page (4×/year manual nationwide; DWLR telemetry is the exception) | https://cgwb.gov.in/en/ground-water-level-monitoring |
| CGWB Guidelines on High-Frequency GW Data (2026): ~84 k wells nation-wide, ~26 k CGWB (62% dug wells, manual mode), ~39 k state manual stations | https://cgwb.gov.in/sites/default/files/2026-02/final_guidelines_on_high_frequency_gw_data.pdf |
| CGWB UP Ground Water Year Book 2022–23 (1,007 UP monitoring wells, 4×/yr) | https://cgwb.gov.in/cgwbpnm/public/uploads/documents/17032384351875462524file.pdf |

**Implication:** a station without a live 2026 telemetry stream is not forecast-able here; forecasting scope is UP-GWL *telemetry* only, while the backend's ingestion scope is NWDP-telemetry countrywide + CGWB assessment Excels.

## Data

- **Sources**: NWIC 6-hourly observed GWL (600 stations, 29 UP districts, ~2.99M rows in the 6h table), IMD/Open-Meteo reanalysis drivers (rain, temp, humidity, solar, wind, pressure), river/canal level, calendar encodings.
- **Splits**: train 2022–2025 (549 stations), validation (same period, held-out slots), test 2026 (378,554 rows, 511 stations, horizon=30 only).
- **Target**: delta 30-day change. Today's GWL is an **anchor feature** and is never itself predicted.
- **Features (30 numeric)**: gwl + 5 lags (1/4/8/28/120 steps), 2 rolling stats (7/30 d), rain 1/7/30 d, weather drivers, river/canal level, year, calendar (month/doy/hour sin+cos), monsoon flag; plus station id / district id ordinals (32 total model inputs).

## Training procedure

- Per-station alignment + spike trimming (0.5–99.5% quantile) in `06_features.py`.
- One random 6-hourly slot per station-day sampled for training (`cumcount() % 4 == 0`).
- XGBoost with early stopping on the validation split (best_iteration ≈ 15, ~few ×100 trees), objectives/min_child_weight tuned in `06_train.py`.
- Feature matrix materialized to `data/features/{train,val,test}.parquet` (reused by spatial CV, ablation, diagnostics).

## Performance (honest numbers)

| Metric | Value |
|---|---|
| Raw 2026 test RMSE (all rows) | 2.242 m |
| **Non-overlap window RMSE (headline)** | **2.339 m** (3,371 independent 30-day windows) |
| Persistence (non-overlap) | 2.382 m → margin **+1.8%** |
| Spatial CV (leave-block-out, 5 folds) | mean 1.973 m / median 1.819 m |
| Effective test sample | ≈6,611 rows (ACF-based; lag-1 residual ACF 0.858) |
| Quantile coverage (q05–q95, stride) | 0.911 vs target 0.80 → widen factor k=1.0 |
| Interval half-width | median ~1.03 m, p90 ~2.58 m (calibrated) |

Pre-row deltas below ~±0.05 m are within overlap noise; driver features are additive but marginal at 30 d (ablation, diagnostics).

## Diagnostics

- **VIF**: nothing > 10 (no collinearity redline).
- **Permutation importance (stride)**: every feature ≈ 0 Δ RMSE on independent windows; `gwl_roll30_std` + `temp_7d` lead. Seasonal/level features (`doy_cos`, `gwl`, `monsoon`) have the largest one-at-a-time swing (up to ~0.39 m).
- **Residuals**: mild positive spatial autocorrelation (Moran's I 0.042, p≈0.051); strong temporal autocorrelation is the binding constraint on sample size.

## Expected behavior & limitations

- Forecasts are **near-flat** at 30 d for most stations; real moves are rare and only meaningful when |Δ| exceeds the calibrated 90% interval width.
- Stations with missing recent drivers or a spike-flagged last reading produce unreliable forecasts (NaN anchor → delta saturation); Fleet gates these as `unreliable` (43/545).
- Static soil/LULC are fetched and shown for context but deliberately **excluded** (proven redundant with station/district ordinals: adding them raised RMSE and collapsed early stopping).
- Re-running: any change to feature engineering or training data requires `06_train` → `07_evaluate` → validation suite → regenerate quantile/fleet/diagnostics.