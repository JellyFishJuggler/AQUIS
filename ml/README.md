# ml — pooled GWL forecasting + correlation

> This is the **single** ML module (post-merge 2026-09). Big data dirs
> (`data/`, `models/*.joblib`, `outputs/`, `venv/`) are git-ignored and
> regenerable from `data/processed/common.parquet` + the numbered pipeline.
> The old per-station `ml/` (Flask API, `agent/`, `artifacts/`, `xgboost_quantile.py`)
> was merged into this module and deleted — see `merge_later` items in the merge spec.

## Setup & selection rule

* **GWL anchors** = stations with **full-2026 coverage**: >= 8 of 9 months Jan..Sep 2026
  represented, first 2026 record <= 2026-01-15, last record >= 2026-08-01.
* **600 / 1,353 stations** across **29 UP districts** qualify.
* All external driver data fetched through NWIC `datastore_search` (authoritative
  source map: `back-end/db/api/api.txt`). Per-district pulls with resume markers.

## Sources probed & fetched

| Source | Rows | Stations | Districts | Notes |
|---|---|---|---|---|
| **gwl** | 2,279,048 | 600 | 29 | From local common.parquet, per-station 0.5–99.5% quantile trim |
| **rainfall** | 42,745 | 50 | 8 | Hourly AWS → IDW 3-nearest to each well (150 km radius) |
| **river_level** | 295,302 | 6 | 2 | No published coordinates → **district-level proxy** |
| **temperature** | 107,950 | 5 | 3 | AWMS, spatial radius |
| **humidity** | 118,947 | 6 | 3 | |
| **solar** | 109,555 | 6 | 3 | |
| **wind_speed** | 122,527 | 6 | 3 | |
| **wind_direction** | 114,213 | 5 | 3 | Categorical; excluded from numeric correlation |
| **pressure** | 110,949 | 4 | 3 | |
| **canal_level** | 88,089 | 17 | 1 | Spatial radius 100 km |
| canal_discharge | — | 0 | 0 | No data within selected districts |
| reservoir discharge (×5) | — | 0 | 0 | Barrages outside selected districts |

### Honest limitation

Within the 29 selected districts, **weather/river gauge coverage is sparse**:
river 2 districts, weather ~3, rainfall 8. Many GWL stations receive no driver
features and drop from pairwise correlation. This is a real data constraint, not
a code defect.

## Key findings (correlation_report.csv)

### By |spearman| (raw)

| Driver | Spearman raw | Deseason | Diff | Stations |
|---|---|---|---|---|
| **river_level** | **0.409** | 0.168 | 0.102 | 78 |
| rain_30d | −0.199 | — | 0.013 | 75 |
| canal_level | −0.146 | −0.104 | −0.010 | 76 |
| pressure | 0.199 | **0.205** | −0.014 | 127 |
| humidity | 0.085 | **−0.158** | −0.014 | 101 |
| temp | −0.041 | **0.092** | 0.019 | 118 |
| solar | −0.055 | −0.073 | 0.004 | 101 |
| wind_speed | −0.095 | −0.006 | 0.002 | 143 |
| rain_7d | 0.012 | — | 0.023 | 325 |
| rain_1d | 0.002 | — | −0.007 | 479 |

### Recharge lag (GWL vs rainfall, Pearson median)

Best lag = **22 days** (med corr +0.136, min 180 valid days per station).
Physically consistent with monsoon recharge delay.

### Descriptions of correlation modes

* **raw**: raw daily correlation (spurious seasonal component possible).
* **deseason**: DoY-mean removed per station — seasonal artefact removed.
* **diff**: first-differenced (day-to-day changes) — eliminates long-term datum trend.

## Association methods

* Rainfall: **IDW (1/d²)**, 3 nearest gauges within 150 km, daily sums aggregated.
* Weather/river/canal: **nearest gauge** within radius; river/canal add `dist_km` column.
* River (no coords): **district-level proxy** (mean across that district's gauges).
  Flagged in README and `data/meta/assoc_*.csv`.

## Visualisation app

Streamlit explorer in `app.py` (8 pages). Run:

```bash
venv/bin/streamlit run app.py
```

Pages: Overview (KPIs + ranked drivers) · Correlation (mode/metric pickers + recharge-lag
curve) · Drivers (per-station driver overlay vs GWL) · Stations (district filter, coverage,
top-series) · **Model (metrics by horizon, RMSE bars, importances)** ·
**Forecast (2026 backtest curves + forward forecast from latest data)** ·
**Assistant (station-locked NL chat, llama3.2:3b via local Ollama — see below)** ·
Sources (manifest quality + association method + soil/extraction status).

### Assistant page (Ollama, local)

Ported from the `ml/` tool but wired to *this* app's data + model: facts from
`data/aligned/table_6h.parquet` and the 30-day outlook from the same pooled
XGBoost model as the Forecast page. Scope is **station-pinned**. The LLM only
phrases pre-computed facts (no invented numbers); the instant facts panel works
even if Ollama is down. Requires the local Ollama server + model:

```bash
ollama serve            # usually already running as a service
ollama pull llama3.2:3b # ~2 GB, one-time
```

Model override: `AQUIS_OLLAMA_MODEL` in the repo `.env` (same read pattern as
`LULC_STATISTICS_API_KEY`). `_assistant.py` = logic, `app_pages/assistant.py` = UI.

## Model pipeline (06–07)

* **Grid**: 6-hourly, production-consistent track (`04b`). GWL native 00/06/12/18;
  drivers (rain 1-min, weather 15-min, river/canal) re-binned to 6 h steps.
* **Target** = change `GWL(t+120) − GWL(t)`, i.e. the single **30-day** horizon
  (120 six-hour steps). Persistence ⇒ keep today's reading unchanged.
* **Features** (per 6h slot): GWL lags (1/4/8/28/120 steps = 6h/1d/2d/7d/30d),
  rolling mean/std (28/120 steps), driver windows (rain_1d/7d/30d, temp(+7d),
  humidity(+7d), solar, wind_speed, pressure, river_level, canal_level), calendar
  (month/doy sin+cos, **hour** sin+cos, monsoon day count), station/district ordinals.
  `gwl` (today) is the anchor feature — never a target.
* **Models**: pooled XGBoost (early stopping, val delta RMSE 1.275 m) + Ridge
  (alpha 0.1) vs persistence & per-station day-of-year climatology. Train sites
  sampled 1/4 (one slot/day) to keep Ridge's dense solve in memory — features are
  grid-derived so values are identical.
* **Split**: train < 2025-10-01, val Oct–Dec 2025, test 2026 (378k rows, 30-day
  targets at 6h cadence).
* **Data hygiene**: |gwl|>500 sentinels dropped (`04`); daily-median GWL (`04`,
  kept for correlation/overview) + 6h-median per slot (`04b`); per-station 30×MAD
  spike mask (`06`); **regime-shift flag** drops stations whose annual median jumps
  >15 m from the 2021-24 baseline (telemetry datum errors like the ~−6→−97 m
  overnight wells; 51 dropped on the 6h grid).
* **2026 test (30-day, level RMSE, 6h)**: **xgboost 2.242 m (beats persistence
  2.290 m, +2.1%)** · ridge 2.456 (−7.3%) · climatology 3.572. The 6h grid tightens
  persistence itself from 2.395 → 2.290 m. Driver value at 30 d stays marginal —
  consistent with "drivers add ~0 at h≤7".
* **Honest re-score** (`validation/*` + `10_ablate.py`, run automatically by
  `07_evaluate`): the 6h grid makes 378k test rows ~99% overlapping, so the valid
  scale of evidence is far smaller. **Non-overlapping 30-day-window ("stride")
  RMSE = 2.339 m** (vs persistence 2.382 m → +1.8%) on 3,371 independent windows;
  effective sample size ≈ 6,600 rows (ACF lag-1 = 0.86) vs 378k raw. **Spatial CV**
  (leave-block-out station retraining, 5 folds) gives 1.97 m mean / 1.82 m median —
  *below* the temporal 2.242, i.e. the 2026 temporal holdout, not spatial leakage,
  is the binding constraint. Residual Moran's I = 0.042 (p=0.051), Geary's C = 1.083
  (ns) — no strong spatial clustering of residual error. **Feature ablation**
  (drop-one-group retrain): rain (+0.03), weather (+0.05) and calendar (+0.05) add
  marginal value; river/canal (−0.005) and GWL lags (≈0) are neutral at 30 d given
  the anchor + calendar — all deltas are within overlap-adjusted noise (~±0.05 m).

## Outputs

* `outputs/correlation_report.csv` — ranked driver×metric×mode table
* `outputs/correlation_by_district.csv` — district-level pooled Spearman/Pearson
* `outputs/lag_curves.csv` — Pearson median at lag 0–30 days
* `outputs/lag_curve.html`, `outputs/gwl_vs_rain_top.html` — interactive plotly charts
* `outputs/model_metrics.csv` — 4-way benchmark (level + delta basis), 30-day horizon
* `outputs/eval_by_district.csv` / `eval_by_station.csv` — split RMSE
* `outputs/predictions_2026.parquet` — test predictions for the app (incl. `window_id`
  + `stride` for overlap-aware evaluation, and calibrated `q05_lvl`/`q95_lvl` test bands)
* `models/xgb_multihorizon.joblib`, `linear_multihorizon.joblib`, `feature_config.json`
* `models/xgb_q{05,50,95}.joblib` + `quantile_calibration.json` — `11_quantile.py`:
  pooled quantile forecasters (`reg:quantileerror`) with empirical coverage
  calibration; raw q05–q95 stride coverage 0.911 ≥ target 0.80 → widen factor k=1.0,
  median half-width ≈1.03 m (old uncalibrated ±1.96σ ≈ ±4.39 m was over-wide)
* `validation/` — P0 honesty suite: `_spatial_folds.py` (block assignment),
  `spatial_cv.py` (leave-block-out retraining → `spatial_cv_metrics.csv`,
  `spatial_cv_summary.json`, `spatial_folds.csv`), `overlap.py`
  (`honest_metrics.csv`, `station_summary.csv`, `overlap.json`), `residual_acf.py`
  (`residual_acf.json`), `spatial_residual.py` (`residual_spatial.json`)
* `outputs/ablation.csv` — `10_ablate.py` drop-group sweep (rain/weather/calendar
  positive; river/canal + GWL lags neutral at 30 d)
* `outputs/diagnostics.json` + `diagnostics_permutation.csv` — `12_diagnostics.py`
  (VIF on a mean-imputed sample, permutation importance on the stride subset, OAT
  driver sensitivity). Verdict: no VIF>10; every feature ≈0 Δ-RMSE on independent
  windows; season/level/monsoon carry the largest single-driver swing (~0.39 m).
* `outputs/fleet_forecast.csv` + `fleet_district.csv` — `11_fleet.py` whole-fleet 30-day
  scan (q50-median headline per station, quality-gated): 502 stations scored (43
  rejected: NaN anchor / stale / implausible), median Δ +0.32 m, decline 0, recovering
  259 at ±0.3 m — post-monsoon, all changes inside their 90% intervals (reset the
  snapshot with `python 11_fleet.py --force`)
* `outputs/correlation_lulc.csv` — static LULC class % × district GWL summary (exploratory)
* `data/meta/lulc_long.csv` + `data/soil/lulc_district.csv` — ISRO Bhuvan LULC 50K, 29 districts
* `data/aligned/table.parquet` / `table.csv` — daily station×day table (correlation/overview)
* `data/aligned/table_6h.parquet` — **6h station×slot table (2.99M rows / 600 stations)**,
  consumed by `06_features`
* `data/soil/soil_stations.csv` — ISRIC SoilGrids 2.0 per-station texture (background fetch)
* `MODEL_CARD.md` — lifecycle card for the pooled model (features, training, honest
  performance, limitations)
* `tests/` — forecast-validation suite (stdlib `unittest`, data-gated, no pytest):
  `python -m unittest discover -s tests -v`
* `data/meta/` — association CSVs, manifest, probe results, per-station flags

## Residual / next steps

1. Re-fetch river_level with Tehsil (already done, district proxy in use).
2. Canal discharge / reservoir discharge: genuinely absent from selected districts.
   If future API expansion covers those districts, re-run `02`.
3. **Soil**: SoilGrids REST fetch running in background (~56/600 cached so far). Once
   complete, `06_features` auto-merges sand/silt/clay/bdod (0-30 + 60-100 cm) and the
   run is replayable. Fetch resumes on restart (station-set skip by CSV);
   NullResponse profiles are skipped.
4. **Extraction**: no groundwater-extraction dataset on NWDP; CGWB dynamic assessments are
   PDF/Excel only (annual, ~1-2 yr lag). Optional district×year CSV at
   `data/soil/cgwb_extraction_district.csv` (columns: `district, year,
   net_annual_gw_draft_mcm, stage_of_development_pct`) joins via carry-forward.
5. Beating persistence at the 30-day horizon by more than ~2% (driver features are
   currently additive at 30 d but marginal). The honest re-score (2.339 m stride
   vs persistence 2.382 m, +1.8%) shows the real margin is ~2%, not ~20x larger on
   per-row terms — treat per-row RMSE deltas below ~±0.05 m as noise.
6. **Quantile uncertainty + calibration** — **DONE** (`11_quantile.py`): pooled
   q05/q50/q95 (`reg:quantileerror`) trained and empirically calibrated on non-overlap
   windows. Raw stride coverage 0.911 vs target 0.80 → `widen_factor_k = 1.0` (no
   widening needed), median half-width ~1.03 m. Forecast page + assistant now use the
   calibrated interval; band_half is per-station `(q95−q05)/2` (median ~1.0 m, p90
   ~2.6 m). Global band shipped; **station-aware bands** (per-station residual scale)
   remain a follow-up.
7. **Fleet scan + recovery ranking** — **DONE** (`11_fleet.py`): 30-day scan across the
   ≥2 y subset (596 eligible), q50-median headline + quality gate for NaN/stale
   anchors. Note: the pooled regression delta saturates to identical values on stations
   whose recent driver/lag columns are empty — the q50 median stays flat there, so fleet
   decline flags use the q50 delta (pooled kept as the `change_pooled_m` column). Result
   snapshot: 502 scored, median +0.32 m, 9→0 declines / 259 recovering at ±0.3 m, no
   station clears its 90% band. Follow-up: flood an **alerts** view for real movers
   (|Δ| > band) and monitor drift on re-scans.
8. **LULC**: Bhuvan 50K district stats fetched for all 29 districts (`08_lulc.py`,
   key in `<repo>/.env`). Class shares correlate with district GWL (fallow ↔ deeper
   GWL r≈+0.42, built-up ↔ shallower r≈−0.42, n=28) but are **kept out of the model**:
   static district constants are redundant with the st_id/dist_id ordinals (test
   XGB 30 d 2.242 → 2.288 m, best_iteration collapsed 15 → 1). Shown in the app's
   Sources page for reference. AOI-wise endpoint + per-station buffer stays unused.
9. **Soil**: ISRIC SoilGrids v2 fetched for all 580 stations (`07_soil.py`, thickness-
   weighted sand/silt/clay/bdod at 0-30/60-100 cm). Same verdict as LULC — included
   in the model, test XGB 30 d 2.242 → 2.277 m, best_iteration 15 → 2, so gated out
   (`ENABLE_SOIL_FEATURES = False`). Mapped stations shown in the Sources page.
10. **Future-rain expectation** (`ENABLE_RAIN_EXP = False` in `06_features.py`): a
   leak-free *trailing* district-day climatology of observed NWIC rainfall summed
   over the next 30 days. Offline ablation (2026 test, XGB 30 d): **2.242 → 2.226 m
   (+2.8% vs +2.1% over persistence)** — small but real, so the feature stays
   available behind the gate. CFSv2 seasonal-lite fetch is staged in `09_cfs_rain.py`
   (NCEI 6h-FLX, quarterly runs 2021–2025, ~2.4 GB). Paused: NCEI gridded services
   (NCSS/OPeNDAP) returned S3-403 and IRI/NMME became login-gated; raw per-step GRIB
   downloads (≈4 MB/file) are the working path — and since the aligned table starts
   2021-01-01, a CFS history fetch would cover **every** training row (no climo-fill).

## Important technical decisions

| Decision | Rationale |
|---|---|
| GWL values kept raw (no global [0,120] cap) | Production parquet datum matches raw NWIC values (73.6% negative = datum reference). Only per-station 0.5–99.5% quantile trim to kill telemetry spikes. |
| Daily GWL = median (not mean) after sentinel drop | Median resists residual spikes; shifts the 2026 persistence RMSE from 33.6 m → 1.6 m. |
| Regime-shift stations dropped (>15 m annual-median jump) | Persistent datum errors (e.g., −6 → −97 m overnight wells) passed spike filters because the whole year is consistently bad; dropping 49 seats the model on physical readings only. |
| Delta target GWL(t+h) − GWL(t) | De-leverages per-station level from tree fitting; persistence is exactly 0-change, so the benchmark is apples-to-apples. |
| 6-hourly grid + single 30-d horizon | Production-consistent cadence; today's GWL is an anchor feature only — day-0 is a KPI, never a prediction. |
| Train sites sampled 1/4 (one slot/day) | Same grid-derived features, 4× smaller Ridge dense solve (~7.5 GB→1.9 GB @ 1.6M train rows) — avoids OOM on 16 GB. |
| District-level proxy for river (no coords) | SW gauges publish null Lat/Long. Approximation documented; flagged `dist_km=NaN`. |
| Empty markers for absent sources | Prevents expensive re-probing of confirmed-empty districts on reruns. |
| `rain_n` / `_dist_km` excluded from correlation | Count/distance columns are not physical drivers. |
| Wind direction excluded from numeric correlation | Circular (0–360°) data requires von Mises statistics; out of scope for V1. |

---
*Module-level details, decisions, residuals, and outputs. Big dirs (`data/`, `models/`,
`outputs/`) are git-ignored and reproducible from the archived pipeline.*
