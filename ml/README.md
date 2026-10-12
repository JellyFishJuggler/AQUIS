# ml — pooled GWL forecasting + trajectory v2 (XGBoost + quantile GRU)

> This is the **single** ML module (post-merge 2026-09). Big data dirs
> (`data/processed`, `data/features*`, `outputs/*.parquet`, `venv/`) are git-ignored
> and regenerable from `data/processed/common.parquet` + the numbered pipeline.
> Models are committed as weights (`models/*.joblib`, `models/traj_*.json`) plus
> git-ignored DL checkpoints (`models/*_dl.pt`) copied out-of-band.
> The old per-station `ml/` (Flask API, `agent/`, `artifacts/`, `xgboost_quantile.py`)
> was merged into this module and deleted — see `merge_later` items in the merge spec.
> Both forecast surfaces — the **trajectory v2** engine (`_trajectory.py`, XGBoost;
> `_dl.py`, GRU) and the pooled direct-30d model — are now **served by quantile GRU**
> models with XGBoost as an automatic fallback/benchmark. See
> [Trajectory v2](#trajectory-v2) and
> [Algorithms evaluated (XGBoost / GRU / SARIMA / GPR)](#algorithms-evaluated-xgboost--gru--sarima--gpr) below.


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

## Serving

### Read-only ML showcase

The multi-page Streamlit showcase (`streamlit_app.py`, `app_pages/`) explains the
ML pipeline, existing forecasts, feature groups, model evaluation protocols, and
data coverage/limitations. It reads existing artifacts and may call the frozen
trajectory inference path when explicitly enabled on the Forecast explorer page.
It does not train models, fetch data, or write outputs. Candidate `HYDRO_COLS`
are clearly marked as carried/experimental and untrained. The evaluation page
includes a paired 2026 actual-versus-forecast view from
`outputs/predictions_2026.parquet`; each displayed metric uses only rows with
both the actual target and selected model prediction. That saved artifact
predates the target-time purge and is labelled historical until regenerated. It
also reads `data/refresh/verification.parquet` to inspect published trajectory
forecasts against the later GWL observation at the exact station/target time,
filterable by model version, horizon, and station.

Run it from this directory in an environment that has the repository's existing
Streamlit dependencies:

```bash
python -m streamlit run streamlit_app.py
```

Pages: Overview · Forecast explorer · Feature explorer · Model evaluation · Data
and limitations.

Production serving remains the Flask API in this directory (`api.py`, used by
the Node back-end gateway) and the refresh pipeline (`refresh/`,
`python -m refresh.cli status`). Pipeline outputs (`fleet_forecast.csv`,
`model_metrics.csv`, honest/ablation/diagnostics) are produced by the numbered
scripts and read via `_model.py` loaders. The Streamlit showcase is for ML
inspection and demonstration; it does not replace the mobile APK or production
serving paths.

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
`LULC_STATISTICS_API_KEY`). `_assistant.py` = logic (served via the
`/assistant/chat` endpoint).

Voice input lives on the mobile client only:
the app transcribes on-device (Android SpeechRecognizer / iOS
SFSpeechRecognizer, Hindi/English) and POSTs the resulting text to the
existing `/assistant/chat` endpoint (`POST /ml/assistant/chat` via the
backend) — the same station-locked pipeline as typed questions. No server
STT model or audio upload involved.

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
  targets at 6h cadence). Boundaries apply to the **anchor**, and every row is
  additionally **purged by its observed target time** (`target_time`): a train
  row survives only if `target_time < 2025-10-01`, a val row only if
  `target_time < 2026-01-01`. Before this, a late-September label was observed
  inside the validation window and a late-December label inside the test window,
  so early stopping selected on later-split labels. The same purge is applied to
  the trajectory long-frame (`30_traj_datasets`) and the refresh candidate build
  (`refresh/model_update`); row counts fall by ≈120 anchors × n_stations.
* **Hydrology block (carried, not yet trained)**: `HYDRO_COLS` in `06_features.py`
  — `gwl_dev{7,30}`, `gwl_slope{7,30,90}` (trailing Theil-Sen on daily medians),
  `gwl_obs{7,30,90}`, `gwl_imputed`, and the exponential antecedent-precipitation
  indices `rain_api{7,22,60}` with `rain_obs{7,22,60}`. Built once in
  `_hydrology.py` and shared verbatim with the serving path so training and
  inference cannot drift. Present in the feature frames and serving row dicts but
  deliberately **excluded from the frozen models' `num_cols`** — enabling them
  requires a deliberate retrain. In serving they are pinned at the anchor
  (state/trend) or advanced per step (rain memory). **`rain_api*` and `rain_obs*`
  mean different things**: `rain_api*` is the numeric memory term and uses the
  rainfall *forcing* available at the step (observations, then Open-Meteo
  forecast / climatology for future steps), whereas `rain_obs*` is a *coverage*
  column counting only slots backed by a real raingauge observation over a fixed
  τ·4-slot denominator — so forecast/climatology rain never counts as observed
  and coverage decays across the horizon. Rainfall is never forward-filled or
  interpolated anywhere in the pipeline, so in batch frames a finite value is a
  genuine observation (the mask is built explicitly in `06_features.build`).
  See `../docs/ml.md` §2.1 and `../docs/ml.md` §3.1.1.
* **Hydrology block is UNTRAINED, and `rain_api*` has a train/serve definition
  mismatch**: batch builds it from rainfall known at the anchor (horizon-invariant),
  while serving advances the recursion with future forecast/climatology rainfall
  (varies with `h`). Reproducing serving in training would need point-in-time
  forecast rainfall, which is **not archived** — the archived forecast JSONs keep
  GWL quantiles and a `driver_source` label, not numeric drivers. Latent today
  because `feature_mode = "flat"`. Full analysis and the two repair options:
  [`../docs/ml.md`](../docs/ml.md).
* **ML showcase app**: `streamlit_app.py` + `app_pages/` — read-only demonstration
  of the pipeline, features, forecasts and evaluation. Run with
  `cd ml && ./venv/bin/python -m streamlit run streamlit_app.py`.
* **Out-of-sample baseline diagnostic** (`31_honest_baseline.py`): recomputes the
  pooled margin over persistence **and** the interval coverage on a held-out half of
  the stations from the existing `outputs/predictions_2026.parquet`, so the headline
  numbers can be read on a fair split without rerunning the pipeline. Reads and
  recomputes only. Caveat: the feature frames are still pre-purge, so its output is
  an upper bound on honest performance.
* **Calibration is fitted out-of-sample**: `32_backtest_traj.py` and
  `refresh/backtest._calibrate` now split the anchors, fit the widening on one half
  and report coverage on the other (`coverage_calibrated_fit` = fit half, held-out
  `coverage_calibrated` = headline). The refresh promotion gate reads the held-out
  value, so coverage can no longer be satisfied by widening the band on the anchors
  being scored.
* **ARIMA + pooled-XGB combination**: `experiments/baselines/arima_compare.py` is a matched-pair
  *comparison* only — **no ensemble exists anywhere in the pipeline**. It now also
  persists per-origin predictions to `outputs/arima_vs_xgb_predictions.csv`, which
  `experiments/baselines/arima_xgb_combine.py` consumes to quantify what a blend would add,
  including a leave-one-origin-out test so fitted weights are not trusted merely
  because they help in sample.
* **Data hygiene**: |gwl|>500 sentinels dropped (`04`); daily-median GWL (`04`,
  kept for correlation/overview) + 6h-median per slot (`04b`); per-station 30×MAD
  spike mask (`06`); **regime-shift flag** drops stations whose annual median jumps
  >15 m from the 2021-24 baseline (telemetry datum errors like the ~−6→−97 m
  overnight wells; 51 dropped on the 6h grid).
* **2026 test (30-day, level RMSE, 6h)**: **xgboost 2.242 m (beats persistence
  2.290 m, +2.1%)** · ridge 2.456 (−7.3%) · climatology 3.572. The 6h grid tightens
  persistence itself from 2.395 → 2.290 m. Driver value at 30 d stays marginal —
  consistent with "drivers add ~0 at h≤7".
* **Stride evaluation** (`experiments/validation/*` + `10_ablate.py`, run automatically by
  `07_evaluate`): the dense 6h test anchors produce temporally correlated endpoint
  errors, not independent 30-day trajectory windows. The saved stride artifact
  reports **RMSE = 2.339 m** (vs persistence 2.382 m → +1.8%) for one 30-day endpoint
  per station every 120 six-hour steps. These points can still be serially
  correlated and clustered by station; the saved `n_windows` count is not an
  independent-replicate count. **All figures in this paragraph predate the
  target-time purge now present in source and have not been regenerated.** Treat
  them as historical artifact results, not a score for the corrected pipeline.
  **Spatial CV** (leave-block-out station retraining, 5 folds) reports 1.97 m mean /
  1.82 m median. This is provisional spatial evidence: held-out stations are absent
  from training, but the global numeric station ordinals are arbitrary values for
  held-out stations, not physical spatial features. Residual Moran's I = 0.042
  (p=0.051), Geary's C = 1.083 (ns) — no strong spatial clustering in that saved
  residual artifact. **Feature ablation**
  (drop-one-group retrain): rain (+0.03), weather (+0.05) and calendar (+0.05) add
  marginal value; river/canal (−0.005) and GWL lags (≈0) are neutral at 30 d given
  the anchor + calendar — all deltas are within overlap-adjusted noise (~±0.05 m).

### Deep-learning models (quantile GRU) — now the served engines

Two pooled PyTorch models (sequence GRU encoder + pinball quantile heads) replace
the XGBoost incumbents **on the served path** (`_dl.py`; dispatched by
`_model.forward_forecast` / `_traj_dispatch`; backend flags in
`data/refresh/runtime.json`). XGBoost is retained as an automatic fallback.

* **Surface (a) — trajectory DL** (`34_train_traj_dl.py` → `models/traj_dl.pt`
  + `traj_dl_config.json`): anchor-centric sequence GRU (hidden 64, AdamW lr 2e-3,
  up to 40 epochs, plateau schedule) over the last 120 observed 6-h rows on the
  station's own axis (left-zero-padded + masked), horizon-conditioned heads for
  q05/q50/q95 and horizon `h ∈ 1..120`. Same frames/target/split as
  `31_train_traj.py`, so the backtest is apples-to-apples. 30 features (`NUM_COLS`).
* **Surface (b) — pooled 30-day DL** (`36_train_pooled_dl.py` → `models/pooled_dl.pt`
  + `pooled_dl_config.json`): daily GRU (hidden 32, window 60, up to 60 epochs,
  3-seed ensemble {42,840,2056}) over a 39-column dense+seasonal frame (incl.
  hydrology slopes/devs, `rain_api*`, `*_obs*`), three pinball heads, z-scored
  inputs, `widen_k = 1.1` calibration. Target = the same `GWL(t+30d) − GWL(t)`
  delta as `06_train`'s pooled XGB.
* **Evidence** — see [Algorithms evaluated](#algorithms-evaluated-xgboost--gru--sarima--gpr).
  `35_traj_paired_ci.py` (DL-vs-XGB per-horizon station bootstrap) and
  `37_backtest_pooled_dl.py` (`outputs/pooled_dl_eval.json`) produce the honest
  numbers; `test_model_dl_parity.py`, `test_trajectory_backend_parity.py`,
  `test_trajectory_dispatch.py` guard serving parity.

## Trajectory v2 — genuine 6-hourly global forecast
The Forecast page's model. Two interchangeable engines on the same frame/target/split:
a **direct multi-horizon shared XGBoost** (`_trajectory.py`) and a **sequence GRU
with horizon-conditioned quantile heads** (`_dl.py` → `34_train_traj_dl.py`). In both,
horizon `h ∈ 1..120` is modelled explicitly, every step is a real model output (no
recursion, no interpolation), `q05 ≤ q50 ≤ q95` at every horizon. **The GRU is the
served engine**, with XGBoost as the automatic fallback/benchmark. Spec + full
results: [`../docs/ml.md`](../docs/ml.md).

| Step | Script | Output |
|---|---|---|
| Datasets | `30_traj_datasets.py` | `data/features_traj/{train,val}.parquet` + `prep_traj.json` (6.53M rows; targets by exact-time lookup, never positional shift across grid gaps) |
| Models (XGBoost) | `31_train_traj.py` | `models/traj_xgb_{q05,q50,q95}.json` + `traj_config.json` (33 features incl. `h,h_sin,h_cos`) |
| Models (GRU) | `34_train_traj_dl.py` | `models/traj_dl.pt` + `traj_dl_config.json` (30 `NUM_COLS`, hidden 64, horizon-cond. quantile heads) |
| Paired CI | `35_traj_paired_ci.py` | `outputs/traj_paired_ci_dl_vs_xgb.json` (per-horizon DL-vs-XGB station bootstrap) |
| Honest backtest | `32_backtest_traj.py [n_stations]` (`TRAJ_MODEL=xgb\|dl`) | `outputs/traj_backtest_metrics{,_dl}.csv`, `traj_backtest_summary{,_dl}.json`, `models/traj{,_dl}_calibration.json` |
| Reliability tables | `33_traj_reliability.py` | `models/traj_reliability.json` (bucket rules + evidence weights) |
| Weather | `20_openmeteo_fetch.py` | `data/cfs/openmeteo_weather_daily.parquet` (37 districts, 365 d history + 16 d forecast) |
| River | `18_cwc_river_forecast.py` | `data/cfs/river_forecast_cwc.parquet` (CWC 3-day forecasts, per district) |

- **Backtest artifact (2026):** anchors are every 14 d and metrics score point
  targets at each horizon; the 30-day forecasts from adjacent anchors are not
  non-overlapping trajectory windows. The saved **XGB** artifact reports 73,881
  anchors and 30-d RMSE **trajectory 2.106 < direct-30d 2.132 < persistence 2.151**,
  `promote_trajectory = True` (historical values; training-target purging has since
  changed in source, not regenerated). The saved **DL** artifact
  (`traj_backtest_summary_dl.json`, 19,406 anchors / 150 stations, 2025-10 → 2026-08)
  reports 30-d RMSE **1.802 vs persistence 1.907**, dir-acc 0.725,
  coverage_calibrated 0.894 (fit-half 0.901, disjoint held-out split), but
  `promote_trajectory = false` because the short-horizon gates do not clear and the
  win is 2026-only. The paired bootstrap (`traj_paired_ci_dl_vs_xgb.json`, 2000
  resamples) shows DL **beats XGB at h48/h360/h720** and ties or slightly loses at
  h6/h12 (`dl_beats_xgb_30d = true`, `all_horizons = false`).
- **Confidence:** per-point HIGH / DIRECTIONAL / LOW from **weighted evidence**
  (0.15 interval quality + 0.20 vs-persistence + 0.20 direction + 0.10 width +
  0.15 station integrity + 0.10 driver availability + 0.05 anchor OOD + 0.05 stability),
  with inference-time downgrades (stale anchor, missing/climatology-only drivers, OOD anchor,
  oscillation) each carrying a reason string.
- **Forward drivers are forecasts, not observations:** Open-Meteo days 1–16, climatology
  beyond, CWC river forecast where the district is covered (`refresh/future_drivers.py`).
- **Forecast UI:** single dark-theme card — "Forecast starts" boundary marker, observed tail,
  q50 + q05/q95 bright uncertainty band (no dot markers), direction banner, 6 metrics
  (anchor/+24h/+7d/+30d/change/confidence).
  The page references only the trajectory forecast.
- **Gate:** `gate_check.py` defines 13 checks — frozen round-1 baselines + trajectory promotion
  and `+30 d` calibrated coverage (latest run: **GATE PASS 13/13**; the recursive-model
  checks skip since the `backtest_6h_*` artifacts were removed).

## Algorithms evaluated (XGBoost / GRU / SARIMA / GPR)

Four families were benchmarked head-to-head. Two are in production — **XGBoost**
(incumbent/fallback) and **quantile GRU** (now served on both surfaces) — and two
were rejected (**SARIMA**, **GPR**). Results kept for provenance.

| Family | Where tested | Result | Verdict |
|---|---|---|---|
| **XGBoost** | surfaces (a)+(b) | trajectory v2 30-d RMSE **2.106 m** (< direct-30d 2.132 < persistence 2.151); pooled 2026 test **2.242 m** (+2.1% vs persistence) | incumbent / fallback |
| **Quantile GRU** | surfaces (a)+(b) | (b) 2026 RMSE **2.082 m**, beats E0-XGB ratio **0.944 [0.879, 0.972]** / persistence 0.952; (a) beats XGB at h360 0.985, **h720 0.990** [0.972, 0.996], ties/loses at h≤12 | **served on both surfaces** |
| **SARIMA** | daily matched-pair (10 st × 5 origins) | RMSE 0.637 vs XGB 0.617, MAE 0.335 vs 0.402, PICP90 **0.726 vs 0.968**; blend beat better single in 0/27 folds | rejected / dropped |
| **GPR** | per-station, time-only feature | pre-2026 → 2026: Tindwara RMSE **7.82 / R² −401**, Banda 2.13 / −7.6; PICP 1.00; O(n³) | rejected (interpolation only) |

### Quantile GRU — the production engine (`34_train_traj_dl.py`, `36_train_pooled_dl.py`)

* **Surface (b) pooled 30-day** — honest non-overlapping 2026 stride
  (`37_backtest_pooled_dl.py` → `outputs/pooled_dl_eval.json`; n=4,355, 511
  stations): persistence RMSE 2.187 · E0-XGB 2.206 · **DL 2.082**. Paired
  station-bootstrap: DL vs E0 ratio **0.944 [0.879, 0.972]** and DL vs persistence
  **0.952 [0.906, 0.972]** — both beat. DL bias −0.030, direction-acc 0.732,
  coverage_90 0.925 (raw 0.908). This is the cleanest DL win.
* **Surface (a) trajectory** — 2026 anchor backtest (`TRAJ_MODEL=dl 32_backtest_traj.py`
  → `traj_backtest_summary_dl.json`; 19,406 anchors, 150 stations, 2025-10 → 2026-08):
  30-d RMSE **1.802 vs persistence 1.907** (direct-30d 1.198 in this artifact), dir
  0.725, coverage_calibrated 0.894 (disjoint fit-half 0.901). `promote_trajectory = false`.
* **Per-horizon DL-vs-XGB** (`35_traj_paired_ci.py` → `traj_paired_ci_dl_vs_xgb.json`,
  2,000 station-bootstrap resamples): DL is **not** better at h6 (ratio 1.012) or
  h12 (1.003); it wins from ~h48 (0.962 [0.957, 0.998]), h360 (0.985), and the
  30-day endpoint h720 (**0.990 [0.972, 0.996]**). `dl_beats_xgb_30d = true`,
  `dl_beats_xgb_all_horizons = false`.
* **Training config:** (a) `models/traj_dl_config.json` — sequence GRU hidden 64,
  30 `NUM_COLS`, 120-step 6-h window, horizon-conditioned quantile heads.
  (b) `models/pooled_dl_config.json` — daily GRU hidden 32, window 60, 39-col
  dense+seasonal frame, 3-seed ensemble, `widen_k = 1.1`.
* **Serving/parity:** `_dl.py` (`forward_forecast`, `trajectory_forecast`,
  `preview`, `clear_caches`); guarded by `test_model_dl_parity.py`,
  `test_trajectory_backend_parity.py`, `test_trajectory_dispatch.py`.
* **Caveat:** the 30-day DL win is a 2026-only signal and short horizons remain
  XGB's edge, so the promotion gate stays conservative.

### SARIMA — `experiments/baselines/arima_compare.py` → `outputs/arima_vs_xgb.csv`, `outputs/arima.md`

* Protocol: **daily means, 10 stations, 5 rolling 2026 origins, 30-day horizon**
  — matched pairs where possible, and **not** comparable to the 6-hourly level
  RMSE tables above.
* `SARIMAX(1,1,1)×(1,0,1,7)` refit **per origin** on univariate daily history
  (≥180 clean days, gaps ≤3 d interpolated), forecast 30 d with 90% intervals;
  matched against the frozen trajectory XGB on the same origins.
* Row-weighted all-days aggregate: ARIMA RMSE **0.637**, MAE **0.335**, PICP90
  **0.726**, width 0.888 · XGB RMSE 0.617, MAE 0.402, PICP90 0.968, width 2.503.
  XGB wins on RMSE + coverage; SARIMA on MAE only.
* Per-station: ARIMA wins on stable low-variance stations (Tindwara RMSE 0.028)
  but collapses on volatile ones (Banda 5.91); XGB's advantage is coverage +
  robustness, not every station.
* **Blend gives nothing out of sample:** `arima_xgb_combine.py`
  leave-one-origin-out — mean blend RMSE 0.501 vs best single 0.401, and the blend
  beat the better single in **0/27 folds**. **No ensemble exists in the pipeline**;
  the module was removed (kept only as `outputs/arima.md` + CSVs).

### GPR — commit `cd07868`, harness once in repo-root `ml-gpr/`

* Per-station **Gaussian Process Regression** with a single `time_hours`
  feature (periodic×RBF kernel), trained batch over all stations
  (`ml/training/train_gpr.py` in that commit).
* In-sample fit is fine (train R² ≈ 0.95), but as a frozen forecaster it
  **extrapolates catastrophically** — pre-2026 train → 2026 daily means:

  | Station | GPR RMSE / R² | SARIMA RMSE | XGB RMSE |
  |---|---:|---:|---:|
  | Inter college, Tindwara | 7.82 / −401 | 0.028 | 0.056 |
  | Banda Badokhar Khurd | 2.13 / −7.6 | 5.91 | 1.97 |

* Why it fails: a time-only GP reverts to its training mean outside the
  training span; predictive variance inflates instead (PICP90 = 1.00 — bands
  cover everything, uninformative). Fit is **O(n³)** — ~85 s at 1.6k rows,
  ~6 min at 3k rows/station (`n_restarts=0`), so no path to 600 pooled
  stations either. Rejected.
* Numbers: `ml-gpr/artifacts/gpr_vs_arima_xgb.json`,
  `ml-gpr/artifacts/gpr_metrics.json` (harness removed with `ml-gpr/`).

## Refresh pipeline (`refresh/`)

The production-facing near-real-time loop. A systemd daemon
(`python -m refresh.cli schedule`) polls on a **fixed 6-hourly wall-clock grid**
(01:00 / 07:00 / 13:00 / 19:00 IST — `refresh/schedule.py`, `refresh_at_hours`
in `refresh_config.json`) and runs fetch → feature/align → inference → publish,
re-arming ~24 h model updates:

| Module | Responsibility |
|---|---|
| `refresh/config.py` | paths + `refresh_config.json` (intervals, grid hours, gates) |
| `refresh/state.py` | `state.json` atomic writes (PID-unique tmp + `os.replace`), corrupt auto-re-arm, journal `journal.jsonl`, refresh lock |
| `refresh/schedule.py` | `next_grid_due()` — next fixed local-time slot (with rolling-interval fallback) |
| `refresh/sources.py` | NWIC incremental fetch (`13_refresh_nwic`, per-station max), Open-Meteo, CWC |
| `refresh/features.py` | align chain + climatology rebuild, backup/restore guarded |
| `refresh/inference.py` | per-station active trajectory forecast for changed anchors (GRU via `_dl`, XGB fallback), publish-or-abort gate |
| `refresh/publish.py` | staged per-station JSON + `forecasts.parquet` + `meta.json` last (commit point), freshness verdict (`data_status` fresh/stale/unknown) |
| `refresh/model_update.py` | drift-gated retrain (trajectory v2 + reliability) — 24 h cadence, memory-bounded |
| `refresh/verification.py` | score archived anchors against realised readings, sign-accuracy ledger + drift detection |
| `refresh/cli.py` | `refresh.cli fetch|forecast|schedule|status|verify` |

Runtime state lives in `data/refresh/` (git-ignored): `state.json`,
`journal.jsonl`, `forecasts/`, `forecast_archive/`, `refresh_config.json`.

The refresh verification ledger remains an ML runtime artifact; the Streamlit
showcase pages documented above do not expose a separate Verification page.

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
* `models/pooled_dl.pt` + `pooled_dl_config.json`, `models/traj_dl.pt` +
  `traj_dl_config.json` + `traj_dl_calibration.json` + `traj_dl_reliability.json` —
  the DL (quantile GRU) engines now served on both surfaces (`*.pt` are git-ignored;
  configs are committed)
* `outputs/pooled_dl_eval.json` — surface (b) honest 2026-stride DL-vs-E0-XGB-vs-persistence
* `outputs/traj_paired_ci_dl_vs_xgb.json` — surface (a) per-horizon DL-vs-XGB paired bootstrap CI
* `outputs/traj_backtest_metrics_dl.csv` + `traj_backtest_summary_dl.json` + `traj_backtest_rows_dl.parquet` — surface (a) DL backtest artifacts
* `models/xgb_q{05,50,95}.joblib` + `quantile_calibration.json` — `11_quantile.py`:
  pooled quantile forecasters (`reg:quantileerror`) with empirical coverage
  calibration; raw q05–q95 stride coverage 0.908 ≥ target 0.80 → widen factor k=1.0,
  median half-width ≈1.03 m (old uncalibrated ±1.96σ ≈ ±4.39 m was over-wide)
* `experiments/validation/` — P0 honesty suite: `_spatial_folds.py` (block assignment),
  `spatial_cv.py` (leave-block-out retraining → `spatial_cv_metrics.csv`,
  `spatial_cv_summary.json`, `spatial_folds.csv`), `overlap.py`
  (`honest_metrics.csv`, `station_summary.csv`, `overlap.json`), `residual_acf.py`
  (`residual_acf.json`), `spatial_residual.py` (`residual_spatial.json`)
* `outputs/ablation.csv` — `10_ablate.py` drop-group sweep (rain/weather/calendar
  positive; river/canal + GWL lags neutral at 30 d)
* `outputs/diagnostics.json` + `diagnostics_permutation.csv` — `12_diagnostics.py`
  (VIF on a mean-imputed sample, permutation importance on the stride subset, OAT
  driver sensitivity). The saved stride points are temporally correlated, not
  independent windows; season/level/monsoon carry the largest single-driver swing
  (~0.39 m) in that artifact.
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
* `tests/` — forecast-validation suite (stdlib `unittest`, data-gated, no pytest).
* `data/meta/` — association CSVs, manifest, probe results, per-station flags
* `outputs/traj_backtest_metrics.csv` + `traj_backtest_summary.json` — saved trajectory v2
  2026 anchor backtest + per-horizon calibration (see the protocol and provenance
  caveats above; artifacts have not been regenerated after the target-time purge)
* `data/cfs/` — Open-Meteo daily weather + CWC river-forecast snapshots (trajectory drivers)

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
   currently additive at 30 d but marginal). The saved historical stride result
   (2.339 m vs persistence 2.382 m, +1.8%) is provisional and predates the
   target-time purge; it is not a re-score of the corrected code. Treat per-row
   RMSE deltas below ~±0.05 m as noise until a clean evaluation is run.
6. **Quantile uncertainty + calibration** — **DONE** (`11_quantile.py`): pooled
   q05/q50/q95 (`reg:quantileerror`) trained and calibrated on the saved stride
    sample (not an independent calibration holdout). Raw stride coverage 0.908 vs target 0.80 → `widen_factor_k = 1.0` (no
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
    over the next 30 days. Offline ablation (2026 test, XGB 30 d): **2.242 → 2.226 m**
    (+2.8% vs +2.1% over persistence) — small but real, so the feature stays
    available behind the gate. The CFSv2 seasonal-lite fetch (`09_cfs_rain.py`) has
    been **removed** — NCEI gridded services returned S3-403 and Open-Meteo replaced
    the driver feed.
11. **Trajectory v2** — **shipped** (see [Trajectory v2](#trajectory-v2)): the Forecast
    page presents the genuine 120-step trajectory with evidence-based confidence.
    The **refresh daemon** now runs the full fetch→inference→publish cycle on a fixed
    6-hourly grid and publishes `data/refresh/forecasts/` + `meta.json` (freshness UI
    reads the verdict). Outstanding: (a) promote the trajectory build via a wet refresh
    so `runtime.json` reports `model_version` (feature_mode is `"flat"`); (b) revisit
    sub-daily DIRECTIONAL-only reads if a denser objective (e.g., pumpage) ever arrives.

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
