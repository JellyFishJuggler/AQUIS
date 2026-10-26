# AQUIS ML module: end-to-end field guide

This guide describes the Python ML module in this directory: what it does, where
its data comes from, how it is transformed, which models it trains and serves,
where outputs are kept, how refresh works, and what the current local artifacts
do and do not prove. It is written from the checked-in code, the module README,
and the local artifacts inspected on **2026-10-25**. It does not claim that a
remote service or scheduled process is currently healthy.

Start here when you want the narrative. The concise algorithm scoreboard is in
[`README.md`](README.md); the consolidated specification is in
[`../docs/ml.md`](../docs/ml.md). The code and generated files are authoritative
when an older document disagrees.

## 1. Quick answers

| Question | Answer |
|---|---|
| What is this ML module for? | Uttar Pradesh groundwater-level telemetry inspection, 30-day forecasting, station/district/fleet summaries, and forecast verification. |
| What is the target? | Groundwater level (GWL) change, not absolute level: `GWL(t + horizon) - GWL(t)`. The displayed level is anchor level plus predicted change. |
| What time step? | The main model grid is 6-hourly (00, 06, 12, 18), four slots per day. |
| How many forecast surfaces? | Two: a pooled model that predicts one 30-day endpoint, and a trajectory model that predicts 120 horizons from 6 hours through 30 days. |
| Where does the historical GWL archive come from? | `data/processed/common.parquet`, an existing NWIC/NWDP telemetry archive. The initial pipeline selects from that local parquet; the refresh pipeline incrementally fetches new NWIC/NWDP observations. |
| Where do historical drivers come from? | NWIC/NWDP CKAN `datastore_search` resources for rainfall, weather, river, canal, and reservoir observations. The configured resource map is in `config.py`; the repository source catalogue is `../back-end/db/api/api.txt`. |
| Which future drivers are available? | Open-Meteo daily weather/rain forecasts up to 16 days, climatology beyond that, and CWC/FMIS river forecasts when covered. Canal levels have no public forecast in this module. |
| What is currently selected for serving? | The local `data/refresh/runtime.json` selects DL for both backends and `feature_mode=flat`; dispatch falls back to XGBoost if the DL path cannot load. This is local configuration, not proof of deployed runtime health. |
| Is there an active verified refresh? | The local `state.json` has a last refresh timestamp on 2026-10-12 but no success value or generated model version. Verification is called by the successful publish cycle, but this snapshot does not confirm a successful cycle. |
| Where should I look for current evidence? | `outputs/pooled_dl_eval.json`, `outputs/traj_backtest_summary*.json`, `outputs/traj_paired_ci_dl_vs_xgb.json`, `models/model_metadata.json`, `data/refresh/state.json`, and `data/refresh/runtime.json`. Always check the artifact's date and protocol. |

## 2. Scope and architecture

`ml` is a Python package-like directory, but its numbered scripts are often run
as flat scripts (for example, `python 06_features.py`) and internal modules are
imported by filename. The principal components are:

| Component | Location | Purpose |
|---|---|---|
| Source configuration and ingestion | `config.py`, `data/raw/nwic.py`, `00_probe.py`–`04b_align_6h.py` | Discover NWIC schemas, select stations, retrieve observations, clean them, and align them in space and time. |
| Feature and model research | numbered scripts, `experiments/` | Build training rows, fit pooled/tree/GRU models, compare baselines, diagnose errors, and preserve research evidence. |
| Shared prediction logic | `_model.py`, `_trajectory.py`, `_dl.py`, `_traj_dispatch.py` | Load model artifacts, construct forecast rows, choose backend, and produce forecast response objects. |
| Operational refresh | `refresh/` | Incrementally fetch live GWL, Open-Meteo weather, and CWC/FMIS river forecasts; rebuild aligned inputs, infer forecasts, publish atomically, verify outcomes, and gate retraining. |
| HTTP service | `api.py` | Flask JSON API for station, district, fleet, model, forecast, and assistant data. |
| ML showcase | `streamlit_app.py`, `app_pages/` | Read and explain existing datasets and artifacts; not the training/refresh service. |
| Tests | `tests/` | Unit and parity checks for source handling, features, inference, API contracts, schedule, and gates. |

### The two model products

1. **Pooled direct 30-day forecast (surface B).** One feature row per station
   anchor. Predicts `GWL(t+30d) - GWL(t)` and q05/q50/q95 intervals. The pooled
   XGBoost model is the incumbent reference; a daily sequence GRU ensemble is
   available and selected in local runtime configuration.
2. **6-hourly trajectory (surface A).** A long-frame contains one row per
   station, anchor, and horizon `h=1..120`. The XGBoost version is one shared
   direct multi-horizon model per quantile; the GRU version encodes recent
   station history and emits horizon-conditioned quantiles. It predicts each
   horizon directly, not by recursively feeding predicted GWL back as new input.

These surfaces are not interchangeable evaluation protocols. The pooled model
is trained on daily rows for a single 30-day target. The trajectory model is
trained/backtested on 6-hourly anchors and many horizons. Their RMSE numbers
must not be compared as if they came from the same sample.

## 3. Data sources: what, who, and how they enter

### 3.1 NWIC/NWDP observations

`config.py` holds the archive/live resource IDs and value hints for each
configured NWIC/NWDP source. `00_probe.py` probes metadata and detects the
numeric value column; `data/meta/probe.json` records the local probe result.
`data/raw/nwic.py` wraps the CKAN API. Retrieval uses `datastore_search`, not
the portal's CSV download links. The repository-side catalogue is
`../back-end/db/api/api.txt`.

| Feed family | Configured names | Use in ML | Spatial caveat |
|---|---|---|---|
| Groundwater telemetry | `gwl` | Anchor/history and supervised target source. | Station-specific; selected from the local cleaned archive for the initial build. |
| Rainfall | `rainfall` | Historical forcing, rolling sums, recharge analysis, and antecedent precipitation memory. | AWS gauges are linked to wells by up to 3 nearest gauges within 150 km, inverse-distance-squared weighted. |
| Weather | `temperature`, `humidity`, `solar`, `wind_speed`, `wind_direction`, `pressure` | Historical context and association analysis; numeric predictors include temperature, humidity, solar, wind speed, pressure. | Nearest gauge within configured radius (generally 150 km); coverage is sparse. Wind direction is not a trained numeric feature. |
| River | `river_level` | Historical context and association analysis. | If coordinates are unavailable, use Tehsil where possible, otherwise a flagged district-level average proxy. |
| Canal | `canal_level`, `canal_discharge` | Historical context / exploratory association. | Canal level uses nearest gauge within 100 km where coordinates exist; configured discharge feed had no usable coverage in selected districts in the saved report. |
| Reservoir/barrage | `res_okhla`, `res_okhla_agra_canal`, `res_matatila`, `res_ganga_rmc`, `res_ganga_1` | Exploratory historical drivers. | Selected barrage feeds had no usable coverage in the saved source report; discharge files keep one detected Gate-1 series as an approximation. |

The full-coverage selection rule is in `config.py` and `01_select.py`: a station
needs at least one 2026 observation in 8 of January–September, a first 2026
record no later than January 15, and a last record on or after August 1. The
checked-in README's data-probe snapshot says **600 of 1,353 stations** passed
across 29 districts. The training metadata in the local directory reports a
different build population (548 stations, 37 districts) because data and
artifacts were generated at different times/configurations; do not assume all
file counts describe one reproducible run without rebuilding from a frozen
input snapshot.

### 3.2 Other external sources

| Source | Script / artifact | What it contributes | Limits / use |
|---|---|---|---|
| Open-Meteo | `20_openmeteo_fetch.py`; `data/cfs/openmeteo_weather_daily.parquet` | District-centroid daily weather/rain archive and short-range forecast. Refresh config requests up to 16 forecast days. | Future values are forcing estimates; daily values are spread across the 6-hour grid. Archive values are suitable for history, but realized future archive values must not be used as point-in-time forecast features in historical backtests. |
| CWC/FMIS river bulletin | `18_cwc_river_forecast.py`; `data/cfs/river_forecast_cwc.parquet` | Parsed river forecasts, up to a short forecast window (currently 7-day logic in future-driver builder). | Used only where a matching district/gauge forecast exists. Its river-level change is applied over a station baseline rather than copying another gauge's raw datum. |
| ISRIC SoilGrids 2.0 | `07_soil.py`; `data/soil/soil_stations.csv` | Static soil texture and bulk density from station coordinates. | Thickness-weighted 0–30 cm aggregation; fetched as supplementary data. Soil features are gated out of the shipped main model after degrading validation. |
| ISRO Bhuvan LULC 50K | `08_lulc.py`; `data/meta/lulc_long.csv`, `data/soil/lulc_district.csv` | District land-cover percentages for two historical cycles; the 2011–12 cycle creates static district features. | Exploratory only; static shares are gated out of the shipped model after no benefit/degradation. API key is read from the repository `.env`; do not put secrets in this guide. |
| Ollama local model | `_assistant.py` | Turns precomputed station facts into conversational wording. | Local LLM dependency; no data-source authority and no permission to invent values. Numeric facts are prepared by Python before prompting. |

### 3.3 Source coverage snapshot

The README's saved fetch report shows 2,279,048 GWL rows for 600 wells and
limited driver coverage: rainfall 50 gauges in 8 districts; river 6 gauges in
2 districts; weather sources generally 4–6 gauges in 3 districts; canal level
17 gauges in 1 district. These are an old local report, not a live availability
check. Many wells therefore have no measured driver linked. A missing driver is
a data coverage limitation, not evidence of zero rainfall or no hydrologic
relationship.

## 4. How raw observations become model-ready data

### 4.1 One-time / research ingestion chain

The numbered scripts are deliberately staged so network pulls can be resumed
and expensive transforms can be rerun independently:

| Order | Script | Input → output | Main job |
|---:|---|---|---|
| 0 | `00_probe.py` | CKAN resource metadata → `data/meta/probe.json` | Inspect schemas/value fields and resource health. |
| 1 | `01_select.py` | `data/processed/common.parquet` → selected station/district CSV/JSON metadata | Apply the full-2026 live-station rule; write `selected_gwl_stations.csv`, `selected_districts.json`, flags, stale watchlist. |
| 2 | `02_fetch_selected.py` | Local GWL parquet + CKAN archive/live resources → `data/selected/*_selected.csv` | Extract GWL for selected wells; fetch each other configured source by selected district and time window. Uses page offsets, retries in the NWIC client, and `.done` markers to resume. |
| 3 | `03_merge_normalize.py` | Selected CSVs → `data/raw/*_norm.parquet`, `data/meta/manifest.json` | Normalize names/types, drop duplicates and invalid timestamps/values, apply caps and GWL station-wise trim, record row counts/ranges/drops. |
| 4 | `04_align.py` | Normalized source parquets → daily `data/aligned/table.parquet` plus `assoc_*.csv` | Build well-to-gauge links; aggregate observations to daily; IDW rainfall and nearest-gauge/proxy join other sources. |
| 4b | `04b_align_6h.py` | Normalized source parquets + associations → `data/aligned/table_6h.parquet` | Build the production-cadence 6-hour grid, aggregate rainfall by sum and other driver values by mean, preserve GWL observation-vs-forward-fill provenance. |
| 5 | `05_correlate.py` | Daily/6-hour aligned tables → correlation/lag CSV/HTML outputs | Calculate raw, deseasonalized, differenced associations and rainfall lag summaries. This is analysis, not training. |
| 6 | `06_features.py` | `table_6h.parquet` (+ optional static sources) → `data/features/{train,val,test}.parquet`, `prep.json` | Remove GWL outliers/regime shifts; build anchor features, label, target time, and split/purge rows. |

The source window defaults in `config.py` are 2021-01-01 through 2026-09-05.
Fetch code extends the driver extraction start by 40 days and end by one day to
cover aggregation boundaries. GWL extraction prefers the newest time in the
local archive, so a fixed configured end does not truncate a refreshed archive.

For a deliberate full research rebuild, the intended command order from `ml/`
is shown below. These commands write data and some make network requests; this
is a runbook only, and none was executed while preparing this guide:

```powershell
python 00_probe.py
python 01_select.py
python 02_fetch_selected.py
python 03_merge_normalize.py
python 04_align.py
python 04b_align_6h.py
python 05_correlate.py
python 06_features.py
python 06_train.py
python 07_evaluate.py
python 11_quantile.py
```

Supplementary source fetches (`07_soil.py`, `08_lulc.py`), future-driver
climatology (`16_future_drivers.py`), external forecast fetches
(`18_cwc_river_forecast.py`, `20_openmeteo_fetch.py`), fleet scoring, and
diagnostics are optional branches. For trajectory v2, build the one-step
6-hourly feature frame and long horizon frame with `14_6h_features.py` only if
the chosen training path requires it; the current direct trajectory dataset
builder consumes `data/features_6h/` and is invoked as `30_traj_datasets.py`,
then run `31_train_traj.py`, `32_backtest_traj.py`, and `33_traj_reliability.py`.
For GRU candidates use `34_train_traj_dl.py` (trajectory) or
`36_train_pooled_dl.py` (pooled) and their corresponding paired/backtest
scripts. Check dependencies, input dates, and artifact provenance before
replacing any saved model.

### 4.2 Retrieval and normalization details

`02_fetch_selected.py` uses CKAN `filters={"District": ...}`, limit 5,000,
offset pagination, and retains records with a parseable acquisition/date field
inside the extraction window. It fetches archive and live resource pairs. Each
source gets a `.done` marker only when rows were written; missing-source markers
in `data/selected/` prevent re-probing known-empty resources on later runs.
GWL is not re-downloaded in this initial stage: it is extracted from
`data/processed/common.parquet` for the selected station names.

`03_merge_normalize.py` writes a canonical long-form schema:
`source`, `Station`, optional `Tehsil`, `District`, `lat`, `lon`, `time`,
`value`. District/station strings are trimmed, timestamps parsed, numeric values
coerced, and `(Station,time)` duplicates removed. Per-source plausibility caps
are declared in `CAPS`. GWL is datum-referenced and is not globally clipped to
nonnegative depth; it is trimmed within each station at 0.5th/99.5th percentiles
and values shallower than −0.1 m are excluded. Rainfall values outside 0–150
mm/hour are excluded. Other source caps are documented alongside `CAPS` in the
script. The manifest records counts, coverage span, extrema, and drops.

### 4.3 Spatial linkage and aggregation

`04_align.py` creates persistent association tables so a well's gauges are
inspectable and reused by the 6-hour aligner:

* Rainfall: up to 3 gauges within 150 km, weights `1 / max(distance_km, 0.05)^2`.
* Weather: nearest eligible gauge within the weather radius (150 km default).
* River and canal: configured nearest radius (100 km); distance columns are kept
  for key sources. If gauge coordinates are missing, use exact District/Tehsil
  when possible; when Tehsil matching is unavailable, district mean is a
  clearly flagged proxy with unknown distance.
* Daily rainfall is summed; temperature, weather, level and discharge signals
  are averaged. Missing source rows remain missing after left join; rainfall is
  not forward-filled or interpolated.

The 6-hour grid is built by flooring event times into 6-hour bins. For GWL,
multiple readings in a slot become the median; sentinel absolute values over
500 are dropped. A GWL value may be carried forward for at most 12 slots (3
days). `gwl_obs=1` means actual reading, `0` means carried, and NaN means no
valid reading after the carry limit. Rain is summed within each 6-hour slot and
then IDW-combined; other drivers are averaged within slot and linked to wells.
These aggregation semantics matter: rain sums are amounts, while the other
variables are average states/rates.

### 4.4 Feature construction

`06_features.py` builds one feature record per station and 6-hour anchor. The
main trained XGBoost trajectory schema is 30 base numeric features plus horizon
fields `h`, `h_sin`, `h_cos` (33 total). The pooled XGBoost model uses the 30
base features plus station/district IDs where configured. Model configs are the
schema contract; adding/reordering features requires a deliberate retrain and
matching serving update.

| Feature group | Examples | Meaning |
|---|---|---|
| Current GWL and memory | `gwl`, `lag1`, `lag4`, `lag8`, `lag28`, `lag120` | Current anchor plus prior 6h/1d/2d/7d/30d values. |
| Rolling GWL | 7d/30d mean and standard deviation | Recent level and variability. |
| Historical rainfall | `rain_1d`, `rain_7d`, `rain_30d` | Trailing sums on the 6-hour grid. |
| Weather and water drivers | temp/humidity current and 7d mean; solar; wind; pressure; river/canal level | Associated available observations. Sparse or absent values remain missing and are handled by model-specific preparation. |
| Calendar/time | year, month/day-of-year sine/cosine, hour sine/cosine, monsoon offset | Encodes season and 6-hour slot without raw cyclic discontinuity. |
| IDs | `st_id`, `dist_id` in applicable pooled model | Numeric ordinal identifiers; station ID is not a physical spatial coordinate and does not provide principled unseen-station generalization. |
| Hydrology block | `gwl_dev*`, `gwl_slope*`, `gwl_obs*`, `gwl_imputed`, `rain_api*`, `rain_obs*` | Causal groundwater trend/state, provenance/coverage and exponential antecedent rainfall memory. Not included in the frozen trajectory XGB/GRU schema. Selected columns are included by the separate pooled daily GRU. |
| Optional static/context | Soil, LULC, extraction, future-rain expectation | Experimental/gated features; not all are trained or present. |

The hydrology functions live in `_hydrology.py` and are shared by batch and
serving builders. GWL trends are Theil–Sen slopes over observed daily medians;
they require at least 3 observed days and 50% window coverage. API rainfall
memory uses `API_t = lambda * API_(t-1) + rain_t`, with 7/22/60 day decay
constants. `rain_obs*` is observed coverage over a fixed denominator; forecast
and climatology values never count as observations.

**Known feature caveat:** `rain_api*` is anchor/history-only in the batch feature
builder but the future-driver builder can advance it with forecast/climatology
rain. Point-in-time future rain matrices were not archived for historical
training. Runtime `feature_mode=flat` avoids using future reconstructed feature
rows for the trajectory path; do not enable `future` mode or train on these
columns without resolving and validating the definition mismatch. The separate
pooled GRU includes rain API and other hydrology columns from its daily feature
builder, so its results should not be assumed to validate every future-mode
trajectory feature behavior.

### 4.5 Labels, time split, and leakage controls

For the pooled 30-day task, the label is the observed GWL value 120 six-hour
slots later minus the anchor GWL. For trajectory training, targets are looked
up by exact station/time, not positional shifting across gaps; each anchor is
expanded across horizons `h=1..120`, and target timestamp is stored as
`target_time`.

The main feature split uses anchor dates: train before 2025-10-01, validation
from 2025-10-01 through 2025-12-31, test from 2026-01-01 onward. The split then
purges each row whose **label timestamp** crosses into the next split (train
labels must remain before 2025-10-01; validation labels before 2026-01-01).
This prevents a late training anchor's future observation from leaking into
validation and likewise at the test boundary. Some saved model/evaluation
artifacts and the historical honest-baseline output predate this correction;
the README and artifact metadata call those results historical. A clean
performance claim requires regenerating features, models, and evaluations from
the current purged pipeline.

## 5. Models, baselines, and training products

### 5.1 Pooled direct-30-day model

`06_train.py` fits pooled XGBoost and Ridge over the feature rows. Target is
30-day delta; conversion to level adds the anchor. Training samples one of each
four 6-hour rows for the dense Ridge solve (roughly one slot per day); features
are already grid-derived. A time-ordered tail of training is used for XGBoost
early stopping. Model output is saved under `models/` with feature schema and
metadata; evaluation is separate (`07_evaluate.py`).

`11_quantile.py` fits pooled q05/q50/q95 XGBoost uncertainty models and writes
`xgb_q05/q50/q95.joblib` plus calibration. The existing pooled daily GRU
(`36_train_pooled_dl.py`) builds daily records from the last 6-hour point per
station/day, uses a 60-day sequence and 39 feature columns (including selected
hydrology features), predicts three quantiles with pinball loss, and averages a
3-seed ensemble (42, 840, 2056). Its honest stride comparison is
`37_backtest_pooled_dl.py` → `outputs/pooled_dl_eval.json`.

### 5.2 Trajectory model

`30_traj_datasets.py` builds train/validation long frames and `prep_traj.json`.
`31_train_traj.py` trains shared direct XGBoost quantile models for q05/q50/q95
with explicit horizon and sinusoidal horizon encoding. `34_train_traj_dl.py`
trains the GRU candidate on the same anchor/horizon targets: it encodes up to
120 previous 6-hour records, left-pads short histories with a mask, conditions
the quantile head on the requested horizon, and uses pinball loss. It does not
recursively reuse predictions as observations.

The post-training chain is:

1. `32_backtest_traj.py` evaluates a station/horizon sample against observations,
   persistence, and direct-30-day references; calibration widening is fit on
   one anchor subset and assessed on a disjoint subset.
2. `33_traj_reliability.py` builds per-horizon reliability/evidence tables used
   to turn model uncertainty and data quality into confidence labels.
3. `35_traj_paired_ci.py` compares DL and XGB by station-bootstrap confidence
   intervals at selected horizons.
4. `gate_check.py` and `refresh/model_update.py` apply promotion criteria; a
   candidate is not promoted simply because its 30-day RMSE is lower.

### 5.3 Other tested algorithms and baselines

* Persistence predicts zero change (today's observed level at every future
  horizon). It is the mandatory basic comparator.
* Per-station day-of-year climatology is another reference baseline.
* SARIMA is an offline daily, per-station benchmark; it is not in the serving
  ensemble. The hypothetical ARIMA/XGB blend did not generalize in
  leave-one-origin-out testing.
* Gaussian Process Regression was tested in a separate historical harness and
  rejected due to future extrapolation failure and cubic scaling.
* Multiple scratch MLP/feature variants were evaluated in strict temporal and
  station-held-out protocols; no candidate cleared promotion. The detailed
  corrected/invalidated experiment record is `data/scratch_exp/REPORT.md`.

The comparable saved scores and their caveats are in the README's
[algorithm table](README.md#algorithms-evaluated-xgboost--gru--sarima--gpr).

## 6. Validation: what each metric does and does not say

* **RMSE/MAE (metres):** error in predicted level after adding predicted delta
  to the anchor. RMSE penalizes large misses more. Always note horizon, time
  cadence, station population, and whether rows overlap.
* **Direction accuracy:** sign agreement of predicted and observed changes;
  does not measure size/calibration of the change.
* **Quantile coverage:** proportion of observed targets between q05 and q95.
  Coverage without interval width is incomplete; extremely wide bands can cover
  everything and still be unhelpful.
* **Stride / non-overlap subset:** reduces the dense 6-hour anchors' temporal
  overlap, but windows may remain correlated across stations/time. The number of
  rows is not automatically an independent sample count.
* **Paired station bootstrap:** resamples stations to estimate model-vs-model
  uncertainty while retaining paired predictions. A CI crossing 1 (for an
  RMSE ratio) is not decisive evidence of a win.
* **Spatial / held-out-station validation:** the scored stations must be wholly
  absent from fitting. Partitioning already-produced predictions is not a true
  station holdout. Numeric `st_id` ordinals are arbitrary for unseen sites.
* **Temporal holdout:** use later periods only after purging by label timestamp.
  Old pre-purge artifacts are not evidence for the corrected pipeline.
* **Promotion gate:** uses multiple criteria, including beating the incumbent,
  persistence, coverage, short horizons, and minimum score counts. Passing one
  metric is insufficient.

The saved GRU pooled result is promising on its own saved 2026 stride protocol;
trajectory GRU beats XGB at the endpoint but not the short horizons. Neither
fact proves robustness to a new station, a different year/regime, or the current
live data. The scratch station-CV report is notably unfavorable to new-station
generalization.

## 7. Inference and future-driver logic

### 7.1 Trajectory inference

`_traj_dispatch.active_trajectory_forecast(station)` reads
`data/refresh/runtime.json`. If `traj_backend=dl`, it attempts
`_dl.trajectory_forecast`; an import/load/inference exception logs a fallback
and uses `_trajectory.trajectory_forecast` (XGBoost). Missing/corrupt runtime
configuration defaults to XGBoost. Both engines load aligned history, validate
the anchor and model artifacts, build recent causal features, and return an
error object with a reason when station data/artifacts/history are unusable.

For each horizon, output levels are anchor plus predicted delta; quantiles are
ordered, calibration widens the interval, and reliability evidence contributes
to HIGH/DIRECTIONAL/LOW confidence plus reasons. Driver availability, station
integrity, anchor out-of-distribution status, interval quality, direction,
performance-vs-persistence, width, and trajectory stability all contribute.

### 7.2 Flat versus future feature mode

`refresh/config.py` allows `flat` or `future`. Flat repeats anchor feature
values for the direct horizon rows; future mode rebuilds future-time driver and
calendar values. In future mode, source precedence is broadly:

1. Weather/rain: Open-Meteo daily data/forecast, with each day mapped across
   6-hour slots; otherwise station/district climatology; otherwise missing.
2. River: covered CWC/FMIS forecast delta applied over the station's own
   baseline; otherwise seasonal station baseline, last observation, or missing.
3. Canal: station seasonal climatology, last observation, or missing.
4. Calendar: calculated from each future timestamp.
5. GWL anchor/history: pinned; the model does not receive future observed GWL.

In flat mode, time-varying future drivers are not introduced as model inputs.
Changing this mode is a modeling change that requires point-in-time backtesting,
especially because the archived forecast snapshots do not contain every numeric
driver matrix used at their original issue time.

### 7.3 Pooled inference

`_model.forward_forecast` is the pooled XGBoost direct-30-day path; it predicts a
single change and interval from the latest usable station anchor. Backend
selection reads `forward_backend` in `runtime.json`; the DL path is provided by
`_dl.forward_forecast`. The HTTP API's `/forecast/<slug>` endpoint is the
120-point trajectory, not the scalar pooled 30-day outlook. Keep this distinction
in clients and dashboards.

## 8. Refresh: how and when operational data changes

The refresh system is in `refresh/`. Code defaults are in `refresh/config.py`;
operator overrides live in the ignored `data/refresh/refresh_config.json` and
some environment variables. Defaults: data/forecast refresh every 6 hours at
01:00, 07:00, 13:00, 19:00 local time; retrain cadence every 24 hours in allowed
windows (02:00, 14:00); stale threshold 30 hours; up to 3 retries with linear
backoff; 8 fetch workers; Open-Meteo forecast up to 16 days. No-new-GWL-data
cycles back off at least 90 minutes rather than spinning.

High-level refresh cycle:

1. `refresh/sources.py` incrementally fetches GWL from live NWIC resources,
   Open-Meteo weather history/forecast, and CWC/FMIS river forecasts. Failures
   preserve the last good parquet and are reported rather than fabricating
   values. Other historical NWIC driver feeds are part of initial ingestion;
   this recurring source fetcher does not incrementally refresh them.
2. The pipeline rebuilds derived features/alignments and climatology with
   backup/restore protections.
3. Inference updates stations whose anchors changed using selected trajectory
   backend and source/climatology logic.
4. `refresh/publish.py` stages per-station JSON plus fleet Parquet; writes
   metadata last as the commit point; archives previous forecasts for later
   verification.
5. Verification joins archived forecasts to later actual GWL at the exact
   station and target timestamp, then contributes coverage/direction/drift
   evidence.
6. Candidate retraining builds a candidate separately, compares it with the
   incumbent on matched windows, and promotes only when all gates pass. A
   feature-schema mismatch blocks promotion. `models_frozen_until` can disable
   model writes during a declared freeze.

Useful read-only/operator commands from `ml/`:

```powershell
python -m refresh.cli status
```

Other CLI commands are `validate-config`, `dry-run`, `forecast`, `retrain`, and
`schedule` (with options shown in `refresh/cli.py`). A successful forecast cycle
automatically invokes verification; there is no standalone `verify` CLI
subcommand in the current parser. Inspect the parser before running a mutating
operation. Refresh writes local state, Parquets, archives, and potentially model
files; it is not part of this guide's read-only inspection.

### Local refresh snapshot inspected 2026-10-25

* `runtime.json`: `feature_mode=flat`, `traj_backend=dl`, `forward_backend=dl`,
  `validated=true`, but `model_version=null` and `written_at=2026-09-10`.
* `state.json`: last refresh timestamp **2026-10-12 11:07 UTC**, but
  `last_forecast_refresh_ok=null`; no forecast generation timestamp/model is
  recorded. The next due timestamp in the snapshot is already in the past.
* Last retraining attempt: **2026-10-08**, `last_retrain_ok=false`; gate error
  says candidate beat persistence and passed coverage/minimum-score checks, but
  did not beat incumbent and failed short-horizon checks.
* `per_source_last_success` is empty in that state snapshot.

This is a stale/incomplete local snapshot. It is not a successful health check
and cannot establish whether a daemon runs on a deployment host.

## 9. Serving surfaces

### 9.1 Flask API (`api.py`)

Run from this directory with `python api.py` (default port 5000; `PORT` can
override). The backend gateway may proxy these under `/ml/*`. API data is loaded
from local aligned tables and saved forecast/fleet artifacts; it is not a live
CKAN query on every request.

| Route | What it returns |
|---|---|
| `GET /` | Service identity/version and route discovery. |
| `GET /health` | Service, Ollama, and dataset recency information. |
| `GET /stations` | Station list with filters/search and recency order. |
| `GET /stations/<slug>` | Station facts and metadata. |
| `GET /stations/<slug>/series` | 6-hour GWL and supported driver series for charts. |
| `GET /stations/<slug>/alerts` | Station zone, precautions/reasons and top drivers; no LLM required. |
| `GET /fleet/alerts` | Fleet alert zones, optionally filtered/sorted. |
| `GET /districts` | District scarcity/status summaries. |
| `GET /districts/<name>` | District detail, stations, stress summaries and model info. |
| `GET /fleet/recovery` | Recovery/decline ranking. |
| `GET /fleet/forecasts` | Saved fleet forecasts with filters. |
| `GET /fleet/scan` | Fleet scan counts, horizon, timestamp/status. |
| `GET /models` | Saved model specifications/calibration metadata, not model weights. |
| `GET /forecast/<slug>?days=N` | Trajectory forecast, default/max 30 days; 4 points/day. |
| `POST /assistant/chat` | Station-pinned question and optional history; returns facts-backed answer. |

Forecast objects include anchor, trajectory values, q05/q50/q95, direction,
confidence and evidence. Missing/stale/invalid input returns structured errors;
the API has machine-readable forecast failure reasons. The assistant's local
Ollama connection can be unavailable while the numeric facts route still works.

### 9.2 Streamlit showcase

Run from `ml/` with `python -m streamlit run streamlit_app.py`. Pages cover map,
monitoring, station, water balance, quality, forecast explorer, model overview,
model trust/evaluation, features, and data/source quality. The app reads saved
artifacts; it does not train models or fetch source datasets. Some interactive
pages may invoke a local inference path when the user requests a forecast. It is
an inspection/demo surface, not the refresh daemon or production API.

### 9.3 Station assistant

`_assistant.py` computes station facts, history summaries, rainfall windows,
driver correlations, forecast summaries, district context, and precaution
signals from local aligned data and saved outputs. The user selects/pins a
station; the LLM is asked to phrase those facts, not calculate a new forecast or
create unsupported measurements. Ollama is local and can be offline; the facts
panel/API facts remain independently useful. The documented default model is
`llama3.2:3b`, configurable via `AQUIS_OLLAMA_MODEL`; voice transcription occurs
on the client and sends text to the same endpoint (no server audio/STT pipeline).

## 10. Directory map: where to find things

| Path | Contents |
|---|---|
| `data/processed/` | Source-of-truth-ish local GWL archive, especially `common.parquet`; large and ignored by Git. |
| `data/selected/` | Selected source CSVs and `.done` resume markers. |
| `data/raw/` | Normalized source Parquets plus NWIC client code. |
| `data/aligned/` | Daily table and production 6-hour table consumed by feature/inference code. |
| `data/features/` | Pooled model train/validation/test features and prep metadata. |
| `data/features_6h/`, `data/features_traj/` | 6-hour feature rows and trajectory long-frame splits. |
| `data/meta/` | Probe, selection, source manifests, associations, climatologies, station/district metadata. |
| `data/cfs/` | Open-Meteo weather and CWC river forecast Parquets. |
| `data/soil/` | Station soil and district LULC supplementary datasets. |
| `data/refresh/` | Runtime config/state, staged/current/archive forecasts, verification ledger, candidate/backups; mostly ignored by Git. |
| `data/scratch_exp/`, `data/scratch_assoc/` | Research scratch scripts, audit records, logs, screenshots; not production inputs by default. |
| `models/` | Model weights, feature schemas, calibration/reliability, and model metadata. Weight files may be binary; local GRU `.pt` files exist in this checkout. |
| `outputs/` | Evaluation scores, diagnostics, CSV/JSON/Parquet results, plots and reports. These are generated snapshots, not automatically fresh. |
| `experiments/` | Validation and baseline comparison code/results. |
| `refresh/` | Operational refresh Python package. |
| `app_pages/` | Streamlit page modules and shared page helpers. |
| `tests/` | Unit, behavior, integration-contract and backend parity test modules. |
| `venv/`, `venv312/` | Local Python virtual environments; ignored and machine-specific. |

Large data/features, refresh runtime, Python environments, caches, and some
checkpoints are Git-ignored. A checkout may therefore contain local files that
are not present in another clone; model configs and runtime payloads can also
refer to ignored checkpoints that must be deployed/copied separately.

The repository-root `requirements.txt` includes common scientific, XGBoost,
Flask, and Streamlit packages but does not pin versions and does not list every
local runtime dependency (notably PyTorch for GRU inference and a Parquet engine
such as PyArrow). The `venv/` and `venv312/` directories are machine-local, not
portable environment specifications. Refresh locking imports POSIX `fcntl`;
the operational scheduler is therefore intended for Linux/POSIX deployment
(the repository docs refer to systemd). Native Windows execution of that refresh
lock path is not established by this checkout. Soil fetching also shells out to
`curl`.

## 11. Numbered script index

| Script | Role |
|---|---|
| `00_probe.py` | Probe configured source resources/schemas. |
| `01_select.py` | Select full-2026 GWL stations and districts. |
| `02_fetch_selected.py` | Extract local GWL plus district-filtered NWIC drivers. |
| `03_merge_normalize.py` | Clean and standardize selected CSVs into Parquets. |
| `04_align.py`, `04b_align_6h.py` | Daily and 6-hour spatial/temporal alignment. |
| `05_correlate.py` | Driver correlations, deseasonalized/differenced analysis, lag curves. |
| `06_features.py` | Feature/label engineering and purged temporal split. |
| `06_train.py`, `07_evaluate.py` | Pooled XGBoost/Ridge training and evaluation. |
| `07_soil.py`, `08_lulc.py` | Supplementary static soil/LULC data fetch and aggregation. |
| `10_ablate.py`, `12_diagnostics.py` | Feature-group ablation, permutation/VIF/sensitivity diagnostics. |
| `11_quantile.py` | Pooled XGBoost quantile models and interval calibration. |
| `11_fleet.py` | Fleet-wide saved forecast scan and district aggregation. |
| `13_refresh_nwic.py` | Incremental NWIC GWL update helper used by refresh. |
| `14_6h_features.py` | Build a one-step-ahead (+6h) recursive-model frame; legacy/experimental, distinct from direct trajectory v2's multi-horizon labels. The direct trajectory dataset builder currently reads `data/features_6h/` produced by this script. |
| `16_future_drivers.py` | Driver climatology/scenario utilities. |
| `18_cwc_river_forecast.py`, `20_openmeteo_fetch.py` | Fetch/parse short-range river and weather forecasts. |
| `30_traj_datasets.py` | Build anchor × horizon trajectory datasets. |
| `31_train_traj.py`, `32_backtest_traj.py`, `33_traj_reliability.py` | Train/backtest XGBoost trajectory, calibration, reliability. |
| `34_train_traj_dl.py`, `35_traj_paired_ci.py` | Train trajectory GRU and compare against trajectory XGB. |
| `36_train_pooled_dl.py`, `37_backtest_pooled_dl.py` | Train and honestly compare pooled daily GRU. |
| `31_honest_baseline.py` | Recompute station-split summaries from existing historical prediction artifact; its input artifact is documented as pre-purge. |
| `gate_check.py` | Check stored model/artifact gates. |

There are intentional numbering gaps and a reused `31_` prefix. Run scripts by
their explicit filename; do not assume that the number alone conveys ordering.

## 12. Saved performance evidence (local snapshot)

The latest concise comparisons are in the README; the main numerical artifacts
are:

| Artifact | What it measures | Local headline |
|---|---|---|
| `outputs/pooled_dl_eval.json` | Pooled daily GRU vs pooled XGB and persistence on 2026 stride | GRU RMSE 2.082 m; XGB 2.206 m; persistence 2.187 m; GRU 90% coverage 0.925. |
| `outputs/traj_backtest_summary_dl.json` | Trajectory GRU sampled anchor/horizon backtest | 30d RMSE 1.802 m vs persistence 1.907 m; coverage 0.894; not promoted. |
| `outputs/traj_backtest_summary.json` | Trajectory XGB backtest | 30d RMSE 1.829 m vs persistence 1.907 m; not promoted. |
| `outputs/traj_paired_ci_dl_vs_xgb.json` | Paired station bootstrap by trajectory horizon | GRU beats XGB at 30d and selected medium/long horizons, not at all horizons. |
| `outputs/arima_xgb_combine.json` | Daily SARIMA/ARIMA comparison and hypothetical blend | ARIMA RMSE 0.735 m; XGB 0.657 m; blend loses to better single model in 0/27 held-out origins. |
| `data/scratch_exp/REPORT.md` | Corrected strict temporal/station-holdout scratch experiments | No model promoted; station-held-out candidates lose to persistence in reported folds. |

Useful exact pooled-GRU details from the saved artifact: on 4,355 stride rows
from 511 stations, RMSE is 2.082 m, MAE 0.751 m, bias −0.030 m, and direction
accuracy 0.732. Its paired station-bootstrap RMSE ratio is **0.944
[0.879, 0.972]** vs pooled XGBoost and **0.952 [0.906, 0.972]** vs persistence;
the confidence intervals exclude 1 in that artifact. These intervals quantify
that particular 2026 sample/protocol, not future-year or new-station guarantees.

For trajectory GRU vs trajectory XGBoost, the saved per-horizon paired bootstrap
reports GRU/XGB RMSE ratios of **1.012** at 6h, **1.003** at 12h, **0.962
[0.957, 0.998]** at 48h, **0.985** at 15d, and **0.990 [0.972, 0.996]** at
30d. The GRU therefore wins at the 30-day endpoint and some medium/long horizons,
but the overall all-horizons criterion is false because short horizons are not
better. Its 30-day calibrated coverage is 0.894 on the held-out calibration
subset. Keep this distinct from pooled-GRU metrics above.

The pooled XGBoost model metadata (`models/model_metadata.json`) reports a
training timestamp of 2026-10-07, 548 stations/37 districts, 1,629,522
train+validation rows, and 419,682 test rows. Its validation level RMSE is
0.973 m (XGB) vs 1.399 m (Ridge); that is a validation score, not the later
2026 test score. This metadata and the test snapshots are not all generated on
the same day/protocol. Recompute from an immutable data snapshot before using
these numbers for a launch, scientific claim, or promotion decision.

## 13. Current limitations and risks in interpretation

1. **Sparse drivers.** Gauge coverage is limited and uneven; correlations and
   forecasts only apply where linked driver data exists.
2. **Local time and source semantics.** Source timestamps and 6-hour bins must
   be interpreted consistently. The code floors times to the 6-hour grid; do
   not silently mix local civil time and UTC in new source integrations.
3. **Target availability.** A forecast needs an observed/usable anchor and
   enough history; stations with absent or stale telemetry may be unavailable.
4. **Datum errors/regime shifts.** A per-station spike rule and an annual-median
   regime-shift filter catch telemetry artifacts, but can exclude a real sudden
   physical change as well as a sensor/datum error. Inspect flags before
   interpreting station-level changes.
5. **Hydrology columns are not one universal model schema.** The trajectory
   frozen schemas omit the carried hydrology block; the pooled GRU includes a
   subset. Never infer training status from a column merely appearing in a
   frame or serving dictionary.
6. **Future rain mismatch.** `rain_api*` semantics differ between batch anchor
   features and future feature construction. Future mode needs careful
   point-in-time data and fresh backtests.
7. **Feature IDs and new stations.** Tree splits on arbitrary `st_id` values do
   not create robust new-station semantics. The corrected scratch station-CV
   evidence does not support a generalization claim.
8. **Old artifacts.** Target-time purge changed after some model and evaluation
   outputs were generated. Historical results should not be presented as
   corrected-pipeline test results.
9. **Promotion is separate from backend selection.** `runtime.json` may select
   DL even though an automated candidate retrain failed. Runtime backend choice
   describes which already-present artifact is attempted; it does not certify
   that a new candidate passed promotion.
10. **Stale local operational state.** Current local snapshots do not show a
    successful refresh status. Deployment health requires querying that host.

## 14. Common questions

### Does opening the Streamlit app download or train anything?

No. It reads local datasets/artifacts. A page can call forecast inference when
requested, but data fetching, training, retraining, and publication are separate
scripts/services.

### Does every station get every driver?

No. A station only gets driver values when a selected gauge can be associated
and has data for that time. Missing values are expected and are surfaced in
coverage/evidence; they should not be interpreted as zero except where the
specific feature logic explicitly defines a missing rain accumulation as zero.

### Is a 30-day forecast one 30-day prediction or 120 predictions?

It depends on the endpoint. The pooled endpoint produces a single day-30 value.
The trajectory endpoint provides 120 six-hour points, including the day-30
endpoint at step 120. The API's `/forecast/<slug>` is trajectory-shaped.

### Why predict change instead of level?

The target is `future GWL - current GWL`; adding it to the observed anchor
returns level. This gives persistence a natural zero-change baseline and avoids
making the learner explain the station datum as its main target.

### What do q05/q50/q95 mean?

They are predicted 5th, 50th, and 95th percentiles of the target distribution.
Calibration can widen intervals to improve empirical coverage. q50 is the
median estimate; q05–q95 is an uncertainty interval, not a guarantee.

### How do I know which data/model produced a forecast?

Inspect the forecast's anchor time, generation metadata, source report,
`data/refresh/meta.json`, `data/refresh/runtime.json`, and relevant model config
and calibration. The local current state has null model version fields, so do
not infer exact provenance when those fields are absent.

### Where do I start debugging a missing station forecast?

Check station presence and latest observed GWL in `table_6h.parquet`, selected
station metadata, spike/regime flags, model artifacts/config, runtime backend,
and the structured error reason from inference/API. Common reasons include no
aligned rows, no observed GWL, insufficient feature history, invalid anchor, or
missing artifacts.

### Where should an edit go?

* Source/schema acquisition: `config.py`, `data/raw/nwic.py`, and `00_probe.py`.
* Historical feature logic: `04_align.py`, `04b_align_6h.py`, `06_features.py`,
  `_hydrology.py`; keep training and serving definitions aligned.
* Model architecture/training: the matching numbered training script and its
  config writer.
* Inference/response shape: `_model.py`, `_trajectory.py`, `_dl.py`,
  `_traj_dispatch.py`; preserve backend parity.
* Scheduled updates/publication: `refresh/`; preserve atomic publish, rollback,
  freshness and promotion gates.
* UI or HTTP response: `app_pages/` or `api.py`; do not put training/fetch side
  effects into the read-only showcase.

When changing a feature or model, update its schema/config, batch builder,
serving builder, calibration/evaluation artifacts, and tests together. A field
being present in one dataframe is not enough to make it safe or active in the
served model.
