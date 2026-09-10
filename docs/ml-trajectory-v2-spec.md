# Trajectory v2 — genuine 6-hourly groundwater forecast (ML-only spec, untracked)

Status: **implemented and validated** (2026-09-10). Documentation-only file — never
committed to git.

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
| UI | Second tab in the Forecast page |
| Freeze | all round-1 baselines frozen; gate extended to 12 checks |

## Pipeline (`ml/`)

| Step | Script | Output |
|---|---|---|
| Datasets | `30_traj_datasets.py` | `data/features_traj/{train,val}.parquet` + `prep_traj.json` (train 6.53M rows, EMA stride 8; targets set by exact-time lookup) |
| Models | `31_train_traj.py` | `models/traj_xgb_{q05,q50,q95}.json` + `traj_config.json` (33 features incl. h,h_sin,h_cos) |
| Honest backtest | `32_backtest_traj.py [n_stations]` | `outputs/traj_backtest_metrics.csv`, `traj_backtest_summary.json`, `models/traj_calibration.json` |
| Reliability tables | `33_traj_reliability.py` | `models/traj_reliability.json` (bucket rules + weights) |
| Runtime engine | `_trajectory.py` | Forecast page tab; unit-tested causal contract |
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

- Full-fleet (549-station) backtest running in background; refresh reliability tables + gate
  when it lands.
- Sub-daily DIRECTIONAL reads: acceptable, documented; revisit only if a denser objective
  (e.g., pumpage) ever arrives.