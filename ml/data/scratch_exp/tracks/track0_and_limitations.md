# Track 0 — future-rain driver availability (read-only probe)

**Date:** 2026-10-08 (all network probes live)

## Question
Is there a genuine *published* 1–30 day rainfall forecast usable **at issue time**
(both at serve time and, crucially, with a point-in-time archive for training)
that could be a legitimate "future rain" driver for the 30-day GWL model?

## Result: NO usable future-rain source (observed-only available)

| Source | Probe result | Horizon | Point-in-time archive of as-issued forecasts? | Verdict |
|---|---|---|---|---|
| meteoblue-style Open-Meteo `/v1/forecast` (daily `precipitation_sum`) | works, no key | 16 days max (`forecast_days=16`; `30` rejected by API) | **No** — current model run only, no historical archive of issued runs | cannot train a driver |
| Open-Meteo `archive-api` (ERA5 / ERA5-Land) | works | full past | no (this is reanalysis = **realized/observed** rain) | would leak the future → forbidden |
| NASA POWER (PRECTOTCORR, MERRA-2) | works | full past | no (analysis/observed) | would leak → forbidden |

All three network sources are reachable. Two of them return only *realized*
rainfall (reanalysis), which the user rule explicitly forbids as a future driver.
The only genuine forecast product (Open-Meteo 16-day current-run) (a) does not
reach 30 days and (b) has no stored archive of past as-issued runs, so it cannot
enter the training schema without inventing point-in-time labels.

The repo's own CFS feed (`data/cfs/openmeteo_weather_daily.parquet`,
`forecast_days=16`, fetched 2026-10-07) is the same current-run product: a
**rolling** feed, not an as-issued history.

## Consequence for the experiment
- No future-rain arm is built. `E_best` uses **trailing data only**; at serve
  time it requires no driver value from the future.
- This is stated as a known limitation: the 30-day skill ceiling is bounded by
  what the *past* 120 k values of gwl + rain + covariates can infer, not by an
  exogenous rainfall forecast.

---

# Frozen-input coverage limitation (Sep–Dec 2025 gap)

The arm-selection protocol asks for hold-out folds in **both** monsoon seasons
(Jun–Dec 2024 and Jun–Dec 2025). The frozen table
(`ml/backups/20261007_pre_ablation/aligned/table_6h.parquet`) has no usable
monsoon in 2025: Sep–Dec 2025 are ~0–15% covered (Oct 2025 ≈ 1.5%).

## Observed gwl coverage (% of a "full" month, 6 h records)

Full ≈ 72,000 records/month (≈548 stations × 4/day × 30 d). Values below are
exact counts from the frozen aligned table.

| month | 2024 records | 2024 % | 2025 records | 2025 % | 2026 records | 2026 % |
|---|---|---|---|---|---|---|
| June  | 71,704 | 99 | 41,010 | 57 | 41,859 | 58 |
| July  | 73,848 | 100 | 36,373 | 50 | 72,370 | 100 |
| August| 72,581 | 100 | 44,019 | 61 | 68,274 | 94 |
| Sep   | 70,580 | 98 | 9,902  | 14 | 50,253 | 70 |
| Oct   | 72,272 | 100 | 860    | 1 | 6,812 (partial, in progress) | – |
| Nov   | 70,213 | 100 | 4,002  | 6 | – | – |
| Dec   | 72,406 | 100 | 14,081 | 20 | – | – |

## Interpretation (known data limitation)

- Coverage collapse is **across nearly all 548 stations simultaneously**
  (e.g., Sep-2025: ~9.9k records ≈ 1 week of data; Oct-2025 ≈ 0.9k records ≈
  a few days from ≤20 stations) — i.e., a **systemic** gap, not per-sensor.
- The collapse is upstream of `ml/06_features.py` (build/drop logic): the raw
  frozen aligned table itself lacks those rows.
- The signal is ingestion/data-availability rather than model design: the feed
  pipeline restores full ~72k/month coverage from 2026-01 onward, including the
  whole 2026 monsoon (Jun–Sep 2026 ≈ 58–100%).
- Cause at the source (NWIC/gateway sensor outage vs. upstream ingestion gap)
  **cannot be determined from the frozen backup alone**; a read of the live raw
  store / feed logs would be needed. Flagged for resolution before the freeze.

## Protocol consequence (per user decision 2026-10-08)

- **Fold-2024** (full July–Dec monsoon) is the **selection fold**.
- **Fold-2025** (n=578 stride rows, no monsoon) is a **sign check only**:
  an arm must have fold-2025 point ratio ≤ E0's, with no significance claim.
- Selection rule applied: arm passes ⇔ fold-2024 paired station CI upper < 1
  AND fold-2024 gain > fold-2024 seed spread AND fold-2025 ratio ≤ E0's.
- The single final model is evaluated **once** on the 2026 test stride grid.