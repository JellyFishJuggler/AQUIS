# AQUIS DL-GRU performance against hydrology thresholds

**Assessment date:** 2026-11-02  
**Scope:** saved 2026 pooled 30-day benchmark and saved 6-hourly trajectory-GRU backtest. These are different evaluation products and should not be combined into one score.

## Summary

The trajectory GRU is strong at 1 day on the saved sample, remains satisfactory by the supplied NSE/R² heuristic at 7 days, and drops below the supplied NSE/R² and range-normalized-error heuristics by 15–30 days when assessed station by station. Its 30-day RMSE is within the supplied 1–2 m seasonal margin. In the newer pooled 30-day benchmark, overall RMSE is 1.299 m and MAE is 0.693 m; analysis of its saved paired rows gives median station NSE/R² 0.345 and median station-range-normalized RMSE 27.27%, so the fleet aggregate hides weak station-wise skill.

These are screening heuristics, not universal acceptance rules. Hydrological performance ratings depend on the application, time scale, observations, and decision cost. The familiar NSE > 0.50 “satisfactory” rating comes from watershed simulation guidance; it should not automatically be treated as a groundwater production certification. Recent groundwater work has used different thresholds and reports substantial well-to-well variation. See [Moriasi et al. (2007)](https://www.ars.usda.gov/research/publications/publication/?seqNo115=197994), [a groundwater-specific AI error assessment](https://jkwra.or.kr/articles/xml/z4yW/), and a [critical review of hydrological performance criteria](https://hess.copernicus.org/articles/27/2397/2023/hess-27-2397-2023.html).

## Results

Trajectory results below are computed from `outputs/traj_backtest_rows_dl.parquet` (saved 2026-10-27; evaluation period 2025-10-01–2026-08-20). For station-wise summaries, stations need at least 10 scored observations at that horizon; 137 stations qualify. The report uses median station NSE/R² and median station RMSE divided by that station’s observed range. This prevents the large differences in absolute groundwater datum between stations from inflating a pooled score.

| Forecast horizon | Rows | Median station NSE / R² | Median station RMSE / observed range | Pooled RMSE (m) | Pooled MAE (m) | Assessment against supplied heuristics |
|---|---:|---:|---:|---:|---:|---|
| 1 day (24 h) | 2,209 | 0.967 | 5.12% | 0.786 | 0.130 | Passes NSE/R² > 0.5 and range-normalized RMSE < 10%; RMSE is above the 0.2–0.5 m short-term band. |
| 7 days (168 h) | 2,133 | 0.772 | 14.13% | 1.402 | 0.294 | Passes NSE/R² > 0.5; misses range-normalized RMSE < 10% and the 0.2–0.5 m short-term band. |
| 15 days (360 h) | 2,053 | 0.475 | 20.90% | 1.355 | 0.432 | Median station NSE/R² is below 0.5 and normalized error is above 10%. |
| 30 days (720 h) | 2,023 | 0.132 | 26.93% | 1.802 | 0.660 | Median station NSE/R² is below 0.5 and normalized error is above 10%; RMSE falls within the supplied 1–2 m seasonal margin. |

The short-term absolute-error band is not met at 1 or 7 days. At 1 day, MAE is 0.130 m but RMSE is 0.786 m, indicating a smaller typical absolute error alongside some larger misses. Do not interpret MAE alone as satisfying a threshold stated for errors generally.

### Pooled 30-day model

The rolling benchmark in [`benchmark_2026.md`](benchmark_2026.md) evaluates nine origins from 2026-01-10 through 2026-09-07, with 3,950 station-origin rows across 510 stations. Paired row-level predictions are in `rows.csv`. The deployed pooled DL-GRU has RMSE **1.299 m**, MAE **0.693 m**, and median bias **−0.028 m**. It reduces RMSE by **7.2%** and MAE by **7.3%** versus persistence in the fleet-pooled calculation. Its overall RMSE is within the user-supplied 1–2 m seasonal margin, but station-wise scores are weaker:

| Model | Stations (≥5 origins) | Median station NSE / R² | 10th–90th percentile NSE | Median station RMSE / observed range | Stations NSE > 0.5 | Stations range-RMSE < 10% |
|---|---:|---:|---:|---:|---:|---:|
| DL-GRU | 500 | 0.345 | −0.594 to 0.693 | 27.27% | 31.8% | 1.4% |
| XGBoost | 500 | 0.368 | −0.390 to 0.680 | 26.43% | 33.0% | 1.2% |
| Persistence | 500 | 0.142 | −0.500 to 0.541 | 30.32% | 13.6% | 0.8% |

Per-station median RMSE/MAE for DL-GRU are **0.693/0.531 m**; the station median RMSE is lower than persistence for 77.4% of these stations, but lower than XGBoost for only 45.2%. This reinforces “modest overall baseline skill, mixed station-level performance,” not a universal GRU win. Each station contributes only 5–9 origins in this file, so station NSE and especially observed-range-normalized RMSE are noisy descriptive statistics, not stable site certification.

The separate saved 3-seed pooled evaluation (`outputs/pooled_dl_eval.json`) reports RMSE 2.082 m and MAE 0.751 m over 4,355 rows / 511 stations, with a paired RMSE ratio versus persistence of 0.952 (95% bootstrap interval 0.906–0.972). It is a different evaluation snapshot and should not be substituted for the rolling benchmark above.

## Metric definitions and interpretation

- **NSE:** `1 − Σ(observed − predicted)² / Σ(observed − mean(observed))²`, computed separately for each station and horizon. A score of 1 is perfect; 0 matches the station’s observed-mean reference; a negative value is worse than that reference.
- **R²:** reported here using the same coefficient-of-determination form, `1 − SSE/SST`, on the same station/horizon rows. Under this definition, it equals NSE. Some software instead reports squared correlation; those definitions are not interchangeable. The table therefore does not claim an independent R² result.
- **Range-normalized RMSE:** per-station `RMSE / (max(observed) − min(observed)) × 100%`, then the median across stations. This is more relevant to the proposed 10% rule than dividing by a pooled range across differently-datumed wells.
- **RMSE/MAE:** RMSE is higher than MAE at each horizon because squaring errors gives more weight to large misses. No universal pass threshold for this ratio was supplied or applied.

## Important limits

1. The trajectory file is a historical saved backtest, not a fresh evaluation through the latest 2026-10-29 observation. The latest pooled benchmark is newer, but only its 30-day aggregate metrics are available here.
2. Trajectory anchors are spaced 14 days apart and forecast windows overlap; they are not independent samples. The saved summary explicitly records `no_overlap_windows: false`.
3. The trajectory evaluation has 137 stations with at least 10 rows per reported horizon, fewer than the 150 stations in the overall saved run. Station medians can hide weak wells: at 30 days the 10th-percentile station NSE is −2.025, while the median is 0.132.
4. `NSE > 0.75` as “production ready,” `R² > 0.8` as “robust,” RMSE under 10% of range, and the absolute-depth bands are supplied comparison rules for this assessment. They do not establish production suitability on their own. Review per-station failures, calibration/coverage, datum shifts, persistence skill, and consequences of forecast errors before operational decisions.

## What to improve next

- Extend the rolling pooled benchmark to preserve paired per-station predictions so NSE/R², station-specific normalized RMSE, and error quantiles can be calculated on the same current origins.
- Report station-level distributions and worst-performing wells alongside fleet averages; investigate the stations below the NSE 0.5 and range-normalized RMSE 10% cutoffs.
- Re-run the trajectory backtest using the current leakage-audited protocol and a newer frozen data snapshot before making a current production-performance claim.
