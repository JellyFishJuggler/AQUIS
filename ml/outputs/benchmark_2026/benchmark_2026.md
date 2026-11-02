# AQUIS 2026 rolling 30-day benchmark — persistence vs XGB vs DL

- Dataset: `data/aligned/table_6h.parquet`, last observation **2026-10-29 06:00:00**, cleaned with `drop_gwl_spikes`.
- Origins (anchor-causal, +30d target = anchor + 30d): 2026-01-10, 2026-02-09, 2026-03-11, 2026-04-10, 2026-05-10, 2026-06-09, 2026-07-09, 2026-08-08, 2026-09-07
- Evaluable rows: **3950** across **510** stations.
- XGB is the tracked q50 artifact loaded from an in-memory copy with `base_score=0` (repo file untouched). DL is the deployed quantile GRU.
- Errors are level errors (`prediction - actual`); persistence predicts the anchor level.

## Per origin

| origin | model | n | median bias (m) | MAE | RMSE | ΔMAE vs persistence | ΔRMSE vs persistence |
|---|---|---|---|---|---|---|---|
| 2026-01-10 | persistence | 498 | +0.100 | 0.513 | 1.109 |  |  |
| 2026-01-10 | XGB (base_score=0) | 498 | +0.011 | 0.494 | 1.095 | +3.7% | +1.2% |
| 2026-01-10 | DL (GRU) | 498 | -0.040 | 0.506 | 1.095 | +1.4% | +1.3% |
| 2026-02-09 | persistence | 498 | +0.215 | 0.561 | 1.043 |  |  |
| 2026-02-09 | XGB (base_score=0) | 498 | +0.039 | 0.498 | 0.965 | +11.2% | +7.5% |
| 2026-02-09 | DL (GRU) | 498 | +0.055 | 0.504 | 0.971 | +10.2% | +6.9% |
| 2026-03-11 | persistence | 497 | +0.058 | 0.500 | 1.178 |  |  |
| 2026-03-11 | XGB (base_score=0) | 497 | -0.111 | 0.514 | 1.193 | -2.7% | -1.2% |
| 2026-03-11 | DL (GRU) | 497 | -0.150 | 0.536 | 1.186 | -7.1% | -0.6% |
| 2026-04-10 | persistence | 494 | +0.327 | 0.536 | 0.856 |  |  |
| 2026-04-10 | XGB (base_score=0) | 494 | +0.074 | 0.407 | 0.769 | +24.2% | +10.2% |
| 2026-04-10 | DL (GRU) | 494 | +0.092 | 0.429 | 0.776 | +19.9% | +9.3% |
| 2026-05-10 | persistence | 194 | +0.333 | 0.637 | 1.016 |  |  |
| 2026-05-10 | XGB (base_score=0) | 194 | +0.080 | 0.508 | 0.885 | +20.3% | +12.9% |
| 2026-05-10 | DL (GRU) | 194 | +0.030 | 0.562 | 0.951 | +11.8% | +6.4% |
| 2026-06-09 | persistence | 495 | +0.182 | 0.491 | 0.894 |  |  |
| 2026-06-09 | XGB (base_score=0) | 495 | +0.128 | 0.468 | 0.881 | +4.8% | +1.4% |
| 2026-06-09 | DL (GRU) | 495 | +0.204 | 0.524 | 0.908 | -6.7% | -1.5% |
| 2026-07-09 | persistence | 489 | -0.214 | 1.030 | 1.533 |  |  |
| 2026-07-09 | XGB (base_score=0) | 489 | +0.009 | 1.015 | 1.438 | +1.4% | +6.2% |
| 2026-07-09 | DL (GRU) | 489 | +0.244 | 0.984 | 1.351 | +4.4% | +11.9% |
| 2026-08-08 | persistence | 447 | -1.011 | 1.763 | 2.747 |  |  |
| 2026-08-08 | XGB (base_score=0) | 447 | -0.642 | 1.485 | 2.475 | +15.8% | +9.9% |
| 2026-08-08 | DL (GRU) | 447 | -0.586 | 1.494 | 2.485 | +15.2% | +9.5% |
| 2026-09-07 | persistence | 338 | -0.281 | 0.728 | 1.095 |  |  |
| 2026-09-07 | XGB (base_score=0) | 338 | +0.109 | 0.650 | 1.014 | +10.7% | +7.4% |
| 2026-09-07 | DL (GRU) | 338 | -0.221 | 0.708 | 1.035 | +2.7% | +5.5% |

## Pooled

| model | n | median bias (m) | MAE | RMSE | ΔMAE vs persistence | ΔRMSE vs persistence |
|---|---|---|---|---|---|---|
| persistence | 3950 | +0.069 | 0.747 | 1.399 |  |  |
| XGB (base_score=0) | 3950 | -0.002 | 0.673 | 1.302 | +9.9% | +6.9% |
| DL (GRU) | 3950 | -0.028 | 0.693 | 1.299 | +7.3% | +7.2% |

## Latest evaluable origin — 2026-09-07 (closest to the current fleet season)

| model | n | median bias (m) | MAE | RMSE |
|---|---|---|---|---|
| persistence | 338 | -0.281 | 0.728 | 1.095 |
| XGB (base_score=0) | 338 | +0.109 | 0.650 | 1.014 |
| DL (GRU) | 338 | -0.221 | 0.708 | 1.035 |

Actual median 30-day change at this origin: **+0.281 m**.

## DL decline flags

| threshold (m) | flagged | actual declines | TP | precision | recall |
|---|---|---|---|---|---|
| 0.3 | 389 | 1184 | 188 | 0.483 | 0.159 |
| 0.6 | 11 | 595 | 4 | 0.364 | 0.007 |

At the latest origin:
| threshold (m) | flagged | actual declines | TP | precision | recall |
|---|---|---|---|---|---|
| 0.3 | 70 | 69 | 25 | 0.357 | 0.362 |
| 0.6 | 6 | 40 | 2 | 0.333 | 0.050 |

## Actual 2026 30-day change

| origin | median actual 30d change (m) |
|---|---|
| pooled | -0.069 |
| 2026-01-10 | -0.100 |
| 2026-02-09 | -0.215 |
| 2026-03-11 | -0.058 |
| 2026-04-10 | -0.327 |
| 2026-05-10 | -0.333 |
| 2026-06-09 | -0.182 |
| 2026-07-09 | +0.214 |
| 2026-08-08 | +1.011 |
| 2026-09-07 | +0.281 |

## Exclusions (clean table)

| origin | scored | regime shift | anchor lag >7d | sparse anchor | no actual ±1d | empty features | errors |
|---|---|---|---|---|---|---|---|
| 2026-01-10 | 498 | 0 | 36 | 2 | 4 | 1 | 0 |
| 2026-02-09 | 498 | 0 | 37 | 1 | 5 | 0 | 0 |
| 2026-03-11 | 497 | 0 | 34 | 0 | 9 | 1 | 0 |
| 2026-04-10 | 494 | 0 | 35 | 3 | 8 | 1 | 0 |
| 2026-05-10 | 194 | 0 | 44 | 3 | 300 | 0 | 0 |
| 2026-06-09 | 495 | 0 | 44 | 1 | 1 | 0 | 0 |
| 2026-07-09 | 489 | 0 | 36 | 0 | 16 | 0 | 0 |
| 2026-08-08 | 447 | 0 | 44 | 0 | 50 | 0 | 0 |
| 2026-09-07 | 338 | 0 | 60 | 5 | 138 | 0 | 0 |

- `sparse anchor` is the Luktra guard: fewer than 14 observations in the 7 days before the anchor.
- Regime-shift stations are excluded from both the forecast and the actuals (their levels carry a datum offset).
- Predictions are produced from the station's causal slice only; the anchor is never a prediction.

