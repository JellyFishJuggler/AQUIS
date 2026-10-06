# Daily SARIMA vs pooled trajectory XGBoost

2026 rolling origins: 5–5 per station; stations: 10.

| Model | MAE | RMSE | PICP (90%) | Mean interval width |
|---|---:|---:|---:|---:|
| ARIMA | 0.502 | 1.947 | 0.697 | 0.920 |
| XGB | 0.493 | 0.885 | 0.964 | 2.551 |

ARIMA choice: SARIMAX(1,1,1)×(1,0,1,7), because weekly seasonality is practical on daily data.
Short gaps up to 3 days were interpolated; origins with longer history gaps or missing future labels were skipped.
Stations the frozen XGB pipeline refuses (yearly-median regime shift > 15 m, telemetry datum error) were excluded from selection — no matched pair can exist there.
ARIMA history rule: >= 180 clean days (no gap > 3d) immediately before each origin; the 365-day rule was unusable because no station has 365 clean days before a 2026 origin.

- MAE winner: xgb (0.493 m).
- RMSE winner: xgb (0.885 m).
- Coverage/width: ARIMA 69.7%/0.920 m; XGB 96.4%/2.551 m; likely reflects pooled driver/trajectory information versus a single-station weekly seasonal model.
