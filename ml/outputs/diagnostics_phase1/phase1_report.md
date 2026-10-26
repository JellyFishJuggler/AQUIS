# ML diagnostics: Phase 1

Scope: read-only analysis of the saved artifacts named in the request. No model was retrained and no files under `models/`, `outputs/` outside this directory, or `data/refresh/` were written. New tables are in this directory.

## Assumptions and limits

- Error is prediction minus observation; RMSE, MAE and median absolute error use paired non-null rows. Skill is `1 - RMSE(model)/RMSE(persistence)` on the same rows. Positive skill means lower RMSE than persistence.
- Seasons are assigned by forecast target time: `time + horizon days` for `predictions_2026.parquet` and `anchor_t + horizon_h` for the trajectory artifact. Pre-monsoon = March-May, monsoon = June-September, post-monsoon = October-February.
- Driver linkage is station-level: a station has a linked driver when any saved 2026 test feature has `rain_obs7`, `rain_obs22`, or `rain_obs60` greater than zero. These columns represent observed rain-gauge coverage. This is an empirical test-frame proxy, not an independent station-to-gauge registry.
- `pooled_dl_eval.json` contains aggregate metrics only. The requested artifact set has no row-level pooled DL predictions, so its season, district, station distribution, median-error distribution, driver split, and top contributors cannot be computed from that JSON.

## Aggregate results

| source | model | n | n_stations | rmse | persistence_rmse | skill_score | mae | median_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled_dl_eval.json | Pooled daily DL | 4355 | 511 | 2.0823 | 2.1872 | 0.0480 | 0.7507 |  |

The `group_metrics.csv` table contains per-horizon, per-season, per-district and driver-group results available from row artifacts.

### Horizon breakdown: 30-day level predictions (days)

| source | model | horizon | n | rmse | persistence_rmse | skill_score | mae | median_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictions_2026 | XGBoost | 30.0000 | 419682 | 2.2594 | 2.3209 | 0.0265 | 0.7721 | 0.3378 |
| predictions_2026 | Ridge | 30.0000 | 419682 | 2.5354 | 2.3209 | -0.0924 | 1.3343 | 0.8660 |

### Horizon breakdown: trajectory backtest (hours)

| source | model | horizon_h | n | rmse | persistence_rmse | skill_score | mae | median_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traj_backtest_rows_dl | Trajectory DL q50 | 6.0000 | 2213 | 0.2961 | 0.2922 | -0.0132 | 0.1381 | 0.0312 |
| traj_backtest_rows_dl | Trajectory DL q50 | 12.0000 | 2211 | 0.3400 | 0.3404 | 0.0010 | 0.1666 | 0.0411 |
| traj_backtest_rows_dl | Trajectory DL q50 | 18.0000 | 2207 | 0.8478 | 0.8770 | 0.0333 | 0.2076 | 0.0475 |
| traj_backtest_rows_dl | Trajectory DL q50 | 24.0000 | 2209 | 0.7863 | 0.8185 | 0.0394 | 0.1300 | 0.0300 |
| traj_backtest_rows_dl | Trajectory DL q50 | 48.0000 | 2188 | 0.8244 | 0.8561 | 0.0370 | 0.1673 | 0.0470 |
| traj_backtest_rows_dl | Trajectory DL q50 | 72.0000 | 2169 | 0.8388 | 0.8722 | 0.0383 | 0.1919 | 0.0600 |
| traj_backtest_rows_dl | Trajectory DL q50 | 168.0000 | 2133 | 1.4018 | 1.4293 | 0.0192 | 0.2935 | 0.1029 |
| traj_backtest_rows_dl | Trajectory DL q50 | 360.0000 | 2053 | 1.3551 | 1.4164 | 0.0433 | 0.4317 | 0.1685 |
| traj_backtest_rows_dl | Trajectory DL q50 | 720.0000 | 2023 | 1.8021 | 1.9071 | 0.0550 | 0.6599 | 0.2703 |

### Seasonal results

| source | model | season | n | rmse | persistence_rmse | skill_score | mae | median_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictions_2026 | XGBoost | monsoon | 174966 | 2.3861 | 2.4849 | 0.0398 | 1.0311 | 0.5229 |
| predictions_2026 | XGBoost | post-monsoon | 62283 | 1.8079 | 1.8225 | 0.0080 | 0.6029 | 0.2625 |
| predictions_2026 | XGBoost | pre-monsoon | 182433 | 2.2732 | 2.3099 | 0.0159 | 0.5815 | 0.2566 |
| predictions_2026 | Ridge | monsoon | 174966 | 2.8897 | 2.4849 | -0.1629 | 1.5169 | 0.9671 |
| predictions_2026 | Ridge | post-monsoon | 62283 | 2.1183 | 1.8225 | -0.1623 | 1.3147 | 0.8977 |
| predictions_2026 | Ridge | pre-monsoon | 182433 | 2.2909 | 2.3099 | 0.0082 | 1.1660 | 0.7726 |
| traj_backtest_rows_dl | Trajectory DL q50 | monsoon | 6385 | 0.8828 | 0.9557 | 0.0762 | 0.3764 | 0.1234 |
| traj_backtest_rows_dl | Trajectory DL q50 | post-monsoon | 4926 | 0.9259 | 0.9285 | 0.0029 | 0.1651 | 0.0432 |
| traj_backtest_rows_dl | Trajectory DL q50 | pre-monsoon | 8095 | 1.2046 | 1.2492 | 0.0356 | 0.2263 | 0.0652 |

### Per-station skill and error distribution

| source | model | horizon | horizon_h | stations | median_station_skill | p10_station_skill | p90_station_skill | median_station_rmse | p10_station_rmse | p90_station_rmse | median_station_mae | median_station_median_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictions_2026 | Ridge | 30.0000 |  | 511 | -0.2436 | -2.2954 | 0.2167 | 1.1226 | 0.5710 | 2.9592 | 0.8981 | 0.8173 |
| predictions_2026 | XGBoost | 30.0000 |  | 511 | 0.0778 | -0.1350 | 0.2328 | 0.6964 | 0.3681 | 2.1939 | 0.5051 | 0.3556 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 6.0000 | 137 | -0.0462 | -2.0020 | 0.0305 | 0.1144 | 0.0170 | 0.4655 | 0.0699 | 0.0356 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 12.0000 | 137 | 0.0002 | -1.5189 | 0.0867 | 0.1392 | 0.0227 | 0.5237 | 0.0907 | 0.0463 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 18.0000 | 137 | 0.0019 | -0.3971 | 0.0666 | 0.1733 | 0.0236 | 0.6798 | 0.0998 | 0.0468 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 24.0000 | 137 | -0.0093 | -0.3713 | 0.0615 | 0.1260 | 0.0239 | 0.3945 | 0.0777 | 0.0343 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 48.0000 | 137 | 0.0008 | -0.2847 | 0.0672 | 0.1934 | 0.0349 | 0.5032 | 0.1182 | 0.0535 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 72.0000 | 137 | -0.0005 | -0.2028 | 0.0806 | 0.2047 | 0.0455 | 0.6118 | 0.1257 | 0.0697 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 168.0000 | 137 | 0.0180 | -0.3044 | 0.1561 | 0.3079 | 0.0838 | 0.8413 | 0.2005 | 0.1087 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 360.0000 | 137 | 0.0642 | -0.3856 | 0.2469 | 0.4066 | 0.1431 | 1.5512 | 0.3099 | 0.1816 |
| traj_backtest_rows_dl | Trajectory DL q50 |  | 720.0000 | 137 | 0.1154 | -0.2564 | 0.3233 | 0.6055 | 0.2171 | 2.1267 | 0.4215 | 0.2954 |

The station-level CSV gives RMSE, MAE and median absolute error for each station. The distribution table reports medians and skill percentiles across stations.

### Per-district results

| source | model | District | n | n_stations | rmse | persistence_rmse | skill_score | mae | median_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictions_2026 | XGBoost | ALIGARH | 7214 | 9 | 0.5140 | 0.5706 | 0.0992 | 0.3689 | 0.2758 |
| predictions_2026 | XGBoost | AZAMGARH | 19008 | 22 | 0.7597 | 0.6899 | -0.1013 | 0.4472 | 0.2578 |
| predictions_2026 | XGBoost | BAGHPAT | 23102 | 28 | 1.4294 | 1.4858 | 0.0380 | 0.6607 | 0.3817 |
| predictions_2026 | XGBoost | BANDA | 15788 | 19 | 4.8616 | 4.8733 | 0.0024 | 1.9079 | 0.5617 |
| predictions_2026 | XGBoost | BAREILLY | 1728 | 2 | 0.3087 | 0.2355 | -0.3105 | 0.1631 | 0.1119 |
| predictions_2026 | XGBoost | BUDAUN | 7492 | 9 | 3.0247 | 3.0554 | 0.0101 | 0.5589 | 0.1655 |
| predictions_2026 | XGBoost | CHITRAKOOT | 5100 | 7 | 2.5629 | 2.4504 | -0.0459 | 1.6035 | 0.9713 |
| predictions_2026 | XGBoost | DEORIA | 14763 | 18 | 0.5381 | 0.5958 | 0.0969 | 0.3736 | 0.2582 |
| predictions_2026 | XGBoost | ETAH | 7006 | 9 | 0.5964 | 0.6468 | 0.0778 | 0.3503 | 0.2007 |
| predictions_2026 | XGBoost | FATEHPUR | 10530 | 13 | 3.1639 | 3.1565 | -0.0023 | 1.0283 | 0.4652 |
| predictions_2026 | XGBoost | FIROZABAD | 495 | 1 | 0.6734 | 0.6808 | 0.0108 | 0.3745 | 0.3346 |
| predictions_2026 | XGBoost | GHAZIABAD | 4302 | 6 | 0.4493 | 0.5256 | 0.1452 | 0.2153 | 0.1197 |
| predictions_2026 | XGBoost | GORAKHPUR | 33084 | 39 | 0.6590 | 0.7431 | 0.1132 | 0.4271 | 0.2697 |
| predictions_2026 | XGBoost | HAMIRPUR | 8066 | 10 | 2.6590 | 2.8396 | 0.0636 | 1.3914 | 0.5053 |
| predictions_2026 | XGBoost | HATHRAS | 1845 | 2 | 5.9659 | 6.1267 | 0.0263 | 3.2793 | 0.4171 |
| predictions_2026 | XGBoost | JALAUN | 11262 | 13 | 1.9344 | 2.0574 | 0.0598 | 1.0712 | 0.3999 |
| predictions_2026 | XGBoost | JAUNPUR | 12234 | 15 | 1.7742 | 1.8266 | 0.0287 | 0.9735 | 0.3425 |
| predictions_2026 | XGBoost | JHANSI | 19084 | 24 | 3.1771 | 3.3449 | 0.0502 | 0.9846 | 0.3250 |
| predictions_2026 | XGBoost | KANSIRAM NAGAR | 962 | 1 | 0.3070 | 0.3857 | 0.2040 | 0.2648 | 0.2699 |
| predictions_2026 | XGBoost | KAUSHAMBI | 4664 | 6 | 0.9794 | 0.9921 | 0.0127 | 0.5529 | 0.3852 |
| predictions_2026 | XGBoost | KUSHINAGAR | 12325 | 15 | 0.6193 | 0.7311 | 0.1529 | 0.3669 | 0.1997 |
| predictions_2026 | XGBoost | LALITPUR | 15268 | 19 | 2.1641 | 2.3444 | 0.0769 | 1.0850 | 0.4182 |
| predictions_2026 | XGBoost | MAHARAJGANJ | 6390 | 8 | 0.6784 | 0.7955 | 0.1472 | 0.4113 | 0.2154 |
| predictions_2026 | XGBoost | MAHOBA | 7778 | 9 | 1.8369 | 2.0521 | 0.1049 | 1.0926 | 0.5375 |
| predictions_2026 | XGBoost | MAINPURI | 4259 | 5 | 0.7555 | 0.8471 | 0.1082 | 0.4707 | 0.2707 |
| predictions_2026 | XGBoost | MATHURA | 6942 | 8 | 0.6953 | 0.6612 | -0.0515 | 0.4404 | 0.2594 |
| predictions_2026 | XGBoost | MEERUT | 18742 | 23 | 0.8295 | 0.9264 | 0.1045 | 0.4624 | 0.2586 |
| predictions_2026 | XGBoost | MIRZAPUR | 18264 | 22 | 1.2482 | 1.3404 | 0.0688 | 0.6563 | 0.3775 |
| predictions_2026 | XGBoost | MUZAFFARNAGAR | 23610 | 28 | 5.5945 | 5.6501 | 0.0099 | 1.2508 | 0.4553 |
| predictions_2026 | XGBoost | PRATAPGARH | 3260 | 4 | 0.5476 | 0.5729 | 0.0442 | 0.4438 | 0.3824 |
| predictions_2026 | XGBoost | SAHARANPUR | 21533 | 27 | 1.2826 | 1.3812 | 0.0714 | 0.5658 | 0.3373 |
| predictions_2026 | XGBoost | SANT RAVIDAS NAGAR | 13665 | 16 | 1.3397 | 1.4501 | 0.0761 | 0.8596 | 0.4841 |
| predictions_2026 | XGBoost | SHAHJAHANPUR | 11582 | 15 | 1.0123 | 1.1350 | 0.1081 | 0.5984 | 0.2622 |
| predictions_2026 | XGBoost | SHRAWASTI | 4971 | 6 | 0.5764 | 0.6120 | 0.0582 | 0.4454 | 0.3610 |
| predictions_2026 | XGBoost | SIDDHARTHNAGAR | 7367 | 9 | 0.6021 | 0.6661 | 0.0961 | 0.4212 | 0.3091 |
| predictions_2026 | XGBoost | SONBHADRA | 20670 | 25 | 1.2190 | 1.3225 | 0.0783 | 0.6990 | 0.4002 |
| predictions_2026 | XGBoost | VARANASI | 15327 | 19 | 2.2771 | 2.3643 | 0.0369 | 1.2008 | 0.5187 |
| predictions_2026 | Ridge | ALIGARH | 7214 | 9 | 1.2694 | 0.5706 | -1.2247 | 1.0186 | 0.7891 |
| predictions_2026 | Ridge | AZAMGARH | 19008 | 22 | 1.6233 | 0.6899 | -1.3531 | 1.3622 | 1.2067 |
| predictions_2026 | Ridge | BAGHPAT | 23102 | 28 | 1.5841 | 1.4858 | -0.0661 | 1.1884 | 0.9807 |
| predictions_2026 | Ridge | BANDA | 15788 | 19 | 5.4668 | 4.8733 | -0.1218 | 2.7167 | 1.6921 |
| predictions_2026 | Ridge | BAREILLY | 1728 | 2 | 0.3163 | 0.2355 | -0.3428 | 0.2640 | 0.2489 |
| predictions_2026 | Ridge | BUDAUN | 7492 | 9 | 2.3315 | 3.0554 | 0.2369 | 0.8995 | 0.3804 |
| predictions_2026 | Ridge | CHITRAKOOT | 5100 | 7 | 2.7071 | 2.4504 | -0.1048 | 1.7671 | 0.8773 |
| predictions_2026 | Ridge | DEORIA | 14763 | 18 | 0.8914 | 0.5958 | -0.4962 | 0.6859 | 0.5444 |
| predictions_2026 | Ridge | ETAH | 7006 | 9 | 1.9010 | 0.6468 | -1.9393 | 1.6122 | 1.5175 |
| predictions_2026 | Ridge | FATEHPUR | 10530 | 13 | 3.9263 | 3.1565 | -0.2439 | 2.0662 | 1.4147 |
| predictions_2026 | Ridge | FIROZABAD | 495 | 1 | 2.4937 | 0.6808 | -2.6631 | 2.2854 | 1.9730 |
| predictions_2026 | Ridge | GHAZIABAD | 4302 | 6 | 0.8015 | 0.5256 | -0.5249 | 0.6469 | 0.6228 |
| predictions_2026 | Ridge | GORAKHPUR | 33084 | 39 | 0.8760 | 0.7431 | -0.1788 | 0.6445 | 0.4769 |
| predictions_2026 | Ridge | HAMIRPUR | 8066 | 10 | 3.2604 | 2.8396 | -0.1482 | 2.2559 | 1.5604 |
| predictions_2026 | Ridge | HATHRAS | 1845 | 2 | 4.8296 | 6.1267 | 0.2117 | 3.2945 | 1.7542 |
| predictions_2026 | Ridge | JALAUN | 11262 | 13 | 1.8833 | 2.0574 | 0.0846 | 1.1910 | 0.6789 |
| predictions_2026 | Ridge | JAUNPUR | 12234 | 15 | 2.1141 | 1.8266 | -0.1574 | 1.7030 | 1.3938 |
| predictions_2026 | Ridge | JHANSI | 19084 | 24 | 3.2236 | 3.3449 | 0.0363 | 1.8185 | 0.9179 |
| predictions_2026 | Ridge | KANSIRAM NAGAR | 962 | 1 | 0.9405 | 0.3857 | -1.4387 | 0.8955 | 0.9010 |
| predictions_2026 | Ridge | KAUSHAMBI | 4664 | 6 | 2.1550 | 0.9921 | -1.1722 | 1.6876 | 1.4263 |
| predictions_2026 | Ridge | KUSHINAGAR | 12325 | 15 | 0.8904 | 0.7311 | -0.2179 | 0.6756 | 0.5153 |
| predictions_2026 | Ridge | LALITPUR | 15268 | 19 | 2.3618 | 2.3444 | -0.0074 | 1.4681 | 0.8295 |
| predictions_2026 | Ridge | MAHARAJGANJ | 6390 | 8 | 0.7553 | 0.7955 | 0.0506 | 0.5597 | 0.4098 |
| predictions_2026 | Ridge | MAHOBA | 7778 | 9 | 3.3099 | 2.0521 | -0.6130 | 2.4298 | 2.0553 |
| predictions_2026 | Ridge | MAINPURI | 4259 | 5 | 0.8459 | 0.8471 | 0.0014 | 0.6489 | 0.5024 |
| predictions_2026 | Ridge | MATHURA | 6942 | 8 | 1.9447 | 0.6612 | -1.9411 | 1.5449 | 1.2564 |
| predictions_2026 | Ridge | MEERUT | 18742 | 23 | 1.0663 | 0.9264 | -0.1510 | 0.7714 | 0.5712 |
| predictions_2026 | Ridge | MIRZAPUR | 18264 | 22 | 1.8914 | 1.3404 | -0.4111 | 1.3350 | 0.9640 |
| predictions_2026 | Ridge | MUZAFFARNAGAR | 23610 | 28 | 5.5925 | 5.6501 | 0.0102 | 1.6686 | 0.8562 |
| predictions_2026 | Ridge | PRATAPGARH | 3260 | 4 | 1.1568 | 0.5729 | -1.0192 | 0.9805 | 0.9013 |
| predictions_2026 | Ridge | SAHARANPUR | 21533 | 27 | 1.3882 | 1.3812 | -0.0051 | 0.9873 | 0.8356 |
| predictions_2026 | Ridge | SANT RAVIDAS NAGAR | 13665 | 16 | 2.0408 | 1.4501 | -0.4073 | 1.7174 | 1.5660 |
| predictions_2026 | Ridge | SHAHJAHANPUR | 11582 | 15 | 1.0562 | 1.1350 | 0.0695 | 0.8088 | 0.6534 |
| predictions_2026 | Ridge | SHRAWASTI | 4971 | 6 | 0.8728 | 0.6120 | -0.4262 | 0.7310 | 0.6630 |
| predictions_2026 | Ridge | SIDDHARTHNAGAR | 7367 | 9 | 1.1628 | 0.6661 | -0.7458 | 0.8250 | 0.5987 |
| predictions_2026 | Ridge | SONBHADRA | 20670 | 25 | 2.1036 | 1.3225 | -0.5906 | 1.5501 | 1.1584 |
| predictions_2026 | Ridge | VARANASI | 15327 | 19 | 2.4790 | 2.3643 | -0.0485 | 1.6864 | 1.1949 |

| source | model | district | n | n_stations | rmse | persistence_rmse | skill_score | mae | median_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traj_backtest_rows_dl | Trajectory DL q50 | ALIGARH | 131 | 1 | 0.3155 | 0.3052 | -0.0339 | 0.2091 | 0.1154 |
| traj_backtest_rows_dl | Trajectory DL q50 | AZAMGARH | 1218 | 8 | 0.3601 | 0.3700 | 0.0265 | 0.2101 | 0.0903 |
| traj_backtest_rows_dl | Trajectory DL q50 | BAGHPAT | 1134 | 8 | 0.4561 | 0.4778 | 0.0455 | 0.2472 | 0.1235 |
| traj_backtest_rows_dl | Trajectory DL q50 | BANDA | 583 | 4 | 0.4776 | 0.5342 | 0.1060 | 0.1816 | 0.0515 |
| traj_backtest_rows_dl | Trajectory DL q50 | BUDAUN | 149 | 1 | 0.1514 | 0.1312 | -0.1542 | 0.0605 | 0.0175 |
| traj_backtest_rows_dl | Trajectory DL q50 | CHITRAKOOT | 319 | 3 | 1.2171 | 1.2554 | 0.0305 | 0.4553 | 0.1006 |
| traj_backtest_rows_dl | Trajectory DL q50 | DEORIA | 695 | 5 | 0.3414 | 0.3584 | 0.0474 | 0.1619 | 0.0344 |
| traj_backtest_rows_dl | Trajectory DL q50 | ETAH | 361 | 3 | 0.2187 | 0.2770 | 0.2107 | 0.0922 | 0.0196 |
| traj_backtest_rows_dl | Trajectory DL q50 | FATEHPUR | 123 | 1 | 0.0466 | 0.0246 | -0.8948 | 0.0269 | 0.0127 |
| traj_backtest_rows_dl | Trajectory DL q50 | FIROZABAD | 94 | 1 | 0.8996 | 0.8994 | -0.0002 | 0.2507 | 0.1249 |
| traj_backtest_rows_dl | Trajectory DL q50 | GHAZIABAD | 274 | 2 | 0.7007 | 0.6840 | -0.0244 | 0.1108 | 0.0143 |
| traj_backtest_rows_dl | Trajectory DL q50 | GORAKHPUR | 789 | 5 | 0.3869 | 0.4331 | 0.1067 | 0.1952 | 0.0834 |
| traj_backtest_rows_dl | Trajectory DL q50 | HAMIRPUR | 275 | 2 | 1.3773 | 1.4973 | 0.0801 | 0.5177 | 0.0689 |
| traj_backtest_rows_dl | Trajectory DL q50 | JALAUN | 587 | 4 | 0.8255 | 0.8926 | 0.0751 | 0.3059 | 0.0728 |
| traj_backtest_rows_dl | Trajectory DL q50 | JAUNPUR | 297 | 2 | 0.4708 | 0.5122 | 0.0808 | 0.1434 | 0.0285 |
| traj_backtest_rows_dl | Trajectory DL q50 | JHANSI | 1052 | 8 | 3.7260 | 3.8427 | 0.0304 | 0.6341 | 0.0660 |
| traj_backtest_rows_dl | Trajectory DL q50 | KUSHINAGAR | 149 | 1 | 0.1517 | 0.2178 | 0.3038 | 0.0679 | 0.0178 |
| traj_backtest_rows_dl | Trajectory DL q50 | LALITPUR | 702 | 5 | 0.6879 | 0.7787 | 0.1166 | 0.2283 | 0.0580 |
| traj_backtest_rows_dl | Trajectory DL q50 | MAHARAJGANJ | 270 | 2 | 0.4060 | 0.5081 | 0.2010 | 0.1947 | 0.0755 |
| traj_backtest_rows_dl | Trajectory DL q50 | MAHOBA | 149 | 1 | 0.5673 | 0.6636 | 0.1451 | 0.2891 | 0.1250 |
| traj_backtest_rows_dl | Trajectory DL q50 | MAINPURI | 569 | 4 | 0.3629 | 0.3924 | 0.0752 | 0.2122 | 0.0990 |
| traj_backtest_rows_dl | Trajectory DL q50 | MATHURA | 595 | 4 | 0.7877 | 0.8045 | 0.0208 | 0.3329 | 0.0670 |
| traj_backtest_rows_dl | Trajectory DL q50 | MEERUT | 1240 | 9 | 0.3362 | 0.4013 | 0.1622 | 0.1809 | 0.0756 |
| traj_backtest_rows_dl | Trajectory DL q50 | MIRZAPUR | 461 | 3 | 0.5506 | 0.6009 | 0.0837 | 0.2605 | 0.0846 |
| traj_backtest_rows_dl | Trajectory DL q50 | MUZAFFARNAGAR | 764 | 5 | 0.5148 | 0.5306 | 0.0297 | 0.2901 | 0.1238 |
| traj_backtest_rows_dl | Trajectory DL q50 | PRATAPGARH | 148 | 1 | 0.4598 | 0.4461 | -0.0306 | 0.3023 | 0.1705 |
| traj_backtest_rows_dl | Trajectory DL q50 | SAHARANPUR | 1302 | 9 | 0.6150 | 0.6639 | 0.0736 | 0.2905 | 0.0940 |
| traj_backtest_rows_dl | Trajectory DL q50 | SANT RAVIDAS NAGAR | 751 | 5 | 0.6339 | 0.6806 | 0.0685 | 0.3186 | 0.1512 |
| traj_backtest_rows_dl | Trajectory DL q50 | SHRAWASTI | 141 | 1 | 0.3866 | 0.4009 | 0.0357 | 0.2581 | 0.1371 |
| traj_backtest_rows_dl | Trajectory DL q50 | SIDDHARTHNAGAR | 443 | 3 | 0.3485 | 0.3648 | 0.0445 | 0.2099 | 0.1124 |
| traj_backtest_rows_dl | Trajectory DL q50 | SONBHADRA | 911 | 6 | 0.6490 | 0.6609 | 0.0180 | 0.3480 | 0.1611 |
| traj_backtest_rows_dl | Trajectory DL q50 | VARANASI | 1003 | 7 | 0.8131 | 0.8343 | 0.0254 | 0.3209 | 0.0740 |

### Driver-group comparison

| source | model | linked_driver | n | n_stations | rmse | persistence_rmse | skill_score | mae | median_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictions_2026 | XGBoost | False | 47635 | 59 | 1.8126 | 1.8633 | 0.0272 | 0.5838 | 0.2405 |
| predictions_2026 | XGBoost | True | 372047 | 452 | 2.3104 | 2.3731 | 0.0264 | 0.7962 | 0.3532 |
| predictions_2026 | Ridge | False | 47635 | 59 | 1.8392 | 1.8633 | 0.0129 | 1.1318 | 0.7243 |
| predictions_2026 | Ridge | True | 372047 | 452 | 2.6112 | 2.3731 | -0.1003 | 1.3603 | 0.8872 |
| traj_backtest_rows_dl | Trajectory DL q50 | False | 3573 | 27 | 0.5727 | 0.6006 | 0.0463 | 0.1899 | 0.0380 |
| traj_backtest_rows_dl | Trajectory DL q50 | True | 15833 | 110 | 1.1175 | 1.1632 | 0.0392 | 0.2760 | 0.0799 |

Interpretation: XGBoost has small positive skill in both groups (2.7% with no linked rain driver; 2.6% with one). Trajectory DL is also positive in both groups (4.6% and 3.9%). The gain is therefore not confined to stations with observed rain-gauge coverage. Ridge gains only in the no-driver group and is worse than persistence in the linked-driver group. Pooled daily DL cannot be split from the aggregate JSON.

### Top stations by squared error

| source | model | station | n | squared_error | share_of_model_squared_error |
| --- | --- | --- | --- | --- | --- |
| predictions_2026 | Ridge | Primary School chandsina | 660 | 695191.7781 | 0.2577 |
| predictions_2026 | Ridge | Primary School Sanda-Sanee | 864 | 314255.9713 | 0.1165 |
| predictions_2026 | Ridge | Upper Primary School Sarahan Bujurg | 864 | 116756.1637 | 0.0433 |
| predictions_2026 | Ridge | Primary School, Dera Devra bujurg | 875 | 83120.4056 | 0.0308 |
| predictions_2026 | Ridge | Hathras Hathras Hathras | 948 | 42426.6763 | 0.0157 |
| predictions_2026 | Ridge | Purva Madhyamik Vidhyalaya Achausa | 864 | 42031.3709 | 0.0156 |
| predictions_2026 | Ridge | Primary School Vamanpura | 864 | 37994.4657 | 0.0141 |
| predictions_2026 | Ridge | Banda Tindwari Bhindowra | 822 | 37211.8138 | 0.0138 |
| predictions_2026 | Ridge | Primary School, Karasdev Mandir | 778 | 36159.1870 | 0.0134 |
| predictions_2026 | Ridge | Mahoba Panwari Didvara | 848 | 31491.1287 | 0.0117 |
| predictions_2026 | Ridge | Hamirpur Sarila Purani | 964 | 30772.3324 | 0.0114 |
| predictions_2026 | Ridge | Purv Madhyamik Vidhyalaya, Doriya | 809 | 27923.3183 | 0.0104 |
| predictions_2026 | Ridge | Inter college, Tindwara | 836 | 26772.4934 | 0.0099 |
| predictions_2026 | Ridge | Goshala Bhadehdu | 864 | 25230.2556 | 0.0094 |
| predictions_2026 | Ridge | Mahoba Kabrai Karaha | 960 | 24192.4810 | 0.0090 |
| predictions_2026 | Ridge | Varansi Cholapur Chaubepur | 840 | 21089.2391 | 0.0078 |
| predictions_2026 | Ridge | Prmary School Rajpura | 856 | 19995.9540 | 0.0074 |
| predictions_2026 | Ridge | Gadhela Panchayat Bhawan | 856 | 16583.0978 | 0.0061 |
| predictions_2026 | Ridge | Swarg Dham, Hardua | 864 | 16569.8761 | 0.0061 |
| predictions_2026 | Ridge | Jaunpur Dobhi Kachhawan | 830 | 16530.5941 | 0.0061 |
| predictions_2026 | XGBoost | Primary School chandsina | 660 | 716845.1666 | 0.3346 |
| predictions_2026 | XGBoost | Primary School Sanda-Sanee | 864 | 237357.6468 | 0.1108 |
| predictions_2026 | XGBoost | Primary School, Dera Devra bujurg | 875 | 152543.7798 | 0.0712 |
| predictions_2026 | XGBoost | Upper Primary School Sarahan Bujurg | 864 | 89270.2697 | 0.0417 |
| predictions_2026 | XGBoost | Primary School Vamanpura | 864 | 67725.2637 | 0.0316 |
| predictions_2026 | XGBoost | Hathras Hathras Hathras | 948 | 65549.8166 | 0.0306 |
| predictions_2026 | XGBoost | Banda Tindwari Bhindowra | 822 | 42147.6069 | 0.0197 |
| predictions_2026 | XGBoost | Goshala Bhadehdu | 864 | 36585.3952 | 0.0171 |
| predictions_2026 | XGBoost | Primary School, Silana | 952 | 28046.5192 | 0.0131 |
| predictions_2026 | XGBoost | Primary School, Karasdev Mandir | 778 | 26262.5357 | 0.0123 |
| predictions_2026 | XGBoost | Hamirpur Sarila Purani | 964 | 23645.6030 | 0.0110 |
| predictions_2026 | XGBoost | Rajkia High School Pakhrauli | 756 | 20914.1377 | 0.0098 |
| predictions_2026 | XGBoost | Varansi Cholapur Chaubepur | 840 | 19850.9758 | 0.0093 |
| predictions_2026 | XGBoost | Prmary School Rajpura | 856 | 17596.1987 | 0.0082 |
| predictions_2026 | XGBoost | Primary School, Kalan Bar Devran | 667 | 16631.1490 | 0.0078 |
| predictions_2026 | XGBoost | Primary School/ Composite, Sidki | 845 | 14581.2325 | 0.0068 |
| predictions_2026 | XGBoost | Gadhela Panchayat Bhawan | 856 | 14406.0483 | 0.0067 |
| predictions_2026 | XGBoost | Goshala Bada Gaon | 856 | 14023.9742 | 0.0065 |
| predictions_2026 | XGBoost | Uccha Prathmik Vidhyalaya, Lohra (Pampawan) | 864 | 12687.1466 | 0.0059 |
| predictions_2026 | XGBoost | Composite School Sagarpur | 856 | 12575.0098 | 0.0059 |
| traj_backtest_rows_dl | Trajectory DL q50 | Primary School, Dera Devra bujurg | 151 | 14126.2040 | 0.6744 |
| traj_backtest_rows_dl | Trajectory DL q50 | Rohari | 112 | 443.8888 | 0.0212 |
| traj_backtest_rows_dl | Trajectory DL q50 | Chitrakoot Manikpur Agarhudha | 142 | 418.4320 | 0.0200 |
| traj_backtest_rows_dl | Trajectory DL q50 | Hamirpur Kurara Jalokhar | 133 | 376.2446 | 0.0180 |
| traj_backtest_rows_dl | Trajectory DL q50 | Swarg Dham, Hardua | 152 | 348.9726 | 0.0167 |
| traj_backtest_rows_dl | Trajectory DL q50 | Mathura Chaumuha Navagaon-11 | 163 | 325.1017 | 0.0155 |
| traj_backtest_rows_dl | Trajectory DL q50 | Primary School, Garher | 132 | 297.3665 | 0.0142 |
| traj_backtest_rows_dl | Trajectory DL q50 | Primary School/ Composite, Landhoura Junardar | 148 | 238.9872 | 0.0114 |
| traj_backtest_rows_dl | Trajectory DL q50 | Varansi Harahua Kakalpur | 131 | 228.2544 | 0.0109 |
| traj_backtest_rows_dl | Trajectory DL q50 | Varansi Chirai Gaon Chhahi | 137 | 201.3109 | 0.0096 |
| traj_backtest_rows_dl | Trajectory DL q50 | Community Center, Ghorawal | 156 | 199.6550 | 0.0095 |
| traj_backtest_rows_dl | Trajectory DL q50 | Primary School, Bilata | 148 | 184.2223 | 0.0088 |
| traj_backtest_rows_dl | Trajectory DL q50 | Saint Ravidas Nagar Abholi Gangarampur | 142 | 168.2691 | 0.0080 |
| traj_backtest_rows_dl | Trajectory DL q50 | Primary School/ Composite, Mehmoodpur | 125 | 158.0565 | 0.0075 |
| traj_backtest_rows_dl | Trajectory DL q50 | Varansi Harahua Gurwat | 144 | 148.6125 | 0.0071 |
| traj_backtest_rows_dl | Trajectory DL q50 | Block Office, Kurara | 142 | 145.4455 | 0.0069 |
| traj_backtest_rows_dl | Trajectory DL q50 | Composite School Basantpur Saitali-1 | 122 | 133.3445 | 0.0064 |
| traj_backtest_rows_dl | Trajectory DL q50 | Composite School, Ghamhapur | 152 | 106.2927 | 0.0051 |
| traj_backtest_rows_dl | Trajectory DL q50 | Primary School, Evani | 142 | 96.9447 | 0.0046 |
| traj_backtest_rows_dl | Trajectory DL q50 | Aagan Wadi Kendra Chandni | 152 | 81.5680 | 0.0039 |

### RMSE concentration

| source | model | top20_squared_error | total_squared_error | top20_share |
| --- | --- | --- | --- | --- |
| predictions_2026 | Ridge | 1662298.6031 | 2697896.7118 | 0.6161 |
| predictions_2026 | XGBoost | 1629245.4759 | 2142412.9861 | 0.7605 |
| traj_backtest_rows_dl | Trajectory DL q50 | 18427.1734 | 20945.9463 | 0.8797 |

Interpretation: a small number of stations dominate squared error: the top 20 contribute 76.0% for XGBoost, 61.6% for Ridge and 88.0% for trajectory DL. `Primary School chandsina` alone contributes 33.5% of XGBoost squared error (25.8% for Ridge); `Primary School, Dera Devra bujurg` contributes 67.4% for trajectory DL. This concentration is compatible with station-specific outliers, but the saved evaluation rows do not establish whether datum errors or heavy pumping caused them.

## Artifact purge audit

| Artifact | Artifact date | Effective input lineage | Purged? | Evidence |
|---|---|---|---|---|
| `outputs/predictions_2026.parquet` | 2026-10-07 21:16 | `data/features/test.parquet`; `prep.json` built 2026-10-07, before purge commit `84935e8` (2026-10-08) | No | Both prediction artifact and feature prep predate commit. |
| `outputs/pooled_dl_eval.json` | 2026-10-11 20:31 | pooled model trained 2026-10-11, but `data/features` prep is 2026-10-07 | No (effective input) | Output is post-commit; upstream saved feature frame is pre-commit. File timestamp alone is insufficient. |
| `outputs/traj_backtest_rows_dl.parquet` | 2026-10-11 19:37 | `data/features_traj` prep built 2026-10-07 | No (effective input) | Output/model is post-commit; saved trajectory feature prep predates commit. |

## Persistence RMSE values found in the module

| RMSE | Protocol / sample | Interpretation |
|---:|---|---|
| 1.907 | 30-day (`h720`) trajectory backtest; 19,406 anchors / 150 stations overall, 2,023 paired h720 rows | Reported in README trajectory results/model comparison and ML_MODULE_GUIDE; saved XGB and DL trajectory summaries use this baseline. Current `traj_backtest_rows_dl.parquet` reproduces 1.9071. |
| 2.187 | Pooled daily GRU evaluation, 4,355 rows, 511 stations | Reported in `pooled_dl_eval.json`, README algorithm comparison and ML_MODULE_GUIDE. It is the saved JSON aggregate, not the 6-hourly model metric. |
| 2.290 | Historical 2026 test on the 6-hour grid; README pairs it with XGBoost RMSE 2.242, Ridge 2.456 and climatology 3.572 | Repeated in README’s 2026 test and algorithm comparison. README says the 6-hour grid lowered persistence from 2.395 to 2.290. Pre-purge; current `predictions_2026.parquet` recomputes to 2.3209. |
| 2.382 | Historical stride evaluation: one 30-day endpoint per station every 120 six-hour steps; README pairs it with model RMSE 2.339 | Reported in README stride-evaluation paragraph; values predate target-time purge. The requested artifacts do not include its row-level evaluation file. |

## `st_id` / `dist_id` feature audit

| Script | Uses station/district ID as model feature? | Detail |
|---|---|---|
| `06_features.py` | Creates and carries both | `st_id` and `dist_id` are ordinal metadata columns; not part of `NUM_COLS`. |
| `06_train.py` | XGBoost: yes; Ridge: no | XGBoost feature columns append both IDs. Ridge one-hot encodes original `Station`/`District` strings instead. |
| `31_train_traj.py` | No | Feature list is numeric predictors plus horizon encodings. |
| `34_train_traj_dl.py` | No | Model input is `NUM_COLS`; station identity is used for sequence grouping only. |
| `36_train_pooled_dl.py` | No | Dense/seasonal columns are model inputs; `st_id` is used for run boundaries/grouping, not as an input. `dist_id` is not used. |

## Files

- `group_metrics.csv`: available metrics by horizon, season, district and linked-driver group.
- `station_metrics.csv`: paired station-level RMSE, MAE, median absolute error and skill.
- `station_skill_distribution.csv`: median/p10/p90 station skill and RMSE.
- `top20_squared_error_contribution.csv`: top 20 stations per row-level model/artifact, including contribution share.
- `error_concentration.csv`: total squared error and top-20 share.
- `pooled_dl_aggregate.csv`: the JSON-only pooled-DL aggregate result.
