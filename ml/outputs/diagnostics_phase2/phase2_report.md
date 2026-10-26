# Phase 2 read-only forensics

## Scope and assumptions

- Read only `data/aligned/table_6h.parquet`, `data/meta/assoc_rainfall.csv`, saved prediction/backtest artifacts, and Phase 1 diagnostics. No training, refresh, or network fetch was run. New files are confined to `outputs/diagnostics_phase2/`. Existing changes in the source Parquet/metadata files were left untouched.
- For outlier one-day jumps, both endpoints are exactly 24 hours apart and flagged `gwl_obs=1`; for 30-day change metrics both endpoints are exactly 30 days apart and observed. The observed-slot share denominator is every table slot for that station during 2021–2026.
- The regime-filter result reproduces `06_features.flag_regime_shift`: median per station/year; baseline median of 2021–2024 yearly medians; fail if any available year median differs by more than 15m. A station is labeled “consistent with datum/telemetry fault” if it fails that filter or has an observed-endpoint 1-day jump >15m; other union stations are labeled “plausible hydrology”. These are evidence-based screening labels, not proof of cause.
- Quality proposal uses only exact 30-day pairs ending before 2025-10-01 and observed at both endpoints. Remove if station p99 absolute 30-day change exceeds 15m. The cutoff was selected as a physical-scale screening threshold and was not tuned using any 2026 model errors.
- Winsorized-RMSE skill caps both model and persistence absolute errors at the 99th percentile of their pooled absolute errors within each reported sample. This is a robustness summary, not the quality filter.
- Nearest rain gauge is the minimum `dist_km` for each GWL station in `assoc_rainfall.csv`; `none` means no matched rain association within 150km. Buckets use `<25`, `[25,75)`, `[75,150]` km.
- For h6/h12/h18/h24, `obs` and `persist` are saved absolute GWL levels (persistence equals the anchor level). Raw flags are matched to exact anchor and target timestamps in `table_6h.parquet`. Missing timestamp matches are reported as unknown, not observed. The regime-filter audit reproduces the existing function over all years present; it is descriptive and can use post-cutoff data, unlike the proposed quality rule.

## 1. Outlier forensics

Union size: 39 stations (20 XGBoost 30d top stations; 20 trajectory DL h720 top stations).

| Station | top_xgb_30d | top_traj_dl_h720 | range_min | range_max | n_1day_jumps_gt5 | n_1day_jumps_gt15 | observed_slot_share | max_abs_30day_change | existing_regime_shift_filter_passed | classification | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Primary School, Dera Devra bujurg | True | True | -11.6640 | 27.5000 | 48 | 48 | 0.5549 | 35.8730 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 35.63 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Banda Tindwari Bhindowra | True | False | -20.9850 | -0.5790 | 122 | 32 | 0.7672 | 17.4150 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 17.24 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Composite School Sagarpur | True | False | -12.9350 | 99.0000 | 18 | 8 | 0.7223 | 71.5000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 71.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Gadhela Panchayat Bhawan | True | False | -22.1070 | 99.0000 | 15 | 14 | 0.7393 | 72.2000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 86.10 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Goshala Bada Gaon | True | False | -18.0720 | 99.0000 | 6 | 6 | 0.7137 | 71.1000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 85.10 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Goshala Bhadehdu | True | False | -30.1300 | 99.0000 | 11 | 11 | 0.7173 | 71.4000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 71.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Hamirpur Sarila Purani | True | False | -27.7830 | -4.9640 | 0 | 0 | 0.7731 | 15.3530 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 2.36 m; threshold 15 m); max observed-endpoint 1-day jump is 2.62 m. |
| Hathras Hathras Hathras | True | False | -22.0350 | -0.2200 | 2805 | 834 | 0.8445 | 19.8880 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 19.97 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Primary School Sanda-Sanee | True | False | -67.4380 | 99.0000 | 10 | 10 | 0.7318 | 73.0000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 85.70 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Primary School Vamanpura | True | False | -64.3000 | 99.0000 | 100 | 39 | 0.6060 | 72.9000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 73.70 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Primary School chandsina | True | False | -75.8890 | 99.0000 | 147 | 75 | 0.7067 | 74.9000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 74.90 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Primary School, Kalan Bar Devran | True | False | -419.1960 | -1.3730 | 88 | 67 | 0.6450 | 403.4600 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 338.06 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Primary School, Karasdev Mandir | True | False | -25.5170 | -2.2930 | 46 | 0 | 0.6987 | 18.3430 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 6.03 m; threshold 15 m); max observed-endpoint 1-day jump is 10.13 m. |
| Primary School, Silana | True | False | -59.7960 | -20.3950 | 58 | 58 | 0.7687 | 39.0290 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 38.64 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Primary School/ Composite, Sidki | True | False | -36.2620 | -9.2060 | 50 | 50 | 0.7429 | 26.7710 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 26.00 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Prmary School Rajpura | True | False | -25.0580 | 99.0000 | 9 | 9 | 0.7082 | 71.9000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 72.70 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Rajkia High School Pakhrauli | True | False | -29.3840 | 99.0000 | 16 | 16 | 0.7286 | 71.7000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 85.80 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Uccha Prathmik Vidhyalaya, Lohra (Pampawan) | True | False | -15.4410 | 99.0000 | 42 | 7 | 0.7279 | 113.8270 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 72.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Upper Primary School Sarahan Bujurg | True | False | -58.4440 | 99.0000 | 9 | 9 | 0.7407 | 72.1000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 72.10 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Varansi Cholapur Chaubepur | True | False | -18.9230 | -0.6840 | 24 | 0 | 0.7660 | 13.9820 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 0.95 m; threshold 15 m); max observed-endpoint 1-day jump is 12.54 m. |
| Aagan Wadi Kendra Chandni | False | True | -8.3490 | 99.0000 | 7 | 7 | 0.7289 | 72.8000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 85.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Block Office, Kurara | False | True | -12.3660 | -1.7090 | 0 | 0 | 0.7777 | 8.2780 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 1.56 m; threshold 15 m); max observed-endpoint 1-day jump is 1.52 m. |
| Chitrakoot Manikpur Agarhudha | False | True | -59.0830 | -0.9050 | 37 | 37 | 0.7575 | 51.0430 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 46.60 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Community Center, Ghorawal | False | True | -16.9860 | -0.1010 | 0 | 0 | 0.7670 | 15.6010 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 3.49 m; threshold 15 m); max observed-endpoint 1-day jump is 4.37 m. |
| Composite School Basantpur Saitali-1 | False | True | -9.1820 | 99.0000 | 14 | 10 | 0.6848 | 73.1000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 73.10 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Composite School, Ghamhapur | False | True | -15.9960 | 99.0000 | 14 | 6 | 0.7179 | 71.4000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 71.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Hamirpur Kurara Jalokhar | False | True | -10.2940 | 0.0560 | 74 | 0 | 0.7420 | 10.0720 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 1.56 m; threshold 15 m); max observed-endpoint 1-day jump is 10.15 m. |
| Mathura Chaumuha Navagaon-11 | False | True | -14.5810 | -0.9910 | 109 | 0 | 0.7848 | 13.5590 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 3.73 m; threshold 15 m); max observed-endpoint 1-day jump is 12.66 m. |
| Primary School, Bilata | False | True | -14.9880 | -1.2890 | 0 | 0 | 0.7194 | 11.2060 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 7.46 m; threshold 15 m); max observed-endpoint 1-day jump is 4.95 m. |
| Primary School, Evani | False | True | -12.9620 | -1.8970 | 0 | 0 | 0.7682 | 7.5420 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 3.05 m; threshold 15 m); max observed-endpoint 1-day jump is 1.18 m. |
| Primary School, Garher | False | True | -12.4130 | -0.1020 | 1 | 0 | 0.7430 | 11.0280 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 1.61 m; threshold 15 m); max observed-endpoint 1-day jump is 5.02 m. |
| Primary School/ Composite, Landhoura Junardar | False | True | -19.1610 | -4.9730 | 0 | 0 | 0.7647 | 9.0820 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 1.93 m; threshold 15 m); max observed-endpoint 1-day jump is 2.84 m. |
| Primary School/ Composite, Mehmoodpur | False | True | -149.0730 | -7.6310 | 13 | 9 | 0.7585 | 140.5670 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 138.58 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Rohari | False | True | -21.3000 | 97.6900 | 12 | 12 | 0.8096 | 72.0900 | False | consistent with datum/telemetry fault | existing annual-median regime filter fails: maximum deviation from 2021-24 station baseline = 36.73 m (>15 m); consistent with a persistent offset, though cause is unverified. |
| Saint Ravidas Nagar Abholi Gangarampur | False | True | -15.6060 | -5.6190 | 0 | 0 | 0.7627 | 7.0250 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 3.67 m; threshold 15 m); max observed-endpoint 1-day jump is 1.15 m. |
| Swarg Dham, Hardua | False | True | -18.7110 | 99.0000 | 12 | 12 | 0.5829 | 71.9000 | True | consistent with datum/telemetry fault | filter passes, but an observed-endpoint 1-day jump reaches 85.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified. |
| Varansi Chirai Gaon Chhahi | False | True | -23.7140 | -13.7890 | 3 | 0 | 0.6983 | 8.3450 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 1.41 m; threshold 15 m); max observed-endpoint 1-day jump is 5.16 m. |
| Varansi Harahua Gurwat | False | True | -12.3110 | -0.1160 | 21 | 0 | 0.6714 | 11.6220 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 5.78 m; threshold 15 m); max observed-endpoint 1-day jump is 11.52 m. |
| Varansi Harahua Kakalpur | False | True | -29.5090 | -12.9120 | 0 | 0 | 0.6931 | 13.4500 | True | plausible hydrology | existing annual-median regime filter passes (max annual-median deviation 2.34 m; threshold 15 m); max observed-endpoint 1-day jump is 1.33 m. |

Each station has a 2021–2026 time-series PNG. These plots show recorded values and `gwl_obs=1` slots; they can reveal abrupt offsets or sustained steps, but alone do not establish a telemetry fault or hydrological cause.

## 2. Pre-cutoff quality-filter proposal

Rule: station p99 of observed, exact 30-day absolute GWL change before 2025-10-01 > 15m. Removed 222 of 600 stations with at least one qualifying pre-cutoff pair; stations without qualifying pairs are not removed by this rule.

| Station | pre_cutoff_p99_abs_30day_change_m | rule_threshold_m |
| --- | --- | --- |
| Jaunpur Rampur Kamaruddin | 442.5422 | 15.0000 |
| Gorakhpur Gagaha Gagaha | 187.3502 | 15.0000 |
| Primary School, Marry | 176.3216 | 15.0000 |
| Primary School, Sanori | 176.1871 | 15.0000 |
| Primary School, Kalan Bar Devran | 175.5329 | 15.0000 |
| Primary School/ Composite, Mundi Kheri | 152.8511 | 15.0000 |
| Composite School Kilhora | 115.5414 | 15.0000 |
| Uccha Prathmik Vidhyalaya, Pathouli | 107.9046 | 15.0000 |
| Primary School Gudha | 105.9643 | 15.0000 |
| Primary School Baijnathpur | 105.1140 | 15.0000 |
| Uccha Prathmik Vidhyalaya, Kapsarh | 104.9202 | 15.0000 |
| Primary School Salawa | 104.4307 | 15.0000 |
| Prathmik Vidhyalaya Sirasiya Pawar | 104.1911 | 15.0000 |
| Primary School Bhata Diha | 103.5781 | 15.0000 |
| Inter Collage Khampar | 103.3660 | 15.0000 |
| Punchayat Bhawan Bhaluwani | 103.2465 | 15.0000 |
| Uccha Prathmik Vidhyalaya, Chero | 103.0453 | 15.0000 |
| Aaganwadi Kendra Pakri Bazaar | 102.8618 | 15.0000 |
| B.D.O. Office Lar | 102.6922 | 15.0000 |
| Pratmik Vidhyalaya, Kakaruwa | 102.6654 | 15.0000 |
| Prathmik Vidhyalaya Pipra Sarvan | 102.3933 | 15.0000 |
| Primary School Dohriya | 100.2855 | 15.0000 |
| Primary School Jagdishpur Pratham | 99.7918 | 15.0000 |
| Composite School katar | 90.5020 | 15.0000 |
| Composite School Sanjarpur | 87.1816 | 15.0000 |
| Prathmik Vidhyalaya, Bali | 84.5686 | 15.0000 |
| Primary School, Sultanpur_1 | 79.8058 | 15.0000 |
| Primary School, Jogiya Khera | 77.2404 | 15.0000 |
| Purva Madhyamik Vidhyalaya, Chackwara | 75.9580 | 15.0000 |
| Primary School Chobla | 75.5000 | 15.0000 |
| Primary School chandsina | 74.9000 | 15.0000 |
| Composite School Nagla Bair | 74.9000 | 15.0000 |
| Prathmik Vidhyalaya, Nehlawa | 74.8000 | 15.0000 |
| Veterinary Hospital Shahjahanpur | 74.5000 | 15.0000 |
| Primary School Kurdi | 74.5000 | 15.0000 |
| Prathmik Vidhyalaya, Shahjahanpur_1 | 74.5000 | 15.0000 |
| Upper Primary School Pohalli | 74.4000 | 15.0000 |
| Purva Madhyamik Vidhyalaya Sihari | 74.4000 | 15.0000 |
| Composite Upper Primary School Sajania | 74.4000 | 15.0000 |
| Primary School No. 1 Barawad | 74.4000 | 15.0000 |
| Primary School Shekhpuri | 74.3000 | 15.0000 |
| Primary School Didhai | 74.3000 | 15.0000 |
| Primary School Mainhe hardo | 74.3000 | 15.0000 |
| Primary School Sonda | 74.3000 | 15.0000 |
| Primary School kakrehwa | 74.2000 | 15.0000 |
| Composite Uccha Prathmik Vidhyalaya, Paltadevi | 74.2000 | 15.0000 |
| Primary School Jatmujheda | 74.0000 | 15.0000 |
| Primary School Harivanshpur | 74.0000 | 15.0000 |
| Prathmik Vidhyalaya, Phaprana | 74.0000 | 15.0000 |
| Primary School Jagannathpur | 74.0000 | 15.0000 |
| Priamry School Mukasim | 73.8000 | 15.0000 |
| Uccha Prathmik Vidhyalaya Odera | 73.7190 | 15.0000 |
| Primary School Campus Aurangabad | 73.7000 | 15.0000 |
| Primary School Chandrapur Bichpuri | 73.6000 | 15.0000 |
| Primary School Rathoda | 73.6000 | 15.0000 |
| Primary School Pinna | 73.5000 | 15.0000 |
| Comosite School Campus Badalganj | 73.4000 | 15.0000 |
| Primary School Pahara | 73.3000 | 15.0000 |
| Primary School Charkonwa | 73.3000 | 15.0000 |
| Grampunchayat Sachiwalaya Deori | 73.3000 | 15.0000 |
| Primary School Masuri | 73.2000 | 15.0000 |
| Primary School Campus Powai | 73.2000 | 15.0000 |
| Prathmik Vidhyalaya Barha | 73.2000 | 15.0000 |
| Primary School Cholahe | 73.2000 | 15.0000 |
| Primary School Sahori | 73.1000 | 15.0000 |
| Composite School Basantpur Saitali-1 | 73.1000 | 15.0000 |
| Primary School Sanda-Sanee | 73.0000 | 15.0000 |
| R.R.C. Center Aata | 73.0000 | 15.0000 |
| Primary School Vamanpura | 72.9000 | 15.0000 |
| Upper Primary School Lehra | 72.9000 | 15.0000 |
| Upper Primary School Rijaula | 72.8000 | 15.0000 |
| Purva Madhyamik Vidhyalaya Piphara | 72.8000 | 15.0000 |
| Aagan Wadi Kendra Chandni | 72.8000 | 15.0000 |
| Upper Primary School Urouliya | 72.8000 | 15.0000 |
| Panchayat Bhawan Kuda Nar Singhpur | 72.7000 | 15.0000 |
| Composite School, Dhamna Khurd | 72.7000 | 15.0000 |
| Primary School Nandapur | 72.7000 | 15.0000 |
| Purva Madhyamik Vidhyalaya Baroda Dang | 72.6000 | 15.0000 |
| Primary School Ashadh | 72.6000 | 15.0000 |
| Primary School Radnool Azizpur | 72.5000 | 15.0000 |
| Primary School Bargdhi | 72.5000 | 15.0000 |
| Prathmik Vidhyalaya, Thavai | 72.4000 | 15.0000 |
| Office of Assistant Director, Resham | 72.4000 | 15.0000 |
| Primary School Madari | 72.3000 | 15.0000 |
| Composite Girls Primary School Mai | 72.3000 | 15.0000 |
| Gadhela Panchayat Bhawan | 72.2000 | 15.0000 |
| Poorv Madhamik Vidhyalaya Kaseti | 72.2000 | 15.0000 |
| Uccha Prathmik Vidhyalaya, Lohra (Pampawan) | 72.2000 | 15.0000 |
| Primary School Bilona | 72.2000 | 15.0000 |
| Goshala Basela | 72.1000 | 15.0000 |
| Upper Primary School Sarahan Bujurg | 72.1000 | 15.0000 |
| Rajkia Balika Inter College Kamasin | 72.1000 | 15.0000 |
| Primary School Kucharam | 72.1000 | 15.0000 |
| Prathmik Vidhyalaya Takuli (Durahi Maafi) | 72.1000 | 15.0000 |
| Primary School, Salhepur | 72.1000 | 15.0000 |
| Primary School Campus Sikraura | 72.0000 | 15.0000 |
| Swarg Dham, Hardua | 71.9000 | 15.0000 |
| Purva Madhyamik Vidhyalaya Bhitaura | 71.9000 | 15.0000 |
| Prathmik Vidhyalaya Manohari | 71.9000 | 15.0000 |
| Panchayat Bhawan Usmanpur | 71.9000 | 15.0000 |
| Prmary School Rajpura | 71.9000 | 15.0000 |
| Office of the Development Officer, Dakore | 71.9000 | 15.0000 |
| Primary School Jairampur | 71.9000 | 15.0000 |
| Primary School Jamunia | 71.8000 | 15.0000 |
| Goshala Pachhahan | 71.8000 | 15.0000 |
| Agan Wadi Kendra Kuchendu | 71.8000 | 15.0000 |
| Prathmik Vidhyalaya Pahalgaon | 71.7000 | 15.0000 |
| Rajkia High School Pakhrauli | 71.7000 | 15.0000 |
| Uccha Prathmik Vidhyalaya, Sonpur | 71.6000 | 15.0000 |
| Primary School Odaraha | 71.6000 | 15.0000 |
| Primary School, Kritikhand | 71.5000 | 15.0000 |
| Purva Madhyamik Vidhyalaya, Chandapur | 71.5000 | 15.0000 |
| Goshala Ghuri | 71.5000 | 15.0000 |
| Prathmik Vidhyalaya, Katiliaha | 71.4000 | 15.0000 |
| Uccha Prathmik Vidhyalaya, Gangapur | 71.4000 | 15.0000 |
| Uccha Prathmik Vidhyalaya, Semra Bhav Nagar | 71.4000 | 15.0000 |
| Goshala Bhadehdu | 71.4000 | 15.0000 |
| Punchayat Bhawan Vijaypur | 71.4000 | 15.0000 |
| Gram Panchayat Sachiwalaya / D.B.R. Ambedkar Samudayak Bhawan, Durkhuroo | 71.4000 | 15.0000 |
| Composite School, Ghamhapur | 71.4000 | 15.0000 |
| Composite School Sagarpur | 71.4000 | 15.0000 |
| Uccha Prathmik Vidhyalaya Deaval | 71.3000 | 15.0000 |
| Goshala Churaha | 71.2000 | 15.0000 |
| Prathmik Vidhyalaya, Makarndpur | 71.2000 | 15.0000 |
| Purva Madhyamik Vidhyalaya Achausa | 71.1000 | 15.0000 |
| Goshala Bada Gaon | 71.1000 | 15.0000 |
| Uccha Prathmik Vidhyalaya Uldan | 70.9000 | 15.0000 |
| Goshala Bankat | 70.7000 | 15.0000 |
| Purva Madhyamik Vidhyalaya Ladwari | 70.1000 | 15.0000 |
| Gundaha | 61.0900 | 15.0000 |
| Fugana_1 | 60.1430 | 15.0000 |
| Badhouli | 58.2072 | 15.0000 |
| Primary School No.1, Bitaoda | 57.2163 | 15.0000 |
| B.D.O. Office, Binauli | 55.6611 | 15.0000 |
| High School & Inter College Boroda | 55.3848 | 15.0000 |
| P.S Engohata | 54.8041 | 15.0000 |
| Narajit | 54.7745 | 15.0000 |
| Amlour | 54.7100 | 15.0000 |
| Bhitari_2 | 54.1500 | 15.0000 |
| Uccha Prathmik Vidhyalaya, Hasanpur Kadim | 53.9850 | 15.0000 |
| Paderi | 53.6900 | 15.0000 |
| Sikhula Pz._1 | 53.5900 | 15.0000 |
| Prathmik Vidhyalaya, Ramnagar | 52.5978 | 15.0000 |
| Palhari_1 | 51.6093 | 15.0000 |
| Madarpur_1 | 51.4000 | 15.0000 |
| Prathmik Vidhyalaya, Khampur | 50.9194 | 15.0000 |
| Naglakabu_1 | 50.5973 | 15.0000 |
| Budh Sainy_1 | 49.8654 | 15.0000 |
| Datt Naga_1 | 49.3485 | 15.0000 |
| Prathmik Vidhyalaya, vilodhpura | 48.3222 | 15.0000 |
| Rohari | 47.5670 | 15.0000 |
| Prathmik Vidhyalaya, Gwalikhera | 47.4752 | 15.0000 |
| Mankuwar | 47.3500 | 15.0000 |
| Kandhari Kala | 45.8300 | 15.0000 |
| Chitrakoot Manikpur Agarhudha | 45.8208 | 15.0000 |
| Prathmik Vidhyalaya, Sikhera_1 | 45.7587 | 15.0000 |
| Terha_1 | 45.7400 | 15.0000 |
| Upper Primary Kasampur | 45.7191 | 15.0000 |
| Prathmik Vidhyalaya, Doula | 45.3610 | 15.0000 |
| Sitapur Mafi | 44.2054 | 15.0000 |
| Perthivipur Nayakh | 44.0400 | 15.0000 |
| Bijijauri_1 | 44.0400 | 15.0000 |
| Baghora_1 | 43.6228 | 15.0000 |
| Prathmik Vidhyalaya, Medpur | 42.9446 | 15.0000 |
| Luktra | 41.3848 | 15.0000 |
| Panchayat Bhawan  Suhel | 40.5897 | 15.0000 |
| Kusarma | 40.2540 | 15.0000 |
| Bhadarwara  P_1 | 39.4300 | 15.0000 |
| Khedki_1 | 39.1377 | 15.0000 |
| Primary School, Salarpur Jalalpur | 38.6521 | 15.0000 |
| Prathmik Vidhyalaya, Biharipur | 38.6127 | 15.0000 |
| Piparaond | 38.1109 | 15.0000 |
| Birdha /Poorabirdha | 37.6200 | 15.0000 |
| Sukwan | 37.5900 | 15.0000 |
| Prathmik Vidhyala, Riawali Nagla | 37.2628 | 15.0000 |
| Chirautha | 36.9192 | 15.0000 |
| Gloragundera/Gewara | 36.8600 | 15.0000 |
| Pathrodi | 36.3900 | 15.0000 |
| Rasin_2 | 36.3592 | 15.0000 |
| Prathmik Vidhyalaya, AhmadShahPur Padra | 36.1506 | 15.0000 |
| Rajkiya High School, Atali | 36.1125 | 15.0000 |
| Jr. High School, Purwa Tarouha | 36.0265 | 15.0000 |
| Devrar_1 | 35.7067 | 15.0000 |
| Chhibaw_1 | 35.6743 | 15.0000 |
| Semara_1 | 35.2707 | 15.0000 |
| primary school jalalpura | 35.1364 | 15.0000 |
| Primary School, Silana | 34.9097 | 15.0000 |
| Composite School Biralsi | 34.6070 | 15.0000 |
| Chandi Kalan | 34.6042 | 15.0000 |
| Jamalpur_2 | 33.8292 | 15.0000 |
| Murwa_1 | 33.6725 | 15.0000 |
| Panchayat Bhawan , Jafarabad Nagla | 33.4789 | 15.0000 |
| Ladaka Purwa | 33.2342 | 15.0000 |
| Jaurahi_1 | 33.0400 | 15.0000 |
| Amchurnirua | 32.8800 | 15.0000 |
| Patanpur_1 | 32.6421 | 15.0000 |
| Dhikauli | 32.3700 | 15.0000 |
| primary school kapasi | 31.9620 | 15.0000 |
| Kamhariya_1 | 31.7685 | 15.0000 |
| Pahara_2 | 31.7650 | 15.0000 |
| Primary School English Medium Qutubpur Labdauna | 31.6478 | 15.0000 |
| Babina Rular | 31.3400 | 15.0000 |
| Prathmik Vidhyalaya, Bichpari Kamali | 30.3984 | 15.0000 |
| Prathmik Vidhyalaya, Didhora  Sadullapur | 30.0439 | 15.0000 |
| Purva Madhyamik Vidyalaya Chalakpur | 29.0666 | 15.0000 |
| Uchha Prathmik Vidhyala, Tadi Mustkil | 28.9505 | 15.0000 |
| Pipraundha | 28.8152 | 15.0000 |
| Shitalpur_1 | 28.4205 | 15.0000 |
| Primary School, Bamhauri | 28.1000 | 15.0000 |
| Jauharpur | 26.6751 | 15.0000 |
| Prathmik Vidhyalaya, Taizullahpur | 25.6082 | 15.0000 |
| Primary School/ Composite, Sidki | 25.5787 | 15.0000 |
| Awagarh (UP-005) | 24.8953 | 15.0000 |
| Chitrakoot Karvi Shivrampur | 24.5135 | 15.0000 |
| Chitrakoot Manikpur Sardar Patel | 24.0730 | 15.0000 |
| Aligarh Chandaus  Somana | 23.8430 | 15.0000 |
| Prathmik Vidhyalaya, Khera Islampur | 23.4851 | 15.0000 |
| Shahjahanpur Sidhauli Mahanand Pur | 22.8473 | 15.0000 |
| Hathras Hathras Hathras | 18.3370 | 15.0000 |
| Primary School, Karasdev Mandir | 16.2062 | 15.0000 |
| Purv Madhyamik Vidhyalaya, Doriya | 15.5585 | 15.0000 |
| Chakara  P_1 | 15.3324 | 15.0000 |

Model skill comparison (paired rows; all stations vs retained stations):

| artifact | subset | n | n_stations | rmse_model | rmse_persistence | skill | mae_model | mae_persistence | mae_skill | median_station_skill | winsor_cap_abs_error_p99 | winsor_rmse_model | winsor_rmse_persistence | winsor_rmse_skill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost 30d | all stations | 419682 | 511 | 2.2594 | 2.3209 | 0.0265 | 0.7721 | 0.8259 | 0.0651 | 0.0778 | 7.1668 | 1.2901 | 1.3673 | 0.0565 |
| XGBoost 30d | retained after pre-cutoff rule | 305660 | 371 | 1.4204 | 1.5137 | 0.0616 | 0.6663 | 0.7156 | 0.0688 | 0.0764 | 5.5944 | 1.0825 | 1.1651 | 0.0709 |
| Trajectory DL h720 | all stations | 2023 | 137 | 1.8021 | 1.9071 | 0.0550 | 0.6599 | 0.7385 | 0.1064 | 0.1154 | 5.9825 | 1.1270 | 1.2321 | 0.0853 |
| Trajectory DL h720 | retained after pre-cutoff rule | 1398 | 93 | 1.9720 | 2.0712 | 0.0479 | 0.6823 | 0.7679 | 0.1115 | 0.1207 | 5.7183 | 1.1038 | 1.2060 | 0.0848 |

The station removal rule uses only pre-cutoff raw GWL history. Model errors were used only for the requested post-filter evaluation, not to choose the rule or threshold.

## 3. Skill by distance to rain gauge

| artifact | distance_bucket | n | n_stations | rmse_model | rmse_persistence | skill | mae_model | mae_persistence | mae_skill | median_station_skill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost 30d | 25-75 km | 178628 | 215 | 2.9295 | 2.9882 | 0.0197 | 0.9006 | 0.9565 | 0.0584 | 0.0832 |
| XGBoost 30d | <25 km | 148830 | 183 | 1.4124 | 1.5060 | 0.0622 | 0.6972 | 0.7645 | 0.0879 | 0.0955 |
| XGBoost 30d | none | 40433 | 50 | 1.9476 | 2.0055 | 0.0289 | 0.6138 | 0.6483 | 0.0531 | 0.0528 |
| XGBoost 30d | 75-150 km | 51791 | 63 | 1.7532 | 1.7869 | 0.0188 | 0.6676 | 0.6910 | 0.0339 | 0.0570 |
| Trajectory DL h720 | 25-75 km | 908 | 60 | 2.3592 | 2.4663 | 0.0434 | 0.7798 | 0.8696 | 0.1033 | 0.1154 |
| Trajectory DL h720 | <25 km | 734 | 50 | 1.0850 | 1.1978 | 0.0941 | 0.5357 | 0.6139 | 0.1274 | 0.1231 |
| Trajectory DL h720 | 75-150 km | 232 | 16 | 1.6216 | 1.7657 | 0.0816 | 0.7762 | 0.8334 | 0.0686 | 0.0584 |
| Trajectory DL h720 | none | 149 | 11 | 0.5296 | 0.6230 | 0.1499 | 0.3600 | 0.4054 | 0.1120 | 0.1379 |

## 4. h18 persistence-RMSE anomaly

The saved `obs` is an absolute target level, not a change. The h18 jump is dominated by `Primary School, Dera Devra bujurg`: its 2026-03-04 00:00 anchor is 27.500m and the observed level 18 hours later is -7.921m (35.421m change); both endpoint flags are `gwl_obs=1`. Its h18 persistence squared-error contribution is 1,255.06 of about 1,697.56 total (73.9%). Of 2,207 h18 rows, 2,148 (97.3%) have a midnight anchor; these contribute 1,690.94 squared error (99.6%). The 0.877-to-0.819 RMSE drop at h24 does not remove the anomaly: Dera Devra remains the leading h24 station with 1,232.46 squared error (83.3% of h24 total), while other stations’ combined squared error drops from about 442.5 at h18 to 247.5 at h24. Dera Devra is not a top-10 station at h12; top-10 station overlap with h18 is 3/10 at h12 and 4/10 at h24. This is a recorded abrupt level discontinuity; the file does not prove whether it was a telemetry fault or real site event.

The horizon summary compares h6/h12/h18/h24, observed-flag rates, and whether saved absolute `obs` matches the exact raw target level. See the station contribution, anchor-hour, flag-group and top-anchor CSVs for attribution.

| horizon_h | n | n_stations | persistence_rmse | persistence_mae | share_anchor_observed | share_target_observed | share_both_observed | artifact_obs_matches_raw_target_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2213.0000 | 137.0000 | 0.2922 | 0.1298 | 0.8906 | 0.9146 | 0.8694 | 1.0000 |
| 12 | 2211.0000 | 137.0000 | 0.3404 | 0.1650 | 0.8915 | 0.8942 | 0.8684 | 1.0000 |
| 18 | 2207.0000 | 137.0000 | 0.8770 | 0.2066 | 0.8931 | 0.9252 | 0.8595 | 1.0000 |
| 24 | 2209.0000 | 137.0000 | 0.8185 | 0.1270 | 0.8923 | 0.9077 | 0.8515 | 1.0000 |

### Persistence error by anchor hour

| horizon_h | anchor_hour | n | n_stations | persistence_rmse | persistence_mae | squared_error |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 0 | 2154 | 137 | 0.2941 | 0.1305 | 186.3528 |
| 6 | 6 | 48 | 43 | 0.1891 | 0.0964 | 1.7173 |
| 6 | 12 | 5 | 5 | 0.0959 | 0.0570 | 0.0460 |
| 6 | 18 | 6 | 6 | 0.3712 | 0.2170 | 0.8267 |
| 12 | 0 | 2152 | 137 | 0.3397 | 0.1646 | 248.3384 |
| 12 | 6 | 48 | 43 | 0.2736 | 0.1433 | 3.5942 |
| 12 | 12 | 5 | 5 | 0.3142 | 0.1728 | 0.4937 |
| 12 | 18 | 6 | 6 | 0.7908 | 0.4830 | 3.7523 |
| 18 | 0 | 2148 | 137 | 0.8873 | 0.2082 | 1690.9363 |
| 18 | 6 | 48 | 43 | 0.1780 | 0.0976 | 1.5214 |
| 18 | 12 | 5 | 5 | 0.3094 | 0.1628 | 0.4788 |
| 18 | 18 | 6 | 6 | 0.8766 | 0.5367 | 4.6108 |
| 24 | 0 | 2150 | 137 | 0.8295 | 0.1288 | 1479.2744 |
| 24 | 6 | 48 | 43 | 0.0948 | 0.0503 | 0.4313 |
| 24 | 12 | 5 | 5 | 0.0841 | 0.0470 | 0.0353 |
| 24 | 18 | 6 | 6 | 0.2039 | 0.1555 | 0.2495 |

### Persistence error by exact observed flags

| horizon_h | anchor_gwl_obs | target_gwl_obs | n | persistence_rmse | persistence_mae | n_stations |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.0000 | 0.0000 | 142 | 0.0000 | 0.0000 | 82 |
| 6 | 0.0000 | 1.0000 | 100 | 0.4201 | 0.1762 | 44 |
| 6 | 1.0000 | 0.0000 | 47 | 0.0000 | 0.0000 | 38 |
| 6 | 1.0000 | 1.0000 | 1924 | 0.2984 | 0.1401 | 136 |
| 12 | 0.0000 | 0.0000 | 183 | 0.1212 | 0.0287 | 81 |
| 12 | 0.0000 | 1.0000 | 57 | 0.5660 | 0.2575 | 41 |
| 12 | 1.0000 | 0.0000 | 51 | 0.2108 | 0.0608 | 42 |
| 12 | 1.0000 | 1.0000 | 1920 | 0.3483 | 0.1780 | 136 |
| 18 | 0.0000 | 0.0000 | 91 | 0.0057 | 0.0009 | 65 |
| 18 | 0.0000 | 1.0000 | 145 | 0.5684 | 0.2961 | 69 |
| 18 | 1.0000 | 0.0000 | 74 | 0.2454 | 0.1119 | 58 |
| 18 | 1.0000 | 1.0000 | 1897 | 0.9316 | 0.2133 | 136 |
| 24 | 0.0000 | 0.0000 | 114 | 0.2162 | 0.0772 | 46 |
| 24 | 0.0000 | 1.0000 | 124 | 0.6385 | 0.3154 | 90 |
| 24 | 1.0000 | 0.0000 | 90 | 0.2224 | 0.1021 | 64 |
| 24 | 1.0000 | 1.0000 | 1881 | 0.8688 | 0.1187 | 137 |

### Largest station contributors by horizon

| horizon_h | Station | n | persistence_rmse | squared_error | share_horizon_sqerr | share_anchor_observed | share_target_observed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Primary School/ Composite, Mehmoodpur | 14 | 1.0118 | 14.3318 | 0.0759 | 1.0000 | 0.9286 |
| 6 | Block Deoria Campus Deoria Sadar | 17 | 0.8942 | 13.5934 | 0.0719 | 0.8824 | 0.8824 |
| 6 | Purv Madhyamik Vidhyalaya, Doriya | 17 | 0.8723 | 12.9356 | 0.0685 | 0.9412 | 0.9412 |
| 6 | MANDHATA PURV MADHYAMIK VIDYALAYA | 17 | 0.8193 | 11.4113 | 0.0604 | 0.9412 | 0.9412 |
| 6 | Varansi Chirai Gaon Chhahi | 16 | 0.8229 | 10.8340 | 0.0573 | 0.3750 | 0.9375 |
| 6 | Primary School, Kevtam | 17 | 0.7701 | 10.0832 | 0.0534 | 0.8235 | 0.8235 |
| 6 | Primary School, Dhikana | 15 | 0.7906 | 9.3757 | 0.0496 | 1.0000 | 1.0000 |
| 6 | Chitrakoot Manikpur Agarhudha | 16 | 0.7464 | 8.9138 | 0.0472 | 0.9375 | 1.0000 |
| 6 | Panchayat Bhwan, Kon | 18 | 0.5827 | 6.1117 | 0.0323 | 0.8333 | 0.8333 |
| 6 | Primary School, Jhaaro Kalan | 18 | 0.5372 | 5.1943 | 0.0275 | 0.8333 | 0.8333 |
| 6 | Navrasia | 16 | 0.5163 | 4.2654 | 0.0226 | 0.9375 | 0.9375 |
| 6 | Prathmik Vidhyalaya, Biharipur | 16 | 0.4971 | 3.9533 | 0.0209 | 0.9375 | 0.8750 |
| 6 | Saint Ravidas Nagar Suriyawan Mond | 17 | 0.4603 | 3.6024 | 0.0191 | 0.9412 | 0.9412 |
| 6 | Mathura Chaumuha Navagaon-11 | 19 | 0.4342 | 3.5816 | 0.0190 | 0.7368 | 0.7895 |
| 6 | Purv Madh Vidhyalaya, Doomradeeh | 17 | 0.4575 | 3.5576 | 0.0188 | 0.8824 | 0.8824 |
| 6 | Primary School, Bandpur | 16 | 0.4477 | 3.2071 | 0.0170 | 0.9375 | 0.8750 |
| 6 | Halwana | 18 | 0.4170 | 3.1298 | 0.0166 | 0.8333 | 0.8333 |
| 6 | Primary School English Medium Qutubpur Labdauna | 17 | 0.4233 | 3.0467 | 0.0161 | 1.0000 | 1.0000 |
| 6 | Meerut Janikhurd Satvai | 16 | 0.4359 | 3.0401 | 0.0161 | 0.9375 | 1.0000 |
| 6 | Azamgarh Jahanganj Godhaura | 17 | 0.4164 | 2.9479 | 0.0156 | 0.9412 | 0.9412 |
| 12 | Primary School/ Composite, Landhoura Junardar | 17 | 1.1335 | 21.8420 | 0.0853 | 0.9412 | 0.9412 |
| 12 | Primary School English Medium Qutubpur Labdauna | 17 | 1.0688 | 19.4210 | 0.0758 | 1.0000 | 1.0000 |
| 12 | Primary School, Dhikana | 15 | 1.1101 | 18.4851 | 0.0722 | 1.0000 | 0.9333 |
| 12 | Primary School, Jhaaro Kalan | 18 | 0.8788 | 13.9000 | 0.0543 | 0.8333 | 0.8333 |
| 12 | Primary School, Kevtam | 17 | 0.8468 | 12.1890 | 0.0476 | 0.8235 | 0.8235 |
| 12 | Primary School/ Composite, Mehmoodpur | 14 | 0.9185 | 11.8119 | 0.0461 | 1.0000 | 0.9286 |
| 12 | Chitrakoot Manikpur Agarhudha | 16 | 0.8362 | 11.1887 | 0.0437 | 0.9375 | 1.0000 |
| 12 | Salhakheri | 17 | 0.7373 | 9.2419 | 0.0361 | 0.8824 | 0.8824 |
| 12 | Mathura Chaumuha Navagaon-11 | 19 | 0.6638 | 8.3711 | 0.0327 | 0.7368 | 0.7895 |
| 12 | Meerut Janikhurd Afjalpur Pavati | 16 | 0.6572 | 6.9102 | 0.0270 | 0.9375 | 1.0000 |
| 12 | Varansi Chirai Gaon Chhahi | 16 | 0.6542 | 6.8479 | 0.0267 | 0.3750 | 0.3750 |
| 12 | Primary School Harinahpur | 18 | 0.5543 | 5.5302 | 0.0216 | 0.8889 | 0.8889 |
| 12 | Alakchandra Inter college Bhinga | 16 | 0.5663 | 5.1315 | 0.0200 | 0.9375 | 0.9375 |
| 12 | Primary School, Bandpur | 16 | 0.5601 | 5.0191 | 0.0196 | 0.9375 | 0.8750 |
| 12 | Primary School, Gauripur Javahar Nagar | 16 | 0.5381 | 4.6331 | 0.0181 | 0.8750 | 0.9375 |
| 12 | Azamgarh Jahanganj Godhaura | 17 | 0.4901 | 4.0839 | 0.0159 | 0.9412 | 0.8824 |
| 12 | Navrasia | 16 | 0.4987 | 3.9799 | 0.0155 | 0.9375 | 0.9375 |
| 12 | Prathmik Vidhyalaya, Biharipur | 16 | 0.4960 | 3.9366 | 0.0154 | 0.9375 | 0.8750 |
| 12 | Primary School Salawa | 16 | 0.4924 | 3.8798 | 0.0151 | 1.0000 | 1.0000 |
| 12 | Purv Madhyamik Vidhyalaya, Doriya | 17 | 0.4538 | 3.5002 | 0.0137 | 0.9412 | 0.9412 |
| 18 | Primary School, Dera Devra bujurg | 17 | 8.5923 | 1255.0569 | 0.7393 | 1.0000 | 0.9412 |
| 18 | Firozabad Madanpur Rudhawali | 11 | 2.5504 | 71.5505 | 0.0421 | 1.0000 | 1.0000 |
| 18 | Hamirpur Kurara Jalokhar | 15 | 2.0225 | 61.3571 | 0.0361 | 1.0000 | 0.9333 |
| 18 | Block Deoria Campus Deoria Sadar | 17 | 1.3227 | 29.7424 | 0.0175 | 0.8824 | 0.9412 |
| 18 | Community Center, Ghorawal | 18 | 1.0703 | 20.6204 | 0.0121 | 0.8333 | 0.8333 |
| 18 | Khursropur | 17 | 1.0806 | 19.8518 | 0.0117 | 0.8824 | 0.9412 |
| 18 | Primary School, Belaukh | 17 | 1.0396 | 18.3715 | 0.0108 | 0.8824 | 0.8824 |
| 18 | Primary School/ Composite, Landhoura Junardar | 17 | 0.9997 | 16.9906 | 0.0100 | 0.9412 | 0.9412 |
| 18 | Primary School, Kevtam | 17 | 0.9745 | 16.1429 | 0.0095 | 0.8235 | 0.8824 |
| 18 | Chitrakoot Manikpur Agarhudha | 16 | 0.9084 | 13.2028 | 0.0078 | 0.9375 | 1.0000 |
| 18 | Goyla | 18 | 0.8494 | 12.9878 | 0.0077 | 0.8889 | 0.8889 |
| 18 | Primary School/ Composite, Mehmoodpur | 14 | 0.8345 | 9.7504 | 0.0057 | 1.0000 | 0.9286 |
| 18 | MANDHATA PURV MADHYAMIK VIDYALAYA | 17 | 0.6982 | 8.2872 | 0.0049 | 0.9412 | 0.8824 |
| 18 | Mathura Chaumuha Navagaon-11 | 19 | 0.6497 | 8.0190 | 0.0047 | 0.7368 | 0.7895 |
| 18 | Mainpuri Karhal  Rajpur | 13 | 0.7216 | 6.7683 | 0.0040 | 1.0000 | 0.9231 |
| 18 | Panchayat Bhwan, Kon | 17 | 0.6295 | 6.7364 | 0.0040 | 0.8824 | 0.9412 |
| 18 | Halwana | 17 | 0.6242 | 6.6232 | 0.0039 | 0.8824 | 0.8235 |
| 18 | Primary School  Maharaji | 18 | 0.5668 | 5.7826 | 0.0034 | 0.7778 | 0.8889 |
| 18 | Meerut Janikhurd Afjalpur Pavati | 16 | 0.5687 | 5.1747 | 0.0030 | 0.9375 | 1.0000 |
| 18 | Primary School English Medium Qutubpur Labdauna | 17 | 0.5504 | 5.1504 | 0.0030 | 1.0000 | 1.0000 |
| 24 | Primary School, Dera Devra bujurg | 17 | 8.5146 | 1232.4586 | 0.8327 | 1.0000 | 1.0000 |
| 24 | Hamirpur Kurara Jalokhar | 15 | 2.0165 | 60.9960 | 0.0412 | 1.0000 | 1.0000 |
| 24 | Mathura Chaumuha Navagaon-11 | 19 | 1.4897 | 42.1662 | 0.0285 | 0.7368 | 0.7895 |
| 24 | Community Center, Ghorawal | 18 | 1.0473 | 19.7438 | 0.0133 | 0.8333 | 0.8333 |
| 24 | Chitrakoot Manikpur Agarhudha | 16 | 0.9661 | 14.9342 | 0.0101 | 0.9375 | 1.0000 |
| 24 | Varansi Chirai Gaon Chhahi | 16 | 0.9033 | 13.0546 | 0.0088 | 0.3750 | 0.3750 |
| 24 | Saint Ravidas Nagar Abholi Gangarampur | 16 | 0.6128 | 6.0092 | 0.0041 | 0.9375 | 1.0000 |
| 24 | Saint Ravidas Nagar Suriyawan Mond | 17 | 0.5580 | 5.2932 | 0.0036 | 0.9412 | 1.0000 |
| 24 | Swarg Dham, Hardua | 17 | 0.5459 | 5.0665 | 0.0034 | 0.9412 | 0.8824 |
| 24 | Goyla | 18 | 0.5189 | 4.8465 | 0.0033 | 0.8889 | 0.8889 |
| 24 | Primary School/ Composite, Mehmoodpur | 14 | 0.5205 | 3.7929 | 0.0026 | 1.0000 | 0.9286 |
| 24 | Saint Ravidas Nagar Gyanpur Kansapur | 17 | 0.4612 | 3.6167 | 0.0024 | 0.9412 | 1.0000 |
| 24 | Rohari | 14 | 0.4556 | 2.9056 | 0.0020 | 0.7143 | 0.7143 |
| 24 | Azamgarh Lalganj Lalganj | 16 | 0.4142 | 2.7445 | 0.0019 | 0.8750 | 1.0000 |
| 24 | Khursropur | 17 | 0.3853 | 2.5239 | 0.0017 | 0.8824 | 0.8824 |
| 24 | Primary School/ Composite, Landhoura Junardar | 17 | 0.3837 | 2.5032 | 0.0017 | 0.9412 | 0.9412 |
| 24 | Primary School, Garher | 15 | 0.3956 | 2.3471 | 0.0016 | 0.8667 | 1.0000 |
| 24 | Primary School, Dhanaura Tikari | 17 | 0.3651 | 2.2655 | 0.0015 | 0.9412 | 1.0000 |
| 24 | Prathmik Vidhyalaya Takuli (Durahi Maafi) | 16 | 0.3719 | 2.2135 | 0.0015 | 1.0000 | 1.0000 |
| 24 | Primary School, Dewari | 17 | 0.3599 | 2.2019 | 0.0015 | 0.9412 | 1.0000 |

## Output files

- `outlier_forensics.csv`: union station metrics, classification, and evidence.
- `precutoff_quality_rule_by_station.csv`, `quality_filter_removed_stations.csv`, `quality_filter_skill_comparison.csv`.
- `rain_gauge_distance_skill.csv`, `rain_gauge_distance_by_station.csv`.
- `h18_horizon_observation_summary.csv`, `h18_observation_flag_groups.csv`, `h18_anchor_hour_summary.csv`, `h18_top_station_contributions.csv`, `h18_top_anchor_errors.csv`, `h18_top10_station_overlap.csv`.
- PNG filenames use a readable station slug and a stable short hash.

### Station plots

- [Aagan Wadi Kendra Chandni](Aagan_Wadi_Kendra_Chandni_c624043c.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 85.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Banda Tindwari Bhindowra](Banda_Tindwari_Bhindowra_5ab415d2.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 17.24 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Block Office, Kurara](Block_Office_Kurara_6a43ebdf.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 1.56 m; threshold 15 m); max observed-endpoint 1-day jump is 1.52 m.
- [Chitrakoot Manikpur Agarhudha](Chitrakoot_Manikpur_Agarhudha_b3a08ed7.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 46.60 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Community Center, Ghorawal](Community_Center_Ghorawal_9da5e2e2.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 3.49 m; threshold 15 m); max observed-endpoint 1-day jump is 4.37 m.
- [Composite School Basantpur Saitali-1](Composite_School_Basantpur_Saitali_1_48733dd2.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 73.10 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Composite School Sagarpur](Composite_School_Sagarpur_6029996a.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 71.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Composite School, Ghamhapur](Composite_School_Ghamhapur_78404e96.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 71.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Gadhela Panchayat Bhawan](Gadhela_Panchayat_Bhawan_5fb00cbd.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 86.10 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Goshala Bada Gaon](Goshala_Bada_Gaon_b159b3e9.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 85.10 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Goshala Bhadehdu](Goshala_Bhadehdu_f3c58ede.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 71.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Hamirpur Kurara Jalokhar](Hamirpur_Kurara_Jalokhar_8a80fdac.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 1.56 m; threshold 15 m); max observed-endpoint 1-day jump is 10.15 m.
- [Hamirpur Sarila Purani](Hamirpur_Sarila_Purani_77769dfd.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 2.36 m; threshold 15 m); max observed-endpoint 1-day jump is 2.62 m.
- [Hathras Hathras Hathras](Hathras_Hathras_Hathras_2a8a1743.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 19.97 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Mathura Chaumuha Navagaon-11](Mathura_Chaumuha_Navagaon_11_d222442f.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 3.73 m; threshold 15 m); max observed-endpoint 1-day jump is 12.66 m.
- [Primary School Sanda-Sanee](Primary_School_Sanda_Sanee_de79c34f.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 85.70 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Primary School Vamanpura](Primary_School_Vamanpura_a5e98d47.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 73.70 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Primary School chandsina](Primary_School_chandsina_19b8f8ca.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 74.90 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Primary School, Bilata](Primary_School_Bilata_6466a5a9.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 7.46 m; threshold 15 m); max observed-endpoint 1-day jump is 4.95 m.
- [Primary School, Dera Devra bujurg](Primary_School_Dera_Devra_bujurg_e457cc5a.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 35.63 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Primary School, Evani](Primary_School_Evani_5c231b53.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 3.05 m; threshold 15 m); max observed-endpoint 1-day jump is 1.18 m.
- [Primary School, Garher](Primary_School_Garher_6bb34643.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 1.61 m; threshold 15 m); max observed-endpoint 1-day jump is 5.02 m.
- [Primary School, Kalan Bar Devran](Primary_School_Kalan_Bar_Devran_e00cb88d.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 338.06 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Primary School, Karasdev Mandir](Primary_School_Karasdev_Mandir_aa4df36d.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 6.03 m; threshold 15 m); max observed-endpoint 1-day jump is 10.13 m.
- [Primary School, Silana](Primary_School_Silana_659344af.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 38.64 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Primary School/ Composite, Landhoura Junardar](Primary_School_Composite_Landhoura_Junardar_df2280a9.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 1.93 m; threshold 15 m); max observed-endpoint 1-day jump is 2.84 m.
- [Primary School/ Composite, Mehmoodpur](Primary_School_Composite_Mehmoodpur_25f733ed.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 138.58 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Primary School/ Composite, Sidki](Primary_School_Composite_Sidki_10430c49.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 26.00 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Prmary School Rajpura](Prmary_School_Rajpura_45acd6c7.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 72.70 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Rajkia High School Pakhrauli](Rajkia_High_School_Pakhrauli_9300882f.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 85.80 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Rohari](Rohari_01564bf3.png) — consistent with datum/telemetry fault. existing annual-median regime filter fails: maximum deviation from 2021-24 station baseline = 36.73 m (>15 m); consistent with a persistent offset, though cause is unverified.
- [Saint Ravidas Nagar Abholi Gangarampur](Saint_Ravidas_Nagar_Abholi_Gangarampur_3f251e95.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 3.67 m; threshold 15 m); max observed-endpoint 1-day jump is 1.15 m.
- [Swarg Dham, Hardua](Swarg_Dham_Hardua_295af2d9.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 85.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Uccha Prathmik Vidhyalaya, Lohra (Pampawan)](Uccha_Prathmik_Vidhyalaya_Lohra_Pampawan_ea14389d.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 72.40 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Upper Primary School Sarahan Bujurg](Upper_Primary_School_Sarahan_Bujurg_68ef875f.png) — consistent with datum/telemetry fault. filter passes, but an observed-endpoint 1-day jump reaches 72.10 m (>15 m); the abrupt discontinuity is consistent with a datum/telemetry fault, though cause is unverified.
- [Varansi Chirai Gaon Chhahi](Varansi_Chirai_Gaon_Chhahi_23f8a40f.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 1.41 m; threshold 15 m); max observed-endpoint 1-day jump is 5.16 m.
- [Varansi Cholapur Chaubepur](Varansi_Cholapur_Chaubepur_82ecaf78.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 0.95 m; threshold 15 m); max observed-endpoint 1-day jump is 12.54 m.
- [Varansi Harahua Gurwat](Varansi_Harahua_Gurwat_9c54f021.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 5.78 m; threshold 15 m); max observed-endpoint 1-day jump is 11.52 m.
- [Varansi Harahua Kakalpur](Varansi_Harahua_Kakalpur_86728333.png) — plausible hydrology. existing annual-median regime filter passes (max annual-median deviation 2.34 m; threshold 15 m); max observed-endpoint 1-day jump is 1.33 m.

## Cleaning addendum: multi-pattern detectors and review plots

### Threshold fitting and safe-treatment rules

All numeric detector cutoffs below were fitted from `gwl_obs=1` data with timestamps strictly before 2025-10-01. The 30-day/120-slot window is the module horizon. Exact repeated-spike masking additionally requires an observed isolated exact value, repeated at least the training-fitted count, with immediate observed exact reversion to the same value on both sides. Smooth-recovery events are recorded and kept. Flat, flip-flop, and change-point segments are flagged for review; they are not silently deleted.

| detector | threshold_slots | fit | window_slots |
| --- | --- | --- | --- |
| constant_value_run | 102.0000 | ceil(p99.99 of observed exact-value run lengths in pre-2025-10-01 training data) | 1 |
| repeated_exact_extreme | 3.0000 | max(2, ceil(p95 of pre-cutoff exact-value frequencies among station tail values <=p0.5 or >=p99.5)) | 1 |
| extreme_value |  | station-specific pre-cutoff p0.5/p99.5; known sentinels 99,-99,999,-999,9999,-9999 are also candidates | 1 |
| near_flat_30d_std | 0.0094 | p1 of pre-cutoff 30d rolling standard deviations; requires >=80 slots | 120 |
| flipflop_step_size |  | station-specific p95 of pre-cutoff observed consecutive-read absolute differences (6h to 48h apart) | 120 |
| flipflop_reversal_count | 434.0000 | ceil(p99.5 of pre-cutoff rolling 30d count of consecutive large sign reversals) | 120 |
| whole_station_invalid_share | 0.0000 | p99 of per-station pre-cutoff share of observed slots covered by flipflop windows | 120 |
| persistent_month_shift |  | station-specific pre-cutoff p99.5 of consecutive observed monthly-median absolute changes; pooled fallback=72.1860m; confirmation uses next available month | 0 |

### 39-station strict labels and disagreements

| Station | phase2_classification | strict_classification | classification_disagrees | placeholder_start_slots | isolated_repeated_extreme_spikes_to_mask | test_stuck_slots | flipflop_invalid_share_pre_observed | flipflop_invalid_share_all_observed | change_points | pumping_like_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aagan Wadi Kendra Chandni | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Banda Tindwari Bhindowra | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 3 | 5 |
| Block Office, Kurara | plausible hydrology | clean | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 0 |
| Chitrakoot Manikpur Agarhudha | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 2 |
| Community Center, Ghorawal | plausible hydrology | clean | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Composite School Basantpur Saitali-1 | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 1 |
| Composite School Sagarpur | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Composite School, Ghamhapur | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Gadhela Panchayat Bhawan | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Goshala Bada Gaon | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Goshala Bhadehdu | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Hamirpur Kurara Jalokhar | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 3 |
| Hamirpur Sarila Purani | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 3 | 4 |
| Hathras Hathras Hathras | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 1 |
| Mathura Chaumuha Navagaon-11 | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 3 |
| Primary School Sanda-Sanee | consistent with datum/telemetry fault | spike-faults on clean series | False | 0 | 2 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Primary School Vamanpura | consistent with datum/telemetry fault | spike-faults on clean series | False | 0 | 1 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Primary School chandsina | consistent with datum/telemetry fault | spike-faults on clean series | False | 0 | 2 | 0 | 0.0000 | 0.0000 | 3 | 0 |
| Primary School, Bilata | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 1 |
| Primary School, Dera Devra bujurg | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 4 | 3 |
| Primary School, Evani | plausible hydrology | clean | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 0 |
| Primary School, Garher | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 1 |
| Primary School, Kalan Bar Devran | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 8 |
| Primary School, Karasdev Mandir | plausible hydrology | clean | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 0 |
| Primary School, Silana | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 5 |
| Primary School/ Composite, Landhoura Junardar | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 3 |
| Primary School/ Composite, Mehmoodpur | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 7 |
| Primary School/ Composite, Sidki | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 3 |
| Prmary School Rajpura | consistent with datum/telemetry fault | spike-faults on clean series | False | 0 | 5 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Rajkia High School Pakhrauli | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Rohari | consistent with datum/telemetry fault | pumping-like real | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 6 |
| Saint Ravidas Nagar Abholi Gangarampur | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 3 |
| Swarg Dham, Hardua | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Uccha Prathmik Vidhyalaya, Lohra (Pampawan) | consistent with datum/telemetry fault | clean | True | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Upper Primary School Sarahan Bujurg | consistent with datum/telemetry fault | spike-faults on clean series | False | 0 | 2 | 0 | 0.0000 | 0.0000 | 1 | 0 |
| Varansi Chirai Gaon Chhahi | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 2 |
| Varansi Cholapur Chaubepur | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 4 | 1 |
| Varansi Harahua Gurwat | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 1 | 1 |
| Varansi Harahua Kakalpur | plausible hydrology | pumping-like real | False | 0 | 0 | 0 | 0.0000 | 0.0000 | 2 | 1 |

Disagreement means Phase 2 called the station a general datum/telemetry-fault candidate but this stricter primary detector label is not one of placeholder-start, stuck, flip-flop, or isolated spike-faults (or vice versa). Individual detectors can overlap; the separate flags remain visible.

### Station-level detector counts

| strict_classification | stations | placeholder_slots | spike_slots | stuck_slots | flip_share_mean |
| --- | --- | --- | --- | --- | --- |
| clean | 412 | 0 | 0 | 0 | 0.0000 |
| flip-flop | 5 | 0 | 185 | 0 | 1.0000 |
| placeholder-start | 4 | 809 | 22 | 0 | 0.0000 |
| pumping-like real | 21 | 0 | 0 | 0 | 0.0000 |
| spike-faults on clean series | 158 | 0 | 2744 | 886 | 0.0000 |

### Whole-station flip-flop exclusion proposal

Proposed train-only rule: exclude a station from one pooled fit if its pre-cutoff observed-slot invalid share (slots covered by a 30-day window with more than 434 large sign reversals) exceeds the pre-cutoff station-share p99 = 0.000000. Large-step thresholds are station-specific pre-cutoff p95 6-hour absolute changes; reversal-count threshold is the pre-cutoff pooled p99.5. `station_detector_review.csv` reports Hathras and all stations. The rule is fitted without test errors; if Hathras is below the historical cutoff, report that it is detected in test but not eligible for pre-only whole-station exclusion.

### Test-window stuck runs and Phase 1 XGBoost error contribution

Test window is 2025-10-01 through 2026-09-30; a stuck run is an exact flat-value run longer than the pre-fitted 102-slot cutoff. Error contribution is XGBoost squared error for 30-day targets in the first 30 days after run entry and for targets inside the stuck interval.

| Station | run_start | run_end | stuck_value | transition_rows | transition_sq_error | transition_share_station_sq_error | targets_inside_stuck_rows | inside_stuck_sq_error | inside_stuck_share_station_sq_error | all_station_xgb_sq_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gloragundera/Gewara | 2026-01-01 00:00:00 | 2026-01-31 18:00:00 | 1.0000 | 0 | 0.0000 |  | 0 | 0.0000 |  | 0.0000 |
| Primary School, Sindhan Kalan | 2026-01-01 06:00:00 | 2026-01-31 18:00:00 | 0.0000 | 0 | 0.0000 |  | 0 | 0.0000 |  | 0.0000 |
| Primary School, Sindhan Kalan | 2026-06-24 06:00:00 | 2026-10-10 18:00:00 | 0.0000 | 0 | 0.0000 |  | 0 | 0.0000 |  | 0.0000 |
| Shitalpur_1 | 2026-01-01 00:00:00 | 2026-01-31 18:00:00 | 1.0000 | 0 | 0.0000 |  | 0 | 0.0000 |  | 0.0000 |
| Shitalpur_1 | 2026-06-24 12:00:00 | 2026-07-24 06:00:00 | 1.0000 | 0 | 0.0000 |  | 0 | 0.0000 |  | 0.0000 |

### Dera Devra bujurg exact 27.5 audit

| Station | value | occurrences_all_slots | occurrences_observed | first_occurrence | last_occurrence | contiguous_runs | longest_run_slots | run_lengths_slots | max_run_months | sentinel_code | pre_tail_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Primary School, Dera Devra bujurg | 27.5000 | 226 | 226 | 2026-01-01 00:00:00 | 2026-03-25 00:00:00 | 14 | 102 | 102;60;20;2;2;1;24;1;2;4;2;2;2;2 | 2 | False | True |

Interpretation: exact 27.5 occurrence/run/month counts are above. The h18 artifact stores absolute levels: anchor 27.500 and target -7.921 both match observed raw slots, so the anomaly is a real recorded level transition in this aligned table. If 27.5 also forms repeated runs, that repetition is reported; it does not by itself prove 27.5 is a placeholder. This evidence supersedes any interpretation that treated `obs` as a change value.

### Pumping-style recovery guard

The detector records large excursions only when they return smoothly to their prior 7-day level within 20-30 days; exact repeated-value immediate reversions are handled by the separate narrow spike mask. The three guard plots show raw values on the left and after-candidate values on the right. `spike_masked_slots=0` means the signal was left untouched.

| requested_station | matched_station | found | spike_masked_slots | pumping_like_events | spike_rule | Station | strict_classification | placeholder_start_slots | test_stuck_slots | flipflop_invalid_share_all_observed | isolated_repeated_extreme_spikes_to_mask |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mathura Chaumuha Navagaon-11 | Mathura Chaumuha Navagaon-11 | True | 0 | 3 | repeated pre-tail/sentinel exact value + observed immediate exact reversion | Mathura Chaumuha Navagaon-11 | pumping-like real | 0 | 0 | 0.0000 | 0 |
| Block Office, Kurara | Block Office, Kurara | True | 0 | 0 | repeated pre-tail/sentinel exact value + observed immediate exact reversion | Block Office, Kurara | clean | 0 | 0 | 0.0000 | 0 |
| Hamirpur Sarila Purani | Hamirpur Sarila Purani | True | 0 | 4 | repeated pre-tail/sentinel exact value + observed immediate exact reversion | Hamirpur Sarila Purani | pumping-like real | 0 | 0 | 0.0000 | 0 |

Smooth-recovery candidate events (all stations):

| Station | event_time | baseline_7d_m | excursion_value_m | excursion_m | recovery_value_20_30d_m | recovery_days | recovery_steps_toward_baseline | precutoff_jump_threshold_m | pre_cutoff_recovery_tolerance_m | exact_value_repetition_count | episode_start | episode_end | candidate_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Banda Tindwari Bhindowra | 2023-07-31 18:00:00 | -16.5505 | -0.6240 | 15.9265 | -16.2010 | 30 | 73 | 11.4542 | 9.2935 | 1 | 2023-07-31 18:00:00 | 2023-08-06 12:00:00 | 3 |
| Banda Tindwari Bhindowra | 2023-09-07 00:00:00 | -16.2040 | -0.5790 | 15.6250 | -13.8330 | 30 | 63 | 11.4542 | 9.2935 | 1 | 2023-09-07 00:00:00 | 2023-09-09 12:00:00 | 2 |
| Banda Tindwari Bhindowra | 2024-06-29 06:00:00 | -18.4590 | -1.2370 | 17.2220 | -18.6520 | 30 | 71 | 11.4542 | 9.2935 | 1 | 2024-06-29 06:00:00 | 2024-06-29 06:00:00 | 1 |
| Banda Tindwari Bhindowra | 2024-08-02 06:00:00 | -18.6530 | -2.1600 | 16.4930 | -16.3990 | 30 | 70 | 11.4542 | 9.2935 | 1 | 2024-08-02 06:00:00 | 2024-08-22 12:00:00 | 3 |
| Banda Tindwari Bhindowra | 2026-02-23 00:00:00 | -1.0670 | -15.4770 | 14.4100 | -1.0700 | 30 | 60 | 11.4542 | 9.2935 | 2 | 2026-02-03 06:00:00 | 2026-03-25 00:00:00 | 32 |
| Chitrakoot Manikpur Agarhudha | 2024-03-17 18:00:00 | -12.7720 | -59.0830 | 46.3110 | -12.9480 | 30 | 58 | 46.0660 | 46.0402 | 1 | 2024-03-17 18:00:00 | 2024-04-23 12:00:00 | 9 |
| Chitrakoot Manikpur Agarhudha | 2025-02-10 18:00:00 | -12.6810 | -59.0360 | 46.3550 | -13.7750 | 30 | 49 | 46.0660 | 46.0402 | 2 | 2025-02-10 18:00:00 | 2025-02-10 18:00:00 | 1 |
| Composite School Basantpur Saitali-1 | 2024-02-18 06:00:00 | 99.0000 | 25.8000 | 73.2000 | 25.9000 | 30 | 3 | 29.7130 | 73.1000 | 3 | 2024-02-18 06:00:00 | 2024-02-18 06:00:00 | 1 |
| Hamirpur Kurara Jalokhar | 2023-07-04 12:00:00 | -8.7000 | -0.3220 | 8.3780 | -5.8700 | 30 | 49 | 8.0416 | 8.3490 | 1 | 2023-07-04 12:00:00 | 2023-07-04 12:00:00 | 1 |
| Hamirpur Kurara Jalokhar | 2024-06-29 06:00:00 | -10.2280 | -0.0880 | 10.1400 | -8.1380 | 30 | 66 | 8.0416 | 8.3490 | 1 | 2024-06-29 06:00:00 | 2024-07-28 00:00:00 | 9 |
| Hamirpur Kurara Jalokhar | 2025-06-22 00:00:00 | -8.6505 | -0.0660 | 8.5845 | -6.2040 | 30 | 11 | 8.0416 | 8.3490 | 13 | 2025-06-22 00:00:00 | 2025-06-22 00:00:00 | 1 |
| Hamirpur Sarila Purani | 2023-03-23 00:00:00 | -27.2330 | -25.1120 | 2.1210 | -26.7580 | 30 | 72 | 1.6339 | 1.8582 | 3 | 2023-03-22 12:00:00 | 2023-03-24 18:00:00 | 4 |
| Hamirpur Sarila Purani | 2023-08-05 06:00:00 | -25.0975 | -22.5440 | 2.5535 | -24.0560 | 30 | 64 | 1.6339 | 1.8582 | 1 | 2023-07-07 18:00:00 | 2023-08-05 12:00:00 | 8 |
| Hamirpur Sarila Purani | 2023-09-08 18:00:00 | -24.0220 | -22.0880 | 1.9340 | -23.6230 | 30 | 78 | 1.6339 | 1.8582 | 1 | 2023-09-08 18:00:00 | 2023-09-08 18:00:00 | 1 |
| Hamirpur Sarila Purani | 2024-08-15 00:00:00 | -25.5420 | -23.6220 | 1.9200 | -24.4460 | 30 | 71 | 1.6339 | 1.8582 | 2 | 2024-08-15 00:00:00 | 2024-08-15 00:00:00 | 1 |
| Hathras Hathras Hathras | 2026-07-09 12:00:00 | -20.3455 | -0.2200 | 20.1255 | -20.1080 | 30 | 53 | 18.5939 | 16.4467 | 1 | 2026-07-09 12:00:00 | 2026-07-09 12:00:00 | 1 |
| Mathura Chaumuha Navagaon-11 | 2023-03-20 18:00:00 | -14.5780 | -0.9910 | 13.5870 | -14.3010 | 30 | 61 | 9.7982 | 11.8885 | 1 | 2023-03-19 00:00:00 | 2023-03-21 12:00:00 | 9 |
| Mathura Chaumuha Navagaon-11 | 2023-06-01 00:00:00 | -14.2870 | -1.9940 | 12.2930 | -12.6710 | 30 | 77 | 9.7982 | 11.8885 | 1 | 2023-06-01 00:00:00 | 2023-06-01 06:00:00 | 2 |
| Mathura Chaumuha Navagaon-11 | 2023-09-10 18:00:00 | -13.7970 | -1.3460 | 12.4510 | -14.0660 | 30 | 96 | 9.7982 | 11.8885 | 1 | 2023-07-10 18:00:00 | 2023-09-10 18:00:00 | 7 |
| Primary School, Bilata | 2023-08-07 00:00:00 | -7.3270 | -2.8520 | 4.4750 | -6.8580 | 30 | 85 | 1.5454 | 4.3916 | 1 | 2023-08-07 00:00:00 | 2023-08-07 00:00:00 | 1 |
| Primary School, Dera Devra bujurg | 2023-08-06 00:00:00 | -7.8020 | -5.8430 | 1.9590 | -7.0360 | 30 | 40 | 0.8340 | 1.8206 | 1 | 2023-08-06 00:00:00 | 2023-08-06 00:00:00 | 1 |
| Primary School, Dera Devra bujurg | 2024-09-21 00:00:00 | -7.0310 | -5.1060 | 1.9250 | -5.5820 | 30 | 67 | 0.8340 | 1.8206 | 1 | 2024-09-20 00:00:00 | 2024-09-21 00:00:00 | 2 |
| Primary School, Dera Devra bujurg | 2026-03-12 18:00:00 | -7.9025 | 27.5000 | 35.4025 | -7.9140 | 30 | 48 | 0.8340 | 1.8206 | 226 | 2026-02-10 18:00:00 | 2026-03-25 00:00:00 | 28 |
| Primary School, Garher | 2024-09-13 00:00:00 | -3.5865 | -1.7660 | 1.8205 | -1.8790 | 30 | 67 | 1.3382 | 1.8202 | 1 | 2024-09-13 00:00:00 | 2024-09-13 00:00:00 | 1 |
| Primary School, Kalan Bar Devran | 2023-06-06 00:00:00 | -16.7120 | -307.7630 | 291.0510 | -19.3800 | 30 | 95 | 198.5627 | 162.2978 | 1 | 2023-06-06 00:00:00 | 2023-06-06 00:00:00 | 1 |
| Primary School, Kalan Bar Devran | 2023-10-10 12:00:00 | -4.9830 | -210.5490 | 205.5660 | -19.5150 | 30 | 29 | 198.5627 | 162.2978 | 1 | 2023-10-10 12:00:00 | 2023-10-10 12:00:00 | 1 |
| Primary School, Kalan Bar Devran | 2024-06-16 12:00:00 | -13.6310 | -349.7220 | 336.0910 | -19.3770 | 30 | 66 | 198.5627 | 162.2978 | 1 | 2024-06-16 12:00:00 | 2024-06-16 18:00:00 | 2 |
| Primary School, Kalan Bar Devran | 2024-09-12 00:00:00 | -4.2030 | -220.4010 | 216.1980 | -3.7550 | 30 | 44 | 198.5627 | 162.2978 | 1 | 2024-09-12 00:00:00 | 2024-09-12 00:00:00 | 1 |
| Primary School, Kalan Bar Devran | 2025-01-20 06:00:00 | -13.8805 | -355.5110 | 341.6305 | -19.3900 | 30 | 65 | 198.5627 | 162.2978 | 1 | 2025-01-20 06:00:00 | 2025-01-20 06:00:00 | 1 |
| Primary School, Kalan Bar Devran | 2026-01-18 06:00:00 | -11.0220 | -268.0990 | 257.0770 | -20.0890 | 30 | 62 | 198.5627 | 162.2978 | 1 | 2026-01-18 06:00:00 | 2026-01-18 06:00:00 | 1 |
| Primary School, Kalan Bar Devran | 2026-04-27 12:00:00 | -19.3190 | -274.6220 | 255.3030 | -15.3340 | 30 | 72 | 198.5627 | 162.2978 | 1 | 2026-04-27 12:00:00 | 2026-04-27 12:00:00 | 1 |
| Primary School, Kalan Bar Devran | 2026-06-01 06:00:00 | -15.5270 | -308.3340 | 292.8070 | -16.6860 | 30 | 21 | 198.5627 | 162.2978 | 1 | 2026-06-01 06:00:00 | 2026-06-01 06:00:00 | 1 |
| Primary School, Silana | 2023-10-02 00:00:00 | -24.3890 | -59.6780 | 35.2890 | -24.2350 | 30 | 49 | 34.9244 | 34.9018 | 1 | 2023-10-02 00:00:00 | 2023-10-15 18:00:00 | 3 |
| Primary School, Silana | 2024-08-26 12:00:00 | -24.6080 | -59.6360 | 35.0280 | -24.7930 | 30 | 74 | 34.9244 | 34.9018 | 1 | 2024-08-06 00:00:00 | 2024-08-26 12:00:00 | 3 |
| Primary School, Silana | 2024-11-05 18:00:00 | -24.8150 | -59.7480 | 34.9330 | -23.5900 | 30 | 60 | 34.9244 | 34.9018 | 2 | 2024-11-05 18:00:00 | 2024-11-05 18:00:00 | 1 |
| Primary School, Silana | 2025-03-14 12:00:00 | -22.6590 | -59.1400 | 36.4810 | -23.1350 | 30 | 74 | 34.9244 | 34.9018 | 1 | 2025-03-14 12:00:00 | 2025-03-14 12:00:00 | 1 |
| Primary School, Silana | 2026-01-04 18:00:00 | -23.0410 | -59.2430 | 36.2020 | -20.5560 | 30 | 71 | 34.9244 | 34.9018 | 2 | 2026-01-04 18:00:00 | 2026-01-08 18:00:00 | 2 |
| Primary School/ Composite, Landhoura Junardar | 2023-06-05 18:00:00 | -13.7375 | -16.3930 | 2.6555 | -13.1710 | 30 | 78 | 1.7383 | 2.5018 | 1 | 2023-06-05 18:00:00 | 2023-06-08 12:00:00 | 2 |
| Primary School/ Composite, Landhoura Junardar | 2024-02-25 18:00:00 | -8.4995 | -11.1100 | 2.6105 | -10.6070 | 30 | 69 | 1.7383 | 2.5018 | 2 | 2024-02-24 12:00:00 | 2024-02-25 18:00:00 | 2 |
| Primary School/ Composite, Landhoura Junardar | 2026-05-20 12:00:00 | -12.3270 | -15.8280 | 3.5010 | -14.6720 | 30 | 84 | 1.7383 | 2.5018 | 1 | 2026-05-18 18:00:00 | 2026-06-25 12:00:00 | 14 |
| Primary School/ Composite, Mehmoodpur | 2023-04-12 18:00:00 | -12.2735 | -14.7670 | 2.4935 | -13.5340 | 30 | 67 | 2.2445 | 2.4811 | 1 | 2023-04-12 18:00:00 | 2023-04-13 18:00:00 | 2 |
| Primary School/ Composite, Mehmoodpur | 2023-05-31 18:00:00 | -14.1865 | -16.9440 | 2.7575 | -13.0910 | 30 | 64 | 2.2445 | 2.4811 | 1 | 2023-05-29 18:00:00 | 2023-05-31 18:00:00 | 3 |
| Primary School/ Composite, Mehmoodpur | 2023-10-02 06:00:00 | -10.0425 | -148.9060 | 138.8635 | -10.6050 | 30 | 68 | 2.2445 | 2.4811 | 1 | 2023-09-03 18:00:00 | 2023-10-02 18:00:00 | 7 |
| Primary School/ Composite, Mehmoodpur | 2023-11-19 18:00:00 | -10.5330 | -149.0730 | 138.5400 | -11.3380 | 30 | 63 | 2.2445 | 2.4811 | 1 | 2023-11-19 18:00:00 | 2023-11-19 18:00:00 | 1 |
| Primary School/ Composite, Mehmoodpur | 2026-04-11 06:00:00 | -10.3110 | -13.8280 | 3.5170 | -11.3700 | 30 | 58 | 2.2445 | 2.4811 | 2 | 2026-04-11 06:00:00 | 2026-04-13 00:00:00 | 3 |
| Primary School/ Composite, Mehmoodpur | 2026-05-24 12:00:00 | -14.0980 | -16.7970 | 2.6990 | -15.3790 | 30 | 18 | 2.2445 | 2.4811 | 1 | 2026-05-24 12:00:00 | 2026-05-24 12:00:00 | 1 |
| Primary School/ Composite, Mehmoodpur | 2026-07-26 06:00:00 | -10.7655 | -148.8880 | 138.1225 | -8.3210 | 30 | 61 | 2.2445 | 2.4811 | 1 | 2026-07-24 12:00:00 | 2026-07-26 18:00:00 | 4 |
| Primary School/ Composite, Sidki | 2023-12-11 06:00:00 | -10.4165 | -36.2510 | 25.8345 | -10.3370 | 30 | 56 | 25.8062 | 25.5750 | 1 | 2023-12-11 06:00:00 | 2023-12-11 06:00:00 | 1 |
| Primary School/ Composite, Sidki | 2024-03-17 00:00:00 | -10.3150 | -36.2360 | 25.9210 | -10.7250 | 30 | 67 | 25.8062 | 25.5750 | 4 | 2024-03-17 00:00:00 | 2024-03-17 00:00:00 | 1 |
| Primary School/ Composite, Sidki | 2026-09-02 12:00:00 | -10.0100 | -36.1910 | 26.1810 | -10.2780 | 30 | 49 | 25.8062 | 25.5750 | 1 | 2026-09-02 12:00:00 | 2026-09-02 12:00:00 | 1 |
| Rohari | 2022-12-02 12:00:00 | 27.4000 | -5.3200 | 32.7200 | 27.9000 | 30 | 86 | 3.7000 | 3.2410 | 1 | 2022-12-02 12:00:00 | 2022-12-02 12:00:00 | 1 |
| Rohari | 2023-06-08 00:00:00 | 29.0500 | 25.2000 | 3.8500 | 26.6000 | 30 | 83 | 3.7000 | 3.2410 | 44 | 2023-06-08 00:00:00 | 2023-06-08 00:00:00 | 1 |
| Rohari | 2024-06-28 12:00:00 | 27.8000 | 23.5450 | 4.2550 | 25.6000 | 30 | 82 | 3.7000 | 3.2410 | 1 | 2024-06-28 12:00:00 | 2024-06-28 12:00:00 | 1 |
| Rohari | 2024-09-13 06:00:00 | 27.2000 | 96.6100 | 69.4100 | 26.5400 | 30 | 33 | 3.7000 | 3.2410 | 13 | 2024-08-31 12:00:00 | 2024-09-13 06:00:00 | 2 |
| Rohari | 2024-10-14 06:00:00 | 26.5300 | -1.0000 | 27.5300 | 26.5200 | 30 | 49 | 3.7000 | 3.2410 | 4 | 2024-10-14 06:00:00 | 2024-10-14 06:00:00 | 1 |
| Rohari | 2025-03-08 00:00:00 | 26.7100 | -1.0000 | 27.7100 | 26.6800 | 30 | 42 | 3.7000 | 3.2410 | 4 | 2025-02-07 18:00:00 | 2025-03-20 12:00:00 | 3 |
| Saint Ravidas Nagar Abholi Gangarampur | 2023-10-13 18:00:00 | -5.8865 | -6.7940 | 0.9075 | -6.2980 | 30 | 49 | 0.6607 | 0.9050 | 1 | 2023-10-13 18:00:00 | 2023-10-13 18:00:00 | 1 |
| Saint Ravidas Nagar Abholi Gangarampur | 2024-09-15 12:00:00 | -11.0610 | -12.2850 | 1.2240 | -10.8340 | 30 | 73 | 0.6607 | 0.9050 | 1 | 2024-09-15 12:00:00 | 2024-10-01 18:00:00 | 3 |
| Saint Ravidas Nagar Abholi Gangarampur | 2026-07-14 12:00:00 | -12.1390 | -13.8760 | 1.7370 | -12.6540 | 30 | 59 | 0.6607 | 0.9050 | 1 | 2026-07-12 12:00:00 | 2026-07-16 12:00:00 | 16 |
| Varansi Chirai Gaon Chhahi | 2024-06-26 06:00:00 | -22.2540 | -20.5760 | 1.6780 | -23.4610 | 30 | 23 | 1.3188 | 1.5892 | 1 | 2024-06-26 06:00:00 | 2024-06-26 06:00:00 | 1 |
| Varansi Chirai Gaon Chhahi | 2026-02-09 06:00:00 | -20.5395 | -15.6490 | 4.8905 | -19.6390 | 30 | 64 | 1.3188 | 1.5892 | 1 | 2026-02-06 18:00:00 | 2026-02-10 18:00:00 | 4 |
| Varansi Cholapur Chaubepur | 2026-09-02 12:00:00 | -2.8180 | -1.3380 | 1.4800 | -1.4590 | 30 | 43 | 0.8120 | 1.4667 | 1 | 2026-09-02 12:00:00 | 2026-09-02 12:00:00 | 1 |
| Varansi Harahua Gurwat | 2024-09-17 12:00:00 | -12.0760 | -0.5710 | 11.5050 | -11.5140 | 30 | 87 | 5.7954 | 7.5343 | 1 | 2024-08-28 06:00:00 | 2024-09-29 12:00:00 | 9 |
| Varansi Harahua Kakalpur | 2024-03-12 12:00:00 | -21.2595 | -23.2730 | 2.0135 | -22.7970 | 30 | 59 | 0.8210 | 1.9999 | 4 | 2024-03-12 00:00:00 | 2024-03-12 12:00:00 | 2 |

### Common observed-data gap and feature-row impact

| stations_affected | gap_station_intervals | first_last_observation | latest_last_observation | earliest_first_return | last_first_return | median_gap_days | common_gap_start | common_gap_end | common_interval_is_intersection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 9 | 2024-07-26 18:00:00 | 2024-12-03 18:00:00 | 2025-02-13 12:00:00 | 2025-05-06 18:00:00 | 154.0000 | 2024-12-03 18:00:00 | 2025-02-13 12:00:00 | True |

Feature row counts use saved 30-day `time` and `target_time` columns. `target_intervals_overlapping_gap` counts forecast intervals intersecting the station-agnostic common gap; `targets_in_gap` counts endpoints in that interval.

| split | scope | rows | anchors_in_gap | targets_in_gap | target_intervals_overlapping_gap | targets_crossing_into_gap |
| --- | --- | --- | --- | --- | --- | --- |
| train | all stations | 1627745 | 151024 | 151894 | 214400 | 63376 |
| train | affected gap stations | 10222 | 0 | 25 | 25 | 25 |
| val | all stations | 1777 | 0 | 0 | 0 | 0 |
| val | affected gap stations | 0 | 0 | 0 | 0 | 0 |

Affected station-specific exact observation gaps:

| Station | last_observed_before_gap | first_observed_after_gap | gap_days |
| --- | --- | --- | --- |
| Primary School, Sindhan Kalan | 2024-09-10 18:00:00 | 2025-04-24 06:00:00 | 225.5000 |
| Uccha Prathmik Vidhyalaya, Sonpur | 2024-08-29 12:00:00 | 2025-03-22 12:00:00 | 205.0000 |
| Primary School Radnool Azizpur | 2024-07-26 18:00:00 | 2025-02-13 12:00:00 | 201.7500 |
| Rasin_2 | 2024-09-10 18:00:00 | 2025-02-17 12:00:00 | 159.7500 |
| Swarg Dham, Hardua | 2024-12-03 18:00:00 | 2025-05-06 18:00:00 | 154.0000 |
| Primary School kakrehwa | 2024-11-07 00:00:00 | 2025-03-25 12:00:00 | 138.5000 |
| Primary School Didhai | 2024-11-11 18:00:00 | 2025-03-26 06:00:00 | 134.5000 |
| Primary School Baijnathpur | 2024-11-21 06:00:00 | 2025-03-25 06:00:00 | 124.0000 |
| Priamry School Mukasim | 2024-12-03 12:00:00 | 2025-03-22 12:00:00 | 109.0000 |

### Before/after PNG review set

Generated 39 union station plots, 10 deterministic random non-union stations (seed 20261026), plus named guard stations if not already included. Left panel is the source series; right panel applies only placeholder-start and exact repeated-spike masks. Flat/stuck and flip-flop slots stay visible and are highlighted; red vertical markers indicate change-point segment boundaries.

| Station | is_union | is_random_other | is_named_guard | plot | masked_placeholder_slots | masked_repeated_spike_slots | flat_slots_flagged | flipflop_slots_flagged | change_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aagan Wadi Kendra Chandni | True | False | False | before_after/Aagan_Wadi_Kendra_Chandni_c624043c_before_after.png | 0 | 0 | 174 | 0 | 1 |
| Aligarh Khair Utwara | False | True | False | before_after/Aligarh_Khair_Utwara_98cfebfa_before_after.png | 0 | 0 | 16 | 0 | 1 |
| Banda Tindwari Bhindowra | True | False | False | before_after/Banda_Tindwari_Bhindowra_5ab415d2_before_after.png | 0 | 0 | 0 | 0 | 3 |
| Block Development Office-Khekra-Shallow | False | True | False | before_after/Block_Development_Office_Khekra_Shallow_04d76e87_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Block Office, Kurara | True | False | True | before_after/Block_Office_Kurara_6a43ebdf_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Chitrakoot Manikpur Agarhudha | True | False | False | before_after/Chitrakoot_Manikpur_Agarhudha_b3a08ed7_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Community Center, Ghorawal | True | False | False | before_after/Community_Center_Ghorawal_9da5e2e2_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Composite School Basantpur Saitali-1 | True | False | False | before_after/Composite_School_Basantpur_Saitali_1_48733dd2_before_after.png | 0 | 0 | 106 | 0 | 1 |
| Composite School Sagarpur | True | False | False | before_after/Composite_School_Sagarpur_6029996a_before_after.png | 0 | 0 | 25 | 0 | 1 |
| Composite School, Ghamhapur | True | False | False | before_after/Composite_School_Ghamhapur_78404e96_before_after.png | 0 | 0 | 125 | 0 | 1 |
| Gadhela Panchayat Bhawan | True | False | False | before_after/Gadhela_Panchayat_Bhawan_5fb00cbd_before_after.png | 0 | 0 | 196 | 0 | 1 |
| Goshala Bada Gaon | True | False | False | before_after/Goshala_Bada_Gaon_b159b3e9_before_after.png | 0 | 0 | 205 | 0 | 1 |
| Goshala Bhadehdu | True | False | False | before_after/Goshala_Bhadehdu_f3c58ede_before_after.png | 0 | 0 | 209 | 0 | 1 |
| Hamirpur Kurara Jalokhar | True | False | False | before_after/Hamirpur_Kurara_Jalokhar_8a80fdac_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Hamirpur Sarila Purani | True | False | True | before_after/Hamirpur_Sarila_Purani_77769dfd_before_after.png | 0 | 0 | 0 | 0 | 3 |
| Hathras Hathras Hathras | True | False | False | before_after/Hathras_Hathras_Hathras_2a8a1743_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Mahoba Jaitpur saguniyamaph | False | True | False | before_after/Mahoba_Jaitpur_saguniyamaph_197ce6e3_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Mainpuri Karhal Naglamdari | False | True | False | before_after/Mainpuri_Karhal_Naglamdari_4ba6b877_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Mathura Chaumuha Navagaon-11 | True | False | True | before_after/Mathura_Chaumuha_Navagaon_11_d222442f_before_after.png | 0 | 0 | 287 | 0 | 1 |
| Mathura Nandgaon Vathainkala | False | True | False | before_after/Mathura_Nandgaon_Vathainkala_05f17913_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Prathmik Vidhyalaya, Didhora  Sadullapur | False | True | False | before_after/Prathmik_Vidhyalaya_Didhora_Sadullapur_db537221_before_after.png | 0 | 0 | 34 | 0 | 1 |
| Primary School Salawa | False | True | False | before_after/Primary_School_Salawa_f6dfc454_before_after.png | 0 | 0 | 146 | 0 | 1 |
| Primary School Sanda-Sanee | True | False | False | before_after/Primary_School_Sanda_Sanee_de79c34f_before_after.png | 0 | 2 | 210 | 0 | 1 |
| Primary School Vamanpura | True | False | False | before_after/Primary_School_Vamanpura_a5e98d47_before_after.png | 0 | 1 | 181 | 0 | 1 |
| Primary School chandsina | True | False | False | before_after/Primary_School_chandsina_19b8f8ca_before_after.png | 0 | 2 | 222 | 0 | 3 |
| Primary School, Belaukh | False | True | False | before_after/Primary_School_Belaukh_87662f8c_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Primary School, Bilata | True | False | False | before_after/Primary_School_Bilata_6466a5a9_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Primary School, Dera Devra bujurg | True | False | False | before_after/Primary_School_Dera_Devra_bujurg_e457cc5a_before_after.png | 0 | 0 | 23 | 0 | 4 |
| Primary School, Evani | True | False | False | before_after/Primary_School_Evani_5c231b53_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Primary School, Garher | True | False | False | before_after/Primary_School_Garher_6bb34643_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Primary School, Kalan Bar Devran | True | False | False | before_after/Primary_School_Kalan_Bar_Devran_e00cb88d_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Primary School, Karasdev Mandir | True | False | False | before_after/Primary_School_Karasdev_Mandir_aa4df36d_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Primary School, Silana | True | False | False | before_after/Primary_School_Silana_659344af_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Primary School/ Composite, Jeevna | False | True | False | before_after/Primary_School_Composite_Jeevna_5b3d0d2a_before_after.png | 0 | 1 | 0 | 0 | 2 |
| Primary School/ Composite, Landhoura Junardar | True | False | False | before_after/Primary_School_Composite_Landhoura_Junardar_df2280a9_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Primary School/ Composite, Mehmoodpur | True | False | False | before_after/Primary_School_Composite_Mehmoodpur_25f733ed_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Primary School/ Composite, Sidki | True | False | False | before_after/Primary_School_Composite_Sidki_10430c49_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Prmary School Rajpura | True | False | False | before_after/Prmary_School_Rajpura_45acd6c7_before_after.png | 0 | 5 | 127 | 0 | 1 |
| Rajkia High School Pakhrauli | True | False | False | before_after/Rajkia_High_School_Pakhrauli_9300882f_before_after.png | 0 | 0 | 202 | 0 | 1 |
| Rohari | True | False | False | before_after/Rohari_01564bf3_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Saint Ravidas Nagar Abholi Gangarampur | True | False | False | before_after/Saint_Ravidas_Nagar_Abholi_Gangarampur_3f251e95_before_after.png | 0 | 0 | 0 | 0 | 2 |
| Shahjahanpur Dadraul Banthara | False | True | False | before_after/Shahjahanpur_Dadraul_Banthara_4d76cdea_before_after.png | 0 | 1 | 214 | 0 | 1 |
| Swarg Dham, Hardua | True | False | False | before_after/Swarg_Dham_Hardua_295af2d9_before_after.png | 0 | 0 | 176 | 0 | 1 |
| Uccha Prathmik Vidhyalaya, Lohra (Pampawan) | True | False | False | before_after/Uccha_Prathmik_Vidhyalaya_Lohra_Pampawan_ea14389d_before_after.png | 0 | 0 | 11 | 0 | 1 |
| Upper Primary School Sarahan Bujurg | True | False | False | before_after/Upper_Primary_School_Sarahan_Bujurg_68ef875f_before_after.png | 0 | 2 | 165 | 0 | 1 |
| Varansi Chirai Gaon Chhahi | True | False | False | before_after/Varansi_Chirai_Gaon_Chhahi_23f8a40f_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Varansi Cholapur Chaubepur | True | False | False | before_after/Varansi_Cholapur_Chaubepur_82ecaf78_before_after.png | 0 | 0 | 0 | 0 | 4 |
| Varansi Harahua Gurwat | True | False | False | before_after/Varansi_Harahua_Gurwat_9c54f021_before_after.png | 0 | 0 | 0 | 0 | 1 |
| Varansi Harahua Kakalpur | True | False | False | before_after/Varansi_Harahua_Kakalpur_86728333_before_after.png | 0 | 0 | 0 | 0 | 2 |

### 27.5 monthly counts

| month | occurrences | observed_occurrences | first | last |
| --- | --- | --- | --- | --- |
| 2026-01 | 123 | 123 | 2026-01-01 | 2026-01-31 18:00:00 |
| 2026-02 | 73 | 73 | 2026-02-01 | 2026-02-28 18:00:00 |
| 2026-03 | 30 | 30 | 2026-03-01 | 2026-03-25 00:00:00 |

### Addendum outputs

- `cleaning_detector_thresholds.csv`; `station_detector_review.csv`; `change_point_segments.csv`; `union_strict_reclassification.csv`.
- `test_stuck_runs.csv`; `test_stuck_phase1_error.csv`; `dera_27_5_audit.csv`; `dera_27_5_month_counts.csv`; `union_repeated_extreme_values.csv`.
- `pumping_like_smooth_recovery_events.csv`; `pumping_guard_review.csv`.
- `common_data_gap_summary.csv`; `common_data_gap_by_station.csv`; `common_data_gap_daily_coverage.csv`; `common_data_gap_monthly_coverage.csv`; `common_data_gap_feature_rows.csv`.
- `before_after_plot_manifest.csv`; PNGs are under `before_after/`.

- [Aagan Wadi Kendra Chandni](before_after/Aagan_Wadi_Kendra_Chandni_c624043c_before_after.png)
- [Banda Tindwari Bhindowra](before_after/Banda_Tindwari_Bhindowra_5ab415d2_before_after.png)
- [Block Office, Kurara](before_after/Block_Office_Kurara_6a43ebdf_before_after.png)
- [Chitrakoot Manikpur Agarhudha](before_after/Chitrakoot_Manikpur_Agarhudha_b3a08ed7_before_after.png)
- [Community Center, Ghorawal](before_after/Community_Center_Ghorawal_9da5e2e2_before_after.png)
- [Composite School Basantpur Saitali-1](before_after/Composite_School_Basantpur_Saitali_1_48733dd2_before_after.png)
- [Composite School Sagarpur](before_after/Composite_School_Sagarpur_6029996a_before_after.png)
- [Composite School, Ghamhapur](before_after/Composite_School_Ghamhapur_78404e96_before_after.png)
- [Gadhela Panchayat Bhawan](before_after/Gadhela_Panchayat_Bhawan_5fb00cbd_before_after.png)
- [Goshala Bada Gaon](before_after/Goshala_Bada_Gaon_b159b3e9_before_after.png)
- [Goshala Bhadehdu](before_after/Goshala_Bhadehdu_f3c58ede_before_after.png)
- [Hamirpur Kurara Jalokhar](before_after/Hamirpur_Kurara_Jalokhar_8a80fdac_before_after.png)
- [Hamirpur Sarila Purani](before_after/Hamirpur_Sarila_Purani_77769dfd_before_after.png)
- [Hathras Hathras Hathras](before_after/Hathras_Hathras_Hathras_2a8a1743_before_after.png)
- [Mathura Chaumuha Navagaon-11](before_after/Mathura_Chaumuha_Navagaon_11_d222442f_before_after.png)
- [Primary School Sanda-Sanee](before_after/Primary_School_Sanda_Sanee_de79c34f_before_after.png)
- [Primary School Vamanpura](before_after/Primary_School_Vamanpura_a5e98d47_before_after.png)
- [Primary School chandsina](before_after/Primary_School_chandsina_19b8f8ca_before_after.png)
- [Primary School, Bilata](before_after/Primary_School_Bilata_6466a5a9_before_after.png)
- [Primary School, Dera Devra bujurg](before_after/Primary_School_Dera_Devra_bujurg_e457cc5a_before_after.png)
- [Primary School, Evani](before_after/Primary_School_Evani_5c231b53_before_after.png)
- [Primary School, Garher](before_after/Primary_School_Garher_6bb34643_before_after.png)
- [Primary School, Kalan Bar Devran](before_after/Primary_School_Kalan_Bar_Devran_e00cb88d_before_after.png)
- [Primary School, Karasdev Mandir](before_after/Primary_School_Karasdev_Mandir_aa4df36d_before_after.png)
- [Primary School, Silana](before_after/Primary_School_Silana_659344af_before_after.png)
- [Primary School/ Composite, Landhoura Junardar](before_after/Primary_School_Composite_Landhoura_Junardar_df2280a9_before_after.png)
- [Primary School/ Composite, Mehmoodpur](before_after/Primary_School_Composite_Mehmoodpur_25f733ed_before_after.png)
- [Primary School/ Composite, Sidki](before_after/Primary_School_Composite_Sidki_10430c49_before_after.png)
- [Prmary School Rajpura](before_after/Prmary_School_Rajpura_45acd6c7_before_after.png)
- [Rajkia High School Pakhrauli](before_after/Rajkia_High_School_Pakhrauli_9300882f_before_after.png)
- [Rohari](before_after/Rohari_01564bf3_before_after.png)
- [Saint Ravidas Nagar Abholi Gangarampur](before_after/Saint_Ravidas_Nagar_Abholi_Gangarampur_3f251e95_before_after.png)
- [Swarg Dham, Hardua](before_after/Swarg_Dham_Hardua_295af2d9_before_after.png)
- [Uccha Prathmik Vidhyalaya, Lohra (Pampawan)](before_after/Uccha_Prathmik_Vidhyalaya_Lohra_Pampawan_ea14389d_before_after.png)
- [Upper Primary School Sarahan Bujurg](before_after/Upper_Primary_School_Sarahan_Bujurg_68ef875f_before_after.png)
- [Varansi Chirai Gaon Chhahi](before_after/Varansi_Chirai_Gaon_Chhahi_23f8a40f_before_after.png)
- [Varansi Cholapur Chaubepur](before_after/Varansi_Cholapur_Chaubepur_82ecaf78_before_after.png)
- [Varansi Harahua Gurwat](before_after/Varansi_Harahua_Gurwat_9c54f021_before_after.png)
- [Varansi Harahua Kakalpur](before_after/Varansi_Harahua_Kakalpur_86728333_before_after.png)

### Detector outcomes that need human review

The pre-cutoff calibration is intentionally reported even where it fails to catch a visually identified pattern. The exact constant-run threshold is 102 six-hour slots (25.5 days). It finds five qualifying test runs across three stations, but none at Chandsina, Sanda-Sanee, Sagarpur, Bilata, Landhoura, or Karasdev. Their flatness indicators remain in `station_detector_review.csv`; the 102-slot rule is too conservative for these examples and should not be used as a complete stuck-segment detector. The five detected runs have zero matched Phase 1 XGBoost 30-day squared error in the first 30 days after entry and within the stuck intervals (`test_stuck_phase1_error.csv`); this is an artifact/sample alignment result, not evidence that the segments are harmless.

The trained flip-flop reversal cutoff is 434 reversals per 30-day window and the pre-period station invalid-share p99 is 0. Hathras has pre/test invalid shares of 0 under this rule and is therefore not selected by the proposed whole-station exclusion rule. The present reversal detector does not substantiate a Hathras flip-flop exclusion; its strict label is `pumping-like real` because the separate recovery-shape detector found smooth-return excursions. That label describes the detected shape only and does not establish a pumping cause. Do not apply whole-station exclusion to Hathras from this run. The rolling flags and source plots are review aids; this calibration needs refinement before an operational exclusion decision.

The monthly-median shift detector also has a high pooled fallback (72.186 m); it reports boundaries in `change_point_segments.csv` but does not guarantee detection of smaller persistent datum changes. All segments are marked kept. In particular, change-point boundaries are proposals to segment feature history, not deletions or a verified causal explanation.

For Dera Devra bujurg, 27.5 occurs 226 times in observed slots from January through March 2026, across 14 runs, with a longest run of 102 six-hour slots. It is beyond that station's pre-cutoff p99.5 and is repeated across three months. This is strong evidence of a repeated ceiling-like level/plateau, though the file alone cannot establish it is a placeholder. The h18 artifact's 27.500 anchor followed by an observed -7.921 target is consequently best described as a transition out of a repeated ceiling-like plateau. It contributes 73.9% of h18 persistence squared error; whether it is a sensor fault or a real site event remains unverified.

The before/after review leaves all three pumping-style guard series untouched by the exact-spike mask: Mathura Chaumuha Navagaon-11, Block Office Kurara, and Hamirpur Sarila Purani each have zero masked slots. The smooth-recovery detector records 3, 0, and 4 candidate episodes respectively; these are candidates, not verified pumping events.
