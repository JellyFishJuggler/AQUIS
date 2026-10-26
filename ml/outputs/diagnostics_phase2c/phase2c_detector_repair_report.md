# Phase 2c: detector repair and acceptance results

Read-only forensic pass. The input aligned table and model directories were not changed. No data rebuild was attempted. Hand-labelled acceptance sets A-E were preserved as supplied; disagreements are shown directly in the station table.

## Detector thresholds and rationale

| detector | threshold | justification |
| --- | --- | --- |
| leading_constant_run | 28 six-hour slots (7 days); sentinel first-level >=4 slots | Groundwater level telemetering should not preserve an identical 3-decimal value for a week; shorter sentinel threshold catches coded initial values. |
| two_step_placeholder_start | repeated sentinel >=4 slots in first 60 days; flag sentinel and subsequent level repeated >=4 times | Catches a placeholder plateau followed by a second flat initialization level; the whole leading initialization prefix is flagged. |
| stuck_test | >=28 occurrences of one exact value in any 30-day window, plus 7-day rolling range <=0.06m and SD <=0.02m | A week with <=6cm total movement and <=2cm standard deviation is a near-flat logger plateau; flagged only, never removed. |
| repeated_extreme_spike | same exact observed value >=3 times and >=8 robust MAD from station median (or sentinel), remove only isolated 6h point with immediate exact reversion | Exact recurrence plus immediate reversal distinguishes telemetry spikes from smooth pumping-style recovery. |
| extreme_burst_review | >=3 robust-range outliers in a 30-day span, even when their exact values differ | Flags bursty extreme events for review without broad magnitude-only removal; masking still requires the exact-repeat and immediate-reversion rule. |
| flipflop_30d | step magnitude >=0.5m; >=6 qualifying adjacent sign reversals; reversal share >=0.60; station invalid-share >=0.20 | Absolute hydrologic level can vary; repeated half-meter-or-larger alternating observed steps in a month and >=20% affected observed slots is an operational sensor-instability criterion. |
| pumping_like_recovery | each observed 6h step <=20m and total excursion <=30m; multi-step smooth return within 30d | A single >20m step or >30m excursion is not credible pumping drawdown/recovery; such events are fault/level-shift candidates. |
| robust_station_range | median +/- 8 * 1.4826 * MAD, fitted after excluding long exact runs and sentinel values | Robust scale is not inflated by flagged initialization or stuck segments; supports repeated-extreme screening only. |

A/B flags are not all removals: stuck and flip-flop segments are flagged for review and retained in the after-series. The after panel masks only detected placeholder initialization and exact repeated spikes with immediate exact reversion. Pumping-like candidates are not removed.

## Acceptance summary

| group | case_type | stations | flagged | passed | failed | precision | recall | negative_control_specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_placeholder | acceptance | 15 | 15 | 15.0 | 0.0 |  | 1.0 |  |
| B_stuck | acceptance | 3 | 3 | 3.0 | 0.0 |  | 1.0 |  |
| B_check | descriptive | 2 | 2 |  |  |  |  |  |
| C_repeated_spike | acceptance | 7 | 6 | 6.0 | 1.0 |  | 0.8571428571428571 |  |
| D_flipflop | acceptance | 1 | 1 | 1.0 | 0.0 |  | 1.0 |  |
| D_check | descriptive | 3 | 2 |  |  |  |  |  |
| E_untouched | acceptance | 7 | 0 | 7.0 | 0.0 |  |  | 1.0 |

A-E station-level pass/fail detail: `acceptance_per_station.csv`. A-D provide positive hand-labelled examples only, so precision cannot be estimated without a labelled negative sample; recall is the share of requested positives flagged. E provides negative controls, so `negative_control_specificity` is the share left unchanged. B_check and D_check are descriptive only: their `flagged` counts report detector output without imposing a hand label.

### Descriptive checks: Sagarpur, Rohari, and other oscillation stations

| group | requested_station | matched_station | detector_value | observed_flag |
| --- | --- | --- | --- | --- |
| B_check | Composite School Sagarpur | Composite School Sagarpur | 480 | True |
| B_check | Rohari | Rohari | 2 | True |
| D_check | Varansi Cholapur Chaubepur | Varansi Cholapur Chaubepur | True | True |
| D_check | Varansi Harahua Gurwat | Varansi Harahua Gurwat | False | False |
| D_check | Varansi Chirai Gaon Chhahi | Varansi Chirai Gaon Chhahi | True | True |


## Corrected 600-station classification

| classification | stations |
| --- | --- |
| stuck | 230 |
| placeholder-start | 156 |
| flip-flop | 133 |
| clean | 58 |
| spike-faults | 19 |
| pumping-like real | 4 |

Station-level details and detector counts are in `classification_600_stations.csv`; per-station flagged-row counts and intervals are in `detector_flagged_rows.csv`.

## Named guard and physical-cap review

| Station | primary_classification | changed_slots | isolated_spike_points | flipflop_invalid_share | pump_candidate_events |
| --- | --- | --- | --- | --- | --- |
| Block Office, Kurara | pumping-like real | 0 | 0 | 0.0 | 4 |
| Hamirpur Sarila Purani | stuck | 0 | 0 | 0.872444011684518 | 62 |
| Mathura Chaumuha Navagaon-11 | stuck | 0 | 0 | 0.0 | 81 |
| Primary School, Evani | pumping-like real | 0 | 0 | 0.0 | 12 |
| Primary School, Garher | stuck | 0 | 0 | 0.0 | 28 |
| Primary School, Kalan Bar Devran | stuck | 0 | 0 | 0.03967043027158987 | 26 |
| Primary School/ Composite, Mehmoodpur | flip-flop | 0 | 0 | 0.28934624697336564 | 416 |
| Rohari | stuck | 0 | 0 | 0.3821934945788157 | 170 |
| Saint Ravidas Nagar Abholi Gangarampur | pumping-like real | 0 | 0 | 0.0 | 50 |
| Varansi Harahua Kakalpur | pumping-like real | 0 | 0 | 0.0 | 22 |

Pumping-like label candidates are limited to smooth multi-step returns with each 6-hour change <=20m and total departure <=30m. Excursions beyond either cap are not called pumping. An abrupt change is still not a proven datum fault without independent evidence.

## Still failing

| group | requested_station | matched_station | detector_value |
| --- | --- | --- | --- |
| C_repeated_spike | Hamirpur Kurara Jalokhar | Hamirpur Kurara Jalokhar | False |

## Plot set

Generated 35 before/after plots for all matched unique stations in A-D and E. Files are listed in `before_after_manifest.csv`.

## Limits

The series-level heuristics are screening rules, not verified sensor diagnoses. Robust station limits are fit on pre-test observations after excluding known sentinels and long exact runs. No physical well datum, logger specification, or station maintenance log was available to verify fault causes. Missing requested station-name matches, if any, appear as failed acceptance rows.
