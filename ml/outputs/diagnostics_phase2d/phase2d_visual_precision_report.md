# Phase 2d: visual precision check

Read-only audit of the Phase 2c detector output. No data or models were changed. Phase 2c A-E acceptance stations are excluded from the random sample. The five requested named stations are then added as targeted diagnostics and are not independent evidence.

## Sample

Fixed seed: 202602. Stratified random sample: 6 each from stuck, flip-flop, placeholder-start, and clean after excluding every matched station in the Phase 2c A-E acceptance file. Added 5 named diagnostics. Total unique visual stations: 29. See `visual_sample_stations.csv`.
| Station | primary_classification | selection |
| --- | --- | --- |
| Banda Badokhar Khurd Chatgan | stuck | stratified random; seed=202602; excludes Phase 2c A-E |
| Banda Bisndha Mrooli | stuck | stratified random; seed=202602; excludes Phase 2c A-E |
| Murwa_1 | stuck | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School, Purakalan, Talbehat, Lalitpur_1 | stuck | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School/ Composite, Jaula Dindauli | stuck | stratified random; seed=202602; excludes Phase 2c A-E |
| Sonbhadra Babhani Bhawar | stuck | stratified random; seed=202602; excludes Phase 2c A-E |
| Chitrakoot Karvi Khoha | flip-flop | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School Jainpur | flip-flop | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School, Brahmpur | flip-flop | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School, Gauripur Javahar Nagar | flip-flop | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School/ Composite, Jhabiran | flip-flop | stratified random; seed=202602; excludes Phase 2c A-E |
| Purv Madhyamik Vidyalaya, Mujehra | flip-flop | stratified random; seed=202602; excludes Phase 2c A-E |
| Bhadarwara  P_1 | placeholder-start | stratified random; seed=202602; excludes Phase 2c A-E |
| Fugana_1 | placeholder-start | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School Shekhpuri | placeholder-start | stratified random; seed=202602; excludes Phase 2c A-E |
| Punchayat Bhawan Bhaluwani | placeholder-start | stratified random; seed=202602; excludes Phase 2c A-E |
| Ranipur_2 | placeholder-start | stratified random; seed=202602; excludes Phase 2c A-E |
| Upper Primary School Rijaula | placeholder-start | stratified random; seed=202602; excludes Phase 2c A-E |
| Anganwadi Kedra, Baini | clean | stratified random; seed=202602; excludes Phase 2c A-E |
| Chitrakoot Karvi Shivrampur | clean | stratified random; seed=202602; excludes Phase 2c A-E |
| Gorakhpur Pali Dumari Niwas | clean | stratified random; seed=202602; excludes Phase 2c A-E |
| Mahoba Kabrai Sisri Kala | clean | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School Sahjanwa , Pratham Pipra | clean | stratified random; seed=202602; excludes Phase 2c A-E |
| Primary School, Dagarpur Tigadi | clean | stratified random; seed=202602; excludes Phase 2c A-E |
| Hamirpur Kurara Jalokhar | stuck | named diagnostic; was used in Phase 2c acceptance |
| Hamirpur Sarila Purani | stuck | named diagnostic; was used in Phase 2c acceptance |
| Mathura Chaumuha Navagaon-11 | stuck | named diagnostic; was used in Phase 2c acceptance |
| Primary School, Kalan Bar Devran | stuck | named diagnostic; was used in Phase 2c acceptance |
| Primary School/ Composite, Mehmoodpur | flip-flop | named diagnostic; was used in Phase 2c acceptance |

Per-station PNGs are in `plots/`; contact sheets contain up to six stations each in a 3x2 layout. Raw observed series and overlay flags are exported to `visual_data/`. Legend colors distinguish placeholder, exact-spike mask, stuck, flip-flop, and extreme-burst points.

## Placeholder truncation check

The Phase 2c prefix threshold selects 29 stations (>=236 flagged slots). Of those, 26 have sentinel repetition or a >=28-slot exact plateau continuing after the first 60 days, within the next 60-day audit interval.
| Station | placeholder_prefix_slots | first_finite_time | search_cap_time | flagged_prefix_end | post_prefix_to_cap_rows | post_prefix_distinct_values | post_prefix_sentinel_slots | post_prefix_max_exact_run_slots | post_prefix_nearflat_slots | real_looking_data_after_prefix | post_cap_to_120d_rows | post_cap_sentinel_slots | max_post_cap_exact_run_slots | post_cap_nearflat_slots | post_cap_placeholder_or_flat_continues | post_cap_series_std | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B.D.O. Office, Binauli | 238 | 2024-01-18 18:00:00 | 2024-03-18 18:00:00 | 2024-03-18 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 8 | True | 9.885835334706448 | placeholder/flat continues beyond 60d |
| Composite Girls Primary School Mai | 238 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 213 | True | 3.407808436262797 | placeholder/flat continues beyond 60d |
| Fugana_1 | 241 | 2022-08-07 00:00:00 | 2022-10-06 00:00:00 | 2022-10-06 00:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 213 | True | 1.94694650249917 | placeholder/flat continues beyond 60d |
| Gadhela Panchayat Bhawan | 239 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 1 | True | 4.487507925588792 | placeholder/flat continues beyond 60d |
| Goshala Bankat | 239 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 1 | True | 4.125741321415176 | placeholder/flat continues beyond 60d |
| Goshala Basela | 236 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-19 18:00:00 | 3 | 3 | 0 | 1 | 0 | False | 240 | 0 | 1 | 0 | False | 0.49878477654793374 | variable data after cap |
| Goshala Bhadehdu | 239 | 2024-01-18 00:00:00 | 2024-03-18 00:00:00 | 2024-03-18 00:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 11 | True | 9.642821656432332 | placeholder/flat continues beyond 60d |
| Goshala Churaha | 239 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 213 | True | 5.513306687937867 | placeholder/flat continues beyond 60d |
| Goshala Ghuri | 239 | 2024-01-19 12:00:00 | 2024-03-19 12:00:00 | 2024-03-19 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 54 | True | 2.25803206252686 | placeholder/flat continues beyond 60d |
| Goshala Pachhahan | 239 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 8 | True | 3.0136931357803007 | placeholder/flat continues beyond 60d |
| Office of the Development Officer, Dakore | 240 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 1 | True | 2.0337751472016277 | placeholder/flat continues beyond 60d |
| Prathmik Vidhyalaya Barha | 239 | 2024-01-19 18:00:00 | 2024-03-19 18:00:00 | 2024-03-19 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 182 | True | 3.647463958774035 | placeholder/flat continues beyond 60d |
| Prathmik Vidhyalaya Manohari | 239 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 2 | True | 2.1462969208616034 | placeholder/flat continues beyond 60d |
| Prathmik Vidhyalaya Takuli (Durahi Maafi) | 237 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 12 | True | 2.1129676969080915 | placeholder/flat continues beyond 60d |
| Primary School Jatmujheda | 239 | 2024-01-19 12:00:00 | 2024-03-19 12:00:00 | 2024-03-19 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 5 | True | 5.303871119481126 | placeholder/flat continues beyond 60d |
| Primary School Kucharam | 236 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-19 18:00:00 | 3 | 3 | 0 | 1 | 0 | False | 240 | 0 | 1 | 0 | False | 2.3790147956453116 | variable data after cap |
| Primary School Kurdi | 239 | 2024-01-19 18:00:00 | 2024-03-19 18:00:00 | 2024-03-19 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 4 | True | 4.4586610425489575 | placeholder/flat continues beyond 60d |
| Primary School No. 1 Barawad | 239 | 2024-01-19 18:00:00 | 2024-03-19 18:00:00 | 2024-03-19 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 135 | True | 6.2799871244504875 | placeholder/flat continues beyond 60d |
| Primary School Pinna | 238 | 2024-01-18 18:00:00 | 2024-03-18 18:00:00 | 2024-03-18 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 8 | True | 7.047741869976224 | placeholder/flat continues beyond 60d |
| Primary School Rathoda | 241 | 2024-01-19 18:00:00 | 2024-03-19 18:00:00 | 2024-03-19 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 4 | True | 6.177416928147772 | placeholder/flat continues beyond 60d |
| Primary School Sanda-Sanee | 239 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 1 | True | 2.933879726526331 | placeholder/flat continues beyond 60d |
| Primary School chandsina | 238 | 2024-01-18 18:00:00 | 2024-03-18 18:00:00 | 2024-03-18 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 8 | True | 7.962754392478181 | placeholder/flat continues beyond 60d |
| Purva Madhyamik Vidhyalaya Bhitaura | 238 | 2024-01-19 18:00:00 | 2024-03-19 18:00:00 | 2024-03-19 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 16 | True | 4.279892162257778 | placeholder/flat continues beyond 60d |
| Rajkia Balika Inter College Kamasin | 239 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 1 | True | 4.039136380153886 | placeholder/flat continues beyond 60d |
| Rajkia High School Pakhrauli | 236 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-19 18:00:00 | 3 | 3 | 0 | 1 | 0 | False | 240 | 0 | 1 | 0 | False | 2.965035001898399 | variable data after cap |
| Uccha Prathmik Vidhyalaya, Semra Bhav Nagar | 239 | 2024-01-20 12:00:00 | 2024-03-20 12:00:00 | 2024-03-20 12:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 10 | True | 2.400948797639985 | placeholder/flat continues beyond 60d |
| Upper Primary Kasampur | 239 | 2024-01-19 18:00:00 | 2024-03-19 18:00:00 | 2024-03-19 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 4 | True | 4.337467625381501 | placeholder/flat continues beyond 60d |
| Upper Primary School Pohalli | 239 | 2024-01-18 18:00:00 | 2024-03-18 18:00:00 | 2024-03-18 18:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 8 | True | 6.228709236264811 | placeholder/flat continues beyond 60d |
| Veterinary Hospital Shahjahanpur | 239 | 2024-01-19 00:00:00 | 2024-03-19 00:00:00 | 2024-03-19 00:00:00 | 0 | 0 | 0 | 0 | 0 | False | 240 | 0 | 1 | 7 | True | 5.894256858514845 | placeholder/flat continues beyond 60d |

This is a direct boundary check: the existing 60-day start search can truncate initialization sequences. Extend the search only when a sentinel/flat run continues, or use run-continuation logic that follows the second plateau until varying in-range observations begin. Do not extend a fixed mask blindly.

## Sensitivity

| detector | setting | stations_flagged |
| --- | --- | --- |
| stuck | current thresholds | 342 |
| stuck | strict range<=0.01, sd<=0.005 | 85 |
| stuck | exact-value contiguous runs >=28 slots only | 0 |
| flip-flop | current: step>=0.5m, reversal share>=0.60 | 222 |
| flip-flop | stricter: step>=1.0m, reversal share>=0.80 | 69 |

`exact-value runs only` means a contiguous exact-valued run >=28 six-hour slots overlapping the test period. The stricter near-flat setting combines that exact rule with a 7-day rolling range <=0.01m and SD <=0.005m. Flip-flop alternatives retain >=6 reversals and >=20% station invalid-share; only step and reversal-share thresholds change.
These are independent detector flags and overlap: Phase 2c exclusive primary labels were 230 stuck and 133 flip-flop stations. Zero stations have a contiguous exact-value test run of >=28 six-hour slots; current stuck flags arise from exact-value occupancy within 30 days and/or near-flat rolling windows.

### Sarila Purani flip-flop share

| Station | finite_slots | observed_slots | large_steps_ge_0_5m | reversing_large_step_pairs | qualifying_window_endpoints | qualifying_observed_slots | current_invalid_share | strict_step1_share08_invalid_share | primary_classification | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hamirpur Sarila Purani | 4745 | 4108 | 154 | 32 | 4168 | 3584 | 0.872444011684518 | 0.0 | stuck | The 0.87 is a coverage share: qualifying endpoints among gwl_obs=1 rows divided by all observed rows. The 30-day windows overlap, so one local burst flags many neighboring endpoints; this is not a claim that 87% of the series is visibly jagged. |

The current share is the proportion of observed timestamps that end in qualifying 30-day reversal windows, not the fraction of calendar duration or the fraction of visual points with a jagged appearance. Sarila has 154 >=0.5m steps and 32 adjacent reversing large-step pairs; overlapping 30-day windows mark 3,584 of 4,108 observed timestamps (0.8724). The plot can look smooth at full-series scale because the threshold also counts sub-meter reversals alongside much larger level movement. At the stricter >=1m / >=0.80 rule Sarila has invalid-share 0. Its mutually exclusive Phase 2c primary label is `stuck`; the independent flip-flop flag is also true.

## Mask-safety recommendations

- Placeholder prefix: safe only as a candidate mask when the prefix is an explicit sentinel/initial flat sequence and run-continuation confirms the boundary; current 60-day truncation requires review for the stations in `placeholder_truncation_check.csv`.
- Exact repeated spikes: safe to mask only with exact repeated value, isolated point, and immediate exact reversion on both adjacent six-hour slots.
- Stuck/near-flat and flip-flop: flag-only. Do not delete or replace these points automatically.
- Extreme-burst points: flag-only unless they independently satisfy the narrow exact-spike mask.

## Limits and files

The sample is a visual precision check, not an independent performance estimate. Extreme-burst points use robust station bounds from pre-test observed values after excluding placeholder prefixes, known sentinels, and long exact runs. A flagged overlay is an algorithmic label, not verified telemetry fault attribution.

Precision is not estimable from the positive-only phase2c class sample without visual labels for negatives. This pass reports overlay counts and lets a reviewer assess false positives in the 24 clean/stuck/flip-flop/placeholder random strata.

Files: `visual_sample_stations.csv`, `visual_detector_counts.csv`, `placeholder_truncation_check.csv`, `sensitivity_summary.csv`, `sensitivity_station_flags.csv`, `sarila_flipflop_explanation.csv`, `visual_plot_manifest.csv`, individual PNGs in `plots/`, and contact sheets in `contact_sheets/`.
