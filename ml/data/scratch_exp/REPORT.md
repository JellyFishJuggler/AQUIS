# ML scratch experiments — consolidated notes

> **Archive:** this document combines historical sprint, red-team, and audit
> notes from September–October 2026. Some runs were invalidated by a metric bug;
> those sections are explicitly marked invalid and must not be used as current
> performance claims. For the concise algorithm comparison and current local
> serving state, start with [`../../README.md`](../../README.md). The detailed
> corrected new-station analysis remains below for provenance.

The reports are retained as experiment records. They may describe old artifacts,
scripts, and configurations that no longer match the active serving setup.


---

<!-- source: tracks/final_report.md -->

# Honest-30-day skill sprint — final report

Date: 2026-10-09 · Scope: `ml/data/scratch_exp/` (research scratch, not production)
Protocol owner: path A (strict, leak-free, no selection on 2026 test)

## 1. Verdict

**Nothing is promoted. `models/` stays frozen at tag `baseline-leakfree-2026-10`;
serve stays unchanged.**

* Strict fold-2024 gate (beat E0, paired-station CI excluding 1, gain > seed
  spread, fold-2025 sign check): **0 of 28 arms pass** (§3).
* One-shot 2026 on the frozen frame: E0 ratio **0.995 [0.9902, 0.9970]**
  (rmse 2.4673 vs persist 2.4798); E_best 0.9978 (§4). This is a *nominal*
  single-shot reading of a frozen config, not a selection result.
* Held-out-station CV: **no model beats persistence** (E_best 1.269 / 1.108;
  E0 1.482 / 1.216) (§5).
* Tier slicing and the DL probe add no robustness evidence (§6, §7).

## 2. The metric bug and the invalid numbers

Errors were computed with an extra `− gwl` (`yd − gwl` instead of `yd − pred`)
in exactly three scripts (`exp_harness.run_arm`, `exp_stationcv.py`,
`exp_eval_2026.py`) before 2026-10-08 10:11. Every number those scripts wrote
before the fix is invalid (run-1 sweep incl. the "C6 winner 0.9798", run-1
station CV 0.9706/0.9718, run-1 one-shot 1.0089/1.0035, coverage 0.0358,
prod-511 1.9582). Full census: `tracks/INVALID_numbers.md`. Scripts scoring in
level space were never affected.

## 3. Strict fold scoreboard (corrected, VALID)

Fixed harness, 3-seed pooled predictions, B=2000 station bootstrap. `n=3672`
fold-2024 stride rows, persist rmse 1.7573 m. Pass requires ratio < 1 **and**
CI lo > 1 **and** gain > spread **and** fold-2025 ratio ≤ E0's.

| arm | f24 ratio | f24 CI | f25 ratio |
|-----|-----------|--------|-----------|
| E0 (production) | 1.0668 | [1.0038, 1.1323] | 1.0427 |
| E_best (C6 feats, huber) | 1.0672 | [1.0110, 1.1245] | 1.0039 |
| E1 | 1.0421 | [0.9824, 1.1041] | 0.9756 |
| E2a | 0.9896 | [0.9395, 1.0374] | 1.2164 |
| E2b | 1.0948 | [1.0269, 1.1633] | 1.0745 |
| E3 | 1.0465 | [0.9831, 1.1117] | 1.0427 |
| E4 | 1.0776 | [1.0224, 1.1400] | 1.1317 |
| E5 | 1.2230 | [1.1278, 1.3323] | 1.0702 |
| R1 | 1.1062 | [1.0298, 1.1844] | 1.0536 |
| C1 | 1.1348 | [1.0567, 1.2162] | 1.1078 |
| C2 | 1.0815 | [1.0134, 1.1526] | 0.9978 |
| C3 | 1.0503 | [0.9898, 1.1103] | 1.0086 |
| C4 | 1.0320 | [0.9789, 1.0823] | 0.9778 |
| C5 | 1.0677 | [1.0033, 1.1357] | 1.0027 |
| C6 | 1.0709 | [1.0082, 1.1362] | 1.0144 |
| C7 | 1.0637 | [1.0071, 1.1204] | 1.0020 |
| C8 | 1.0571 | [1.0030, 1.1112] | 1.0149 |
| C9 | 1.0882 | [1.0239, 1.1537] | 1.0340 |
| T1 | 0.9674 | [0.9224, 1.0090] | 1.1447 |
| T2 | 1.0442 | [0.9837, 1.1073] | 1.0748 |
| T3 | 1.0325 | [0.9748, 1.0881] | 0.9962 |
| T4 | 1.1706 | [1.0922, 1.2565] | 1.0201 |
| T5 | 0.9635 | [0.9181, 1.0070] | 1.0899 |
| T6 | 1.0757 | [1.0096, 1.1446] | 1.0568 |
| T7 | 1.0697 | [1.0047, 1.1371] | 0.9876 |
| T8 | 1.1004 | [1.0335, 1.1697] | 1.0184 |

Only E2a, T1, T5 dip below 1 on fold-2024 — all three have CIs that include 1
*and* blow up on fold-2025 (1.2164 / 1.1447 / 1.0899). Source:
`arm_reports.json`.

## 4. One-shot 2026 (frozen frame) and reconciliation

`eval_2026.json` — frozen inputs, identical windows, 3-seed mean, seed 42/840/
2056:

| config | ratio | CI | rmse model | rmse persist | coverage | width |
|--------|-------|----|-----------|--------------|----------|-------|
| E0 | 0.9950 | [0.9902, 0.9970] | 2.4673 | 2.4798 | — | — |
| E_best | 0.9978 | [0.9956, 0.9987] | 2.4743 | 2.4798 | 0.898 | 3.1267 m |

Reconciliation of the three honest runs (Q2 of the brief):

| run | n rows | rmse model | rmse persist | ratio | frame / train / seed |
|-----|--------|-----------|--------------|-------|----------------------|
| production | 3764 | 2.4182 | 2.4759 | 0.9767 [0.9522, 0.9866] | live @2026-10-07 21:16 · downsample `cumcount%4==0` · seed 42 · CPU |
| ablation A | 3766 | 2.4607 | 2.4798 | 0.9919 | live · full train · 3-seed |
| one-shot E0 | 3766 | 2.4673 | 2.4798 | 0.9950 | frozen · full train · 3-seed · GPU |

**Exact closure of the persist gap (2.4798 vs 2.4759):** the 3764 overlapping
rows agree to 1e-8; the only 2 extra rows are both 2026-09-05 — Aligarh Lodha
(Δ = 0.70 m) and Hathras (Δ = −9.25 m). Required Σd² = 85.7, actual 86.0.
No hidden subsetting.

## 5. Held-out-station CV (corrected)

`stationcv.json` — stations split into disjoint train/score folds; the
score-side station is never seen in training. Runtime 5.9 min.

| config | fold | ratio | CI | rmse model | rmse persist | n |
|--------|------|-------|----|-----------|--------------|---|
| E0 | 2024 | 1.4819 | [1.1940, 1.8536] | 2.6041 | 1.7573 | 3672 |
| E0 | 2025 | 1.2158 | [0.9917, 1.4852] | 2.1216 | 1.7451 | 578 |
| E_best | 2024 | 1.2690 | [1.1472, 1.4114] | 2.2299 | 1.7573 | 3672 |
| E_best | 2025 | 1.1075 | [0.9893, 1.2414] | 1.9326 | 1.7451 | 578 |

Everything is ≥ 1 and 2024 CIs sit entirely above 1: **on unseen stations the
models are worse than persistence.** This is the strongest robustness
counter-signal in the sprint.

## 6. Tier tables (Track 2, 2026 stride, labels from training data only)

`tiers_2026.json`; headline assertions reproduced the one-shot exactly (E0
0.995, E_best 0.9978, coverage 0.898), so tier slices are on the same rows as
§4. `n=3766` rows / 511 stations; every test station got a label.

Axis A — volatility (per-station persist RMSE on fold-2024 stride, a_q1 calmest
… a_q4 most volatile):

| tier | stations | rows | E0 | E0 CI | E_best | E_best CI | coverage | width |
|------|----------|------|----|-------|--------|-----------|----------|-------|
| a_q1 | 107 | 799 | 0.9973 | [0.9880, 1.0018] | 0.9973 | [0.9899, 0.9991] | 0.934 | 2.79 |
| a_q2 | 134 | 990 | 0.9795 | [0.9733, 0.9854] | 0.9883 | [0.9855, 0.9907] | 0.926 | 3.08 |
| a_q3 | 134 | 988 | 0.9962 | [0.9764, 0.9988] | 0.9987 | [0.9902, 0.9998] | 0.920 | 2.94 |
| a_q4 | 136 | 989 | 0.9951 | [0.9900, 0.9972] | 0.9979 | [0.9958, 0.9988] | 0.819 | 3.64 |

Axis B — 6h record completeness over the training window (b_q1 sparsest …
b_q4 densest):

| tier | stations | rows | E0 | E0 CI | E_best | E_best CI | coverage | width |
|------|----------|------|----|-------|--------|-----------|----------|-------|
| b_q1 | 131 | 974 | 0.9945 | [0.9835, 0.9975] | 0.9978 | [0.9927, 0.9989] | 0.881 | 3.14 |
| b_q2 | 110 | 801 | 0.9976 | [0.9891, 0.9991] | 0.9987 | [0.9941, 0.9995] | 0.858 | 3.35 |
| b_q3 | 135 | 998 | 0.9847 | [0.9757, 0.9903] | 0.9945 | [0.9902, 0.9985] | 0.916 | 2.98 |
| b_q4 | 135 | 993 | 0.9935 | [0.9828, 0.9973] | 0.9964 | [0.9910, 0.9983] | 0.930 | 3.08 |

Read: no stratum shows a decisive win. Skill is at most ~2% and only in
a_q2/b_q3; the most volatile tier (a_q4) has the worst interval coverage
(0.819) and widest intervals — the interval method degrades exactly where it
matters. Right-sizes the 0.995 headline: it is a small aggregate tilt, not
uniform skill.

## 7. Track 3 — DL probe (time-boxed)

`track3_result.json`: **DROP.** 50 stations (top-50 by fold-2024 train count),
daily grid (`time.hour==0`), per-station every-30th-day stride; E0-XGB and a
2-layer MLP (256→128, dropout 0.1, Adam 1e-3) fit on the *same* rows.

| fold | rows | stations | persist rmse | XGB ratio | MLP ratio |
|------|------|----------|--------------|-----------|-----------|
| 2024 | 326 | 50 | 0.7777 | 1.2446 [1.0733, 1.4618] | 8.3322 [1.0116, 14.8465] |
| 2025 | 35 | 24 | 1.9315 | 1.2936 [0.8386, 1.3409] | 1.1091 [1.0558, 1.5854] |

The MLP diverged on fold-2024 (also hit all-NaN train columns, imputed to 0 per
the pre-registered rule) → DROP by the pre-agreed rule (MLP ≥ XGB on
fold-2024). Runtime 0.7 min, well inside the 3 h box. No DL follow-up
recommended at this data scale.

## 8. Limitations carried forward

* Frozen 2025 table has a monsoon hole (Sep–Dec 2025 coverage ~0–15%), so the
  fold-2025 sign check is weak by construction; documented in
  `tracks/track0_and_limitations.md`.
* Track 0: no usable future-rain driver (Open-Meteo max 16 d current-run;
  ERA5/NASA POWER would leak) → E_best is trailing-only.
* Q3 (monsoon): exclusion of 2024 monsoon from fold-2024 train is a fact by
  construction (`exp_harness.py:101`); the smaller-2024-delta effect is tested
  (persist stride RMSE fold-2024 1.7573 vs fold-2025 1.7451 vs 2026 2.4798,
  +41% on 2026); the "monsoon-in-training causes the deficit" story is a
  hypothesis with counter-evidence (fold-2025 still loses despite recent
  monsoon in train; non-monsoon E0 1.0063 fold-24 / 0.8577 fold-25).
* Serve note: production path untouched. The q05/q95 interval
  (coverage 0.898 @ 3.13 m) is experimental and not wired into serve.

## 9. Environment / ops notes

* Track 3 ran in `ml/venv312` (python 3.12): torch 2.14.1+cu130 (CUDA ok on RTX
  3050), xgboost 3.4.1, scikit-learn 1.9.1, scipy 1.18.1, pandas 3.0.6,
  pyarrow 25.0.1. All other scripts use `ml/venv` (python 3.14).
* The machine rebooted mid-sprint (2026-10-09 ~01:04); all background jobs had
  to be relaunched under `setsid`. The refresh daemon
  (`refresh.cli schedule --poll 60`, formerly PID 49011) is **not running**
  after the reboot.

## 10. Artifacts and provenance

| path | what |
|------|------|
| `arm_reports.json` | corrected 28-arm fold sweep (10:11 2026-10-08) |
| `eval_2026.json` | corrected one-shot 2026 (10:17 2026-10-08) |
| `stationcv.json` | corrected held-out-station CV |
| `tracks/tiers_2026.json`, `tracks/preds_2026_stride.parquet` | Track 2 |
| `tracks/track3_result.json` | Track 3 verdict |
| `tracks/INVALID_numbers.md` | bug census + invalid-number list |
| `tracks/track0_and_limitations.md` | Track 0 / frozen-data limitation |
| `frozen/` | input snapshot + `inputs_sha256.txt` |
| `run_remaining.sh` | driver (Track 2 → Track 3) |
| `tracks/artifact_sha256.txt` | hashes of the ignored large artifacts |

Suggested local tag: `honest30d-scratch-20261008` (no push). Nothing under
`models/` or the production serve path was modified.


---

<!-- source: tracks/persistence_audit.md -->

# Persistence Audit (30-day horizon, 6h grid)

## Definitions (verified on tracks/base_features.parquet)
- delta identity: `target_d == target - gwl` on 2,059,444 rows (|err| max 5.55e-06, mean 2.19e-07). The 30-day delta label is exactly the observed level change.
- horizon: `target_time - time = 720h` exactly (30d) on 2,059,444 finite rows.
- anchor/grid gap: `date - time` centred -9.0h, range [-18,+0]h (features' as-of date precedes the 6h issue time; documented, not a leak).

## Persistence rule under test
- Persistence {delta, level} forecast at t+30d: **delta = 0**, **level = anchor gwl**.
- Hence persistence delta error on every row is exactly `yd` (identity above); level error = `target - gwl = yd`.

## Persistence RMSE on all 3,766 2026-stride rows
- rows = 3766, stations = 511
- RMSE = 2.480 m, MAE = 0.829 m
- |error| quantiles: q50 0.325 q75 0.777 q90 1.757 q95 2.914 q99 7.839
- mean per-station |error|: min 0.020 median 0.516 max 19.367

## 5 sample anchors x six dashboard stations

| station | time | anchor_date | target_time | gwl_anchor | target_level | delta_actual | delta_persist | level_pred_persist | level_err |
|---|---|---|---|---|---|---|---|---|---|
| ASHADHA PRATHMIK VIDYALAYA | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -6.411 | -6.267 | 0.144 | 0 | -6.411 | 0.144 |
| ASHADHA PRATHMIK VIDYALAYA | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -6.267 | -6.414 | -0.147 | 0 | -6.267 | -0.147 |
| ASHADHA PRATHMIK VIDYALAYA | 2026-04-02 06:00:00 | 2026-04-02 00:00:00 | 2026-05-02 06:00:00 | -6.29 | -7.049 | -0.759 | 0 | -6.29 | -0.759 |
| ASHADHA PRATHMIK VIDYALAYA | 2026-05-03 12:00:00 | 2026-05-03 00:00:00 | 2026-06-02 12:00:00 | -7.049 | -6.713 | 0.336 | 0 | -7.049 | 0.336 |
| ASHADHA PRATHMIK VIDYALAYA | 2026-07-26 12:00:00 | 2026-07-26 00:00:00 | 2026-08-25 12:00:00 | -7.692 | -6.982 | 0.71 | 0 | -7.692 | 0.71 |
| Azamgarh Ahiraula Bahera | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -5.237 | -5.312 | -0.075 | 0 | -5.237 | -0.075 |
| Azamgarh Ahiraula Bahera | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -5.312 | -5.348 | -0.036 | 0 | -5.312 | -0.036 |
| Azamgarh Ahiraula Bahera | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -5.425 | -5.605 | -0.18 | 0 | -5.425 | -0.18 |
| Azamgarh Ahiraula Bahera | 2026-05-31 00:00:00 | 2026-05-31 00:00:00 | 2026-06-30 00:00:00 | -5.838 | -6.039 | -0.201 | 0 | -5.838 | -0.201 |
| Azamgarh Ahiraula Bahera | 2026-07-31 00:00:00 | 2026-07-31 00:00:00 | 2026-08-30 00:00:00 | -6.016 | -5.837 | 0.179 | 0 | -6.016 | 0.179 |
| Azamgarh Ahiraula Kandara | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -3.289 | -3.245 | 0.044 | 0 | -3.289 | 0.044 |
| Azamgarh Ahiraula Kandara | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -3.245 | -3.465 | -0.22 | 0 | -3.245 | -0.22 |
| Azamgarh Ahiraula Kandara | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -3.288 | -3.612 | -0.324 | 0 | -3.288 | -0.324 |
| Azamgarh Ahiraula Kandara | 2026-05-31 00:00:00 | 2026-05-31 00:00:00 | 2026-06-30 00:00:00 | -3.344 | -3.802 | -0.458 | 0 | -3.344 | -0.458 |
| Azamgarh Ahiraula Kandara | 2026-07-31 00:00:00 | 2026-07-31 00:00:00 | 2026-08-30 00:00:00 | -3.724 | -4.02 | -0.296 | 0 | -3.724 | -0.296 |
| Azamgarh Atrauliya Atrauliya | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -1.644 | -1.715 | -0.071 | 0 | -1.644 | -0.071 |
| Azamgarh Atrauliya Atrauliya | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -1.715 | -1.793 | -0.078 | 0 | -1.715 | -0.078 |
| Azamgarh Atrauliya Atrauliya | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -1.853 | -1.959 | -0.106 | 0 | -1.853 | -0.106 |
| Azamgarh Atrauliya Atrauliya | 2026-05-01 00:00:00 | 2026-05-01 00:00:00 | 2026-05-31 00:00:00 | -1.959 | -1.996 | -0.037 | 0 | -1.959 | -0.037 |
| Azamgarh Atrauliya Atrauliya | 2026-08-07 00:00:00 | 2026-08-07 00:00:00 | 2026-09-06 00:00:00 | -2.128 | -1.223 | 0.905 | 0 | -2.128 | 0.905 |
| Azamgarh Jahanganj Godhaura | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -4.974 | -4.98 | -0.006 | 0 | -4.974 | -0.006 |
| Azamgarh Jahanganj Godhaura | 2026-03-02 00:00:00 | 2026-03-02 00:00:00 | 2026-04-01 00:00:00 | -5.047 | -5.048 | -0.001 | 0 | -5.047 | -0.001 |
| Azamgarh Jahanganj Godhaura | 2026-05-01 00:00:00 | 2026-05-01 00:00:00 | 2026-05-31 00:00:00 | -5.755 | -5.282 | 0.473 | 0 | -5.755 | 0.473 |
| Azamgarh Jahanganj Godhaura | 2026-06-30 00:00:00 | 2026-06-30 00:00:00 | 2026-07-30 00:00:00 | -4.52 | -5.145 | -0.625 | 0 | -4.52 | -0.625 |
| Azamgarh Jahanganj Godhaura | 2026-08-31 18:00:00 | 2026-08-31 00:00:00 | 2026-09-30 18:00:00 | -5.452 | -5.117 | 0.335 | 0 | -5.452 | 0.335 |
| Azamgarh Lalganj Lalganj | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -3.861 | -3.825 | 0.036 | 0 | -3.861 | 0.036 |
| Azamgarh Lalganj Lalganj | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -3.825 | -3.741 | 0.084 | 0 | -3.825 | 0.084 |
| Azamgarh Lalganj Lalganj | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -3.65 | -4.126 | -0.476 | 0 | -3.65 | -0.476 |
| Azamgarh Lalganj Lalganj | 2026-05-31 00:00:00 | 2026-05-31 00:00:00 | 2026-06-30 00:00:00 | -3.073 | -3.265 | -0.192 | 0 | -3.073 | -0.192 |
| Azamgarh Lalganj Lalganj | 2026-08-06 06:00:00 | 2026-08-06 00:00:00 | 2026-09-05 06:00:00 | -3.619 | -3.115 | 0.504 | 0 | -3.619 | 0.504 |
| Azamgarh Palhana Balahi Lahucha | 2026-01-01 06:00:00 | 2026-01-01 00:00:00 | 2026-01-31 06:00:00 | -5.363 | -5.388 | -0.025 | 0 | -5.363 | -0.025 |
| Azamgarh Palhana Balahi Lahucha | 2026-01-31 06:00:00 | 2026-01-31 00:00:00 | 2026-03-02 06:00:00 | -5.388 | -5.417 | -0.029 | 0 | -5.388 | -0.029 |
| Azamgarh Palhana Balahi Lahucha | 2026-04-01 06:00:00 | 2026-04-01 00:00:00 | 2026-05-01 06:00:00 | -5.43 | -5.412 | 0.018 | 0 | -5.43 | 0.018 |
| Azamgarh Palhana Balahi Lahucha | 2026-05-01 06:00:00 | 2026-05-01 00:00:00 | 2026-05-31 06:00:00 | -5.412 | -5.423 | -0.011 | 0 | -5.412 | -0.011 |
| Azamgarh Palhana Balahi Lahucha | 2026-08-07 06:00:00 | 2026-08-07 00:00:00 | 2026-09-06 06:00:00 | -5.494 | -5.529 | -0.035 | 0 | -5.494 | -0.035 |
| Azamgarh Palhana Jamui Bhatauli | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -2.692 | -2.616 | 0.076 | 0 | -2.692 | 0.076 |
| Azamgarh Palhana Jamui Bhatauli | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -2.616 | -2.651 | -0.035 | 0 | -2.616 | -0.035 |
| Azamgarh Palhana Jamui Bhatauli | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -2.801 | -2.757 | 0.044 | 0 | -2.801 | 0.044 |
| Azamgarh Palhana Jamui Bhatauli | 2026-05-01 00:00:00 | 2026-05-01 00:00:00 | 2026-05-31 00:00:00 | -2.757 | -2.755 | 0.002 | 0 | -2.757 | 0.002 |
| Azamgarh Palhana Jamui Bhatauli | 2026-08-01 00:00:00 | 2026-08-01 00:00:00 | 2026-08-31 00:00:00 | -2.049 | -1.764 | 0.285 | 0 | -2.049 | 0.285 |
| Azamgarh Pavai Velasiya | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -2.227 | -2.286 | -0.059 | 0 | -2.227 | -0.059 |
| Azamgarh Pavai Velasiya | 2026-03-02 00:00:00 | 2026-03-02 00:00:00 | 2026-04-01 00:00:00 | -2.372 | -3.65 | -1.278 | 0 | -2.372 | -1.278 |
| Azamgarh Pavai Velasiya | 2026-05-01 00:00:00 | 2026-05-01 00:00:00 | 2026-05-31 00:00:00 | -3.503 | -3.013 | 0.49 | 0 | -3.503 | 0.49 |
| Azamgarh Pavai Velasiya | 2026-06-30 00:00:00 | 2026-06-30 00:00:00 | 2026-07-30 00:00:00 | -3.135 | -3.633 | -0.498 | 0 | -3.135 | -0.498 |
| Azamgarh Pavai Velasiya | 2026-08-31 00:00:00 | 2026-08-31 00:00:00 | 2026-09-30 00:00:00 | -2.886 | -0.555 | 2.331 | 0 | -2.886 | 2.331 |
| Azamgarh Sathiyavn Gajahada | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -0.286 | -0.553 | -0.267 | 0 | -0.286 | -0.267 |
| Azamgarh Sathiyavn Gajahada | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -0.553 | -0.828 | -0.275 | 0 | -0.553 | -0.275 |
| Azamgarh Sathiyavn Gajahada | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -0.98 | -1.181 | -0.201 | 0 | -0.98 | -0.201 |
| Azamgarh Sathiyavn Gajahada | 2026-05-01 00:00:00 | 2026-05-01 00:00:00 | 2026-05-31 00:00:00 | -1.181 | -1.368 | -0.187 | 0 | -1.181 | -0.187 |
| Azamgarh Sathiyavn Gajahada | 2026-08-07 00:00:00 | 2026-08-07 00:00:00 | 2026-09-06 00:00:00 | 0.463 | 0.418 | -0.045 | 0 | 0.463 | -0.045 |
| Azamgarh Thekama Bardah | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -1.167 | -1.117 | 0.05 | 0 | -1.167 | 0.05 |
| Azamgarh Thekama Bardah | 2026-03-02 00:00:00 | 2026-03-02 00:00:00 | 2026-04-01 00:00:00 | -1.112 | -1.098 | 0.014 | 0 | -1.112 | 0.014 |
| Azamgarh Thekama Bardah | 2026-05-01 00:00:00 | 2026-05-01 00:00:00 | 2026-05-31 00:00:00 | -1.645 | -1.596 | 0.049 | 0 | -1.645 | 0.049 |
| Azamgarh Thekama Bardah | 2026-06-30 00:00:00 | 2026-06-30 00:00:00 | 2026-07-30 00:00:00 | -1.696 | -1.527 | 0.169 | 0 | -1.696 | 0.169 |
| Azamgarh Thekama Bardah | 2026-08-31 00:00:00 | 2026-08-31 00:00:00 | 2026-09-30 00:00:00 | -1.654 | -2.969 | -1.315 | 0 | -1.654 | -1.315 |
| Banda Baberu Anvan | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -1.127 | -1.494 | -0.367 | 0 | -1.127 | -0.367 |
| Banda Baberu Anvan | 2026-02-05 00:00:00 | 2026-02-05 00:00:00 | 2026-03-07 00:00:00 | -1.518 | -2.347 | -0.829 | 0 | -1.518 | -0.829 |
| Banda Baberu Anvan | 2026-04-06 00:00:00 | 2026-04-06 00:00:00 | 2026-05-06 00:00:00 | -2.452 | -2.922 | -0.47 | 0 | -2.452 | -0.47 |
| Banda Baberu Anvan | 2026-05-06 00:00:00 | 2026-05-06 00:00:00 | 2026-06-05 00:00:00 | -2.922 | -3.29 | -0.368 | 0 | -2.922 | -0.368 |
| Banda Baberu Anvan | 2026-08-13 06:00:00 | 2026-08-13 00:00:00 | 2026-09-12 06:00:00 | -1.083 | 0.24 | 1.323 | 0 | -1.083 | 1.323 |
| Banda Baberu Umrehani | 2026-01-01 06:00:00 | 2026-01-01 00:00:00 | 2026-01-31 06:00:00 | -1.321 | -1.569 | -0.248 | 0 | -1.321 | -0.248 |
| Banda Baberu Umrehani | 2026-01-31 06:00:00 | 2026-01-31 00:00:00 | 2026-03-02 06:00:00 | -1.569 | -2.135 | -0.566 | 0 | -1.569 | -0.566 |
| Banda Baberu Umrehani | 2026-04-01 06:00:00 | 2026-04-01 00:00:00 | 2026-05-01 06:00:00 | -2.832 | -3.638 | -0.806 | 0 | -2.832 | -0.806 |
| Banda Baberu Umrehani | 2026-05-01 06:00:00 | 2026-05-01 00:00:00 | 2026-05-31 06:00:00 | -3.638 | -4.235 | -0.597 | 0 | -3.638 | -0.597 |
| Banda Baberu Umrehani | 2026-08-12 18:00:00 | 2026-08-12 00:00:00 | 2026-09-11 18:00:00 | -0.346 | -0.11 | 0.236 | 0 | -0.346 | 0.236 |
| Banda Badokhar Khurd Chatgan | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -8.958 | -8.789 | 0.169 | 0 | -8.958 | 0.169 |
| Banda Badokhar Khurd Chatgan | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -8.789 | -8.679 | 0.11 | 0 | -8.789 | 0.11 |
| Banda Badokhar Khurd Chatgan | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -8.658 | -8.744 | -0.086 | 0 | -8.658 | -0.086 |
| Banda Badokhar Khurd Chatgan | 2026-06-05 12:00:00 | 2026-06-05 00:00:00 | 2026-07-05 12:00:00 | -8.99 | -9.239 | -0.249 | 0 | -8.99 | -0.249 |
| Banda Badokhar Khurd Chatgan | 2026-08-11 12:00:00 | 2026-08-11 00:00:00 | 2026-09-10 12:00:00 | -8.716 | -7.946 | 0.77 | 0 | -8.716 | 0.77 |
| Banda Bisndha Mrooli | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -1.926 | -3.202 | -1.276 | 0 | -1.926 | -1.276 |
| Banda Bisndha Mrooli | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -3.202 | -4.502 | -1.3 | 0 | -3.202 | -1.3 |
| Banda Bisndha Mrooli | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -4.928 | -5.288 | -0.36 | 0 | -4.928 | -0.36 |
| Banda Bisndha Mrooli | 2026-07-06 00:00:00 | 2026-07-06 00:00:00 | 2026-08-05 00:00:00 | -5.914 | -3.378 | 2.536 | 0 | -5.914 | 2.536 |
| Banda Bisndha Mrooli | 2026-09-05 00:00:00 | 2026-09-05 00:00:00 | 2026-10-05 00:00:00 | -0.114 | -0.485 | -0.371 | 0 | -0.114 | -0.371 |
| Banda Kamasin Tilowsa | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -0.733 | -2.868 | -2.135 | 0 | -0.733 | -2.135 |
| Banda Kamasin Tilowsa | 2026-02-01 00:00:00 | 2026-02-01 00:00:00 | 2026-03-03 00:00:00 | -2.867 | -4.301 | -1.434 | 0 | -2.867 | -1.434 |
| Banda Kamasin Tilowsa | 2026-04-02 00:00:00 | 2026-04-02 00:00:00 | 2026-05-02 00:00:00 | -3.643 | -3.578 | 0.065 | 0 | -3.643 | 0.065 |
| Banda Kamasin Tilowsa | 2026-05-02 00:00:00 | 2026-05-02 00:00:00 | 2026-06-01 00:00:00 | -3.578 | -3.592 | -0.014 | 0 | -3.578 | -0.014 |
| Banda Kamasin Tilowsa | 2026-08-12 18:00:00 | 2026-08-12 00:00:00 | 2026-09-11 18:00:00 | -0.202 | -0.089 | 0.113 | 0 | -0.202 | 0.113 |
| Banda Tindwari Bhindowra | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -12.604 | -14 | -1.396 | 0 | -12.604 | -1.396 |
| Banda Tindwari Bhindowra | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -14 | -1.055 | 12.945 | 0 | -14 | 12.945 |
| Banda Tindwari Bhindowra | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -14.757 | -13.672 | 1.085 | 0 | -14.757 | 1.085 |
| Banda Tindwari Bhindowra | 2026-05-02 00:00:00 | 2026-05-02 00:00:00 | 2026-06-01 00:00:00 | -13.672 | -13.7 | -0.028 | 0 | -13.672 | -0.028 |
| Banda Tindwari Bhindowra | 2026-08-07 12:00:00 | 2026-08-07 00:00:00 | 2026-09-06 12:00:00 | -15.051 | -6.495 | 8.556 | 0 | -15.051 | 8.556 |
| Deoria Bhaluwani Markna | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -2.581 | -2.952 | -0.371 | 0 | -2.581 | -0.371 |
| Deoria Bhaluwani Markna | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -2.952 | -3.14 | -0.188 | 0 | -2.952 | -0.188 |
| Deoria Bhaluwani Markna | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -3.137 | -3.221 | -0.084 | 0 | -3.137 | -0.084 |
| Deoria Bhaluwani Markna | 2026-07-06 00:00:00 | 2026-07-06 00:00:00 | 2026-08-05 00:00:00 | -3.585 | -2.381 | 1.204 | 0 | -3.585 | 1.204 |
| Deoria Bhaluwani Markna | 2026-09-04 00:00:00 | 2026-09-04 00:00:00 | 2026-10-04 00:00:00 | -1.949 | -0.315 | 1.634 | 0 | -1.949 | 1.634 |
| Deoria Bhatani Mishrauli Dichit | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -2.515 | -2.703 | -0.188 | 0 | -2.515 | -0.188 |
| Deoria Bhatani Mishrauli Dichit | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -2.703 | -2.849 | -0.146 | 0 | -2.703 | -0.146 |
| Deoria Bhatani Mishrauli Dichit | 2026-03-02 00:00:00 | 2026-03-02 00:00:00 | 2026-04-01 00:00:00 | -2.849 | -2.909 | -0.06 | 0 | -2.849 | -0.06 |
| Deoria Bhatani Mishrauli Dichit | 2026-07-10 00:00:00 | 2026-07-10 00:00:00 | 2026-08-09 00:00:00 | -3.217 | -1.956 | 1.261 | 0 | -3.217 | 1.261 |
| Deoria Bhatani Mishrauli Dichit | 2026-08-09 00:00:00 | 2026-08-09 00:00:00 | 2026-09-08 00:00:00 | -1.956 | -1.727 | 0.229 | 0 | -1.956 | 0.229 |
| Deoria Bhatpar Rani Dharkwan | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -2.906 | -2.887 | 0.019 | 0 | -2.906 | 0.019 |
| Deoria Bhatpar Rani Dharkwan | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -2.887 | -2.754 | 0.133 | 0 | -2.887 | 0.133 |
| Deoria Bhatpar Rani Dharkwan | 2026-04-05 00:00:00 | 2026-04-05 00:00:00 | 2026-05-05 00:00:00 | -2.756 | -2.924 | -0.168 | 0 | -2.756 | -0.168 |
| Deoria Bhatpar Rani Dharkwan | 2026-05-05 00:00:00 | 2026-05-05 00:00:00 | 2026-06-04 00:00:00 | -2.924 | -2.48 | 0.444 | 0 | -2.924 | 0.444 |
| Deoria Bhatpar Rani Dharkwan | 2026-08-09 00:00:00 | 2026-08-09 00:00:00 | 2026-09-08 00:00:00 | -1.235 | -2.342 | -1.107 | 0 | -1.235 | -1.107 |
| Deoria Salempur Manihari | 2026-01-06 00:00:00 | 2026-01-06 00:00:00 | 2026-02-05 00:00:00 | -5.408 | -4.088 | 1.32 | 0 | -5.408 | 1.32 |
| Deoria Salempur Manihari | 2026-02-05 00:00:00 | 2026-02-05 00:00:00 | 2026-03-07 00:00:00 | -4.088 | -4.179 | -0.091 | 0 | -4.088 | -0.091 |
| Deoria Salempur Manihari | 2026-03-08 00:00:00 | 2026-03-08 00:00:00 | 2026-04-07 00:00:00 | -4.192 | -4.241 | -0.049 | 0 | -4.192 | -0.049 |
| Deoria Salempur Manihari | 2026-04-08 00:00:00 | 2026-04-08 00:00:00 | 2026-05-08 00:00:00 | -4.24 | -4.357 | -0.117 | 0 | -4.24 | -0.117 |
| Deoria Salempur Manihari | 2026-07-28 18:00:00 | 2026-07-28 00:00:00 | 2026-08-27 18:00:00 | -4.21 | -3.414 | 0.796 | 0 | -4.21 | 0.796 |
| Deoria Tarkulwa Tarkulwa | 2026-01-06 06:00:00 | 2026-01-06 00:00:00 | 2026-02-05 06:00:00 | -8.813 | -7.522 | 1.291 | 0 | -8.813 | 1.291 |
| Deoria Tarkulwa Tarkulwa | 2026-02-05 06:00:00 | 2026-02-05 00:00:00 | 2026-03-07 06:00:00 | -7.522 | -7.73 | -0.208 | 0 | -7.522 | -0.208 |
| Deoria Tarkulwa Tarkulwa | 2026-04-06 06:00:00 | 2026-04-06 00:00:00 | 2026-05-06 06:00:00 | -7.881 | -8.064 | -0.183 | 0 | -7.881 | -0.183 |
| Deoria Tarkulwa Tarkulwa | 2026-05-06 06:00:00 | 2026-05-06 00:00:00 | 2026-06-05 06:00:00 | -8.064 | -8.356 | -0.292 | 0 | -8.064 | -0.292 |
| Deoria Tarkulwa Tarkulwa | 2026-08-10 06:00:00 | 2026-08-10 00:00:00 | 2026-09-09 06:00:00 | -6.859 | -6.44 | 0.419 | 0 | -6.859 | 0.419 |
| Meerut Janikhurd Afjalpur Pavati | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -8.881 | -8.709 | 0.172 | 0 | -8.881 | 0.172 |
| Meerut Janikhurd Afjalpur Pavati | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -8.709 | -9.542 | -0.833 | 0 | -8.709 | -0.833 |
| Meerut Janikhurd Afjalpur Pavati | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -9.543 | -10.589 | -1.046 | 0 | -9.543 | -1.046 |
| Meerut Janikhurd Afjalpur Pavati | 2026-05-01 00:00:00 | 2026-05-01 00:00:00 | 2026-05-31 00:00:00 | -10.589 | -11.547 | -0.958 | 0 | -10.589 | -0.958 |
| Meerut Janikhurd Afjalpur Pavati | 2026-08-07 00:00:00 | 2026-08-07 00:00:00 | 2026-09-06 00:00:00 | -8.757 | -7.374 | 1.383 | 0 | -8.757 | 1.383 |
| Meerut Janikhurd Satvai | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -4.297 | -4.131 | 0.166 | 0 | -4.297 | 0.166 |
| Meerut Janikhurd Satvai | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -4.131 | -4.528 | -0.397 | 0 | -4.131 | -0.397 |
| Meerut Janikhurd Satvai | 2026-04-01 00:00:00 | 2026-04-01 00:00:00 | 2026-05-01 00:00:00 | -4.659 | -4.878 | -0.219 | 0 | -4.659 | -0.219 |
| Meerut Janikhurd Satvai | 2026-05-01 00:00:00 | 2026-05-01 00:00:00 | 2026-05-31 00:00:00 | -4.878 | -5.366 | -0.488 | 0 | -4.878 | -0.488 |
| Meerut Janikhurd Satvai | 2026-08-05 00:00:00 | 2026-08-05 00:00:00 | 2026-09-04 00:00:00 | -3.64 | -2.08 | 1.56 | 0 | -3.64 | 1.56 |
| Meerut Sardhana Chhabariya | 2026-01-01 00:00:00 | 2026-01-01 00:00:00 | 2026-01-31 00:00:00 | -11.862 | -11.868 | -0.006 | 0 | -11.862 | -0.006 |
| Meerut Sardhana Chhabariya | 2026-01-31 00:00:00 | 2026-01-31 00:00:00 | 2026-03-02 00:00:00 | -11.868 | -11.481 | 0.387 | 0 | -11.868 | 0.387 |
| Meerut Sardhana Chhabariya | 2026-03-02 00:00:00 | 2026-03-02 00:00:00 | 2026-04-01 00:00:00 | -11.481 | -11.56 | -0.079 | 0 | -11.481 | -0.079 |
| Meerut Sardhana Chhabariya | 2026-06-22 18:00:00 | 2026-06-22 00:00:00 | 2026-07-22 18:00:00 | -12.044 | -11.949 | 0.095 | 0 | -12.044 | 0.095 |
| Meerut Sardhana Chhabariya | 2026-08-21 18:00:00 | 2026-08-21 00:00:00 | 2026-09-20 18:00:00 | -11.475 | -11.004 | 0.471 | 0 | -11.475 | 0.471 |

Peak RSS 3661 MB.

**Conclusion:** persistence is the honest 'level persists' baseline; every model ratio divides by this RMSE on the same rows.


---

<!-- source: tracks/track0_and_limitations.md -->

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


---

<!-- source: tracks/track3_two_attempts.md -->

# Track 3 — two attempts and the divergence diagnosis

Same 50 stations (top-50 by fold-2024 train rows), daily grid (`time.hour==0`),
per-station every-30th-day stride, identical scoreboard (ratio = RMSE(model) /
RMSE(persistence) on stride rows, station bootstrap B=2000), E0-XGB refit on the
**same rows**. External hard box 90 min. venv312 (torch 2.14.1+cu130, CUDA).

| fold | rows / st | persist | XGB ratio (CI) | attempt-1 MLP (CI) | attempt-2 MLP (CI) |
|------|-----------|---------|----------------|--------------------|--------------------|
| 2024 | 326 / 50 | 0.7777 | 1.2446 [1.0733, 1.4618] | **8.3322** [1.0116, 14.8465] | **1.0245** [0.9565, 1.1108] |
| 2025 | 35 / 24 | 1.9315 | 1.2936 [0.8386, 1.3409] | 1.1091 [1.0558, 1.5854] | **1.0141** [0.9829, 1.2974] |

* attempt 1 (`exp_track3_dl.py`): **DROP** — MLP ≥ XGB on fold-2024.
* attempt 2 (`exp_track3_dl2.py`): **KEEP** — MLP 1.0245 < XGB 1.2446 on
  fold-2024, and 1.0141 < 1.2936 on fold-2025. Runtime 0.6 min.

**Important caveat:** KEEP means "beats the E0-XGB baseline on this tiny
50-station daily subset". The MLP's own ratio is still **> 1** (1.0245), with a
CI that includes 1 — i.e. the neural net still does **not** beat *persistence*
on unseen future data, it only beats the local XGB reference. No promotion
implication.

## What attempt 2 changed (each mapped to an attempt-1 weakness)

| change | attempt 1 | attempt 2 |
|--------|-----------|-----------|
| loss | MSE on raw deltas | `HuberLoss(delta=1.0)` on standardised target |
| target | raw metres (range ~±20) | standardised (fit on train) |
| normalisation | none | `LayerNorm` after each hidden Linear |
| gradients | unclipped | `clip_grad_norm_(..., 1.0)` |
| schedule | constant lr 1e-3 | `ReduceLROnPlateau(factor 0.5, patience 2)` |
| patience | 8 | 10 |
| NaN handling | silent median impute | median impute **+ explicit all-NaN audit** |

## Diagnosis — why attempt 1 diverged

1. **MSE on unstandardised delta targets.** The 30-day GWL deltas include a few
   large-magnitude labels (tens of metres). Squared error lets those points
   dominate: a single-batch gradient spike can blow the weights between two
   validation checks, which is exactly the failure mode seen (one fold's MLP
   ratio ~8.3, CI up to 14.8 — one or more seeds ran away while others stayed
   ~1.1, giving the huge spread).
2. **No gradient clipping / no schedule**, so a spike was never damped — a bad
   update persisted.
3. **No normalisation layers.** A 2-layer MLP on ~31 unscaled-in-hidden-space
   features with a large-target MSE objective has no mechanism to keep
   activations bounded.
4. **Silent all-NaN column.** `river_level` is all-NaN in this 50-station
   daily subset; attempt 1 imputed it to 0 (a constant) without reporting it.
   That alone is harmless, but it was a blind spot — attempt 2 records the
   audit (`nan_diag`: `{'2024': ['river_level'], '2025': ['river_level']}`).

## Evidence attempt 2 is stable (auditable)

Per-seed train-loss traces recorded in `tracks/track3_result2.json`
(`mlp_loss_trace`): fold-2024 seeds 42/840/2056 train loss
0.220→0.084 / 0.205→0.085 / 0.194→0.088, val-best 0.331 / 0.350 / 0.374; all
finite, all converged (hit the 120-epoch cap, i.e. still improving). Fold-2025
early-stopped at 44/50/76 epochs with decreasing train loss and finite val.
No `All-NaN`-driven NaN loss, no divergence.

## Conclusion

The DL probe is not evidence for promoting a neural net: attempt 2 beats the
local XGB baseline but still loses to persistence (ratio > 1, CI includes 1) on
the fold that matters. The divergence was an optimisation-stability bug (MSE +
no clipping/norm/schedule on raw-scale deltas), not a statement about model
capacity at this data scale.


---

<!-- source: tracks/stationcv_reconciliation.md -->

# Station-CV reconciliation — "does it work on new stations?"

Three numbers have been cited with different meanings; only one is a
new-station test.

| protocol | script | what it actually does | scored on stations unseen in fit? | result |
|----------|--------|-----------------------|-----------------------------------|--------|
| corrected station CV | `exp_stationcv.py` | retrain from scratch per fold on group g train/val rows, score the **other** group's stride rows | **yes** (rows of scored stations are absent from train+val) | E0 **1.482** (2024) / 1.216 (2025); E_best 1.269 / 1.108 → **loses to persistence** |
| "OOS held-out stations" | `31_honest_baseline.py` | read `outputs/predictions_2026.parquet` (produced by the model trained on **all** stations) and partition the *already-scored rows* by station parity | **no** — every station was in the fit | xgb 2.7627 vs persist 2.8218 = **+2.09%** |
| "spatial CV ~0.99" | `scratch_assoc_ablate.py` | genuine station-held-out retrain (fit half the stations, score the other half) | yes | but the reported 0.9925 is **ratio C/B between two model variants**, not model/persistence |

## Why the +2.1% is not a new-station result

`31_honest_baseline.py:102-107` only partitions rows. The xgb model in
`predictions_2026.parquet` (`06_train.py`) was fit with every station,
including the 228 "held-out" ones, and the file's own caveat says it is
recomputed from **pre-purge** frames, so target leakage is still present
(`honest_baseline_oos.json.caveat`). It is an upper bound on a *row* split, not
a test of generalization. The +2.09% (`oos_heldout_stations`) is exactly that
row-split artifact.

## Why the ablation ~0.99 is not model/persistence

`scratch_assoc_ablate.py:319-339` pools the two spatial folds and reports
`ratio_C_over_B` = 0.9925, CI [0.9757, 1.0043], `p_C_better` = 0.869. Both
numerator and denominator are *model* RMSE (C = rainfall-association variant,
B = plain association variant). It answers "do the static association columns
hurt on held-out stations?" — not "does the model beat persistence?". The
`~0.99` also matches the arms' **overall test** ratio (A/B `ratio_all` =
0.9919), which is the production stride protocol, not a station holdout.

## st_id / dist_id handling for unseen stations

All three protocols carry `st_id` and `dist_id` (they are in `BASE_XCOLS` /
`F.NUM_COLS + ["st_id","dist_id"]`). `st_id = sha1(Station)[:16]` truncated to
an int, i.e. an **arbitrary numeric label with no ordering**. For a station
absent from training, the tree therefore splits on an out-of-range integer and
lands wherever the numeric threshold puts it — pure extrapolation, no semantic
"new station" handling. **Yes, st_id is an unseen category** in both
station-held-out protocols (dist_id may be partially seen, since districts
contain several stations). A deployment-correct new-station protocol must drop
`st_id` (or map it to a single unknown code) and keep only features that exist
for a brand-new site (its own trailing GWL lags, weather, and district/neighbour
context). Neither station-held-out script does this, so:
* the 1.48 result is the honest direction (it does not even get the benefit of a
  neutral ID), and
* dropping st_id would not be expected to *rescue* the margin, it would remove
  an arbitrary-but-informative-for-seen-stations feature.

## Verdict

The **only** protocol that both retrains without the scored stations and scores
against persistence is `exp_stationcv.py`, and it says the models do **not**
generalize to unseen stations (ratio 1.27–1.48). For the claim "works on new
stations" that protocol is the right one; the `31_honest_baseline` OOS and the
ablation spatial-CV numbers do **not** support that claim. No promotion on this
basis.


---

<!-- source: tracks/INVALID_numbers.md -->

# INVALID numbers — double-subtraction error bug (fixed 2026-10-08)

## The bug

Three scripts computed scoring errors with an extra `−gwl` (the anchor level)
on **both** the model and the persistence side:

| Script | Buggy expression (before fix) | Correct form (now in the file) |
|---|---|---|
| `ml/data/scratch_exp/exp_harness.py` (`run_arm`) | `Em.append(yd − (gwl+pred))`, `Ep.append(yd − gwl)` | `err_m = yd − pred`, `err_p = yd` |
| `ml/data/scratch_exp/exp_stationcv.py` | `em_g = yd − (gwl+pred)`, `all_err_p.append(yd − gwl)` | `em_g = yd − pred`, `all_err_p.append(yd)` |
| `ml/data/scratch_exp/exp_eval_2026.py` | `err_p = yd − gwl`; model via `yd − (gwl+pred)` | `err_p = yd`; model via `yd − pred` |

Here `yd = target_d` (30-day GWL delta), `gwl` = anchor level, `pred` = model
delta forecast. The persistence baseline predicts delta 0, so its error is the
delta itself (`yd`), **not** `yd − gwl`.

**Scope of the bug:** repo-wide audit (`ml/*.py`, `ml/data/**/*.py`) shows the
pattern occurs **only** in the three scripts above. Everything else scores in
level space and was never affected — `scratch_assoc_ablate.py`
(`true_l − (anchor+pred)` / `true_l − anchor`), `data/scratch_assoc/diag_4c_glitches/run.py`
(`(xgb−target)²` / `(persist−target)²`), `31_honest_baseline.py`, `06_train.py`,
`gate_check.py`.

**Rule of thumb:** every number reported from the three scripts BEFORE the
fix timestamp (`arm_reports.json` rewritten 2026-10-08 10:11, `eval_2026.json`
rewritten 2026-10-08 10:17) is INVALID. An arbitrary corrupted persistence
denominator (`yd − gwl`) inflated ratios toward ~1 and made nearly every
configuration look like a small gain vs persistence.

---

## INVALID — run-1 fold sweep (`exp_harness.py`, CPU partial run + full GPU run)

All ratios/CIs below are model-RMSE over the **corrupted** persistence RMSE.
Row counts, fold windows and seed spreads are structural and remain valid;
every *ratio* does not.

### Arms

| Arm | fold-2024 (INVALID) | fold-2025 (INVALID) |
|---|---|---|
| E0 (control) | 0.9880 [0.9846, 0.9915] | 1.0106 |
| R1 | 0.9972 — FAIL | — |
| E4 | 1.0053 — FAIL | — |
| E2b | 0.9829 — "pass" | 0.9683 |
| E2a | 0.9950 — "pass" | 0.9579 |
| E1 | 0.9842 — "pass" | 0.9658 |
| E5 | 0.9894 — "pass" | 0.9889 |
| E3 | 0.9862 — "pass (marginal)" | 1.0106 |
| E_best naive union | 0.9933 | — |
| C1 | 0.9812 | — |
| C2 | 0.9838 | — |
| C3 | 0.9813 | — |
| C4 | 0.9938 | — |
| C5 (C3+neighbours) | 0.9812 | — |
| **C6 ("E_best winner")** | **0.9798 [0.9759, 0.9835]** gain 0.0202 spread 0.0033 js24 0.9765 | **0.9588** |
| C7 (full union) | 0.9931 | — |
| C8 (full union) | 0.9935 | — |
| C9 (C6, d7/lr0.07) | 0.9817 | 0.9541 |
| T1 (C2 cols, d4/lr0.03) | 0.9674-range values (run-1) | — |
| T3 | 0.9813 (d7/lr0.03) | — |
| T5–T8 | T6 0.9815; T8 0.9821 | T8 0.9524 |

### INVALID conclusions drawn from run-1

- "Passing arms exist: E2b, E2a, E1, E5, E3 (+ all candidates C1–C9, T1–T8
  beat persistence pointwise)" — **invalid**.
- "E_best = C6 (base + rain_api7/22/60 + rain_cum90/180/365 + e3_chg_k3/k5,
  pseudo-Huber) selected as winner" — **invalid as a selection** (the fold
  metric it won on was wrong). C6's config is *kept as the E_best candidate
  name only*; the strict corrected protocol selects nothing.
- "E_best beats E0 on fold-2024" — **invalid**.
- Fold row counts (train 815,104 / val 128,303 / holdout 390,668 / stride
  3,672 for fold-2024; 1,558,809 / 102,435 / 42,289 / 578 for fold-2025) are
  **valid** (structural).

## INVALID — run-1 held-out-station CV (`exp_stationcv.py`)

(`stationcv.json` from that run was deleted before the corrected re-run.)

| Config | fold-2024 (INVALID) | fold-2025 (INVALID) |
|---|---|---|
| E_best | 0.9706 [0.9641, 0.9773], gain 0.0294, spread 0.0028, n=3672 | 0.9531 [0.9457, 0.9600] |
| E0 | 0.9718 [0.9646, 0.9792], gain 0.0282, spread 0.0037 | 0.9542 [0.9448, 0.9622] |

- INVALID conclusion: "E_best ≥ E0 on both folds (station CV)". Corrected
  re-run supersedes this file (see `stationcv.json` + final report).

## INVALID — run-1 one-shot 2026 evals (`exp_eval_2026.py`, 2 buggy runs)

| Quantity | E0 (INVALID) | E_best (INVALID) |
|---|---|---|
| ratio | 1.0089 [1.0081, 1.0097] | 1.0035 [1.0032, 1.0039] |
| rmse (model / persist) | 12.1011 / 11.9946 | 12.0367 / 11.9946 |
| per-seed rmse | 12.1068 / 12.0865 / 12.1101 | 12.0329 / 12.0380 / 12.0394 |
| prod511-restricted ratio | 1.9582 | 1.9526 |
| interval coverage (q05/q95) | 0.0358 (level/delta unit mix) | same run |

- The rmse scale (~12 m) is the corrupted `|yd − gwl|` quantity, not a
  30-day delta error; the true persist stride RMSE on the same grid is
  ~2.48 m.
- `width_mean` 3.1267 m is unit-correct (`hi − lo` cancels the anchor) but was
  reported next to the invalid coverage.
- Corrected one-shot supersedes: `eval_2026.json` (2026-10-08 10:17) —
  E0 0.995 [0.9902, 0.9970], E_best 0.9978, coverage 0.898.

---

## VALID — never touched by the bug (do not relitigate)

| Artifact | Numbers | Why valid |
|---|---|---|
| Rain-link ablation (`scratch_assoc_ablate.py`, `diagnosis/2026-10-08_association_ablation/ablation_report.json`) | A 0.9919 [0.9865, 0.9942], B 0.9919 [0.9865, 0.9942], C 0.9989 [0.9977, 0.9997]; spatial CV C/B 0.9932 (p=0.869); decision: C does not beat B; seed-spread rule numbers | level-space errors: `true_l − (anchor+pred)` / `true_l − anchor` |
| Diag 4c glitch scan (`summary.json`) | all-rows 2.4182/2.4759 → 0.9767 [0.9522, 0.9866]; excl-glitch 1.9577/2.0209 → 0.9687 [0.9448, 0.9825]; fleet event counts | level-space from `predictions_2026` |
| Diag 4b best-lag (`fleet_summary.json`, `best_lag.csv`) | lag distribution (mean 10.6 d, median 4 d), 511 st / 3764 rows | correlation/lag analysis, no error formula |
| Production honest metrics (`outputs/honest_metrics.csv`, `outputs/honest_baseline_oos.csv`) | xgb stride 2.4182, persist 2.4759, ridge 2.6699, clim 3.7499; raw/window stats | level-space (`target` vs `anchor+pred`) |
| Gate (`gate_check.py`) | GATE PASS 13/13; stride RMSE 2.418; g30 traj 1.0947 / direct 1.0764 / persist 1.2183 | reads production artifacts above |
| Track 0 findings (`tracks/track0_and_limitations.md`) | no future-rain source; 2025 coverage tables | record counts / source audit, no error formula |
| Fold windows & row counts | see above | structural |

## VALID — corrected artifacts (supersede all run-1 numbers)

| File | Written | Contents |
|---|---|---|
| `arm_reports.json` | 2026-10-08 10:11 | full 26-arm corrected sweep; **strict result: no arm passes fold-2024 (CI excluding 1) — E0 stands** |
| `eval_2026.json` | 2026-10-08 10:17 | corrected one-shot: E0 0.995 [0.9902, 0.9970] rmse 2.4673; E_best 0.9978; interval coverage 0.898 @ width 3.1267 |
| `stationcv.json` | (corrected re-run) | corrected held-out-station CV, E0 vs E_best — see final report |
| `tracks/final_report.md` | (final) | strict scoreboard, headline, reconciliation, tiers, Track 3 |

The corruption direction matters for interpreting history: the buggy
persistence denominator (`yd − gwl`) is *larger* than the true one for most
rows, so run-1 ratios were pushed **below** 1, making losing configurations
look like winners. All "beats persistence" claims from run-1 are artefacts.


---

<!-- source: tracks/dl_redteam.md -->

# DL Red-Team Verdict (independent audit, 2026-10-10)

## ONE-LINE VERDICT

**NOT ROBUST / NOT PROMOTABLE.** The GRU's stationcv edge over persistence
(0.9348 vs persist) is **reproduced by a feature-less pooled seasonal-delta
forecast** (0.9506; paired station-bootstrap CI [0.974,1.059] on the ratio
seasonal/GRU **includes 1.0**), i.e. the model wires to seasonal shape, not to
transferable skill.  Applying the agreed decision rule, step 2 (controls)
ended in **early-stop → write seasonal-shape finding**.  No promotion; freeze
stays; seasonal-delta is the incumbent to beat before any rework.

---

## 1. Gate scoreboard (original definition, for reference)

- Station-group holdout (district-stratified stations ~75/10/15 per district):
  hold stations=96, rows scored GRU/XGB-E0 = 764
  - persist RMSE 1.725;  XGB-E0 RMSE 1.797 (ref, gating only);  GRU RMSE 1.635
  - ratio: XGB-E0 1.0414 [0.989,1.118];  GRU **0.9480 [0.902,0.986]** → G1 PASS
- Fold 2024: GRU ratio **1.2288** [1.124,1.346] (persist 1.757) — GRU FAILS folds
- Fold 2025: GRU ratio **1.0330** [0.966,1.104] (persist 1.745) — GRU FAILS folds
- Gate 2 (beats persistence on BOTH folds): **FAIL** (both folds > 1).
- 2026 one-shot: GRU ratio 0.9485, coverage90 0.973, width90 **7.74 m** (> 3.1 m
  limit) → Gate 3 **FAIL** (over-covered, over-wide).  MLP ratio 0.9043, cov 0.974,
  width 7.38 m.  2026 used only once for final scoring.

## 2. Controls on the SAME 764-row stationcv holdout (mean of 3 seeds, same paired bootstrap)

| control | ratio vs persist | CI | paired vs GRU | GRU clearly better? |
|---|---|---|---|---|
| (a) pooled seasonal-delta by DOY, **no station/lag/weather info** | **0.9506** | [0.931,0.966] | 1.017 [**0.974,1.059**] | **NO → early stop** |
| (b) E0 XGBoost without st_id/dist_id | 1.0515 | [1.007,1.118] | 1.125 [1.061,1.217] | YES (but irrelevant) |
| (c) GRU, own trailing GWL only (no context) | 1.0103 | [1.002,1.019] | 1.081 [1.040,1.129] | YES (features add value) |
| (d) GRU, shuffled labels (negative) | 1.0151 | [1.010,1.024] | 1.086 [1.043,1.142] | YES (labels matter) |

Reference GRU main (same 3 seeds as the 0.9480 gate): **ratio 0.9348 [0.893,0.973]**
(seed RMSEs 1.644 / 1.625 / 1.672; peak RSS 5.1 GB).

Interpretation:
- **(a)**: seasonal-delty timing alone gets 0.9506 — the GRU's 0.9348 is within the
  paired noise.  Margin over a calendar-only model ≈ 1.6% of persist RMSE,
  NOT significant (paired CI includes 1.0).  This is the decisive control.
- **(c)**: stripping context drops the GRU to ~1.01 — i.e. worst case it only
  recovers the seasonal shape from its own trajectory.  Consistent with (a).
- **(d)**: shuffled labels floor ~1.015 ≈ a near-constant (persist-level) predictor;
  the main GRU's gain over this floor is the seasonal wiring, not memorization
  of station identity (identity is not in the transferable features anyway).

## 3. Leakage audit (summary)
- FTR stats computed on train rows only, NaN-aware (per fold).
- Target-time purge enforced by exp_harness folds/stride (train/val rows require
  target_time < origin; holdout scored on stride windows only).
- `target_d == target - gwl` identity holds (max |err| 5.5e-06) and
  `target_time - time = 720h` exactly on all 2,059,444 finite rows
  (see tracks/persistence_audit.md).
- st_id/dist_id excluded from MLP/GRU transferable schema; no neighbours.
- 2026 trained on < 2025-10-01 only, scored once; early stopping on leak-free val.

## 4. Persistence audit
- tracks/persistence_audit.md (2026-10-10): persistence = delta 0 / level anchor.
  Persist RMSE = 2.480 m on all 3,766 2026-stride rows (511 stations); 1.725 on
  the 764 stationcv holdout rows.  Contained leakage checks (anchor marketing date,
  -9 h as-of offset) all clean.

## 5. Sanity / environment
- production XGB E0 untouched; models/ 14/14 sha-identical; freeze
  `models_frozen_until 2026-11-01` still active.
- train cost ~8-9 min per stage on RTX 3050; peak RSS ~4.8-5.1 GB.

## 6. Decision-rule trace (as-agreed)
1. persistence audit → tracks/persistence_audit.md (done).
2. controls on identical holdout + same bootstrap (done) → control (a) matched GRU
   (paired CI includes 1) → **EARLY-STOP BRANCH, as pre-agreed**.
3. → stop.  Did NOT run district-held-out / calibration refit / numbers stage,
   because the seasonal-shape finding already fails the promotion bar.
   Fold-2024=1.2288 is additionally reported as a hard FAIL (GRU loses to
   persistence on held-out 2024 by 23%).

## Recommendation
Treat the **pooled seasonal-delta-by-DOY** model as the new incumbent baseline.
Any rework must beat it (paired station-bootstrap on a true whole-district
holdout, plus calibrated 90% width ≤ 3.1 m on a time-ordered purged block)
before MLP/GRU can be re-promoted.


---

<!-- source: dl_rebuild/report.md -->

# DL Rebuild (global, no st_id/dist_id) — Gate Results

## Gates (aligned to exp_harness protocol)
- Folds 2024/2025: per-station stride holdouts; train/val time-buffer purged by target_time.
- Station-CV: district-stratified unseen stations (holdout 96 stations) — test group never seen in training; same stride scoring.
- 2026 one-shot: train < 2025-10-01; test stride from 2026-01-01 (511 stations); coverage on 90% empirical band (delta pred).

## Fold results (vs persistence RMSE)
| Fold | MLP (no ID) | MLP CI(5,95) | GRU | GRU CI(5,95) | Persist RMSE |
|---|---|---|---|---|---|
| 2024 | 1.0075 | [0.956,1.052] | 1.2288 | [1.124,1.346] | 1.757 |
| 2025 | 1.1400 | [0.931,1.378] | 1.0330 | [0.966,1.104] | 1.745 |

*MLP narrowly misses 2024 (<1 but not strongly better); GRU weak on 2024 but improves on 2025.*

## Unseen-station CV (new stations, zero-shot)
| Model | Ratio | CI(5,95) | vs XGB-E0 (same protocol) |
|---|---|---|---|
| XGB-E0 (with st_id/dist_id) | 1.0414 | [0.989,1.118] | reference |
| MLP (transferable) | 1.0152 | [0.972,1.059] | beats E0, ratio<1 CI excludes? CI upper 1.059>1 |
| **GRU (transferable)** | **0.9480** | **[0.902,0.986]** | **< 1, CI upper < 1** — passes unseen-station gate |

## 2026 one-shot (production test slice)
| Model | Ratio | 90% cov | 90% width (m) | Stations |
|---|---|---|---|---|
| MLP | 0.9043 | 0.974 | 7.38 | 511 |
| **GRU** | **0.9485** | **0.973** | **7.74** | 511 |

## Verdict
- **New-station reliability:** GRU achieves ratio 0.9480 with CI [0.902, 0.986] on unseen-station CV (beats XGB-E0 and meets ratio<1 with bootstrap CI<1 in upper tail). MLP does not clear the strict CI<1 bar.
- **Transfer design (no identity):** successful zero-shot on held-out stations (this was the XGB failure point).
- **Coverage:** ~97.3% at 90% band (width ~7.7 m) on 2026 stride. This is **over-covered and over-wide vs the 3.1 m limit** — NOT acceptable uncalibrated; would need a time-ordered purged calibration block (see red-team addendum below).
- **Recommendation:** ~~promote GRU~~ **DO NOT promote** — see red-team verdict in `tracks/dl_redteam.md` (2026-10-10). Keep fallback/persistence ramp-up for brand-new stations (<~30d history) as planned. Respect freeze until 2026-11-01T00:00:00+00:00.

## Red-team addendum (independent audit, 2026-10-10)
- Decisive control: a **feature-less pooled seasonal-delta-by-DOY** forecast scores
  **0.9506 [0.931,0.966]** on the very same 764-row stationcv holdout, vs GRU
  0.9348 [0.893,0.973]; paired station-bootstrap ratio **1.017 [0.974,1.059]** —
  CI includes 1 → GRU is **not** significantly better than calendar-only timing.
- GRU fails both time folds (2024: 1.2288; 2025: 1.0330). Gate scoreboard is
  G1 PASS / G2 FAIL / G3 FAIL.
- Controls: (c) own-GWL-only GRU 1.0103, (d) shuffled-labels 1.0151; both confirm
  the residual skill is seasonal wiring, not transferable dynamics.
- Verdict (one line): **NOT ROBUST / NOT PROMOTABLE.** Details + decision-rule
  trace in `tracks/dl_redteam.md`; raw numbers in `result_controls.json`.

**Files:** `result.json`, `folds.log`, `stationcv.log`, `y2026.log`, `report.md` under `data/scratch_exp/dl_rebuild/`.
