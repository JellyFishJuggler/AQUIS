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
