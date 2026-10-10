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