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
