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
