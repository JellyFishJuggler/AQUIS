# Spatial GNN station-influence study — paper-ready summary

All numbers below are copied from `ml/outputs/gnn_influence/eval.json`,
`folds.json`, `influence_meta.json`, `station_influence.json`,
`district_influence.json`, `influence_consistency.json`, `graph.json`, the
training logs, or `ml/research/gnn/README.md`. Nothing here is tuned on the
2026 test period. The production forecast path was not modified.

---

## 1. What it is

This is a **research-only** study of a spatial graph neural network for
groundwater-level (GWL) forecasting and, more importantly, of **which station's
reading moves which other station's forecast**. The model predicts the same
target as the production trajectory GRU — `delta = GWL(t + h·6h) − GWL(t)` with
level = observed anchor + delta, and quantiles `q05 / q50 / q95` trained with
pinball loss — but inserts two graph-attention layers between the GRU encoder
and the quantile head, so neighbouring stations can exchange information. It is
**not a forecasting replacement**: production forecasts continue to come from
the existing GRU (`models/traj_dl.pt`), the promotion gate is unchanged, and the
influence endpoints are off by default. The scientific contribution is the
perturbation-based influence matrix (not attention weights) plus an honest
assessment of whether the graph helps on unseen stations.

---

## 2. Architecture

```
window (120 × 30 features + mask) -> GRU(64) -> LayerNorm
  -> 2 × GATConv(64, heads=2, edge_dim=1, concat=False)
     with residual connection + LayerNorm
  -> [embedding(64) ‖ static row(30) ‖ horizon(3: h, sin h, cos h)]
  -> MLP 64 -> 64 -> 32 -> 3
  -> (q05, q50, q95) of the delta
```

* **Input scaling:** per-column **train-only** z-score, clipped to `±8σ`
  (`CLIP_SIGMA = 8.0`), stored as `feat_mean` / `feat_std` in each checkpoint
  config. This was necessary: raw columns span `rain_30d ≈ 15,120 mm`,
  `pressure ≈ 1,070 hPa`, `year ≈ 2,024`; unscaled inputs saturated the GRU
  gates (14–16 % of hidden units at `|h| > 0.99`) and a 1 m nudge of `gwl`
  changed the embedding by exactly `0.0`. The production GRU is scored
  **unscaled**, exactly as served, so `gnn` vs `gru_matched` is the clean
  architecture comparison and `gnn` vs `gru_production` is the as-served one.
* **Quantile ordering:** the three outputs are sorted (`q05 ≤ q50 ≤ q95`)
  before scoring; there is no explicit crossing penalty in the loss.
* **Loss:** pinball / quantile loss, identical to the production GRU
  (`pipelines.traj_dl.pinball`), masked to valid rows.
* **Optimiser:** AdamW, `lr = 2e-3`, `weight_decay = 1e-4`,
  `ReduceLROnPlateau(factor=0.5, patience=2)`, gradient-norm clip `5.0`.
* **Budget:** `hidden = 64`, `nodes_per_batch = 1024`, `epochs = 30`,
  `patience = 5`, seed `42`. Best epoch 5, best val loss `0.065994`.
* **Parameters:** GNN `45,123`; matched-budget GRU `27,331`.
* **Training time:** `162.8 s` for 10 epochs (GNN) and `160.7 s` for 12 epochs
  (matched GRU) on an RTX 3050 Laptop (4 GB), ≈200 ms/step at ~1,024 nodes.
* **Checkpoints:** `ml/models/research/gnn_traj.pt`,
  `gru_matched.pt`, `gnn_traj_holdout.pt`, `gru_matched_holdout.pt`
  (local only; `*.pt` is gitignored).

---

## 3. Graph

* **Nodes:** 548 stations that have coordinates in
  `data/meta/selected_gwl_stations.csv` and appear on the feature axis.
* **Edges:** k-nearest neighbours (`k = 5`, haversine) **union** all
  same-district pairs, deduplicated, undirected, no self-loops. District names
  compared case-folded and stripped.
* **Counts:** 5,705 edges = 1,699 kNN + 4,006 same-district-only; 37 districts.
* **Degree:** min 5, median 21, max 43 (mean 20.82).
* **Edge feature:** haversine distance in km, standardised with the checkpoint's
  train statistics (mean `31.42 km`, std `21.80 km`; range `0.0–225.98 km`).
* `graph.json` stores nodes (station, district, lat, lon) and edges
  `{i, j, km}`.

---

## 4. Data and protocol

| split | window |
|---|---|
| train | `< 2025-10-01` |
| val | `2025-10-01 .. 2025-12-31` |
| test | `>= 2026-01-01` |

* **Purge rule:** labels that cross a split boundary are removed upstream
  (existing purged splits; no new split logic). `data.load_split` reproduces
  `pipelines.traj_dl._load_split` byte-identically on val.
* **Train size:** 5,230 snapshots / 217,853 anchors.
* **Val size:** 13,743 anchors; 375,037 scored rows for every model.
* **2026 test:** anchors built in memory from `data/features_6h/test.parquet`
  through the existing `_expand` / `_axis_and_targets` helpers, stride 4 (one
  anchor per day), horizons 4/28/60/120 only; 431,575 scored rows, 511 stations.
  No production feature file is written.
* **Held-out-station protocol:** one spatial k-means block (block 1, **103 of
  548 stations**) removed from training for **both** models; early stopping on
  seen stations only; scored on val restricted to the unseen block (111,662
  rows). Single-fold result is reported below; the 5-fold × 3-seed repeat is the
  main spatial result.
* **Leakage control:** the 2026 test is used only for scoring; the checkpoint is
  the one early-stopped on val. Nothing is tuned on test. Influence is measured
  on val + train timestamps only, never on test.

---

## 5. Results

### (a) Val — temporal comparison (same rows for all models)

| horizon | GNN RMSE | GRU production RMSE | GRU matched RMSE | GNN pinball | prod pinball | matched pinball |
|---|---|---|---|---|---|---|
| 1 d | 0.2100 | 0.2119 | 0.2090 | | | |
| 7 d | 0.3511 | 0.3516 | 0.3521 | | | |
| 15 d | 0.5405 | 0.5423 | 0.5377 | | | |
| 30 d | 0.9527 | 0.9511 | 0.9532 | | | |
| all | | | | 0.065994 | 0.068294 | 0.066101 |

Paired station bootstrap, ratio `RMSE(gnn)/RMSE(gru)` (2,000 resamples):

| horizon | vs production GRU | vs matched GRU |
|---|---|---|
| 1 d | 0.9911 [0.9833, 0.9989] | 1.0044 [1.0007, 1.0110] |
| 7 d | 0.9966 [0.9581, 1.0270] | 0.9965 [0.9877, 1.0084] |
| 15 d | 0.9969 [0.9651, 1.0329] | 1.0041 [0.9784, 1.0194] |
| 30 d | 1.0043 [0.9814, 1.0829] | 0.9992 [0.9709, 1.0254] |

**Read:** on val the GNN is statistically indistinguishable from both GRUs.
There is no temporal-accuracy gain.

### (b) 2026 test — single run (checkpoint early-stopped on val)

| horizon | GNN RMSE | matched GRU RMSE | production GRU RMSE | ratio vs matched [95% CI] | ratio vs production [95% CI] |
|---|---|---|---|---|---|
| 1 d | 0.9125 | 0.9077 | 0.8863 | 1.0050 [1.0014, 1.0106] | 1.0281 [1.0149, 1.0599] |
| 7 d | 1.4123 | 1.4044 | 1.3852 | 1.0056 [1.0037, 1.0103] | 1.0188 [1.0075, 1.0375] |
| 15 d | 1.7375 | 1.7252 | 1.7119 | 1.0073 [1.0047, 1.0128] | 1.0143 [1.0030, 1.0297] |
| 30 d | 2.3690 | 2.3456 | 2.3467 | 1.0104 [1.0062, 1.0189] | 1.0093 [0.9996, 1.0176] |
| all | pinball 0.125001 | 0.123230 | 0.125606 | | |

Rows: 124,699 / 114,495 / 104,540 / 87,841; 511 stations. The matched-GRU CI
excludes 1 at every horizon (`p_a_better` = 0.0005, 0.0, 0.0, 0.0): the GNN is
**0.5–1.0 % worse** than the matched GRU on the untouched 2026 test. It is
1.0–2.8 % worse than the production GRU as served.

### (c) 5 spatial folds × 3 seeds — held-out stations (val only)

30 runs; block seed 42; training seeds 42/43/44; 2,000-resample nested
bootstrap; epochs 20, patience 4, 1,024 nodes. Fold sizes 145 / 103 / 115 / 124
/ 61 stations (548 total).

| horizon | RMSE GNN | RMSE GRU | ratio mean ± std (15 runs) | std of fold means | 95% CI | P(GNN better) |
|---|---|---|---|---|---|---|
| 1 d | 0.2107 | 0.2121 | 0.9913 ± 0.0116 | 0.0066 | [0.9825, 0.9985] | 0.994 |
| 7 d | 0.3561 | 0.3555 | 0.9963 ± 0.0358 | 0.0168 | [0.9865, 1.0088] | 0.7865 |
| 15 d | 0.5616 | 0.5607 | 0.9778 ± 0.0928 | 0.0598 | [0.9572, 0.9953] | 0.993 |
| 30 d | 0.9923 | 0.9952 | 0.9746 ± 0.0917 | 0.0305 | [0.9139, 1.0006] | 0.9695 |

Pinball (all horizons pooled per run): `0.9852 ± 0.0592`.

Per-fold mean ratios over the 3 seeds:

| fold | unseen stations | 1 d | 7 d | 15 d | 30 d |
|---|---|---|---|---|---|
| 0 | 145 | 0.9835 | 0.9998 | 1.0004 | 1.0054 |
| 1 | 103 | 1.0005 | 1.0092 | 1.0214 | 0.9609 |
| 2 | 115 | 0.9953 | 0.9814 | 0.9686 | 0.9432 |
| 3 | 124 | 0.9890 | 1.0147 | 1.0207 | 1.0090 |
| 4 | 61 | 0.9883 | 0.9765 | 0.8779 | 0.9544 |

Per-fold CIs are in `folds.json`; the 15 d fold-4 CI is
`[0.7861, 0.9524]` on 61 stations.

### (d) Per-fold winners and the honest conclusion

* Folds **2** and **4** have mean ratio `< 1` at all four horizons. Fold 4 has
  the smallest held-out set (61 stations) and supplies most of the 15 d gain
  (`0.8779 [0.7861, 0.9524]`).
* Folds **1** and **3** are `> 1` at 7 d and 15 d; fold **0** is `> 1` at 15 d
  and 30 d.
* The pooled mean favours the GNN slightly at every horizon and the nested CI
  excludes 1 at 1 d and 15 d — but the **sign flips by fold** and the
  fold-to-fold spread (std of fold means `0.0598` at 15 d) is the same order as
  the effect.

**Honest conclusion:** the graph carries usable spatial information for stations
the model has never seen at mid horizons, but the improvement is
**fold-dependent, not reliable**, and it does **not** survive as a served-model
replacement: on the untouched 2026 test the GNN is 0.5–1.0 % worse than the
matched GRU. The correct claim is “worth revisiting for stations with little
history”, not “beats the production model”.

Single-block held-out check (block 1, 103 unseen stations):
7 d `0.2826` vs `0.2863` (ratio `0.9870 [0.9699, 1.0005]`), 15 d `0.3410` vs
`0.3586` (`0.9510 [0.9173, 0.9751]`), 30 d `0.3567` vs `0.3551`
(`1.0027 [0.9394, 1.0915]`), pinball `0.054262` vs `0.055826`.

---

## 6. Influence method

For a sampled timestamp, each station's own recent input is nudged by `±m` on
the `gwl` feature (last row of its 120-step window and the matching static
entry), with `m ∈ {0.05, 0.10, 0.25}` m. The central difference of every other
station's predicted `q50` gives

```
sensitivity[i, j] = median_m ( q50_j(+m) − q50_j(−m) ) / (2m)
```

* **No-edge ablation:** the same computation is repeated with all graph edges
  removed (self-loops only, identical weights) and subtracted. The ablation is a
  pure GRU with no cross-station path, so its off-diagonal response is exactly
  `0`; every reported off-diagonal value is therefore **graph-only**.
* **Aggregation:** per-timestamp matrices are averaged by **median**; district
  values are the **signed mean** over station pairs (most pairs are exactly 0,
  so a median would collapse the district matrix).
* **Derived-feature consistency:** perturbing `GWL(t)` must also move the
  rolling inputs that contain that row. The audit shows:
  `lag1/lag4/lag8/lag28/lag120` do **not** move (they are `GWL(t−k)`);
  `gwl_roll7_mean/std` and `gwl_roll30_mean/std` **do** move; deviations,
  slopes, observation flags and the imputation flag are not in the frozen
  `NUM_COLS` and are never seen by the model. The update is **analytic, not a
  replay**: the mean shifts by `δ/n`, the variance is rebuilt from
  `Σx² − n·mean²` with one element replaced, and `n7`/`n30` come from the axis
  timestamps because the stored windows reset across observation gaps.
  Replaying the axis windows reproduces the stored columns to
  `2.09e-6 m` max (mean `1.56e-7 m`, 64 anchors); median window counts are 6,
  max 14, and 6/64 anchors have a single 30 d observation.
* **Lag audit:** `lag1` verified on 58/64 anchors (`|Δ| ≤ 1.23e-6 m`), `lag4`
  52/64, `lag8` 16/64; `lag28`/`lag120` are structurally invariant but cannot be
  verified this way (break purges remove rows after feature construction) and
  are reported as skipped, not mismatched.
* **Effect of the consistency fix:** on 4 val snapshots, comparing raw-only vs
  consistent perturbation:

| h | snapshots | rel. Frobenius ‖raw−consistent‖/‖raw‖ | max abs Δ | corr. off-diagonal | top-10 source overlap |
|---|---|---|---|---|---|
| 28 (7 d) | 2 | 17.3 – 30.7 | `2.4e-2` | `−0.62 … −0.15` | 3–8 / 10 |
| 120 (30 d) | 1 | 9.16 | `1.6e-2` | `0.786` | 10 / 10 |

  The relative size exceeds 1 because the derived-feature response dominates the
  raw-`gwl` response. At 7 d the ranking changes materially (only 3 of the top
  10 sources survive; off-diagonal correlation goes negative); at 30 d the
  pattern is similar but rescaled. Published artifacts use
  `--consistent-derived`; raw-only numbers are preserved in
  `influence_consistency.json`. Consistent magnitudes are roughly 4–16× the
  raw-only ones.
* **Linearity check:** per-metre sensitivities from 0.05 m and 5 m perturbations
  agree with correlation `1.000` (README), so the magnitudes are not numerical
  noise.
* **Horizons:** 7 d (`h = 28`) and 30 d (`h = 120`).
* **Coverage:** 499 of 548 stations appear in at least one scored snapshot;
  481 (7 d) / 471 (30 d) carry at least one finite link. Stations with no
  measurement are **omitted** from `station_influence.json`, not published as
  zeros.
* **Snapshots:** 40 sampled timestamps = 24 val + 16 train; scored counts
  `{28: 29, 120: 23}`; the consistency report uses **4 val snapshots** only.

---

## 7. Influence findings (final consistent numbers)

* **Strongest station link:** `7.38e-3` per metre at 7 d and `9.35e-3` per
  metre at 30 d.
* **Mean |sensitivity| over all published links:** `7.90e-5` per metre at 7 d,
  `2.18e-4` per metre at 30 d.
* **Same-district vs cross-district** (full-matrix means, README):
  `1.2e-4` vs `1.5e-5` per metre at 7 d (**8.1×**) and `3.4e-4` vs `2.6e-5` at
  30 d (**13.1×**). The published JSON stores only the top 10 links per
  direction; recomputing from that truncated file gives `1.17e-4` vs
  `1.19e-5` (9.9×) at 7 d and `3.35e-4` vs `2.10e-5` (15.9×) at 30 d.
* **District matrix:** diagonal max `5.24e-4` (MAINPURI) at 7 d and `3.72e-3`
  (KAUSHAMBI) at 30 d. Largest off-diagonal pairs:
  PRATAPGARH → KAUSHAMBI `+3.51e-4` (7 d) / `−5.40e-4` (30 d);
  MAINPURI → BAREILLY `−5.25e-5` (7 d) / `−2.39e-4` (30 d);
  ALIGARH → MATHURA `+3.16e-5` (7 d) / `+5.49e-5` (30 d).
* **Example station — “Panchayat Bhawan, Dohri” (MIRZAPUR), 7 d:**
  `strength_in = 0.010537`, `strength_out = 0.010750`.
  *Influenced by:* Primary School Harinahpur `−2.7950e-3`, Primary School
  Biraura `+1.9185e-3`, Primary School Todarpur `−1.9918e-4`, Madhyamik
  Vidhyalaya Dulha Daul `+1.6376e-4`, Primary School Koilar `+8.7644e-5`,
  Primary School Babura `−8.5771e-5`.
  *Influences:* Primary School Babura `−1.9703e-3`, Purv Madhyamik Vidyalaya
  Bharpura `−1.7638e-3`, Primary School Todarpur `−4.5304e-4`, Primary School
  Harinahpur `−3.8675e-4`, Primary School Biraura `−3.6663e-4`, Punchayat Bhawan
  Vijaypur `+3.1506e-4`.
  At 30 d the same station: `strength_in = 0.002940`, `strength_out = 0.003417`;
  top incoming Madhyamik Vidhyalaya Dulha Daul `+4.2643e-4`, top outgoing
  Madhyamik Vidhyalaya Dulha Daul `+3.5496e-4`.
* **Units:** every sensitivity is **per metre of source GWL input**. The
  perturbation is applied internally as `m / std(gwl)` because the model sees
  standardised features.
* **Signs:** both positive and negative links occur; a negative value means the
  target station's median forecast falls when the source station's GWL rises.
  Signs are not stable across horizons (e.g. PRATAPGARH → KAUSHAMBI flips from
  positive at 7 d to negative at 30 d).

---

## 8. Serving

Flask (`ml/api.py`), all read-only JSON:

```
GET /influence/meta
GET /influence/graph
GET /influence/station/<path:name>?top=5     (1..25)
GET /influence/district/<path:name>?top=10   (1..40)
GET /influence/consistency
```

plus identical `/api/influence/...` aliases. The loader
(`ml/serving/influence.py`) imports **no torch and no torch_geometric** (verified:
`torch` is not in `sys.modules` before or after a request), caches by file
mtime, and never raises.

* **Flag:** `data/refresh/runtime.json` → `"influence_enabled": true`. Absent,
  `false`, or unreadable ⇒ off. The feature only reads this file; it never
  writes it.
* **Fallback contract:** every route returns HTTP 200 with
  `{"available": false, "reason": "..."}` when the flag is off
  (`influence_disabled`) or the artifact is missing/corrupt
  (`meta_missing`, `graph_missing`, `station_missing`, `station_not_found`,
  `district_missing`, `district_not_found`, `consistency_missing`). No 500 is
  surfaced to forecast clients.
* **Artifacts:** `ml/outputs/gnn_influence/{influence_meta,graph,
  station_influence,district_influence,influence_consistency}.json` plus
  `district_influence_h28.png` and `district_influence_h120.png`.
* **UI status:** a Streamlit expander exists in `ml/app_pages/map_view.py`
  (hidden unless the flag is on). There is **no Node/Express proxy for
  `/consistency`** in this commit and **no React Native app** in this
  repository. The Next.js front-end (`front-end/app`, `front-end/components`)
  contains **no influence screen, no station “Most influenced by / Most
  influences” panel, no map edge overlay, no district view, no Experimental
  badge, and no mocked influence data**. The Flask endpoints are the only
  serving surface added here.
* **Isolation:** `data/refresh/runtime.json` SHA-256 is unchanged
  (`2EFFE19416EDB9F4A10523753943B715E814D2FC97B12EF2C05ADC9D9F3DE0F0`);
  the full suite is 254/254 OK and `python -m checks.gate_check` is 13/13 PASS.

---

## 9. Limitations and wording rules

* **Sensitivity, not causation.** Perturbation measures how the trained model
  responds, not a physical cause. The exact caveat stored verbatim in
  `influence_meta.json` is:
  > Experimental. Shows model sensitivity between stations, not proven causation.
  > Pumping and local geology are not observed.
* **No pumping / abstraction data** and no local geology are observed; both are
  plausible confounders of any real inter-station effect.
* **Weak spatial clustering in the residuals:** Moran's I = `0.042`
  (p = 0.051) on spatial-CV residuals (`docs/ml.md`); a separate XGBoost
  test-residual run gives `0.074` (p = 0.02). Either way the residuals show no
  strong spatial structure, so the graph is not exploiting a large missed
  spatial signal.
* **Same-district strength is partly by construction:** 4,006 of the 5,705
  edges are same-district-only edges. Stronger same-district links are therefore
  expected even if geography alone drove the effect; they are not independent
  evidence of district-level physics.
* **Small absolute magnitudes:** the strongest station link is `9.35e-3` m of
  target median per metre of source input. Do not describe these as large
  physical effects.
* **No served-horizon gain:** on val the GNN ties the GRU; on the untouched 2026
  test it is 0.5–1.0 % worse than the matched GRU and 1.0–2.8 % worse than the
  production GRU. It must not be presented as an accuracy improvement.
* **Fold-dependent spatial result:** the 5-fold mean favours the GNN, but the
  sign flips by fold and the 15 d gain is carried by the smallest fold
  (61 stations). Call it “encouraging / worth revisiting”, not “reliable”.
* **Coverage is partial:** influence is measured at 40 sampled timestamps
  (24 val + 16 train), not all timestamps; 499/548 stations are covered, and
  the consistency audit uses only 4 val snapshots. The test period is never used
  for influence.
* **Artifacts are regenerated / dated:** `eval.json` 2026-11-18 19:45,
  `folds.json` 2026-11-18 21:49, consistency report 2026-11-18 21:14, influence
  artifacts 2026-11-18 21:58 (all after the upstream label-time purge; the val
  window's last days carry no 7 d / 30 d label and those snapshots score
  nothing). The four research checkpoints were trained on 2026-11-18 and are
  local-only; `ml/models/research/` is gitignored, so reproducing from scratch
  requires retraining.
* **Production untouched:** no production model, refresh loop, promotion gate,
  or `runtime.json` value was changed. The flag is off by default and the
  endpoints only read JSON.

---

## 10. Reproduce

From `ml/` with the project venv (`ml/.venv/Scripts/python.exe` on Windows):

```bash
python -m research.gnn.graph
python -m research.gnn.data
python -m research.gnn.train
python -m research.gnn.train --arch gru --out models/research/gru_matched.pt
python -m research.gnn.train --holdout-block 1 --out models/research/gnn_traj_holdout.pt
python -m research.gnn.train --arch gru --holdout-block 1 \
       --out models/research/gru_matched_holdout.pt
python -m research.gnn.eval --test --holdout
python -m research.gnn.folds
python -m research.gnn.influence --consistent-derived
python -m research.gnn.influence --check-derived 4
```

Outputs: `ml/outputs/gnn_influence/` (`eval.json`, `eval.md`, `folds.json`,
`folds.md`, `graph.json`, `influence_meta.json`, `station_influence.json`,
`district_influence.json`, `influence_consistency.json`, two heatmaps).
Fold cache: `ml/outputs/gnn_influence/folds/runs/` (gitignored). Tests:

```bash
python -m unittest discover -s tests
python -m checks.gate_check
```

`python -m research.gnn.train` refuses to write outside `models/research/`.

---

## 11. Suggested LaTeX-ready claims

**Safe to state**

1. We add a research-only spatial GNN that predicts the same quantile deltas as
   the production trajectory GRU and inserts two graph-attention layers over a
   548-station, 5,705-edge graph built from 5-nearest-neighbour haversine links
   and same-district pairs.
2. On the val window the GNN is statistically indistinguishable from the
   production and matched GRUs (30 d RMSE `0.9527` vs `0.9511` / `0.9532`;
   pinball `0.0660` vs `0.0683` / `0.0661`).
3. On the untouched 2026 test the GNN is slightly worse than the matched GRU
   (30 d RMSE `2.3690` vs `2.3456`; ratio `1.0104 [1.0062, 1.0189]`), so the
   graph does not improve temporal generalisation.
4. Over 5 spatial folds × 3 seeds, the held-out-station mean RMSE ratio is
   `0.9913 / 0.9963 / 0.9778 / 0.9746` at 1/7/15/30 d, but the sign flips by
   fold and the 15 d gain is carried by the 61-station fold; the effect is
   fold-dependent.
5. Perturbation-based influence (with the no-edge ablation subtracted and
   rolling GWL features updated consistently) is local and grows with horizon:
   the strongest station link is `7.38e-3` per metre at 7 d and `9.35e-3` at
   30 d, and same-district links are about 8–13× stronger than cross-district
   links.

**Must NOT claim**

1. That the GNN replaces, improves, or should replace the production forecast —
   it is 0.5–1.0 % worse than the matched GRU on 2026 test and was never
   promoted.
2. That the influence values are causal, physical, or evidence about pumping —
   they are model sensitivities only; pumping and local geology are unobserved.
3. That same-district influence proves district-level hydrogeology — the graph
   deliberately contains all same-district pairs, so that strength is partly
   built in.
4. That the held-out-station gain is a reliable generalisation result — it is
   fold-dependent, and the 15 d mean is dominated by the smallest fold.
5. That the influence matrices cover every station or the test period — coverage
   is 499/548 stations from 40 sampled timestamps (24 val + 16 train), and the
   consistency audit uses 4 val snapshots; the 2026 test is never used for
   influence.
