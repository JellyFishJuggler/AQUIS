# Research: spatial GNN ("station influence")

**Research only. Nothing here is promoted, and nothing here is used to produce a
served forecast.** The production path (`ml/api.py` forecast routes,
`ml/serving/`, `ml/refresh/`, `models/traj_dl.pt`, `models/pooled_dl.pt`, the XGB
fallbacks and `data/refresh/runtime.json`) is untouched by this package: it is
not imported by it, and it does not write to it.

---

## What it does

A graph neural network that predicts the same target as the trajectory GRU —
`delta = GWL(t + h·6h) − GWL(t)`, level = observed anchor + delta, quantiles
`q05 / q50 / q95` trained with pinball loss — but with a station graph between
the encoder and the quantile head, so neighbouring stations can exchange
information. The interesting question is not the accuracy (it is roughly on par
with the production GRU) but **which station moves which other station's
forecast**, measured by perturbation, not by attention weights.

## Files

| path | role |
| --- | --- |
| `graph.py` | station graph → `outputs/gnn_influence/graph.json` |
| `data.py` | snapshot batching over the **existing** purged splits (no new split logic) |
| `model.py` | fresh trajectory-GRU encoder + 2 `GATConv` + horizon-conditioned quantile head |
| `train.py` | `python -m research.gnn.train` (GNN or matched-budget GRU) |
| `eval.py` | `python -m research.gnn.eval` → `eval.json` / `eval.md` |
| `folds.py` | `python -m research.gnn.folds` → 5 folds × 3 seeds → `folds.json` / `folds.md` |
| `influence.py` | `python -m research.gnn.influence` → influence artifacts + heatmaps |

Checkpoints and configs go to `ml/models/research/` only (`gnn_*.pt`,
`*_config.json`, fold runs under `folds/`). Artifacts go to
`ml/outputs/gnn_influence/` (fold run cache under `folds/runs/`).

## Graph

* nodes = the 548 stations that have coordinates in
  `data/meta/selected_gwl_stations.csv` and appear on the feature axis;
* edges = kNN (`k = 5`, haversine) **union** same-district pairs, deduplicated,
  no self-loops (district names compared case-folded — the feature tables say
  `KANSIRAM NAGAR`, the CSV says `Kansiram Nagar`);
* one edge feature: haversine distance in km, standardised with the mean/std
  stored in the checkpoint config.

`graph.json` → 548 nodes, 5,705 edges (1,699 kNN + 4,006 same-district-only),
37 districts, degree min 5 / median 21 / max 43.

## Data and splits

No new split logic. `data.load_split` reads `data/features_traj/{train,val}.parquet`
and reproduces `pipelines.traj_dl._load_split` **byte-identically**
(`python -m research.gnn.data` asserts `a_axis / a_runstart / a_static /
a_static_m / ptr / h / delta` are equal for `val`).

| split | window |
| --- | --- |
| train | `< 2025-10-01` |
| val | `2025-10-01 .. 2025-12-31` |
| test | `>= 2026-01-01` |

Labels crossing a boundary are already purged upstream. The 2026 test anchors
used by `eval.py --test` are built **in memory** from
`data/features_6h/test.parquet` through the existing
`pipelines.traj._expand` / `_axis_and_targets` helpers (default stride 4 = one
anchor per day, horizons 4/28/60/120 only). No production feature file is
written or overwritten.

## Model

```
window (120 × 31 features + mask)  ->  GRU(64) -> LayerNorm        # fresh weights
   -> 2 × GATConv(64, heads=2, edge_dim=1) with residual + LayerNorm
   -> [embedding ‖ static row ‖ (h, sin h, cos h)]  -> MLP  ->  (q05, q50, q95)
```

* the encoder has the **same shape** as the production `TrajGRU` but is
  instantiated from scratch — `models/traj_dl.pt` is never loaded;
* one graph snapshot per anchor timestamp; a station missing at that timestamp
  is simply absent from that snapshot's node set (that is the masking);
* 45,123 trainable parameters; ~200 ms/step at ~1,024 nodes on an RTX 3050
  Laptop (4 GB);
* fixed seed (`42`), AdamW, `ReduceLROnPlateau`, early stopping on val pinball.

### Input scaling (and why it was necessary)

The raw feature columns are on wildly different scales — `rain_30d` reaches
15,120 mm, `pressure` ~1,070 hPa, `year` ~2,024. Fed straight into a GRU, those
columns saturate the gates: 14–16 % of hidden units sit at `|h| > 0.99`, and the
encoder output becomes almost independent of its input. The diagnostic signature
was exact — nudging `gwl` by 1 m changed the encoded embedding by **exactly
`0.0`**, so every graph-derived sensitivity was `0`.

The research models therefore standardise with **train-only** statistics
(per-column mean/std over the train axis, clipped to `±8σ`, `CLIP_SIGMA = 8.0`),
stored as `feat_mean` / `feat_std` in each checkpoint config so eval and
influence reproduce the same transform. The influence perturbation is applied as
`m / std(gwl)` internally, so reported values stay **per metre**.

`gru_production` is *not* scaled — it is scored exactly as it is served. That
means:

* `gnn` vs `gru_matched` is the clean architecture comparison (identical
  preprocessing and budget);
* `gnn` vs `gru_production` is the "as served" comparison, where the production
  GRU sees the raw features it was trained on.

Legacy checkpoints without the stats fall back to unscaled inputs with a warning.

## Evaluation

`eval.py` scores **identical rows** for every model (it raises if they diverge),
and reports RMSE, MAE, bias, direction accuracy, calibrated 90 % coverage +
half-width and median station NSE at 1 d / 7 d / 15 d / 30 d, plus the pinball
loss. Calibration widening is fitted on half the rows and reported on the other
half. Paired station-level bootstrap (2,000 resamples) gives a CI on the RMSE
ratio.

Models compared:

* `gnn` — this model;
* `gru_production` — the real `models/traj_dl.pt`, read-only;
* `gru_matched` — a fresh `TrajGRU` trained by the same loop with the same
  budget, so architecture is the only difference.

Held-out-station test: one spatial k-means block (~19 % of stations) is removed
from training for **both** the GNN and the matched GRU, early stopping uses seen
stations only, and both are scored on the unseen block. That is the honest
spatial-generalisation number; the temporal val number is not.

Nothing is tuned on the test period — the checkpoint is the one early-stopped on
val.

### Multi-fold held-out comparison

A single block is one draw, so `folds.py` repeats the held-out-station protocol
over **5 spatial k-means folds × 3 training seeds** (`42, 43, 44`):

* the block assignment is computed **once** with a fixed `--block-seed 42`, so a
  fold is the same station set for every training seed — the seed only changes
  the training run;
* for each fold the held-out stations are removed from **both** models' training
  data and the matched GRU is retrained on the identical subset, with early
  stopping on seen stations only;
* scoring is the val split restricted to that fold's held-out stations (no test
  period, nothing tuned).

Per horizon it reports the row-pooled `ratio = RMSE(gnn) / RMSE(gru)` for every
(fold, seed) run, the mean ± std across all 15 runs, the std of the 5 fold means,
and a station-level bootstrap CI (2,000 resamples, seeded) where each fold is
resampled inside itself and the fold ratios are averaged — a nested bootstrap,
so the CI reflects both station sampling and fold-to-fold spread.

Fold sizes are 145 / 103 / 115 / 124 / 61 stations (548 total). Results
(`folds.md`):

| h | RMSE gnn | RMSE gru | ratio mean ± std (15 runs) | std of fold means | 95% CI | P(gnn better) |
|---|---|---|---|---|---|---|
| 1 d | 0.2107 | 0.2121 | 0.9913 ± 0.0116 | 0.0066 | [0.9825, 0.9985] | 0.99 |
| 7 d | 0.3561 | 0.3555 | 0.9963 ± 0.0358 | 0.0168 | [0.9865, 1.0088] | 0.79 |
| 15 d | 0.5616 | 0.5607 | 0.9778 ± 0.0928 | 0.0598 | [0.9572, 0.9953] | 0.99 |
| 30 d | 0.9923 | 0.9952 | 0.9746 ± 0.0917 | 0.0305 | [0.9139, 1.0006] | 0.97 |

Pinball (all horizons pooled per run) `gnn/gru = 0.9852 ± 0.0592`.

Per fold (`ratio mean ± std` over the 3 seeds):

| fold | unseen | 1 d | 7 d | 15 d | 30 d |
|---|---|---|---|---|---|
| 0 | 145 | 0.9835 ± 0.0166 | 0.9998 ± 0.0058 | 1.0004 ± 0.0137 | 1.0054 ± 0.0072 |
| 1 | 103 | 1.0005 ± 0.0052 | 1.0092 ± 0.0310 | 1.0214 ± 0.0211 | 0.9609 ± 0.1517 |
| 2 | 115 | 0.9953 ± 0.0138 | 0.9814 ± 0.0413 | 0.9686 ± 0.0042 | 0.9432 ± 0.0122 |
| 3 | 124 | 0.9890 ± 0.0062 | 1.0147 ± 0.0466 | 1.0207 ± 0.1437 | 1.0090 ± 0.1349 |
| 4 | 61 | 0.9883 ± 0.0119 | 0.9765 ± 0.0489 | 0.8779 ± 0.1323 | 0.9544 ± 0.1088 |

**Reading it honestly.** The average favours the GNN slightly at every horizon,
and the nested CI excludes 1 at 1 d and 15 d — but the sign flips by fold: only
fold 2 wins at all four horizons, folds 1 and 3 lose at 7 d and 15 d, and the
overall 15 d win is largely fold 4 (`0.878 [0.786, 0.952]` on 61 stations).
Fold-to-fold spread (std of fold means `0.060` at 15 d) is the same order as the
effect, so this is a *fold-dependent* improvement, not a reliable one. It also
does not contradict the single-fold 2026-test result above (ratio `1.005–1.010`,
GNN slightly worse): the folds are scored on val only, and nothing here is tuned
on test.

## Influence (perturbation, not attention)

For a sampled set of timestamps, each station's own recent input is nudged by
`±m` on the `gwl` feature (last row of its window + matching static entry),
with `m ∈ {0.05, 0.10, 0.25}` m. The central difference of every other station's
predicted `q50` gives

```
sensitivity[i, j] = median_m ( q50_j(+m) − q50_j(−m) ) / (2m)
```

The same computation is repeated with **all graph edges removed** (self-loops
only, identical weights) and subtracted, so what is reported is only the
influence that exists *because of the graph*. Per-timestamp matrices are
averaged by median; district values are the **signed mean** over station pairs
(the effect is sparse — most pairs are exactly `0`, so a median would collapse
the whole district matrix to zero).

Because the no-edge ablation is a pure GRU with no cross-station path, its
off-diagonal entries are exactly `0` — so the reported off-diagonal matrix *is*
the graph contribution, and a non-zero off-diagonal entry is direct evidence
that information moved between stations. Magnitudes are per metre of source
input (`m / std(gwl)` is applied internally because the model sees standardised
features).

Outputs: `station_influence.json`, `district_influence.json`,
`influence_meta.json`, `district_influence_h28.png`,
`district_influence_h120.png`.

> **Caveat (also stored verbatim in `influence_meta.json`):**
> Experimental. Shows model sensitivity between stations, not proven causation.
> Pumping and local geology are not observed.

### Snapshot sampling

Influence is measured at 24 val + 16 train timestamps. Reporting is bursty: a
timestamp can carry 1 station or 250, and the dense ones mostly cover the same
~250 stations, so *time diversity*, not snapshot size, drives coverage.

The sampler keeps the timestamps at or above the median snapshot size and then
spreads the picks evenly over time (deterministic, no seed). Measured
alternatives on this dataset:

| strategy | stations covered |
| --- | --- |
| evenly over all timestamps | 484 / 548 |
| **evenly over the better-populated half (used)** | **499 / 548** |
| same, after dropping timestamps with no valid target | 462 / 548 |
| one (largest) snapshot per equal-time bin | 406 / 548 |
| greedy set-cover over stations | 278 / 548 |

The last days of the val window carry no 7 d / 30 d label (labels crossing into
test are purged), so 7 of the 40 sampled snapshots score nothing. Pre-filtering
them looks tidy but changes the even spacing onto a worse set of timestamps
(462/548), so they are kept: a snapshot with no valid target is detected before
any forward pass and skips in ~0 s. `influence_meta.json` reports both
`snapshots` and `snapshots_scored`.

### Derived-feature consistency

The perturbation above moves `GWL(t)` on the anchor row. Four model inputs are
functions of that row, so a *consistent* perturbation has to move them too. The
audit (`feature_audit` in `influence_consistency.json`) classifies every derived
GWL column:

| column group | in `NUM_COLS`? | moves with `GWL(t)`? |
| --- | --- | --- |
| `lag1/lag4/lag8/lag28/lag120` | yes | no — they are `GWL(t−k)` |
| `gwl_roll7_mean/std`, `gwl_roll30_mean/std` | yes | yes — the window contains `t` |
| `gwl_dev7/30`, `gwl_slope7/30/90`, `gwl_obs*`, `gwl_imputed` | no | never seen by the model |

So exactly two feature *groups* need updating, and only the rolling one does
anything. `lag_audit` confirms the lag claim empirically on the anchors whose
trailing `k` axis rows are a complete 6 h run: `lag1` 58/64 anchors
(`|Δ| ≤ 1.3e-6` m), `lag4` 52/64, `lag8` 16/64; `lag28`/`lag120` cannot be
verified this way because break purges and observation gaps remove rows from the
axis *after* the features were built. Those anchors are reported as skipped
instead of being counted as mismatches, and the invariance is structural anyway
(`groupby(...).shift(k)`, never a window).

The update is **analytic, not a replay**. The stored features were built with
time-based `7D`/`30D` windows on a superset of the filtered axis, so recomputing
them from `seq` would move the *unperturbed* base by up to a metre. Instead
`roll_delta` applies the exact single-element effect: the mean shifts by
`δ/n` and the variance is rebuilt from `Σx² − n·mean²` with one element replaced.
The window counts `n7`/`n30` come from the axis timestamps, because the stored
windows reset across observation gaps. Replaying the axis windows reproduces the
stored columns to `2.1e-6` m (mean `1.6e-7` m, 64 anchors), which is the residual
of those dropped rows — small enough to confirm the window definition, not small
enough to replace the stored values. Window counts are small because NWIC GWL
readings are sparse: median 6 observations per window, max 14, and 6 of 64
anchors have a single 30 d observation.

For a `+5 cm` step with `n7 = n30 = 14`: `roll7_mean` `+3.6 mm`,
`roll7_std` `+8.2 mm`, `roll30_mean` `+3.6 mm`, `roll30_std` `+8.2 mm`. The
std moves more than the mean because a single new observation changes the
spread of a 14-point window far more than it changes its level.

**Does it change the answer? Yes, materially.** On 4 val snapshots
(`--check-derived 4`), comparing graph-only sensitivity with and without the
consistent update:

| h | snapshots | rel. Frobenius ‖raw−consistent‖/‖raw‖ | max abs Δ | corr. off-diagonal | top-10 source overlap |
| --- | --- | --- | --- | --- | --- |
| 28 | 2 | 17.3 – 30.7 | `2.4e-2` | `−0.62` … `−0.15` | 3 – 8 / 10 |
| 120 | 1 | 9.16 | `1.6e-2` | `0.78` | 10 / 10 |

The relative size exceeds 1 because the derived-feature response dominates the
raw-`gwl`-only response — the model leans on the rolling std as much as on the
level. At 7 d the *ranking* changes (only 3 of the top 10 sources survive, and
the off-diagonal correlation goes negative), while at 30 d the pattern is
similar but rescaled. The published artifacts are therefore generated with
`--consistent-derived`; the raw-only numbers are kept in
`influence_consistency.json` for comparison. The structure of the answer is
unchanged (local, horizon-growing, graph-only), but the magnitudes are roughly
4–16× larger.

## Findings

**Accuracy — no temporal gain.** On val the GNN is indistinguishable from the
production GRU and from the matched-budget GRU (30 d RMSE `0.9527 / 0.9511 /
0.9532`, pinball `0.0660 / 0.0683 / 0.0661`; gnn vs matched bootstrap ratio
`0.999 [0.971, 1.025]`). On the untouched 2026 test it is slightly *worse* than
the matched GRU: 30 d RMSE `2.3690` vs `2.3456`, pinball `0.1250` vs `0.1232`,
bootstrap ratio `1.005–1.010` with the CI excluding 1 at every horizon. So the
graph does not improve temporal generalisation — it costs ~0.5–1 %.

**Accuracy — a real gain on unseen stations, but fold-dependent.** With a
spatial k-means block (103 of 548 stations) removed from training for both
models, the GNN wins where neighbours matter most: 7 d RMSE `0.2826` vs `0.2863`
(ratio `0.987 [0.970, 1.001]`), 15 d `0.3410` vs `0.3586` (ratio `0.951
[0.917, 0.975]`), pinball `0.0543` vs `0.0558`. At 1 d (`0.997`) and 30 d
(`1.003 [0.939, 1.091]`) it is a tie. Repeating over **5 folds × 3 seeds** on
val gives mean ratios `0.991 / 0.996 / 0.978 / 0.975` at 1/7/15/30 d (see
"Multi-fold held-out comparison"), but only fold 2 wins everywhere and the
15 d mean is carried by the smallest fold, so the gain is real but not stable
across folds. Conclusion: the graph carries usable spatial information for
stations the model has never seen at mid horizons — worth revisiting, not a
served-model replacement.

**Influence — local, and horizon-growing.** The published artifacts use the
consistent perturbation (rolling GWL features follow the anchor). The response
is locally linear (per-metre sensitivities from 0.05 m and 5 m perturbations
agree with correlation `1.000`), so the magnitudes are real, not numerical
noise. Same-district links are still stronger than cross-district ones — mean
`|sensitivity|` `1.2e-4` vs `1.5e-5` per metre at 7 d (8.1×) and `3.4e-4` vs
`2.6e-5` at 30 d (13.1×) — and the effect grows with horizon: strongest station
link `7.4e-3` per metre at 7 d, `9.4e-3` at 30 d; district diagonal max
`5.2e-4` / `3.7e-3`. The no-edge ablation has *exactly* zero off-diagonal
response, so every reported off-diagonal value is graph-only.

These are roughly 4–16× the raw-`gwl`-only magnitudes (`5.2e-4` / `1.4e-3` max
link, `1.4e-4` / `2.3e-4` district diagonal), because the model reacts to the
implied change in the rolling std as well as to the level. Pairs that are
exactly `0` stay in the same range (15–34 % of scored pairs in the consistency
snapshots, vs 16–31 % for the raw perturbation), so the matrix is still sparse
and local. Raw-only numbers are preserved in `influence_consistency.json`.

Coverage: 499 of 548 stations appear in at least one scored snapshot, and 481
(7 d) / 471 (30 d) carry at least one finite link; stations with no measurement
are omitted from `station_influence.json` rather than published as zeros.

**Practical read.** The GNN is not a candidate to replace the production
forecast — it does not beat it on the served horizon set. It is a research tool:
the influence matrices answer "which station's reading moves which other
station's forecast, and by how much", and the held-out-station result says the
graph is worth revisiting for stations with little history.

## Serving the artifacts

`ml/serving/influence.py` reads only JSON — **no torch, no torch_geometric at
request time** — caches by file mtime, and returns `{"available": false,
"reason": ...}` for missing / corrupt artifacts or when the flag is off.

Flask (`ml/api.py`):

```
GET /influence/meta
GET /influence/graph
GET /influence/station/<slug-or-name>?top=5
GET /influence/district/<name>?top=10
GET /influence/consistency
```

plus `/api/influence/...` aliases. The Node gateway and the web front-end are
**not** part of this change; the artifacts are exposed by Flask only.

The feature is **off by default**. To show it, add `"influence_enabled": true`
to `data/refresh/runtime.json` (that file is only read, never written, by this
feature). With the key absent the routes answer `available:false` and the
Streamlit Map page hides the section entirely.

## Reproducing

```bash
cd ml
python -m research.gnn.graph                  # graph.json
python -m research.gnn.data                   # split self-check
python -m research.gnn.train                  # -> models/research/gnn_traj.pt
python -m research.gnn.train --arch gru --out models/research/gru_matched.pt
python -m research.gnn.train --holdout-block 1 --out models/research/gnn_traj_holdout.pt
python -m research.gnn.train --arch gru --holdout-block 1 \
       --out models/research/gru_matched_holdout.pt
python -m research.gnn.eval --test --holdout   # eval.json / eval.md
python -m research.gnn.folds                   # 5 folds x 3 seeds -> folds.json / folds.md
python -m research.gnn.influence               # influence artifacts + heatmaps
python -m research.gnn.influence --consistent-derived   # same, derived features follow
python -m research.gnn.influence --check-derived 4      # consistency report only
```

`python -m research.gnn.train` refuses to write outside `models/research/`.

## Tests

`tests/test_influence_api.py` covers the availability contract (default off,
flag on with temp artifacts, `top` clamping, case-folded district lookup,
corrupt/missing files, mtime cache, `runtime.json` byte-identical) and the
consistency endpoint's summary-only payload.

`tests/test_influence_derived.py` pins the analytic rolling update against a
brute-force recompute (n = 1 … 120, zero-delta identity), the time-window counts
including the gap reset, and the feature-audit classification.

`tests/test_influence_isolation.py` covers import isolation (loader pulls in no
torch/PyG; `api` pulls in neither `research` nor PyG), the training write guard,
and that artifacts live under `outputs/`.
