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
