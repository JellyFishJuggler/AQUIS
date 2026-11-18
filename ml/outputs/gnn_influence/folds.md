# Held-out-station comparison — 5 spatial folds × 3 seeds

Generated: 2026-11-18 21:49:14  |  device: `cuda`

> **Research only.** Val split restricted to the held-out stations. Nothing tuned on the test period.

Protocol: 5 spatial k-means folds (block seed `42`), 3 training seeds [42, 43, 44], identical station subset for both models, 20 max epochs / patience 4.

`ratio = RMSE(gnn) / RMSE(gru)` — below 1 means the GNN wins. CI = station bootstrap (2000 resamples, nested over folds).

## Per horizon

| h | RMSE gnn | RMSE gru | ratio mean ± std (15 runs) | std of fold means | 95% CI | P(gnn better) | min | max |
|---|---|---|---|---|---|---|---|---|
| 1d | 0.2107 | 0.2121 | 0.9913 ± 0.0116 | 0.0066 | [0.9825, 0.9985] | 0.99 | 0.9645 | 1.0102 |
| 7d | 0.3561 | 0.3555 | 0.9963 ± 0.0358 | 0.0168 | [0.9865, 1.0088] | 0.79 | 0.9205 | 1.0664 |
| 15d | 0.5616 | 0.5607 | 0.9778 ± 0.0928 | 0.0598 | [0.9572, 0.9953] | 0.99 | 0.7498 | 1.1791 |
| 30d | 0.9923 | 0.9952 | 0.9746 ± 0.0917 | 0.0305 | [0.9139, 1.0006] | 0.97 | 0.8099 | 1.1571 |

## Per fold

| fold | unseen | 1d | 7d | 15d | 30d |
|---|---|---|---|---|---|
| 0 | 145 | 0.9835 ± 0.0166 | 0.9998 ± 0.0058 | 1.0004 ± 0.0137 | 1.0054 ± 0.0072 |
| 1 | 103 | 1.0005 ± 0.0052 | 1.0092 ± 0.0310 | 1.0214 ± 0.0211 | 0.9609 ± 0.1517 |
| 2 | 115 | 0.9953 ± 0.0138 | 0.9814 ± 0.0413 | 0.9686 ± 0.0042 | 0.9432 ± 0.0122 |
| 3 | 124 | 0.9890 ± 0.0062 | 1.0147 ± 0.0466 | 1.0207 ± 0.1437 | 1.0090 ± 0.1349 |
| 4 | 61 | 0.9883 ± 0.0119 | 0.9765 ± 0.0489 | 0.8779 ± 0.1323 | 0.9544 ± 0.1088 |

## Pinball loss

ratio `gnn/gru` = **0.9852 ± 0.0592** (15 runs).

