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
