"""dl_rebuild/gates — scoring protocol (mirrors exp_harness exactly).

  * fold train/val/holdout + per-station stride (cumcount %120==0) reuse
    exp_harness.build_folds / holdout_stride so numbers line up with E0.
  * persistence baseline = delta=0 (RMSE_p = ||target_d|| root mean square).
  * station bootstrap CI on RMSE(model)/RMSE(persistence) with seed 7, 2000
    reps (identical to exp_harness._score_set).
  * unseen-station CV: a (val) + (holdout) station group never appears in
    training; scored on the same fold strides -> ratio < 1 with CI<1 is the
    new-station reliability gate.
  * coverage/width for the 2026 one-shot: empirical sd of errors, 90% band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import exp_harness as H

B = 2000
RNG_SEED = 7


def folds(df: object) -> dict:
    return H.build_folds(df)


def stride(df: object, folds: dict) -> dict:
    return H.holdout_stride(df, folds)


def line_up(d: dict) -> dict:
    """Thin alias so callers read naturally."""
    return d


def window_ok(rs: np.ndarray, p: np.ndarray, W: int) -> np.ndarray:
    """True where a full W-length trailing window fits inside the station run."""
    return (p - rs[p]) >= W


def selector(d: dict, vsel: dict, W: int | None) -> tuple[np.ndarray, dict]:
    """Return (indices, meta) for a named window reflecting fold cfg keys."""
    ix = np.flatnonzero(
        (d["fin"] == 1)
        & (vsel["mask"])
        & (d["tgt_time"] <= vsel["tt_cap"].astype("datetime64[ns]").astype("int64"))
        & (d["date"] >= vsel["date_lo"].astype("datetime64[ns]").astype("int64"))
        & (d["date"] <= vsel["date_hi"].astype("datetime64[ns]").astype("int64"))
    )
    if W is not None:
        ok = window_ok(d["run_start"], ix, W)
        ix = ix[ok]
    return ix, {"n": int(ix.size), "mask": len(vsel.get("mask_meta", ""))}


def bootstrap_ci(em: np.ndarray, ep: np.ndarray, stE: np.ndarray):
    em = np.asarray(em, float); ep = np.asarray(ep, float); stE = np.asarray(stE)
    st_u = np.unique(stE)
    pos = {int(s): np.flatnonzero(stE == s) for s in st_u}
    rng = np.random.default_rng(RNG_SEED)
    if st_u.size < 15:
        r = float(np.sqrt(np.mean(em ** 2)) / max(np.sqrt(np.mean(ep ** 2)), 1e-9))
        return r, r, r, int(len(st_u))
    sample = rng.choice(st_u, size=(B, len(st_u)), replace=True)
    ratios = np.empty(B)
    for i in range(B):
        sel = np.concatenate([pos[int(s)] for s in sample[i]])
        ratios[i] = (np.sqrt(np.mean(em[sel] ** 2))
                     / max(np.sqrt(np.mean(ep[sel] ** 2)), 1e-9))
    clo, chi = np.quantile(ratios, [0.05, 0.95])
    point = float(np.sqrt(np.mean(em ** 2)) / max(np.sqrt(np.mean(ep ** 2)), 1e-9))
    return point, float(clo), float(chi), int(len(st_u))


def score_set(pred_d: np.ndarray, target_d: np.ndarray, st: np.ndarray,
              month: np.ndarray | None = None) -> dict:
    em = pred_d - target_d
    ep = target_d
    ratio, clo, chi, nst = bootstrap_ci(em, ep, st)
    r = {"n": int(len(target_d)), "n_stations": nst,
         "rmse_model": float(np.sqrt(np.mean(em ** 2))),
         "rmse_persist": float(np.sqrt(np.mean(ep ** 2))),
         "ratio": ratio, "ci_lo": clo, "ci_hi": chi,
         "mae_model": float(np.mean(np.abs(em)))}
    if month is not None:
        js = month <= 9
        r["ratio_junssep"] = (float(np.sqrt(np.mean(em[js] ** 2))
                                     / max(np.sqrt(np.mean(ep[js] ** 2)), 1e-9))
                              if js.any() else None)
    return r


def coverage_90(em: np.ndarray) -> dict:
    """90% coverage from an empirical error band around the mean delta pred."""
    sd = float(np.sqrt(np.mean(em ** 2)))
    width = 2 * 1.645 * sd
    cov = float(np.mean(np.abs(em) <= 1.645 * sd))
    return {"sigma": sd, "width_90": width, "coverage_90": cov,
            "n": int(len(em))}