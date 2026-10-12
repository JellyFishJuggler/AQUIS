"""35_traj_paired_ci.py

Paired station-bootstrap CI comparing the DL trajectory model against the
incumbent XGB trajectory model on the *same* backtest anchors.

Reads outputs/traj_backtest_rows_dl.parquet (from `TRAJ_MODEL=dl 32_...`) and
outputs/traj_backtest_rows.parquet (from `TRAJ_MODEL=xgb 32_...`), joins on
(station, anchor_t, horizon_h) and reports, per horizon and overall, the
pooled-RMSE difference and ratio with a 2.5/97.5 CI from resampling stations
with replacement.

A ratio CI whose upper bound is < 1 means DL beats XGB at that horizon.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
KEYS = ["station", "anchor_t", "horizon_h"]
N_BOOT = 2000
SEED = 20261011


def _rmse(y, p):
    return float(math.sqrt(np.mean((y - p) ** 2)))


def _load(suffix):
    return pd.read_parquet(OUT / f"traj_backtest_rows{suffix}.parquet")


def _pair(horizon):
    dl = _load("_dl")
    xgb = _load("")
    dl = dl[dl["horizon_h"] == horizon][KEYS + ["obs", "traj_q50"]]
    xgb = xgb[xgb["horizon_h"] == horizon][KEYS + ["obs", "traj_q50"]]
    m = dl.merge(xgb, on=KEYS, suffixes=("_dl", "_xgb"))
    assert np.allclose(m["obs_dl"], m["obs_xgb"]), "obs mismatch across models"
    return m


def _bootstrap(m, rng):
    stations = m["station"].to_numpy()
    uniq = np.unique(stations)
    idx_by_station = {s: np.where(stations == s)[0] for s in uniq}
    y = m["obs_dl"].to_numpy()
    p_dl = m["traj_q50_dl"].to_numpy()
    p_xgb = m["traj_q50_xgb"].to_numpy()
    deltas = np.empty(N_BOOT)
    ratios = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by_station[s] for s in pick])
        r_dl = _rmse(y[ix], p_dl[ix])
        r_xgb = _rmse(y[ix], p_xgb[ix])
        deltas[b] = r_dl - r_xgb
        ratios[b] = r_dl / r_xgb
    return deltas, ratios


def main():
    rng = np.random.default_rng(SEED)
    horizons = sorted(pd.unique(_load("_dl")["horizon_h"]).tolist())
    rows = []
    print(f"{'h':>5} {'n':>6} {'rmse_dl':>9} {'rmse_xgb':>9} "
          f"{'delta':>9} {'ratio':>7} {'ratio_lo':>9} {'ratio_hi':>9} beat")
    for h in horizons:
        m = _pair(h)
        y = m["obs_dl"].to_numpy()
        r_dl = _rmse(y, m["traj_q50_dl"].to_numpy())
        r_xgb = _rmse(y, m["traj_q50_xgb"].to_numpy())
        deltas, ratios = _bootstrap(m, rng)
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        beat = bool(hi < 1.0)
        print(f"{h:>5} {len(m):>6} {r_dl:>9.4f} {r_xgb:>9.4f} "
              f"{r_dl - r_xgb:>9.4f} {r_dl / r_xgb:>7.4f} "
              f"{lo:>9.4f} {hi:>9.4f} {beat}")
        rows.append({"horizon_h": int(h), "n": int(len(m)),
                     "rmse_dl": r_dl, "rmse_xgb": r_xgb,
                     "delta": r_dl - r_xgb, "ratio": r_dl / r_xgb,
                     "ratio_lo": float(lo), "ratio_hi": float(hi),
                     "dl_beats_xgb": beat,
                     "delta_lo": float(np.percentile(deltas, 2.5)),
                     "delta_hi": float(np.percentile(deltas, 97.5))})

    res = {"n_boot": N_BOOT, "seed": SEED, "per_horizon": rows,
           "dl_beats_xgb_all_horizons": all(r["dl_beats_xgb"] for r in rows),
           "dl_beats_xgb_30d": next(r["dl_beats_xgb"] for r in rows
                                    if r["horizon_h"] == 720)}
    (OUT / "traj_paired_ci_dl_vs_xgb.json").write_text(json.dumps(res, indent=2))
    print("\n30d (h=720) DL beats XGB:", res["dl_beats_xgb_30d"],
          "| all horizons:", res["dl_beats_xgb_all_horizons"])
    print("wrote", OUT / "traj_paired_ci_dl_vs_xgb.json")


if __name__ == "__main__":
    main()
