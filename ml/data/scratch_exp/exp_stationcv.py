"""exp_stationcv — held-out-station 2-fold CV for finalists (E_best vs E0).

For each fold y in {2024, 2025}, stations are split deterministically into two
groups (even/odd st_id, sorted). For each group g: train on fold-y train rows
in g (early stop on fold-y val rows in g), score on fold-y HOLD-OUT stride rows
in the OTHER group. Metrics pooled over both groups per fold, 3 seeds, same
scoreboard (ratio vs persistence + station bootstrap CI + Jun-Sep split).
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

os.environ.setdefault("OMP_NUM_THREADS", "6")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import exp_harness as H

REPORT = H.HERE / "stationcv.json"

RAIN_MEM = ["rain_api7", "rain_api22", "rain_api60",
            "rain_cum90", "rain_cum180", "rain_cum365"]
E3 = ["e3_chg_k3", "e3_chg_k5"]

CONFIGS = {
    "E0": {"cols": H.BASE_XCOLS, "huber": False, "params": {}},
    "E_best": {"cols": H.BASE_XCOLS + RAIN_MEM + E3, "huber": True,
               "params": {}},
}

def main() -> None:
    t0 = time.time()
    df = H.load_frame()
    df = H.ensure_aux(df)
    df = H.build_neighbour(df)
    folds = H.build_folds(df)
    stride = H.holdout_stride(df, folds)

    st_all = np.sort(np.unique(df["st_id"].to_numpy()))
    grp = {"A": st_all[0::2], "B": st_all[1::2]}
    in_g = {}
    for k, g in grp.items():
        in_g[k] = np.isin(df["st_id"].to_numpy(), g)

    gwl = df["gwl"].to_numpy(float)
    yd = df["target_d"].to_numpy(float)
    stE = df["st_id"].to_numpy(np.int32)
    moE = df["date"].dt.month.to_numpy(np.int8)

    out = {}
    for name, cfg in CONFIGS.items():
        cols = cfg["cols"]
        per_fold = {}
        for y in H.FOLDS:
            ho = stride[y]
            all_err_m, all_err_p, all_st, all_mo = [], [], [], []
            seed_rmses = {s: [] for s in H.SEEDS}
            for g in ("A", "B"):
                hg = ho[~in_g[g][ho]]
                other = in_g[g]
                tr = folds[y]["train"][other[folds[y]["train"]]]
                va = folds[y]["val"][other[folds[y]["val"]]]
                Xtr = np.ascontiguousarray(df.iloc[tr][cols].to_numpy("float32"))
                Xva = np.ascontiguousarray(df.iloc[va][cols].to_numpy("float32"))
                Xho = np.ascontiguousarray(df.iloc[hg][cols].to_numpy("float32"))
                yv = yd[va].astype(np.float64)
                for seed in H.SEEDS:
                    params = dict(H.XGB_PARAMS, random_state=seed)
                    params.update(cfg["params"])
                    if cfg["huber"]:
                        params.update(objective="reg:pseudohubererror",
                                      huber_slope=2.0)
                    m_ = xgb.XGBRegressor(**params)
                    try:
                        m_.fit(Xtr, yd[tr].astype(np.float64),
                               eval_set=[(Xva, yv)], early_stopping_rounds=60,
                               verbose=False)
                    except TypeError:
                        m_.set_params(early_stopping_rounds=60)
                        m_.fit(Xtr, yd[tr].astype(np.float64),
                               eval_set=[(Xva, yv)], verbose=False)
                    pred = m_.predict(Xho).astype(np.float64)
                    em_g = yd[hg] - pred      # model delta error
                    all_err_m.append(em_g)
                    all_err_p.append(yd[hg])  # persistence delta error (delta=0)
                    all_st.append(stE[hg])
                    all_mo.append(moE[hg])
                    seed_rmses[seed].append(float(np.sqrt(np.mean(em_g ** 2))))
                del Xtr, Xva, Xho, m_
                gc.collect()
            em = np.concatenate(all_err_m)
            ep = np.concatenate(all_err_p)
            st = np.concatenate(all_st)
            mo = np.concatenate(all_mo)
            seed_rmse = {int(s): float(np.mean(v)) for s, v in seed_rmses.items()}
            per_fold[y] = H._score_set(em, ep, st, mo, seed_rmse)
            per_fold[y]["n_scores"] = len(em) // len(H.SEEDS)
            print(f"{name} fold{y}: r={per_fold[y]['ratio']} "
                  f"[{per_fold[y]['ci_lo']},{per_fold[y]['ci_hi']}] "
                  f"n={per_fold[y]['n_scores']} ({time.time()-t0:.0f}s)",
                  flush=True)
        out[name] = {"cols": cols, "folds": per_fold,
                     "params": cfg["params"], "huber": cfg["huber"]}
        REPORT.write_text(json.dumps(out, indent=1))

    print("\n=== held-out-station CV ===")
    for n, r in out.items():
        for y in H.FOLDS:
            f = r["folds"][y]
            print(f"{n:6s} {y} r={f['ratio']} [{f['ci_lo']},{f['ci_hi']}] "
                  f"gain={f['gain']} spread={f['spread']} n={f['n_scores']}")
    print(f"TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()