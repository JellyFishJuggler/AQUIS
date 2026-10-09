"""exp_eval_2026 — ONE-SHOT evaluation of the chosen config on the 2026 test
stride grid (frozen inputs only; run exactly once after selection).

Windows mirror production (06_features):
  train : anchor + target_time both < 2025-10-01
  val   : date in [2025-10-01, 2026-01-01) with target_time < 2026-01-01
          (early stopping only)
  test  : date >= 2026-01-01, label finite; scored on per-station stride rows
          (cumcount % 120 == 0 inside the test window) — the honest protocol.

Models (all on GPU):
  E0      : production point config retrained (squared, 3 seeds)
  E_best  : chosen config (C6: base + rain_api7/22/60 + rain_cum90/180/365
            + e3_chg_k3/k5, pseudo-Huber, depth6 lr0.05), 3 seeds
  q05/q95 : quantile companions (E_best features, reg:quantileerror) for the
            serve-time interval; coverage reported on the same stride rows.
Reported numbers: rmse(model) / rmse(persistence), station bootstrap CI, per-
seed + seed-mean values, Jun-Sep split, plus same metrics restricted to the
production 511-station test set.
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

OUT = H.HERE / "eval_2026.json"

RAIN_MEM = ["rain_api7", "rain_api22", "rain_api60",
            "rain_cum90", "rain_cum180", "rain_cum365"]
E3 = ["e3_chg_k3", "e3_chg_k5"]
E_BEST_COLS = H.BASE_XCOLS + RAIN_MEM + E3

VAL_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2026-01-01")

# production test-station set (from the frozen features/test.parquet)
PROD_TEST = (ROOT / "data" / "scratch_exp" / "frozen" /
             "features_test_stations.json")


def mask_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = df["date"].to_numpy("datetime64[ns]")
    t = df["target_time"].to_numpy("datetime64[ns]")
    fin = np.isfinite(df["target_d"].to_numpy(float))
    vs, ts = VAL_START.to_datetime64(), TEST_START.to_datetime64()
    tr = fin & (d < vs) & (t < vs)
    va = fin & (d >= vs) & (d < ts) & (t < ts)
    te_cand = fin & (d >= ts)
    return tr, va, te_cand


def te_stride(df: pd.DataFrame, te_cand: np.ndarray) -> np.ndarray:
    ix = np.flatnonzero(te_cand)
    st = df["st_id"].to_numpy(np.int32)[ix]
    cc = pd.Series(st).groupby(st, sort=False).cumcount().to_numpy()
    return ix[cc % 120 == 0]


def _fit(yobj: str, alpha: float | None, cols: list[str], seed: int,
         Xtr, ytr, Xva, yva):
    params = dict(H.XGB_PARAMS, random_state=seed)
    if yobj == "huber":
        params.update(objective="reg:pseudohubererror", huber_slope=2.0)
    elif yobj in ("q05", "q95"):
        params.update(objective="reg:quantileerror", quantile_alpha=alpha,
                      eval_metric="mae")
    m_ = xgb.XGBRegressor(**params)
    try:
        m_.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=60,
               verbose=False)
    except TypeError:
        m_.set_params(early_stopping_rounds=60)
        m_.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m_


def boot_ratio(em, ep, stE):
    st_u = np.unique(stE)
    pos = {int(s): np.flatnonzero(stE == s) for s in st_u}
    rng = np.random.default_rng(7)
    reps = 2000
    sample = rng.choice(st_u, size=(reps, len(st_u)), replace=True)
    ratios = np.empty(reps)
    for i in range(reps):
        sel = np.concatenate([pos[int(s)] for s in sample[i]])
        ratios[i] = np.sqrt(np.mean(em[sel] ** 2)) / np.sqrt(np.mean(ep[sel] ** 2))
    return np.quantile(ratios, [0.05, 0.95]).tolist()


def main() -> None:
    t0 = time.time()
    df = H.load_frame()
    df = H.ensure_aux(df)
    df = H.build_neighbour(df)
    tr, va, te_cand = mask_windows(df)
    ho = te_stride(df, te_cand)
    print(f"train={tr.sum():,} val={va.sum():,} test-cand={te_cand.sum():,} "
          f"stride={len(ho):,} ({len(np.unique(df.st_id.to_numpy()[ho]))} st)")

    gwl = df["gwl"].to_numpy(float)
    yd = df["target_d"].to_numpy(float)
    stE = df["st_id"].to_numpy(np.int32)
    moE = df["date"].dt.month.to_numpy(np.int8)
    station = df["Station"].astype(str).to_numpy()

    prod_st = set()
    if PROD_TEST.exists():
        prod_st = set(json.loads(PROD_TEST.read_text()))
    else:
        feats_test = pd.read_parquet(ROOT / "data" / "scratch_exp" /
                                     "frozen" / "test.parquet",
                                     columns=["Station"])
        prod_st = set(feats_test["Station"].astype(str))
        PROD_TEST.write_text(json.dumps(sorted(prod_st)))

    preds = {}
    for cfg_name, cfg in {"E0": ("sq", H.BASE_XCOLS),
                          "E_best": ("huber", E_BEST_COLS),
                          "E1": ("huber", H.BASE_XCOLS),
                          "E2a": ("sq", H.ARMS["E2a"]["cols"]),
                          "C2": ("huber", H.ARMS["C2"]["cols"]),
                          "C4": ("huber", H.ARMS["C4"]["cols"]),
                          "T3": ("huber", H.ARMS["T3"]["cols"]),
                          "T7": ("huber", H.ARMS["T7"]["cols"])}.items():
        yobj, cols = cfg
        Xtr = np.ascontiguousarray(df.iloc[tr][cols].to_numpy("float32"))
        Xva = np.ascontiguousarray(df.iloc[va][cols].to_numpy("float32"))
        Xho = np.ascontiguousarray(df.iloc[ho][cols].to_numpy("float32"))
        ytr = yd[tr].astype(np.float64)
        yva = yd[va].astype(np.float64)
        seed_pred = {}
        for seed in H.SEEDS:
            m_ = _fit(yobj, None, cols, seed, Xtr, ytr, Xva, yva)
            seed_pred[str(seed)] = np.full(len(df), np.nan)
            seed_pred[str(seed)][ho] = m_.predict(Xho).astype(np.float64)
        preds[cfg_name] = {"point": seed_pred}
        del Xtr, Xva, Xho, m_
        gc.collect()

    # quantile companions on E_best features
    qq = {}
    Xtr = np.ascontiguousarray(df.iloc[tr][E_BEST_COLS].to_numpy("float32"))
    Xva = np.ascontiguousarray(df.iloc[va][E_BEST_COLS].to_numpy("float32"))
    Xho = np.ascontiguousarray(df.iloc[ho][E_BEST_COLS].to_numpy("float32"))
    for a, label in ((0.05, "q05"), (0.95, "q95")):
        ps = {}
        for seed in H.SEEDS:
            m_ = _fit(label, a, E_BEST_COLS, seed, Xtr, ytr, Xva, yva)
            ps[str(seed)] = np.full(len(df), np.nan)
            ps[str(seed)][ho] = m_.predict(Xho).astype(np.float64)
        qq[label] = ps
    preds["E_best"]["quantile"] = qq
    del Xtr, Xva, Xho, m_
    gc.collect()

    err_p = yd[ho]                       # persistence delta error (delta=0)
    res = {}
    for cfg in preds:
        ps = preds[cfg]["point"]
        em_ens = np.mean([gwl[ho] + ps[s][ho] for s in ps], axis=0)  # level
        em_ens_d = em_ens - gwl[ho]       # delta forecast = pred mean
        rmse_ens = float(np.sqrt(np.mean((yd[ho] - em_ens_d) ** 2)))
        rmse_p = float(np.sqrt(np.mean(err_p ** 2)))
        ratio = rmse_ens / rmse_p
        ci = boot_ratio(yd[ho] - em_ens_d, err_p, stE[ho])
        js = moE[ho] <= 9
        r_js = (float(np.sqrt(np.mean((yd[ho][js] - em_ens_d[js]) ** 2))) /
                float(np.sqrt(np.mean(err_p[js] ** 2)))) if js.any() else None
        res[cfg] = {
            "rmse_persist": round(rmse_p, 4),
            "rmse_seedmean": round(rmse_ens, 4),
            "ratio": round(ratio, 4), "ci": [round(x, 4) for x in ci],
            "ratio_junssep": round(r_js, 4),
            "per_seed_rmse": {s: round(float(np.sqrt(np.mean(
                (yd[ho] - ps[s][ho]) ** 2))), 4) for s in ps},
            "n_scores": len(ho),
            "n_stations": int(len(np.unique(stE[ho]))),
        }
        # restrict to production test stations
        sel = np.isin(station[ho], list(prod_st))
        if sel.sum():
            e = yd[ho[sel]] - em_ens_d[sel]
            p = err_p[sel]
            res[cfg]["ratio_prod511"] = round(
                float(np.sqrt(np.mean(e ** 2))) / float(np.sqrt(np.mean(p ** 2))), 4)
            res[cfg]["n_prod511"] = int(sel.sum())
    # quantile coverage (E_best) — deltas: level = anchor + delta, so coverage
    # is evaluated on the delta forecast interval [mean(q05->pred), mean(q95->pred)]
    qqq = preds["E_best"]["quantile"]
    lo_d = np.mean([qqq["q05"][s][ho] for s in qqq["q05"]], axis=0)
    hi_d = np.mean([qqq["q95"][s][ho] for s in qqq["q95"]], axis=0)
    cov = float(np.mean((yd[ho] >= lo_d) & (yd[ho] <= hi_d)))
    res["E_best"]["interval"] = {
        "coverage": round(cov, 4),
        "width_mean": round(float(np.mean(hi_d - lo_d)), 4),
        "method": "q05/q95 quantile companions (E_best features), 3-seed mean; "
                  "level = anchor + delta; coverage on delta interval",
    }
    OUT.write_text(json.dumps({"test_window": str(TEST_START.date()), **res},
                              indent=1))

    print("\n=== 2026 one-shot test (stride grid) ===")
    for c in preds:
        r = res[c]
        print(f"{c:7s} ratio={r['ratio']} ci={r['ci']} rmse={r['rmse_seedmean']} "
              f"persist={r['rmse_persist']} js={r['ratio_junssep']} "
              f"per-seed={r['per_seed_rmse']} prod511={r.get('ratio_prod511')}")
    print("interval:", res["E_best"]["interval"])
    print(f"TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()