"""dl_rebuild/run_all — full DL rebuild pipeline + gate scoreboard.

Protocol (mirrors exp_harness/exp_eval_2026 so numbers are comparable):
  * folds 2024/2025 + per-station stride from exp_harness
  * MLP (transferable features, NO st_id/dist_id) and GRU (trailing GWL
    trajectory + transferable features) trained per fold, 3 seeds, early
    stop on the leak-free time buffer; mean-prediction averaged over seeds
  * XGB-E0 reference on the SAME protocol for the unseen-station CV
  * unseen-station CV: district-stratified holdout (~15%) + val (~10%),
    trained only on the rest; scored on both fold strides → the
    new-station reliability gate
  * 2026 one-shot: train anchor/target < 2025-10-01, test stride from
    2026-01-01, coverage/width on the 90% empirical band

Stage selection: --only build|folds|stationcv|y2026
All writes stay under dl_rebuild/.  No production files touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]       # ml/
HERE = Path(__file__).resolve().parent           # dl_rebuild
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))             # for exp_harness

import build_data as BD
import gates as G
import train as TR
from models import GRUNet, MLPNet

SEEDS = (42, 840, 2056)
W = 36                        # shorter, regularized
OUT = HERE / "result.json"
REP = HERE / "report.md"


def minimal_frame(d: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "st_id": d["st"],
        "date": d["date"].astype("datetime64[ns]"),
        "target_time": d["tgt_time"].astype("datetime64[ns]"),
        "target_d": d["y"],
    })
    return df


def prep(folds_year: dict, d: dict, W_ok: bool) -> dict:
    """Standardise FTR + G on the fold train (NaN-aware); return arrays + idx."""
    t = folds_year["train"]
    if W_ok:
        ok = G.window_ok(d["run_start"], t, W)
        t = t[ok]
    mu = d["FTR"][t].mean(0, keepdims=True)
    sd = d["FTR"][t].std(0, keepdims=True) + 1e-6
    F = ((d["FTR"] - mu) / sd).astype(np.float32)
    gG = d["G"][t]                                   # train rows only
    gmu = np.nanmean(gG, axis=0, keepdims=True)
    gsd = np.nanstd(gG, axis=0, keepdims=True) + 1e-6
    gmu = np.where(np.isfinite(gmu), gmu, 0.0)
    gsd = np.where(np.isfinite(gsd), gsd, 1.0)
    Gs = ((d["G"] - gmu) / gsd).astype(np.float32)   # NaN stays NaN (mask keeps it)
    meta = {"train_rows": int(t.size)}
    return F, Gs, t, mu, sd, gmu, gsd, meta


def run_folds(d: dict, device) -> dict:
    import exp_harness as H
    df = minimal_frame(d)
    folds = H.build_folds(df)
    stride = H.holdout_stride(df, folds)
    out = {}
    for y in (2024, 2025):
        fo = folds[y]
        ho = stride[y]
        ys = d["y"]
        month = pd.to_datetime(d["date"].astype("datetime64[ns]")).month.to_numpy()
        fold_scores = {}
        # ---- MLP ----
        F, Gs, tr, mu, sd, gmu, gsd, meta = prep(fo, d, False)
        preds = []
        ym, ysd = float(ys[fo["train"]].mean()), float(ys[fo["train"]].std() + 1e-6)
        ystd = ((ys - ym) / ysd).astype(np.float32)
        for s in SEEDS:
            m = MLPNet(F.shape[1]).to(device)
            m, _ = TR.train_mlp(m, F, ystd, fo["train"], fo["val"], device, s)
            preds.append(TR.predict_mlp(m, F, ho, device, ym, ysd))
        pm = np.mean(preds, axis=0)
        fold_scores["mlp"] = G.score_set(pm, ys[ho], d["st"][ho], month[ho])
        # ---- GRU ----
        Fg, Gg, trg, mu, sd, gmu, gsd, meta2 = prep(fo, d, True)
        vok = G.window_ok(d["run_start"], fo["val"], W)
        va_g = fo["val"][vok]
        hog = ho[G.window_ok(d["run_start"], ho, W)]
        preds_g = []
        for s in SEEDS:
            m = GRUNet(Fg.shape[1], W=W).to(device)
            m, diag = TR.train_gru(m, Gg, trg, Fg, None, ys, None,
                                   Gg, va_g, Fg, ys, None, W, device, s)
            ym, ysd = float(ys[trg].mean()), float(ys[trg].std() + 1e-6)
            preds_g.append(TR.predict_gru(m, Gg, Fg, hog, W, device, ym, ysd))
        pgg = np.mean(preds_g, axis=0)
        fold_scores["gru"] = G.score_set(pgg, ys[hog], d["st"][hog], month[hog])
        out[str(y)] = {
            "train_rows": int(fo["train"].size),
            "val_rows": int(fo["val"].size),
            "stride_rows": int(len(ho)),
            "gru_stride_rows": int(len(hog)),
            "scores": fold_scores,
        }
        print(f"fold {y}: mlp={fold_scores['mlp']['ratio']:.4f} "
              f"gru={fold_scores['gru']['ratio']:.4f} "
              f"(persist {fold_scores['gru']['rmse_persist']:.3f})",
              flush=True)
    return out


def run_stationcv(d: dict, device) -> dict:
    """District-stratified unseen-station CV (train<2024-06-01)."""
    import exp_harness as H
    df = minimal_frame(d)
    folds = H.build_folds(df)
    stride = H.holdout_stride(df, folds)
    st = d["st"]
    stU = np.unique(st)
    dist = pd.Series(d["district"]).fillna("UNK").to_numpy()
    st_dist = pd.Series([dist[np.where(st == s)[0][0]] for s in stU], index=stU)
    rng = np.random.default_rng(123)
    part: dict[str, set] = {"train": set(), "val": set(), "hold": set()}
    for _, codes in st_dist.groupby(st_dist).groups.items():
        codes = np.asarray(list(codes), dtype=np.int64)
        lab = rng.choice(["train", "val", "hold"], size=len(codes), replace=True,
                         p=[0.75, 0.10, 0.15])
        for g in part:
            part[g].update(codes[lab == g].tolist())
    def in_part(p):
        return np.array([int(s) in p for s in st])
    tr_mask = in_part(part["train"])
    va_mask = in_part(part["val"])
    ho_mask = in_part(part["hold"])
    ys = d["y"]
    month = pd.to_datetime(d["date"].astype("datetime64[ns]")).month.to_numpy()
    tr = folds[2024]["train"][tr_mask[folds[2024]["train"]]]
    va = folds[2024]["val"][va_mask[folds[2024]["val"]]]
    hodx = np.concatenate([stride[y][ho_mask[stride[y]]] for y in (2024, 2025)])
    print(f"stationcv holdout stations={len(part['hold'])} "
          f"train={len(tr):,} val={len(va):,} scored={len(hodx):,}", flush=True)

    # ---- XGB-E0 reference (identity included, raw NaN, as production) ------
    xcols = [c for c in H.BASE_XCOLS]
    xres = {}
    import xgboost as xgb
    dfr = H.load_frame()                                # raw features, same order
    Xtr = dfr.iloc[tr][xcols].to_numpy("float32")
    Xva = dfr.iloc[va][xcols].to_numpy("float32")
    Xho = dfr.iloc[hodx][xcols].to_numpy("float32")
    del dfr
    preds = []
    for s in SEEDS:
        m = xgb.XGBRegressor(**H.XGB_PARAMS, random_state=s)
        try:
            m.fit(Xtr, ys[tr], eval_set=[(Xva, ys[va])],
                  early_stopping_rounds=60, verbose=False)
        except TypeError:
            m.set_params(early_stopping_rounds=60)
            m.fit(Xtr, ys[tr], eval_set=[(Xva, ys[va])], verbose=False)
        preds.append(m.predict(Xho).astype(np.float64))
    pm = np.mean(preds, axis=0)
    xres = G.score_set(pm, ys[hodx], st[hodx], month[hodx])
    print(f"stationcv xgb-e0 ratio={xres['ratio']:.4f} "
          f"ci=[{xres['ci_lo']:.3f},{xres['ci_hi']:.3f}]", flush=True)

    out = {}
    F, Gs, trS, mu, sd, gmu, gsd, meta = prep(
        {"train": tr, "val": va}, d, False)
    for name in ("mlp", "gru"):
        preds = []
        trg = tr
        va_g = va
        hog = hodx
        if name == "gru":
            ok = G.window_ok(d["run_start"], tr, W)
            trg = tr[ok]
            vok = G.window_ok(d["run_start"], va, W)
            va_g = va[vok]
            hog = hodx[G.window_ok(d["run_start"], hodx, W)]
            for s in SEEDS:
                m = GRUNet(F.shape[1], W=W).to(device)
                m, diag = TR.train_gru(m, Gs, trg, F, None, ys, None,
                                       Gs, va_g, F, ys, None, W, device, s)
                ym, ysd = float(ys[trg].mean()), float(ys[trg].std() + 1e-6)
                preds.append(TR.predict_gru(m, Gs, F, hog, W, device, ym, ysd))
            sc = G.score_set(np.mean(preds, axis=0), ys[hog], st[hog], month[hog])
            sc["stride_rows"] = int(len(hog))
        else:
            ym, ysd = float(ys[tr].mean()), float(ys[tr].std() + 1e-6)
            ystd = ((ys - ym) / ysd).astype(np.float32)
            for s in SEEDS:
                m = MLPNet(F.shape[1]).to(device)
                m, _ = TR.train_mlp(m, F, ystd, tr, va, device, s)
                preds.append(TR.predict_mlp(m, F, hog, device, ym, ysd))
            sc = G.score_set(np.mean(preds, axis=0), ys[hog], st[hog], month[hog])
            sc["stride_rows"] = int(len(hog))
        out[name] = sc
        print(f"stationcv {name} ratio={sc['ratio']:.4f} "
              f"ci=[{sc['ci_lo']:.3f},{sc['ci_hi']:.3f}]", flush=True)
    return {"stations": {"train": len(part["train"]), "val": len(part["val"]),
                         "hold": len(part["hold"])},
            "xgb_e0": xres, "dl": out}


def run_y2026(d: dict, device) -> dict:
    import exp_harness as H
    df = minimal_frame(d)
    st = d["st"]
    t = d["tgt_time"]
    dt = d["date"]
    # 2026 one-shot: train < 2025-10-01 (anchor+target), val [2025-10-01,2026-01-01),
    # test stride >= 2026-01-01
    dt_p = pd.to_datetime(d["date"].astype("datetime64[ns]"))
    tt_p = pd.to_datetime(d["tgt_time"].astype("datetime64[ns]"))
    T1 = pd.Timestamp("2025-10-01")
    T2 = pd.Timestamp("2026-01-01")
    tr = np.flatnonzero((d["fin"] == 1) & (dt_p < T1) & (tt_p < T1))
    va = np.flatnonzero((d["fin"] == 1) & (dt_p >= T1) & (dt_p < T2) &
                        (tt_p < T2))
    cand = np.flatnonzero((d["fin"] == 1) & (dt_p >= T2))
    cc = pd.Series(st[cand]).groupby(st[cand], sort=False).cumcount().to_numpy()
    ho = cand[cc % 120 == 0]
    ys = d["y"]
    month = pd.to_datetime(d["date"].astype("datetime64[ns]")).month.to_numpy()
    print(f"y2026 train={len(tr):,} val={len(va):,} stride={len(ho):,}",
          flush=True)
    out = {}
    F, Gs, trS, mu, sd, gmu, gsd, meta = prep({"train": tr, "val": va}, d, False)
    # MLP
    ym, ysd = float(ys[tr].mean()), float(ys[tr].std() + 1e-6)
    ystd = ((ys - ym) / ysd).astype(np.float32)
    preds = []
    for s in SEEDS:
        m = MLPNet(F.shape[1]).to(device)
        m, _ = TR.train_mlp(m, F, ystd, tr, va, device, s)
        preds.append(TR.predict_mlp(m, F, ho, device, ym, ysd))
    pm = np.mean(preds, axis=0)
    s = G.score_set(pm, ys[ho], st[ho], month[ho])
    out["mlp"] = {**s, **G.coverage_90(pm - ys[ho])}
    # GRU
    trg = tr[G.window_ok(d["run_start"], tr, W)]
    vok = G.window_ok(d["run_start"], va, W)
    hog = ho[G.window_ok(d["run_start"], ho, W)]
    Fg, Gg, _, mu, sd, gmu, gsd, _ = prep({"train": trg, "val": va[vok]}, d, True)
    preds = []
    for s in SEEDS:
        m = GRUNet(Fg.shape[1], W=W).to(device)
        m, diag = TR.train_gru(m, Gg, trg, Fg, None, ys, None,
                               Gg, va[vok], Fg, ys, None, W, device, s)
        ym, ysd = float(ys[trg].mean()), float(ys[trg].std() + 1e-6)
        preds.append(TR.predict_gru(m, Gg, Fg, hog, W, device, ym, ysd))
    pm = np.mean(preds, axis=0).astype(np.float64)
    s = G.score_set(pm, ys[hog], st[hog], month[hog])
    out["gru"] = {**s, **G.coverage_90(pm - ys[hog])}
    print(f"y2026 mlp ratio={out['mlp']['ratio']:.4f} cov="
          f"{out['mlp']['coverage_90']:.3f} | gru ratio="
          f"{out['gru']['ratio']:.4f} cov={out['gru']['coverage_90']:.3f}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["build", "folds", "stationcv", "y2026", "all"])
    args = ap.parse_args()
    t0 = time.time()
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False          # native GRU: tiny workspace
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)
    print(f"device={device} torch={torch.__version__}", flush=True)

    d = BD.ly_load()
    print(f"data ready n={d['n']:,} st={d['meta']['stations']} "
          f"ftr={d['FTR'].shape[1]}", flush=True)

    res = {}
    if OUT.exists():
        res = json.loads(OUT.read_text())
    if args.only in ("folds", "all"):
        res["folds"] = run_folds(d, device)
        OUT.write_text(json.dumps(res, indent=1))
        print(f"folds done {time.time()-t0:.0f}s", flush=True)
    if args.only in ("stationcv", "all"):
        res["stationcv"] = run_stationcv(d, device)
        OUT.write_text(json.dumps(res, indent=1))
        print(f"stationcv done {time.time()-t0:.0f}s", flush=True)
    if args.only in ("y2026", "all"):
        res["y2026"] = run_y2026(d, device)
        OUT.write_text(json.dumps(res, indent=1))
        print(f"y2026 done {time.time()-t0:.0f}s", flush=True)
    print(f"TOTAL {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()