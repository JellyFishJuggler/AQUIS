"""dl_rebuild/redteam — red-team reruns (scratch only, 30-day horizon).

Stage selection: --only persistence|controls|district|calibration|numbers|all
Guardrails: everything writes under dl_rebuild/ or tracks/.  models/ untouched,
freeze untouched, one heavy job at a time (caller uses --only + nohup).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
TRACKS = HERE.parent / "tracks"
LOGS = HERE / "logs"
LOGS.mkdir(exist_ok=True)

import build_data as BD
import gates as G
import train as TR
from models import GRUNet
import exp_harness as H
import run_all as R

SEEDS = R.SEEDS
W = R.W
PAIRED_B, PAIRED_SEED = 2000, 7


def ram_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _st_name_map(df: pd.DataFrame) -> dict:
    d = {}
    for k, v in df.groupby("st_id", sort=False)["Station"].first().items():
        d[int(k)] = str(v)
    return d


def doy_from(dt: np.ndarray) -> np.ndarray:
    s = pd.to_datetime(np.asarray(dt).astype("datetime64[ns]"))
    return pd.DatetimeIndex(s).dayofyear.to_numpy()


def paired_station_ci(em_g: np.ndarray, em_a: np.ndarray, stE: np.ndarray):
    """Station-bootstrap of RMSE(a)/RMSE(g).  ci_hi<1 => control worse than GRU."""
    em_g = np.asarray(em_g, float); em_a = np.asarray(em_a, float)
    stE = np.asarray(stE)
    st_u = np.unique(stE)
    pos = {int(s): np.flatnonzero(stE == s) for s in st_u}
    rng = np.random.default_rng(PAIRED_SEED)
    if st_u.size < 15:
        r = float(np.sqrt(np.mean(em_a ** 2)) / np.sqrt(np.mean(em_g ** 2) + 1e-9))
        return r, r, r
    pick = rng.choice(st_u, size=(PAIRED_B, len(st_u)), replace=True)
    rr = np.empty(PAIRED_B)
    for i in range(PAIRED_B):
        sel = np.concatenate([pos[int(s)] for s in pick[i]])
        rr[i] = np.sqrt(np.mean(em_a[sel] ** 2)) / np.sqrt(np.mean(em_g[sel] ** 2) + 1e-9)
    point = float(np.sqrt(np.mean(em_a ** 2)) / np.sqrt(np.mean(em_g ** 2) + 1e-9))
    lo, hi = np.quantile(rr, [0.05, 0.95])
    return point, float(lo), float(hi)


def seasonal_delta_pooled(yd: np.ndarray, doy: np.ndarray, tr_ix, pred_ix):
    """Global doy-mean delta over TRAINING stations only (no station info)."""
    g = pd.DataFrame({"doy": doy[tr_ix], "y": yd[tr_ix]}).groupby("doy")["y"].mean()
    pred = g.reindex(doy[pred_ix]).to_numpy()
    fallback = float(np.nanmean(yd[tr_ix])) if yd[tr_ix].size else 0.0
    pred = np.where(np.isfinite(pred), pred, fallback)
    return pred.astype(np.float64)


def stationcv_split(d, df):
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
        lab = rng.choice(["train", "val", "hold"], size=len(codes),
                         replace=True, p=[0.75, 0.10, 0.15])
        for g in part:
            part[g].update(codes[lab == g].tolist())

    def in_part(p):
        return np.array([int(s) in p for s in st])

    tr_mask = in_part(part["train"])
    va_mask = in_part(part["val"])
    ho_mask = in_part(part["hold"])
    tr = folds[2024]["train"][tr_mask[folds[2024]["train"]]]
    va = folds[2024]["val"][va_mask[folds[2024]["val"]]]
    hodx = np.concatenate([stride[y][ho_mask[stride[y]]] for y in (2024, 2025)])
    return tr, va, hodx, part


def y2026_split(d):
    dt_p = pd.to_datetime(d["date"].astype("datetime64[ns]"))
    tt_p = pd.to_datetime(d["tgt_time"].astype("datetime64[ns]"))
    T1 = pd.Timestamp("2025-10-01"); T2 = pd.Timestamp("2026-01-01")
    tr = np.flatnonzero((d["fin"] == 1) & (dt_p < T1) & (tt_p < T1))
    va = np.flatnonzero((d["fin"] == 1) & (dt_p >= T1) & (dt_p < T2) & (tt_p < T2))
    cand = np.flatnonzero((d["fin"] == 1) & (dt_p >= T2))
    cc = pd.Series(d["st"][cand]).groupby(d["st"][cand], sort=False).cumcount().to_numpy()
    ho = cand[cc % 120 == 0]
    return tr, va, ho


def prep_gru(d, tr, va, use_fts):
    ok = G.window_ok(d["run_start"], tr, W)
    trg = tr[ok]
    vok = G.window_ok(d["run_start"], va, W)
    va_g = va[vok]
    if not use_fts:
        gG = d["G"][trg]
        gmu = np.nanmean(gG, axis=0, keepdims=True)
        gsd = np.nanstd(gG, axis=0, keepdims=True) + 1e-6
        gmu = np.where(np.isfinite(gmu), gmu, 0.0)
        gsd = np.where(np.isfinite(gsd), gsd, 1.0)
        Gs = ((d["G"] - gmu) / gsd).astype(np.float32)
        F = np.zeros((d["G"].shape[0], 1), dtype=np.float32)
        return F, Gs, trg, va_g
    F, Gs, _, mu, sd, gmu, gsd, meta = R.prep({"train": trg, "val": va_g}, d, True)
    return F, Gs, trg, va_g


# ---------------------------------------------------------------------------
# 1. PERSISTENCE AUDIT
# ---------------------------------------------------------------------------
def stage_persistence():
    d = BD.ly_load()
    df = H.load_frame()
    nm = _st_name_map(df)
    t = pd.to_datetime(d["tgt_time"].astype("datetime64[ns]"))
    dt = pd.to_datetime(d["date"].astype("datetime64[ns]"))
    yd = d["y"].astype(float)
    fin = d["fin"] == 1
    T2 = pd.Timestamp("2026-01-01")
    cand = np.flatnonzero(fin & (dt >= T2))
    cc = pd.Series(d["st"][cand]).groupby(d["st"][cand], sort=False).cumcount().to_numpy()
    ho = cand[cc % 120 == 0]
    assert len(ho) == 3766, f"2026 stride rows = {len(ho)} (expected 3766)"

    base = df.dropna(subset=["target_d", "target", "gwl"])
    idf = (base["target_d"] - (base["target"] - base["gwl"])).abs()
    hrs720 = (base["target_time"] - base["time"]).dt.total_seconds().div(3600)
    dg = (base["date"] - base["time"]).dt.total_seconds().div(3600)

    ep = yd[ho]
    rmse_p = float(np.sqrt(np.mean(ep ** 2)))
    mae_p = float(np.mean(np.abs(ep)))
    qs = np.quantile(np.abs(ep), [0.5, 0.75, 0.9, 0.95, 0.99])
    st_ho = d["st"][ho]
    per_st = pd.DataFrame({"st": st_ho, "e": np.abs(ep)}).groupby("st")["e"].mean()
    print(f"persist(2026 stride) rows={len(ho)} st={np.unique(st_ho).size} "
          f"RMSE={rmse_p:.3f} MAE={mae_p:.3f} q90={qs[2]:.3f}", flush=True)

    prefixes = ["Azamgarh", "Deoria", "Banda", "Meerut", "Varanasi", "ASHADHA"]
    dash = sorted({v for v in nm.values()
                   if any(str(v).startswith(p) for p in prefixes)})
    st_to_rows = {}
    for i in ho:
        if nm.get(int(d["st"][i])) in dash:
            st_to_rows.setdefault(nm[int(d["st"][i])], []).append(int(i))
    rows_out = []
    for nm_i in dash:
        ix = st_to_rows.get(nm_i, [])
        if not ix:
            rows_out.append({"station": nm_i, "note": "no 2026 stride rows"})
            continue
        ix = np.asarray(ix)
        pick = np.unique(np.linspace(0, len(ix) - 1, 5).astype(int))
        for k in pick:
            i = int(ix[k])
            rows_out.append({
                "station": nm_i, "time": str(df["time"].iloc[i]),
                "anchor_date": str(dt[i]), "target_time": str(t[i]),
                "gwl_anchor": float(df["gwl"].iloc[i]),
                "target_level": float(df["target"].iloc[i]),
                "delta_actual": float(yd[i]), "delta_persist": 0.0,
                "level_pred_persist": float(df["gwl"].iloc[i]),
                "level_err": float(yd[i]),
            })

    lines = ["# Persistence Audit (30-day horizon, 6h grid)", "",
             "## Definitions (verified on tracks/base_features.parquet)",
             f"- delta identity: `target_d == target - gwl` on {len(idf):,} rows "
             f"(|err| max {idf.max():.2e}, mean {idf.mean():.2e}). "
             "The 30-day delta label is exactly the observed level change.",
             f"- horizon: `target_time - time = 720h` exactly (30d) on "
             f"{int(hrs720.count()):,} finite rows.",
             f"- anchor/grid gap: `date - time` centred {dg.mean():+.1f}h, "
             f"range [{dg.min():+.0f},{dg.max():+.0f}]h (features' as-of date "
             "precedes the 6h issue time; documented, not a leak).", "",
             "## Persistence rule under test",
             "- Persistence {delta, level} forecast at t+30d: **delta = 0**, "
             "**level = anchor gwl**.",
             "- Hence persistence delta error on every row is exactly `yd` "
             "(identity above); level error = `target - gwl = yd`.", "",
             f"## Persistence RMSE on all {len(ho):,} 2026-stride rows",
             f"- rows = {len(ho)}, stations = {np.unique(st_ho).size}",
             f"- RMSE = {rmse_p:.3f} m, MAE = {mae_p:.3f} m",
             f"- |error| quantiles: q50 {qs[0]:.3f} q75 {qs[1]:.3f} "
             f"q90 {qs[2]:.3f} q95 {qs[3]:.3f} q99 {qs[4]:.3f}",
             f"- mean per-station |error|: min {per_st.min():.3f} "
             f"median {per_st.median():.3f} max {per_st.max():.3f}", "",
             "## 5 sample anchors x six dashboard stations", ""]
    hdr = ["station", "time", "anchor_date", "target_time", "gwl_anchor",
           "target_level", "delta_actual", "delta_persist", "level_pred_persist",
           "level_err"]
    if rows_out:
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for r in rows_out:
            if "note" in r:
                lines.append(f"| {r['station']} | no rows |")
                continue
            lines.append("| " + " | ".join(f"{r[c]:.5g}" if isinstance(r[c], float)
                                           else str(r[c]) for c in hdr) + " |")
    lines += ["", f"Peak RSS {ram_kb()/1024.0:.0f} MB.", "",
              "**Conclusion:** persistence is the honest 'level persists' "
              "baseline; every model ratio divides by this RMSE on the same rows."]
    (TRACKS / "persistence_audit.md").write_text("\n".join(lines))
    print(f"wrote {TRACKS/'persistence_audit.md'}", flush=True)


# ---------------------------------------------------------------------------
# 2. CONTROLS (same stationcv holdout + same 2026 rows, 3 seeds)
# ---------------------------------------------------------------------------
def stage_controls():
    d = BD.ly_load()
    noid = [c for c in H.BASE_XCOLS if c not in ("st_id", "dist_id")]
    lb = pd.read_parquet(H.BASE, columns=noid)
    for c in lb.columns:
        lb[c] = lb[c].astype(np.float32)
    Lb = lb.to_numpy("float32")
    del lb; gc.collect()
    if Lb.shape[0] != int(d["n"]):
        raise SystemExit(f"base parquet rows {Lb.shape[0]} != d.n {d['n']}")
    df_all = H.load_frame()
    tr, va, hodx, part = stationcv_split(d, df_all)
    del df_all; gc.collect()
    yd = d["y"].astype(float)
    st = d["st"]
    month = pd.to_datetime(d["date"].astype("datetime64[ns]")).month.to_numpy()
    doy = doy_from(d["date"])
    Fg, Gs, trg, va_g = prep_gru(d, tr, va, use_fts=True)
    hog = hodx[G.window_ok(d["run_start"], hodx, W)]
    print(f"controls stationcv: hold st={len(part['hold'])} rows={len(hodx)} "
          f"gru rows={len(hog)}", flush=True)

    out = {"split": "stationcv-holdout", "hold_stations": len(part["hold"]),
           "rows": int(len(hodx)), "gru_rows": int(len(hog)), "controls": {}}

    # (a) pooled seasonal-delta baseline (aligned to hog)
    pred_a = seasonal_delta_pooled(yd, doy, trg, hog)
    sc_a = G.score_set(pred_a, yd[hog], st[hog], month[hog])
    print(f"a seasonal-delta ratio={sc_a['ratio']:.4f} "
          f"ci=[{sc_a['ci_lo']:.3f},{sc_a['ci_hi']:.3f}]", flush=True)

    # (b) E0 XGBoost WITHOUT st_id/dist_id (aligned to hog)
    Xtr = Lb[trg]
    Xva = Lb[va_g]
    Xho = Lb[hog]
    preds_b = []
    seed_b = {}
    for s in SEEDS:
        m = xgb.XGBRegressor(**H.XGB_PARAMS, random_state=s)
        try:
            m.fit(Xtr, yd[trg], eval_set=[(Xva, yd[va_g])],
                  early_stopping_rounds=60, verbose=False)
        except TypeError:
            m.set_params(early_stopping_rounds=60)
            m.fit(Xtr, yd[trg], eval_set=[(Xva, yd[va_g])], verbose=False)
        p = m.predict(Xho).astype(np.float64)
        preds_b.append(p)
        seed_b[s] = float(np.sqrt(np.mean((p - yd[hog]) ** 2)))
        del m; gc.collect()
    pm_b = np.mean(preds_b, axis=0)
    del Xtr, Xva, Xho, Lb; gc.collect()
    sc_b = G.score_set(pm_b, yd[hog], st[hog], month[hog])
    print(f"b e0-noid ratio={sc_b['ratio']:.4f} "
          f"ci=[{sc_b['ci_lo']:.3f},{sc_b['ci_hi']:.3f}]", flush=True)

    # (c) GRU own-GWL only (context off)
    Fz, Gz, trgz, vaz = prep_gru(d, tr, va, use_fts=False)
    hogz = hodx[G.window_ok(d["run_start"], hodx, W)]
    preds_c = []
    seed_c = {}
    for s in SEEDS:
        m = GRUNet(1, W=W).to(R.DEVICE)
        m, diag = TR.train_gru(m, Gz, trgz, Fz, None, yd, None,
                               Gz, vaz, Fz, yd, None, W, R.DEVICE, s)
        ym, ysd = float(yd[trgz].mean()), float(yd[trgz].std() + 1e-6)
        p = TR.predict_gru(m, Gz, Fz, hogz, W, R.DEVICE, ym, ysd)
        preds_c.append(p)
        seed_c[s] = float(np.sqrt(np.mean((p - yd[hogz]) ** 2)))
        del m; gc.collect(); torch.cuda.empty_cache()
    pm_c = np.mean(preds_c, axis=0)
    sc_c = G.score_set(pm_c, yd[hogz], st[hogz], month[hogz])
    print(f"c gru-owngwl ratio={sc_c['ratio']:.4f} "
          f"ci=[{sc_c['ci_lo']:.3f},{sc_c['ci_hi']:.3f}]", flush=True)

    # (d) GRU shuffled labels (negative control)
    ys_shuf = yd.astype(np.float32)
    perm = np.random.default_rng(123).permutation(trg.size)
    vis = trg[perm]
    ys_shuf[trg] = yd[vis]
    preds_d = []
    seed_d = {}
    for s in SEEDS:
        m = GRUNet(Fg.shape[1], W=W).to(R.DEVICE)
        m, diag = TR.train_gru(m, Gs, trg, Fg, None, ys_shuf, None,
                               Gs, va_g, Fg, ys_shuf, None, W, R.DEVICE, s)
        ym, ysd = float(ys_shuf[trg].mean()), float(ys_shuf[trg].std() + 1e-6)
        p = TR.predict_gru(m, Gs, Fg, hog, W, R.DEVICE, ym, ysd)
        preds_d.append(p)
        seed_d[s] = float(np.sqrt(np.mean((p - yd[hog]) ** 2)))
        del m; gc.collect(); torch.cuda.empty_cache()
    pm_d = np.mean(preds_d, axis=0)
    sc_d = G.score_set(pm_d, yd[hog], st[hog], month[hog])
    print(f"d gru-shuffled ratio={sc_d['ratio']:.4f} "
          f"ci=[{sc_d['ci_lo']:.3f},{sc_d['ci_hi']:.3f}]", flush=True)
    del ys_shuf, perm, vis, Fz, Gz, trgz, vaz; gc.collect(); torch.cuda.empty_cache()

    # reference GRU main (same protocol as the 0.9480 gate)
    preds_m = []
    seed_m = {}
    for s in SEEDS:
        m = GRUNet(Fg.shape[1], W=W).to(R.DEVICE)
        m, diag = TR.train_gru(m, Gs, trg, Fg, None, yd, None,
                               Gs, va_g, Fg, yd, None, W, R.DEVICE, s)
        ym, ysd = float(yd[trg].mean()), float(yd[trg].std() + 1e-6)
        p = TR.predict_gru(m, Gs, Fg, hog, W, R.DEVICE, ym, ysd)
        preds_m.append(p)
        seed_m[s] = float(np.sqrt(np.mean((p - yd[hog]) ** 2)))
        del m; gc.collect(); torch.cuda.empty_cache()
    pm_m = np.mean(preds_m, axis=0)
    sc_m = G.score_set(pm_m, yd[hog], st[hog], month[hog])
    print(f"GRU main ratio={sc_m['ratio']:.4f} "
          f"ci=[{sc_m['ci_lo']:.3f},{sc_m['ci_hi']:.3f}]", flush=True)
    del Fg, Gs, trg, va_g; gc.collect(); torch.cuda.empty_cache()

    em_g = pm_m - yd[hog]
    for nm, em_a in (("a_seasonal", pred_a - yd[hog]),
                     ("b_e0noid", pm_b - yd[hog]),
                     ("c_owngwl", pm_c - yd[hogz]),
                     ("d_shuf", pm_d - yd[hog])):
        ref = em_a if nm != "c_owngwl" else em_a
        st_ref = st[hog] if nm != "c_owngwl" else st[hogz]
        pt, lo, hi = paired_station_ci(em_g, ref, st[hog] if nm != "c_owngwl"
                                       else st_ref)
        out["controls"][nm] = {"pair_ratio": pt, "pair_ci": [lo, hi],
                               "gru_clearly_better": bool(hi < 1.0)}

    out["controls"]["a_seasonal"]["sc"] = sc_a
    out["controls"]["b_e0noid"]["sc"] = sc_b
    out["controls"]["c_owngwl"]["sc"] = sc_c
    out["controls"]["d_shuf"]["sc"] = sc_d
    out["reference_gru"] = sc_m
    out["seed_rmse"] = {"b_e0noid": seed_b, "c_owngwl": seed_c,
                        "d_shuf": seed_d, "gru_main": seed_m, "a_seasonal": None}
    out["ram_kb"] = ram_kb()

    a_p = out["controls"]["a_seasonal"]
    out["decision"] = {
        "seasonal_ratio": sc_a["ratio"], "gru_ratio": sc_m["ratio"],
        "pair_seasonal_over_gru": [a_p["pair_ratio"], a_p["pair_ci"]],
        "gru_beats_seasonal_ci": bool(a_p["pair_ci"][1] < 1.0),
        "early_stop": ("WRITE seasonal-shape finding"
                       if not a_p["gru_clearly_better"]
                       else "continue to district-held-out"),
    }
    (HERE / "result_controls.json").write_text(json.dumps(out, indent=1,
                                                          default=float))
    print("controls saved -> dl_rebuild/result_controls.json", flush=True)


# ---------------------------------------------------------------------------
# 3. TRUE DISTRICT-HELD-OUT (whole districts absent from training)
# ---------------------------------------------------------------------------
def district_holdout_split(d, df, n_hold=3):
    folds = H.build_folds(df)
    stride = H.holdout_stride(df, folds)
    st = d["st"]
    dist = pd.Series(d["district"]).fillna("UNK").to_numpy()
    stU = np.unique(st)
    st_dist = pd.Series([dist[np.where(st == s)[0][0]] for s in stU], index=stU)
    hodx = np.concatenate([stride[y] for y in (2024, 2025)])
    cnt = pd.Series(st[hodx]).map(st_dist).value_counts()
    hold_d = [str(x) for x in cnt.head(n_hold).index]
    hold_st = {int(s) for s in stU if str(st_dist[s]) in hold_d}

    def has_st(ix):
        return np.array([int(s) in hold_st for s in st[ix]])

    tr = folds[2024]["train"][~has_st(folds[2024]["train"])]
    va = folds[2024]["val"][~has_st(folds[2024]["val"])]
    hox = hodx[has_st(hodx)]
    return tr, va, hox, hold_d, sorted(hold_st)


def stage_district():
    d = BD.ly_load()
    df = H.load_frame()
    yd = d["y"].astype(float)
    st = d["st"]
    month = pd.to_datetime(d["date"].astype("datetime64[ns]")).month.to_numpy()
    doy = doy_from(d["date"])
    tr, va, hodx, hold_d, hold_st = district_holdout_split(d, df)
    print(f"district-holdout: districts={hold_d} st={len(hold_st)} "
          f"rows={len(hodx)}", flush=True)

    Fg, Gs, trg, va_g = prep_gru(d, tr, va, use_fts=True)
    hog = hodx[G.window_ok(d["run_start"], hodx, W)]
    out = {"split": "district-held-out", "hold_districts": hold_d,
           "hold_stations": int(len(hold_st)), "rows": int(len(hodx)),
           "gru_rows": int(len(hog)), "controls": {}}

    pred_a = seasonal_delta_pooled(yd, doy, trg, hodx)
    out["controls"]["a_seasonal"] = G.score_set(pred_a, yd[hodx], st[hodx],
                                                month[hodx])
    noid = [c for c in H.BASE_XCOLS if c not in ("st_id", "dist_id")]
    xcols = [c for c in noid if c in df.columns]
    Xtr = df.iloc[trg][xcols].to_numpy("float32")
    Xva = df.iloc[va_g][xcols].to_numpy("float32")
    Xho = df.iloc[hog][xcols].to_numpy("float32")
    preds_b = []
    for s in SEEDS:
        m = xgb.XGBRegressor(**H.XGB_PARAMS, random_state=s)
        try:
            m.fit(Xtr, yd[trg], eval_set=[(Xva, yd[va_g])],
                  early_stopping_rounds=60, verbose=False)
        except TypeError:
            m.set_params(early_stopping_rounds=60)
            m.fit(Xtr, yd[trg], eval_set=[(Xva, yd[va_g])], verbose=False)
        preds_b.append(m.predict(Xho).astype(np.float64))
    pm_b = np.mean(preds_b, axis=0)
    out["controls"]["b_e0noid"] = G.score_set(pm_b, yd[hog], st[hog], month[hog])
    print(f"district b_e0noid ratio={out['controls']['b_e0noid']['ratio']:.4f}",
          flush=True)

    preds_m = []
    seed_m = {}
    for s in SEEDS:
        m = GRUNet(Fg.shape[1], W=W).to(R.DEVICE)
        m, diag = TR.train_gru(m, Gs, trg, Fg, None, yd, None,
                               Gs, va_g, Fg, yd, None, W, R.DEVICE, s)
        ym, ysd = float(yd[trg].mean()), float(yd[trg].std() + 1e-6)
        p = TR.predict_gru(m, Gs, Fg, hog, W, R.DEVICE, ym, ysd)
        preds_m.append(p)
        seed_m[s] = float(np.sqrt(np.mean((p - yd[hog]) ** 2)))
    pm = np.mean(preds_m, axis=0)
    out["gru"] = G.score_set(pm, yd[hog], st[hog], month[hog])
    out["gru_seed_rmse"] = seed_m
    print(f"district GRU ratio={out['gru']['ratio']:.4f} "
          f"ci=[{out['gru']['ci_lo']:.3f},{out['gru']['ci_hi']:.3f}]", flush=True)

    em_g = pm - yd[hog]
    for nm, em_a in (("a_seasonal", pred_a[hog] - yd[hog]),
                     ("b_e0noid", pm_b - yd[hog])):
        pt, lo, hi = paired_station_ci(em_g, em_a, st[hog])
        out["controls"][nm]["pair_vs_gru"] = [pt, lo, hi]
    out["ram_kb"] = ram_kb()
    (HERE / "result_district.json").write_text(json.dumps(out, indent=1,
                                                          default=float))
    print("district saved -> dl_rebuild/result_district.json", flush=True)


# ---------------------------------------------------------------------------
# 4. CALIBRATION refit on time-ordered purged training block (30d only)
# ---------------------------------------------------------------------------
def stage_calibration():
    d = BD.ly_load()
    yd = d["y"].astype(float)
    st = d["st"]
    month = pd.to_datetime(d["date"].astype("datetime64[ns]")).month.to_numpy()
    tr, va, ho = y2026_split(d)
    cal = np.flatnonzero((d["fin"] == 1) &
                         (pd.to_datetime(d["tgt_time"].astype("datetime64[ns]")) >=
                          pd.Timestamp("2025-07-01")) &
                         (pd.to_datetime(d["tgt_time"].astype("datetime64[ns]")) <
                          pd.Timestamp("2025-10-01")) &
                         (pd.to_datetime(d["date"].astype("datetime64[ns]")) <
                          pd.Timestamp("2025-07-01")))
    print(f"calibration purged block rows={len(cal)} "
          "(tgt in [2025-07-01,2025-10-01), date<2025-07-01)", flush=True)

    Fg, Gs, trg, va_g = prep_gru(d, tr, va, use_fts=True)
    cal_g = cal[G.window_ok(d["run_start"], cal, W)]
    hog = ho[G.window_ok(d["run_start"], ho, W)]

    preds_m = []
    for s in SEEDS:
        m = GRUNet(Fg.shape[1], W=W).to(R.DEVICE)
        m, diag = TR.train_gru(m, Gs, trg, Fg, None, yd, None,
                               Gs, va_g, Fg, yd, None, W, R.DEVICE, s)
        ym, ysd = float(yd[trg].mean()), float(yd[trg].std() + 1e-6)
        preds_m.append(TR.predict_gru(m, Gs, Fg, np.concatenate([cal_g, hog]),
                                      W, R.DEVICE, ym, ysd))
    pm_all = np.mean(preds_m, axis=0)
    pm_cal = pm_all[:len(cal_g)]
    pm_hog = pm_all[len(cal_g):]
    em_cal = pm_cal - yd[cal_g]
    em_ho = pm_hog - yd[hog]

    k = np.quantile(np.abs(em_cal), 0.90)
    cov_cal = float(np.mean(np.abs(em_cal) <= k))
    width_cal = 2 * k
    cov_ho = float(np.mean(np.abs(em_ho) <= k))
    g90 = G.coverage_90(em_ho)
    out = {
        "block": {"rows": int(len(cal)), "gru_rows": int(len(cal_g)),
                  "window": "tgt[2025-07-01,2025-10-01) & date<2025-07-01"},
        "2026_stride": {"rows": int(len(hog)),
                        "n_stations": int(np.unique(st[hog]).size)},
        "cal_band": {"k": float(k), "width_90_cal": width_cal,
                     "coverage_90_cal": cov_cal},
        "applied_to_2026": {"coverage_90": cov_ho, "width_90": width_cal,
                            "width_limit_m": 3.1,
                            "width_ok": bool(width_cal <= 3.1),
                            "over_covered": bool(cov_ho > 0.93)},
        "empirical_band_old": g90,
        "gru_ratio_2026": G.score_set(pm_hog, yd[hog], st[hog], month[hog]),
        "read_once_note": "2026 stride read exactly once (scoring only).",
        "ram_kb": ram_kb(),
    }
    print(f"cal: cov={cov_cal:.3f} width={width_cal:.2f}m | 2026 applied "
          f"cov={cov_ho:.3f} | old emp cov={g90['coverage_90']:.3f} "
          f"width={g90['width_90']:.2f}m", flush=True)
    (HERE / "result_calibration.json").write_text(json.dumps(out, indent=1,
                                                            default=float))
    print("calibration saved -> dl_rebuild/result_calibration.json", flush=True)


# ---------------------------------------------------------------------------
# 5. NUMBERS: per-seed, per-district, Jun-Sep, RAM, fold-2024=1.2288 explain
# ---------------------------------------------------------------------------
def stage_numbers():
    d = BD.ly_load()
    df = H.load_frame()
    yd = d["y"].astype(float)
    st = d["st"]
    month = pd.to_datetime(d["date"].astype("datetime64[ns]")).month.to_numpy()
    dist = pd.Series(d["district"]).fillna("UNK").to_numpy()
    folds = H.build_folds(df)
    stride = H.holdout_stride(df, folds)

    fo = folds[2024]
    ho = stride[2024]
    Fg, Gs, trg, va_g = prep_gru(d, fo, fo["val"], use_fts=True)
    hog = ho[G.window_ok(d["run_start"], ho, W)]
    drop = len(ho) - len(hog)
    preds = []
    seed_m = {}
    for s in SEEDS:
        m = GRUNet(Fg.shape[1], W=W).to(R.DEVICE)
        m, diag = TR.train_gru(m, Gs, trg, Fg, None, yd, None,
                               Gs, va_g, Fg, yd, None, W, R.DEVICE, s)
        ym, ysd = float(yd[trg].mean()), float(yd[trg].std() + 1e-6)
        p = TR.predict_gru(m, Gs, Fg, hog, W, R.DEVICE, ym, ysd)
        preds.append(p)
        seed_m[s] = float(np.sqrt(np.mean((p - yd[hog]) ** 2)))
    pm = np.mean(preds, axis=0)
    sc = G.score_set(pm, yd[hog], st[hog], month[hog])
    per_st = pd.DataFrame({"st": st[hog], "em": np.abs(pm - yd[hog]),
                           "ep": np.abs(yd[hog]), "dist": dist[hog]}).groupby("st").mean()
    worst = per_st.assign(ratio=per_st["em"] / (per_st["ep"] + 1e-9)) \
        .sort_values("ratio", ascending=False).head(10)
    per_dist = pd.DataFrame({"dist": dist[hog], "em": (pm - yd[hog]) ** 2,
                             "ep": yd[hog] ** 2}).groupby("dist").agg(
        rmse_m=lambda x: np.sqrt(x["em"].mean()),
        rmse_p=lambda x: np.sqrt(x["ep"].mean()),
        n=("em", "size")).assign(ratio=lambda x: x["rmse_m"] / (x["rmse_p"] + 1e-9))

    f25 = folds[2025]
    h25 = stride[2025]
    hog25 = h25[G.window_ok(d["run_start"], h25, W)]
    out = {
        "fold2024_gru": {
            "stride_rows": int(len(ho)), "gru_stride_rows": int(len(hog)),
            "dropped_by_window": int(drop),
            "train_rows": int(fo["train"].size), "val_rows": int(fo["val"].size),
            "gru_train_rows": int(trg.size),
            "seed_rmse": seed_m,
            "spread": float((max(seed_m.values()) - min(seed_m.values())) /
                            min(seed_m.values())),
            "score": sc,
            "worst_stations": worst.reset_index().to_dict(orient="records"),
            "per_district": per_dist.reset_index().to_dict(orient="records"),
        },
        "fold2025_gru": {
            "stride_rows": int(len(h25)), "gru_stride_rows": int(len(hog25)),
            "train_rows": int(f25["train"].size),
        },
        "ram_kb": ram_kb(),
    }
    (HERE / "result_numbers.json").write_text(json.dumps(out, indent=1,
                                                        default=float))
    print(f"fold2024 GRU ratio={sc['ratio']:.4f} spread="
          f"{out['fold2024_gru']['spread']:.4f} window_drop={drop} "
          f"(train {fo['train'].size:,} -> gru {trg.size:,})", flush=True)
    print("numbers saved -> dl_rebuild/result_numbers.json", flush=True)


def main():
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = True
    R.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)
    print(f"device={R.DEVICE} torch={torch.__version__} seeds={SEEDS} W={W}",
          flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["persistence", "controls", "district",
                             "calibration", "numbers", "all"])
    a = ap.parse_args()
    t0 = time.time()
    if a.only in ("persistence", "all"):
        stage_persistence()
    if a.only in ("controls", "all"):
        gc.collect()
        stage_controls()
    if a.only in ("district", "all"):
        gc.collect()
        stage_district()
    if a.only in ("calibration", "all"):
        gc.collect()
        stage_calibration()
    if a.only in ("numbers", "all"):
        gc.collect()
        stage_numbers()
    print(f"TOTAL {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()