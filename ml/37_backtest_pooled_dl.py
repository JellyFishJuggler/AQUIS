"""37_backtest_pooled_dl.py - honest 2026-stride backtest for surface (b) DL.

Compares the pooled daily-GRU quantile ensemble (models/pooled_dl.pt) against the
incumbent pooled XGBoost (models/xgb_multihorizon.joblib, "E0") + persistence on
an identical, non-overlapping 2026 stride (one anchor per station ~every 30 days,
only where the +30d target is observed).

Trains on nothing >= 2026-01-01 (36 uses train<Oct2025 / val Oct-Dec2025), so the
2026 rows are genuinely out of sample. Writes outputs/pooled_dl_eval.json.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
FEATS = ROOT / "data" / "features"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"

T = importlib.import_module("36_train_pooled_dl")
SEEDS = T.SEEDS


def metrics(pred_d, y, st):
    e = pred_d - y
    return {"n": int(len(y)), "n_stations": int(np.unique(st).size),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "dir_acc": float(np.mean(np.sign(pred_d) == np.sign(y)))}


def paired_ci(em, ep, st, n_boot=2000, seed=20261011):
    rng = np.random.default_rng(seed)
    st = np.asarray(st)
    uniq = np.unique(st)
    idx = {s: np.where(st == s)[0] for s in uniq}
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx[s] for s in pick])
        r_m = math.sqrt(np.mean(em[ix] ** 2))
        r_p = math.sqrt(np.mean(ep[ix] ** 2))
        ratios[b] = r_m / r_p
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    pt = math.sqrt(np.mean(em ** 2)) / math.sqrt(np.mean(ep ** 2))
    return {"ratio": float(pt), "ci": [float(lo), float(hi)], "beats": bool(hi < 1.0)}


def stride_per_station(t_days, st, cand, min_days=30):
    keep = []
    for s in np.unique(st[cand]):
        ix = cand[st[cand] == s]
        ix = ix[np.argsort(t_days[ix])]
        last = None
        for i in ix:
            if last is None or (t_days[i] - last) >= min_days:
                keep.append(i); last = t_days[i]
    return np.array(sorted(keep))


def main():
    device = torch.device("cpu")
    cfg = json.loads((MODELS / "pooled_dl_config.json").read_text())
    states = torch.load(MODELS / "pooled_dl.pt", map_location="cpu", weights_only=True)
    norm = cfg["norm"]

    frames = [pd.read_parquet(FEATS / n) for n in ("train.parquet", "val.parquet", "test.parquet")]
    D = T.build_daily(frames)
    del frames
    last = D["last"]

    t = pd.to_datetime(D["time"])
    t_days = t.to_numpy("datetime64[D]").astype("int64")
    test_start = np.datetime64(cfg["test_start"])
    cand = np.where((t >= test_start) & D["fin"])[0]
    keep = stride_per_station(t_days, D["st"], cand, 30)
    print(f"2026 stride anchors={len(keep)} stations={np.unique(D['st'][keep]).size}", flush=True)

    # ---- normalize + DL ensemble prediction ----
    mu = np.asarray(norm["f_mu"], np.float32); sd = np.asarray(norm["f_sd"], np.float32)
    F = ((D["FTR"] - mu) / sd).astype(np.float32)
    gmu = np.asarray(norm["g_mu"], np.float32); gsd = np.asarray(norm["g_sd"], np.float32)
    Gs = ((D["G"] - gmu) / gsd).astype(np.float32)
    ym, ys = float(norm["y_mu"]), float(norm["y_sd"])
    preds = [T._predict_state(st, Gs, F, keep, device, ym, ys, F.shape[1]) for st in states]
    dl = np.mean(preds, axis=0)  # (n,3) deltas
    k = float(cfg.get("widen_k", 1.0))
    dl_lo = dl[:, 1] - k * (dl[:, 1] - dl[:, 0])
    dl_hi = dl[:, 1] + k * (dl[:, 2] - dl[:, 1])

    # ---- incumbent pooled XGB (E0), same rows ----
    xgb_b = joblib.load(MODELS / "xgb_multihorizon.joblib")
    cols = getattr(xgb_b, "feature_names_in_", None)
    if cols is None:
        cols = list(json.loads((MODELS / "feature_config.json").read_text())["num_cols"]) + ["st_id", "dist_id"]
    cols = list(cols)
    e0 = xgb_b.predict(last.iloc[keep][cols].to_numpy(np.float32))

    y = D["y"][keep]
    st = D["st"][keep]
    res = {
        "model": "pooled daily GRU + 3 pinball quantile heads (3-seed ensemble)",
        "incumbent": "pooled XGBoost (xgb_multihorizon.joblib)",
        "period_start": str(pd.Timestamp(test_start).date()),
        "widen_k": k,
        "persistence": metrics(np.zeros_like(y), y, st),
        "E0": metrics(e0, y, st),
        "DL": metrics(dl[:, 1], y, st),
        "DL_coverage_90": float(np.mean((y >= dl_lo) & (y <= dl_hi))),
        "DL_coverage_raw": float(np.mean((y >= dl[:, 0]) & (y <= dl[:, 2]))),
    }
    res["DL"]["paired_vs_E0"] = paired_ci(dl[:, 1] - y, e0 - y, st)
    res["DL"]["paired_vs_persist"] = paired_ci(dl[:, 1] - y, -y, st)
    OUT.mkdir(exist_ok=True)
    (OUT / "pooled_dl_eval.json").write_text(json.dumps(res, indent=2, default=float))
    for a in ("persistence", "E0", "DL"):
        m = res[a]
        pb = m.get("paired_vs_E0", {})
        print(f"  {a:11} rmse={m['rmse']:.4f} mae={m['mae']:.4f} bias={m['bias']:+.3f} "
              f"dir={m['dir_acc']:.3f} paired_vs_E0={pb.get('ratio', float('nan')):.4f} "
              f"{pb.get('ci','')} beats={pb.get('beats')}")
    print(f"  DL coverage_90={res['DL_coverage_90']:.3f} raw={res['DL_coverage_raw']:.3f}")
    print(f"wrote {OUT/'pooled_dl_eval.json'}")


if __name__ == "__main__":
    main()
