"""exp_track3_dl2 — Track 3 SECOND attempt (hard 90-min box, `timeout 5400`).

Same question, same 50 stations / daily grid / stride / scoreboard as
exp_track3_dl.py (attempt 1, MLP diverged: fold-2024 ratio 8.33). Attempt-2
changes, each aimed at a specific instability in attempt 1:

  * all-NaN train columns are reported and median-imputed (attempt 1 imputed
    silently; the warning was the only clue),
  * feature standardisation fit on train (kept) AND target standardisation,
  * LayerNorm after each hidden Linear (attempt 1 had none),
  * Huber loss (attempt 1 used MSE, which lets huge deltas blow the weights),
  * gradient clipping (norm 1.0),
  * Adam lr 1e-3 with ReduceLROnPlateau(factor 0.5, patience 2),
  * early stopping on val RMSE (patience 10), best state restored,
  * per-epoch train-loss trace is recorded so "train loss decreases" is auditable.

Verdict rule unchanged: DROP if fold-2024 MLP ratio >= XGB ratio on the SAME
rows, or if it fails to converge; KEEP otherwise. XGB reference on the same
rows is refit here too (identical to attempt 1).

Output: tracks/track3_result2.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_harness as H

OUT = H.HERE / "tracks" / "track3_result2.json"
B = 2000
N_STATIONS = 50
MLP_SEEDS = (42, 840, 2056)
MAX_EPOCHS, PATIENCE, LR, BATCH = 120, 10, 1e-3, 4096
HUBER_DELTA, CLIP = 1.0, 1.0


def verdict(v: str, reason: str, extra: dict | None = None) -> None:
    OUT.write_text(json.dumps({"verdict": v, "reason": reason,
                               **(extra or {})}, indent=1))
    print(f"VERDICT {v}: {reason}", flush=True)


def boot(em, ep, st_tier, seed=7):
    st_u = np.unique(st_tier)
    if st_u.size < 5:
        return None
    pos = {int(s): np.flatnonzero(st_tier == s) for s in st_u}
    rng = np.random.default_rng(seed)
    sample = rng.choice(st_u, size=(B, st_u.size), replace=True)
    ratios = np.empty(B)
    for i in range(B):
        sel = np.concatenate([pos[int(s)] for s in sample[i]])
        ratios[i] = np.sqrt(np.mean(em[sel] ** 2)) / np.sqrt(np.mean(ep[sel] ** 2))
    return [float(x) for x in np.quantile(ratios, [0.05, 0.95])]


def daily_stride(df: pd.DataFrame, folds: dict) -> dict[int, np.ndarray]:
    out = {}
    d = df["date"].to_numpy("datetime64[ns]")
    t = df["target_time"].to_numpy("datetime64[ns]")
    fin = np.isfinite(df["target_d"].to_numpy(float))
    st = df["st_id"].to_numpy(np.int32)
    for y, cfg in H.FOLDS.items():
        o = cfg["origin"].to_datetime64()
        hh = cfg["hh"].to_datetime64()
        cand = np.flatnonzero(fin & (d >= o) & (t <= hh))
        cc = pd.Series(st[cand]).groupby(st[cand], sort=False).cumcount().to_numpy()
        out[y] = cand[cc % 30 == 0]
    return out


def pick_stations(df: pd.DataFrame) -> list[str]:
    tr = (df["date"] < H.FOLDS[2024]["vl"]) & \
         (df["target_time"] < H.FOLDS[2024]["origin"])
    cnt = df.loc[tr.to_numpy() if isinstance(tr, pd.Series) else tr] \
        .groupby("Station").size()
    top = cnt.sort_values(ascending=False, kind="mergesort").head(N_STATIONS)
    return sorted(top.index.astype(str))


def main() -> None:
    t0 = time.time()
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        verdict("SKIP", f"torch unavailable: {e}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)
    print(f"device={device} torch={torch.__version__}", flush=True)

    df = H.load_frame()
    df = H.ensure_aux(df)
    keep = pick_stations(df)
    df = df[df["Station"].astype(str).isin(keep)].reset_index(drop=True)
    df = df[pd.to_datetime(df["time"]).dt.hour == 0].reset_index(drop=True)
    print(f"subset: {df['Station'].nunique()} st, {len(df):,} daily rows",
          flush=True)

    folds = H.build_folds(df)
    stride = daily_stride(df, folds)
    for y in H.FOLDS:
        print(f"fold {y}: train={len(folds[y]['train']):,} "
              f"val={len(folds[y]['val']):,} stride={len(stride[y]):,}",
              flush=True)

    yd = df["target_d"].to_numpy(float)
    stE = df["st_id"].to_numpy(np.int32)
    cols = H.BASE_XCOLS
    X_all = df[cols].to_numpy("float32")

    class MLP(nn.Module):
        def __init__(self, p: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(p, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    diag = {}
    results = {}
    for y in H.FOLDS:
        tr, va, ho = folds[y]["train"], folds[y]["val"], stride[y]
        Xtr_raw, Xva_raw, Xho_raw = X_all[tr], X_all[va], X_all[ho]
        ytr, yva = yd[tr], yd[va]
        ep = yd[ho]

        # ---- NaN audit (attempt-1 blind spot) + train-median imputation ----
        all_nan = [cols[j] for j in range(Xtr_raw.shape[1])
                   if not np.isfinite(Xtr_raw[:, j]).any()]
        med = np.nanmedian(Xtr_raw, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        Xtr = np.where(np.isfinite(Xtr_raw), Xtr_raw, med).astype(np.float32)
        Xva = np.where(np.isfinite(Xva_raw), Xva_raw, med).astype(np.float32)
        Xho = np.where(np.isfinite(Xho_raw), Xho_raw, med).astype(np.float32)
        diag[str(y)] = {"all_nan_cols": all_nan,
                        "n_all_nan": len(all_nan)}
        print(f"fold {y}: all-NaN train cols={all_nan or 'none'}", flush=True)

        # ---- XGB reference on the SAME rows (raw NaN, as production) -------
        import xgboost as xgb
        x_pred = []
        for s in H.SEEDS:
            m = xgb.XGBRegressor(**H.XGB_PARAMS, random_state=s)
            try:
                m.fit(Xtr_raw, ytr, eval_set=[(Xva_raw, yva)],
                      early_stopping_rounds=60, verbose=False)
            except TypeError:
                m.set_params(early_stopping_rounds=60)
                m.fit(Xtr_raw, ytr, eval_set=[(Xva_raw, yva)], verbose=False)
            x_pred.append(m.predict(Xho_raw).astype(np.float64))

        # ---- MLP -----------------------------------------------------------
        m_pred, losses = [], []
        for s in MLP_SEEDS:
            torch.manual_seed(s)
            mu = Xtr.mean(0, keepdims=True)
            sd = Xtr.std(0, keepdims=True) + 1e-6
            ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
            xtr = torch.from_numpy((Xtr - mu) / sd).to(device)
            ytr_t = torch.from_numpy(((ytr - ym) / ys).astype(np.float32)).to(device)
            xva = torch.from_numpy((Xva - mu) / sd).to(device)
            yva_t = torch.from_numpy(((yva - ym) / ys).astype(np.float32)).to(device)
            model = MLP(Xtr.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=2)
            lossfn = nn.HuberLoss(delta=HUBER_DELTA)
            best, best_state, bad = float("inf"), None, 0
            n = xtr.shape[0]
            trace = []
            for ep_i in range(MAX_EPOCHS):
                model.train()
                perm = torch.randperm(n, device=device)
                run, nb = 0.0, 0
                for i in range(0, n, BATCH):
                    idx = perm[i:i + BATCH]
                    opt.zero_grad()
                    loss = lossfn(model(xtr[idx]), ytr_t[idx])
                    if not torch.isfinite(loss):
                        verdict("DROP", f"non-finite train loss (seed {s}, fold {y})")
                        return
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                    opt.step()
                    run += float(loss.detach()); nb += 1
                model.eval()
                with torch.no_grad():
                    v = float(nn.functional.mse_loss(model(xva), yva_t).sqrt())
                sched.step(v)
                train_loss = run / max(nb, 1)
                trace.append(round(train_loss, 5))
                if not math.isfinite(v):
                    verdict("DROP", f"non-finite val RMSE (seed {s}, fold {y})")
                    return
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t
                                  in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            if best_state is None:
                verdict("DROP", f"no best state (seed {s}, fold {y})")
                return
            model.load_state_dict(best_state)
            with torch.no_grad():
                pred_scaled = model(torch.from_numpy(
                    (Xho - mu) / sd).to(device)).cpu().numpy()
            m_pred.append((pred_scaled * ys + ym).astype(np.float64))
            losses.append({"seed": s, "epochs": len(trace),
                           "train_first": trace[0], "train_last": trace[-1],
                           "train_min": min(trace), "val_best": round(best, 5)})
            print(f"  fold {y} seed {s}: epochs={len(trace)} train "
                  f"{trace[0]}->{trace[-1]} (min {min(trace)}) val_best={best:.4f}",
                  flush=True)
            torch.cuda.empty_cache() if device.type == "cuda" else None

        ex = np.mean(x_pred, axis=0) - ep
        em = np.mean(m_pred, axis=0) - ep
        rx = float(np.sqrt(np.mean(ex ** 2)) / np.sqrt(np.mean(ep ** 2)))
        rm = float(np.sqrt(np.mean(em ** 2)) / np.sqrt(np.mean(ep ** 2)))
        results[str(y)] = {
            "n_stride": len(ho), "n_stations": int(len(np.unique(stE[ho]))),
            "persist_rmse": round(float(np.sqrt(np.mean(ep ** 2))), 4),
            "xgb_ratio": round(rx, 4), "xgb_ci": boot(ex, ep, stE[ho]),
            "mlp_ratio": round(rm, 4), "mlp_ci": boot(em, ep, stE[ho]),
            "mlp_loss_trace": losses,
        }
        print(f"fold {y}: xgb={rx:.4f} mlp={rm:.4f} "
              f"(persist {results[str(y)]['persist_rmse']}) n={len(ho)}",
              flush=True)

    f24, f25 = results["2024"], results["2025"]
    converged = all(math.isfinite(v["mlp_ratio"]) for v in results.values())
    extra = {"2024": results["2024"], "2025": results["2025"],
             "nan_diag": diag,
             "config": {"loss": f"huber(delta={HUBER_DELTA})", "clip": CLIP,
                        "lr": LR, "sched": "ReduceLROnPlateau(0.5,patience2)",
                        "norm": "LayerNorm", "target_std": True,
                        "max_epochs": MAX_EPOCHS, "patience": PATIENCE}}
    if not converged:
        verdict("DROP", "non-finite metric", extra)
    elif f24["mlp_ratio"] >= f24["xgb_ratio"]:
        verdict("DROP",
                f"MLP not better than E0 on fold-2024 "
                f"({f24['mlp_ratio']} >= {f24['xgb_ratio']})", extra)
    else:
        verdict("KEEP",
                f"MLP beats E0 on fold-2024 ({f24['mlp_ratio']} < "
                f"{f24['xgb_ratio']}); fold-2025 {f25['mlp_ratio']} vs "
                f"{f25['xgb_ratio']}", extra)
    print(f"TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
