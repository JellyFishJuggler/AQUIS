"""exp_track3_dl — Track 3, time-boxed DL probe (external `timeout 10800`).

Question: does a neural net beat the E0 XGBoost on the honest scoreboard?

Scope (kept small on purpose — the box is the design):
  * 50 stations chosen from TRAINING data only (top-50 by fold-2024 train
    row count, ties broken by name).
  * daily grid: rows with time.hour == 0 (one row per station-day).
  * same rolling-origin folds as exp_harness (origins 2024-06-01,
    2025-06-01; val = [origin-90d, origin-30d), holdout = origin..Dec-31,
    target purge), stride = per-station daily cumcount % 30 == 0 inside the
    purged holdout (= every 30 days, mirroring the 6h cumcount % 120).
  * models, fit on the SAME rows: E0-XGB (production params, 3 seeds) vs a
    tabular MLP (torch, Adam, early stop on val RMSE, 3 init seeds).
  * same scoreboard as every other track: ratio = RMSE(model)/RMSE(persist)
    on stride rows, station bootstrap CI (B=2000), both folds, pooled 3-seed
    mean predictions.

Drop rules (protocol): MLP is DROPPED if fold-2024 ratio >= E0-XGB ratio on
the same rows, or it fails to converge (val RMSE NaN / no improvement at
all), or the wall clock hits the external timeout (verdict recorded as
TIMEOUT by the caller). If torch/venv is missing: verdict SKIP.

Outputs: tracks/track3_result.json
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

OUT = H.HERE / "tracks" / "track3_result.json"
B = 2000
N_STATIONS = 50
MLP_SEEDS = (42, 840, 2056)
MAX_EPOCHS, PATIENCE, LR, BATCH = 100, 8, 1e-3, 4096


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
    """Per-station every-30th-day stride inside the purged holdout."""
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


def fit_xgb(Xtr, ytr, Xva, yva, seed):
    m = H.xgb.XGBRegressor(**H.XGB_PARAMS, random_state=seed)
    try:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=60,
              verbose=False)
    except TypeError:
        m.set_params(early_stopping_rounds=60)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def main() -> None:
    t0 = time.time()
    try:
        import torch
        import torch.nn as nn
    except Exception as e:                                    # offline/venv missing
        verdict("SKIP", f"torch unavailable: {e}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)

    df = H.load_frame()
    df = H.ensure_aux(df)          # MUST run on the full frame: aux cache
    #                              # overwrites the base feature columns
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
    gwl = df["gwl"].to_numpy(float)
    stE = df["st_id"].to_numpy(np.int32)
    cols = H.BASE_XCOLS
    X_all = df[cols].to_numpy("float32")

    # ---- MLP definition ---------------------------------------------------
    class MLP(nn.Module):
        def __init__(self, p: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(p, 256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def fit_mlp(Xtr, ytr, Xva, yva, seed):
        # train-only median imputation (XGB gets raw NaN; torch does not)
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med).astype(np.float32)
        Xva = np.where(np.isfinite(Xva), Xva, med).astype(np.float32)
        torch.manual_seed(seed)
        mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
        xtr = torch.from_numpy((Xtr - mu) / sd).to(device)
        ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
        xva = torch.from_numpy((Xva - mu) / sd).to(device)
        yva_t = torch.from_numpy(yva.astype(np.float32)).to(device)
        model = MLP(Xtr.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best, best_state, bad = float("inf"), None, 0
        n = xtr.shape[0]
        for ep in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xtr[idx]), ytr_t[idx])
                if not torch.isfinite(loss):
                    return None, float("nan")
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = float(nn.functional.mse_loss(model(xva), yva_t).sqrt())
            if not math.isfinite(v):
                return None, float("nan")
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t
                              in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is None:
            return None, float("nan")
        model.load_state_dict(best_state)
        return model, (mu, sd, med)

    results = {}
    for y in H.FOLDS:
        tr, va, ho = folds[y]["train"], folds[y]["val"], stride[y]
        Xtr, ytr = X_all[tr], yd[tr]
        Xva, yva = X_all[va], yd[va]
        Xho = X_all[ho]
        ep = yd[ho]                       # persistence delta error

        x_pred, m_pred = [], []
        for s in H.SEEDS:
            x_pred.append(fit_xgb(Xtr, ytr, Xva, yva, s).predict(Xho)
                          .astype(np.float64))
        for s in MLP_SEEDS:
            model, scal = fit_mlp(Xtr, ytr, Xva, yva, s)
            if model is None:
                verdict("DROP", "MLP failed to converge (val NaN / no best)",
                        {"fold": y})
                return
            mu, sd, med = scal
            with torch.no_grad():
                xh = np.where(np.isfinite(Xho), Xho, med).astype(np.float32)
                xt = torch.from_numpy((xh - mu) / sd).to(device)
                m_pred.append(model(xt).cpu().numpy().astype(np.float64))
            del xt
            torch.cuda.empty_cache() if device.type == "cuda" else None

        ex = np.mean(x_pred, axis=0) - ep
        em = np.mean(m_pred, axis=0) - ep
        rx = float(np.sqrt(np.mean(ex ** 2)) / np.sqrt(np.mean(ep ** 2)))
        rm = float(np.sqrt(np.mean(em ** 2)) / np.sqrt(np.mean(ep ** 2)))
        results[str(y)] = {
            "n_stride": len(ho), "n_stations": int(len(np.unique(stE[ho]))),
            "persist_rmse": round(float(np.sqrt(np.mean(ep ** 2))), 4),
            "xgb_ratio": round(rx, 4),
            "xgb_ci": boot(ex, ep, stE[ho]),
            "mlp_ratio": round(rm, 4),
            "mlp_ci": boot(em, ep, stE[ho]),
        }
        print(f"fold {y}: xgb={rx:.4f} mlp={rm:.4f} "
              f"(persist {results[str(y)]['persist_rmse']}) "
              f"n={len(ho)}", flush=True)

    f24, f25 = results["2024"], results["2025"]
    converged = all(math.isfinite(v["mlp_ratio"]) for v in results.values())
    if not converged:
        verdict("DROP", "non-finite metric", results)
    elif f24["mlp_ratio"] >= f24["xgb_ratio"]:
        verdict("DROP",
                f"MLP not better than E0 on fold-2024 "
                f"({f24['mlp_ratio']} >= {f24['xgb_ratio']})", results)
    else:
        verdict("KEEP",
                f"MLP beats E0 on fold-2024 ({f24['mlp_ratio']} < "
                f"{f24['xgb_ratio']}); fold-2025 sign check "
                f"{f25['mlp_ratio']} vs {f25['xgb_ratio']}", results)
    print(f"TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
