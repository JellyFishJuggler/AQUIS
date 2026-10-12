"""36_train_pooled_dl.py - pooled 30-day DL (surface b replacement candidate).

Ports the validated daily GRU backbone from ml-temp stage3_G (build_daily: last
6-hourly slot per station-day, W=60 daily steps; GRUNet) and extends its single
Huber point head to THREE pinball quantile heads (q05/q50/q95), matching the
locked "sequence GRU + quantile heads" principle.

Target: ``target_d`` = GWL(t+30d) - GWL(t) (the same target as 06_train's pooled
xgb_multihorizon), so this is a schema-compatible candidate for surface (b).

Split (identical to 06_train): train < 2025-10-01, val = Oct-Dec 2025, test =
2026. Trains an ensemble of 3 seeds (seed-mean, as in dl2026_validate).

Artifacts (never overwrites the incumbent xgb_multihorizon.joblib):
  * models/pooled_dl.pt           - list of 3 state_dicts (ensemble)
  * models/pooled_dl_config.json  - feature cols, window, norm stats, quantiles

Honest evaluation vs the incumbent pooled XGB on the 2026 stride is done by
37_backtest_pooled_dl.py.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
FEATS = ROOT / "data" / "features"
MODELS = ROOT / "models"

DENSE = ["gwl", "lag1", "lag4", "lag8", "lag28", "lag120",
         "gwl_roll7_mean", "gwl_roll7_std", "gwl_roll30_mean", "gwl_roll30_std",
         "gwl_slope7", "gwl_slope30", "gwl_slope90", "gwl_dev7", "gwl_dev30",
         "rain_1d", "rain_7d", "rain_30d", "rain_api7", "rain_api22", "rain_api60",
         "temp", "temp_7d", "humidity", "humidity_7d", "solar", "wind_speed",
         "pressure", "river_level", "canal_level", "gwl_obs7", "gwl_obs30",
         "gwl_obs90", "gwl_imputed"]
SEASONAL = ["month_sin", "month_cos", "doy_sin", "doy_cos", "monsoon"]
COLS = DENSE + SEASONAL

W = 60
MAX_EPOCHS, PATIENCE, BATCH, LR = 60, 8, 2048, 1e-3
CLIP = 1.0
QUANTILES = (0.05, 0.5, 0.95)
SEEDS = (42, 840, 2056)
VAL_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2026-01-01")


def build_daily(frames: list[pd.DataFrame]) -> dict:
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["Station", "time"])
    last = d.groupby(["Station", "date"], sort=False).tail(1).reset_index(drop=True)
    last = last.sort_values(["Station", "date"]).reset_index(drop=True)
    X = last[COLS].to_numpy(np.float32)
    M = np.isfinite(X).astype(np.float32)
    FTR = np.concatenate([np.where(M, X, 0.0).astype(np.float32), M], axis=1)
    gwl = last["gwl"].to_numpy(np.float64)
    dw = np.empty_like(gwl)
    dw[0] = 0.0
    dw[1:] = gwl[1:] - gwl[:-1]
    dw[np.isnan(gwl)] = np.nan
    G = np.stack([gwl, dw], axis=1).astype(np.float32)
    st = last["st_id"].to_numpy(np.int32)
    rid = np.cumsum(np.r_[True, np.diff(st) != 0]) - 1
    rlen = np.bincount(rid)
    run_start = np.repeat(np.cumsum(np.r_[0, rlen[:-1]]), rlen)
    y = last["target_d"].to_numpy(np.float64)
    return {"FTR": FTR, "G": G, "st": st, "n": len(last),
            "time": last["time"].to_numpy("datetime64[ns]"),
            "date": last["date"].to_numpy("datetime64[ns]"),
            "y": y, "fin": np.isfinite(y), "run_start": run_start, "cols": COLS,
            "last": last}


class GRUNetQ(nn.Module):
    """Daily sequence GRU encoder + linear head emitting QUANTILES outputs."""

    def __init__(self, feat_p, nq=len(QUANTILES), hidden=32, seq_in=2, layers=1):
        super().__init__()
        self.rnn = nn.GRU(seq_in + 1, hidden, num_layers=layers, batch_first=True)
        self.onorm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + feat_p, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, nq))

    def forward(self, seq, mask, feat):
        x = torch.cat([torch.where(torch.isfinite(seq), seq, torch.zeros_like(seq)),
                       mask[:, :, :1]], dim=-1)
        h, _ = self.rnn(x)
        return self.head(torch.cat([self.onorm(h[:, -1]), feat], dim=-1))


def dev(x, device):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)


def window_ix(G, ix, w):
    p = np.asarray(ix, int)
    return G[p[:, None] - w + 1 + np.arange(w)[None, :]]


def window_ok(run_start, ix, w):
    ix = np.asarray(ix, int)
    return (ix - run_start[ix]) >= (w - 1)


def pinball(pred, target, qs):
    e = target[:, None] - pred
    return torch.mean(torch.maximum(qs * e, (qs - 1.0) * e))


def train_seed(F, Gs, ystd, run_start, tr, va, device, seed, feat_p):
    torch.manual_seed(seed)
    net = GRUNetQ(feat_p).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2)
    qs = dev(np.array(QUANTILES, np.float32), device)
    best, best_state, bad = float("inf"), None, 0
    n = len(tr)
    for ep in range(MAX_EPOCHS):
        net.train()
        perm = np.random.default_rng(seed + ep).permutation(n)
        for i in range(0, n, BATCH):
            ix = tr[perm[i:i + BATCH]]
            seq = dev(window_ix(Gs, ix, W), device)
            pred = net(seq, torch.isfinite(seq).float(), dev(F[ix], device))
            loss = pinball(pred, dev(ystd[ix], device), qs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP); opt.step()
        net.eval()
        with torch.no_grad():
            v = 0.0
            for j in range(0, len(va), 4096):
                ch = va[j:j + 4096]
                seq = dev(window_ix(Gs, ch, W), device)
                pred = net(seq, torch.isfinite(seq).float(), dev(F[ch], device))
                v += float(pinball(pred, dev(ystd[ch], device), qs)) * len(ch)
            v /= max(len(va), 1)
        sched.step(v)
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state); net.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return net, best


def predict_seed(net, Gs, F, ix, device, ym, ys):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), 4096):
            b = np.asarray(ix[i:i + 4096], int)
            seq = dev(window_ix(Gs, b, W), device)
            out.append(net(seq, torch.isfinite(seq).float(), dev(F[b], device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64) * ys + ym


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)
    print(f"device={device} torch={torch.__version__}", flush=True)

    trf = pd.read_parquet(FEATS / "train.parquet")
    vaf = pd.read_parquet(FEATS / "val.parquet")
    D = build_daily([trf, vaf])
    del trf, vaf

    t = pd.to_datetime(D["time"])
    tr = np.where((t < VAL_START.to_datetime64()) & D["fin"] &
                  window_ok(D["run_start"], np.arange(D["n"]), W))[0]
    va = np.where((t >= VAL_START.to_datetime64()) & (t < TEST_START.to_datetime64())
                  & D["fin"] & window_ok(D["run_start"], np.arange(D["n"]), W))[0]
    print(f"daily rows={D['n']:,} train={len(tr):,} val={len(va):,}", flush=True)

    mu = D["FTR"][tr].mean(0, keepdims=True)
    sd = D["FTR"][tr].std(0, keepdims=True) + 1e-6
    F = ((D["FTR"] - mu) / sd).astype(np.float32)
    gmu = np.nanmean(D["G"][tr], axis=0, keepdims=True)
    gsd = np.nanstd(D["G"][tr], axis=0, keepdims=True) + 1e-6
    Gs = ((D["G"] - gmu) / gsd).astype(np.float32)
    ym = float(np.mean(D["y"][tr])); ys = float(np.std(D["y"][tr]) + 1e-6)
    ystd = ((D["y"] - ym) / ys).astype(np.float32)

    states, bests = [], []
    for s in SEEDS:
        net, best = train_seed(F, Gs, ystd, D["run_start"], tr, va, device, s, F.shape[1])
        states.append({k: v for k, v in net.state_dict().items()})
        bests.append(float(best))
        rm = float(np.sqrt(np.mean((predict_seed(net, Gs, F, va, device, ym, ys)[:, 1]
                                    - D["y"][va]) ** 2)))
        print(f"  seed {s}: val_pinball={best:.5f} val_rmse={rm:.4f}", flush=True)

    # calibration check on val (nominal 90% for q05/q95) -> widening factor
    pv = _ensemble_predict(states, Gs, F, va, device, ym, ys, F.shape[1])
    lo, med, hi = pv[:, 0], pv[:, 1], pv[:, 2]
    k = 1.0
    for _ in range(60):
        cov = float(np.mean((D["y"][va] >= med - k * (med - lo)) &
                            (D["y"][va] <= med + k * (hi - med))))
        if cov >= 0.90:
            break
        k += 0.02
    print(f"  val coverage@k=1: {float(np.mean((D['y'][va] >= lo) & (D['y'][va] <= hi))):.3f} "
          f"-> widen_k={k:.2f}", flush=True)

    MODELS.mkdir(exist_ok=True)
    torch.save(states, MODELS / "pooled_dl.pt")
    cfg = {
        "paradigm": "daily sequence GRU encoder + 3 pinball quantile heads (q05/q50/q95)",
        "target": "GWL(t+30d) - GWL(t); level = anchor + delta (anchor observed only)",
        "feature_cols": COLS, "window_w": W, "hidden": 32, "quantiles": list(QUANTILES),
        "seeds": list(SEEDS), "seed_val_pinball": bests,
        "norm": {"f_mu": mu.tolist(), "f_sd": sd.tolist(),
                 "g_mu": gmu.tolist(), "g_sd": gsd.tolist(), "y_mu": ym, "y_sd": ys},
        "widen_k": k, "val_start": str(VAL_START.date()),
        "test_start": str(TEST_START.date()),
        "train_rows_daily": int(len(tr)), "val_rows_daily": int(len(va)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (MODELS / "pooled_dl_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {MODELS/'pooled_dl.pt'} + pooled_dl_config.json  ({time.time()-t0:.0f}s)",
          flush=True)


def _load(state, feat_p, device):
    net = GRUNetQ(feat_p).to(device)
    net.load_state_dict(state); net.eval()
    return net


def _ensemble_predict(states, Gs, F, ix, device, ym, ys, feat_p):
    preds = [_predict_state(st, Gs, F, ix, device, ym, ys, feat_p) for st in states]
    return np.mean(preds, axis=0)


def _predict_state(state, Gs, F, ix, device, ym, ys, feat_p):
    net = _load(state, feat_p, device)
    return predict_seed(net, Gs, F, ix, device, ym, ys)


if __name__ == "__main__":
    main()
