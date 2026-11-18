"""Trajectory and pooled GRU trainers plus the pooled honest backtest.

Merges the old 34/36/37 numbered scripts. Class names and the handful of
helpers used by the serving layer (``TrajGRU``, ``GRUNetQ``, ``build_daily``,
``_predict_state``, ``SEEDS``) keep their original names.

Run a stage with:  python -m pipeline_traj <train_dl|train_pooled_dl|backtest_pooled_dl>
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
F = importlib.import_module("pipeline_model")


# ====================== trajectory GRU (from 34_train_traj_dl.py)
ROOT = Path(__file__).resolve().parent
FE6 = ROOT / "data" / "features_6h"
FTRAJ = ROOT / "data" / "features_traj"
OUTM = ROOT / "models"
OUTM.mkdir(parents=True, exist_ok=True)

NUM_COLS = list(F.NUM_COLS)
META = ["Station", "time"]
MAX_H = 120
QUANTILES = (0.05, 0.50, 0.95)
PAD = -1


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _load_axis() -> dict:
    """Union of all splits = the station's observed 6h axis (features only)."""
    parts = []
    for sub in ("train", "val", "test"):
        p = FE6 / f"{sub}.parquet"
        if not p.exists():
            continue
        parts.append(pd.read_parquet(p, columns=META + NUM_COLS))
    d = pd.concat(parts, ignore_index=True)
    d["Station"] = d["Station"].astype(str)
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.sort_values(["Station", "time"], kind="stable").reset_index(drop=True)
    st = d["Station"].to_numpy()
    X = d[NUM_COLS].to_numpy(np.float32)
    finite = np.isfinite(X)
    X = np.where(finite, X, 0.0).astype(np.float32)
    starts = np.r_[0, np.flatnonzero(st[1:] != st[:-1]) + 1]
    ends = np.r_[starts[1:], len(st)]
    times = d["time"].to_numpy("datetime64[ns]")
    print(f"[axis] rows={len(d):,} stations={len(starts)}", flush=True)
    return {"X": X, "finite": finite.astype(np.float32), "st": st,
            "times": times, "starts": starts, "ends": ends}


def _load_split(name: str, axis: dict) -> dict:
    lf = pd.read_parquet(FTRAJ / f"{name}.parquet",
                         columns=META + NUM_COLS + ["h", "target_delta", "target_time"])
    lf["Station"] = lf["Station"].astype(str)
    lf["time"] = pd.to_datetime(lf["time"], errors="coerce")
    lf = lf.sort_values(["Station", "time", "h"], kind="stable").reset_index(drop=True)
    stc = lf["Station"].to_numpy()
    tmc = lf["time"].to_numpy("datetime64[ns]")
    same = np.r_[False, (stc[1:] == stc[:-1]) & (tmc[1:] == tmc[:-1])]
    grp = np.cumsum(~same) - 1
    A = int(grp[-1]) + 1
    counts = np.bincount(grp, minlength=A)
    ptr = np.r_[0, np.cumsum(counts)]
    a_station = stc[ptr[:-1]]
    a_time = tmc[ptr[:-1]]
    a_static = lf[NUM_COLS].to_numpy(np.float32)[ptr[:-1]]
    a_static_m = np.isfinite(a_static).astype(np.float32)
    a_static = np.where(np.isfinite(a_static), a_static, 0.0).astype(np.float32)
    h_arr = lf["h"].to_numpy(np.int16)
    d_arr = lf["target_delta"].to_numpy(np.float32)

    # map each anchor to its row on the union axis (station slice + searchsorted)
    a_axis = np.empty(A, np.int64)
    a_runstart = np.empty(A, np.int64)
    st_index = {axis["st"][s]: i for i, s in enumerate(axis["starts"])}
    for sname in np.unique(a_station):
        s0 = st_index[sname]
        sl0, sl1 = axis["starts"][s0], axis["ends"][s0]
        sel = np.flatnonzero(a_station == sname)
        pos = np.searchsorted(axis["times"][sl0:sl1], a_time[sel])
        pos = np.clip(pos, 0, (sl1 - sl0) - 1)
        ok = axis["times"][sl0:sl1][pos] == a_time[sel]
        if not ok.all():
            missed = int((~ok).sum())
            print(f"  WARN {sname}: {missed} anchors not found on axis", flush=True)
        a_axis[sel] = sl0 + pos
        a_runstart[sel] = sl0
    print(f"[{name}] anchors={A:,} rows={len(lf):,} "
          f"h~{h_arr.min()}..{h_arr.max()}", flush=True)
    return {"a_axis": a_axis, "a_runstart": a_runstart, "a_static": a_static,
            "a_static_m": a_static_m, "ptr": ptr, "h": h_arr, "delta": d_arr,
            "A": A}


def _make_batch(d_idx: np.ndarray, axis: dict, sp: dict, W: int, device):
    """Build padded tensors for a batch of anchor ids."""
    B = len(d_idx)
    seq = np.zeros((B, W, len(NUM_COLS)), np.float32)
    seqm = np.zeros((B, W, 1), np.float32)
    for i, a in enumerate(d_idx):
        end = int(sp["a_axis"][a]) + 1
        start = int(sp["a_runstart"][a])
        lo = max(start, end - W)
        L = end - lo
        if L <= 0:
            continue
        seq[i, W - L:] = axis["X"][lo:end]
        seqm[i, W - L:, 0] = 1.0
    # horizons per anchor -> pad to max length
    hp = [sp["h"][sp["ptr"][a]:sp["ptr"][a + 1]].astype(np.float32) for a in d_idx]
    dp = [sp["delta"][sp["ptr"][a]:sp["ptr"][a + 1]] for a in d_idx]
    lens = np.array([len(x) for x in hp], np.int64)
    Lmax = int(lens.max())
    H = np.full((B, Lmax), PAD, np.float32)
    D = np.zeros((B, Lmax), np.float32)
    M = np.zeros((B, Lmax), np.float32)
    for i, (hh, dd) in enumerate(zip(hp, dp)):
        H[i, :len(hh)] = hh
        D[i, :len(dd)] = dd
        M[i, :len(dd)] = 1.0
    hvalid = np.clip(H, 1, MAX_H)
    hf = np.stack([hvalid / MAX_H,
                   np.sin(2 * np.pi * hvalid / MAX_H),
                   np.cos(2 * np.pi * hvalid / MAX_H)], axis=-1)
    return {
        "seq": torch.from_numpy(seq).to(device),
        "seqm": torch.from_numpy(seqm).to(device),
        "static": torch.from_numpy(sp["a_static"][d_idx]).to(device),
        "h": torch.from_numpy(hvalid).to(device),
        "hf": torch.from_numpy(hf.astype(np.float32)).to(device),
        "delta": torch.from_numpy(D).to(device),
        "mask": torch.from_numpy(M).to(device),
    }


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class TrajGRU(nn.Module):
    def __init__(self, nfeat: int, hidden: int = 48, static: int = 30,
                 nq: int = 3, layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(nfeat + 1, hidden, num_layers=layers, batch_first=True)
        self.onorm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + static + 3, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, nq))

    def forward(self, seq, seqm, static, hf):
        x = torch.cat([seq, seqm], dim=-1)
        h, _ = self.rnn(x)
        ctx = self.onorm(h[:, -1])
        B, L = hf.shape[0], hf.shape[1]
        inp = torch.cat([ctx[:, None, :].expand(B, L, -1),
                         static[:, None, :].expand(B, L, -1), hf], dim=-1)
        return self.head(inp)


def pinball(pred, target, mask):
    q = torch.tensor(QUANTILES, device=pred.device)
    e = target[..., None] - pred
    loss = torch.maximum(q * e, (q - 1) * e)
    loss = (loss * mask[..., None]).sum() / (mask.sum() * len(QUANTILES) + 1e-9)
    return loss


# --------------------------------------------------------------------------- #
# train / val
# --------------------------------------------------------------------------- #
def _val_per_h(model, axis, sp, W, device, steps):
    model.eval()
    order = np.arange(sp["A"])
    hs, ds, ps = [], [], []
    with torch.no_grad():
        for i in range(0, len(order), 512):
            idx = order[i:i + 512]
            b = _make_batch(idx, axis, sp, W, device)
            pred = model(b["seq"], b["seqm"], b["static"], b["hf"]).cpu().numpy()
            m = b["mask"].cpu().numpy() > 0
            hv = b["h"].cpu().numpy().astype(np.int16)
            hs.append(hv[m]); ds.append(b["delta"].cpu().numpy()[m]); ps.append(pred[m])
    h_all = np.concatenate(hs); d_all = np.concatenate(ds); p_all = np.concatenate(ps)
    p_all = np.sort(p_all, axis=1)
    rows = []
    for h in steps:
        sel = h_all == h
        if not sel.any():
            continue
        y = d_all[sel]; p = p_all[sel]
        e = y - p[:, 1]
        rows.append({"h": int(h), "n": int(sel.sum()),
                     "rmse": float(np.sqrt(np.mean(e ** 2))),
                     "mae": float(np.mean(np.abs(e))),
                     "bias": float(np.mean(p[:, 1] - y)),
                     "coverage": float(np.mean((p[:, 0] <= y) & (y <= p[:, 2]))),
                     "halfwidth_median": float(np.median((p[:, 2] - p[:, 0]) / 2))})
    return rows


def train_dl_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--w", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    torch.set_num_threads(6)
    print(f"device={device} torch={torch.__version__} W={args.w} "
          f"hidden={args.hidden} batch={args.batch}", flush=True)
    t0 = time.time()

    axis = _load_axis()
    tr = _load_split("train", axis)
    va = _load_split("val", axis)
    if args.smoke:
        rng = np.random.default_rng(0)
        tr["sel"] = rng.choice(tr["A"], size=min(6000, tr["A"]), replace=False)
        va["sel"] = rng.choice(va["A"], size=min(800, va["A"]), replace=False)
    else:
        tr["sel"] = np.arange(tr["A"]); va["sel"] = np.arange(va["A"])

    model = TrajGRU(len(NUM_COLS), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2)

    best, bad, best_state = float("inf"), 0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.default_rng(args.seed + ep).permutation(tr["sel"])
        run, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            b = _make_batch(idx, axis, tr, args.w, device)
            loss = pinball(model(b["seq"], b["seqm"], b["static"], b["hf"]),
                           b["delta"], b["mask"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            run += float(loss.detach()); nb += 1
        v = _val_rmse30(model, axis, va, args.w, device)
        sched.step(v)
        print(f"  ep{ep:02d} train_loss={run/max(nb,1):.4f} val_rmse30={v:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stop", flush=True)
                break
    model.load_state_dict(best_state)

    steps = [1, 2, 3, 4, 8, 12, 28, 60, 120]
    vmetrics = _val_per_h(model, axis, va, args.w, device,
                          [1, 2, 3, 4, 8, 12, 28, 60, 120] if not args.smoke else [1, 2, 8, 30])
    torch.save(best_state, OUTM / "traj_dl.pt")
    cfg = {
        "paradigm": "sequence GRU encoder + horizon-conditioned quantile heads (pinball)",
        "target": "GWL(t+h*6h) - GWL(t); level = anchor + delta (anchor observed only)",
        "feature_cols": NUM_COLS, "horizon_cols": ["h", "h_sin", "h_cos"],
        "quantiles": {"q05": QUANTILES[0], "q50": QUANTILES[1], "q95": QUANTILES[2]},
        "max_h": MAX_H, "window_w": args.w, "hidden": args.hidden,
        "train_anchors": int(len(tr["sel"])), "val_anchors": int(len(va["sel"])),
        "best_val_rmse30": float(best),
        "val_per_h_metrics": vmetrics,
        "smoke": bool(args.smoke),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUTM / "traj_dl_config.json").write_text(json.dumps(cfg, indent=2))
    print("\nper-horizon validation (delta RMSE m / coverage / half-width):")
    print(pd.DataFrame(vmetrics).round(4).to_string(index=False))
    print(f"\nelapsed {(time.time()-t0):.0f}s -> models/traj_dl.pt + traj_dl_config.json")


def _val_rmse30(model, axis, sp, W, device):
    rows = _val_per_h(model, axis, sp, W, device, [MAX_H])
    return rows[0]["rmse"] if rows else float("nan")


# ====================== pooled daily GRU (from 36_train_pooled_dl.py)
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
POOLED_QUANTILES = (0.05, 0.5, 0.95)
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
    """Daily sequence GRU encoder + linear head emitting POOLED_QUANTILES outputs."""

    def __init__(self, feat_p, nq=len(POOLED_QUANTILES), hidden=32, seq_in=2, layers=1):
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


def pooled_pinball(pred, target, qs):
    e = target[:, None] - pred
    return torch.mean(torch.maximum(qs * e, (qs - 1.0) * e))


def train_seed(F, Gs, ystd, run_start, tr, va, device, seed, feat_p):
    torch.manual_seed(seed)
    net = GRUNetQ(feat_p).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2)
    qs = dev(np.array(POOLED_QUANTILES, np.float32), device)
    best, best_state, bad = float("inf"), None, 0
    n = len(tr)
    for ep in range(MAX_EPOCHS):
        net.train()
        perm = np.random.default_rng(seed + ep).permutation(n)
        for i in range(0, n, BATCH):
            ix = tr[perm[i:i + BATCH]]
            seq = dev(window_ix(Gs, ix, W), device)
            pred = net(seq, torch.isfinite(seq).float(), dev(F[ix], device))
            loss = pooled_pinball(pred, dev(ystd[ix], device), qs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP); opt.step()
        net.eval()
        with torch.no_grad():
            v = 0.0
            for j in range(0, len(va), 4096):
                ch = va[j:j + 4096]
                seq = dev(window_ix(Gs, ch, W), device)
                pred = net(seq, torch.isfinite(seq).float(), dev(F[ch], device))
                v += float(pooled_pinball(pred, dev(ystd[ch], device), qs)) * len(ch)
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


def train_pooled_dl_main():
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
        "paradigm": "daily sequence GRU encoder + 3 pooled_pinball quantile heads (q05/q50/q95)",
        "target": "GWL(t+30d) - GWL(t); level = anchor + delta (anchor observed only)",
        "feature_cols": COLS, "window_w": W, "hidden": 32, "quantiles": list(POOLED_QUANTILES),
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


# ====================== pooled GRU backtest (from 37_backtest_pooled_dl.py)
ROOT = Path(__file__).resolve().parent
FEATS = ROOT / "data" / "features"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"



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


def backtest_pooled_dl_main():
    device = torch.device("cpu")
    cfg = json.loads((MODELS / "pooled_dl_config.json").read_text())
    states = torch.load(MODELS / "pooled_dl.pt", map_location="cpu", weights_only=True)
    norm = cfg["norm"]

    frames = [pd.read_parquet(FEATS / n) for n in ("train.parquet", "val.parquet", "test.parquet")]
    D = build_daily(frames)
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
    preds = [_predict_state(st, Gs, F, keep, device, ym, ys, F.shape[1]) for st in states]
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
    OUmkdir(exist_ok=True)
    (OUT / "pooled_dl_eval.json").write_text(json.dumps(res, indent=2, default=float))
    for a in ("persistence", "E0", "DL"):
        m = res[a]
        pb = m.get("paired_vs_E0", {})
        print(f"  {a:11} rmse={m['rmse']:.4f} mae={m['mae']:.4f} bias={m['bias']:+.3f} "
              f"dir={m['dir_acc']:.3f} paired_vs_E0={pb.get('ratio', float('nan')):.4f} "
              f"{pb.get('ci','')} beats={pb.get('beats')}")
    print(f"  DL coverage_90={res['DL_coverage_90']:.3f} raw={res['DL_coverage_raw']:.3f}")
    print(f"wrote {OUT/'pooled_dl_eval.json'}")
