"""34_train_traj_dl — sequence GRU + quantile heads for the trajectory v2 DL.

Replaces the incumbent pooled direct multi-horizon XGBoost
(31_train_traj.py -> models/traj_xgb_q{05,50,95}.json) with a single pooled
PyTorch model.  Same frame, same target, same split as the incumbent so the
backtest is apples-to-apples.

Design (locked):
  * anchor-centric: one history window per (station, t) anchor -> encode ONCE;
    each anchor's observed (h, target_delta) pairs are scored by the quantile
    heads.  No 5.9M x 120 sequence matrix is ever materialised.
  * sequence = the last W observed 6h rows on the station's own axis
    (features_6h union), left-zero-padded + masked when history < W.
  * one encoder + 3 quantile heads (q05/q50/q95), pinball loss, post-hoc sort.

Targets/splits are read from the incumbent long-frame
(data/features_traj/{train,val}.parquet, built by 30_traj_datasets.py) so the
per-anchor+horizon time purge is inherited, not re-implemented.

Outputs -> models/traj_dl.pt + models/traj_dl_config.json  (NEVER overwrites the
incumbent traj_xgb_*.json).
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

F = importlib.import_module("06_features")

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


def main():
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


if __name__ == "__main__":
    main()
