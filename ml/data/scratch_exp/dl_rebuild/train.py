"""dl_rebuild/train — stable, audited training loop (single 30-day horizon).

Early stops on the leak-free time-buffer val window ([origin-90d, origin-30d)),
restores the best state, records a per-epoch train-loss trace (like the track-3
attempt-2 audit) and clips gradients.  Standardisation is fit on the TRAIN
rows only (target too), so scores/Gates stay honest.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

MAX_EPOCHS, PATIENCE, BATCH, LR = 60, 10, 2048, 1e-3
HUBER_DELTA, CLIP = 1.0, 1.0


def _batch_loader(tr_ix: np.ndarray, B: int, seed: int):
    n = len(tr_ix)
    perm = np.random.default_rng(seed).permutation(n)
    for i in range(0, n, B):
        yield tr_ix[perm[i:i + B]]


def _dev(x: np.ndarray, device):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)


def train_mlp(model, Xf, ystd, tr_ix, va_ix, device, seed,
              max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR):
    """Xf (n,p) standardized; ystd (n,) standardized on train; both FULL length.
    Slices are moved to the device per batch (fits a 4 GB laptop GPU)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=2)
    lossfn = nn.HuberLoss(delta=HUBER_DELTA)
    best, best_state, bad, trace = float("inf"), None, 0, []
    for ep in range(max_epochs):
        model.train()
        run, nb = 0.0, 0
        for ix in _batch_loader(tr_ix, BATCH, seed * 1000 + ep):
            opt.zero_grad()
            loss = lossfn(model(_dev(Xf[ix], device)), _dev(ystd[ix], device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            run += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(
                model(_dev(Xf[va_ix], device)), _dev(ystd[va_ix], device)).sqrt())
        sched.step(v)
        trace.append((ep + 1, float(run / max(nb, 1)), v))
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite val RMSE (seed {seed})")
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t
                          in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("no improving epoch")
    model.load_state_dict(best_state)
    model.eval()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return model, {"best_val": best, "epochs": len(trace), "trace": trace}


def window_ix(G: np.ndarray, ix: np.ndarray, W: int) -> np.ndarray:
    """(len(ix), W, C) tray of each row's trailing W-window from the series."""
    p = np.asarray(ix, int)
    idx = p[:, None] - W + 1 + np.arange(W)[None, :]     # (B, W)
    return G[idx]


def train_gru(model, G, tr_ix, ftr, tr_fr, ytr, ytr_ix, G_va, va_ix,
              ftr_va, yva, yva_ix, W, device, seed,
              max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR):
    torch.manual_seed(seed)
    y_all = ytr                                   # numpy full length
    ym, ys = float(y_all[tr_ix].mean()), float(y_all[tr_ix].std() + 1e-6)

    def make_batch(ix):
        ix = np.asarray(ix, int)
        seq = torch.from_numpy(np.ascontiguousarray(
            window_ix(G, ix, W), dtype=np.float32)).to(device)
        sm = torch.isfinite(seq).float()
        ff = _dev(ftr[ix], device)
        yy = _dev((y_all[ix] - ym) / ys, device)
        return seq, sm, ff, yy

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=2)
    lossfn = nn.HuberLoss(delta=HUBER_DELTA)
    best, best_state, bad, trace = float("inf"), None, 0, []
    n = len(tr_ix)
    for ep in range(max_epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(n)
        run, nb = 0.0, 0
        for i in range(0, n, BATCH):
            seq, sm, ff, yy = make_batch(tr_ix[perm[i:i + BATCH]])
            opt.zero_grad()
            p = model(seq, sm, ff)
            loss = lossfn(p, yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            run += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            sse, cnt = 0.0, 0
            vix = np.asarray(va_ix, int)
            for j in range(0, len(vix), 2048):
                ch = vix[j:j + 2048]
                vseq = _dev(window_ix(G_va, ch, W), device)
                vf = _dev(ftr_va[ch], device)
                vyy = _dev((yva[ch] - ym) / ys, device)
                pv = model(vseq, torch.isfinite(vseq).float(), vf)
                sse += float(nn.functional.mse_loss(pv, vyy,
                                                    reduction="sum"))
                cnt += len(ch)
            v = math.sqrt(sse / max(cnt, 1))
        sched.step(v)
        trace.append((ep + 1, float(run / max(nb, 1)), v))
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite val RMSE (seed {seed})")
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t
                          in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("no improving epoch")
    model.load_state_dict(best_state)
    model.eval()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return model, {"best_val": best, "epochs": len(trace), "trace": trace}


def predict_mlp(model, X, ix, device, ym=0.0, ys=1.0):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X[ix]).to(device)
        p = []
        for i in range(0, len(ix), 8192):
            p.append(model(Xt[i:i + 8192]).cpu().numpy())
        return (np.concatenate(p) * ys + ym).astype(np.float64)


def predict_gru(model, G, ftr, ix, W, device, ym=0.0, ys=1.0):
    """Return unscaled deltas for rows ix (full windows required; caller ok)."""
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(ix), 2048):
            b = np.asarray(ix[i:i + 2048], int)
            seq = _dev(window_ix(G, b, W), device)
            out.append(model(seq, torch.isfinite(seq).float(),
                             _dev(ftr[b], device)).cpu().numpy())
        return (np.concatenate(out) * ys + ym).astype(np.float64)