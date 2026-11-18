"""Train the research GNN (and the matched-budget GRU baseline).

    python -m research.gnn.train                       # GNN on all stations
    python -m research.gnn.train --arch gru --out models/research/gru_matched.pt
    python -m research.gnn.train --holdout-block 0 --out models/research/gnn_traj_holdout.pt
    python -m research.gnn.train --arch gru --holdout-block 0 --out models/research/gru_matched_holdout.pt

Hard constraints honoured here:

* the split is the existing purged one — train/val are read from
  ``data/features_traj`` through ``research.gnn.data`` (which reproduces
  ``pipelines.traj_dl._load_split`` exactly); nothing is re-split;
* the encoder starts from fresh weights — ``models/traj_dl.pt`` is never read;
* checkpoints are written only under ``models/research/``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import graph as Gr
from . import model as M
from .eval import (config_path, holdout_block_of, predict_all, subset_split,
                   _subset_anchors)

ROOT = Path(__file__).resolve().parents[2]          # ml/
RESEARCH = ROOT / "models" / "research"
W = 120
SPLIT_NOTE = {"train": "< 2025-10-01", "val": "2025-10-01..2025-12-31",
              "test": ">= 2026-01-01", "purging": "labels crossing a boundary removed"}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(arch: str, nfeat: int, hidden: int, seed: int) -> torch.nn.Module:
    if arch == "gnn":
        return M.GNNTraj(nfeat=nfeat, hidden=hidden, static=nfeat)
    tdl = importlib.import_module("pipelines.traj_dl")
    return M.GRUAdapter(tdl.TrajGRU(nfeat, hidden=hidden))


def fit(arch: str, seed: int, sp_tr: dict, sp_va_fit: dict, axis: dict, G: dict,
        device, *, epochs: int = 20, patience: int = 4, nodes: int = 1024,
        hidden: int = 64, lr: float = 2e-3, weight_decay: float = 1e-4,
        max_snapshots: int = 0, log=print) -> tuple[torch.nn.Module, float, int, list, dict]:
    """Train one model on ``sp_tr``, early-stopping on ``sp_va_fit``.

    Shared by ``main`` and the multi-fold experiment so both use exactly the
    same protocol (fresh weights, same optimiser, same early-stopping rule).
    Returns ``(model_at_best_epoch, best_val_loss, best_epoch, history, info)``.
    """
    set_seed(seed)
    groups = D.snapshot_groups(sp_tr)
    if max_snapshots and len(groups) > max_snapshots:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(groups), max_snapshots, replace=False))
        groups = [groups[i] for i in sel]
    info = {"train_snapshots": len(groups),
            "train_anchors": int(sum(len(g) for g in groups))}
    model = build(arch, len(D.NUM_COLS), hidden, seed).to(device)
    log(f"parameters={M.count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2)

    best, bad, best_state, best_ep = float("inf"), 0, None, 0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + ep)
        batches = D.pack_groups(groups, nodes, rng)
        run, nb = 0.0, 0
        t0 = time.time()
        for b in batches:
            t = D.build_batch(b, axis, sp_tr, W, G["node_of"], G["edge_i"],
                              G["edge_j"], G["edge_attr"], G["n"], device)
            pred = model(t["seq"], t["seqm"], t["static"], t["hf"],
                         t["edge_index"], t["edge_attr"])
            loss = M.pinball(pred, t["delta"], t["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss.detach())
            nb += 1
        tr_loss = run / max(nb, 1)

        vp = predict_all(model, axis, sp_va_fit, G, device, nodes)
        va_loss = vp["loss"]
        sched.step(va_loss)
        h120 = next((r for r in D.HORIZONS_EVAL if r == 120), 120)
        sel = vp["h"] == h120
        rmse30 = (float(np.sqrt(np.mean((vp["p"][sel, 1] - vp["y"][sel]) ** 2)))
                  if sel.any() else float("nan"))
        dt = time.time() - t0
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_rmse30": rmse30, "seconds": round(dt, 1)})
        log(f"ep {ep:02d} train {tr_loss:.4f} val {va_loss:.4f} "
            f"rmse30 {rmse30:.4f} {dt:.1f}s")

        if va_loss < best - 1e-6:
            best, bad, best_ep = va_loss, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop at epoch {ep} (best {best:.4f} @ {best_ep})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_ep, history, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the research GNN / matched GRU.")
    ap.add_argument("--arch", choices=("gnn", "gru"), default="gnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=1024, help="target nodes per step")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-snapshots", type=int, default=0,
                    help="cap the number of training snapshots (0 = all)")
    ap.add_argument("--holdout-block", type=int, default=-1,
                    help="spatial k-means block to remove from training (-1 = none)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    name = args.out or str(RESEARCH / (
        "gnn_traj.pt" if args.arch == "gnn" else "gru_matched.pt"))
    out = Path(name)
    if out.parent.name != "research":
        raise SystemExit(f"refusing to write outside models/research/: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"arch={args.arch} device={device} torch={torch.__version__}", flush=True)
    axis = D.load_axis()
    fmean, fstd = D.feature_stats(axis)          # train-only statistics
    axis = D.scale_axis(axis, fmean, fstd)
    graph = Gr.ensure_graph()
    G = Gr.graph_arrays(graph)
    G["graph_nodes"] = graph["nodes"]
    ea, emean, estd = D.edge_attr_from_km(G["edge_km"])
    G["edge_attr"] = ea

    sp_tr = D.scale_split(D.load_split("train", axis), fmean, fstd)
    sp_va = D.scale_split(D.load_split("val", axis), fmean, fstd)

    holdout_block = int(args.holdout_block)
    unseen: set[str] = set()
    seen: set[str] = set(G["stations"])
    if holdout_block >= 0:
        block = holdout_block_of(G, seed=args.seed)
        unseen = {G["stations"][i] for i in np.flatnonzero(block == holdout_block)}
        seen = set(G["stations"]) - unseen
        sp_tr = _subset_anchors(sp_tr, np.array([s in seen for s in sp_tr["a_station"]]))
        # early stopping on *seen* stations only — the unseen block is never used
        # for model selection, so the held-out table stays honest.
        sp_va_fit = subset_split(sp_va, seen)
        print(f"holdout block {holdout_block}: unseen={len(unseen)} seen={len(seen)}",
              flush=True)
    else:
        sp_va_fit = sp_va

    n_groups_est = len(D.snapshot_groups(sp_tr))
    if args.max_snapshots:
        n_groups_est = min(n_groups_est, args.max_snapshots)
    print(f"train snapshots<={n_groups_est:,} val anchors={sp_va_fit['A']:,}",
          flush=True)

    model, best, best_ep, history, info = fit(
        args.arch, args.seed, sp_tr, sp_va_fit, axis, G, device,
        epochs=args.epochs, patience=args.patience, nodes=args.nodes,
        hidden=args.hidden, lr=args.lr, weight_decay=args.weight_decay,
        max_snapshots=args.max_snapshots,
        log=lambda m: print(m, flush=True))
    n_anchors = info["train_anchors"]
    torch.save(model.state_dict(), out)

    # final val metrics from the selected checkpoint
    vp = predict_all(model, axis, sp_va_fit, G, device, args.nodes)
    from .eval import calibrated_metrics, horizon_metrics
    cfg = {
        "arch": "GRUAdapter" if args.arch == "gru" else "GNNTraj",
        "nfeat": len(D.NUM_COLS), "hidden": args.hidden,
        "static": len(D.NUM_COLS), "nq": 3,
        "gat_layers": 2, "gat_heads": 2, "dropout": 0.1,
        "quantiles": list(D.QUANTILES),
        "window_w": W, "max_h": D.MAX_H,
        "feat_mean": [float(x) for x in fmean],
        "feat_std": [float(x) for x in fstd],
        "feat_scaling": ("train-only z-score, clipped to +/-%.0f sigma; "
                         "gru_production is scored unscaled" % D.CLIP_SIGMA),
        "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
        "nodes_per_batch": args.nodes, "lr": args.lr,
        "weight_decay": args.weight_decay, "max_snapshots": args.max_snapshots,
        "split": SPLIT_NOTE,
        "target": "GWL(t+h*6h) - GWL(t); level = anchor + delta (anchor observed only)",
        "train_anchors": n_anchors, "train_snapshots": info["train_snapshots"],
        "val_anchors": int(sp_va_fit["A"]),
        "holdout_block": holdout_block,
        "holdout_unseen_stations": len(unseen),
        "edge_km_mean": emean, "edge_km_std": estd,
        "graph": {"path": "outputs/gnn_influence/graph.json",
                  "n_nodes": graph["meta"]["n_nodes"],
                  "n_edges": graph["meta"]["n_edges"]},
        "best_val_loss": best, "best_epoch": best_ep,
        "val_metrics": horizon_metrics(vp),
        "val_calibration": calibrated_metrics(vp),
        "history": history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "date": time.strftime("%Y-%m-%d"),
        "note": ("Research-only checkpoint. Production models in ml/models/ "
                 "(traj_dl.pt, pooled_dl.pt, XGB) were not read or written."),
    }
    cp = config_path(out)
    cp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"saved {out} and {cp}")
    print(f"best val loss {best:.4f} @ epoch {best_ep}")


if __name__ == "__main__":
    main()
