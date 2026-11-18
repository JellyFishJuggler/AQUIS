"""Multi-fold, multi-seed held-out-station comparison (GNN vs matched GRU).

    python -m research.gnn.folds --device cuda

Protocol
--------
* **5 spatial k-means folds** (``n_blocks=5``). The block assignment is computed
  once with a **fixed** ``--block-seed`` (42) so a fold is the same station set
  for every training seed; the seed only changes the training run.
* For each fold the held-out stations are removed from **both** models' training
  set; early stopping uses seen stations only, exactly like
  ``train --holdout-block``. The baseline GRU therefore always trains on the
  *identical* station subset as the GNN.
* **3 training seeds** (42, 43, 44) per fold, same budget/optimiser/patience.
* Evaluation is the val split restricted to the held-out stations (no test
  period is touched, nothing is tuned).

Reported
--------
Per horizon: RMSE ratio ``gnn / gru`` (row-pooled, same definition as
``eval.md``), mean ± std across the 15 (fold, seed) runs, per-fold means, and a
station-level bootstrap CI (2,000 resamples of held-out stations, seeded) on the
row-pooled RMSE ratio. Overall CI comes from a nested bootstrap: resample
stations inside every fold, average the fold ratios.

Runs are cached under ``outputs/gnn_influence/folds/runs/`` so an interrupted
experiment resumes instead of retraining finished runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import graph as Gr
from .eval import (holdout_block_of, predict_all, subset_split, _subset_anchors)
from .train import fit

ROOT = Path(__file__).resolve().parents[2]          # ml/
OUT = ROOT / "outputs" / "gnn_influence"
RUNS = OUT / "folds" / "runs"
CKPT = ROOT / "models" / "research" / "folds"
HORIZONS = (4, 28, 60, 120)
ARCHS = ("gnn", "gru")


def _run_path(fold: int, seed: int, arch: str) -> Path:
    return RUNS / f"f{fold}_s{seed}_{arch}.json"


def _ckpt_path(fold: int, seed: int, arch: str) -> Path:
    return CKPT / f"f{fold}_s{seed}_{arch}.pt"


def run_metrics(pr: dict) -> dict:
    """Row-pooled metrics + per-station squared errors (for the bootstrap)."""
    h, y, st = pr["h"], pr["y"], pr["station"]
    q50 = pr["p"][:, 1]
    out: dict = {"pinball": float(pr["loss"]), "horizons": {}}
    for hh in HORIZONS:
        sel = h == hh
        if not sel.any():
            continue
        err2 = (q50[sel] - y[sel]) ** 2
        sse: dict[str, float] = {}
        cnt: dict[str, int] = {}
        for s in np.unique(st[sel]):
            m = st[sel] == s
            sse[str(s)] = float(err2[m].sum())
            cnt[str(s)] = int(m.sum())
        out["horizons"][str(hh)] = {
            "n": int(sel.sum()),
            "rmse": float(np.sqrt(err2.mean())),
            "mae": float(np.abs(q50[sel] - y[sel]).mean()),
            "station_sse": sse,
            "station_n": cnt,
        }
    return out


def _pooled(runs: list[dict], arch: str, h: str) -> float:
    use = [r for r in runs if r["arch"] == arch and h in r["horizons"]]
    sse = sum(sum(r["horizons"][h]["station_sse"].values()) for r in use)
    n = sum(sum(r["horizons"][h]["station_n"].values()) for r in use)
    return float(np.sqrt(sse / max(n, 1)))


def _ratio_from_sse(g_sse: np.ndarray, g_n: np.ndarray,
                    u_sse: np.ndarray, u_n: np.ndarray) -> np.ndarray:
    """Row-pooled RMSE ratio; works for 1-D point estimates and (B, S) samples."""
    g = np.sqrt(g_sse.sum(axis=-1) / np.maximum(g_n.sum(axis=-1), 1))
    u = np.sqrt(u_sse.sum(axis=-1) / np.maximum(u_n.sum(axis=-1), 1))
    return g / u


def _bootstrap_fold(g_runs: list[dict], u_runs: list[dict], h: str,
                    n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Station bootstrap of the row-pooled RMSE ratio for one fold.

    Per-station squared errors are averaged over the fold's seeds first, so the
    bootstrap reflects the fold, not a single training run.
    """
    stations = sorted(set(g_runs[0]["horizons"][h]["station_sse"]) &
                      set(u_runs[0]["horizons"][h]["station_sse"]))
    g_sse = np.array([np.mean([r["horizons"][h]["station_sse"][s] for r in g_runs])
                      for s in stations], float)
    g_n = np.array([np.mean([r["horizons"][h]["station_n"][s] for r in g_runs])
                    for s in stations], float)
    u_sse = np.array([np.mean([r["horizons"][h]["station_sse"][s] for r in u_runs])
                      for s in stations], float)
    u_n = np.array([np.mean([r["horizons"][h]["station_n"][s] for r in u_runs])
                    for s in stations], float)
    idx = rng.integers(0, len(stations), size=(n_boot, len(stations)))
    return _ratio_from_sse(g_sse[idx], g_n[idx], u_sse[idx], u_n[idx])


def aggregate(runs: list[dict], folds: int, seeds: list[int],
              n_boot: int = 2000, boot_seed: int = 0) -> dict:
    by = {(r["fold"], r["seed"], r["arch"]): r for r in runs}
    rng = np.random.default_rng(boot_seed)
    out: dict = {"n_runs": len(runs), "folds": folds, "seeds": seeds,
                 "n_boot": n_boot, "per_horizon": {}}

    for hh in HORIZONS:
        h = str(hh)
        rows = []                                  # per (fold, seed) ratio
        per_fold = {}
        for f in range(folds):
            g_runs = [by[(f, s, "gnn")] for s in seeds if (f, s, "gnn") in by]
            u_runs = [by[(f, s, "gru")] for s in seeds if (f, s, "gru") in by]
            if not g_runs or not u_runs:
                continue
            ratios = []
            for g, u in zip(g_runs, u_runs):
                if h not in g["horizons"] or h not in u["horizons"]:
                    continue
                ratios.append(g["horizons"][h]["rmse"] / u["horizons"][h]["rmse"])
            if not ratios:
                continue
            boot = _bootstrap_fold(g_runs, u_runs, h, n_boot, rng)
            per_fold[f] = {
                "n_unseen": int(g_runs[0]["n_unseen"]),
                "n_rows": int(g_runs[0]["horizons"][h]["n"]),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
                "ratios": [float(x) for x in ratios],
                "ci95": [float(np.percentile(boot, 2.5)),
                         float(np.percentile(boot, 97.5))],
                "p_gnn_better": float(np.mean(boot < 1.0)),
                "_boot": boot,
            }
            rows.extend(ratios)

        if not per_fold:
            continue
        # each fold contributes one 1-D bootstrap array of length n_boot
        boots = np.stack([v["_boot"] for v in per_fold.values()], axis=0)
        nested = boots.mean(axis=0)
        for v in per_fold.values():
            v.pop("_boot")
        fold_means = [v["ratio_mean"] for v in per_fold.values()]
        out["per_horizon"][h] = {
            "rmse_gnn": _pooled(runs, "gnn", h),
            "rmse_gru": _pooled(runs, "gru", h),
            "ratio_mean": float(np.mean(rows)),
            "ratio_std": float(np.std(rows, ddof=1)) if len(rows) > 1 else 0.0,
            "ratio_fold_mean_std": float(np.std(fold_means, ddof=1))
            if len(fold_means) > 1 else 0.0,
            "ratio_min": float(np.min(rows)),
            "ratio_max": float(np.max(rows)),
            "ci95": [float(np.percentile(nested, 2.5)),
                     float(np.percentile(nested, 97.5))],
            "p_gnn_better": float(np.mean(nested < 1.0)),
            "per_fold": per_fold,
        }

    # pinball (all horizons pooled per run, like eval.md)
    rows = []
    per_fold = {}
    for f in range(folds):
        g_runs = [by[(f, s, "gnn")] for s in seeds if (f, s, "gnn") in by]
        u_runs = [by[(f, s, "gru")] for s in seeds if (f, s, "gru") in by]
        if not g_runs or not u_runs:
            continue
        r = [g["pinball"] / u["pinball"] for g, u in zip(g_runs, u_runs)]
        per_fold[f] = {"ratio_mean": float(np.mean(r)),
                       "ratio_std": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0}
        rows.extend(r)
    if rows:
        out["pinball"] = {
            "ratio_mean": float(np.mean(rows)),
            "ratio_std": float(np.std(rows, ddof=1)) if len(rows) > 1 else 0.0,
            "per_fold": per_fold,
        }
    return out


def render_md(res: dict, meta: dict) -> str:
    L = ["# Held-out-station comparison — 5 spatial folds × 3 seeds", "",
         f"Generated: {meta['generated_at']}  |  device: `{meta['device']}`", "",
         "> **Research only.** Val split restricted to the held-out stations. "
         "Nothing tuned on the test period.", "",
         f"Protocol: {res['folds']} spatial k-means folds (block seed "
         f"`{meta['block_seed']}`), {len(res['seeds'])} training seeds "
         f"{res['seeds']}, identical station subset for both models, "
         f"{meta['epochs']} max epochs / patience {meta['patience']}.", "",
         "`ratio = RMSE(gnn) / RMSE(gru)` — below 1 means the GNN wins. "
         f"CI = station bootstrap ({res['n_boot']} resamples, nested over folds).",
         ""]

    L += ["## Per horizon", "",
          "| h | RMSE gnn | RMSE gru | ratio mean ± std (15 runs) | "
          "std of fold means | 95% CI | P(gnn better) | min | max |",
          "|---|---|---|---|---|---|---|---|---|"]
    for hh in HORIZONS:
        b = res["per_horizon"].get(str(hh))
        if not b:
            continue
        L.append(f"| {hh / 4:.0f}d | {b['rmse_gnn']:.4f} | {b['rmse_gru']:.4f} | "
                 f"{b['ratio_mean']:.4f} ± {b['ratio_std']:.4f} | "
                 f"{b['ratio_fold_mean_std']:.4f} | "
                 f"[{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] | "
                 f"{b['p_gnn_better']:.2f} | {b['ratio_min']:.4f} | "
                 f"{b['ratio_max']:.4f} |")

    L += ["", "## Per fold", "",
          "| fold | unseen | " + " | ".join(f"{h / 4:.0f}d" for h in HORIZONS) + " |",
          "|---|---|" + "---|" * len(HORIZONS)]
    for f in sorted({int(k) for b in res["per_horizon"].values() for k in b["per_fold"]}):
        cells, unseen = [], ""
        for hh in HORIZONS:
            pf = res["per_horizon"].get(str(hh), {}).get("per_fold", {})
            b = pf.get(f, pf.get(str(f)))      # int keys in memory, str in JSON
            if not b:
                cells.append("—")
                continue
            unseen = b["n_unseen"]
            cells.append(f"{b['ratio_mean']:.4f} ± {b['ratio_std']:.4f}")
        L.append(f"| {f} | {unseen} | " + " | ".join(cells) + " |")

    if "pinball" in res:
        p = res["pinball"]
        L += ["", "## Pinball loss", "",
              f"ratio `gnn/gru` = **{p['ratio_mean']:.4f} ± {p['ratio_std']:.4f}** "
              f"(15 runs).", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-fold held-out-station comparison.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--block-seed", type=int, default=42,
                    help="fixed seed for the spatial block assignment (fold identity)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="retrain cached runs")
    ap.add_argument("--out-json", default=str(OUT / "folds.json"))
    ap.add_argument("--out-md", default=str(OUT / "folds.md"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device={device} folds={args.folds} seeds={seeds} "
          f"block_seed={args.block_seed}", flush=True)

    axis = D.load_axis()
    fmean, fstd = D.feature_stats(axis)
    axis = D.scale_axis(axis, fmean, fstd)
    graph = Gr.ensure_graph()
    G = Gr.graph_arrays(graph)
    G["graph_nodes"] = graph["nodes"]
    ea, _, _ = D.edge_attr_from_km(G["edge_km"])
    G["edge_attr"] = ea
    sp_tr = D.scale_split(D.load_split("train", axis), fmean, fstd)
    sp_va = D.scale_split(D.load_split("val", axis), fmean, fstd)

    block = holdout_block_of(G, n_blocks=args.folds, seed=args.block_seed)
    stations = np.asarray(G["stations"])
    sizes = [int((block == f).sum()) for f in range(args.folds)]
    print(f"fold sizes: {sizes} (total {sum(sizes)})", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t_start = time.time()
    done = 0
    total = args.folds * len(seeds) * len(ARCHS)

    for f in range(args.folds):
        unseen = set(stations[block == f])
        seen = set(stations) - unseen
        sp_tr_f = _subset_anchors(
            sp_tr, np.array([s in seen for s in sp_tr["a_station"]]))
        sp_va_seen = subset_split(sp_va, seen)
        sp_va_unseen = subset_split(sp_va, unseen)
        print(f"fold {f}: unseen={len(unseen)} seen={len(seen)} "
              f"train_anchors={sp_tr_f['A']:,} unseen_rows={sp_va_unseen['A']:,}",
              flush=True)

        for seed in seeds:
            for arch in ARCHS:
                done += 1
                p = _run_path(f, seed, arch)
                if p.exists() and not args.force:
                    r = json.loads(p.read_text(encoding="utf-8"))
                    print(f"  [{done}/{total}] f{f} s{seed} {arch}: cached", flush=True)
                    results.append(r)
                    continue
                t0 = time.time()
                model, best, best_ep, hist, info = fit(
                    arch, seed, sp_tr_f, sp_va_seen, axis, G, device,
                    epochs=args.epochs, patience=args.patience, nodes=args.nodes,
                    hidden=args.hidden,
                    log=lambda m: print(f"    {m}", flush=True))
                pr = predict_all(model, axis, sp_va_unseen, G, device, args.nodes)
                r = {"fold": f, "seed": seed, "arch": arch,
                     "n_unseen": len(unseen),
                     "best_val_loss": float(best), "best_epoch": int(best_ep),
                     "train_anchors": info["train_anchors"],
                     "train_snapshots": info["train_snapshots"],
                     "seconds": round(time.time() - t0, 1),
                     "feat_mean": [float(x) for x in fmean],
                     "feat_std": [float(x) for x in fstd],
                     **run_metrics(pr)}
                p.write_text(json.dumps(r), encoding="utf-8")
                torch.save(model.state_dict(), _ckpt_path(f, seed, arch))
                results.append(r)
                eta = (time.time() - t_start) / done * (total - done) / 60
                print(f"  [{done}/{total}] f{f} s{seed} {arch}: best {best:.4f} @ "
                      f"{best_ep} in {r['seconds']:.0f}s  "
                      f"30d rmse {r['horizons'].get('120', {}).get('rmse', float('nan')):.4f}  "
                      f"ETA {eta:.0f} min", flush=True)

    res = aggregate(results, args.folds, seeds)
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device), "block_seed": args.block_seed,
            "epochs": args.epochs, "patience": args.patience,
            "nodes": args.nodes, "hidden": args.hidden,
            "block_sizes": sizes, "seconds": round(time.time() - t_start, 1),
            "note": ("Research only. Both models trained on the identical seen "
                     "station subset; early stopping on seen stations; val split "
                     "restricted to the held-out block for scoring.")}
    res["meta"] = meta
    Path(args.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(args.out_md).write_text(render_md(res, meta), encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}", flush=True)
    for hh in HORIZONS:
        b = res["per_horizon"].get(str(hh))
        if b:
            print(f"  {hh / 4:.0f}d ratio {b['ratio_mean']:.4f} ± "
                  f"{b['ratio_std']:.4f}  CI [{b['ci95'][0]:.4f}, "
                  f"{b['ci95'][1]:.4f}]", flush=True)


if __name__ == "__main__":
    main()
