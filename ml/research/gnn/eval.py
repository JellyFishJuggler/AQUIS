"""Evaluation for the research GNN — honest, identical-row comparison.

Metrics: RMSE, MAE, bias, direction accuracy, calibrated 90% coverage +
interval half-width, median station NSE, at h = 4 / 28 / 60 / 120
(1d / 7d / 15d / 30d on the 6-hourly grid).

Two protocols:

* **temporal** — both models score the same anchors of the purged ``val`` split
  (2025-10-01 .. 2025-12-31) and, secondarily, the 2026 test anchors built in
  memory from ``data/features_6h/test.parquet``.
* **held-out stations** — a spatial k-means block (~20% of stations) is removed
  from training for *both* models; both are scored on that block's val anchors
  and reported explicitly as the unseen-station test.

Nothing here tunes on the test period: the checkpoint is the one early-stopped
on val.

    python -m research.gnn.eval                 # temporal val
    python -m research.gnn.eval --test          # + in-memory 2026 test
    python -m research.gnn.eval --holdout       # + held-out-station table
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
from . import model as M

ROOT = Path(__file__).resolve().parents[2]          # ml/
OUT = ROOT / "outputs" / "gnn_influence"
RESEARCH = ROOT / "models" / "research"
PROD_TRAJ = ROOT / "models" / "traj_dl.pt"
PROD_CFG = ROOT / "models" / "traj_dl_config.json"

W = 120
TARGET_COVERAGE = 0.90
WIDEN_GRID = np.arange(0.50, 3.001, 0.01)


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def config_path(ckpt: Path) -> Path:
    return ckpt.with_name(ckpt.stem + "_config.json")


def load_checkpoint(ckpt: Path, device) -> tuple[torch.nn.Module, dict]:
    ckpt = Path(ckpt)
    cfg = json.loads(config_path(ckpt).read_text(encoding="utf-8"))
    if cfg.get("arch") == "GRUAdapter":
        import importlib
        tdl = importlib.import_module("pipelines.traj_dl")
        net = tdl.TrajGRU(int(cfg["nfeat"]), hidden=int(cfg["hidden"]))
        model = M.GRUAdapter(net)
    else:
        model = M.build_model(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


def load_production_gru(device) -> tuple[torch.nn.Module, dict]:
    """The currently served trajectory GRU (read-only, CPU weights)."""
    import importlib
    tdl = importlib.import_module("pipelines.traj_dl")
    cfg = json.loads(PROD_CFG.read_text(encoding="utf-8"))
    net = tdl.TrajGRU(len(cfg["feature_cols"]), hidden=int(cfg["hidden"]))
    net.load_state_dict(torch.load(PROD_TRAJ, map_location="cpu", weights_only=True))
    net.eval()
    return M.GRUAdapter(net).to(device).eval(), cfg


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_all(model: torch.nn.Module, axis: dict, sp: dict, G: dict, device,
                batch_nodes: int = 1024, include_edges: bool = True) -> dict:
    """Predict every anchor in ``sp``, grouped into timestamp snapshots."""
    model.eval()
    batches = D.pack_groups(D.snapshot_groups(sp), batch_nodes)
    H, Y, P, A = [], [], [], []
    tot_loss = 0.0
    for b in batches:
        t = D.build_batch(b, axis, sp, W, G["node_of"], G["edge_i"], G["edge_j"],
                          G["edge_attr"], G["n"], device, include_edges=include_edges)
        pred = model(t["seq"], t["seqm"], t["static"], t["hf"],
                     t["edge_index"], t["edge_attr"])
        tot_loss += float(M.pinball(pred, t["delta"], t["mask"]))
        m = t["mask"].cpu().numpy() > 0
        rows, cols = np.nonzero(m)
        hv = t["h"].cpu().numpy()
        dv = t["delta"].cpu().numpy()
        pv = pred.cpu().numpy()
        av = t["anchor"].cpu().numpy()
        H.append(hv[rows, cols].astype(np.int16))
        Y.append(dv[rows, cols])
        P.append(pv[rows, cols])
        A.append(av[rows])
    h = np.concatenate(H)
    y = np.concatenate(Y)
    p = np.sort(np.concatenate(P), axis=1)          # ordering guarantee
    a = np.concatenate(A)
    return {"h": h, "y": y, "p": p, "anchor": a,
            "station": sp["a_station"][a], "time": sp["a_time"][a],
            "loss": tot_loss / max(len(batches), 1)}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _nse_median(y: np.ndarray, p50: np.ndarray, station: np.ndarray,
                min_rows: int = 10) -> tuple[float, int]:
    vals = []
    for s in np.unique(station):
        sel = station == s
        if int(sel.sum()) < min_rows:
            continue
        yy, pp = y[sel], p50[sel]
        sst = float(np.sum((yy - yy.mean()) ** 2))
        if sst <= 1e-12:
            continue
        vals.append(1.0 - float(np.sum((yy - pp) ** 2)) / sst)
    if not vals:
        return float("nan"), 0
    return float(np.median(vals)), len(vals)


def horizon_metrics(pred: dict, horizons=D.HORIZONS_EVAL) -> list[dict]:
    rows = []
    for h in horizons:
        sel = pred["h"] == h
        n = int(sel.sum())
        if n == 0:
            continue
        y = pred["y"][sel]
        p = pred["p"][sel]
        p50 = p[:, 1]
        e = p50 - y
        nse, n_st = _nse_median(y, p50, pred["station"][sel])
        rows.append({
            "h": int(h), "days": round(h / 4.0, 2), "n": n, "n_stations_nse": n_st,
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "direction_acc": float(np.mean(np.sign(p50) == np.sign(y))),
            "coverage_raw": float(np.mean((p[:, 0] <= y) & (y <= p[:, 2]))),
            "halfwidth_median_raw": float(np.median((p[:, 2] - p[:, 0]) / 2)),
            "nse_median": nse,
        })
    return rows


def _coverage(y: np.ndarray, p: np.ndarray, s: float) -> float:
    p50 = p[:, 1]
    lo = p50 - s * (p50 - p[:, 0])
    hi = p50 + s * (p[:, 2] - p50)
    return float(np.mean((lo <= y) & (y <= hi)))


def _halfwidth(y: np.ndarray, p: np.ndarray, s: float) -> float:
    p50 = p[:, 1]
    lo = p50 - s * (p50 - p[:, 0])
    hi = p50 + s * (p[:, 2] - p50)
    return float(np.median((hi - lo) / 2))


def calibrated_metrics(pred: dict, horizons=D.HORIZONS_EVAL,
                       seed: int = 42) -> list[dict]:
    """Widening fitted on half the rows, coverage reported on the other half."""
    rows = []
    rng = np.random.default_rng(seed)
    for h in horizons:
        sel = pred["h"] == h
        n = int(sel.sum())
        if n < 20:
            continue
        y, p = pred["y"][sel], pred["p"][sel]
        idx = rng.permutation(n)
        cut = n // 2
        fit, ho = idx[:cut], idx[cut:]
        best_s, best_d = 1.0, float("inf")
        for s in WIDEN_GRID:
            d = abs(_coverage(y[fit], p[fit], s) - TARGET_COVERAGE)
            if d < best_d:
                best_d, best_s = d, float(s)
        rows.append({
            "h": int(h), "days": round(h / 4.0, 2), "n_fit": int(cut),
            "n_heldout": int(n - cut), "widening_s": best_s,
            "coverage_fit": _coverage(y[fit], p[fit], best_s),
            "coverage": _coverage(y[ho], p[ho], best_s),
            "halfwidth_median": _halfwidth(y[ho], p[ho], best_s),
        })
    return rows


def bootstrap_rmse_ratio(pred_a: dict, pred_b: dict, horizon: int,
                         n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired station-level bootstrap of RMSE(a)/RMSE(b) at one horizon.

    Stations are resampled with replacement; both models are scored on exactly
    the same rows, so the pairing is exact.
    """
    sa = pred_a["h"] == horizon
    sb = pred_b["h"] == horizon
    if int(sa.sum()) != int(sb.sum()) or not np.array_equal(pred_a["y"][sa], pred_b["y"][sb]):
        raise ValueError("models are not scored on identical rows")
    y = pred_a["y"][sa]
    ea = (pred_a["p"][sa][:, 1] - y) ** 2
    eb = (pred_b["p"][sb][:, 1] - y) ** 2
    st = pred_a["station"][sa]
    uniq, inv = np.unique(st, return_inverse=True)
    k = len(uniq)
    sse_a = np.bincount(inv, weights=ea, minlength=k)
    sse_b = np.bincount(inv, weights=eb, minlength=k)
    cnt = np.bincount(inv, minlength=k).astype(np.float64)
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, size=k)
        na = float(np.sum(sse_a[pick]))
        nb = float(np.sum(sse_b[pick]))
        ratios[i] = np.sqrt(na / max(nb, 1e-12))
    return {
        "h": int(horizon), "days": round(horizon / 4.0, 2), "n_stations": int(k),
        "ratio_median": float(np.median(ratios)),
        "ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        "p_a_better": float(np.mean(ratios < 1.0)),
        "a": "gnn", "b": "gru",
    }


# --------------------------------------------------------------------------- #
# split helpers
# --------------------------------------------------------------------------- #
def holdout_block_of(G: dict, n_blocks: int = 5, seed: int = 42) -> np.ndarray:
    """Spatial k-means block label per graph node (reuses the existing helper)."""
    import importlib.util
    p = ROOT / "experiments" / "validation" / "_spatial_folds.py"
    spec = importlib.util.spec_from_file_location("_spatial_folds", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # only the pure function is used
    nodes = G["graph_nodes"]
    coords = np.array([[n["lat"], n["lon"]] for n in nodes], dtype=float)
    return mod.spatial_block_labels(coords, n_blocks=n_blocks, random_state=seed)


def subset_split(sp: dict, stations: set[str]) -> dict:
    """Anchors whose station is in ``stations`` (new dict, same conventions)."""
    keep = np.array([str(s) in stations for s in sp["a_station"]])
    return _subset_anchors(sp, keep)


def _subset_anchors(sp: dict, keep: np.ndarray) -> dict:
    idx = np.flatnonzero(keep)
    ptr_old, h_old, d_old = sp["ptr"], sp["h"], sp["delta"]
    lens = ptr_old[idx + 1] - ptr_old[idx]
    ptr = np.r_[0, np.cumsum(lens)]
    h = np.concatenate([h_old[ptr_old[a]:ptr_old[a + 1]] for a in idx])
    d = np.concatenate([d_old[ptr_old[a]:ptr_old[a + 1]] for a in idx])
    return {"a_axis": sp["a_axis"][idx], "a_runstart": sp["a_runstart"][idx],
            "a_static": sp["a_static"][idx], "a_static_m": sp["a_static_m"][idx],
            "ptr": ptr, "h": h, "delta": d, "A": len(idx),
            "a_station": sp["a_station"][idx], "a_time": sp["a_time"][idx]}


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _md_table(rows: list[dict], cols: list[tuple[str, str]], fmt: dict) -> str:
    head = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key)
            f = fmt.get(key)
            if v is None:
                cells.append("—")
            elif f:
                cells.append(f.format(v))
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the research GNN.")
    ap.add_argument("--test", action="store_true", help="also score in-memory 2026 test")
    ap.add_argument("--holdout", action="store_true", help="also score held-out stations")
    ap.add_argument("--test-stride", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-nodes", type=int, default=1024)
    ap.add_argument("--out", default=str(OUT / "eval.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    print(f"device={device}", flush=True)

    axis = D.load_axis()
    graph = Gr.ensure_graph()
    G = Gr.graph_arrays(graph)
    G["graph_nodes"] = graph["nodes"]
    ea, emean, estd = D.edge_attr_from_km(G["edge_km"])
    G["edge_attr"] = ea

    gnn, gcfg = load_checkpoint(RESEARCH / "gnn_traj.pt", device)
    stats = D.stats_from_cfg(gcfg)
    if stats is None:                            # legacy checkpoint: no scaling
        fmean = np.zeros(len(D.NUM_COLS), np.float32)
        fstd = np.ones(len(D.NUM_COLS), np.float32)
        print("warning: checkpoint has no feat_mean/feat_std, scoring unscaled", flush=True)
    else:
        fmean, fstd = stats
    axis_raw = axis
    axis = D.scale_axis(axis_raw, fmean, fstd)

    prod, pcfg = load_production_gru(device)
    models = {"gnn": gnn, "gru_production": prod}
    matched_path = RESEARCH / "gru_matched.pt"
    if matched_path.exists():
        models["gru_matched"] = load_checkpoint(matched_path, device)[0]

    def views(sp_raw: dict) -> dict[str, tuple[dict, dict]]:
        """Per-model (axis, split): research models scaled, production raw."""
        sp_scaled = D.scale_split(sp_raw, fmean, fstd)
        return {"gnn": (axis, sp_scaled), "gru_matched": (axis, sp_scaled),
                "gru_production": (axis_raw, sp_raw)}

    result: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "graph": graph["meta"],
        "horizons": list(D.HORIZONS_EVAL),
        "checkpoints": {"gnn": gcfg, "gru_production": {
            "hidden": pcfg.get("hidden"), "trained_at": pcfg.get("trained_at")}},
        "splits": {},
    }

    # ---------------- temporal: val ----------------
    sp_val_raw = D.load_split("val", axis_raw)
    vv = views(sp_val_raw)
    val: dict = {}
    preds: dict[str, dict] = {}
    for label, mdl in models.items():
        t0 = time.time()
        ax, spv = vv[label]
        preds[label] = predict_all(mdl, ax, spv, G, device, args.batch_nodes)
        val[label] = {
            "n_rows": int(len(preds[label]["h"])),
            "pinball": round(float(preds[label]["loss"]), 6),
            "metrics": horizon_metrics(preds[label]),
            "calibration": calibrated_metrics(preds[label]),
            "seconds": round(time.time() - t0, 1),
        }
    val["bootstrap_vs_production"] = [
        bootstrap_rmse_ratio(preds["gnn"], preds["gru_production"], h)
        for h in D.HORIZONS_EVAL if int((preds["gnn"]["h"] == h).sum())]
    if "gru_matched" in preds:
        val["bootstrap_vs_matched"] = [
            bootstrap_rmse_ratio(preds["gnn"], preds["gru_matched"], h)
            for h in D.HORIZONS_EVAL if int((preds["gnn"]["h"] == h).sum())]
    result["splits"]["val"] = val
    print("val done", flush=True)

    # ---------------- temporal: in-memory 2026 test ----------------
    if args.test:
        sp_test_raw = D.build_test_split(axis_raw, stride=args.test_stride)
        tv = views(sp_test_raw)
        test: dict = {}
        tpred: dict[str, dict] = {}
        for label, mdl in models.items():
            t0 = time.time()
            ax, spv = tv[label]
            tpred[label] = predict_all(mdl, ax, spv, G, device, args.batch_nodes)
            test[label] = {
                "n_rows": int(len(tpred[label]["h"])),
                "pinball": round(float(tpred[label]["loss"]), 6),
                "metrics": horizon_metrics(tpred[label]),
                "calibration": calibrated_metrics(tpred[label]),
                "seconds": round(time.time() - t0, 1),
            }
        test["bootstrap_vs_production"] = [
            bootstrap_rmse_ratio(tpred["gnn"], tpred["gru_production"], h)
            for h in D.HORIZONS_EVAL if int((tpred["gnn"]["h"] == h).sum())]
        if "gru_matched" in tpred:
            test["bootstrap_vs_matched"] = [
                bootstrap_rmse_ratio(tpred["gnn"], tpred["gru_matched"], h)
                for h in D.HORIZONS_EVAL if int((tpred["gnn"]["h"] == h).sum())]
        result["splits"]["test_2026"] = test
        print("2026 test done", flush=True)

    # ---------------- held-out stations ----------------
    if args.holdout:
        ho_path = RESEARCH / "gnn_traj_holdout.pt"
        ho_gru_path = RESEARCH / "gru_matched_holdout.pt"
        if not ho_path.exists() or not ho_gru_path.exists():
            print("skip holdout: checkpoints missing", flush=True)
        else:
            block = holdout_block_of(G)
            ho_cfg = json.loads(config_path(ho_path).read_text(encoding="utf-8"))
            ho_block = int(ho_cfg["holdout_block"])
            unseen = {G["stations"][i] for i in np.flatnonzero(block == ho_block)}
            sp_unseen_raw = subset_split(sp_val_raw, unseen)
            hp: dict[str, dict] = {}
            for label, ckpt in (("gnn_holdout", ho_path),
                                ("gru_matched_holdout", ho_gru_path)):
                mdl, cfg_h = load_checkpoint(ckpt, device)
                st = D.stats_from_cfg(cfg_h)
                ax_h, sp_h = ((axis_raw, sp_unseen_raw) if st is None
                              else (D.scale_axis(axis_raw, *st),
                                    D.scale_split(sp_unseen_raw, *st)))
                hp[label] = predict_all(mdl, ax_h, sp_h, G, device, args.batch_nodes)
            ho: dict = {"holdout_block": ho_block, "n_unseen_stations": len(unseen),
                        "n_rows": int(len(hp["gnn_holdout"]["h"])), "models": {}}
            for label, pr in hp.items():
                ho["models"][label] = {"n_rows": int(len(pr["h"])),
                                       "pinball": round(float(pr["loss"]), 6),
                                       "metrics": horizon_metrics(pr),
                                       "calibration": calibrated_metrics(pr)}
            ho["bootstrap_gnn_vs_gru"] = [
                bootstrap_rmse_ratio(hp["gnn_holdout"], hp["gru_matched_holdout"], h)
                for h in D.HORIZONS_EVAL if int((hp["gnn_holdout"]["h"] == h).sum())]
            result["splits"]["heldout_stations"] = ho
            print("held-out done", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    md = render_md(result)
    Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.md')}")
    print(md)


def render_md(result: dict) -> str:
    cols = [("days", "h"), ("n", "n"), ("rmse", "RMSE"), ("mae", "MAE"),
            ("bias", "bias"), ("direction_acc", "dir acc"),
            ("coverage", "cov90*"), ("halfwidth_median", "half-width*"),
            ("nse_median", "median NSE")]
    fmt = {"days": "{:.0f}d", "n": "{:,}", "rmse": "{:.4f}", "mae": "{:.4f}",
           "bias": "{:+.4f}", "direction_acc": "{:.3f}", "coverage": "{:.3f}",
           "halfwidth_median": "{:.4f}", "nse_median": "{:.3f}"}

    def table_for(block: dict, labels: list[str]) -> str:
        out = []
        for label in labels:
            m = block.get(label)
            if not m:
                continue
            rows = []
            cal = {c["h"]: c for c in m.get("calibration", [])}
            for r in m.get("metrics", []):
                rr = dict(r)
                c = cal.get(r["h"], {})
                rr["coverage"] = c.get("coverage")
                rr["halfwidth_median"] = c.get("halfwidth_median")
                rows.append(rr)
            out.append(f"**{label}** ({m.get('n_rows', 0):,} rows"
                       + (f", pinball {m['pinball']:.4f}" if "pinball" in m else "")
                       + ")\n\n" + _md_table(rows, cols, fmt))
        return "\n\n".join(out)

    L = ["# AQUIS research GNN — evaluation", ""]
    L.append(f"Generated: {result.get('generated_at')}  |  device: `{result.get('device')}`")
    L.append("")
    L.append("> **Research only.** The GNN is not promoted and is not used by the "
             "serving forecast path. Metrics below are on the existing purged split; "
             "nothing was tuned on the test period.")
    L.append("")
    L.append("`*` coverage / half-width use a 90% widening fitted on half the rows "
             "and reported on the other half.")
    L.append("")
    g = result.get("graph", {})
    L.append(f"Graph: {g.get('n_nodes')} nodes, {g.get('n_edges')} edges "
             f"({g.get('n_knn_edges')} kNN + {g.get('n_same_district_only_edges')} "
             f"same-district-only), k={g.get('k')}.")
    L.append("")

    for name, block in result.get("splits", {}).items():
        if name == "heldout_stations":
            continue
        L.append(f"## {name}")
        L.append("")
        labels = [k for k in block if k not in ("bootstrap_vs_production",
                                                "bootstrap_vs_matched")]
        L.append(table_for(block, labels))
        for key in ("bootstrap_vs_production", "bootstrap_vs_matched"):
            boot = block.get(key)
            if not boot:
                continue
            L.append("")
            L.append(f"Paired station bootstrap (RMSE ratio gnn/other, `{key}`):")
            L.append("")
            rows = [{"days": b["days"], "ratio": b["ratio_median"],
                     "lo": b["ci95"][0], "hi": b["ci95"][1],
                     "p": b["p_a_better"]} for b in boot]
            L.append(_md_table(rows,
                               [("days", "h"), ("ratio", "ratio"), ("lo", "2.5%"),
                                ("hi", "97.5%"), ("p", "P(gnn better)")],
                               {"days": "{:.0f}d", "ratio": "{:.3f}", "lo": "{:.3f}",
                                "hi": "{:.3f}", "p": "{:.2f}"}))
        L.append("")

    ho = result.get("splits", {}).get("heldout_stations")
    if ho:
        L.append("## Held-out stations (unseen-station test)")
        L.append("")
        L.append(f"Spatial k-means block **{ho['holdout_block']}** removed from training "
                 f"for *both* models: {ho['n_unseen_stations']} unseen stations, "
                 f"{ho['n_rows']:,} scored rows. This is the honest spatial-generalisation "
                 "number, not a temporal one.")
        L.append("")
        L.append(table_for(ho["models"], list(ho["models"])))
        if ho.get("bootstrap_gnn_vs_gru"):
            L.append("")
            L.append("Paired station bootstrap (RMSE ratio gnn/gru on unseen stations):")
            L.append("")
            rows = [{"days": b["days"], "ratio": b["ratio_median"],
                     "lo": b["ci95"][0], "hi": b["ci95"][1],
                     "p": b["p_a_better"]} for b in ho["bootstrap_gnn_vs_gru"]]
            L.append(_md_table(rows,
                               [("days", "h"), ("ratio", "ratio"), ("lo", "2.5%"),
                                ("hi", "97.5%"), ("p", "P(gnn better)")],
                               {"days": "{:.0f}d", "ratio": "{:.3f}", "lo": "{:.3f}",
                                "hi": "{:.3f}", "p": "{:.2f}"}))
        L.append("")

    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
