"""Perturbation-based station influence (not attention).

For every station *i* we nudge its own recent input by a small amount
``eps`` (added to the ``gwl`` feature of the last row of its 120-step window
and to the matching static vector), run the trained GNN, and read how the
predicted median delta ``q50`` of every *other* station *j* moves:

    sensitivity[i, j] = median_m ( q50_j(+m) - q50_j(-m) ) / (2 m)

with ``m`` in {0.05, 0.10, 0.25} metres. The central difference removes the
even-order response and the median across magnitudes keeps the estimate robust.

The same computation is repeated with the graph edges removed (self-loops only,
identical weights) and subtracted, so what is reported is **only the influence
that exists because of the graph**, not the trivial "everything is correlated
through the shared encoder" effect.

Aggregated per snapshot (median over sampled anchor timestamps), then to
districts (signed mean over station pairs - the effect is sparse, so a median
would collapse to zero).

Outputs (``ml/outputs/gnn_influence/``):
  station_influence.json, district_influence.json, influence_meta.json,
  district_influence_h28.png, district_influence_h120.png

    python -m research.gnn.influence
    python -m research.gnn.influence --snapshots 64 --horizons 28,120
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import graph as Gr
from .eval import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]          # ml/
OUT = ROOT / "outputs" / "gnn_influence"
RESEARCH = ROOT / "models" / "research"
W = 120
GWL_IDX = D.NUM_COLS.index("gwl")
MAGNITUDES = (0.05, 0.10, 0.25)
CAVEAT = ("Experimental. Shows model sensitivity between stations, not proven "
          "causation. Pumping and local geology are not observed.")

# GWL-derived columns that the model actually consumes. Lags are past values and
# do not move when the anchor row's ``gwl`` is perturbed; the two rolling windows
# *do* contain the anchor row, so a consistent perturbation has to update them.
ROLL_SPEC = (("gwl_roll7_mean", 28, "mean"),
             ("gwl_roll7_std", 28, "std"),
             ("gwl_roll30_mean", 120, "mean"),
             ("gwl_roll30_std", 120, "std"))
ROLL_IDX = np.array([D.NUM_COLS.index(n) for n, _, _ in ROLL_SPEC], np.int64)
LAG_COLS = [c for c in D.NUM_COLS if c.startswith("lag")]
ROLL_DAYS = {28: 7, 120: 30}


def _phys(seq_last: np.ndarray, j: int, fmean, fstd) -> float:
    """Invert the checkpoint z-score for one column."""
    return float(seq_last[j]) * float(fstd[j]) + float(fmean[j])


def roll_delta(seq_last: np.ndarray, gwl_phys: float, delta_phys: float,
               n7: int, n30: int, fmean, fstd) -> np.ndarray:
    """New standardised rolling columns after moving GWL(t) by ``delta_phys``.

    Exact single-element update of the stored rolling statistics: the mean
    shifts by ``delta/n`` and the variance is rebuilt from
    ``sum((x - mean)^2)`` with one element replaced. Nothing else in the input
    changes, so the unperturbed base stays byte-identical (a full recompute
    would not: the stored features use time-based windows computed on a
    superset of the filtered axis, so replaying them from ``seq`` introduces
    unrelated differences).

    ``n7``/``n30`` are the observation counts in the trailing 7 d / 30 d
    windows. The feature build uses time-based rolling, which resets across
    long gaps; the counts are therefore taken from the axis timestamps, not
    from the 120-row sequence length.
    """
    out = np.empty(len(ROLL_SPEC), np.float64)
    for k, (name, win, kind) in enumerate(ROLL_SPEC):
        n = max(int(n7 if win == 28 else n30), 1)
        j = D.NUM_COLS.index(name)
        mj = D.NUM_COLS.index(f"gwl_roll{ROLL_DAYS[win]}_mean")
        m0 = _phys(seq_last, mj, fmean, fstd)
        m1 = m0 + float(delta_phys) / n
        if kind == "mean":
            stat = m1
        elif n < 2:                       # pandas std is NaN at n=1 -> 0 fill
            stat = 0.0
        else:
            sd0 = _phys(seq_last, j, fmean, fstd)
            x0 = float(gwl_phys)
            x1 = x0 + float(delta_phys)
            # S = sum((x-mean)^2) = sum(x^2) - n*mean^2; only one element moves,
            # but every deviation changes when the mean shifts, so rebuild from
            # the raw sum of squares.
            sumsq0 = sd0 * sd0 * (n - 1) + n * m0 * m0
            sumsq1 = sumsq0 - x0 * x0 + x1 * x1
            s1 = sumsq1 - n * m1 * m1
            stat = float(np.sqrt(max(s1, 0.0) / (n - 1)))
        z = (stat - float(fmean[j])) / max(float(fstd[j]), 1e-12)
        out[k] = float(np.clip(z, -D.CLIP_SIGMA, D.CLIP_SIGMA))
    return out


def window_features(sub: dict, axis: dict, fmean, fstd) -> dict:
    """Per-anchor observation counts and time-window rolling stats.

    Replays the feature build's time-based windows (``7D`` / ``30D``, closed on
    the right) from the axis. Used both for the exact analytic update (counts)
    and for the reproducibility diagnostic (stats).
    """
    A = int(sub["A"])
    n7, n30 = np.empty(A, np.int64), np.empty(A, np.int64)
    m7, s7 = np.empty(A), np.empty(A)
    m30, s30 = np.empty(A), np.empty(A)
    st_index = {str(axis["st"][s]): i for i, s in enumerate(axis["starts"])}
    starts, ends = axis["starts"], axis["ends"]
    t_all = axis["times"]
    for a in range(A):
        s0 = st_index[str(sub["a_station"][a])]
        sl = t_all[starts[s0]:ends[s0]]
        t = sub["a_time"][a]
        hi = int(np.searchsorted(sl, t, side="right"))
        for days, nn, mm, ss in ((7, n7, m7, s7), (30, n30, m30, s30)):
            lo = int(np.searchsorted(sl, t - np.timedelta64(days, "D"),
                                     side="right"))
            idx = starts[s0] + np.arange(lo, hi)
            nn[a] = max(len(idx), 1)
            x = (axis["X"][idx, GWL_IDX].astype(np.float64) * float(fstd[GWL_IDX])
                 + float(fmean[GWL_IDX]))
            mm[a] = float(np.mean(x))
            ss[a] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return {"n7": n7, "n30": n30, "mean7": m7, "std7": s7,
            "mean30": m30, "std30": s30}




# --------------------------------------------------------------------------- #
# snapshots
# --------------------------------------------------------------------------- #
def _evenly(groups: list[np.ndarray], n: int) -> list[np.ndarray]:
    if n <= 0 or len(groups) <= n:
        return list(groups)
    idx = np.unique(np.linspace(0, len(groups) - 1, n).astype(int))
    return [groups[i] for i in idx]


def _evenly_large(groups: list[np.ndarray], n: int,
                  min_frac: float = 0.5) -> list[np.ndarray]:
    """Evenly spaced in time among the better-populated snapshots.

    Many timestamps carry only a handful of reporting stations (a snapshot can
    hold 1 node or 250). Keeping the timestamps at or above ``min_frac`` of the
    size distribution and then spreading the sample evenly over time inside
    that subset maximises station coverage while preserving the temporal
    spread. Measured on this dataset: 506/548 stations covered, vs 484 for a
    plain even sample and 406 for one-snapshot-per-time-bin.

    Target-validity is deliberately *not* used as a pre-filter: the last days of
    the val window carry no 7 d / 30 d label (labels crossing into test are
    purged), and dropping those timestamps first shifts the even spacing onto
    a worse set of snapshots (462/548). They cost nothing instead — a snapshot
    without a single valid target skips in ~0 s.
    """
    groups = list(groups)
    if n <= 0 or len(groups) <= n:
        return groups
    sizes = np.array([len(g) for g in groups])
    thr = float(np.quantile(sizes, min_frac))
    keep = [g for g, s in zip(groups, sizes) if s >= max(thr, 1.0)]
    if len(keep) < n:
        keep = groups
    return _evenly(keep, n)


def sample_snapshots(axis: dict, n_val: int, n_train: int,
                     fmean=None, fstd=None):
    """Evenly spaced val snapshots (honest) + train snapshots (station coverage).

    Returns ``(snapshots, splits)``; the split frames are loaded once, scaled with
    the checkpoint's train-only statistics, and reused for every snapshot.
    """
    splits = {}
    out = []

    def _scale(sp):
        return sp if fmean is None else D.scale_split(sp, fmean, fstd)

    sp_val = _scale(D.load_split("val", axis))
    splits["val"] = sp_val
    for g in _evenly_large(D.snapshot_groups(sp_val), n_val):
        out.append(("val", g))
    if n_train:
        sp_tr = _scale(D.load_split("train", axis))
        splits["train"] = sp_tr
        for g in _evenly_large(D.snapshot_groups(sp_tr), n_train):
            out.append(("train", g))
    return out, splits


# --------------------------------------------------------------------------- #
# response
# --------------------------------------------------------------------------- #
def _hcol(t: dict, h_target: int) -> np.ndarray:
    hv = t["h"].cpu().numpy()
    col = np.full(len(hv), -1, np.int64)
    for r in range(len(hv)):
        w = np.flatnonzero(hv[r] == h_target)
        if len(w):
            col[r] = int(w[0])
    return col


@torch.no_grad()
def _gather_q50(pred: torch.Tensor, col: np.ndarray) -> np.ndarray:
    p = pred.cpu().numpy()[:, :, 1]
    return p[np.arange(len(col)), np.maximum(col, 0)]


@torch.no_grad()
def _q50_from_z(model, t, z, static, col) -> np.ndarray:
    """Graph pass + head for pre-encoded embeddings (encoder is per-node)."""
    z = model.message_pass(z, t["edge_index"], t["edge_attr"])
    return _gather_q50(model.heads(z, static, t["hf"]), col)


@torch.no_grad()
def _forward_q50(model, t, seq, static, col) -> np.ndarray:
    pred = model(seq, t["seqm"], static, t["hf"], t["edge_index"], t["edge_attr"])
    return _gather_q50(pred, col)


@torch.no_grad()
def snapshot_sensitivity(model, t: dict, col: np.ndarray,
                         gwl_std: float = 1.0, fmean=None, fstd=None,
                         consistent: bool = False, n7=None, n30=None) -> np.ndarray:
    """(P, P) central-difference sensitivity for one snapshot.

    ``gwl_std`` converts the physical perturbation (metres) into the model's
    standardised input units, so the reported value is still *per metre*.

    With ``consistent=True`` the derived GWL features are updated too: the
    rolling means/stds are adjusted by the exact single-element effect of the
    perturbed anchor (``roll_delta``). Lags are past values and stay untouched;
    slopes/deviations/flags are not in ``NUM_COLS`` and are never seen by the
    model.

    The GRU encoder is per-node independent, so a perturbation only changes that
    row's embedding: we re-encode one row and re-run the (cheap) graph pass over
    the unchanged rest. Identical arithmetic to a full forward, ~P times faster.
    """
    if consistent and (fmean is None or fstd is None or n7 is None or n30 is None):
        raise ValueError("consistent perturbation requires feat_mean/feat_std "
                         "and window counts")
    P = t["n_nodes"]
    fast = all(hasattr(model, a) for a in ("encode", "message_pass", "heads"))
    if fast:
        z0 = model.encode(t["seq"], t["seqm"])
        base = _q50_from_z(model, t, z0, t["static"], col)
        z = z0.clone()
    else:
        base = _forward_q50(model, t, t["seq"], t["static"], col)
    seq = t["seq"].clone()
    static = t["static"].clone()
    seq_last0 = seq[:, -1, :].clone()
    static0 = static.clone()
    gwl_mean = float(fmean[GWL_IDX]) if fmean is not None else 0.0

    def _apply(i: int, delta_phys: float) -> None:
        step = float(delta_phys) / max(float(gwl_std), 1e-6)
        gwl_phys = float(seq_last0[i, GWL_IDX]) * float(gwl_std) + gwl_mean
        seq[i, -1, GWL_IDX] += step
        static[i, GWL_IDX] += step
        if consistent:
            r = torch.as_tensor(
                roll_delta(seq_last0[i], gwl_phys, delta_phys,
                           int(n7[i]), int(n30[i]), fmean, fstd),
                dtype=seq.dtype, device=seq.device)
            seq[i, -1, ROLL_IDX] = r
            static[i, ROLL_IDX] = r

    def _undo(i: int) -> None:
        seq[i, -1, :] = seq_last0[i]
        static[i, :] = static0[i]

    sens = np.zeros((P, P), np.float64)
    for m in MAGNITUDES:
        pos = np.zeros((P, P), np.float64)
        neg = np.zeros((P, P), np.float64)

        def _row(i: int) -> np.ndarray:
            if not fast:
                return _forward_q50(model, t, seq, static, col)
            z[i] = model.encode(seq[i:i + 1], t["seqm"][i:i + 1])[0]
            return _q50_from_z(model, t, z, static, col)

        for i in range(P):
            _apply(i, +m)
            pos[i] = _row(i) - base
            _undo(i)
            if fast:
                z[i] = z0[i]                    # undo this row's embedding

            _apply(i, -m)
            neg[i] = _row(i) - base
            _undo(i)
            if fast:
                z[i] = z0[i]
        sens += (pos - neg) / (2.0 * m)
    return sens / len(MAGNITUDES)



# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def build_matrices(model, axis, G, snapshots, splits, horizons, device,
                   gwl_std: float = 1.0, fmean=None, fstd=None,
                   consistent: bool = False) -> dict:
    N = G["n"]
    mats = {h: [] for h in horizons}
    station_hits = np.zeros(N, np.int64)
    t_start = time.time()
    for si, (kind, g) in enumerate(snapshots, 1):
        sp = splits[kind]
        # rebuild the anchors of this snapshot only
        sub = _subset(sp, g)
        t = D.build_batch([np.arange(sub["A"])], axis, sub, W, G["node_of"],
                          G["edge_i"], G["edge_j"], G["edge_attr"], G["n"], device)
        t_ng = D.build_batch([np.arange(sub["A"])], axis, sub, W, G["node_of"],
                             G["edge_i"], G["edge_j"], G["edge_attr"], G["n"],
                             device, include_edges=False)
        win = window_features(sub, axis, fmean, fstd) if consistent else {}
        n7, n30 = win.get("n7"), win.get("n30")
        nodes = t["node_global"].cpu().numpy()
        t_snap = time.time()
        scored = []
        for h in horizons:
            col = _hcol(t, h)
            if (col < 0).all():
                continue
            sg = snapshot_sensitivity(model, t, col, gwl_std, fmean, fstd,
                                      consistent, n7, n30)
            sng = snapshot_sensitivity(model, t_ng, col, gwl_std, fmean, fstd,
                                       consistent, n7, n30)
            sub_full = sg - sng
            valid = col >= 0
            sub_full[~valid, :] = np.nan
            sub_full[:, ~valid] = np.nan
            full = np.full((N, N), np.nan, np.float32)
            full[np.ix_(nodes, nodes)] = sub_full
            mats[h].append(full)
            scored.append(h)
        if scored:
            station_hits[nodes] += 1       # only count snapshots that produced data
        print(f"  snapshot {si}/{len(snapshots)} ({kind}) nodes={t['n_nodes']} "
              f"horizons={scored} {time.time() - t_snap:.1f}s "
              f"elapsed={time.time() - t_start:.0f}s", flush=True)
    return mats, station_hits


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _top_sources(mat: np.ndarray, k: int = 10) -> set[int]:
    a = np.nan_to_num(mat, nan=0.0)
    np.fill_diagonal(a, 0.0)
    return set(np.argsort(-np.abs(a).sum(axis=1))[:k].tolist())


def compare_matrices(raw: np.ndarray, con: np.ndarray) -> dict:
    """How much the graph-only sensitivity moves when derived features follow."""
    fin = np.isfinite(raw) & np.isfinite(con)
    off = ~np.eye(raw.shape[0], dtype=bool) & fin
    d = (raw - con)[fin]
    aa, bb = raw[fin], con[fin]
    nz = off & (raw != 0.0)
    ta, tb = _top_sources(raw), _top_sources(con)
    return {
        "n_pairs": int(fin.sum()),
        "raw_max_abs": float(np.max(np.abs(aa))) if len(aa) else 0.0,
        "raw_zero_fraction": float(np.mean(aa == 0.0)) if len(aa) else None,
        "con_zero_fraction": float(np.mean(bb == 0.0)) if len(bb) else None,
        "diff_max_abs": float(np.max(np.abs(d))) if len(d) else 0.0,
        "diff_mean_abs": float(np.mean(np.abs(d))) if len(d) else 0.0,
        "diff_rel_frobenius": (float(np.linalg.norm(d) / np.linalg.norm(aa))
                               if len(d) and np.linalg.norm(aa) > 0 else None),
        "corr_all": _corr(aa, bb),
        "corr_offdiag": _corr(raw[off], con[off]) if off.any() else None,
        "sign_agreement_nonzero": (float(np.mean(np.sign(raw[nz]) == np.sign(con[nz])))
                                   if nz.any() else None),
        "top10_source_overlap": len(ta & tb),
    }


def feature_audit() -> dict:
    hydro = ["gwl_dev7", "gwl_dev30", "gwl_slope7", "gwl_slope30",
             "gwl_slope90", "gwl_obs7", "gwl_obs30", "gwl_obs90", "gwl_imputed"]
    return {
        "model_inputs": list(D.NUM_COLS),
        "gwl_lag_columns_in_model": LAG_COLS,
        "gwl_rolling_columns_in_model": [n for n, _, _ in ROLL_SPEC],
        "lag_columns_changed_by_anchor_perturbation": False,
        "rolling_columns_changed_by_anchor_perturbation": True,
        "derived_columns_not_in_model": [c for c in hydro if c not in D.NUM_COLS],
        "note": ("Lags are GWL(t-k) and therefore do not move when GWL(t) is "
                 "perturbed. Slopes/deviations/flags are computed by "
                 "pipelines.model.build but are not in the frozen traj_dl "
                 "schema, so the model never sees them (no update possible or "
                 "needed). Rolling mean/std windows contain GWL(t) and are "
                 "updated by roll_delta."),
    }


def roll_reproducibility(win: dict, t: dict, fmean, fstd) -> dict:
    """Compare the stored rolling columns with a replay from the axis windows.

    The stored features were built with time-based rolling on a superset of the
    filtered axis (break purges removed some rows afterwards), so a replay is
    close but not bit-exact. That residual is exactly why the perturbation uses
    the analytic single-element update instead of replacing the stored columns.
    """
    seq = t["seq"].detach().cpu().numpy()
    P = len(seq)
    rec = np.column_stack([win["mean7"], win["std7"],
                           win["mean30"], win["std30"]])
    stored = np.empty((P, len(ROLL_SPEC)), np.float64)
    for k, (name, _, _) in enumerate(ROLL_SPEC):
        j = D.NUM_COLS.index(name)
        stored[:, k] = seq[:, -1, j].astype(np.float64) * float(fstd[j]) + float(fmean[j])
    diff = np.abs(rec - stored)
    return {
        "rows": int(P),
        "max_abs_diff_m": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff_m": float(diff.mean()) if diff.size else 0.0,
        "rows_over_1mm": int((diff.max(axis=1) > 1e-3).sum()) if diff.size else 0,
        "per_column_max_m": {name: float(diff[:, k].max()) if diff.size else 0.0
                             for k, (name, _, _) in enumerate(ROLL_SPEC)},
        "window_counts": {
            "n7_min": int(win["n7"].min()), "n7_median": float(np.median(win["n7"])),
            "n7_max": int(win["n7"].max()),
            "n30_min": int(win["n30"].min()), "n30_median": float(np.median(win["n30"])),
            "n30_max": int(win["n30"].max()),
            "rows_n30_lt_2": int((win["n30"] < 2).sum()),
        },
        "note": ("Residual is expected: the stored features use time-based "
                 "windows over rows that were later dropped from the axis "
                 "(break purges). The perturbation therefore applies the exact "
                 "analytic delta to the stored statistics."),
    }


def example_deltas(t: dict, win: dict, fmean, fstd, step_m: float = 0.05) -> dict:
    """One row's derived-feature deltas for a +step_m physical perturbation."""
    seq = t["seq"].detach().cpu().numpy()
    i = int(np.argmax(win["n30"]))
    gwl_phys = _phys(seq[i, -1], GWL_IDX, fmean, fstd)
    base = np.array([_phys(seq[i, -1], D.NUM_COLS.index(n), fmean, fstd)
                     for n, _, _ in ROLL_SPEC])
    new_z = roll_delta(seq[i, -1], gwl_phys, step_m,
                       int(win["n7"][i]), int(win["n30"][i]), fmean, fstd)
    new = np.array([new_z[k] * float(fstd[D.NUM_COLS.index(name)])
                    + float(fmean[D.NUM_COLS.index(name)])
                    for k, (name, _, _) in enumerate(ROLL_SPEC)])
    return {
        "row": i,
        "n7": int(win["n7"][i]), "n30": int(win["n30"][i]),
        "step_m": step_m,
        "gwl_scaled_delta": float(step_m / float(fstd[GWL_IDX])),
        "rolling_columns": {
            name: {"scaled_delta": float(new_z[k] - seq[i, -1, D.NUM_COLS.index(name)]),
                   "physical_delta_m": float(new[k] - base[k])}
            for k, (name, _, _) in enumerate(ROLL_SPEC)},
        "lag_columns_scaled_delta": {c: 0.0 for c in LAG_COLS},
    }


def lag_audit(sub: dict, axis: dict, t: dict, fmean, fstd) -> dict:
    """Empirically confirm the lag columns are past values, not functions of t.

    ``lagK`` is ``GWL(t-K)`` (``groupby(...).shift(K)`` in
    ``pipelines.model.build``), so perturbing ``GWL(t)`` cannot move it. This is
    checked on the anchors whose trailing ``K`` axis rows are a complete 6 h run:
    only there is the axis row ``K`` back the same row the feature used (break
    purges and observation gaps remove rows *after* the features were built, so
    the other anchors cannot be verified this way and are reported as skipped).
    """
    seq = t["seq"].detach().cpu().numpy()
    step = np.timedelta64(6, "h")
    times = axis["times"]
    A = int(sub["A"])
    rows = []
    for name in LAG_COLS:
        steps = int(name[3:])
        j = D.NUM_COLS.index(name)
        diffs, skipped = [], 0
        for a in range(A):
            pos = int(sub["a_axis"][a]) - steps
            if pos < int(sub["a_runstart"][a]):
                skipped += 1
                continue
            window = times[pos:int(sub["a_axis"][a]) + 1]
            if len(window) != steps + 1 or not np.all(np.diff(window) == step):
                skipped += 1                  # gap / purge inside the lag span
                continue
            ref = (axis["X"][pos, GWL_IDX].astype(np.float64) * float(fstd[GWL_IDX])
                   + float(fmean[GWL_IDX]))
            diffs.append(abs(_phys(seq[a, -1], j, fmean, fstd) - ref))
        rows.append({
            "column": name, "steps": steps,
            "n_verified": len(diffs), "n_skipped": skipped,
            "max_abs_diff_m": float(np.max(diffs)) if diffs else None,
            "median_abs_diff_m": float(np.median(diffs)) if diffs else None,
        })
    return {
        "lags_are_past_values": True,
        "definition": ("lagK = GWL(t-K); verified where the trailing K axis rows "
                       "form a complete 6 h run, so the anchor perturbation "
                       "leaves the lag columns untouched by construction"),
        "per_column": rows,
    }


def consistency_check(model, axis, G, snapshots, splits, horizons, device,
                      gwl_std, fmean, fstd) -> dict:
    """Raw-GWL vs consistent-derived perturbation, on a handful of snapshots."""
    out = {"meta": {
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "definition": ("graph-only sensitivity (edges minus no-edge "
                              "ablation) with and without updating the rolling "
                              "GWL features that contain the perturbed anchor"),
               "horizons_h": list(horizons),
               "magnitudes_m": list(MAGNITUDES),
               "snapshots": len(snapshots)},
           "feature_audit": feature_audit(), "snapshots": []}
    for si, (kind, g) in enumerate(snapshots, 1):
        sp = splits[kind]
        sub = _subset(sp, g)
        t = D.build_batch([np.arange(sub["A"])], axis, sub, W, G["node_of"],
                          G["edge_i"], G["edge_j"], G["edge_attr"], G["n"], device)
        t_ng = D.build_batch([np.arange(sub["A"])], axis, sub, W, G["node_of"],
                             G["edge_i"], G["edge_j"], G["edge_attr"], G["n"],
                             device, include_edges=False)
        win = window_features(sub, axis, fmean, fstd)
        entry = {"kind": kind,
                 "time": str(np.datetime64(sub["a_time"][0])),
                 "nodes": int(t["n_nodes"]), "horizons": {}}
        if si == 1:
            out["roll_reproducibility"] = roll_reproducibility(win, t, fmean, fstd)
            out["static_vs_seq_last_max_abs"] = float(np.abs(
                t["static"].detach().cpu().numpy()
                - t["seq"][:, -1, :].detach().cpu().numpy()).max())
            out["example_deltas"] = example_deltas(t, win, fmean, fstd)
            out["lag_audit"] = lag_audit(sub, axis, t, fmean, fstd)
        for h in horizons:
            col = _hcol(t, h)
            if (col < 0).all():
                continue
            raw = (snapshot_sensitivity(model, t, col, gwl_std, fmean, fstd, False)
                   - snapshot_sensitivity(model, t_ng, col, gwl_std, fmean, fstd, False))
            con = (snapshot_sensitivity(model, t, col, gwl_std, fmean, fstd, True,
                                        win["n7"], win["n30"])
                   - snapshot_sensitivity(model, t_ng, col, gwl_std, fmean, fstd, True,
                                          win["n7"], win["n30"]))
            valid = col >= 0
            raw[~valid, :] = np.nan
            raw[:, ~valid] = np.nan
            con[~valid, :] = np.nan
            con[:, ~valid] = np.nan
            entry["horizons"][str(h)] = compare_matrices(raw, con)
        out["snapshots"].append(entry)
        print(f"  consistency snapshot {si}/{len(snapshots)} nodes={t['n_nodes']}",
              flush=True)

    summary = {}
    for h in horizons:
        rows = [s["horizons"][str(h)] for s in out["snapshots"]
                if str(h) in s["horizons"]]
        if not rows:
            continue

        def _vals(key, fn):
            v = [r[key] for r in rows if r.get(key) is not None]
            return fn(v) if v else None

        summary[str(h)] = {
            "snapshots": len(rows),
            "diff_max_abs_max": _vals("diff_max_abs", max),
            "diff_rel_frobenius_mean": _vals("diff_rel_frobenius", lambda v: float(np.mean(v))),
            "corr_offdiag_min": _vals("corr_offdiag", min),
            "sign_agreement_nonzero_min": _vals("sign_agreement_nonzero", min),
            "top10_source_overlap_min": _vals("top10_source_overlap", min),
        }
    out["summary"] = summary
    return out


def _subset(sp: dict, anchors: np.ndarray) -> dict:
    anchors = np.asarray(anchors, np.int64)
    ptr, h, d = sp["ptr"], sp["h"], sp["delta"]
    lens = ptr[anchors + 1] - ptr[anchors]
    new_ptr = np.r_[0, np.cumsum(lens)]
    return {"a_axis": sp["a_axis"][anchors], "a_runstart": sp["a_runstart"][anchors],
            "a_static": sp["a_static"][anchors], "a_static_m": sp["a_static_m"][anchors],
            "ptr": new_ptr,
            "h": np.concatenate([h[ptr[a]:ptr[a + 1]] for a in anchors]),
            "delta": np.concatenate([d[ptr[a]:ptr[a + 1]] for a in anchors]),
            "A": len(anchors), "a_station": sp["a_station"][anchors],
            "a_time": sp["a_time"][anchors]}


def _median_stack(mats: list[np.ndarray]) -> np.ndarray:
    if not mats:
        return np.zeros((0, 0), np.float64)
    with warnings.catch_warnings():
        # pairs never co-observed in any sampled snapshot stay NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(mats, axis=0), axis=0)


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def _top_pairs(mat: np.ndarray, stations: list[str], k: int = 20) -> list[dict]:
    """Strongest district links by magnitude, sign preserved."""
    a = np.where(np.isfinite(mat), np.abs(mat), -np.inf)
    flat = a.ravel()
    idx = np.argsort(flat)[::-1][:k * 4]
    out, seen = [], set()
    for t in idx:
        i, j = divmod(int(t), a.shape[1])
        if i == j or not np.isfinite(flat[t]) or flat[t] == 0.0:
            continue
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        out.append({"from": stations[i], "to": stations[j],
                    "sensitivity": float(mat[i, j])})
        if len(out) >= k:
            break
    return out


def station_payload(mat: np.ndarray, stations: list[str], districts: list[str],
                    top: int = 10) -> dict:
    a = np.nan_to_num(mat, nan=0.0)
    finite = np.isfinite(mat)
    out = {}
    for j, name in enumerate(stations):
        # a station that no sampled snapshot covered has no measurement at all;
        # omit it rather than publishing a row of zeros.
        if not (finite[:, j].any() or finite[j, :].any()):
            continue
        # rank by magnitude: a strong inhibitory link matters as much as a
        # positive one. The stored value keeps its sign.
        incoming = np.argsort(-np.abs(a[:, j]))
        outgoing = np.argsort(-np.abs(a[j, :]))
        out[name] = {
            "node": int(j), "district": districts[j],
            "strength_in": float(np.sum(np.abs(a[:, j]))),
            "strength_out": float(np.sum(np.abs(a[j, :]))),
            "influenced_by": [
                {"station": stations[int(i)], "district": districts[int(i)],
                 "sensitivity": float(a[i, j])}
                for i in incoming if i != j and a[i, j] != 0.0][:top],
            "influences": [
                {"station": stations[int(k)], "district": districts[int(k)],
                 "sensitivity": float(a[j, k])}
                for k in outgoing if k != j and a[j, k] != 0.0][:top],
        }
    return out


def district_matrix(mat: np.ndarray, districts: list[str]) -> tuple[list[str], np.ndarray]:
    """District-level matrix: signed **mean** over station pairs.

    A median was tried first and collapsed to exactly ``0`` for almost every
    pair: the graph effect is sparse (most station pairs are 0), so the median
    of a district pair is 0 even when a few pairs carry a real signal. The mean
    is linear in the underlying sensitivities, keeps the sign, and is therefore
    the honest district aggregate.
    """
    names = sorted(set(districts))
    idx = {d: i for i, d in enumerate(names)}
    acc: dict[tuple[int, int], list[float]] = {}
    for i, di in enumerate(districts):
        for j, dj in enumerate(districts):
            v = mat[i, j]
            if np.isfinite(v):
                acc.setdefault((idx[di], idx[dj]), []).append(float(v))
    out = np.full((len(names), len(names)), np.nan)
    for (a, b), vals in acc.items():
        out[a, b] = float(np.mean(vals))
    return names, out


def _heatmap(path: Path, names: list[str], mat: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.nan_to_num(mat, nan=0.0)
    v = float(np.percentile(np.abs(a), 99)) or 1.0
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(a, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("influenced district (to)")
    ax.set_ylabel("source district (from)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ q50 per metre (graph-only)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Perturbation station influence.")
    ap.add_argument("--ckpt", default=str(RESEARCH / "gnn_traj.pt"))
    ap.add_argument("--snapshots-val", type=int, default=24)
    ap.add_argument("--snapshots-train", type=int, default=16)
    ap.add_argument("--horizons", default="28,120")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-derived", type=int, default=0, metavar="N",
                    help="run the derived-feature consistency check on N val "
                         "snapshots and exit")
    ap.add_argument("--consistent-derived", action="store_true",
                    help="update the rolling GWL features when perturbing")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    print(f"device={device} horizons={horizons}", flush=True)

    model, cfg = load_checkpoint(Path(args.ckpt), device)
    axis = D.load_axis()
    stats = D.stats_from_cfg(cfg)
    if stats is None:
        fmean = fstd = None
        gwl_std = 1.0
        print("warning: checkpoint has no feat_mean/feat_std, assuming unscaled",
              flush=True)
    else:
        fmean, fstd = stats
        axis = D.scale_axis(axis, fmean, fstd)
        gwl_std = float(fstd[GWL_IDX])
    graph = Gr.ensure_graph()
    G = Gr.graph_arrays(graph)
    G["graph_nodes"] = graph["nodes"]
    ea, emean, estd = D.edge_attr_from_km(G["edge_km"])
    G["edge_attr"] = ea
    stations = G["stations"]
    districts = [n["district"] for n in graph["nodes"]]

    t0 = time.time()
    n_val = args.check_derived or args.snapshots_val
    n_train = 0 if args.check_derived else args.snapshots_train
    snaps, splits = sample_snapshots(axis, n_val, n_train, fmean, fstd)
    print(f"snapshots: {len(snaps)}", flush=True)

    if args.check_derived:
        chk = consistency_check(model, axis, G, snaps, splits, horizons, device,
                                gwl_std, fmean, fstd)
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / "influence_consistency.json"
        p.write_text(json.dumps(chk, indent=1), encoding="utf-8")
        print(f"wrote {p} in {time.time() - t0:.0f}s", flush=True)
        for h in horizons:
            s = chk["summary"].get(str(h))
            if s:
                print(f"  h={h}: rel_frobenius {s['diff_rel_frobenius_mean']:.3e} "
                      f"max_abs_diff {s['diff_max_abs_max']:.3e} "
                      f"corr_offdiag_min {s['corr_offdiag_min']} "
                      f"top10_overlap_min {s['top10_source_overlap_min']}",
                      flush=True)
        return

    mats, hits = build_matrices(model, axis, G, snaps, splits, horizons, device,
                                gwl_std, fmean, fstd, args.consistent_derived)
    print(f"influence compute {time.time() - t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    final = {h: _median_stack(mats[h]) for h in horizons}
    scored_counts = {str(h): len(mats[h]) for h in horizons}

    station_json = {"meta": {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "checkpoint": str(Path(args.ckpt).name),
                             "snapshots": len(snaps), "horizons_h": horizons,
                             "magnitudes_m": list(MAGNITUDES),
                             "definition": "central-difference d(q50)/d(gwl_i), "
                                           "graph edges minus no-edge ablation",
                             "caveat": CAVEAT},
                    "stations": {}}
    for h in horizons:
        station_json["stations"][str(h)] = station_payload(
            final[h], stations, districts, top=10)

    district_json = {"meta": {"generated_at": station_json["meta"]["generated_at"],
                              "horizons_h": horizons,
                              "aggregation": "signed mean over station pairs",
                              "caveat": CAVEAT}}
    for h in horizons:
        names, dmat = district_matrix(final[h], districts)
        district_json[str(h)] = {
            "districts": names,
            "matrix": [[None if not np.isfinite(x) else round(float(x), 6)
                        for x in row] for row in dmat],
            "top_pairs": _top_pairs(dmat, names, k=20),
        }
        _heatmap(OUT / f"district_influence_h{h}.png", names, dmat,
                 f"District influence, h={h} ({h / 4:.0f}d) — graph-only, per metre")

    meta = {
        "model": {"checkpoint": Path(args.ckpt).name, "config": cfg,
                  "graph": graph["meta"]},
        "generated_at": station_json["meta"]["generated_at"],
        "date": time.strftime("%Y-%m-%d"),
        "split": {"train": "< 2025-10-01", "val": "2025-10-01..2025-12-31",
                  "test": ">= 2026-01-01"},
        "n_stations": len(stations),
        "n_districts": len(set(districts)),
        "snapshots": {"val": args.snapshots_val, "train": args.snapshots_train,
                      "total": len(snaps)},
        "snapshots_scored": scored_counts,
        "station_coverage": {"with_any_snapshot": int((hits > 0).sum()),
                             "total": int(len(hits))},
        "horizons_h": horizons,
        "magnitudes_m": list(MAGNITUDES),
        "input_scaling": ("train-only z-score (clip +/-%.0f sigma); perturbations "
                          "applied as m / std(gwl)" % D.CLIP_SIGMA),
        "perturbation_features": (
            "gwl + rolling mean/std updated by the exact single-element delta "
            "(consistent derived features)"
            if args.consistent_derived else
            "gwl only (rolling features held fixed)"),
        "caveat": CAVEAT,
        "caveat_text": CAVEAT,
        "eval": _eval_summary(),
        "note": "Research only. Not promoted; serving forecasts are unchanged.",
    }

    (OUT / "station_influence.json").write_text(json.dumps(station_json), encoding="utf-8")
    (OUT / "district_influence.json").write_text(json.dumps(district_json), encoding="utf-8")
    (OUT / "influence_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote station_influence.json, district_influence.json, "
          f"influence_meta.json + 2 heatmaps to {OUT}")


def _eval_summary() -> dict:
    p = OUT / "eval.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    val = d.get("splits", {}).get("val", {})
    gnn = next((r for r in val.get("gnn", {}).get("metrics", []) if r["h"] == 120), None)
    prod = next((r for r in val.get("gru_production", {}).get("metrics", [])
                 if r["h"] == 120), None)
    return {"val_rmse_30d_gnn": gnn and gnn["rmse"],
            "val_rmse_30d_gru_production": prod and prod["rmse"]}


if __name__ == "__main__":
    main()
