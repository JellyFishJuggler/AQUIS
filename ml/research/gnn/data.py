"""Snapshot batching over the existing purged splits.

One graph snapshot = one anchor timestamp, with every station observed at that
timestamp as a node. Stations missing at a timestamp are simply absent from
that snapshot's node set (that *is* the masking: no zero-filled node, no fake
edge).

The split boundaries are **not** redefined here. ``load_split`` reuses the
already-materialised ``data/features_traj/{train,val}.parquet`` produced by
``pipelines.traj.datasets`` and reproduces ``pipelines.traj_dl._load_split``
exactly (``tests``-style check in ``selfcheck``). The 2026 test targets are
built in memory from ``data/features_6h/test.parquet`` with the existing
``pipelines.traj._expand`` / ``_axis_and_targets`` helpers, so no production
feature file is written or overwritten.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]          # ml/
FE6 = ROOT / "data" / "features_6h"
FTRAJ = ROOT / "data" / "features_traj"

_TDL = importlib.import_module("pipelines.traj_dl")
_TRAJ = importlib.import_module("pipelines.traj")

NUM_COLS = list(_TDL.NUM_COLS)
META = ["Station", "time"]
MAX_H = int(_TDL.MAX_H)
PAD = int(_TDL.PAD)
QUANTILES = tuple(_TDL.QUANTILES)
HORIZONS_EVAL = (4, 28, 60, 120)      # 1d / 7d / 15d / 30d on the 6h grid


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def load_axis() -> dict:
    """Union axis, identical to the production loader (read-only)."""
    return _TDL._load_axis()


# --------------------------------------------------------------------------- #
# feature scaling
#
# The raw feature columns are on wildly different scales (``rain_30d`` reaches
# 15,120 mm, ``pressure`` ~1,070 hPa, ``year`` ~2,024). Feeding that straight
# into a GRU saturates the gates: the encoder output barely responds to its
# input, so the graph never receives any signal. The research models therefore
# standardise with **train-only** statistics (mean/std over the train axis,
# clipped to +/-8 sigma), stored in the checkpoint config so eval / influence
# reproduce exactly the same transform.
#
# ``gru_production`` is *not* scaled — it is scored exactly as it is served.
# --------------------------------------------------------------------------- #
CLIP_SIGMA = 8.0


def feature_stats(axis: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std over the train axis (finite values only)."""
    X = np.asarray(axis["X"], np.float64)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _zscore(v: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (np.asarray(v, np.float32) - mean) / std
    return np.clip(z, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)


def scale_axis(axis: dict, mean: np.ndarray, std: np.ndarray) -> dict:
    """New axis dict with ``X`` standardised (the original is not modified)."""
    out = dict(axis)
    out["X"] = _zscore(axis["X"], mean, std)
    return out


def scale_split(sp: dict, mean: np.ndarray, std: np.ndarray) -> dict:
    """New split dict with ``a_static`` standardised (mask untouched)."""
    out = dict(sp)
    out["a_static"] = _zscore(sp["a_static"], mean, std)
    return out


def stats_from_cfg(cfg: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Scaling stats stored in a checkpoint config, or ``None`` for legacy ones."""
    m, s = cfg.get("feat_mean"), cfg.get("feat_std")
    if not m or not s:
        return None
    return (np.asarray(m, np.float32), np.asarray(s, np.float32))



def anchors_from_frame(lf: pd.DataFrame, axis: dict) -> dict:
    """Group a long ``(station, time, h)`` frame into anchors.

    Byte-for-byte the logic of ``pipelines.traj_dl._load_split`` after the
    parquet read, so train/val anchors are identical to the production ones and
    the in-memory 2026 frame is grouped the same way.
    """
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
            raise ValueError(f"{sname}: {int((~ok).sum())} anchors not found on axis")
        a_axis[sel] = sl0 + pos
        a_runstart[sel] = sl0

    return {"a_axis": a_axis, "a_runstart": a_runstart, "a_static": a_static,
            "a_static_m": a_static_m, "ptr": ptr, "h": h_arr, "delta": d_arr,
            "A": A, "a_station": a_station, "a_time": a_time}


def load_split(name: str, axis: dict) -> dict:
    """Purged train/val split exactly as the production GRU sees it."""
    lf = pd.read_parquet(FTRAJ / f"{name}.parquet",
                         columns=META + NUM_COLS + ["h", "target_delta", "target_time"])
    lf["Station"] = lf["Station"].astype(str)
    lf["time"] = pd.to_datetime(lf["time"], errors="coerce")
    return anchors_from_frame(lf, axis)


def build_test_split(axis: dict, horizons: tuple[int, ...] = HORIZONS_EVAL,
                     stride: int = 4) -> dict:
    """2026 test anchors built in memory from ``features_6h/test.parquet``.

    ``stride`` subsamples the anchor grid (default 4 = one anchor per day) so
    the in-memory expansion stays small; every horizon in ``horizons`` is
    expanded on the same anchors for both models. Nothing is written to disk.
    """
    te = pd.read_parquet(FE6 / "test.parquet", columns=META + NUM_COLS)
    te["Station"] = te["Station"].astype(str)
    te["time"] = pd.to_datetime(te["time"], errors="coerce")
    te = te.dropna(subset=["time"]).sort_values(["Station", "time"], kind="stable")
    if stride > 1:
        te = te.iloc[::stride].reset_index(drop=True)
    else:
        te = te.reset_index(drop=True)

    frames = []
    for st, sub in te.groupby("Station", sort=True):
        times_st, gwl_st = _TRAJ._axis_and_targets([sub[["time", "gwl"]]])
        frames.append(_TRAJ._expand(sub, times_st, gwl_st, list(horizons)))
    lf = pd.concat(frames, ignore_index=True)
    return anchors_from_frame(lf, axis)


# --------------------------------------------------------------------------- #
# snapshots
# --------------------------------------------------------------------------- #
def snapshot_groups(sp: dict, anchor_ids: np.ndarray | None = None) -> list[np.ndarray]:
    """Anchor ids grouped by timestamp (one group = one graph snapshot)."""
    ids = np.arange(sp["A"]) if anchor_ids is None else np.asarray(anchor_ids, np.int64)
    t = sp["a_time"][ids]
    order = np.argsort(t, kind="stable")
    ids = ids[order]
    t = t[order]
    bounds = np.r_[0, np.flatnonzero(t[1:] != t[:-1]) + 1, len(ids)]
    return [ids[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def pack_groups(groups: list[np.ndarray], target_nodes: int,
                rng: np.random.Generator | None = None) -> list[list[np.ndarray]]:
    """Greedily pack snapshots into mini-batches of ~``target_nodes`` nodes.

    ``groups`` order is preserved unless ``rng`` is given, in which case it is
    shuffled first (training). Snapshots are never split across batches, so a
    station's neighbours are always in the same forward pass.
    """
    order = np.arange(len(groups))
    if rng is not None:
        order = rng.permutation(order)
    batches, cur, n = [], [], 0
    for i in order:
        g = groups[i]
        cur.append(g)
        n += len(g)
        if n >= target_nodes:
            batches.append(cur)
            cur, n = [], 0
    if cur:
        batches.append(cur)
    return batches


# --------------------------------------------------------------------------- #
# tensors
# --------------------------------------------------------------------------- #
def build_batch(groups: list[np.ndarray], axis: dict, sp: dict, W: int,
                node_of: dict[str, int], edge_i: np.ndarray, edge_j: np.ndarray,
                edge_attr: np.ndarray, n_nodes: int,
                device: torch.device | str | None = None,
                self_loops: bool = True, include_edges: bool = True) -> dict:
    """Tensors for a list of snapshots.

    Returns ``seq/seqm/static/hf/delta/mask`` (one row per present station) plus
    ``edge_index``/``edge_attr`` (local indices, both directions) and
    ``batch`` (which snapshot each row belongs to) for PyG.

    ``include_edges=False`` keeps only the self-loops: that is the *no-edge
    ablation* used by the influence analysis (same weights, graph removed).
    """
    n_feat = len(NUM_COLS)
    P = int(sum(len(g) for g in groups))
    Lmax = 1
    for g in groups:
        for a in g:
            Lmax = max(Lmax, int(sp["ptr"][a + 1] - sp["ptr"][a]))

    seq = np.zeros((P, W, n_feat), np.float32)
    seqm = np.zeros((P, W, 1), np.float32)
    static = np.zeros((P, n_feat), np.float32)
    H = np.full((P, Lmax), PAD, np.float32)
    D = np.zeros((P, Lmax), np.float32)
    M = np.zeros((P, Lmax), np.float32)
    node_global = np.empty(P, np.int64)
    batch_vec = np.empty(P, np.int64)
    anchor_vec = np.empty(P, np.int64)

    present_flag = np.zeros(n_nodes, bool)
    src_parts, dst_parts, ea_parts = [], [], []
    off = 0
    for b, g in enumerate(groups):
        remap = np.full(n_nodes, -1, np.int64)
        for k, a in enumerate(g):
            row = off + k
            end = int(sp["a_axis"][a]) + 1
            start = int(sp["a_runstart"][a])
            lo = max(start, end - W)
            L = end - lo
            if L > 0:
                seq[row, W - L:] = axis["X"][lo:end]
                seqm[row, W - L:, 0] = 1.0
            static[row] = sp["a_static"][a]
            p0, p1 = int(sp["ptr"][a]), int(sp["ptr"][a + 1])
            ln = p1 - p0
            H[row, :ln] = sp["h"][p0:p1]
            D[row, :ln] = sp["delta"][p0:p1]
            M[row, :ln] = 1.0
            node = node_of[str(sp["a_station"][a])]
            node_global[row] = node
            batch_vec[row] = b
            anchor_vec[row] = a
            remap[node] = off + k
            present_flag[node] = True

        if include_edges:
            mask_e = present_flag[edge_i] & present_flag[edge_j]
            if mask_e.any():
                li = remap[edge_i[mask_e]]
                lj = remap[edge_j[mask_e]]
                ek = edge_attr[mask_e]
                src_parts += [li, lj]
                dst_parts += [lj, li]
                ea_parts += [ek, ek]
        off += len(g)
        present_flag[:] = False

    if self_loops:
        loop = np.arange(P, dtype=np.int64)
        src_parts.append(loop)
        dst_parts.append(loop)
        ea_parts.append(np.zeros(P, np.float32))

    if src_parts:
        edge_index = np.stack([np.concatenate(src_parts), np.concatenate(dst_parts)])
        ea = np.concatenate(ea_parts)
    else:                                   # pragma: no cover - P>0 always has loops
        edge_index = np.zeros((2, 0), np.int64)
        ea = np.zeros(0, np.float32)

    hvalid = np.clip(H, 1, MAX_H)
    hf = np.stack([hvalid / MAX_H,
                   np.sin(2 * np.pi * hvalid / MAX_H),
                   np.cos(2 * np.pi * hvalid / MAX_H)], axis=-1).astype(np.float32)

    out = {
        "seq": torch.from_numpy(seq),
        "seqm": torch.from_numpy(seqm),
        "static": torch.from_numpy(static),
        "h": torch.from_numpy(hvalid),
        "hf": torch.from_numpy(hf),
        "delta": torch.from_numpy(D),
        "mask": torch.from_numpy(M),
        "edge_index": torch.from_numpy(edge_index),
        "edge_attr": torch.from_numpy(ea[:, None]),
        "batch": torch.from_numpy(batch_vec),
        "node_global": torch.from_numpy(node_global),
        "anchor": torch.from_numpy(anchor_vec),
        "n_nodes": P,
    }
    if device is not None:
        for k in ("seq", "seqm", "static", "h", "hf", "delta", "mask",
                  "edge_index", "edge_attr", "batch", "node_global", "anchor"):
            out[k] = out[k].to(device, non_blocking=True)
    return out


def edge_attr_from_km(edge_km: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardise the km edge feature (stats are stored in the model config)."""
    m = float(np.mean(edge_km)) if len(edge_km) else 0.0
    s = float(np.std(edge_km)) if len(edge_km) else 1.0
    s = s if s > 1e-6 else 1.0
    return ((edge_km - m) / s).astype(np.float32), m, s


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def selfcheck() -> None:
    """Prove ``anchors_from_frame`` reproduces ``traj_dl._load_split`` exactly."""
    axis = load_axis()
    for name in ("val",):
        mine = load_split(name, axis)
        ref = _TDL._load_split(name, axis)
        for k in ("a_axis", "a_runstart", "a_static", "a_static_m", "ptr", "h", "delta"):
            a, b = mine[k], ref[k]
            assert a.shape == b.shape, (name, k, a.shape, b.shape)
            assert np.array_equal(a, b), (name, k, "values differ")
        assert mine["A"] == ref["A"]
        print(f"selfcheck {name}: A={mine['A']:,} identical to traj_dl._load_split")
    g = snapshot_groups(load_split("val", axis))
    print(f"selfcheck val: {len(g)} snapshots, "
          f"median {int(np.median([len(x) for x in g]))} stations/snapshot")


if __name__ == "__main__":
    selfcheck()
