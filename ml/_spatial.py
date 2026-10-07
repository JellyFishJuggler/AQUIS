"""_spatial — station-neighbour features and the ST-GNN justification gate.

Objective (d): decide whether a spatio-temporal graph model is *justified* before
paying for PyTorch, a graph, and a new evaluation protocol. An ST-GNN is a
high-capacity model; this pipeline's entire margin over "assume no change" is
~1.3%, so capacity added before evidence is the most expensive mistake available.

The gate is deliberately cheap and behavioural, not architectural:

    If a neighbour's *anomaly* carries no information about this well's 30-day
    change beyond what this well's own anomaly already carries, then a graph
    model has no signal to exploit — it would only have capacity.

Why anomalies and not levels
----------------------------
Groundwater levels are on per-station datums (73.6% of readings are negative in
this corpus). Averaging raw levels across neighbouring wells averages datums, not
hydrology. Every spatial feature here is built on `gwl_anomaly` — the deviation
from the station's **own trailing baseline** — which is the only cross-station
comparable quantity.

Causality
---------
The trailing baseline is shifted one step, so at time *t* it covers
``[t-window, t-1]`` and never includes ``value[t]``. Neighbour features at *t*
therefore use only information available at *t*. `tests/test_spatial.py` pins
this.

Pure functions only — no filesystem, no pipeline imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _association import haversine_km

__all__ = [
    "K_VALUES", "LENGTH_KM", "haversine_km", "neighbour_graph", "station_anomaly",
    "neighbour_features", "neighbour_lag_features", "spatial_gate",
    "graph_inputs_available",
]

# Neighbour counts to expose. 3/5/10 bracket the plausible interaction range;
# a GNN would learn its own topology, so the gate only needs to know whether
# *any* of these carries signal.
K_VALUES = (3, 5, 10)

# Distance decay for the weighted neighbour mean. Larger than the rainfall scale
# because groundwater heads are smoother than rainfall fields, but still local.
LENGTH_KM = 40.0
D_MIN_KM = 0.05

# Anomaly baseline window in 6 h steps (120 = 30 d), with a long alternative to
# separate "recent departure" from "seasonal departure".
WINDOWS = {"30d": 120, "365d": 1460}


# ---------------------------------------------------------------------------
# neighbour graph
# ---------------------------------------------------------------------------
def neighbour_graph(anchors: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """k-nearest station graph from coordinates.

    `anchors` needs Station, lat, lon. Returns one row per (station, neighbour)
    with `dist_km` and a Gaussian `weight` normalised per station, plus each
    neighbour's `lat`/`lon` so a graph model can be built from the same table.
    """
    need = {"Station", "lat", "lon"}
    missing = need - set(anchors.columns)
    if missing:
        raise KeyError(f"neighbour_graph needs {sorted(need)}; missing {sorted(missing)}")
    a = anchors.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if a.empty:
        return pd.DataFrame(columns=["station", "neighbour", "dist_km", "weight",
                                     "rank", "neighbour_lat", "neighbour_lon"])
    lat = a["lat"].to_numpy(float)
    lon = a["lon"].to_numpy(float)
    sta = a["Station"].astype(str).to_numpy()
    rows: list[dict] = []
    for i in range(len(a)):
        d = haversine_km(lat[i], lon[i], lat, lon)
        d[i] = np.inf                                   # never its own neighbour
        order = np.argsort(d)[:k]
        order = order[np.isfinite(d[order]) & (d[order] > 0)]
        if order.size == 0:
            continue
        dd = d[order]
        w = np.exp(-((np.maximum(dd, D_MIN_KM) / LENGTH_KM) ** 2))
        wsum = float(w.sum())
        w = w / wsum if wsum > 0 else np.full(w.shape, 1.0 / w.shape[0])
        for rank, (ix, dist, wt) in enumerate(zip(order, dd, w), start=1):
            rows.append({
                "station": sta[i], "neighbour": sta[ix],
                "dist_km": float(dist), "weight": float(wt), "rank": rank,
                "neighbour_lat": float(lat[ix]), "neighbour_lon": float(lon[ix]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# anomalies
# ---------------------------------------------------------------------------
def station_anomaly(obs: pd.DataFrame, window_steps: int = 120) -> pd.DataFrame:
    """Per-station deviation from its own **trailing**, causally-shifted baseline.

    `obs` needs Station, time, gwl. Returns Station, time, gwl, anomaly. The
    first step of each station is NaN (no history), which is correct rather than
    fabricated as zero.
    """
    need = {"Station", "time", "gwl"}
    missing = need - set(obs.columns)
    if missing:
        raise KeyError(f"station_anomaly needs {sorted(need)}; missing {sorted(missing)}")
    v = obs[["Station", "time", "gwl"]].sort_values(["Station", "time"]).copy()
    v["gwl"] = pd.to_numeric(v["gwl"], errors="coerce")
    base = (v.groupby("Station", sort=False)["gwl"]
              .transform(lambda s: s.rolling(window_steps, min_periods=1).mean()
                                   .shift(1)))
    v["anomaly"] = v["gwl"] - base
    return v


# ---------------------------------------------------------------------------
# neighbour features
# ---------------------------------------------------------------------------
def neighbour_features(anom: pd.DataFrame, graph: pd.DataFrame,
                       ks: tuple[int, ...] = K_VALUES) -> pd.DataFrame:
    """Weighted neighbour-anomaly statistics per (station, time).

    `anom` needs Station, time, anomaly. Returns Station, time and, per k:
    ``knn{k}_mean``, ``knn{k}_std``, ``knn{k}_dmin_km``, ``knn{k}_n``.

    `n` is the number of neighbours that actually had an anomaly at that
    timestamp — a coverage column, so a sparse neighbourhood can be discounted
    rather than silently contributing a mean over one well.
    """
    if anom.empty or graph.empty:
        return pd.DataFrame(columns=["Station", "time"])
    a = anom[["Station", "time", "anomaly"]].dropna(subset=["time"])
    out: pd.DataFrame | None = None
    for k in ks:
        g = graph[graph["rank"] <= k]
        if g.empty:
            continue
        m = g.merge(a, left_on="neighbour", right_on="Station", how="inner",
                    suffixes=("", "_n"))
        m = m.dropna(subset=["anomaly"])
        if m.empty:
            continue
        m["_wv"] = m["weight"] * m["anomaly"]
        m["_w2"] = m["weight"] ** 2
        grp = m.groupby(["station", "time"], as_index=False).agg(
            _wv=("_wv", "sum"), w=("weight", "sum"),
            _w2=("_w2", "sum"), n=("anomaly", "size"),
            dmin=("dist_km", "min"),
        )
        grp["mean"] = grp["_wv"] / grp["w"].replace(0.0, np.nan)
        # weighted population std, guarding the single-neighbour case
        var = (grp["_w2"] - grp["w"] ** 2 / grp["n"].replace(0, np.nan))
        grp["std"] = np.sqrt(np.maximum(var, 0.0) / grp["w"].replace(0.0, np.nan))
        grp["std"] = grp["std"].where(grp["n"] > 1, np.nan)
        part = grp.rename(columns={"station": "Station"})[
            ["Station", "time", "mean", "std", "dmin", "n"]].rename(columns={
                "mean": f"knn{k}_mean", "std": f"knn{k}_std",
                "dmin": f"knn{k}_dmin_km", "n": f"knn{k}_n"})
        out = part if out is None else out.merge(part, on=["Station", "time"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["Station", "time"])


def neighbour_lag_features(anom: pd.DataFrame, graph: pd.DataFrame, k: int = 5,
                           lags: tuple[int, ...] = (4, 28, 120)) -> pd.DataFrame:
    """Neighbour mean anomaly at earlier steps (1 d / 7 d / 30 d back).

    The physical case for a graph is that a well responds to its neighbours'
    *prior* state, not their simultaneous state. Simultaneous neighbour means are
    partly contemporaneous noise; lagged ones are more plausibly causal.
    """
    base = neighbour_features(anom, graph[graph["rank"] <= k], ks=(k,))
    if base.empty:
        return base
    col = f"knn{k}_mean"
    wide = base.pivot_table(index="time", columns="Station", values=col)
    frames = []
    for lag in lags:
        shifted = wide.shift(lag)
        part = shifted.stack(dropna=False).rename(f"knn{k}_lag{lag}").reset_index()
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["Station", "time"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["Station", "time"], how="outer")
    return out


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """corr(x, y | z) — residualise both on z, then correlate."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 30:
        return float("nan")
    x, y, z = x[ok], y[ok], z[ok]
    Z = np.column_stack([np.ones_like(z), z])
    try:
        bx = np.linalg.lstsq(Z, x, rcond=None)[0]
        by = np.linalg.lstsq(Z, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")
    rx, ry = x - Z @ bx, y - Z @ by
    if rx.std() <= 0 or ry.std() <= 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spatial_gate(target: pd.DataFrame, anom: pd.DataFrame, graph: pd.DataFrame,
                 target_col: str = "target_d", own_anomaly_col: str = "anomaly",
                 k: int = 5) -> dict:
    """Decide whether neighbour information justifies a graph model.

    `target` needs Station, time, `target_col` (the quantity being forecast — the
    30-day change). `anom` needs Station, time, `own_anomaly_col`.

    Reports:
      * raw corr(neighbour anomaly, target)
      * PARTIAL corr, controlling for the well's OWN anomaly — the honest test,
        because a neighbouring well's anomaly is highly correlated with this
        well's own, and the model already has the well's own history
      * the increment in explained variance (partial r^2)
      * residual spatial autocorrelation of the target after removing the well's
        own anomaly, which is the ceiling any spatial model could reach

    `passes` requires |partial r| >= min_partial_r (default 0.10) AND a
    meaningful sample. The threshold is a judgement call, stated so it can be
    argued with — 0.10 is ~1% of variance, deliberately low because the whole
    signal here is thin, and a GNN needs more than 1% to beat a tree.
    """
    min_partial_r = 0.10
    out: dict = {"k": int(k), "min_partial_r": min_partial_r, "passes": False}

    nf = neighbour_features(anom, graph[graph["rank"] <= k], ks=(k,))
    mean_col = f"knn{k}_mean"
    if nf.empty or mean_col not in nf.columns:
        out["reason"] = "no neighbour features could be built"
        return out

    # Select only what we need from `target`: callers routinely pass a frame
    # that already carries `anomaly` (and even `knn*_mean`), and merging those
    # would suffix them to `anomaly_x`/`anomaly_y` — KeyError at dropna.
    t = target[["Station", "time", target_col]]
    own = (anom[["Station", "time", own_anomaly_col]]
           .rename(columns={own_anomaly_col: "_own_anomaly"}))
    df = (t.merge(nf[["Station", "time", mean_col, f"knn{k}_n"]],
                  on=["Station", "time"], how="inner")
           .merge(own, on=["Station", "time"], how="inner"))
    df = df.dropna(subset=[target_col, mean_col, "_own_anomaly"])
    if len(df) < 200:
        out["reason"] = f"only {len(df)} usable rows"
        out["n"] = int(len(df))
        return out

    y = df[target_col].to_numpy(float)
    nb = df[mean_col].to_numpy(float)
    own = df["_own_anomaly"].to_numpy(float)

    raw_r = float(np.corrcoef(nb, y)[0, 1]) if nb.std() > 0 else float("nan")
    part_r = _partial_corr(nb, y, own)
    own_r = float(np.corrcoef(own, y)[0, 1]) if own.std() > 0 else float("nan")

    out.update({
        "n": int(len(df)),
        "n_stations": int(df["Station"].nunique()),
        "raw_neighbour_r": round(raw_r, 4) if np.isfinite(raw_r) else None,
        "own_anomaly_r": round(own_r, 4) if np.isfinite(own_r) else None,
        "partial_neighbour_r": round(part_r, 4) if np.isfinite(part_r) else None,
        "partial_r2": round(part_r ** 2, 5) if np.isfinite(part_r) else None,
        "median_neighbours_used": (float(df[f"knn{k}_n"].median())
                                   if f"knn{k}_n" in df.columns else None),
    })
    if not np.isfinite(part_r):
        out["reason"] = "partial correlation undefined (degenerate covariance)"
        return out
    out["passes"] = bool(abs(part_r) >= min_partial_r)
    out["verdict"] = (
        f"neighbour anomaly carries {100 * part_r ** 2:.2f}% of the target variance "
        f"beyond the well's own anomaly — "
        + ("a graph model is worth attempting" if out["passes"]
           else "NOT enough to justify a GNN; capacity would chase noise")
    )
    return out


def graph_inputs_available(anchors: pd.DataFrame) -> dict:
    """Preconditions for a graph model, reported rather than assumed.

    A GNN needs three things this repository does not automatically have, and
    each has burned time elsewhere in this project:
      * usable coordinates for every node (wells without them are unmappable)
      * no reliance on the global `st_id`/`dist_id` ordinals as a content feature
      * a provenance flag, because the 6 h grid forward-fills up to 3 days
    """
    need = {"Station", "lat", "lon"}
    if not need <= set(anchors.columns):
        return {"ok": False, "reason": f"anchors lack {sorted(need - set(anchors.columns))}"}
    n = int(anchors["Station"].nunique())
    mapped = int(anchors.dropna(subset=["lat", "lon"])["Station"].nunique())
    return {
        "nodes": n,
        "nodes_with_coordinates": mapped,
        "unmappable": n - mapped,
        "coordinates_ok": mapped == n,
        "warns": [
            "`st_id`/`dist_id` are global ordinals and currently dominate importance "
            "— they must be removed before any spatial model, or it will learn identity",
            "the 6 h grid forward-fills up to 3 days (04b_align_6h MAX_FFILL_STEPS=12); "
            "message passing over unflagged filled values propagates fabrication",
        ],
    }
