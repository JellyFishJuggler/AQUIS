"""Station graph for the AQUIS research GNN.

Nodes are groundwater stations. Edges are the union of

* k-nearest-neighbour pairs by haversine distance (``k = 5``), and
* every pair of stations that share a district (district names are compared
  case-folded, because the feature tables and the station CSV disagree on
  ``KANSIRAM NAGAR`` / ``Kansiram Nagar``),

deduplicated, without self-loops. The single edge feature is the haversine
distance in kilometres.

Station coordinates come from ``data/meta/selected_gwl_stations.csv``; the
station set is the union of the feature axis (``data/features_6h/*.parquet``),
i.e. exactly the stations the model can score.

Research only: the serving forecast path never imports this module.

    python -m research.gnn.graph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # ml/
META = ROOT / "data" / "meta"
FE6 = ROOT / "data" / "features_6h"
OUT_DIR = ROOT / "outputs" / "gnn_influence"
GRAPH_JSON = OUT_DIR / "graph.json"

K = 5
R_EARTH_KM = 6371.0088
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# stations
# --------------------------------------------------------------------------- #
def axis_stations() -> list[str]:
    """Station names on the feature axis (union of the 6h feature splits)."""
    names: set[str] = set()
    for sub in SPLITS:
        p = FE6 / f"{sub}.parquet"
        if not p.exists():
            continue
        col = pd.read_parquet(p, columns=["Station"])["Station"]
        names.update(col.astype(str).unique())
    return sorted(names)


def load_stations(stations: list[str] | None = None) -> pd.DataFrame:
    """``Station, District, Latitude, Longitude`` for every axis station.

    District is taken from the 6h feature tables (the canonical casing used by
    the model) and falls back to the station CSV for any station missing there.
    Raises if a station has no coordinates — the graph cannot be built without
    them, and silently dropping nodes would change the station set.
    """
    stations = sorted(stations or axis_stations())
    coord = pd.read_csv(META / "selected_gwl_stations.csv",
                        usecols=["Station", "District", "Latitude", "Longitude"])
    coord["Station"] = coord["Station"].astype(str)
    coord = coord.drop_duplicates("Station").set_index("Station")

    district: dict[str, str] = {}
    for sub in SPLITS:
        p = FE6 / f"{sub}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["Station", "District"])
        d["Station"] = d["Station"].astype(str)
        for st, di in d.drop_duplicates("Station").itertuples(index=False):
            district.setdefault(str(st), str(di))

    rows = []
    missing_coord, missing_dist = [], []
    for st in stations:
        if st not in coord.index:
            missing_coord.append(st)
            continue
        rec = coord.loc[st]
        lat, lon = float(rec["Latitude"]), float(rec["Longitude"])
        if not (np.isfinite(lat) and np.isfinite(lon)):
            missing_coord.append(st)
            continue
        di = district.get(st) or str(rec["District"])
        if not di or di == "nan":
            missing_dist.append(st)
        rows.append({"station": st, "district": di, "lat": lat, "lon": lon})

    if missing_coord:
        raise ValueError(
            f"{len(missing_coord)} station(s) have no coordinates: {missing_coord[:5]}")
    if missing_dist:
        raise ValueError(
            f"{len(missing_dist)} station(s) have no district: {missing_dist[:5]}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise haversine distance (km) for ``lat``/``lon`` in degrees."""
    la = np.radians(np.asarray(lat, dtype=np.float64))[:, None]
    lo = np.radians(np.asarray(lon, dtype=np.float64))[:, None]
    dlat = la - la.T
    dlon = lo - lo.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _knn_pairs(dist: np.ndarray, k: int) -> set[tuple[int, int]]:
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    k = min(k, d.shape[0] - 1)
    nn = np.argsort(d, axis=1, kind="stable")[:, :k]
    pairs = set()
    for i in range(d.shape[0]):
        for j in nn[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                pairs.add((a, b))
    return pairs


def _district_pairs(districts: np.ndarray) -> set[tuple[int, int]]:
    key = np.array([str(d).casefold().strip() for d in districts])
    pairs = set()
    for d in np.unique(key):
        idx = np.flatnonzero(key == d)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = int(idx[a]), int(idx[b])
                pairs.add((i, j) if i < j else (j, i))
    return pairs


# --------------------------------------------------------------------------- #
# build / io
# --------------------------------------------------------------------------- #
def build_graph(stations: pd.DataFrame | None = None, k: int = K) -> dict:
    """Build the node/edge payload (no file write)."""
    stations = load_stations() if stations is None else stations.reset_index(drop=True)
    lat = stations["lat"].to_numpy(np.float64)
    lon = stations["lon"].to_numpy(np.float64)
    dist = haversine_matrix(lat, lon)

    knn = _knn_pairs(dist, k)
    same = _district_pairs(stations["district"].to_numpy())
    pairs = sorted(knn | same)

    nodes = [
        {"i": int(i), "station": str(r.station), "district": str(r.district),
         "lat": float(r.lat), "lon": float(r.lon)}
        for i, r in enumerate(stations.itertuples(index=False))
    ]
    edges = [{"i": int(i), "j": int(j), "km": float(dist[i, j])} for i, j in pairs]
    n_same_only = len(same - knn)

    return {
        "meta": {
            "kind": "station_graph",
            "rule": "kNN(k=%d, haversine) UNION same-district pairs" % k,
            "k": int(k),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_knn_edges": len(knn),
            "n_same_district_only_edges": n_same_only,
            "n_districts": int(stations["district"].nunique()),
            "edge_feature": "haversine distance (km)",
            "directed": False,
            "self_loops": False,
            "district_matching": "casefold+strip",
            "note": "Research artifact. Not used by the serving forecast path.",
        },
        "nodes": nodes,
        "edges": edges,
    }


def save_graph(graph: dict, path: Path = GRAPH_JSON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=1), encoding="utf-8")
    return path


def load_graph(path: Path = GRAPH_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def graph_arrays(graph: dict) -> dict:
    """Numpy/torch-friendly view: station order + edge index + edge distance."""
    stations = [n["station"] for n in graph["nodes"]]
    node_of = {s: i for i, s in enumerate(stations)}
    ei = np.array([e["i"] for e in graph["edges"]], dtype=np.int64)
    ej = np.array([e["j"] for e in graph["edges"]], dtype=np.int64)
    km = np.array([e["km"] for e in graph["edges"]], dtype=np.float32)
    return {"stations": stations, "node_of": node_of, "edge_i": ei,
            "edge_j": ej, "edge_km": km, "n": len(stations)}


def ensure_graph(path: Path = GRAPH_JSON, rebuild: bool = False) -> dict:
    """Load ``graph.json`` or build+save it when missing / ``rebuild``."""
    if path.exists() and not rebuild:
        return load_graph(path)
    g = build_graph()
    save_graph(g, path)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the AQUIS station graph.")
    ap.add_argument("--out", default=str(GRAPH_JSON))
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()
    g = build_graph(k=args.k)
    p = save_graph(g, Path(args.out))
    m = g["meta"]
    print(f"graph -> {p}")
    print(f"  nodes={m['n_nodes']} edges={m['n_edges']} "
          f"(knn={m['n_knn_edges']}, same-district-only={m['n_same_district_only_edges']}) "
          f"districts={m['n_districts']}")
    deg = np.zeros(m["n_nodes"], np.int64)
    for e in g["edges"]:
        deg[e["i"]] += 1
        deg[e["j"]] += 1
    print(f"  degree min={deg.min()} median={int(np.median(deg))} max={deg.max()}")


if __name__ == "__main__":
    main()
