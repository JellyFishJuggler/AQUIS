"""Spatial GNN ("station influence") research package.

Modules
-------
graph      station graph (kNN haversine UNION same-district) -> outputs/gnn_influence/graph.json
data       snapshot batching over the existing purged splits (no new split logic)
model      fresh trajectory-GRU encoder + 2 GAT layers + horizon-conditioned quantile heads
train      python -m research.gnn.train
eval       python -m research.gnn.eval
influence  python -m research.gnn.influence

Everything here is research-only. Checkpoints go to ``ml/models/research/`` and
artifacts to ``ml/outputs/gnn_influence/``; production files are never touched.
"""

__all__ = ["graph", "data", "model", "train", "eval", "influence"]
