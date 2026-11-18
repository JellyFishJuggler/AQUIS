"""Research-only experiments for AQUIS.

Nothing in this package is imported by the serving path (``ml/api.py``,
``ml/serving/``, ``ml/refresh/``). It may import torch / torch_geometric and it
writes only inside ``ml/models/research/`` and ``ml/outputs/gnn_influence/``.
"""
