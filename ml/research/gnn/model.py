"""Fresh trajectory-GRU encoder + 2 GAT layers + horizon-conditioned quantile heads.

The encoder is the *same architecture* as the production ``TrajGRU``
(``nn.GRU(nfeat+1, hidden)`` + ``LayerNorm``), instantiated with **fresh
weights** — the production checkpoint ``ml/models/traj_dl.pt`` is never loaded
and never written. After the encoder, station embeddings are refined by two
``GATConv`` layers over the station graph (edge feature = standardised km),
then the unchanged horizon-conditioned quantile head maps
``[embedding ‖ static row ‖ horizon features]`` to ``(q05, q50, q95)``.

Ordering ``q05 <= q50 <= q95`` is guaranteed at inference by sorting the three
outputs, exactly like ``pipelines.traj_dl._val_per_h``.
"""

from __future__ import annotations

import importlib

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

_TDL = importlib.import_module("pipelines.traj_dl")

pinball = _TDL.pinball          # identical loss to the production GRU
QUANTILES = tuple(_TDL.QUANTILES)


class GNNTraj(nn.Module):
    """Trajectory GRU encoder -> GAT message passing -> quantile heads."""

    def __init__(self, nfeat: int, hidden: int = 64, static: int = 30,
                 nq: int = 3, heads: int = 2, dropout: float = 0.1,
                 layers: int = 2):
        super().__init__()
        self.nfeat = int(nfeat)
        self.hidden = int(hidden)
        self.static = int(static)
        self.nq = int(nq)
        self.layers = int(layers)

        # --- encoder: same shape as TrajGRU, fresh weights -------------------
        self.rnn = nn.GRU(nfeat + 1, hidden, num_layers=1, batch_first=True)
        self.onorm = nn.LayerNorm(hidden)

        # --- graph: 2 GAT layers with residual + LayerNorm -------------------
        self.gat = nn.ModuleList([
            GATConv(hidden, hidden, heads=heads, concat=False,
                    edge_dim=1, dropout=dropout, add_self_loops=False)
            for _ in range(self.layers)
        ])
        self.gnorm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(self.layers)])

        # --- head: identical to TrajGRU.head --------------------------------
        self.head = nn.Sequential(
            nn.Linear(hidden + static + 3, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, nq))

    # ------------------------------------------------------------------ #
    def encode(self, seq: torch.Tensor, seqm: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(torch.cat([seq, seqm], dim=-1))
        return self.onorm(h[:, -1])

    def message_pass(self, z: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor) -> torch.Tensor:
        for gat, norm in zip(self.gat, self.gnorm):
            z = norm(z + torch.relu(gat(z, edge_index, edge_attr=edge_attr)))
        return z

    def heads(self, z: torch.Tensor, static: torch.Tensor,
              hf: torch.Tensor) -> torch.Tensor:
        P, L = hf.shape[0], hf.shape[1]
        inp = torch.cat([z[:, None, :].expand(P, L, -1),
                         static[:, None, :].expand(P, L, -1), hf], dim=-1)
        return self.head(inp)

    def forward(self, seq, seqm, static, hf, edge_index, edge_attr):
        z = self.message_pass(self.encode(seq, seqm), edge_index, edge_attr)
        return self.heads(z, static, hf)

    # ------------------------------------------------------------------ #
    def config(self) -> dict:
        return {"arch": "GNNTraj", "nfeat": self.nfeat, "hidden": self.hidden,
                "static": self.static, "nq": self.nq, "gat_layers": self.layers,
                "quantiles": list(QUANTILES)}


def build_model(cfg: dict) -> GNNTraj:
    return GNNTraj(nfeat=int(cfg["nfeat"]), hidden=int(cfg["hidden"]),
                   static=int(cfg.get("static", 30)),
                   nq=int(cfg.get("nq", 3)),
                   heads=int(cfg.get("gat_heads", 2)),
                   dropout=float(cfg.get("dropout", 0.1)),
                   layers=int(cfg.get("gat_layers", 2)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GRUAdapter(nn.Module):
    """Production ``TrajGRU`` behind the GNN forward signature (graph ignored).

    Used for the matched-budget baseline so one training/eval loop drives both
    architectures. It never loads production weights by itself — the caller
    decides what to put inside.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, seq, seqm, static, hf, edge_index, edge_attr):
        return self.net(seq, seqm, static, hf)

    def config(self) -> dict:
        return {"arch": "GRUAdapter", "hidden": int(self.net.rnn.hidden_size)}
