"""dl_rebuild/models — deep nets with NO station identity.

Both nets predict the 30-day level delta (mean; Huber loss) on a standardized
target, which keeps every gate ratio directly comparable with E0/persistence.
Deliberately simple + stable (the track-3 lesson): LayerNorm after every
Linear/Recurrent cell, ReLU, small dropout, grad clip, ReduceLROnPlateau.

  * MLPNet   — feedforward on the transferable row features (35 dense + 35
               masks + 7 seasonal).  Cheap; isolates the "st_id was the
               problem" hypothesis on the full 600-station frame.
  * GRUNet   — GRU encoder over the trailing GWL trajectory (window W) -> h,
               concat row features -> MLP head.  The flagship sequential
               model for new-station transfer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, p: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GRUNet(nn.Module):
    """GRU over the station's own trailing 6h trajectory + row covariates."""

    def __init__(self, feat_p: int, hidden: int = 32, W: int = 36,
                 seq_in: int = 2, layers: int = 1):
        super().__init__()
        self.W = W
        self.rnn = nn.GRU(seq_in + 1, hidden, num_layers=layers,
                          batch_first=True, dropout=0.2 if layers > 1 else 0.0)
        self.onorm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + feat_p, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, seq: torch.Tensor, seq_mask: torch.Tensor,
                feat: torch.Tensor) -> torch.Tensor:
        # seq: (B, W, C); nan -> 0, first-channel finiteness feeds the RNN
        x = torch.cat([torch.where(torch.isfinite(seq), seq,
                                   torch.zeros_like(seq)),
                       seq_mask[:, :, :1]], dim=-1)
        h, _ = self.rnn(x)
        h = self.onorm(h[:, -1])
        return self.head(torch.cat([h, feat], dim=-1)).squeeze(-1)


def restore_best(model: nn.Module, best_state: dict) -> nn.Module:
    model.load_state_dict(best_state)
    return model