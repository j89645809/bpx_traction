"""45-D BPX force estimator trained from the upstream policy history."""

from __future__ import annotations

import torch
from torch import nn

from .contracts import BPX_INPUT_DIM, FORCE_DIM, HISTORY_LENGTH
from .world_model import CausalBlock


class BpxForceTCN(nn.Module):
    """Causal TCN with force and confidence heads for BPX.

    Input is the upstream policy frame (45 values) over 25 samples. The force
    head returns body-frame newtons; confidence is a scalar in [0, 1].
    """

    def __init__(self, input_dim: int = BPX_INPUT_DIM, channels: int = 128):
        super().__init__()
        if input_dim != BPX_INPUT_DIM:
            raise ValueError(f"BPX input dimension is fixed at {BPX_INPUT_DIM}")
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, channels, 1), nn.SiLU(),
            CausalBlock(channels, 1), CausalBlock(channels, 2),
            CausalBlock(channels, 4), CausalBlock(channels, 8),
        )
        self.force = nn.Linear(channels, FORCE_DIM)
        self.confidence = nn.Linear(channels, 1)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[1] != HISTORY_LENGTH or history.shape[2] != self.input_dim:
            raise ValueError(f"expected [B, {HISTORY_LENGTH}, {self.input_dim}], got {tuple(history.shape)}")
        encoded = self.encoder(history.transpose(1, 2))[..., -1]
        force = 80.0 * torch.tanh(self.force(encoded))
        confidence = torch.sigmoid(self.confidence(encoded))
        return force, confidence
