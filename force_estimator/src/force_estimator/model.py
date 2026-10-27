"""Temporal proprioceptive force estimator recovered from the Go2 pipeline."""

from __future__ import annotations

import torch
from torch import nn

from .contracts import FORCE_DIM, HISTORY_LENGTH


class ForceEstimator(nn.Module):
    """Estimate a three-component force representation from 25 history frames.

    Input shape is ``[batch, 25, input_dim]``. The model preserves the archived
    architecture and returns ``[batch, 3]``. Its output uses the training target
    units of the archived estimator; use the training normalization/target
    metadata before interpreting it as newtons.
    """

    def __init__(self, input_dim: int, history_length: int = HISTORY_LENGTH, hidden_dim: int = 32):
        super().__init__()
        if history_length != HISTORY_LENGTH:
            raise ValueError(f"the archived architecture requires history_length={HISTORY_LENGTH}")
        self.input_dim = input_dim
        self.history_length = history_length
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=3),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.ELU(),
        )
        self.head = nn.Linear(hidden_dim, FORCE_DIM)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] != self.history_length or history.shape[2] != self.input_dim:
            raise ValueError(f"expected [B, {self.history_length}, {self.input_dim}], got {tuple(history.shape)}")
        encoded = self.encoder(history).transpose(1, 2)
        return self.head(self.temporal(encoded).squeeze(-1))
