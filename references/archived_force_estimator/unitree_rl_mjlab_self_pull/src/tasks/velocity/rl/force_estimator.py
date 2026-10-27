"""GuideDog-style temporal external-force estimator."""

from __future__ import annotations

import torch
from torch import nn


class ForceEstimator(nn.Module):
  """Estimate normalized base-frame force from a proprioceptive history."""

  def __init__(self, input_dim: int, history_length: int = 25, hidden_dim: int = 32):
    super().__init__()
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
    self.head = nn.Linear(hidden_dim, 3)

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    """Args: history [batch, 25, proprioceptive-observation-dim]."""
    encoded = self.encoder(history).transpose(1, 2)
    return self.head(self.temporal(encoded).squeeze(-1))
