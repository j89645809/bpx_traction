"""Causal multi-task TCN for deployable missing-state estimation."""

from __future__ import annotations

import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = 2 * dilation
        self.pad = pad
        self.conv = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(4, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad:
            y = y[..., :-self.pad]
        return x + self.act(self.norm(y))


class MissingStateTCN(nn.Module):
    """Input [B,T,D]; outputs values [B,15] and confidence [B,4]."""

    def __init__(self, input_dim: int = 64, channels: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, channels, 1),
            nn.SiLU(),
            CausalBlock(channels, 1),
            CausalBlock(channels, 2),
            CausalBlock(channels, 4),
            CausalBlock(channels, 8),
        )
        self.values = nn.Linear(channels, 15)
        self.confidence = nn.Linear(channels, 4)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(history.transpose(1, 2))[..., -1]
        raw = self.values(encoded)
        velocity = 2.0 * torch.tanh(raw[:, :3])
        height = 0.15 + 0.45 * torch.sigmoid(raw[:, 3:4])
        contact = torch.sigmoid(raw[:, 4:8])
        load = 2.0 * torch.sigmoid(raw[:, 8:12])
        force = 80.0 * torch.tanh(raw[:, 12:15])
        values = torch.cat((velocity, height, contact, load, force), dim=1)
        return values, torch.sigmoid(self.confidence(encoded))
