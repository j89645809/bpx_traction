from __future__ import annotations

import os

import torch


def residual_gate_placeholder(env) -> torch.Tensor:
    """Allocate the stored gate observation; the residual runner fills it."""
    return torch.zeros((env.num_envs, 1), device=env.device)


def emergency_features_placeholder(env) -> torch.Tensor:
    """Allocate stored emergency-controller inputs; runner fills them per step."""
    feature_dim = 60 if os.environ.get("FINETUNE_EMERGENCY_HISTORY", "0") == "1" else 12
    return torch.zeros((env.num_envs, feature_dim), device=env.device)


def emergency_gate_placeholder(env) -> torch.Tensor:
    """Allocate the true-force emergency gate; runner fills it per step."""
    return torch.zeros((env.num_envs, 1), device=env.device)
