"""Shape-safe preprocessing for estimator and BPX handoff tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .contracts import FORCE_DIM, POLICY_FORCE_SCALE_N, WORLD_INPUT_DIM


def load_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a ``mean``/``std`` pair and validate its feature dimension."""
    data = np.load(path)
    missing = {"mean", "std"} - set(data.files)
    if missing:
        raise ValueError(f"normalization file is missing keys: {sorted(missing)}")
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    if mean.shape != (WORLD_INPUT_DIM,) or std.shape != (WORLD_INPUT_DIM,):
        raise ValueError(
            f"expected mean/std shape ({WORLD_INPUT_DIM},), got {mean.shape}/{std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization mean/std must be finite and std must be positive")
    return mean, std


def normalize_history(history: Any, mean: Any, std: Any) -> Any:
    """Normalize ``[..., 64]`` history while preserving the array/tensor type."""
    shape = getattr(history, "shape", ())
    if len(shape) < 2 or shape[-1] != WORLD_INPUT_DIM:
        raise ValueError(f"history must have at least two dimensions and end in {WORLD_INPUT_DIM}")
    if getattr(mean, "shape", ()) != (WORLD_INPUT_DIM,) or getattr(std, "shape", ()) != (WORLD_INPUT_DIM,):
        raise ValueError("mean and std must each have shape (64,)")
    return (history - mean) / std


def normalize_force_for_policy(force_body_n: Any) -> Any:
    """Convert body-frame force in newtons to the BPX policy's [-1, 1] input."""
    if getattr(force_body_n, "shape", ())[-1] != FORCE_DIM:
        raise ValueError("force must have a final dimension of 3")
    # Works for NumPy and torch tensors because both implement clip/clamp differently;
    # callers may use the matching helper below for torch tensors.
    if isinstance(force_body_n, np.ndarray):
        return np.clip(force_body_n / POLICY_FORCE_SCALE_N, -1.0, 1.0)
    return force_body_n.div(POLICY_FORCE_SCALE_N).clamp(-1.0, 1.0)
