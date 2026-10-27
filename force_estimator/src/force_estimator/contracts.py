"""Tensor contracts shared by the force estimator and the BPX policy."""

from __future__ import annotations

from dataclasses import dataclass

HISTORY_LENGTH = 25
WORLD_INPUT_DIM = 64  # legacy MissingStateTCN artifact
BPX_INPUT_DIM = 45
WORLD_VALUES_DIM = 15
WORLD_CONFIDENCE_DIM = 4
FORCE_DIM = 3
POLICY_FORCE_SCALE_N = 45.0
WORLD_FORCE_LIMIT_N = 80.0

# MissingStateTCN values: velocity(3), height(1), contact(4), load(4), force(3).
FORCE_VALUE_SLICE = slice(12, 15)


@dataclass(frozen=True)
class BpxForceContract:
    """Direct BPX force-estimator contract from the upstream 45-D policy frame."""

    history_length: int = HISTORY_LENGTH
    input_dim: int = BPX_INPUT_DIM
    force_dim: int = FORCE_DIM
    confidence_dim: int = 1
    sample_period_s: float = 0.02
    output_frame: str = "world"
    policy_frame: str = "body"


@dataclass(frozen=True)
class WorldForceContract:
    """Runtime contract for the causal world-force model."""

    history_length: int = HISTORY_LENGTH
    input_dim: int = WORLD_INPUT_DIM
    values_dim: int = WORLD_VALUES_DIM
    confidence_dim: int = WORLD_CONFIDENCE_DIM
    sample_period_s: float = 0.02
    output_frame: str = "world"
    policy_frame: str = "body"

    @property
    def history_shape(self) -> tuple[int, int, int]:
        return (1, self.history_length, self.input_dim)

    @property
    def output_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((1, self.values_dim), (1, self.confidence_dim))
