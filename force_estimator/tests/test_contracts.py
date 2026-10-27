from pathlib import Path

import numpy as np
import pytest

from force_estimator.contracts import HISTORY_LENGTH, WORLD_INPUT_DIM, WorldForceContract
from force_estimator.preprocessing import load_normalization, normalize_force_for_policy, normalize_history

ROOT = Path(__file__).parents[1]


def test_world_contract_is_explicit():
    c = WorldForceContract()
    assert c.history_shape == (1, HISTORY_LENGTH, WORLD_INPUT_DIM)
    assert c.output_shapes == ((1, 15), (1, 4))
    assert c.sample_period_s == 0.02


def test_normalization_statistics_are_valid():
    mean, std = load_normalization(ROOT / "config" / "normalization.npz")
    assert mean.shape == (64,)
    assert std.shape == (64,)
    assert np.all(std > 0)
    normalized = normalize_history(np.zeros((2, 25, 64), dtype=np.float32), mean, std)
    assert normalized.shape == (2, 25, 64)


def test_policy_force_is_scaled_and_clipped():
    force = np.array([[45.0, -90.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(normalize_force_for_policy(force), [[1.0, -1.0, 0.0]])


@pytest.mark.parametrize("shape", [(64,), (25, 63), (2, 25, 63)])
def test_normalization_rejects_wrong_feature_dimension(shape):
    mean = np.zeros(64, dtype=np.float32)
    std = np.ones(64, dtype=np.float32)
    with pytest.raises(ValueError):
        normalize_history(np.zeros(shape, dtype=np.float32), mean, std)
