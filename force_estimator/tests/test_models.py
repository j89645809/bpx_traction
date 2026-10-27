import pytest

torch = pytest.importorskip("torch")

from force_estimator.model import ForceEstimator
from force_estimator.world_model import MissingStateTCN


def test_force_estimator_shape():
    model = ForceEstimator(input_dim=64)
    y = model(torch.zeros(2, 25, 64))
    assert y.shape == (2, 3)


def test_world_model_shape_and_bounds():
    model = MissingStateTCN()
    values, confidence = model(torch.zeros(2, 25, 64))
    assert values.shape == (2, 15)
    assert confidence.shape == (2, 4)
    assert torch.all(values[:, :3].abs() <= 2.0)
    assert torch.all(values[:, 12:15].abs() <= 80.0)
    assert torch.all((confidence >= 0) & (confidence <= 1))


def test_world_model_is_causal_at_last_timestep():
    model = MissingStateTCN().eval()
    prefix = torch.randn(1, 12, 64)
    a = torch.cat((prefix, torch.zeros(1, 13, 64)), dim=1)
    b = torch.cat((prefix, torch.randn(1, 13, 64)), dim=1)
    va, ca = model(a)
    vb, cb = model(b)
    torch.testing.assert_close(va, vb)
    torch.testing.assert_close(ca, cb)
