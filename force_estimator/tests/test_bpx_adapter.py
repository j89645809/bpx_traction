import numpy as np

from force_estimator.bpx_adapter import BpxForceEstimator


class FakeIO:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def get_inputs(self):
        return [FakeIO("sensor_history")]

    def get_outputs(self):
        return [FakeIO("values"), FakeIO("confidence")]

    def run(self, _, feed):
        batch = next(iter(feed.values())).shape[0]
        values = np.zeros((batch, 15), dtype=np.float32)
        values[:, 12:15] = [45.0, 0.0, 0.0]
        confidence = np.zeros((batch, 4), dtype=np.float32)
        confidence[:, 3] = 0.8
        return values, confidence


def test_bpx_adapter_scales_force_and_confidence(tmp_path):
    np.savez(tmp_path / "normalization.npz", mean=np.zeros(64, np.float32), std=np.ones(64, np.float32))
    estimator = BpxForceEstimator("unused.onnx", tmp_path / "normalization.npz", session=FakeSession())
    force, confidence = estimator.predict(np.zeros((1, 25, 64), np.float32), np.array([[1, 0, 0, 0]], np.float32))
    np.testing.assert_allclose(force, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(confidence, [[0.8]])


def test_bpx_adapter_rotates_world_force_and_masks_invalid_input(tmp_path):
    np.savez(tmp_path / "normalization.npz", mean=np.zeros(64, np.float32), std=np.ones(64, np.float32))
    estimator = BpxForceEstimator("unused.onnx", tmp_path / "normalization.npz", session=FakeSession())
    # 90-degree body-to-world yaw: world +X is body -Y.
    q = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]], np.float32)
    history = np.zeros((1, 25, 64), np.float32)
    history[0, 0, 0] = np.nan
    force, confidence = estimator.predict(history, q)
    np.testing.assert_allclose(force, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(confidence, [[0.0]])
