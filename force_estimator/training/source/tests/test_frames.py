import math
import unittest
import numpy as np

from world_force.frames import body_to_world, world_to_body


class FrameContractTests(unittest.TestCase):
    def test_roundtrip_and_norm(self):
        rng = np.random.default_rng(41)
        q = rng.normal(size=(1000, 4))
        f = rng.normal(size=(1000, 3)) * 80.0
        fw = body_to_world(f, q)
        fb = world_to_body(fw, q)
        self.assertLess(np.max(np.abs(fb - f)), 1.0e-10)
        rel = np.abs(np.linalg.norm(fw, axis=1) - np.linalg.norm(f, axis=1)) / np.maximum(np.linalg.norm(f, axis=1), 1.0e-12)
        self.assertLess(np.max(rel), 1.0e-10)

    def test_fixed_world_force_after_yaw_90(self):
        q = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
        fw = np.array([20.0, 0.0, 0.0])
        fb = world_to_body(fw, q)
        np.testing.assert_allclose(fb, [0.0, -20.0, 0.0], atol=1.0e-10)
        np.testing.assert_allclose(body_to_world(fb, q), fw, atol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
