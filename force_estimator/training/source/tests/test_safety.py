import unittest
from world_force.safety import SafetyState, SafetySupervisor


class SafetyTests(unittest.TestCase):
    def test_hard_faults_stop(self):
        for field, value in (("roll", 1.0), ("pitch", 1.0), ("height", 0.1), ("communication_age_s", 0.2)):
            args = dict(roll=0.0, pitch=0.0, height=0.3, confidence=1.0, communication_age_s=0.0, dt=0.02)
            args[field] = value
            self.assertEqual(SafetySupervisor().evaluate(**args), SafetyState.SAFE_STOP)

    def test_frame_change_forces_warmup(self):
        s = SafetySupervisor()
        s.reset_frame("W0-a")
        s.state = SafetyState.FOLLOW_READY
        s.reset_frame("W0-b")
        self.assertEqual(s.state, SafetyState.ESTIMATOR_WARMUP)


if __name__ == "__main__":
    unittest.main()
