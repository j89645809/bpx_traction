import unittest
from pathlib import Path
import yaml


class ObservationContractTests(unittest.TestCase):
    def test_exact_contiguous_73_dimensions(self):
        path = Path(__file__).parents[1] / "contracts" / "observation_contract.yaml"
        contract = yaml.safe_load(path.read_text())
        cursor = 0
        for term in contract["terms"]:
            start, end = map(int, term["index"].split(":"))
            self.assertEqual(start, cursor)
            self.assertEqual(end - start, term["dimension"])
            for required in ("unit", "frame", "scale", "latency_ms", "source", "fallback"):
                self.assertIn(required, term)
            cursor = end
        self.assertEqual(cursor, contract["total_dimension"])

    def test_no_privileged_actor_source(self):
        path = Path(__file__).parents[1] / "contracts" / "observation_contract.yaml"
        contract = yaml.safe_load(path.read_text())
        forbidden = set(contract["validation"]["privileged_sources_forbidden_in_actor"])
        self.assertFalse(forbidden & {term["source"] for term in contract["terms"]})

    def test_estimator_input_contract_is_64d_and_deployable(self):
        path = Path(__file__).parents[1] / "contracts" / "estimator_io.yaml"
        contract = yaml.safe_load(path.read_text())
        self.assertEqual(sum(group["dimension"] for group in contract["input_groups"]), 64)
        self.assertEqual(contract["input_dimension"], 64)
        self.assertTrue(all(not group["privileged"] for group in contract["input_groups"]))
        self.assertIn("q_wb", {group["name"] for group in contract["input_groups"]})


if __name__ == "__main__":
    unittest.main()
