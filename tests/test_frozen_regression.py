import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from shaft_diagnosis.regression import (canonical_digest, check_endpoints,
    check_synthetic, load_endpoint_bundle, read_json, score_endpoint_bundle,
    validate_endpoint_bundle)


ROOT = Path(__file__).absolute().parents[1]


class FrozenRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic = read_json(ROOT / "configs/regression/synthetic_reference.json")

    def fixture(self):
        return {key: np.asarray(value, dtype=np.int64)
                for key, value in self.synthetic["scoring_input"].items()}

    def test_original_function_synthetic_reference(self):
        result = check_synthetic(ROOT)
        self.assertTrue(result["passed"])
        self.assertFalse(result["experimental_data_used"])
        self.assertFalse(result["neural_inference_executed"])
        self.assertFalse(result["full_raw_data_reproduction"])

    def test_hash_independent_of_integer_storage_width(self):
        self.assertEqual(canonical_digest(np.asarray([1, -1], dtype=np.int8)),
                         canonical_digest(np.asarray([1, -1], dtype=np.int64)))
        self.assertNotEqual(canonical_digest(np.asarray([[1, -1]], dtype=np.int64)),
                            canonical_digest(np.asarray([1, -1], dtype=np.int64)))

    def test_authorized_bundle_round_trip(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "endpoints.npz"
            np.savez(path, **self.fixture())
            arrays = load_endpoint_bundle(path)
        result = check_endpoints(arrays, self.synthetic["scoring_reference"])
        self.assertTrue(result["passed"])

    def test_identical_metrics_do_not_override_identity_mismatch(self):
        arrays = self.fixture()
        arrays["endpoint_key"][:, 0] += 100
        expected = self.synthetic["scoring_reference"]
        self.assertEqual(score_endpoint_bundle(arrays), expected["metrics"])
        result = check_endpoints(arrays, expected)
        self.assertFalse(result["passed"])
        self.assertEqual(result["array_hash_mismatches"], ["endpoint_key"])

    def test_changed_decision_is_not_hidden_by_matching_aggregate_metrics(self):
        arrays = self.fixture()
        arrays["prediction_u_prediction"][0] = 0
        expected = self.synthetic["scoring_reference"]
        self.assertEqual(score_endpoint_bundle(arrays), expected["metrics"])
        result = check_endpoints(arrays, expected, "current-endpoints")
        self.assertFalse(result["passed"])
        self.assertIn("prediction_u_prediction", result["array_hash_mismatches"])
        self.assertFalse(result["neural_inference_executed"])

    def test_incomplete_reference_is_rejected(self):
        expected = copy.deepcopy(self.synthetic["scoring_reference"])
        del expected["array_sha256"]["truth_u_degree"]
        with self.assertRaises(ValueError):
            check_endpoints(self.fixture(), expected)
        expected = copy.deepcopy(self.synthetic["scoring_reference"])
        del expected["metrics"]["u.accuracy"]
        with self.assertRaises(ValueError):
            check_endpoints(self.fixture(), expected)

    def test_invalid_array_schema_is_rejected(self):
        for field, value in (("truth_u_degree", np.zeros(16, dtype=float)),
                             ("prediction_u_ready", np.full(16, 2, dtype=int)),
                             ("prediction_m_joint_prediction", np.full(16, -1, dtype=int)),
                             ("endpoint_key", np.zeros((16, 2), dtype=int)),
                             ("truth_structure_index", np.full(16, 8, dtype=int))):
            arrays = self.fixture()
            arrays[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_endpoint_bundle(arrays)

    def test_missing_authorized_data_is_skip_not_pass(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.run([sys.executable, str(ROOT / "scripts/check_frozen_results.py"),
                                  "--mode", "frozen-endpoints"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, env=env, check=False)
        result = json.loads(process.stdout)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(result["status"], "skipped")
        self.assertFalse(result["passed"])

    def test_nonfinite_reference_is_rejected(self):
        expected = copy.deepcopy(self.synthetic["scoring_reference"])
        expected["metrics"]["structure_accuracy"] = float("nan")
        with self.assertRaises(ValueError):
            check_endpoints(self.fixture(), expected)


if __name__ == "__main__":
    unittest.main()
