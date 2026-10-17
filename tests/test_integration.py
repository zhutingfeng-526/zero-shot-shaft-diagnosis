import unittest
from pathlib import Path
import numpy as np
from shaft_diagnosis import integration as api
from shaft_diagnosis.evaluation import evaluate_complete


class IntegrationTests(unittest.TestCase):
    def test_load_frozen_presence(self):
        models, config = api.load_presence_models(Path(__file__).absolute().parents[1])
        self.assertEqual(set(models), {"u", "m", "r"})
        self.assertEqual(config["horizon_windows"], 281)

    def test_record_boundary(self):
        segment = np.repeat([0, 1], 300)
        rank = np.tile(np.arange(300), 2)
        self.assertEqual(api.context_indices(segment, rank, [299, 599]).shape, (2, 281))
        with self.assertRaises(ValueError):
            api.context_indices(segment, rank, [300])
        with self.assertRaises(ValueError):
            api.context_indices(segment, rank, [299, 299])

    def test_m_context(self):
        p = np.broadcast_to([0.2, 0.3, 0.5], (2, 281, 3))
        d = np.broadcast_to([0.7, 0.3], (2, 281, 2))
        pos, degree = api.aggregate_m_context(p, d, np.ones((2, 281)))
        np.testing.assert_allclose(pos, [[0.2, 0.3, 0.5]] * 2, atol=1e-7)
        np.testing.assert_allclose(degree, [[0.7, 0.3]] * 2, atol=1e-7)

    def test_u_history(self):
        p = np.broadcast_to([0.1, 0.2, 0.7], (300, 3))
        ids, output = api.aggregate_u(p, np.zeros(300))
        self.assertEqual(ids.tolist(), list(range(280, 300)))
        np.testing.assert_allclose(output, np.tile([0.1, 0.2, 0.7], (20, 1)), atol=1e-7)

    def test_m_history_contract(self):
        with self.assertRaises(ValueError):
            api.aggregate_m_context(np.ones((1, 2, 3)), np.ones((1, 2, 2)), np.ones((1, 2)))
        with self.assertRaises(ValueError):
            api.aggregate_m_context(np.ones((1, 281, 3)), np.ones((1, 281, 2)), np.full((1, 281), 0.5))

    def test_conditioned_output(self):
        out = api.assemble_diagnosis([0, 4, 7], [2, 2, 2], [1, 1, 1], [0, 0, 0],
                                    [2, 2, 2], [True] * 3, [True] * 3, [True, True, False])
        self.assertEqual(out["u_prediction"].tolist(), [-1, 2, 2])
        self.assertEqual(out["m_joint_prediction"].tolist(), [-1, 2, 2])
        self.assertEqual(out["i_prediction"].tolist(), [-1, -1, -1])

    def test_complete_requires_available_attributes(self):
        out = api.assemble_diagnosis([0, 7], [2, 2], [1, 1], [0, 0], [2, 2],
                                    [True] * 2, [True] * 2, [True, False])
        truth = {"structure_index": [0, 7], "u_degree": [-1, 2], "m_position": [-1, 1],
                 "m_degree": [-1, 0], "i_position": [-1, 2]}
        result = evaluate_complete(out, truth, [1, 2], [1, 1])
        self.assertEqual(result["structure_accuracy"], 1.0)
        self.assertEqual(result["complete_diagnosis_accuracy"], 0.5)

    def test_coverage_requires_predicted_presence(self):
        out = api.assemble_diagnosis([0], [0], [0], [0], [0], [True], [True], [True])
        truth = {"structure_index": [1], "u_degree": [0], "m_position": [-1],
                 "m_degree": [-1], "i_position": [-1]}
        result = evaluate_complete(out, truth, [1], [1])
        self.assertEqual(result["attributes"]["u"]["coverage"], 0.0)
        self.assertEqual(result["complete_diagnosis_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
