import os
from pathlib import Path
import unittest
import numpy as np
from shaft_diagnosis.unbalance_consensus import ordinal_probabilities, build_shift_consensus

ROOT = Path(os.path.abspath(__file__)).parents[1]


class UnbalanceTests(unittest.TestCase):
    def test_median_ignores_one_extreme_member_and_preserves_raw_rows(self):
        contract = {'threshold_init': [.2, .8], 'transport_scale': .5}
        anchor = np.array([0., 1.])
        raw = {'anchor': anchor, 'evidence': anchor.copy(), 'deep_evidence': np.zeros(2),
               'probabilities': ordinal_probabilities(anchor, contract), 'confidence': np.ones(2), 'uncertainty': np.zeros(2)}
        shared = {i: {'deep_evidence': np.full(2, v)} for i, v in enumerate([.1, .2, .2, .3, 100])}
        out = build_shift_consensus(raw, shared, list(shared), np.array([False, True]), contract, 'median')
        np.testing.assert_array_equal(out['probabilities'][0], raw['probabilities'][0])
        np.testing.assert_allclose(out['deep_evidence'], [0, .2])

    def test_released_models_raw_inference(self):
        from shaft_diagnosis.unbalance import UnbalancePredictor
        rng = np.random.RandomState(17)
        x = rng.normal(size=(2, 10, 6000)).astype(np.float32)
        model = UnbalancePredictor(ROOT / 'models/unbalance')
        out = model.predict(x, np.array([1200, 1500]), np.array([False, True]))
        self.assertEqual(out['probabilities'].shape, (2, 3))
        np.testing.assert_allclose(out['probabilities'].sum(1), 1., atol=1e-6)
        self.assertEqual(float(out['deep_evidence'][0]), 0.)


if __name__ == '__main__':
    unittest.main()
