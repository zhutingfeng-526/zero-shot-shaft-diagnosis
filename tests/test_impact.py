"""Public synthetic tests; no experiment data or historical scripts required."""
import unittest
import numpy as np
from shaft_diagnosis.impact import ImpactExpert, numerical_modules


class TestImpact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ImpactExpert()
        cls.modules = numerical_modules()

    def test_no_events_are_unavailable(self):
        result = self.model.predict(np.zeros((10, 48000), dtype=np.float32), 1200)
        self.assertFalse(result['ready'])
        self.assertIsNone(result['label'])
        np.testing.assert_array_equal(result['probabilities'], np.full(3, 1/3, dtype=np.float32))

    def test_input_contract(self):
        with self.assertRaises(ValueError):
            self.model.predict(np.zeros((10, 6000)), 1200)
        with self.assertRaises(ValueError):
            self.model.predict(np.zeros((10, 48000)), 0)

    def test_directed_hierarchy_preserves_endpoint(self):
        base = np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]], dtype=np.float32)
        p, labels, stages = self.modules['r16'].apply_hierarchy(
            base, np.array([False,False,False]), np.array([True,True,True]))
        np.testing.assert_array_equal(labels, [1,1,2])
        np.testing.assert_array_equal(p[2], base[2])

    def test_c2_eligibility_and_exact_inactive_identity(self):
        probability = np.array([.6,.3,.1], dtype=np.float32)
        for count, margin, expected in [(11,1.,False),(12,-1.,False),(12,1.,True)]:
            score = {'ready':True,'h1_prediction':0,'event_count':count,
                     'acoustic_margin':margin,'h1_probability':probability}
            output,label,gate = self.modules['causal'].apply_c2(self.modules, score, 0.)
            self.assertEqual(gate, expected)
            np.testing.assert_array_equal(output, probability[[1,0,2]] if expected else probability)

    def test_maturity_requires_complete_carrier(self):
        masks = self.modules['causal'].event_coordinate_masks(
            np.array([511,512,47000,47900]), np.array([6,6,6,6]), 0, 48000, 0, 48000)
        np.testing.assert_array_equal(masks['mature'], [False,True,True,False])

    def test_impulse_pipeline_is_finite_and_deterministic(self):
        rng = np.random.RandomState(4)
        raw = rng.normal(0, .01, (10,48000)).astype(np.float32)
        raw[:6,1000:47000:2000] += 10
        first = self.model.predict(raw, 1200)
        second = self.model.predict(raw, 1200)
        self.assertTrue(first['ready'])
        self.assertGreaterEqual(first['event_count'], 12)
        self.assertTrue(np.isfinite(first['probabilities']).all())
        self.assertAlmostEqual(float(np.sum(first['probabilities'])), 1., places=6)
        np.testing.assert_array_equal(first['probabilities'], second['probabilities'])


if __name__ == '__main__':
    unittest.main()
