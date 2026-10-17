"""Synthetic M contracts; no experimental records or labels are needed."""
import unittest
import numpy as np

from shaft_diagnosis.misalignment import (
    MisalignmentExpert, ORBIT_Q20, predict_misalignment, prepare_rotation_packet,
    predict_synchronous_experts,
)


class MisalignmentTests(unittest.TestCase):
    def test_final_tv1_only_confirmation(self):
        hp = np.array([[.7, .2, .1]], np.float32)
        proposal = np.array([[.1, .8, .1]], np.float32)
        degree = np.array([[.8, .2]], np.float32)
        quality = np.zeros((1, 9), np.float32)
        a1 = predict_misalignment(hp, degree, hp, proposal, degree, degree, quality,
                                  position_mode='a1')
        final = predict_misalignment(hp, degree, hp, proposal, degree, degree, quality,
                                     tv1_position=proposal, position_mode='final')
        self.assertEqual(int(a1['position'][0]), 0)
        self.assertEqual(int(final['position'][0]), 1)
        with self.assertRaises(ValueError):
            predict_misalignment(hp, degree, hp, proposal, degree, degree, quality, position_mode='final')

    def test_position_mode_must_be_explicit(self):
        position = np.array([[.7, .2, .1]], np.float32)
        degree = np.array([[.8, .2]], np.float32)
        quality = np.zeros((1, 9), np.float32)
        arguments = (position, degree, position, position, degree, degree, quality)
        with self.assertRaisesRegex(ValueError, 'explicitly set'):
            predict_misalignment(*arguments)
        with self.assertRaisesRegex(ValueError, 'explicitly set'):
            predict_misalignment(*arguments, position_mode='unknown')

    def test_final_tv1_probability_contract(self):
        position = np.array([[.7, .2, .1]], np.float32)
        degree = np.array([[.8, .2]], np.float32)
        quality = np.zeros((1, 9), np.float32)
        arguments = (position, degree, position, position, degree, degree, quality)
        for invalid in (
            np.array([[.7, .3]], np.float32),
            np.array([[np.nan, .2, .1]], np.float32),
            np.array([[.7, .2, .2]], np.float32),
            np.array([[1.1, -.1, 0.]], np.float32),
        ):
            with self.subTest(tv1_position=invalid):
                with self.assertRaisesRegex(ValueError, 'tv1_position'):
                    predict_misalignment(*arguments, tv1_position=invalid,
                                         position_mode='final')

    def test_frozen_gates(self):
        h0p = np.tile([.7, .2, .1], (4, 1)).astype(np.float32)
        h0d = np.tile([.8, .2], (4, 1)).astype(np.float32)
        dp = np.tile([.1, .8, .1], (4, 1)).astype(np.float32)
        cp = dp.copy()
        cp[1] = h0p[1]
        dd = np.tile([.1, .9], (4, 1)).astype(np.float32)
        quality = np.zeros((4, 9), np.float32)
        quality[:, 4] = [1., 1., 1., ORBIT_Q20 - .1]
        quality[:, 7] = [1., 0., 1., 1.]
        quality[2, 8] = 1.
        out = predict_misalignment(h0p, h0d, cp, dp, dd, dd, quality,
                                   position_mode='a1')
        np.testing.assert_array_equal(out['position'], [1, 0, 1, 1])
        np.testing.assert_array_equal(out['degree'], [1, 0, 0, 0])
        np.testing.assert_array_equal(out['degree_route_weight'], [1, 0, 0, 0])

    def test_rotation_mask_preserves_original_fallback(self):
        matrices = [np.zeros((4, 211), np.float32)]
        rows = [[{'state': 'CORE', 'overlap_core': True}] * 3 + [{'state': 'CLEAN_OTHER'}]]
        out = prepare_rotation_packet(matrices, rows, [True], np.zeros((1, 3), np.float32),
                                      np.zeros((1, 2), np.float32),
                                      {'center': np.zeros(211), 'scale': np.ones(211)})
        # Original position exclusion is abandoned when fewer than 3 turns remain.
        self.assertTrue(out['position_valid'].all())
        self.assertTrue(out['degree_valid'].all())
        self.assertEqual(float(out['gate'][0]), 1.)

    def test_raw_inference(self):
        import torch
        torch.set_num_threads(2)
        t = np.arange(6000) / 5000.
        raw = np.stack([np.sin(2 * np.pi * 20 * t + .17 * channel)
                        + .2 * np.cos(2 * np.pi * 40 * t + .05 * channel)
                        for channel in range(10)])[None].astype(np.float32)
        model = MisalignmentExpert()
        self.assertEqual(model.position_mode, 'final')
        result = model.predict(raw, np.array([1200], np.float32))
        self.assertEqual(result['position_prob'].shape, (1, 3))
        self.assertEqual(result['degree_prob'].shape, (1, 2))
        for key in ['position_prob', 'degree_prob']:
            self.assertTrue(np.isfinite(result[key]).all())
            np.testing.assert_allclose(result[key].sum(1), 1., atol=1e-6)

    def test_bad_raw_contract_rejected(self):
        with self.assertRaises(ValueError):
            predict_synchronous_experts(np.zeros((1, 10, 5999)))


if __name__ == '__main__':
    unittest.main()
