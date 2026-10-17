import unittest
import numpy as np
from shaft_diagnosis.preprocessing import (
    FILTER_SPEC_HASH, filter_spec_hash, preprocess_record, window_at,
)


class PreprocessingTests(unittest.TestCase):
    def test_filter_configuration_matches_frozen_data4(self):
        self.assertEqual(filter_spec_hash(), FILTER_SPEC_HASH)
        self.assertEqual(FILTER_SPEC_HASH, "8b0189e0d2")

    def test_full_acquisition_tail_requires_training_duration(self):
        with self.assertRaisesRegex(ValueError, '300 seconds'):
            preprocess_record(np.zeros((10, 1000), dtype=np.float32))

    def test_window_preserves_float16_storage_rounding(self):
        x = np.random.RandomState(17).normal(size=(10, 6200)).astype(np.float32)
        expected = x[:, 150:6150].astype(np.float16)
        np.testing.assert_array_equal(window_at(x, 150, as_float32=False), expected)
        np.testing.assert_array_equal(window_at(x, 150), expected.astype(np.float32))
        with self.assertRaises(ValueError):
            window_at(x, 201)


if __name__ == '__main__':
    unittest.main()
