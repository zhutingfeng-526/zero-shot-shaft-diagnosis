"""Observable warnings for the existing preprocessing and feature fallbacks."""

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np

from shaft_diagnosis import i_features_causal, m_features_synchronous, preprocessing


class FallbackTests(unittest.TestCase):
    def test_decimation_failure_warns_without_changing_the_fir_fallback(self):
        signal = np.arange(600, dtype=np.float32) / 600.0
        cutoff = 2200.0 / 5000.0
        index = np.arange(401)
        kernel = np.sinc(cutoff * (index - 200.0))
        kernel *= 0.54 - 0.46 * np.cos(2 * np.pi * (index / 400))
        kernel /= np.sum(kernel)
        expected = np.convolve(signal, kernel, mode="same")[::2].astype(np.float32)
        with patch("scipy.signal.decimate", side_effect=ValueError("test failure")):
            with self.assertWarnsRegex(RuntimeWarning, "convolution-based.*fallback"):
                actual = preprocessing.anti_alias_and_decimate(signal)
        np.testing.assert_array_equal(actual, expected)

    def test_successful_decimation_does_not_warn(self):
        expected = np.arange(10, dtype=np.float32)
        with patch("scipy.signal.decimate", return_value=expected) as decimate:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                actual = preprocessing.anti_alias_and_decimate(np.zeros(20))
        self.assertEqual(caught, [])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(decimate.call_args.kwargs, {"q": 2, "ftype": "fir", "zero_phase": True})

    def test_event_mask_failure_warns_and_returns_the_existing_zero_mask(self):
        detector = Mock(side_effect=RuntimeError("test failure"))
        modules = {"r13": SimpleNamespace(p7a=SimpleNamespace(robust_event_mask=detector))}
        with self.assertWarnsRegex(RuntimeWarning, "all-false mask"):
            actual = i_features_causal.causal_event_mask(
                modules, np.zeros((10, 64)), np.zeros(0), np.zeros(0),
            )
        np.testing.assert_array_equal(actual, np.zeros(64, dtype=bool))

    def test_wrong_length_event_mask_warns_and_returns_the_existing_zero_mask(self):
        detector = Mock(return_value=np.ones(63, dtype=bool))
        modules = {"r13": SimpleNamespace(p7a=SimpleNamespace(robust_event_mask=detector))}
        with self.assertWarnsRegex(RuntimeWarning, "unexpected length"):
            actual = i_features_causal.causal_event_mask(
                modules, np.zeros((10, 64)), np.zeros(0), np.zeros(0),
            )
        np.testing.assert_array_equal(actual, np.zeros(64, dtype=bool))

    def test_successful_event_mask_does_not_warn(self):
        expected = np.arange(64) % 3 == 0
        detector = Mock(return_value=expected)
        modules = {"r13": SimpleNamespace(p7a=SimpleNamespace(robust_event_mask=detector))}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = i_features_causal.causal_event_mask(
                modules, np.zeros((10, 64)), np.zeros(0), np.zeros(0),
            )
        self.assertEqual(caught, [])
        np.testing.assert_array_equal(actual, expected)

    def test_explicit_event_mask_path_is_unchanged(self):
        expected = np.arange(64) % 2 == 0
        detector = Mock(return_value=expected)
        modules = {"r13": SimpleNamespace(event_free_mask=detector)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = i_features_causal.causal_event_mask(
                modules, np.zeros((10, 64)), np.array([10]), np.array([3]),
            )
        self.assertEqual(caught, [])
        np.testing.assert_array_equal(actual, expected)

    def test_scipy_import_failure_warns_and_keeps_the_existing_phase_fallback(self):
        original_import = builtins.__import__

        def import_without_scipy_signal(name, *args, **kwargs):
            if name == "scipy.signal":
                raise ImportError("test failure")
            return original_import(name, *args, **kwargs)

        specification = importlib.util.spec_from_file_location(
            "_synchronous_fallback_test", Path(m_features_synchronous.__file__),
        )
        isolated = importlib.util.module_from_spec(specification)
        with patch("builtins.__import__", side_effect=import_without_scipy_signal):
            with self.assertWarnsRegex(RuntimeWarning, "without Savitzky-Golay"):
                specification.loader.exec_module(isolated)
        self.assertIsNone(isolated.hilbert)
        self.assertIsNone(isolated.savgol_filter)
        time = np.arange(6000) / 5000.0
        phase, coherence = isolated.analytic_vibration_phase(
            np.cos(2 * np.pi * 20 * time), np.zeros(6000, dtype=bool),
        )
        self.assertTrue(np.isfinite(phase).all())
        self.assertEqual(phase.shape, (6000,))
        self.assertAlmostEqual(coherence, 1.0)


if __name__ == "__main__":
    unittest.main()
