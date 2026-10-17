import unittest
from types import SimpleNamespace

import numpy as np

from shaft_diagnosis.system import DiagnosisPredictor, history_windows


class SystemTests(unittest.TestCase):
    def test_window_geometry_and_rounding(self):
        raw = np.broadcast_to(np.arange(48000, dtype=np.float32) / 1000, (10, 48000))
        history, windows = history_windows(raw)
        self.assertEqual(windows.shape, (281, 10, 6000))
        np.testing.assert_array_equal(windows[0], history[:, :6000])
        np.testing.assert_array_equal(windows[-1], history[:, 42000:])
        np.testing.assert_array_equal(windows[:-1, :, 150:], windows[1:, :, :-150])
        np.testing.assert_array_equal(history, raw.astype(np.float16).astype(np.float32))

    def test_reject_bad_history(self):
        for raw in (np.zeros((10, 47999)), np.full((10, 48000), np.nan),
                    np.full((10, 48000), 70000)):
            with self.assertRaises(ValueError):
                history_windows(raw)

    def test_complete_path_and_base_i_selection(self):
        # Fixed branch doubles isolate orchestration from neural numerical tests.
        predictor = DiagnosisPredictor.__new__(DiagnosisPredictor)
        seen = []

        def structure(raw):
            n = len(raw)
            return {"probabilities": np.tile([.001] * 7 + [.993], (n, 1)),
                    "structure_index": np.full(n, 7), "bits": np.ones((n, 3), dtype=np.int8)}

        def unbalance(raw, rpm, mask):
            seen.extend(mask.tolist())
            return {"probabilities": np.tile([.1, .2, .7], (len(raw), 1))}

        predictor.structure = SimpleNamespace(predict=structure)
        predictor.unbalance = SimpleNamespace(predict=unbalance)
        predictor.misalignment = SimpleNamespace(predict=lambda raw, rpm, workers: {
            "position_prob": np.tile([.2, .6, .2], (len(raw), 1)),
            "degree_prob": np.tile([.8, .2], (len(raw), 1)),
            "degree_route_guard": np.ones(len(raw), dtype=np.uint8)})
        predictor.impact = SimpleNamespace(predict=lambda raw, rpm, **kw: {
            "probabilities": np.array([1/3] * 3), "legacy_prediction": 0,
            "ready": False, "event_count": 0, "c2_applied": False})
        from pathlib import Path
        from shaft_diagnosis.integration import load_presence_models
        predictor.presence, predictor.config = load_presence_models(Path(__file__).absolute().parents[1])
        result = predictor.predict(np.zeros((10, 48000)), 1200, batch_size=19)
        self.assertEqual(len(seen), 281)
        self.assertTrue(all(seen))
        self.assertEqual(result["structure_label"].tolist(), ["N"])
        self.assertEqual(result["u_prediction"].tolist(), [-1])
        self.assertEqual(result["m_joint_prediction"].tolist(), [-1])
        self.assertEqual(result["u_probability"].argmax(1).tolist(), [2])
        self.assertEqual(result["m_position_probability"].argmax(1).tolist(), [1])
        self.assertEqual(result["i_prediction"].tolist(), [-1])
        self.assertEqual(result["i_ready"].tolist(), [False])


if __name__ == "__main__":
    unittest.main()
