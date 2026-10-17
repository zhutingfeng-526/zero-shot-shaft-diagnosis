import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from shaft_diagnosis.structure_decoder import equal_consensus, decode_margin_strict_v12
from shaft_diagnosis.labels import PUBLIC_STRUCTURES, public_label

ROOT = Path(os.path.abspath(__file__)).parents[1]


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from shaft_diagnosis.structure import StructurePredictor
        torch.set_num_threads(2)
        cls.predictor = StructurePredictor(ROOT / 'models/structure')

    def test_tied_vote_uses_latest_then_source(self):
        votes = {56: np.array([1, 1]), 57: np.array([1, 1]),
                 58: np.array([2, 2]), 59: np.array([2, 2]), 60: np.array([3, 2])}
        reliability = {56: .9, 57: .9, 58: .5, 59: .5, 60: .4}
        pred, prob, entropy, tie, _ = equal_consensus(votes, list(votes), reliability)
        np.testing.assert_array_equal(pred, [1, 2])
        np.testing.assert_array_equal(tie, [True, False])
        np.testing.assert_allclose(prob.sum(1), 1.0)

    def test_public_labels(self):
        self.assertEqual(public_label('UMR'), 'UMI')
        self.assertEqual(PUBLIC_STRUCTURES[3], 'I')

    def test_frozen_decoder_empty_and_large_margins(self):
        calibration = json.loads((ROOT / 'models/structure/calibration.json').read_text(encoding='utf-8'))
        result = decode_margin_strict_v12(np.array([[-10, -10, -10], [10, 10, 10]]), np.array([0, 2]), calibration['decoder'])
        np.testing.assert_array_equal(result, [[0, 0, 0], [1, 1, 1]])

    def test_missing_prototypes_are_explicit(self):
        from shaft_diagnosis.structure import StructurePredictor
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            shutil.copyfile(ROOT / 'models/structure/manifest.json', Path(temp) / 'manifest.json')
            with self.assertRaisesRegex(FileNotFoundError, 'prototypes_epoch_056'):
                StructurePredictor(temp)

    def test_five_models_predict_from_windows(self):
        result = self.predictor.predict(np.zeros((2, 10, 6000), dtype=np.float32))
        self.assertEqual(result['checkpoint_decisions'].shape, (5, 2))
        self.assertEqual(result['structure_index'].shape, (2,))
        np.testing.assert_allclose(result['probabilities'].sum(1), 1.0)
        self.assertTrue(np.isfinite(result['entropy']).all())

    def test_prototype_hash_schema_and_checkpoint_binding(self):
        from shaft_diagnosis.structure import _load_prototypes_npz
        import torch
        source = ROOT / 'models/structure/prototypes_epoch_056.npz'
        model = self.predictor.models[56]
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            _load_prototypes_npz(source, '0' * 64, model, torch.device('cpu'))
        with np.load(source, allow_pickle=False) as stored:
            arrays = {k: stored[k].copy() for k in stored.files}
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            bad = Path(temp) / 'bad.npz'
            altered = dict(arrays)
            altered['P_M'] = arrays['P_M'][:2]
            np.savez(bad, **altered)
            with self.assertRaisesRegex(ValueError, 'schema'):
                _load_prototypes_npz(bad, hashlib.sha256(bad.read_bytes()).hexdigest(), model, torch.device('cpu'))
            altered = dict(arrays)
            altered['p_U0'] = arrays['p_U0'] + 0.01
            np.savez(bad, **altered)
            with self.assertRaisesRegex(ValueError, 'does not match checkpoint'):
                _load_prototypes_npz(bad, hashlib.sha256(bad.read_bytes()).hexdigest(), model, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
