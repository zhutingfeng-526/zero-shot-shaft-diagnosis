"""Raw M frontend: frozen physical views, K5 features, anchors and event states."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import m_features_p0 as p0
from . import m_features_p1 as p1
from . import m_features_p2 as p2
from .m_features_background import robust_transform
from .m_features_events import detector_state_contract, rotation_state_rows, rotation_feature_matrix
from .m_features_k5 import k5_batch_output
from .misalignment_anchors import (
    DualAttributeNet, TopologyResidualPosition, prepare_inputs, predict_net,
    position_prob_from_model, probability_logits,
)
from .structure_model import RealSemanticSameSpaceModel


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def read_model(directory, stem):
    meta = json.loads((directory / (stem + '.json')).read_text(encoding='utf-8'))
    arrays = read_arrays(directory / (stem + '.npz'))
    meta.update({key[5:]: value for key, value in arrays.items() if key.startswith('meta.')})
    state = {key: torch.from_numpy(value) for key, value in arrays.items() if not key.startswith('meta.')}
    return meta, state


class MRawFrontend:
    """Uses shared K5 encoder weights directly; no structure prototypes are needed."""

    def __init__(self, model_dir, device='cpu', structure_dir=None):
        self.directory = Path(model_dir)
        self.device = torch.device(device)
        self.schema = json.loads((self.directory / 'background_feature_schema.json').read_text(encoding='utf-8'))
        self.indices = read_arrays(self.directory / 'background_feature_indices.npz')
        self.normalizers = read_arrays(self.directory / 'background_feature_normalizers.npz')
        self.anchors = {}
        for name in ('p4_position_anchor', 'p4_degree_anchor'):
            meta, state = read_model(self.directory, name)
            model = DualAttributeNet(len(meta['position_indices']), len(meta['degree_indices']),
                                     meta['hidden_dim'], meta['shared'])
            model.load_state_dict(state, strict=True)
            self.anchors[name] = (model.to(self.device).eval(), meta)
        meta, state = read_model(self.directory, 'p4_topology_anchor')
        model = TopologyResidualPosition(meta['feature_count'], meta['hidden_dim'], meta['topology_hidden_dim'])
        model.load_state_dict(state, strict=True)
        self.topology = (model.to(self.device).eval(), meta)
        self.structure_dir = Path(structure_dir) if structure_dir is not None else self.directory.parent / 'structure'
        manifest = json.loads((self.structure_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.structure_config = manifest['model_config']
        self.k5_members = manifest['members']
        if [int(member['epoch']) for member in self.k5_members] != [56, 57, 58, 59, 60]:
            raise ValueError('M background requires frozen K5 epochs 56 through 60')
        self.k5_models = None
        config = json.loads((self.directory / 'rotation_detectors.json').read_text(encoding='utf-8'))
        self.detector_args = SimpleNamespace(**config['parameters'])
        self.detectors = config['detectors']
        self.selected_detector = next(value for value in self.detectors.values() if value['id'] == config['selected'])

    def _k5(self):
        if self.k5_models is None:
            from .unbalance import _load_state_npz
            self.k5_models = []
            for member in self.k5_members:
                model = RealSemanticSameSpaceModel(self.structure_config)
                _load_state_npz(model, self.structure_dir / member['state_file'], member['sha256'])
                self.k5_models.append(model.to(self.device).eval())
        return self.k5_models

    @torch.no_grad()
    def feature_matrix(self, raw, rpm):
        raw = np.asarray(raw, dtype=np.float32)
        rpm = np.asarray(rpm, dtype=np.float32).reshape(-1)
        if raw.ndim != 3 or raw.shape[1:] != (10, 6000) or len(raw) != len(rpm) or not len(raw):
            raise ValueError('M raw inputs must have shape (N,10,6000), rpm (N,)')
        if not np.isfinite(raw).all() or not np.isfinite(rpm).all() or np.any(rpm <= 0):
            raise ValueError('raw and rpm must be finite; rpm must be positive')
        pieces, non_names = [], None
        for start in range(0, len(raw), 192):
            value = torch.from_numpy(raw[start:start + 192]).to(self.device)
            speed = torch.from_numpy(rpm[start:start + 192]).to(self.device)
            physical, names0 = p0.extract_physical_features(value, speed, 5000.0)
            robust, names1 = p1.extract_robust_evidence(value, speed, 5000.0)
            multiview, names2 = p2.extract_p2_features(value, speed, 5000.0)
            non_names = ['p0::' + name for name in names0] + ['p1::' + name for name in names1] + ['p2::' + name for name in names2]
            values = torch.cat((physical, robust, multiview), dim=1).cpu().numpy().astype(np.float32)
            pieces.append(np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6))
        epoch_values, base_names = [], None
        for model in self._k5():
            chunks = []
            for start in range(0, len(raw), 256):
                values, base_names = k5_batch_output(model, self.structure_config, raw[start:start + 256], self.device)
                chunks.append(values)
            epoch_values.append(np.concatenate(chunks, axis=0))
        stack = np.stack(epoch_values, axis=0)
        names = non_names + ['k5::k5_' + name + '_mean' for name in base_names] + ['k5::k5_' + name + '_std' for name in base_names]
        if names != self.schema['names']:
            raise RuntimeError('M raw feature schema does not match the frozen source contract')
        return np.concatenate((np.concatenate(pieces, axis=0), np.mean(stack, axis=0), np.std(stack, axis=0)), axis=1).astype(np.float32)

    def background_features(self, features):
        result = {}
        for output, group in (
            ('station', 'station_physics'), ('orbit', 'orbit_physics'), ('acoustic', 'acoustic_physics'),
            ('station_raw', 'station_raw'), ('orbit_raw', 'orbit_raw'), ('acoustic_raw', 'acoustic_raw'),
            ('global_position', 'global_position'), ('event_context', 'event_context'),
        ):
            result[output] = robust_transform(features[:, self.indices[group]],
                                               self.normalizers[group + '_center'], self.normalizers[group + '_scale'])
        probabilities = {}
        for name, (model, meta) in self.anchors.items():
            xp, xd = prepare_inputs({'x': features}, meta['position_indices'], meta['degree_indices'],
                                    meta['position_center'], meta['position_scale'], meta['degree_center'], meta['degree_scale'])
            probabilities[name] = predict_net(model, xp, xd, self.device)
        topology, meta = self.topology
        position, _ = position_prob_from_model(topology, {'x': features}, probabilities['p4_position_anchor'][0], meta, self.device)
        result['position_anchor_logits'] = probability_logits(position)
        result['degree_anchor_logits'] = probability_logits(probabilities['p4_degree_anchor'][1])
        return result

    def rotation_features(self, raw, rpm):
        features, states, events = [], [], []
        for value, speed in zip(np.asarray(raw, dtype=np.float32), np.asarray(rpm).reshape(-1)):
            contract = detector_state_contract(value, int(speed), self.detectors['E1'], self.detectors['E2'],
                                               self.selected_detector, self.detector_args)
            matrix, names = rotation_feature_matrix(value, int(speed))
            if matrix.shape[1] != 211:
                raise RuntimeError('P4ZE rotation schema must contain 211 features')
            features.append(matrix)
            states.append(rotation_state_rows(contract, int(speed), value.shape[1]))
            events.append(int(contract['event_count']) > 0)
        return features, states, np.asarray(events, dtype=bool)

    def prepare(self, raw, rpm):
        matrix = self.feature_matrix(raw, rpm)
        rotation, states, events = self.rotation_features(raw, rpm)
        return self.background_features(matrix), rotation, states, events
