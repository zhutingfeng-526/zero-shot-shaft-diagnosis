"""I0/I1/I2 causal impact expert (legacy numerical names R0/R1/R2).

Input is an ordered, simultaneous 10-channel, 5000 Hz trailing 9.6 s
buffer. No original experiment directory, event cache or target label is read.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import warnings
import numpy as np

_CORE_FILES = {'causal':'causal', 'r1':'event_detection', 'r13':'projective',
               'r13b':'baseline', 'r14':'carrier', 'r15':'relations',
               'r16':'hierarchy', 'r18':'acoustic', 'p7a':'event_mask',
               'p4zc':'phase_detector', 'p4zb':'event_trace', 'r6':'background'}
_NUMERICAL_LOCK = RLock()


def numerical_modules():
    """Bind event detection, spatial representation and local review modules."""
    modules = {name: importlib.import_module('.i_features_' + filename, __package__)
               for name, filename in _CORE_FILES.items()}
    extracted = list(modules.values())
    modules['r0'] = SimpleNamespace(p4zc=modules['p4zc'])
    for module in extracted:
        module._bind(modules)
    return modules


def load_model(model_dir=None):
    """Load inference-only frozen parameters; verify archive SHA-256."""
    root = Path(model_dir) if model_dir else Path(__file__).absolute().parents[2] / 'models' / 'impact'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    archive = root / manifest['arrays_file']
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest['arrays_sha256']:
        raise RuntimeError('Impact frozen model checksum mismatch')
    with np.load(archive, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    modules = numerical_modules()
    return modules['causal'].decode_model_tree(manifest['model_tree'], arrays), manifest


def build_causal_cell(raw, rpm, parameters, config, *, condition_id=0,
                      run_id=0, start_sample=0, modules=None):
    """Extract one causal cell. Coordinates are samples within the continuous run.

    Event detection and spatial features use the same trailing buffer.
    The caller's numerical lock protects the shared normalization binding.
    """
    modules = modules or numerical_modules()
    c = modules['causal']
    value = np.asarray(raw, dtype=np.float32)
    expected = (10, int(config['buffer_samples']))
    if value.shape != expected:
        raise ValueError('Impact raw must have shape %r, got %r' % (expected, value.shape))
    if not np.isfinite(value).all():
        raise ValueError('Impact raw contains non-finite values')
    if not np.isfinite(rpm) or int(rpm) <= 0:
        raise ValueError('rpm must be a positive finite value')
    fragment = c.SourceFragment('inference', int(condition_id), int(run_id), 0,
                                int(start_sample), value, -1, int(rpm), np.zeros(0, dtype=np.int64))
    contract = {'detector': parameters['detector'],
                'args': SimpleNamespace(**config['detector_args']),
                'duplicate_tolerance_samples': config['duplicate_tolerance_samples'],
                'manifest': {'lock_hash': 'frozen-model-archive-verified'}}
    events = c.detect_causal_prefix_events(modules, fragment, 0, value.shape[1], contract)
    centers = np.asarray(events['center_buffer'], dtype=np.int64)
    widths = np.asarray(events['event_width'], dtype=np.int64)
    normalized, normalization_report = c.causal_normalize(
        modules, value, value.shape[1], 0, centers, widths,
        parameters['n0'], config['normalization_id'])
    matrices, matrix_report = c.extract_matrices_from_normalized(
        modules, normalized, events['event_id'], centers, widths, events['nuisance'])
    mature = np.asarray(events['mature_mask'], dtype=bool)
    mature &= np.asarray(matrices['short_valid'], dtype=bool) | np.asarray(matrices['long_valid'], dtype=bool)
    indices = np.where(mature)[0]
    count = len(indices)
    models = parameters['models']
    if count:
        selected = {name: np.asarray(array)[indices]
                    if np.asarray(array).shape and np.asarray(array).shape[0] == len(centers)
                    else np.asarray(array) for name, array in matrices.items()}
        v1, v2, a1 = c.projectors_from_time_covariance(modules, selected['time_covariance'])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
            relation = modules['r15'].build_event_feature_package(selected, models['background'], models['band'], 'K0_clean')
        feature = np.asarray(relation['feature'], dtype=np.float32)
        if not np.isfinite(feature).all():
            raise RuntimeError('Impact relation feature is non-finite')
        valid = np.asarray(relation['valid'], dtype=bool)
        process = modules['r15'].event_process_features(selected)
        vs, acs = c.strength_from_covariance(selected['time_covariance'])
        gv1, gv2, ga1 = (c.mean_projector(modules, x, rank) for x, rank in ((v1, 1), (v2, 2), (a1, 1)))
    else:
        v1 = np.zeros((0, 4, 6, 6), dtype=np.complex64)
        v2 = np.zeros_like(v1)
        a1 = np.zeros((0, 4, 2, 2), dtype=np.complex64)
        gv1 = np.zeros((4, 6, 6), dtype=np.complex64)
        gv2 = np.zeros_like(gv1)
        ga1 = np.zeros((4, 2, 2), dtype=np.complex64)
        feature = np.zeros((0, len(modules['r15'].SUMMARY_FEATURE_NAMES)), dtype=np.float32)
        valid = np.zeros(0, dtype=bool)
        process = np.zeros((0, 6), dtype=np.float32)
        vs = acs = np.zeros((0, 4), dtype=np.float32)
    ready = bool(count and valid.any())
    arrays = {'event_id': events['event_id'][indices], 'center_sample': centers[indices],
              'center_run_sample': events['center_run'][indices],
              'event_width': widths[indices].astype(np.int32),
              'vibration_raw_rank1': v1, 'vibration_raw_rank2': v2, 'acoustic_raw_rank1': a1,
              'group_vibration_raw_rank1': gv1, 'group_vibration_raw_rank2': gv2, 'group_acoustic_raw_rank1': ga1,
              'vibration_strength': vs, 'acoustic_strength': acs,
              'relation_feature': feature, 'relation_valid': valid.astype(np.uint8),
              'process_feature': process, 'truth': np.array([-1], dtype=np.int16),
              'ready': np.array([int(ready)], dtype=np.uint8), 'event_count': np.array([count], dtype=np.int32)}
    return {'arrays': arrays, 'key': fragment.group_key, 'split': 'inference',
            'manifest': {'normalization': normalization_report, 'matrix': matrix_report,
                         'maturity': events['report']}}


class ImpactExpert:
    """Frozen I expert. `ready` must be checked before using its probabilities.

    `predict` takes exactly one buffer: channels [B1x, B1y, B2x, B2y,
    B3x, B3y, orbit-x, orbit-y, acoustic-A, acoustic-B], shape (10,48000).
    `rpm` is the causal buffer's supplied speed, truncated to integer as in R1.
    """
    def __init__(self, model_dir=None, config_path=None):
        self.parameters, self.model_manifest = load_model(model_dir)
        path = Path(config_path) if config_path else Path(__file__).absolute().parents[2] / 'configs' / 'impact.json'
        self.config = json.loads(path.read_text(encoding='utf-8'))
        self.modules = numerical_modules()

    def predict(self, raw, rpm, *, condition_id=0, run_id=0, start_sample=0):
        with _NUMERICAL_LOCK:
            cell = build_causal_cell(raw, rpm, self.parameters, self.config,
                                     condition_id=condition_id, run_id=run_id,
                                     start_sample=start_sample, modules=self.modules)
            score = self.modules['causal'].score_h1_cell(self.modules, cell, self.parameters['models'])
            probability, legacy_prediction, gate = self.modules['causal'].apply_c2(
                self.modules, score, self.parameters['models']['legacy_threshold'])
        # The legacy no-evidence path returns uniform probabilities and argmax 0.
        # Keep that numerical result; expose an unambiguous external unavailable label.
        label = 'I%d' % legacy_prediction if score['ready'] else None
        return {'probabilities': probability, 'class_names': ('I0', 'I1', 'I2'),
                'label': label, 'ready': bool(score['ready']),
                'event_count': int(score['event_count']), 'c2_applied': bool(gate),
                'h1_probability': score['h1_probability'], 'd2_probability': score['d2_probability'],
                'stage': int(score['stage']), 'acoustic_margin': float(score['acoustic_margin']),
                'legacy_prediction': int(legacy_prediction), 'diagnostics': cell['manifest']}


def fit_carrier_prototypes(source_bundles, included_splits):
    """Original group-balanced source-only projector fitting primitive.

    Each split contains truth, group_valid, and group_* projector arrays as in
    `i_features_carrier.fit_carrier_prototypes`; labels are integers 0, 1, 2.
    This helper does not reproduce source split construction or model selection.
    """
    return numerical_modules()['r14'].fit_carrier_prototypes(source_bundles, included_splits)


def fit_orbit_classifier(feature, labels, groups, ridge):
    """Original R15 weighted logistic fit on source orbit features (binary 0/1).

    Gain/cap calibration, frozen band selection, and source split certification
    are separate original operations, not implicitly rerun by this helper.
    """
    return numerical_modules()['r15'].fit_weighted_logistic(feature, labels, groups, ridge)


def fit_background_operators(source_background):
    """Original source context-balanced background operator fitting primitive.

    Expects the original matrix schema (family labels, time/spectral covariance
    arrays). It does not construct the experimental background event bank.
    """
    return numerical_modules()['r15'].fit_background_operator_model(source_background)
