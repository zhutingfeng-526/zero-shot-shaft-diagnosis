"""Misalignment position and severity inference with fixed confirmation gates."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from .m_features_synchronous import (
    PHASE_QUALITY_NAMES, align_batch, rotation_coefficients, t2_features,
    transform_features, predict_dual_linear, softmax_np,
)
from .m_features_rotation import state_vector
from .misalignment_routes import (
    normalize_probability, route_position, build_gates,
    normalize_endpoint_probability, endpoint_softmax,
)

ORBIT_Q20 = 0.981577706336975
BACKGROUND_KEYS = (
    'station', 'orbit', 'acoustic', 'station_raw', 'orbit_raw', 'acoustic_raw',
    'global_position', 'event_context', 'position_anchor_logits', 'degree_anchor_logits',
)


def default_model_dir():
    return Path(__file__).absolute().parents[2] / 'models' / 'misalignment'


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _probability_contract(value, count, classes, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (count, classes) or not np.isfinite(array).all():
        raise ValueError('%s must have finite shape (%d, %d)' % (name, count, classes))
    if np.any(array < 0) or not np.allclose(array.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError('%s must contain nonnegative normalized probabilities' % name)
    return array


def predict_misalignment(h0_position, h0_degree, p4j_position, d2_position,
                        d2_degree, t2_degree, phase_quality, orbit_q20=ORBIT_Q20,
                        tv1_position=None, position_mode=None):
    """Apply an explicitly selected position gate and the fixed severity gate.

    position_mode='final' selects the published position confirmation rule and
    requires tv1_position. position_mode='a1' selects the single-confirmer
    comparator. There is no implicit mode for this lower-level helper.
    Position order is B1/B2/B3; degree order D0/D1. Phase quality has the nine
    columns in PHASE_QUALITY_NAMES. Inputs must refer to the same ordered windows.
    This function performs no fitting and does not use family or target labels.
    """
    if position_mode not in ('final', 'a1'):
        raise ValueError("position_mode must be explicitly set to 'final' or 'a1'")
    count = len(h0_position)
    hp = _probability_contract(h0_position, count, 3, 'h0_position')
    hd = _probability_contract(h0_degree, count, 2, 'h0_degree')
    cp = _probability_contract(p4j_position, count, 3, 'p4j_position')
    dp = _probability_contract(d2_position, count, 3, 'd2_position')
    dd = _probability_contract(d2_degree, count, 2, 'd2_degree')
    td = _probability_contract(t2_degree, count, 2, 't2_degree')
    quality = np.asarray(phase_quality, dtype=np.float64)
    if quality.shape != (count, len(PHASE_QUALITY_NAMES)) or not np.isfinite(quality).all():
        raise ValueError('phase_quality must be a finite N x 9 matrix')
    proposal = np.argmax(dp, axis=1)
    position_gate = (proposal != np.argmax(hp, axis=1)) & (np.argmax(cp, axis=1) == proposal)
    if position_mode == 'final':
        if tv1_position is None:
            raise ValueError("position_mode='final' requires the frozen three-member TV1 probability")
        tv1 = _probability_contract(tv1_position, count, 3, 'tv1_position')
        # The final gate does not use the diagnostic T2 position labels.
        gates, _ = build_gates(hp, dp, cp, tv1, np.zeros(count, dtype=np.int64))
        position_gate = gates['E0_P4JorTV1']
    position = route_position(hp, dp, position_gate)
    hd = normalize_probability(hd)
    mixed = 0.5 * np.asarray(dd, dtype=np.float64) + 0.5 * np.asarray(td, dtype=np.float64)
    expert_degree = normalize_probability(mixed)
    physical = (quality[:, 7] >= 0.5) & (quality[:, 8] < 0.5)
    disagreement = np.argmax(expert_degree, axis=1) != np.argmax(hd, axis=1)
    route_guard = physical & disagreement
    route_weight = (quality[:, 4] >= float(orbit_q20)).astype(np.float64)
    route_weight *= route_guard.astype(np.float64)
    degree = normalize_probability(
        (1.0 - route_weight[:, None]) * np.asarray(hd, dtype=np.float64)
        + route_weight[:, None] * np.asarray(expert_degree, dtype=np.float64)
    )
    if position_mode == 'final':
        # Preserve the final window interface's probability renormalization.
        position = normalize_endpoint_probability(position)
        degree = normalize_endpoint_probability(degree)
    return {
        'position_prob': position, 'degree_prob': degree,
        'position': position.argmax(1).astype(np.int64),
        'degree': degree.argmax(1).astype(np.int64),
        'position_route_gate': position_gate,
        'degree_route_weight': route_weight.astype(np.float32),
        'degree_route_guard': route_guard.astype(np.uint8),
    }


def predict_synchronous_experts(raw, model_dir=None, workers=1):
    """Run final D2/T2 heads from N x 10 x 6000 raw windows at 5000 Hz.

    Channel order: B1x, B1y, B2x, B2y, B3x, B3y, orbit x, orbit y, sound A,
    sound B. Units/scaling must match the experiment input; no channel standard-
    ization is introduced. Original optional-SciPy and nominal-phase fallback
    behavior is retained. Use SciPy for the validated release environment.
    """
    value = np.asarray(raw, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (10, 6000) or len(value) == 0:
        raise ValueError('raw must have nonempty shape (N, 10, 6000)')
    if not np.isfinite(value).all():
        raise ValueError('raw contains nonfinite values')
    directory = Path(model_dir) if model_dir is not None else default_model_dir()
    reference = _arrays(directory / 'synchronous_normal_reference.npz')
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        aligned, quality = align_batch(value, executor)
    coefficients = rotation_coefficients(aligned)
    features, names = t2_features(coefficients, aligned, {
        'vib_inv': reference['main_vib_inv'], 'orbit_inv': reference['main_orbit_inv'],
    })
    output = {'phase_quality': quality}
    for key, filename in [('D2', 'D2_T2YOnly.npz'), ('T2', 'T2_VibOrbitCORS.npz')]:
        model = _arrays(directory / filename)
        scaled = transform_features(features, model['indices'], model['center'], model['scale'])
        output[key] = predict_dual_linear(scaled, model)
    return output


def prepare_rotation_packet(rotation_features, rotation_states, event_present,
                            base_position, base_degree, normalization):
    """Prepare normalized per-turn features and masks for residual inference.

    rotation_features is a sequence of R_i x 211 matrices; rotation_states is
    a matching sequence of row dictionaries consumed by state_vector. No event
    detection is performed here. event_present is the source-frozen detector's
    event_count > 0 flag per window, not a fault/degree truth label.
    """
    count = len(rotation_features)
    if not count or len(rotation_states) != count:
        raise ValueError('rotation feature/state window counts must agree and be positive')
    max_rotations = max(len(value) for value in rotation_features)
    if any(len(value) == 0 for value in rotation_features):
        raise ValueError('every window must have at least one rotation')
    center = np.asarray(normalization['center'], dtype=np.float32)
    scale = np.asarray(normalization['scale'], dtype=np.float32)
    tokens = np.zeros((count, max_rotations, len(center)), dtype=np.float32)
    states = np.zeros((count, max_rotations, 15), dtype=np.float32)
    valid = np.zeros((count, max_rotations), dtype=bool)
    core = np.zeros_like(valid)
    for sample, (matrix, rows) in enumerate(zip(rotation_features, rotation_states)):
        matrix = np.asarray(matrix, dtype=np.float32)
        n = len(matrix)
        if matrix.shape != (n, len(center)) or len(rows) != n or not np.isfinite(matrix).all():
            raise ValueError('rotation feature/state shape or finite contract failed')
        tokens[sample, :n] = np.clip((matrix - center[None, :]) / scale[None, :], -10.0, 10.0)
        valid[sample, :n] = True
        for rotation, row in enumerate(rows):
            states[sample, rotation] = state_vector(row)
            core[sample, rotation] = bool(row.get('overlap_core', False))
    position_valid = valid & ~core
    for sample in range(count):
        if int(np.sum(position_valid[sample])) < 3:
            position_valid[sample] = valid[sample]
    gate = np.asarray(event_present, dtype=np.float32)
    if gate.shape != (count,) or not np.isin(gate, (0, 1)).all():
        raise ValueError('event_present must be one Boolean event-count flag per window')
    return {'tokens': np.clip(tokens, -12.0, 12.0).astype(np.float32), 'states': states,
            'position_valid': position_valid, 'degree_valid': valid, 'gate': gate,
            'base_position': np.asarray(base_position, dtype=np.float32),
            'base_degree': np.asarray(base_degree, dtype=np.float32)}


class MisalignmentExpert:
    """Load published M weights; raw inference also uses shared K5 encoders."""

    def __init__(self, model_dir=None, device='cpu', position_mode='final'):
        import torch
        from .misalignment_models import PositionTopologyAdapter, SafeInnovationRoutingAdapter, TaskSetEncoder, load_adapter_state
        self.directory = Path(model_dir) if model_dir is not None else default_model_dir()
        self.device = torch.device(device)
        if position_mode not in ('final', 'a1'):
            raise ValueError("position_mode must be 'final' or 'a1'")
        self.position_mode = position_mode

        def metadata(stem):
            return json.loads((self.directory / (stem + '.json')).read_text(encoding='utf-8'))

        def state(stem):
            return {key: torch.from_numpy(value) for key, value in _arrays(self.directory / (stem + '.npz')).items()
                    if not key.startswith('normalization.')}

        a = metadata('background_a0')
        self.background_dims = a['dims']
        self.routing = {}
        members = [('baseline', 'background_routing'), ('confirmer', 'position_confirmer')]
        if self.position_mode == 'final':
            members.extend(('tv1_%d' % seed, 'tv1_routing_%d' % seed) for seed in (202762, 202763))
        for key, stem in members:
            anchor_stem = 'tv1_a0_' + key[4:] if key.startswith('tv1_') else 'background_a0'
            anchor_spec = metadata(anchor_stem)
            anchor = PositionTopologyAdapter(anchor_spec['dims'], anchor_spec['hidden_dim'],
                                             anchor_spec['architecture'], anchor_spec['edges'])
            anchor.load_state_dict(state(anchor_stem), strict=True)
            for parameter in anchor.parameters():
                parameter.requires_grad_(False)
            spec = metadata(stem)
            model = SafeInnovationRoutingAdapter(anchor, spec['hidden_dim'], spec['architecture'])
            load_adapter_state(model, state(stem))
            self.routing[key] = model.to(self.device).eval()
        r = metadata('rotation_residual')
        self.rotation = TaskSetEncoder(r['feature_dim'], r['state_dim'], r['hidden_dim'],
                                       r['mode'], r['bounds']['position'], r['bounds']['degree'])
        self.rotation.load_state_dict(state('rotation_residual'), strict=True)
        self.rotation.to(self.device).eval()
        values = _arrays(self.directory / 'rotation_residual.npz')
        self.rotation_normalization = {key: values['normalization.' + key] for key in ('center', 'scale')}
        self.raw_frontend = None

    def predict_from_prepared(self, background_features, rotation_features,
                              rotation_states, event_present, raw, workers=1):
        """Infer M attributes from background groups, per-turn features and raw windows.

        background_features requires BACKGROUND_KEYS. Spatial group shapes are
        N x 3 x d for station, N x 2 x d for acoustic, otherwise N x d.
        d is recorded in background_a0.json. Anchors are N x 3 / N x 2 logits.
        The caller must supply the original source-frozen feature/anchor contract.
        """
        import torch
        missing = set(BACKGROUND_KEYS) - set(background_features)
        if missing:
            raise ValueError('missing background fields: %s' % sorted(missing))
        n = len(raw)
        packet = {}
        for key in BACKGROUND_KEYS:
            value = np.asarray(background_features[key], dtype=np.float32)
            if key in ('position_anchor_logits', 'degree_anchor_logits'):
                shape = (n, 3 if key == 'position_anchor_logits' else 2)
            else:
                prefix = (n, 3) if key.startswith('station') else (n, 2) if key.startswith('acoustic') else (n,)
                shape = prefix + (self.background_dims[key],)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('%s must have finite shape %s' % (key, shape))
            packet[key] = torch.from_numpy(value).to(self.device)
        with torch.no_grad():
            baseline = self.routing['baseline'](packet)
            confirmer = self.routing['confirmer'](packet)
            tv1_probability = None
            if self.position_mode == 'final':
                tv1_logits = [baseline['position_logits']]
                tv1_logits.extend(self.routing['tv1_%d' % seed](packet)['position_logits'] for seed in (202762, 202763))
                tv1_probability = normalize_endpoint_probability(np.median(np.stack([
                    endpoint_softmax(value.cpu().numpy()) for value in tv1_logits
                ], axis=0), axis=0))
            rotation = prepare_rotation_packet(
                rotation_features, rotation_states, event_present,
                baseline['position_logits'].cpu().numpy(), baseline['degree_logits'].cpu().numpy(),
                self.rotation_normalization,
            )
            tensors = {key: torch.from_numpy(value).to(self.device) for key, value in rotation.items()}
            h0 = self.rotation(**tensors)
        synchronous = predict_synchronous_experts(raw, self.directory, workers)
        probability = endpoint_softmax if self.position_mode == 'final' else softmax_np
        return predict_misalignment(
            probability(h0['position_logits'].cpu().numpy()), probability(h0['degree_logits'].cpu().numpy()),
            probability(confirmer['position_logits'].cpu().numpy()), synchronous['D2']['position_prob'],
            synchronous['D2']['degree_prob'], synchronous['T2']['degree_prob'], synchronous['phase_quality'],
            tv1_position=tv1_probability, position_mode=self.position_mode,
        )

    def predict(self, raw, rpm, workers=1):
        """Infer per-window M position/degree from N x 10 x 6000 windows and RPM.

        Output is conditional on M presence. The upstream structure decision
        determines whether this expert's attributes are reported. The phase
        quality gate retains the original 1200-RPM validity/fallback contract.
        """
        from .m_features_raw import MRawFrontend
        if self.raw_frontend is None:
            self.raw_frontend = MRawFrontend(self.directory, self.device)
        background, rotation, states, events = self.raw_frontend.prepare(raw, rpm)
        return self.predict_from_prepared(background, rotation, states, events, raw, workers)

    predict_raw = predict
