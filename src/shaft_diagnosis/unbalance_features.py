"""Frozen U physical features and phase folds, with the original transient suppression."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
FRONTEND_VERSION = 'level2_shared_physics_frontend_v1'
PHASE_CHANNELS: Tuple[int, ...] = tuple(range(8))
PHASE_BINS = 64
EPS = 1e-08
SENSOR_CONTRACT: Tuple[Mapping[str, object], ...] = ({'id': 'ch01', 'idx': 0, 'name': 'bearing1_x', 'modality': 'vibration', 'location': 'B1', 'axis': 'x'}, {'id': 'ch02', 'idx': 1, 'name': 'bearing1_y', 'modality': 'vibration', 'location': 'B1', 'axis': 'y'}, {'id': 'ch03', 'idx': 2, 'name': 'bearing2_x', 'modality': 'vibration', 'location': 'B2', 'axis': 'x'}, {'id': 'ch04', 'idx': 3, 'name': 'bearing2_y', 'modality': 'vibration', 'location': 'B2', 'axis': 'y'}, {'id': 'ch05', 'idx': 4, 'name': 'bearing3_x', 'modality': 'vibration', 'location': 'B3', 'axis': 'x'}, {'id': 'ch06', 'idx': 5, 'name': 'bearing3_y', 'modality': 'vibration', 'location': 'B3', 'axis': 'y'}, {'id': 'ch07', 'idx': 6, 'name': 'shaft_orbit_x', 'modality': 'shaft_orbit', 'location': 'B1-B2', 'axis': 'x'}, {'id': 'ch08', 'idx': 7, 'name': 'shaft_orbit_y', 'modality': 'shaft_orbit', 'location': 'B1-B2', 'axis': 'y'}, {'id': 'ch09', 'idx': 8, 'name': 'sound_A_bearing1_side', 'modality': 'acoustic', 'location': 'B1', 'axis': 'sound'}, {'id': 'ch10', 'idx': 9, 'name': 'sound_B_bearing3_side', 'modality': 'acoustic', 'location': 'B3', 'axis': 'sound'})
U_PHYSICS_NAMES: Tuple[str, ...] = ('U_bg_B1_order1_log_amp', 'U_bg_B2_order1_log_amp', 'U_bg_B3_order1_log_amp', 'U_bg_B1_order1_share', 'U_bg_B2_order1_share', 'U_bg_B3_order1_share', 'U_bg_B1_order2_to_order1', 'U_bg_B2_order2_to_order1', 'U_bg_B3_order2_to_order1', 'U_bg_B1_order3_to_order1', 'U_bg_B2_order3_to_order1', 'U_bg_B3_order3_to_order1', 'U_bg_order1_endpoint_signed', 'U_bg_order1_midness_log', 'U_bg_order1_spatial_barycenter', 'U_orbit_order1_log_amp', 'U_orbit_order2_to_order1', 'U_orbit_order3_to_order1', 'U_orbit_xy_order1_phase_sin', 'U_orbit_xy_order1_phase_cos')
U_ANCHOR_NAME = 'U_bg_B3_order1_log_amp'
U_ANCHOR_INDEX = U_PHYSICS_NAMES.index(U_ANCHOR_NAME)
CONFIG = {"event_count": 3, "event_min_distance": 64, "event_mask_radius": 24}

def robust_z_time(signal: torch.Tensor) -> torch.Tensor:
    med = torch.median(signal, dim=1, keepdim=True).values
    mad = torch.median(torch.abs(signal - med), dim=1, keepdim=True).values.clamp_min(1e-06)
    return (signal - med) / (1.4826 * mad)


def event_centers(score: torch.Tensor, count: int, min_distance: int) -> torch.Tensor:
    width = int(min_distance) * 2 + 1
    pooled = torch.nn.functional.max_pool1d(score.unsqueeze(1), kernel_size=width, stride=1, padding=int(min_distance)).squeeze(1)
    candidate = torch.where(score >= pooled, score, torch.full_like(score, -1e+20))
    return torch.topk(candidate, k=int(count), dim=1).indices


def transient_suppressed_waveform(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    centered = x - torch.median(x, dim=2, keepdim=True).values
    vib_point = torch.sqrt(torch.mean(centered[:, 0:6, :] ** 2, dim=1).clamp_min(1e-08))
    ac_point = torch.sqrt(torch.mean(centered[:, 8:10, :] ** 2, dim=1).clamp_min(1e-08))
    score = 0.5 * robust_z_time(vib_point) + 0.5 * robust_z_time(ac_point)
    margin = max(int(CONFIG['event_mask_radius']) + 1, int(CONFIG['event_min_distance']))
    score = score.clone()
    score[:, :margin] = -1e+20
    score[:, -margin:] = -1e+20
    centers = event_centers(score, int(CONFIG['event_count']), int(CONFIG['event_min_distance']))
    time_ids = torch.arange(x.shape[2], device=x.device).view(1, 1, -1)
    distance = torch.abs(time_ids - centers.unsqueeze(2))
    mask = torch.any(distance <= int(CONFIG['event_mask_radius']), dim=1)
    fill = torch.median(centered, dim=2, keepdim=True).values
    clean = torch.where(mask.unsqueeze(1), fill, centered)
    return (clean, mask.float())

def validate_sensor_contract(channels: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Return an auditable contract comparison and fail on any semantic drift."""
    expected = [dict(row) for row in SENSOR_CONTRACT]
    actual = [dict(row) for row in channels]
    fields = ('id', 'idx', 'name', 'modality', 'location', 'axis')
    mismatches: List[Dict[str, object]] = []
    if len(actual) != len(expected):
        mismatches.append({'field': 'length', 'expected': len(expected), 'actual': len(actual)})
    for i in range(min(len(actual), len(expected))):
        for field in fields:
            if actual[i].get(field) != expected[i].get(field):
                mismatches.append({'row': i, 'field': field, 'expected': expected[i].get(field), 'actual': actual[i].get(field)})
    if mismatches:
        raise RuntimeError('Level2 sensor contract drift: %s' % mismatches[:8])
    return {'frontend_version': FRONTEND_VERSION, 'channels': len(expected), 'fields': list(fields), 'mismatches': []}


def _gather_fft_bin(spec: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    index = bins.long().clamp(0, spec.shape[2] - 1).view(-1, 1, 1)
    index = index.expand(-1, spec.shape[1], 1)
    return torch.gather(spec, 2, index).squeeze(2)


def periodic_reconstruction(x: torch.Tensor, rpm: torch.Tensor, fs_hz: float, max_order: int=3, radius: int=1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct shaft-order components with the same convention as P2."""
    if x.ndim != 3 or x.shape[1] != 10:
        raise ValueError('Expected x shaped [B,10,T], got %s' % (tuple(x.shape),))
    centered = x.float() - torch.mean(x.float(), dim=2, keepdim=True)
    spec = torch.fft.rfft(centered, dim=2)
    periodic_spec = torch.zeros_like(spec)
    base = torch.round(rpm.float() / 60.0 * float(x.shape[2]) / float(fs_hz)).long()
    base = base.clamp(1, spec.shape[2] - 1)
    order_values: List[torch.Tensor] = []
    for order in range(1, int(max_order) + 1):
        center = (base * int(order)).clamp(1, spec.shape[2] - 1)
        order_values.append(torch.abs(_gather_fft_bin(spec, center)) / float(x.shape[2]))
        for offset in range(-int(radius), int(radius) + 1):
            bins = (center + int(offset)).clamp(1, spec.shape[2] - 1)
            index = bins.view(-1, 1, 1).expand(-1, spec.shape[1], 1)
            periodic_spec.scatter_(2, index, torch.gather(spec, 2, index))
    periodic = torch.fft.irfft(periodic_spec, n=x.shape[2], dim=2)
    amplitudes = torch.stack(order_values, dim=2)
    return (periodic, centered - periodic, amplitudes)


def rpm_synchronous_fold(background: torch.Tensor, rpm: torch.Tensor, fs_hz: float, bins: int=PHASE_BINS, channel_indices: Sequence[int]=PHASE_CHANNELS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average samples with the same shaft phase and normalize only the shape view."""
    if background.ndim != 3 or background.shape[1] != 10:
        raise ValueError('Expected background [B,10,T]')
    b, _c, t = background.shape
    ids = torch.as_tensor(list(channel_indices), dtype=torch.long, device=background.device)
    selected = background.index_select(1, ids).float()
    time = torch.arange(t, dtype=torch.float32, device=background.device).view(1, t)
    cycles = time * (rpm.float().view(b, 1) / 60.0) / float(fs_hz)
    phase_index = torch.floor(torch.remainder(cycles, 1.0) * float(bins)).long()
    phase_index = phase_index.clamp(0, int(bins) - 1)
    expanded = phase_index.view(b, 1, t).expand(b, selected.shape[1], t)
    folded = torch.zeros((b, selected.shape[1], int(bins)), dtype=selected.dtype, device=selected.device)
    folded.scatter_add_(2, expanded, selected)
    counts = torch.zeros((b, 1, int(bins)), dtype=selected.dtype, device=selected.device)
    counts.scatter_add_(2, phase_index.view(b, 1, t), torch.ones((b, 1, t), device=selected.device))
    folded = folded / counts.clamp_min(1.0)
    folded = folded - torch.mean(folded, dim=2, keepdim=True)
    amplitude = torch.sqrt(torch.mean(folded ** 2, dim=2).clamp_min(EPS))
    shape = folded / amplitude.unsqueeze(2).clamp_min(1e-06)
    return (torch.nan_to_num(shape), torch.nan_to_num(amplitude))


def u_physics_features_from_background(background: torch.Tensor, rpm: torch.Tensor, fs_hz: float, periodic_radius: int=1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Extract the frozen U-degree physics vector from transient-suppressed data."""
    _periodic, _aperiodic, amplitudes = periodic_reconstruction(background, rpm, fs_hz, max_order=3, radius=periodic_radius)
    bearing = []
    for c0, c1 in ((0, 1), (2, 3), (4, 5)):
        bearing.append(torch.sqrt(amplitudes[:, c0, :] ** 2 + amplitudes[:, c1, :] ** 2).clamp_min(EPS))
    bearing_orders = torch.stack(bearing, dim=1)
    order1 = bearing_orders[:, :, 0]
    total1 = torch.sum(order1, dim=1).clamp_min(EPS)
    share1 = order1 / total1.unsqueeze(1)
    order2_ratio = bearing_orders[:, :, 1] / order1.clamp_min(EPS)
    order3_ratio = bearing_orders[:, :, 2] / order1.clamp_min(EPS)
    endpoint = (order1[:, 2] - order1[:, 0]) / total1
    midness = torch.log((order1[:, 1] + EPS) / torch.sqrt(order1[:, 0] * order1[:, 2] + EPS))
    barycenter = (share1[:, 1] + 2.0 * share1[:, 2]) / 2.0
    orbit_orders = torch.sqrt(amplitudes[:, 6, :] ** 2 + amplitudes[:, 7, :] ** 2).clamp_min(EPS)
    centered = background - torch.mean(background, dim=2, keepdim=True)
    spec = torch.fft.rfft(centered, dim=2)
    base = torch.round(rpm.float() / 60.0 * float(background.shape[2]) / float(fs_hz)).long()
    base = base.clamp(1, spec.shape[2] - 1)
    orbit_complex = _gather_fft_bin(spec, base)
    orbit_phase = torch.angle(orbit_complex[:, 6] * torch.conj(orbit_complex[:, 7]))
    values = [torch.log(order1[:, 0] + EPS), torch.log(order1[:, 1] + EPS), torch.log(order1[:, 2] + EPS), share1[:, 0], share1[:, 1], share1[:, 2], order2_ratio[:, 0], order2_ratio[:, 1], order2_ratio[:, 2], order3_ratio[:, 0], order3_ratio[:, 1], order3_ratio[:, 2], endpoint, midness, barycenter, torch.log(orbit_orders[:, 0] + EPS), orbit_orders[:, 1] / orbit_orders[:, 0].clamp_min(EPS), orbit_orders[:, 2] / orbit_orders[:, 0].clamp_min(EPS), torch.sin(orbit_phase), torch.cos(orbit_phase)]
    physics = torch.stack(values, dim=1)
    physics = torch.nan_to_num(physics, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    details = {'amplitudes': amplitudes, 'bearing_orders': bearing_orders, 'orbit_orders': orbit_orders, 'order1_share': share1}
    if physics.shape[1] != len(U_PHYSICS_NAMES):
        raise RuntimeError('U physics schema mismatch')
    return (physics, details)


def extract_u_frontend(x10: torch.Tensor, rpm: torch.Tensor, fs_hz: float, transient_suppressor, phase_bins: int=PHASE_BINS, periodic_radius: int=1) -> Dict[str, torch.Tensor]:
    """Run the complete U frontend; transient_suppressor is the frozen P1 function."""
    raw = x10.float() - torch.median(x10.float(), dim=2, keepdim=True).values
    background, event_mask = transient_suppressor(raw)
    physics, details = u_physics_features_from_background(background, rpm, fs_hz, periodic_radius=periodic_radius)
    phase, phase_amplitude = rpm_synchronous_fold(background, rpm, fs_hz, bins=int(phase_bins), channel_indices=PHASE_CHANNELS)
    return {'physics': physics, 'phase': phase, 'phase_amplitude': phase_amplitude, 'background': background, 'event_mask': event_mask, 'bearing_orders': details['bearing_orders']}
