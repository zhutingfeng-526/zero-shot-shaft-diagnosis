"""Window-level order, spatial and transient response features for M."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



FS_HZ_DEFAULT = 5000.0


CHANNELS: List[Dict[str, Any]] = [
    {"id": "ch01", "idx": 0, "name": "bearing1_x", "modality": "vibration", "location": "B1", "axis": "x"},
    {"id": "ch02", "idx": 1, "name": "bearing1_y", "modality": "vibration", "location": "B1", "axis": "y"},
    {"id": "ch03", "idx": 2, "name": "bearing2_x", "modality": "vibration", "location": "B2", "axis": "x"},
    {"id": "ch04", "idx": 3, "name": "bearing2_y", "modality": "vibration", "location": "B2", "axis": "y"},
    {"id": "ch05", "idx": 4, "name": "bearing3_x", "modality": "vibration", "location": "B3", "axis": "x"},
    {"id": "ch06", "idx": 5, "name": "bearing3_y", "modality": "vibration", "location": "B3", "axis": "y"},
    {"id": "ch07", "idx": 6, "name": "shaft_orbit_x", "modality": "shaft_orbit", "location": "B1-B2", "axis": "x"},
    {"id": "ch08", "idx": 7, "name": "shaft_orbit_y", "modality": "shaft_orbit", "location": "B1-B2", "axis": "y"},
    {"id": "ch09", "idx": 8, "name": "sound_A_bearing1_side", "modality": "acoustic", "location": "B1", "axis": "sound"},
    {"id": "ch10", "idx": 9, "name": "sound_B_bearing3_side", "modality": "acoustic", "location": "B3", "axis": "sound"},
]


CONFIG: Dict[str, Any] = {
    "gpu_id": 1,
    "expected_gpu_token": "A4000",
    "seed": 202707,
    "fs_hz": FS_HZ_DEFAULT,
    "physical_batch_size": 192,
    "model_batch_size": 256,
    "loader_workers": 0,
    "pin_memory": True,
    "source_fit_per_condition": 384,
    "source_audit_per_condition": 192,
    "real_pair_per_condition": 128,
    "triple_per_condition": 64,
    "pseudo_per_pair_alpha": 96,
    "pseudo_alphas": [0.35, 0.50, 0.65],
    "record_block_windows": 256,
    "ridge_lambda": 1.0e-2,
    "cv_folds": 3,
    "event_count": 3,
    "event_min_distance": 64,
    "event_core_radius": 8,
    "event_bg_pre": 80,
    "event_bg_gap": 16,
    "event_lag_radius": 32,
}


def _gather_frequency(spec: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Gather one complex FFT bin per sample and channel."""
    b, c, _ = spec.shape
    index = bins.long().clamp(0, spec.shape[-1] - 1).view(b, 1, 1).expand(b, c, 1)
    return torch.gather(spec, 2, index).squeeze(2)


def _band_energy(power: torch.Tensor, freqs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    mask = (freqs >= float(lo)) & (freqs < float(hi))
    if not bool(mask.any()):
        return torch.zeros(power.shape[:2], dtype=power.dtype, device=power.device)
    total = torch.sum(power[:, :, 1:], dim=2).clamp_min(1.0e-8)
    return torch.sum(power[:, :, mask], dim=2) / total


def _robust_z_time(signal: torch.Tensor) -> torch.Tensor:
    med = torch.median(signal, dim=1, keepdim=True).values
    mad = torch.median(torch.abs(signal - med), dim=1, keepdim=True).values.clamp_min(1.0e-6)
    return (signal - med) / (1.4826 * mad)


def _event_centers(score: torch.Tensor, count: int, min_distance: int) -> torch.Tensor:
    width = int(min_distance) * 2 + 1
    pooled = F.max_pool1d(score.unsqueeze(1), kernel_size=width, stride=1, padding=int(min_distance)).squeeze(1)
    candidate = torch.where(score >= pooled, score, torch.full_like(score, -1.0e20))
    return torch.topk(candidate, k=int(count), dim=1).indices


def _window_gather(x: torch.Tensor, centers: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Return [B,E,C,W] samples around centers."""
    b, c, t = x.shape
    e = centers.shape[1]
    ids = (centers.unsqueeze(2) + offsets.view(1, 1, -1)).clamp(0, t - 1)
    expanded = x.unsqueeze(1).expand(b, e, c, t)
    gather_ids = ids.unsqueeze(2).expand(b, e, c, ids.shape[2])
    return torch.gather(expanded, 3, gather_ids)


def extract_event_features(x: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
    eps = 1.0e-8
    b, _c, t = x.shape
    vib_point = torch.sqrt(torch.mean(x[:, 0:6, :] ** 2, dim=1).clamp_min(eps))
    ac_point = torch.sqrt(torch.mean(x[:, 8:10, :] ** 2, dim=1).clamp_min(eps))
    vib_z = _robust_z_time(vib_point)
    ac_z = _robust_z_time(ac_point)
    mixed = 0.5 * vib_z + 0.5 * ac_z
    event_margin = max(int(CONFIG["event_bg_pre"]), int(CONFIG["event_core_radius"]) + 1)
    center_score = mixed.clone()
    center_score[:, :event_margin] = -1.0e20
    center_score[:, -event_margin:] = -1.0e20
    centers = _event_centers(center_score, int(CONFIG["event_count"]), int(CONFIG["event_min_distance"]))

    core_r = int(CONFIG["event_core_radius"])
    core_offsets = torch.arange(-core_r, core_r + 1, device=x.device)
    bg_offsets = torch.arange(-int(CONFIG["event_bg_pre"]), -int(CONFIG["event_bg_gap"]), device=x.device)
    core = _window_gather(x, centers, core_offsets)
    bg = _window_gather(x, centers, bg_offsets)
    core_energy = torch.mean(core ** 2, dim=3).clamp_min(eps)
    bg_energy = torch.mean(bg ** 2, dim=3).clamp_min(eps)
    core_peak = torch.amax(torch.abs(core), dim=3).clamp_min(eps)

    b1 = torch.sum(core_energy[:, :, 0:2], dim=2)
    b2 = torch.sum(core_energy[:, :, 2:4], dim=2)
    b3 = torch.sum(core_energy[:, :, 4:6], dim=2)
    vib_total = (b1 + b2 + b3).clamp_min(eps)
    shares = torch.stack([b1 / vib_total, b2 / vib_total, b3 / vib_total], dim=2)
    ac_a = core_energy[:, :, 8]
    ac_b = core_energy[:, :, 9]
    ac_total = (ac_a + ac_b).clamp_min(eps)

    endpoint_signed = (b3 - b1) / vib_total
    endpoint_strength = torch.abs(endpoint_signed)
    bary = (-b1 + b3) / vib_total
    center_margin = b2 / vib_total - torch.maximum(b1, b3) / vib_total
    midness = torch.log((b2 + eps) / torch.sqrt(b1 * b3 + eps))
    acoustic_signed = (ac_b - ac_a) / ac_total
    vib_gain = torch.log((vib_total + eps) / (torch.sum(bg_energy[:, :, 0:6], dim=2) + eps))
    ac_gain = torch.log((ac_total + eps) / (torch.sum(bg_energy[:, :, 8:10], dim=2) + eps))
    peak_concentration = torch.sum(core_peak[:, :, 0:6], dim=2) / torch.sqrt(vib_total).clamp_min(eps)

    half = core.shape[-1] // 2
    early = torch.mean(torch.abs(core[:, :, :, :half]), dim=(2, 3))
    late = torch.mean(torch.abs(core[:, :, :, half + 1:]), dim=(2, 3))
    rise_decay = torch.log((early + eps) / (late + eps))

    vib_center = torch.argmax(vib_z, dim=1)
    ac_center = torch.argmax(ac_z, dim=1)
    shared_center = centers[:, 0]
    center_lag = (ac_center - vib_center).float() / float(max(1, t))
    shared_vib_lag = (shared_center - vib_center).float() / float(max(1, t))
    shared_ac_lag = (shared_center - ac_center).float() / float(max(1, t))
    independent_gain = (
        torch.gather(vib_point, 1, vib_center[:, None]).squeeze(1)
        + torch.gather(ac_point, 1, ac_center[:, None]).squeeze(1)
    ) / (
        torch.gather(vib_point + ac_point, 1, shared_center[:, None]).squeeze(1).clamp_min(eps)
    )

    vote = torch.argmax(shares, dim=2)
    vote_frac = torch.stack([(vote == k).float().mean(dim=1) for k in range(3)], dim=1)
    vote_entropy = -torch.sum(vote_frac * torch.log(vote_frac.clamp_min(eps)), dim=1) / math.log(3.0)
    endpoint_sign_consistency = torch.abs(torch.mean(torch.sign(endpoint_signed), dim=1))
    event_score = torch.gather(mixed, 1, centers)
    event_coverage = torch.median(event_score, dim=1).values / torch.amax(event_score, dim=1).clamp_min(eps)

    per_event = {
        "b1_share": shares[:, :, 0], "b2_share": shares[:, :, 1], "b3_share": shares[:, :, 2],
        "endpoint_signed": endpoint_signed, "endpoint_strength": endpoint_strength, "bary": bary,
        "center_margin": center_margin, "midness": midness, "acoustic_signed": acoustic_signed,
        "vib_core_background_log_gain": vib_gain, "acoustic_core_background_log_gain": ac_gain,
        "peak_concentration": peak_concentration, "rise_decay_log_ratio": rise_decay,
    }
    values: List[torch.Tensor] = []
    names: List[str] = []
    for name, val in per_event.items():
        values.extend([torch.median(val, dim=1).values, torch.std(val, dim=1, unbiased=False)])
        names.extend(["event_%s_median" % name, "event_%s_std" % name])
    values.extend([
        vote_frac[:, 0], vote_frac[:, 1], vote_frac[:, 2], vote_entropy,
        endpoint_sign_consistency, event_coverage, center_lag, shared_vib_lag,
        shared_ac_lag, independent_gain,
    ])
    names.extend([
        "event_vote_B1_frac", "event_vote_B2_frac", "event_vote_B3_frac", "event_vote_entropy",
        "event_endpoint_sign_consistency", "event_peak_coverage", "event_acoustic_vibration_center_lag_norm",
        "event_shared_vibration_center_lag_norm", "event_shared_acoustic_center_lag_norm",
        "event_independent_vs_shared_peak_gain",
    ])
    return torch.stack(values, dim=1), names


def extract_physical_features(x10: torch.Tensor, rpm: torch.Tensor, fs_hz: float) -> Tuple[torch.Tensor, List[str]]:
    eps = 1.0e-8
    x = x10.float()
    centered = x - torch.mean(x, dim=2, keepdim=True)
    mean = torch.mean(x, dim=2)
    std = torch.std(x, dim=2, unbiased=False).clamp_min(eps)
    rms = torch.sqrt(torch.mean(x ** 2, dim=2).clamp_min(eps))
    abs_mean = torch.mean(torch.abs(x), dim=2).clamp_min(eps)
    p2p = torch.amax(x, dim=2) - torch.amin(x, dim=2)
    var = torch.mean(centered ** 2, dim=2).clamp_min(eps)
    skew = torch.mean(centered ** 3, dim=2) / torch.pow(var, 1.5)
    kurt = torch.mean(centered ** 4, dim=2) / (var ** 2)
    crest = torch.amax(torch.abs(x), dim=2) / rms
    impulse = torch.amax(torch.abs(x), dim=2) / abs_mean
    shape = rms / abs_mean
    diff_rms = torch.sqrt(torch.mean(torch.diff(x, dim=2) ** 2, dim=2).clamp_min(eps))

    spec = torch.fft.rfft(centered, dim=2)
    power = torch.abs(spec) ** 2
    freqs = torch.fft.rfftfreq(x.shape[2], d=1.0 / float(fs_hz)).to(x.device)
    bins_1x = torch.round((rpm.float() / 60.0) * float(x.shape[2]) / float(fs_hz)).long().clamp(1, spec.shape[2] - 1)
    harmonics = []
    for order in (1, 2, 3):
        bins = (bins_1x * int(order)).clamp(1, spec.shape[2] - 1)
        harmonics.append(torch.abs(_gather_frequency(spec, bins)) / float(x.shape[2]))
    amp1, amp2, amp3 = harmonics
    order_ratio21 = amp2 / amp1.clamp_min(eps)
    order_ratio31 = amp3 / amp1.clamp_min(eps)
    spec_centroid = torch.sum(power * freqs.view(1, 1, -1), dim=2) / torch.sum(power, dim=2).clamp_min(eps)
    bands = [
        _band_energy(power, freqs, 0.0, 50.0), _band_energy(power, freqs, 50.0, 150.0),
        _band_energy(power, freqs, 150.0, 500.0), _band_energy(power, freqs, 500.0, 1500.0),
        _band_energy(power, freqs, 1500.0, float(fs_hz) / 2.0 + 1.0),
    ]
    per_ch = torch.stack([
        mean, std, rms, abs_mean, p2p, skew, kurt, crest, impulse, shape, diff_rms,
        amp1, amp2, amp3, order_ratio21, order_ratio31, spec_centroid,
    ] + bands, dim=2)
    stat_names = [
        "mean", "std", "rms", "abs_mean", "p2p", "skewness", "kurtosis", "crest", "impulse",
        "shape", "diff_rms", "order1_amp", "order2_amp", "order3_amp", "order2_order1_ratio",
        "order3_order1_ratio", "spectral_centroid_hz", "band_0_50", "band_50_150", "band_150_500",
        "band_500_1500", "band_1500_nyquist",
    ]
    names = ["%s__%s__%s" % (ch["id"], ch["name"], stat) for ch in CHANNELS for stat in stat_names]
    values = [per_ch.reshape(x.shape[0], -1)]

    layout_values: List[torch.Tensor] = []
    layout_names: List[str] = []
    bearing_energy = []
    for bi, (c0, c1) in enumerate(((0, 1), (2, 3), (4, 5)), start=1):
        energy = rms[:, c0] ** 2 + rms[:, c1] ** 2
        bearing_energy.append(energy)
        layout_values.extend([torch.sqrt(energy), torch.log((rms[:, c0] + eps) / (rms[:, c1] + eps))])
        layout_names.extend(["bearing%d_xy_rms" % bi, "bearing%d_x_y_log_rms_ratio" % bi])
    be = torch.stack(bearing_energy, dim=1)
    total = torch.sum(be, dim=1).clamp_min(eps)
    shares = be / total[:, None]
    layout_values.extend([shares[:, 0], shares[:, 1], shares[:, 2]])
    layout_names.extend(["spatial_B1_energy_share", "spatial_B2_energy_share", "spatial_B3_energy_share"])
    layout_values.extend([
        (be[:, 2] - be[:, 0]) / total,
        (-be[:, 0] + be[:, 2]) / total,
        shares[:, 1] - torch.maximum(shares[:, 0], shares[:, 2]),
        torch.log((be[:, 1] + eps) / torch.sqrt(be[:, 0] * be[:, 2] + eps)),
    ])
    layout_names.extend(["spatial_endpoint_signed", "spatial_barycenter", "spatial_center_margin", "spatial_midness"])

    ox = centered[:, 6, :]
    oy = centered[:, 7, :]
    cov_xx = torch.mean(ox * ox, dim=1)
    cov_yy = torch.mean(oy * oy, dim=1)
    cov_xy = torch.mean(ox * oy, dim=1)
    trace = cov_xx + cov_yy
    disc = torch.sqrt(((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy ** 2).clamp_min(eps))
    eig1 = (0.5 * (trace + disc)).clamp_min(eps)
    eig2 = (0.5 * (trace - disc)).clamp_min(eps)
    major = torch.sqrt(eig1)
    minor = torch.sqrt(eig2)
    orbit_area = math.pi * major * minor
    orbit_ecc = torch.sqrt((1.0 - eig2 / eig1).clamp(0.0, 1.0))
    orbit_angle = 0.5 * torch.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    layout_values.extend([major, minor, orbit_area, orbit_ecc, torch.sin(orbit_angle), torch.cos(orbit_angle)])
    layout_names.extend([
        "orbit_major_axis", "orbit_minor_axis", "orbit_ellipse_area", "orbit_eccentricity",
        "orbit_orientation_sin", "orbit_orientation_cos",
    ])

    ac_energy = rms[:, 8] ** 2 + rms[:, 9] ** 2
    layout_values.extend([
        torch.sqrt(ac_energy), (rms[:, 9] ** 2 - rms[:, 8] ** 2) / ac_energy.clamp_min(eps),
        torch.log((ac_energy + eps) / (total + eps)),
        torch.log((eig1 + eig2 + eps) / (total + eps)),
    ])
    layout_names.extend([
        "acoustic_AB_rms", "acoustic_endpoint_signed", "cross_acoustic_vibration_log_gain",
        "cross_orbit_vibration_log_gain",
    ])

    phase_pairs = [(0, 1, "B1_xy"), (2, 3, "B2_xy"), (4, 5, "B3_xy"), (6, 7, "orbit_xy"), (8, 9, "acoustic_AB")]
    s1 = _gather_frequency(spec, bins_1x)
    s2 = _gather_frequency(spec, (bins_1x * 2).clamp(1, spec.shape[2] - 1))
    for c0, c1, label in phase_pairs:
        for order, sval in ((1, s1), (2, s2)):
            delta = torch.angle(sval[:, c0] * torch.conj(sval[:, c1]))
            layout_values.extend([torch.sin(delta), torch.cos(delta)])
            layout_names.extend(["phase_%s_%dx_sin" % (label, order), "phase_%s_%dx_cos" % (label, order)])

    values.append(torch.stack(layout_values, dim=1))
    names.extend(layout_names)
    event_values, event_names = extract_event_features(x)
    values.append(event_values)
    names.extend(event_names)
    feat = torch.cat(values, dim=1)
    if feat.shape[1] != len(names):
        raise RuntimeError("Physical feature name/shape mismatch: %s vs %d" % (tuple(feat.shape), len(names)))
    feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return feat, names


def apply_model_input_norm(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "none":
        return x
    if mode == "demean":
        return x - x.mean(dim=2, keepdim=True)
    if mode == "zscore":
        return (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True, unbiased=False).clamp_min(1.0e-8)
    raise ValueError("Unsupported input_norm=%s" % mode)
