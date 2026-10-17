"""Periodic background, event morphology and cross-channel response features."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hashlib
from . import m_features_p0 as p0
from . import m_features_p1 as p1



CONFIG: Dict[str, Any] = {
    "gpu_id": 1,
    "expected_gpu_token": "A4000",
    "seed": 202707,
    "fs_hz": 5000.0,
    "batch_size": 96,
    "ridge_lambda": 1.0e-2,
    "cv_folds": 4,
    "event_count": 3,
    "event_min_distance": 64,
    "event_mask_radius": 24,
    "periodic_bin_radius": 1,
    "max_order": 8,
    "event_lag_radius": 24,
    "raw_sketch_dim": 192,
    "pseudo_per_pair_alpha": 96,
    "pseudo_recipe_block_size": 8,
    "pseudo_alphas": [0.35, 0.50, 0.65],
}


GROUPS = (
    ("B1_vibration", (0, 1), "vibration", "B1"),
    ("B2_vibration", (2, 3), "vibration", "B2"),
    ("B3_vibration", (4, 5), "vibration", "B3"),
    ("orbit_B1_B2", (6, 7), "shaft_orbit", "B1-B2"),
    ("sound_A_B1", (8,), "acoustic", "B1"),
    ("sound_B_B3", (9,), "acoustic", "B3"),
)


_PROJECTION_CACHE: Dict[Tuple[int, int, int, str], torch.Tensor] = {}


def gather_fft_bin(spec: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    index = bins.long().clamp(0, spec.shape[2] - 1).view(-1, 1, 1).expand(-1, spec.shape[1], 1)
    return torch.gather(spec, 2, index).squeeze(2)


def periodic_reconstruction(
    x: torch.Tensor, rpm: torch.Tensor, fs_hz: float, max_order: int, radius: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered = x - torch.mean(x, dim=2, keepdim=True)
    spec = torch.fft.rfft(centered, dim=2)
    periodic_spec = torch.zeros_like(spec)
    base = torch.round((rpm.float() / 60.0) * float(x.shape[2]) / float(fs_hz)).long()
    base = base.clamp(1, spec.shape[2] - 1)
    order_values: List[torch.Tensor] = []
    for order in range(1, int(max_order) + 1):
        center = (base * int(order)).clamp(1, spec.shape[2] - 1)
        order_values.append(torch.abs(gather_fft_bin(spec, center)) / float(x.shape[2]))
        for offset in range(-int(radius), int(radius) + 1):
            bins = (center + int(offset)).clamp(1, spec.shape[2] - 1)
            index = bins.view(-1, 1, 1).expand(-1, spec.shape[1], 1)
            periodic_spec.scatter_(2, index, torch.gather(spec, 2, index))
    periodic = torch.fft.irfft(periodic_spec, n=x.shape[2], dim=2)
    amplitudes = torch.stack(order_values, dim=2)
    return periodic, centered - periodic, amplitudes


def group_power(x: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    ids = list(int(v) for v in indices)
    return torch.sum(x[:, :, ids, :] ** 2, dim=2)


def group_envelopes(x: torch.Tensor) -> torch.Tensor:
    values = []
    for _name, indices, _modality, _location in GROUPS:
        ids = list(indices)
        values.append(torch.sqrt(torch.mean(x[:, ids, :] ** 2, dim=1).clamp_min(1.0e-8)))
    return torch.stack(values, dim=1)


def robust_z(signal: torch.Tensor) -> torch.Tensor:
    med = torch.median(signal, dim=1, keepdim=True).values
    mad = torch.median(torch.abs(signal - med), dim=1, keepdim=True).values.clamp_min(1.0e-6)
    return (signal - med) / (1.4826 * mad)


def event_centers_from_views(residual: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    envelopes = group_envelopes(residual)
    vib = torch.sqrt(torch.mean(residual[:, 0:6, :] ** 2, dim=1).clamp_min(1.0e-8))
    acoustic = torch.sqrt(torch.mean(residual[:, 8:10, :] ** 2, dim=1).clamp_min(1.0e-8))
    vib_z = robust_z(vib)
    acoustic_z = robust_z(acoustic)
    mixed = 0.5 * vib_z + 0.5 * acoustic_z
    margin = 144
    score = mixed.clone()
    score[:, :margin] = -1.0e20
    score[:, -margin:] = -1.0e20
    centers = p0._event_centers(score, int(CONFIG["event_count"]), int(CONFIG["event_min_distance"]))
    return centers, {
        "envelopes": envelopes, "vibration_score": vib_z,
        "acoustic_score": acoustic_z, "mixed_score": mixed,
    }


def add_feature(
    values: List[torch.Tensor], names: List[str], value: torch.Tensor, name: str
) -> None:
    if value.ndim != 1:
        raise RuntimeError("Feature %s must be [B], got %s" % (name, tuple(value.shape)))
    values.append(value)
    names.append(name)


def add_event_summary(
    values: List[torch.Tensor], names: List[str], value: torch.Tensor, name: str
) -> None:
    if value.ndim != 2:
        raise RuntimeError("Event feature %s must be [B,E], got %s" % (name, tuple(value.shape)))
    add_feature(values, names, torch.median(value, dim=1).values, name + "__median")
    add_feature(values, names, torch.std(value, dim=1, unbiased=False), name + "__event_std")


def background_features(
    clean: torch.Tensor, rpm: torch.Tensor, fs_hz: float
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[str]]:
    eps = 1.0e-8
    periodic, aperiodic, amplitudes = periodic_reconstruction(
        clean, rpm, fs_hz, int(CONFIG["max_order"]), int(CONFIG["periodic_bin_radius"])
    )
    rms = torch.sqrt(torch.mean(clean ** 2, dim=2).clamp_min(eps))
    values: List[torch.Tensor] = []
    names: List[str] = []
    max_order = amplitudes.shape[2]

    for channel in p0.CHANNELS:
        ci = int(channel["idx"])
        for order in range(max_order):
            amp = amplitudes[:, ci, order]
            add_feature(values, names, amp, "bg_order__%s__order%d_amp" % (channel["id"], order + 1))
            add_feature(
                values, names, amp / rms[:, ci].clamp_min(eps),
                "bg_order__%s__order%d_to_rms" % (channel["id"], order + 1),
            )

    bearing_orders = []
    for c0, c1 in ((0, 1), (2, 3), (4, 5)):
        bearing_orders.append(torch.sqrt(amplitudes[:, c0, :] ** 2 + amplitudes[:, c1, :] ** 2).clamp_min(eps))
    bearing = torch.stack(bearing_orders, dim=1)
    for order in range(max_order):
        current = bearing[:, :, order]
        total = torch.sum(current, dim=1).clamp_min(eps)
        share = current / total[:, None]
        for bi in range(3):
            add_feature(
                values, names, share[:, bi],
                "bg_spatial__B%d__order%d_share" % (bi + 1, order + 1),
            )
        add_feature(values, names, (current[:, 2] - current[:, 0]) / total,
                    "bg_spatial__order%d_endpoint_signed" % (order + 1))
        add_feature(values, names, torch.log((current[:, 1] + eps) / torch.sqrt(current[:, 0] * current[:, 2] + eps)),
                    "bg_spatial__order%d_midness" % (order + 1))

    spec = torch.fft.rfft(clean - torch.mean(clean, dim=2, keepdim=True), dim=2)
    base = torch.round((rpm.float() / 60.0) * float(clean.shape[2]) / float(fs_hz)).long()
    base = base.clamp(1, spec.shape[2] - 1)
    phase_pairs = (
        (0, 1, "B1_xy", "vibration", "B1"),
        (2, 3, "B2_xy", "vibration", "B2"),
        (4, 5, "B3_xy", "vibration", "B3"),
        (6, 7, "orbit_xy", "shaft_orbit", "B1-B2"),
        (8, 9, "sound_AB", "acoustic", "B1-B3"),
    )
    for order in (1, 2, 3):
        sval = gather_fft_bin(spec, (base * order).clamp(1, spec.shape[2] - 1))
        for c0, c1, label, _modality, _location in phase_pairs:
            delta = torch.angle(sval[:, c0] * torch.conj(sval[:, c1]))
            add_feature(values, names, torch.sin(delta), "bg_phase__%s__order%d_sin" % (label, order))
            add_feature(values, names, torch.cos(delta), "bg_phase__%s__order%d_cos" % (label, order))

    for group_name, indices, _modality, _location in GROUPS:
        ids = list(indices)
        clean_energy = torch.mean(clean[:, ids, :] ** 2, dim=(1, 2)).clamp_min(eps)
        periodic_energy = torch.mean(periodic[:, ids, :] ** 2, dim=(1, 2)).clamp_min(eps)
        aperiodic_energy = torch.mean(aperiodic[:, ids, :] ** 2, dim=(1, 2)).clamp_min(eps)
        add_feature(values, names, periodic_energy / clean_energy,
                    "bg_periodicity__%s__periodic_fraction" % group_name)
        add_feature(values, names, aperiodic_energy / clean_energy,
                    "bg_aperiodic__%s__residual_fraction" % group_name)
        centered = aperiodic[:, ids, :] - torch.mean(aperiodic[:, ids, :], dim=2, keepdim=True)
        var = torch.mean(centered ** 2, dim=(1, 2)).clamp_min(eps)
        kurt = torch.mean(centered ** 4, dim=(1, 2)) / (var ** 2)
        add_feature(values, names, kurt, "bg_aperiodic__%s__kurtosis" % group_name)

    return periodic, aperiodic, values, names


def event_morphology_features(
    residual: torch.Tensor, centers: torch.Tensor, event_views: Mapping[str, torch.Tensor]
) -> Tuple[List[torch.Tensor], List[str], torch.Tensor]:
    eps = 1.0e-8
    values: List[torch.Tensor] = []
    names: List[str] = []
    segments = {
        "pre": torch.arange(-96, -24, device=residual.device),
        "core": torch.arange(-8, 9, device=residual.device),
        "early": torch.arange(9, 49, device=residual.device),
        "late": torch.arange(49, 129, device=residual.device),
    }
    windows = {key: p0._window_gather(residual, centers, offsets) for key, offsets in segments.items()}
    group_energy: Dict[str, List[torch.Tensor]] = {}
    for segment, window in windows.items():
        group_energy[segment] = []
        for _group_name, indices, _modality, _location in GROUPS:
            energy = torch.mean(group_power(window, indices), dim=2).clamp_min(eps)
            group_energy[segment].append(energy)

    for gi, (group_name, _indices, _modality, _location) in enumerate(GROUPS):
        pre = group_energy["pre"][gi]
        core = group_energy["core"][gi]
        early = group_energy["early"][gi]
        late = group_energy["late"][gi]
        add_event_summary(values, names, torch.log((core + eps) / (pre + eps)),
                          "event_morph__%s__core_pre_log_gain" % group_name)
        add_event_summary(values, names, torch.log((early + eps) / (pre + eps)),
                          "event_morph__%s__early_pre_log_gain" % group_name)
        add_event_summary(values, names, torch.log((late + eps) / (early + eps)),
                          "event_morph__%s__late_early_log_decay" % group_name)

    b1, b2, b3 = [group_energy["core"][i] for i in range(3)]
    vib_total = (b1 + b2 + b3).clamp_min(eps)
    for bi, energy in enumerate((b1, b2, b3), start=1):
        add_event_summary(values, names, energy / vib_total,
                          "event_morph__B%d__core_energy_share" % bi)
    add_event_summary(values, names, (b3 - b1) / vib_total, "event_morph__topology__endpoint_signed")
    add_event_summary(values, names, torch.log((b2 + eps) / torch.sqrt(b1 * b3 + eps)),
                      "event_morph__topology__midness")
    ac_a, ac_b = group_energy["core"][4], group_energy["core"][5]
    add_event_summary(values, names, (ac_b - ac_a) / (ac_a + ac_b).clamp_min(eps),
                      "event_morph__sound__endpoint_signed")

    offsets = torch.arange(-64, 129, device=residual.device)
    event_window = p0._window_gather(residual, centers, offsets)
    vib_env = torch.sqrt(torch.mean(event_window[:, :, 0:6, :] ** 2, dim=2).clamp_min(eps))
    ac_env = torch.sqrt(torch.mean(event_window[:, :, 8:10, :] ** 2, dim=2).clamp_min(eps))
    mixed_env = 0.5 * vib_env / torch.amax(vib_env, dim=2, keepdim=True).clamp_min(eps)
    mixed_env = mixed_env + 0.5 * ac_env / torch.amax(ac_env, dim=2, keepdim=True).clamp_min(eps)
    peak = torch.amax(mixed_env, dim=2).clamp_min(eps)
    peak_index = torch.argmax(mixed_env, dim=2)
    half_width = torch.mean((mixed_env >= 0.5 * peak[:, :, None]).float(), dim=2)
    add_event_summary(values, names, (peak_index.float() - 64.0) / float(len(offsets)),
                      "event_morph__mixed__peak_offset_norm")
    add_event_summary(values, names, half_width, "event_morph__mixed__halfmax_width_fraction")

    vib_score = event_views["vibration_score"]
    ac_score = event_views["acoustic_score"]
    vib_peak = torch.argmax(vib_score, dim=1)
    ac_peak = torch.argmax(ac_score, dim=1)
    shared = centers[:, 0]
    add_feature(values, names, (ac_peak - vib_peak).float() / float(residual.shape[2]),
                "event_morph__cross_modal__acoustic_vibration_peak_lag_norm")
    add_feature(values, names, (shared - vib_peak).float() / float(residual.shape[2]),
                "event_morph__cross_modal__shared_vibration_peak_lag_norm")
    score_at_centers = torch.gather(event_views["mixed_score"], 1, centers)
    add_feature(values, names, torch.std(score_at_centers, dim=1, unbiased=False),
                "event_morph__multi_event__score_std")
    return values, names, event_window


def local_time_frequency_features(event_window: torch.Tensor, fs_hz: float) -> Tuple[List[torch.Tensor], List[str]]:
    eps = 1.0e-8
    values: List[torch.Tensor] = []
    names: List[str] = []
    centered = event_window - torch.mean(event_window, dim=3, keepdim=True)
    spec = torch.fft.rfft(centered, dim=3)
    power = torch.abs(spec) ** 2
    freqs = torch.fft.rfftfreq(event_window.shape[3], d=1.0 / float(fs_hz)).to(event_window.device)
    bands = ((0.0, 100.0), (100.0, 300.0), (300.0, 800.0), (800.0, 1500.0), (1500.0, fs_hz / 2.0 + 1.0))
    for group_name, indices, _modality, _location in GROUPS:
        gp = torch.sum(power[:, :, list(indices), :], dim=2)
        total = torch.sum(gp[:, :, 1:], dim=2).clamp_min(eps)
        shares = []
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            share = torch.sum(gp[:, :, mask], dim=2) / total
            shares.append(share)
            add_event_summary(
                values, names, share,
                "event_tf__%s__band_%d_%d_share" % (group_name, int(lo), int(min(hi, fs_hz / 2.0))),
            )
        low = shares[0] + shares[1]
        high = shares[3] + shares[4]
        add_event_summary(values, names, torch.log((high + eps) / (low + eps)),
                          "event_tf__%s__high_low_log_ratio" % group_name)
        centroid = torch.sum(gp * freqs.view(1, 1, -1), dim=2) / torch.sum(gp, dim=2).clamp_min(eps)
        add_event_summary(values, names, centroid, "event_tf__%s__centroid_hz" % group_name)
    return values, names


def best_lag_correlation(a: torch.Tensor, b: torch.Tensor, radius: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = a - torch.mean(a, dim=2, keepdim=True)
    b = b - torch.mean(b, dim=2, keepdim=True)
    a = a / torch.sqrt(torch.mean(a ** 2, dim=2, keepdim=True).clamp_min(1.0e-8))
    b = b / torch.sqrt(torch.mean(b ** 2, dim=2, keepdim=True).clamp_min(1.0e-8))
    corr_values = []
    lags = list(range(-int(radius), int(radius) + 1))
    for lag in lags:
        if lag < 0:
            corr = torch.mean(a[:, :, :lag] * b[:, :, -lag:], dim=2)
        elif lag > 0:
            corr = torch.mean(a[:, :, lag:] * b[:, :, :-lag], dim=2)
        else:
            corr = torch.mean(a * b, dim=2)
        corr_values.append(corr)
    stack = torch.stack(corr_values, dim=2)
    best_idx = torch.argmax(stack, dim=2)
    best_corr = torch.gather(stack, 2, best_idx.unsqueeze(2)).squeeze(2)
    lag_tensor = torch.tensor(lags, dtype=torch.float32, device=a.device)
    best_lag = lag_tensor[best_idx]
    return best_lag, best_corr


def propagation_features(event_window: torch.Tensor, fs_hz: float) -> Tuple[List[torch.Tensor], List[str]]:
    values: List[torch.Tensor] = []
    names: List[str] = []
    env = []
    for _name, indices, _modality, _location in GROUPS:
        env.append(torch.sqrt(torch.mean(event_window[:, :, list(indices), :] ** 2, dim=2).clamp_min(1.0e-8)))
    envelopes = torch.stack(env, dim=2)
    pairs = (
        (0, 1, "B1_B2_vibration"), (1, 2, "B2_B3_vibration"), (0, 2, "B1_B3_vibration"),
        (4, 5, "sound_A_B"), (0, 4, "B1_vibration_soundA"),
        (2, 5, "B3_vibration_soundB"), (1, 3, "B2_vibration_orbit"),
    )
    for a, b, label in pairs:
        lag, corr = best_lag_correlation(envelopes[:, :, a, :], envelopes[:, :, b, :], int(CONFIG["event_lag_radius"]))
        add_event_summary(values, names, 1000.0 * lag / float(fs_hz), "event_graph__%s__lag_ms" % label)
        add_event_summary(values, names, corr, "event_graph__%s__max_corr" % label)
    return values, names


def fixed_shape_projection(base: torch.Tensor, output_dim: int, seed: int) -> torch.Tensor:
    key = (int(base.shape[1]), int(output_dim), int(seed), str(base.device))
    if key not in _PROJECTION_CACHE:
        rng = np.random.RandomState(int(seed))
        matrix = rng.choice([-1.0, 1.0], size=(int(base.shape[1]), int(output_dim))).astype(np.float32)
        matrix /= math.sqrt(float(output_dim))
        _PROJECTION_CACHE[key] = torch.from_numpy(matrix).to(base.device)
    normalized = (base - torch.mean(base, dim=1, keepdim=True)) / torch.std(base, dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return normalized.matmul(_PROJECTION_CACHE[key])


def raw_shape_sketch(
    raw: torch.Tensor, background: torch.Tensor, residual: torch.Tensor, event_window: torch.Tensor
) -> Tuple[List[torch.Tensor], List[str]]:
    pooled = []
    for view in (raw, background):
        pooled.append(F.adaptive_avg_pool1d(view, 12).flatten(1))
        pooled.append(torch.sqrt(F.adaptive_avg_pool1d(view ** 2, 12).clamp_min(1.0e-8)).flatten(1))
    pooled.append(torch.sqrt(F.adaptive_avg_pool1d(residual ** 2, 16).clamp_min(1.0e-8)).flatten(1))
    event_median = torch.median(event_window, dim=1).values
    pooled.append(F.adaptive_avg_pool1d(event_median, 16).flatten(1))
    base = torch.cat(pooled, dim=1)
    sketch = fixed_shape_projection(base, int(CONFIG["raw_sketch_dim"]), int(CONFIG["seed"]) + 1701)
    values = [sketch[:, j] for j in range(sketch.shape[1])]
    names = ["raw_sketch__dim_%03d" % j for j in range(sketch.shape[1])]
    return values, names


@torch.no_grad()
def extract_p2_features(x: torch.Tensor, rpm: torch.Tensor, fs_hz: float) -> Tuple[torch.Tensor, List[str]]:
    raw = x.float() - torch.median(x.float(), dim=2, keepdim=True).values
    background, _mask = p1.transient_suppressed_waveform(raw)
    periodic, aperiodic, values, names = background_features(background, rpm, fs_hz)
    event_residual = raw - periodic
    centers, event_views = event_centers_from_views(event_residual)
    morph_values, morph_names, event_window = event_morphology_features(event_residual, centers, event_views)
    tf_values, tf_names = local_time_frequency_features(event_window, fs_hz)
    graph_values, graph_names = propagation_features(event_window, fs_hz)
    sketch_values, sketch_names = raw_shape_sketch(raw, background, event_residual, event_window)
    values.extend(morph_values + tf_values + graph_values + sketch_values)
    names.extend(morph_names + tf_names + graph_names + sketch_names)
    features = torch.stack(values, dim=1)
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    if features.shape[1] != len(names):
        raise RuntimeError("P2 feature schema mismatch: %s names=%d" % (tuple(features.shape), len(names)))
    return features, names
