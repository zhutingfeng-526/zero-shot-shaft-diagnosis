"""Transient-suppressed and periodic-residual evidence for M background readout."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import m_features_p0 as p0



CONFIG: Dict[str, Any] = {
    "gpu_id": 1,
    "expected_gpu_token": "A4000",
    "seed": 202707,
    "fs_hz": 5000.0,
    "batch_size": 192,
    "event_count": 3,
    "event_min_distance": 64,
    "event_mask_radius": 24,
    "periodic_bin_radius": 1,
    "pseudo_per_pair_alpha": 96,
    "pseudo_alphas": [0.35, 0.50, 0.65],
    "ridge_lambda": 1.0e-2,
    "bootstrap_repeats": 200,
}


def robust_z_time(signal: torch.Tensor) -> torch.Tensor:
    med = torch.median(signal, dim=1, keepdim=True).values
    mad = torch.median(torch.abs(signal - med), dim=1, keepdim=True).values.clamp_min(1.0e-6)
    return (signal - med) / (1.4826 * mad)


def event_centers(score: torch.Tensor, count: int, min_distance: int) -> torch.Tensor:
    width = int(min_distance) * 2 + 1
    pooled = torch.nn.functional.max_pool1d(
        score.unsqueeze(1), kernel_size=width, stride=1, padding=int(min_distance)
    ).squeeze(1)
    candidate = torch.where(score >= pooled, score, torch.full_like(score, -1.0e20))
    return torch.topk(candidate, k=int(count), dim=1).indices


def transient_suppressed_waveform(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    centered = x - torch.median(x, dim=2, keepdim=True).values
    vib_point = torch.sqrt(torch.mean(centered[:, 0:6, :] ** 2, dim=1).clamp_min(1.0e-8))
    ac_point = torch.sqrt(torch.mean(centered[:, 8:10, :] ** 2, dim=1).clamp_min(1.0e-8))
    score = 0.5 * robust_z_time(vib_point) + 0.5 * robust_z_time(ac_point)
    margin = max(int(CONFIG["event_mask_radius"]) + 1, int(CONFIG["event_min_distance"]))
    score = score.clone()
    score[:, :margin] = -1.0e20
    score[:, -margin:] = -1.0e20
    centers = event_centers(score, int(CONFIG["event_count"]), int(CONFIG["event_min_distance"]))
    time_ids = torch.arange(x.shape[2], device=x.device).view(1, 1, -1)
    distance = torch.abs(time_ids - centers.unsqueeze(2))
    mask = torch.any(distance <= int(CONFIG["event_mask_radius"]), dim=1)
    fill = torch.median(centered, dim=2, keepdim=True).values
    clean = torch.where(mask.unsqueeze(1), fill, centered)
    return clean, mask.float()


def gather_fft_bin(spec: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    index = bins.long().clamp(0, spec.shape[2] - 1).view(-1, 1, 1).expand(-1, spec.shape[1], 1)
    return torch.gather(spec, 2, index).squeeze(2)


def periodic_residual(x: torch.Tensor, rpm: torch.Tensor, fs_hz: float) -> torch.Tensor:
    centered = x - torch.mean(x, dim=2, keepdim=True)
    spec = torch.fft.rfft(centered, dim=2)
    periodic_spec = torch.zeros_like(spec)
    base_bins = torch.round((rpm.float() / 60.0) * float(x.shape[2]) / float(fs_hz)).long().clamp(1, spec.shape[2] - 1)
    radius = int(CONFIG["periodic_bin_radius"])
    for order in (1, 2, 3):
        center = (base_bins * int(order)).clamp(1, spec.shape[2] - 1)
        for offset in range(-radius, radius + 1):
            bins = (center + int(offset)).clamp(1, spec.shape[2] - 1)
            index = bins.view(-1, 1, 1).expand(-1, spec.shape[1], 1)
            values = torch.gather(spec, 2, index)
            periodic_spec.scatter_(2, index, values)
    periodic = torch.fft.irfft(periodic_spec, n=x.shape[2], dim=2)
    return centered - periodic


@torch.no_grad()
def extract_robust_evidence(x: torch.Tensor, rpm: torch.Tensor, fs_hz: float) -> Tuple[torch.Tensor, List[str]]:
    eps = 1.0e-8
    clean, mask = transient_suppressed_waveform(x.float())
    rms = torch.sqrt(torch.mean(clean ** 2, dim=2).clamp_min(eps))
    spec = torch.fft.rfft(clean, dim=2)
    base_bins = torch.round((rpm.float() / 60.0) * float(clean.shape[2]) / float(fs_hz)).long().clamp(1, spec.shape[2] - 1)
    amps = []
    for order in (1, 2, 3):
        bins = (base_bins * int(order)).clamp(1, spec.shape[2] - 1)
        amps.append(torch.abs(gather_fft_bin(spec, bins)) / float(clean.shape[2]))
    amp1, amp2, amp3 = amps

    values: List[torch.Tensor] = []
    names: List[str] = []
    bearing_amp = []
    bearing_rms = []
    for bi, (c0, c1) in enumerate(((0, 1), (2, 3), (4, 5)), start=1):
        a = torch.sqrt(amp1[:, c0] ** 2 + amp1[:, c1] ** 2).clamp_min(eps)
        r = torch.sqrt(rms[:, c0] ** 2 + rms[:, c1] ** 2).clamp_min(eps)
        bearing_amp.append(a)
        bearing_rms.append(r)
        values.extend([a, a / r, torch.log((rms[:, c0] + eps) / (rms[:, c1] + eps))])
        names.extend([
            "U_masked_B%d_1x_xy" % bi,
            "U_masked_B%d_1x_to_rms" % bi,
            "M_masked_B%d_x_y_log_rms_ratio" % bi,
        ])
    ba = torch.stack(bearing_amp, dim=1)
    br = torch.stack(bearing_rms, dim=1)
    ba_total = torch.sum(ba, dim=1).clamp_min(eps)
    br_energy = br ** 2
    br_total = torch.sum(br_energy, dim=1).clamp_min(eps)
    ba_share = ba / ba_total[:, None]
    br_share = br_energy / br_total[:, None]
    values.extend([ba_share[:, 0], ba_share[:, 1], ba_share[:, 2]])
    names.extend(["U_masked_B1_1x_share", "U_masked_B2_1x_share", "U_masked_B3_1x_share"])
    values.append(torch.std(ba, dim=1, unbiased=False) / torch.mean(ba, dim=1).clamp_min(eps))
    names.append("U_masked_bearing_1x_cv")
    orbit_amp = torch.sqrt(amp1[:, 6] ** 2 + amp1[:, 7] ** 2).clamp_min(eps)
    orbit_rms = torch.sqrt(rms[:, 6] ** 2 + rms[:, 7] ** 2).clamp_min(eps)
    order_total = torch.sqrt(torch.sum(torch.stack(amps, dim=2)[:, 6:8, :] ** 2, dim=(1, 2))).clamp_min(eps)
    values.extend([orbit_amp, orbit_amp / orbit_rms, orbit_amp / order_total])
    names.extend(["U_masked_orbit_1x_radius", "U_masked_orbit_1x_to_rms", "U_masked_orbit_1x_to_order123"])

    values.extend([br_share[:, 0], br_share[:, 1], br_share[:, 2]])
    names.extend(["M_masked_B1_energy_share", "M_masked_B2_energy_share", "M_masked_B3_energy_share"])
    values.extend([
        (br_energy[:, 2] - br_energy[:, 0]) / br_total,
        torch.log((br_energy[:, 1] + eps) / torch.sqrt(br_energy[:, 0] * br_energy[:, 2] + eps)),
        torch.sqrt(br_total),
    ])
    names.extend(["M_masked_barycenter", "M_masked_midness", "M_masked_total_bearing_rms"])
    values.append(torch.mean(mask, dim=1))
    names.append("transient_mask_fraction")

    residual = periodic_residual(x.float(), rpm, fs_hz)
    r_features, r_names = p0.extract_event_features(residual)
    for j, name in enumerate(r_names):
        values.append(r_features[:, j])
        names.append("R_periodic_residual_%s" % name)
    return torch.stack(values, dim=1), names
