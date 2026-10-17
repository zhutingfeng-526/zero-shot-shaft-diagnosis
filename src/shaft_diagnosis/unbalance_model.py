"""Final U Softsign bounded residual transport and exact ordinal/ranking training losses."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import unbalance_features as frontend
CONFIG = {"ordinal_slope": 4.0, "ordinal_lambda": 0.25, "rank_lambda": 0.05, "rank_margin_fraction": 0.10}

def bounded_map(raw_shift: torch.Tensor, bound: torch.Tensor, mode: str) -> torch.Tensor:
    normalized = raw_shift / bound.clamp_min(1e-06)
    if mode == 'tanh':
        return bound * torch.tanh(normalized)
    if mode == 'softsign':
        return bound * normalized / (1.0 + torch.abs(normalized))
    raise RuntimeError('Unknown boundary map: %s' % mode)


def ordinal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = torch.stack([(labels > 0).float(), (labels > 1).float()], dim=1)
    return F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(dim=1)


def rank_loss(corrected: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    means: List[torch.Tensor] = []
    for degree in range(3):
        mask = labels == degree
        if not bool(mask.any()):
            return corrected.sum() * 0.0
        means.append(corrected[mask].mean())
    return 0.5 * (F.softplus(margin - (means[1] - means[0])) + F.softplus(margin - (means[2] - means[1])))


class RobustTransportNet(nn.Module):
    """Shared residual transport with exact clean gate and zero residual start."""

    def __init__(self, spec: Mapping[str, Any], contract: Mapping[str, Any]):
        super().__init__()
        self.spec = dict(spec)
        self.boundary = str(spec['boundary'])
        self.orientation = float(contract['orientation'])
        self.anchor_index = int(frontend.U_ANCHOR_INDEX)
        self.register_buffer('physics_center', torch.tensor(np.asarray(contract['center']), dtype=torch.float32))
        self.register_buffer('physics_scale', torch.tensor(np.asarray(contract['scale']), dtype=torch.float32))
        self.register_buffer('thresholds', torch.tensor(np.asarray(contract['threshold_init']), dtype=torch.float32))
        self.register_buffer('correction_bound', torch.tensor(float(contract['correction_bound']), dtype=torch.float32))
        self.register_buffer('transport_scale', torch.tensor(float(contract['transport_scale']), dtype=torch.float32))
        self.phase_encoder = nn.Sequential(nn.Conv1d(8, 8, kernel_size=7, padding=3, groups=8, bias=False), nn.Conv1d(8, 16, kernel_size=1), nn.GroupNorm(4, 16), nn.GELU(), nn.AvgPool1d(2), nn.Conv1d(16, 16, kernel_size=5, padding=2, groups=16, bias=False), nn.Conv1d(16, 20, kernel_size=1), nn.GroupNorm(4, 20), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.physics_encoder = nn.Sequential(nn.Linear(len(frontend.U_PHYSICS_NAMES), 20), nn.GELU(), nn.Linear(20, 12), nn.GELU())
        self.shift_head = nn.Sequential(nn.Linear(32, 20), nn.GELU(), nn.Linear(20, 1))
        nn.init.zeros_(self.shift_head[-1].weight)
        nn.init.zeros_(self.shift_head[-1].bias)

    def forward(self, physics: torch.Tensor, phase: torch.Tensor, family_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        standardized = (physics.float() - self.physics_center) / self.physics_scale.clamp_min(1e-06)
        raw_anchor = self.orientation * standardized[:, self.anchor_index]
        shift_physics = standardized.clone()
        shift_physics[:, self.anchor_index] = 0.0
        latent = torch.cat([self.phase_encoder(phase.float()), self.physics_encoder(shift_physics)], dim=1)
        raw_shift = self.shift_head(latent)[:, 0]
        mapped_shift = bounded_map(raw_shift, self.correction_bound, self.boundary)
        compound_gate = (family_id != 0).float()
        shift = compound_gate * mapped_shift
        corrected = raw_anchor - shift
        slope = float(CONFIG['ordinal_slope'])
        cumulative_logits = slope * (corrected.unsqueeze(1) - self.thresholds.unsqueeze(0))
        gt = torch.sigmoid(cumulative_logits)
        probabilities = torch.stack([1.0 - gt[:, 0], gt[:, 0] - gt[:, 1], gt[:, 1]], dim=1)
        probabilities = probabilities.clamp_min(1e-08)
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        return {'raw_anchor': raw_anchor, 'predicted_shift': shift, 'bounded_shift': shift, 'corrected_anchor': corrected, 'cumulative_logits': cumulative_logits, 'probabilities': probabilities, 'uncertainty': torch.zeros_like(raw_shift), 'confidence': torch.ones_like(raw_shift)}


def neural_loss(model: RobustTransportNet, batch: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    physics, phase, label, _context_id, family_id, delta_true = [value.to(device, non_blocking=True) for value in batch]
    out = model(physics, phase, family_id)
    scale = model.transport_scale.clamp_min(1e-06)
    regression = F.smooth_l1_loss(out['predicted_shift'] / scale, delta_true / scale, reduction='none')
    sample = regression + float(CONFIG['ordinal_lambda']) * ordinal_loss(out['cumulative_logits'], label)
    margin = float(CONFIG['rank_margin_fraction']) * float(model.transport_scale.detach().cpu().item())
    return sample.mean() + float(CONFIG['rank_lambda']) * rank_loss(out['corrected_anchor'], label, margin)


def update_average_state(average: Optional[Dict[str, torch.Tensor]], state: Mapping[str, torch.Tensor], count: int) -> Dict[str, torch.Tensor]:
    if average is None:
        return {name: value.detach().cpu().clone() for name, value in state.items()}
    updated: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        current = value.detach().cpu()
        if current.is_floating_point():
            updated[name] = average[name] + (current - average[name]) / float(count)
        else:
            updated[name] = current.clone()
    return updated
