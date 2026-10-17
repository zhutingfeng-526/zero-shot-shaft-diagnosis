"""Prototype distances and margins; prototypes must belong to the same trained checkpoint."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
COMPONENTS = ("U", "M", "R")
def pairwise_distance(z: torch.Tensor, p: torch.Tensor, metric: str='cosine') -> torch.Tensor:
    metric = str(metric).lower()
    if metric == 'cosine':
        z_n = F.normalize(z, p=2, dim=-1, eps=1e-12)
        p_n = F.normalize(p, p=2, dim=-1, eps=1e-12)
        return 1.0 - z_n @ p_n.T
    if metric == 'l2':
        return torch.cdist(z.float(), p.float(), p=2)
    raise ValueError(f'Unsupported distance metric: {metric}')

@torch.no_grad()
def prototype_margin_and_grade(out: Dict[str, Any], protos: Dict[str, Any], device: torch.device, metric: str):
    margins = []
    grades: Dict[str, torch.Tensor] = {}
    for c in COMPONENTS:
        z_c = out['z_c'][c].float()
        p0 = protos[f'p_{c}0'].to(device).float()
        P = protos[f'P_{c}'].to(device).float()
        valid = protos[f'P_{c}_valid'].to(device).bool()
        grade_ids = protos[f'P_{c}_grade_ids'].to(device).long()
        d_null = pairwise_distance(z_c, p0.unsqueeze(0), metric).squeeze(1)
        d_all = pairwise_distance(z_c, P, metric)
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        d_valid = d_all[:, valid_idx]
        d_min, row_local = torch.min(d_valid, dim=1)
        pred_row = valid_idx[row_local]
        grades[c] = grade_ids[pred_row].long()
        margins.append((d_null - d_min).float())
    return (torch.stack(margins, dim=1), grades)
