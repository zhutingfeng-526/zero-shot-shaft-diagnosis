"""Tensor parsing of component-presence and physical attribute training targets."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
def derive_real_semantic_labels_torch_fast(y_umr: Any, y_reg7: Any, exist_threshold: float=0.5, m_degree_threshold: float=0.3, eps: float=1e-08) -> Dict[str, Any]:
    """Native torch parser for training loops. Keeps tensors on device and avoids CPU round-trips."""
    import torch
    if not torch.is_tensor(y_umr):
        y_umr = torch.as_tensor(y_umr)
    if not torch.is_tensor(y_reg7):
        y_reg7 = torch.as_tensor(y_reg7, device=y_umr.device)
    y_umr = y_umr.float()
    y_reg7 = y_reg7.float()
    if y_umr.ndim == 1:
        y_umr = y_umr.reshape(1, -1)
    if y_reg7.ndim == 1:
        y_reg7 = y_reg7.reshape(1, -1)
    if y_umr.shape[1] != 3:
        raise ValueError(f'y_umr must have shape [N,3], got {tuple(y_umr.shape)}')
    if y_reg7.shape[1] != 7:
        raise ValueError(f'y_reg7 must have shape [N,7], got {tuple(y_reg7.shape)}')
    n = int(y_umr.shape[0])
    device = y_umr.device
    exist = (y_umr > float(exist_threshold)).long()
    u_degree = torch.full((n,), -1, dtype=torch.long, device=device)
    m_position = torch.full((n,), -1, dtype=torch.long, device=device)
    m_degree = torch.full((n,), -1, dtype=torch.long, device=device)
    m_joint = torch.full((n,), -1, dtype=torch.long, device=device)
    r_position = torch.full((n,), -1, dtype=torch.long, device=device)
    legacy_grade_m = torch.full((n,), -1, dtype=torch.long, device=device)
    u_mask = exist[:, 0] == 1
    if bool(u_mask.any()):
        u_raw = torch.round(y_reg7[:, 0]).long() - 1
        u_degree[u_mask] = torch.clamp(u_raw[u_mask], min=0, max=2)
    m_vals = y_reg7[:, 1:4]
    m_amp, m_pos_all = torch.max(m_vals, dim=-1)
    m_mask = exist[:, 1] == 1
    if bool(m_mask.any()):
        valid = m_mask & (m_amp > float(eps))
        deg = (m_amp >= float(m_degree_threshold)).long()
        m_position[valid] = m_pos_all[valid].long()
        m_degree[valid] = deg[valid]
        m_joint[valid] = m_position[valid] * 2 + m_degree[valid]
        legacy_grade_m[valid] = deg[valid]
    r_vals = y_reg7[:, 4:7]
    r_amp, r_pos_all = torch.max(r_vals, dim=-1)
    r_mask = exist[:, 2] == 1
    if bool(r_mask.any()):
        valid = r_mask & (r_amp > float(eps))
        r_position[valid] = r_pos_all[valid].long()
    legacy_grade_umr = torch.stack([u_degree, legacy_grade_m, r_position], dim=1)
    return {'U_exist': exist[:, 0], 'M_exist': exist[:, 1], 'R_exist': exist[:, 2], 'exist_umr': exist, 'structure_count': torch.sum(exist, dim=1).long(), 'U_degree': u_degree, 'M_position': m_position, 'M_degree': m_degree, 'M_joint': m_joint, 'R_position': r_position, 'M_amp': m_amp.float(), 'M_vector': m_vals.float(), 'R_vector': r_vals.float(), 'legacy_grade_M': legacy_grade_m, 'legacy_grade_umr': legacy_grade_umr}

def derive_real_labels_torch(y_umr: torch.Tensor, y_reg7: torch.Tensor, exist_threshold: float=0.5, m_degree_threshold: float=0.3, eps: float=1e-08) -> Dict[str, torch.Tensor]:
    return derive_real_semantic_labels_torch_fast(y_umr, y_reg7, exist_threshold=exist_threshold, m_degree_threshold=m_degree_threshold, eps=eps)
