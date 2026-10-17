"""Exact foundational structure losses and pseudo-pair operators. Full final training orchestration is not included."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .structure_model import RealSemanticSameSpaceModel
from .structure_targets import derive_real_labels_torch
COMPONENTS = ("U", "M", "R")

def pair_type_components(pair_type: str) -> Tuple[str, str, str]:
    pair_type = str(pair_type).upper()
    if pair_type == 'UM':
        return ('U', 'M', 'R')
    if pair_type == 'UR':
        return ('U', 'R', 'M')
    if pair_type == 'MR':
        return ('M', 'R', 'U')
    raise ValueError(f'Unsupported pair_type={pair_type}')


def count_constrained_exist(logits_exist: torch.Tensor, count_target: torch.Tensor) -> torch.Tensor:
    counts = torch.clamp(count_target.long(), min=0, max=int(logits_exist.shape[1]))
    pred = torch.zeros_like(logits_exist, dtype=torch.long)
    for k in range(1, int(logits_exist.shape[1]) + 1):
        row_idx = torch.nonzero(counts == k, as_tuple=False).squeeze(1)
        if int(row_idx.numel()) <= 0:
            continue
        top_idx = torch.topk(logits_exist[row_idx], k=k, dim=-1).indices
        pred[row_idx.unsqueeze(1), top_idx] = 1
    return pred


def loss_exist(out: Dict[str, Any], y_umr: torch.Tensor) -> torch.Tensor:
    losses = []
    for ci, c in enumerate(COMPONENTS):
        losses.append(F.binary_cross_entropy_with_logits(out['exist_logits'][c], y_umr[:, ci]))
    return torch.stack(losses).sum()


def loss_real_grade(out: Dict[str, Any], labels: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    main_losses: List[torch.Tensor] = []
    aux_losses: List[torch.Tensor] = []
    u_mask = labels['U_exist'].bool() & (labels['U_degree'] >= 0)
    if bool(u_mask.any()):
        main_losses.append(F.cross_entropy(out['grade_logits']['U'][u_mask], labels['U_degree'][u_mask]))
    m_mask = labels['M_exist'].bool() & (labels['M_joint'] >= 0)
    if bool(m_mask.any()):
        main_losses.append(F.cross_entropy(out['grade_logits']['M'][m_mask], labels['M_joint'][m_mask]))
        aux_losses.append(F.cross_entropy(out['m_position_logits'][m_mask], labels['M_position'][m_mask]))
        aux_losses.append(F.cross_entropy(out['m_degree_logits'][m_mask], labels['M_degree'][m_mask]))
    r_mask = labels['R_exist'].bool() & (labels['R_position'] >= 0)
    if bool(r_mask.any()):
        main_losses.append(F.cross_entropy(out['grade_logits']['R'][r_mask], labels['R_position'][r_mask]))
    zero = out['grade_logits']['U'].new_zeros(())
    main = torch.stack(main_losses).sum() if main_losses else zero
    aux = torch.stack(aux_losses).sum() if aux_losses else zero
    return (main, aux)


def loss_null(model: RealSemanticSameSpaceModel, out: Dict[str, Any], y_umr: torch.Tensor) -> torch.Tensor:
    losses = []
    for ci, c in enumerate(COMPONENTS):
        z = out['z_c'][c]
        p0 = model.p_null[c].unsqueeze(0)
        mask = (y_umr[:, ci] < 0.5).float().unsqueeze(-1)
        losses.append(((mask * (z - p0)) ** 2).mean())
    return torch.stack(losses).sum()


def loss_decorr(out: Dict[str, Any], y_umr: torch.Tensor, exclude_normal: bool=True, eps: float=1e-06) -> Tuple[torch.Tensor, Dict[str, float]]:
    if exclude_normal:
        valid = y_umr.sum(dim=1) > 0
    else:
        valid = torch.ones(y_umr.shape[0], dtype=torch.bool, device=y_umr.device)
    if int(valid.sum().item()) < 2:
        zero = y_umr.new_zeros(())
        return (zero, {'corr_UM': 0.0, 'corr_UR': 0.0, 'corr_MR': 0.0})

    def stdz(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        return z / (z.std(dim=0, keepdim=True, unbiased=False) + eps)
    z_u = stdz(out['z_c']['U'][valid])
    z_m = stdz(out['z_c']['M'][valid])
    z_r = stdz(out['z_c']['R'][valid])
    b = float(z_u.shape[0])
    c_um = (z_u.T @ z_m / b).pow(2).mean()
    c_ur = (z_u.T @ z_r / b).pow(2).mean()
    c_mr = (z_m.T @ z_r / b).pow(2).mean()
    loss = c_um + c_ur + c_mr
    return (loss, {'corr_UM': float(c_um.detach().item()), 'corr_UR': float(c_ur.detach().item()), 'corr_MR': float(c_mr.detach().item())})


def ramp_weight(epoch: int, base_w: float, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < int(start_epoch):
        return 0.0
    if int(ramp_epochs) <= 0:
        return float(base_w)
    return float(base_w) * min(1.0, (epoch - int(start_epoch) + 1) / float(ramp_epochs))


def split_single_pools(y_umr: torch.Tensor) -> Dict[str, torch.Tensor]:
    single = y_umr.sum(dim=1) == 1
    return {'U': torch.nonzero(single & (y_umr[:, 0] > 0.5), as_tuple=False).squeeze(1), 'M': torch.nonzero(single & (y_umr[:, 1] > 0.5), as_tuple=False).squeeze(1), 'R': torch.nonzero(single & (y_umr[:, 2] > 0.5), as_tuple=False).squeeze(1)}


def sample_pair_indices(pool_a: torch.Tensor, pool_b: torch.Tensor, max_pairs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(int(pool_a.numel()), int(pool_b.numel()), int(max_pairs))
    if n <= 0:
        empty = pool_a.new_empty((0,))
        return (empty, empty)
    ia = torch.randperm(int(pool_a.numel()), device=pool_a.device)[:n]
    ib = torch.randperm(int(pool_b.numel()), device=pool_b.device)[:n]
    return (pool_a[ia], pool_b[ib])


def make_pseudo_batch(x10: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor, alpha_min: float, alpha_max: float) -> torch.Tensor:
    n = int(idx_a.numel())
    if n <= 0:
        return x10.new_empty((0, x10.shape[1], x10.shape[2]))
    alpha = torch.empty((n,), dtype=x10.dtype, device=x10.device).uniform_(float(alpha_min), float(alpha_max))
    return alpha.view(n, 1, 1) * x10[idx_a] + (1.0 - alpha).view(n, 1, 1) * x10[idx_b]


def pseudo_exist_target(n: int, pair_type: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    target = torch.zeros((n, 3), device=device, dtype=dtype)
    if pair_type == 'UM':
        target[:, 0] = 1.0
        target[:, 1] = 1.0
    elif pair_type == 'UR':
        target[:, 0] = 1.0
        target[:, 2] = 1.0
    elif pair_type == 'MR':
        target[:, 1] = 1.0
        target[:, 2] = 1.0
    else:
        raise ValueError(pair_type)
    return target


def pseudo_keep_loss(model: RealSemanticSameSpaceModel, out_clean: Dict[str, Any], out_mix: Dict[str, Any], idx_a: torch.Tensor, idx_b: torch.Tensor, pair_type: str, detach_ref: bool=True, rest_null_weight: float=1.0) -> torch.Tensor:
    comp_a, comp_b, comp_rest = pair_type_components(pair_type)
    z_ref_a = out_clean['z_c'][comp_a][idx_a]
    z_ref_b = out_clean['z_c'][comp_b][idx_b]
    if detach_ref:
        z_ref_a = z_ref_a.detach()
        z_ref_b = z_ref_b.detach()
    z_a = out_mix['z_c'][comp_a]
    z_b = out_mix['z_c'][comp_b]
    z_rest = out_mix['z_c'][comp_rest]
    p0 = model.p_null[comp_rest].unsqueeze(0).expand_as(z_rest).detach()
    z_ref_a = F.normalize(z_ref_a, p=2, dim=-1, eps=1e-12)
    z_ref_b = F.normalize(z_ref_b, p=2, dim=-1, eps=1e-12)
    z_a = F.normalize(z_a, p=2, dim=-1, eps=1e-12)
    z_b = F.normalize(z_b, p=2, dim=-1, eps=1e-12)
    z_rest = F.normalize(z_rest, p=2, dim=-1, eps=1e-12)
    p0 = F.normalize(p0, p=2, dim=-1, eps=1e-12)
    return F.mse_loss(z_a, z_ref_a) + F.mse_loss(z_b, z_ref_b) + float(rest_null_weight) * F.mse_loss(z_rest, p0)


def pseudo_exist_loss_for_pair(logits_exist: torch.Tensor, exist_target: torch.Tensor, pair_type: str, cfg: Dict[str, Any]) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits_exist, exist_target, reduction='none')
    weights = torch.ones_like(bce)
    pair_type = str(pair_type).upper()
    for ci, comp in enumerate(COMPONENTS):
        state = 'active' if float(exist_target[0, ci].detach().item()) > 0.5 else 'inactive'
        weights[:, ci] = float(cfg.get(f'pseudo_exist_weight_{pair_type}_{comp}_{state}', 1.0))
    return (bce * weights).sum() / weights.sum().clamp_min(1.0)


def pseudo_grade_loss_for_pair(out_mix: Dict[str, Any], labels: Dict[str, torch.Tensor], idx_a: torch.Tensor, idx_b: torch.Tensor, pair_type: str, w_m_aux: float) -> torch.Tensor:
    comp_a, comp_b, _ = pair_type_components(pair_type)
    pairs = [(comp_a, idx_a), (comp_b, idx_b)]
    losses: List[torch.Tensor] = []
    aux_losses: List[torch.Tensor] = []
    for comp, src_idx in pairs:
        if comp == 'U':
            losses.append(F.cross_entropy(out_mix['grade_logits']['U'], labels['U_degree'][src_idx]))
        elif comp == 'M':
            losses.append(F.cross_entropy(out_mix['grade_logits']['M'], labels['M_joint'][src_idx]))
            aux_losses.append(F.cross_entropy(out_mix['m_position_logits'], labels['M_position'][src_idx]))
            aux_losses.append(F.cross_entropy(out_mix['m_degree_logits'], labels['M_degree'][src_idx]))
        elif comp == 'R':
            losses.append(F.cross_entropy(out_mix['grade_logits']['R'], labels['R_position'][src_idx]))
        else:
            raise ValueError(comp)
    out_zero = out_mix['grade_logits']['U'].new_zeros(())
    main = torch.stack(losses).sum() if losses else out_zero
    aux = torch.stack(aux_losses).sum() if aux_losses else out_zero
    return main + float(w_m_aux) * aux


def count_target_from_kind(kind_int: torch.Tensor) -> torch.Tensor:
    return torch.clamp(kind_int.long(), min=0, max=2)


def count_loss_and_acc(count_logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
    loss = F.cross_entropy(count_logits, target.long())
    with torch.no_grad():
        pred = torch.argmax(count_logits, dim=-1)
        acc = float((pred == target.long()).float().mean().item())
    return (loss, acc)
