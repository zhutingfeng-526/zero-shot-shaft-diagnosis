"""Final PB6 per-epoch update with geometry, view consistency and group-DRO losses."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from . import structure_losses as train_mod
COMPONENTS = ('U', 'M', 'R')
COMP_INDEX = {c: i for i, c in enumerate(COMPONENTS)}
PAIR_TYPES = ('UM', 'UR', 'MR')

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-12)


def make_weighted_pseudo_batch(x10: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor, alpha: float) -> torch.Tensor:
    a = x10[idx_a]
    b = x10[idx_b]
    return float(alpha) * a + (1.0 - float(alpha)) * b


def deterministic_pair_indices(pool_a: torch.Tensor, pool_b: torch.Tensor, max_pairs: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(int(pool_a.numel()), int(pool_b.numel()), int(max_pairs))
    if n <= 0:
        return (pool_a[:0], pool_b[:0])
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed) & 2147483647)
    pa = torch.randperm(int(pool_a.numel()), generator=g, device='cpu')[:n].to(pool_a.device)
    pb = torch.randperm(int(pool_b.numel()), generator=g, device='cpu')[:n].to(pool_b.device)
    return (pool_a[pa], pool_b[pb])


def sample_pair_indices_for_job(train_mod: Any, pools: Dict[str, torch.Tensor], pair_type: str, cfg: Dict[str, Any], epoch: int, batch_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    comp_a, comp_b, _ = train_mod.pair_type_components(pair_type)
    max_pairs = int(cfg['pseudo_max_pairs_per_type'])
    if bool(cfg.get('deterministic_pseudo', False)):
        base = int(cfg.get('seed', cfg.get('seed_base', 202700)))
        pair_offset = {'UM': 101, 'UR': 211, 'MR': 307}[pair_type]
        seed = base * 1000003 + int(epoch) * 1009 + int(batch_id) * 37 + pair_offset
        return deterministic_pair_indices(pools[comp_a], pools[comp_b], max_pairs, seed)
    return train_mod.sample_pair_indices(pools[comp_a], pools[comp_b], max_pairs)


def set_stageb_trainability(model: torch.nn.Module, cfg: Dict[str, Any], epoch: int) -> int:
    freeze_epochs = int(cfg.get('stageb_encoder_freeze_epochs', 0))
    start = int(cfg.get('pseudo_start_epoch', 16))
    freeze = freeze_epochs > 0 and start <= int(epoch) < start + freeze_epochs
    frozen = 0
    for name, p in model.named_parameters():
        is_encoder = name.startswith('base_model.enc_vib') or name.startswith('base_model.enc_orb') or name.startswith('base_model.enc_acou')
        if is_encoder:
            p.requires_grad = not freeze
            if freeze:
                frozen += int(p.numel())
        else:
            p.requires_grad = True
    return frozen


def source_geometry_loss(model: torch.nn.Module, out: Dict[str, Any], labels: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> torch.Tensor:
    label_keys = {'U': 'U_degree', 'M': 'M_joint', 'R': 'R_position'}
    exist_keys = {'U': 'U_exist', 'M': 'M_exist', 'R': 'R_exist'}
    losses: List[torch.Tensor] = []
    active_null_min = float(cfg.get('source_geom_active_null_min', 0.55))
    class_sep_min = float(cfg.get('source_geom_class_sep_min', 0.08))
    for comp in COMPONENTS:
        z = out['z_c'][comp]
        cls = labels[label_keys[comp]]
        mask = labels[exist_keys[comp]].bool() & (cls >= 0)
        if int(mask.sum().item()) <= 1:
            continue
        centroids: List[torch.Tensor] = []
        for k in torch.unique(cls[mask]).detach().cpu().tolist():
            km = mask & (cls == int(k))
            if int(km.sum().item()) <= 1:
                continue
            c = F.normalize(z[km].float().mean(dim=0, keepdim=True), p=2, dim=-1)
            centroids.append(c)
            p0 = F.normalize(model.p_null[comp].float().unsqueeze(0), p=2, dim=-1)
            losses.append(F.relu(active_null_min - cosine_distance(c, p0)).mean())
        if len(centroids) >= 2:
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    losses.append(F.relu(class_sep_min - cosine_distance(centroids[i], centroids[j])).mean())
    if not losses:
        return out['count_logits'].new_zeros(())
    return torch.stack(losses).mean()


def evidence_logits(out: Dict[str, Any]) -> torch.Tensor:
    return torch.stack([out['exist_logits'][c] for c in COMPONENTS], dim=-1)


def pseudo_view_consistency_loss(out_a: Dict[str, Any], out_b: Dict[str, Any]) -> torch.Tensor:
    e_a = evidence_logits(out_a).float()
    e_b = evidence_logits(out_b).float()
    e_loss = F.mse_loss(e_a, e_b)
    p_a = F.log_softmax(out_a['count_logits'].float(), dim=-1)
    p_b = F.softmax(out_b['count_logits'].float(), dim=-1)
    c_loss = F.kl_div(p_a, p_b, reduction='batchmean')
    return e_loss + 0.25 * c_loss


def false_component_loss_for_pair(train_mod: Any, model: torch.nn.Module, out_mix: Dict[str, Any], pair_type: str, cfg: Dict[str, Any]) -> torch.Tensor:
    comp_a, comp_b, comp_rest = train_mod.pair_type_components(pair_type)
    logits = evidence_logits(out_mix)
    ia = COMP_INDEX[comp_a]
    ib = COMP_INDEX[comp_b]
    ir = COMP_INDEX[comp_rest]
    rest_logit = logits[:, ir]
    active_min_logit = torch.minimum(logits[:, ia], logits[:, ib])
    temp = float(cfg.get('stageJ_false_softplus_temp', 1.0))
    tau_logit = float(cfg.get('stageJ_tau_inactive_logit', 0.0))
    gamma = float(cfg.get('stageJ_rank_gamma', 0.15))
    l_abs = F.softplus((rest_logit - tau_logit) / temp).mean() * temp
    l_rank = F.softplus((rest_logit - active_min_logit + gamma) / temp).mean() * temp
    z_rest = out_mix['z_c'][comp_rest]
    p0 = model.p_null[comp_rest].unsqueeze(0).expand_as(z_rest)
    d_null = cosine_distance(z_rest, p0)
    tau_null = float(cfg.get('stageJ_null_cosine_dist_tau', 0.08))
    null_temp = float(cfg.get('stageJ_null_softplus_temp', 0.05))
    l_null = F.softplus((d_null - tau_null) / null_temp).mean() * null_temp
    return float(cfg.get('stageJ_false_abs_weight', 1.0)) * l_abs + float(cfg.get('stageJ_false_rank_weight', 1.0)) * l_rank + float(cfg.get('stageJ_false_null_weight', 0.25)) * l_null


def run_train_epoch_directed(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, cfg: Dict[str, Any], epoch: int) -> Dict[str, float]:
    from . import structure_losses as train_mod
    model.train(True)
    frozen_params = set_stageb_trainability(model, cfg, epoch)
    amp = bool(cfg.get('use_amp', True)) and device.type == 'cuda'
    w_decorr = train_mod.ramp_weight(epoch, cfg['w_decorr'], cfg['decorr_start_epoch'], cfg['decorr_ramp_epochs'])
    w_pseudo = train_mod.ramp_weight(epoch, cfg['w_pseudo'], cfg['pseudo_start_epoch'], cfg['pseudo_ramp_epochs'])
    w_pseudo_exist = train_mod.ramp_weight(epoch, cfg['w_pseudo_exist'], cfg['pseudo_start_epoch'], cfg['pseudo_ramp_epochs'])
    w_pseudo_grade = train_mod.ramp_weight(epoch, cfg['w_pseudo_grade'], cfg['pseudo_start_epoch'], cfg['pseudo_ramp_epochs'])
    w_false = train_mod.ramp_weight(epoch, float(cfg.get('stageJ_w_false_component', 0.0)), int(cfg.get('stageJ_start_epoch', cfg['pseudo_start_epoch'])), int(cfg.get('stageJ_ramp_epochs', cfg['pseudo_ramp_epochs'])))
    w_source_geom = train_mod.ramp_weight(epoch, float(cfg.get('w_source_geometry', 0.0)), int(cfg.get('pseudo_start_epoch', 16)), int(cfg.get('pseudo_ramp_epochs', 6)))
    w_view = train_mod.ramp_weight(epoch, float(cfg.get('w_pseudo_view_consistency', 0.0)), int(cfg.get('pseudo_start_epoch', 16)), int(cfg.get('pseudo_ramp_epochs', 6)))
    sums = defaultdict(float)
    n_batches = 0
    n_samples = 0
    for batch_id, batch in enumerate(loader):
        x10, y_umr, y_reg7, kind_int, *_ = batch
        x10 = x10.to(device, non_blocking=True)
        y_umr = y_umr.to(device, non_blocking=True)
        y_reg7 = y_reg7.to(device, non_blocking=True)
        kind_int = kind_int.to(device, non_blocking=True)
        labels = train_mod.derive_real_labels_torch(y_umr, y_reg7, exist_threshold=float(cfg['exist_threshold']), m_degree_threshold=float(cfg['m_degree_threshold']))
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x10)
            l_exist = train_mod.loss_exist(out, y_umr)
            l_grade, l_m_aux = train_mod.loss_real_grade(out, labels)
            l_null = train_mod.loss_null(model, out, y_umr)
            if w_decorr > 0.0:
                l_decorr, corr = train_mod.loss_decorr(out, y_umr, exclude_normal=bool(cfg['decorr_exclude_normal']))
            else:
                l_decorr = y_umr.new_zeros(())
                corr = {'corr_UM': 0.0, 'corr_UR': 0.0, 'corr_MR': 0.0}
            l_count_real, count_acc_real = train_mod.count_loss_and_acc(out['count_logits'], train_mod.count_target_from_kind(kind_int))
            l_source_geometry = source_geometry_loss(model, out, labels, cfg) if w_source_geom > 0.0 else y_umr.new_zeros(())
            l_pseudo = y_umr.new_zeros(())
            l_pseudo_exist = y_umr.new_zeros(())
            l_pseudo_grade = y_umr.new_zeros(())
            l_count_pseudo = y_umr.new_zeros(())
            l_false_component = y_umr.new_zeros(())
            l_view = y_umr.new_zeros(())
            pseudo_counts = {'UM': 0, 'UR': 0, 'MR': 0}
            count_acc_pseudo_vals: List[float] = []
            group_loss: Dict[str, torch.Tensor] = {}
            group_weight = {'UM': 0.0, 'UR': 0.0, 'MR': 0.0}
            if w_pseudo > 0.0 or w_pseudo_exist > 0.0 or w_pseudo_grade > 0.0 or (w_false > 0.0) or (w_view > 0.0):
                pools = train_mod.split_single_pools(y_umr)
                group_terms: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
                for pair_type in PAIR_TYPES:
                    idx_a, idx_b = sample_pair_indices_for_job(train_mod, pools, pair_type, cfg, epoch, batch_id)
                    if int(idx_a.numel()) <= 0:
                        continue
                    alpha = float(cfg.get('pseudo_alpha_fixed', 0.5))
                    x_mix = make_weighted_pseudo_batch(x10, idx_a, idx_b, alpha=alpha)
                    out_mix = model(x_mix)
                    keep = train_mod.pseudo_keep_loss(model, out, out_mix, idx_a, idx_b, pair_type=pair_type, detach_ref=bool(cfg['pseudo_detach_ref']), rest_null_weight=float(cfg.get(f'pseudo_rest_null_weight_{pair_type}', 1.0)))
                    logits_exist = evidence_logits(out_mix)
                    exist_target = train_mod.pseudo_exist_target(int(idx_a.numel()), pair_type, logits_exist.device, logits_exist.dtype)
                    exist = train_mod.pseudo_exist_loss_for_pair(logits_exist, exist_target, pair_type=pair_type, cfg=cfg)
                    grade = train_mod.pseudo_grade_loss_for_pair(out_mix, labels, idx_a, idx_b, pair_type=pair_type, w_m_aux=float(cfg['w_m_aux']))
                    count_target = torch.full((int(idx_a.numel()),), 2, dtype=torch.long, device=out_mix['count_logits'].device)
                    count_l, count_acc = train_mod.count_loss_and_acc(out_mix['count_logits'], count_target)
                    false_l = false_component_loss_for_pair(train_mod, model, out_mix, pair_type, cfg)
                    view_l = y_umr.new_zeros(())
                    if w_view > 0.0:
                        delta = float(cfg.get('pseudo_view_alpha_delta', 0.06))
                        x_mix_alt = make_weighted_pseudo_batch(x10, idx_a, idx_b, alpha=min(0.95, max(0.05, alpha + delta)))
                        out_alt = model(x_mix_alt)
                        view_l = pseudo_view_consistency_loss(out_mix, out_alt)
                    total = w_pseudo * keep + w_pseudo_exist * exist + w_pseudo_grade * grade + float(cfg['w_count']) * float(cfg['count_lambda_pseudo']) * count_l + w_false * false_l + w_view * view_l
                    group_terms.append((pair_type, total, keep, exist, grade, count_l, false_l, view_l))
                    pseudo_counts[pair_type] = int(idx_a.numel())
                    count_acc_pseudo_vals.append(float(count_acc))
                if group_terms:
                    totals = torch.stack([g[1] for g in group_terms])
                    if bool(cfg.get('pseudo_group_dro', False)):
                        eta = float(cfg.get('pseudo_group_dro_eta', 2.0))
                        q = torch.softmax(eta * totals.detach(), dim=0)
                    else:
                        q = torch.ones_like(totals) / float(len(group_terms))
                    pseudo_total = (q * totals).sum()
                    for qi, item in zip(q, group_terms):
                        pair_type, total, keep, exist, grade, count_l, false_l, view_l = item
                        group_loss[pair_type] = total.detach()
                        group_weight[pair_type] = float(qi.detach().item())
                    l_pseudo = torch.stack([g[2] for g in group_terms]).mean()
                    l_pseudo_exist = torch.stack([g[3] for g in group_terms]).mean()
                    l_pseudo_grade = torch.stack([g[4] for g in group_terms]).mean()
                    l_count_pseudo = torch.stack([g[5] for g in group_terms]).mean()
                    l_false_component = torch.stack([false_l for _, _, _, _, _, _, false_l, _view_l in group_terms]).mean()
                    l_view = torch.stack([view_l for _, _, _, _, _, _, _false_l, view_l in group_terms]).mean()
                else:
                    pseudo_total = y_umr.new_zeros(())
            else:
                pseudo_total = y_umr.new_zeros(())
            l_count = l_count_real + float(cfg['count_lambda_pseudo']) * l_count_pseudo
            real_loss = float(cfg['w_exist']) * l_exist + float(cfg['w_grade']) * l_grade + float(cfg['w_m_aux']) * l_m_aux + float(cfg['w_null']) * l_null + w_decorr * l_decorr + float(cfg['w_count']) * l_count_real + w_source_geom * l_source_geometry
            loss = real_loss + pseudo_total
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n_batches += 1
        n_samples += int(x10.shape[0])
        sums['loss'] += float(loss.detach().item())
        sums['L_exist'] += float(l_exist.detach().item())
        sums['L_grade'] += float(l_grade.detach().item())
        sums['L_m_aux'] += float(l_m_aux.detach().item())
        sums['L_null'] += float(l_null.detach().item())
        sums['L_decorr'] += float(l_decorr.detach().item())
        sums['L_pseudo'] += float(l_pseudo.detach().item())
        sums['L_pseudo_exist'] += float(l_pseudo_exist.detach().item())
        sums['L_pseudo_grade'] += float(l_pseudo_grade.detach().item())
        sums['L_count'] += float(l_count.detach().item())
        sums['L_false_component'] += float(l_false_component.detach().item())
        sums['L_source_geometry'] += float(l_source_geometry.detach().item())
        sums['L_pseudo_view_consistency'] += float(l_view.detach().item())
        sums['corr_UM'] += float(corr['corr_UM'])
        sums['corr_UR'] += float(corr['corr_UR'])
        sums['corr_MR'] += float(corr['corr_MR'])
        sums['count_acc_real'] += float(count_acc_real)
        sums['count_acc_pseudo'] += float(np.mean(count_acc_pseudo_vals)) if count_acc_pseudo_vals else 0.0
        sums['pseudo_pairs_UM'] += float(pseudo_counts['UM'])
        sums['pseudo_pairs_UR'] += float(pseudo_counts['UR'])
        sums['pseudo_pairs_MR'] += float(pseudo_counts['MR'])
        for pair_type in PAIR_TYPES:
            sums[f'L_group_{pair_type}'] += float(group_loss.get(pair_type, y_umr.new_zeros(())).detach().item()) if pair_type in group_loss else 0.0
            sums[f'group_w_{pair_type}'] += float(group_weight.get(pair_type, 0.0))
        sums['stageb_encoder_frozen'] += float(frozen_params > 0)
    denom = max(1, n_batches)
    out_m = {k: v / denom for k, v in sums.items()}
    out_m['n_samples'] = float(n_samples)
    out_m['w_decorr_cur'] = float(w_decorr)
    out_m['w_pseudo_cur'] = float(w_pseudo)
    out_m['w_pseudo_exist_cur'] = float(w_pseudo_exist)
    out_m['w_pseudo_grade_cur'] = float(w_pseudo_grade)
    out_m['w_false_component_cur'] = float(w_false)
    return out_m
