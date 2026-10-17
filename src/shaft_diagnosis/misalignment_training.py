"""Linear M attribute training and weighted source-domain losses."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
POSITION_CLASSES = 3
DEGREE_CLASSES = 2



def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class DualAttributeLinear(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(int(input_dim), POSITION_CLASSES + DEGREE_CLASSES)

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(value)
        return logits[:, :POSITION_CLASSES], logits[:, POSITION_CLASSES:]


def equal_group_weights(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups, dtype=str)
    counts = {group: int(np.sum(values == group)) for group in sorted(set(values.tolist()))}
    weights = np.asarray([1.0 / float(counts[group]) for group in values], dtype=np.float32)
    weights *= float(len(weights)) / float(np.sum(weights))
    return weights


def train_dual_linear(
    features: np.ndarray,
    position: np.ndarray,
    degree: np.ndarray,
    weights: np.ndarray,
    ids: np.ndarray,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    selected = np.asarray(ids, dtype=np.int64)
    if len(selected) == 0:
        raise RuntimeError("Empty source training fold")
    set_seed(int(seed))
    model = DualAttributeLinear(features.shape[1]).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=float(weight_decay))
    position_np = np.asarray(position, dtype=np.int64)
    degree_np = np.asarray(degree, dtype=np.int64)
    weights_np = np.asarray(weights, dtype=np.float32)
    model.train()
    for epoch in range(int(epochs)):
        permutation = selected[torch.randperm(len(selected), generator=generator).numpy()]
        for start in range(0, len(permutation), int(batch_size)):
            current = permutation[start:start + int(batch_size)]
            xb = torch.from_numpy(np.asarray(features[current], dtype=np.float32)).to(device, non_blocking=True)
            yp = torch.from_numpy(position_np[current]).to(device)
            yd = torch.from_numpy(degree_np[current]).to(device)
            wb = torch.from_numpy(weights_np[current]).to(device)
            optimizer.zero_grad(set_to_none=True)
            position_logits, degree_logits = model(xb)
            loss_p = F.cross_entropy(position_logits, yp, reduction="none")
            loss_d = F.cross_entropy(degree_logits, yd, reduction="none")
            loss = torch.sum((loss_p + loss_d) * wb) / torch.clamp(torch.sum(wb), min=1.0e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            del xb, yp, yd, wb, position_logits, degree_logits, loss_p, loss_d, loss
    state = {
        "weight": model.linear.weight.detach().cpu().numpy().astype(np.float32),
        "bias": model.linear.bias.detach().cpu().numpy().astype(np.float32),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state


def class_domain_centroid_loss(
    logits: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, classes: int
) -> torch.Tensor:
    terms: List[torch.Tensor] = []
    for cls in range(int(classes)):
        clean = logits[(labels == cls) & (domains == 0)]
        if clean.shape[0] == 0:
            continue
        clean_center = clean.mean(dim=0)
        for domain in (1, 2):
            context = logits[(labels == cls) & (domains == domain)]
            if context.shape[0] > 0:
                terms.append(torch.mean((context.mean(dim=0) - clean_center).square()))
    if not terms:
        return logits.new_zeros(())
    return torch.stack(terms).mean()


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    return torch.sum(losses * weights) / torch.clamp(weights.sum(), min=1.0e-6)


def anchor_kl(final_logits: torch.Tensor, anchor_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return final_logits.new_zeros(())
    target = torch.softmax(anchor_logits[mask].detach(), dim=1)
    return F.kl_div(torch.log_softmax(final_logits[mask], dim=1), target, reduction="batchmean")
