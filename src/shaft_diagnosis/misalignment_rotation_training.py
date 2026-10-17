"""Training packets and balanced optimization for per-rotation M residuals."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import hashlib
from collections import defaultdict
from .misalignment_models import TaskSetEncoder



POSITION_CLASSES = 3


DEGREE_CLASSES = 2


JOINT_CLASSES = 6


PROTOCOL_ID = "level2_p4zf_event_state_task_decoupled_safe_evidence_routing_phase1_r1"


STATE_NAMES = (
    "CORE", "TAIL", "UNCERTAIN", "POST_EARLY", "POST_MID", "POST_LATE",
    "PRE_STABLE", "INTER_STABLE", "CLEAN_OTHER",
)


ENVIRONMENTS = (
    "E0_clean_M",
    "E1_event_schedule_drop",
    "E2_core_local_corruption",
    "E3_sensor_propagation_variation",
    "E4_evidence_missing",
)


def stable_seed(parts: Iterable[Any]) -> int:
    value = "|".join(str(item) for item in parts)
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def materialize_environment(
    bank: Mapping[str, Any],
    environment: str,
    noise_scales: Mapping[str, float],
    group_ids: Mapping[str, np.ndarray],
    seed: int,
) -> Dict[str, np.ndarray]:
    tokens = np.asarray(bank["tokens"], dtype=np.float32).copy()
    states = np.asarray(bank["states"], dtype=np.float32).copy()
    valid = np.asarray(bank["valid"], dtype=bool).copy()
    core = np.asarray(bank["core"], dtype=bool).copy()
    tail = np.asarray(bank["tail"], dtype=bool).copy()
    uncertain = np.asarray(bank["uncertain"], dtype=bool).copy()
    base_position = np.asarray(bank["base_position"], dtype=np.float32).copy()
    base_degree = np.asarray(bank["base_degree"], dtype=np.float32).copy()
    count = len(tokens)
    gate = np.ones(count, dtype=np.float32)
    if environment == "E0_clean_M":
        states[:] = 0.0
        states[:, :, STATE_NAMES.index("CLEAN_OTHER")] = valid.astype(np.float32)
        states[:, :, len(STATE_NAMES) + 2] = valid.astype(np.float32)
        core[:] = False; tail[:] = False; uncertain[:] = False
        gate[:] = 0.0
    else:
        for sample in range(count):
            rng = np.random.RandomState(stable_seed((PROTOCOL_ID, seed, environment, bank["sample_key"][sample])))
            n = int(np.sum(valid[sample]))
            if environment == "E1_event_schedule_drop":
                candidates = np.where(valid[sample] & (tail[sample] | uncertain[sample]))[0]
                if len(candidates):
                    drop = candidates[rng.rand(len(candidates)) < 0.35]
                    valid[sample, drop] = False
            elif environment == "E2_core_local_corruption":
                ids = np.where(valid[sample] & core[sample])[0]
                if len(ids):
                    tokens[sample, ids] += rng.normal(0.0, 0.45, size=(len(ids), tokens.shape[2])).astype(np.float32)
            elif environment == "E3_sensor_propagation_variation":
                choices = ("B1_vibration", "B2_vibration", "B3_vibration", "shaft_orbit", "sound_A", "sound_B")
                selected = choices[int(rng.randint(0, len(choices)))]
                ids = np.asarray(group_ids.get(selected, np.zeros(0, dtype=np.int64)), dtype=np.int64)
                if len(ids):
                    tokens[sample, :n, ids] *= float(rng.uniform(0.55, 1.45))
            elif environment == "E4_evidence_missing":
                ids = np.where(valid[sample])[0]
                rng.shuffle(ids)
                keep_min = min(len(ids), 3)
                drop_n = min(max(0, len(ids) - keep_min), max(1, int(round(0.20 * len(ids)))))
                valid[sample, ids[:drop_n]] = False
                choices = ("shaft_orbit", "sound_A", "sound_B")
                selected = choices[int(rng.randint(0, len(choices)))]
                feature_ids = np.asarray(group_ids.get(selected, np.zeros(0, dtype=np.int64)), dtype=np.int64)
                if len(feature_ids):
                    tokens[sample, :, feature_ids] = 0.0
            else:
                raise ValueError("Unknown environment: %s" % environment)
            pos_noise = rng.normal(0.0, float(noise_scales["position"]), size=POSITION_CLASSES)
            deg_noise = rng.normal(0.0, float(noise_scales["degree"]), size=DEGREE_CLASSES)
            base_position[sample] += (pos_noise - np.mean(pos_noise)).astype(np.float32)
            base_degree[sample] += (deg_noise - np.mean(deg_noise)).astype(np.float32)
    position_valid = valid & ~core
    for sample in range(count):
        if int(np.sum(position_valid[sample])) < 3:
            position_valid[sample] = valid[sample]
        if not np.any(valid[sample]):
            raise RuntimeError("Environment removed every token for sample %d" % sample)
    return {
        "tokens": np.clip(tokens, -12.0, 12.0).astype(np.float32),
        "states": states.astype(np.float32),
        "position_valid": position_valid,
        "degree_valid": valid,
        "gate": gate,
        "base_position": base_position,
        "base_degree": base_degree,
    }


def tensor_slice(environment: Mapping[str, np.ndarray], ids: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "tokens": torch.from_numpy(np.asarray(environment["tokens"])[ids]).to(device),
        "states": torch.from_numpy(np.asarray(environment["states"])[ids]).to(device),
        "position_valid": torch.from_numpy(np.asarray(environment["position_valid"])[ids]).to(device),
        "degree_valid": torch.from_numpy(np.asarray(environment["degree_valid"])[ids]).to(device),
        "base_position": torch.from_numpy(np.asarray(environment["base_position"])[ids]).to(device),
        "base_degree": torch.from_numpy(np.asarray(environment["base_degree"])[ids]).to(device),
        "gate": torch.from_numpy(np.asarray(environment["gate"])[ids]).to(device),
    }


def forward_packet(model: TaskSetEncoder, packet: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        packet["tokens"], packet["states"], packet["position_valid"], packet["degree_valid"],
        packet["base_position"], packet["base_degree"], packet["gate"],
    )


def balanced_epoch_indices(position: np.ndarray, degree: np.ndarray, seed: int) -> np.ndarray:
    joint = np.asarray(position) * 2 + np.asarray(degree)
    cells = [np.where(joint == cell)[0] for cell in range(JOINT_CLASSES)]
    if any(len(ids) == 0 for ids in cells):
        raise RuntimeError("Balanced sampler requires all six M cells")
    target = max(len(ids) for ids in cells)
    rng = np.random.RandomState(int(seed))
    values = np.concatenate([rng.choice(ids, size=target, replace=len(ids) < target) for ids in cells])
    rng.shuffle(values)
    return values.astype(np.int64)


def train_model(
    method_id: str,
    mode: str,
    seed: int,
    bank: Mapping[str, Any],
    environments: Mapping[str, Mapping[str, np.ndarray]],
    bounds: Mapping[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[TaskSetEncoder, List[Dict[str, Any]]]:
    set_seed(seed)
    model = TaskSetEncoder(
        int(np.asarray(bank["tokens"]).shape[2]), int(np.asarray(bank["states"]).shape[2]),
        int(args.p4zf_hidden_dim), mode, float(bounds["position"]), float(bounds["degree"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.p4zf_learning_rate), weight_decay=float(args.p4zf_weight_decay),
    )
    labels_p = torch.from_numpy(np.asarray(bank["position"], dtype=np.int64)).to(device)
    labels_d = torch.from_numpy(np.asarray(bank["degree"], dtype=np.int64)).to(device)
    average: Optional[Dict[str, torch.Tensor]] = None
    average_count = 0
    history: List[Dict[str, Any]] = []
    batch_size = int(args.p4zf_batch_size)
    for epoch in range(1, int(args.p4zf_epochs) + 1):
        model.train()
        order = balanced_epoch_indices(bank["position"], bank["degree"], stable_seed((seed, epoch)))
        epoch_values: Dict[str, List[float]] = defaultdict(list)
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            idx = torch.from_numpy(ids).to(device)
            outputs: List[Dict[str, torch.Tensor]] = []
            environment_losses: List[torch.Tensor] = []
            for environment in ENVIRONMENTS:
                output = forward_packet(model, tensor_slice(environments[environment], ids, device))
                outputs.append(output)
                classification = F.cross_entropy(output["position_logits"], labels_p[idx]) + F.cross_entropy(output["degree_logits"], labels_d[idx])
                environment_losses.append(classification)
            losses = torch.stack(environment_losses)
            classification = torch.mean(losses)
            vrex = torch.var(losses, unbiased=False)
            probabilities_p = torch.stack([torch.softmax(output["position_logits"], dim=1) for output in outputs])
            probabilities_d = torch.stack([torch.softmax(output["degree_logits"], dim=1) for output in outputs])
            consistency = torch.mean((probabilities_p - probabilities_p.mean(dim=0, keepdim=True)) ** 2)
            consistency = consistency + torch.mean((probabilities_d - probabilities_d.mean(dim=0, keepdim=True)) ** 2)
            clean = outputs[0]
            anchor = torch.mean((torch.softmax(clean["position_logits"], dim=1) - torch.softmax(tensor_slice(environments[ENVIRONMENTS[0]], ids, device)["base_position"], dim=1)) ** 2)
            anchor = anchor + torch.mean((torch.softmax(clean["degree_logits"], dim=1) - torch.softmax(tensor_slice(environments[ENVIRONMENTS[0]], ids, device)["base_degree"], dim=1)) ** 2)
            residual = torch.mean(torch.stack([
                torch.mean(output["position_delta"] ** 2) + torch.mean(output["degree_delta"] ** 2)
                for output in outputs
            ]))
            loss = classification
            loss = loss + float(args.p4zf_vrex_weight) * vrex
            loss = loss + float(args.p4zf_consistency_weight) * consistency
            loss = loss + float(args.p4zf_anchor_weight) * anchor
            loss = loss + float(args.p4zf_residual_weight) * residual
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.p4zf_grad_clip))
            optimizer.step()
            epoch_values["loss"].append(float(loss.item()))
            epoch_values["classification"].append(float(classification.item()))
            epoch_values["vrex"].append(float(vrex.item()))
            epoch_values["consistency"].append(float(consistency.item()))
            epoch_values["anchor"].append(float(anchor.item()))
            epoch_values["residual"].append(float(residual.item()))
        if epoch >= int(args.p4zf_swa_start):
            average_count += 1
            state = model.state_dict()
            if average is None:
                average = {key: value.detach().cpu().clone() for key, value in state.items()}
            else:
                for key, value in state.items():
                    if torch.is_floating_point(value):
                        average[key].add_((value.detach().cpu() - average[key]) / float(average_count))
                    else:
                        average[key] = value.detach().cpu().clone()
        if epoch == 1 or epoch % int(args.p4zf_log_every) == 0 or epoch == int(args.p4zf_epochs):
            row = {"method_id": method_id, "seed": seed, "epoch": epoch}
            row.update({key: float(np.mean(value)) for key, value in epoch_values.items()})
            history.append(row)
            print("[P4ZF %s seed=%d ep=%03d] loss=%.4f cls=%.4f" % (
                method_id, seed, epoch, row["loss"], row["classification"],
            ), flush=True)
    if average is None:
        raise RuntimeError("SWA collected no state for %s" % method_id)
    model.load_state_dict(average, strict=True)
    model.eval()
    return model, history
