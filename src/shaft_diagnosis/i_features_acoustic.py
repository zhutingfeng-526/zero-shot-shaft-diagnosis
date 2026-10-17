"""Numerical acoustic functions."""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Tuple
import numpy as np

SOURCE_SPLITS = ("train_single", "dev_single", "final_single")


TIME_SCALES_MS = (4, 12, 32, 80)


PRIMARY_SCALE = 1


def promote_r0_to_r1(probability: np.ndarray, gate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    output = normalize_probability(np.asarray(probability, dtype=np.float32).copy())
    base_prediction = np.argmax(output, axis=1)
    active = np.asarray(gate, dtype=bool) & (base_prediction == 0)
    left = output[active, 0].copy()
    output[active, 0] = output[active, 1]
    output[active, 1] = left
    output = normalize_probability(output)
    prediction = np.argmax(output, axis=1).astype(np.int16)
    checks = {
        "only_R0_changes": not bool(np.any((base_prediction != 0) & (prediction != base_prediction))),
        "gate_changes_to_R1": not bool(np.any(active & (prediction != 1))),
        "R2_probability_unchanged": bool(np.allclose(output[:, 2], probability[:, 2], atol=1.0e-7)),
    }
    if not all(checks.values()):
        raise RuntimeError("R18 bounded residual contract failed: %s" % checks)
    return output.astype(np.float32), prediction


def event_margin_bank(
    bundle: Mapping[str, Any],
    model: Mapping[str, Any],
    control: str = "K0_clean",
) -> Dict[str, Any]:
    views = r14.control_event_views(bundle, control)
    distances = r14.carrier_scale_distances(views, model)
    active = np.asarray(views["active"], dtype=bool)
    output: Dict[str, Any] = {"active": active}
    for field in ("side_acoustic", "side_full", "side_vibration"):
        current = np.asarray(distances[field], dtype=np.float32)
        output[field] = {
            "m01": (current[:, :, 0] - current[:, :, 1]).astype(np.float32),
            "m21": (current[:, :, 2] - current[:, :, 1]).astype(np.float32),
        }
    return output


def ensemble_event_margin_bank(
    bundle: Mapping[str, Any],
    models: Mapping[str, Any],
    control: str = "K0_clean",
) -> Dict[str, Any]:
    banks = [event_margin_bank(bundle, models[held], control) for held in SOURCE_SPLITS]
    output: Dict[str, Any] = {"active": np.logical_and.reduce([bank["active"] for bank in banks])}
    for field in ("side_acoustic", "side_full", "side_vibration"):
        output[field] = {}
        for margin in ("m01", "m21"):
            output[field][margin] = np.median(
                np.stack([bank[field][margin] for bank in banks], axis=0), axis=0
            ).astype(np.float32)
    return output


def _episode_groups(centers: np.ndarray, refractory: int) -> List[np.ndarray]:
    order = np.argsort(np.asarray(centers, dtype=np.int64), kind="mergesort")
    if not len(order):
        return []
    groups: List[List[int]] = [[int(order[0])]]
    ordered_centers = np.asarray(centers, dtype=np.int64)[order]
    for position in range(1, len(order)):
        if int(ordered_centers[position] - ordered_centers[position - 1]) <= int(refractory):
            groups[-1].append(int(order[position]))
        else:
            groups.append([int(order[position])])
    return [np.asarray(group, dtype=np.int64) for group in groups]


def summarize_margin(
    values: np.ndarray,
    active: np.ndarray,
    group_index: np.ndarray,
    centers: np.ndarray,
    strength: np.ndarray,
    n_groups: int,
    refractory: int,
) -> Dict[str, Any]:
    score = np.asarray(values, dtype=np.float64)
    if score.ndim != 2 or score.shape[1] != len(TIME_SCALES_MS):
        raise RuntimeError("R18 event-margin shape mismatch: %s" % (score.shape,))
    shape = (n_groups, len(TIME_SCALES_MS))
    output: Dict[str, Any] = {
        "record": np.full(shape, np.nan, dtype=np.float32),
        "q25": np.full(shape, np.nan, dtype=np.float32),
        "q75": np.full(shape, np.nan, dtype=np.float32),
        "event_mad": np.full(shape, np.nan, dtype=np.float32),
        "block_q25": np.full(shape, np.nan, dtype=np.float32),
        "block_mad": np.full(shape, np.nan, dtype=np.float32),
        "strength_balanced": np.full(shape, np.nan, dtype=np.float32),
        "episode": np.full(shape, np.nan, dtype=np.float32),
        "strength_strata": np.full((n_groups, len(TIME_SCALES_MS), 3), np.nan, dtype=np.float32),
        "event_count": np.zeros(n_groups, dtype=np.int32),
        "episode_count": np.zeros(n_groups, dtype=np.int32),
        "block_count": np.zeros(n_groups, dtype=np.int16),
        "event_groups": [],
    }
    active_value = np.asarray(active, dtype=bool)
    groups = np.asarray(group_index, dtype=np.int64)
    center_value = np.asarray(centers, dtype=np.int64)
    strength_value = np.asarray(strength, dtype=np.float64)
    if strength_value.ndim == 2:
        strength_scalar = strength_value[:, PRIMARY_SCALE]
    else:
        strength_scalar = strength_value
    for group in range(n_groups):
        ids = np.where(active_value & (groups == group) & np.all(np.isfinite(score), axis=1))[0]
        ids = ids[np.argsort(center_value[ids], kind="mergesort")]
        output["event_groups"].append((ids.copy(), score[ids].astype(np.float32)))
        output["event_count"][group] = int(len(ids))
        if not len(ids):
            continue
        current = score[ids]
        output["record"][group] = np.median(current, axis=0)
        output["q25"][group] = np.quantile(current, 0.25, axis=0)
        output["q75"][group] = np.quantile(current, 0.75, axis=0)
        output["event_mad"][group] = 1.4826 * np.median(
            np.abs(current - np.median(current, axis=0, keepdims=True)), axis=0
        )
        block_count = min(8, max(3, int(len(ids) // 8)))
        blocks = [part for part in np.array_split(np.arange(len(ids)), block_count) if len(part)]
        block_values = np.stack([np.median(current[part], axis=0) for part in blocks], axis=0)
        output["block_q25"][group] = np.quantile(block_values, 0.25, axis=0)
        output["block_mad"][group] = 1.4826 * np.median(
            np.abs(block_values - np.median(block_values, axis=0, keepdims=True)), axis=0
        )
        output["block_count"][group] = int(len(blocks))
        strength_order = np.argsort(strength_scalar[ids], kind="mergesort")
        strata = [part for part in np.array_split(strength_order, 3) if len(part)]
        stratum_values = np.stack([np.median(current[part], axis=0) for part in strata], axis=0)
        output["strength_strata"][group, :, :len(strata)] = stratum_values.T
        output["strength_balanced"][group] = np.median(stratum_values, axis=0)
        episodes = _episode_groups(center_value[ids], int(refractory))
        episode_values = np.stack([np.median(current[part], axis=0) for part in episodes], axis=0)
        output["episode"][group] = np.median(episode_values, axis=0)
        output["episode_count"][group] = int(len(episodes))
    return output


def build_record_summary(
    bundle: Mapping[str, Any], bank: Mapping[str, Any], refractory: int
) -> Dict[str, Any]:
    group_index = np.asarray(bundle["event"]["group_index"], dtype=np.int64)
    centers = np.asarray(bundle["center_sample"], dtype=np.int64)
    n_groups = len(bundle["keys"])
    output: Dict[str, Any] = {"keys": np.asarray(bundle["keys"]).astype(object)}
    for margin in ("m01", "m21"):
        output[margin] = {}
        for field in ("side_acoustic", "side_full", "side_vibration"):
            strength = bundle["acoustic_strength"] if field != "side_vibration" else bundle["vibration_strength"]
            output[margin][field] = summarize_margin(
                bank[field][margin], bank["active"], group_index, centers,
                strength, n_groups, refractory,
            )
    acoustic_strength = np.asarray(bundle["acoustic_strength"], dtype=np.float64)
    acoustic_scalar = acoustic_strength[:, PRIMARY_SCALE] if acoustic_strength.ndim == 2 else acoustic_strength
    active = np.asarray(bank["active"], dtype=bool)
    median_strength = np.full(n_groups, np.nan, dtype=np.float32)
    for group in range(n_groups):
        ids = np.where(active & (group_index == group) & np.isfinite(acoustic_scalar))[0]
        if len(ids):
            median_strength[group] = float(np.median(acoustic_scalar[ids]))
    output["median_acoustic_strength"] = median_strength
    output["event_count"] = np.asarray(output["m01"]["side_acoustic"]["event_count"], dtype=np.int32)
    return output

def _bind(modules):
    globals().update(modules)
    global normalize_probability
    normalize_probability = r16.normalize_probability
