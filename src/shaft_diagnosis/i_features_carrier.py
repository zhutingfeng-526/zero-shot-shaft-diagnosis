"""Numerical carrier functions."""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np

R_CLASSES = 3


TIME_SCALES_MS = (4, 12, 32, 80)


EPS = 1.0e-8


STATION_CHANNELS: Dict[str, Tuple[int, int]] = {
    "B1": (0, 1),
    "B2": (2, 3),
    "B3": (4, 5),
}


def normalize_probability(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 1.0e-12)
    array /= np.maximum(np.sum(array, axis=-1, keepdims=True), 1.0e-12)
    return array.astype(np.float32)


def normalized_projector(vectors: np.ndarray, rank: int) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.complex128)
    if value.ndim == 1:
        value = value[:, None]
    q, _unused = np.linalg.qr(value)
    q = q[:, : int(rank)]
    return ((q.dot(q.conj().T) + q.dot(q.conj().T).conj().T) * 0.5).astype(np.complex64)


def batch_block_distance(
    samples: np.ndarray, prototypes: np.ndarray, indices: Sequence[int], rank: int
) -> np.ndarray:
    """Vectorized equivalent of block_distance for [event, scale] x [class, scale]."""
    left = np.asarray(samples, dtype=np.complex128)
    right = np.asarray(prototypes, dtype=np.complex128)
    ids = np.asarray(indices, dtype=np.int64)
    a = np.take(np.take(left, ids, axis=-2), ids, axis=-1)[:, :, None, :, :]
    b = np.transpose(
        np.take(np.take(right, ids, axis=-2), ids, axis=-1), (1, 0, 2, 3)
    )[None, :, :, :, :]
    trace_a = np.real(np.trace(a, axis1=-2, axis2=-1))
    trace_b = np.real(np.trace(b, axis1=-2, axis2=-1))
    mass_a = trace_a / max(float(rank), 1.0)
    mass_b = trace_b / max(float(rank), 1.0)
    a_shape = a / np.maximum(trace_a, EPS)[..., None, None]
    b_shape = b / np.maximum(trace_b, EPS)[..., None, None]
    norm_a = np.sqrt(np.sum(np.abs(a_shape) ** 2, axis=(-2, -1)))
    norm_b = np.sqrt(np.sum(np.abs(b_shape) ** 2, axis=(-2, -1)))
    shape = np.sqrt(np.sum(np.abs(a_shape - b_shape) ** 2, axis=(-2, -1)))
    shape /= np.maximum(norm_a + norm_b, EPS)
    mass = np.abs(mass_a - mass_b)
    return np.clip(0.65 * shape + 0.35 * mass, 0.0, 1.0).astype(np.float32)


def batch_cross_block_distance(
    samples: np.ndarray, prototypes: np.ndarray,
    first: Sequence[int], second: Sequence[int],
) -> np.ndarray:
    """Vectorized equivalent of cross_block_distance."""
    left = np.asarray(samples, dtype=np.complex128)
    right = np.asarray(prototypes, dtype=np.complex128)
    first_ids = np.asarray(first, dtype=np.int64)
    second_ids = np.asarray(second, dtype=np.int64)
    a = np.take(np.take(left, first_ids, axis=-2), second_ids, axis=-1)[:, :, None, :, :]
    b = np.transpose(
        np.take(np.take(right, first_ids, axis=-2), second_ids, axis=-1), (1, 0, 2, 3)
    )[None, :, :, :, :]
    norm_a = np.sqrt(np.sum(np.abs(a) ** 2, axis=(-2, -1)))
    norm_b = np.sqrt(np.sum(np.abs(b) ** 2, axis=(-2, -1)))
    cosine = np.real(np.sum(np.conj(a) * b, axis=(-2, -1)))
    cosine /= np.maximum(norm_a * norm_b, EPS)
    angular = 0.5 * (1.0 - np.clip(cosine, -1.0, 1.0))
    magnitude = np.abs(norm_a - norm_b) / np.maximum(norm_a + norm_b, EPS)
    distance = np.clip(0.70 * angular + 0.30 * magnitude, 0.0, 1.0)
    both_zero = (norm_a <= EPS) & (norm_b <= EPS)
    return np.where(both_zero, 0.0, distance).astype(np.float32)


def batch_projector_distance(
    samples: np.ndarray, prototypes: np.ndarray, rank: int
) -> np.ndarray:
    """Vectorized equivalent of projector_distance."""
    left = np.asarray(samples, dtype=np.complex128)[:, :, None, :, :]
    right = np.transpose(np.asarray(prototypes, dtype=np.complex128), (1, 0, 2, 3))[
        None, :, :, :, :
    ]
    overlap = np.real(np.sum(left * np.swapaxes(right, -1, -2), axis=(-2, -1)))
    return np.clip(1.0 - overlap / max(float(rank), 1.0), 0.0, 1.0).astype(np.float32)


def robust_location(values: np.ndarray) -> float:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    if not len(current):
        return 0.0
    median = float(np.median(current))
    scale = 1.4826 * float(np.median(np.abs(current - median)))
    if scale <= 1.0e-7:
        return float(np.mean(current))
    clipped = np.clip(current, median - 2.5 * scale, median + 2.5 * scale)
    return float(np.mean(clipped))


def _mean_projectors(values: Sequence[np.ndarray], rank: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex64)
    if array.ndim != 3 or not len(array):
        raise RuntimeError("R14 projector prototype has no records")
    matrix = np.mean(array.astype(np.complex128), axis=0)
    matrix = 0.5 * (matrix + matrix.conj().T)
    _eigenvalue, eigenvector = np.linalg.eigh(matrix)
    return normalized_projector(eigenvector[:, -int(rank):], int(rank))


def fit_carrier_prototypes(
    source_bundles: Mapping[str, Mapping[str, Any]], included_splits: Sequence[str]
) -> Dict[str, Any]:
    prototypes = {
        "vibration_raw_rank1": np.zeros((R_CLASSES, len(TIME_SCALES_MS), 6, 6), dtype=np.complex64),
        "vibration_raw_rank2": np.zeros((R_CLASSES, len(TIME_SCALES_MS), 6, 6), dtype=np.complex64),
        "acoustic_raw_rank1": np.zeros((R_CLASSES, len(TIME_SCALES_MS), 2, 2), dtype=np.complex64),
    }
    record_rows: List[Dict[str, Any]] = []
    for label in range(R_CLASSES):
        records: List[Tuple[str, int]] = []
        for split in included_splits:
            bundle = source_bundles[split]
            ids = np.where(np.asarray(bundle["truth"], dtype=np.int64) == int(label))[0]
            records.extend((split, int(index)) for index in ids.tolist() if bool(bundle["group_valid"][index]))
        if not records:
            raise RuntimeError("R14 source prototype class missing: %d" % label)
        for scale_position in range(len(TIME_SCALES_MS)):
            prototypes["vibration_raw_rank1"][label, scale_position] = _mean_projectors(
                [source_bundles[split]["group_vibration_raw_rank1"][index, scale_position] for split, index in records], 1
            )
            prototypes["vibration_raw_rank2"][label, scale_position] = _mean_projectors(
                [source_bundles[split]["group_vibration_raw_rank2"][index, scale_position] for split, index in records], 2
            )
            prototypes["acoustic_raw_rank1"][label, scale_position] = _mean_projectors(
                [source_bundles[split]["group_acoustic_raw_rank1"][index, scale_position] for split, index in records], 1
            )
        record_rows.append({"R_position": int(label), "records": int(len(records))})
    return {
        "prototypes": prototypes,
        "included_splits": list(included_splits),
        "record_rows": record_rows,
        "target_labels_accessed": False,
        "target_parameters_fitted": False,
    }


def similarity_transform_batch(projector: np.ndarray, transform: np.ndarray) -> np.ndarray:
    value = np.asarray(projector, dtype=np.complex128)
    matrix = np.asarray(transform, dtype=np.complex128)
    output = np.matmul(matrix, value)
    output = np.matmul(output, matrix.conj().T)
    return output.astype(np.complex64)


def control_event_views(bundle: Mapping[str, Any], control: str) -> Dict[str, np.ndarray]:
    event = bundle["event"]
    if control == "K6_nonimpact_center_jitter":
        return {
            "vibration_raw_rank1": np.asarray(event["jitter_vibration_raw_rank1"], dtype=np.complex64),
            "vibration_raw_rank2": np.asarray(event["jitter_vibration_raw_rank2"], dtype=np.complex64),
            "acoustic_raw_rank1": np.asarray(event["jitter_acoustic_raw_rank1"], dtype=np.complex64),
            "active": np.asarray(event["jitter_active"], dtype=bool),
        }
    vibration_rank1 = np.asarray(event["vibration_raw_rank1"], dtype=np.complex64).copy()
    vibration_rank2 = np.asarray(event["vibration_raw_rank2"], dtype=np.complex64).copy()
    acoustic_rank1 = np.asarray(event["acoustic_raw_rank1"], dtype=np.complex64).copy()
    active = np.asarray(event["active"], dtype=bool).copy()
    if control == "K1_B1_B3_vibration_swap":
        transform = np.eye(6, dtype=np.float64)[[4, 5, 2, 3, 0, 1]]
        vibration_rank1 = similarity_transform_batch(vibration_rank1, transform)
        vibration_rank2 = similarity_transform_batch(vibration_rank2, transform)
    elif control == "K2_B2_deterministic_permutation":
        left = np.eye(6, dtype=np.float64)[[2, 3, 0, 1, 4, 5]]
        right = np.eye(6, dtype=np.float64)[[0, 1, 4, 5, 2, 3]]
        even = np.arange(len(vibration_rank1)) % 2 == 0
        vibration_rank1[even] = similarity_transform_batch(vibration_rank1[even], left)
        vibration_rank2[even] = similarity_transform_batch(vibration_rank2[even], left)
        vibration_rank1[~even] = similarity_transform_batch(vibration_rank1[~even], right)
        vibration_rank2[~even] = similarity_transform_batch(vibration_rank2[~even], right)
    elif control == "K3_acoustic_A_B_swap":
        transform = np.eye(2, dtype=np.float64)[[1, 0]]
        acoustic_rank1 = similarity_transform_batch(acoustic_rank1, transform)
    elif control == "K7_record_acoustic_pairing_shuffle":
        if len(acoustic_rank1) > 1:
            shift = max(1, len(acoustic_rank1) // 3)
            acoustic_rank1 = np.roll(acoustic_rank1, shift=shift, axis=0)
    elif control in ("K0_clean", "K4_acoustic_joint_sign_flip", "K5_common_scale_0p5"):
        # Projectors are exactly invariant to a joint sign flip or common positive scaling.
        pass
    else:
        raise RuntimeError("Unknown R14 control: %s" % control)
    return {
        "vibration_raw_rank1": vibration_rank1,
        "vibration_raw_rank2": vibration_rank2,
        "acoustic_raw_rank1": acoustic_rank1,
        "active": active,
    }


def carrier_scale_distances(
    views: Mapping[str, np.ndarray], model: Mapping[str, Any]
) -> Dict[str, np.ndarray]:
    vibration_rank1 = np.asarray(views["vibration_raw_rank1"])
    vibration_rank2 = np.asarray(views["vibration_raw_rank2"])
    acoustic_rank1 = np.asarray(views["acoustic_raw_rank1"])
    prototypes = model["prototypes"]
    n = len(vibration_rank1)
    shape = (n, len(TIME_SCALES_MS), R_CLASSES)
    center_rank1 = np.zeros(shape, dtype=np.float32)
    center_rank2 = np.zeros(shape, dtype=np.float32)
    side_full = np.zeros(shape, dtype=np.float32)
    side_vibration = np.zeros(shape, dtype=np.float32)
    side_acoustic = np.zeros(shape, dtype=np.float32)
    b1 = STATION_CHANNELS["B1"]
    b2 = STATION_CHANNELS["B2"]
    b3 = STATION_CHANNELS["B3"]
    # Chunking bounds temporary broadcast tensors while preserving every
    # scalar formula used by the original implementation.
    for start in range(0, n, 2048):
        stop = min(n, start + 2048)
        sample_r1 = vibration_rank1[start:stop]
        sample_r2 = vibration_rank2[start:stop]
        prototype_r1 = prototypes["vibration_raw_rank1"]
        prototype_r2 = prototypes["vibration_raw_rank2"]
        center_rank1[start:stop] = (
            0.40 * batch_block_distance(sample_r1, prototype_r1, b2, 1)
            + 0.30 * batch_cross_block_distance(sample_r1, prototype_r1, b2, b1)
            + 0.30 * batch_cross_block_distance(sample_r1, prototype_r1, b2, b3)
        )
        center_rank2[start:stop] = (
            0.40 * batch_block_distance(sample_r2, prototype_r2, b2, 2)
            + 0.30 * batch_cross_block_distance(sample_r2, prototype_r2, b2, b1)
            + 0.30 * batch_cross_block_distance(sample_r2, prototype_r2, b2, b3)
        )
        vibration_distance = (
            0.25 * batch_block_distance(sample_r1, prototype_r1, b1, 1)
            + 0.25 * batch_block_distance(sample_r1, prototype_r1, b3, 1)
            + 0.20 * batch_cross_block_distance(sample_r1, prototype_r1, b1, b3)
        ) / 0.70
        acoustic_distance = batch_projector_distance(
            acoustic_rank1[start:stop], prototypes["acoustic_raw_rank1"], 1
        )
        side_vibration[start:stop] = vibration_distance
        side_acoustic[start:stop] = acoustic_distance
        side_full[start:stop] = 0.55 * vibration_distance + 0.45 * acoustic_distance
    return {
        "center_rank1": center_rank1,
        "center_rank12": 0.5 * center_rank1 + 0.5 * center_rank2,
        "side_full": side_full,
        "side_vibration": side_vibration,
        "side_acoustic": side_acoustic,
    }


def group_scale_scores(
    bundle: Mapping[str, Any], distances: Mapping[str, np.ndarray], center_field: str, side_field: str
) -> Tuple[np.ndarray, np.ndarray]:
    group_index = np.asarray(bundle["event"]["group_index"], dtype=np.int64)
    active = np.asarray(distances.get("active", bundle["event"]["active"]), dtype=bool)
    center_distance = np.asarray(distances[center_field], dtype=np.float64)
    side_distance = np.asarray(distances[side_field], dtype=np.float64)
    n_groups = len(bundle["keys"])
    center = np.zeros((n_groups, len(TIME_SCALES_MS)), dtype=np.float32)
    side = np.zeros((n_groups, len(TIME_SCALES_MS)), dtype=np.float32)
    for group in range(n_groups):
        ids = np.where(active & (group_index == group))[0]
        if not len(ids):
            continue
        for scale_position in range(len(TIME_SCALES_MS)):
            center_event = (
                np.minimum(center_distance[ids, scale_position, 0], center_distance[ids, scale_position, 2])
                - center_distance[ids, scale_position, 1]
            )
            side_event = side_distance[ids, scale_position, 2] - side_distance[ids, scale_position, 0]
            center[group, scale_position] = robust_location(center_event)
            side[group, scale_position] = robust_location(side_event)
    return center, side


def factorized_probability(
    center_score: np.ndarray, side_score: np.ndarray, center_temperature: float, side_temperature: float
) -> np.ndarray:
    center = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(center_score) / max(float(center_temperature), 1.0e-3), -60.0, 60.0)))
    side = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(side_score) / max(float(side_temperature), 1.0e-3), -60.0, 60.0)))
    probability = np.column_stack(((1.0 - center) * side, center, (1.0 - center) * (1.0 - side)))
    return normalize_probability(probability)


def aggregate_variant_scores(
    center_scale: np.ndarray,
    side_scale: np.ndarray,
    center_weight: np.ndarray,
    side_weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.sum(np.asarray(center_scale, dtype=np.float64) * np.asarray(center_weight)[None, :], axis=1),
        np.sum(np.asarray(side_scale, dtype=np.float64) * np.asarray(side_weight)[None, :], axis=1),
    )

def _bind(modules):
    globals().update(modules)
