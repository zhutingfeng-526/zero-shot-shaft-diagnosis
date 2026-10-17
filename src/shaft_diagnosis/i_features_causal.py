"""Numerical causal functions."""
from __future__ import annotations
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple
import numpy as np

SOURCE_SPLITS = ("train_single", "dev_single", "final_single")


FS_HZ = 5000.0


R_MAX_SAMPLES = 512


MIN_LOCAL_VALID_SAMPLES = 1024


MINIMUM_C2_EVENTS = 12


PRIMARY_SCALE_INDEX = 1


@dataclass
class SourceFragment:
    split: str
    cond_int: int
    run_id: int
    fragment_id: int
    start_sample: int
    raw: np.ndarray
    truth: int
    rpm: int
    event_indices: np.ndarray

    @property
    def group_key(self) -> str:
        return "%s:C%03d:run%d:fragment%d" % (
            self.split, self.cond_int, self.run_id, self.fragment_id,
        )


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def array_hash(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    if array.dtype.hasobject or array.dtype.kind in ("U", "S"):
        payload = canonical_json([
            bytes(item).decode("utf-8", errors="surrogateescape")
            if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in array.reshape(-1).tolist()
        ]).encode("utf-8")
    else:
        payload = np.ascontiguousarray(array).tobytes(order="C")
    digest.update(payload)
    return digest.hexdigest()


def robust_center_scale(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    current = np.asarray(value, dtype=np.float64)
    center = np.median(current, axis=-1)
    scale = 1.4826 * np.median(np.abs(current - center[..., None]), axis=-1)
    fallback = np.std(current, axis=-1)
    scale = np.where(scale > 1.0e-7, scale, np.maximum(fallback, 1.0e-5))
    return center.astype(np.float32), scale.astype(np.float32)


def event_coordinate_masks(
    center_run: np.ndarray, width: np.ndarray, fragment_start: int,
    fragment_length: int, buffer_start: int, cutoff: int,
) -> Dict[str, np.ndarray]:
    run_center = np.asarray(center_run, dtype=np.int64)
    event_width = np.asarray(width, dtype=np.int64)
    if run_center.shape != event_width.shape:
        raise RuntimeError("Event center/width shape mismatch")
    center_fragment = run_center - int(fragment_start)
    in_fragment = (center_fragment >= 0) & (center_fragment < int(fragment_length))
    detector_close = center_fragment + np.maximum(event_width, 1)
    observed = (
        in_fragment
        & (detector_close <= int(cutoff))
        & (center_fragment >= int(buffer_start) - R_MAX_SAMPLES)
    )
    maturity = np.maximum(detector_close, center_fragment + R_MAX_SAMPLES)
    mature = (
        observed
        & (maturity <= int(cutoff))
        & (center_fragment - R_MAX_SAMPLES >= int(buffer_start))
    )
    return {
        "center_run": run_center,
        "center_fragment": center_fragment,
        "center_buffer": center_fragment - int(buffer_start),
        "event_width": event_width,
        "detector_close_fragment": detector_close,
        "maturity_fragment": maturity,
        "in_fragment": in_fragment,
        "observed": observed,
        "mature": mature,
        "expired": center_fragment + R_MAX_SAMPLES < int(buffer_start),
    }


def causal_process_nuisance(
    scores: np.ndarray, widths: np.ndarray, centers_fragment: np.ndarray,
    buffer_start: int, cutoff: int,
) -> np.ndarray:
    center = np.asarray(centers_fragment, dtype=np.int64)
    width = np.asarray(widths, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float32)
    if not (center.shape == width.shape == score.shape):
        raise RuntimeError("Causal process arrays are not aligned")
    if len(center) and np.any(np.diff(center) < 0):
        raise RuntimeError("Causal events must be ordered by fragment coordinate")
    previous = np.zeros(len(center), dtype=np.float32)
    following = np.zeros(len(center), dtype=np.float32)
    if len(center) > 1:
        intervals = np.diff(center).astype(np.float32)
        previous[1:] = intervals
        following[:-1] = intervals
    duration_seconds = max((int(cutoff) - int(buffer_start)) / FS_HZ, 1.0e-6)
    density = float(len(center) / duration_seconds)
    nuisance = np.zeros((len(center), 7), dtype=np.float32)
    nuisance[:, 0] = score
    nuisance[:, 2] = width.astype(np.float32)
    nuisance[:, 3] = previous
    nuisance[:, 4] = following
    nuisance[:, 5] = density
    nuisance[:, 6] = 1.0
    return nuisance


def detect_causal_prefix_events(
    modules: Mapping[str, Any], fragment: SourceFragment, start: int, end: int,
    detector_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    if not 0 <= int(start) < int(end) <= int(fragment.raw.shape[1]):
        raise RuntimeError("Invalid prefix detector bounds %d:%d/%d" % (start, end, fragment.raw.shape[1]))
    r1 = modules["r1"]
    prefix = r1.Fragment(
        cond_int=int(fragment.cond_int), run_id=int(fragment.run_id),
        fragment_id=int(fragment.fragment_id), start_sample=int(fragment.start_sample + int(start)),
        raw=np.asarray(fragment.raw[:, int(start):int(end)], dtype=np.float32),
        window_indices=np.zeros(0, dtype=np.int64),
        window_starts=np.zeros(0, dtype=np.int64),
    )
    detections = r1.cluster_unique_events(
        r1.detect_fragment(prefix, detector_contract["detector"], int(fragment.rpm), detector_contract["args"]),
        int(detector_contract["duplicate_tolerance_samples"]),
    )
    center_run = np.asarray([row.center for row in detections], dtype=np.int64)
    width = np.asarray([row.width for row in detections], dtype=np.int64)
    score = np.asarray([row.score for row in detections], dtype=np.float32)
    coordinates = event_coordinate_masks(
        center_run, width, fragment.start_sample, fragment.raw.shape[1], start, end,
    )
    if not bool(np.all(coordinates["in_fragment"])):
        raise RuntimeError("Prefix detector emitted an event outside its stitched fragment")
    observed_positions = np.where(coordinates["observed"])[0]
    mature_positions = np.where(coordinates["mature"])[0]
    observed_center_run = center_run[observed_positions]
    observed_center_fragment = coordinates["center_fragment"][observed_positions]
    observed_center_buffer = coordinates["center_buffer"][observed_positions]
    observed_width = width[observed_positions]
    observed_score = score[observed_positions]
    event_id = np.asarray([
        r1.stable_hash((int(fragment.cond_int), int(fragment.run_id), int(value))).encode("ascii")
        for value in observed_center_run.tolist()
    ], dtype="S64")
    nuisance = causal_process_nuisance(
        observed_score, observed_width, observed_center_fragment, start, end,
    )
    mature_ids = set(
        r1.stable_hash((int(fragment.cond_int), int(fragment.run_id), int(center_run[position]))).encode("ascii")
        for position in mature_positions.tolist()
    )
    mature_mask = np.asarray([value in mature_ids for value in event_id.tolist()], dtype=bool)
    return {
        "event_id": event_id,
        "center_run": observed_center_run,
        "center_fragment": observed_center_fragment,
        "center_buffer": observed_center_buffer,
        "event_width": observed_width,
        "event_score": observed_score,
        "nuisance": nuisance,
        "mature_mask": mature_mask,
        "report": {
            "candidate_events": int(len(detections)),
            "observed_events": int(len(observed_positions)),
            "mature_events": int(np.sum(mature_mask)),
            "expired_events": int(np.sum(coordinates["expired"])),
            "production_event_source": "frozen_R1_detector_rerun_on_causal_trailing_buffer",
            "detector_input": "trailing_buffer_only_[cutoff-T,cutoff]",
            "future_samples_accessed": 0,
            "detector_lock_hash": detector_contract["manifest"]["lock_hash"],
        },
    }


def causal_event_mask(
    modules: Mapping[str, Any], raw: np.ndarray, centers: np.ndarray, widths: np.ndarray,
) -> np.ndarray:
    r13 = modules["r13"]
    if len(centers):
        return np.asarray(r13.event_free_mask(raw.shape[1], centers, widths, raw), dtype=bool)
    try:
        detected = np.asarray(r13.p7a.robust_event_mask(np.asarray(raw, dtype=np.float32)), dtype=bool)
        if len(detected) == raw.shape[1]:
            return detected
    except Exception as exc:
        warnings.warn(
            "Fallback event-mask detection failed; returning an all-false mask. "
            "No events will be masked on this path. "
            "Original error: %s: %s" % (type(exc).__name__, exc),
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            "Fallback event-mask detection returned an unexpected length; "
            "returning an all-false mask. No events will be masked on this path.",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.zeros(raw.shape[1], dtype=bool)


def causal_normalize(
    modules: Mapping[str, Any], fragment_raw: np.ndarray, cutoff: int, start: int,
    observed_centers: np.ndarray, observed_widths: np.ndarray,
    n0: Mapping[str, np.ndarray], normalization_id: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if int(cutoff) > fragment_raw.shape[1] or int(start) < 0 or int(start) >= int(cutoff):
        raise RuntimeError("Invalid causal buffer bounds %d:%d/%d" % (start, cutoff, fragment_raw.shape[1]))
    raw = np.asarray(fragment_raw[:, int(start):int(cutoff)], dtype=np.float32)
    local_center = np.asarray(observed_centers, dtype=np.int64) - int(start)
    mask = causal_event_mask(modules, raw, local_center, observed_widths)
    valid = ~mask
    fallback = False
    if normalization_id == "N1_trailing_robust" and int(np.sum(valid)) >= MIN_LOCAL_VALID_SAMPLES:
        center, scale = robust_center_scale(raw[:, valid])
    else:
        center = np.asarray(n0["center"], dtype=np.float32)
        scale = np.asarray(n0["scale"], dtype=np.float32)
        fallback = normalization_id == "N1_trailing_robust"
    normalized = ((raw - center[:, None]) / scale[:, None]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise RuntimeError("Non-finite causal normalization")
    return normalized, {
        "normalization_id": normalization_id,
        "fallback_to_N0": bool(fallback),
        "valid_samples": int(np.sum(valid)),
        "buffer_samples": int(raw.shape[1]),
        "center_hash": array_hash(center),
        "scale_hash": array_hash(scale),
    }


def identity_record_normalize(raw: np.ndarray, excluded: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(raw, dtype=np.float32)
    return value, np.zeros(value.shape[0], dtype=np.float32), np.ones(value.shape[0], dtype=np.float32)


def extract_matrices_from_normalized(
    modules: Mapping[str, Any], normalized: np.ndarray, event_ids: np.ndarray,
    centers: np.ndarray, widths: np.ndarray, nuisance: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    owner = modules["r15"].r14.r13
    original = owner.robust_record_normalize
    owner.robust_record_normalize = identity_record_normalize
    try:
        matrices, report = modules["r15"].extract_fragment_matrices(
            normalized, event_ids, centers, widths, nuisance,
        )
    finally:
        owner.robust_record_normalize = original
    if owner.robust_record_normalize is not original:
        raise RuntimeError("Causal normalization monkey-patch was not restored")
    return matrices, report


def projectors_from_time_covariance(
    modules: Mapping[str, Any], covariance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r14 = modules["r14"]
    value = np.asarray(covariance, dtype=np.complex64)
    if value.ndim != 4 or value.shape[1:] != (4, 10, 10):
        raise RuntimeError("Unexpected causal time covariance shape: %s" % (value.shape,))
    n = len(value)
    v1 = np.zeros((n, 4, 6, 6), dtype=np.complex64)
    v2 = np.zeros((n, 4, 6, 6), dtype=np.complex64)
    a1 = np.zeros((n, 4, 2, 2), dtype=np.complex64)
    for event in range(n):
        for scale in range(4):
            vibration = 0.5 * (value[event, scale, :6, :6] + value[event, scale, :6, :6].conj().T)
            acoustic = 0.5 * (value[event, scale, 8:10, 8:10] + value[event, scale, 8:10, 8:10].conj().T)
            _ev, vector = np.linalg.eigh(vibration.astype(np.complex128))
            _ea, acoustic_vector = np.linalg.eigh(acoustic.astype(np.complex128))
            v1[event, scale] = r14.normalized_projector(vector[:, -1:], 1)
            v2[event, scale] = r14.normalized_projector(vector[:, -2:], 2)
            a1[event, scale] = r14.normalized_projector(acoustic_vector[:, -1:], 1)
    return v1, v2, a1


def mean_projector(modules: Mapping[str, Any], value: np.ndarray, rank: int) -> np.ndarray:
    r13 = modules["r13"]
    current = np.asarray(value, dtype=np.complex64)
    if current.ndim != 4 or not len(current):
        raise RuntimeError("Cannot aggregate empty event projectors")
    output = np.zeros(current.shape[1:], dtype=np.complex64)
    for scale in range(current.shape[1]):
        output[scale] = r13._safe_projector_mean(current[:, scale], int(rank))
    return output


def strength_from_covariance(covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(covariance)
    vibration = np.maximum(np.real(np.trace(value[:, :, :6, :6], axis1=2, axis2=3)) / 6.0, 0.0)
    acoustic = np.maximum(np.real(np.trace(value[:, :, 8:10, 8:10], axis1=2, axis2=3)) / 2.0, 0.0)
    return np.log1p(vibration).astype(np.float32), np.log1p(acoustic).astype(np.float32)


def cell_bundle(cell: Mapping[str, Any]) -> Dict[str, Any]:
    arrays = cell["arrays"]
    count = int(np.asarray(arrays["event_count"])[0])
    active = np.asarray(arrays["relation_valid"], dtype=bool)
    if len(active) != count:
        raise RuntimeError("Causal cell active/event count mismatch: %s" % cell["key"])
    return {
        "split": str(cell["split"]),
        "unit": "causal_finite_buffer",
        "keys": np.asarray([cell["key"]], dtype=object),
        "group_vibration_raw_rank1": np.asarray(arrays["group_vibration_raw_rank1"])[None],
        "group_vibration_raw_rank2": np.asarray(arrays["group_vibration_raw_rank2"])[None],
        "group_acoustic_raw_rank1": np.asarray(arrays["group_acoustic_raw_rank1"])[None],
        "group_valid": np.asarray([bool(np.asarray(arrays["ready"])[0])], dtype=bool),
        "event": {
            "group_index": np.zeros(count, dtype=np.int32),
            "active": active,
            "vibration_raw_rank1": np.asarray(arrays["vibration_raw_rank1"]),
            "vibration_raw_rank2": np.asarray(arrays["vibration_raw_rank2"]),
            "acoustic_raw_rank1": np.asarray(arrays["acoustic_raw_rank1"]),
            "jitter_active": active.copy(),
            "jitter_vibration_raw_rank1": np.asarray(arrays["vibration_raw_rank1"]),
            "jitter_vibration_raw_rank2": np.asarray(arrays["vibration_raw_rank2"]),
            "jitter_acoustic_raw_rank1": np.asarray(arrays["acoustic_raw_rank1"]),
            "event_id": np.asarray(arrays["event_id"]),
        },
        "center_sample": np.asarray(arrays["center_sample"], dtype=np.int64),
        "event_width": np.asarray(arrays["event_width"], dtype=np.float32),
        "acoustic_strength": np.asarray(arrays["acoustic_strength"], dtype=np.float32),
        "vibration_strength": np.asarray(arrays["vibration_strength"], dtype=np.float32),
        "truth": np.asarray(arrays["truth"], dtype=np.int64),
    }


def descriptor_from_cell(cell: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    arrays = cell["arrays"]
    ready = bool(np.asarray(arrays["ready"])[0])
    return {
        "keys": np.asarray([cell["key"]], dtype=object),
        "vibration_raw_rank1": np.asarray(arrays["group_vibration_raw_rank1"])[None],
        "acoustic_raw_rank1": np.asarray(arrays["group_acoustic_raw_rank1"])[None],
        "projective_active": np.asarray([ready], dtype=bool),
    }


def event_acoustic_margin(
    modules: Mapping[str, Any], bundle: Mapping[str, Any], acoustic_models: Mapping[str, Any],
    refractory_samples: int,
) -> Tuple[float, Dict[str, Any]]:
    if not bool(np.asarray(bundle["group_valid"])[0]):
        return float("nan"), {"event_count": 0, "active": 0}
    if set(acoustic_models) == set(SOURCE_SPLITS):
        bank = modules["r18"].ensemble_event_margin_bank(bundle, acoustic_models, "K0_clean")
    elif "all_source" in acoustic_models:
        bank = modules["r18"].event_margin_bank(bundle, acoustic_models["all_source"], "K0_clean")
    else:
        raise RuntimeError("Unknown acoustic model registry")
    summary = modules["r18"].build_record_summary(bundle, bank, int(refractory_samples))
    margin = float(np.asarray(summary["m01"]["side_acoustic"]["record"])[0, PRIMARY_SCALE_INDEX])
    return margin, {
        "event_count": int(np.asarray(summary["event_count"])[0]),
        "active": int(np.sum(np.asarray(bank["active"], dtype=bool))),
    }


def score_h1_cell(
    modules: Mapping[str, Any], cell: Mapping[str, Any], models: Mapping[str, Any],
) -> Dict[str, Any]:
    arrays = cell["arrays"]
    ready = bool(np.asarray(arrays["ready"])[0])
    count = int(np.asarray(arrays["event_count"])[0])
    if not ready:
        return {
            "ready": False, "event_count": count,
            "d2_probability": np.full(3, 1.0 / 3.0, dtype=np.float32),
            "h1_probability": np.full(3, 1.0 / 3.0, dtype=np.float32),
            "h1_prediction": -1, "endpoint_support": False, "orbit_support": False,
            "acoustic_margin": float("nan"), "stage": -1,
        }
    descriptor = descriptor_from_cell(cell)
    d2_probability, _distance = modules["r14"].r13b.raw_covariance_probability(descriptor, models["d2"])
    bundle = cell_bundle(cell)
    endpoint = modules["r16"].e2_score_bundle(bundle, models["endpoint"], "K0_clean")
    endpoint_support = bool(np.argmax(np.asarray(endpoint["record_probability"])[0]) == 2)
    feature = np.asarray(arrays["relation_feature"], dtype=np.float32)
    indices = np.asarray(modules["r15"].FEATURE_SETS["orbit"], dtype=np.int64)
    orbit_feature = feature[:, indices]
    process = np.asarray(arrays["process_feature"], dtype=np.float32)
    valid = np.asarray(arrays["relation_valid"], dtype=bool) & np.isfinite(orbit_feature).all(axis=1)
    if np.any(valid):
        event_score = modules["r16"].e1_score(models["orbit"], orbit_feature[valid], process[valid])
        orbit_delta = float(modules["r16"].robust_location(event_score))
        orbit_probability = modules["r16"].left_relation_probability(d2_probability, np.asarray([orbit_delta]))
        orbit_support = bool(np.argmax(orbit_probability[0]) == 1)
    else:
        orbit_delta = float("nan")
        orbit_support = False
    h1_probability, h1_prediction, stage = modules["r16"].apply_hierarchy(
        d2_probability, np.asarray([endpoint_support]), np.asarray([orbit_support]),
    )
    acoustic_margin, acoustic_report = event_acoustic_margin(
        modules, bundle, models["acoustic_models"], int(models["refractory_samples"]),
    )
    return {
        "ready": True,
        "event_count": count,
        "d2_probability": np.asarray(d2_probability[0], dtype=np.float32),
        "h1_probability": np.asarray(h1_probability[0], dtype=np.float32),
        "h1_prediction": int(h1_prediction[0]),
        "endpoint_support": endpoint_support,
        "orbit_support": orbit_support,
        "orbit_delta": orbit_delta,
        "acoustic_margin": acoustic_margin,
        "acoustic_report": acoustic_report,
        "stage": int(stage[0]),
    }


def apply_c2(
    modules: Mapping[str, Any], score: Mapping[str, Any], threshold: float,
) -> Tuple[np.ndarray, int, bool]:
    probability = np.asarray(score["h1_probability"], dtype=np.float32)[None]
    prediction = int(score["h1_prediction"])
    gate = bool(
        score["ready"] and prediction == 0 and int(score["event_count"]) >= MINIMUM_C2_EVENTS
        and np.isfinite(float(score["acoustic_margin"]))
        and float(score["acoustic_margin"]) > float(threshold)
    )
    updated, current = modules["r18"].promote_r0_to_r1(probability, np.asarray([gate]))
    return np.asarray(updated[0], dtype=np.float32), int(current[0]), gate


def decode_model_tree(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, Mapping) and set(value) == {"__array__"}:
        return np.asarray(arrays[str(value["__array__"])])
    if isinstance(value, Mapping) and set(value) == {"__tuple__"}:
        return tuple(decode_model_tree(item, arrays) for item in value["__tuple__"])
    if isinstance(value, Mapping):
        return {str(key): decode_model_tree(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_model_tree(item, arrays) for item in value]
    return value

def _bind(modules):
    globals().update(modules)
