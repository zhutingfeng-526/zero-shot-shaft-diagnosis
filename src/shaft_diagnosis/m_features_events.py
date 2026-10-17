"""Phase-residual event states and per-rotation M response descriptors."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np

import argparse
NODE_CHANNELS = {"B1": (0, 1), "B2": (2, 3), "B3": (4, 5), "O12": (6, 7), "A1": (8,), "A3": (9,)}



def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    width = max(1, min(int(width), x.shape[-1]))
    left, right = width // 2, width - 1 - width // 2
    padded = np.pad(x, ((0, 0), (left, right)), mode="reflect")
    csum = np.cumsum(padded, axis=1, dtype=np.float64)
    csum = np.concatenate((np.zeros((len(x), 1)), csum), axis=1)
    return (csum[:, width:] - csum[:, :-width]) / float(width)


def node_event_trace(raw: np.ndarray, channel_scale: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64)
    z = (x - moving_average(x, 251)) / np.maximum(np.asarray(channel_scale)[:, None], 1.0e-8)
    traces = []
    for width in (5, 17, 49):
        energy = np.sqrt(np.maximum(moving_average(z * z, width), 0.0))
        node_values = np.stack([
            np.sqrt(np.mean(energy[np.asarray(ids)] ** 2, axis=0))
            for ids in NODE_CHANNELS.values()
        ], axis=0)
        # Median prevents one noisy sensor from defining all events; max keeps local impacts visible.
        traces.append(0.55 * np.max(node_values, axis=0) + 0.45 * np.median(node_values, axis=0))
    return np.max(np.stack(traces, axis=0), axis=0)


def close_binary(mask: np.ndarray, gap: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    true_ids = np.where(result)[0]
    if len(true_ids) < 2:
        return result
    for left, right in zip(true_ids[:-1], true_ids[1:]):
        if 0 < int(right - left - 1) <= int(gap):
            result[left:right + 1] = True
    return result


def connected_intervals(mask: np.ndarray, minimum: int) -> List[Tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    delta = np.diff(padded)
    starts, ends = np.where(delta == 1)[0], np.where(delta == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if int(b - a) >= int(minimum)]


FS_HZ = 5000.0


def samples_per_rotation(rpm: int, fs_hz: float = FS_HZ) -> int:
    if int(rpm) <= 0:
        raise ValueError("rpm must be positive")
    return max(8, int(round(float(fs_hz) * 60.0 / float(rpm))))


def rotation_matrix(raw: np.ndarray, rpm: int, offset: int = 0) -> Tuple[np.ndarray, int, int]:
    x = np.asarray(raw, dtype=np.float64)
    spr = samples_per_rotation(rpm)
    offset = int(offset) % spr
    usable = ((x.shape[1] - offset) // spr) * spr
    if usable < 3 * spr:
        raise RuntimeError("At least three complete rotations are required")
    matrix = x[:, offset:offset + usable].reshape(x.shape[0], -1, spr)
    return matrix, spr, offset


def phase_template(raw: np.ndarray, rpm: int, valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    matrix, spr, offset = rotation_matrix(raw, rpm)
    rotations = matrix.shape[1]
    if valid_mask is None:
        valid = np.ones((rotations, spr), dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)[offset:offset + rotations * spr].reshape(rotations, spr)
        valid = ~mask
    initial = np.median(matrix, axis=1)
    template = initial.copy()
    valid_counts = np.sum(valid, axis=0)
    for phase in range(spr):
        ids = np.where(valid[:, phase])[0]
        if len(ids) >= max(3, int(math.ceil(0.25 * rotations))):
            template[:, phase] = np.median(matrix[:, ids, phase], axis=1)
    tiled = np.tile(template, rotations)
    if offset:
        prefix = np.asarray(raw, dtype=np.float64)[:, :offset]
        suffix = np.asarray(raw, dtype=np.float64)[:, offset + rotations * spr:]
        tiled = np.concatenate((prefix, tiled, suffix), axis=1)
    elif rotations * spr < raw.shape[1]:
        tiled = np.concatenate((tiled, np.asarray(raw, dtype=np.float64)[:, rotations * spr:]), axis=1)
    return tiled[:, :raw.shape[1]], {
        "samples_per_rotation": spr,
        "rotation_n": rotations,
        "phase_valid_fraction_min": float(np.min(valid_counts) / float(rotations)),
        "phase_valid_fraction_mean": float(np.mean(valid_counts) / float(rotations)),
    }


def phase_residual(raw: np.ndarray, rpm: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    template, diagnostics = phase_template(raw, rpm)
    return np.asarray(raw, dtype=np.float64) - template, template, diagnostics


def detector_trace(raw: np.ndarray, rpm: int, channel_scale: np.ndarray, mode: str) -> np.ndarray:
    if mode == "phase_residual":
        value, _template, _diag = phase_residual(raw, rpm)
    elif mode == "raw_multiscale":
        value = np.asarray(raw, dtype=np.float64)
    else:
        raise ValueError("Unknown detector mode: %s" % mode)
    return node_event_trace(value, np.asarray(channel_scale, dtype=np.float64))


def trace_states(trace: np.ndarray, rpm: int, detector: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    values = np.asarray(trace, dtype=np.float64)
    hard = close_binary(values >= float(detector["hard_threshold"]), int(args.event_close_gap))
    intervals = connected_intervals(hard, int(args.event_min_samples))
    core = np.zeros(len(values), dtype=bool)
    expanded: List[Tuple[int, int]] = []
    for start, end in intervals:
        left = max(0, start - int(args.core_expand)); right = min(len(values), end + int(args.core_expand))
        core[left:right] = True; expanded.append((left, right))
    envelope = moving_average(values[None, :], int(args.tail_smooth_samples))[0]
    tail = np.zeros(len(values), dtype=bool)
    max_tail = int(round(float(args.tail_max_rotations) * samples_per_rotation(rpm)))
    settle = int(args.tail_settle_samples)
    for _start, end in expanded:
        limit = min(len(values), end + max_tail)
        stop = limit
        low_run = 0
        for index in range(end, limit):
            if envelope[index] < float(detector["tail_threshold"]):
                low_run += 1
                if low_run >= settle:
                    stop = index - settle + 1
                    break
            else:
                low_run = 0
        tail[end:stop] = True
    tail &= ~core
    uncertain = (values >= float(detector["tail_threshold"])) & ~(core | tail)
    states = np.zeros(len(values), dtype=np.int8)
    states[core] = 1; states[tail] = 2; states[uncertain] = 3
    return {
        "core": core, "tail": tail, "uncertain": uncertain, "states": states,
        "intervals": expanded, "event_count": int(len(expanded)),
        "core_coverage": float(np.mean(core)), "tail_coverage": float(np.mean(tail)),
        "uncertain_coverage": float(np.mean(uncertain)), "trace": values,
    }


ORDERS = (1, 2, 3)


SENSOR_CONTRACT = (
    {"channel": 0, "id": "B1_x", "modality": "vibration", "location": "B1"},
    {"channel": 1, "id": "B1_y", "modality": "vibration", "location": "B1"},
    {"channel": 2, "id": "B2_x", "modality": "vibration", "location": "B2"},
    {"channel": 3, "id": "B2_y", "modality": "vibration", "location": "B2"},
    {"channel": 4, "id": "B3_x", "modality": "vibration", "location": "B3"},
    {"channel": 5, "id": "B3_y", "modality": "vibration", "location": "B3"},
    {"channel": 6, "id": "orbit_x", "modality": "shaft_orbit", "location": "B1-B2"},
    {"channel": 7, "id": "orbit_y", "modality": "shaft_orbit", "location": "B1-B2"},
    {"channel": 8, "id": "sound_A", "modality": "acoustic", "location": "B1"},
    {"channel": 9, "id": "sound_B", "modality": "acoustic", "location": "B3"},
)


def intervals_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    value = np.asarray(mask, dtype=bool)
    padded = np.concatenate((np.asarray([False]), value, np.asarray([False])))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(int(left), int(right)) for left, right in zip(starts, ends)]


def close_mask(mask: np.ndarray, gap: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool).copy()
    intervals = intervals_from_mask(value)
    for (_left, right), (next_left, _next_right) in zip(intervals[:-1], intervals[1:]):
        if next_left - right <= int(gap):
            value[right:next_left] = True
    return value


def detector_state_contract(
    raw: np.ndarray,
    rpm: int,
    raw_detector: Mapping[str, Any],
    phase_detector: Mapping[str, Any],
    selected_detector: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    detectors = (raw_detector, phase_detector)
    states: Dict[str, Dict[str, Any]] = {}
    traces: Dict[str, np.ndarray] = {}
    for detector in detectors:
        trace = detector_trace(
            raw, rpm, np.asarray(detector["channel_scale"]), str(detector["mode"]),
        )
        traces[str(detector["id"])] = np.asarray(trace, dtype=np.float64)
        states[str(detector["id"])] = trace_states(trace, rpm, detector, args)
    e1 = states[str(raw_detector["id"])]
    e2 = states[str(phase_detector["id"])]
    selected = states[str(selected_detector["id"])]
    e1_event = np.asarray(e1["core"]) | np.asarray(e1["tail"])
    e2_event = np.asarray(e2["core"]) | np.asarray(e2["tail"])
    high_core = close_mask(np.asarray(e1["core"]) & np.asarray(e2["core"]), int(args.event_close_gap))
    high_event = high_core | (np.asarray(e1["tail"]) & np.asarray(e2["tail"]))
    union_event = (
        e1_event | e2_event | np.asarray(e1["uncertain"]) | np.asarray(e2["uncertain"])
    )
    uncertain = union_event & ~high_event
    anchors = intervals_from_mask(high_core)
    selected_intervals = [(int(left), int(right)) for left, right in selected["intervals"]]
    if not anchors:
        anchors = selected_intervals
    selected_trace = traces[str(selected_detector["id"])]
    event_widths = [right - left for left, right in anchors]
    return {
        "e1": e1,
        "e2": e2,
        "selected": selected,
        "high_core": high_core,
        "high_event": high_event,
        "strict_exclusion": union_event,
        "uncertain": uncertain,
        "anchors": anchors,
        "selected_intervals": selected_intervals,
        "selected_trace": selected_trace,
        "event_count": int(len(anchors)),
        "trace_peak": float(np.max(selected_trace)),
        "trace_q99": float(np.quantile(selected_trace, 0.99)),
        "event_width_mean": float(np.mean(event_widths)) if event_widths else 0.0,
        "e1_e2_iou": float(np.sum(e1_event & e2_event) / max(1, np.sum(e1_event | e2_event))),
        "strict_exclusion_coverage": float(np.mean(union_event)),
    }


def rotation_state_rows(contract: Mapping[str, Any], rpm: int, sample_n: int) -> List[Dict[str, Any]]:
    spr = int(samples_per_rotation(int(rpm)))
    rotation_n = int(sample_n // spr)
    anchors = list(contract["anchors"])
    strict = np.asarray(contract["strict_exclusion"], dtype=bool)
    high_core = np.asarray(contract["high_core"], dtype=bool)
    high_event = np.asarray(contract["high_event"], dtype=bool)
    uncertain = np.asarray(contract["uncertain"], dtype=bool)
    rows: List[Dict[str, Any]] = []
    for rotation in range(rotation_n):
        left, right = rotation * spr, (rotation + 1) * spr
        overlap_exclusion = bool(np.any(strict[left:right]))
        overlap_core = bool(np.any(high_core[left:right]))
        overlap_tail = bool(np.any(high_event[left:right] & ~high_core[left:right]))
        overlap_uncertain = bool(np.any(uncertain[left:right]))
        previous = [end for start, end in anchors if end <= left]
        upcoming = [start for start, end in anchors if start >= right]
        d_prev = float((left - max(previous)) / float(spr)) if previous else float("inf")
        d_next = float((min(upcoming) - right) / float(spr)) if upcoming else float("inf")
        if overlap_core:
            state = "CORE"
        elif overlap_tail:
            state = "TAIL"
        elif overlap_exclusion or overlap_uncertain:
            state = "UNCERTAIN"
        elif 0.0 <= d_prev < 0.5:
            state = "POST_EARLY"
        elif 0.5 <= d_prev < 1.25:
            state = "POST_MID"
        elif 1.25 <= d_prev < 2.5:
            state = "POST_LATE"
        elif d_prev >= 2.5 and 0.5 <= d_next < 1.5:
            state = "PRE_STABLE"
        elif d_prev >= 2.5 and d_next >= 1.5:
            state = "INTER_STABLE"
        else:
            state = "CLEAN_OTHER"
        rows.append({
            "rotation": int(rotation), "start": int(left), "end": int(right),
            "state": state, "strict_clean": not overlap_exclusion,
            "d_prev_rot": d_prev, "d_next_rot": d_next,
            "overlap_core": overlap_core, "overlap_tail": overlap_tail,
            "overlap_uncertain": overlap_uncertain,
        })
    return rows


def channel_harmonics(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(value, dtype=np.float64)
    centered = x - np.mean(x, axis=1, keepdims=True)
    n = centered.shape[1]
    phase = 2.0 * np.pi * np.arange(n, dtype=np.float64) / float(n)
    coefficients = np.zeros((x.shape[0], len(ORDERS)), dtype=np.complex128)
    for local, order in enumerate(ORDERS):
        coefficients[:, local] = (2.0 / float(n)) * np.sum(
            centered * np.exp(-1j * float(order) * phase)[None, :], axis=1,
        )
    rms = np.sqrt(np.mean(centered ** 2, axis=1) + 1.0e-12)
    return coefficients, rms, centered


def rotation_feature_vector(value: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 10:
        raise ValueError("rotation_feature_vector expects [10,T]")
    coefficients, rms, centered = channel_harmonics(x)
    values: List[float] = []
    names: List[str] = []

    def add(name: str, number: float) -> None:
        names.append(name)
        values.append(float(number))

    channel_ids = [row["id"] for row in SENSOR_CONTRACT]
    for channel, channel_id in enumerate(channel_ids):
        local = centered[channel]
        p2p = float(np.max(local) - np.min(local))
        crest = float(np.max(np.abs(local)) / max(rms[channel], 1.0e-12))
        kurt = float(np.mean(local ** 4) / max(np.mean(local ** 2) ** 2, 1.0e-12))
        add("%s__log_rms" % channel_id, math.log(max(float(rms[channel]), 1.0e-12)))
        add("%s__log_p2p" % channel_id, math.log(max(p2p, 1.0e-12)))
        add("%s__crest" % channel_id, crest)
        add("%s__kurtosis" % channel_id, kurt)
        for local_order, order in enumerate(ORDERS):
            coefficient = coefficients[channel, local_order]
            amplitude = float(abs(coefficient))
            add("%s__order%d__real" % (channel_id, order), float(np.real(coefficient)))
            add("%s__order%d__imag" % (channel_id, order), float(np.imag(coefficient)))
            add("%s__order%d__log_amp" % (channel_id, order), math.log(max(amplitude, 1.0e-12)))
            add("%s__order%d__amp_to_rms" % (channel_id, order), amplitude / max(float(rms[channel]), 1.0e-12))

    bearing_pairs = ((0, 1, "B1"), (2, 3, "B2"), (4, 5, "B3"))
    bearing_amplitudes = np.zeros((3, len(ORDERS)), dtype=np.float64)
    for bearing_index, (xch, ych, bearing) in enumerate(bearing_pairs):
        for local_order, order in enumerate(ORDERS):
            cx, cy = coefficients[xch, local_order], coefficients[ych, local_order]
            bearing_amplitudes[bearing_index, local_order] = math.sqrt(abs(cx) ** 2 + abs(cy) ** 2)
            phase_difference = np.angle(cx * np.conj(cy))
            add("%s_relation__order%d__log_vector_amp" % (bearing, order), math.log(max(bearing_amplitudes[bearing_index, local_order], 1.0e-12)))
            add("%s_relation__order%d__xy_phase_sin" % (bearing, order), math.sin(float(phase_difference)))
            add("%s_relation__order%d__xy_phase_cos" % (bearing, order), math.cos(float(phase_difference)))

    for local_order, order in enumerate(ORDERS):
        current = bearing_amplitudes[:, local_order]
        total = float(np.sum(current)) + 1.0e-12
        for bearing_index, bearing in enumerate(("B1", "B2", "B3")):
            add("spatial_relation__order%d__%s_share" % (order, bearing), float(current[bearing_index] / total))
        add("spatial_relation__order%d__endpoint_signed" % order, float((current[2] - current[0]) / total))
        add("spatial_relation__order%d__midness" % order, float(np.log((current[1] + 1.0e-12) / math.sqrt(current[0] * current[2] + 1.0e-12))))

    orbit = np.asarray(x[6:8], dtype=np.float64)
    orbit_center = np.mean(orbit, axis=1)
    orbit_zero = orbit - orbit_center[:, None]
    covariance = np.cov(orbit_zero)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance + 1.0e-12 * np.eye(2))
    order = np.argsort(eigenvalues)[::-1]
    major, minor = float(eigenvalues[order[0]]), float(eigenvalues[order[1]])
    vector = eigenvectors[:, order[0]]
    angle = math.atan2(float(vector[1]), float(vector[0]))
    add("orbit_relation__log_radius", math.log(max(math.sqrt(max(major + minor, 1.0e-12)), 1.0e-12)))
    add("orbit_relation__ellipticity", math.sqrt(max(minor, 0.0) / max(major, 1.0e-12)))
    add("orbit_relation__orientation_sin2", math.sin(2.0 * angle))
    add("orbit_relation__orientation_cos2", math.cos(2.0 * angle))
    add("orbit_relation__centroid_x", float(orbit_center[0]))
    add("orbit_relation__centroid_y", float(orbit_center[1]))

    b1_rms = math.sqrt(float(rms[0] ** 2 + rms[1] ** 2))
    b2_rms = math.sqrt(float(rms[2] ** 2 + rms[3] ** 2))
    b3_rms = math.sqrt(float(rms[4] ** 2 + rms[5] ** 2))
    add("crossmodal_relation__B1_soundA_log_rms_ratio", math.log(max(b1_rms, 1.0e-12) / max(float(rms[8]), 1.0e-12)))
    add("crossmodal_relation__B3_soundB_log_rms_ratio", math.log(max(b3_rms, 1.0e-12) / max(float(rms[9]), 1.0e-12)))
    add("crossmodal_relation__B2_orbit_log_rms_ratio", math.log(max(b2_rms, 1.0e-12) / max(math.sqrt(float(rms[6] ** 2 + rms[7] ** 2)), 1.0e-12)))
    vector = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    return vector, names


def rotation_feature_matrix(raw: np.ndarray, rpm: int) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(raw, dtype=np.float64)
    spr = int(samples_per_rotation(int(rpm)))
    rotation_n = int(x.shape[1] // spr)
    if rotation_n < 3:
        raise RuntimeError("At least three complete rotations are required")
    rows: List[np.ndarray] = []
    names: Optional[List[str]] = None
    for rotation in range(rotation_n):
        vector, current_names = rotation_feature_vector(x[:, rotation * spr:(rotation + 1) * spr])
        if names is None:
            names = list(current_names)
        elif names != list(current_names):
            raise RuntimeError("Rotation feature schema changed within a sample")
        rows.append(vector)
    return np.stack(rows).astype(np.float32), list(names or [])
