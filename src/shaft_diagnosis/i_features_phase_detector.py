"""Numerical phase detector functions."""
from __future__ import annotations
import argparse
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np

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
    return p4zb.node_event_trace(value, np.asarray(channel_scale, dtype=np.float64))


def trace_states(trace: np.ndarray, rpm: int, detector: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    values = np.asarray(trace, dtype=np.float64)
    hard = p4zb.close_binary(values >= float(detector["hard_threshold"]), int(args.event_close_gap))
    intervals = p4zb.connected_intervals(hard, int(args.event_min_samples))
    core = np.zeros(len(values), dtype=bool)
    expanded: List[Tuple[int, int]] = []
    for start, end in intervals:
        left = max(0, start - int(args.core_expand)); right = min(len(values), end + int(args.core_expand))
        core[left:right] = True; expanded.append((left, right))
    envelope = p4zb.moving_average(values[None, :], int(args.tail_smooth_samples))[0]
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

def _bind(modules):
    globals().update(modules)
