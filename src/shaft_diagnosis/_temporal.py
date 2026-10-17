"""Numerical primitives for causal attribute aggregation."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

EPS = 1.0e-8


WINSOR_FRACTION = 0.10


HUBER_C = 1.345


HUBER_ITERATIONS = 6


QUALITY_LOW_WEIGHT = 0.50


def normalize_probability(value: Any) -> np.ndarray:
    probability = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    if probability.ndim < 2 or not np.isfinite(probability).all():
        raise RuntimeError("Invalid probability array: %s" % (probability.shape,))
    total = np.sum(probability, axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise RuntimeError("Probability row has no mass")
    return probability / total


def softmax(value: Any) -> np.ndarray:
    score = np.asarray(value, dtype=np.float64)
    score = score - np.max(score, axis=-1, keepdims=True)
    exp = np.exp(score)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def sigmoid(value: Any) -> np.ndarray:
    score = np.asarray(value, dtype=np.float64)
    output = np.empty_like(score)
    positive = score >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
    exp = np.exp(score[~positive])
    output[~positive] = exp / (1.0 + exp)
    return output


def huber_location(log_odds: np.ndarray, external_weight: Optional[np.ndarray] = None) -> np.ndarray:
    value = np.asarray(log_odds, dtype=np.float64)
    center = np.median(value, axis=1)
    mad = np.median(np.abs(value - center[:, None]), axis=1)
    scale = np.maximum(1.4826 * mad, 1.0e-6)
    external = np.ones_like(value) if external_weight is None else np.asarray(external_weight, dtype=np.float64)
    if external.shape != value.shape or np.any(external <= 0.0) or not np.isfinite(external).all():
        raise RuntimeError("Invalid external Huber weight")
    for _ in range(HUBER_ITERATIONS):
        residual = np.abs((value - center[:, None]) / scale[:, None])
        robust = np.ones_like(residual)
        large = residual > HUBER_C
        robust[large] = HUBER_C / residual[large]
        weight = robust * external
        center = np.sum(weight * value, axis=1) / np.maximum(np.sum(weight, axis=1), EPS)
    return center


def degree_from_log_odds(log_odds: np.ndarray) -> np.ndarray:
    p1 = sigmoid(log_odds)
    return np.column_stack((1.0 - p1, p1))

def segment_bounds(segment_id: Any) -> List[Tuple[int, int]]:
    values = np.asarray(segment_id, dtype=np.int64)
    if not len(values):
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), changes))
    ends = np.concatenate((changes, np.asarray([len(values)], dtype=np.int64)))
    output = [(int(left), int(right)) for left, right in zip(starts.tolist(), ends.tolist())]
    if any(np.any(values[left:right] != values[left]) for left, right in output):
        raise RuntimeError("Timeline segment IDs are not contiguous")
    return output


def rolling_group_mean(probability: Any, segment_id: Any, count: int) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(probability, dtype=np.float64)
    segment = np.asarray(segment_id, dtype=np.int64)
    if value.ndim != 2 or len(value) != len(segment) or int(count) < 1:
        raise RuntimeError("Invalid causal rolling input")
    positions: List[np.ndarray] = []
    means: List[np.ndarray] = []
    for left, right in segment_bounds(segment):
        if right - left < int(count):
            continue
        current = value[left:right]
        prefix = np.vstack((np.zeros((1, current.shape[1]), dtype=np.float64), np.cumsum(current, axis=0)))
        means.append((prefix[int(count):] - prefix[:-int(count)]) / float(count))
        positions.append(np.arange(left + int(count) - 1, right, dtype=np.int64))
    if not positions:
        return np.zeros(0, dtype=np.int64), np.zeros((0, value.shape[1]), dtype=np.float64)
    return np.concatenate(positions), np.concatenate(means, axis=0)
