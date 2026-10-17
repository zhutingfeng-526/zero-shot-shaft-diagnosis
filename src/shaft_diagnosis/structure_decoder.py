"""Frozen V12 component-threshold decoder and categorical epoch consensus. No decoder search or report logic."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import numpy as np
COMPONENTS = ("U", "M", "R")
STRUCTURES = ["normal", "U", "M", "R", "UM", "UR", "MR", "UMR"]
EPSILON = 1.0e-3
BITS = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int64)

def topk_bits(scores: np.ndarray, k: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    k = np.asarray(k, dtype=np.int64)
    k = np.clip(k, 0, scores.shape[1])
    order = np.argsort(-scores, axis=1)
    out = np.zeros_like(scores, dtype=np.int64)
    for kk in range(1, scores.shape[1] + 1):
        rows = np.where(k == kk)[0]
        if rows.size == 0:
            continue
        out[rows[:, None], order[rows, :kk]] = 1
    return out


def decode_margin(margin: np.ndarray, count_pred: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    margin = np.asarray(margin, dtype=np.float32)
    family = str(spec['family'])
    if family == 'proto_threshold':
        return (margin > float(spec.get('tau', 0.0))).astype(np.int64)
    if family == 'global_tau':
        return (margin >= float(spec.get('tau', 0.0))).astype(np.int64)
    if family == 'count_head_topk':
        return topk_bits(margin, count_pred)
    order = np.argsort(-margin, axis=1)
    sorted_m = np.take_along_axis(margin, order, axis=1)
    m1, m2, m3 = (sorted_m[:, 0], sorted_m[:, 1], sorted_m[:, 2])
    if family == 'gap_guard':
        tau_abs = float(spec['tau_abs'])
        gap12 = float(spec['gap12'])
        gap23 = float(spec['gap23'])
        k = np.full((margin.shape[0],), 2, dtype=np.int64)
        k[m1 < tau_abs] = 0
        single = (m1 >= tau_abs) & ((m2 < tau_abs) | (m1 - m2 >= gap12))
        k[single] = 1
        triple = (m1 >= tau_abs) & (m2 >= tau_abs) & (m3 >= tau_abs) & (m2 - m3 <= gap23)
        k[triple] = 3
        return topk_bits(margin, k)
    if family == 'hybrid':
        tau_abs = float(spec['tau_abs'])
        tau_soft = float(spec['tau_soft'])
        gap12 = float(spec['gap12'])
        gap23 = float(spec['gap23'])
        active_count = np.sum(margin >= tau_abs, axis=1).astype(np.int64)
        k = active_count.copy()
        k[m1 < tau_abs] = 0
        soft_pair = (m1 >= tau_abs) & (active_count <= 1) & (m2 >= tau_abs - tau_soft) & (m1 - m2 <= gap12)
        k[soft_pair] = 2
        hard_single = (m1 >= tau_abs) & (active_count <= 1) & ~soft_pair
        k[hard_single] = 1
        drop_third = (active_count >= 3) & (m2 - m3 > gap23)
        k[drop_third] = 2
        keep_triple = (active_count >= 3) & (m2 - m3 <= gap23)
        k[keep_triple] = 3
        return topk_bits(margin, k)
    raise ValueError(f'Unsupported decoder family: {family}')

def decode_margin_strict(margin: np.ndarray, count_pred: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    margin = np.asarray(margin, dtype=np.float32)
    family = str(spec['family'])
    if family in ('proto_threshold', 'global_tau', 'count_head_topk', 'gap_guard', 'hybrid'):
        return decode_margin(margin, count_pred, spec)
    if family == 'component_tau':
        tau = np.asarray([float(spec[f'tau_{c}']) for c in COMPONENTS], dtype=np.float32)
        return (margin >= tau.reshape(1, 3)).astype(np.int64)
    if family == 'component_tau_soft_hybrid':
        tau = np.asarray([float(spec[f'tau_{c}']) for c in COMPONENTS], dtype=np.float32)
        soft = float(spec.get('tau_soft', 0.25))
        gap12 = float(spec.get('gap12', 1.0))
        gap23 = float(spec.get('gap23', 0.5))
        active_count = np.sum(margin >= tau.reshape(1, 3), axis=1).astype(np.int64)
        order = np.argsort(-margin, axis=1)
        sorted_m = np.take_along_axis(margin, order, axis=1)
        sorted_tau = np.take_along_axis(np.tile(tau.reshape(1, 3), (margin.shape[0], 1)), order, axis=1)
        m1, m2, m3 = (sorted_m[:, 0], sorted_m[:, 1], sorted_m[:, 2])
        t1, t2 = (sorted_tau[:, 0], sorted_tau[:, 1])
        k = active_count.copy()
        k[m1 < t1] = 0
        soft_pair = (m1 >= t1) & (active_count <= 1) & (m2 >= t2 - soft) & (m1 - m2 <= gap12)
        k[soft_pair] = 2
        hard_single = (m1 >= t1) & (active_count <= 1) & ~soft_pair
        k[hard_single] = 1
        drop_third = (active_count >= 3) & (m2 - m3 > gap23)
        k[drop_third] = 2
        keep_triple = (active_count >= 3) & (m2 - m3 <= gap23)
        k[keep_triple] = 3
        return topk_bits(margin, k)
    raise ValueError(f'Unsupported strict decoder family: {family}')

def _margin3(margin: np.ndarray) -> np.ndarray:
    arr = np.asarray(margin, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f'Expected margin with shape [N, >=3], got {arr.shape}')
    return arr[:, :3]

def _tau3_for_spec(spec: Dict[str, Any]) -> np.ndarray:
    if str(spec.get('family')) == 'component_tau_soft_hybrid':
        return np.asarray([float(spec['tau_U']), float(spec['tau_M']), float(spec['tau_R'])], dtype=np.float32)
    tau_abs = float(spec.get('tau_abs', 0.0))
    return np.asarray([tau_abs, tau_abs, tau_abs], dtype=np.float32)


def _excess3(margin_or_arrays: Any, spec: Dict[str, Any]) -> np.ndarray:
    return _margin3(margin_or_arrays) - _tau3_for_spec(spec).reshape(1, 3)


def _normal_guard_mask(margin: np.ndarray, count_pred: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    mode = str(spec.get('normal_guard_mode', ''))
    if not mode:
        return np.zeros(np.asarray(margin).shape[0], dtype=bool)
    ex = _excess3(margin, spec)
    sorted_ex = np.sort(ex, axis=1)[:, ::-1]
    max_ex = sorted_ex[:, 0]
    second_ex = sorted_ex[:, 1]
    mean_ex = np.mean(ex, axis=1)
    count0 = np.asarray(count_pred, dtype=np.int64).reshape(-1) == 0
    thr = float(spec.get('normal_max_excess', 0.0))
    loose = float(spec.get('normal_loose_excess', thr))
    strict = float(spec.get('normal_strict_excess', thr))
    second_thr = float(spec.get('normal_second_excess', np.inf))
    mean_thr = float(spec.get('normal_mean_excess', np.inf))
    if mode == 'max_band':
        return max_ex <= thr
    if mode == 'count0_max_band':
        return count0 & (max_ex <= thr)
    if mode == 'dual_band':
        return (max_ex <= strict) | count0 & (max_ex <= loose)
    if mode == 'shape_band':
        return (max_ex <= thr) & (second_ex <= second_thr) & (mean_ex <= mean_thr)
    raise ValueError(f'Unsupported V12 normal guard mode: {mode}')


def decode_margin_strict_v12(margin: np.ndarray, count_pred: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    pred = decode_margin_strict(margin, count_pred, spec).astype(np.int64, copy=True)
    guard = _normal_guard_mask(margin, count_pred, spec)
    if np.any(guard):
        pred[guard] = 0
    return pred

def count_pred(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    logits = np.asarray(arrays['count_logits'], dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise RuntimeError('Expected three count logits, got %s' % (logits.shape,))
    return np.argmax(logits, axis=1).astype(np.int64)


def categorical_pool(decision_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    decision_matrix = np.asarray(decision_matrix, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if decision_matrix.ndim != 2:
        raise RuntimeError('Decision matrix must be [E,N].')
    if weights.shape != (decision_matrix.shape[0],):
        raise RuntimeError('Temporal weight shape mismatch.')
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-08):
        raise RuntimeError('Temporal weights must sum to one.')
    out = np.zeros((decision_matrix.shape[1], len(STRUCTURES)), dtype=np.float64)
    rows = np.arange(decision_matrix.shape[1])
    for ei in range(decision_matrix.shape[0]):
        out[rows, decision_matrix[ei]] += float(weights[ei])
    return (1.0 - EPSILON) * out + EPSILON / float(len(STRUCTURES))


def normalized_entropy(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / math.log(float(p.shape[1]))

def structure_indices(gt: np.ndarray) -> np.ndarray:
    lookup = {tuple((int(x) for x in row)): i for i, row in enumerate(BITS)}
    return np.asarray([lookup[tuple((int(x) for x in row))] for row in gt], dtype=np.int64)

def vote_counts(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.int64)
    out = np.zeros((matrix.shape[1], len(STRUCTURES)), dtype=np.int16)
    rows = np.arange(matrix.shape[1])
    for ei in range(matrix.shape[0]):
        out[rows, matrix[ei]] += 1
    return out


def equal_consensus(split_decisions: Dict[int, np.ndarray], epochs: Sequence[int], reliability: Dict[int, float], tie_rule: str='latest_then_source') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs = [int(x) for x in epochs]
    matrix = np.stack([np.asarray(split_decisions[ep], dtype=np.int64) for ep in epochs], axis=0)
    counts = vote_counts(matrix)
    prob = categorical_pool(matrix, np.full(len(epochs), 1.0 / float(len(epochs))))
    max_count = counts.max(axis=1, keepdims=True)
    tied = counts == max_count
    tie_mask = tied.sum(axis=1) > 1
    pred = np.argmax(counts, axis=1).astype(np.int64)
    rows = np.arange(matrix.shape[1])
    latest = matrix[-1]
    if tie_rule not in ('index', 'latest', 'source', 'latest_then_source'):
        raise ValueError('Unknown tie rule: %s' % tie_rule)
    if tie_rule in ('latest', 'latest_then_source'):
        latest_is_tied = tie_mask & tied[rows, latest]
        pred[latest_is_tied] = latest[latest_is_tied]
    else:
        latest_is_tied = np.zeros_like(tie_mask)
    unresolved = tie_mask & ~latest_is_tied if tie_rule == 'latest_then_source' else tie_mask
    if tie_rule in ('source', 'latest_then_source') and bool(np.any(unresolved)):
        source_score = np.zeros_like(prob)
        for ei, ep in enumerate(epochs):
            source_score[rows, matrix[ei]] += float(reliability[int(ep)])
        source_score[~tied] = -np.inf
        source_max = np.max(source_score, axis=1, keepdims=True)
        source_tied = tied & np.isclose(source_score, source_max, rtol=0.0, atol=1e-12)
        source_pred = np.argmax(source_score, axis=1).astype(np.int64)
        pred[unresolved] = source_pred[unresolved]
        score_tie = unresolved & (source_tied.sum(axis=1) > 1)
        for ei in range(matrix.shape[0] - 1, -1, -1):
            candidate = matrix[ei]
            choose = score_tie & source_tied[rows, candidate]
            pred[choose] = candidate[choose]
            score_tie[choose] = False
    entropy = normalized_entropy(prob)
    return (pred, prob, entropy, tie_mask, matrix)
