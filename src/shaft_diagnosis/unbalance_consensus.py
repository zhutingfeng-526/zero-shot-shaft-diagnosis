"""Final sample-wise median transport; selected rows use the companion-I correction, other rows retain raw physics."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
ORDINAL_SLOPE = 4.0

def validate_inferred(name: str, inferred: Mapping[str, np.ndarray], n: int) -> None:
    required = ('probabilities', 'evidence', 'anchor', 'deep_evidence', 'confidence', 'uncertainty')
    missing = [key for key in required if key not in inferred]
    if missing:
        raise RuntimeError('%s prediction cache missing fields: %s' % (name, missing))
    if any((len(np.asarray(inferred[key])) != int(n) for key in required)):
        raise RuntimeError('%s prediction cache length mismatch' % name)
    if not all((np.isfinite(np.asarray(inferred[key])).all() for key in required)):
        raise RuntimeError('%s prediction cache contains non-finite values' % name)
    probabilities = np.asarray(inferred['probabilities'], dtype=np.float64)
    if probabilities.shape != (int(n), 3):
        raise RuntimeError('%s probability shape mismatch: %s' % (name, probabilities.shape))
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-05, rtol=0.0):
        raise RuntimeError('%s probabilities are not normalized' % name)

def ordinal_probabilities(corrected_anchor: np.ndarray, contract: Mapping[str, Any]) -> np.ndarray:
    corrected = np.asarray(corrected_anchor, dtype=np.float64).reshape(-1)
    thresholds = np.asarray(contract['threshold_init'], dtype=np.float64).reshape(2)
    slope = float(ORDINAL_SLOPE)
    logits = slope * (corrected[:, None] - thresholds[None, :])
    logits = np.clip(logits, -60.0, 60.0)
    gt = 1.0 / (1.0 + np.exp(-logits))
    probabilities = np.stack([1.0 - gt[:, 0], gt[:, 0] - gt[:, 1], gt[:, 1]], axis=1)
    probabilities = np.maximum(probabilities, 1e-08)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float64)


def robust_disagreement(shift_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(shift_stack, dtype=np.float64)
    center = np.median(stack, axis=0)
    mad = 1.4826 * np.median(np.abs(stack - center[None, :]), axis=0)
    return (center, mad)


def build_shift_consensus(raw: Mapping[str, np.ndarray], shared_by_seed: Mapping[int, Mapping[str, np.ndarray]], members: Sequence[int], use_shared: np.ndarray, contract: Mapping[str, Any], aggregation: str) -> Dict[str, np.ndarray]:
    members = tuple((int(seed) for seed in members))
    if not members:
        raise RuntimeError('Transport consensus has no members')
    mask = np.asarray(use_shared, dtype=bool)
    n = len(mask)
    shifts = np.stack([np.asarray(shared_by_seed[seed]['deep_evidence'], dtype=np.float64) for seed in members], axis=0)
    if shifts.shape != (len(members), n):
        raise RuntimeError('Transport shift stack shape mismatch: %s' % (shifts.shape,))
    median_shift, mad = robust_disagreement(shifts)
    if aggregation == 'median':
        selected_shift = median_shift
    elif aggregation == 'mean':
        selected_shift = np.mean(shifts, axis=0)
    else:
        raise RuntimeError('Unknown transport aggregation: %s' % aggregation)
    anchor = np.asarray(raw['anchor'], dtype=np.float64)
    corrected = anchor - selected_shift
    reconstructed = ordinal_probabilities(corrected, contract)
    scale = max(float(contract['transport_scale']), 1e-12)
    normalized_mad = mad / scale
    consensus: Dict[str, np.ndarray] = {key: np.asarray(value).copy() for key, value in raw.items()}
    consensus['probabilities'][mask] = reconstructed[mask]
    consensus['evidence'][mask] = corrected[mask]
    consensus['deep_evidence'][mask] = selected_shift[mask]
    consensus['uncertainty'][mask] = normalized_mad[mask]
    consensus['confidence'][mask] = 1.0 / (1.0 + normalized_mad[mask])
    validate_inferred('%s_shift_consensus' % aggregation, consensus, n)
    return consensus
