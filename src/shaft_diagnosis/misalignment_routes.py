"""Probability normalization and fixed acceptance rules for M spatial updates."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np



def normalize_probability(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 1.0e-12)
    array /= np.maximum(array.sum(axis=1, keepdims=True), 1.0e-12)
    output = array.astype(np.float32)
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-5):
        raise RuntimeError("Invalid probability array")
    return output


POSITION_CLASSES = 3


REFERENCE_ID = "A0_FixedA2"


P1_ID = "A1_P4JConfirmed"


PRIMARY_ID = "E0_P4JorTV1"


CANDIDATE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "id": REFERENCE_ID,
        "kind": "anchor",
        "tv1_ratio_min": None,
        "position_policy": "H0",
        "complexity": 0,
    },
    {
        "id": P1_ID,
        "kind": "confirmed_comparator",
        "tv1_ratio_min": None,
        "position_policy": "P4J_only",
        "complexity": 1,
    },
    {
        "id": PRIMARY_ID,
        "kind": "primary",
        "tv1_ratio_min": 1.0,
        "position_policy": "P4J_or_TV1",
        "complexity": 2,
    },
    {
        "id": "E15_TV1Ratio150",
        "kind": "safety_fallback",
        "tv1_ratio_min": 1.5,
        "position_policy": "P4J_or_TV1",
        "complexity": 3,
    },
    {
        "id": "E20_TV1Ratio200",
        "kind": "safety_fallback",
        "tv1_ratio_min": 2.0,
        "position_policy": "P4J_or_TV1",
        "complexity": 4,
    },
)


def build_gates(
    h0_probability: np.ndarray,
    d2_probability: np.ndarray,
    p4j_probability: np.ndarray,
    tv1_probability: np.ndarray,
    t2_label: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    h0 = np.argmax(h0_probability, axis=1)
    proposal = np.argmax(d2_probability, axis=1)
    p4j = np.argmax(p4j_probability, axis=1)
    tv1 = np.argmax(tv1_probability, axis=1)
    t2 = np.asarray(t2_label, dtype=np.int64)
    rows = np.arange(len(h0), dtype=np.int64)
    denominator = np.maximum(tv1_probability[rows, h0], 1.0e-12)
    ratio = np.asarray(tv1_probability[rows, proposal] / denominator, dtype=np.float32)
    different = proposal != h0
    p4j_support = different & (p4j == proposal)
    tv1_support = different & (tv1 == proposal)
    tv1_only = tv1_support & (~p4j_support)
    gates: Dict[str, np.ndarray] = {
        REFERENCE_ID: np.zeros(len(h0), dtype=bool),
        P1_ID: p4j_support,
    }
    for spec in CANDIDATE_SPECS:
        candidate_id = str(spec["id"])
        threshold = spec.get("tv1_ratio_min")
        if threshold is not None:
            gates[candidate_id] = p4j_support | (tv1_only & (ratio >= float(threshold)))
    diagnostics = {
        "h0_label": h0.astype(np.int8),
        "proposal_label": proposal.astype(np.int8),
        "p4j_label": p4j.astype(np.int8),
        "tv1_label": tv1.astype(np.int8),
        "t2_label": t2.astype(np.int8),
        "different": different.astype(np.uint8),
        "p4j_support": p4j_support.astype(np.uint8),
        "tv1_support": tv1_support.astype(np.uint8),
        "tv1_only": tv1_only.astype(np.uint8),
        "both_support": (p4j_support & tv1_support).astype(np.uint8),
        "tv1_relative_support": ratio,
    }
    if not (
        np.all(gates[P1_ID] <= gates["E20_TV1Ratio200"])
        and np.all(gates["E20_TV1Ratio200"] <= gates["E15_TV1Ratio150"])
        and np.all(gates["E15_TV1Ratio150"] <= gates[PRIMARY_ID])
    ):
        raise RuntimeError("P7E nested route contract failed")
    return gates, diagnostics


def route_position(h0: np.ndarray, proposal: np.ndarray, gate: np.ndarray) -> np.ndarray:
    baseline = np.asarray(h0, dtype=np.float32)
    alternative = np.asarray(proposal, dtype=np.float32)
    active = np.asarray(gate, dtype=bool)
    if baseline.shape != alternative.shape or baseline.ndim != 2 or baseline.shape[1] != POSITION_CLASSES:
        raise RuntimeError("Position route probability shape mismatch")
    if len(active) != len(baseline):
        raise RuntimeError("Position route gate length mismatch")
    for value in (baseline, alternative):
        if not np.isfinite(value).all() or not np.allclose(value.sum(axis=1), 1.0, atol=2.0e-5):
            raise RuntimeError("Invalid frozen position probability")
    output = np.array(baseline, copy=True)
    output[active] = alternative[active]
    return output


def normalize_endpoint_probability(value: Any) -> np.ndarray:
    probability = np.asarray(value, dtype=np.float64)
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise RuntimeError("Invalid probability matrix: %s" % (probability.shape,))
    total = probability.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise RuntimeError("Probability has non-positive row mass")
    return (probability / total).astype(np.float32)


def endpoint_softmax(logits: Any) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value = value - np.max(value, axis=1, keepdims=True)
    value = np.exp(np.clip(value, -60.0, 60.0))
    return normalize_endpoint_probability(value)
