"""Event-state encoding and source normalization for per-rotation M features."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np



STATE_NAMES = (
    "CORE", "TAIL", "UNCERTAIN", "POST_EARLY", "POST_MID", "POST_LATE",
    "PRE_STABLE", "INTER_STABLE", "CLEAN_OTHER",
)


def state_vector(row: Mapping[str, Any]) -> np.ndarray:
    vector = np.zeros(len(STATE_NAMES) + 6, dtype=np.float32)
    state = str(row.get("state", "CLEAN_OTHER"))
    vector[STATE_NAMES.index(state) if state in STATE_NAMES else STATE_NAMES.index("CLEAN_OTHER")] = 1.0
    d_prev = float(row.get("d_prev_rot", float("inf")))
    d_next = float(row.get("d_next_rot", float("inf")))
    vector[len(STATE_NAMES) + 0] = float(np.log1p(min(d_prev, 8.0))) if np.isfinite(d_prev) else float(np.log(9.0))
    vector[len(STATE_NAMES) + 1] = float(np.log1p(min(d_next, 8.0))) if np.isfinite(d_next) else float(np.log(9.0))
    vector[len(STATE_NAMES) + 2] = float(bool(row.get("strict_clean", True)))
    vector[len(STATE_NAMES) + 3] = float(bool(row.get("overlap_core", False)))
    vector[len(STATE_NAMES) + 4] = float(bool(row.get("overlap_tail", False)))
    vector[len(STATE_NAMES) + 5] = float(bool(row.get("overlap_uncertain", False)))
    return vector


def clean_state_vector() -> np.ndarray:
    return state_vector({"state": "CLEAN_OTHER", "strict_clean": True})


def feature_normalization(bank: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    matrix = np.concatenate([np.asarray(value, dtype=np.float64) for value in bank["rotation_features"]], axis=0)
    center = np.median(matrix, axis=0)
    scale = np.percentile(matrix, 75.0, axis=0) - np.percentile(matrix, 25.0, axis=0)
    scale = np.maximum(scale, 1.0e-4)
    return {"center": center.astype(np.float32), "scale": scale.astype(np.float32)}


def quantile_map(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise RuntimeError("Empty or non-finite source coherence values")
    return {"q%d" % q: float(np.quantile(array, float(q) / 100.0)) for q in (10, 20, 30, 40)}
