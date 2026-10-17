"""Numerical event mask functions."""
from __future__ import annotations
import numpy as np

def robust_event_mask(raw: np.ndarray) -> np.ndarray:
    difference = np.diff(np.asarray(raw, dtype=np.float64), axis=1, prepend=raw[:, :1])
    score = np.sqrt(np.mean(difference[:6] ** 2, axis=0)) + 0.25 * np.sqrt(np.mean(difference[8:] ** 2, axis=0))
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    z = (score - median) / max(1.4826 * mad, 1.0e-9)
    mask = z > 8.0
    return np.convolve(mask.astype(np.int64), np.ones(17, dtype=np.int64), mode="same") > 0 if np.any(mask) else mask

def _bind(modules):
    globals().update(modules)
