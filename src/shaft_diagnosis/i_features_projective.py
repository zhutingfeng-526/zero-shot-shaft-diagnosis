"""Numerical projective functions."""
from __future__ import annotations
from typing import Mapping, Tuple
import numpy as np
from scipy import signal

FS_HZ = 5000.0


R_CLASSES = 3


TIME_SCALES_MS = (4, 12, 32, 80)


EVENT_GUARD_MS = 80.0


def normalize_probability(value: np.ndarray) -> np.ndarray:
    probability = np.maximum(np.asarray(value, dtype=np.float64), 1.0e-12)
    return (probability / np.maximum(np.sum(probability, axis=-1, keepdims=True), 1.0e-12)).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value -= np.max(value, axis=-1, keepdims=True)
    return normalize_probability(np.exp(value))


def hermitian(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def normalized_projector(vectors: np.ndarray, rank: int) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.complex128)
    if value.ndim == 1:
        value = value[:, None]
    q, _ = np.linalg.qr(value)
    q = q[:, : int(rank)]
    return hermitian(q.dot(q.conj().T)).astype(np.complex64)


def projector_distance(left: np.ndarray, right: np.ndarray, rank: int) -> float:
    overlap = float(np.real(np.trace(hermitian(left).dot(hermitian(right)))))
    return float(np.clip(1.0 - overlap / max(float(rank), 1.0), 0.0, 1.0))


def normalize_projector_average(value: np.ndarray, rank: int) -> np.ndarray:
    matrix = hermitian(value)
    _eigenvalue, eigenvector = np.linalg.eigh(matrix)
    return normalized_projector(eigenvector[:, -int(rank):], int(rank))


def event_free_mask(length: int, centers: np.ndarray, widths: np.ndarray, raw: np.ndarray) -> np.ndarray:
    mask = np.zeros(int(length), dtype=bool)
    guard = int(round(EVENT_GUARD_MS * FS_HZ / 1000.0))
    for center, width in zip(np.asarray(centers, dtype=np.int64), np.asarray(widths, dtype=np.int64)):
        radius = max(guard, int(width) + guard // 2)
        left = max(0, int(center) - radius)
        right = min(int(length), int(center) + radius + 1)
        mask[left:right] = True
    detected = np.asarray(p7a.robust_event_mask(np.asarray(raw, dtype=np.float32)), dtype=bool)
    if len(detected) != int(length):
        raise RuntimeError("P7A event-mask length mismatch")
    return mask | detected


def robust_record_normalize(raw: np.ndarray, excluded: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(raw, dtype=np.float64)
    valid = ~np.asarray(excluded, dtype=bool)
    if int(np.sum(valid)) < 1024:
        valid = np.ones(value.shape[1], dtype=bool)
    center = np.median(value[:, valid], axis=1)
    scale = 1.4826 * np.median(np.abs(value[:, valid] - center[:, None]), axis=1)
    fallback = np.std(value[:, valid], axis=1)
    scale = np.where(scale > 1.0e-7, scale, np.maximum(fallback, 1.0e-5))
    normalized = ((value - center[:, None]) / scale[:, None]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise RuntimeError("Non-finite same-record normalization")
    return normalized, center.astype(np.float32), scale.astype(np.float32)


def analytic_channels(raw: np.ndarray) -> np.ndarray:
    if signal is None:
        raise RuntimeError("SciPy is required for analytic covariance")
    return np.asarray(signal.hilbert(np.asarray(raw, dtype=np.float64), axis=1), dtype=np.complex64)


def _safe_projector_mean(values: np.ndarray, rank: int) -> np.ndarray:
    current = np.asarray(values, dtype=np.complex128)
    if current.ndim != 3 or not len(current):
        raise RuntimeError("Projector mean requires events x channels x channels")
    return normalize_projector_average(np.mean(current, axis=0), int(rank))


def projector_distance_matrix(
    descriptor: Mapping[str, np.ndarray],
    prototypes: Mapping[str, np.ndarray],
    vibration_field: str,
    acoustic_field: str,
    vibration_rank: int,
    blend_acoustic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(np.asarray(descriptor["keys"]))
    vibration = np.zeros((n, R_CLASSES), dtype=np.float64)
    acoustic = np.zeros((n, R_CLASSES), dtype=np.float64)
    for index in range(n):
        for label in range(R_CLASSES):
            vibration[index, label] = np.mean([
                projector_distance(
                    np.asarray(descriptor[vibration_field])[index, scale_position],
                    np.asarray(prototypes[vibration_field])[label, scale_position],
                    vibration_rank,
                )
                for scale_position in range(len(TIME_SCALES_MS))
            ])
            acoustic[index, label] = np.mean([
                projector_distance(
                    np.asarray(descriptor[acoustic_field])[index, scale_position],
                    np.asarray(prototypes[acoustic_field])[label, scale_position],
                    1,
                )
                for scale_position in range(len(TIME_SCALES_MS))
            ])
    total = 0.55 * vibration + 0.45 * acoustic if blend_acoustic else vibration
    return total.astype(np.float32), vibration.astype(np.float32), acoustic.astype(np.float32)


def distance_probability(distance: np.ndarray, temperature: float) -> np.ndarray:
    value = np.asarray(distance, dtype=np.float64)
    finite = np.isfinite(value).all(axis=1)
    safe = np.where(np.isfinite(value), value, 0.0)
    probability = softmax(-safe / max(float(temperature), 1.0e-3))
    probability[~finite] = 1.0 / R_CLASSES
    return probability

def _bind(modules):
    globals().update(modules)
