"""Numerical baseline functions."""
from __future__ import annotations
from typing import Any, Mapping, Tuple
import numpy as np

R_CLASSES = 3


def raw_covariance_probability(
    descriptor: Mapping[str, np.ndarray], model: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    distance, _vibration, _acoustic = r13.projector_distance_matrix(
        descriptor, model["prototypes"], "vibration_raw_rank1", "acoustic_raw_rank1", 1
    )
    probability = r13.distance_probability(distance, float(model["projective_temperature"]))
    active = np.asarray(descriptor["projective_active"], dtype=bool)
    probability[~active] = 1.0 / R_CLASSES
    return np.asarray(probability, dtype=np.float32), np.asarray(distance, dtype=np.float32)

def _bind(modules):
    globals().update(modules)
