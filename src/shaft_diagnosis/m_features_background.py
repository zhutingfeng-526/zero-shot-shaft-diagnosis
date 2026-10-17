"""Source-fitted scaling and grouping of M background response features."""
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np



def robust_center_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    center = np.median(values, axis=0)
    q25 = np.percentile(values, 25.0, axis=0)
    q75 = np.percentile(values, 75.0, axis=0)
    scale = np.maximum(q75 - q25, 1.0e-6)
    return center.astype(np.float32), scale.astype(np.float32)


def robust_transform(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = (np.asarray(x, dtype=np.float32) - center) / scale
    return np.clip(
        np.nan_to_num(values, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0
    ).astype(np.float32)


def fit_feature_normalizers(source_fit: Mapping[str, np.ndarray], contract: Mapping[str, Any]) -> Dict[str, Any]:
    x = np.asarray(source_fit["x"], dtype=np.float32)
    values: Dict[str, np.ndarray] = {
        "station_physics": x[:, np.asarray(contract["station_physics"], dtype=np.int64)].reshape(-1, np.asarray(contract["station_physics"]).shape[1]),
        "station_raw": x[:, np.asarray(contract["station_raw"], dtype=np.int64)].reshape(-1, np.asarray(contract["station_raw"]).shape[1]),
        "orbit_physics": x[:, np.asarray(contract["orbit_physics"], dtype=np.int64)],
        "orbit_raw": x[:, np.asarray(contract["orbit_raw"], dtype=np.int64)],
        "acoustic_physics": x[:, np.asarray(contract["acoustic_physics"], dtype=np.int64)].reshape(-1, np.asarray(contract["acoustic_physics"]).shape[1]),
        "acoustic_raw": x[:, np.asarray(contract["acoustic_raw"], dtype=np.int64)].reshape(-1, np.asarray(contract["acoustic_raw"]).shape[1]),
        "global_position": x[:, np.asarray(contract["global_position"], dtype=np.int64)],
        "global_physics": x[:, np.asarray(contract["global_physics"], dtype=np.int64)],
        "global_residual": x[:, np.asarray(contract["global_residual"], dtype=np.int64)],
        "event_context": x[:, np.asarray(contract["event_context"], dtype=np.int64)],
    }
    normalizers: Dict[str, Any] = {}
    for key, value in values.items():
        center, scale = robust_center_scale(value)
        normalizers[key] = {"center": center, "scale": scale}
    return normalizers


def prepare_features(
    records: Mapping[str, np.ndarray], contract: Mapping[str, Any], normalizers: Mapping[str, Any]
) -> Dict[str, np.ndarray]:
    x = np.asarray(records["x"], dtype=np.float32)

    def transform_group(key: str, values: np.ndarray) -> np.ndarray:
        return robust_transform(
            values,
            np.asarray(normalizers[key]["center"], dtype=np.float32),
            np.asarray(normalizers[key]["scale"], dtype=np.float32),
        )

    station_physics = x[:, np.asarray(contract["station_physics"], dtype=np.int64)]
    station_raw = x[:, np.asarray(contract["station_raw"], dtype=np.int64)]
    acoustic_physics = x[:, np.asarray(contract["acoustic_physics"], dtype=np.int64)]
    acoustic_raw = x[:, np.asarray(contract["acoustic_raw"], dtype=np.int64)]
    result = {
        "station": transform_group("station_physics", station_physics),
        "orbit": transform_group("orbit_physics", x[:, np.asarray(contract["orbit_physics"], dtype=np.int64)]),
        "acoustic": transform_group("acoustic_physics", acoustic_physics),
        "station_raw": transform_group("station_raw", station_raw),
        "orbit_raw": transform_group("orbit_raw", x[:, np.asarray(contract["orbit_raw"], dtype=np.int64)]),
        "acoustic_raw": transform_group("acoustic_raw", acoustic_raw),
        "global_position": transform_group("global_position", x[:, np.asarray(contract["global_position"], dtype=np.int64)]),
        "global_physics": transform_group("global_physics", x[:, np.asarray(contract["global_physics"], dtype=np.int64)]),
        "global_residual": transform_group("global_residual", x[:, np.asarray(contract["global_residual"], dtype=np.int64)]),
        "event_context": transform_group("event_context", x[:, np.asarray(contract["event_context"], dtype=np.int64)]),
    }
    for key in ("position", "degree", "joint", "key", "cond_id", "structure", "rpm", "window_count", "split"):
        result[key] = np.asarray(records[key])
    return result


def normalizer_arrays(normalizers: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for key, values in normalizers.items():
        arrays["%s_center" % key] = np.asarray(values["center"], dtype=np.float32)
        arrays["%s_scale" % key] = np.asarray(values["scale"], dtype=np.float32)
    return arrays
