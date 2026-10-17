"""Numerical hierarchy functions."""
from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
import numpy as np

R_CLASSES = 3


def normalize_probability(value: np.ndarray) -> np.ndarray:
    probability = np.maximum(np.asarray(value, dtype=np.float64), 1.0e-12)
    probability /= np.maximum(np.sum(probability, axis=-1, keepdims=True), 1.0e-12)
    return probability.astype(np.float32)


def robust_location(value: np.ndarray) -> float:
    current = np.asarray(value, dtype=np.float64)
    current = current[np.isfinite(current)]
    if not len(current):
        return 0.0
    if len(current) < 5:
        return float(np.median(current))
    low, high = np.quantile(current, (0.10, 0.90))
    trimmed = current[(current >= low) & (current <= high)]
    return float(np.median(trimmed if len(trimmed) else current))


def apply_residualizer(model: Mapping[str, np.ndarray], feature: np.ndarray, process: np.ndarray) -> np.ndarray:
    x = np.asarray(feature, dtype=np.float64)
    p = np.asarray(process, dtype=np.float64)
    zp = np.clip((p - np.asarray(model["process_center"])) / np.asarray(model["process_scale"]), -8.0, 8.0)
    zx = np.clip((x - np.asarray(model["feature_center"])) / np.asarray(model["feature_scale"]), -8.0, 8.0)
    design = np.column_stack((np.ones(len(zp)), zp))
    return (zx - design @ np.asarray(model["coefficient"], dtype=np.float64)).astype(np.float32)


def e1_score(model: Mapping[str, Any], feature: np.ndarray, process: np.ndarray) -> np.ndarray:
    x = np.asarray(feature, dtype=np.float32)
    if str(model.get("kind", "plain")) == "residual":
        x = apply_residualizer(model["residualizer"], x, process)
    raw = r15.logistic_score(model["classifier"], x)
    return np.clip(float(model.get("gain", 1.0)) * raw, -float(model.get("cap", 4.0)), float(model.get("cap", 4.0))).astype(np.float32)


def e2_score_bundle(bundle: Mapping[str, Any], model: Mapping[str, Any], control: str = "K0_clean") -> Dict[str, np.ndarray]:
    if control == "K8_B1_B3_full_endpoint_swap":
        # B1/B3 are observed jointly by endpoint vibration stations and by
        # acoustic A/B.  Swapping only vibration leaves a valid acoustic
        # endpoint identity behind and is therefore not a destructive
        # topology control for E2.  The full control swaps both observations.
        views = dict(r15.r14.control_event_views(bundle, "K1_B1_B3_vibration_swap"))
        acoustic_swap = np.eye(2, dtype=np.float64)[[1, 0]]
        views["acoustic_raw_rank1"] = r15.r14.similarity_transform_batch(
            np.asarray(views["acoustic_raw_rank1"]), acoustic_swap
        )
    else:
        views = r15.r14.control_event_views(bundle, control)
    distances = r15.r14.carrier_scale_distances(views, model)
    distances["active"] = np.asarray(views["active"], dtype=bool)
    center_event = (
        np.minimum(np.asarray(distances["center_rank1"])[:, :, 0], np.asarray(distances["center_rank1"])[:, :, 2])
        - np.asarray(distances["center_rank1"])[:, :, 1]
    )
    side_event = np.asarray(distances["side_full"])[:, :, 2] - np.asarray(distances["side_full"])[:, :, 0]
    center = np.mean(center_event, axis=1)
    side = np.mean(side_event, axis=1)
    temperature = model["temperatures"]["C1_RawRank1Factorized"]
    event_probability = r15.r14.factorized_probability(center, side, temperature["center"], temperature["side"])
    event_score = np.log(np.maximum(event_probability[:, 2], 1.0e-8)) - np.log(np.maximum(np.max(event_probability[:, :2], axis=1), 1.0e-8))
    center_scale, side_scale = r15.r14.group_scale_scores(bundle, distances, "center_rank1", "side_full")
    record_center, record_side = r15.r14.aggregate_variant_scores(
        center_scale, side_scale, np.asarray(model["uniform_weight"]), np.asarray(model["uniform_weight"])
    )
    record_probability = r15.r14.factorized_probability(
        record_center, record_side, temperature["center"], temperature["side"]
    )
    return {
        "event_score": np.asarray(event_score, dtype=np.float32),
        "event_valid": np.asarray(views["active"], dtype=bool),
        "record_probability": np.asarray(record_probability, dtype=np.float32),
        "record_score": (
            np.log(np.maximum(record_probability[:, 2], 1.0e-8))
            - np.log(np.maximum(np.max(record_probability[:, :2], axis=1), 1.0e-8))
        ).astype(np.float32),
    }


def force_winner_probability(base_probability: np.ndarray, winners: np.ndarray) -> np.ndarray:
    base = normalize_probability(base_probability)
    output = np.asarray(base, dtype=np.float64).copy()
    for index, winner in enumerate(np.asarray(winners, dtype=np.int64).tolist()):
        if int(np.argmax(base[index])) == int(winner):
            continue
        other = np.asarray([label for label in range(R_CLASSES) if label != int(winner)], dtype=np.int64)
        winner_mass = max(0.500001, float(np.max(base[index, other])) + 1.0e-6)
        winner_mass = min(winner_mass, 0.75)
        remainder = 1.0 - winner_mass
        denominator = float(np.sum(base[index, other]))
        output[index, int(winner)] = winner_mass
        output[index, other] = remainder * base[index, other] / max(denominator, 1.0e-12)
    output = normalize_probability(output)
    if not np.array_equal(np.argmax(output, axis=1), np.asarray(winners, dtype=np.int64)):
        raise RuntimeError("R16 routed probability does not reproduce routed prediction")
    return output


def apply_hierarchy(
    base_probability: np.ndarray,
    e2_gate: np.ndarray,
    e1_gate: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = normalize_probability(base_probability)
    prediction = np.argmax(base, axis=1).astype(np.int16)
    original = prediction.copy()
    stage = np.zeros(len(prediction), dtype=np.int8)
    endpoint = np.asarray(e2_gate, dtype=bool) & (prediction != 2)
    prediction[endpoint] = 2
    stage[endpoint] = 1
    adjacent = np.asarray(e1_gate, dtype=bool) & (prediction == 0)
    prediction[adjacent] = 1
    stage[adjacent] = 2
    if np.any((original == 2) & (prediction != 2)):
        raise RuntimeError("R16 changed a protected baseline-R2 decision")
    if np.any((original == 1) & (prediction == 0)) or np.any((original == 2) & (prediction == 0)):
        raise RuntimeError("R16 introduced a forbidden backward transition")
    probability = force_winner_probability(base, prediction)
    return probability, prediction, stage


def left_relation_probability(base: np.ndarray, record_delta: np.ndarray) -> np.ndarray:
    return r15.left_mass_update(np.asarray(base, dtype=np.float32), np.asarray(record_delta, dtype=np.float32), False)

def _bind(modules):
    globals().update(modules)
