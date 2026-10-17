"""Evaluate composition and complete diagnosis with equal condition/run weight."""
from __future__ import annotations
import numpy as np
from ._metrics import condition_run_weights, confusion_metrics
from ._routing import STRUCTURE_BITS


def evaluate_complete(prediction, truth, condition, run):
    """Truth uses zero-based structure, U degree, M position/degree, and I position.

    An unavailable attribute of a truly present component counts as incorrect.
    Attribute metrics are conditional on the component being truly present.
    """
    y = np.asarray(truth["structure_index"], dtype=np.int64)
    pred = np.asarray(prediction["structure_index"], dtype=np.int64)
    if y.ndim != 1 or y.shape != pred.shape or np.any((y < 0) | (y >= 8)) or np.any((pred < 0) | (pred >= 8)):
        raise ValueError("Truth and predictions must have aligned valid structures")
    weights = condition_run_weights(np.ones(len(y), dtype=bool), condition, run)
    exact = pred == y
    structure_accuracy = float(np.sum(weights * exact))
    bits = STRUCTURE_BITS[y]
    predicted_bits = STRUCTURE_BITS[pred]
    attributes = {}
    fields = (("u", 0, "u_degree", "u_prediction", 3),
              ("m_position", 1, "m_position", "m_position_prediction", 3),
              ("m_degree", 1, "m_degree", "m_degree_prediction", 2),
              ("m_joint", 1, "m_joint", "m_joint_prediction", 6),
              ("i", 2, "i_position", "i_prediction", 3))
    values = dict(truth)
    values["m_joint"] = 2 * np.asarray(truth["m_position"]) + np.asarray(truth["m_degree"])
    for name, bit, target, field, classes in fields:
        ready_key = ("m" if bit == 1 else name) + "_ready"
        ready = np.asarray(prediction[ready_key], dtype=bool) & (predicted_bits[:, bit] == 1)
        active = bits[:, bit] == 1
        correct = ready & (np.asarray(prediction[field]) == np.asarray(values[target]))
        exact &= (~active) | correct
        attributes[name] = confusion_metrics(values[target], prediction[field], ready,
                                              active, condition, run, classes)
    return {"structure_accuracy": structure_accuracy,
            "complete_diagnosis_accuracy": float(np.sum(weights * exact)),
            "attributes": attributes}
