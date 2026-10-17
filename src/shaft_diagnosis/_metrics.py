"""Condition- and recording-balanced evaluation primitives."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def condition_run_weights(mask: Any, condition: Any, run: Any) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    cond = np.asarray(condition, dtype=np.int64)
    physical_run = np.asarray(run, dtype=np.int64)
    if not np.any(active) or not (len(active) == len(cond) == len(physical_run)):
        raise RuntimeError("Cannot construct condition/run weights")
    weights = np.zeros(len(active), dtype=np.float64)
    conditions = np.unique(cond[active])
    for current_condition in conditions.tolist():
        condition_mask = active & (cond == int(current_condition))
        runs = np.unique(physical_run[condition_mask])
        for current_run in runs.tolist():
            ids = np.where(condition_mask & (physical_run == int(current_run)))[0]
            weights[ids] = 1.0 / float(len(conditions) * len(runs) * len(ids))
    if abs(float(np.sum(weights)) - 1.0) > 1.0e-10:
        raise RuntimeError("weights do not sum to one")
    return weights


def confusion_metrics(
    truth: Any, prediction: Any, ready: Any, active: Any,
    condition: Any, run: Any, classes: int,
) -> Dict[str, Any]:
    y = np.asarray(truth, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    enabled = np.asarray(ready, dtype=bool)
    mask = np.asarray(active, dtype=bool)
    if not np.any(mask):
        return {
            "n": 0, "accuracy": 0.0, "macro_f1": 0.0, "macro_recall": 0.0,
            "min_recall": 0.0, "coverage": 0.0, "confusion": [], "per_class": {},
        }
    weights = condition_run_weights(mask, condition, run)
    matrix = np.zeros((int(classes), int(classes) + 1), dtype=np.float64)
    for index in np.where(mask)[0].tolist():
        target = int(y[index])
        if target < 0 or target >= int(classes):
            raise RuntimeError("Truth class outside metric range")
        column = int(pred[index]) if enabled[index] and 0 <= int(pred[index]) < int(classes) else int(classes)
        matrix[target, column] += float(weights[index])
    recalls: List[float] = []
    f1s: List[float] = []
    per_class: Dict[str, Any] = {}
    for target in range(int(classes)):
        support = float(np.sum(matrix[target]))
        true_positive = float(matrix[target, target])
        predicted = float(np.sum(matrix[:, target]))
        recall = true_positive / support if support > 0.0 else 0.0
        precision = true_positive / predicted if predicted > 0.0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        recalls.append(recall); f1s.append(f1)
        per_class[str(target)] = {
            "support": support, "predicted": predicted, "true_positive": true_positive,
            "recall": recall, "precision": precision, "f1": f1,
        }
    return {
        "n": int(np.sum(mask)), "accuracy": float(np.trace(matrix[:, :classes])),
        "macro_f1": float(np.mean(f1s)), "macro_recall": float(np.mean(recalls)),
        "min_recall": float(np.min(recalls)),
        "coverage": float(np.sum(weights[mask] * enabled[mask].astype(np.float64))),
        "confusion": matrix.tolist(), "per_class": per_class,
    }
