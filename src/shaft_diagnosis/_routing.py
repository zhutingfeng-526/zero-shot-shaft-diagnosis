"""Frozen physical presence features and source-calibrated structure routing."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np

STRUCTURE_BITS = np.asarray(
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)), dtype=np.int8,
)


FS_HZ = 5000.0


WINDOW_SAMPLES = 6000


HORIZON_WINDOWS = 281


EPS = 1.0e-8


SOURCE_FOLDS = 3


SOURCE_GATE = {
    "auroc_min": 0.80,
    "balanced_accuracy_min": 0.70,
    "worst_fold_balanced_accuracy_min": 0.60,
    "wrong_switch_rate_max": 0.01,
}


SENSOR_CONTRACT = (
    "B1_x", "B1_y_vertical", "B2_x", "B2_y_vertical", "B3_x",
    "B3_y_vertical", "orbit_x", "orbit_y_vertical", "sound_A_AC_OUT",
    "sound_B_AC_OUT",
)


PHYSICS_FEATURES = {
    "u": (
        "orbit_1x_fraction", "vibration_1x_fraction", "orbit_1x_to_2x",
        "station_1x_uniformity", "station_1x_phase_coherence",
    ),
    "m": (
        "vertical_2x_to_1x", "vertical_horizontal_1x", "vertical_station_contrast",
        "orbit_yx_1x", "vertical_phase_dispersion",
    ),
    "r": (
        "vibration_crest", "vibration_kurtosis", "acoustic_crest",
        "vibration_high_fraction", "difference_impulsiveness",
    ),
}


BRANCHES = ("u", "m", "r")


FAMILIES = ("N", "U", "M", "R", "UM", "UR", "MR", "UMR")


def normalize_probability(value: Any) -> np.ndarray:
    probability = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise RuntimeError("Invalid probability matrix: %s" % (probability.shape,))
    total = np.sum(probability, axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise RuntimeError("Probability row has no mass")
    return probability / total


def canonical_family(value: Any) -> np.ndarray:
    output: List[str] = []
    for item in np.asarray(value).reshape(-1).tolist():
        current = bytes(item).decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        upper = current.upper().replace("NORMAL", "N")
        output.append("".join(letter for letter in "UMR" if letter in upper) or "N")
    return np.asarray(output, dtype="U4")


def causal_temporal_scores(
    probability: np.ndarray, segment_id: np.ndarray, rank: np.ndarray,
    endpoints: np.ndarray, mode: str,
) -> np.ndarray:
    p = normalize_probability(probability)
    segment = np.asarray(segment_id, dtype=np.int64)
    local_rank = np.asarray(rank, dtype=np.int64)
    ids = np.asarray(endpoints, dtype=np.int64)
    if len(p) != len(segment) or len(segment) != len(local_rank):
        raise RuntimeError("Temporal-score arrays are not aligned")
    if mode == "mean_log":
        values = np.log(np.maximum(p, 1.0e-6))
    elif mode == "median_log_odds":
        values = np.log(np.maximum(p, 1.0e-6)) - np.log(np.maximum(1.0 - p, 1.0e-6))
    else:
        raise RuntimeError("Unknown temporal mode: %s" % mode)
    output = np.empty((len(ids), p.shape[1]), dtype=np.float64)
    if mode == "mean_log":
        endpoint_lookup = {int(value): position for position, value in enumerate(ids.tolist())}
        for current in np.unique(segment).tolist():
            group = np.flatnonzero(segment == int(current))
            cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values[group], axis=0)))
            for local_endpoint, global_endpoint in enumerate(group.tolist()):
                out_index = endpoint_lookup.get(int(global_endpoint))
                if out_index is None:
                    continue
                local_start = local_endpoint - HORIZON_WINDOWS + 1
                if local_start < 0 or local_rank[global_endpoint] - local_rank[group[local_start]] != HORIZON_WINDOWS - 1:
                    raise RuntimeError("Temporal mean crosses a record boundary")
                output[out_index] = (
                    cumulative[local_endpoint + 1] - cumulative[local_start]
                ) / float(HORIZON_WINDOWS)
    else:
        for out_index, endpoint in enumerate(ids.tolist()):
            start = int(endpoint) - HORIZON_WINDOWS + 1
            if start < 0 or segment[start] != segment[endpoint] or local_rank[endpoint] - local_rank[start] != HORIZON_WINDOWS - 1:
                raise RuntimeError("Temporal median crosses a record boundary")
            output[out_index] = np.median(values[start:endpoint + 1], axis=0)
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite temporal Level1 score")
    return output


def physics_feature_matrix(raw: np.ndarray, rpm: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    speed = np.asarray(rpm, dtype=np.float64).reshape(-1)
    if value.ndim != 3 or value.shape[1:] != (10, WINDOW_SAMPLES) or len(value) != len(speed):
        raise RuntimeError("Unexpected physics batch shape: %s" % (value.shape,))
    centered = value - np.mean(value, axis=2, keepdims=True)
    second = np.mean(centered * centered, axis=2) + EPS
    rms = np.sqrt(second)
    crest = np.max(np.abs(centered), axis=2) / rms
    kurtosis = np.mean(centered ** 4, axis=2) / (second ** 2) - 3.0
    difference = np.diff(centered[:, :6], axis=2)
    diff_rms = np.sqrt(np.mean(difference * difference, axis=2) + EPS)
    diff_crest = np.max(np.abs(difference), axis=2) / diff_rms

    window = np.hanning(WINDOW_SAMPLES).astype(np.float64)
    spectrum = np.fft.rfft(centered * window[None, None, :], axis=2)
    power = np.abs(spectrum) ** 2
    frequency = np.fft.rfftfreq(WINDOW_SAMPLES, 1.0 / FS_HZ)
    rotation = speed / 60.0

    def variable_band(low: float, high: float) -> np.ndarray:
        return (frequency[None, :] >= low * rotation[:, None]) & (frequency[None, :] <= high * rotation[:, None])

    def energy(mask: np.ndarray, channels: Sequence[int]) -> np.ndarray:
        selected = power[:, np.asarray(channels, dtype=np.int64), :]
        return np.sum(selected * mask[:, None, :], axis=2) + EPS

    one_mask = variable_band(0.90, 1.10)
    two_mask = variable_band(1.80, 2.20)
    harmonic_mask = variable_band(0.50, 4.50)
    high_mask = np.broadcast_to(((frequency >= 100.0) & (frequency <= 1000.0))[None, :], one_mask.shape)
    one = energy(one_mask, range(10))
    two = energy(two_mask, range(10))
    harmonic = energy(harmonic_mask, range(10))
    high_energy = energy(high_mask, range(6))
    total_vibration = np.sum(power[:, :6], axis=2) + EPS
    one_complex = np.sum(spectrum * one_mask[:, None, :], axis=2) / np.maximum(np.sum(one_mask, axis=1)[:, None], 1.0)

    station_one = np.stack((one[:, 0] + one[:, 1], one[:, 2] + one[:, 3], one[:, 4] + one[:, 5]), axis=1)
    station_fraction = station_one / np.stack(
        (harmonic[:, 0] + harmonic[:, 1], harmonic[:, 2] + harmonic[:, 3], harmonic[:, 4] + harmonic[:, 5]), axis=1
    )
    vertical_complex = one_complex[:, (1, 3, 5)]
    vertical_unit = vertical_complex / np.maximum(np.abs(vertical_complex), EPS)
    phase_coherence = np.abs(np.mean(vertical_unit, axis=1))
    station_cv = np.std(station_fraction, axis=1) / np.maximum(np.mean(station_fraction, axis=1), EPS)

    features = np.column_stack((
        np.log((one[:, 6] + one[:, 7] + EPS) / (harmonic[:, 6] + harmonic[:, 7] + EPS)),
        np.log((np.sum(one[:, :6], axis=1) + EPS) / (np.sum(harmonic[:, :6], axis=1) + EPS)),
        np.log((one[:, 6] + one[:, 7] + EPS) / (two[:, 6] + two[:, 7] + EPS)),
        1.0 / (1.0 + station_cv),
        phase_coherence,
        np.log((np.sum(two[:, (1, 3, 5)], axis=1) + EPS) / (np.sum(one[:, (1, 3, 5)], axis=1) + EPS)),
        np.log((np.sum(one[:, (1, 3, 5)], axis=1) + EPS) / (np.sum(one[:, (0, 2, 4)], axis=1) + EPS)),
        np.std(station_one, axis=1) / np.maximum(np.mean(station_one, axis=1), EPS),
        np.log((one[:, 7] + EPS) / (one[:, 6] + EPS)),
        1.0 - phase_coherence,
        np.log1p(np.max(crest[:, :6], axis=1)),
        np.arcsinh(np.max(kurtosis[:, :6], axis=1)),
        np.log1p(np.max(crest[:, 8:10], axis=1)),
        np.log((np.sum(high_energy, axis=1) + EPS) / (np.sum(total_vibration, axis=1) + EPS)),
        np.log1p(np.max(diff_crest, axis=1)),
    )).astype(np.float32)
    if features.shape != (len(value), 15) or not np.isfinite(features).all():
        raise RuntimeError("Physics feature extraction produced invalid values")
    return features


def feature_names_flat() -> Tuple[str, ...]:
    return tuple(name for branch in BRANCHES for name in PHYSICS_FEATURES[branch])


def robust_center_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1.0e-6, scale, np.maximum(fallback, 1.0e-3))
    return center, scale


def equal_group_class_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    group = np.asarray(groups).astype(str)
    weights = np.zeros(len(labels), dtype=np.float64)
    for current in sorted(set(group.tolist())):
        ids = np.flatnonzero(group == current)
        weights[ids] = 1.0 / float(len(ids))
    for label in (0, 1):
        active = labels == label
        mass = float(np.sum(weights[active]))
        if mass <= 0.0:
            raise RuntimeError("Binary source class is absent")
        weights[active] *= 0.5 / mass
    return weights / np.sum(weights)


def fit_binary_logistic(x: np.ndarray, y: np.ndarray, groups: np.ndarray, ridge: float) -> Dict[str, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    center, scale = robust_center_scale(values)
    z = np.clip((values - center) / scale, -8.0, 8.0)
    design = np.column_stack((np.ones(len(z)), z))
    sample_weight = equal_group_class_weights(labels.astype(np.int64), groups)
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.diag(np.asarray([0.0] + [float(ridge)] * z.shape[1], dtype=np.float64))
    for _iteration in range(80):
        linear = np.clip(design @ coefficient, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        gradient = design.T @ (sample_weight * (probability - labels)) + penalty @ coefficient
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = design.T @ (curvature[:, None] * design) + penalty + np.eye(design.shape[1]) * 1.0e-9
        step = np.linalg.solve(hessian, gradient)
        coefficient -= step
        if float(np.max(np.abs(step))) < 1.0e-8:
            break
    return {"center": center, "scale": scale, "coefficient": coefficient}


def logistic_score(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    z = np.clip((values - np.asarray(model["center"])) / np.asarray(model["scale"]), -8.0, 8.0)
    return np.column_stack((np.ones(len(z)), z)) @ np.asarray(model["coefficient"])


def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(y, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1) + 1.0
        position = end
    positive = labels == 1
    n_pos = int(np.sum(positive)); n_neg = int(np.sum(~positive))
    if not n_pos or not n_neg:
        raise RuntimeError("AUROC requires both classes")
    return float((np.sum(ranks[positive]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balanced_accuracy(y: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(y, dtype=np.int64)
    prediction = np.asarray(score, dtype=np.float64) >= 0.0
    recalls = [float(np.mean(prediction[labels == label] == bool(label))) for label in (0, 1)]
    return float(np.mean(recalls))


def fit_presence_validators(
    source_features: np.ndarray, source_structures: np.ndarray,
    source_groups: np.ndarray, folds: np.ndarray, ridge: float,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    structures = canonical_family(source_structures)
    feature = np.asarray(source_features, dtype=np.float64)
    models: Dict[str, Any] = {}
    oof = np.empty((len(feature), 3), dtype=np.float64)
    report: Dict[str, Any] = {}
    offsets = {"u": (0, 5), "m": (5, 10), "r": (10, 15)}
    for branch_index, branch in enumerate(BRANCHES):
        left, right = offsets[branch]
        x = feature[:, left:right]
        y = np.asarray([int(branch.upper() in family) for family in structures], dtype=np.int64)
        fold_metrics: List[float] = []
        for fold in range(SOURCE_FOLDS):
            train = folds != fold; held = folds == fold
            if len(np.unique(y[train])) != 2 or len(np.unique(y[held])) != 2:
                raise RuntimeError("Grouped source fold lacks a binary class: %s/%d" % (branch, fold))
            model = fit_binary_logistic(x[train], y[train], np.asarray(source_groups)[train], ridge)
            oof[held, branch_index] = logistic_score(model, x[held])
            fold_metrics.append(balanced_accuracy(y[held], oof[held, branch_index]))
        final_model = fit_binary_logistic(x, y, source_groups, ridge)
        oof_median = float(np.median(oof[:, branch_index]))
        scale = float(1.4826 * np.median(np.abs(oof[:, branch_index] - oof_median)))
        scale = max(scale, float(np.median(np.abs(oof[:, branch_index]))), 1.0e-3)
        metrics = {
            "auroc": rank_auc(y, oof[:, branch_index]),
            "balanced_accuracy": balanced_accuracy(y, oof[:, branch_index]),
            "fold_balanced_accuracy": fold_metrics,
            "worst_fold_balanced_accuracy": float(min(fold_metrics)),
            "zero_is_presence_decision_boundary": True,
        }
        metrics["gate_pass"] = bool(
            metrics["auroc"] >= SOURCE_GATE["auroc_min"]
            and metrics["balanced_accuracy"] >= SOURCE_GATE["balanced_accuracy_min"]
            and metrics["worst_fold_balanced_accuracy"] >= SOURCE_GATE["worst_fold_balanced_accuracy_min"]
        )
        models[branch] = {**final_model, "oof_center": 0.0, "oof_scale": scale}
        report[branch] = metrics
    standardized = np.column_stack([
        np.clip((oof[:, index] - models[branch]["oof_center"]) / models[branch]["oof_scale"], -3.0, 3.0)
        for index, branch in enumerate(BRANCHES)
    ])
    report["all_gate_pass"] = bool(all(report[branch]["gate_pass"] for branch in BRANCHES))
    return models, standardized, report


def apply_presence_validators(models: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    offsets = {"u": (0, 5), "m": (5, 10), "r": (10, 15)}
    columns: List[np.ndarray] = []
    for branch in BRANCHES:
        left, right = offsets[branch]
        model = models[branch]
        score = logistic_score(model, values[:, left:right])
        columns.append(np.clip((score - float(model["oof_center"])) / float(model["oof_scale"]), -3.0, 3.0))
    return np.column_stack(columns)


def compatibility_scores(standardized_presence: np.ndarray) -> np.ndarray:
    z = np.asarray(standardized_presence, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 3:
        raise RuntimeError("Presence scores must have three columns")
    signs = 2.0 * STRUCTURE_BITS.astype(np.float64) - 1.0
    return 0.5 * (z @ signs.T)


def robust_margin_scale(scores: np.ndarray) -> float:
    order = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    margin = order[:, -1] - order[:, -2]
    median = float(np.median(margin))
    mad = float(1.4826 * np.median(np.abs(margin - median)))
    return max(mad, float(np.median(np.abs(margin))), 0.05)


def same_direction_mask(base: np.ndarray, proposed: np.ndarray, z: np.ndarray) -> np.ndarray:
    base_bits = STRUCTURE_BITS[np.asarray(base, dtype=np.int64)]
    proposed_bits = STRUCTURE_BITS[np.asarray(proposed, dtype=np.int64)]
    changed = base_bits != proposed_bits
    direction = np.where(proposed_bits > base_bits, np.asarray(z) >= 0.0, np.asarray(z) <= 0.0)
    return np.all((~changed) | direction, axis=1)


def quantile_higher(values: np.ndarray, q: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if not len(array):
        return 0.0
    index = int(math.ceil(float(q) * len(array)) - 1)
    return float(array[min(max(index, 0), len(array) - 1)])


def calibrate_safe_switch(
    temporal_scores: np.ndarray, standardized_presence: np.ndarray,
    truth_index: np.ndarray, weight: float, temporal_scale: float,
) -> Dict[str, Any]:
    temporal = np.asarray(temporal_scores, dtype=np.float64) / float(temporal_scale)
    compatibility = compatibility_scores(standardized_presence)
    top2 = np.argsort(-temporal, axis=1, kind="mergesort")[:, :2]
    base = top2[:, 0]
    combined = temporal[np.arange(len(temporal))[:, None], top2] + float(weight) * compatibility[np.arange(len(temporal))[:, None], top2]
    local = np.argmax(combined, axis=1)
    proposed = top2[np.arange(len(top2)), local]
    advantage = combined[np.arange(len(top2)), local] - combined[:, 0]
    direction = same_direction_mask(base, proposed, standardized_presence)
    dangerous = (base == np.asarray(truth_index)) & (proposed != base) & direction
    tau = quantile_higher(advantage[dangerous], 0.99)
    apply = (proposed != base) & direction & (advantage > tau + 1.0e-12)
    output = base.copy(); output[apply] = proposed[apply]
    protected = base == np.asarray(truth_index)
    wrong_switch = protected & (output != base)
    wrong_rate = float(np.sum(wrong_switch) / max(np.sum(protected), 1))
    if wrong_rate > SOURCE_GATE["wrong_switch_rate_max"] + 1.0e-12:
        raise RuntimeError("Source wrong-switch calibration exceeded its contract")
    return {
        "tau": tau, "weight": float(weight), "route": output,
        "base_route": base, "proposed_route": proposed,
        "advantage": advantage, "direction_pass": direction,
        "applied": apply, "source_wrong_switch_rate": wrong_rate,
    }


def apply_safe_switch(
    temporal_scores: np.ndarray, standardized_presence: np.ndarray,
    weight: float, temporal_scale: float, tau: float,
) -> Dict[str, np.ndarray]:
    temporal = np.asarray(temporal_scores, dtype=np.float64) / float(temporal_scale)
    compatibility = compatibility_scores(standardized_presence)
    top2 = np.argsort(-temporal, axis=1, kind="mergesort")[:, :2]
    base = top2[:, 0]
    combined = temporal[np.arange(len(temporal))[:, None], top2] + float(weight) * compatibility[np.arange(len(temporal))[:, None], top2]
    local = np.argmax(combined, axis=1)
    proposed = top2[np.arange(len(top2)), local]
    advantage = combined[np.arange(len(top2)), local] - combined[:, 0]
    direction = same_direction_mask(base, proposed, standardized_presence)
    applied = (proposed != base) & direction & (advantage > float(tau) + 1.0e-12)
    route = base.copy(); route[applied] = proposed[applied]
    return {"route": route, "proposed": proposed, "advantage": advantage, "direction": direction, "applied": applied}
