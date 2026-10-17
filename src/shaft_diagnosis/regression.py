"""Reference checks without bundled experimental signals or predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation import evaluate_complete
from .integration import aggregate_m_context
from . import _routing
from . import structure_decoder as decoder


TRUTH_FIELDS = ("structure_index", "u_degree", "m_position", "m_degree", "i_position")
PREDICTION_FIELDS = ("structure_index", "u_prediction", "m_position_prediction",
                     "m_degree_prediction", "m_joint_prediction", "i_prediction",
                     "u_ready", "m_ready", "i_ready")
METADATA_FIELDS = ("condition", "run", "endpoint_key")
BUNDLE_FIELDS = tuple("truth_" + k for k in TRUTH_FIELDS) + tuple(
    "prediction_" + k for k in PREDICTION_FIELDS) + METADATA_FIELDS
REFERENCE_VERSION = 1


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def canonical_digest(value):
    """Hash shape and little-endian int64 values, independent of NPZ storage."""
    array = np.ascontiguousarray(value, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"shaft-diagnosis-int64-v1\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_endpoint_bundle(path):
    """Read the fixed numeric schema; never load pickle/object arrays.

    ``endpoint_key`` is an ordered (N,2) integer record/window key supplied
    privately with authorized data. The public reference contains only its hash.
    """
    with np.load(str(path), allow_pickle=False) as archive:
        if set(archive.files) != set(BUNDLE_FIELDS):
            missing = sorted(set(BUNDLE_FIELDS) - set(archive.files))
            extra = sorted(set(archive.files) - set(BUNDLE_FIELDS))
            raise ValueError("Endpoint bundle fields differ: missing=%s, extra=%s" % (missing, extra))
        arrays = {key: archive[key] for key in BUNDLE_FIELDS}
    return validate_endpoint_bundle(arrays)


def validate_endpoint_bundle(arrays):
    """Reject misaligned, fractional, empty, or inconsistent endpoint arrays."""
    if set(arrays) != set(BUNDLE_FIELDS):
        raise ValueError("Endpoint bundle has an incomplete or unexpected schema")
    n = np.asarray(arrays["truth_structure_index"]).size
    if n == 0:
        raise ValueError("An endpoint bundle must not be empty")
    canonical = {}
    for key in BUNDLE_FIELDS:
        value = np.asarray(arrays[key])
        shape = (n, 2) if key == "endpoint_key" else (n,)
        if value.shape != shape or value.dtype.kind not in "biu":
            raise ValueError("%s must be an aligned integer array of shape %s" % (key, shape))
        if value.dtype.kind == "u" and value.size and np.max(value) > np.iinfo(np.int64).max:
            raise ValueError("%s exceeds int64 range" % key)
        canonical[key] = np.asarray(value, dtype=np.int64)
    if len(np.unique(canonical["endpoint_key"], axis=0)) != n:
        raise ValueError("Endpoint keys must be unique")
    for prefix in ("truth_", "prediction_"):
        route = canonical[prefix + "structure_index"]
        if np.any((route < 0) | (route > 7)):
            raise ValueError("Invalid structure index")
    ranges = {"truth_u_degree": 3, "truth_m_position": 3, "truth_m_degree": 2,
              "truth_i_position": 3, "prediction_u_prediction": 3,
              "prediction_m_position_prediction": 3, "prediction_m_degree_prediction": 2,
              "prediction_m_joint_prediction": 6, "prediction_i_prediction": 3}
    for key, classes in ranges.items():
        if np.any((canonical[key] < -1) | (canonical[key] >= classes)):
            raise ValueError("Invalid class value for " + key)
    bits = _routing.STRUCTURE_BITS[canonical["truth_structure_index"]]
    for key, bit in (("u_degree", 0), ("m_position", 1), ("m_degree", 1), ("i_position", 2)):
        value = canonical["truth_" + key]
        if np.any((bits[:, bit] == 1) & (value < 0)) or np.any((bits[:, bit] == 0) & (value != -1)):
            raise ValueError("Truth attributes disagree with truth structure: " + key)
    for key in ("u_ready", "m_ready", "i_ready"):
        if not np.isin(canonical["prediction_" + key], (0, 1)).all():
            raise ValueError("Readiness flags must be Boolean")
    position = canonical["prediction_m_position_prediction"]
    degree = canonical["prediction_m_degree_prediction"]
    joint = np.where((position >= 0) & (degree >= 0), 2 * position + degree, -1)
    if not np.array_equal(joint, canonical["prediction_m_joint_prediction"]):
        raise ValueError("M joint prediction disagrees with position/degree predictions")
    return canonical


def summarize_metrics(result):
    summary = {key: result[key] for key in ("structure_accuracy", "complete_diagnosis_accuracy")}
    for branch, metrics in result["attributes"].items():
        for name in ("accuracy", "macro_f1", "coverage"):
            summary[branch + "." + name] = metrics[name]
    return summary


def score_endpoint_bundle(arrays):
    arrays = validate_endpoint_bundle(arrays)
    truth = {key: arrays["truth_" + key] for key in TRUTH_FIELDS}
    prediction = {key: arrays["prediction_" + key] for key in PREDICTION_FIELDS}
    return summarize_metrics(evaluate_complete(prediction, truth, arrays["condition"], arrays["run"]))


def check_endpoints(arrays, reference, mode="frozen-endpoints"):
    """Check authorized endpoint results, not neural inference execution.

    Frozen mode re-scores original predictions. Current mode compares a caller's
    newly inferred endpoints. Both require exact input identities and decisions;
    matching rounded aggregate metrics alone is never a pass.
    """
    if mode not in ("frozen-endpoints", "current-endpoints"):
        raise ValueError("Unknown endpoint regression mode")
    arrays = validate_endpoint_bundle(arrays)
    required = set(BUNDLE_FIELDS)
    if set(reference.get("array_sha256", {})) != required:
        raise ValueError("The reference must bind every endpoint array")
    if set(reference.get("metrics", {})) != set(score_endpoint_bundle(arrays)):
        raise ValueError("The reference has an incomplete metric schema")
    fingerprints = {key: canonical_digest(value) for key, value in arrays.items()}
    mismatches = [key for key in BUNDLE_FIELDS if fingerprints[key] != reference["array_sha256"][key]]
    metrics = score_endpoint_bundle(arrays)
    differences = {key: float(metrics[key]) - float(reference["metrics"][key]) for key in metrics}
    if not all(np.isfinite(list(differences.values()))):
        raise ValueError("The reference contains non-finite metrics")
    maximum_error = max(abs(value) for value in differences.values())
    n = len(arrays["condition"])
    passed = not mismatches and n == reference["n"] and maximum_error <= 1.0e-12
    return {"mode": mode, "status": "passed" if passed else "failed", "passed": passed,
            "scope": "Endpoint decisions and condition/run-balanced re-scoring; no raw-signal inference is run.",
            "rows": n, "array_hash_mismatches": mismatches, "metrics": metrics,
            "metric_difference": differences, "maximum_metric_absolute_error": maximum_error,
            "neural_inference_executed": False, "full_raw_data_reproduction": False}


def synthetic_temporal_inputs():
    """Deterministic artificial histories; no measured signals or record keys."""
    rank = np.arange(281)
    p = np.empty((3, 281, 3), dtype=np.float64)
    d = np.empty((3, 281, 2), dtype=np.float64)
    q = np.empty((3, 281), dtype=np.uint8)
    for row in range(3):
        p[row] = np.asarray((0.2, 0.3, 0.5))[None, :] + 0.01 * ((rank[:, None] + np.arange(3) + row) % 7)
        d[row, :, 1] = 0.2 + 0.1 * row + 0.01 * (rank % 11)
        d[row, :, 0] = 1.0 - d[row, :, 1]
        q[row] = (rank + row) % 2
    p[1, ::10] = (1.0e-30, 1.0e-30, 1.0)
    d[2, ::13] = (1.0e-30, 1.0)
    return p, d, q


def check_synthetic(repository_root):
    """Replay original-function numerical fixtures, not experimental inference."""
    root = Path(repository_root)
    reference = read_json(root / "configs/regression/synthetic_reference.json")
    if reference.get("schema_version") != REFERENCE_VERSION:
        raise ValueError("Unsupported synthetic reference schema")
    for relative, expected in reference["bound_assets_sha256"].items():
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Synthetic reference asset checksum mismatch: " + relative)
    calibration = read_json(root / "models/structure/calibration.json")
    margins = np.asarray(reference["decoder_input"]["margins"], dtype=np.float32)
    count = np.asarray(reference["decoder_input"]["count_prediction"], dtype=np.int64)
    epochs = reference["decoder_input"]["epochs"]
    votes = {int(epoch): decoder.structure_indices(decoder.decode_margin_strict_v12(
        margins[index], count[index], calibration["decoder"])) for index, epoch in enumerate(epochs)}
    reliability = {int(key): value for key, value in calibration["source_reliability"].items()}
    route, probability, entropy, tie, matrix = decoder.equal_consensus(votes, epochs, reliability)
    actual = {"checkpoint_decisions": matrix, "structure_index": route,
              "probabilities": probability, "entropy": entropy, "tie": tie.astype(np.int64)}
    p, d, q = synthetic_temporal_inputs()
    position, degree = aggregate_m_context(p, d, q)
    actual.update(m_position_probability=position, m_degree_probability=degree)
    integration_config = read_json(root / "configs/integration.json")
    switch = reference["switch_input"]
    output = _routing.apply_safe_switch(switch["temporal_score"], switch["presence"],
        integration_config["presence_weight"], integration_config["temporal_scale"],
        integration_config["safe_switch_margin"])
    for key, value in output.items():
        actual["switch_" + key] = value
    expected = reference["expected"]
    if set(actual) != set(expected):
        raise ValueError("Synthetic reference output fields differ")
    checks = {}
    for key, value in actual.items():
        target = np.asarray(expected[key])
        value = np.asarray(value)
        same_shape = value.shape == target.shape
        if value.dtype.kind in "biu":
            passed = same_shape and np.array_equal(value, target)
            tolerance = 0.0
        else:
            tolerance = 1.0e-7 if value.dtype == np.float32 else 1.0e-12
            passed = same_shape and np.allclose(value, target, rtol=0.0, atol=tolerance)
        error = float(np.max(np.abs(value.astype(np.float64) - target))) if same_shape and value.size else None
        checks[key] = {"passed": bool(passed), "maximum_absolute_error": error,
                       "absolute_tolerance": tolerance}
    arrays = validate_endpoint_bundle({key: np.asarray(value, dtype=np.int64)
                                      for key, value in reference["scoring_input"].items()})
    metric_result = check_endpoints(arrays, reference["scoring_reference"])
    passed = all(value["passed"] for value in checks.values()) and metric_result["passed"]
    return {"mode": "synthetic", "status": "passed" if passed else "failed", "passed": passed,
            "scope": "Artificial decoder, consensus, temporal aggregation, safe-switch and scoring fixtures generated by original frozen functions.",
            "experimental_data_used": False, "neural_inference_executed": False,
            "full_raw_data_reproduction": False, "checks": checks, "scoring": metric_result}
