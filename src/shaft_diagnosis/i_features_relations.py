"""Numerical relations functions."""
from __future__ import annotations
import hashlib
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
from scipy import signal
from scipy.signal.windows import dpss

FS_HZ = 5000.0


CHANNEL_COUNT = 10


TIME_SCALES_MS = (4, 12, 32, 80)


FIXED_BANDS_HZ: Tuple[Tuple[float, float], ...] = (
    (20.0, 40.0),
    (40.0, 80.0),
    (80.0, 160.0),
    (160.0, 315.0),
    (315.0, 630.0),
    (630.0, 1000.0),
    (1000.0, 1500.0),
    (1500.0, 2000.0),
    (618.0, 837.0),
    (751.0, 1016.0),
    (822.0, 1112.0),
)


SHORT_SPECTRAL_SAMPLES = 256


LONG_SPECTRAL_SAMPLES = 512


DPSS_NW = 2.5


DPSS_TAPERS = 3


EPS = 1.0e-8


B1 = (0, 1)


B2 = (2, 3)


B3 = (4, 5)


ORBIT = (6, 7)


ACOUSTIC_A = (8,)


NEIGHBOR = B1 + B3


BASE_METRIC_NAMES = (
    "cond_operator_share",
    "cond_schur_share",
    "cond_operator_schur_agreement",
    "cond_residual_B2_B1_ratio",
    "cond_vertical_share",
    "cond_anisotropy",
    "orbit_coherence_B2_minus_B1",
    "orbit_transfer_B2_B1_log_ratio",
    "orbit_B1_B2_subspace_alignment",
    "acousticA_coherence_B2_minus_B1",
    "acousticA_transfer_B2_B1_log_ratio",
    "orbit_acoustic_ratio_agreement",
    "B1_B2_coherence",
    "B2_B1_energy_log_ratio",
)


SUMMARY_SUFFIXES = ("time_median", "spectral_median", "spectral_iqr", "long_minus_short")


SUMMARY_FEATURE_NAMES = tuple(
    "%s__%s" % (metric, suffix)
    for metric in BASE_METRIC_NAMES
    for suffix in SUMMARY_SUFFIXES
)


CONDITIONAL_METRICS = BASE_METRIC_NAMES[:6]


ORBIT_METRICS = BASE_METRIC_NAMES[6:9]


ACOUSTIC_METRICS = BASE_METRIC_NAMES[9:12]


DIRECT_METRICS = BASE_METRIC_NAMES[12:]


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        bytes(value).decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).tolist()
    ], dtype=object)


def normalize_probability(value: np.ndarray) -> np.ndarray:
    probability = np.asarray(value, dtype=np.float64)
    probability = np.maximum(np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = np.sum(probability, axis=-1, keepdims=True)
    return (probability / np.maximum(total, 1.0e-12)).astype(np.float32)


def sigmoid(value: np.ndarray) -> np.ndarray:
    current = np.clip(np.asarray(value, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-current))


def safe_inverse(value: np.ndarray, ridge: float = 1.0e-6) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    dimension = matrix.shape[-1]
    scale = np.real(np.trace(matrix, axis1=-2, axis2=-1)) / max(1, dimension)
    regularized = matrix + (float(ridge) * np.maximum(scale, EPS))[..., None, None] * np.eye(dimension)
    return np.linalg.pinv(regularized, rcond=1.0e-8)


def hermitian(value: np.ndarray) -> np.ndarray:
    current = np.asarray(value)
    return 0.5 * (current + np.swapaxes(current.conj(), -1, -2))


def normalize_psd(value: np.ndarray) -> np.ndarray:
    current = hermitian(np.asarray(value, dtype=np.complex128))
    trace = np.maximum(np.real(np.trace(current, axis1=-2, axis2=-1)), EPS)
    return (current / trace[..., None, None]).astype(np.complex64)


def submatrix(value: np.ndarray, rows: Sequence[int], columns: Sequence[int]) -> np.ndarray:
    current = np.asarray(value)
    return np.take(np.take(current, np.asarray(rows, dtype=np.int64), axis=-2), np.asarray(columns, dtype=np.int64), axis=-1)


def canonical_coherence(value: np.ndarray, left: Sequence[int], right: Sequence[int]) -> np.ndarray:
    current = normalize_psd(value).astype(np.complex128)
    sxx = submatrix(current, left, left)
    syy = submatrix(current, right, right)
    sxy = submatrix(current, left, right)
    ex, ux = np.linalg.eigh(hermitian(sxx))
    ey, uy = np.linalg.eigh(hermitian(syy))
    ix = np.einsum("...ij,...j,...kj->...ik", ux, 1.0 / np.sqrt(np.maximum(ex, 1.0e-6)), ux.conj(), optimize=True)
    iy = np.einsum("...ij,...j,...kj->...ik", uy, 1.0 / np.sqrt(np.maximum(ey, 1.0e-6)), uy.conj(), optimize=True)
    whitened = np.einsum("...ij,...jk,...kl->...il", ix, sxy, iy, optimize=True)
    singular = np.linalg.svd(whitened, compute_uv=False)
    return np.clip(np.real(singular[..., 0]) ** 2, 0.0, 1.0).astype(np.float32)


def transfer_norm(value: np.ndarray, left: Sequence[int], reference: Sequence[int]) -> np.ndarray:
    current = normalize_psd(value).astype(np.complex128)
    sxr = submatrix(current, left, reference)
    srr = submatrix(current, reference, reference)
    transfer = np.einsum("...ij,...jk->...ik", sxr, safe_inverse(srr), optimize=True)
    return np.sqrt(np.sum(np.abs(transfer) ** 2, axis=(-2, -1))).astype(np.float32)


def phase_invariant_alignment(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lvalue = np.asarray(left, dtype=np.complex128).reshape(left.shape[:-2] + (-1,))
    rvalue = np.asarray(right, dtype=np.complex128).reshape(right.shape[:-2] + (-1,))
    numerator = np.abs(np.sum(lvalue.conj() * rvalue, axis=-1))
    denominator = np.sqrt(np.sum(np.abs(lvalue) ** 2, axis=-1) * np.sum(np.abs(rvalue) ** 2, axis=-1))
    return (numerator / np.maximum(denominator, EPS)).astype(np.float32)


def numeric_payload_finite(payload: Mapping[str, np.ndarray], keys: Sequence[str]) -> bool:
    return bool(all(
        np.isfinite(np.asarray(payload[key])).all()
        for key in keys
        if np.issubdtype(np.asarray(payload[key]).dtype, np.number)
    ))


def robust_location_scale(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    current = np.asarray(value, dtype=np.float64)
    center = np.median(current, axis=0)
    q25 = np.quantile(current, 0.25, axis=0)
    q75 = np.quantile(current, 0.75, axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1.0e-4)
    return center.astype(np.float32), scale.astype(np.float32)


def band_name(band: Tuple[float, float]) -> str:
    return "%g-%gHz" % (float(band[0]), float(band[1]))


def left_mass_update(base_probability: np.ndarray, delta: np.ndarray, hard_lock_r2: bool) -> np.ndarray:
    base = normalize_probability(base_probability)
    left_mass = np.asarray(base[:, 0] + base[:, 1], dtype=np.float64)
    base_logit = np.log(np.maximum(base[:, 1], 1.0e-8)) - np.log(np.maximum(base[:, 0], 1.0e-8))
    q = sigmoid(base_logit + np.asarray(delta, dtype=np.float64))
    output = np.column_stack((left_mass * (1.0 - q), left_mass * q, base[:, 2]))
    output = normalize_probability(output)
    if bool(hard_lock_r2):
        baseline_prediction = np.argmax(base, axis=1)
        locked = baseline_prediction == 2
        output[locked] = base[locked]
        escaped = (~locked) & (np.argmax(output, axis=1) == 2)
        for index in np.where(escaped)[0].tolist():
            winner = 1 if q[index] >= 0.5 else 0
            other = 1 - winner
            # Preserve P(R2) and left mass while keeping the corrected left winner
            # at least tied with R2; np.argmax then deterministically favors R0/R1.
            winner_mass = min(
                float(left_mass[index]),
                max(float(base[index, 2]), 0.5 * float(left_mass[index] + base[index, 2])),
            )
            output[index, winner] = winner_mass
            output[index, other] = float(left_mass[index]) - winner_mass
        if not np.array_equal(np.argmax(output, axis=1) == 2, baseline_prediction == 2):
            raise RuntimeError("R15 bidirectional R2 decision lock failed")
    return output.astype(np.float32)


MATRIX_KEYS = (
    "event_id",
    "process_feature",
    "time_covariance",
    "spectral_covariance",
    "long_spectral_covariance",
    "pre_event_covariance",
    "jitter_time_covariance",
    "jitter_spectral_covariance",
    "real_orbit_coherence",
    "surrogate_orbit_coherence",
    "real_acoustic_coherence",
    "surrogate_acoustic_coherence",
    "short_valid",
    "long_valid",
    "dense_event",
)


def empty_matrix_extension(n: int) -> Dict[str, np.ndarray]:
    return {
        "event_id": np.empty(int(n), dtype="S64"),
        "process_feature": np.zeros((int(n), 6), dtype=np.float32),
        "time_covariance": np.zeros((int(n), len(TIME_SCALES_MS), CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.complex64),
        "spectral_covariance": np.zeros((int(n), len(FIXED_BANDS_HZ), CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.complex64),
        "long_spectral_covariance": np.zeros((int(n), len(FIXED_BANDS_HZ), CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.complex64),
        "pre_event_covariance": np.zeros((int(n), CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.complex64),
        "jitter_time_covariance": np.zeros((int(n), len(TIME_SCALES_MS), CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.complex64),
        "jitter_spectral_covariance": np.zeros((int(n), len(FIXED_BANDS_HZ), CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.complex64),
        "real_orbit_coherence": np.zeros((int(n), len(FIXED_BANDS_HZ)), dtype=np.float32),
        "surrogate_orbit_coherence": np.zeros((int(n), len(FIXED_BANDS_HZ)), dtype=np.float32),
        "real_acoustic_coherence": np.zeros((int(n), len(FIXED_BANDS_HZ)), dtype=np.float32),
        "surrogate_acoustic_coherence": np.zeros((int(n), len(FIXED_BANDS_HZ)), dtype=np.float32),
        "short_valid": np.zeros(int(n), dtype=np.uint8),
        "long_valid": np.zeros(int(n), dtype=np.uint8),
        "dense_event": np.zeros(int(n), dtype=np.uint8),
    }


def padded_segment(signal: np.ndarray, start: int, width: int) -> np.ndarray:
    value = np.asarray(signal)
    output = np.zeros((value.shape[0], int(width)), dtype=value.dtype)
    left = max(0, int(start))
    right = min(value.shape[1], int(start) + int(width))
    if right <= left:
        return output
    target_left = left - int(start)
    output[:, target_left:target_left + (right - left)] = value[:, left:right]
    return output


def covariance_from_segment(segment: np.ndarray) -> np.ndarray:
    value = np.asarray(segment, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != CHANNEL_COUNT:
        raise RuntimeError("R15 covariance segment shape mismatch: %s" % (value.shape,))
    value = value - np.mean(value, axis=1, keepdims=True)
    covariance = value @ value.conj().T / max(1, value.shape[1])
    covariance += 1.0e-7 * np.eye(CHANNEL_COUNT)
    return normalize_psd(covariance)


def event_seed(event_id: Any) -> int:
    raw = bytes(event_id) if isinstance(event_id, (bytes, np.bytes_)) else str(event_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], byteorder="little", signed=False)


def multitaper_covariance(
    segment: np.ndarray,
    tapers: np.ndarray,
    bands: Sequence[Tuple[float, float]],
    surrogate_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(segment, dtype=np.float64)
    if value.shape != (CHANNEL_COUNT, tapers.shape[1]):
        raise RuntimeError("R15 multitaper segment shape mismatch: %s" % (value.shape,))
    value = value - np.mean(value, axis=1, keepdims=True)
    tapered = np.asarray(tapers, dtype=np.float64)[:, None, :] * value[None, :, :]
    spectrum = np.fft.rfft(tapered, axis=-1).astype(np.complex128)
    frequencies = np.fft.rfftfreq(value.shape[1], d=1.0 / FS_HZ)
    taper_index = np.arange(spectrum.shape[0], dtype=np.float64)[:, None, None] + 1.0
    channel_index = np.arange(CHANNEL_COUNT, dtype=np.float64)[None, :, None] + 1.0
    frequency_index = np.arange(spectrum.shape[-1], dtype=np.float64)[None, None, :] + 1.0
    offset = float(int(surrogate_seed) % 104729) / 104729.0
    phase_fraction = np.mod(
        offset + 0.61803398875 * taper_index * channel_index + 0.41421356237 * channel_index * frequency_index,
        1.0,
    )
    randomized = spectrum * np.exp(2j * np.pi * phase_fraction)
    real_output = np.zeros((len(bands), CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.complex64)
    surrogate_output = np.zeros_like(real_output)
    for position, (low, high) in enumerate(bands):
        bins = np.where((frequencies >= float(low)) & (frequencies < float(high)))[0]
        if not len(bins):
            center = 0.5 * (float(low) + float(high))
            bins = np.asarray([int(np.argmin(np.abs(frequencies - center)))], dtype=np.int64)
        current = spectrum[:, :, bins]
        surrogate = randomized[:, :, bins]
        real_covariance = np.einsum("kcf,kdf->cd", current, current.conj(), optimize=True) / max(1, current.shape[0] * current.shape[2])
        surrogate_covariance = np.einsum("kcf,kdf->cd", surrogate, surrogate.conj(), optimize=True) / max(1, surrogate.shape[0] * surrogate.shape[2])
        real_output[position] = normalize_psd(real_covariance)
        surrogate_output[position] = normalize_psd(surrogate_covariance)
    return real_output, surrogate_output


def reference_observability(covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(covariance)
    orbit = np.maximum(canonical_coherence(value, B1, ORBIT), canonical_coherence(value, B2, ORBIT))
    acoustic = np.maximum(canonical_coherence(value, B1, ACOUSTIC_A), canonical_coherence(value, B2, ACOUSTIC_A))
    return np.asarray(orbit, dtype=np.float32), np.asarray(acoustic, dtype=np.float32)


def extract_fragment_matrices(
    raw: np.ndarray,
    event_ids: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    nuisance: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    value = np.asarray(raw, dtype=np.float32)
    output = empty_matrix_extension(len(event_ids))
    output["event_id"] = np.asarray(event_ids, dtype="S64")
    if not len(event_ids):
        return output, {"events": 0, "short_valid": 0, "long_valid": 0, "dense_events": 0}
    nuisance_value = np.asarray(nuisance, dtype=np.float32)
    if nuisance_value.ndim != 2 or nuisance_value.shape[0] != len(event_ids) or nuisance_value.shape[1] < 6:
        raise RuntimeError("R15 event nuisance shape mismatch: %s" % (nuisance_value.shape,))
    excluded = r14.r13.event_free_mask(value.shape[1], centers, widths, value)
    normalized, _center, _scale = r14.r13.robust_record_normalize(value, excluded)
    analytic = r14.r13.analytic_channels(normalized)
    short_tapers = np.asarray(dpss(SHORT_SPECTRAL_SAMPLES, DPSS_NW, Kmax=DPSS_TAPERS, sym=False, norm=2), dtype=np.float64)
    long_tapers = np.asarray(dpss(LONG_SPECTRAL_SAMPLES, DPSS_NW, Kmax=DPSS_TAPERS, sym=False, norm=2), dtype=np.float64)
    ordered_centers = np.asarray(centers, dtype=np.int64)
    for event_position, center in enumerate(ordered_centers.tolist()):
        neighbor = np.delete(np.abs(ordered_centers - int(center)), event_position)
        nearest = int(np.min(neighbor)) if len(neighbor) else LONG_SPECTRAL_SAMPLES * 2
        dense = nearest < SHORT_SPECTRAL_SAMPLES
        output["dense_event"][event_position] = int(dense)
        output["short_valid"][event_position] = int(not dense)
        output["long_valid"][event_position] = int(nearest >= LONG_SPECTRAL_SAMPLES)
        process_left = max(0, int(center) - 512)
        process_right = min(normalized.shape[1], int(center) + 512)
        station = np.sqrt(
            normalized[0:6:2, process_left:process_right] ** 2
            + normalized[1:6:2, process_left:process_right] ** 2
        )
        if not station.shape[1]:
            raise RuntimeError("R15 empty process segment for event %r" % event_ids[event_position])
        common_trace = np.median(station, axis=0)
        common_peak = float(np.max(common_trace))
        common_energy = float(np.mean(station ** 2))
        current_nuisance = nuisance_value[event_position]
        output["process_feature"][event_position] = np.asarray((
            math.log1p(max(common_peak, 0.0)),
            math.log1p(max(common_energy, 0.0)),
            math.log1p(max(float(current_nuisance[2]), 0.0)),
            math.log1p(max(float(current_nuisance[3]) / FS_HZ, 0.0)),
            math.log1p(max(float(current_nuisance[4]) / FS_HZ, 0.0)),
            math.log1p(max(float(current_nuisance[5]), 0.0)),
        ), dtype=np.float32)
        shift = 10 if (event_seed(event_ids[event_position]) % 2 == 0) else -10
        for scale_position, scale_ms in enumerate(TIME_SCALES_MS):
            sample_width = max(8, int(round(float(scale_ms) * FS_HZ / 1000.0)))
            start = int(center) - max(2, sample_width // 8)
            output["time_covariance"][event_position, scale_position] = covariance_from_segment(
                padded_segment(analytic, start, sample_width)
            )
            output["jitter_time_covariance"][event_position, scale_position] = covariance_from_segment(
                padded_segment(analytic, start + shift, sample_width)
            )
        pre_width = max(16, int(round(80.0 * FS_HZ / 1000.0)))
        pre_gap = max(4, int(round(5.0 * FS_HZ / 1000.0)))
        output["pre_event_covariance"][event_position] = covariance_from_segment(
            padded_segment(analytic, int(center) - pre_gap - pre_width, pre_width)
        )
        short_start = int(center) - 32
        short_segment = padded_segment(normalized, short_start, SHORT_SPECTRAL_SAMPLES)
        short_covariance, short_surrogate = multitaper_covariance(
            short_segment, short_tapers, FIXED_BANDS_HZ, event_seed(event_ids[event_position])
        )
        output["spectral_covariance"][event_position] = short_covariance
        orbit, acoustic = reference_observability(short_covariance)
        surrogate_orbit, surrogate_acoustic = reference_observability(short_surrogate)
        output["real_orbit_coherence"][event_position] = orbit
        output["surrogate_orbit_coherence"][event_position] = surrogate_orbit
        output["real_acoustic_coherence"][event_position] = acoustic
        output["surrogate_acoustic_coherence"][event_position] = surrogate_acoustic
        jitter_segment = padded_segment(normalized, short_start + shift, SHORT_SPECTRAL_SAMPLES)
        jitter_covariance, _unused = multitaper_covariance(
            jitter_segment, short_tapers, FIXED_BANDS_HZ, event_seed(event_ids[event_position]) + 17
        )
        output["jitter_spectral_covariance"][event_position] = jitter_covariance
        long_start = int(center) - 64
        long_segment = padded_segment(normalized, long_start, LONG_SPECTRAL_SAMPLES)
        long_covariance, _unused_long = multitaper_covariance(
            long_segment, long_tapers, FIXED_BANDS_HZ, event_seed(event_ids[event_position]) + 31
        )
        output["long_spectral_covariance"][event_position] = long_covariance
    report = {
        "events": int(len(event_ids)),
        "short_valid": int(np.sum(output["short_valid"])),
        "long_valid": int(np.sum(output["long_valid"])),
        "dense_events": int(np.sum(output["dense_event"])),
        "finite": numeric_payload_finite(output, MATRIX_KEYS),
    }
    if not report["finite"]:
        raise RuntimeError("R15 non-finite fragment matrices")
    return output, report


def balanced_covariance_mean(covariance: np.ndarray, families: np.ndarray, included: Sequence[str]) -> np.ndarray:
    current = np.asarray(covariance)
    names = np.asarray(families).astype(str)
    means = []
    for family in included:
        ids = np.where(names == str(family))[0]
        if not len(ids):
            raise RuntimeError("R15 background family missing: %s" % family)
        means.append(np.mean(current[ids], axis=0))
    return normalize_psd(np.mean(np.stack(means), axis=0))


def fit_conditional_operator(covariance: np.ndarray, ridge: float) -> Tuple[np.ndarray, float]:
    current = normalize_psd(covariance).astype(np.complex128)
    s2n = submatrix(current, B2, NEIGHBOR)
    snn = submatrix(current, NEIGHBOR, NEIGHBOR)
    regularized = snn + float(ridge) * max(float(np.real(np.trace(snn))) / len(NEIGHBOR), EPS) * np.eye(len(NEIGHBOR))
    condition = float(np.linalg.cond(regularized))
    operator = s2n @ np.linalg.pinv(regularized, rcond=1.0e-8)
    return operator.astype(np.complex64), condition


def operator_residual_ratio(covariance: np.ndarray, operator: np.ndarray) -> float:
    current = normalize_psd(covariance).astype(np.complex128)
    s22 = submatrix(current, B2, B2)
    s2n = submatrix(current, B2, NEIGHBOR)
    sn2 = submatrix(current, NEIGHBOR, B2)
    snn = submatrix(current, NEIGHBOR, NEIGHBOR)
    w = np.asarray(operator, dtype=np.complex128)
    residual = hermitian(s22 - w @ sn2 - s2n @ w.conj().T + w @ snn @ w.conj().T)
    return float(np.real(np.trace(residual)) / max(float(np.real(np.trace(s22))), EPS))


def fit_background_operator_model(background: Mapping[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    families = decode_strings(np.asarray(background["family"]))
    ridge_grid = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
    model: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for field in ("time_covariance", "spectral_covariance", "long_spectral_covariance"):
        covariance = np.asarray(background[field])
        view_count = covariance.shape[1]
        operators = np.zeros((view_count, len(B2), len(NEIGHBOR)), dtype=np.complex64)
        ridges = np.zeros(view_count, dtype=np.float32)
        conditions = np.zeros(view_count, dtype=np.float32)
        for view in range(view_count):
            scores: List[Tuple[float, float, float]] = []
            for ridge in ridge_grid:
                errors: List[float] = []
                condition_values: List[float] = []
                for held in r6.CONTEXT_FAMILIES:
                    fit_families = [family for family in r6.CONTEXT_FAMILIES if family != held]
                    fit_covariance = balanced_covariance_mean(covariance[:, view], families, fit_families)
                    operator, condition = fit_conditional_operator(fit_covariance, ridge)
                    held_covariance = balanced_covariance_mean(covariance[:, view], families, [held])
                    errors.append(operator_residual_ratio(held_covariance, operator))
                    condition_values.append(condition)
                penalty = 0.02 * float(np.std(errors)) + (1.0 if max(condition_values) > 1.0e6 else 0.0)
                scores.append((float(np.mean(errors) + penalty), float(ridge), float(max(condition_values))))
            best = min(scores, key=lambda row: (row[0], row[1]))
            all_covariance = balanced_covariance_mean(covariance[:, view], families, r6.CONTEXT_FAMILIES)
            operator, condition = fit_conditional_operator(all_covariance, best[1])
            operators[view] = operator
            ridges[view] = float(best[1])
            conditions[view] = float(condition)
            rows.append({
                "matrix_field": field,
                "view": int(view),
                "view_name": (str(TIME_SCALES_MS[view]) + "ms") if field == "time_covariance" else band_name(FIXED_BANDS_HZ[view]),
                "ridge": float(best[1]),
                "leave_family_error": float(best[0]),
                "condition_number": float(condition),
                "condition_cap_pass": bool(condition <= 1.0e6),
            })
        stem = field.replace("_covariance", "")
        model[stem + "_operator"] = operators
        model[stem + "_ridge"] = ridges
        model[stem + "_condition"] = conditions
    return model, rows


def trace_real(value: np.ndarray) -> np.ndarray:
    return np.real(np.trace(np.asarray(value), axis1=-2, axis2=-1))


def normalized_matrix_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    first = normalize_psd(left).astype(np.complex128)
    second = normalize_psd(right).astype(np.complex128)
    return np.sqrt(np.sum(np.abs(first - second) ** 2, axis=(-2, -1))).astype(np.float32)


def relation_metric_tensor(
    covariance: np.ndarray,
    operators: np.ndarray,
    ridges: np.ndarray,
) -> np.ndarray:
    current = normalize_psd(covariance).astype(np.complex128)
    if current.ndim != 4 or current.shape[-2:] != (CHANNEL_COUNT, CHANNEL_COUNT):
        raise RuntimeError("R15 relation covariance shape mismatch: %s" % (current.shape,))
    if np.asarray(operators).shape != (current.shape[1], len(B2), len(NEIGHBOR)):
        raise RuntimeError("R15 conditional operator shape mismatch")
    n, views = current.shape[:2]
    output = np.zeros((n, views, len(BASE_METRIC_NAMES)), dtype=np.float32)
    for view in range(views):
        matrix = current[:, view]
        s22 = submatrix(matrix, B2, B2)
        s11 = submatrix(matrix, B1, B1)
        s2n = submatrix(matrix, B2, NEIGHBOR)
        sn2 = submatrix(matrix, NEIGHBOR, B2)
        snn = submatrix(matrix, NEIGHBOR, NEIGHBOR)
        operator = np.asarray(operators[view], dtype=np.complex128)
        operator_residual = hermitian(
            s22
            - np.einsum("ij,njk->nik", operator, sn2, optimize=True)
            - np.einsum("nij,kj->nik", s2n, operator.conj(), optimize=True)
            + np.einsum("ij,njk,lk->nil", operator, snn, operator.conj(), optimize=True)
        )
        ridge = float(np.asarray(ridges)[view])
        inverse = safe_inverse(snn, ridge=max(ridge, 1.0e-6))
        schur_residual = hermitian(s22 - np.einsum("nij,njk,nkl->nil", s2n, inverse, sn2, optimize=True))
        trace22 = np.maximum(trace_real(s22), EPS)
        trace11 = np.maximum(trace_real(s11), EPS)
        operator_trace = np.maximum(trace_real(operator_residual), EPS)
        schur_trace = np.maximum(trace_real(schur_residual), EPS)
        eigenvalue = np.maximum(np.linalg.eigvalsh(operator_residual), 0.0)
        vertical = np.real(operator_residual[:, 1, 1]) / operator_trace
        anisotropy = (eigenvalue[:, -1] - eigenvalue[:, 0]) / np.maximum(np.sum(eigenvalue, axis=1), EPS)

        orbit_coh_b1 = canonical_coherence(matrix, B1, ORBIT)
        orbit_coh_b2 = canonical_coherence(matrix, B2, ORBIT)
        orbit_t1 = np.einsum(
            "nij,njk->nik", submatrix(matrix, B1, ORBIT), safe_inverse(submatrix(matrix, ORBIT, ORBIT)), optimize=True
        )
        orbit_t2 = np.einsum(
            "nij,njk->nik", submatrix(matrix, B2, ORBIT), safe_inverse(submatrix(matrix, ORBIT, ORBIT)), optimize=True
        )
        orbit_norm1 = np.sqrt(np.sum(np.abs(orbit_t1) ** 2, axis=(-2, -1)))
        orbit_norm2 = np.sqrt(np.sum(np.abs(orbit_t2) ** 2, axis=(-2, -1)))
        orbit_ratio = np.log(np.maximum(orbit_norm2, EPS)) - np.log(np.maximum(orbit_norm1, EPS))

        acoustic_coh_b1 = canonical_coherence(matrix, B1, ACOUSTIC_A)
        acoustic_coh_b2 = canonical_coherence(matrix, B2, ACOUSTIC_A)
        acoustic_norm1 = transfer_norm(matrix, B1, ACOUSTIC_A)
        acoustic_norm2 = transfer_norm(matrix, B2, ACOUSTIC_A)
        acoustic_ratio = np.log(np.maximum(acoustic_norm2, EPS)) - np.log(np.maximum(acoustic_norm1, EPS))

        output[:, view, 0] = np.log(operator_trace) - np.log(trace22)
        output[:, view, 1] = np.log(schur_trace) - np.log(trace22)
        output[:, view, 2] = 1.0 - normalized_matrix_distance(operator_residual, schur_residual) / math.sqrt(2.0)
        output[:, view, 3] = np.log(operator_trace) - np.log(trace11)
        output[:, view, 4] = np.clip(vertical, 0.0, 1.0)
        output[:, view, 5] = np.clip(anisotropy, 0.0, 1.0)
        output[:, view, 6] = orbit_coh_b2 - orbit_coh_b1
        output[:, view, 7] = orbit_ratio
        output[:, view, 8] = phase_invariant_alignment(orbit_t1, orbit_t2)
        output[:, view, 9] = acoustic_coh_b2 - acoustic_coh_b1
        output[:, view, 10] = acoustic_ratio
        output[:, view, 11] = -np.abs(orbit_ratio - acoustic_ratio)
        output[:, view, 12] = canonical_coherence(matrix, B1, B2)
        output[:, view, 13] = np.log(trace22) - np.log(trace11)
    return np.nan_to_num(output, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)


def nan_summary(value: np.ndarray, reducer: str) -> np.ndarray:
    current = np.asarray(value, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        if reducer == "median":
            output = np.nanmedian(current, axis=1)
        elif reducer == "iqr":
            output = np.nanquantile(current, 0.75, axis=1) - np.nanquantile(current, 0.25, axis=1)
        else:
            raise ValueError("Unknown R15 summary reducer: %s" % reducer)
    return np.nan_to_num(output, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)


def summarize_relation_features(
    time_metrics: np.ndarray,
    spectral_metrics: np.ndarray,
    long_metrics: np.ndarray,
    selected_bands: Sequence[int],
    short_valid: np.ndarray,
    long_valid: np.ndarray,
) -> np.ndarray:
    time_value = np.asarray(time_metrics, dtype=np.float32)
    selected = np.asarray(list(selected_bands), dtype=np.int64)
    if len(selected):
        short = np.asarray(spectral_metrics, dtype=np.float32)[:, selected]
        long = np.asarray(long_metrics, dtype=np.float32)[:, selected]
        short = short.copy()
        long = long.copy()
        short[~np.asarray(short_valid, dtype=bool)] = np.nan
        long[~np.asarray(long_valid, dtype=bool)] = np.nan
        spectral_median = nan_summary(short, "median")
        spectral_iqr = nan_summary(short, "iqr")
        long_difference = nan_summary(long - short, "median")
    else:
        shape = (len(time_value), len(BASE_METRIC_NAMES))
        spectral_median = np.zeros(shape, dtype=np.float32)
        spectral_iqr = np.zeros(shape, dtype=np.float32)
        long_difference = np.zeros(shape, dtype=np.float32)
    time_median = np.median(time_value, axis=1).astype(np.float32)
    output = np.zeros((len(time_value), len(SUMMARY_FEATURE_NAMES)), dtype=np.float32)
    for metric in range(len(BASE_METRIC_NAMES)):
        offset = metric * len(SUMMARY_SUFFIXES)
        output[:, offset + 0] = time_median[:, metric]
        output[:, offset + 1] = spectral_median[:, metric]
        output[:, offset + 2] = spectral_iqr[:, metric]
        output[:, offset + 3] = long_difference[:, metric]
    return np.clip(np.nan_to_num(output, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0).astype(np.float32)


def pre_event_features(pre_covariance: np.ndarray, background_model: Mapping[str, np.ndarray]) -> np.ndarray:
    time_operator = np.asarray(background_model["time_operator"])[-1:]
    time_ridge = np.asarray(background_model["time_ridge"])[-1:]
    metric = relation_metric_tensor(np.asarray(pre_covariance)[:, None], time_operator, time_ridge)[:, 0]
    return np.asarray(metric, dtype=np.float32)


def transform_covariance(value: np.ndarray, control: str) -> np.ndarray:
    current = np.asarray(value, dtype=np.complex64)
    if control == "K0_clean" or control == "K6_center_jitter_2ms":
        return current
    if control == "K1_B1_B2_swap":
        order = np.asarray((2, 3, 0, 1, 4, 5, 6, 7, 8, 9), dtype=np.int64)
        return current[..., order, :][..., :, order]
    if control == "K4_complex_conjugate":
        return current.conj()
    if control == "K5_common_scale_0p5":
        return 0.25 * current
    output = current.copy()
    if control == "K2_orbit_cross_block_destroyed":
        reference = ORBIT
    elif control == "K3_acoustic_A_cross_block_destroyed":
        reference = ACOUSTIC_A
    else:
        raise ValueError("Unknown R15 covariance control: %s" % control)
    other = tuple(index for index in range(CHANNEL_COUNT) if index not in reference)
    for row in other:
        output[..., row, reference] = 0.0
        output[..., reference, row] = 0.0
    return output


def build_event_feature_package(
    matrices: Mapping[str, np.ndarray],
    background_model: Mapping[str, np.ndarray],
    band_screen: Mapping[str, Any],
    control: str = "K0_clean",
) -> Dict[str, np.ndarray]:
    if control == "K6_center_jitter_2ms":
        time_covariance = np.asarray(matrices["jitter_time_covariance"])
        spectral_covariance = np.asarray(matrices["jitter_spectral_covariance"])
        long_covariance = np.asarray(matrices["long_spectral_covariance"])
    else:
        time_covariance = transform_covariance(np.asarray(matrices["time_covariance"]), control)
        spectral_covariance = transform_covariance(np.asarray(matrices["spectral_covariance"]), control)
        long_covariance = transform_covariance(np.asarray(matrices["long_spectral_covariance"]), control)
    time_metrics = relation_metric_tensor(
        time_covariance, np.asarray(background_model["time_operator"]), np.asarray(background_model["time_ridge"])
    )
    spectral_metrics = relation_metric_tensor(
        spectral_covariance, np.asarray(background_model["spectral_operator"]), np.asarray(background_model["spectral_ridge"])
    )
    long_metrics = relation_metric_tensor(
        long_covariance, np.asarray(background_model["long_spectral_operator"]), np.asarray(background_model["long_spectral_ridge"])
    )
    feature = summarize_relation_features(
        time_metrics, spectral_metrics, long_metrics,
        band_screen.get("union_selected", ()),
        np.asarray(matrices["short_valid"], dtype=bool),
        np.asarray(matrices["long_valid"], dtype=bool),
    )
    pre = pre_event_features(transform_covariance(np.asarray(matrices["pre_event_covariance"]), control), background_model)
    return {
        "feature": feature,
        "pre_event": pre,
        "valid": np.isfinite(feature).all(axis=1),
    }


def feature_indices(metrics: Sequence[str]) -> np.ndarray:
    names: List[str] = []
    for metric in metrics:
        names.extend(["%s__%s" % (metric, suffix) for suffix in SUMMARY_SUFFIXES])
    lookup = {name: position for position, name in enumerate(SUMMARY_FEATURE_NAMES)}
    return np.asarray([lookup[name] for name in names], dtype=np.int64)


FEATURE_SETS: Dict[str, np.ndarray] = {
    "conditional": feature_indices(CONDITIONAL_METRICS),
    "bridge": feature_indices(ORBIT_METRICS + ACOUSTIC_METRICS + DIRECT_METRICS),
    "full": feature_indices(BASE_METRIC_NAMES),
    "orbit": feature_indices(ORBIT_METRICS),
    "acoustic": feature_indices(ACOUSTIC_METRICS),
}


def event_process_features(matrices: Mapping[str, np.ndarray]) -> np.ndarray:
    process = np.asarray(matrices["process_feature"], dtype=np.float32)
    if process.ndim != 2 or process.shape[1] != 6 or not np.isfinite(process).all():
        raise RuntimeError("R15 current-event process feature contract failed: %s" % (process.shape,))
    return process


def equal_record_weights(keys: np.ndarray, labels: np.ndarray) -> np.ndarray:
    groups = np.asarray(keys).astype(str)
    y = np.asarray(labels, dtype=np.int64)
    output = np.zeros(len(groups), dtype=np.float64)
    for label in np.unique(y).tolist():
        label_ids = np.where(y == int(label))[0]
        label_groups = np.unique(groups[label_ids])
        for group in label_groups.tolist():
            ids = np.where((groups == str(group)) & (y == int(label)))[0]
            output[ids] = 1.0 / max(1, len(label_groups) * len(ids))
    output /= max(float(np.mean(output[output > 0])), EPS)
    return output


def fit_weighted_logistic(
    feature: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    ridge: float,
) -> Dict[str, np.ndarray]:
    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    center, scale = robust_location_scale(x)
    z = np.clip((x - center) / scale, -8.0, 8.0)
    design = np.column_stack((np.ones(len(z)), z))
    weight = equal_record_weights(np.asarray(groups), np.asarray(labels, dtype=np.int64))
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    for _iteration in range(80):
        probability = sigmoid(design @ coefficient)
        variance = np.maximum(probability * (1.0 - probability), 1.0e-5)
        gradient = design.T @ (weight * (y - probability)) - penalty @ coefficient
        hessian = design.T @ ((weight * variance)[:, None] * design) + penalty
        step = np.linalg.pinv(hessian, rcond=1.0e-8) @ gradient
        coefficient += step
        if float(np.max(np.abs(step))) < 1.0e-7:
            break
    return {
        "center": np.asarray(center, dtype=np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
        "intercept": np.asarray([coefficient[0]], dtype=np.float32),
        "coefficient": np.asarray(coefficient[1:], dtype=np.float32),
        "ridge": np.asarray([float(ridge)], dtype=np.float32),
    }


def logistic_score(model: Mapping[str, np.ndarray], feature: np.ndarray) -> np.ndarray:
    x = np.asarray(feature, dtype=np.float64)
    center = np.asarray(model["center"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    intercept = float(np.asarray(model["intercept"]).ravel()[0])
    z = np.clip((x - center) / np.maximum(scale, 1.0e-4), -8.0, 8.0)
    return (intercept + z @ coefficient).astype(np.float32)

def _bind(modules):
    globals().update(modules)
