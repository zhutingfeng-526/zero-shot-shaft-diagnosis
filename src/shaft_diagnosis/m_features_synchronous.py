"""Common-phase alignment and synchronous response features for the M expert."""
from __future__ import annotations
import math
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np

from concurrent.futures import ThreadPoolExecutor
try:
    from scipy.signal import hilbert, savgol_filter
except Exception as exc:
    warnings.warn(
        "SciPy signal functions could not be imported; using the FFT analytic "
        "signal fallback without Savitzky-Golay phase smoothing. Check the "
        "SciPy installation. Original error: %s: %s" % (type(exc).__name__, exc),
        RuntimeWarning,
        stacklevel=2,
    )
    hilbert = None
    savgol_filter = None



FS_HZ = 5000.0


WINDOW_SAMPLES = 6000


SAMPLES_PER_REV = 250


REVOLUTIONS = 23


RESAMPLED_SAMPLES = REVOLUTIONS * SAMPLES_PER_REV


POSITION_CLASSES = 3


DEGREE_CLASSES = 2


ORDERS = (1, 2, 3, 4)


SENSOR_CONTRACT: Tuple[Dict[str, Any], ...] = (
    {"channel": 1, "index": 0, "name": "B1_x", "modality": "vibration", "location": "B1_motor_end", "axis": "x"},
    {"channel": 2, "index": 1, "name": "B1_y", "modality": "vibration", "location": "B1_motor_end", "axis": "y_vertical"},
    {"channel": 3, "index": 2, "name": "B2_x", "modality": "vibration", "location": "B2_middle", "axis": "x"},
    {"channel": 4, "index": 3, "name": "B2_y", "modality": "vibration", "location": "B2_middle", "axis": "y_vertical"},
    {"channel": 5, "index": 4, "name": "B3_x", "modality": "vibration", "location": "B3_rotor_end", "axis": "x"},
    {"channel": 6, "index": 5, "name": "B3_y", "modality": "vibration", "location": "B3_rotor_end", "axis": "y_vertical"},
    {"channel": 7, "index": 6, "name": "orbit_x", "modality": "shaft_orbit", "location": "between_B1_B2", "axis": "x"},
    {"channel": 8, "index": 7, "name": "orbit_y", "modality": "shaft_orbit", "location": "between_B1_B2", "axis": "y_vertical"},
    {"channel": 9, "index": 8, "name": "sound_A", "modality": "acoustic", "location": "near_B1", "axis": "sound"},
    {"channel": 10, "index": 9, "name": "sound_B", "modality": "acoustic", "location": "near_B3", "axis": "sound"},
)


CHANNEL_NAMES = tuple(row["name"] for row in SENSOR_CONTRACT)


VIBRATION = (0, 1, 2, 3, 4, 5)


ORBIT = (6, 7)


PHASE_QUALITY_NAMES = (
    "estimated_rpm", "rpm_iqr", "cycle_coverage", "monotonic_fraction", "orbit_coherence",
    "vibration_coherence", "cross_phase_residual", "phase_valid", "fallback_nominal",
)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value -= np.max(value, axis=1, keepdims=True)
    value = np.exp(value)
    value /= np.maximum(value.sum(axis=1, keepdims=True), 1.0e-12)
    return value.astype(np.float32)


def robust_event_mask(raw: np.ndarray) -> np.ndarray:
    difference = np.diff(np.asarray(raw, dtype=np.float64), axis=1, prepend=raw[:, :1])
    score = np.sqrt(np.mean(difference[:6] ** 2, axis=0)) + 0.25 * np.sqrt(np.mean(difference[8:] ** 2, axis=0))
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    z = (score - median) / max(1.4826 * mad, 1.0e-9)
    mask = z > 8.0
    return np.convolve(mask.astype(np.int64), np.ones(17, dtype=np.int64), mode="same") > 0 if np.any(mask) else mask


def fft_bandpass(signal: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    value = np.asarray(signal, dtype=np.float64)
    frequency = np.fft.fftfreq(len(value), d=1.0 / FS_HZ)
    spectrum = np.fft.fft(value)
    mask = (np.abs(frequency) >= float(lo_hz)) & (np.abs(frequency) <= float(hi_hz))
    return np.fft.ifft(spectrum * mask).real


def smooth_phase(phase: np.ndarray, event_mask: np.ndarray) -> np.ndarray:
    value = np.asarray(phase, dtype=np.float64).copy()
    valid = ~np.asarray(event_mask, dtype=bool)
    index = np.arange(len(value), dtype=np.float64)
    if int(np.sum(valid)) >= 16:
        value[~valid] = np.interp(index[~valid], index[valid], value[valid])
    if savgol_filter is not None and len(value) >= 101:
        value = savgol_filter(value, 101, 3, mode="interp")
    if float(np.polyfit(index, value, 1)[0]) < 0.0:
        value = -value
    return np.maximum.accumulate(value + np.arange(len(value)) * 1.0e-9)


def orbit_phase(raw: np.ndarray, event_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    xy = np.asarray(raw[6:8], dtype=np.float64).T
    xy -= np.median(xy, axis=0, keepdims=True)
    covariance = np.cov(xy, rowvar=False) + np.eye(2) * 1.0e-9
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    whitener = eigenvector.dot(np.diag(1.0 / np.sqrt(np.maximum(eigenvalue, 1.0e-9)))).dot(eigenvector.T)
    complex_orbit = xy.dot(whitener)[:, 0] + 1j * xy.dot(whitener)[:, 1]
    frequency = np.fft.fftfreq(len(complex_orbit), d=1.0 / FS_HZ)
    spectrum = np.fft.fft(complex_orbit)
    positive = (frequency >= 12.0) & (frequency <= 28.0)
    negative = (frequency <= -12.0) & (frequency >= -28.0)
    use = positive if np.sum(np.abs(spectrum[positive]) ** 2) >= np.sum(np.abs(spectrum[negative]) ** 2) else negative
    filtered = np.fft.ifft(spectrum * use)
    phase = smooth_phase(np.unwrap(np.angle(filtered)), event_mask)
    direct = np.unwrap(np.angle(complex_orbit))
    if float(np.polyfit(np.arange(len(direct)), direct, 1)[0]) < 0.0:
        direct = -direct
    clean = ~np.asarray(event_mask, dtype=bool)
    reference = np.where(clean)[0]
    if len(reference) > 200:
        reference = reference[100:-100]
    lag = float(np.angle(np.mean(np.exp(1j * (phase[reference] - direct[reference])))))
    phase -= lag
    phase += round(float(np.median((direct - phase) / (2.0 * math.pi)))) * 2.0 * math.pi
    residual = np.unwrap(np.angle(np.exp(1j * (direct - phase))))
    if savgol_filter is not None and len(residual) >= 101:
        residual = savgol_filter(residual, 101, 2, mode="interp")
    phase = smooth_phase(phase + residual, event_mask)
    coherence = float(np.median(np.abs(filtered)) / max(np.sqrt(np.mean(np.abs(complex_orbit) ** 2)), 1.0e-9))
    return phase, coherence


def analytic_vibration_phase(signal: np.ndarray, event_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    filtered = fft_bandpass(signal, 12.0, 28.0)
    if hilbert is None:
        spectrum = np.fft.fft(filtered)
        multiplier = np.zeros(len(filtered), dtype=np.float64)
        multiplier[0] = 1.0
        multiplier[1:(len(filtered) + 1) // 2] = 2.0
        if len(filtered) % 2 == 0:
            multiplier[len(filtered) // 2] = 1.0
        analytic = np.fft.ifft(spectrum * multiplier)
    else:
        analytic = hilbert(filtered)
    phase = smooth_phase(np.unwrap(np.angle(analytic)), event_mask)
    coherence = float(np.sqrt(np.mean(filtered ** 2)) / max(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2)), 1.0e-9))
    return phase, coherence


def estimate_consensus_phase(raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    event_mask = robust_event_mask(raw)
    orbit, orbit_coherence = orbit_phase(raw, event_mask)
    phases: List[np.ndarray] = []
    weights: List[float] = []
    residuals: List[float] = []
    for channel in range(6):
        phase, coherence = analytic_vibration_phase(raw[channel], event_mask)
        aligned = phase + round(float(np.median((orbit - phase) / (2.0 * math.pi)))) * 2.0 * math.pi
        fixed_lag = float(np.angle(np.mean(np.exp(1j * (aligned - orbit)))))
        aligned -= fixed_lag
        aligned += round(float(np.median((orbit - aligned) / (2.0 * math.pi)))) * 2.0 * math.pi
        residual = float(np.median(np.abs(np.angle(np.exp(1j * (aligned - orbit))))))
        weight = max(coherence, 1.0e-4) / max(residual + 0.05, 0.05)
        phases.append(aligned); weights.append(weight); residuals.append(residual)
    orbit_weight = max(orbit_coherence, 0.05) * 4.0
    correction = np.zeros_like(orbit)
    denominator = orbit_weight
    for phase, weight in zip(phases, weights):
        correction += weight * np.clip(phase - orbit, -0.20, 0.20)
        denominator += weight
    value = smooth_phase(orbit + correction / max(denominator, 1.0e-9), event_mask)
    instantaneous = np.diff(value) * FS_HZ / (2.0 * math.pi)
    central = instantaneous[100:-100]
    positive = central[(central > 5.0) & (central < 60.0)]
    median_hz = float(np.median(positive)) if len(positive) else 0.0
    rpm = median_hz * 60.0
    rpm_iqr = float((np.quantile(positive, 0.75) - np.quantile(positive, 0.25)) * 60.0) if len(positive) else 1.0e9
    coverage = float((value[-1] - value[0]) / (2.0 * math.pi))
    monotonic = float(np.mean(np.diff(value) > 0.0))
    valid = bool(18.0 <= coverage <= 30.0 and 1080.0 <= rpm <= 1320.0 and monotonic >= 0.995)
    return value, {
        "estimated_rpm": rpm, "rpm_iqr": rpm_iqr, "cycle_coverage": coverage,
        "monotonic_fraction": monotonic, "orbit_coherence": orbit_coherence,
        "vibration_coherence": float(np.mean(weights)), "cross_phase_residual": float(np.mean(residuals)),
        "phase_valid": float(valid),
    }


def resample_consensus_window(raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    phase, quality = estimate_consensus_phase(raw)
    lower, upper = float(phase[0]), float(phase[-1])
    target_span = (RESAMPLED_SAMPLES - 1) * (2.0 * math.pi / SAMPLES_PER_REV)
    middle = 0.5 * (lower + upper)
    ideal = int(round((middle - 0.5 * target_span) / (2.0 * math.pi)))
    minimum = int(math.ceil(lower / (2.0 * math.pi)))
    maximum = int(math.floor((upper - target_span) / (2.0 * math.pi)))
    target_start = float(min(max(ideal, minimum), maximum)) * 2.0 * math.pi
    target = target_start + np.arange(RESAMPLED_SAMPLES, dtype=np.float64) * (2.0 * math.pi / SAMPLES_PER_REV)
    quality["unlocked_estimated_rpm"] = float(quality["estimated_rpm"])
    if target[0] < lower or target[-1] > upper or not bool(quality["phase_valid"]):
        phase = np.arange(WINDOW_SAMPLES, dtype=np.float64) * (2.0 * math.pi * 20.0 / FS_HZ)
        lower, upper = float(phase[0]), float(phase[-1])
        target_start = 0.5 * (lower + upper) - 0.5 * target_span
        target = target_start + np.arange(RESAMPLED_SAMPLES, dtype=np.float64) * (2.0 * math.pi / SAMPLES_PER_REV)
        quality["estimated_rpm"] = 1200.0
        quality["fallback_nominal"] = 1.0
    else:
        quality["fallback_nominal"] = 0.0
    aligned = np.stack([
        np.interp(target, phase, np.asarray(raw[channel], dtype=np.float64)) for channel in range(10)
    ], axis=0).astype(np.float32)
    return aligned, quality


def align_batch(raw: np.ndarray, executor: ThreadPoolExecutor) -> Tuple[np.ndarray, np.ndarray]:
    results = list(executor.map(resample_consensus_window, [np.asarray(row, dtype=np.float32) for row in raw]))
    aligned = np.stack([row[0] for row in results], axis=0)
    quality = np.asarray([[float(row[1][name]) for name in PHASE_QUALITY_NAMES] for row in results], dtype=np.float32)
    if not np.isfinite(aligned).all() or not np.isfinite(quality).all():
        raise RuntimeError("Non-finite phase-aligned output")
    return aligned, quality


def rotation_coefficients(aligned: np.ndarray) -> np.ndarray:
    cycles = np.asarray(aligned, dtype=np.float32).reshape(len(aligned), 10, REVOLUTIONS, SAMPLES_PER_REV)
    spectrum = np.fft.rfft(cycles, axis=3) / float(SAMPLES_PER_REV)
    return np.asarray(spectrum[:, :, :, 1:5], dtype=np.complex64)


def feature_name(base: str, *tags: str) -> str:
    return "tags=%s;%s" % (",".join(sorted(set(str(tag) for tag in tags if tag))), str(base))


def feature_tags(name: str) -> set:
    prefix = str(name).split(";", 1)[0]
    return set(prefix[len("tags="):].split(",")) if prefix.startswith("tags=") and prefix != "tags=" else set()


def shrink_normalize_hpd(matrix: np.ndarray, shrink: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(matrix, dtype=np.complex128)
    dimension = value.shape[-1]
    diagonal = np.zeros_like(value)
    indices = np.arange(dimension)
    diagonal[..., indices, indices] = np.real(value[..., indices, indices])
    value = (1.0 - float(shrink)) * value + float(shrink) * diagonal
    trace = np.real(np.trace(value, axis1=-2, axis2=-1))
    epsilon = np.maximum(trace, 1.0e-12)[..., None, None] * (1.0e-6 / float(dimension))
    value = value + epsilon * np.eye(dimension, dtype=np.complex128)
    trace = np.real(np.trace(value, axis1=-2, axis2=-1))
    value = value / np.maximum(trace, 1.0e-12)[..., None, None]
    return value, trace


def coefficient_hpd(coefficients: np.ndarray, channels: Sequence[int]) -> np.ndarray:
    z = np.asarray(coefficients[:, tuple(channels), :, :], dtype=np.complex128)
    # B x channel x cycle x order -> B x order x channel x channel
    return np.einsum("bdck,beck->bkde", z, np.conjugate(z), optimize=True) / float(z.shape[2])


def inverse_sqrt_hpd(matrix: np.ndarray) -> np.ndarray:
    value, _trace = shrink_normalize_hpd(np.asarray(matrix, dtype=np.complex128))
    eigenvalue, eigenvector = np.linalg.eigh(value)
    return np.einsum(
        "...ij,...j,...kj->...ik",
        eigenvector,
        1.0 / np.sqrt(np.maximum(eigenvalue, 1.0e-8)),
        np.conjugate(eigenvector),
        optimize=True,
    )


def log_euclidean_residual(matrix: np.ndarray, reference_inverse_sqrt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normalized, trace = shrink_normalize_hpd(matrix)
    left = np.asarray(reference_inverse_sqrt, dtype=np.complex128)[None]
    transported = np.matmul(np.matmul(left, normalized), np.swapaxes(np.conjugate(left), -1, -2))
    transported = 0.5 * (transported + np.swapaxes(np.conjugate(transported), -1, -2))
    eigenvalue, eigenvector = np.linalg.eigh(transported)
    logged = np.einsum(
        "...ij,...j,...kj->...ik",
        eigenvector,
        np.log(np.maximum(eigenvalue, 1.0e-7)),
        np.conjugate(eigenvector),
        optimize=True,
    )
    return logged, trace


def flatten_hermitian(
    matrix: np.ndarray,
    channel_names: Sequence[str],
    prefix: str,
    base_tags: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    value = np.asarray(matrix)
    columns: List[np.ndarray] = []
    names: List[str] = []
    for left in range(value.shape[-1]):
        axis_tags = channel_axis_tags((channel_names[left],))
        columns.append(np.real(value[..., left, left]))
        names.append(feature_name("%s_diag_%s" % (prefix, channel_names[left]), *(list(base_tags) + list(axis_tags))))
    for left in range(value.shape[-1]):
        for right in range(left + 1, value.shape[-1]):
            axis_tags = channel_axis_tags((channel_names[left], channel_names[right]))
            tags = list(base_tags) + list(axis_tags) + ["phase"]
            columns.append(np.sqrt(2.0) * np.real(value[..., left, right]))
            names.append(feature_name("%s_off_%s_%s_re" % (prefix, channel_names[left], channel_names[right]), *tags))
            columns.append(np.sqrt(2.0) * np.imag(value[..., left, right]))
            names.append(feature_name("%s_off_%s_%s_im" % (prefix, channel_names[left], channel_names[right]), *tags))
    return np.stack(columns, axis=-1).astype(np.float32), names


def channel_axis_tags(names: Sequence[str]) -> Tuple[str, ...]:
    tags: List[str] = []
    for name in names:
        if str(name).endswith("_x") or str(name) == "orbit_x":
            tags.append("axis_x")
        elif str(name).endswith("_y") or str(name) == "orbit_y":
            tags.append("axis_y")
    return tuple(sorted(set(tags)))


def hpd_feature_block(
    coefficients: np.ndarray,
    channels: Sequence[int],
    reference_inverse_sqrt: np.ndarray,
    prefix: str,
    extra_tags: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    names_local = [CHANNEL_NAMES[index] for index in channels]
    matrix = coefficient_hpd(coefficients, channels)
    logged, trace = log_euclidean_residual(matrix, reference_inverse_sqrt)
    pieces: List[np.ndarray] = []
    names: List[str] = []
    z = np.asarray(coefficients[:, tuple(channels), :, :], dtype=np.complex128)
    for order_index, order in enumerate(ORDERS):
        flattened, flattened_names = flatten_hermitian(
            logged[:, order_index], names_local, "%s_logres_k%d" % (prefix, order),
            tuple(extra_tags) + ("reference",),
        )
        pieces.append(flattened); names.extend(flattened_names)
        pieces.append(np.log1p(np.maximum(trace[:, order_index], 0.0))[:, None].astype(np.float32))
        tags = tuple(extra_tags) + ("magnitude",)
        names.append(feature_name("%s_logtrace_k%d" % (prefix, order), *tags))
        amplitude = np.abs(z[..., order_index])
        mean_amplitude = np.mean(amplitude, axis=2)
        coefficient_variation = np.std(amplitude, axis=2) / np.maximum(mean_amplitude, 1.0e-9)
        unit = z[..., order_index] / np.maximum(amplitude, 1.0e-9)
        resultant = np.abs(np.mean(unit, axis=2))
        for channel_index, channel_name in enumerate(names_local):
            axis_tags = channel_axis_tags((channel_name,))
            current_tags = tuple(extra_tags) + tuple(axis_tags)
            pieces.extend((
                np.log1p(mean_amplitude[:, channel_index])[:, None].astype(np.float32),
                np.clip(coefficient_variation[:, channel_index], 0.0, 20.0)[:, None].astype(np.float32),
                resultant[:, channel_index][:, None].astype(np.float32),
            ))
            names.extend((
                feature_name("%s_amp_k%d_%s" % (prefix, order, channel_name), *(current_tags + ("magnitude",))),
                feature_name("%s_ampcv_k%d_%s" % (prefix, order, channel_name), *(current_tags + ("magnitude",))),
                feature_name("%s_phase_resultant_k%d_%s" % (prefix, order, channel_name), *(current_tags + ("phase",))),
            ))
    return np.concatenate(pieces, axis=1), names


def normalized_coherence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.mean(np.asarray(left) * np.conjugate(np.asarray(right)), axis=1)
    denominator = np.sqrt(np.mean(np.abs(left) ** 2, axis=1) * np.mean(np.abs(right) ** 2, axis=1))
    return numerator / np.maximum(denominator, 1.0e-12)


def orbit_geometry(aligned: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    orbit = np.asarray(aligned[:, 6:8], dtype=np.float64)
    orbit -= np.mean(orbit, axis=2, keepdims=True)
    covariance = np.einsum("bct,bdt->bcd", orbit, orbit, optimize=True) / float(orbit.shape[2])
    eigenvalue, eigenvector = np.linalg.eigh(covariance + np.eye(2)[None] * 1.0e-12)
    major = eigenvector[:, :, 1]
    orientation = np.arctan2(major[:, 1], major[:, 0])
    ratio = eigenvalue[:, 1] / np.maximum(eigenvalue[:, 0], 1.0e-12)
    correlation = covariance[:, 0, 1] / np.sqrt(np.maximum(covariance[:, 0, 0] * covariance[:, 1, 1], 1.0e-12))
    values = np.stack((
        np.log1p(np.maximum(covariance[:, 0, 0], 0.0)),
        np.log1p(np.maximum(covariance[:, 1, 1], 0.0)),
        np.log1p(np.maximum(ratio, 0.0)),
        np.sin(2.0 * orientation), np.cos(2.0 * orientation),
        np.clip(correlation, -1.0, 1.0),
        np.log1p(np.maximum(np.trace(covariance, axis1=1, axis2=2), 0.0)),
    ), axis=1).astype(np.float32)
    names = [
        feature_name("orbit_variance_x", "orbit", "axis_x", "magnitude"),
        feature_name("orbit_variance_y", "orbit", "axis_y", "magnitude"),
        feature_name("orbit_ellipse_ratio", "orbit", "magnitude"),
        feature_name("orbit_orientation_sin2", "orbit", "phase", "axis_x", "axis_y"),
        feature_name("orbit_orientation_cos2", "orbit", "phase", "axis_x", "axis_y"),
        feature_name("orbit_xy_correlation", "orbit", "phase", "axis_x", "axis_y"),
        feature_name("orbit_total_energy", "orbit", "magnitude"),
    ]
    return values, names


def t1_features(coefficients: np.ndarray, references: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    return hpd_feature_block(coefficients, VIBRATION, references["vib_inv"], "vib", ("vibration",))


def t2_features(
    coefficients: np.ndarray,
    aligned: np.ndarray,
    references: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    t1, names = t1_features(coefficients, references)
    orbit_hpd, orbit_names = hpd_feature_block(coefficients, ORBIT, references["orbit_inv"], "orbit", ("orbit",))
    geometry, geometry_names = orbit_geometry(aligned)
    pieces = [t1, orbit_hpd, geometry]
    names = names + orbit_names + geometry_names
    z = np.asarray(coefficients, dtype=np.complex128)
    # Fixed-coordinate circular components; no physical forward/backward label is assigned.
    for order_index, order in enumerate(ORDERS):
        x_value = z[:, 6, :, order_index]
        y_value = z[:, 7, :, order_index]
        component_a = x_value + 1j * y_value
        component_b = x_value - 1j * y_value
        amp_a = np.mean(np.abs(component_a), axis=1)
        amp_b = np.mean(np.abs(component_b), axis=1)
        relation = normalized_coherence(component_a, component_b)
        circular = np.stack((
            np.log1p(amp_a), np.log1p(amp_b),
            (amp_a - amp_b) / np.maximum(amp_a + amp_b, 1.0e-9),
            np.real(relation), np.imag(relation),
        ), axis=1).astype(np.float32)
        pieces.append(circular)
        names.extend((
            feature_name("orbit_circularA_amp_k%d" % order, "orbit", "magnitude"),
            feature_name("orbit_circularB_amp_k%d" % order, "orbit", "magnitude"),
            feature_name("orbit_circular_balance_k%d" % order, "orbit", "magnitude"),
            feature_name("orbit_circular_relation_re_k%d" % order, "orbit", "phase"),
            feature_name("orbit_circular_relation_im_k%d" % order, "orbit", "phase"),
        ))
        for vibration_channel in (0, 1, 2, 3):
            for orbit_channel in ORBIT:
                coherence = normalized_coherence(
                    z[:, vibration_channel, :, order_index], z[:, orbit_channel, :, order_index],
                )
                pieces.append(np.stack((np.real(coherence), np.imag(coherence), np.abs(coherence)), axis=1).astype(np.float32))
                axis_tags = channel_axis_tags((CHANNEL_NAMES[vibration_channel], CHANNEL_NAMES[orbit_channel]))
                base = "vib_orbit_%s_%s_k%d" % (CHANNEL_NAMES[vibration_channel], CHANNEL_NAMES[orbit_channel], order)
                names.extend((
                    feature_name(base + "_re", "orbit", "crossmodal", "phase", *axis_tags),
                    feature_name(base + "_im", "orbit", "crossmodal", "phase", *axis_tags),
                    feature_name(base + "_mag", "orbit", "crossmodal", "magnitude", *axis_tags),
                ))
    for channel in (0, 1, 2, 3, 6, 7):
        base = z[:, channel, :, 0]
        for harmonic_index, harmonic in enumerate((2, 3, 4), start=1):
            target = z[:, channel, :, harmonic_index]
            numerator = np.mean((base ** harmonic) * np.conjugate(target), axis=1)
            denominator = np.sqrt(np.mean(np.abs(base) ** (2 * harmonic), axis=1) * np.mean(np.abs(target) ** 2, axis=1))
            closure = numerator / np.maximum(denominator, 1.0e-12)
            pieces.append(np.stack((np.real(closure), np.imag(closure), np.abs(closure)), axis=1).astype(np.float32))
            axis_tags = channel_axis_tags((CHANNEL_NAMES[channel],))
            prefix = "harmonic_closure_%s_1x%d_%dx" % (CHANNEL_NAMES[channel], harmonic, harmonic)
            names.extend((
                feature_name(prefix + "_re", "phase", *axis_tags, *("orbit",) if channel in ORBIT else ()),
                feature_name(prefix + "_im", "phase", *axis_tags, *("orbit",) if channel in ORBIT else ()),
                feature_name(prefix + "_mag", "magnitude", *axis_tags, *("orbit",) if channel in ORBIT else ()),
            ))
    return np.concatenate(pieces, axis=1), names


def robust_standardizer(features: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(features[:, np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    center = np.median(selected, axis=0).astype(np.float32)
    q25 = np.quantile(selected, 0.25, axis=0).astype(np.float32)
    q75 = np.quantile(selected, 0.75, axis=0).astype(np.float32)
    robust = (q75 - q25) / 1.349
    standard = np.std(selected.astype(np.float64), axis=0).astype(np.float32)
    scale = np.maximum(robust, 0.05 * standard)
    active = np.isfinite(center) & np.isfinite(scale) & (scale > 1.0e-7)
    if int(np.sum(active)) < 4:
        raise RuntimeError("Robust standardizer retained too few features")
    return center[active], scale[active], np.asarray(indices, dtype=np.int64)[active]


def transform_features(
    features: np.ndarray,
    indices: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    value = np.asarray(features[:, np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    value = (value - np.asarray(center, dtype=np.float32)[None]) / np.asarray(scale, dtype=np.float32)[None]
    return np.nan_to_num(np.clip(value, -12.0, 12.0), nan=0.0, posinf=12.0, neginf=-12.0).astype(np.float32)


def predict_dual_linear(features: np.ndarray, state: Mapping[str, np.ndarray], batch_size: int = 8192) -> Dict[str, np.ndarray]:
    weight = np.asarray(state["weight"], dtype=np.float32)
    bias = np.asarray(state["bias"], dtype=np.float32)
    position = np.empty((len(features), POSITION_CLASSES), dtype=np.float32)
    degree = np.empty((len(features), DEGREE_CLASSES), dtype=np.float32)
    for start in range(0, len(features), int(batch_size)):
        end = min(len(features), start + int(batch_size))
        logits = np.asarray(features[start:end], dtype=np.float32).dot(weight.T) + bias[None]
        position[start:end] = softmax_np(logits[:, :POSITION_CLASSES])
        degree[start:end] = softmax_np(logits[:, POSITION_CLASSES:])
    return {"position_prob": position, "degree_prob": degree}
