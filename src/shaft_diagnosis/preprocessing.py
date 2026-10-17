"""Record preprocessing for the frozen 5 kHz, ten-channel DATA4 input contract.

Apply these stages to complete acquisition records before selecting windows.
Input values retain their acquisition-export amplitude units; no sensor gain or
unit conversion is applied. Existing DATA4 windows must not be filtered again.
"""
import hashlib
import json
import warnings
import numpy as np
from scipy.signal import butter, medfilt, savgol_filter, sosfiltfilt

ACQUISITION_HZ = 10000
SAMPLE_RATE_HZ = 5000
TRAINING_RECORD_SECONDS = 300
WINDOW_SAMPLES = 6000
HOP_SAMPLES = 150
FILTER_SPEC_HASH = "8b0189e0d2"
CHANNEL_ORDER = (
    "bearing1_x", "bearing1_y", "bearing2_x", "bearing2_y",
    "bearing3_x", "bearing3_y", "shaft_orbit_x", "shaft_orbit_y",
    "sound_A_bearing1_side", "sound_B_bearing3_side",
)
FILTER_CFG = {
    "vibration": dict(median_size=3, fc_baseline=2.0, polyorder=3,
                      gauss_sigma=0.0, demean=True),
    "orbit": dict(use_sg_baseline=True, fc_baseline=0.2, polyorder=3,
                  median_size=0, sg_pad_enable=True, sg_pad_len=None,
                  sg_pad_mult=1.0, sg_pad_mode="reflect", demean=True,
                  hp_cutoff_hz=0.0, hp_order=2, lp_cutoff_hz=500.0, lp_order=4),
    "acoustic": dict(median_size=3, fc_baseline=2.0, polyorder=3,
                     gauss_sigma=0.0, demean=True),
}


def filter_spec_hash():
    specification = json.dumps(FILTER_CFG, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(specification.encode("utf-8")).hexdigest()[:10]


def anti_alias_and_decimate(x, fs0=ACQUISITION_HZ, q=2, cutoff_hz=2200.0):
    """Original FIR zero-phase decimation, including its original fallback."""
    try:
        from scipy.signal import decimate
        y = decimate(x, q=q, ftype="fir", zero_phase=True)
        return np.asarray(y, dtype=np.float32)
    except Exception as exc:
        warnings.warn(
            "SciPy decimation failed; using the convolution-based decimation "
            "fallback. Check the input and SciPy installation. "
            "Original error: %s: %s" % (type(exc).__name__, exc),
            RuntimeWarning,
            stacklevel=2,
        )
        nyq = fs0 / 2.0
        fc = min(cutoff_hz, nyq * 0.9) / nyq
        numtaps = 401
        n = np.arange(numtaps) - (numtaps - 1) / 2.0
        h = np.sinc(fc * n)
        w = 0.54 - 0.46 * np.cos(2 * np.pi * (np.arange(numtaps) / (numtaps - 1)))
        h = h * w
        h = h / np.sum(h)
        xf = np.convolve(x, h, mode="same")
        y = xf[::q]
        return np.asarray(y, dtype=np.float32)


def channel_type(ch_1based):
    if 1 <= ch_1based <= 6:
        return "vibration"
    if 7 <= ch_1based <= 8:
        return "orbit"
    if 9 <= ch_1based <= 10:
        return "acoustic"
    raise ValueError("Channel number must be 1–10")


def sg_window_from_fc(fs, fc_baseline, n, polyorder):
    if fc_baseline is None or fc_baseline <= 0:
        return max(polyorder + 2, 101)
    M_est = int(0.443 * fs / fc_baseline)
    max_win = n if (n % 2 == 1) else (n - 1)
    M = max(polyorder + 2, min(M_est, max_win))
    if M % 2 == 0:
        M -= 1
    if M < polyorder + 2:
        M = max_win
        if M % 2 == 0:
            M -= 1
    return int(M)


def median_sg_gaussian(x, fs, cfg):
    """Median filtering, SG baseline removal and demeaning; Gaussian is disabled."""
    x = np.asarray(x, dtype=np.float32)
    k = int(cfg.get("median_size", 3))
    if k and k > 1:
        if k % 2 == 0:
            k += 1
        x = medfilt(x, kernel_size=k).astype(np.float32)
    fc = float(cfg.get("fc_baseline", 2.0))
    poly = int(cfg.get("polyorder", 3))
    win = sg_window_from_fc(fs, fc, len(x), poly)
    if win >= len(x):
        win = len(x) - 1 if (len(x) % 2 == 0) else len(x)
    if win % 2 == 0:
        win -= 1
    baseline = savgol_filter(x.astype(np.float64), window_length=win,
                             polyorder=poly, mode="interp")
    x = (x.astype(np.float64) - baseline).astype(np.float32)
    if bool(cfg.get("demean", True)):
        x = (x - np.mean(x, dtype=np.float64)).astype(np.float32)
    return x.astype(np.float32, copy=False)


def orbit_baseline_demean_hp_lp(x, fs, cfg):
    """Reflected SG baseline removal, demeaning and zero-phase orbit low-pass."""
    x = np.asarray(x, dtype=np.float32)
    k = int(cfg.get("median_size", 0))
    if k and k > 1:
        if k % 2 == 0:
            k += 1
        x = medfilt(x, kernel_size=k).astype(np.float32)
    if bool(cfg.get("use_sg_baseline", True)):
        fc = float(cfg.get("fc_baseline", 0.5))
        poly = int(cfg.get("polyorder", 3))
        n0 = len(x)
        win = sg_window_from_fc(fs, fc, n0, poly)
        if win >= n0:
            win = n0 - 1 if (n0 % 2 == 0) else n0
        if win % 2 == 0:
            win -= 1
        pad_enable = bool(cfg.get("sg_pad_enable", True))
        pad_mode = str(cfg.get("sg_pad_mode", "reflect"))
        pad_mult = float(cfg.get("sg_pad_mult", 1.0))
        pad_len_cfg = cfg.get("sg_pad_len", None)
        if pad_enable:
            if pad_len_cfg is None:
                pad_len = int(round(win * pad_mult))
            else:
                pad_len = int(pad_len_cfg)
            pad_len = max(0, min(pad_len, n0 - 1))
        else:
            pad_len = 0
        if pad_len > 0:
            x_pad = np.pad(x.astype(np.float64), (pad_len, pad_len), mode=pad_mode)
            baseline_pad = savgol_filter(x_pad, window_length=win,
                                         polyorder=poly, mode="interp")
            x = (x_pad - baseline_pad)[pad_len:-pad_len].astype(np.float32, copy=False)
        else:
            baseline = savgol_filter(x.astype(np.float64), window_length=win,
                                      polyorder=poly, mode="interp")
            x = (x.astype(np.float64) - baseline).astype(np.float32)
    if bool(cfg.get("demean", True)):
        x = x - np.mean(x, dtype=np.float64).astype(np.float32)
    hp_cut = float(cfg.get("hp_cutoff_hz", 0.0))
    hp_order = int(cfg.get("hp_order", 2))
    if hp_cut is not None and hp_cut > 0:
        wn_hp = hp_cut / (fs / 2.0)
        sos_hp = butter(hp_order, wn_hp, btype="highpass", output="sos")
        x = sosfiltfilt(sos_hp, x.astype(np.float64)).astype(np.float32, copy=False)
    lp_cut = float(cfg.get("lp_cutoff_hz", 500.0))
    lp_order = int(cfg.get("lp_order", 4))
    wn_lp = lp_cut / (fs / 2.0)
    if wn_lp >= 1.0:
        return x.astype(np.float32, copy=False)
    if wn_lp <= 0:
        raise ValueError("lp_cutoff_hz must be positive")
    sos_lp = butter(lp_order, wn_lp, btype="lowpass", output="sos")
    return sosfiltfilt(sos_lp, x.astype(np.float64)).astype(np.float32, copy=False)


def filter_record(record_5khz):
    """Filter a complete, synchronized 5 kHz record shaped [10,T]."""
    signal = np.asarray(record_5khz, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] != 10 or signal.shape[1] < 32:
        raise ValueError("Expected a complete 5 kHz record [10,T], T >= 32")
    if not np.isfinite(signal).all():
        raise ValueError("Record contains non-finite samples")
    result = np.empty_like(signal, dtype=np.float32)
    for channel in range(1, 11):
        kind = channel_type(channel)
        cfg = FILTER_CFG[kind]
        fn = orbit_baseline_demean_hp_lp if kind == "orbit" else median_sg_gaussian
        result[channel - 1] = fn(signal[channel - 1], SAMPLE_RATE_HZ, cfg)
    return result


def preprocess_record(record_10khz, trim_to_training_duration=True):
    """Apply the acquisition-to-DATA2 pipeline; default takes the last 300 seconds.

    Disabling trimming processes the supplied complete record at its own length.
    That option does not reproduce the original 300-second boundary conditions.
    """
    signal = np.asarray(record_10khz, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] != 10 or not np.isfinite(signal).all():
        raise ValueError("Expected a finite synchronized 10 kHz record [10,T]")
    if trim_to_training_duration:
        samples = ACQUISITION_HZ * TRAINING_RECORD_SECONDS
        if signal.shape[1] < samples:
            raise ValueError("Frozen acquisition protocol requires at least 300 seconds at 10 kHz")
        signal = signal[:, -samples:]
    downsampled = np.stack([anti_alias_and_decimate(row) for row in signal])
    return filter_record(downsampled)


def window_at(preprocessed_record, start_sample, as_float32=True):
    """Select one 6000-sample DATA4 window, preserving the saved float16 rounding."""
    signal = np.asarray(preprocessed_record)
    start = int(start_sample)
    if signal.ndim != 2 or signal.shape[0] != 10 or start < 0 or start + WINDOW_SAMPLES > signal.shape[1]:
        raise ValueError("Window is outside the preprocessed ten-channel record")
    # Frozen raw_norm is 'none'; float16 storage is followed by float32 model input.
    value = signal[:, start:start + WINDOW_SAMPLES].astype(np.float32, copy=False)
    value = value.astype(np.float16, copy=False)
    return value.astype(np.float32) if as_float32 else value
