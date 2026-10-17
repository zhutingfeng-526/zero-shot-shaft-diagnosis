"""Frozen five-member U inference from raw windows or frontend tensors."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .unbalance_features import extract_u_frontend, transient_suppressed_waveform
from .unbalance_model import RobustTransportNet
from .unbalance_output import normalize_output
from .unbalance_consensus import build_shift_consensus


def _load_state_npz(model, path, expected_sha256=None):
    """Load numeric arrays only; validate exact tensor keys and shapes."""
    path = Path(path)
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Model SHA-256 mismatch: " + str(path))
    with np.load(path, allow_pickle=False) as arrays:
        expected = model.state_dict()
        if set(arrays.files) != set(expected):
            raise ValueError("Model tensor keys do not match the architecture")
        state = {}
        for name, tensor in expected.items():
            value = arrays[name]
            if value.dtype.kind not in 'biufc' or tuple(value.shape) != tuple(tensor.shape):
                raise ValueError("Model tensor schema mismatch: " + name)
            state[name] = torch.from_numpy(value.copy())
    model.load_state_dict(state, strict=True)
    return model


class UnbalancePredictor:
    """Companion-I rows use median transport; remaining rows keep raw U physics.

    `companion_i` is the upstream structure routing decision, never a test label.
    This branch returns U degree conditional on U being present; the caller is
    responsible for the upstream U-presence gate.
    """
    def __init__(self, model_dir, device="cpu"):
        directory = Path(model_dir)
        self.contract = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.device = torch.device(device)
        self.models = {}
        for member in manifest["members"]:
            model = RobustTransportNet(member["spec"], self.contract)
            _load_state_npz(model, directory / member["state_file"], member["sha256"])
            self.models[int(member["seed"])] = model.to(self.device).eval()
        if list(self.models) != list(range(202731, 202736)):
            raise ValueError("Final U method requires the five frozen seeds 202731–202735")

    @torch.no_grad()
    def predict_features(self, physics, phase, companion_i):
        physics = torch.as_tensor(physics, dtype=torch.float32, device=self.device)
        phase = torch.as_tensor(phase, dtype=torch.float32, device=self.device)
        mask = np.asarray(companion_i, dtype=bool).reshape(-1)
        n = len(mask)
        if tuple(physics.shape) != (n, 20) or tuple(phase.shape) != (n, 8, 64):
            raise ValueError("Expected physics [N,20], phase [N,8,64], and companion_i [N]")
        if not bool(torch.isfinite(physics).all() and torch.isfinite(phase).all()):
            raise ValueError("U frontend tensors must be finite")
        zeros = torch.zeros(n, dtype=torch.long, device=self.device)
        ones = torch.ones(n, dtype=torch.long, device=self.device)
        raw = normalize_output(next(iter(self.models.values()))(physics, phase, zeros))
        shared = {seed: normalize_output(model(physics, phase, ones)) for seed, model in self.models.items()}
        return build_shift_consensus(raw, shared, list(self.models), mask, self.contract, "median")

    @torch.no_grad()
    def predict(self, windows, rpm, companion_i, fs_hz=5000.0):
        x = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
        speed = torch.as_tensor(rpm, dtype=torch.float32, device=self.device).reshape(-1)
        if x.ndim != 3 or x.shape[1] != 10 or len(speed) != len(x):
            raise ValueError("Expected windows [N,10,T] and rpm [N]")
        if x.shape[2] < 256 or not bool(torch.isfinite(x).all()):
            raise ValueError("Windows must be finite with at least 256 samples")
        if not bool(torch.isfinite(speed).all() and (speed > 0).all()) or not np.isfinite(fs_hz) or fs_hz <= 0:
            raise ValueError("Sampling rate and RPM must be positive and finite")
        features = extract_u_frontend(x, speed, fs_hz, transient_suppressed_waveform)
        # The frozen window interface stores phase folds as float16 before inference.
        phase = features["phase"].to(torch.float16).to(torch.float32)
        return self.predict_features(features["physics"], phase, companion_i)
