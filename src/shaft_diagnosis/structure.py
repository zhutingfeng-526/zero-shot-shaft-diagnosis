"""Frozen structure runtime; explicit prototype assets are required for raw inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .structure_decoder import BITS, decode_margin_strict_v12, equal_consensus, structure_indices
from .structure_model import RealSemanticSameSpaceModel
from .structure_evidence import prototype_margin_and_grade
from .unbalance import _load_state_npz
from .labels import PUBLIC_STRUCTURES


def _load_prototypes_npz(path, expected_sha256, model, device):
    """Validate the frozen prototype schema and its checkpoint null vectors."""
    path = Path(path)
    if not expected_sha256 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Prototype SHA-256 mismatch: " + str(path))
    dimensions = {"U": 3, "M": 6, "R": 3}
    z_dim = int(model.base_model.p_null["U"].numel())
    schema = {}
    for component, grades in dimensions.items():
        schema.update({
            "P_" + component: ((grades, z_dim), np.dtype("float32")),
            "P_" + component + "_valid": ((grades,), np.dtype("bool")),
            "P_" + component + "_grade_ids": ((grades,), np.dtype("int64")),
            "p_" + component + "0": ((z_dim,), np.dtype("float32")),
        })
    result = {}
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(schema):
            raise ValueError("Prototype tensor keys do not match the schema")
        for name, (shape, dtype) in schema.items():
            value = arrays[name]
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError("Prototype tensor schema mismatch: " + name)
            result[name] = torch.from_numpy(value.copy()).to(device)
    for component, grades in dimensions.items():
        if not bool(result["P_" + component + "_valid"].all()):
            raise ValueError("Every component-grade prototype must be valid")
        if not torch.equal(result["P_" + component + "_grade_ids"], torch.arange(grades, device=device)):
            raise ValueError("Prototype grade IDs are not in the frozen order")
        if not torch.equal(result["p_" + component + "0"], model.base_model.p_null[component]):
            raise ValueError("Prototype null vector does not match checkpoint: " + component)
    return result


def predict_from_evidence(margins_by_epoch, count_logits_by_epoch, calibration):
    """Decode checkpoint-specific prototype margins with the frozen shared V12 rule."""
    epochs = [56, 57, 58, 59, 60]
    decisions = {}
    for epoch in epochs:
        margin = np.asarray(margins_by_epoch[epoch], dtype=np.float32)
        logits = np.asarray(count_logits_by_epoch[epoch], dtype=np.float32)
        if margin.ndim != 2 or margin.shape[1] != 3 or logits.shape != margin.shape:
            raise ValueError("Each epoch requires margin and count logits [N,3]")
        if not np.isfinite(margin).all() or not np.isfinite(logits).all():
            raise ValueError("Evidence must be finite")
        bits = decode_margin_strict_v12(margin, logits.argmax(1), calibration["decoder"])
        decisions[epoch] = structure_indices(bits)
    reliability = {int(k): float(v) for k, v in calibration["source_reliability"].items()}
    pred, prob, entropy, tie, matrix = equal_consensus(decisions, epochs, reliability)
    return {"structure_index": pred, "structure": np.asarray(PUBLIC_STRUCTURES)[pred],
            "bits": BITS[pred], "probabilities": prob, "entropy": entropy,
            "tie": tie, "checkpoint_decisions": matrix}


class StructurePredictor:
    """Five-checkpoint inference with matching source prototypes."""
    def __init__(self, model_dir, device="cpu"):
        directory = Path(model_dir)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if [int(m["epoch"]) for m in manifest["members"]] != [56, 57, 58, 59, 60]:
            raise ValueError("Final structure method requires epochs 56 through 60 in order")
        missing = [str(directory / m["prototype_file"]) for m in manifest["members"]
                   if not (directory / m["prototype_file"]).is_file()]
        if missing:
            raise FileNotFoundError("Missing checkpoint-specific source prototypes: " + "; ".join(missing))
        self.calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        self.config = manifest["model_config"]
        self.device = torch.device(device)
        self.use_amp = bool(manifest.get("inference_use_amp", True)) and self.device.type == "cuda"
        self.models, self.prototypes = {}, {}
        for m in manifest["members"]:
            epoch = int(m["epoch"])
            model = RealSemanticSameSpaceModel(self.config)
            _load_state_npz(model, directory / m["state_file"], m["sha256"])
            self.models[epoch] = model.to(self.device).eval()
            self.prototypes[epoch] = _load_prototypes_npz(
                directory / m["prototype_file"], m.get("prototype_sha256"),
                self.models[epoch], self.device)

    @torch.no_grad()
    def predict(self, windows):
        x = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
        if x.ndim != 3 or x.shape[1] != 10 or not bool(torch.isfinite(x).all()):
            raise ValueError("Expected finite windows [N,10,T]")
        mode = self.config.get("input_norm", "none")
        if mode == "demean":
            x = x - x.mean(dim=2, keepdim=True)
        elif mode == "zscore":
            x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True, unbiased=False) + 1e-8)
        elif mode != "none":
            raise ValueError("Unsupported input normalization: " + mode)
        margins, counts = {}, {}
        for epoch, model in self.models.items():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = model(x)
            margin, _grades = prototype_margin_and_grade(out, self.prototypes[epoch], self.device, "cosine")
            margins[epoch] = margin.cpu().numpy()
            counts[epoch] = out["count_logits"].float().cpu().numpy()
        return predict_from_evidence(margins, counts, self.calibration)
