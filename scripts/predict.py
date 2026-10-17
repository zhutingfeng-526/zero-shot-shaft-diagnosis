"""Run complete diagnosis or one branch on preprocessed numeric NPZ waveforms."""
from pathlib import Path
import argparse
import sys
import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def predict(expert, inputs, device="cpu"):
    raw = inputs["raw"]
    if expert == "structure":
        from shaft_diagnosis.structure import StructurePredictor
        return StructurePredictor(ROOT / "models/structure", device).predict(raw)
    rpm = inputs["rpm"]
    if expert == "system":
        from shaft_diagnosis.system import DiagnosisPredictor
        if np.asarray(rpm).size != 1:
            raise ValueError("Complete diagnosis requires one speed for its 9.6 s history")
        return DiagnosisPredictor(ROOT, device).predict(raw, float(np.asarray(rpm).reshape(-1)[0]))
    if expert == "u":
        from shaft_diagnosis.unbalance import UnbalancePredictor
        return UnbalancePredictor(ROOT / "models/unbalance", device).predict(raw, rpm, inputs["companion_i"])
    if expert == "m":
        from shaft_diagnosis.misalignment import MisalignmentExpert
        return MisalignmentExpert(ROOT / "models/misalignment", device).predict(raw, rpm)
    from shaft_diagnosis.impact import ImpactExpert
    if np.asarray(rpm).size != 1:
        raise ValueError("The I branch requires one speed for its single 9.6 s buffer")
    result = ImpactExpert(ROOT / "models/impact", ROOT / "configs/impact.json").predict(
        raw, float(np.asarray(rpm).reshape(-1)[0]))
    return {"probabilities": result["probabilities"], "ready": result["ready"],
            "position": result["legacy_prediction"] if result["ready"] else -1,
            "event_count": result["event_count"], "class_names": result["class_names"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=("system", "structure", "u", "m", "i"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    with np.load(args.input, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    result = {key: np.asarray(value) for key, value in predict(args.expert, inputs, args.device).items()}
    if any(value.dtype.hasobject for value in result.values()):
        raise TypeError("Output must contain only numeric or fixed-width string arrays")
    with args.output.open("xb") as output:
        np.savez_compressed(output, **result)
    print("Saved " + str(args.output))


if __name__ == "__main__":
    main()
