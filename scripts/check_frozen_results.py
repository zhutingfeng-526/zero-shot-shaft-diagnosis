"""Run synthetic references or re-score an authorized numeric endpoint bundle.

Synthetic tests do not reproduce paper metrics. Experimental predictions, truth,
and record/window keys are not distributed. Missing authorized data is a skip
(exit 2), not a pass. No model fitting or inference settings are changed.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shaft_diagnosis.regression import (REFERENCE_VERSION, check_endpoints,
                                       check_synthetic, load_endpoint_bundle, read_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "frozen-endpoints", "current-endpoints"), required=True)
    parser.add_argument("--bundle", type=Path, help="Authorized endpoint NPZ; see docs/REGRESSION.md")
    parser.add_argument("--split", choices=("final_pair", "final_triple"))
    parser.add_argument("--candidate", choices=("B0_C0_T0", "E0_C3_T7"))
    args = parser.parse_args(argv)
    try:
        if args.mode == "synthetic":
            if args.bundle is not None or args.split is not None or args.candidate is not None:
                parser.error("Synthetic mode does not accept experimental bundle options")
            result = check_synthetic(ROOT)
        elif args.bundle is None or not args.bundle.is_file():
            result = {"mode": args.mode, "status": "skipped", "passed": False,
                      "reason": "An authorized endpoint bundle is required; no experimental data are bundled.",
                      "neural_inference_executed": False, "full_raw_data_reproduction": False}
        elif args.split is None or args.candidate is None:
            parser.error("Endpoint modes require --split and --candidate")
        else:
            reference = read_json(ROOT / "configs/regression/frozen_endpoint_reference.json")
            if reference.get("schema_version") != REFERENCE_VERSION:
                raise ValueError("Unsupported frozen endpoint reference schema")
            group = reference["groups"][args.split + "/" + args.candidate]
            result = check_endpoints(load_endpoint_bundle(args.bundle), group, args.mode)
            result.update(split=args.split, candidate=args.candidate)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result = {"mode": args.mode, "status": "failed", "passed": False,
                  "reason": type(exc).__name__ + ": " + str(exc),
                  "neural_inference_executed": False, "full_raw_data_reproduction": False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
