# Hierarchical compound-fault diagnosis

Code and frozen model parameters for *Hierarchical zero-shot compound-fault diagnosis for marine propulsion shaft systems via structure-conditioned attribute inference*.

## Status

The code and bundled model assets are licensed under the [Non-Commercial Research License](LICENSE).

Repository URL: https://github.com/zhutingfeng-526/zero-shot-shaft-diagnosis

The frozen structure model, its five matching source-prototype sets, U/M/I experts and temporal integration are included. Complete inference takes a preprocessed 9.6 s history and returns the fault composition, available attributes and branch probabilities. See the [model assets](docs/REQUIRED_ASSETS.md) and [input contract](docs/INPUTS.md).

## Setup and checks

Use a fresh Python 3.8 environment; see the [installation instructions](docs/INSTALL.md) for creation and activation commands. The default installation is CPU-only. Run from the repository root with that environment's Python:

```sh
python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_release.py --tests
python scripts/report_environment.py --driver
```

CPU installation uses PyTorch 1.11.0+cpu, NumPy 1.24.4 and SciPy 1.10.1. The optional `requirements-cuda113.txt` selects the CUDA build used in the local GPU checks; install it in a separate environment. These are release-validation environments, not a recovered historical GPU image. The [environment record](docs/ENVIRONMENT.md) distinguishes historical evidence and local validation.

The release check reports dependency versions, runtime flags, import errors, file checksums and missing assets. Exit code 0 means those checks passed; exit code 2 identifies an unresolved asset or failed check. It does not certify a full experimental rerun. The environment command only inspects settings; it does not change precision or backend flags. Retained numerical fallbacks now emit a warning if used; investigate such warnings before comparing results.

[Frozen-result regression checks](docs/REGRESSION.md) separately cover synthetic numerical references and rescoring of authorized predictions and labels. Experimental waveforms, predictions and record indices are not bundled.

Use the source checkout. For imports in another script, an editable installation is supported:

```sh
python -m pip install --no-deps --no-build-isolation -e .
```

Keep `models/` and `configs/` alongside `src/`. A standalone wheel is not provided; moving only the Python package omits the model resources.

## Inference

Inputs are the study's preprocessed, amplitude-preserving waveforms, not unfiltered acquisition voltages. The [input contract](docs/INPUTS.md) defines channel order, shapes and labels.

To exercise the complete interface without experimental data:

```sh
python scripts/make_example.py
python scripts/predict.py --expert system --input data/example_history.npz --output data/example_prediction.npz --device cpu
```

This generates artificial sine waves and runs the actual released models. Its predictions are an interface check, not experimental performance. See the [output fields](docs/INPUTS.md#complete-system-output-fields) for interpreting the saved arrays.

For authorized experimental inputs:

```sh
python scripts/predict.py --expert system --input data/history.npz --output outputs/diagnosis.npz
python scripts/predict.py --expert structure --input data/windows.npz --output outputs/structure.npz
python scripts/predict.py --expert u --input data/u_windows.npz --output outputs/u_predictions.npz
python scripts/predict.py --expert m --input data/m_windows.npz --output outputs/m_predictions.npz
python scripts/predict.py --expert i --input data/i_history.npz --output outputs/i_predictions.npz
```

Create the output directory first. Existing output files are not overwritten. Add `--device cuda` to use a compatible CUDA installation; the default is CPU. The standalone experts return attributes conditional on their component being present. Complete inference applies the final structure gate automatically.

`integration.py` contains the frozen structure refinement, trailing-history aggregation and composition-conditioned output assembly. `evaluation.py` implements condition/run-balanced structure and complete-diagnosis metrics. The U correction uses the **base structure's predicted I bit**, not a ground-truth label or a revised final structure decision.

`MisalignmentExpert` defaults to the final M configuration. The lower-level `predict_misalignment` helper requires an explicit `position_mode='final'` or `'a1'`; final mode also requires its three-member confirmation probability. Omitting the mode is an error, not an implicit choice of a historical comparator.

The exported parameters were checked against their original tensors. Sampled real-history tests compared the extracted computations with the original implementations and the saved final diagnosis labels. Floating-point scores can differ from archived GPU caches; bitwise cross-device reproducibility is not assumed. These checks are not a rerun of every experimental result.

## Contents

- `src/shaft_diagnosis/`: preprocessing, structure model/decoder, U/M/I experts, integration, metrics and core training/loss functions.
- `models/`: selected numerical weights, source statistics, prototypes and calibrations. No optimizer states or training histories.
- `configs/`: selected settings only.
- `tests/`: synthetic regression and contract checks; these are not experimental performance tests.
- `release_manifest.json`: file checksums and required-asset status.

Training functions expose the retained learning operations. A complete train-from-scratch command, acquisition-file reader and experiment scheduler are not included. The code does not rerun hyperparameter searches or reproduce every manuscript table with one command.

## Data and reuse

The experimental data are not publicly available due to confidentiality restrictions but may be obtained from the corresponding author, Xiaoming Ye (`xmye@hust.edu.cn`), upon reasonable request, subject to these restrictions. Raw measurements, private indices, prediction caches, manuscript drafts and development logs are not bundled.

The [Non-Commercial Research License](LICENSE) covers this project's code, accompanying configurations and documentation, and bundled model assets. It permits use, modification and redistribution solely for non-commercial research, with copyright and license notices retained and modifications identified. Commercial use requires separate written authorization.

Copyright (c) 2026 Tingfeng Zhu. Third-party dependencies retain their own licenses, and experimental data are not covered. This is a custom non-commercial research license, not an OSI-approved open-source license.

Author, title and software-version metadata are provided in [CITATION.cff](CITATION.cff). No publication DOI has been assigned.
