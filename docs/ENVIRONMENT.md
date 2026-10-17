# Numerical environment

The release preserves the selected weights, prototypes, calibration and inference operations. Environment evidence is recorded separately from model configuration in [runtime_evidence.json](../configs/runtime_evidence.json); inference does not read that file or apply a precision profile.

## Historical records and local validation

| Evidence | Recorded environment | Scope |
| --- | --- | --- |
| Selected structure-training preflight | Python 3.8.20; NumPy 1.24.4; PyTorch 2.1.2; Linux; RTX A4000 | Direct record for training, not final inference |
| Related training/replay audits | PyTorch 2.1.2; CUDA build 11.8; cuDNN 8.7 | Direct records in related historical runs |
| Final dense inference preflight | RTX A4000 | Complete PyTorch/CUDA/cuDNN/driver lock not recovered |
| Local release validation | Python 3.8.20; NumPy 1.24.4; SciPy 1.10.1; PyTorch 1.11.0+cu113; CUDA build 11.3; cuDNN 8.2; Windows; RTX 4060 Ti | CPU and CUDA portability checks, not the historical execution image |
| Clean CPU installation | Python 3.8.20; NumPy 1.24.4; SciPy 1.10.1; PyTorch 1.11.0+cpu; Windows | Isolated virtual environment, editable installation, dependency checks, tests and complete artificial-history inference |

`requirements-cpu.txt` and `requirements-cuda113.txt` select explicit local CPU and CUDA builds; the shared `requirements.txt` and package metadata retain their common dependency versions. They do not reconstruct the historical GPU environment. The related CUDA 11.8/cuDNN 8.7 records identify a candidate stack for further historical validation, not a fully recovered final-run lock. No container or environment file is presented as that missing lock. Historical source identifiers and SHA-256 values are retained in the evidence JSON; private logs are not bundled.

## Precision and batching

- Structure inference and the M expert's shared K5 feature extractor use CUDA autocast. The selected structure-training run had AMP disabled; inference explicitly enabled it. Its training source also enabled deterministic algorithms with warnings, disabled cuDNN benchmark, enabled cuDNN deterministic and supplied `CUBLAS_WORKSPACE_CONFIG=:4096:8`. These training settings are not restored by loading model tensors in a fresh inference process.
- Outside these contexts, the U and remaining M neural paths normally use float32. The I expert uses NumPy/SciPy operations. Input/storage float16 rounding is a separate retained operation, not an AMP setting.
- The release does not set TF32, cuDNN benchmark/deterministic, or deterministic-algorithm flags at inference startup. They follow the current process. A related old-runner control recorded matrix TF32 off, cuDNN TF32 on and deterministic/benchmark flags off; the final dense inference process did not retain an equivalent flag snapshot.
- Historical sparse model inference used batches of 2048; dense evidence generation used 256. `DiagnosisPredictor.predict` defaults to 16 windows per batch. These are execution settings, not different model parameters. Restoring batch size alone did not recover all archived probabilities in local diagnostics.

Backend defaults vary with PyTorch versions. In particular, matrix TF32 defaulted to on in PyTorch 1.11 and off from 1.12; cuDNN TF32 is separate. See [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.1/notes/cuda.html). Do not change thresholds or precision settings merely to make archived decisions match.

## Capture the environment used for a run

```sh
python scripts/report_environment.py --driver
```

This read-only command prints versions, CUDA build, cuDNN, GPU/driver information and backend flags. The CUDA build is the version associated with PyTorch, not the maximum CUDA version displayed by `nvidia-smi`. The optional driver query uses `nvidia-smi` if available.

For a precise run record, call the helper **in the inference process**, after any application setup:

```python
from shaft_diagnosis.runtime import runtime_report
environment = runtime_report(include_driver=True)
```

It does not change flags, seeds, precision or thread counts. A separate invocation reports a new process's defaults, not another process's settings. `autocast_enabled_outside_branch=False` does not mean branch-local CUDA autocast is disabled. The prediction CLI sets CPU threads to 1 by default; the reporting command leaves the existing thread count unchanged.

## Interpreting a regression check

Asset hashes establish parameter identity. Synthetic numerical references check decoder, aggregation and scoring operations. Authorized frozen predictions and labels can be rescored against historical aggregate results; this is distinct from recomputing predictions from waveforms. See [regression checks](REGRESSION.md).

Local diagnostics found that small feature changes can cross fixed decoder or acceptance boundaries. Discrete votes and gated probabilities can therefore differ substantially even when underlying feature differences are small. A failed frozen comparison must be reported and investigated, not hidden by changing calibration or widening a tolerance after observing the result. Bitwise cross-device inference is not assumed.

Repeated CUDA execution can also vary on the same device. The retained U phase-folding operation uses `scatter_add_`, followed by the original float16 storage rounding. Different accumulation order can cross a rounding boundary and change a folded value. Local isolation reproduced both observed U probability variants using the unchanged original implementation. The [PyTorch scatter-add documentation](https://docs.pytorch.org/docs/2.1/generated/torch.Tensor.scatter_add_.html) describes CUDA nondeterminism for this operation. The release preserves this calculation rather than replacing it to force agreement.
