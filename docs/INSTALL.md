# Installation and smoke test

Run the commands from the source checkout. Models and configurations remain next to `src/`; installing only a wheel is not supported.

## CPU environment

The CPU profile pins Python dependencies independently of an existing research environment. Use Python 3.8; the tested interpreter is 3.8.20. Python itself is not installed by pip.

Windows PowerShell, without activating scripts:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_release.py --tests
.\.venv\Scripts\python.exe scripts/check_frozen_results.py --mode synthetic
.\.venv\Scripts\python.exe scripts/make_example.py
.\.venv\Scripts\python.exe scripts/predict.py --expert system --input data/example_history.npz --output data/example_prediction.npz --device cpu
```

On Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. Linux installation is not part of the Windows clean-environment acceptance test.

Alternatively, activate the environment and use `python` for subsequent commands. The CPU profile selects `torch==1.11.0+cpu` from the official PyTorch wheel index; it also pins NumPy, SciPy, typing extensions and the editable-build tools. Existing model parameters are unchanged.

Checks should exit with code 0. The example produces numeric NPZ input and output; neither is experimental data. Both example and prediction commands reject existing output files. Use different filenames or retain previous outputs outside the checkout before rerunning.

Acceptance on Windows/Python 3.8.20 used a new virtual environment with system site-packages disabled. Editable installation and `pip check` passed, all 55 tests passed, and the complete example returned 25 output fields with 281 supporting windows and finite numerical values. The original research environment was not modified.

## Optional CUDA environment

Create a separate environment and replace `requirements-cpu.txt` with `requirements-cuda113.txt` in the installation command. That profile selects `torch==1.11.0+cu113`; it requires a compatible NVIDIA GPU and driver. NumPy and SciPy pins are the same. The unused torchvision and torchaudio packages are not required.

Run `python scripts/report_environment.py --driver` and verify `cuda_available` before selecting `--device cuda`. The CUDA profile matches the existing local GPU validation environment; it is not presented as a separately tested clean GPU installation or as the historical server image. Wheel sources follow the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v1110).

Neither installation profile changes TF32, AMP, deterministic flags or calibration. See [ENVIRONMENT.md](ENVIRONMENT.md) for numerical settings and [REGRESSION.md](REGRESSION.md) for the distinction between interface tests and experimental result checks.
