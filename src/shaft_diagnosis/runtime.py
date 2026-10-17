"""Read numerical runtime settings without changing inference configuration."""
import importlib
import os
import platform
import shutil
import subprocess
import sys


def runtime_report(include_driver=False):
    """Return process settings, not a claim of historical environment identity.

    Call this in the inference process to capture effective backend flags.
    Running the command separately reports that new process's settings only.
    No seed, precision, thread count, or backend flag is changed here.
    """
    result = {
        "scope": "Current process; read-only inspection, not historical validation",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "TORCH_CUDNN_V8_API_DISABLED", "TORCH_CUDNN_V8_API_DEBUG",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    for name in ("numpy", "scipy", "torch"):
        try:
            module = importlib.import_module(name)
            result["packages"][name] = module.__version__
        except ImportError:
            result["packages"][name] = None
    if result["packages"]["torch"] is None:
        result["torch_error"] = "PyTorch is not installed"
        return result
    import torch
    result["torch_git_version"] = torch.version.git_version
    result["cuda_build"] = torch.version.cuda
    result["cudnn_version"] = torch.backends.cudnn.version()
    result["cuda_available"] = torch.cuda.is_available()
    result["backend_flags"] = {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "autocast_enabled_outside_branch": bool(torch.is_autocast_enabled()),
        "default_dtype": str(torch.get_default_dtype()),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
    }
    optional = (("float32_matmul_precision", "get_float32_matmul_precision"),
                ("deterministic_warn_only", "is_deterministic_algorithms_warn_only_enabled"))
    for key, method in optional:
        if hasattr(torch, method):
            result["backend_flags"][key] = getattr(torch, method)()
    if hasattr(torch, "get_autocast_gpu_dtype"):
        result["backend_flags"]["cuda_autocast_target_dtype"] = str(torch.get_autocast_gpu_dtype())
    result["gpus"] = []
    if result["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(index)
            result["gpus"].append({"index": index, "name": gpu.name,
                                   "compute_capability": [gpu.major, gpu.minor],
                                   "total_memory_bytes": gpu.total_memory})
    if include_driver:
        executable = shutil.which("nvidia-smi")
        result["driver_query"] = {"status": "unavailable"}
        if executable:
            try:
                query = subprocess.run(
                    [executable, "--query-gpu=index,driver_version", "--format=csv,noheader"],
                    check=True, capture_output=True, text=True, timeout=10)
                result["driver_query"] = {"status": "ok", "nvidia_smi_gpu_index_and_driver":
                                           query.stdout.strip().splitlines()}
            except (OSError, subprocess.SubprocessError) as exc:
                result["driver_query"] = {"status": "error", "error_type": type(exc).__name__}
    return result
