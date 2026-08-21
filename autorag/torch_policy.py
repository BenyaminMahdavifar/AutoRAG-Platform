"""
Canonical PyTorch Installation Policy and Embedding Device Separation for AutoRAG.

This module enforces the strict conceptual separation between:
1. Environment Installation: Always install & maintain a CUDA-enabled PyTorch build.
2. Embedding Runtime Placement: Resolved at runtime per Connection Profile (CPU, CUDA, Auto).
"""

import os
import sys
import subprocess
from typing import Dict, Any, Optional

# Canonical PyTorch CUDA Wheel Specification
PYTORCH_CUDA_VARIANT = "cu124"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
PYTORCH_PACKAGE = "torch"


class CUDAUnavailableError(RuntimeError):
    """Raised when a Connection Profile explicitly requests CUDA compute, but CUDA is unavailable."""
    def __init__(self, message: str = "This profile explicitly requires CUDA, but CUDA runtime is not currently available on this system."):
        super().__init__(message)
        self.code = "CUDA_RUNTIME_UNAVAILABLE"
        self.recommendations = [
            "Verify that an NVIDIA GPU and compatible NVIDIA graphics drivers are installed.",
            "Run environment diagnostics in Connection Settings -> Environment Lifecycle.",
            "Set Device Compute to 'Auto (Best Available)' or 'CPU' in your Connection Profile."
        ]

    def to_dict(self, torch_version: Optional[str] = None, cuda_runtime: Optional[str] = None) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "torchBuild": "cuda" if cuda_runtime else "unknown",
            "torchVersion": torch_version,
            "torchCudaVersion": cuda_runtime,
            "recommendations": self.recommendations
        }


def resolve_embedding_device(configured_device: Optional[str] = "auto") -> str:
    """
    Canonically resolves the runtime device for embedding models.

    Behavior matrix:
    - 'cpu': Returns 'cpu'. The embedding model runs on CPU using the CUDA-enabled PyTorch build.
    - 'cuda': Checks torch.cuda.is_available(). If available -> 'cuda'. If NOT available -> raises CUDAUnavailableError.
    - 'auto': Checks torch.cuda.is_available(). If available -> 'cuda', otherwise falls back to 'cpu'.
    - 'mps': Checks torch.backends.mps.is_available(). If available -> 'mps', otherwise falls back to 'cpu'.

    Note:
    - Connection Profile settings NEVER trigger PyTorch package reinstalls or .venv rebuilds.
    - Resolving to 'cpu' does NOT mean CPU-only PyTorch is installed.
    """
    dev = (configured_device or "auto").strip().lower()

    if dev == "cpu":
        return "cpu"

    if dev == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                torch_ver = getattr(torch, "__version__", "unknown")
                cuda_ver = getattr(torch.version, "cuda", None)
                raise CUDAUnavailableError(
                    f"Connection Profile requested CUDA compute, but torch.cuda.is_available() is False. (PyTorch: {torch_ver}, Built CUDA: {cuda_ver or 'None'})."
                )
            return "cuda"
        except ImportError:
            raise CUDAUnavailableError("PyTorch is not installed in the current environment.")

    if dev == "mps":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    if dev in ("auto", "best", "default"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    raise ValueError(f"Invalid embedding device setting: '{configured_device}'. Expected 'auto', 'cpu', or 'cuda'.")


def get_nvidia_driver_info() -> Dict[str, Any]:
    """Safely query nvidia-smi for host GPU / driver diagnostics without requiring local nvcc/CUDA SDK."""
    info: Dict[str, Any] = {
        "nvidia_smi_available": False,
        "gpu_detected": False,
        "gpu_name": "N/A",
        "driver_version": "N/A",
        "cuda_driver_version": "N/A"
    }

    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout:
            info["nvidia_smi_available"] = True
            info["gpu_detected"] = True
            lines = res.stdout.splitlines()
            for line in lines:
                if "Driver Version:" in line:
                    parts = line.split("Driver Version:")
                    if len(parts) > 1:
                        sub = parts[1].split()
                        if sub:
                            info["driver_version"] = sub[0]
                    if "CUDA Version:" in line:
                        sub_cuda = line.split("CUDA Version:")[1].split()
                        if sub_cuda:
                            info["cuda_driver_version"] = sub_cuda[0]
                if "NVIDIA" in line and "%" in line:
                    # Line with GPU name
                    name_part = line.split("|")[1].strip()
                    if name_part:
                        info["gpu_name"] = name_part
    except Exception:
        pass

    return info


def inspect_installed_torch(python_exec: Optional[str] = None) -> Dict[str, Any]:
    """
    Examines the installed PyTorch backend and validates CPU tensor readiness + CUDA status.
    Can be run in-process or via python_exec subprocess.
    """
    if python_exec and python_exec != sys.executable:
        code = """
import sys
import json
try:
    import torch
    
    # Check CPU tensor execution
    cpu_ok = False
    try:
        t = torch.tensor([1.0, 2.0], device="cpu")
        cpu_ok = bool(t.shape == (2,))
    except Exception:
        cpu_ok = False
        
    cuda_avail = False
    gpu_name = "N/A"
    gpu_count = 0
    gpu_mem = "N/A"
    try:
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory
            gpu_mem = f"{mem / (1024**3):.2f} GB"
    except Exception:
        pass
        
    version = getattr(torch, "__version__", "")
    cuda_runtime = getattr(torch.version, "cuda", None)
    
    is_cpu_build = "+cpu" in version or cuda_runtime is None
    is_cuda_build = cuda_runtime is not None and not is_cpu_build
    
    health_code = "TORCH_CUDA_BUILD_READY" if is_cuda_build else ("TORCH_CPU_BUILD_DETECTED" if is_cpu_build else "TORCH_IMPORT_FAILED")
    
    print(json.dumps({
        "installed": True,
        "version": version,
        "build": "cuda" if is_cuda_build else "cpu",
        "cudaRuntime": cuda_runtime,
        "isCpuBuild": is_cpu_build,
        "isCudaBuild": is_cuda_build,
        "cpuExecution": cpu_ok,
        "cudaAvailable": cuda_avail,
        "deviceCount": gpu_count,
        "deviceName": gpu_name,
        "gpuMemory": gpu_mem,
        "healthCode": health_code
    }))
except ImportError:
    print(json.dumps({
        "installed": False,
        "version": None,
        "build": "missing",
        "cudaRuntime": None,
        "isCpuBuild": False,
        "isCudaBuild": False,
        "cpuExecution": False,
        "cudaAvailable": False,
        "deviceCount": 0,
        "deviceName": "N/A",
        "gpuMemory": "N/A",
        "healthCode": "TORCH_MISSING"
    }))
"""
        try:
            res = subprocess.run([python_exec, "-c", code], capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                lines = res.stdout.strip().splitlines()
                for line in reversed(lines):
                    try:
                        import json
                        return json.loads(line)
                    except Exception:
                        pass
        except Exception:
            pass

        return {
            "installed": False,
            "version": None,
            "build": "unknown",
            "cudaRuntime": None,
            "isCpuBuild": False,
            "isCudaBuild": False,
            "cpuExecution": False,
            "cudaAvailable": False,
            "deviceCount": 0,
            "deviceName": "N/A",
            "gpuMemory": "N/A",
            "healthCode": "TORCH_IMPORT_FAILED"
        }

    # In-process inspection
    try:
        import torch
        version = getattr(torch, "__version__", "")
        cuda_runtime = getattr(torch.version, "cuda", None)

        cpu_ok = False
        try:
            t = torch.tensor([1.0, 2.0], device="cpu")
            cpu_ok = bool(t.shape == (2,))
        except Exception:
            cpu_ok = False

        cuda_avail = False
        gpu_name = "N/A"
        gpu_count = 0
        gpu_mem = "N/A"
        try:
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory
                gpu_mem = f"{mem / (1024**3):.2f} GB"
        except Exception:
            pass

        is_cpu_build = "+cpu" in version or cuda_runtime is None
        is_cuda_build = cuda_runtime is not None and not is_cpu_build
        health_code = "TORCH_CUDA_BUILD_READY" if is_cuda_build else ("TORCH_CPU_BUILD_DETECTED" if is_cpu_build else "TORCH_IMPORT_FAILED")

        return {
            "installed": True,
            "version": version,
            "build": "cuda" if is_cuda_build else "cpu",
            "cudaRuntime": cuda_runtime,
            "isCpuBuild": is_cpu_build,
            "isCudaBuild": is_cuda_build,
            "cpuExecution": cpu_ok,
            "cudaAvailable": cuda_avail,
            "deviceCount": gpu_count,
            "deviceName": gpu_name,
            "gpuMemory": gpu_mem,
            "healthCode": health_code
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "build": "missing",
            "cudaRuntime": None,
            "isCpuBuild": False,
            "isCudaBuild": False,
            "cpuExecution": False,
            "cudaAvailable": False,
            "deviceCount": 0,
            "deviceName": "N/A",
            "gpuMemory": "N/A",
            "healthCode": "TORCH_MISSING"
        }
