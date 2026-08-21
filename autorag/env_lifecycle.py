import os
import sys
import json
import subprocess
import venv
from pathlib import Path
from typing import List, Dict, Any, Optional
from autorag.runtime_discovery import RuntimeDiscovery
from autorag.torch_policy import (
    PYTORCH_CUDA_VARIANT, PYTORCH_INDEX_URL, PYTORCH_PACKAGE,
    inspect_installed_torch, resolve_embedding_device, CUDAUnavailableError, get_nvidia_driver_info
)

class EnvironmentManager:
    """Manages the Python virtual environment for the workspace."""
    def __init__(self, workspace_dir: Path):
        self.venv_dir = workspace_dir / ".venv"
        
        if os.name == 'nt':
            self.venv_python = self.venv_dir / "Scripts" / "python.exe"
            self.venv_pip = self.venv_dir / "Scripts" / "pip.exe"
        else:
            self.venv_python = self.venv_dir / "bin" / "python"
            self.venv_pip = self.venv_dir / "bin" / "pip"

    def ensure_environment(self, emit_progress_fn) -> str:
        if not self.venv_python.exists():
            emit_progress_fn(10, "Create Virtual Environment", ["Runtime Discovery"], f"Creating .venv at {self.venv_dir}...")
            global_env = RuntimeDiscovery.discover()
            
            # Use venv module to create it (with_pip=False to avoid debian ensurepip issues)
            builder = venv.EnvBuilder(with_pip=False)
            builder.create(self.venv_dir)
            
            emit_progress_fn(15, "Upgrade pip", ["Runtime Discovery", "Create Virtual Environment"], "Bootstrapping pip...")
            import urllib.request
            get_pip_path = self.venv_dir / "get-pip.py"
            try:
                urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', get_pip_path)
                subprocess.run([str(self.venv_python), str(get_pip_path)], capture_output=True, check=True)
            except Exception as e:
                emit_progress_fn(15, "Upgrade pip", ["Runtime Discovery", "Create Virtual Environment"], f"Failed to bootstrap pip: {e}")
            finally:
                if get_pip_path.exists():
                    get_pip_path.unlink()
            
        return str(self.venv_python)

class DependencyResolver:
    """Resolves required packages based on feature configuration."""
    @staticmethod
    def resolve(payload: Dict[str, Any]) -> List[str]:
        packages = set()
        packages.add("requests")
        packages.add("pypdf")
        
        emb_model = payload.get("embedding_model")
        if emb_model and emb_model != "local-tfidf-512":
            packages.update(["transformers", "sentence-transformers", "huggingface_hub", "tokenizers"])
            
        return list(packages)

class DependencyInstaller:
    """Installs missing dependencies smartly, enforcing CUDA-enabled PyTorch canonical policy."""
    def __init__(self, python_exec: str):
        self.python_exec = python_exec

    def get_installed_packages(self) -> set:
        try:
            res = subprocess.run([self.python_exec, "-m", "pip", "list", "--format=json"], capture_output=True, text=True, check=True)
            installed = json.loads(res.stdout)
            return {pkg["name"].lower().replace("_", "-") for pkg in installed}
        except Exception:
            return set()

    def install_torch_if_needed(self, emit_progress_fn, base_progress: int = 25):
        """Enforces canonical CUDA-enabled PyTorch installation with repair if CPU-only build is found."""
        torch_info = inspect_installed_torch(self.python_exec)
        
        if not torch_info["installed"] or torch_info["isCpuBuild"]:
            reason = "CPU-only build detected (Backend Mismatch)" if torch_info["isCpuBuild"] else "PyTorch not installed"
            emit_progress_fn(
                base_progress,
                "Install PyTorch (CUDA)",
                ["Runtime Discovery", "Create Virtual Environment", "Upgrade pip"],
                f"{reason}. Installing canonical CUDA PyTorch ({PYTORCH_CUDA_VARIANT})..."
            )
            
            cmd = [
                self.python_exec, "-m", "pip", "install",
                "--upgrade", "--force-reinstall", "torch",
                "--index-url", PYTORCH_INDEX_URL
            ] if torch_info["isCpuBuild"] else [
                self.python_exec, "-m", "pip", "install",
                "torch", "--index-url", PYTORCH_INDEX_URL
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Post-check verification
            verified = inspect_installed_torch(self.python_exec)
            if not verified["installed"]:
                raise RuntimeError("Failed to verify PyTorch installation.")
        else:
            emit_progress_fn(
                base_progress + 5,
                "Install PyTorch (CUDA)",
                ["Runtime Discovery", "Create Virtual Environment", "Upgrade pip"],
                f"PyTorch CUDA build ({torch_info['version']}) is ready."
            )

    def install(self, packages: List[str], emit_progress_fn, base_progress: int = 40):
        installed_packages = self.get_installed_packages()
        missing = [pkg for pkg in packages if pkg.lower().replace("_", "-") not in installed_packages]
        
        if not missing:
            emit_progress_fn(base_progress + 25, "Install Packages", ["Runtime Discovery", "Create Virtual Environment", "Upgrade pip", "Resolve Dependencies"], "All auxiliary dependencies are installed.")
            return

        total_missing = len(missing)
        for i, pkg in enumerate(missing):
            progress = base_progress + int((i / total_missing) * 25)
            msg = f"Installing package {i+1}/{total_missing}: {pkg}"
            emit_progress_fn(progress, "Install Packages", ["Runtime Discovery", "Create Virtual Environment", "Upgrade pip", "Resolve Dependencies"], msg)
            
            try:
                # Use --extra-index-url to protect CUDA torch wheels
                cmd = [self.python_exec, "-m", "pip", "install", pkg, "--extra-index-url", PYTORCH_INDEX_URL]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                # Try standard pip install as fallback
                try:
                    subprocess.run([self.python_exec, "-m", "pip", "install", pkg], capture_output=True, text=True, check=True)
                except Exception:
                    pass

class EnvironmentValidator:
    """Verifies the environment after installation with structured diagnostics."""
    def __init__(self, python_exec: str):
        self.python_exec = python_exec

    def validate(self, payload: Dict[str, Any], emit_progress_fn) -> Dict[str, Any]:
        emit_progress_fn(75, "Validate Environment", ["Runtime Discovery", "Create Virtual Environment", "Upgrade pip", "Resolve Dependencies", "Install Packages"], "Running validation checks and tensor diagnostics...")
        
        torch_info = inspect_installed_torch(self.python_exec)
        driver_info = get_nvidia_driver_info()
        
        configured_dev = payload.get("embedding_device", "auto")
        emb_model = payload.get("embedding_model")
        
        resolved_dev = "cpu"
        try:
            resolved_dev = resolve_embedding_device(configured_dev)
        except Exception:
            resolved_dev = "unresolvable"

        result = {
            "python_runtime": sys.version.split(' ')[0],
            "torch": torch_info,
            "cuda_available": torch_info.get("cudaAvailable", False),
            "gpu_name": torch_info.get("deviceName", "N/A"),
            "memory_available": torch_info.get("gpuMemory", "N/A"),
            "driver_info": driver_info,
            "configured_device": configured_dev,
            "resolved_device": resolved_dev,
            "hf_authenticated": False,
            "model_available": False,
            "llm_connection": False,
            "health_code": torch_info.get("healthCode", "UNKNOWN"),
            "details": []
        }

        # Detailed logging
        if torch_info.get("installed"):
            build_type = "CUDA-enabled" if torch_info.get("isCudaBuild") else ("CPU-only (Backend Mismatch)" if torch_info.get("isCpuBuild") else "Unknown")
            result["details"].append(f"PyTorch Version: {torch_info.get('version')} ({build_type}, CUDA Runtime: {torch_info.get('cudaRuntime') or 'N/A'})")
            result["details"].append(f"CPU Tensor Execution: {'PASS' if torch_info.get('cpuExecution') else 'FAIL'}")
        else:
            result["details"].append("PyTorch is not installed in the active environment.")

        if torch_info.get("cudaAvailable"):
            result["details"].append(f"CUDA Hardware: {torch_info.get('deviceName')} (Memory: {torch_info.get('gpuMemory')})")
        else:
            result["details"].append("CUDA Hardware: Not detected or unavailable. CPU execution ready.")

        hf_token = payload.get("hf_token")
        if hf_token:
            try:
                from huggingface_hub import login, HfApi
                login(token=hf_token, add_to_git_credential=False)
                api = HfApi()
                user = api.whoami()
                result["hf_authenticated"] = True
                result["details"].append(f"HF Authenticated as {user.get('name')}")
            except Exception as e:
                result["details"].append(f"HF Auth failed: {str(e)}")
        else:
            result["details"].append("No HF token provided.")

        if emb_model and emb_model != "local-tfidf-512":
            try:
                from sentence_transformers import SentenceTransformer
                # Test loading with canonically resolved device
                target_dev = resolve_embedding_device(configured_dev)
                model = SentenceTransformer(emb_model, device=target_dev)
                result["model_available"] = True
                result["details"].append(f"Embedding model '{emb_model}' loaded successfully on device '{target_dev}'.")
            except CUDAUnavailableError as e:
                result["model_available"] = False
                result["details"].append(f"CUDA Placement Error: {str(e)}")
                result["error"] = str(e)
            except Exception as e:
                result["model_available"] = False
                result["details"].append(f"Embedding model load failed: {str(e)}")

        return result
