import os
import sys
import platform
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RuntimeEnvironment:
    python_executable: str
    pip_executable: str
    is_virtual_env: bool
    os_name: str
    architecture: str
    package_manager: str
    diagnostics: Dict[str, str]

class RuntimeDiscovery:
    """Discovers cross-platform python runtime execution details."""
    
    @classmethod
    def discover(cls) -> RuntimeEnvironment:
        diagnostics = {}
        
        # 1. OS & Architecture
        os_name = platform.system()
        architecture = platform.machine()
        diagnostics["os"] = os_name
        diagnostics["architecture"] = architecture
        
        # 2. Virtual Environment Detection
        is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        diagnostics["is_virtual_env"] = str(is_venv)
        
        # 3. Python Executable Discovery
        python_exec = cls._find_python_executable(diagnostics)
        
        # 4. Pip Executable Discovery
        pip_exec = cls._find_pip_executable(python_exec, diagnostics)
        
        # 5. Package Manager Detection
        package_manager = "pip"
        if shutil.which("conda") is not None:
            package_manager = "conda"
        elif shutil.which("poetry") is not None:
            package_manager = "poetry"
        elif shutil.which("uv") is not None:
            package_manager = "uv"
            
        diagnostics["package_manager"] = package_manager
        
        return RuntimeEnvironment(
            python_executable=python_exec,
            pip_executable=pip_exec,
            is_virtual_env=is_venv,
            os_name=os_name,
            architecture=architecture,
            package_manager=package_manager,
            diagnostics=diagnostics
        )
        
    @classmethod
    def _find_python_executable(cls, diagnostics: Dict[str, str]) -> str:
        candidates = [
            ("sys.executable", sys.executable),
            ("python", shutil.which("python")),
            ("python3", shutil.which("python3")),
            ("py", shutil.which("py"))
        ]
        
        for name, path in candidates:
            if path and os.path.exists(path):
                diagnostics[f"python_candidate_{name}"] = f"Found at {path}"
                return path
            else:
                diagnostics[f"python_candidate_{name}"] = "Not found or invalid"
                
        # Fallback
        diagnostics["python_candidate_fallback"] = "Used 'python' as absolute fallback"
        return "python"

    @classmethod
    def _find_pip_executable(cls, python_exec: str, diagnostics: Dict[str, str]) -> str:
        candidates = [
            ("pip", shutil.which("pip")),
            ("pip3", shutil.which("pip3")),
        ]
        
        # Check if pip is next to python
        py_dir = os.path.dirname(python_exec)
        pip_in_py_dir = os.path.join(py_dir, "pip")
        if os.name == 'nt':
            pip_in_py_dir += ".exe"
            
        if os.path.exists(pip_in_py_dir):
            candidates.insert(0, ("pip_in_py_dir", pip_in_py_dir))
            
        for name, path in candidates:
            if path and os.path.exists(path):
                diagnostics[f"pip_candidate_{name}"] = f"Found at {path}"
                return path
            else:
                diagnostics[f"pip_candidate_{name}"] = "Not found or invalid"
                
        diagnostics["pip_candidate_fallback"] = "Used 'pip' as absolute fallback"
        return "pip"
