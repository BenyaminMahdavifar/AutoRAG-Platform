import https from "https";
import path from "path";
import fs from "fs";
import os from "os";
import { spawnSync, spawn } from "child_process";

// Canonical PyTorch CUDA Wheel Target Policy
export const PYTORCH_CUDA_VARIANT = "cu124";
export const PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124";
export const PYTORCH_PACKAGE = "torch";

export interface TorchInspectionResult {
  installed: boolean;
  version: string | null;
  build: "cuda" | "cpu" | "missing" | "unknown";
  cudaRuntime: string | null;
  isCpuBuild: boolean;
  isCudaBuild: boolean;
  cpuExecution: boolean;
  cudaAvailable: boolean;
  deviceCount: number;
  deviceName: string;
  gpuMemory: string;
  healthCode: string;
}

export class WorkspaceRootResolver {
  static resolve(): string {
    if (process.env.WORKSPACE_ROOT && fs.existsSync(process.env.WORKSPACE_ROOT)) {
      return process.env.WORKSPACE_ROOT;
    }
    
    let current = process.cwd();
    const scriptDir = process.cwd(); // fallback
    const markers = ["pyproject.toml", "requirements.txt", "package.json", "src", "autorag"];
    
    function findRoot(start: string) {
      let dir = start;
      while (true) {
        if (markers.some(marker => fs.existsSync(path.join(dir, marker)))) {
          return dir;
        }
        const parent = path.dirname(dir);
        if (parent === dir) break;
        dir = parent;
      }
      return null;
    }
    
    const root = findRoot(current) || findRoot(scriptDir);
    if (root) return root;
    
    return current;
  }
}

export function getVenvPython(workspaceRoot: string): string {
  const venvDir = path.join(workspaceRoot, ".venv");
  return os.platform() === "win32" 
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");
}

export function isEnvironmentReady(): boolean {
  const root = WorkspaceRootResolver.resolve();
  const python = getVenvPython(root);
  return fs.existsSync(python);
}

export function getGlobalPython(): string {
  if (process.env.VIRTUAL_ENV) {
    const venvPython = os.platform() === "win32" 
      ? path.join(process.env.VIRTUAL_ENV, "Scripts", "python.exe")
      : path.join(process.env.VIRTUAL_ENV, "bin", "python");
    if (fs.existsSync(venvPython)) {
      return venvPython;
    }
  }
  return os.platform() === "win32" ? "python" : "python3";
}

export function inspectVenvTorch(venvPython: string): TorchInspectionResult {
  if (!fs.existsSync(venvPython)) {
    return {
      installed: false,
      version: null,
      build: "missing",
      cudaRuntime: null,
      isCpuBuild: false,
      isCudaBuild: false,
      cpuExecution: false,
      cudaAvailable: false,
      deviceCount: 0,
      deviceName: "N/A",
      gpuMemory: "N/A",
      healthCode: "TORCH_MISSING"
    };
  }

  const script = `
import sys, json
try:
    import torch
    ver = getattr(torch, "__version__", "")
    cuda_rt = getattr(torch.version, "cuda", None)
    
    cpu_ok = False
    try:
        t = torch.tensor([1.0, 2.0], device="cpu")
        cpu_ok = bool(t.shape == (2,))
    except Exception:
        cpu_ok = False
        
    cuda_avail = False
    dev_name = "N/A"
    dev_count = 0
    gpu_mem = "N/A"
    try:
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            dev_count = torch.cuda.device_count()
            dev_name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory
            gpu_mem = f"{mem / (1024**3):.2f} GB"
    except Exception:
        pass
        
    is_cpu = "+cpu" in ver or cuda_rt is None
    is_cuda = cuda_rt is not None and not is_cpu
    hcode = "TORCH_CUDA_BUILD_READY" if is_cuda else ("TORCH_CPU_BUILD_DETECTED" if is_cpu else "TORCH_IMPORT_FAILED")
    
    print(json.dumps({
        "installed": True,
        "version": ver,
        "build": "cuda" if is_cuda else "cpu",
        "cudaRuntime": cuda_rt,
        "isCpuBuild": is_cpu,
        "isCudaBuild": is_cuda,
        "cpuExecution": cpu_ok,
        "cudaAvailable": cuda_avail,
        "deviceCount": dev_count,
        "deviceName": dev_name,
        "gpuMemory": gpu_mem,
        "healthCode": hcode
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
`;

  try {
    const res = spawnSync(venvPython, ["-c", script], { encoding: "utf8", timeout: 10000 });
    if (res.status === 0 && res.stdout) {
      const lines = res.stdout.trim().split("\n");
      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i].trim();
        if (line.startsWith("{")) {
          try {
            return JSON.parse(line);
          } catch (e) {}
        }
      }
    }
  } catch (e) {}

  return {
    installed: false,
    version: null,
    build: "unknown",
    cudaRuntime: null,
    isCpuBuild: false,
    isCudaBuild: false,
    cpuExecution: false,
    cudaAvailable: false,
    deviceCount: 0,
    deviceName: "N/A",
    gpuMemory: "N/A",
    healthCode: "TORCH_IMPORT_FAILED"
  };
}

export async function setupEnvironmentJob(
  payload: any, 
  emitProgress: (progress: number, stage: string, completed_stages: string[], message: string) => void,
  logMessage: (msg: string) => void
): Promise<any> {
  const workspaceRoot = WorkspaceRootResolver.resolve();
  emitProgress(5, "Resolve Workspace Root", [], `Resolved workspace root: ${workspaceRoot}`);
  
  const venvDir = path.join(workspaceRoot, ".venv");
  const venvPython = getVenvPython(workspaceRoot);

  const venvExists = fs.existsSync(venvPython);
  logMessage(`[Environment] Workspace Root: ${workspaceRoot}`);
  logMessage(`[Environment] Virtual Environment: ${venvDir}`);
  logMessage(`[Environment] Python Executable: ${venvPython}`);
  logMessage(`[Environment] Using Venv: ${process.env.VIRTUAL_ENV || venvDir.includes(".venv") ? 'yes' : 'no'}`);
  logMessage(`[Environment] Venv Status: ${venvExists ? 'reused' : 'newly created'}`);
  logMessage(`[Environment Policy] Target PyTorch Wheel: CUDA-enabled (${PYTORCH_CUDA_VARIANT}) via ${PYTORCH_INDEX_URL}`);
  
  emitProgress(10, "Detect Runtime", ["Resolve Workspace Root"], "Detecting Python runtime...");
  
  if (!fs.existsSync(venvPython)) {
    emitProgress(15, "Create .venv", ["Resolve Workspace Root", "Detect Runtime"], `Creating virtual environment in ${venvDir}...`);
    const globalPython = getGlobalPython();
    const result = spawnSync(globalPython, ["-m", "venv", "--without-pip", venvDir], { encoding: "utf8" });
    if (result.error || result.status !== 0) {
      const err = `Failed to create venv: ${result.stderr || result.error?.message}`;
      logMessage(`ERROR: ${err}`);
      throw new Error(err);
    }
    
    emitProgress(20, "Upgrade pip", ["Resolve Workspace Root", "Detect Runtime", "Create .venv"], "Bootstrapping pip in venv...");
    
    const getPipPath = path.join(venvDir, "get-pip.py");
    try {
      await new Promise<void>((resolve, reject) => {
        const file = fs.createWriteStream(getPipPath);
        https.get("https://bootstrap.pypa.io/get-pip.py", (response: any) => {
          response.pipe(file);
          file.on("finish", () => { file.close(); resolve(); });
        }).on("error", (err: any) => {
          fs.unlink(getPipPath, () => reject(err));
        });
      });
      
      const pipResult = spawnSync(venvPython, [getPipPath], { encoding: "utf8" });
      if (pipResult.error || pipResult.status !== 0) {
        logMessage(`WARNING: Failed to bootstrap pip: ${pipResult.stderr || pipResult.error?.message}`);
      }
      
      fs.unlinkSync(getPipPath);
    } catch (e: any) {
      logMessage(`WARNING: Failed to download get-pip.py: ${e.message}`);
    }
  } else {
    emitProgress(20, "Create .venv", ["Resolve Workspace Root", "Detect Runtime"], `Using existing virtual environment at ${venvDir}`);
  }

  emitProgress(25, "Resolve Dependencies", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip"], "Resolving required packages...");
  
  const emb_model = payload.embedding_model;
  const requiresTorch = emb_model && emb_model !== "local-tfidf-512";

  // Step 1: Inspect PyTorch build & enforce CUDA wheel policy
  if (requiresTorch) {
    emitProgress(30, "Install PyTorch (CUDA)", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip", "Resolve Dependencies"], "Checking PyTorch backend...");
    const torchInfo = inspectVenvTorch(venvPython);
    
    if (!torchInfo.installed || torchInfo.isCpuBuild) {
      const reason = torchInfo.isCpuBuild 
        ? `CPU-only PyTorch build (${torchInfo.version}) detected (Backend Mismatch)` 
        : "PyTorch not installed";
      logMessage(`[PyTorch Policy] ${reason}. Installing canonical CUDA-enabled PyTorch (${PYTORCH_CUDA_VARIANT}) from ${PYTORCH_INDEX_URL}...`);
      
      emitProgress(35, "Install PyTorch (CUDA)", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip", "Resolve Dependencies"], `Installing CUDA-enabled PyTorch (${PYTORCH_CUDA_VARIANT})...`);
      
      const installArgs = torchInfo.isCpuBuild
        ? ["-m", "pip", "install", "--upgrade", "--force-reinstall", "torch", "--index-url", PYTORCH_INDEX_URL]
        : ["-m", "pip", "install", "torch", "--index-url", PYTORCH_INDEX_URL];
        
      const pResult = spawnSync(venvPython, installArgs, { encoding: "utf8" });
      if (pResult.status !== 0) {
        const err = `Failed to install CUDA PyTorch: ${pResult.stderr || pResult.stdout}`;
        logMessage(`ERROR: ${err}`);
        throw new Error(err);
      }
      
      // Verify PyTorch installation after install
      const verifiedTorch = inspectVenvTorch(venvPython);
      if (!verifiedTorch.installed) {
        throw new Error("PyTorch installation verification failed: module could not be imported.");
      }
      logMessage(`[PyTorch Policy] Verified PyTorch: ${verifiedTorch.version} (CUDA runtime: ${verifiedTorch.cudaRuntime || 'none'}, CPU execution: ${verifiedTorch.cpuExecution ? 'passed' : 'failed'})`);
    } else {
      logMessage(`[PyTorch Policy] Existing PyTorch is already CUDA-enabled: ${torchInfo.version} (CUDA ${torchInfo.cudaRuntime}). CPU Execution: ${torchInfo.cpuExecution ? 'passed' : 'failed'}`);
      emitProgress(45, "Install PyTorch (CUDA)", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip", "Resolve Dependencies"], `PyTorch CUDA build (${torchInfo.version}) is ready.`);
    }
  }

  // Step 2: Install remaining helper packages
  const packages = new Set<string>();
  packages.add("requests");
  packages.add("pypdf");
  if (requiresTorch) {
    packages.add("transformers");
    packages.add("sentence-transformers");
    packages.add("huggingface_hub");
    packages.add("tokenizers");
  }

  const packageList = Array.from(packages);
  
  emitProgress(50, "Install Packages", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip", "Resolve Dependencies"], "Checking auxiliary packages...");
  
  const listResult = spawnSync(venvPython, ["-m", "pip", "list", "--format=json"], { encoding: "utf8" });
  let installed = new Set<string>();
  if (listResult.status === 0) {
    try {
      const installedArr = JSON.parse(listResult.stdout);
      installedArr.forEach((pkg: any) => installed.add(pkg.name.toLowerCase().replace(/_/g, "-")));
    } catch (e) {}
  }

  const missing = packageList.filter(pkg => !installed.has(pkg.toLowerCase().replace(/_/g, "-")));
  
  if (missing.length > 0) {
    for (let i = 0; i < missing.length; i++) {
      const pkg = missing[i];
      const p = 50 + Math.floor((i / missing.length) * 25);
      emitProgress(p, "Install Packages", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip", "Resolve Dependencies"], `Installing package ${i+1}/${missing.length}: ${pkg}`);
      
      // Use --extra-index-url to protect CUDA torch wheels from downgrade
      const installArgs = requiresTorch
        ? ["-m", "pip", "install", pkg, "--extra-index-url", PYTORCH_INDEX_URL]
        : ["-m", "pip", "install", pkg];
        
      const pResult = spawnSync(venvPython, installArgs, { encoding: "utf8" });
      if (pResult.status !== 0) {
        const err = `Failed to install ${pkg}: ${pResult.stderr || pResult.stdout}`;
        logMessage(`ERROR: ${err}`);
        throw new Error(err);
      }
    }
  } else {
    emitProgress(75, "Install Packages", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip", "Resolve Dependencies"], "All auxiliary dependencies are installed.");
  }
  
  emitProgress(80, "Validate Imports", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip", "Resolve Dependencies", "Install Packages"], "Validating imports, tensor execution, and runtime readiness...");
  
  return new Promise((resolve, reject) => {
    const payloadStr = JSON.stringify(payload);
    const child = spawn(venvPython, ["cli.py", "validate_environment", "--payload", payloadStr], {
      cwd: workspaceRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1" }
    });
    
    let stdout = "";
    let stderr = "";
    
    child.stdout.on("data", data => {
      stdout += data.toString("utf8");
    });
    
    child.stderr.on("data", data => {
      stderr += data.toString("utf8");
    });
    
    child.on("close", code => {
      if (code === 0) {
        try {
          const lines = stdout.trim().split(/\r?\n/);
          let parsed = null;
          for (let i = lines.length - 1; i >= 0; i--) {
            const line = lines[i].trim();
            if (line.startsWith('{')) {
              try {
                parsed = JSON.parse(line);
                break;
              } catch (e) {
                // Ignore parse errors, try next line
              }
            }
          }
          if (!parsed) {
            throw new Error("Could not find valid JSON in script output.");
          }
          
          emitProgress(100, "Ready", ["Resolve Workspace Root", "Detect Runtime", "Create .venv", "Upgrade pip", "Resolve Dependencies", "Install Packages", "Validate Imports"], "Environment preparation complete.");
          resolve({
            success: true,
            validation: parsed.validation || parsed,
            python_exec: venvPython
          });
        } catch(e) {
          logMessage(`ERROR: Failed to parse validation result: ${stdout}`);
          reject(new Error("Failed to parse validation result"));
        }
      } else {
        logMessage(`ERROR: Validation script failed: ${stderr || stdout}`);
        reject(new Error("Validation script failed"));
      }
    });
  });
}
