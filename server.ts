import express from "express";
import path from "path";
import fs from "fs";
import os from "os";
import { execFile, spawn, spawnSync, ChildProcess } from "child_process";
import { GoogleGenAI } from "@google/genai";
import { createServer as createViteServer } from "vite";
import { WorkspaceRootResolver, getVenvPython, isEnvironmentReady, setupEnvironmentJob } from "./server/environment";


const app = express();
const PORT = 3000;

app.use(express.json({ limit: "200mb" }));

// Persistent Job Store Infrastructure
interface ServerJob {
  job_id: string;
  type: string;
  title: string;
  action?: string;
  payload?: any;
  status: "Waiting" | "Queued" | "Starting" | "Running" | "Completed" | "Failed" | "Cancelled" | "Skipped";
  progress: number;
  current_stage: string;
  completed_stages: string[];
  started_at: string;
  execution_started_at?: string;
  completed_at?: string;
  updated_at: string;
  estimated_remaining?: string;
  logs: string[];
  result?: any;
  error?: string;
  suggested_fix?: string;
  metrics?: {
    queue_time_ms?: number;
    execution_time_ms?: number;
    total_time_ms?: number;
  };
}

const JOBS_FILE = path.join(process.cwd(), "workspace", "jobs.json");
const runningProcesses = new Map<string, ChildProcess>();
let jobsMap = new Map<string, ServerJob>();

function loadJobsFromDisk() {
  try {
    if (fs.existsSync(JOBS_FILE)) {
      const data = JSON.parse(fs.readFileSync(JOBS_FILE, "utf-8"));
      if (Array.isArray(data)) {
        data.forEach((job: ServerJob) => {
          // If server restarted while job was running, mark as failed
          if (job.status === "Running" || job.status === "Queued" || job.status === "Starting") {
            job.status = "Failed";
            job.error = "Process terminated unexpectedly during server restart.";
            job.suggested_fix = "Click Retry Job to execute again.";
          }
          jobsMap.set(job.job_id, job);
        });
      }
    }
  } catch (e) {
    console.error("Failed to load jobs from disk:", e);
  }
}

function saveJobsToDisk() {
  try {
    const workspaceDir = path.join(process.cwd(), "workspace");
    if (!fs.existsSync(workspaceDir)) {
      fs.mkdirSync(workspaceDir, { recursive: true });
    }
    const jobsArray = Array.from(jobsMap.values());
    fs.writeFileSync(JOBS_FILE, JSON.stringify(jobsArray, null, 2));
  } catch (e) {
    console.error("Failed to save jobs to disk:", e);
  }
}

loadJobsFromDisk();

function createBackgroundJob(type: string, title: string, action: string, payload: any): ServerJob {
  const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  const now = new Date().toISOString();
  
  const job: ServerJob = {
    job_id: jobId,
    type,
    title,
    action,
    payload,
    status: "Starting",
    progress: 5,
    current_stage: "Initializing Job...",
    completed_stages: [],
    started_at: now,
    updated_at: now,
    estimated_remaining: "Calculating...",
    logs: [`[${now.slice(11, 19)}] Starting ${title}...`],
  };

  jobsMap.set(jobId, job);
  saveJobsToDisk();
  
  // Set job running immediately
  job.status = "Running";
  job.execution_started_at = new Date().toISOString();
  saveJobsToDisk();

  if (action === "setup_environment") {
    setupEnvironmentJob(payload, (progress, stage, completed_stages, message) => {
      job.progress = progress;
      job.current_stage = stage;
      if (completed_stages) job.completed_stages = completed_stages;
      if (message) job.logs.push(`[${new Date().toISOString().slice(11, 19)}] ${message}`);
      job.updated_at = new Date().toISOString();
      saveJobsToDisk();
    }, (msg) => {
      job.logs.push(`[${new Date().toISOString().slice(11, 19)}] ${msg}`);
      saveJobsToDisk();
    }).then((result: any) => {
      job.status = "Completed";
      job.progress = 100;
      job.result = result;
      job.completed_at = new Date().toISOString();
      job.logs.push(`[${new Date().toISOString().slice(11, 19)}] Job finished successfully.`);
      job.updated_at = new Date().toISOString();
      saveJobsToDisk();
    }).catch((err: any) => {
      job.status = "Failed";
      job.error = err.message || "Environment setup failed";
      job.completed_at = new Date().toISOString();
      job.logs.push(`[${new Date().toISOString().slice(11, 19)}] ERROR: ${job.error}`);
      job.updated_at = new Date().toISOString();
      saveJobsToDisk();
    });
    
    return job;
  }
  
  // For other actions, check if environment is ready
  if (!isEnvironmentReady()) {
    job.status = "Failed";
    job.error = "Environment not ready. Please run Environment Setup first.";
    job.completed_at = new Date().toISOString();
    job.logs.push(`[${new Date().toISOString().slice(11, 19)}] ERROR: ${job.error}`);
    job.updated_at = new Date().toISOString();
    saveJobsToDisk();
    return job;
  }

  const payloadStr = JSON.stringify(payload || {});
  const workspaceRoot = WorkspaceRootResolver.resolve();
  const venvPython = getVenvPython(workspaceRoot);
  
  const payloadPath = path.join(os.tmpdir(), `autorag-payload-${jobId}.json`);
  fs.writeFileSync(payloadPath, payloadStr, "utf-8");
  
  const child = spawn(venvPython, ["-u", "cli.py", action, "--payload-file", payloadPath], {
    cwd: workspaceRoot,
    env: { ...process.env, PYTHONUNBUFFERED: "1" }
  });

  child.on("exit", () => {
    try {
      if (fs.existsSync(payloadPath)) {
        fs.unlinkSync(payloadPath);
      }
    } catch (e) {}
  });

  runningProcesses.set(jobId, child);

  let buffer = "";
  
  child.stdout.on("data", (chunk: Buffer) => {
    buffer += chunk.toString("utf-8");
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const parsed = JSON.parse(trimmed);
        if (parsed.type === "progress") {
          job.progress = parsed.progress || job.progress;
          job.current_stage = parsed.current_stage || job.current_stage;
          if (parsed.completed_stages) {
            job.completed_stages = parsed.completed_stages;
          }
          if (parsed.message) {
            job.logs.push(`[${new Date().toISOString().slice(11, 19)}] ${parsed.message}`);
          }
          job.updated_at = new Date().toISOString();
          saveJobsToDisk();
        } else if (parsed.success !== undefined) {
          const nowMs = new Date().getTime();
          job.completed_at = new Date().toISOString();
          if (!job.metrics) job.metrics = {};
          if (job.execution_started_at) {
            job.metrics.execution_time_ms = nowMs - new Date(job.execution_started_at).getTime();
          }
          job.metrics.total_time_ms = nowMs - new Date(job.started_at).getTime();

          if (parsed.success) {
            job.status = "Completed";
            job.progress = 100;
            job.result = parsed;
            job.logs.push(`[${new Date().toISOString().slice(11, 19)}] Job finished successfully.`);
          } else {
            job.status = "Failed";
            job.error = parsed.error || "Execution failed";
            job.suggested_fix = parsed.suggested_fix || "Check configuration parameters and try again.";
            job.logs.push(`[${new Date().toISOString().slice(11, 19)}] ERROR: ${job.error}`);
          }
          job.updated_at = new Date().toISOString();
          saveJobsToDisk();
        }
      } catch (e) {
        job.logs.push(`[${new Date().toISOString().slice(11, 19)}] ${trimmed}`);
      }
    }
  });

  child.stderr.on("data", (data: Buffer) => {
    const errText = data.toString("utf-8").trim();
    if (errText) {
      job.logs.push(`[${new Date().toISOString().slice(11, 19)}] STDERR: ${errText}`);
    }
  });

  child.on("close", (code) => {
    runningProcesses.delete(jobId);
    if (job.status === "Running") {
      const nowMs = new Date().getTime();
      job.completed_at = new Date().toISOString();
      if (!job.metrics) job.metrics = {};
      if (job.execution_started_at) {
        job.metrics.execution_time_ms = nowMs - new Date(job.execution_started_at).getTime();
      }
      job.metrics.total_time_ms = nowMs - new Date(job.started_at).getTime();

      if (code === 0) {
        job.status = "Completed";
        job.progress = 100;
      } else {
        job.status = "Failed";
        job.error = `Process exited with code ${code}`;
        job.suggested_fix = "Review connection configuration and retry the operation.";
      }
      job.updated_at = new Date().toISOString();
      saveJobsToDisk();
    }
  });

  return job;
}

// Helper to execute python cli.py bridge synchronously
function runPythonCli(action: string, payload: any = {}): Promise<any> {
  return new Promise((resolve) => {
    const payloadStr = JSON.stringify(payload);
    
    if (!isEnvironmentReady()) {
      return resolve({ success: false, error: "Environment not ready." });
    }
    
    const workspaceRoot = WorkspaceRootResolver.resolve();
    const venvPython = getVenvPython(workspaceRoot);
    
    execFile(
      venvPython,
      ["cli.py", action, "--payload", payloadStr],
      { cwd: workspaceRoot, maxBuffer: 10 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          console.error(`Python CLI error (${action}):`, stderr || error.message);
          return resolve({ success: false, error: stderr || error.message });
        }
        try {
          const lines = stdout.trim().split("\n");
          const lastLine = lines[lines.length - 1];
          const parsed = JSON.parse(lastLine);
          resolve(parsed);
        } catch (e) {
          resolve({ success: false, raw_output: stdout, error: "Failed to parse JSON output" });
        }
      }
    );
  });
}

// API Routes
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", platform: "AutoRAG Optimization Platform" });
});

// Job Management Endpoints
app.post("/api/jobs/start", (req, res) => {
  console.log("Job start request received for:", req.body.action, "Payload keys:", req.body.payload ? Object.keys(req.body.payload) : "none");
  const { action, payload, title, type } = req.body;
  if (!action) {
    return res.status(400).json({ error: "Missing action parameter" });
  }
  const jobTitle = title || `Execute ${action}`;
  const jobType = type || action;
  const job = createBackgroundJob(jobType, jobTitle, action, payload);
  res.json({ job_id: job.job_id, job });
});

app.get("/api/jobs", (req, res) => {
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  const jobsList = Array.from(jobsMap.values()).sort(
    (a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
  );
  res.json({ jobs: jobsList });
});

app.get("/api/jobs/:job_id", (req, res) => {
  const { job_id } = req.params;
  const job = jobsMap.get(job_id);
  if (!job) {
    return res.status(404).json({ error: "Job not found" });
  }
  res.json({ job });
});

app.post("/api/jobs/:job_id/cancel", (req, res) => {
  const { job_id } = req.params;
  const job = jobsMap.get(job_id);
  if (!job) {
    return res.status(404).json({ error: "Job not found" });
  }

  const proc = runningProcesses.get(job_id);
  if (proc) {
    proc.kill("SIGTERM");
    runningProcesses.delete(job_id);
  }

  job.status = "Cancelled";
  job.updated_at = new Date().toISOString();
  job.logs.push(`[${new Date().toISOString().slice(11, 19)}] Job cancelled by user.`);
  saveJobsToDisk();

  res.json({ success: true, job });
});

app.post("/api/jobs/clear", (req, res) => {
  const activeJobs = new Map<string, ServerJob>();
  jobsMap.forEach((job, id) => {
    if (job.status === "Running" || job.status === "Queued" || job.status === "Starting") {
      activeJobs.set(id, job);
    }
  });
  jobsMap = activeJobs;
  saveJobsToDisk();
  res.json({ success: true, count: jobsMap.size });
});

// Instant Cache Metadata Endpoint
app.get("/api/workspace/cache_metadata", (req, res) => {
  try {
    const wsDir = path.join(process.cwd(), "workspace");
    const kbDir = path.join(wsDir, "kb");
    const expDir = path.join(wsDir, "experiments");
    const repDir = path.join(wsDir, "reports");
    const datasetsDir = path.join(wsDir, "datasets");

    const docsCount = fs.existsSync(kbDir) ? fs.readdirSync(kbDir).length : 0;
    const experimentsCount = fs.existsSync(expDir) ? fs.readdirSync(expDir).length : 0;
    const reportsCount = fs.existsSync(repDir) ? fs.readdirSync(repDir).length : 0;
    let datasetItemsCount = 0;
    if (fs.existsSync(datasetsDir)) {
      try {
        const files = fs.readdirSync(datasetsDir).filter(f => f.startsWith("dataset_") && f.endsWith(".json"));
        if (files.length > 0) {
          // Sort by modified time descending to get the latest
          files.sort((a, b) => fs.statSync(path.join(datasetsDir, b)).mtimeMs - fs.statSync(path.join(datasetsDir, a)).mtimeMs);
          for (const file of files) {
            const ds = JSON.parse(fs.readFileSync(path.join(datasetsDir, file), "utf-8"));
            if (ds.items && ds.items.length > 0) {
              datasetItemsCount = ds.items.length;
              break;
            }
          }
        }
      } catch (e) {}
    }

    res.json({
      cache_metadata: {
        documentsCount: docsCount,
        experimentsCount: experimentsCount,
        reportsCount: reportsCount,
        datasetItemsCount: datasetItemsCount,
        lastUpdated: new Date().toISOString()
      }
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Generic CLI RPC endpoint (for synchronous calls)
app.post("/api/autorag/:action", async (req, res) => {
  const { action } = req.params;
  const payload = req.body || {};
  const result = await runPythonCli(action, payload);
  res.json(result);
});

// Download/view generated reports
app.get("/api/reports/file/:filename", (req, res) => {
  const { filename } = req.params;
  const reportPath = path.join(process.cwd(), "workspace", "reports", filename);
  if (fs.existsSync(reportPath)) {
    res.sendFile(reportPath);
  } else {
    res.status(404).json({ error: "Report file not found" });
  }
});

// Server-side LLM API endpoint for Explanations
app.post("/api/llm/generate", async (req, res) => {
  try {
    const { prompt, systemInstruction, connection } = req.body;
    if (!connection || !connection.provider) {
      return res.status(400).json({ error: "Connection configuration is missing" });
    }

    let endpoint = connection.base_url;
    if (endpoint.endsWith("/")) endpoint = endpoint.slice(0, -1);
    if (!endpoint.endsWith("/chat/completions") && connection.provider !== "gemini") {
      endpoint = `${endpoint}/chat/completions`;
    }

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (connection.api_key) {
      headers["Authorization"] = `Bearer ${connection.api_key}`;
    }

    if (connection.provider === "openrouter" || endpoint.includes("openrouter.ai")) {
      headers["HTTP-Referer"] = "https://github.com/google/aistudio-build";
      headers["X-Title"] = "RAG Optimizer Sweep";
    }

    let payload: any = {};
    if (connection.provider === "gemini") {
      const ai = new GoogleGenAI({
        apiKey: connection.api_key || process.env.GEMINI_API_KEY,
        httpOptions: { headers: { "User-Agent": "aistudio-build" } },
      });
      const response = await ai.models.generateContent({
        model: connection.model_name || "gemini-2.5-flash",
        contents: prompt,
        config: systemInstruction ? { systemInstruction } : undefined,
      });
      return res.json({ text: response.text });
    } else {
      const messages = [];
      if (systemInstruction) {
        messages.push({ role: "system", content: systemInstruction });
      }
      messages.push({ role: "user", content: prompt });
      payload = {
        model: connection.model_name,
        messages: messages,
        temperature: connection.temperature || 0.7,
        max_tokens: connection.max_tokens || 1000,
      };
    }

    const response = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Provider Error ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    const text = data.choices?.[0]?.message?.content || "";
    res.json({ text });
  } catch (error: any) {
    console.error("LLM Generate Error:", error);
    res.status(500).json({ error: error.message || "Failed to generate content" });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {

    const workspaceRoot = WorkspaceRootResolver.resolve();
    const venvDir = path.join(workspaceRoot, ".venv");
    const venvPython = getVenvPython(workspaceRoot);
    const venvExists = fs.existsSync(venvPython);
    console.log(`[Environment] Workspace Root: ${workspaceRoot}`);
    console.log(`[Environment] Virtual Environment: ${venvDir}`);
    console.log(`[Environment] Python Executable: ${venvPython}`);
    console.log(`[Environment] Using Venv: ${process.env.VIRTUAL_ENV || venvDir.includes(".venv") ? 'yes' : 'no'}`);
    console.log(`[Environment] Venv Status: ${venvExists ? 'reused' : 'newly created'}`);
    console.log(`AutoRAG Platform server running on http://localhost:${PORT}`);
  });
}

startServer();

