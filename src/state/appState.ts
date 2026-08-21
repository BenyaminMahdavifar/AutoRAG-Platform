import { useState, useEffect } from "react";
import {
  Job,
  JobType,
  ConnectionProfile,
  WorkspaceState,
  PlaygroundSession,
  OptimizerSession,
  KbSession,
  DatasetSession,
  ReportsSession,
  CacheMetadata,
  Manifest,
  DocumentMeta,
  Dataset,
  LeaderboardEntry,
  ConnectionConfig
} from "../types";

const LOCAL_STORAGE_KEY = "autorag_app_state_v2";

const DEFAULT_PROFILES: ConnectionProfile[] = [
  {
    profile_id: "prof_openai_default",
    name: "OpenAI Official (gpt-4o-mini)",
    provider: "openai",
    base_url: "https://api.openai.com/v1",
    default_model: "gpt-4o-mini",
    api_key_reference: "key_ref_openai_default",
    is_default: true,
  },
  {
    profile_id: "prof_gemini_default",
    name: "Google Gemini (gemini-3.6-flash)",
    provider: "gemini",
    base_url: "https://generativelanguage.googleapis.com/v1beta",
    default_model: "gemini-3.6-flash",
    api_key_reference: "key_ref_gemini_default",
  },
  {
    profile_id: "prof_ollama_local",
    name: "Ollama Local (llama3)",
    provider: "ollama",
    base_url: "http://localhost:11434/v1",
    default_model: "llama3",
    api_key_reference: "key_ref_none",
  },
  {
    profile_id: "prof_lmstudio_local",
    name: "LM Studio Local (local-model)",
    provider: "lmstudio",
    base_url: "http://localhost:1234/v1",
    default_model: "local-model",
    api_key_reference: "key_ref_none",
  },
  {
    profile_id: "prof_openrouter_default",
    name: "OpenRouter (claude-3.5-sonnet)",
    provider: "openrouter",
    base_url: "https://openrouter.ai/api/v1",
    default_model: "anthropic/claude-3.5-sonnet",
    api_key_reference: "key_ref_openrouter_default",
  },
];

export interface Sessions {
  playground: PlaygroundSession;
  optimizer: OptimizerSession;
  kb: KbSession;
  dataset: DatasetSession;
  reports: ReportsSession;
}

const DEFAULT_SESSIONS: Sessions = {
  playground: {
    chunkStrategy: "recursive",
    chunkSize: 512,
    chunkOverlap: 64,
    retrieverStrategy: "hybrid",
    distanceMetric: "cosine",
    topK: 4,
    systemPrompt: "You are a helpful assistant. Use ONLY the provided context to answer the question.",
    lastTrialResult: null,
  },
  optimizer: {
    strategy: "llm_guided",
    maxTrials: 4,
    usePreviousOptimizationHistory: true,
    lastSweepSummary: null,
  },
  kb: {
    selectedDocId: null,
    uploadName: "",
    uploadContent: "",
    chunkStrategy: "recursive",
    chunkSize: 256,
  },
  dataset: {
    filterText: "",
  },
  reports: {
    filterText: "",
  },
};

const DEFAULT_WORKSPACE_STATE: WorkspaceState = {
  activeTab: "dashboard",
  selectedDatasetId: null,
  selectedPipelineId: null,
  selectedExperimentId: null,
  activeConnectionProfileId: "prof_openai_default",
  lastOpenedPage: "dashboard",
};

const DEFAULT_CACHE_METADATA: CacheMetadata = {
  documentsCount: 0,
  experimentsCount: 0,
  reportsCount: 0,
  datasetItemsCount: 0,
  lastUpdated: new Date().toISOString(),
};

class AppStateStore {
  private listeners: Set<() => void> = new Set();
  private pollInterval: any = null;
  public version = 0;

  // Application State Layer Store Properties
  public workspaceState: WorkspaceState = { ...DEFAULT_WORKSPACE_STATE };
  public connectionProfiles: ConnectionProfile[] = [...DEFAULT_PROFILES];
  public apiKeySecrets: Record<string, string> = {};
  public sessions: Sessions = JSON.parse(JSON.stringify(DEFAULT_SESSIONS));
  public jobs: Job[] = [];
  public cacheMetadata: CacheMetadata = { ...DEFAULT_CACHE_METADATA };

  // Cached Domain Entities
  public manifest: Manifest | null = null;
  public documents: DocumentMeta[] = [];
  public dataset: Dataset | null = null;
  public leaderboard: LeaderboardEntry[] = [];
  public reports: any[] = [];

  constructor() {
    this.loadFromLocalStorage();
    this.fetchInitialBackendData();
    this.startPollingIfJobActive();
  }

  public subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  public notify() {
    this.version += 1;
    this.saveToLocalStorage();
    this.listeners.forEach((listener) => listener());
  }

  // Active Connection Config Computation
  public getActiveConnectionConfig(): ConnectionConfig {
    const activeProfile =
      this.connectionProfiles.find(
        (p) => p.profile_id === this.workspaceState.activeConnectionProfileId
      ) || this.connectionProfiles[0];

    const secret = this.apiKeySecrets[activeProfile.api_key_reference] || "";
    const hfToken = activeProfile.hf_token_reference ? (this.apiKeySecrets[activeProfile.hf_token_reference] || "") : "";

    return {
      provider: activeProfile.provider as any,
      base_url: activeProfile.base_url,
      api_key: secret,
      model_name: activeProfile.default_model,
      hf_token: hfToken,
      embedding_model: activeProfile.embedding_model,
      embedding_device: activeProfile.embedding_device,
      timeout_sec: activeProfile.timeout_sec || 30,
      temperature: activeProfile.temperature ?? 0.2,
      top_p: activeProfile.top_p ?? 0.95,
      max_tokens: activeProfile.max_tokens ?? 1024,
    };
  }

  // Workspace Navigation
  public setActiveTab(tab: string) {
    this.workspaceState.activeTab = tab;
    this.workspaceState.lastOpenedPage = tab;
    this.notify();
  }

  // Session Update Helper
  public updateSession<K extends keyof Sessions>(key: K, partial: Partial<Sessions[K]>) {
    this.sessions[key] = { ...this.sessions[key], ...partial };
    this.notify();
  }

  // Connection Profiles Management
  public setActiveConnectionProfile(profileId: string) {
    this.workspaceState.activeConnectionProfileId = profileId;
    this.notify();
  }

  public addConnectionProfile(profile: ConnectionProfile, secretKey: string = "", hfToken: string = "") {
    this.connectionProfiles.push(profile);
    if (secretKey) {
      this.apiKeySecrets[profile.api_key_reference] = secretKey;
    }
    if (hfToken && profile.hf_token_reference) {
      this.apiKeySecrets[profile.hf_token_reference] = hfToken;
    }
    this.workspaceState.activeConnectionProfileId = profile.profile_id;
    this.notify();
  }

  public updateConnectionProfile(profile: ConnectionProfile, secretKey?: string, hfToken?: string) {
    const idx = this.connectionProfiles.findIndex((p) => p.profile_id === profile.profile_id);
    if (idx !== -1) {
      this.connectionProfiles[idx] = profile;
    }
    if (secretKey !== undefined) {
      this.apiKeySecrets[profile.api_key_reference] = secretKey;
    }
    if (hfToken !== undefined && profile.hf_token_reference) {
      this.apiKeySecrets[profile.hf_token_reference] = hfToken;
    }
    this.notify();
  }

  public deleteConnectionProfile(profileId: string) {
    this.connectionProfiles = this.connectionProfiles.filter((p) => p.profile_id !== profileId);
    if (this.workspaceState.activeConnectionProfileId === profileId) {
      this.workspaceState.activeConnectionProfileId =
        this.connectionProfiles[0]?.profile_id || "prof_openai_default";
    }
    this.notify();
  }

  public setApiKeySecret(keyRef: string, secret: string) {
    this.apiKeySecrets[keyRef] = secret;
    this.notify();
  }

  // Job Operations
  public async startJob(type: JobType, title: string, action: string, payloadOverride: any = {}) {
    const connConfig = this.getActiveConnectionConfig();
    const payload = {
      ...connConfig,
      ...payloadOverride,
    };

    if (action === "run_optimizer") {
      this.leaderboard = [];
      this.sessions.optimizer.lastSweepSummary = null;
      this.notify();
    }

    try {
      const res = await fetch("/api/jobs/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, payload, title, type }),
      });
      const data = await res.json();
      if (data.job) {
        this.upsertJob(data.job);
        this.startPollingIfJobActive();
      }
    } catch (e) {
      console.error("Failed to start job", e);
    }
  }

  public async cancelJob(jobId: string) {
    try {
      const res = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
      const data = await res.json();
      if (data.job) {
        this.upsertJob(data.job);
      }
    } catch (e) {
      console.error("Failed to cancel job", e);
    }
  }

  public async retryJob(jobId: string) {
    const targetJob = this.jobs.find((j) => j.job_id === jobId);
    if (targetJob) {
      this.startJob(targetJob.type, `Retry: ${targetJob.title.replace(/^Retry:\s*/, '')}`, targetJob.action || targetJob.type, targetJob.payload || {});
    }
  }

  public async clearJobHistory() {
    try {
      await fetch("/api/jobs/clear", { method: "POST" });
      this.jobs = this.jobs.filter((j) => j.status === "Running" || j.status === "Queued" || j.status === "Starting");
      this.notify();
    } catch (e) {
      console.error("Failed to clear jobs", e);
    }
  }

  private upsertJob(job: Job) {
    const idx = this.jobs.findIndex((j) => j.job_id === job.job_id);
    if (idx !== -1) {
      this.jobs[idx] = job;
    } else {
      this.jobs.unshift(job);
    }
    this.notify();
  }

  private startPollingIfJobActive() {
    const hasActive = this.jobs.some((j) => j.status === "Running" || j.status === "Queued" || j.status === "Starting");
    if (hasActive && !this.pollInterval) {
      this.pollInterval = setInterval(() => this.pollJobs(), 1200);
    }
  }

  private async pollJobs() {
    try {
      const res = await fetch(`/api/jobs?t=${Date.now()}`);
      if (!res.ok) return;
      const contentType = res.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) return;
      
      const data = await res.json();
      if (data.jobs && Array.isArray(data.jobs)) {
        let changed = false;
        let finishedJobTypes = new Set<string>();

        data.jobs.forEach((serverJob: Job) => {
          const localJob = this.jobs.find((j) => j.job_id === serverJob.job_id);
          if (!localJob || localJob.updated_at !== serverJob.updated_at || localJob.status !== serverJob.status) {
            changed = true;
            if ((!localJob || localJob.status !== "Completed") && serverJob.status === "Completed") {
              finishedJobTypes.add(serverJob.type);
              if (serverJob.action) finishedJobTypes.add(serverJob.action);
            }
          }
        });

        this.jobs = data.jobs;

        if (finishedJobTypes.size > 0) {
          this.handleJobCompletionEffects(finishedJobTypes);
        }

        const stillHasActive = this.jobs.some((j) => j.status === "Running" || j.status === "Queued" || j.status === "Starting");
        if (!stillHasActive && this.pollInterval) {
          clearInterval(this.pollInterval);
          this.pollInterval = null;
        }

        if (changed) {
          this.notify();
        }
      }
    } catch (e) {
      console.error("Error polling jobs", e);
    }
  }

  private handleJobCompletionEffects(types: Set<string>) {
    if (types.has("scan_kb") || types.has("upload_doc") || types.has("import_kb") || types.has("clear_kb")) {
      this.fetchKnowledgeBase();
    }
    if (types.has("build_dataset")) {
      this.fetchDataset();
    }
    if (types.has("run_experiment") || types.has("run_optimizer") || types.has("run_generalization_test")) {
      this.fetchLeaderboard();
    }
    if (types.has("export_reports")) {
      this.fetchReports();
    }
    this.refreshCacheMetadata();
  }

  // Initial Data & Entity Refresh Handlers
  public async syncAll() {
    await this.fetchInitialBackendData();
  }

  public async fetchInitialBackendData() {
    await Promise.all([
      this.fetchJobs(),
      this.fetchKnowledgeBase(),
      this.fetchLeaderboard(),
      this.refreshCacheMetadata(),
    ]);
  }

  public async fetchJobs() {
    try {
      const res = await fetch("/api/jobs");
      if (!res.ok) return;
      const contentType = res.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) return;
      const data = await res.json();
      if (data.jobs) {
        this.jobs = data.jobs;
        this.startPollingIfJobActive();
        this.notify();
      }
    } catch (e) {
      console.error("Failed to fetch jobs", e);
    }
  }

  public async fetchKnowledgeBase() {
    try {
      const conn = this.getActiveConnectionConfig();
      const res = await fetch("/api/autorag/scan_kb", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(conn),
      });
      const data = await res.json();
      if (data.manifest) this.manifest = data.manifest;
      if (data.documents) {
        this.documents = data.documents;
        if (!this.sessions.kb.selectedDocId && data.documents.length > 0) {
          this.sessions.kb.selectedDocId = data.documents[0].doc_id;
        }
      }
      this.notify();
    } catch (e) {
      console.error("Failed to fetch KB", e);
    }
  }

  public async fetchLeaderboard() {
    try {
      const conn = this.getActiveConnectionConfig();
      const res = await fetch("/api/autorag/list_experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(conn),
      });
      const data = await res.json();
      if (data.experiments && Array.isArray(data.experiments)) {
        this.leaderboard = data.experiments.map((exp: any, idx: number) => {
          const resObj = exp.results || exp.result || {};
          const score = typeof exp.composite_score === "number" && Number.isFinite(exp.composite_score)
            ? exp.composite_score
            : (typeof resObj.composite_score === "number" && Number.isFinite(resObj.composite_score)
              ? resObj.composite_score
              : null);
          const ansCorrectness = resObj.answer_metrics?.answer_correctness 
            ?? resObj.answer_metrics?.accuracy 
            ?? resObj.answer_metrics?.faithfulness 
            ?? exp.answer_correctness 
            ?? exp.answer_faithfulness 
            ?? null;
          return {
            trial: exp.trial_number ?? exp.trial ?? (idx + 1),
            experiment_id: exp.experiment_id || "",
            composite_score: score,
            retrieval_hit_rate: resObj.retrieval_metrics?.hit_rate ?? exp.retrieval_hit_rate ?? null,
            answer_correctness: ansCorrectness,
            answer_faithfulness: ansCorrectness,
            config: exp.config || {},
            result: {
              ...resObj,
              composite_score: score,
              metrics_valid: resObj.metrics_valid ?? (score !== null),
            },
            timestamp: exp.timestamp || "",
            status: exp.status || resObj.status || (score !== null ? "completed" : "failed"),
            generalization_test: exp.generalization_test || null,
          };
        });
        this.notify();
      }
    } catch (e) {
      console.error("Failed to fetch leaderboard", e);
    }
  }

  public async runGeneralizationTest(experimentId: string, testSize: number = 5) {
    return this.startJob(
      "run_generalization_test",
      `Generalization Test (${testSize} items): ${experimentId.slice(0, 14)}`,
      "run_generalization_test",
      {
        experiment_id: experimentId,
        test_size: testSize,
      }
    );
  }

  public async clearLeaderboard() {
    try {
      const conn = this.getActiveConnectionConfig();
      const res = await fetch("/api/autorag/clear_experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(conn),
      });
      if (res.ok) {
        this.leaderboard = [];
        this.sessions.optimizer.lastSweepSummary = null;
        this.notify();
      }
    } catch (e) {
      console.error("Failed to clear leaderboard", e);
    }
  }

  public async fetchDataset() {
    try {
      const conn = this.getActiveConnectionConfig();
      const res = await fetch("/api/autorag/get_dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(conn),
      });
      const data = await res.json();
      if (data.dataset) {
        this.dataset = data.dataset;
        this.notify();
      }
    } catch (e) {
      console.error("Failed to fetch dataset", e);
    }
  }

  public async fetchReports() {
    try {
      const conn = this.getActiveConnectionConfig();
      const res = await fetch("/api/autorag/export_reports", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(conn),
      });
      const data = await res.json();
      if (data.reports) {
        this.reports = data.reports;
        this.notify();
      }
    } catch (e) {
      console.error("Failed to fetch reports", e);
    }
  }

  public async refreshCacheMetadata() {
    try {
      const res = await fetch("/api/workspace/cache_metadata");
      const data = await res.json();
      if (data.cache_metadata) {
        this.cacheMetadata = data.cache_metadata;
        this.notify();
      }
    } catch (e) {
      console.error("Failed to fetch cache metadata", e);
    }
  }

  // Import / Export Operations
  public exportConnectionProfiles(): string {
    const exportData = {
      version: "1.0",
      exported_at: new Date().toISOString(),
      profiles: this.connectionProfiles,
      secrets: this.apiKeySecrets,
    };
    return JSON.stringify(exportData, null, 2);
  }

  public importConnectionProfiles(jsonStr: string): boolean {
    try {
      const parsed = JSON.parse(jsonStr);
      if (Array.isArray(parsed.profiles)) {
        this.connectionProfiles = parsed.profiles;
        if (parsed.secrets && typeof parsed.secrets === "object") {
          this.apiKeySecrets = { ...this.apiKeySecrets, ...parsed.secrets };
        }
        if (this.connectionProfiles.length > 0) {
          this.workspaceState.activeConnectionProfileId = this.connectionProfiles[0].profile_id;
        }
        this.notify();
        return true;
      }
      return false;
    } catch (e) {
      console.error("Failed to import connection profiles", e);
      return false;
    }
  }

  public exportWorkspaceSettings(): string {
    const exportData = {
      version: "1.0",
      exported_at: new Date().toISOString(),
      workspaceState: this.workspaceState,
      sessions: this.sessions,
      connectionProfiles: this.connectionProfiles,
    };
    return JSON.stringify(exportData, null, 2);
  }

  public importWorkspaceSettings(jsonStr: string): boolean {
    try {
      const parsed = JSON.parse(jsonStr);
      if (parsed.workspaceState) {
        this.workspaceState = { ...DEFAULT_WORKSPACE_STATE, ...parsed.workspaceState };
      }
      if (parsed.sessions) {
        this.sessions = { ...DEFAULT_SESSIONS, ...parsed.sessions };
      }
      if (Array.isArray(parsed.connectionProfiles)) {
        this.connectionProfiles = parsed.connectionProfiles;
      }
      this.notify();
      return true;
    } catch (e) {
      console.error("Failed to import workspace settings", e);
      return false;
    }
  }

  // Local Storage Persistence
  private saveToLocalStorage() {
    try {
      const payload = {
        workspaceState: this.workspaceState,
        connectionProfiles: this.connectionProfiles,
        apiKeySecrets: this.apiKeySecrets,
        sessions: this.sessions,
        cacheMetadata: this.cacheMetadata,
      };
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(payload));
    } catch (e) {
      console.error("Failed to write to localStorage", e);
    }
  }

  private loadFromLocalStorage() {
    try {
      const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed.workspaceState) {
          this.workspaceState = { ...DEFAULT_WORKSPACE_STATE, ...parsed.workspaceState };
        }
        if (Array.isArray(parsed.connectionProfiles) && parsed.connectionProfiles.length > 0) {
          this.connectionProfiles = parsed.connectionProfiles;
        }
        if (parsed.apiKeySecrets) {
          this.apiKeySecrets = parsed.apiKeySecrets;
        }
        if (parsed.sessions) {
          this.sessions = { ...DEFAULT_SESSIONS, ...parsed.sessions };
        }
        if (parsed.cacheMetadata) {
          this.cacheMetadata = { ...DEFAULT_CACHE_METADATA, ...parsed.cacheMetadata };
        }
      }
    } catch (e) {
      console.error("Failed to parse localStorage state", e);
    }
  }
}

// Global Singleton Application State Instance
export const appState = new AppStateStore();

// React Hook for Reactive Component Binding
export function useAppState() {
  const [, setVersion] = useState(appState.version);
  useEffect(() => {
    return appState.subscribe(() => {
      setVersion(appState.version);
    });
  }, []);
  return appState;
}
