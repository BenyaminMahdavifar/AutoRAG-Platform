export interface DocumentMeta {
  doc_id: string;
  filename: string;
  filepath: string;
  file_type: string;
  size_bytes: number;
  checksum: string;
  content_preview?: string;
  metadata?: Record<string, any>;
}

export interface Manifest {
  scanned_at: string;
  total_docs: number;
  total_size_bytes: number;
  kb_checksum: string;
  files: DocumentMeta[];
}

export interface DatasetItem {
  item_id: string;
  doc_id: string;
  chunk_id: string;
  question: string;
  ground_truth: string;
  metadata?: Record<string, any>;
}

export interface Dataset {
  dataset_id: string;
  created_at: string;
  version: string;
  items: DatasetItem[];
}

export interface RetrievalMetrics {
  precision: number;
  recall: number;
  hit_rate: number;
  mrr: number;
  ndcg: number;
}

export interface AnswerMetrics {
  answer_correctness?: number;
  accuracy?: number;
  faithfulness?: number;
  completeness?: number;
  relevance?: number;
  answer_relevance?: number;
  semantic_similarity?: number;
  context_utilization?: number;
}

export interface PipelineConfig {
  experiment_name: string;
  llm_config?: {
    provider?: string;
    model_name?: string;
    base_url?: string;
  };
  embedding_config?: {
    provider?: string;
    model_name?: string;
  };
  chunking_config: {
    strategy: "recursive" | "fixed" | "paragraph" | "semantic";
    chunk_size: number;
    chunk_overlap: number;
  };
  retriever_config: {
    strategy: "hybrid" | "dense" | "sparse";
    distance_metric: "cosine" | "dot" | "euclidean";
    top_k: number;
    hybrid_alpha: number;
  };
  system_prompt: string;
}

export interface TrialResult {
  experiment_id: string;
  composite_score: number | null;
  retrieval_metrics: RetrievalMetrics;
  answer_metrics: AnswerMetrics;
  avg_latency_ms: number;
  total_tokens: number;
  sample_evaluations: any[];
  timestamp: string;
  status?: string;
  metrics_valid?: boolean;
  failure_reason?: string;
  completed_batches?: number;
  failed_batches?: number;
}

export interface GeneralizationSampleEvaluation {
  question: string;
  ground_truth: string;
  generated_answer: string;
  retrieved_chunks_count: number;
  hit_rate: number;
  precision: number;
  answer_correctness?: number;
  faithfulness?: number;
  accuracy?: number;
}

export interface GeneralizationTestResult {
  test_id: string;
  experiment_id: string;
  kb_checksum: string;
  test_size: number;
  optimization_composite_score: number | null;
  generalization_composite_score: number;
  score_delta: number | null;
  retrieval_metrics: RetrievalMetrics;
  answer_metrics: AnswerMetrics;
  avg_latency_ms: number;
  total_tokens: number;
  sample_evaluations: GeneralizationSampleEvaluation[];
  status: "completed" | "failed" | "running";
  metrics_valid: boolean;
  summary_text: string;
  dataset_id: string;
  timestamp: string;
  failure_reason?: string;
}

export interface LeaderboardEntry {
  trial: number;
  experiment_id: string;
  composite_score: number | null;
  retrieval_hit_rate: number | null;
  answer_correctness?: number | null;
  answer_faithfulness: number | null;
  config: PipelineConfig;
  result: TrialResult;
  timestamp: string;
  status?: string;
  generalization_test?: GeneralizationTestResult;
}

export interface ConnectionConfig {
  provider: "openai" | "ollama" | "lmstudio" | "openrouter" | "gemini";
  base_url: string;
  api_key: string;
  model_name: string;
  hf_token?: string;
  embedding_model?: string;
  embedding_device?: "auto" | "cpu" | "cuda" | "mps";
  timeout_sec?: number;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
}

export type JobStatus = "Starting" | "Queued" | "Running" | "Completed" | "Failed" | "Cancelled";
export type JobType = "scan_kb" | "upload_doc" | "import_kb" | "clear_kb" | "build_dataset" | "run_experiment" | "run_optimizer" | "run_generalization_test" | "export_reports" | "setup_environment" | "export_rag";

export interface JobStage {
  id: string;
  label: string;
  status: "completed" | "current" | "pending";
}

export interface Job {
  job_id: string;
  type: JobType;
  action?: string;
  title: string;
  status: JobStatus;
  progress: number;
  current_stage: string;
  completed_stages: string[];
  started_at: string;
  updated_at: string;
  estimated_remaining?: string;
  logs: string[];
  result?: any;
  error?: string;
  suggested_fix?: string;
  payload?: any;
}

export interface ConnectionProfile {
  profile_id: string;
  name: string;
  provider: "openai" | "ollama" | "lmstudio" | "openrouter" | "gemini" | "custom";
  base_url: string;
  default_model: string;
  api_key_reference: string;
  hf_token_reference?: string;
  embedding_model?: string;
  embedding_device?: "auto" | "cpu" | "cuda" | "mps";
  auto_login?: boolean;
  is_default?: boolean;
  timeout_sec?: number;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
}

export interface WorkspaceState {
  activeTab: string;
  selectedDatasetId: string | null;
  selectedPipelineId: string | null;
  selectedExperimentId: string | null;
  activeConnectionProfileId: string;
  lastOpenedPage: string;
}

export interface PlaygroundSession {
  chunkStrategy: "recursive" | "fixed" | "paragraph" | "semantic";
  chunkSize: number;
  chunkOverlap: number;
  retrieverStrategy: "hybrid" | "dense" | "sparse";
  distanceMetric: "cosine" | "dot" | "euclidean";
  topK: number;
  systemPrompt: string;
  lastTrialResult: TrialResult | null;
}

export interface OptimizerSession {
  strategy: "llm_guided" | "grid" | "random";
  maxTrials: number;
  usePreviousOptimizationHistory?: boolean;
  learnFromGeneralizationTest?: boolean;
  lastSweepSummary: any | null;
}

export interface DatasetSession {
  filterText: string;
}

export interface KbSession {
  selectedDocId: string | null;
  uploadName: string;
  uploadContent: string;
  chunkStrategy: "recursive" | "fixed" | "paragraph" | "semantic";
  chunkSize: number;
}

export interface ReportsSession {
  filterText: string;
}

export interface CacheMetadata {
  documentsCount: number;
  experimentsCount: number;
  reportsCount: number;
  datasetItemsCount: number;
  lastUpdated: string;
}

