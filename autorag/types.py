"""
Type definitions and data classes for AutoRAG-Platform.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import json

import hashlib

@dataclass
class Artifact:
    artifact_id: str
    type: str  # e.g., "manifest", "chunks", "embeddings", "index", "retrieved_results", "generated_answers", "report"
    path: str
    hash: str
    creation_time: str

@dataclass
class EnvironmentSnapshot:
    python_version: str
    os: str
    architecture: str
    virtual_env: bool
    pip_executable: str
    python_executable: str




@dataclass
class LLMConfig:

    provider: str = "openai"  # "openai", "ollama", "lmstudio", "openrouter", "gemini", "custom"
    base_url: str = "https://api.openai.com/v1"

    api_key: str = ""

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_p: float = 0.95
    timeout_sec: int = 30
    max_tokens: int = 1024

    def get_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()



@dataclass
class EmbeddingConfig:
    provider: str = "local"  # "local", "openai", "gemini"
    model_name: str = "local-tfidf-512"
    dimension: int = 512
    base_url: str = "https://api.openai.com/v1"
    device: str = "auto"
    api_key: str = ""

    def get_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()



@dataclass
class ChunkingConfig:
    split_method: Literal["fixed", "recursive", "paragraph", "sentence", "semantic"] = "recursive"
    strategy: str = "recursive" # alias
    chunk_size: int = 512

    chunk_overlap: int = 64

    def get_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()



@dataclass
class RetrieverConfig:
    strategy: Literal["dense", "sparse", "hybrid"] = "hybrid"
    distance_metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    top_k: int = 4
    score_threshold: float = 0.0
    hybrid_alpha: float = 0.7
    use_mmr: bool = False

    index_type: str = "memory"  # "memory", "faiss", "chroma"

    def get_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()



@dataclass
class PipelineConfig:
    experiment_name: str
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever_config: RetrieverConfig = field(default_factory=RetrieverConfig)
    system_prompt: str = (
        "You are a helpful assistant. Use ONLY the provided context to answer the question. "
        "If you do not know the answer from the context, state that you don't know."
    )

    @property
    def pipeline_id(self) -> str:
        d = asdict(self)
        d.pop("experiment_name", None)
        j = json.dumps(d, sort_keys=True)
        return hashlib.sha256(j.encode("utf-8")).hexdigest()


@dataclass
class Document:
    doc_id: str
    filepath: str
    filename: str
    file_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    size_bytes: int = 0


@dataclass
class KnowledgeBaseManifest:
    scanned_at: str
    total_docs: int
    total_size_bytes: int
    kb_checksum: str
    files: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetItem:
    item_id: str
    doc_id: str
    chunk_id: str
    question: str
    ground_truth: str
    reference_answer: str = ""
    gold_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    dataset_id: str
    created_at: str
    version: str
    kb_checksum: str
    items: List[DatasetItem] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    precision: float = 0.0
    recall: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0


@dataclass
class AnswerMetrics:
    accuracy: float = 0.0
    completeness: float = 0.0
    relevance: float = 0.0
    answer_correctness: float = 0.0
    faithfulness: float = 0.0  # Backward compatibility alias for answer_correctness/accuracy
    answer_relevance: float = 0.0  # for compatibility
    semantic_similarity: float = 0.0  # for compatibility
    context_utilization: float = 0.0  # for compatibility

    def __post_init__(self):
        val = self.answer_correctness or self.accuracy or self.faithfulness or 0.0
        if not self.answer_correctness and val:
            self.answer_correctness = val
        if not self.accuracy and val:
            self.accuracy = val
        if not self.faithfulness and val:
            self.faithfulness = val



@dataclass
class EvaluationResult:
    experiment_id: str
    pipeline_config: PipelineConfig
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    avg_latency_ms: float
    total_tokens: int
    sample_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    orchestrator_metrics: Optional[Dict[str, Any]] = None
    
    status: str = "completed"
    metrics_valid: bool = True
    failure_reason: Optional[str] = None
    completed_batches: int = 0
    failed_batches: int = 0
    retry_count: int = 0
    
    composite_score: Optional[float] = None


@dataclass
class Experiment:
    experiment_id: str
    pipeline_id: str
    dataset_id: str
    timestamp: str
    runtime: float
    results: Optional[EvaluationResult] = None
    artifacts: List[Artifact] = field(default_factory=list)
    environment: Optional[EnvironmentSnapshot] = None
    
    name: str = ""
    created_at: str = ""
    config: Optional[PipelineConfig] = None
    status: str = "created"  # "created", "running", "completed", "failed", "partial_failed"
    composite_score: Optional[float] = None
    trial_number: Optional[int] = None
    optimization_run_id: Optional[str] = None
    kb_checksum: Optional[str] = None
    used_previous_history: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)



@dataclass
class OptimizationSpec:
    strategy: Literal["grid", "random", "llm_guided"] = "llm_guided"
    max_trials: int = 6
    target_metric: str = "composite_score"  # "composite_score", "answer_correctness", "faithfulness", "hit_rate"
    use_previous_history: bool = True
    usePreviousOptimizationHistory: bool = True  # alias support
    learn_from_generalization_test: bool = False
    learnFromGeneralizationTest: bool = False  # alias support
    chunk_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024])
    chunk_overlaps: List[int] = field(default_factory=lambda: [0, 64, 128])
    top_k_values: List[int] = field(default_factory=lambda: [2, 4, 8])
    strategies: List[str] = field(default_factory=lambda: ["dense", "hybrid"])


@dataclass
class GeneralizationSampleEvaluation:
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_chunks_count: int
    hit_rate: float
    precision: float
    accuracy: float = 0.0
    answer_correctness: float = 0.0
    faithfulness: float = 0.0  # Backward compatibility alias
    latency_ms: float = 0.0

    def __post_init__(self):
        val = self.answer_correctness or self.accuracy or self.faithfulness or 0.0
        if not self.answer_correctness and val:
            self.answer_correctness = val
        if not self.accuracy and val:
            self.accuracy = val
        if not self.faithfulness and val:
            self.faithfulness = val


@dataclass
class GeneralizationTestResult:
    test_id: str
    experiment_id: str
    kb_checksum: str
    test_size: int
    optimization_composite_score: Optional[float]
    generalization_composite_score: Optional[float]
    score_delta: Optional[float]
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    avg_latency_ms: float
    total_tokens: int
    sample_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "completed"  # "completed", "failed", "partial_failed"
    metrics_valid: bool = True
    failure_reason: Optional[str] = None
    summary_text: Optional[str] = None
    dataset_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

