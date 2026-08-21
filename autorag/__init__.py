"""
AutoRAG-Platform Package.
Contains 10 modular, deterministic, type-safe subsystems.
"""

from .types import (
    LLMConfig, EmbeddingConfig, ChunkingConfig, RetrieverConfig, PipelineConfig,
    Document, KnowledgeBaseManifest, Chunk, DatasetItem, Dataset,
    RetrievalMetrics, AnswerMetrics, EvaluationResult, Experiment, OptimizationSpec,
    GeneralizationTestResult
)
from .workspace import WorkspaceManager
from .connections import OpenAICompatibleClient, LocalTFIDFEmbedder, EmbeddingClient
from .torch_policy import (
    PYTORCH_CUDA_VARIANT, PYTORCH_INDEX_URL, PYTORCH_PACKAGE,
    CUDAUnavailableError, resolve_embedding_device, inspect_installed_torch, get_nvidia_driver_info
)
from .knowledge_base import KnowledgeBaseScanner
from .dataset_builder import DatasetBuilder
from .index_builder import TextChunker, VectorIndex, IndexBuilder
from .retrieval_engine import cosine_similarity, dot_product, BM25Scorer, RetrievalEngine
from .generation_engine import GenerationEngine
from .evaluation_engine import EvaluationEngine
from .optimization_engine import OptimizationEngine
from .generalization_engine import GeneralizationEngine
from .report_engine import ReportEngine

__version__ = "1.0.0"
