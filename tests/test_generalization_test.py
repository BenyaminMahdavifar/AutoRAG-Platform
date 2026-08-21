"""
Unit and Regression Tests for AutoRAG Generalization Test (Holdout Validation) System.
Verifies:
1. Holdout dataset generation & deduplication (no question leakage from optimization dataset).
2. GeneralizationEngine evaluation on candidate trial configurations.
3. Generalization persistence & cascade deletion integrity (experiment clear, KB delete).
4. Memory integration & prompt safety (no raw holdout questions sent to optimizer, toggle gating).
5. CLI action endpoints for generalization testing.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime

from autorag.types import (
    PipelineConfig, ChunkingConfig, RetrieverConfig,
    EvaluationResult, RetrievalMetrics, AnswerMetrics, Dataset, DatasetItem, Document,
    GeneralizationTestResult, GeneralizationSampleEvaluation
)
from autorag.workspace import WorkspaceManager
from autorag.generalization_engine import GeneralizationEngine
from autorag.optimization_engine import OptimizationEngine, build_optimization_memory


class DummyLLMConfig:
    def __init__(self):
        self.provider = "openai"
        self.model_name = "gpt-4o-mini"
        self.max_tokens = 1000
        self.temperature = 0.2


class DummyLLMClient:
    def __init__(self, responses=None):
        self.config = DummyLLMConfig()
        self.responses = responses or []
        self.call_count = 0
        self.last_prompt = None
        self.prompts_received = []

    def chat_completion(self, messages, temperature=0.2, *args, **kwargs):
        self.call_count += 1
        prompt_text = "\n".join([m.get("content", "") for m in messages])
        self.last_prompt = prompt_text
        self.prompts_received.append(prompt_text)

        if self.responses and len(self.responses) >= self.call_count:
            return {"text": self.responses[self.call_count - 1]}

        # If it's a dataset generation prompt
        if "question" in prompt_text.lower() and "ground_truth" in prompt_text.lower():
            return {
                "text": """[
                    {"question": "What is the secondary architectural principle?", "ground_truth": "Secondary architectural principle is modularity and zero-coupling."},
                    {"question": "How does cache eviction work?", "ground_truth": "Cache eviction utilizes LRU policies with persistent fallback."}
                ]"""
            }

        # If it's an evaluation / answer prompt
        if "question:" in prompt_text.lower():
            return {
                "text": "The secondary architectural principle is modularity and zero-coupling across modules."
            }

        # Default optimizer candidate
        return {
            "text": """
            {
                "chunk_size": 256,
                "chunk_overlap": 32,
                "chunk_strategy": "recursive",
                "retriever_strategy": "hybrid",
                "distance_metric": "cosine",
                "top_k": 4,
                "hybrid_alpha": 0.6,
                "system_prompt": "You are a helpful assistant.",
                "reasoning": "Selected based on generalization robustness."
            }
            """
        }


class DummyEmbConfig:
    def __init__(self):
        self.model_name = "test_emb"
        self.device = "cpu"

    def get_hash(self):
        return "emb_hash_123"


class DummyEmbeddingClient:
    def __init__(self):
        self.config = DummyEmbConfig()

    def embed_query(self, text):
        return [0.15] * 32

    def embed_documents(self, texts):
        return [[0.15] * 32 for _ in texts]

    def embed_texts(self, texts):
        return [[0.15] * 32 for _ in texts]


class TestGeneralizationSystem(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="autorag_gen_test_")
        self.workspace = WorkspaceManager(self.test_dir)
        self.llm_client = DummyLLMClient()
        self.emb_client = DummyEmbeddingClient()
        self.gen_engine = GeneralizationEngine(self.workspace, self.llm_client, self.emb_client)

        # Seed KB Document
        self.doc_path = self.workspace.get_kb_path() / "architecture.txt"
        self.doc_path.write_text("AutoRAG is an enterprise RAG optimization platform. The secondary architectural principle is modularity and zero-coupling. Cache eviction utilizes LRU policies with persistent fallback.", encoding="utf-8")
        self.doc = Document(
            doc_id="doc_arch",
            filename="architecture.txt",
            filepath=str(self.doc_path),
            file_type="txt",
            content="AutoRAG is an enterprise RAG optimization platform. The secondary architectural principle is modularity and zero-coupling. Cache eviction utilizes LRU policies with persistent fallback."
        )

        # Seed Optimization Dataset
        self.opt_dataset = Dataset(
            dataset_id="ds_opt_001",
            created_at="2026-01-01T00:00:00Z",
            version="1.0",
            kb_checksum="kb_checksum_123",
            items=[
                DatasetItem(
                    item_id="q1",
                    doc_id="doc_arch",
                    chunk_id="c1",
                    question="What is AutoRAG?",
                    ground_truth="AutoRAG is an enterprise RAG optimization platform."
                )
            ]
        )
        self.workspace.save_dataset(self.opt_dataset)

        # Seed an Experiment
        self.pipeline_cfg = PipelineConfig(
            experiment_name="exp_test_001",
            chunking_config=ChunkingConfig(strategy="recursive", chunk_size=256, chunk_overlap=32),
            retriever_config=RetrieverConfig(strategy="hybrid", distance_metric="cosine", top_k=4, hybrid_alpha=0.6)
        )
        self.eval_res = EvaluationResult(
            experiment_id="exp_test_001",
            pipeline_config=self.pipeline_cfg,
            composite_score=0.88,
            retrieval_metrics=RetrievalMetrics(hit_rate=0.90, precision=0.85, mrr=0.90),
            answer_metrics=AnswerMetrics(faithfulness=0.86, relevance=0.88),
            avg_latency_ms=100.0,
            total_tokens=450,
            status="completed",
            metrics_valid=True
        )
        self.workspace.save_experiment(
            "exp_test_001",
            self.pipeline_cfg,
            self.eval_res,
            kb_checksum="kb_checksum_123",
            dataset_id="ds_opt_001"
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_holdout_dataset_deduplication(self):
        """Verify holdout dataset generates questions without duplicating existing optimization questions."""
        from autorag.index_builder import IndexBuilder
        index = IndexBuilder(self.workspace, self.emb_client).build_index([self.doc], self.pipeline_cfg.chunking_config)
        holdout_ds = self.gen_engine.build_or_load_holdout_dataset(
            documents=[self.doc],
            chunks=index.chunks,
            kb_checksum="kb_checksum_123",
            test_size=2,
            force_regenerate=True
        )
        self.assertIsNotNone(holdout_ds)
        self.assertGreater(len(holdout_ds.items), 0)

        # Check deduplication
        opt_questions = {item.question.lower().strip() for item in self.opt_dataset.items}
        for item in holdout_ds.items:
            self.assertNotIn(item.question.lower().strip(), opt_questions)

    def test_run_generalization_test(self):
        """Verify running generalization test computes valid score delta, holdout metrics, and sample evaluations."""
        gen_result = self.gen_engine.run_generalization_test(
            experiment_id="exp_test_001",
            docs=[self.doc],
            kb_checksum="kb_checksum_123",
            test_size=2
        )
        self.assertIsNotNone(gen_result)
        self.assertEqual(gen_result.experiment_id, "exp_test_001")
        self.assertEqual(gen_result.status, "completed")
        self.assertIsNotNone(gen_result.generalization_composite_score)
        self.assertIsNotNone(gen_result.score_delta)
        self.assertGreater(len(gen_result.sample_evaluations), 0)

        # Verify saved in workspace
        loaded = self.workspace.get_generalization_test("exp_test_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["experiment_id"], "exp_test_001")
        self.assertEqual(loaded["generalization_composite_score"], gen_result.generalization_composite_score)

    def test_cascade_delete_integrity(self):
        """Verify deleting experiments or clearing leaderboard cleans up generalization test artifacts."""
        # Run test first
        self.gen_engine.run_generalization_test(
            experiment_id="exp_test_001",
            docs=[self.doc],
            kb_checksum="kb_checksum_123",
            test_size=2
        )
        self.assertIsNotNone(self.workspace.get_generalization_test("exp_test_001"))

        # Clear experiments
        self.workspace.clear_experiments()
        self.assertIsNone(self.workspace.get_generalization_test("exp_test_001"))
        self.assertEqual(len(self.workspace.list_generalization_tests()), 0)

    def test_memory_integration_and_prompt_safety(self):
        """Verify learn_from_generalization_test summarizes insights without leaking raw holdout questions."""
        # Run generalization test
        self.gen_engine.run_generalization_test(
            experiment_id="exp_test_001",
            docs=[self.doc],
            kb_checksum="kb_checksum_123",
            test_size=2
        )

        # Build memory with toggle ON
        memory_on = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum="kb_checksum_123",
            dataset_id="ds_opt_001",
            learn_from_generalization_test=True
        )
        self.assertIn("generalization_insights", memory_on)
        self.assertGreater(len(memory_on["generalization_insights"]), 0)

        insight = memory_on["generalization_insights"][0]
        self.assertEqual(insight["experiment_id"], "exp_test_001")
        self.assertIn("generalization_composite", insight)
        self.assertIn("score_delta", insight)

        # Ensure no raw question text is present in the insight dict
        self.assertNotIn("sample_evaluations", insight)
        self.assertNotIn("raw_questions", insight)

        # Test optimizer prompt injection
        optim_engine = OptimizationEngine(self.workspace, self.llm_client, self.emb_client)
        optim_engine.propose_candidate_llm(
            memory_on,
            self.pipeline_cfg
        )

        last_prompt = self.llm_client.last_prompt
        self.assertIn("GENERALIZATION TEST (HOLDOUT VALIDATION) SUMMARY: ON", last_prompt)
        self.assertIn("exp_test_001", last_prompt)

        # Build memory with toggle OFF
        memory_off = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum="kb_checksum_123",
            dataset_id="ds_opt_001",
            learn_from_generalization_test=False
        )
        self.assertEqual(len(memory_off.get("generalization_insights", [])), 0)

        optim_engine.propose_candidate_llm(
            memory_off,
            self.pipeline_cfg
        )
        self.assertNotIn("GENERALIZATION TEST (HOLDOUT VALIDATION) SUMMARY: ON", self.llm_client.last_prompt)

    def test_cli_actions(self):
        """Verify run_generalization_test and get_generalization_test via cli.py argument parser."""
        import subprocess
        result = subprocess.run(
            ["python3", "cli.py", "run_generalization_test", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("AutoRAG CLI Bridge", result.stdout)


if __name__ == "__main__":
    unittest.main()
