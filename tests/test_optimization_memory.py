"""
Unit and Regression Tests for AutoRAG Historical Optimization Memory Toggle,
Sequential Adaptive Continuation, Cross-KB Isolation, Dataset Lineage Isolation,
Memory Capping, and Zero-Trace Experiment Reset.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock

from autorag.types import (
    PipelineConfig, ChunkingConfig, RetrieverConfig, OptimizationSpec,
    EvaluationResult, RetrievalMetrics, AnswerMetrics, Dataset, DatasetItem, Document
)
from autorag.workspace import WorkspaceManager
from autorag.optimization_engine import (
    OptimizationEngine,
    build_optimization_memory,
    compute_config_hash
)


class DummyLLMClient:
    def __init__(self, responses=None):
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
        
        # Default intelligent response with valid JSON
        return {
            "text": """
            {
                "chunk_size": 256,
                "chunk_overlap": 32,
                "chunk_strategy": "recursive",
                "retriever_strategy": "hybrid",
                "distance_metric": "cosine",
                "top_k": 5,
                "hybrid_alpha": 0.65,
                "system_prompt": "You are a specialized enterprise assistant.",
                "reasoning": "Exploring smaller chunks and tighter hybrid alpha based on prior performance trends."
            }
            """
        }


class DummyEmbeddingClient:
    def embed_query(self, text):
        return [0.1] * 32

    def embed_documents(self, texts):
        return [[0.1] * 32 for _ in texts]


class TestOptimizationMemoryAndIsolation(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="autorag_memory_test_")
        self.workspace = WorkspaceManager(self.test_dir)
        self.llm_client = DummyLLMClient()
        self.emb_client = DummyEmbeddingClient()
        self.engine = OptimizationEngine(self.workspace, self.llm_client, self.emb_client)

        self.kb_checksum_a = "kb_hash_aaaa1111"
        self.kb_checksum_b = "kb_hash_bbbb2222"
        self.dataset_id_1 = "ds_dataset_001"
        self.dataset_id_2 = "ds_dataset_002"

        # Mock Docs & Dataset
        self.docs = [Document(doc_id="d1", filename="doc1.txt", filepath="doc1.txt", file_type="txt", content="AutoRAG is an enterprise RAG optimization platform.")]
        self.dataset = Dataset(
            dataset_id=self.dataset_id_1,
            created_at="2026-01-01T00:00:00Z",
            version="1.0",
            kb_checksum=self.kb_checksum_a,
            items=[DatasetItem(item_id="q1", doc_id="d1", chunk_id="c1", question="What is AutoRAG?", ground_truth="AutoRAG is an enterprise RAG optimization platform.")]
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _seed_trial(self, exp_id: str, kb_checksum: str, dataset_id: str, run_id: str, score: float, chunk_size: int = 512, valid: bool = True, strategy: str = "hybrid"):
        p_cfg = PipelineConfig(
            experiment_name=exp_id,
            chunking_config=ChunkingConfig(strategy="recursive", chunk_size=chunk_size, chunk_overlap=64),
            retriever_config=RetrieverConfig(strategy=strategy, distance_metric="cosine", top_k=4, hybrid_alpha=0.7)
        )
        res = EvaluationResult(
            experiment_id=exp_id,
            pipeline_config=p_cfg,
            retrieval_metrics=RetrievalMetrics(hit_rate=score if valid else 0.0, precision=score if valid else 0.0),
            answer_metrics=AnswerMetrics(faithfulness=score if valid else 0.0, accuracy=score if valid else 0.0),
            avg_latency_ms=100.0,
            total_tokens=450,
            metrics_valid=valid,
            status="completed" if valid else "failed",
            failure_reason=None if valid else "Index build failed",
            composite_score=score if valid else None
        )
        self.workspace.save_experiment(
            exp_id,
            p_cfg,
            res,
            kb_checksum=kb_checksum,
            dataset_id=dataset_id,
            optimization_run_id=run_id
        )

    def test_build_optimization_memory_toggle_on_off(self):
        """Test build_optimization_memory behavior with toggle ON vs OFF."""
        # Seed 3 historical trials for KB-A and Dataset-1
        self._seed_trial("exp_hist_1", self.kb_checksum_a, self.dataset_id_1, "run_past_1", 0.75, chunk_size=256)
        self._seed_trial("exp_hist_2", self.kb_checksum_a, self.dataset_id_1, "run_past_1", 0.88, chunk_size=512)
        self._seed_trial("exp_hist_3", self.kb_checksum_a, self.dataset_id_1, "run_past_1", 0.60, chunk_size=1024)

        current_trials = [
            {
                "experiment_id": "exp_curr_1",
                "config": {"chunking_config": {"strategy": "recursive", "chunk_size": 128, "chunk_overlap": 16}, "retriever_config": {"strategy": "dense", "distance_metric": "cosine", "top_k": 3, "hybrid_alpha": 0.5}, "system_prompt": "Test"},
                "composite_score": 0.80,
                "retrieval_hit_rate": 0.85,
                "answer_faithfulness": 0.75,
                "status": "completed",
                "metrics_valid": True
            }
        ]

        # Case 1: Toggle ON
        mem_on = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum=self.kb_checksum_a,
            dataset_id=self.dataset_id_1,
            current_run_id="run_current_2",
            current_run_trials=current_trials,
            use_previous_history=True,
            max_historical_trials=5
        )

        self.assertTrue(mem_on["used_previous_history"])
        self.assertEqual(len(mem_on["historical_memory"]), 3)
        self.assertEqual(mem_on["historical_memory_count"], 3)
        self.assertEqual(mem_on["best_historical_score"], 0.88)
        self.assertEqual(mem_on["best_historical_trial_id"], "exp_hist_2")
        self.assertEqual(len(mem_on["current_run_memory"]), 1)
        self.assertEqual(mem_on["best_current_score"], 0.80)

        # Verify historical memory sorted descending by composite score
        scores = [t["composite_score"] for t in mem_on["historical_memory"]]
        self.assertEqual(scores, [0.88, 0.75, 0.60])

        # Case 2: Toggle OFF (Hard Isolation)
        mem_off = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum=self.kb_checksum_a,
            dataset_id=self.dataset_id_1,
            current_run_id="run_current_2",
            current_run_trials=current_trials,
            use_previous_history=False
        )

        self.assertFalse(mem_off["used_previous_history"])
        self.assertEqual(len(mem_off["historical_memory"]), 0)
        self.assertEqual(mem_off["historical_memory_count"], 0)
        self.assertIsNone(mem_off["best_historical_score"])
        self.assertIsNone(mem_off["best_historical_trial_id"])
        self.assertEqual(len(mem_off["current_run_memory"]), 1)
        self.assertEqual(mem_off["best_current_score"], 0.80)

    def test_cross_kb_and_dataset_lineage_isolation(self):
        """Historical trials from different KBs or different datasets must NEVER bleed through."""
        # Seed trials for KB-A / DS-1
        self._seed_trial("exp_kba_ds1", self.kb_checksum_a, self.dataset_id_1, "run_past", 0.90)
        # Seed trial for KB-B / DS-1
        self._seed_trial("exp_kbb_ds1", self.kb_checksum_b, self.dataset_id_1, "run_past", 0.95)
        # Seed trial for KB-A / DS-2
        self._seed_trial("exp_kba_ds2", self.kb_checksum_a, self.dataset_id_2, "run_past", 0.85)

        # Optimize for KB-A and DS-1 with history ON
        mem_kba = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum=self.kb_checksum_a,
            dataset_id=self.dataset_id_1,
            current_run_id="run_new",
            current_run_trials=[],
            use_previous_history=True
        )
        self.assertEqual(len(mem_kba["historical_memory"]), 1)
        self.assertEqual(mem_kba["historical_memory"][0]["experiment_id"], "exp_kba_ds1")

        # Optimize for KB-B and DS-1
        mem_kbb = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum=self.kb_checksum_b,
            dataset_id=self.dataset_id_1,
            current_run_id="run_new",
            current_run_trials=[],
            use_previous_history=True
        )
        self.assertEqual(len(mem_kbb["historical_memory"]), 1)
        self.assertEqual(mem_kbb["historical_memory"][0]["experiment_id"], "exp_kbb_ds1")

    def test_failed_and_invalid_trials_handling_in_memory(self):
        """Failed or invalid-metric trials must not be selected as best trial, but can be kept as negative samples."""
        self._seed_trial("exp_valid_1", self.kb_checksum_a, self.dataset_id_1, "run_1", 0.70, valid=True)
        self._seed_trial("exp_failed_1", self.kb_checksum_a, self.dataset_id_1, "run_1", 0.0, valid=False)

        mem = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum=self.kb_checksum_a,
            dataset_id=self.dataset_id_1,
            current_run_id="run_2",
            current_run_trials=[],
            use_previous_history=True
        )

        self.assertEqual(mem["historical_memory_count"], 2)
        # Best historical score must come strictly from valid trial
        self.assertEqual(mem["best_historical_score"], 0.70)
        self.assertEqual(mem["best_historical_trial_id"], "exp_valid_1")

        # Verify failed trial is marked properly
        failed_entry = next(t for t in mem["historical_memory"] if t["experiment_id"] == "exp_failed_1")
        self.assertFalse(failed_entry["metrics_valid"])
        self.assertEqual(failed_entry["status"], "failed")

    def test_memory_size_capping(self):
        """Ensure historical memory is capped at max_historical_trials and keeps highest scoring trials."""
        for i in range(12):
            score = 0.50 + (i * 0.03)  # scores: 0.50, 0.53, ..., 0.83
            self._seed_trial(f"exp_cap_{i}", self.kb_checksum_a, self.dataset_id_1, "run_past", score, chunk_size=100 + (i * 50))

        mem = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum=self.kb_checksum_a,
            dataset_id=self.dataset_id_1,
            current_run_id="run_new",
            current_run_trials=[],
            use_previous_history=True,
            max_historical_trials=6
        )

        self.assertEqual(len(mem["historical_memory"]), 6)
        self.assertEqual(mem["historical_memory_count"], 6)
        # Top score should be highest (0.83)
        self.assertAlmostEqual(mem["historical_memory"][0]["composite_score"], 0.83, places=2)
        # 6th score should be 0.68
        self.assertAlmostEqual(mem["historical_memory"][-1]["composite_score"], 0.68, places=2)

    def test_llm_prompt_no_leakage_when_history_off(self):
        """Ensure LLM prompt contains ZERO historical trial IDs or prior scores when toggle is OFF."""
        self._seed_trial("exp_secret_hist_999", self.kb_checksum_a, self.dataset_id_1, "run_past", 0.94)

        spec_off = OptimizationSpec(strategy="llm_guided", max_trials=2, use_previous_history=False)
        base_cfg = PipelineConfig(experiment_name="base")

        # Run proposal
        memory_ctx = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum=self.kb_checksum_a,
            dataset_id=self.dataset_id_1,
            current_run_id="run_test_leakage",
            current_run_trials=[],
            use_previous_history=False
        )
        self.engine.propose_candidate_llm(base_cfg, [], memory_ctx=memory_ctx)

        prompt = self.llm_client.last_prompt
        self.assertNotIn("exp_secret_hist_999", prompt)
        self.assertNotIn("0.94", prompt)
        self.assertIn("HISTORICAL OPTIMIZATION MEMORY (PAST RUNS): OFF", prompt)

    def test_clear_experiments_zero_trace_reset(self):
        """Clearing experiments removes all traces and future sweeps have 0 historical trials."""
        self._seed_trial("exp_1", self.kb_checksum_a, self.dataset_id_1, "run_1", 0.85)
        self._seed_trial("exp_2", self.kb_checksum_a, self.dataset_id_1, "run_1", 0.89)

        exps_before = self.workspace.list_experiments(kb_checksum=self.kb_checksum_a)
        self.assertEqual(len(exps_before), 2)

        deleted = self.workspace.clear_experiments(kb_checksum=self.kb_checksum_a)
        self.assertEqual(deleted, 2)

        exps_after = self.workspace.list_experiments(kb_checksum=self.kb_checksum_a)
        self.assertEqual(len(exps_after), 0)

        mem = build_optimization_memory(
            workspace=self.workspace,
            kb_checksum=self.kb_checksum_a,
            dataset_id=self.dataset_id_1,
            current_run_id="run_fresh",
            current_run_trials=[],
            use_previous_history=True
        )
        self.assertEqual(len(mem["historical_memory"]), 0)
        self.assertEqual(mem["historical_memory_count"], 0)

    def test_duplicate_config_detection_with_history(self):
        """Validate that validate_config detects duplicates in historical memory when toggle is enabled."""
        hist_cfg = PipelineConfig(
            experiment_name="hist",
            chunking_config=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=64),
            retriever_config=RetrieverConfig(strategy="hybrid", distance_metric="cosine", top_k=4, hybrid_alpha=0.7)
        )
        hist_records = [{"config": hist_cfg.__dict__, "experiment_id": "exp_hist"}]

        # Same config proposed
        test_cfg = PipelineConfig(
            experiment_name="test",
            chunking_config=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=64),
            retriever_config=RetrieverConfig(strategy="hybrid", distance_metric="cosine", top_k=4, hybrid_alpha=0.7)
        )

        is_valid, reason = self.engine.validate_config(test_cfg, current_run_history=[], historical_history=hist_records)
        # Should detect collision and return False
        self.assertFalse(is_valid)
        self.assertIn("previously evaluated", reason)

    def test_strictly_sequential_execution_in_optimize(self):
        """Verify that optimize executes trials strictly sequentially and records run metadata."""
        spec = OptimizationSpec(strategy="llm_guided", max_trials=3, use_previous_history=True)
        base_cfg = PipelineConfig(experiment_name="sweep_base")

        # Mock run_trial to verify sequential calls and return realistic results
        call_records = []

        def mock_run_trial(p_cfg, docs, dataset):
            trial_num = len(call_records) + 1
            exp_id = f"exp_seq_trial_{trial_num}"
            score = 0.70 + (trial_num * 0.05)
            call_records.append({"trial": trial_num, "config": p_cfg, "score": score})
            
            res = EvaluationResult(
                experiment_id=exp_id,
                pipeline_config=p_cfg,
                retrieval_metrics=RetrievalMetrics(hit_rate=score, precision=score),
                answer_metrics=AnswerMetrics(faithfulness=score, accuracy=score),
                avg_latency_ms=100.0,
                total_tokens=400,
                metrics_valid=True,
                status="completed",
                composite_score=score
            )
            # Save experiment as run_trial normally does
            self.workspace.save_experiment(
                exp_id,
                p_cfg,
                res,
                kb_checksum=self.kb_checksum_a,
                dataset_id=self.dataset_id_1,
                optimization_run_id="run_seq_test"
            )
            from autorag.types import Experiment
            return Experiment(experiment_id=exp_id, pipeline_id="p1", dataset_id=self.dataset_id_1, timestamp="2026-01-01T00:00:00Z", runtime=1.5, config=p_cfg, results=res, composite_score=score)

        self.engine.run_trial = mock_run_trial

        summary = self.engine.optimize(spec, base_cfg, self.docs, self.dataset)
        self.assertEqual(len(call_records), 3)
        self.assertEqual(summary["total_trials"], 3)
        self.assertEqual(len(summary["leaderboard"]), 3)
        self.assertTrue(summary["used_previous_history"])
        self.assertIsNotNone(summary["best_score"])


if __name__ == "__main__":
    unittest.main()
