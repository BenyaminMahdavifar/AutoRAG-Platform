"""
Unit and Regression Tests for AutoRAG Canonical Optimization State,
Workspace Consistency, and Persistence Integrity.
"""

import os
import math
import json
import shutil
import tempfile
import unittest
from datetime import datetime

from autorag.types import (
    EvaluationResult, Experiment, RetrievalMetrics, AnswerMetrics,
    PipelineConfig, ChunkingConfig, RetrieverConfig
)
from autorag.evaluation_engine import EvaluationEngine
from autorag.workspace import WorkspaceManager, normalize_experiment_record, _sanitize_floats
from autorag.report_engine import ReportEngine


class TestStateConsistencyAndPersistence(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="autorag_test_workspace_")
        self.workspace = WorkspaceManager(self.test_dir)
        self.eval_engine = EvaluationEngine()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_composite_score_calculation(self):
        # 1. Normal valid metrics: 0.7 * hit_rate + 0.3 * faithfulness
        rm = RetrievalMetrics(hit_rate=0.8, precision=0.8)
        am = AnswerMetrics(faithfulness=0.8, accuracy=0.8)
        score = self.eval_engine.calculate_composite_score(rm, am)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 0.8, places=4)

        # 2. Perfect metrics
        rm_perfect = RetrievalMetrics(hit_rate=1.0, precision=1.0)
        am_perfect = AnswerMetrics(faithfulness=1.0, accuracy=1.0)
        score_perfect = self.eval_engine.calculate_composite_score(rm_perfect, am_perfect)
        self.assertAlmostEqual(score_perfect, 1.0, places=4)

        # 3. Weighted metrics check (0.7 * 1.0 + 0.3 * 0.0 = 0.7)
        rm_weight = RetrievalMetrics(hit_rate=1.0)
        am_weight = AnswerMetrics(faithfulness=0.0)
        score_weight = self.eval_engine.calculate_composite_score(rm_weight, am_weight)
        self.assertAlmostEqual(score_weight, 0.7, places=4)

    def test_sanitize_non_finite_floats(self):
        raw_data = {
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "neg_inf_val": float("-inf"),
            "normal_float": 0.85,
            "nested": {
                "nan_nested": float("nan"),
                "valid": 42
            },
            "list_val": [float("nan"), 1.23, float("inf")]
        }
        sanitized = _sanitize_floats(raw_data)
        self.assertIsNone(sanitized["nan_val"])
        self.assertIsNone(sanitized["inf_val"])
        self.assertIsNone(sanitized["neg_inf_val"])
        self.assertEqual(sanitized["normal_float"], 0.85)
        self.assertIsNone(sanitized["nested"]["nan_nested"])
        self.assertEqual(sanitized["nested"]["valid"], 42)
        self.assertEqual(sanitized["list_val"], [None, 1.23, None])

    def test_normalize_experiment_record_cleans_legacy_keys(self):
        legacy_record = {
            "experiment_id": "exp_1001",
            "composite_score": 0.88,
            "_composite_score": 0.88,
            "overall_score": 0.88,
            "score": 0.88,
            "status": "completed",
            "results": {
                "composite_score": 0.88,
                "_composite_score": 0.88,
                "overall_score": 0.88,
                "score": 0.88,
                "metrics_valid": True,
                "status": "completed",
                "retrieval_metrics": {"hit_rate": 0.9},
                "answer_metrics": {"faithfulness": 0.85}
            }
        }
        normalized = normalize_experiment_record(legacy_record)
        self.assertEqual(normalized["composite_score"], 0.88)
        self.assertEqual(normalized["results"]["composite_score"], 0.88)
        self.assertNotIn("_composite_score", normalized)
        self.assertNotIn("overall_score", normalized)
        self.assertNotIn("score", normalized)
        self.assertNotIn("_composite_score", normalized["results"])
        self.assertNotIn("overall_score", normalized["results"])
        self.assertNotIn("score", normalized["results"])

    def test_normalize_experiment_record_failed_trial(self):
        failed_record = {
            "experiment_id": "exp_failed_1",
            "composite_score": 0.75,  # Stale score from partial run
            "status": "failed",
            "results": {
                "composite_score": 0.75,
                "metrics_valid": False,
                "status": "failed",
                "failure_reason": "API Key Forbidden"
            }
        }
        normalized = normalize_experiment_record(failed_record)
        self.assertIsNone(normalized["composite_score"])
        self.assertIsNone(normalized["results"]["composite_score"])
        self.assertFalse(normalized["results"]["metrics_valid"])
        self.assertEqual(normalized["status"], "failed")

    def test_workspace_roundtrip_persistence(self):
        exp_data = {
            "experiment_id": "exp_roundtrip_test",
            "composite_score": 0.9123,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "chunking_config": {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 64},
                "retriever_config": {"strategy": "hybrid", "top_k": 4}
            },
            "results": {
                "composite_score": 0.9123,
                "retrieval_metrics": {"hit_rate": 0.95, "precision": 0.90},
                "answer_metrics": {"faithfulness": 0.90, "answer_relevance": 0.88},
                "metrics_valid": True,
                "status": "completed"
            },
            "status": "completed",
            "trial_number": 1
        }
        self.workspace.save_experiment(exp_data)
        
        # Verify get_experiment
        fetched = self.workspace.get_experiment("exp_roundtrip_test")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["composite_score"], 0.9123)
        self.assertEqual(fetched["results"]["composite_score"], 0.9123)
        self.assertTrue(fetched["results"]["metrics_valid"])

        # Verify list_experiments
        exps = self.workspace.list_experiments()
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0]["experiment_id"], "exp_roundtrip_test")
        self.assertEqual(exps[0]["composite_score"], 0.9123)

    def test_workspace_persistence_nan_safety(self):
        # Even if unhandled NaN is passed into save_experiment, it must be safely sanitized
        data_with_nan = {
            "experiment_id": "exp_nan_test",
            "composite_score": float("nan"),
            "results": {
                "composite_score": float("nan"),
                "avg_latency_ms": float("nan"),
                "metrics_valid": False,
                "status": "failed"
            }
        }
        # Must not raise ValueError: Out of range float values are not JSON compliant
        self.workspace.save_experiment(data_with_nan)
        fetched = self.workspace.get_experiment("exp_nan_test")
        self.assertIsNotNone(fetched)
        self.assertIsNone(fetched["composite_score"])
        self.assertIsNone(fetched["results"]["composite_score"])
        self.assertIsNone(fetched["results"]["avg_latency_ms"])

    def test_deterministic_leaderboard_sorting(self):
        trials = [
            {"experiment_id": "exp_mid", "composite_score": 0.75, "status": "completed", "trial_number": 1, "results": {"composite_score": 0.75, "metrics_valid": True, "status": "completed"}},
            {"experiment_id": "exp_fail", "composite_score": None, "status": "failed", "trial_number": 2, "results": {"composite_score": None, "metrics_valid": False, "status": "failed"}},
            {"experiment_id": "exp_best", "composite_score": 0.94, "status": "completed", "trial_number": 3, "results": {"composite_score": 0.94, "metrics_valid": True, "status": "completed"}},
            {"experiment_id": "exp_low", "composite_score": 0.60, "status": "completed", "trial_number": 4, "results": {"composite_score": 0.60, "metrics_valid": True, "status": "completed"}}
        ]
        for t in trials:
            self.workspace.save_experiment(t)

        sorted_exps = self.workspace.list_experiments()
        self.assertEqual(len(sorted_exps), 4)
        self.assertEqual(sorted_exps[0]["experiment_id"], "exp_best")
        self.assertEqual(sorted_exps[0]["composite_score"], 0.94)
        self.assertEqual(sorted_exps[1]["experiment_id"], "exp_mid")
        self.assertEqual(sorted_exps[1]["composite_score"], 0.75)
        self.assertEqual(sorted_exps[2]["experiment_id"], "exp_low")
        self.assertEqual(sorted_exps[2]["composite_score"], 0.60)
        self.assertEqual(sorted_exps[3]["experiment_id"], "exp_fail")
        self.assertIsNone(sorted_exps[3]["composite_score"])

    def test_report_engine_canonical_score_exports(self):
        reporter = ReportEngine(self.workspace)
        leaderboard = [
            {
                "experiment_id": "exp_best",
                "composite_score": 0.9250,
                "config": {
                    "llm_config": {"provider": "openai", "model_name": "gpt-4o-mini"},
                    "embedding_config": {"model_name": "text-embedding-3-small"},
                    "chunking_config": {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 64},
                    "retriever_config": {"strategy": "hybrid", "distance_metric": "cosine", "top_k": 4}
                },
                "results": {
                    "composite_score": 0.9250,
                    "retrieval_metrics": {"hit_rate": 0.95, "precision": 0.90, "mrr": 0.92},
                    "answer_metrics": {"accuracy": 0.91, "completeness": 0.90, "relevance": 0.94},
                    "metrics_valid": True,
                    "status": "completed"
                }
            }
        ]
        md_file = reporter.export_markdown(leaderboard, "test_summary")
        self.assertTrue(os.path.exists(md_file))
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
            self.assertIn("0.9250", md_content)
            self.assertNotIn("None", md_content)

        csv_file = reporter.export_csv(leaderboard, "test_data")
        self.assertTrue(os.path.exists(csv_file))
        with open(csv_file, "r", encoding="utf-8") as f:
            csv_content = f.read()
            self.assertIn("0.925", csv_content)

        html_file = reporter.export_html(leaderboard, "test_dashboard")
        self.assertTrue(os.path.exists(html_file))
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
            self.assertIn("0.9250", html_content)

    def test_cascading_kb_clear_zero_trace(self):
        """Test that clearing KB removes all artifacts, trials, memory, caches with zero residual traces."""
        # 1. Create simulated KB file
        kb_path = self.workspace.get_kb_path()
        doc_file = kb_path / "sample_doc.txt"
        doc_file.write_text("Test knowledge base content for zero trace isolation.")

        # 2. Create simulated index and chunk cache
        cache_dir = self.workspace.get_cache_path()
        (cache_dir / "chunks_kb1.json").write_text('{"chunks": []}')
        (cache_dir / "index_kb1.json").write_text('{"index": []}')

        # 3. Create simulated dataset
        ds_dir = self.workspace.get_datasets_path()
        (ds_dir / "dataset_kb1.json").write_text('{"items": []}')

        # 4. Create simulated experiment and memory
        self.workspace.save_experiment({
            "experiment_id": "exp_kb1_trial1",
            "kb_checksum": "kb1_chk",
            "metadata": {"kb_checksum": "kb1_chk"},
            "composite_score": 0.85,
            "status": "completed"
        })
        self.workspace.save_optimization_memory("kb1_chk", {"best_so_far": {"composite_score": 0.85}})

        # 5. Verify items exist
        self.assertTrue(doc_file.exists())
        self.assertEqual(len(self.workspace.list_experiments()), 1)
        self.assertIsNotNone(self.workspace.load_optimization_memory("kb1_chk"))

        # 6. Execute Zero-Trace Purge
        counts = self.workspace.clear_kb_artifacts()
        self.assertGreaterEqual(counts["kb_files"], 1)
        self.assertGreaterEqual(counts["experiments"], 1)
        self.assertGreaterEqual(counts["datasets"], 1)

        # 7. Verify everything is cleanly purged
        self.assertFalse(doc_file.exists())
        self.assertEqual(len(self.workspace.list_experiments()), 0)
        self.assertIsNone(self.workspace.load_optimization_memory("kb1_chk"))
        self.assertEqual(len(list(cache_dir.iterdir())), 0)
        self.assertEqual(len(list(ds_dir.iterdir())), 0)

    def test_scoped_experiments_and_cross_kb_isolation(self):
        """Test that experiments and memory are isolated per KB checksum."""
        self.workspace.save_experiment({
            "experiment_id": "exp_kb_A",
            "kb_checksum": "checksum_A",
            "metadata": {"kb_checksum": "checksum_A"},
            "composite_score": 0.90,
            "status": "completed"
        })
        self.workspace.save_experiment({
            "experiment_id": "exp_kb_B",
            "kb_checksum": "checksum_B",
            "metadata": {"kb_checksum": "checksum_B"},
            "composite_score": 0.70,
            "status": "completed"
        })
        self.workspace.save_optimization_memory("checksum_A", {"best_so_far": {"composite_score": 0.90}})
        self.workspace.save_optimization_memory("checksum_B", {"best_so_far": {"composite_score": 0.70}})

        # Query scoped
        exps_a = self.workspace.list_experiments(kb_checksum="checksum_A")
        exps_b = self.workspace.list_experiments(kb_checksum="checksum_B")
        self.assertEqual(len(exps_a), 1)
        self.assertEqual(exps_a[0]["experiment_id"], "exp_kb_A")
        self.assertEqual(len(exps_b), 1)
        self.assertEqual(exps_b[0]["experiment_id"], "exp_kb_B")

        # Clear only KB A
        deleted_count = self.workspace.clear_experiments(kb_checksum="checksum_A")
        self.assertEqual(deleted_count, 1)

        # KB B still exists
        self.assertEqual(len(self.workspace.list_experiments(kb_checksum="checksum_A")), 0)
        self.assertEqual(len(self.workspace.list_experiments(kb_checksum="checksum_B")), 1)
        self.assertIsNone(self.workspace.load_optimization_memory("checksum_A"))
        self.assertIsNotNone(self.workspace.load_optimization_memory("checksum_B"))

    def test_sequential_optimization_memory_and_deduplication(self):
        """Test canonical memory builder, hash consistency, and duplicate detection."""
        from autorag.optimization_engine import OptimizationEngine
        from autorag.connections import OpenAICompatibleClient, EmbeddingClient
        from autorag.types import OptimizationSpec, LLMConfig, EmbeddingConfig

        mock_llm = OpenAICompatibleClient(LLMConfig(provider="mock", model_name="mock-llm"))
        mock_embed = EmbeddingClient(EmbeddingConfig(provider="mock", model_name="mock-embed"))
        opt_engine = OptimizationEngine(self.workspace, mock_llm, mock_embed)
        base_config = PipelineConfig(
            experiment_name="test_pipeline",
            llm_config=LLMConfig(provider="mock", model_name="mock-llm"),
            embedding_config=EmbeddingConfig(provider="mock", model_name="mock-embed"),
            chunking_config=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=64),
            retriever_config=RetrieverConfig(strategy="hybrid", distance_metric="cosine", top_k=4)
        )

        history = [
            {
                "trial_number": 1,
                "config": base_config.__dict__,
                "composite_score": 0.80,
                "status": "completed",
                "results": {
                    "composite_score": 0.80,
                    "metrics_valid": True,
                    "retrieval_metrics": {"hit_rate": 0.85},
                    "answer_metrics": {"faithfulness": 0.75},
                    "avg_latency_ms": 120.0,
                    "status": "completed"
                }
            }
        ]

        spec = OptimizationSpec(strategy="llm_guided", max_trials=3)
        memory = opt_engine.build_optimization_memory(history, spec, base_config, "test_kb_chk")

        self.assertEqual(memory["kb_checksum"], "test_kb_chk")
        self.assertEqual(len(memory["evaluated_trials"]), 1)
        self.assertEqual(memory["best_so_far"]["composite_score"], 0.80)

        # Verify duplicate detection
        is_valid_dup, reason = opt_engine.validate_config(base_config, history)
        self.assertFalse(is_valid_dup)
        self.assertIn("previously evaluated", reason)

        # Mutate to valid candidate
        new_cand = opt_engine._mutate_config_adaptive(base_config, history, memory)
        is_valid_new, _ = opt_engine.validate_config(new_cand, history)
        self.assertTrue(is_valid_new)


if __name__ == "__main__":
    unittest.main()
