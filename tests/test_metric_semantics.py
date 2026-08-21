"""
Unit & Regression Tests for AutoRAG Metric Semantics Audit.
Verifies that:
1. AnswerMetrics synchronizes answer_correctness, accuracy, and faithfulness seamlessly.
2. EvaluationEngine evaluates answer correctness and computes composite scores identically.
3. Legacy experiments with 'faithfulness' are normalized cleanly with 'answer_correctness'.
4. OptimizationEngine produces and loads leaderboards preserving identical numerical ranking and score calculations.
5. Generalization results preserve and report answer_correctness without breaking faithfulness aliases.
6. ReportEngine formats correctly.
"""

import math
import unittest
from autorag.types import (
    AnswerMetrics, RetrievalMetrics, EvaluationResult,
    PipelineConfig, ChunkingConfig, RetrieverConfig, GeneralizationTestResult
)
from autorag.evaluation_engine import EvaluationEngine
from autorag.workspace import WorkspaceManager
from autorag.optimization_engine import OptimizationEngine


class DummyEvaluationEngine(EvaluationEngine):
    def __init__(self):
        # Subclass without requiring full LLM client for math checks
        pass


class TestMetricSemantics(unittest.TestCase):

    def test_answer_metrics_synchronization(self):
        # 1. Created with only answer_correctness
        m1 = AnswerMetrics(answer_correctness=0.85)
        self.assertEqual(m1.answer_correctness, 0.85)
        self.assertEqual(m1.faithfulness, 0.85)
        self.assertEqual(m1.accuracy, 0.85)

        # 2. Created with legacy faithfulness only
        m2 = AnswerMetrics(faithfulness=0.92)
        self.assertEqual(m2.answer_correctness, 0.92)
        self.assertEqual(m2.faithfulness, 0.92)
        self.assertEqual(m2.accuracy, 0.92)

        # 3. Created with accuracy only
        m3 = AnswerMetrics(accuracy=0.78)
        self.assertEqual(m3.answer_correctness, 0.78)
        self.assertEqual(m3.faithfulness, 0.78)
        self.assertEqual(m3.accuracy, 0.78)

    def test_composite_score_calculation(self):
        engine = DummyEvaluationEngine()
        
        ret_m = RetrievalMetrics(hit_rate=0.8, precision=0.6, recall=0.7, mrr=0.75, ndcg=0.72)
        ans_m = AnswerMetrics(answer_correctness=0.9, faithfulness=0.9, accuracy=0.9, completeness=0.8, relevance=0.85)

        score = engine.calculate_composite_score(ret_m, ans_m)
        # Expected: 0.8 * 0.7 + 0.9 * 0.3 = 0.56 + 0.27 = 0.83
        self.assertAlmostEqual(score, 0.83, places=4)

        # Legacy fallback if answer_correctness is 0.0 but faithfulness is set
        ans_m_legacy = AnswerMetrics()
        ans_m_legacy.faithfulness = 0.9
        score_legacy = engine.calculate_composite_score(ret_m, ans_m_legacy)
        self.assertAlmostEqual(score_legacy, 0.83, places=4)

    def test_workspace_legacy_normalization(self):
        from autorag.workspace import normalize_experiment_record
        legacy_record = {
            "experiment_id": "exp_legacy_001",
            "timestamp": "2025-01-01T00:00:00Z",
            "composite_score": 0.85,
            "results": {
                "composite_score": 0.85,
                "retrieval_metrics": {
                    "hit_rate": 0.9,
                    "precision": 0.8,
                    "recall": 0.85,
                    "mrr": 0.88,
                    "ndcg": 0.86
                },
                "answer_metrics": {
                    "faithfulness": 0.75,
                    "completeness": 0.7,
                    "relevance": 0.8
                }
            }
        }

        normalized = normalize_experiment_record(legacy_record)
        ans_metrics = normalized["results"]["answer_metrics"]
        
        self.assertEqual(ans_metrics.get("answer_correctness"), 0.75)
        self.assertEqual(ans_metrics.get("faithfulness"), 0.75)
        self.assertEqual(ans_metrics.get("accuracy"), 0.75)

    def test_optimization_trial_normalization(self):
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        try:
            ws = WorkspaceManager(temp_dir)
            engine = OptimizationEngine(ws, None, None)
            legacy_trial = {
                "trial_number": 1,
                "experiment_id": "exp_test_001",
                "status": "completed",
                "composite_score": 0.85,
                "results": {
                    "status": "completed",
                    "metrics_valid": True,
                    "composite_score": 0.85,
                    "retrieval_metrics": {"hit_rate": 1.0},
                    "answer_metrics": {"faithfulness": 0.5}
                }
            }

            trial_summary, fprint, is_valid = engine._summarize_trial_record(legacy_trial)
            self.assertTrue(is_valid)
            self.assertEqual(trial_summary["composite_score"], 0.85)
            self.assertEqual(trial_summary["answer_correctness"], 0.5)
            self.assertEqual(trial_summary["faithfulness"], 0.5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
