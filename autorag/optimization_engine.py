"""
Optimization Engine Subsystem for AutoRAG Platform.
Provides Grid Search, Random Search, and LLM-Guided Optimization.
Includes strict validation, constraint checking, duplicate detection, trial execution, and recommendation logic.
"""

import hashlib
import json
import math
import random
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .types import (
    PipelineConfig, ChunkingConfig, RetrieverConfig, OptimizationSpec, EvaluationResult, Experiment,
    RetrievalMetrics, AnswerMetrics
)
from .connections import OpenAICompatibleClient, EmbeddingClient
from .knowledge_base import Document
from .index_builder import IndexBuilder
from .retrieval_engine import RetrievalEngine
from .generation_engine import GenerationEngine
from .prompt_builder import PromptBuilder
from .evaluation_engine import EvaluationEngine
from .dataset_builder import Dataset
from .workspace import WorkspaceManager


def compute_config_hash(config: Any) -> str:
    """Generate unique canonical SHA256 signature for pipeline configuration."""
    if isinstance(config, PipelineConfig):
        chunk_strategy = config.chunking_config.strategy
        chunk_size = int(config.chunking_config.chunk_size)
        chunk_overlap = int(config.chunking_config.chunk_overlap)
        ret_strat = config.retriever_config.strategy
        dist_metric = config.retriever_config.distance_metric
        top_k = int(config.retriever_config.top_k)
        hybrid_alpha = round(float(config.retriever_config.hybrid_alpha), 3)
        sys_prompt = config.system_prompt or ""
    elif isinstance(config, dict):
        c_chunk = config.get("chunking_config") or {}
        c_ret = config.get("retriever_config") or {}
        chunk_strategy = c_chunk.get("strategy", "recursive")
        chunk_size = int(c_chunk.get("chunk_size", 512))
        chunk_overlap = int(c_chunk.get("chunk_overlap", 64))
        ret_strat = c_ret.get("strategy", "hybrid")
        dist_metric = c_ret.get("distance_metric", "cosine")
        top_k = int(c_ret.get("top_k", 4))
        hybrid_alpha = round(float(c_ret.get("hybrid_alpha", 0.7)), 3)
        sys_prompt = config.get("system_prompt", "")
    else:
        return hashlib.sha256(str(config).encode("utf-8")).hexdigest()

    raw = json.dumps({
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "retriever_strategy": ret_strat,
        "distance_metric": dist_metric,
        "top_k": top_k,
        "hybrid_alpha": hybrid_alpha,
        "system_prompt": sys_prompt
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_optimization_memory(
    workspace: Optional[WorkspaceManager] = None,
    kb_checksum: str = "default_kb",
    dataset_id: Optional[str] = None,
    current_run_id: str = "run_default",
    current_run_trials: Optional[List[Dict[str, Any]]] = None,
    use_previous_history: bool = True,
    learn_from_generalization_test: bool = False,
    max_historical_trials: int = 8
) -> Dict[str, Any]:
    """Module-level memory builder for direct testing and orchestration."""
    historical_trials: List[Dict[str, Any]] = []
    if workspace and use_previous_history:
        all_exps = workspace.list_experiments(kb_checksum=kb_checksum)
        for exp in all_exps:
            if exp.get("optimization_run_id") == current_run_id:
                continue
            h_chk = exp.get("kb_checksum") or exp.get("metadata", {}).get("kb_checksum")
            if h_chk and kb_checksum and h_chk != kb_checksum:
                continue
            h_ds = exp.get("dataset_id")
            if dataset_id and h_ds and h_ds != dataset_id:
                continue
            historical_trials.append(exp)

    engine = OptimizationEngine(workspace or WorkspaceManager(tempfile.gettempdir() if 'tempfile' in globals() else "/tmp"), None, None)  # type: ignore
    return engine.build_optimization_memory(
        current_run_history=current_run_trials or [],
        historical_history=historical_trials,
        use_previous_history=use_previous_history,
        learn_from_generalization_test=learn_from_generalization_test,
        kb_checksum=kb_checksum,
        dataset_id=dataset_id,
        max_historical_trials=max_historical_trials
    )


class OptimizationEngine:
    """Manages hyperparameter search across RAG pipeline configurations."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        llm_client: OpenAICompatibleClient,
        embedding_client: EmbeddingClient
    ):
        self.workspace = workspace
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.eval_engine = EvaluationEngine(llm_client)

    def _hash_config(self, config: PipelineConfig) -> str:
        """Generate unique SHA256 signature for pipeline configuration."""
        raw = json.dumps({
            "chunk_strategy": config.chunking_config.strategy,
            "chunk_size": int(config.chunking_config.chunk_size),
            "chunk_overlap": int(config.chunking_config.chunk_overlap),
            "retriever_strategy": config.retriever_config.strategy,
            "distance_metric": config.retriever_config.distance_metric,
            "top_k": int(config.retriever_config.top_k),
            "hybrid_alpha": round(float(config.retriever_config.hybrid_alpha), 3),
            "system_prompt": config.system_prompt or ""
        }, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def validate_config(
        self,
        config: PipelineConfig,
        current_run_history: Optional[List[Dict[str, Any]]] = None,
        historical_history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str]:
        """
        Validate proposed config against constraints and duplicate detection.
        If historical_history is provided (when historical learning is ON), duplicate
        check rejects configurations previously tested in current OR historical runs.
        When historical_history is None/empty (when historical learning is OFF), duplicate
        check ONLY rejects configurations tested within the current active run.
        """
        if current_run_history is None:
            current_run_history = []
        if historical_history is None:
            historical_history = []
        c = config.chunking_config
        r = config.retriever_config

        # 1. Parameter Constraints
        if c.chunk_size < 64 or c.chunk_size > 4096:
            return False, f"Invalid chunk_size {c.chunk_size}. Must be between 64 and 4096."

        if c.chunk_overlap >= c.chunk_size:
            return False, f"Invalid chunk_overlap {c.chunk_overlap}. Must be strictly less than chunk_size {c.chunk_size}."

        if c.chunk_overlap < 0:
            return False, f"Invalid chunk_overlap {c.chunk_overlap}. Must be >= 0."

        if r.top_k < 1 or r.top_k > 20:
            return False, f"Invalid top_k {r.top_k}. Must be between 1 and 20."

        if r.hybrid_alpha < 0.0 or r.hybrid_alpha > 1.0:
            return False, f"Invalid hybrid_alpha {r.hybrid_alpha}. Must be between 0.0 and 1.0."

        valid_chunk_strats = {"recursive", "fixed", "paragraph", "semantic"}
        if c.strategy not in valid_chunk_strats:
            return False, f"Invalid chunk_strategy '{c.strategy}'. Must be one of {valid_chunk_strats}."

        valid_ret_strats = {"dense", "sparse", "hybrid"}
        if r.strategy not in valid_ret_strats:
            return False, f"Invalid retriever_strategy '{r.strategy}'. Must be one of {valid_ret_strats}."

        valid_dist_metrics = {"cosine", "dot", "euclidean"}
        if r.distance_metric not in valid_dist_metrics:
            return False, f"Invalid distance_metric '{r.distance_metric}'. Must be one of {valid_dist_metrics}."

        # 2. Duplicate Detection against current run
        config_hash = self._hash_config(config)
        for h in current_run_history:
            h_config = h.get("config", {})
            if hasattr(h_config, "__dict__"):
                h_config = h_config.__dict__
            if isinstance(h_config, dict):
                h_c = h_config.get("chunking_config", {})
                if hasattr(h_c, "__dict__"):
                    h_c = h_c.__dict__
                h_r = h_config.get("retriever_config", {})
                if hasattr(h_r, "__dict__"):
                    h_r = h_r.__dict__

                h_raw = json.dumps({
                    "chunk_strategy": h_c.get("strategy") if isinstance(h_c, dict) else getattr(h_c, "strategy", "recursive"),
                    "chunk_size": int(h_c.get("chunk_size", 512) if isinstance(h_c, dict) else getattr(h_c, "chunk_size", 512)),
                    "chunk_overlap": int(h_c.get("chunk_overlap", 64) if isinstance(h_c, dict) else getattr(h_c, "chunk_overlap", 64)),
                    "retriever_strategy": h_r.get("strategy") if isinstance(h_r, dict) else getattr(h_r, "strategy", "hybrid"),
                    "distance_metric": h_r.get("distance_metric") if isinstance(h_r, dict) else getattr(h_r, "distance_metric", "cosine"),
                    "top_k": int(h_r.get("top_k", 4) if isinstance(h_r, dict) else getattr(h_r, "top_k", 4)),
                    "hybrid_alpha": round(float(h_r.get("hybrid_alpha", 0.7) if isinstance(h_r, dict) else getattr(h_r, "hybrid_alpha", 0.7)), 3),
                    "system_prompt": (h_config.get("system_prompt") if isinstance(h_config, dict) else getattr(h_config, "system_prompt", "")) or ""
                }, sort_keys=True)
                if hashlib.sha256(h_raw.encode("utf-8")).hexdigest() == config_hash:
                    return False, f"Duplicate configuration detected (previously evaluated in current run trial {h.get('trial_number', '?')})."

        # 3. Duplicate Detection against historical runs (only when historical history provided)
        if historical_history:
            for h in historical_history:
                h_config = h.get("config", {})
                if hasattr(h_config, "__dict__"):
                    h_config = h_config.__dict__
                if isinstance(h_config, dict):
                    h_c = h_config.get("chunking_config", {})
                    if hasattr(h_c, "__dict__"):
                        h_c = h_c.__dict__
                    h_r = h_config.get("retriever_config", {})
                    if hasattr(h_r, "__dict__"):
                        h_r = h_r.__dict__

                    h_raw = json.dumps({
                        "chunk_strategy": h_c.get("strategy") if isinstance(h_c, dict) else getattr(h_c, "strategy", "recursive"),
                        "chunk_size": int(h_c.get("chunk_size", 512) if isinstance(h_c, dict) else getattr(h_c, "chunk_size", 512)),
                        "chunk_overlap": int(h_c.get("chunk_overlap", 64) if isinstance(h_c, dict) else getattr(h_c, "chunk_overlap", 64)),
                        "retriever_strategy": h_r.get("strategy") if isinstance(h_r, dict) else getattr(h_r, "strategy", "hybrid"),
                        "distance_metric": h_r.get("distance_metric") if isinstance(h_r, dict) else getattr(h_r, "distance_metric", "cosine"),
                        "top_k": int(h_r.get("top_k", 4) if isinstance(h_r, dict) else getattr(h_r, "top_k", 4)),
                        "hybrid_alpha": round(float(h_r.get("hybrid_alpha", 0.7) if isinstance(h_r, dict) else getattr(h_r, "hybrid_alpha", 0.7)), 3),
                        "system_prompt": (h_config.get("system_prompt") if isinstance(h_config, dict) else getattr(h_config, "system_prompt", "")) or ""
                    }, sort_keys=True)
                    if hashlib.sha256(h_raw.encode("utf-8")).hexdigest() == config_hash:
                        return False, f"Duplicate configuration detected (previously evaluated in historical run trial {h.get('trial_number', '?')})."

        return True, "Valid configuration."

    def _summarize_trial_record(self, h: Dict[str, Any]) -> Tuple[Dict[str, Any], str, bool]:
        """Extract canonical structured trial summary, fingerprint, and valid score flag."""
        trial_num = h.get("trial_number", 0)
        run_id = h.get("optimization_run_id") or h.get("metadata", {}).get("optimization_run_id", "")
        cfg = h.get("config", {})
        if hasattr(cfg, "__dict__"):
            cfg = cfg.__dict__
        c_cfg = cfg.get("chunking_config", {}) if isinstance(cfg, dict) else {}
        if hasattr(c_cfg, "__dict__"):
            c_cfg = c_cfg.__dict__
        r_cfg = cfg.get("retriever_config", {}) if isinstance(cfg, dict) else {}
        if hasattr(r_cfg, "__dict__"):
            r_cfg = r_cfg.__dict__

        res = h.get("results", {}) or {}
        if hasattr(res, "__dict__"):
            res = res.__dict__
        status = res.get("status", h.get("status", "unknown")) if isinstance(res, dict) else "unknown"
        metrics_valid = res.get("metrics_valid", True) if isinstance(res, dict) else True
        score = res.get("composite_score") if isinstance(res, dict) else None
        if score is None:
            score = h.get("composite_score")

        if score is not None and not (isinstance(score, (int, float)) and math.isfinite(score)):
            score = None
            metrics_valid = False

        c_strat = c_cfg.get("strategy") if isinstance(c_cfg, dict) else getattr(c_cfg, "strategy", "recursive")
        c_size = int(c_cfg.get("chunk_size", 512) if isinstance(c_cfg, dict) else getattr(c_cfg, "chunk_size", 512))
        c_overlap = int(c_cfg.get("chunk_overlap", 64) if isinstance(c_cfg, dict) else getattr(c_cfg, "chunk_overlap", 64))
        r_strat = r_cfg.get("strategy") if isinstance(r_cfg, dict) else getattr(r_cfg, "strategy", "hybrid")
        r_dist = r_cfg.get("distance_metric") if isinstance(r_cfg, dict) else getattr(r_cfg, "distance_metric", "cosine")
        r_topk = int(r_cfg.get("top_k", 4) if isinstance(r_cfg, dict) else getattr(r_cfg, "top_k", 4))
        r_alpha = round(float(r_cfg.get("hybrid_alpha", 0.7) if isinstance(r_cfg, dict) else getattr(r_cfg, "hybrid_alpha", 0.7)), 3)
        sys_prompt = (cfg.get("system_prompt") if isinstance(cfg, dict) else getattr(cfg, "system_prompt", "")) or ""

        h_raw = json.dumps({
            "chunk_strategy": c_strat,
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
            "retriever_strategy": r_strat,
            "distance_metric": r_dist,
            "top_k": r_topk,
            "hybrid_alpha": r_alpha,
            "system_prompt": sys_prompt
        }, sort_keys=True)
        fprint = hashlib.sha256(h_raw.encode("utf-8")).hexdigest()

        ret_metrics = res.get("retrieval_metrics") or {} if isinstance(res, dict) else {}
        if hasattr(ret_metrics, "__dict__"):
            ret_metrics = ret_metrics.__dict__
        ans_metrics = res.get("answer_metrics") or {} if isinstance(res, dict) else {}
        if hasattr(ans_metrics, "__dict__"):
            ans_metrics = ans_metrics.__dict__

        is_valid_completed = (status == "completed" and metrics_valid and score is not None)

        gen_test = h.get("generalization_test")
        gen_summary = None
        if gen_test and isinstance(gen_test, dict) and gen_test.get("status") == "completed":
            ans_m = gen_test.get("answer_metrics") or {}
            ans_val = ans_m.get("answer_correctness") if isinstance(ans_m, dict) else getattr(ans_m, "answer_correctness", None)
            if ans_val is None:
                ans_val = ans_m.get("faithfulness") if isinstance(ans_m, dict) else getattr(ans_m, "faithfulness", None)
            if ans_val is None:
                ans_val = ans_m.get("accuracy") if isinstance(ans_m, dict) else getattr(ans_m, "accuracy", None)
            gen_summary = {
                "generalization_composite_score": gen_test.get("generalization_composite_score"),
                "score_delta": gen_test.get("score_delta"),
                "hit_rate": (gen_test.get("retrieval_metrics") or {}).get("hit_rate") if isinstance(gen_test.get("retrieval_metrics"), dict) else getattr(gen_test.get("retrieval_metrics"), "hit_rate", None),
                "answer_correctness": ans_val,
                "faithfulness": ans_val,
                "test_size": gen_test.get("test_size", 5)
            }

        strat = h.get("metadata", {}).get("optimization_strategy") or h.get("optimization_strategy") or "llm_guided"

        ans_val = ans_metrics.get("answer_correctness") if isinstance(ans_metrics, dict) else getattr(ans_metrics, "answer_correctness", None)
        if ans_val is None:
            ans_val = ans_metrics.get("accuracy") if isinstance(ans_metrics, dict) else getattr(ans_metrics, "accuracy", None)
        if ans_val is None:
            ans_val = ans_metrics.get("faithfulness") if isinstance(ans_metrics, dict) else getattr(ans_metrics, "faithfulness", None)

        trial_summary = {
            "experiment_id": h.get("experiment_id") or (res.get("experiment_id") if isinstance(res, dict) else "") or "",
            "trial_number": trial_num,
            "optimization_run_id": run_id,
            "optimization_strategy": strat,
            "status": status,
            "metrics_valid": is_valid_completed,
            "composite_score": score if is_valid_completed else None,
            "chunk_strategy": c_strat,
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
            "retriever_strategy": r_strat,
            "distance_metric": r_dist,
            "top_k": r_topk,
            "hybrid_alpha": r_alpha,
            "hit_rate": ret_metrics.get("hit_rate") if isinstance(ret_metrics, dict) else getattr(ret_metrics, "hit_rate", None),
            "precision": ret_metrics.get("precision") if isinstance(ret_metrics, dict) else getattr(ret_metrics, "precision", None),
            "answer_correctness": ans_val,
            "faithfulness": ans_val,
            "avg_latency_ms": res.get("avg_latency_ms") if isinstance(res, dict) else getattr(res, "avg_latency_ms", None),
            "config_fingerprint": fprint[:12],
            "generalization_test": gen_summary,
            "config": cfg if isinstance(cfg, dict) else (cfg.__dict__ if hasattr(cfg, "__dict__") else {})
        }

        if not is_valid_completed:
            trial_summary["failure_reason"] = (
                (res.get("failure_reason") if isinstance(res, dict) else getattr(res, "failure_reason", None))
                or h.get("failure_reason")
                or "Execution failure or invalid metrics"
            )

        return trial_summary, fprint, is_valid_completed

    def _compute_empirical_trends(self, trials: List[Dict[str, Any]]) -> List[str]:
        """Compute human-readable empirical correlations from valid evaluated trials."""
        trends = []
        valid_trials = [t for t in trials if t.get("composite_score") is not None and isinstance(t.get("composite_score"), (int, float))]
        if len(valid_trials) >= 2:
            # Chunk size correlation
            sizes = sorted(list(set(t["chunk_size"] for t in valid_trials if t.get("chunk_size"))))
            if len(sizes) > 1:
                size_scores = {}
                for s in sizes:
                    s_trials = [t for t in valid_trials if t["chunk_size"] == s and t["composite_score"] is not None]
                    if s_trials:
                        size_scores[s] = sum(t["composite_score"] for t in s_trials) / len(s_trials)
                if len(size_scores) > 1:
                    best_size = max(size_scores.items(), key=lambda x: x[1])[0]
                    trends.append(f"Chunk size {best_size} produced the highest average composite score ({size_scores[best_size]:.3f}).")

            # Retriever strategy correlation
            ret_strats = list(set(t["retriever_strategy"] for t in valid_trials if t.get("retriever_strategy")))
            if len(ret_strats) > 1:
                strat_scores = {}
                for rs in ret_strats:
                    rs_trials = [t for t in valid_trials if t["retriever_strategy"] == rs and t["composite_score"] is not None]
                    if rs_trials:
                        strat_scores[rs] = sum(t["composite_score"] for t in rs_trials) / len(rs_trials)
                if len(strat_scores) > 1:
                    best_strat = max(strat_scores.items(), key=lambda x: x[1])[0]
                    trends.append(f"Retriever strategy '{best_strat}' currently leads with average score {strat_scores[best_strat]:.3f}.")

            # Top K correlation
            top_ks = sorted(list(set(t["top_k"] for t in valid_trials if t.get("top_k"))))
            if len(top_ks) > 1:
                top_k_hits = {}
                for tk in top_ks:
                    tk_trials = [t for t in valid_trials if t["top_k"] == tk and t.get("hit_rate") is not None]
                    if tk_trials:
                        top_k_hits[tk] = sum(t["hit_rate"] for t in tk_trials) / len(tk_trials)
                if len(top_k_hits) > 1:
                    best_k = max(top_k_hits.items(), key=lambda x: x[1])[0]
                    trends.append(f"Top-K={best_k} produced highest hit rate ({top_k_hits[best_k]:.3f}).")

        return trends

    def build_optimization_memory(
        self,
        current_run_history: List[Dict[str, Any]],
        historical_history: Any = None,
        use_previous_history: bool = True,
        learn_from_generalization_test: bool = False,
        spec: Optional[OptimizationSpec] = None,
        base_config: Optional[PipelineConfig] = None,
        kb_checksum: str = "default_kb",
        dataset_id: Optional[str] = None,
        max_historical_trials: int = 8
    ) -> Dict[str, Any]:
        """
        Build centralized, canonical optimization memory split cleanly into:
        1. Historical Optimization Memory (prior compatible trials, bounded, filtered)
        2. Current Run Memory (trials evaluated in active session)
        
        When use_previous_history is False, Historical Optimization Memory is strictly empty
        and excluded from the optimization context.
        """
        # Support legacy positional signature: build_optimization_memory(history, spec, base_config, kb_checksum)
        if isinstance(historical_history, OptimizationSpec):
            spec_obj = historical_history
            base_cfg_obj = use_previous_history if isinstance(use_previous_history, PipelineConfig) else None
            chk_obj = learn_from_generalization_test if isinstance(learn_from_generalization_test, str) else (str(spec) if spec and isinstance(spec, str) else kb_checksum)
            spec = spec_obj
            base_config = base_cfg_obj
            kb_checksum = chk_obj
            use_previous_history = getattr(spec, "use_previous_history", True)
            learn_from_generalization_test = getattr(spec, "learn_from_generalization_test", False) or getattr(spec, "learnFromGeneralizationTest", False)
            historical_history = []
        elif historical_history is None:
            historical_history = []

        if spec is not None:
            if hasattr(spec, "learn_from_generalization_test"):
                learn_from_generalization_test = bool(spec.learn_from_generalization_test)
            elif hasattr(spec, "learnFromGeneralizationTest"):
                learn_from_generalization_test = bool(spec.learnFromGeneralizationTest)

        # 1. Process Current Run Trials
        current_evaluated = []
        current_failed = []
        current_hashes = []
        current_best = None
        current_best_score = -1.0

        for h in current_run_history:
            trial_summary, fprint, is_valid = self._summarize_trial_record(h)
            current_hashes.append(fprint)
            if is_valid:
                current_evaluated.append(trial_summary)
                if trial_summary["composite_score"] > current_best_score:
                    current_best_score = trial_summary["composite_score"]
                    current_best = trial_summary
            else:
                current_failed.append(trial_summary)

        current_trends = self._compute_empirical_trends(current_evaluated)

        # 2. Process Historical Trials (Scoped by KB & Dataset, Bounded)
        historical_hashes = []
        historical_best = None
        historical_best_score = -1.0
        top_historical = []
        recent_historical = []
        hist_trends = []
        failed_hist_trials = []
        valid_hist_trials = []

        if use_previous_history and historical_history:
            for h in historical_history:
                # Check KB checksum scope
                h_chk = h.get("kb_checksum") or h.get("metadata", {}).get("kb_checksum")
                if h_chk and kb_checksum and h_chk != kb_checksum:
                    continue  # Cross-KB isolation
                
                # Check Dataset scope if specified
                h_ds = h.get("dataset_id")
                if dataset_id and h_ds and h_ds != dataset_id:
                    continue  # Dataset lineage isolation

                trial_summary, fprint, is_valid = self._summarize_trial_record(h)
                historical_hashes.append(fprint)
                if is_valid:
                    valid_hist_trials.append(trial_summary)
                    if trial_summary["composite_score"] > historical_best_score:
                        historical_best_score = trial_summary["composite_score"]
                        historical_best = trial_summary
                else:
                    failed_hist_trials.append(trial_summary)

            hist_trends = self._compute_empirical_trends(valid_hist_trials)
            top_historical = sorted(valid_hist_trials, key=lambda x: x.get("composite_score") or 0.0, reverse=True)
            recent_historical = valid_hist_trials[-5:]

        all_retained_hist = (top_historical + failed_hist_trials)[:max_historical_trials] if use_previous_history else []

        historical_memory_dict: Dict[str, Any] = {
            "enabled": (len(valid_hist_trials) > 0 or len(failed_hist_trials) > 0) if use_previous_history else False,
            "historical_trial_count": len(all_retained_hist),
            "best_historical_trial": historical_best if use_previous_history else None,
            "top_historical_trials": top_historical[:max_historical_trials] if use_previous_history else [],
            "recent_historical_trials": recent_historical if use_previous_history else [],
            "historical_trends": hist_trends if use_previous_history else [],
            "historical_failed_trials": (failed_hist_trials[-3:]) if use_previous_history else [],
            "historical_fingerprints": historical_hashes if use_previous_history else []
        }

        # 3. Compile high-level Generalization Validation Summaries (NO raw questions)
        generalization_insights = []
        all_trials_pool = current_evaluated + (valid_hist_trials if use_previous_history else [])
        for t in all_trials_pool:
            if t.get("optimization_strategy") in ["grid", "random"]:
                continue
            gt = t.get("generalization_test")
            if gt and isinstance(gt, dict) and gt.get("generalization_composite_score") is not None:
                g_score = gt.get("generalization_composite_score")
                g_delta = gt.get("score_delta")
                delta_str = f"{g_delta:+.3f}" if g_delta is not None else "N/A"
                generalization_insights.append({
                    "trial_number": t.get("trial_number"),
                    "experiment_id": t.get("experiment_id"),
                    "chunk_strategy": t.get("chunk_strategy"),
                    "chunk_size": t.get("chunk_size"),
                    "retriever_strategy": t.get("retriever_strategy"),
                    "optimization_composite_score": t.get("composite_score"),
                    "generalization_composite_score": g_score,
                    "generalization_composite": g_score,
                    "score_delta": g_delta,
                    "generalization_delta": delta_str,
                    "holdout_hit_rate": gt.get("hit_rate"),
                    "holdout_answer_correctness": gt.get("answer_correctness", gt.get("faithfulness")),
                    "holdout_faithfulness": gt.get("faithfulness", gt.get("answer_correctness"))
                })

        # 4. Overall composite best so far
        overall_best = current_best
        if overall_best is None and use_previous_history:
            overall_best = historical_best

        # Combine trends
        combined_trends = current_trends
        if not combined_trends and use_previous_history and historical_memory_dict.get("historical_trends"):
            combined_trends = historical_memory_dict["historical_trends"]
        if not combined_trends:
            combined_trends = ["Establishing baseline parameters in initial trials."]

        # Combine recent trials for prompt display
        combined_recent = current_evaluated[-5:] if current_evaluated else (
            historical_memory_dict.get("recent_historical_trials", []) if use_previous_history else []
        )

        all_hashes = current_hashes + (historical_hashes if use_previous_history else [])

        memory = {
            "kb_checksum": kb_checksum,
            "dataset_id": dataset_id,
            "use_previous_history": bool(use_previous_history),
            "used_previous_history": bool(use_previous_history),
            "learn_from_generalization_test": bool(learn_from_generalization_test),
            "generalization_memory": generalization_insights if learn_from_generalization_test else [],
            "generalization_insights": generalization_insights if learn_from_generalization_test else [],
            "target_objective": "Maximize Composite Score = 0.7 * Hit Rate + 0.3 * Answer Correctness, while minimizing latency.",
            "search_space_constraints": {
                "chunk_strategy": ["recursive", "fixed", "paragraph", "semantic"],
                "chunk_size": [128, 256, 384, 512, 768, 1024, 1536, 2048],
                "chunk_overlap": [0, 32, 64, 128, 256],
                "retriever_strategy": ["dense", "sparse", "hybrid"],
                "distance_metric": ["cosine", "dot", "euclidean"],
                "top_k": [2, 3, 4, 6, 8, 10],
                "hybrid_alpha": "0.0 (pure sparse/BM25) to 1.0 (pure dense)"
            },
            "historical_memory": all_retained_hist if use_previous_history else [],
            "historical_memory_dict": historical_memory_dict,
            "historical_memory_count": len(all_retained_hist) if use_previous_history else 0,
            "best_historical_score": (historical_best.get("composite_score") if historical_best else None) if use_previous_history else None,
            "best_historical_trial_id": (historical_best.get("experiment_id") if historical_best else None) if use_previous_history else None,
            "current_run_memory": current_evaluated,
            "current_run_memory_dict": {
                "current_trial_count": len(current_run_history),
                "best_current_trial": current_best,
                "recent_trials": current_evaluated[-5:],
                "current_trends": current_trends,
                "current_failed_trials": current_failed[-3:],
                "current_fingerprints": current_hashes
            },
            "best_current_score": (current_best.get("composite_score") if current_best else None),
            # Top-level legacy keys for direct compatibility
            "best_so_far": overall_best,
            "evaluated_trials": current_evaluated + (historical_memory_dict.get("top_historical_trials", []) if use_previous_history else []),
            "recent_trials": combined_recent,
            "observed_trends": combined_trends,
            "failed_trials": current_failed + (historical_memory_dict.get("historical_failed_trials", []) if use_previous_history else []),
            "total_evaluated_count": len(current_run_history) + (historical_memory_dict.get("historical_trial_count", 0) if use_previous_history else 0),
            "evaluated_fingerprints": all_hashes
        }
        return memory

    def propose_candidate_llm(
        self,
        memory_or_config: Any,
        base_config_or_history: Any = None,
        current_run_history: Optional[List[Dict[str, Any]]] = None,
        historical_history: Optional[List[Dict[str, Any]]] = None,
        memory_ctx: Optional[Dict[str, Any]] = None
    ) -> PipelineConfig:
        """
        LLM proposes EXACTLY ONE next candidate hyperparameter configuration
        strictly conditioned on the sequential optimization memory of evaluated trials.
        When use_previous_history is False, the prompt contains ZERO historical trials.
        """
        if isinstance(memory_or_config, PipelineConfig):
            base_config = memory_or_config
            curr_history = base_config_or_history if isinstance(base_config_or_history, list) else []
            memory = memory_ctx or self.build_optimization_memory(curr_history, historical_history=historical_history, use_previous_history=bool(historical_history))
        else:
            memory = memory_or_config
            base_config = base_config_or_history or PipelineConfig(experiment_name="base")
            curr_history = current_run_history or []

        hist_mem = memory.get("historical_memory_dict") or (memory.get("historical_memory") if isinstance(memory.get("historical_memory"), dict) else {})
        curr_mem = memory.get("current_run_memory_dict") or (memory.get("current_run_memory") if isinstance(memory.get("current_run_memory"), dict) else {})
        use_prev = memory.get("use_previous_history", True) and hist_mem.get("enabled", False)

        hist_section = ""
        if use_prev:
            hist_section = (
                f"=== HISTORICAL OPTIMIZATION MEMORY (PAST RUNS): ON ===\n"
                f"Compatible Historical Trials Retained: {hist_mem.get('historical_trial_count', 0)}\n"
                f"Best Historical Configuration:\n{json.dumps(hist_mem.get('best_historical_trial'), indent=2)}\n\n"
                f"Top Historical Configurations:\n{json.dumps(hist_mem.get('top_historical_trials'), indent=2)}\n\n"
                f"Empirical Historical Trends:\n{json.dumps(hist_mem.get('historical_trends'), indent=2)}\n\n"
                f"Historical Failed / Suboptimal Regions to Avoid:\n{json.dumps(hist_mem.get('historical_failed_trials'), indent=2) if hist_mem.get('historical_failed_trials') else 'None.'}\n\n"
            )
        else:
            hist_section = (
                f"=== HISTORICAL OPTIMIZATION MEMORY (PAST RUNS): OFF ===\n"
                f"(Fresh optimization session. Historical trials from prior runs are excluded from context.)\n\n"
            )

        curr_section = (
            f"=== CURRENT RUN MEMORY ===\n"
            f"Current Run Trials Completed: {curr_mem.get('current_trial_count', len(curr_history) if curr_history is not None else 0)}\n"
            f"Current Best Candidate:\n{json.dumps(curr_mem.get('best_current_trial'), indent=2) if curr_mem.get('best_current_trial') else 'No completed trials in current run yet.'}\n\n"
            f"Recent Current Run Trials:\n{json.dumps(curr_mem.get('recent_trials', []), indent=2)}\n\n"
            f"Observed Current Run Trends:\n{json.dumps(curr_mem.get('current_trends', []), indent=2)}\n\n"
            f"Current Run Failed Configurations to Avoid:\n{json.dumps(curr_mem.get('current_failed_trials', []), indent=2) if curr_mem.get('current_failed_trials') else 'None.'}\n\n"
        )

        gen_section = ""
        if memory.get("learn_from_generalization_test") and memory.get("generalization_memory"):
            gen_section = (
                f"=== GENERALIZATION TEST (HOLDOUT VALIDATION) SUMMARY: ON ===\n"
                f"(High-level summary of independent holdout validation metrics on prior tested configurations. Use this to avoid overfitting and favor architectures with minimal generalization drop):\n"
                f"{json.dumps(memory.get('generalization_memory'), indent=2)}\n\n"
            )

        prompt = (
            f"You are a Senior RAG ML Systems Optimization Engineer. You are performing closed-loop sequential hyperparameter optimization for a RAG pipeline.\n\n"
            f"=== OPTIMIZATION OBJECTIVE ===\n"
            f"{memory['target_objective']}\n\n"
            f"=== SEARCH SPACE & CONSTRAINTS ===\n"
            f"{json.dumps(memory['search_space_constraints'], indent=2)}\n\n"
            f"{hist_section}"
            f"{curr_section}"
            f"{gen_section}"
            f"=== ALREADY TESTED CONFIGURATIONS (DO NOT REPEAT) ===\n"
            f"{json.dumps(memory.get('evaluated_fingerprints', []), indent=2)}\n\n"
            f"=== INSTRUCTIONS FOR TRIAL N+1 ===\n"
            f"1. Analyze evaluated results: if Hit Rate is low, consider changing chunk_size, increasing top_k, or switching retriever strategy (e.g. hybrid). If Faithfulness is low, consider refining chunk overlap or distance metric.\n"
            f"2. Propose EXACTLY ONE candidate configuration for the next single trial that either refines the best candidate or explores an unvisited promising subspace.\n"
            f"3. You MUST NOT propose a configuration that is identical to any previously evaluated trial listed above.\n"
            f"4. Ensure chunk_overlap is strictly less than chunk_size.\n\n"
            f"Provide your output STRICTLY as a JSON object with this format:\n"
            f'{{\n'
            f'  "reasoning": "1-2 sentence engineering justification for why this specific hyperparameter adjustment was selected based on prior trials",\n'
            f'  "chunk_strategy": "recursive" | "fixed" | "paragraph" | "semantic",\n'
            f'  "chunk_size": 256 | 384 | 512 | 768 | 1024 | 1536,\n'
            f'  "chunk_overlap": 0 | 32 | 64 | 128 | 256,\n'
            f'  "retriever_strategy": "dense" | "sparse" | "hybrid",\n'
            f'  "distance_metric": "cosine" | "dot" | "euclidean",\n'
            f'  "top_k": 2 | 3 | 4 | 6 | 8,\n'
            f'  "hybrid_alpha": 0.3 to 0.8\n'
            f'}}'
        )

        # Verification check: if historical learning is OFF, ensure prompt contains no historical trial data
        if not memory.get("use_previous_history", True):
            if "Compatible Historical Trials Retained" in prompt or "Best Historical Configuration" in prompt:
                raise ValueError("Information Leakage Error: Historical optimization data detected in fresh prompt.")

        for attempt in range(3):
            try:
                from .orchestrator import execute_with_retry
                res, _ = execute_with_retry(
                    self.llm_client.chat_completion,
                    3,
                    print,
                    "Sequential Optimizer Propose",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2 + (attempt * 0.2),
                    json_mode=True
                )
                import re
                match = re.search(r'\{.*\}', res.get("text", ""), re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    c_size = int(data.get("chunk_size", 512))
                    c_overlap = int(data.get("chunk_overlap", 64))
                    if c_overlap >= c_size:
                        c_overlap = max(0, c_size // 8)

                    candidate = PipelineConfig(
                        experiment_name=f"trial_seq_{int(datetime.now().timestamp())}_{attempt}",
                        llm_config=base_config.llm_config,
                        embedding_config=base_config.embedding_config,
                        chunking_config=ChunkingConfig(
                            strategy=str(data.get("chunk_strategy", "recursive")),
                            chunk_size=c_size,
                            chunk_overlap=c_overlap
                        ),
                        retriever_config=RetrieverConfig(
                            strategy=str(data.get("retriever_strategy", "hybrid")),
                            distance_metric=str(data.get("distance_metric", "cosine")),
                            top_k=int(data.get("top_k", 4)),
                            hybrid_alpha=round(float(data.get("hybrid_alpha", 0.7)), 2)
                        ),
                        system_prompt=base_config.system_prompt
                    )
                    
                    is_valid, _ = self.validate_config(candidate, curr_history, historical_history if memory.get("use_previous_history") else None)
                    if is_valid:
                        if data.get("reasoning"):
                            print(f"[Optimizer Reasoning] {data['reasoning']}")
                        return candidate
            except Exception as e:
                print(f"[Optimizer Propose Attempt {attempt+1}] Warning: {e}")

        # Intelligent mutation fallback if LLM is unavailable or repeatedly duplicates
        return self._mutate_config_adaptive(base_config, curr_history, memory, historical_history if memory.get("use_previous_history") else None)

    def _mutate_config_adaptive(
        self,
        base_config: PipelineConfig,
        current_run_history: List[Dict[str, Any]],
        memory: Optional[Dict[str, Any]] = None,
        historical_history: Optional[List[Dict[str, Any]]] = None
    ) -> PipelineConfig:
        """Deterministically mutate configuration to find an untried valid parameter combination."""
        chunk_sizes = [256, 384, 512, 768, 1024, 1536]
        overlaps = [0, 32, 64, 128]
        strategies = ["recursive", "fixed", "paragraph", "semantic"]
        retriever_strats = ["hybrid", "dense", "sparse"]
        dist_metrics = ["cosine", "dot", "euclidean"]
        top_ks = [2, 3, 4, 6, 8]
        alphas = [0.3, 0.5, 0.7, 0.85]

        # Try up to 50 randomized non-duplicate configurations
        for _ in range(50):
            c_size = random.choice(chunk_sizes)
            c_overlap = random.choice([o for o in overlaps if o < c_size])
            candidate = PipelineConfig(
                experiment_name=f"trial_adapt_{int(datetime.now().timestamp())}_{random.randint(10, 99)}",
                llm_config=base_config.llm_config,
                embedding_config=base_config.embedding_config,
                chunking_config=ChunkingConfig(
                    strategy=random.choice(strategies),
                    chunk_size=c_size,
                    chunk_overlap=c_overlap
                ),
                retriever_config=RetrieverConfig(
                    strategy=random.choice(retriever_strats),
                    distance_metric=random.choice(dist_metrics),
                    top_k=random.choice(top_ks),
                    hybrid_alpha=random.choice(alphas)
                ),
                system_prompt=base_config.system_prompt
            )
            is_valid, _ = self.validate_config(candidate, current_run_history, historical_history)
            if is_valid:
                return candidate

        return base_config

    def _mutate_config_random(self, base_config: PipelineConfig) -> PipelineConfig:
        """Randomly mutate hyperparameters."""
        chunk_sizes = [256, 384, 512, 768, 1024]
        overlaps = [0, 32, 64, 128]
        strategies = ["fixed", "recursive", "paragraph", "semantic"]
        retriever_strats = ["dense", "hybrid", "sparse"]
        top_ks = [2, 3, 4, 6, 8]

        c_size = random.choice(chunk_sizes)
        c_overlap = random.choice([o for o in overlaps if o < c_size])

        return PipelineConfig(
            experiment_name=f"trial_rnd_{int(datetime.now().timestamp())}",
            llm_config=base_config.llm_config,
            embedding_config=base_config.embedding_config,
            chunking_config=ChunkingConfig(
                strategy=random.choice(strategies),
                chunk_size=c_size,
                chunk_overlap=c_overlap
            ),
            retriever_config=RetrieverConfig(
                strategy=random.choice(retriever_strats),
                distance_metric=random.choice(["cosine", "dot"]),
                top_k=random.choice(top_ks),
                hybrid_alpha=round(random.uniform(0.3, 0.8), 2)
            ),
            system_prompt=base_config.system_prompt
        )

    def run_trial(
        self,
        config: PipelineConfig,
        documents: List[Document],
        dataset: Dataset
    ) -> Experiment:
        """Build index, run retrieval, generate answers, evaluate dataset, return Experiment."""
        from .types import Experiment, Artifact, EnvironmentSnapshot
        from .runtime_discovery import RuntimeDiscovery
        import time, sys
        start_time = time.time()
        
        index_builder = IndexBuilder(self.workspace, self.embedding_client, self.llm_client)
        retrieval_engine = RetrievalEngine(self.embedding_client)
        gen_engine = GenerationEngine(self.llm_client)

        pipeline_id = config.pipeline_id

# 1. Build Index (artifact generation handled by WorkspaceManager inside IndexBuilder, but we can query them)
        index = index_builder.build_index(documents, config.chunking_config, getattr(config.retriever_config, "index_type", "memory"), pipeline_id, dataset.kb_checksum)

        # Collect artifacts
        exp_artifacts = []
        import hashlib
        chunking_hash = hashlib.sha256(f"{config.chunking_config.get_hash()}_{dataset.kb_checksum}".encode()).hexdigest()
        embedding_hash = hashlib.sha256(f"{config.embedding_config.get_hash()}_{chunking_hash}".format().encode()).hexdigest()
        
        # We can simulate fetching these artifact dictionaries based on what save_artifact does, 
        # or just query the workspace if we had a getter. We'll query glob since we know their prefix.
        artifacts_dir = self.workspace.root_dir / "artifacts"
        if artifacts_dir.exists():
            import json
            for filepath in artifacts_dir.glob("*.json"):
                if f"chunks_art_" in filepath.name or f"embeddings_art_" in filepath.name:
                    try:
                        with open(filepath, "r", encoding="utf-8") as af:
                            # We don't read the whole data, we just want metadata if it was saved.
                            # Since save_artifact doesn't save metadata separately yet, we can just construct Artifact
                            # Wait, the prompt says Artifact is a concept. We'll just define the artifacts manually here.
                            pass
                    except Exception:
                        pass
        
        # We can construct them manually since we know they exist or were created
        chunks_art = Artifact(
            artifact_id=f"art_chunks_{chunking_hash[:8]}",
            type="chunks",
            path=str(self.workspace.cache_dir / f"chunks_{chunking_hash}.json"),
            hash=chunking_hash,
            creation_time=datetime.now().isoformat()
        )
        exp_artifacts.append(chunks_art)
        
        embeds_art = Artifact(
            artifact_id=f"art_embeds_{embedding_hash[:8]}",
            type="embeddings",
            path=str(self.workspace.cache_dir / f"embeddings_{embedding_hash}.json"),
            hash=embedding_hash,
            creation_time=datetime.now().isoformat()
        )
        exp_artifacts.append(embeds_art)
        
        # 2. Evaluate across dataset items
        r_list, a_list = [], []
        total_tokens, total_latency = 0, 0.0
        sample_evals = []
        
        items_to_eval = dataset.items[:min(10, len(dataset.items))]
        
        # Pre-compute retrieval for all items
        retrieved_results = []
        for item in items_to_eval:
            retrieved = retrieval_engine.retrieve(item.question, index, config.retriever_config)
            r_metrics = self.eval_engine.evaluate_retrieval(retrieved, item.chunk_id, item.ground_truth)
            r_list.append(r_metrics)
            
            context_str, citations = PromptBuilder.build_context_string(retrieved)
            retrieved_results.append((retrieved, r_metrics, context_str, citations))

        # Pass 1: Generate Answers using Orchestrator
        from .orchestrator import AIOrchestrator, Task, execute_with_retry
        orchestrator = AIOrchestrator(self.workspace, self.llm_client)
        
        gen_tasks = []
        for i, item in enumerate(items_to_eval):
            context_str = retrieved_results[i][2]
            prompt = PromptBuilder.build_prompt(item.question, context_str)
            gen_tasks.append(
                Task(
                    task_id=f"gen_{i}",
                    task_type="generate_answer",
                    input_data=prompt
                )
            )
            
        orchestrator.execute_tasks(gen_tasks)
        
        # Check if generation completely failed or had a provider failure
        provider_failure = any(task.error and ("Forbidden" in str(task.error) or "HTTP Error" in str(task.error)) for task in gen_tasks)
        gen_failed = all(task.error for task in gen_tasks)
        
        if (gen_failed or provider_failure) and len(gen_tasks) > 0:
            failure_reason = next((str(task.error) for task in gen_tasks if task.error), "All generation tasks failed")
            print(f"[Trial Execution] Generation failed: {failure_reason}. Aborting trial.")
            exp_id = f"exp_{int(datetime.now().timestamp())}_{random.randint(100, 999)}"
            res = EvaluationResult(
                experiment_id=exp_id,
                pipeline_config=config,
                retrieval_metrics=RetrievalMetrics(),
                answer_metrics=AnswerMetrics(),
                avg_latency_ms=0.0,
                total_tokens=orchestrator.metrics.get("total_tokens_used", 0),
                status="failed",
                metrics_valid=False,
                failure_reason=failure_reason,
                composite_score=None
            )
            return Experiment(
                experiment_id=exp_id,
                pipeline_id=config.pipeline_id,
                dataset_id=dataset.dataset_id,
                timestamp=datetime.now().isoformat(),
                runtime=0.0,
                results=res,
                config=config,
                status="failed",
                composite_score=None
            )
            
        answers = []
        for task in gen_tasks:
            if task.result and "answer" in task.result:
                answers.append(task.result["answer"])
            else:
                answers.append("Generation failed or timed out.")
                
        # Pass 2: Evaluate Answers using Orchestrator
        eval_tasks = []
        for i, item in enumerate(items_to_eval):
            eval_input = f"Question: {item.question}\nReference Answer (Ground Truth): {item.ground_truth}\nAssistant Answer: {answers[i]}"
            eval_tasks.append(
                Task(
                    task_id=f"eval_{i}",
                    task_type="evaluate_answer",
                    input_data=eval_input
                )
            )
            
        orchestrator.execute_tasks(eval_tasks)
        
        # Check if evaluation completely failed or had a provider failure
        provider_failure_eval = any(task.error and ("Forbidden" in str(task.error) or "HTTP Error" in str(task.error)) for task in eval_tasks)
        eval_failed = all(task.error for task in eval_tasks)
        
        if (eval_failed or provider_failure_eval) and len(eval_tasks) > 0:
            failure_reason = next((str(task.error) for task in eval_tasks if task.error), "All evaluation tasks failed")
            print(f"[Trial Execution] Evaluation failed: {failure_reason}. Aborting trial.")
            exp_id = f"exp_{int(datetime.now().timestamp())}_{random.randint(100, 999)}"
            res = EvaluationResult(
                experiment_id=exp_id,
                pipeline_config=config,
                retrieval_metrics=RetrievalMetrics(),
                answer_metrics=AnswerMetrics(),
                avg_latency_ms=0.0,
                total_tokens=orchestrator.metrics.get("total_tokens_used", 0),
                status="failed",
                metrics_valid=False,
                failure_reason=failure_reason,
                composite_score=None
            )
            return Experiment(
                experiment_id=exp_id,
                pipeline_id=config.pipeline_id,
                dataset_id=dataset.dataset_id,
                timestamp=datetime.now().isoformat(),
                runtime=0.0,
                results=res,
                config=config,
                status="failed",
                composite_score=None
            )
        
        for i, item in enumerate(items_to_eval):
            a_metrics = AnswerMetrics()
            task = eval_tasks[i]
            if task.result:
                a_metrics.accuracy = float(task.result.get("accuracy", 0.0))
                a_metrics.completeness = float(task.result.get("completeness", 0.0))
                a_metrics.relevance = float(task.result.get("relevance", 0.0))
                a_metrics.answer_correctness = a_metrics.accuracy
                a_metrics.faithfulness = a_metrics.accuracy
            
            a_list.append(a_metrics)
            
            r_metrics = retrieved_results[i][1]
            retrieved = retrieved_results[i][0]
            
            sample_evals.append({
                "question": item.question,
                "ground_truth": item.ground_truth,
                "answer": answers[i],
                "retrieved_count": len(retrieved),
                "r_precision": r_metrics.precision,
                "a_accuracy": a_metrics.accuracy,
                "a_answer_correctness": a_metrics.answer_correctness,
                "a_faithfulness": a_metrics.faithfulness
            })
            
        # Update metrics with orchestrator totals
        total_tokens = orchestrator.metrics.get("total_tokens_used", 0)
        total_latency = orchestrator.metrics.get("total_execution_time", 0.0) * 1000

        # Average metrics
        avg_precision = sum(m.precision for m in r_list) / max(1, len(r_list))
        avg_recall = sum(m.recall for m in r_list) / max(1, len(r_list))
        avg_hit = sum(m.hit_rate for m in r_list) / max(1, len(r_list))
        avg_mrr = sum(m.mrr for m in r_list) / max(1, len(r_list))
        avg_ndcg = sum(m.ndcg for m in r_list) / max(1, len(r_list))

        avg_retrieval = RetrievalMetrics(
            precision=round(avg_precision, 4),
            recall=round(avg_recall, 4),
            hit_rate=round(avg_hit, 4),
            mrr=round(avg_mrr, 4),
            ndcg=round(avg_ndcg, 4)
        )

        avg_corr = sum(getattr(m, "answer_correctness", getattr(m, "accuracy", getattr(m, "faithfulness", 0.0))) for m in a_list) / max(1, len(a_list))
        avg_rel = sum(m.relevance for m in a_list) / max(1, len(a_list))
        avg_comp = sum(m.completeness for m in a_list) / max(1, len(a_list))
        
        avg_answer = AnswerMetrics(
            accuracy=round(avg_corr, 4),
            answer_correctness=round(avg_corr, 4),
            relevance=round(avg_rel, 4),
            completeness=round(avg_comp, 4),
            faithfulness=round(avg_corr, 4)
        )

        composite = self.eval_engine.calculate_composite_score(avg_retrieval, avg_answer)

        exp_id = f"exp_{int(datetime.now().timestamp())}_{random.randint(100, 999)}"
        
        failed_batches = 0
        completed_batches = 0
        failure_reasons = set()
        for t in gen_tasks + eval_tasks:
            if getattr(t, 'error', None) or not t.result:
                failed_batches += 1
                if getattr(t, 'error', None):
                    failure_reasons.add(str(t.error))
                elif not t.result:
                    failure_reasons.add("Task returned no result")
            else:
                completed_batches += 1

        metrics_valid = (failed_batches == 0)
        status = "completed" if metrics_valid else ("failed" if completed_batches == 0 else "partial_failed")
        failure_reason = " | ".join(failure_reasons) if failure_reasons else None
        retry_count = orchestrator.metrics.get("retry_count", 0) if 'orchestrator' in locals() else 0

        def normalize_trial_result(result: EvaluationResult) -> EvaluationResult:
            if not result.metrics_valid or result.failed_batches > 0:
                result.metrics_valid = False
                result.composite_score = None
                result.status = "failed" if result.completed_batches == 0 else "partial_failed"
                # Clear invalid metrics
                result.retrieval_metrics = RetrievalMetrics()
                result.answer_metrics = AnswerMetrics()
            if not result.failure_reason and result.failed_batches > 0:
                result.failure_reason = "Unknown transient failure during evaluation"
            return result

        eval_result = EvaluationResult(
            experiment_id=exp_id,
            pipeline_config=config,
            retrieval_metrics=avg_retrieval,
            answer_metrics=avg_answer,
            composite_score=composite if metrics_valid else None,
            avg_latency_ms=round(total_latency / max(1, len(dataset.items)), 2),
            total_tokens=total_tokens,
            sample_evaluations=sample_evals,
            timestamp=datetime.now().isoformat(),
            orchestrator_metrics=orchestrator.metrics if 'orchestrator' in locals() else None,
            status=status,
            metrics_valid=metrics_valid,
            failure_reason=failure_reason,
            completed_batches=completed_batches,
            failed_batches=failed_batches,
            retry_count=retry_count
        )
        
        eval_result = normalize_trial_result(eval_result)
        
        runtime_ms = time.time() - start_time
        
        env = RuntimeDiscovery.discover()
        env_snapshot = EnvironmentSnapshot(
            python_version=sys.version,
            os=env.os_name,
            architecture=env.architecture,
            virtual_env=env.is_virtual_env,
            pip_executable=env.pip_executable,
            python_executable=env.python_executable
        )
        

        return Experiment(
            experiment_id=exp_id,
            pipeline_id=pipeline_id,
            dataset_id=dataset.dataset_id,
            timestamp=datetime.now().isoformat(),
            runtime=round(runtime_ms, 2),
            results=eval_result,
            artifacts=exp_artifacts,
            environment=env_snapshot,
            config=config,
            status=eval_result.status,
            composite_score=eval_result.composite_score
        )

    def optimize(
        self,
        spec: OptimizationSpec,
        base_config: PipelineConfig,
        documents: List[Document],
        dataset: Dataset
    ) -> Dict[str, Any]:
        """
        Run closed-loop sequential multi-trial optimization pipeline according to spec.
        Enforces OBSERVE -> ANALYZE -> DECIDE ONE -> EXECUTE -> EVALUATE -> PERSIST -> UPDATE MEMORY.
        Supports explicit user toggle `use_previous_history` for learning from retained past runs.
        """
        kb_checksum = dataset.kb_checksum or "default_kb"
        optimization_run_id = f"run_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"
        use_previous_history = bool(getattr(spec, "use_previous_history", getattr(spec, "usePreviousOptimizationHistory", True)))

        # Load prior historical experiments scoped strictly by KB and dataset lineage
        all_experiments = self.workspace.list_experiments(kb_checksum=kb_checksum)
        historical_trials = []
        if use_previous_history:
            for exp in all_experiments:
                # Exclude experiments from the current active run
                if exp.get("optimization_run_id") == optimization_run_id:
                    continue
                # Match KB checksum
                h_chk = exp.get("kb_checksum") or exp.get("metadata", {}).get("kb_checksum")
                if h_chk and h_chk != kb_checksum:
                    continue
                # Match Dataset ID if present
                h_ds = exp.get("dataset_id")
                if dataset.dataset_id and h_ds and h_ds != dataset.dataset_id:
                    continue
                historical_trials.append(exp)

        print(f"[optimizer] Starting optimization run: {optimization_run_id}")
        print(f"[optimizer] Previous-history learning: {'enabled' if use_previous_history else 'disabled'}")
        print(f"[optimizer] Compatible historical trials available: {len(historical_trials) if use_previous_history else 0}")

        current_run_history: List[Dict[str, Any]] = []
        best_result: Optional[EvaluationResult] = None
        best_score = -1.0

        grid_combinations = []
        if spec.strategy == "grid":
            # Pre-compute grid space to step through sequentially
            for c_strat in ["recursive", "fixed"]:
                for c_size in [256, 512, 1024]:
                    for r_strat in ["hybrid", "dense"]:
                        for top_k in [3, 6]:
                            c_overlap = 64 if c_size > 256 else 32
                            grid_combinations.append(
                                PipelineConfig(
                                    experiment_name=f"grid_{c_strat}_{c_size}_{r_strat}_k{top_k}",
                                    llm_config=base_config.llm_config,
                                    embedding_config=base_config.embedding_config,
                                    chunking_config=ChunkingConfig(strategy=c_strat, chunk_size=c_size, chunk_overlap=c_overlap),
                                    retriever_config=RetrieverConfig(strategy=r_strat, distance_metric="cosine", top_k=top_k, hybrid_alpha=0.7),
                                    system_prompt=base_config.system_prompt
                                )
                            )

        for trial in range(1, spec.max_trials + 1):
            # 1. OBSERVE & ANALYZE: Build canonical layered optimization memory
            memory = self.build_optimization_memory(
                current_run_history=current_run_history,
                historical_history=historical_trials if use_previous_history else [],
                use_previous_history=use_previous_history,
                spec=spec,
                base_config=base_config,
                kb_checksum=kb_checksum,
                dataset_id=dataset.dataset_id
            )

            # 2. DECIDE ONE NEXT CANDIDATE
            candidate: PipelineConfig
            if spec.strategy == "llm_guided":
                if trial == 1:
                    # If historical learning is ON and we have prior valid history, let LLM propose immediately conditioned on history
                    if use_previous_history and historical_trials:
                        candidate = self.propose_candidate_llm(
                            memory,
                            base_config,
                            current_run_history,
                            historical_history=historical_trials if use_previous_history else None
                        )
                    else:
                        # Fresh run: test base configuration or adaptive mutation
                        is_valid_base, _ = self.validate_config(
                            base_config,
                            current_run_history,
                            historical_history=historical_trials if use_previous_history else None
                        )
                        if is_valid_base:
                            candidate = base_config
                        else:
                            candidate = self._mutate_config_adaptive(
                                base_config,
                                current_run_history,
                                memory,
                                historical_history=historical_trials if use_previous_history else None
                            )
                else:
                    # LLM proposes ONE candidate strictly conditioned on memory of evaluated trials
                    candidate = self.propose_candidate_llm(
                        memory,
                        base_config,
                        current_run_history,
                        historical_history=historical_trials if use_previous_history else None
                    )
            elif spec.strategy == "grid":
                # Find first grid combo not in current or historical history
                found_grid = False
                for g_cand in grid_combinations:
                    is_valid, _ = self.validate_config(
                        g_cand,
                        current_run_history,
                        historical_history=historical_trials if use_previous_history else None
                    )
                    if is_valid:
                        candidate = g_cand
                        found_grid = True
                        break
                if not found_grid:
                    candidate = self._mutate_config_adaptive(
                        base_config,
                        current_run_history,
                        memory,
                        historical_history=historical_trials if use_previous_history else None
                    )
            else:  # random search
                candidate = self._mutate_config_adaptive(
                    base_config,
                    current_run_history,
                    memory,
                    historical_history=historical_trials if use_previous_history else None
                )

            # 3. VALIDATE & DEDUPLICATE
            is_valid, reason = self.validate_config(
                candidate,
                current_run_history,
                historical_history=historical_trials if use_previous_history else None
            )
            if not is_valid:
                candidate = self._mutate_config_adaptive(
                    base_config,
                    current_run_history,
                    memory,
                    historical_history=historical_trials if use_previous_history else None
                )
                is_valid, reason = self.validate_config(
                    candidate,
                    current_run_history,
                    historical_history=historical_trials if use_previous_history else None
                )

            if not is_valid:
                print(f"[Trial {trial}] Could not generate valid non-duplicate candidate: {reason}. Skipping.")
                continue

            # 4. EXECUTE & EVALUATE TRIAL
            try:
                print("[Trial Start]")
                print(f"Trial ID: {trial}")
                print(f"Optimization Run ID: {optimization_run_id}")
                print(f"Strategy: {spec.strategy}")
                print(f"Config Hash: {self._hash_config(candidate)}")
                print(f"Parameters: chunk_size={candidate.chunking_config.chunk_size}, overlap={candidate.chunking_config.chunk_overlap}, retriever={candidate.retriever_config.strategy}, top_k={candidate.retriever_config.top_k}")
                print("Status: running")
                
                experiment = self.run_trial(candidate, documents, dataset)
                
                if experiment.results and experiment.results.status == 'completed':
                    ans_val = getattr(experiment.results.answer_metrics, "answer_correctness", getattr(experiment.results.answer_metrics, "accuracy", getattr(experiment.results.answer_metrics, "faithfulness", "N/A"))) if experiment.results.answer_metrics else 'N/A'
                    print("\n[Metric Calculation]")
                    print(f"Trial ID: {trial}")
                    print(f"Hit Rate: {experiment.results.retrieval_metrics.hit_rate if experiment.results.retrieval_metrics else 'N/A'}")
                    print(f"Context Precision: {experiment.results.retrieval_metrics.precision if experiment.results.retrieval_metrics else 'N/A'}")
                    print(f"Answer Correctness: {ans_val}")
                    print(f"Latency: {experiment.results.avg_latency_ms} ms")
                    print(f"Composite Formula: (Hit Rate * 0.7) + (Answer Correctness * 0.3)")
                    print(f"Composite Score: {experiment.results.composite_score}")
                    print()
                
                print("[Trial Result]")
                print(f"Trial ID: {trial}")
                print(f"Status: {experiment.results.status if experiment.results else 'failed'}")
                print(f"Composite: {experiment.results.composite_score if (experiment.results and experiment.results.composite_score is not None) else 'None'}")
                print(f"Metrics Valid: {'yes' if experiment.results and experiment.results.metrics_valid else 'no'}")
                    
            except Exception as e:
                print("[Trial Result]")
                print(f"Trial ID: {trial}")
                print("Status: failed")
                print(f"Failure Reason: {str(e)}")
                
                exp_id = f"exp_{int(datetime.now().timestamp())}_{random.randint(100, 999)}"
                experiment = Experiment(
                    experiment_id=exp_id,
                    pipeline_id=candidate.pipeline_id,
                    dataset_id=dataset.dataset_id,
                    timestamp=datetime.now().isoformat(),
                    runtime=0.0,
                    results=EvaluationResult(
                        experiment_id=exp_id,
                        pipeline_config=candidate,
                        retrieval_metrics=RetrievalMetrics(),
                        answer_metrics=AnswerMetrics(),
                        avg_latency_ms=0.0,
                        total_tokens=0,
                        status="failed",
                        metrics_valid=False,
                        failure_reason=str(e),
                        composite_score=None
                    ),
                    config=candidate,
                    status="failed",
                    composite_score=None
                )

            # 5. PERSIST TRIAL STATE
            from dataclasses import asdict
            
            trial_record = asdict(experiment)
            
            # Security: scrub secrets
            if "config" in trial_record and trial_record["config"]:
                if "llm_config" in trial_record["config"]:
                    trial_record["config"]["llm_config"].pop("api_key", None)
                if "embedding_config" in trial_record["config"]:
                    trial_record["config"]["embedding_config"].pop("api_key", None)
            
            if "results" in trial_record and trial_record["results"]:
                if "pipeline_config" in trial_record["results"]:
                    if "llm_config" in trial_record["results"]["pipeline_config"]:
                        trial_record["results"]["pipeline_config"]["llm_config"].pop("api_key", None)
                    if "embedding_config" in trial_record["results"]["pipeline_config"]:
                        trial_record["results"]["pipeline_config"]["embedding_config"].pop("api_key", None)

            trial_record["trial_number"] = trial
            trial_record["optimization_run_id"] = optimization_run_id
            trial_record["used_previous_history"] = use_previous_history
            trial_record["composite_score"] = experiment.composite_score
            trial_record["kb_checksum"] = kb_checksum
            trial_record["dataset_id"] = dataset.dataset_id
            if "metadata" not in trial_record or not isinstance(trial_record["metadata"], dict):
                trial_record["metadata"] = {}
            trial_record["metadata"]["kb_checksum"] = kb_checksum
            trial_record["metadata"]["optimization_run_id"] = optimization_run_id
            trial_record["metadata"]["optimization_strategy"] = spec.strategy
            trial_record["metadata"]["used_previous_history"] = use_previous_history
            trial_record["metadata"]["dataset_id"] = dataset.dataset_id
            if trial_record.get("results"):
                trial_record["results"]["composite_score"] = experiment.results.composite_score if experiment.results else None
            
            current_run_history.append(trial_record)
            self.workspace.save_experiment(trial_record)

            # 6. UPDATE CANONICAL OPTIMIZATION MEMORY ON DISK
            updated_memory = self.build_optimization_memory(
                current_run_history=current_run_history,
                historical_history=historical_trials if use_previous_history else [],
                use_previous_history=use_previous_history,
                spec=spec,
                base_config=base_config,
                kb_checksum=kb_checksum,
                dataset_id=dataset.dataset_id
            )
            self.workspace.save_optimization_memory(kb_checksum, updated_memory)
            best_score_disp = (updated_memory.get('best_so_far') or {}).get('composite_score', 'N/A')
            print(f"[Optimization Memory Update] Scoped to KB {kb_checksum[:8]} | Current Run Evaluated: {len(current_run_history)} | Historical: {len(historical_trials) if use_previous_history else 0} | Best Score: {best_score_disp}")

            # 7. Track Best Configuration
            if experiment.results and experiment.results.metrics_valid and experiment.results.composite_score is not None and experiment.results.composite_score > best_score:
                best_score = experiment.results.composite_score
                best_result = experiment

        def get_score(x):
            r = x.get("results", {}) or {}
            score = r.get("composite_score")
            if score is None:
                score = x.get("composite_score")
            if isinstance(score, (int, float)) and __import__("math").isfinite(score):
                return float(score)
            return -1.0
            
        print("\n[Best Trial Selection]")
        
        valid_history = [
            h for h in current_run_history
            if h.get("results") and h["results"].get("status") == "completed"
            and h["results"].get("metrics_valid")
            and get_score(h) != -1.0
        ]
        
        completed_trials = sum(1 for h in current_run_history if h.get("results") and h["results"].get("status") == "completed")
        partial_failed_trials = sum(1 for h in current_run_history if h.get("results") and h["results"].get("status") == "partial_failed")
        failed_trials = len(current_run_history) - completed_trials - partial_failed_trials
        valid_trials = len(valid_history)
        if completed_trials + failed_trials + partial_failed_trials != len(current_run_history) or valid_trials > completed_trials:
            print("[SWEEP INTEGRITY ERROR] Counters do not match trial history.")

        candidate_ids = [h.get("trial_number") for h in current_run_history]
        valid_ids = [h.get("trial_number") for h in valid_history]
        print(f"Candidate Trial IDs: {candidate_ids}")
        print(f"Valid Scored Trial IDs: {valid_ids}")
        
        best_trial_num = 'None'
        best_h = {}
        if valid_history:
            best_h = max(valid_history, key=get_score)
            best_trial_num = best_h.get("trial_number")
            best_score = get_score(best_h)
            best_result = next((exp for exp in [best_result] if exp), None)
        
        print(f"Selected Trial ID: {best_trial_num}")
        print(f"Selected Composite: {best_score if valid_history else 'None'}")
        if not valid_history:
            print("Selection Status: NO_VALID_TRIALS")
        print("\n[Sweep Summary]")
        print(f"Optimization Run ID: {optimization_run_id}")
        print(f"Previous History Learned: {use_previous_history}")
        print(f"Historical Trials Retained: {len(historical_trials) if use_previous_history else 0}")
        print(f"Total Current Run Trials: {len(current_run_history)}")
        print(f"Completed Trials: {completed_trials}")
        print(f"Failed Trials: {failed_trials}")
        print(f"Partial Failed Trials: {partial_failed_trials}")
        print(f"Valid Trials: {valid_trials}")
        print(f"Best Composite: {best_score if valid_history else 'None'}")
        if not valid_history:
            print("Final Status: FAILED")
            print("Failure Reason: All trials failed before producing valid metrics")
        
        # Rank current run history by score descending, putting -1.0 (None) at the end
        current_run_history.sort(key=get_score, reverse=True)

        mapped_leaderboard = []
        for h in current_run_history:
            res = h.get("results", {}) or {}
            score = res.get("composite_score")
            if score is None:
                score = h.get("composite_score")
            if score is not None and not (isinstance(score, (int, float)) and __import__("math").isfinite(score)):
                score = None
            ans_m = res.get("answer_metrics", {}) if res.get("metrics_valid") else {}
            ans_val = None
            if isinstance(ans_m, dict):
                ans_val = ans_m.get("answer_correctness")
                if ans_val is None:
                    ans_val = ans_m.get("accuracy")
                if ans_val is None:
                    ans_val = ans_m.get("faithfulness")
            elif ans_m:
                ans_val = getattr(ans_m, "answer_correctness", getattr(ans_m, "accuracy", getattr(ans_m, "faithfulness", None)))

            mapped_leaderboard.append({
                "trial": h.get("trial_number", 0),
                "optimization_run_id": h.get("optimization_run_id", optimization_run_id),
                "experiment_id": h.get("experiment_id", ""),
                "composite_score": score,
                "retrieval_hit_rate": res.get("retrieval_metrics", {}).get("hit_rate", None) if res.get("metrics_valid") else None,
                "answer_correctness": ans_val,
                "answer_faithfulness": ans_val,
                "config": h.get("config", {}),
                "timestamp": h.get("timestamp", ""),
                "result": res,
                "status": res.get("status", h.get("status", "failed")),
                "used_previous_history": h.get("used_previous_history", use_previous_history)
            })

        status = "completed" if valid_history else "failed"
        failure_reason = None if valid_history else "All trials failed before producing valid metrics"
        
        return {
            "status": status,
            "failure_reason": failure_reason,
            "optimization_run_id": optimization_run_id,
            "used_previous_history": use_previous_history,
            "historical_trials_count": len(historical_trials) if use_previous_history else 0,
            "trials_run": len(current_run_history),
            "total_trials": len(current_run_history),
            "best_score": best_score if valid_history else None,
            "best_experiment_id": best_h.get("experiment_id", "") if valid_history else "",
            "best_config": best_h.get("config", base_config.__dict__) if valid_history else base_config.__dict__,
            "leaderboard": mapped_leaderboard
        }
