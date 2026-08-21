"""
Generalization Test Engine for AutoRAG Platform.
Evaluates completed optimization Trials against independent, frozen holdout datasets
to measure true generalization capability and prevent overfitting.
"""

import os
import sys
import re
import json
import time
import math
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable

from .types import (
    Document, Chunk, Dataset, DatasetItem, PipelineConfig, ChunkingConfig,
    RetrieverConfig, RetrievalMetrics, AnswerMetrics, GeneralizationTestResult,
    EvaluationResult
)
from .workspace import WorkspaceManager
from .connections import OpenAICompatibleClient, EmbeddingClient
from .index_builder import IndexBuilder
from .retrieval_engine import RetrievalEngine
from .prompt_builder import PromptBuilder
from .evaluation_engine import EvaluationEngine
from .orchestrator import AIOrchestrator, Task


def _normalize_text(s: str) -> str:
    """Normalize text for strict deduplication comparisons."""
    return re.sub(r'\W+', '', s.lower())


class GeneralizationEngine:
    """
    Coordinates generation of independent holdout evaluation datasets,
    execution of generalization validation against completed trials,
    and non-destructive persistence of generalization test results.
    """

    def __init__(
        self,
        workspace: WorkspaceManager,
        llm_client: Optional[OpenAICompatibleClient] = None,
        emb_client: Optional[EmbeddingClient] = None,
        eval_engine: Optional[EvaluationEngine] = None,
        logger: Optional[Callable[[str], None]] = None
    ):
        self.workspace = workspace
        self.llm_client = llm_client
        self.emb_client = emb_client or EmbeddingClient()
        self.eval_engine = eval_engine or EvaluationEngine()
        self.logger = logger

    def log(self, message: str):
        if self.logger:
            self.logger(message)
        else:
            print(f"[GeneralizationEngine] {message}", file=sys.stderr)

    def get_existing_optimization_questions(self, kb_checksum: str) -> set:
        """Collect all existing optimization question strings for a KB to prevent leakage."""
        existing_questions = set()
        if not self.workspace.datasets_dir.exists():
            return existing_questions

        for filepath in self.workspace.datasets_dir.glob("dataset_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("kb_checksum") == kb_checksum:
                        for item in data.get("items", []):
                            q = item.get("question")
                            if q:
                                existing_questions.add(_normalize_text(q))
            except Exception:
                continue
        return existing_questions

    def build_or_load_holdout_dataset(
        self,
        documents: List[Document],
        chunks: List[Chunk],
        kb_checksum: str,
        test_size: int = 5,
        force_regenerate: bool = False
    ) -> Dataset:
        """
        Build or load a frozen holdout dataset of size test_size that is strictly
        independent and deduplicated against the primary optimization datasets.
        """
        holdout_file_pattern = f"holdout_dataset_v1_{kb_checksum[:8]}_n{test_size}.json"
        holdout_path = self.workspace.datasets_dir / holdout_file_pattern

        if not force_regenerate and holdout_path.exists():
            try:
                with open(holdout_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = [DatasetItem(**item) for item in data.get("items", [])]
                if len(items) >= test_size:
                    self.log(f"Loaded frozen holdout dataset with {len(items)} items from disk.")
                    return Dataset(
                        dataset_id=data.get("dataset_id", f"holdout_{kb_checksum[:8]}"),
                        created_at=data.get("created_at", datetime.now().isoformat()),
                        version=data.get("version", "1.0"),
                        kb_checksum=kb_checksum,
                        items=items[:test_size]
                    )
            except Exception as e:
                self.log(f"Failed to load cached holdout dataset: {e}. Generating a new one.")

        # Gather questions already used in optimization dataset(s) for strict deduplication
        existing_opt_questions = self.get_existing_optimization_questions(kb_checksum)
        self.log(f"Found {len(existing_opt_questions)} existing optimization questions to deduplicate against.")

        # Sample chunks with a reverse/offset strategy or distinct slice for holdout coverage
        if not chunks:
            raise ValueError("Cannot generate holdout dataset from empty chunks.")

        # Sample chunks spread across the documents
        step = max(1, len(chunks) // max(1, test_size * 2))
        sampled_chunks = [chunks[i] for i in range(len(chunks) - 1, -1, -step)]
        if len(sampled_chunks) < test_size:
            sampled_chunks = chunks.copy()
            random.Random(42).shuffle(sampled_chunks)

        holdout_items: List[DatasetItem] = []
        seen_holdout_questions = set()
        item_counter = 0

        for chunk in sampled_chunks:
            if len(holdout_items) >= test_size:
                break

            if not self.llm_client:
                raise RuntimeError("LLM Client is required to generate holdout validation datasets.")

            prompt = (
                f"You are a rigorous ML evaluation system constructing an independent holdout test set.\n"
                f"Given the text below, generate 2 specific, challenging, and factual question-and-answer pairs.\n"
                f"The questions MUST be answerable strictly from the provided text.\n\n"
                f"Text:\n{chunk.text}\n\n"
                f"Format output strictly as JSON array:\n"
                f'[{{\"question\": \"...\", \"ground_truth\": \"...\"}}]'
            )

            try:
                res = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                text = res.get("text", "")
                json_match = re.search(r'\[.*\]', text, re.DOTALL)
                if json_match:
                    pairs = json.loads(json_match.group(0))
                    for pair in pairs:
                        q = pair.get("question", "").strip()
                        gt = pair.get("ground_truth", "").strip()
                        if not q or not gt:
                            continue

                        norm_q = _normalize_text(q)
                        # Check deduplication against optimization data and previously added holdout items
                        if norm_q in existing_opt_questions:
                            self.log(f"Skipping duplicate optimization question: {q}")
                            continue
                        if norm_q in seen_holdout_questions:
                            continue

                        seen_holdout_questions.add(norm_q)
                        item_counter += 1
                        holdout_items.append(DatasetItem(
                            item_id=f"gen_q_{item_counter}",
                            doc_id=chunk.doc_id,
                            chunk_id=chunk.chunk_id,
                            question=q,
                            ground_truth=gt,
                            reference_answer=gt,
                            gold_context=chunk.text,
                            metadata={"holdout": True, "source_chunk": chunk.chunk_id}
                        ))
                        if len(holdout_items) >= test_size:
                            break
            except Exception as e:
                self.log(f"Holdout chunk Q&A generation warning: {e}")
                continue

        if len(holdout_items) == 0:
            raise ValueError("Failed to generate any valid holdout Q&A items for generalization test.")

        dataset_id = f"gen_holdout_{kb_checksum[:8]}_{int(time.time())}"
        holdout_ds = Dataset(
            dataset_id=dataset_id,
            created_at=datetime.now().isoformat(),
            version="1.0",
            kb_checksum=kb_checksum,
            items=holdout_items
        )

        # Freeze holdout dataset to disk
        try:
            data = {
                "dataset_id": dataset_id,
                "created_at": holdout_ds.created_at,
                "version": holdout_ds.version,
                "kb_checksum": kb_checksum,
                "is_generalization_holdout": True,
                "items": [item.__dict__ for item in holdout_items]
            }
            with open(holdout_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.log(f"Saved frozen holdout dataset of size {len(holdout_items)} to {holdout_path.name}")
        except Exception as e:
            self.log(f"Warning: Could not freeze holdout dataset to disk: {e}")

        return holdout_ds

    def run_generalization_test(
        self,
        experiment_id: str,
        docs: List[Document],
        kb_checksum: str,
        test_size: int = 5,
        emit_progress_fn: Optional[Callable[[int, str, List[str], str], None]] = None
    ) -> GeneralizationTestResult:
        """
        Execute an independent Generalization Test on an existing Trial.
        """
        def report(progress: int, stage: str, completed: List[str], message: str):
            if emit_progress_fn:
                emit_progress_fn(progress, stage, completed, message)
            self.log(f"[{stage}] {message}")

        report(10, "Validating Target Trial", [], f"Verifying Trial {experiment_id} in workspace...")

        exp_data = self.workspace.get_experiment(experiment_id)
        if not exp_data:
            raise ValueError(f"Trial {experiment_id} not found in workspace experiments.")

        cfg_dict = exp_data.get("config", {})
        if not cfg_dict:
            raise ValueError(f"Trial {experiment_id} does not contain a valid configuration dictionary.")

        # Reconstruct PipelineConfig
        chunking_dict = cfg_dict.get("chunking_config", {})
        if hasattr(chunking_dict, "__dict__"):
            chunking_dict = chunking_dict.__dict__
        elif not isinstance(chunking_dict, dict):
            chunking_dict = {}

        retriever_dict = cfg_dict.get("retriever_config", {})
        if hasattr(retriever_dict, "__dict__"):
            retriever_dict = retriever_dict.__dict__
        elif not isinstance(retriever_dict, dict):
            retriever_dict = {}

        p_cfg = PipelineConfig(
            experiment_name=exp_data.get("name", experiment_id),
            chunking_config=ChunkingConfig(
                strategy=chunking_dict.get("strategy", "recursive"),
                chunk_size=int(chunking_dict.get("chunk_size", 512)),
                chunk_overlap=int(chunking_dict.get("chunk_overlap", 64))
            ),
            retriever_config=RetrieverConfig(
                strategy=retriever_dict.get("strategy", "hybrid"),
                distance_metric=retriever_dict.get("distance_metric", "cosine"),
                top_k=int(retriever_dict.get("top_k", 4)),
                hybrid_alpha=float(retriever_dict.get("hybrid_alpha", 0.7))
            ),
            system_prompt=cfg_dict.get("system_prompt", "You are a helpful assistant.")
        )

        opt_score = exp_data.get("composite_score")
        if opt_score is None and isinstance(exp_data.get("results"), dict):
            opt_score = exp_data.get("results", {}).get("composite_score")

        report(30, "Building Target Index", ["Validating Target Trial"], f"Chunking & embedding with trial configuration (chunk_size={p_cfg.chunking_config.chunk_size})...")
        index_builder = IndexBuilder(self.workspace, self.emb_client)
        index = index_builder.build_index(docs, p_cfg.chunking_config)

        report(50, "Generating Independent Holdout Dataset", ["Validating Target Trial", "Building Target Index"], f"Synthesizing {test_size} unseen holdout questions...")
        holdout_dataset = self.build_or_load_holdout_dataset(docs, index.chunks, kb_checksum, test_size=test_size)

        report(70, "Executing Holdout Benchmark", ["Validating Target Trial", "Building Target Index", "Generating Independent Holdout Dataset"], f"Evaluating {len(holdout_dataset.items)} holdout items...")

        items_to_eval = holdout_dataset.items[:test_size]
        retrieval_engine = RetrievalEngine(self.emb_client)

        r_list: List[RetrievalMetrics] = []
        retrieved_results = []
        start_time = time.time()

        for item in items_to_eval:
            retrieved = retrieval_engine.retrieve(item.question, index, p_cfg.retriever_config)
            r_metrics = self.eval_engine.evaluate_retrieval(retrieved, item.chunk_id, item.ground_truth)
            r_list.append(r_metrics)
            context_str, citations = PromptBuilder.build_context_string(retrieved)
            retrieved_results.append((retrieved, r_metrics, context_str, citations))

        # Pass 1: Generate Answers using AIOrchestrator
        orchestrator = AIOrchestrator(self.workspace, self.llm_client)
        gen_tasks = []
        for i, item in enumerate(items_to_eval):
            context_str = retrieved_results[i][2]
            prompt = PromptBuilder.build_prompt(item.question, context_str)
            gen_tasks.append(
                Task(
                    task_id=f"gen_holdout_{i}",
                    task_type="generate_answer",
                    input_data=prompt
                )
            )

        orchestrator.execute_tasks(gen_tasks)

        answers = []
        for task in gen_tasks:
            if task.result and "answer" in task.result:
                answers.append(task.result["answer"])
            else:
                answers.append("Generation failed or timed out.")

        # Pass 2: Evaluate Answers using AIOrchestrator
        eval_tasks = []
        for i, item in enumerate(items_to_eval):
            eval_input = f"Question: {item.question}\nReference Answer (Ground Truth): {item.ground_truth}\nAssistant Answer: {answers[i]}"
            eval_tasks.append(
                Task(
                    task_id=f"eval_holdout_{i}",
                    task_type="evaluate_answer",
                    input_data=eval_input
                )
            )

        orchestrator.execute_tasks(eval_tasks)

        a_list: List[AnswerMetrics] = []
        sample_evals: List[Dict[str, Any]] = []

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
                "generated_answer": answers[i],
                "retrieved_chunks_count": len(retrieved),
                "hit_rate": r_metrics.hit_rate,
                "precision": r_metrics.precision,
                "accuracy": a_metrics.accuracy,
                "answer_correctness": a_metrics.answer_correctness,
                "faithfulness": a_metrics.faithfulness
            })

        n_r = max(1, len(r_list))
        avg_r_metrics = RetrievalMetrics(
            hit_rate=round(sum(r.hit_rate for r in r_list) / n_r, 4),
            precision=round(sum(r.precision for r in r_list) / n_r, 4),
            recall=round(sum(r.recall for r in r_list) / n_r, 4),
            mrr=round(sum(r.mrr for r in r_list) / n_r, 4),
            ndcg=round(sum(r.ndcg for r in r_list) / n_r, 4)
        )
        n_a = max(1, len(a_list))
        avg_corr = round(sum(getattr(a, "answer_correctness", getattr(a, "accuracy", getattr(a, "faithfulness", 0.0))) for a in a_list) / n_a, 4)
        avg_a_metrics = AnswerMetrics(
            accuracy=avg_corr,
            answer_correctness=avg_corr,
            faithfulness=avg_corr,
            completeness=round(sum(a.completeness for a in a_list) / n_a, 4),
            relevance=round(sum(a.relevance for a in a_list) / n_a, 4)
        )
        elapsed_ms = (time.time() - start_time) * 1000

        # Generalization Composite Score: (Hit Rate * 0.7) + (Answer Correctness * 0.3)
        gen_composite_score = round(
            float(avg_r_metrics.hit_rate * 0.7 + avg_a_metrics.answer_correctness * 0.3),
            4
        )

        score_delta: Optional[float] = None
        if opt_score is not None and isinstance(opt_score, (int, float)) and math.isfinite(opt_score):
            score_delta = round(float(gen_composite_score - opt_score), 4)

        # Generate summary description
        delta_str = f"Δ {score_delta:+.1%}" if score_delta is not None else "N/A"
        if score_delta is not None:
            if score_delta >= -0.03:
                generalization_assessment = "Robust generalizability with minimal performance drop across unseen holdout queries."
            elif score_delta >= -0.10:
                generalization_assessment = "Moderate generalization gap observed; candidate pipeline retains decent domain coverage."
            else:
                generalization_assessment = "Significant generalization drop detected; configuration exhibits potential overfitting to the optimization dataset."
        else:
            generalization_assessment = "Independent holdout benchmark completed successfully."

        summary_text = (
            f"Holdout Test (N={len(items_to_eval)}): Generalization Score {(gen_composite_score * 100):.1f}% "
            f"({delta_str} vs Opt {(opt_score * 100):.1f}% if available). "
            f"Hit Rate: {(avg_r_metrics.hit_rate * 100):.1f}%, Answer Correctness: {(avg_a_metrics.answer_correctness * 100):.1f}%. "
            f"{generalization_assessment}"
        )

        test_id = f"gen_test_{experiment_id}_{int(time.time())}"
        result = GeneralizationTestResult(
            test_id=test_id,
            experiment_id=experiment_id,
            kb_checksum=kb_checksum,
            test_size=len(items_to_eval),
            optimization_composite_score=opt_score,
            generalization_composite_score=gen_composite_score,
            score_delta=score_delta,
            retrieval_metrics=avg_r_metrics,
            answer_metrics=avg_a_metrics,
            avg_latency_ms=round(elapsed_ms / max(1, len(items_to_eval)), 1),
            total_tokens=orchestrator.metrics.get("total_tokens_used", 0),
            sample_evaluations=sample_evals,
            status="completed",
            metrics_valid=True,
            summary_text=summary_text,
            dataset_id=holdout_dataset.dataset_id,
            timestamp=datetime.now().isoformat()
        )

        report(95, "Saving Generalization Artifacts", ["Validating Target Trial", "Building Target Index", "Generating Independent Holdout Dataset", "Executing Holdout Benchmark"], "Writing isolated test results to workspace...")
        self.workspace.save_generalization_test(result)

        score_disp = f"{(gen_composite_score * 100):.1f}%"
        report(100, "Generalization Test Complete", ["Validating Target Trial", "Building Target Index", "Generating Independent Holdout Dataset", "Executing Holdout Benchmark", "Saving Generalization Artifacts"], f"Validation completed: Score {score_disp} ({delta_str}).")

        return result
