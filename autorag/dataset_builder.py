"""
Dataset Builder Subsystem for AutoRAG Platform.
Generates evaluation Q&A datasets from documents.
Handles dataset versioning, caching, and reuse based on KB checksum.
"""

import hashlib
import json
import re
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .types import Document, Chunk, Dataset, DatasetItem
from .connections import OpenAICompatibleClient
from .workspace import WorkspaceManager


class DatasetBuilder:
    """Builds and manages Q&A evaluation datasets from document collection."""

    def __init__(self, workspace: WorkspaceManager, llm_client: Optional[OpenAICompatibleClient] = None, logger: Optional[Any] = None):
        self.workspace = workspace
        self.llm_client = llm_client
        self.logger = logger

    def log(self, message: str):
        if self.logger:
            self.logger(message)
        else:
            print(f"[DatasetBuilder] {message}", file=sys.stderr)

    def generate_questions_from_chunk(self, chunk: Chunk) -> List[Tuple[str, str]]:
        """Extract synthetic Q&A pairs from a chunk deterministically or via LLM."""
        from typing import Tuple
        
        pairs: List[Tuple[str, str]] = []

        # If LLM client is available, attempt LLM question generation
        if self.llm_client:
            prompt = (
                f"Given the following text, create 2 clear, specific factual questions "
                f"and concise ground truth answers based ONLY on the text.\n\n"
                f"Text:\n{chunk.text}\n\n"
                f"Format output strictly as JSON array:\n"
                f'[{{\"question\": \"...\", \"ground_truth\": \"...\"}}]'
            )
            try:
                res = self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                text = res.get("text", "")
                self.log(f"LLM Response: {text[:200]}...") # Log response
                json_match = re.search(r'\[.*\]', text, re.DOTALL)
                if json_match:
                    items = json.loads(json_match.group(0))
                    for item in items:
                        if "question" in item and "ground_truth" in item:
                            pairs.append((item["question"], item["ground_truth"]))
                            self.log(f"Generated Q: {item['question']}")
                else:
                    raise ValueError("No JSON array found in LLM response.")
            except Exception as e:
                import sys
                self.log(f"LLM Generation Failed: {e}")
                raise RuntimeError(f"LLM Dataset Generation Failed: {e}") from e
        else:
            raise RuntimeError("LLM Client is required to build a dataset. No mock data allowed.")

        # Deterministic fallback question generation removed (No Mock Data)
        if not pairs:
            raise ValueError(f"Failed to extract any Q&A pairs from chunk {chunk.chunk_id}.")

        return pairs

    def build_dataset(self, documents: List[Document], chunks: List[Chunk], kb_checksum: str, force_regenerate: bool = False) -> Dataset:
        """Build or load a cached Q&A evaluation dataset."""
        import time
        start_time = time.time()
        
        # 1. Check if an existing dataset matches the kb_checksum
        if not force_regenerate:
            existing = self.workspace.load_latest_dataset(kb_checksum=kb_checksum)
            if existing:
                items = [DatasetItem(**item) for item in existing.get("items", [])]
                return Dataset(
                    dataset_id=existing.get("dataset_id", "cached_ds"),
                    created_at=existing.get("created_at", datetime.now().isoformat()),
                    version=existing.get("version", "1.0"),
                    kb_checksum=kb_checksum,
                    items=items
                )

        # 2. Build new synthetic dataset
        items: List[DatasetItem] = []
        item_counter = 0

        # Sample chunks across documents for balanced coverage
        sampled_chunks = chunks[:min(20, len(chunks))]
        
        # --- Batching with Orchestrator ---
        if self.llm_client:
            from .orchestrator import AIOrchestrator, Task
            
            orchestrator = AIOrchestrator(self.workspace, self.llm_client)
            tasks = []
            
            for chunk in sampled_chunks:
                tasks.append(
                    Task(
                        task_id=f"task_{chunk.chunk_id}",
                        task_type="generate_qa",
                        input_data=chunk.text
                    )
                )
            
            orchestrator.execute_tasks(tasks, logger_fn=self.log)
            
            for task, chunk in zip(tasks, sampled_chunks):
                if task.result:
                    for item in task.result:
                        if "question" in item and "ground_truth" in item:
                            item_counter += 1
                            item_id = f"ds_item_{item_counter:03d}"
                            items.append(
                                DatasetItem(
                                    item_id=item_id,
                                    doc_id=chunk.doc_id,
                                    chunk_id=chunk.chunk_id,
                                    question=item["question"],
                                    ground_truth=item["ground_truth"], reference_answer=item["ground_truth"], gold_context=chunk.text,
                                    metadata={"filename": chunk.filename, "chunk_index": chunk.chunk_index}
                                )
                            )
                            self.log(f"Generated Q: {item['question']}")
                else:
                    self.log(f"Failed to generate QA for chunk {chunk.chunk_id}: {task.error}")
            
            if hasattr(orchestrator, "metrics"):
                setattr(self, "_last_orchestrator_metrics", orchestrator.metrics)
        else:
            raise RuntimeError("LLM Client is required to build a dataset. No mock data allowed.")

        generation_duration = time.time() - start_time
        dataset_id = hashlib.md5(f"dataset_{kb_checksum[:12]}_{time.time()}".encode("utf-8")).hexdigest()[:10]
        
        # Include transparency metadata
        provider = self.llm_client.config.provider if self.llm_client else "unknown"
        model_name = self.llm_client.config.model_name if self.llm_client else "unknown"
        
        dataset = Dataset(
            dataset_id=dataset_id,
            created_at=datetime.now().isoformat(),
            version="1.0",
            kb_checksum=kb_checksum,
            items=items
        )
        # Store metadata inside the dataset object (we can attach it to the dict when saving)
        execution_metadata = {
            "generation_duration_sec": round(generation_duration, 2),
            "model_used": model_name,
            "provider": provider,
            "questions_count": len(items),
            "answers_count": len(items)
        }
        
        if hasattr(self, "_last_orchestrator_metrics"):
            execution_metadata["orchestrator_metrics"] = self._last_orchestrator_metrics
            
        setattr(dataset, "_execution_metadata", execution_metadata)

        # Save to workspace
        self.workspace.save_dataset(dataset)
        return dataset
