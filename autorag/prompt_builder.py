"""Prompt Builder Subsystem for AutoRAG Platform.
Assembles prompts from contexts and templates. Never calls LLM directly."""

from typing import List, Tuple, Dict, Any
from .types import Chunk

class PromptBuilder:
    """Assembles context-aware prompts deterministically."""

    @staticmethod
    def build_context_string(retrieved_chunks: List[Tuple[Chunk, float]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Convert retrieved chunks into a formatted context block and extract citations."""
        context_parts = []
        citations = []
        for idx, (chunk, score) in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {idx} - Document: {chunk.filename} (Score: {score:.3f})]:\n{chunk.text}"
            )
            citations.append({
                "source_num": idx,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "score": score,
                "snippet": chunk.text[:150]
            })
        context_str = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        return context_str, citations

    @staticmethod
    def build_prompt(query: str, context_str: str) -> str:
        """Combine query and context into a deterministic prompt."""
        return (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Please answer the question accurately using ONLY the provided context."
        )
