"""
Index Builder Subsystem for AutoRAG Platform.
Handles document chunking strategies, embedding generation, vector index construction, and artifact caching.
"""

import hashlib
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .types import Document, Chunk, ChunkingConfig, EmbeddingConfig
from .connections import EmbeddingClient
from .workspace import WorkspaceManager


class TextChunker:
    """Implements multiple deterministic text chunking strategies."""

    @staticmethod
    def chunk_document(doc: Document, config: ChunkingConfig, llm_client=None) -> List[Chunk]:
        text = doc.content
        size = max(50, config.chunk_size)
        overlap = min(config.chunk_overlap, size - 10)
        
        raw_chunks: List[Tuple[str, int, int]] = []

        if getattr(config, "split_method", config.strategy) == "fixed":
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                raw_chunks.append((text[start:end], start, end))
                if end == len(text):
                    break
                start += (size - overlap)

        elif getattr(config, "split_method", config.strategy) == "paragraph":
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current = ""
            start_idx = 0
            for p in paragraphs:
                if len(current) + len(p) <= size:
                    current = f"{current}\n\n{p}".strip() if current else p
                else:
                    if current:
                        raw_chunks.append((current, start_idx, start_idx + len(current)))
                        start_idx += len(current)
                    current = p
            if current:
                raw_chunks.append((current, start_idx, start_idx + len(current)))

        elif getattr(config, "split_method", config.strategy) == "semantic" and llm_client is not None:
            # LLM-assisted semantic splitting via orchestrator
            from .orchestrator import AIOrchestrator, Task
            # We can't easily pass the workspace here unless it's available, but we can pass None if it's fine.
            # Actually we can just run chat_completion directly since it's just one chunking call per document.
            # But let's adapt it to use a Task so it fits the pattern, or just leave it.
            # I will leave it using chat_completion for now to avoid breaking the signature of chunk_document.
            prompt = f"Split the following document into semantic chunks of roughly {size} characters each. Output a JSON array of strings.\n\n{text}"
            try:
                res = llm_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt="You are a JSON chunking engine. Output ONLY a valid JSON array of strings.",
                    temperature=0.1
                )
                import json
                chunk_texts = json.loads(res.get("text", "[]"))
                if not chunk_texts:
                    raise ValueError("Empty chunks")
                start_idx = 0
                for c in chunk_texts:
                    raw_chunks.append((c, start_idx, start_idx + len(c)))
                    start_idx += len(c)
            except Exception as e:
                raise RuntimeError(f"Semantic chunking failed during LLM call: {str(e)}") from e

        elif getattr(config, "split_method", config.strategy) == "sentence" or (getattr(config, "split_method", config.strategy) == "semantic" and llm_client is None):
            # Sentence boundary splitting with size accumulation
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            current = ""
            start_idx = 0
            for sent in sentences:
                if len(current) + len(sent) <= size:
                    current = f"{current} {sent}".strip() if current else sent
                else:
                    if current:
                        raw_chunks.append((current, start_idx, start_idx + len(current)))
                        start_idx += len(current)
                    current = sent
            if current:
                raw_chunks.append((current, start_idx, start_idx + len(current)))

        else:  # "recursive" default
            separators = ["\n\n", "\n", ". ", " ", ""]
            def _split(t: str, start: int, sep_idx: int = 0) -> List[Tuple[str, int, int]]:
                if len(t) <= size or sep_idx >= len(separators):
                    if len(t) <= size:
                        return [(t, start, start + len(t))]
                    step = max(20, size - overlap)
                    return [(t[i:i+size], start+i, start+i+len(t[i:i+size])) for i in range(0, len(t), step)]

                sep = separators[sep_idx]
                if sep in t:
                    parts = t.split(sep)
                    res = []
                    curr = ""
                    c_start = start
                    for p in parts:
                        piece = p + sep if sep != "" else p
                        if len(curr) + len(piece) <= size:
                            curr += piece
                        else:
                            if curr:
                                if len(curr) == len(t):
                                    res.extend(_split(curr, c_start, sep_idx + 1))
                                else:
                                    res.extend(_split(curr, c_start, 0))
                                c_start += len(curr)
                            curr = piece
                    if curr:
                        if len(curr) == len(t):
                            res.extend(_split(curr, c_start, sep_idx + 1))
                        else:
                            res.extend(_split(curr, c_start, 0))
                    return res
                else:
                    return _split(t, start, sep_idx + 1)

            raw_chunks = _split(text, 0)

        chunks: List[Chunk] = []
        for idx, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            chunk_id = hashlib.md5(f"{doc.doc_id}:{idx}:{start_char}".encode("utf-8")).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,

                    chunk_index=idx,
                    metadata={
                        "source_file": doc.filepath,
                        "position": start_char,
                        "page": 1, # default, text docs don't have page numbers
                        "chunk_id": chunk_id,
                        "file_type": doc.file_type, 
                        "doc_checksum": doc.checksum
                    }
                )
            )

        return chunks


class VectorIndex:
    """In-memory & cache-backed Vector Index."""
    def __init__(self, chunks: List[Chunk], vectors: List[List[float]], embedding_config: EmbeddingConfig, index_type: str = "memory"):
        self.chunks = chunks
        self.vectors = vectors
        self.embedding_config = embedding_config
        self.index_type = index_type
        self._faiss_index = None
        self._chroma_collection = None
        
        if index_type == "faiss":
            self._build_faiss()
        elif index_type == "chroma":
            self._build_chroma()

    def _build_faiss(self):
        try:
            import faiss
            import numpy as np
            if not self.vectors:
                return
            dim = len(self.vectors[0])
            self._faiss_index = faiss.IndexFlatL2(dim)
            self._faiss_index.add(np.array(self.vectors).astype('float32'))
        except ImportError:
            pass

    def _build_chroma(self):
        try:
            import chromadb
            if not self.vectors:
                return
            client = chromadb.Client()
            self._chroma_collection = client.create_collection("temp_rag")
            ids = [str(i) for i in range(len(self.chunks))]
            self._chroma_collection.add(
                embeddings=self.vectors,
                documents=[c.text for c in self.chunks],
                ids=ids
            )
        except ImportError:
            pass


class IndexBuilder:
    """Builds Vector Index from documents and chunking configurations."""

    def __init__(self, workspace: WorkspaceManager, embedding_client: EmbeddingClient, llm_client=None):
        self.llm_client = llm_client
        self.workspace = workspace
        self.embedding_client = embedding_client

    def build_index(self, documents: List[Document], chunking_config: ChunkingConfig, index_type: str = "memory", pipeline_id: str = None, kb_checksum: str = "") -> VectorIndex:
        """Chunk documents, embed chunks, and build index."""
        # Pipeline aware artifact caching
        chunking_hash = hashlib.sha256(f"{chunking_config.get_hash()}_{kb_checksum}".encode()).hexdigest()
        
        # 1. Chunks
        all_chunks: List[Chunk] = []
        cached_chunks = self.workspace.get_cached_item(f"chunks_{chunking_hash}")
        if cached_chunks:
            all_chunks = [Chunk(**c) for c in cached_chunks]
        else:
            if getattr(chunking_config, "split_method", chunking_config.strategy) == "semantic" and self.llm_client is not None:
                from .orchestrator import AIOrchestrator, Task
                orchestrator = AIOrchestrator(self.workspace, self.llm_client)
                tasks = []
                size = max(50, chunking_config.chunk_size)
                for doc in documents:
                    tasks.append(Task(task_id=f"chunk_{doc.doc_id}", task_type="semantic_chunk", input_data=f"{size}|{doc.content}"))
                
                orchestrator.execute_tasks(tasks)
                
                for task, doc in zip(tasks, documents):
                    if task.result and isinstance(task.result, list):
                        start_idx = 0
                        for c in task.result:
                            all_chunks.append(Chunk(
                                chunk_id=hashlib.md5(f"{doc.doc_id}_{start_idx}".encode()).hexdigest(),
                                doc_id=doc.doc_id,
                                text=c,
                                chunk_index=len([ch for ch in all_chunks if ch.doc_id == doc.doc_id]),
                                metadata={"filename": doc.filename, "start_idx": start_idx, "end_idx": start_idx + len(c)},
                                filename=doc.filename
                            ))
                            start_idx += len(c)
                    else:
                        # Fallback to single chunking if semantic fails
                        all_chunks.extend(TextChunker.chunk_document(doc, chunking_config, self.llm_client))
            else:
                for doc in documents:
                    chunks = TextChunker.chunk_document(doc, chunking_config, self.llm_client)
                    all_chunks.extend(chunks)
                    
            if all_chunks:
                self.workspace.set_cached_item(f"chunks_{chunking_hash}", [c.__dict__ for c in all_chunks])
                self.workspace.save_artifact("chunks", [c.__dict__ for c in all_chunks], pipeline_id)
                
        if not all_chunks:
            return VectorIndex([], [], self.embedding_client.config)

        # 2. Embeddings
        embedding_hash = hashlib.sha256(f"{self.embedding_client.config.get_hash()}_{chunking_hash}".format().encode()).hexdigest()
        texts = [c.text for c in all_chunks]
        
        vectors = self.workspace.get_cached_item(f"embeddings_{embedding_hash}")
        if not vectors or len(vectors) != len(texts):
            vectors = self.embedding_client.embed_texts(texts)
            self.workspace.set_cached_item(f"embeddings_{embedding_hash}", vectors)
            self.workspace.save_artifact("embeddings", vectors, pipeline_id)

        return VectorIndex(all_chunks, vectors, self.embedding_client.config, index_type=index_type)
