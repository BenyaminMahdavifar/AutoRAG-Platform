"""
Retrieval Engine Subsystem for AutoRAG Platform.
Provides Dense, Sparse (BM25/TF-IDF), and Hybrid retrieval algorithms.
Supports Cosine, Dot Product, and Euclidean similarity metrics.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter

from .types import Chunk, RetrieverConfig
from .index_builder import VectorIndex
from .connections import EmbeddingClient


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def dot_product(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


def euclidean_distance_score(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    # Convert distance to similarity score in [0, 1]
    return 1.0 / (1.0 + dist)


class BM25Scorer:
    """Deterministic BM25 Sparse Keyword Scorer."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_tokens = [re.findall(r'\b\w+\b', doc.lower()) for doc in corpus]
        self.avgdl = sum(len(d) for d in self.doc_tokens) / max(1, self.corpus_size)

        self.doc_freqs: Counter = Counter()
        for d in self.doc_tokens:
            for t in set(d):
                self.doc_freqs[t] += 1

    def score(self, query: str) -> List[float]:
        q_tokens = re.findall(r'\b\w+\b', query.lower())
        scores = [0.0] * self.corpus_size

        for q in q_tokens:
            df = self.doc_freqs.get(q, 0)
            if df == 0:
                continue
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for idx, doc in enumerate(self.doc_tokens):
                tf = doc.count(q)
                if tf > 0:
                    doc_len = len(doc)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom

        # Normalize BM25 scores to [0, 1]
        max_s = max(scores) if scores and max(scores) > 0 else 1.0
        return [s / max_s for s in scores]



def maximal_marginal_relevance(
    query_vector: List[float],
    doc_vectors: List[List[float]],
    top_k: int,
    lambda_mult: float = 0.5
) -> List[int]:
    """Calculate MMR and return indices."""
    if not doc_vectors:
        return []
        
    # Calculate similarity of all docs to query
    q_sims = [cosine_similarity(query_vector, v) for v in doc_vectors]
    
    # Calculate similarity between all docs
    doc_sims = []
    for i in range(len(doc_vectors)):
        row = []
        for j in range(len(doc_vectors)):
            row.append(cosine_similarity(doc_vectors[i], doc_vectors[j]))
        doc_sims.append(row)
        
    selected = []
    unselected = list(range(len(doc_vectors)))
    
    while len(selected) < min(top_k, len(doc_vectors)):
        best_score = -float("inf")
        best_idx = -1
        
        for i in unselected:
            sim_to_query = q_sims[i]
            sim_to_selected = max([doc_sims[i][j] for j in selected]) if selected else 0.0
            
            score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_idx != -1:
            selected.append(best_idx)
            unselected.remove(best_idx)
        else:
            break
            
    return selected

class RetrievalEngine:

    """Executes retrieval according to RetrieverConfig strategy."""

    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client

    def retrieve(
        self, query: str, index: VectorIndex, config: RetrieverConfig
    ) -> List[Tuple[Chunk, float]]:
        """Retrieve top-K matching chunks with similarity scores."""
        if not index.chunks:
            return []

        # 1. Dense Vector Scoring
        query_vector = self.embedding_client.embed_texts([query])[0]
        dense_scores = []
        
        if getattr(index, "index_type", "memory") == "faiss" and index._faiss_index is not None:
            import numpy as np
            k = min(config.top_k * 2, len(index.vectors))
            distances, indices = index._faiss_index.search(np.array([query_vector]).astype('float32'), k)
            dense_scores = [0.0] * len(index.chunks)
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    # convert distance to score
                    dense_scores[idx] = max(0.0, 1.0 / (1.0 + distances[0][i]))
        elif getattr(index, "index_type", "memory") == "chroma" and index._chroma_collection is not None:
            k = min(config.top_k * 2, len(index.vectors))
            res = index._chroma_collection.query(query_embeddings=[query_vector], n_results=k)
            dense_scores = [0.0] * len(index.chunks)
            for i, idx_str in enumerate(res["ids"][0]):
                dist = res["distances"][0][i]
                idx = int(idx_str)
                dense_scores[idx] = max(0.0, 1.0 / (1.0 + dist))
        else:
            for v in index.vectors:
                if config.distance_metric == "cosine":
                    score = cosine_similarity(query_vector, v)
                elif config.distance_metric == "dot":
                    score = dot_product(query_vector, v)
                else:  # "euclidean"
                    score = euclidean_distance_score(query_vector, v)
                dense_scores.append(max(0.0, min(1.0, score)))

        # 2. Sparse Scoring
        bm25 = BM25Scorer([c.text for c in index.chunks])
        sparse_scores = bm25.score(query)

        # 3. Combine scores based on strategy
        final_scored_chunks: List[Tuple[Chunk, float]] = []
        for idx, chunk in enumerate(index.chunks):
            if config.strategy == "dense":
                score = dense_scores[idx]
            elif config.strategy == "sparse":
                score = sparse_scores[idx]
            else:  # "hybrid"
                alpha = config.hybrid_alpha
                score = alpha * dense_scores[idx] + (1.0 - alpha) * sparse_scores[idx]

            if score >= config.score_threshold:
                final_scored_chunks.append((chunk, score))

# Sort by score descending
        if getattr(config, "use_mmr", False) and config.strategy == "dense":
            # Pass vectors associated with the filtered docs to MMR
            doc_vectors = [index.vectors[index.chunks.index(c)] for c, s in final_scored_chunks]
            mmr_indices = maximal_marginal_relevance(query_vector, doc_vectors, config.top_k)
            final_scored_chunks = [final_scored_chunks[i] for i in mmr_indices]
        else:
            final_scored_chunks.sort(key=lambda x: x[1], reverse=True)
            final_scored_chunks = final_scored_chunks[:config.top_k]
            
        return final_scored_chunks
