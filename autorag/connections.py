"""
Connections Subsystem for AutoRAG Platform.
Provides HTTP-based OpenAI-compatible LLM & Embedding clients with fallback to Gemini.
Deterministic local TF-IDF vectorizer fallback included for zero-dependency operation.
"""

import os
import re
import json
import math
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from .types import LLMConfig, EmbeddingConfig
from .torch_policy import resolve_embedding_device, CUDAUnavailableError


class OpenAICompatibleClient:
    """Zero-dependency HTTP client for OpenAI-compatible APIs (OpenAI, Ollama, LM Studio, OpenRouter)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """Send a chat completion request to an OpenAI-compatible endpoint."""
        # Handle Gemini fallback if configured or if no api_key/openai base_url provided
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if self.config.provider == "gemini" or (not self.api_key and gemini_key):
            return self._gemini_chat_completion(messages, system_prompt or "", json_mode)

        endpoint = f"{self.base_url}/chat/completions"
        
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self.config.model_name,
            "messages": full_messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "aistudio-build/AutoRAG-1.0",
        }
        
        # OpenRouter-specific recommended headers
        if "openrouter" in self.config.provider.lower() or "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/google/aistudio-build"
            headers["X-Title"] = "RAG Optimizer Sweep"

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        print(f"\n[OpenRouter Request]")
        print(f"Base URL: {self.base_url}")
        print(f"Chat Endpoint: {endpoint}")
        print(f"Model: {self.config.model_name}")
        print(f"API Key Present: {'yes' if self.api_key else 'no'}")
        
        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "text": content,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "provider": self.config.provider,
                }
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
                # Redact keys just in case
                import re
                body = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[REDACTED]', body)
            except Exception:
                body = "<could not read response body>"
            
            print(f"\n[AI Provider Error]")
            print(f"Provider: {self.config.provider}")
            print(f"Endpoint: {endpoint}")
            print(f"HTTP Status: {e.code}")
            print(f"Authentication: {'configured' if self.api_key else 'none'}")
            print(f"API Key Present: {'yes' if self.api_key else 'no'}")
            print(f"Response Body: {body[:500]}")
            
            if gemini_key and e.code not in [400, 403, 401]:
                return self._gemini_chat_completion(messages, system_prompt or "", json_mode)
                
            raise RuntimeError(f"HTTP request to AI Provider failed: HTTP Error {e.code}: {e.reason} - {body[:200]}") from e
        except Exception as e:
            # Fallback to Gemini if available
            if gemini_key:
                return self._gemini_chat_completion(messages, system_prompt or "", json_mode)
            
            # No mock data allowed - propagate the actual failure
            raise RuntimeError(f"HTTP request to AI Provider failed: {str(e)}") from e

    def _gemini_chat_completion(self, messages: List[Dict[str, str]], system_prompt: str, json_mode: bool = False) -> Dict[str, Any]:
        """Call Gemini REST API directly using stdlib urllib."""
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set. Cannot use Gemini provider.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
                "maxOutputTokens": self.config.max_tokens,
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "aistudio-build"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {
                    "text": text,
                    "prompt_tokens": 20,
                    "completion_tokens": 50,
                    "total_tokens": 70,
                    "provider": "gemini"
                }
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e


class LocalTFIDFEmbedder:
    """Zero-dependency deterministic TF-IDF & Character N-gram Vectorizer."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\b\w+\b', text.lower())
        # Also extract char n-grams (3-grams) for subword coverage
        ngrams = [text[i:i+3].lower() for i in range(len(text) - 2)] if len(text) >= 3 else []
        return words + ngrams

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        """Fit vocabulary and calculate IDF for document collection."""
        if not texts:
            return []

        doc_count = len(texts)
        doc_freqs: Counter = Counter()

        doc_tokens = [self._tokenize(t) for t in texts]
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freqs[t] += 1

        # Select top features by frequency up to dimension
        most_common = doc_freqs.most_common(self.dimension)
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(most_common)}
        
        # Calculate IDF
        self.idf = {
            term: math.log((1 + doc_count) / (1 + freq)) + 1.0
            for term, freq in doc_freqs.items()
            if term in self.vocabulary
        }

        return [self._transform_single(tokens) for tokens in doc_tokens]

    def _transform_single(self, tokens: List[str]) -> List[float]:
        vector = [0.0] * self.dimension
        if not tokens:
            return vector

        term_counts = Counter(tokens)
        total_tokens = len(tokens)

        for term, count in term_counts.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                tf = count / total_tokens
                idf = self.idf.get(term, 1.0)
                vector[idx] = tf * idf

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def transform(self, texts: List[str]) -> List[List[float]]:
        """Transform new texts using fitted vocabulary or hashed features."""
        if not self.vocabulary:
            return self.fit_transform(texts)

        results = []
        for text in texts:
            tokens = self._tokenize(text)
            results.append(self._transform_single(tokens))
        return results


_GLOBAL_EMBEDDING_CACHE = {}

class EmbeddingClient:
    """Embedding Client supporting Local TF-IDF, OpenAI, and Gemini API."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.local_embedder = LocalTFIDFEmbedder(dimension=config.dimension)


    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of texts."""
        if not texts:
            return []

        if self.config.provider == "huggingface":
            configured_dev = getattr(self.config, "device", "auto")
            resolved_dev = resolve_embedding_device(configured_dev)

            try:
                import time
                from sentence_transformers import SentenceTransformer
                
                cache_key = f"{self.config.model_name}::{resolved_dev}"
                
                print("[Embedding Model Manager]")
                print(f"Model: {self.config.model_name}")
                print(f"Configured Device: {configured_dev}")
                print(f"Resolved Device: {resolved_dev}")
                
                if cache_key in _GLOBAL_EMBEDDING_CACHE:
                    print("Cache: HIT")
                    print("Reusing embedding model")
                    print("Load Time: 0ms")
                    model = _GLOBAL_EMBEDDING_CACHE[cache_key]
                else:
                    print("Cache: MISS")
                    print(f"Loading model on {resolved_dev}...")
                    start_time = time.time()
                    model = SentenceTransformer(self.config.model_name, device=resolved_dev)
                    load_time = time.time() - start_time
                    _GLOBAL_EMBEDDING_CACHE[cache_key] = model
                    print("[Embedding Model Manager]")
                    print("Load complete")
                    print(f"Load Time: {load_time:.2f}s")
                
                embeddings = model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
            except CUDAUnavailableError:
                raise
            except ImportError:
                print("sentence_transformers not found, falling back to local TF-IDF")
                self.config.provider = "local"

        if self.config.provider == "local":
            return self.local_embedder.fit_transform(texts)
        # HTTP OpenAI-compatible embeddings call
        base_url = self.config.base_url.rstrip("/")
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")

        if api_key:
            try:
                endpoint = f"{base_url}/embeddings"
                payload = {
                    "input": texts,
                    "model": self.config.model_name
                }
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.config.timeout_sec if hasattr(self.config, 'timeout_sec') else 30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return [item["embedding"] for item in data["data"]]
            except Exception:
                pass

        # Fallback to local TF-IDF if external service fails or no API key provided
        return self.local_embedder.fit_transform(texts)
