"""Generation Engine Subsystem for AutoRAG Platform.
Invokes LLM generation via OpenAI-compatible endpoints."""

import time
from typing import List, Dict, Any
from .types import PipelineConfig
from .connections import OpenAICompatibleClient

class GenerationEngine:
    """Executes RAG context-aware answer generation."""

    def __init__(self, llm_client: OpenAICompatibleClient):
        self.llm_client = llm_client

    def generate_answer(
        self, prompt: str, config: PipelineConfig
    ) -> Dict[str, Any]:
        """Generate answer from assembled prompt."""
        start_time = time.time()
        
        res = self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=config.system_prompt,
            temperature=config.llm_config.temperature,
            max_tokens=config.llm_config.max_tokens
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "answer": res.get("text", ""),
            "latency_ms": latency_ms,
            "tokens": res.get("total_tokens", 0),
            "provider": res.get("provider", "unknown")
        }
