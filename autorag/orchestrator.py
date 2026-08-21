import json
import hashlib
import time
import re
from typing import List, Dict, Any, Optional

from .connections import OpenAICompatibleClient

class TokenBudgetManager:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        
        if "gpt-4o" in model_name:
            self.max_context = 128000
            self.max_output = 4096
        elif "gemini-1.5" in model_name or "gemini-2" in model_name:
            self.max_context = 2000000
            self.max_output = 8192
        elif "claude-3" in model_name:
            self.max_context = 200000
            self.max_output = 4096
        else:
            self.max_context = 32000
            self.max_output = 2048

        self.target_batch_size = min(40000, self.max_context // 2)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4 + 10

class Task:
    def __init__(self, task_id: str, task_type: str, input_data: str, expected_output_tokens: int = 500):
        self.task_id = task_id
        self.task_type = task_type
        self.input_data = input_data
        self.expected_output_tokens = expected_output_tokens
        self.result = None
        self.error = None
        self.estimated_input_tokens = len(input_data) // 4

class Batch:
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.tasks: List[Task] = []
        self.estimated_input_tokens = 0
        self.estimated_output_tokens = 0
        
    def can_add(self, task: Task, limit: int) -> bool:
        return self.estimated_input_tokens + task.estimated_input_tokens < limit
        
    def add(self, task: Task):
        self.tasks.append(task)
        self.estimated_input_tokens += task.estimated_input_tokens
        self.estimated_output_tokens += task.expected_output_tokens

class CacheManager:
    def __init__(self, workspace):
        self.workspace = workspace
        self.cache = {} 
        
    def get(self, key: str) -> Optional[Dict]:
        return self.cache.get(key)
        
    def set(self, key: str, value: Dict):
        self.cache[key] = value

class RequestPlanner:
    def __init__(self, budget_manager: TokenBudgetManager):
        self.budget_manager = budget_manager
        
    def plan_batches(self, tasks: List[Task], logger_fn=None) -> List[Batch]:
        def log(msg):
            if logger_fn:
                logger_fn(msg)
            else:
                print(msg)
                
        batches = []
        current_batch = Batch(f"batch_{len(batches)}")
        
        for task in tasks:
            if task.estimated_input_tokens > self.budget_manager.target_batch_size:
                if task.estimated_input_tokens > self.budget_manager.max_context - self.budget_manager.max_output - 1000:
                    task.error = "Input too large for context window."
                    log(f"[Batch Planner] Task {task.task_id} exceeds max context window. Skipping.")
                    continue
                oversized_batch = Batch(f"batch_{len(batches)}")
                oversized_batch.add(task)
                batches.append(oversized_batch)
                log(f"[Batch Planner] Decision: SINGLE OVERSIZED TASK | Reason: individual task {task.task_id} estimated tokens {task.estimated_input_tokens} exceeds target budget {self.budget_manager.target_batch_size}")
                continue
                
            if not current_batch.can_add(task, self.budget_manager.target_batch_size) and current_batch.tasks:
                batches.append(current_batch)
                current_batch = Batch(f"batch_{len(batches)}")
                
            current_batch.add(task)
            
        if current_batch.tasks:
            batches.append(current_batch)
            
        return batches

class PromptComposer:
    def compose(self, batch: Batch) -> str:
        prompt = "Perform the following tasks and return the results as a single JSON object. The keys of the JSON should be the Task IDs.\n\n"
        for task in batch.tasks:
            prompt += f"--- TASK ID: {task.task_id} ---\n"
            prompt += f"Task Type: {task.task_type}\n"
            prompt += f"Input:\n{task.input_data}\n\n"
            
            if task.task_type == "generate_qa":
                prompt += "Instruction: Create 2 clear, specific factual questions and concise ground truth answers based ONLY on the input text. Format: list of objects with 'question' and 'ground_truth'.\n"
            elif task.task_type == "evaluate_answer":
                prompt += "Instruction: Evaluate the provided answer against the ground truth. Format: object with 'accuracy', 'completeness', 'relevance' (scores 0.0 to 1.0) and 'reasoning'.\n"
            elif task.task_type == "generate_answer":
                prompt += f"Instruction: Generate an answer based ONLY on the provided context.\n"
            elif task.task_type == "semantic_chunk":
                prompt += f"Instruction: Split the following document into semantic chunks of roughly {task.input_data.split('|')[0]} characters each. Output a JSON array of strings.\n"
        
        prompt += "\nOUTPUT FORMAT:\n"
        prompt += "```json\n"
        prompt += "{\n"
        for i, task in enumerate(batch.tasks):
            if task.task_type == "generate_qa":
                prompt += f'  "{task.task_id}": [{{"question": "...", "ground_truth": "..."}}]'
            elif task.task_type == "evaluate_answer":
                prompt += f'  "{task.task_id}": {{"accuracy": 0.0, "completeness": 0.0, "relevance": 0.0, "reasoning": "..."}}'
            elif task.task_type == "generate_answer":
                prompt += f'  "{task.task_id}": {{"answer": "..."}}'
            elif task.task_type == "semantic_chunk":
                prompt += f'  "{task.task_id}": ["chunk1", "chunk2", "..."]'
            
            if i < len(batch.tasks) - 1:
                prompt += ",\n"
            else:
                prompt += "\n"
        prompt += "}\n"
        prompt += "```\n"
        return prompt

class ResponseParser:
    def parse(self, text: str, batch: Batch) -> bool:
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    data = json.loads(text)
                    
            success_count = 0
            for task in batch.tasks:
                if task.task_id in data:
                    task.result = data[task.task_id]
                    success_count += 1
                else:
                    task.error = "Missing from LLM response"
            return success_count > 0
        except Exception as e:
            for task in batch.tasks:
                task.error = f"Parse error: {e}"
            return False

def categorize_error(error_msg: str) -> dict:
    msg = error_msg.lower()
    if "403" in msg or "forbidden" in msg:
        return {"category": "FORBIDDEN", "retryable": False}
    if "401" in msg or "unauthorized" in msg:
        return {"category": "AUTH_ERROR", "retryable": False}
    if "timeout" in msg or "timed out" in msg or "timeout" in error_msg.lower():
        return {"category": "TIMEOUT", "retryable": True}
    if "closed connection" in msg or "connection reset" in msg or "connection refused" in msg or "remote disconnect" in msg:
        return {"category": "CONNECTION_ERROR", "retryable": True}
    if "429" in msg or "rate limit" in msg:
        return {"category": "RATE_LIMIT", "retryable": True}
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg or "server error" in msg:
        return {"category": "SERVER_ERROR", "retryable": True}
    return {"category": "UNKNOWN", "retryable": False}

def execute_with_retry(func, max_retries, logger, batch_name, *args, **kwargs):
    last_err = None
    retries = 0
    delays = [0, 2, 4, 8, 16]
    import time
    
    start_time = time.time()
    
    for attempt in range(max_retries):
        if attempt == 0:
            logger(f"[{batch_name} Request]\nAttempt: {attempt + 1}/{max_retries}")
        else:
            logger(f"[{batch_name} Request]\nAttempt: {attempt + 1}/{max_retries}")
            
        try:
            res = func(*args, **kwargs)
            if attempt > 0:
                latency = round(time.time() - start_time, 1)
                logger(f"...\nSuccess\nLatency: {latency} seconds")
            return res, retries
        except Exception as e:
            last_err = e
            error_category = categorize_error(str(e))
            if attempt < max_retries - 1:
                if error_category["retryable"]:
                    delay = delays[attempt + 1] if (attempt + 1) < len(delays) else 16
                    logger(f"...\n{str(e)}\nRetryable: Yes\nWaiting {delay} seconds before retry...\n--------------------------------")
                    retries += 1
                    time.sleep(delay)
                else:
                    logger(f"...\n{str(e)}\nRetryable: No\nError is non-retryable. Aborting batch.")
                    break
            else:
                logger(f"[AI Provider]\nAttempts: {max_retries}\nFinal Status: FAILED\nReason:\n{str(e)}\nTrial Status:\nFAILED")
                break
    raise last_err

class AIOrchestrator:
    def __init__(self, workspace, llm_client: OpenAICompatibleClient):
        self.workspace = workspace
        self.llm_client = llm_client
        self.budget_manager = TokenBudgetManager(llm_client.config.provider, llm_client.config.model_name)
        self.planner = RequestPlanner(self.budget_manager)
        self.composer = PromptComposer()
        self.parser = ResponseParser()
        self.cache = CacheManager(workspace)
        
        self.metrics = {
            "original_requests": 0,
            "optimized_requests": 0,
            "estimated_tokens_saved": 0,
            "cache_hits": 0,
            "total_tokens_used": 0,
            "total_execution_time": 0,
            "retry_count": 0
        }

    def execute_tasks(self, tasks: List[Task], logger_fn=None):
        def log(msg):
            if logger_fn:
                logger_fn(msg)
            else:
                print(msg)
                
        self.metrics["original_requests"] += len(tasks)
        
        uncached_tasks = []
        for task in tasks:
            cache_key = hashlib.md5(f"{task.task_type}_{task.input_data}".encode()).hexdigest()
            cached = self.cache.get(cache_key)
            if cached:
                task.result = cached
                self.metrics["cache_hits"] += 1
            else:
                task._cache_key = cache_key
                uncached_tasks.append(task)
                
        if not uncached_tasks:
            return

        batches = self.planner.plan_batches(uncached_tasks, logger_fn=log)
        self.metrics["optimized_requests"] += len(batches)
        
        self.metrics["estimated_tokens_saved"] += len(uncached_tasks) * 100 - len(batches) * 100
        
        for idx, batch in enumerate(batches):
            log(f"[Orchestrator] Executing Batch {idx+1}/{len(batches)} ({len(batch.tasks)} tasks, ~{batch.estimated_input_tokens} tokens)")
            prompt = self.composer.compose(batch)
            
            try:
                start_time = time.time()
                res, retries = execute_with_retry(
                    self.llm_client.chat_completion,
                    5,
                    log,
                    f"Batch {idx+1}",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    json_mode=True
                )
                self.metrics["retry_count"] += retries
                
                text = res.get("text", "")
                self.metrics["total_execution_time"] += (time.time() - start_time)
                
                usage = res.get("usage", {})
                self.metrics["total_tokens_used"] += usage.get("total_tokens", self.budget_manager.estimate_tokens(prompt) + self.budget_manager.estimate_tokens(text))
                
                success = self.parser.parse(text, batch)
                if success:
                    for task in batch.tasks:
                        if task.result is not None:
                            self.cache.set(task._cache_key, task.result)
                else:
                    log(f"[Orchestrator] Batch {idx+1} parsing failed.")
                    for task in batch.tasks:
                        task.error = "Parse error"
            except Exception as e:
                log(f"[Orchestrator] Batch {idx+1} failed: {e}")
                for task in batch.tasks:
                    task.error = str(e)

