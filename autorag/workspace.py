"""
Workspace subsystem for AutoRAG Platform.
Manages project folders, caching, file persistence, and artifact lifecycle.
"""

import os
import json
import math
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .types import Experiment, Dataset, KnowledgeBaseManifest, PipelineConfig


def _sanitize_floats(obj: Any) -> Any:
    """Recursively sanitize data structure replacing non-finite floats with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def normalize_experiment_record(experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure experiment dictionary conforms to canonical schema with consistent composite_score and metric semantics."""
    data = _sanitize_floats(dict(experiment_data))
    
    results = data.get("results")
    if isinstance(results, dict):
        results = dict(results)
        status = results.get("status", data.get("status", "completed"))
        metrics_valid = results.get("metrics_valid", True)
        failed_batches = results.get("failed_batches", 0)
        
        # Synchronize answer metrics across answer_correctness / accuracy / faithfulness
        ans_m = results.get("answer_metrics")
        if isinstance(ans_m, dict):
            ans_m = dict(ans_m)
            ans_val = ans_m.get("answer_correctness")
            if ans_val is None:
                ans_val = ans_m.get("accuracy")
            if ans_val is None:
                ans_val = ans_m.get("faithfulness")
            if ans_val is not None and isinstance(ans_val, (int, float)) and math.isfinite(ans_val):
                ans_m["answer_correctness"] = round(float(ans_val), 4)
                if "accuracy" not in ans_m or ans_m["accuracy"] is None:
                    ans_m["accuracy"] = round(float(ans_val), 4)
                if "faithfulness" not in ans_m or ans_m["faithfulness"] is None:
                    ans_m["faithfulness"] = round(float(ans_val), 4)
            results["answer_metrics"] = ans_m

        if metrics_valid is False or status in ("failed", "partial_failed") or failed_batches > 0:
            results["metrics_valid"] = False
            results["composite_score"] = None
            data["composite_score"] = None
            data["status"] = status if status in ("failed", "partial_failed") else "failed"
        else:
            score = results.get("composite_score")
            if score is None:
                score = results.get("_composite_score")
            if score is None:
                score = data.get("composite_score")
            if score is None and results.get("retrieval_metrics") and isinstance(results.get("answer_metrics"), dict):
                r_hit = results.get("retrieval_metrics", {}).get("hit_rate", 0.0)
                a_corr = results.get("answer_metrics", {}).get("answer_correctness", results.get("answer_metrics", {}).get("faithfulness", 0.0))
                if isinstance(r_hit, (int, float)) and isinstance(a_corr, (int, float)):
                    score = round(0.7 * float(r_hit) + 0.3 * float(a_corr), 4)
            if score is not None and isinstance(score, (int, float)) and math.isfinite(score):
                score = round(float(score), 4)
            else:
                score = None
            results["composite_score"] = score
            data["composite_score"] = score
            
        # Remove legacy alias keys
        results.pop("_composite_score", None)
        results.pop("overall_score", None)
        results.pop("score", None)
        data["results"] = results
    else:
        score = data.get("composite_score")
        if score is not None and isinstance(score, (int, float)) and math.isfinite(score):
            score = round(float(score), 4)
        else:
            score = None
        data["composite_score"] = score
        
    data.pop("_composite_score", None)
    data.pop("overall_score", None)
    data.pop("score", None)
    return data


class WorkspaceManager:
    """Manages project directories and persistent state."""

    def __init__(self, root_dir: str = "workspace"):
        self.root_dir = Path(root_dir)
        self.cache_dir = self.root_dir / "cache"
        self.datasets_dir = self.root_dir / "datasets"
        self.experiments_dir = self.root_dir / "experiments"
        self.reports_dir = self.root_dir / "reports"
        self.configs_dir = self.root_dir / "configs"
        self.kb_dir = self.root_dir / "kb"
        self.generalization_dir = self.root_dir / "generalization_tests"

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create workspace directory structure if missing."""
        for d in [
            self.root_dir,
            self.cache_dir,
            self.datasets_dir,
            self.experiments_dir,
            self.reports_dir,
            self.configs_dir,
            self.kb_dir,
            self.generalization_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def get_kb_path(self) -> Path:
        return self.kb_dir

    def get_cache_path(self) -> Path:
        return self.cache_dir

    def get_datasets_path(self) -> Path:
        return self.datasets_dir

    def get_experiments_path(self) -> Path:
        return self.experiments_dir

    def get_reports_path(self) -> Path:
        return self.reports_dir

    def get_configs_path(self) -> Path:
        return self.configs_dir

    def get_artifacts_path(self) -> Path:
        artifacts_dir = self.root_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir

    # Cache Operations
    def get_cached_item(self, cache_key: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set_cached_item(self, cache_key: str, data: Any) -> None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # Manifest Operations
    def save_manifest(self, manifest: KnowledgeBaseManifest) -> None:
        manifest_file = self.cache_dir / "kb_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.__dict__, f, indent=2)

    def load_manifest(self) -> Optional[KnowledgeBaseManifest]:
        manifest_file = self.cache_dir / "kb_manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return KnowledgeBaseManifest(**data)
            except Exception:
                return None
        return None

    # Dataset Operations
    def save_dataset(self, dataset: Dataset) -> None:
        filepath = self.datasets_dir / f"dataset_v{dataset.version}_{dataset.dataset_id[:8]}.json"
        data = {
            "dataset_id": dataset.dataset_id,
            "created_at": dataset.created_at,
            "version": dataset.version,
            "kb_checksum": dataset.kb_checksum,
            "items": [item.__dict__ for item in dataset.items],
        }
        if hasattr(dataset, "_execution_metadata"):
            data["execution_metadata"] = getattr(dataset, "_execution_metadata")
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_latest_dataset(self, kb_checksum: Optional[str] = None) -> Optional[Dict[str, Any]]:
        dataset_files = sorted(self.datasets_dir.glob("dataset_*.json"), key=os.path.getmtime, reverse=True)
        for filepath in dataset_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Only consider datasets that actually have items
                    if not data.get("items"):
                        continue
                    if kb_checksum is None or data.get("kb_checksum") == kb_checksum:
                        return data
            except Exception:
                continue
        return None

    # Experiment Operations
    def save_experiment(self, experiment_data: Any, *args, **kwargs) -> None:
        if isinstance(experiment_data, str) and args:
            exp_id = experiment_data
            p_cfg = args[0] if len(args) > 0 else kwargs.get("config")
            results = args[1] if len(args) > 1 else kwargs.get("results")
            record = {
                "experiment_id": exp_id,
                "config": p_cfg.__dict__ if hasattr(p_cfg, "__dict__") else (p_cfg or {}),
                "results": results.__dict__ if hasattr(results, "__dict__") else (results or {}),
                "kb_checksum": kwargs.get("kb_checksum", ""),
                "dataset_id": kwargs.get("dataset_id", ""),
                "optimization_run_id": kwargs.get("optimization_run_id", "")
            }
            if hasattr(results, "composite_score"):
                record["composite_score"] = results.composite_score
            experiment_data = record
        elif hasattr(experiment_data, "__dict__") and not isinstance(experiment_data, dict):
            experiment_data = experiment_data.__dict__

        normalized = normalize_experiment_record(experiment_data)
        exp_id = normalized.get("experiment_id") or f"exp_{int(datetime.now().timestamp())}"
        filepath = self.experiments_dir / f"{exp_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, default=str, allow_nan=False)

    def list_experiments(self, kb_checksum: Optional[str] = None) -> List[Dict[str, Any]]:
        experiments = []
        for filepath in sorted(self.experiments_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    normalized = normalize_experiment_record(data)
                    if kb_checksum:
                        exp_kb = normalized.get("kb_checksum") or normalized.get("metadata", {}).get("kb_checksum")
                        if exp_kb and exp_kb != kb_checksum:
                            continue
                    
                    # Attach generalization test if available
                    exp_id = normalized.get("experiment_id")
                    if exp_id and "generalization_test" not in normalized:
                        gen_test = self.get_generalization_test(exp_id)
                        if gen_test:
                            normalized["generalization_test"] = gen_test
                    experiments.append(normalized)
            except Exception:
                continue
        
        def _get_sort_key(e):
            score = e.get("composite_score")
            if score is None and isinstance(e.get("results"), dict):
                score = e.get("results", {}).get("composite_score")
            if isinstance(score, (int, float)) and math.isfinite(score):
                return float(score)
            return -1.0

        experiments.sort(key=_get_sort_key, reverse=True)
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        filepath = self.experiments_dir / f"{experiment_id}.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    normalized = normalize_experiment_record(data)
                    if "generalization_test" not in normalized:
                        gen_test = self.get_generalization_test(experiment_id)
                        if gen_test:
                            normalized["generalization_test"] = gen_test
                    return normalized
            except Exception:
                return None
        return None

    # Generalization Test Operations
    def save_generalization_test(self, test_data: Any) -> None:
        """Save a generalization test result independently from optimization trial metrics."""
        if hasattr(test_data, "__dict__") and not isinstance(test_data, dict):
            from dataclasses import asdict
            data = asdict(test_data)
        else:
            data = dict(test_data)
            
        data = _sanitize_floats(data)
        exp_id = data.get("experiment_id")
        if not exp_id:
            raise ValueError("experiment_id is required to save generalization test result.")
            
        filepath = self.generalization_dir / f"{exp_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            
        # Also attach to experiment file without modifying optimization composite score
        exp_file = self.experiments_dir / f"{exp_id}.json"
        if exp_file.exists():
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    exp_record = json.load(f)
                exp_record["generalization_test"] = data
                with open(exp_file, "w", encoding="utf-8") as f:
                    json.dump(exp_record, f, indent=2, default=str)
            except Exception:
                pass

    def get_generalization_test(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve generalization test result for a given experiment ID."""
        filepath = self.generalization_dir / f"{experiment_id}.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return _sanitize_floats(data)
            except Exception:
                return None
        return None

    def list_generalization_tests(self, kb_checksum: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all generalization test results, optionally filtered by kb_checksum."""
        results = []
        if self.generalization_dir.exists():
            for filepath in sorted(self.generalization_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if kb_checksum and data.get("kb_checksum") != kb_checksum:
                            continue
                        results.append(_sanitize_floats(data))
                except Exception:
                    continue
        return results

    def clear_experiments(self, kb_checksum: Optional[str] = None) -> int:
        """Clear experiment results, associated optimization memory, and generalization tests from the workspace."""
        deleted_count = 0
        for filepath in list(self.experiments_dir.glob("*.json")):
            try:
                if kb_checksum:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    exp_kb = data.get("kb_checksum") or data.get("metadata", {}).get("kb_checksum")
                    if exp_kb and exp_kb != kb_checksum:
                        continue
                filepath.unlink()
                deleted_count += 1
            except Exception:
                continue

        # Also clear generalization test files for this KB or all
        if self.generalization_dir.exists():
            for gen_file in list(self.generalization_dir.glob("*.json")):
                try:
                    if kb_checksum:
                        with open(gen_file, "r", encoding="utf-8") as f:
                            gdata = json.load(f)
                        if gdata.get("kb_checksum") != kb_checksum:
                            continue
                    gen_file.unlink()
                except Exception:
                    pass

        # Also clear any optimization memory files in cache
        if kb_checksum:
            mem_file = self.cache_dir / f"opt_memory_{kb_checksum}.json"
            if mem_file.exists():
                try:
                    mem_file.unlink()
                except Exception:
                    pass
        else:
            for mem_file in list(self.cache_dir.glob("opt_memory_*.json")):
                try:
                    mem_file.unlink()
                except Exception:
                    pass

        # Clear generated reports to avoid stale data
        if self.reports_dir.exists():
            for rep_file in list(self.reports_dir.glob("*")):
                try:
                    if rep_file.is_file():
                        rep_file.unlink()
                except Exception:
                    pass

        return deleted_count

    # Optimization Memory Persistence
    def save_optimization_memory(self, kb_checksum: str, memory_data: Dict[str, Any]) -> None:
        """Save canonical optimization memory for a specific KB."""
        filepath = self.cache_dir / f"opt_memory_{kb_checksum}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, default=str)

    def load_optimization_memory(self, kb_checksum: str) -> Optional[Dict[str, Any]]:
        """Load canonical optimization memory for a specific KB."""
        filepath = self.cache_dir / f"opt_memory_{kb_checksum}.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def clear_kb_artifacts(self, kb_checksum: Optional[str] = None) -> Dict[str, int]:
        """
        Cascade delete all derived artifacts (manifests, caches, chunks, embeddings,
        datasets, trials, generalization tests, memory, reports) for a specific KB or all KBs without touching
        connection profiles or environment.
        """
        counts = {
            "kb_files": 0,
            "manifests": 0,
            "datasets": 0,
            "experiments": 0,
            "generalization_tests": 0,
            "artifacts": 0,
            "caches": 0,
            "reports": 0
        }

        # 1. Clear KB files and scanner cache if clearing all
        if kb_checksum is None:
            if self.kb_dir.exists():
                for item in list(self.kb_dir.iterdir()):
                    try:
                        if item.is_file():
                            item.unlink()
                            counts["kb_files"] += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            counts["kb_files"] += 1
                    except Exception:
                        pass
            # Also clear manifest and scanner cache
            manifest_file = self.cache_dir / "kb_manifest.json"
            if manifest_file.exists():
                try:
                    manifest_file.unlink()
                    counts["manifests"] += 1
                except Exception:
                    pass
            scanner_cache = self.kb_dir / ".scanner_cache.json"
            if scanner_cache.exists():
                try:
                    scanner_cache.unlink()
                    counts["caches"] += 1
                except Exception:
                    pass

        # 2. Datasets
        if self.datasets_dir.exists():
            for ds_file in list(self.datasets_dir.glob("dataset_*.json")):
                try:
                    if kb_checksum:
                        with open(ds_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("kb_checksum") != kb_checksum:
                            continue
                    ds_file.unlink()
                    counts["datasets"] += 1
                except Exception:
                    pass

        # 3. Generalization Tests count
        if self.generalization_dir.exists():
            for gen_file in list(self.generalization_dir.glob("*.json")):
                try:
                    if kb_checksum:
                        with open(gen_file, "r", encoding="utf-8") as f:
                            gdata = json.load(f)
                        if gdata.get("kb_checksum") != kb_checksum:
                            continue
                    gen_file.unlink()
                    counts["generalization_tests"] += 1
                except Exception:
                    pass

        # 4. Experiments & Trials
        counts["experiments"] = self.clear_experiments(kb_checksum=kb_checksum)

        # 5. Cache files (chunks, embeddings, opt_memory)
        if self.cache_dir.exists():
            for cache_file in list(self.cache_dir.glob("*.json")):
                try:
                    if kb_checksum:
                        if kb_checksum in cache_file.name:
                            cache_file.unlink()
                            counts["caches"] += 1
                    else:
                        cache_file.unlink()
                        counts["caches"] += 1
                except Exception:
                    pass

        # 6. Artifacts
        artifacts_dir = self.root_dir / "artifacts"
        if artifacts_dir.exists():
            for art_file in list(artifacts_dir.glob("*.json")):
                try:
                    if not kb_checksum:
                        art_file.unlink()
                        counts["artifacts"] += 1
                except Exception:
                    pass

        # 7. Reports
        if not kb_checksum and self.reports_dir.exists():
            for rep_file in list(self.reports_dir.glob("*")):
                try:
                    if rep_file.is_file():
                        rep_file.unlink()
                        counts["reports"] += 1
                except Exception:
                    pass

        return counts

    
# Artifact Operations
    def save_artifact(self, artifact_type: str, data: Any, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Save a pipeline artifact."""
        artifacts_dir = self.root_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        content = json.dumps(data, indent=2, default=str).encode("utf-8")
        data_hash = hashlib.sha256(content).hexdigest()
        
        artifact_id = f"art_{data_hash[:12]}"
        filename = f"{artifact_type}_{artifact_id}.json"
        
        filepath = artifacts_dir / filename
        if not filepath.exists():
            with open(filepath, "wb") as f:
                f.write(content)
            
        return {
            "artifact_id": artifact_id,
            "type": artifact_type,
            "path": str(filepath),
            "hash": data_hash,
            "creation_time": datetime.now().isoformat()
        }

    def load_artifact_by_hash(self, data_hash: str) -> Optional[Any]:
        # not typically used because we cache by config, but just in case
        return None
        return None

    # Reports Operations
    def save_report(self, filename: str, content: str, extension: str = "txt") -> str:
        filepath = self.reports_dir / f"{filename}.{extension}"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return str(filepath)

    def list_reports(self) -> List[Dict[str, Any]]:
        reports = []
        for filepath in sorted(self.reports_dir.glob("*"), key=os.path.getmtime, reverse=True):
            if filepath.is_file():
                reports.append({
                    "name": filepath.name,
                    "path": str(filepath),
                    "size_bytes": filepath.stat().st_size,
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                })
        return reports
