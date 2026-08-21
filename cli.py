"""
CLI Interface Bridge for AutoRAG Platform.
Provides JSON RPC interface for web backend and CLI orchestration.
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from autorag.exporter import RagExporter

from autorag import (
    WorkspaceManager, KnowledgeBaseScanner, DatasetBuilder,
    IndexBuilder, RetrievalEngine, GenerationEngine, EvaluationEngine,
    OptimizationEngine, ReportEngine, GeneralizationEngine, PipelineConfig, LLMConfig,
    EmbeddingConfig, ChunkingConfig, RetrieverConfig, OptimizationSpec,
    inspect_installed_torch, resolve_embedding_device, CUDAUnavailableError, get_nvidia_driver_info
)


def emit_progress(progress, stage, completed_stages, message=""):
    event = {
        "type": "progress",
        "progress": progress,
        "current_stage": stage,
        "completed_stages": completed_stages,
        "message": message
    }
    print(json.dumps(event, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description="AutoRAG CLI Bridge")
    parser.add_argument("action", choices=["scan_kb", "build_dataset", "run_experiment", "run_optimizer", "list_experiments", "clear_experiments", "export_reports", "upload_doc", "import_kb", "clear_kb", "get_dataset", "validate_environment", "setup_environment", "export_rag", "run_generalization_test", "get_generalization_test"])
    parser.add_argument("--payload", type=str, default="{}", help="JSON payload")
    parser.add_argument("--payload-file", type=str, help="Path to JSON payload file")
    args = parser.parse_args()

    try:
        if args.payload_file and os.path.exists(args.payload_file):
            with open(args.payload_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = json.loads(args.payload)
    except Exception as e:
        print(f"ERROR reading payload: {e}")
        payload = {}

    try:
        workspace = WorkspaceManager("workspace")
        scanner = KnowledgeBaseScanner(workspace.get_kb_path())

        llm_cfg = LLMConfig(
            provider=payload.get("provider", "openai"),
            base_url=payload.get("base_url", "https://api.openai.com/v1"),
            api_key=payload.get("api_key", ""),
            model_name=payload.get("model_name", "gpt-4o-mini"),
            timeout_sec=int(payload.get("timeout_sec", 30)),
            temperature=float(payload.get("temperature", 0.2)),
            top_p=float(payload.get("top_p", 0.95)),
            max_tokens=int(payload.get("max_tokens", 1024))
        )
        
        emb_model_name = payload.get("embedding_model")
        hf_token = payload.get("hf_token")
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            except:
                pass
        
        emb_device = payload.get("embedding_device", "auto")
        if emb_model_name and emb_model_name != "local-tfidf-512":
            emb_cfg = EmbeddingConfig(provider="huggingface", model_name=emb_model_name, device=emb_device)
        else:
            emb_cfg = EmbeddingConfig(provider="local", model_name="local-tfidf-512")


        from autorag.connections import OpenAICompatibleClient, EmbeddingClient
        llm_client = OpenAICompatibleClient(llm_cfg)
        emb_client = EmbeddingClient(emb_cfg)


        import sys
        hf_token_status = "configured" if hf_token else "missing"
        model_source = "download required"
        if emb_cfg.provider == "huggingface":
            try:
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                model_dir = "models--" + emb_cfg.model_name.replace("/", "--")
                if os.path.exists(os.path.join(cache_dir, model_dir)):
                    model_source = "cache"
            except Exception as e:
                pass
        else:
            model_source = "local"
            
        print("[Embedding Configuration]")
        print(f"Model: {emb_cfg.model_name}")
        print(f"Device: {emb_cfg.device}")
        print(f"Auto Login: {'enabled' if hf_token else 'disabled'}")
        print(f"HF Auth: {hf_token_status}")
        print(f"Model Source: {model_source}")

        if args.action in ["run_optimizer", "run_experiment", "validate_environment"]:
            try:
                import transformers
                deps_avail = "yes"
            except ImportError:
                deps_avail = "no"

            workspace_root = os.getcwd()
            venv_path = os.path.abspath(".venv")
            using_venv = "yes" if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else "no"
            print("[Environment]")
            print(f"Workspace Root: {workspace_root}")
            print(f"Virtual Environment: {venv_path}")
            print(f"Python Executable: {sys.executable}")
            print(f"Using Venv: {using_venv}")
            print("[Sweep Setup]")
            print(f"LLM Provider: {llm_cfg.provider.capitalize()}")
            print(f"Endpoint: {llm_cfg.base_url}")
            print(f"Embedding Model: {emb_cfg.model_name}")
            print(f"Dependencies Available: {deps_avail}")
            if llm_cfg.provider.lower() == "ollama":
                print("Connection Mode: local")
        output = {"success": True, "action": args.action}

        def validate_llm_config():
            if llm_cfg.provider in ["openai", "gemini"] and not llm_cfg.api_key and not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
                raise ValueError(f"API Key is required for provider: {llm_cfg.provider}. Please configure it in settings.")
            if not llm_cfg.model_name:
                raise ValueError("Model name must be configured.")
            return True

        if args.action == "scan_kb":
            emit_progress(20, "Scanning Documents Directory", [], "Accessing workspace/kb directory...")
            docs, manifest = scanner.scan()
            emit_progress(70, "Verifying Document Checksums", ["Scanning Documents Directory"], f"Found {len(docs)} documents. Calculating SHA256 hashes...")
            workspace.save_manifest(manifest)
            emit_progress(100, "Manifest Saved", ["Scanning Documents Directory", "Verifying Document Checksums"], "Knowledge Base Manifest updated successfully.")
            output["manifest"] = manifest.__dict__
            output["documents"] = [
                {
                    "doc_id": d.doc_id,
                    "filename": d.filename,
                    "filepath": d.filepath,
                    "file_type": d.file_type,
                    "size_bytes": d.size_bytes,
                    "checksum": d.checksum,
                    "content_preview": d.content[:300],
                    "metadata": d.metadata
                }
                for d in docs
            ]

        elif args.action == "upload_doc":
            filename = payload.get("filename", "uploaded_doc.txt")
            content = payload.get("content", "")
            emit_progress(25, "Writing Document File", [], f"Writing {filename} to knowledge base...")
            filepath = workspace.get_kb_path() / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            emit_progress(60, "Scanning Updated Directory", ["Writing Document File"], "Rescanning Knowledge Base...")
            docs, manifest = scanner.scan()
            workspace.save_manifest(manifest)
            emit_progress(100, "Directory Manifest Updated", ["Writing Document File", "Scanning Updated Directory"], f"File {filename} uploaded successfully.")
            output["message"] = f"File {filename} uploaded successfully."
            output["manifest"] = manifest.__dict__

        elif args.action == "import_kb":
            import base64
            import zipfile
            import io
            
            import_type = payload.get("import_type")
            batch_id = payload.get("batch_id", "unknown_batch")
            
            emit_progress(10, "Initializing Import", [], f"Starting {import_type} import...")
            kb_dir = workspace.get_kb_path()
            
            total_discovered = 0
            imported = 0
            skipped = 0
            failed = 0
            
            def process_file(rel_path, file_bytes):
                nonlocal imported, skipped, failed
                try:
                    import hashlib
                    ext = Path(rel_path).suffix.lower()
                    if ext not in KnowledgeBaseScanner.SUPPORTED_EXTENSIONS:
                        skipped += 1
                        return
                    
                    target_path = kb_dir / rel_path
                    
                    if target_path.exists():
                        existing_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
                        new_hash = hashlib.sha256(file_bytes).hexdigest()
                        if existing_hash == new_hash:
                            skipped += 1
                            return

                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(file_bytes)
                    imported += 1
                except Exception as e:
                    print(f"Failed to process {rel_path}: {e}")
                    failed += 1

            if import_type == "folder":
                files = payload.get("files", [])
                total_discovered = len(files)
                for i, f in enumerate(files):
                    rel_path = f.get("relative_path")
                    content_b64 = f.get("content_b64", "")
                    if i % 10 == 0:
                        emit_progress(20 + int((i / max(1, total_discovered)) * 60), "Processing Files", ["Initializing Import"], f"Extracting {rel_path}...")
                    try:
                        file_bytes = base64.b64decode(content_b64)
                        process_file(rel_path, file_bytes)
                    except Exception as e:
                        failed += 1
                        
            elif import_type == "zip":
                zip_b64 = payload.get("zip_b64", "")
                try:
                    zip_bytes = base64.b64decode(zip_b64)
                    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                        file_list = [info for info in z.infolist() if not info.is_dir()]
                        total_discovered = len(file_list)
                        for i, info in enumerate(file_list):
                            if i % 10 == 0:
                                emit_progress(20 + int((i / max(1, total_discovered)) * 60), "Extracting ZIP", ["Initializing Import"], f"Extracting {info.filename}...")
                            try:
                                file_bytes = z.read(info.filename)
                                process_file(info.filename, file_bytes)
                            except Exception as e:
                                failed += 1
                except Exception as e:
                    raise ValueError(f"Failed to extract ZIP archive: {e}")
            
            emit_progress(80, "Scanning Updated Directory", ["Initializing Import", "Processing Files"], "Rescanning Knowledge Base...")
            docs, manifest = scanner.scan()
            workspace.save_manifest(manifest)
            
            summary = {
                "total_discovered": total_discovered,
                "imported": imported,
                "skipped": skipped,
                "failed": failed
            }
            emit_progress(100, "Import Complete", ["Initializing Import", "Processing Files", "Scanning Updated Directory"], f"Imported {imported} files. Skipped: {skipped}, Failed: {failed}.")
            output["summary"] = summary
            output["manifest"] = manifest.__dict__

        elif args.action == "clear_kb":
            emit_progress(20, "Initializing Zero-Trace Purge", [], "Preparing to clear Knowledge Base and all derived artifacts...")
            
            # Cascade delete all derived artifacts, datasets, trials, reports, and caches
            deleted_counts = workspace.clear_kb_artifacts()
            
            emit_progress(60, "Scanning Updated Directory", ["Initializing Zero-Trace Purge"], "Rescanning Knowledge Base...")
            docs, manifest = scanner.scan()
            workspace.save_manifest(manifest)
            
            emit_progress(100, "Knowledge Base Purged", ["Initializing Zero-Trace Purge", "Scanning Updated Directory"], f"Successfully cleared Knowledge Base. Removed {deleted_counts.get('kb_files', 0)} files, {deleted_counts.get('datasets', 0)} datasets, {deleted_counts.get('experiments', 0)} trials.")
            output["message"] = "Knowledge Base and all derived artifacts cleared successfully."
            output["deleted_counts"] = deleted_counts
            output["manifest"] = manifest.__dict__

        elif args.action == "build_dataset":
            emit_progress(5, "Validating Requirements", [], "Checking configuration and API keys...")
            validate_llm_config()
            
            emit_progress(15, "Scan Knowledge Base", ["Validating Requirements"], "Scanning documents...")
            docs, manifest = scanner.scan()
            if not docs:
                raise ValueError("No documents found in the Knowledge Base. Please upload documents first.")

            emit_progress(35, "Chunking Documents", ["Validating Requirements", "Scan Knowledge Base"], f"Splitting {len(docs)} documents into chunks...")
            index_builder = IndexBuilder(workspace, emb_client)
            index = index_builder.build_index(docs, ChunkingConfig(chunk_size=512))
            emit_progress(65, "Generating Vector Embeddings", ["Validating Requirements", "Scan Knowledge Base", "Chunking Documents"], f"Embedding {len(index.chunks)} chunks with local TFIDF engine...")
            emit_progress(85, "Synthesizing Ground Truth Q&A", ["Validating Requirements", "Scan Knowledge Base", "Chunking Documents", "Generating Vector Embeddings"], f"Connecting to AI Provider ({llm_cfg.provider}) and generating synthetic question/ground truth pairs via LLM...")
            def logger(msg):
                emit_progress(85, "Synthesizing Ground Truth Q&A", ["Validating Requirements", "Scan Knowledge Base", "Chunking Documents", "Generating Vector Embeddings"], msg)
            
            builder = DatasetBuilder(workspace, llm_client, logger=logger)
            dataset = builder.build_dataset(docs, index.chunks, manifest.kb_checksum, force_regenerate=True)
            
            emit_progress(95, "Validating Artifacts", ["Validating Requirements", "Scan Knowledge Base", "Chunking Documents", "Generating Vector Embeddings", "Synthesizing Ground Truth Q&A"], "Verifying generated dataset integrity...")
            if not dataset or not dataset.items:
                raise ValueError("Generated dataset is empty. Model failed to produce valid Q&A pairs.")
            if len(dataset.items) == 0:
                raise ValueError("Dataset question count must be > 0.")
            for item in dataset.items:
                if not item.question or not item.ground_truth:
                    raise ValueError(f"Invalid dataset entry found (Missing question or answer): {item.item_id}")
                    
            # Verify physically on disk
            ds_path = workspace.datasets_dir / f"dataset_v{dataset.version}_{dataset.dataset_id[:8]}.json"
            if not ds_path.exists():
                raise ValueError("Artifact Validation Failed: Dataset file does not exist on disk.")
            try:
                with open(ds_path, "r", encoding="utf-8") as f:
                    ds_data = json.load(f)
                    if not ds_data.get("items"):
                        raise ValueError("Artifact Validation Failed: Saved dataset file has no items.")
            except Exception as e:
                raise ValueError(f"Artifact Validation Failed: Dataset file is not readable or invalid JSON: {e}")
            
            emit_progress(100, "Dataset Version Saved", ["Validating Requirements", "Scan Knowledge Base", "Chunking Documents", "Generating Vector Embeddings", "Synthesizing Ground Truth Q&A", "Validating Artifacts"], f"Dataset v{dataset.version} created with {len(dataset.items)} test cases.")
            dataset_output = {
                "dataset_id": dataset.dataset_id,
                "created_at": dataset.created_at,
                "version": dataset.version,
                "items": [item.__dict__ for item in dataset.items]
            }
            if hasattr(dataset, "_execution_metadata"):
                dataset_output["execution_metadata"] = getattr(dataset, "_execution_metadata")
            
            output["dataset"] = dataset_output

        elif args.action == "run_experiment":
            emit_progress(5, "Validating Requirements", [], "Checking configuration and API keys...")
            validate_llm_config()
            
            emit_progress(10, "Scan Knowledge Base", ["Validating Requirements"], "Loading documents...")
            docs, manifest = scanner.scan()
            if not docs:
                raise ValueError("No documents found in the Knowledge Base. Please upload documents first.")

            c_cfg = payload.get("chunking_config", {})
            r_cfg = payload.get("retriever_config", {})
            
            p_cfg = PipelineConfig(
                experiment_name=payload.get("experiment_name", "web_experiment"),
                llm_config=llm_cfg,
                embedding_config=emb_cfg,
                chunking_config=ChunkingConfig(
                    strategy=c_cfg.get("strategy", "recursive"),
                    chunk_size=int(c_cfg.get("chunk_size", 512)),
                    chunk_overlap=int(c_cfg.get("chunk_overlap", 64))
                ),
                retriever_config=RetrieverConfig(
                    strategy=r_cfg.get("strategy", "hybrid"),
                    distance_metric=r_cfg.get("distance_metric", "cosine"),
                    top_k=int(r_cfg.get("top_k", 4)),
                    hybrid_alpha=float(r_cfg.get("hybrid_alpha", 0.7))
                )
            )

            emit_progress(30, "Building Vector Index", ["Validating Requirements", "Scan Knowledge Base"], f"Chunking strategy: {p_cfg.chunking_config.strategy} (size={p_cfg.chunking_config.chunk_size})...")
            opt_engine = OptimizationEngine(workspace, llm_client, emb_client)
            index = IndexBuilder(workspace, emb_client).build_index(docs, p_cfg.chunking_config)
            
            emit_progress(55, "Loading Evaluation Dataset", ["Validating Requirements", "Scan Knowledge Base", "Building Vector Index"], "Loading ground-truth evaluation dataset...")
            dataset = DatasetBuilder(workspace, llm_client).build_dataset(docs, index.chunks, manifest.kb_checksum)
            if not dataset or not dataset.items:
                raise ValueError("Evaluation dataset is empty. Please build a dataset first.")

            emit_progress(80, "Executing Retrieval & Generation Benchmark", ["Validating Requirements", "Scan Knowledge Base", "Building Vector Index", "Loading Evaluation Dataset"], "Running trial queries, measuring Hit Rate and Faithfulness...")
            res = opt_engine.run_trial(p_cfg, docs, dataset)
            eval_res = res.results
            if eval_res and eval_res.metrics_valid is False:
                emit_progress(100, "Trial Failed", ["Validating Requirements", "Scan Knowledge Base", "Building Vector Index", "Loading Evaluation Dataset", "Executing Retrieval & Generation Benchmark"], f"Trial {res.experiment_id} failed: {eval_res.failure_reason}")
            else:
                score_str = f"{(res.composite_score * 100):.1f}%" if res.composite_score is not None else "N/A"
                emit_progress(100, "Trial Completed", ["Validating Requirements", "Scan Knowledge Base", "Building Vector Index", "Loading Evaluation Dataset", "Executing Retrieval & Generation Benchmark"], f"Trial {res.experiment_id} completed with composite score {score_str}.")
            
            from dataclasses import asdict
            res_dict = asdict(res)
            res_dict["composite_score"] = res.composite_score
            res_dict["metrics_valid"] = res.results.metrics_valid if res.results else False
            res_dict["failure_reason"] = res.results.failure_reason if res.results else None
            res_dict["status"] = res.results.status if res.results else "failed"
            if eval_res:
                res_dict["retrieval_metrics"] = asdict(eval_res.retrieval_metrics) if eval_res.retrieval_metrics else None
                res_dict["answer_metrics"] = asdict(eval_res.answer_metrics) if eval_res.answer_metrics else None
                res_dict["avg_latency_ms"] = eval_res.avg_latency_ms
                res_dict["total_tokens"] = eval_res.total_tokens
                res_dict["sample_evaluations"] = eval_res.sample_evaluations
                res_dict["orchestrator_metrics"] = eval_res.orchestrator_metrics
            output["result"] = res_dict

        elif args.action == "run_optimizer":
            emit_progress(5, "Validating Requirements", [], "Checking configuration and API keys...")
            validate_llm_config()
            
            emit_progress(10, "Initializing Search Space", ["Validating Requirements"], "Loading workspace and target pipeline parameters...")
            docs, manifest = scanner.scan()
            if not docs:
                raise ValueError("No documents found in the Knowledge Base. Please upload documents first.")
                
            p_cfg = PipelineConfig(experiment_name="sweep_base", llm_config=llm_cfg, embedding_config=emb_cfg)
            opt_engine = OptimizationEngine(workspace, llm_client, emb_client)
            
            emit_progress(30, "Building Index & Dataset", ["Validating Requirements", "Initializing Search Space"], "Preparing index and test dataset for sweep...")
            index = IndexBuilder(workspace, emb_client).build_index(docs, p_cfg.chunking_config)
            dataset = DatasetBuilder(workspace, llm_client).build_dataset(docs, index.chunks, manifest.kb_checksum)
            if not dataset or not dataset.items:
                raise ValueError("Evaluation dataset is empty. Please build a dataset first.")

            use_previous = payload.get("use_previous_history", payload.get("usePreviousOptimizationHistory", True))
            learn_gen = payload.get("learn_from_generalization_test", payload.get("learnFromGeneralizationTest", False))
            spec = OptimizationSpec(
                strategy=payload.get("strategy", "llm_guided"),
                max_trials=int(payload.get("max_trials", 4)),
                use_previous_history=bool(use_previous),
                learn_from_generalization_test=bool(learn_gen)
            )
            emit_progress(50, f"Running {spec.max_trials} Optimization Trials ({spec.strategy})", ["Validating Requirements", "Initializing Search Space", "Building Index & Dataset"], f"Evaluating parameter combinations...")
            summary = opt_engine.optimize(spec, p_cfg, docs, dataset)
            if summary['best_score'] is None or summary['best_score'] < 0:
                emit_progress(100, "Optimization Sweep Complete", ["Validating Requirements", "Initializing Search Space", "Building Index & Dataset", f"Running {spec.max_trials} Optimization Trials ({spec.strategy})"], "Sweep finished, but all trials failed.")
            else:
                emit_progress(100, "Optimization Sweep Complete", ["Validating Requirements", "Initializing Search Space", "Building Index & Dataset", f"Running {spec.max_trials} Optimization Trials ({spec.strategy})"], f"Sweep finished. Best composite score: {(summary['best_score'] * 100):.1f}%.")
            output["summary"] = summary

        elif args.action == "run_generalization_test":
            from autorag.generalization_engine import GeneralizationEngine
            from dataclasses import asdict

            exp_id = payload.get("experiment_id") or payload.get("trialId")
            if not exp_id:
                raise ValueError("experiment_id or trialId is required for generalization test.")

            emit_progress(5, "Scanning Knowledge Base", [], "Scanning documents for holdout validation...")
            docs, manifest = scanner.scan()
            if not docs:
                raise ValueError("Knowledge base contains no documents. Please add documents first.")

            gen_engine = GeneralizationEngine(
                workspace=workspace,
                llm_client=llm_client,
                emb_client=emb_client,
                eval_engine=EvaluationEngine(llm_client)
            )

            test_size = int(payload.get("test_size", 5))
            result = gen_engine.run_generalization_test(
                experiment_id=exp_id,
                docs=docs,
                kb_checksum=manifest.kb_checksum,
                test_size=test_size,
                emit_progress_fn=lambda p, s, c, m: emit_progress(p, s, c, m)
            )

            res_dict = asdict(result) if hasattr(result, "__dict__") else dict(result)
            output["result"] = res_dict
            output["generalization_result"] = res_dict
            output["experiment_id"] = exp_id
            output["status"] = "success"
            output["success"] = True

        elif args.action == "get_generalization_test":
            exp_id = payload.get("experiment_id") or payload.get("trialId")
            if not exp_id:
                raise ValueError("experiment_id is required to fetch generalization test.")
            gen_res = workspace.get_generalization_test(exp_id)
            output["result"] = gen_res
            output["generalization_test"] = gen_res
            output["experiment_id"] = exp_id
            output["status"] = "success" if gen_res else "not_found"
            output["success"] = True

        elif args.action == "list_experiments":
            kb_checksum = payload.get("kb_checksum")
            output["experiments"] = workspace.list_experiments(kb_checksum=kb_checksum)

        elif args.action == "clear_experiments":
            kb_checksum = payload.get("kb_checksum")
            deleted_count = workspace.clear_experiments(kb_checksum=kb_checksum)
            output["status"] = "success"
            output["deleted_count"] = deleted_count
            output["message"] = f"Cleared {deleted_count} experiment trials and reset optimization memory."

        

        elif args.action == "setup_environment":
            from autorag.env_lifecycle import EnvironmentManager, DependencyResolver, DependencyInstaller, EnvironmentValidator
            
            emit_progress(5, "Runtime Discovery", [], "Discovering Python runtime...")
            
            env_manager = EnvironmentManager(workspace.root_dir)
            python_exec = env_manager.ensure_environment(emit_progress_fn=lambda p, s, c, m: emit_progress(p, s, c, m))
            
            emit_progress(20, "Resolve Dependencies", ["Runtime Discovery", "Create Virtual Environment", "Upgrade pip"], "Resolving required packages...")
            resolver = DependencyResolver()
            packages = resolver.resolve(payload)
            
            installer = DependencyInstaller(python_exec)
            installer.install(packages, emit_progress_fn=lambda p, s, c, m: emit_progress(p, s, c, m), base_progress=30)
            
            validator = EnvironmentValidator(python_exec)
            validation_result = validator.validate(payload, emit_progress_fn=lambda p, s, c, m: emit_progress(p, s, c, m))
            
            try:
                from autorag.connections import OpenAICompatibleClient
                llm_cfg = LLMConfig(
                    provider=payload.get("provider", "openai"),
                    base_url=payload.get("base_url", "https://api.openai.com/v1"),
                    api_key=payload.get("api_key", ""),
                    model_name=payload.get("model_name", "gpt-4o-mini"),
                )
                llm_client = OpenAICompatibleClient(llm_cfg)
                res = llm_client.chat_completion([{"role": "user", "content": "Hi"}])
                if "error" in res:
                    validation_result.setdefault("details", []).append(f"LLM Connection failed: {res['error']}")
                    validation_result["llm_connection"] = False
                else:
                    validation_result.setdefault("details", []).append("LLM Connection verified successfully.")
                    validation_result["llm_connection"] = True
            except Exception as e:
                validation_result.setdefault("details", []).append(f"LLM Connection failed: {str(e)}")
                validation_result["llm_connection"] = False

            output["validation"] = validation_result
            output["python_exec"] = python_exec
            emit_progress(100, "Ready", ["Runtime Discovery", "Create Virtual Environment", "Upgrade pip", "Resolve Dependencies", "Install Packages", "Validate Environment"], "Environment preparation complete.")

        elif args.action == "validate_environment":
            torch_info = inspect_installed_torch()
            driver_info = get_nvidia_driver_info()
            configured_dev = payload.get("embedding_device", "auto")
            
            resolved_dev = "cpu"
            try:
                resolved_dev = resolve_embedding_device(configured_dev)
            except Exception:
                resolved_dev = "unresolvable"

            result = {
                "python_runtime": sys.version.split(' ')[0],
                "torch": torch_info,
                "cuda_available": torch_info.get("cudaAvailable", False),
                "gpu_name": torch_info.get("deviceName", "N/A"),
                "memory_available": torch_info.get("gpuMemory", "N/A"),
                "driver_info": driver_info,
                "configured_device": configured_dev,
                "resolved_device": resolved_dev,
                "hf_authenticated": False,
                "model_available": False,
                "llm_connection": False,
                "health_code": torch_info.get("healthCode", "UNKNOWN"),
                "details": []
            }
            
            if torch_info.get("installed"):
                build_label = "CUDA-enabled" if torch_info.get("isCudaBuild") else ("CPU-only (Backend Mismatch)" if torch_info.get("isCpuBuild") else "Unknown")
                result["details"].append(f"PyTorch: {torch_info.get('version')} ({build_label}, CUDA Runtime: {torch_info.get('cudaRuntime') or 'N/A'})")
                result["details"].append(f"CPU Tensor Execution: {'PASS' if torch_info.get('cpuExecution') else 'FAIL'}")
            else:
                result["details"].append("PyTorch is not installed.")
                
            if torch_info.get("cudaAvailable"):
                result["details"].append(f"CUDA Hardware: {torch_info.get('deviceName')} (Memory: {torch_info.get('gpuMemory')})")
            else:
                result["details"].append("CUDA Hardware: Not available on host. CPU execution active.")
            
            hf_token = payload.get("hf_token")
            if hf_token:
                try:
                    from huggingface_hub import login, HfApi
                    login(token=hf_token, add_to_git_credential=False)
                    api = HfApi()
                    user = api.whoami()
                    result["hf_authenticated"] = True
                    result["details"].append(f"HF Authenticated as {user.get('name')}")
                except Exception as e:
                    result["details"].append(f"HF Auth failed: {str(e)}")
            else:
                result["details"].append("No HF token provided.")
                
            emb_model = payload.get("embedding_model")
            if emb_model and emb_model != "local-tfidf-512":
                try:
                    from sentence_transformers import SentenceTransformer
                    target_device = resolve_embedding_device(configured_dev)
                    model = SentenceTransformer(emb_model, device=target_device)
                    result["model_available"] = True
                    result["details"].append(f"Embedding model '{emb_model}' ready on '{target_device}'.")
                except CUDAUnavailableError as e:
                    result["model_available"] = False
                    result["details"].append(f"CUDA Placement Error: {str(e)}")
                    result["error"] = str(e)
                except Exception as e:
                    result["model_available"] = False
                    result["details"].append(f"Embedding model load failed: {str(e)}")
            else:
                result["model_available"] = True
                result["details"].append("Using local TF-IDF vectorizer (no embedding download needed).")
                
            try:
                res = llm_client.chat_completion([{"role": "user", "content": "Hi"}])
                if res and "text" in res:
                    result["llm_connection"] = True
                    result["details"].append("LLM Connection verified.")
            except Exception as e:
                result["details"].append(f"LLM Connection: Not configured or unreachable ({str(e)})")
                
            output["validation"] = result
        elif args.action == "get_dataset":
            dataset = workspace.load_latest_dataset()
            if dataset:
                output["dataset"] = dataset
            else:
                output["dataset"] = None

        elif args.action == "export_rag":
            exp_id = payload.get("experiment_id")
            if not exp_id:
                raise ValueError("experiment_id is required for export_rag")
            exporter = RagExporter(workspace)
            zip_url = exporter.export_trial(exp_id, emit_progress)
            output["download_url"] = zip_url
            output["success"] = True

        elif args.action == "export_reports":
            emit_progress(25, "Gathering Experiment Trials", [], "Listing workspace experiments...")
            exps = workspace.list_experiments()
            
            def get_score(e):
                score = e.get("composite_score")
                if score is None and isinstance(e.get("results"), dict):
                    score = e.get("results", {}).get("composite_score")
                if isinstance(score, (int, float)) and math.isfinite(score):
                    return float(score)
                return -1.0
                
            exps.sort(key=get_score, reverse=True)
            reporter = ReportEngine(workspace)
            emit_progress(60, "Generating Markdown & CSV Reports", ["Gathering Experiment Trials"], "Compiling leaderboard summary table and CSV metrics...")
            output["md_path"] = reporter.export_markdown(exps, "leaderboard_summary")
            output["csv_path"] = reporter.export_csv(exps, "leaderboard_data")
            emit_progress(85, "Rendering HTML Dashboard", ["Gathering Experiment Trials", "Generating Markdown & CSV Reports"], "Rendering standalone HTML dashboard...")
            output["html_path"] = reporter.export_html(exps, "leaderboard_dashboard")
            output["reports"] = workspace.list_reports()
            emit_progress(100, "All Reports Exported", ["Gathering Experiment Trials", "Generating Markdown & CSV Reports", "Rendering HTML Dashboard"], "Artifacts successfully generated and saved to workspace/reports.")

        print(json.dumps(output, default=str), flush=True)

    except Exception as e:
        
        err_output = {
            "success": False,
            "action": args.action,
            "error": str(e),
            "suggested_fix": "Verify that your LLM provider endpoint is reachable and API key is valid in Connection Settings."
        }
        print(json.dumps(err_output, default=str), flush=True)


if __name__ == "__main__":
    main()
