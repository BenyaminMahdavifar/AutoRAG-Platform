"""
Streamlit UI for AutoRAG-Platform.
Thin presentation layer that delegates all business logic to `autorag` subsystems.
Run with: `streamlit run app.py`
"""

import os
import json
import streamlit as st
from pathlib import Path

from autorag import (
    WorkspaceManager, KnowledgeBaseScanner, DatasetBuilder,
    IndexBuilder, RetrievalEngine, GenerationEngine, EvaluationEngine,
    OptimizationEngine, ReportEngine, PipelineConfig, LLMConfig,
    EmbeddingConfig, ChunkingConfig, RetrieverConfig, OptimizationSpec
)

st.set_page_config(
    page_title="AutoRAG - RAG Pipeline Optimizer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AutoRAG-Platform")
st.caption("Automated RAG construction, retrieval evaluation, hyperparameter optimization, and reporting engine.")

# Initialize workspace
workspace = WorkspaceManager("workspace")
scanner = KnowledgeBaseScanner(workspace.get_kb_path())

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Subsystem",
    ["1. Knowledge Base", "2. Dataset Builder", "3. Experiment Playground", "4. Optimizer Sweep", "5. Reports & Exports", "6. Connections Config"]
)

# Sidebar Endpoint Config
st.sidebar.markdown("---")
st.sidebar.subheader("LLM Endpoint Settings")
llm_provider = st.sidebar.selectbox("Provider", ["openai", "ollama", "lmstudio", "openrouter", "gemini"], index=0)
base_url = st.sidebar.text_input("Base URL", "https://api.openai.com/v1" if llm_provider != "gemini" else "")
api_key = st.sidebar.text_input("API Key", type="password")
model_name = st.sidebar.text_input("Model Name", "gpt-4o-mini" if llm_provider == "openai" else "gemini-3.6-flash")

llm_cfg = LLMConfig(provider=llm_provider, base_url=base_url, api_key=api_key, model_name=model_name)
emb_cfg = EmbeddingConfig(provider="local", model_name="local-tfidf-512")

# Page 1: Knowledge Base
if page == "1. Knowledge Base":
    st.header("📂 Knowledge Base Inspector")
    docs, manifest = scanner.scan()
    workspace.save_manifest(manifest)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", manifest.total_docs)
    col2.metric("Total Size (KB)", f"{manifest.total_size_bytes / 1024:.1f}")
    col3.metric("KB Checksum", manifest.kb_checksum[:10])

    st.subheader("Document Files")
    for doc in docs:
        with st.expander(f"📄 {doc.filename} ({doc.size_bytes} bytes)"):
            st.json(doc.metadata)
            st.text_area("Content Preview", doc.content[:1000], height=150, key=f"preview_{doc.doc_id}")

# Page 2: Dataset Builder
elif page == "2. Dataset Builder":
    st.header("📊 Evaluation Dataset Builder")
    docs, manifest = scanner.scan()
    
    if st.button("Generate / Load Synthetic Dataset"):
        with st.spinner("Building chunks and generating dataset..."):
            index_builder = IndexBuilder(workspace, EmbeddingClient(emb_cfg), OpenAICompatibleClient(llm_cfg))
            index = index_builder.build_index(docs, ChunkingConfig(chunk_size=512), index_type="memory")
            
            builder = DatasetBuilder(workspace)
            dataset = builder.build_dataset(docs, index.chunks, manifest.kb_checksum)
            
            st.success(f"Dataset Loaded! Total Items: {len(dataset.items)}")
            
            for item in dataset.items:
                st.markdown(f"**Question:** {item.question}")
                st.info(f"**Ground Truth:** {item.ground_truth}")
                st.caption(f"Source Doc: {item.metadata.get('filename')} | ID: {item.item_id}")
                st.markdown("---")

# Page 3: Single Experiment
elif page == "3. Experiment Playground":
    st.header("🔬 Experiment Playground")
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_strategy = st.selectbox("Chunk Strategy", ["recursive", "fixed", "paragraph", "semantic"])
        chunk_size = st.slider("Chunk Size", 128, 2048, 512, step=64)
        chunk_overlap = st.slider("Chunk Overlap", 0, 256, 64, step=16)
    with col2:
        ret_strategy = st.selectbox("Retriever Strategy", ["hybrid", "dense", "sparse"])
        dist_metric = st.selectbox("Distance Metric", ["cosine", "dot", "euclidean"])
        top_k = st.slider("Top K Chunks", 1, 10, 4)

    test_query = st.text_input("Test Query", "What are the core components of a RAG architecture?")

    if st.button("Run Experiment & Evaluate"):
        docs, manifest = scanner.scan()
        p_cfg = PipelineConfig(
            experiment_name="playground_test",
            llm_config=llm_cfg,
            embedding_config=emb_cfg,
            chunking_config=ChunkingConfig(strategy=chunk_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            retriever_config=RetrieverConfig(strategy=ret_strategy, distance_metric=dist_metric, top_k=top_k)
        )

        with st.spinner("Executing RAG Pipeline..."):
            opt_engine = OptimizationEngine(workspace, OpenAICompatibleClient(llm_cfg), EmbeddingClient(emb_cfg))
            index_builder = IndexBuilder(workspace, EmbeddingClient(emb_cfg), OpenAICompatibleClient(llm_cfg))
            index = index_builder.build_index(docs, p_cfg.chunking_config, index_type=getattr(p_cfg.retriever_config, "index_type", "memory"))
            
            builder = DatasetBuilder(workspace)
            dataset = builder.build_dataset(docs, index.chunks, manifest.kb_checksum)

            res = opt_engine.run_trial(p_cfg, docs, dataset)

            st.subheader("Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Composite Score", f"{res.composite_score:.4f}")
            m2.metric("Hit Rate", f"{res.retrieval_metrics.hit_rate:.4f}")
            m3.metric("Answer Correctness", f"{getattr(res.answer_metrics, 'answer_correctness', res.answer_metrics.accuracy):.4f}")
            m4.metric("Avg Latency (ms)", f"{res.avg_latency_ms:.1f}")

# Page 4: Optimizer Sweep
elif page == "4. Optimizer Sweep":
    st.header("🚀 Hyperparameter Optimization Sweep")
    strategy = st.selectbox("Optimization Strategy", ["llm_guided", "grid", "random"])
    max_trials = st.slider("Max Trials", 2, 10, 4)

    if st.button("Start Optimization Sweep"):
        docs, manifest = scanner.scan()
        p_cfg = PipelineConfig(
            experiment_name="sweep_base",
            llm_config=llm_cfg,
            embedding_config=emb_cfg
        )

        with st.spinner("Running Optimization Trials..."):
            opt_engine = OptimizationEngine(workspace, OpenAICompatibleClient(llm_cfg), EmbeddingClient(emb_cfg))
            index_builder = IndexBuilder(workspace, EmbeddingClient(emb_cfg), OpenAICompatibleClient(llm_cfg))
            index = index_builder.build_index(docs, p_cfg.chunking_config, index_type=getattr(p_cfg.retriever_config, "index_type", "memory"))
            
            dataset = DatasetBuilder(workspace).build_dataset(docs, index.chunks, manifest.kb_checksum)

            spec = OptimizationSpec(strategy=strategy, max_trials=max_trials)
            summary = opt_engine.optimize(spec, p_cfg, docs, dataset)

            st.success(f"Sweep Completed! Highest Composite Score: {summary['best_score']:.4f}")
            st.json(summary["leaderboard"])

# Page 5: Reports
elif page == "5. Reports & Exports":
    st.header("📈 Reports & Exports")
    experiments = workspace.list_experiments()
    st.subheader(f"Saved Experiments ({len(experiments)})")
    
    if st.button("Export Full Leaderboard to Markdown & HTML"):
        reporter = ReportEngine(workspace)
        md_path = reporter.export_markdown(experiments, "latest_summary")
        html_path = reporter.export_html(experiments, "latest_dashboard")
        st.success(f"Reports saved to workspace/reports!")

    for exp in experiments:
        st.json(exp)

# Page 6: Connections Config
elif page == "6. Connections Config":
    st.header("⚙️ Connections & Environment")
    st.info("Configured for OpenAI, Ollama, LM Studio, OpenRouter, and Gemini.")
    st.json({
        "provider": llm_cfg.provider,
        "base_url": llm_cfg.base_url,
        "model_name": llm_cfg.model_name,
        "has_api_key": bool(llm_cfg.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    })
