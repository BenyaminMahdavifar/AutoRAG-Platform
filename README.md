# AutoRAG Platform

AutoRAG-Platform is a production-grade RAG (Retrieval-Augmented Generation) Optimization Platform designed for evaluating, experimenting, hyperparameter tuning, and exporting optimized RAG pipelines.

## What It Does
AutoRAG provides a full-stack environment to import documents, synthesize ground-truth datasets, evaluate retrieval and generation strategies, run hyperparameter optimization sweeps, and export the winning pipeline as a standalone artifact.

## Core Capabilities
- **Automated Synthetic Dataset Builder:** Synthesize Q&A evaluation datasets from document collections.
- **Hyperparameter Optimization Sweep:** Execute Grid Search, Random Sampling Search, and LLM-Guided Strategy (Adaptive Feedback) across parameters like chunk size, chunk overlap, and retriever strategy.
- **Modular Retrieval Algorithms:** Dense vector retrieval, Sparse (TF-IDF/BM25) matching, and Hybrid retrieval with Maximal Marginal Relevance (MMR) re-ranking.
- **Multi-Provider Connections:** Native support for OpenAI, Ollama, LM Studio, OpenRouter, and local models.
- **Standalone Package Exporter:** Export trial configurations as self-contained ZIP packages containing vector stores and code.
- **Generalization Test:** Validate optimized configurations against hold-out synthetic questions.

## Architecture
- **Frontend:** React (Vite) Single Page Application.
- **Backend Server:** Node.js (Express + TSX/esbuild) orchestrating requests.
- **Engine:** Python AutoRAG Engine (`cli.py` & `autorag/`) containing modular subsystems for embeddings, knowledge base, optimization, and evaluation.

## Requirements
- **Node.js**: v18+ (for frontend and backend server).
- **Python**: v3.9+ (for AutoRAG Engine).
- **OS**: Linux, macOS, or Windows.

## Installation
The application uses npm for node dependencies and automates Python virtual environment resolution.
```bash
npm install
```

## Running the Application
### Development
```bash
npm run dev
```
The application runs on `http://localhost:3000`. Python dependencies and the virtual environment will be automatically bootstrapped on first use.

### Production Build
```bash
npm run build
npm start
```

## Connection Profiles
The platform supports modular connection profiles. You can configure LLM and Embedding providers independently (e.g., OpenAI for generation, Local TF-IDF for embeddings). Environment variables are injected into the Python engine at runtime.

## Knowledge Base
Users can ingest documents into the Workspace. The Knowledge Base engine calculates checksums, manages chunking (Fixed, Recursive, Paragraph, Semantic), and maintains document manifests for downstream tasks.

## Dataset Builder
The Dataset Builder generates synthetic Q&A pairs (ground truth datasets) using the configured LLM, anchoring questions to document chunks for accurate evaluation.

## Hyperparameter Optimization Engine
The Optimization Engine explores the design space using three primary search strategies:
- **Grid Search (Exhaustive Combinations):** Exhaustively evaluates all combinations of the provided parameters.
- **Random Sampling Search:** Selects random combinations up to a maximum trial count.
- **LLM-Guided Strategy (Adaptive Feedback):** Uses an LLM to sequentially propose hyperparameter candidates based on the results and metrics of previous trials.

Compatible retained optimization history may guide a new LLM-Guided optimization run if "Learn from Previous Trials" is enabled.

## Generalization Test
The Generalization Test evaluates a completed optimization Trial on newly generated evaluation questions that were not used during the optimization process. This provides validation against overfitting. 
Users control the requested question count. Generalization metrics remain separate from optimization metrics.

## Workspace
The workspace is a file-system-based manager (default: `workspace/` in the project root) handling state, manifest storage, cached artifacts, experiment logs, and local databases.

## Report Engine
Generates detailed reports (Markdown, HTML, CSV) summarizing trial results and optimization leaderboards.

## Environment Lifecycle
The application features an automated environment lifecycle manager that dynamically provisions a Python virtual environment (`.venv`). It enforces CUDA-enabled PyTorch canonical installations and installs auxiliary libraries (`transformers`, `sentence-transformers`, `requests`, `pypdf`) at runtime. Embedding device compute (CPU vs. CUDA) is resolved dynamically.

## Development
```bash
# Type check / Lint
npm run lint

# Build artifacts
npm run build

# Clean
npm run clean
```

## Repository Structure
- `src/` - React frontend source code.
- `server/` - Node.js Express backend configurations.
- `server.ts` - Express backend entry point.
- `autorag/` - Python AutoRAG engine modules.
- `cli.py` - JSON RPC boundary for the Python engine.
- `tests/` - Python test suite.

## Security
Do not commit active API keys. Use connection profiles and `.env` for managing secrets locally. The application manages local runtime state; ensure `.env` and `workspace/` remain ignored in version control.

## License
MIT License.
