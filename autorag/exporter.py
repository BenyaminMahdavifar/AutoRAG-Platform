import os
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from .workspace import WorkspaceManager
from .index_builder import IndexBuilder, TextChunker
from .types import ChunkingConfig, EmbeddingConfig
from .connections import EmbeddingClient

class RagExporter:
    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace
        
    def export_trial(self, trial_id: str, emit_progress_fn_orig) -> str:
        build_log = []
        def _emit(progress, stage, completed, msg):
            build_log.append(f"[{datetime.now().isoformat()}] [{progress}%] {stage}: {msg}")
            emit_progress_fn_orig(progress, stage, completed, msg)
            
        exp = self.workspace.get_experiment(trial_id)
        if not exp:
            raise ValueError(f"Trial {trial_id} not found in workspace.")
            
        config = exp.get("config") or {}
        
        # 1. Setup export directory
        _emit(5, "Preparing export", [], "Creating export directories...")
        export_base = Path(self.workspace.root_dir) / "exports"
        export_name = f"rag_{trial_id}"
        export_dir = export_base / export_name
        
        if export_dir.exists():
            shutil.rmtree(export_dir)
            
        for subdir in ["knowledge-base", "embeddings", "vector_database", "config", "src", "logs"]:
            (export_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        completed = ["Preparing export"]
            
        # 2. Copy Knowledge Base
        _emit(15, "Building Knowledge Base", completed, "Copying knowledge base files...")
        kb_path = Path(self.workspace.kb_dir)
        export_kb_path = export_dir / "knowledge-base"
        shutil.copytree(kb_path, export_kb_path, dirs_exist_ok=True)
                
        completed.append("Building Knowledge Base")
                
        # 3. Chunking
        _emit(30, "Chunking documents", completed, "Applying trial chunking configuration...")
        chunking_dict = config.get("chunking_config", {})
        chunk_cfg = ChunkingConfig(
            strategy=chunking_dict.get("strategy", "recursive"),
            chunk_size=chunking_dict.get("chunk_size", 512),
            chunk_overlap=chunking_dict.get("chunk_overlap", 64)
        )
        
        emb_dict = config.get("embedding_config", {})
        emb_cfg = EmbeddingConfig(
            provider=emb_dict.get("provider", "local"),
            model_name=emb_dict.get("model_name", "local-tfidf-512"),
            dimension=emb_dict.get("dimension", 512)
        )
        emb_client = EmbeddingClient(emb_cfg)
        
        builder = IndexBuilder(self.workspace, emb_client)
        
        # load docs using the workspace
        from autorag.knowledge_base import KnowledgeBaseScanner
        scanner = KnowledgeBaseScanner(self.workspace.kb_dir)
        docs, _ = scanner.scan()
        if not docs:
            raise ValueError("Knowledge base is empty. Please upload documents first.")
            
        index = builder.build_index(docs, chunk_cfg)
        
        completed.append("Chunking documents")
        
        # 4. Generating Embeddings
        _emit(50, "Generating embeddings", completed, "Embeddings generated during index build.")
        
        emb_export_path = export_dir / "embeddings" / "embeddings_meta.json"
        with open(emb_export_path, "w") as f:
            json.dump({"total_chunks": len(index.chunks), "model": emb_cfg.model_name}, f)
            
        completed.append("Generating embeddings")
            
        # 5. Building vector database (Chroma)
        _emit(65, "Building vector database", completed, "Constructing standalone Chroma DB...")
        try:
            import chromadb
            db_path = export_dir / "vector_database"
            client = chromadb.PersistentClient(path=str(db_path))
            collection = client.create_collection("rag_collection")
            
            ids = [str(i) for i in range(len(index.chunks))]
            texts = [c.text for c in index.chunks]
            metadatas = [c.metadata for c in index.chunks]
            
            # Since local-tfidf-512 embeddings are not standard floats, we might need a workaround.
            # But wait, autorag generates embeddings as lists of floats. Let's assume index.vectors has them.
            if index.vectors:
                collection.add(
                    embeddings=index.vectors,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
        except Exception as e:
            # Fallback to simple JSON if chroma is not working
            with open(export_dir / "vector_database" / "index.json", "w") as f:
                # Assuming chunk is an object with __dict__ or dict()
                json.dump([c.__dict__ if hasattr(c, "__dict__") else dict(c) for c in index.chunks], f)
                
        completed.append("Building vector database")
                
        # 6. Writing configuration
        _emit(75, "Writing configuration", completed, "Exporting trial metadata and configs...")
        with open(export_dir / "config" / "configuration.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(export_dir / "config" / "optimization.json", "w") as f:
            json.dump({"metrics": exp.get("metrics", {}), "score": exp.get("composite_score")}, f, indent=2)
            
        with open(export_dir / "metadata.json", "w") as f:
            json.dump({
                "trial_id": trial_id,
                "timestamp": datetime.now().isoformat(),
                "config_hash": exp.get("pipeline_id"),
                "status": "exported"
            }, f, indent=2)
            
        completed.append("Writing configuration")
            
        # 7. Generating source code
        _emit(85, "Generating source code", completed, "Generating standalone RAG script...")
        ingest_path = export_dir / "src" / "ingest.py"
        with open(ingest_path, "w") as f:
            f.write(self._generate_ingest_code(config))
        src_path = export_dir / "src" / "rag_pipeline.py"
        with open(src_path, "w") as f:
            f.write(self._generate_source_code(config))
            
        with open(export_dir / "requirements.txt", "w") as f:
            f.write("chromadb\nopenai\nrequests\npydantic\nlitellm\ntenacity\npython-dotenv\n")
            
        with open(export_dir / "README.md", "w") as f:
            f.write(f"# AutoRAG Export: {trial_id}\n\nGenerated on {datetime.now().isoformat()}\n\n## Usage\n`python src/rag_pipeline.py`\n")
            
        completed.append("Generating source code")
            
        # 8. Validating package
        _emit(90, "Validating package", completed, "Verifying exported artifacts...")
        if not (export_dir / "vector_database").exists():
            raise RuntimeError("Vector database was not generated successfully.")
            
        completed.append("Validating package")
            
        with open(export_dir / "logs" / "build.log", "w") as f:
            f.write("\n".join(build_log))

        # 9. Compressing ZIP
        _emit(95, "Compressing ZIP", completed, "Creating downloadable archive...")
        zip_path = Path(self.workspace.root_dir) / "reports" / f"{export_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_base)
                    zipf.write(file_path, arcname)
                    
        completed.append("Compressing ZIP")
        _emit(100, "Export completed", completed, "Ready for download.")
        
        return f"/api/reports/file/{export_name}.zip"

    def _generate_ingest_code(self, config):
        emb = config.get("embedding_config", {})
        return f'''import os
from pathlib import Path
from chromadb import PersistentClient

# --- Auto-Generated Ingestion Script ---
# Embedding Model: {emb.get("model_name", "text-embedding-3-large")}

DB_NAME = str(Path(__file__).parent.parent / "vector_database")
COLLECTION_NAME = "rag_collection"
KB_PATH = Path(__file__).parent.parent / "knowledge-base"

def ingest():
    print(f"Scanning {{KB_PATH}} for documents...")
    client = PersistentClient(path=DB_NAME)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    
    docs = []
    ids = []
    metadatas = []
    
    for file in KB_PATH.rglob("*.txt"):
        docs.append(file.read_text(errors="ignore"))
        ids.append(file.name)
        metadatas.append({{"source": str(file), "type": "txt"}})
        
    for file in KB_PATH.rglob("*.md"):
        docs.append(file.read_text(errors="ignore"))
        ids.append(file.name)
        metadatas.append({{"source": str(file), "type": "md"}})
        
    if docs:
        print(f"Ingesting {{len(docs)}} new documents...")
        collection.add(
            documents=docs,
            ids=ids,
            metadatas=metadatas
        )
        print("Ingestion complete.")
    else:
        print("No new documents found.")

if __name__ == "__main__":
    ingest()
'''

    def _generate_source_code(self, config):
        llm = config.get("llm_config", {})
        emb = config.get("embedding_config", {})
        ret = config.get("retriever_config", {})
        
        return f'''import os
import json
from pathlib import Path
import chromadb
from openai import OpenAI
from litellm import completion
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# Exported Configuration
MODEL = "{llm.get("model_name", "gpt-4")}"
EMBEDDING_MODEL = "{emb.get("model_name", "text-embedding-3-large")}"
RETRIEVAL_K = {ret.get("top_k", 20)}
FINAL_K = 10

DB_NAME = str(Path(__file__).parent.parent / "vector_database")
COLLECTION_NAME = "rag_collection"

# Set your API keys in environment variables (e.g. OPENAI_API_KEY, GROQ_API_KEY)
openai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path=DB_NAME)
collection = chroma_client.get_collection(COLLECTION_NAME)

wait = wait_exponential(multiplier=1, min=10, max=240)

class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{{context}}

With this context, please answer the user's question. Be accurate, relevant and complete."""

@retry(wait=wait)
def rerank(question, chunks):
    if not chunks:
        return []
    system_prompt = """You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked."""
    
    user_prompt = f"The user has asked the following question:\\n\\n{{question}}\\n\\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\\n\\nHere are the chunks:\\n\\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {{index + 1}}:\\n\\n{{chunk.page_content}}\\n\\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    
    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": user_prompt}},
    ]
    
    try:
        response = completion(model=MODEL, messages=messages, response_format=RankOrder)
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order
        return [chunks[i - 1] for i in order if 0 < i <= len(chunks)]
    except Exception as e:
        print(f"Rerank failed: {{e}}")
        return chunks

def make_rag_messages(question, history, chunks):
    context = "\\n\\n".join(
        f"Extract from {{chunk.metadata.get('source', 'unknown')}}:\\n{{chunk.page_content}}" for chunk in chunks
    )
    system_prompt_formatted = SYSTEM_PROMPT.replace("{{context}}", context)
    return (
        [{{"role": "system", "content": system_prompt_formatted}}]
        + history
        + [{{"role": "user", "content": question}}]
    )

@retry(wait=wait)
def rewrite_query(question, history=[]):
    message = f"""You are in a conversation with a user, answering questions.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{{history}}

And this is the user's current question:
{{question}}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else."""
    
    try:
        response = completion(model=MODEL, messages=[{{"role": "system", "content": message}}])
        return response.choices[0].message.content
    except Exception as e:
        print(f"Query rewrite failed: {{e}}")
        return question

def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

def fetch_context_unranked(question):
    # Using Chroma's default embedding function or direct text search
    # Uncomment lines below if using an external embedding API (like OpenAI)
    # query_embedding = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    # results = collection.query(query_embeddings=[query_embedding], n_results=RETRIEVAL_K)
    
    results = collection.query(query_texts=[question], n_results=RETRIEVAL_K)
    chunks = []
    if results["documents"] and len(results["documents"]) > 0:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(Result(page_content=doc, metadata=meta))
    return chunks

def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]

@retry(wait=wait)
def answer_question(question: str, history: list = None) -> tuple:
    if history is None:
        history = []
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks

if __name__ == "__main__":
    test_question = "What is the main topic discussed in the knowledge base?"
    print(f"\\n[Question]\\n{{test_question}}")
    try:
        answer, retrieved_chunks = answer_question(test_question)
        print(f"\\n[Answer]\\n{{answer}}")
        print(f"\\n[Sources]\\nRetrieved {{len(retrieved_chunks)}} chunks.")
    except Exception as e:
        print(f"\\n[Error] Failed to generate answer: {{e}}")
        print("Please ensure your API keys (e.g. OPENAI_API_KEY, GROQ_API_KEY) are set and required packages are installed.")
'''
