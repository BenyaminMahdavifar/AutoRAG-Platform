"""
Knowledge Base Subsystem for AutoRAG Platform.
Handles document scanning, format parsing, metadata extraction, checksum generation, and manifest building.
Deterministic code only.
"""

import os
import json
import csv
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .types import Document, KnowledgeBaseManifest


class KnowledgeBaseScanner:
    """Scans directory, loads documents, extracts metadata, builds deterministic manifest."""

    SUPPORTED_EXTENSIONS = {
        ".md", ".txt", ".json", ".pdf"
    }

    def __init__(self, kb_dir: str = "workspace/kb"):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.kb_dir / ".scanner_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cache(self):
        try:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def compute_sha256(self, content: str) -> str:
        """Compute SHA256 hash of text content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def parse_file(self, filepath: Path) -> Optional[Document]:
        """Parse file into Document dataclass with extracted metadata."""
        if not filepath.is_file() or filepath.name == ".scanner_cache.json":
            return None

        ext = filepath.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return None

        try:
            raw_bytes = filepath.read_bytes()
            size_bytes = len(raw_bytes)
            
            file_hash = hashlib.sha256(raw_bytes).hexdigest()
            cache_key = f"{filepath.name}_{size_bytes}_{file_hash}"
            
            if cache_key in self.cache:
                # Need to convert dict back to Document if necessary, but **kwargs works if matching
                return Document(**self.cache[cache_key])

            text_content = ""

            try:
                relative_path = str(filepath.relative_to(self.kb_dir)).replace("\\", "/")
            except ValueError:
                relative_path = filepath.name

            folder_path = str(Path(relative_path).parent).replace("\\", "/")
            if folder_path == ".":
                folder_path = ""

            doc_id = hashlib.md5(f"{relative_path}:{size_bytes}".encode("utf-8")).hexdigest()[:12]

            metadata = {
                "document_id": doc_id,
                "source_type": "file",
                "source_path": str(filepath),
                "relative_path": relative_path,
                "filename": filepath.name,
                "extension": ext,
                "size": size_bytes,
                "folder_path": folder_path,
            }

            if ext == ".pdf":
                try:
                    import pypdf
                    import io
                    reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
                    pages = []
                    page_meta = []
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                            page_meta.append({"page": i + 1, "chars": len(text)})
                    text_content = "\n\n".join(pages)
                    metadata["page_info"] = page_meta
                except ImportError:
                    print(f"pypdf not installed, cannot read PDF {filepath}")
                    return None
                except Exception as e:
                    print(f"Error extracting PDF {filepath}: {e}")
                    return None
            else:
                text_content = raw_bytes.decode("utf-8", errors="replace")
                
            text_content = text_content.replace("\r\n", "\n").replace("\u200b", "")

            if ext == ".md":
                headings = re.findall(r"^#+\s+(.+)$", text_content, flags=re.MULTILINE)
                metadata["headings"] = headings
            elif ext == ".json":
                try:
                    metadata["raw_json_text"] = text_content
                    parsed_json = json.loads(text_content)
                    if isinstance(parsed_json, dict):
                        metadata["keys"] = list(parsed_json.keys())
                        text_content = json.dumps(parsed_json, indent=2)
                    elif isinstance(parsed_json, list):
                        metadata["array_length"] = len(parsed_json)
                        text_content = json.dumps(parsed_json, indent=2)
                except Exception:
                    pass

            metadata["char_count"] = len(text_content)
            metadata["line_count"] = len(text_content.splitlines())
            metadata["estimated_tokens"] = max(1, len(text_content) // 4)

            checksum = self.compute_sha256(text_content)

            doc = Document(
                doc_id=doc_id,
                filepath=relative_path,
                filename=filepath.name,
                file_type=ext.lstrip("."),
                content=text_content,
                metadata=metadata,
                checksum=checksum,
                size_bytes=size_bytes
            )
            
            self.cache[cache_key] = doc.__dict__
            return doc

        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")
            return None

    def scan(self) -> Tuple[List[Document], KnowledgeBaseManifest]:
        """Scan directory, returning list of Document objects and a KnowledgeBaseManifest."""
        from typing import Tuple
        
        documents: List[Document] = []
        files_metadata: List[Dict[str, Any]] = []

        all_files = sorted(self.kb_dir.rglob("*"))
        for filepath in all_files:
            if filepath.is_file():
                doc = self.parse_file(filepath)
                if doc:
                    documents.append(doc)
                    files_metadata.append({
                        "doc_id": doc.doc_id,
                        "filename": doc.filename,
                        "filepath": doc.filepath,
                        "size_bytes": doc.size_bytes,
                        "checksum": doc.checksum,
                        "file_type": doc.file_type,
                        "char_count": doc.metadata.get("char_count", 0),
                    })

        # Calculate combined KB checksum
        combined_checksums = "".join([f["checksum"] for f in files_metadata])
        kb_checksum = hashlib.sha256(combined_checksums.encode("utf-8")).hexdigest()

        total_size = sum(f["size_bytes"] for f in files_metadata)
        manifest = KnowledgeBaseManifest(
            scanned_at=datetime.now().isoformat(),
            total_docs=len(documents),
            total_size_bytes=total_size,
            kb_checksum=kb_checksum,
            files=files_metadata
        )

        self._save_cache()

        return documents, manifest

    def scan_and_load(self) -> Tuple[List[Document], KnowledgeBaseManifest]:
        """Convenience method to scan and load documents and manifest."""
        return self.scan()
