"""Tenant-scoped Knowledge Hub services for safe local document ingestion."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from backend.api.schemas import KnowledgeDocumentResponse, KnowledgeSearchResponse
from backend.config.settings import AppSettings
from backend.rag.ingestion import DocumentIngestionPipeline
from backend.rag.retriever import RAGContextRetriever
from backend.rag.vector_store import BaseVectorStore, get_vector_store

ALLOWED_DOCUMENT_EXTENSIONS = {".md", ".pdf", ".txt"}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
SAFE_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class KnowledgeHubService:
    """Coordinates uploads, staged reindexing, and retrieval testing per tenant."""

    def __init__(self, settings: AppSettings, data_dir: Path | None = None) -> None:
        self._settings = settings
        self._data_dir = data_dir or Path("data")
        self._uploads_dir = self._data_dir / "uploads"
        self._manifest_path = self._data_dir / "knowledge_documents.json"
        self._stores: dict[str, BaseVectorStore] = {}
        self._lock = RLock()

    @staticmethod
    def validate_tenant_id(tenant_id: str) -> str:
        """Validate tenant isolation keys before accessing local file or vector paths."""
        normalized = tenant_id.strip().lower()
        if not SAFE_TENANT_ID.fullmatch(normalized):
            raise ValueError("Tenant ID may contain lowercase letters, digits, hyphens, and underscores only.")
        return normalized

    @staticmethod
    def safe_filename(filename: str) -> str:
        """Return a stable, traversal-safe filename for document replacement semantics."""
        name = Path(filename).name.strip()
        extension = Path(name).suffix.lower()
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).stem).strip(".-_")
        if not stem or extension not in ALLOWED_DOCUMENT_EXTENSIONS:
            raise ValueError("Only PDF, Markdown, and text documents are supported.")
        return f"{stem[:120]}{extension}"

    def _store_for(self, tenant_id: str) -> BaseVectorStore:
        with self._lock:
            if tenant_id not in self._stores:
                self._stores[tenant_id] = get_vector_store(
                    persist_dir=self._settings.chroma_persist_dir,
                    tenant_id=tenant_id,
                )
            return self._stores[tenant_id]

    def _load_manifest(self) -> list[dict[str, object]]:
        if not self._manifest_path.exists():
            return []
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _write_manifest(self, entries: list[dict[str, object]]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._manifest_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        temporary_path.replace(self._manifest_path)

    def list_documents(self, tenant_id: str) -> list[KnowledgeDocumentResponse]:
        """List successfully indexed documents for the requested tenant."""
        tenant_id = self.validate_tenant_id(tenant_id)
        return [
            KnowledgeDocumentResponse.model_validate(entry)
            for entry in self._load_manifest()
            if entry.get("tenant_id") == tenant_id
        ]

    def document_count(self, tenant_id: str = "grand-meridian") -> int:
        """Return indexed document count for dashboard health metadata."""
        return len(self.list_documents(tenant_id))

    def save_and_index(
        self,
        tenant_id: str,
        original_filename: str,
        content: bytes,
    ) -> KnowledgeDocumentResponse:
        """Persist an upload and stage it into the tenant's active knowledge base."""
        tenant_id = self.validate_tenant_id(tenant_id)
        filename = self.safe_filename(original_filename)
        if not content:
            raise ValueError("The uploaded document is empty.")
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("Documents must be 10 MB or smaller.")

        # SEC-06 / 2.4: Magic-byte and payload inspection to prevent polyglot files
        extension = Path(filename).suffix.lower()
        if extension == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise ValueError("Invalid PDF format: file header must start with %PDF-.")
        elif extension in {".txt", ".md"}:
            if b"\x00" in content:
                raise ValueError("Text files must not contain binary null bytes.")
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Text and Markdown files must be valid UTF-8 encoded text.") from exc

        tenant_dir = self._uploads_dir / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        destination = tenant_dir / filename
        temporary_path = destination.with_suffix(f"{destination.suffix}.upload")
        temporary_path.write_bytes(content)
        temporary_path.replace(destination)

        store = self._store_for(tenant_id)
        pipeline = DocumentIngestionPipeline(vector_store=store)
        indexed = pipeline.reindex_atomically(
            destination,
            metadata={"tenant_id": tenant_id, "source": "admin_upload"},
            document_id=filename,
        )
        if indexed < 1:
            raise ValueError("No readable text was found in the uploaded document.")

        record = KnowledgeDocumentResponse(
            name=original_filename,
            tenant_id=tenant_id,
            chunks_indexed=indexed,
            uploaded_at=_utc_now(),
            status="indexed",
        )
        with self._lock:
            entries = [
                entry
                for entry in self._load_manifest()
                if not (entry.get("tenant_id") == tenant_id and entry.get("name") == original_filename)
            ]
            entries.append(record.model_dump(mode="json"))
            self._write_manifest(entries)
        return record

    def search(self, tenant_id: str, query: str) -> KnowledgeSearchResponse:
        """Run a voice-guarded retrieval test against one tenant collection."""
        tenant_id = self.validate_tenant_id(tenant_id)
        retriever = RAGContextRetriever(vector_store=self._store_for(tenant_id))
        result = retriever.query_with_context(query, metadata_filter={"tenant_id": tenant_id})
        chunks = [
            {
                "source": str(chunk.metadata.get("doc_name", "Knowledge document")),
                "excerpt": chunk.text[:800],
                "score": chunk.score,
            }
            for chunk in result["chunks"]
        ]
        return KnowledgeSearchResponse(
            has_context=result["has_context"],
            chunks=chunks,
            fallback_spoken=result["fallback_spoken"],
        )
