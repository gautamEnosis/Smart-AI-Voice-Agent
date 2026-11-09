"""Tenant-scoped Knowledge Hub services for safe local document ingestion."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from threading import RLock
from uuid import uuid4

from backend.api.schemas import KnowledgeDocumentResponse, KnowledgeSearchResponse
from backend.config.settings import AppSettings, get_settings
from backend.rag.ingestion import DocumentIngestionPipeline
from backend.rag.retriever import RAGContextRetriever
from backend.rag.vector_store import (
    BaseVectorStore,
    MemoryVectorStore,
    UnavailableVectorStore,
    VectorStoreUnavailableError,
    get_vector_store,
)

logger = logging.getLogger(__name__)

ALLOWED_DOCUMENT_EXTENSIONS = {".md", ".pdf", ".txt"}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
SAFE_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class KnowledgeStoreUnavailableError(VectorStoreUnavailableError):
    """Raised when durable tenant knowledge storage is not available."""


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
        self._tenant_write_locks: dict[str, RLock] = {}
        self._lock = RLock()
        self._manifest_lock = RLock()

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

    def _write_lock_for(self, tenant_id: str) -> RLock:
        with self._lock:
            if tenant_id not in self._tenant_write_locks:
                self._tenant_write_locks[tenant_id] = RLock()
            return self._tenant_write_locks[tenant_id]

    def _durable_store_for(self, tenant_id: str) -> BaseVectorStore:
        """Return the tenant store only when it is backed by persistent ChromaDB."""
        store = self._store_for(tenant_id)
        if isinstance(store, MemoryVectorStore):
            raise KnowledgeStoreUnavailableError(
                "Persistent knowledge storage is unavailable. No document was indexed; "
                "check the ChromaDB installation and data directory."
            )
        return store

    def retriever_for(self, tenant_id: str) -> RAGContextRetriever:
        """Create a tenant retriever without making voice startup depend on ChromaDB."""
        tenant_id = self.validate_tenant_id(tenant_id)
        store = self._store_for(tenant_id)
        if isinstance(store, MemoryVectorStore):
            return RAGContextRetriever(
                vector_store=UnavailableVectorStore(
                    "Persistent ChromaDB is unavailable; uploaded knowledge cannot be searched."
                )
            )
        return RAGContextRetriever(vector_store=store)

    def _load_manifest(self) -> list[dict[str, object]]:
        if not self._manifest_path.exists():
            return []
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("The knowledge document manifest cannot be read.") from exc
        if not isinstance(raw, list) or any(not isinstance(entry, dict) for entry in raw):
            raise RuntimeError("The knowledge document manifest has an invalid format.")
        return raw

    def _write_manifest(self, entries: list[dict[str, object]]) -> None:
        with self._manifest_lock:
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._manifest_path.with_name(
                f".{self._manifest_path.name}.{uuid4().hex}.tmp"
            )
            try:
                temporary_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
                temporary_path.replace(self._manifest_path)
            finally:
                temporary_path.unlink(missing_ok=True)

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

    def storage_health(self, tenant_id: str = "grand-meridian") -> tuple[bool, str]:
        """Report whether this tenant's knowledge store is persistent and usable."""
        tenant_id = self.validate_tenant_id(tenant_id)
        store = self._store_for(tenant_id)
        if isinstance(store, MemoryVectorStore):
            return (
                False,
                "Persistent ChromaDB is unavailable; knowledge uploads and live retrieval are disabled.",
            )
        return (
            True,
            "Persistent tenant-isolated ChromaDB is available; "
            f"configured embeddings: {self._settings.embedding_provider}.",
        )

    def save_and_index(
        self,
        tenant_id: str,
        original_filename: str,
        content: bytes,
    ) -> KnowledgeDocumentResponse:
        """Persist an upload and stage it into the tenant's active knowledge base."""
        tenant_id = self.validate_tenant_id(tenant_id)
        with self._write_lock_for(tenant_id):
            return self._save_and_index_locked(tenant_id, original_filename, content)

    def _save_and_index_locked(
        self,
        tenant_id: str,
        original_filename: str,
        content: bytes,
    ) -> KnowledgeDocumentResponse:
        """Validate, persist, and index one upload while holding its tenant write lock."""
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

        store = self._durable_store_for(tenant_id)
        tenant_dir = self._uploads_dir / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        destination = tenant_dir / filename
        upload_id = uuid4().hex
        temporary_path = destination.with_name(f".{destination.stem}.{upload_id}.upload{destination.suffix}")
        backup_path = destination.with_name(f".{destination.name}.{upload_id}.backup")
        pipeline = DocumentIngestionPipeline(vector_store=store)
        staged = False
        previous_entries: list[dict[str, object]] | None = None
        manifest_updated = False
        destination_backed_up = False
        destination_promoted = False
        try:
            temporary_path.write_bytes(content)
            staged = True
            store.begin_staged_reindex()
            store.delete_by_doc(filename)
            indexed = pipeline.ingest_file(
                temporary_path,
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
            with self._manifest_lock:
                previous_entries = self._load_manifest()
                entries = [
                    entry
                    for entry in previous_entries
                    if not (entry.get("tenant_id") == tenant_id and entry.get("name") == original_filename)
                ]
                entries.append(record.model_dump(mode="json"))
                if destination.exists():
                    destination.replace(backup_path)
                    destination_backed_up = True
                temporary_path.replace(destination)
                destination_promoted = True
                self._write_manifest(entries)
                manifest_updated = True
                store.commit_staged_reindex()
                staged = False

            if destination_backed_up:
                try:
                    backup_path.unlink()
                except OSError:
                    logger.warning("Could not remove successful upload backup %s", backup_path, exc_info=True)
            return record
        except Exception as exc:
            rollback_errors: list[Exception] = []
            if manifest_updated and previous_entries is not None:
                try:
                    with self._manifest_lock:
                        self._write_manifest(previous_entries)
                except OSError as rollback_exc:
                    rollback_errors.append(rollback_exc)
                    logger.exception("Could not restore the knowledge document manifest")
            if destination_promoted:
                try:
                    destination.unlink(missing_ok=True)
                except OSError as rollback_exc:
                    rollback_errors.append(rollback_exc)
                    logger.exception("Could not remove the failed promoted upload %s", destination)
            if destination_backed_up:
                try:
                    backup_path.replace(destination)
                except OSError as rollback_exc:
                    rollback_errors.append(rollback_exc)
                    logger.exception("Could not restore the previous upload %s", destination)
            if staged:
                try:
                    store.discard_staged_reindex()
                except Exception as rollback_exc:
                    rollback_errors.append(rollback_exc)
                    logger.exception("Could not discard staged knowledge index changes")
            if rollback_errors:
                raise RuntimeError(
                    "Knowledge upload failed and rollback was incomplete; inspect backend logs."
                ) from exc
            raise
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary knowledge upload %s", temporary_path, exc_info=True)

    def search(self, tenant_id: str, query: str) -> KnowledgeSearchResponse:
        """Run a voice-guarded retrieval test against one tenant collection."""
        tenant_id = self.validate_tenant_id(tenant_id)
        retriever = RAGContextRetriever(vector_store=self._durable_store_for(tenant_id))
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


@lru_cache(maxsize=1)
def get_knowledge_hub() -> KnowledgeHubService:
    """Return the process-wide Knowledge Hub shared by Admin and voice sessions."""
    return KnowledgeHubService(get_settings())
