"""Focused tests for Phase 5 admin runtime and knowledge staging behavior."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from backend.api.knowledge import KnowledgeHubService, KnowledgeStoreUnavailableError
from backend.api.runtime import ProviderRuntimeStore
from backend.api.schemas import ProviderSettingsUpdate
from backend.config.settings import AppSettings
from backend.rag.chunker import DocumentChunk
from backend.rag.embeddings import DefaultLocalEmbeddingFunction
from backend.rag.ingestion import DocumentIngestionPipeline
from backend.rag.vector_store import BaseVectorStore, ChromaVectorStore, MemoryVectorStore


class DurableMemoryVectorStore(BaseVectorStore):
    """Test adapter for persistent-store behavior without filesystem Chroma state."""

    def __init__(self) -> None:
        self._memory = MemoryVectorStore()

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        return self._memory.add_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 2,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[DocumentChunk]:
        return self._memory.search(query, top_k, metadata_filter)

    def delete_by_doc(self, doc_name: str) -> int:
        return self._memory.delete_by_doc(doc_name)

    def clear(self) -> None:
        self._memory.clear()

    def begin_staged_reindex(self) -> None:
        self._memory.begin_staged_reindex()

    def commit_staged_reindex(self) -> None:
        self._memory.commit_staged_reindex()

    def discard_staged_reindex(self) -> None:
        self._memory.discard_staged_reindex()


def test_provider_runtime_persists_only_non_secret_selection(tmp_path: Path) -> None:
    """Provider switching persists model metadata without embedding an API credential."""
    state_path = tmp_path / "provider-state.json"
    runtime = ProviderRuntimeStore(AppSettings(), state_path=state_path)

    response = runtime.update(
        ProviderSettingsUpdate(
            provider="ollama",
            model="qwen2.5:7b",
            base_url="http://localhost:11434/v1",
        )
    )

    assert response.active.provider == "ollama"
    assert response.active.model == "qwen2.5:7b"
    persisted = state_path.read_text(encoding="utf-8")
    assert "qwen2.5:7b" in persisted
    assert "api_key" not in persisted


def test_staged_reindex_replaces_one_document_and_keeps_other_knowledge(tmp_path: Path) -> None:
    """A replacement upload must not delete a tenant's unrelated documents."""
    store = MemoryVectorStore()
    pipeline = DocumentIngestionPipeline(vector_store=store)
    first_document = tmp_path / "arrival.md"
    second_document = tmp_path / "parking.md"
    replacement_document = tmp_path / "arrival.md"

    first_document.write_text("Check-in begins at 3 PM.", encoding="utf-8")
    pipeline.reindex_atomically(first_document, metadata={"tenant_id": "grand-meridian"})
    second_document.write_text("Valet parking costs 35 dollars per night.", encoding="utf-8")
    pipeline.reindex_atomically(second_document, metadata={"tenant_id": "grand-meridian"})
    replacement_document.write_text("Check-in begins at 4 PM.", encoding="utf-8")
    pipeline.reindex_atomically(replacement_document, metadata={"tenant_id": "grand-meridian"})

    check_in = store.search("When does check-in begin?", metadata_filter={"tenant_id": "grand-meridian"})
    parking = store.search("How much is valet parking?", metadata_filter={"tenant_id": "grand-meridian"})

    assert check_in and "4 PM" in check_in[0].text
    assert parking and "35 dollars" in parking[0].text


def test_failed_empty_reindex_keeps_previous_document_live(tmp_path: Path) -> None:
    """A failed replacement cannot promote a staged collection with the old file removed."""
    store = MemoryVectorStore()
    pipeline = DocumentIngestionPipeline(vector_store=store)
    document = tmp_path / "policy.md"
    document.write_text("The spa opens at 8 AM.", encoding="utf-8")
    pipeline.reindex_atomically(document, metadata={"tenant_id": "grand-meridian"})

    document.write_text(" \n ", encoding="utf-8")
    try:
        pipeline.reindex_atomically(document, metadata={"tenant_id": "grand-meridian"})
    except ValueError as exc:
        assert "No readable text" in str(exc)
    else:
        raise AssertionError("An empty replacement must fail rather than promote.")

    results = store.search("When does the spa open?", metadata_filter={"tenant_id": "grand-meridian"})
    assert results and "8 AM" in results[0].text


def test_knowledge_upload_fails_explicitly_when_only_memory_storage_is_available(
    tmp_path: Path,
) -> None:
    """An ephemeral fallback must not be reported as a durable indexed upload."""
    service = KnowledgeHubService(AppSettings(), data_dir=tmp_path / "knowledge")
    with patch("backend.api.knowledge.get_vector_store", return_value=MemoryVectorStore()):
        try:
            service.save_and_index("grand-meridian", "policy.md", b"The spa opens at 8 AM.")
        except KnowledgeStoreUnavailableError as exc:
            assert "Persistent knowledge storage is unavailable" in str(exc)
        else:
            raise AssertionError("An in-memory fallback must not report a successful durable upload.")
    assert not (tmp_path / "knowledge" / "uploads" / "grand-meridian").exists()
    assert service.list_documents("grand-meridian") == []


def test_voice_retriever_reports_unavailable_without_blocking_session_setup(tmp_path: Path) -> None:
    """Voice can construct a retriever that reports the unavailable store per turn."""
    service = KnowledgeHubService(AppSettings(), data_dir=tmp_path / "knowledge")
    with patch("backend.api.knowledge.get_vector_store", return_value=MemoryVectorStore()):
        retriever = service.retriever_for("grand-meridian")

    messages = [{"role": "user", "content": "What time is check-in?"}]
    result = asyncio.run(retriever.update_turn_context(messages, "What time is check-in?"))

    assert result["status"] == "unavailable"
    assert "search is unavailable" in messages[-1]["content"]


def test_upload_rolls_back_file_and_index_when_manifest_write_fails(tmp_path: Path) -> None:
    """A manifest failure retains the previous file and searchable document version."""
    service = KnowledgeHubService(AppSettings(), data_dir=tmp_path / "knowledge")
    store = DurableMemoryVectorStore()
    service._stores["grand-meridian"] = store
    service.save_and_index("grand-meridian", "policy.md", b"The spa opens at 8 AM.")
    destination = tmp_path / "knowledge" / "uploads" / "grand-meridian" / "policy.md"

    with patch.object(service, "_write_manifest", side_effect=OSError("manifest write failed")):
        try:
            service.save_and_index("grand-meridian", "policy.md", b"The spa opens at 9 AM.")
        except OSError as exc:
            assert "manifest write failed" in str(exc)
        else:
            raise AssertionError("A failed manifest update must fail the upload.")

    assert destination.read_bytes() == b"The spa opens at 8 AM."
    assert service.list_documents("grand-meridian")[0].chunks_indexed == 1
    results = store.search("When does the spa open?", metadata_filter={"tenant_id": "grand-meridian"})
    assert results and "8 AM" in results[0].text


def test_upload_rolls_back_file_and_manifest_when_index_promotion_fails(tmp_path: Path) -> None:
    """A failed active-index pointer promotion restores the previous file and manifest."""
    service = KnowledgeHubService(AppSettings(), data_dir=tmp_path / "knowledge")
    store = DurableMemoryVectorStore()
    service._stores["grand-meridian"] = store
    service.save_and_index("grand-meridian", "policy.md", b"The spa opens at 8 AM.")
    destination = tmp_path / "knowledge" / "uploads" / "grand-meridian" / "policy.md"

    with patch.object(store, "commit_staged_reindex", side_effect=OSError("index promotion failed")):
        try:
            service.save_and_index("grand-meridian", "policy.md", b"The spa opens at 9 AM.")
        except OSError as exc:
            assert "index promotion failed" in str(exc)
        else:
            raise AssertionError("A failed index promotion must fail the upload.")

    assert destination.read_bytes() == b"The spa opens at 8 AM."
    assert service.list_documents("grand-meridian")[0].chunks_indexed == 1
    results = store.search("When does the spa open?", metadata_filter={"tenant_id": "grand-meridian"})
    assert results and "8 AM" in results[0].text


def test_corrupt_manifest_blocks_upload_without_discarding_existing_index(tmp_path: Path) -> None:
    """Unreadable metadata must not be silently replaced with an empty manifest."""
    service = KnowledgeHubService(AppSettings(), data_dir=tmp_path / "knowledge")
    store = DurableMemoryVectorStore()
    service._stores["grand-meridian"] = store
    store.add_chunks(
        [
            DocumentChunk(
                id="existing-policy",
                text="The spa opens at 8 AM.",
                metadata={"tenant_id": "grand-meridian", "doc_name": "existing.md"},
            )
        ]
    )
    service._manifest_path.parent.mkdir(parents=True, exist_ok=True)
    service._manifest_path.write_text("{invalid json", encoding="utf-8")

    try:
        service.save_and_index("grand-meridian", "new-policy.md", b"The pool closes at 10 PM.")
    except RuntimeError as exc:
        assert "manifest cannot be read" in str(exc)
    else:
        raise AssertionError("A corrupt manifest must not be interpreted as empty.")

    assert not (tmp_path / "knowledge" / "uploads" / "grand-meridian" / "new-policy.md").exists()
    assert store.search("When does the spa open?", metadata_filter={"tenant_id": "grand-meridian"})
    assert service._manifest_path.read_text(encoding="utf-8") == "{invalid json"


def test_concurrent_tenant_uploads_preserve_shared_manifest_entries(tmp_path: Path) -> None:
    """Different tenants can upload concurrently without losing manifest records."""
    service = KnowledgeHubService(AppSettings(), data_dir=tmp_path / "knowledge")
    stores = {"tenant-a": DurableMemoryVectorStore(), "tenant-b": DurableMemoryVectorStore()}

    def get_test_store(*, tenant_id: str, **kwargs: object) -> BaseVectorStore:
        return stores[tenant_id]

    barrier = Barrier(2)

    def upload(tenant_id: str) -> None:
        barrier.wait()
        service.save_and_index(tenant_id, "policy.md", f"{tenant_id} hotel policy.".encode())

    with patch("backend.api.knowledge.get_vector_store", side_effect=get_test_store):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(upload, tenant_id) for tenant_id in stores]
            for future in futures:
                future.result()

    manifest = json.loads((tmp_path / "knowledge" / "knowledge_documents.json").read_text(encoding="utf-8"))
    assert {entry["tenant_id"] for entry in manifest} == set(stores)
    for tenant_id in stores:
        documents = service.list_documents(tenant_id)
        assert len(documents) == 1
        assert documents[0].name == "policy.md"


def test_document_identity_includes_the_extension_when_requested(tmp_path: Path) -> None:
    """Admin uploads with matching stems keep distinct indexed source identities."""
    store = MemoryVectorStore()
    pipeline = DocumentIngestionPipeline(vector_store=store)
    markdown_document = tmp_path / "policies.md"
    text_document = tmp_path / "policies.txt"
    markdown_document.write_text("The pool closes at 10 PM.", encoding="utf-8")
    text_document.write_text("Valet parking costs 35 dollars.", encoding="utf-8")

    pipeline.reindex_atomically(
        markdown_document,
        metadata={"tenant_id": "grand-meridian"},
        document_id="policies.md",
    )
    pipeline.reindex_atomically(
        text_document,
        metadata={"tenant_id": "grand-meridian"},
        document_id="policies.txt",
    )

    sources = {chunk.metadata["doc_name"] for chunk in store.search("pool parking")}
    assert sources == {"policies.md", "policies.txt"}


def test_chroma_staging_promotes_a_complete_collection_without_deleting_current_reader(
    tmp_path: Path,
) -> None:
    """Chroma promotion changes only the active pointer, preserving current readers."""
    store = ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="phase_five_staging",
        embedding_function=DefaultLocalEmbeddingFunction(),
    )
    assert store.is_available
    pipeline = DocumentIngestionPipeline(vector_store=store)
    document = tmp_path / "policy.md"
    document.write_text("The spa opens at 8 AM.", encoding="utf-8")
    pipeline.reindex_atomically(document, metadata={"tenant_id": "grand-meridian"})
    current_reader = store._collection

    document.write_text("The spa opens at 9 AM.", encoding="utf-8")
    pipeline.reindex_atomically(document, metadata={"tenant_id": "grand-meridian"})

    assert current_reader.get().get("ids")
    reloaded = ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="phase_five_staging",
        embedding_function=DefaultLocalEmbeddingFunction(),
    )
    results = reloaded.search("When does the spa open?", metadata_filter={"tenant_id": "grand-meridian"})
    assert results and "9 AM" in results[0].text
