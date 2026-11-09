"""Vector Store Abstraction and Connectors.

Provides a unified interface with two implementations:
1. ChromaVectorStore: Production embedded ChromaDB vector store.
2. MemoryVectorStore: Zero-dependency in-memory vector store with TF-IDF cosine similarity.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.rag.chunker import DocumentChunk
from backend.rag.embeddings import DefaultLocalEmbeddingFunction, get_embedding_function

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when a persistent vector store cannot complete an operation."""


class VectorStoreUnavailableError(VectorStoreError):
    """Raised when the configured persistent vector store is unavailable."""


def collection_name_for_tenant(tenant_id: str) -> str:
    """Returns a Chroma-safe collection name for a tenant knowledge base."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", (tenant_id or "default").strip().lower())
    return f"kb_{cleaned or 'default'}"


class BaseVectorStore(ABC):
    """Abstract interface for RAG vector storage and semantic retrieval."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Indexes document chunks into the store. Returns count of chunks added."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 2,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Searches store for chunks most similar to query, sorted by descending score."""

    @abstractmethod
    def delete_by_doc(self, doc_name: str) -> int:
        """Deletes all chunks belonging to a document. Returns count of chunks removed."""

    @abstractmethod
    def clear(self) -> None:
        """Clears all stored chunks."""

    def begin_staged_reindex(self) -> None:
        """Opens a staged collection so live callers are not blocked during re-index."""
        return None

    def commit_staged_reindex(self) -> None:
        """Atomically promotes the staged collection to live."""
        return None

    def discard_staged_reindex(self) -> None:
        """Discards an incomplete staged collection after an ingestion failure."""
        return None


class UnavailableVectorStore(BaseVectorStore):
    """Vector-store adapter that reports unavailable storage to live retrieval."""

    def __init__(self, message: str = "Persistent knowledge storage is unavailable.") -> None:
        self._message = message

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        raise VectorStoreUnavailableError(self._message)

    def search(
        self,
        query: str,
        top_k: int = 2,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        raise VectorStoreUnavailableError(self._message)

    def delete_by_doc(self, doc_name: str) -> int:
        raise VectorStoreUnavailableError(self._message)

    def clear(self) -> None:
        raise VectorStoreUnavailableError(self._message)


class MemoryVectorStore(BaseVectorStore):
    """Lightweight, zero-dependency in-memory vector store with TF-IDF similarity.

    Guarantees 100% deterministic local testing and zero external dependencies.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._tokenized_docs: dict[str, list[str]] = {}
        self._doc_frequencies: Counter[str] = Counter()
        self._staged_store: MemoryVectorStore | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple alphanumeric tokenizer with stopword stripping."""
        words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        stopwords = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
            "is", "are", "was", "were", "it", "of", "with", "that", "this",
            "you", "your", "can", "do", "we", "our", "my", "have", "has",
            "how", "much", "many", "what", "where", "when", "who", "why", "which",
            "tell", "me", "about", "there", "any", "some", "get", "will", "would", "could",
        }
        return [w for w in words if w not in stopwords]

    def _compute_tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        """Computes a normalized TF-IDF vector for a list of tokens."""
        if not tokens:
            return {}
        tf = Counter(tokens)
        total_docs = max(1, len(self._chunks))

        vec: dict[str, float] = {}
        for token, count in tf.items():
            df = self._doc_frequencies.get(token, 0)
            idf = math.log((total_docs + 1) / (df + 1)) + 1.0
            vec[token] = (count / len(tokens)) * idf

        # L2 normalization
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            return {k: v / norm for k, v in vec.items()}
        return vec

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if self._staged_store is not None:
            return self._staged_store.add_chunks(chunks)
        added = 0
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            if chunk.id in self._chunks:
                self._remove_chunk(chunk.id)
            self._chunks[chunk.id] = chunk
            tokens = self._tokenize(chunk.text)
            self._tokenized_docs[chunk.id] = tokens
            for token in set(tokens):
                self._doc_frequencies[token] += 1
            added += 1
        return added

    def search(
        self,
        query: str,
        top_k: int = 2,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self._chunks:
            return []

        query_set = set(query_tokens)
        query_vec = self._compute_tfidf_vector(query_tokens)
        total_query_idf = sum(self._compute_tfidf_vector([t]).get(t, 1.0) for t in query_tokens) or 1.0

        scored_chunks: list[tuple[float, DocumentChunk]] = []

        for chunk_id, chunk in self._chunks.items():
            # Apply metadata filters
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            doc_tokens = self._tokenized_docs.get(chunk_id, [])
            doc_set = set(doc_tokens)
            matching_tokens = query_set.intersection(doc_set)

            if not matching_tokens:
                continue

            # Query keyword coverage: ratio of query terms present in document
            matched_idf = sum(self._compute_tfidf_vector([t]).get(t, 1.0) for t in matching_tokens)
            coverage = matched_idf / total_query_idf

            # Cosine component
            doc_vec = self._compute_tfidf_vector(doc_tokens)
            cosine = sum(query_vec.get(term, 0.0) * weight for term, weight in doc_vec.items())

            # Combined score favoring query term presence
            score = round(0.75 * coverage + 0.25 * cosine, 4)

            scored_chunk = DocumentChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata.copy(),
                score=score,
            )
            scored_chunks.append((score, scored_chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def _remove_chunk(self, chunk_id: str) -> None:
        tokens = self._tokenized_docs.pop(chunk_id, [])
        for token in set(tokens):
            self._doc_frequencies[token] -= 1
            if self._doc_frequencies[token] <= 0:
                del self._doc_frequencies[token]
        self._chunks.pop(chunk_id, None)

    def delete_by_doc(self, doc_name: str) -> int:
        if self._staged_store is not None:
            return self._staged_store.delete_by_doc(doc_name)
        to_delete = [
            cid for cid, chunk in self._chunks.items()
            if chunk.metadata.get("doc_name") == doc_name
        ]
        for cid in to_delete:
            self._remove_chunk(cid)
        return len(to_delete)

    def clear(self) -> None:
        if self._staged_store is not None:
            self._staged_store.clear()
            return
        self._chunks.clear()
        self._tokenized_docs.clear()
        self._doc_frequencies.clear()

    def begin_staged_reindex(self) -> None:
        """Clone live data so staged writes never affect concurrent readers."""
        staged = MemoryVectorStore()
        staged.add_chunks(list(self._chunks.values()))
        self._staged_store = staged

    def commit_staged_reindex(self) -> None:
        """Promote the staged copy after a successful ingestion."""
        if self._staged_store is None:
            return
        self._chunks = self._staged_store._chunks
        self._tokenized_docs = self._staged_store._tokenized_docs
        self._doc_frequencies = self._staged_store._doc_frequencies
        self._staged_store = None

    def discard_staged_reindex(self) -> None:
        """Drops the staged clone while leaving the live store unchanged."""
        self._staged_store = None


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB embedded vector store connector with tenant collections."""

    LOCAL_HASH_MIN_KEYWORD_COVERAGE = 0.40
    LOCAL_HASH_CANDIDATE_LIMIT = 32

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        tenant_id: str = "grand-meridian",
        embedding_function: Any | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.persist_dir = persist_dir or "./data/chroma_db"
        self.collection_name = collection_name or collection_name_for_tenant(tenant_id)
        self._pointer_path = Path(self.persist_dir) / f"{self.collection_name}.active.json"
        self._embedding_function = embedding_function or DefaultLocalEmbeddingFunction()
        self._client: Any = None
        self._collection: Any = None
        self._staged_collection: Any = None
        self._staged_collection_name: str | None = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            active_name = self._read_active_collection_name() or self.collection_name
            self._collection = self._get_or_create_collection(active_name)
            logger.info(
                "ChromaDB initialized at %s (collection: %s)",
                self.persist_dir,
                self.collection_name,
            )
        except Exception as exc:
            logger.warning("ChromaDB unavailable (%s). Falling back to MemoryVectorStore.", exc)
            self._client = None
            self._collection = None

    def _get_or_create_collection(self, name: str) -> Any:
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def _read_active_collection_name(self) -> str | None:
        """Read the tenant's persisted active collection pointer, if present."""
        if not self._pointer_path.exists():
            return None
        try:
            payload = loads(self._pointer_path.read_text(encoding="utf-8"))
            name = payload.get("active_collection")
            return name if isinstance(name, str) and name else None
        except (OSError, JSONDecodeError, AttributeError) as exc:
            logger.warning("Ignoring invalid active collection pointer %s: %s", self._pointer_path, exc)
            return None

    def _write_active_collection_name(self, collection_name: str) -> None:
        """Atomically point future readers at a complete collection."""
        temporary_path = self._pointer_path.with_suffix(".tmp")
        temporary_path.write_text(
            dumps({"active_collection": collection_name}),
            encoding="utf-8",
        )
        temporary_path.replace(self._pointer_path)

    @property
    def is_available(self) -> bool:
        return self._collection is not None

    def _active_write_collection(self) -> Any:
        if self._staged_collection is not None:
            return self._staged_collection
        return self._collection

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        target = self._active_write_collection()
        if target is None:
            raise VectorStoreUnavailableError("ChromaDB is unavailable; document chunks were not indexed.")
        if not chunks:
            return 0

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        # ChromaDB metadata values must be primitive types (str, int, float, bool)
        metadatas = [
            {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in c.metadata.items()}
            for c in chunks
        ]

        try:
            target.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError("ChromaDB could not index document chunks.") from exc
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 2,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        if not self.is_available:
            raise VectorStoreUnavailableError("ChromaDB is unavailable; knowledge could not be searched.")
        if not query.strip():
            return []

        where_clause = None
        if metadata_filter:
            # Format ChromaDB where filter
            if len(metadata_filter) == 1:
                k, v = next(iter(metadata_filter.items()))
                where_clause = {k: v}
            elif len(metadata_filter) > 1:
                where_clause = {"$and": [{k: v} for k, v in metadata_filter.items()]}

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
        }
        uses_local_hash = isinstance(self._embedding_function, DefaultLocalEmbeddingFunction)
        if where_clause:
            kwargs["where"] = where_clause

        try:
            if uses_local_hash:
                collection_count = self._collection.count()
                if collection_count == 0:
                    return []
                kwargs["n_results"] = min(
                    collection_count,
                    max(top_k * 10, self.LOCAL_HASH_CANDIDATE_LIMIT),
                )
            results = self._collection.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError("ChromaDB query failed.") from exc

        retrieved: list[DocumentChunk] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        query_tokens = {self._normalize_keyword(token) for token in MemoryVectorStore._tokenize(query)}
        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, chunk_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            # Cosine distance to similarity: similarity = 1 - distance
            dist = distances[i] if i < len(distances) else 1.0
            similarity = round(max(0.0, min(1.0, 1.0 - dist)), 4)

            if uses_local_hash:
                document_tokens = {
                    self._normalize_keyword(token)
                    for token in MemoryVectorStore._tokenize(text)
                }
                coverage = len(query_tokens.intersection(document_tokens)) / max(1, len(query_tokens))
                if coverage < self.LOCAL_HASH_MIN_KEYWORD_COVERAGE:
                    continue
                similarity = round((0.75 * coverage) + (0.25 * similarity), 4)
            retrieved.append(
                DocumentChunk(
                    id=chunk_id,
                    text=text,
                    metadata=meta,
                    score=similarity,
                )
            )

        if uses_local_hash:
            retrieved.sort(key=lambda chunk: chunk.score, reverse=True)
        return retrieved[:top_k]

    @staticmethod
    def _normalize_keyword(token: str) -> str:
        """Normalize common inflections for local-hash lexical reranking."""
        if len(token) > 5 and token.endswith("ies"):
            return f"{token[:-3]}y"
        if len(token) > 5 and token.endswith("ing"):
            stem = token[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def delete_by_doc(self, doc_name: str) -> int:
        if not self.is_available:
            raise VectorStoreUnavailableError("ChromaDB is unavailable; document chunks were not deleted.")
        try:
            target = self._active_write_collection()
            existing = target.get(where={"doc_name": doc_name})
            ids = existing.get("ids") or []
            if not ids:
                return 0
            target.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError("ChromaDB could not delete document chunks.") from exc

    def clear(self) -> None:
        if self.is_available:
            replacement_name = f"{self.collection_name}__clear__{uuid4().hex[:12]}"
            replacement = self._get_or_create_collection(replacement_name)
            self._write_active_collection_name(replacement_name)
            self._collection = replacement

    def begin_staged_reindex(self) -> None:
        """Writes new embeddings into ``{collection}_staged`` without locking live reads."""
        if not self.is_available:
            return
        staged_name = f"{self.collection_name}__staged__{uuid4().hex[:12]}"
        self._staged_collection = self._get_or_create_collection(staged_name)
        self._staged_collection_name = staged_name
        live_data = self._collection.get(include=["documents", "metadatas"])
        live_ids = live_data.get("ids") or []
        live_documents = live_data.get("documents") or []
        live_metadatas = live_data.get("metadatas") or []
        if live_ids:
            self._staged_collection.upsert(
                ids=live_ids,
                documents=live_documents,
                metadatas=live_metadatas,
            )

    def commit_staged_reindex(self) -> None:
        """Promotes the staged collection into the live tenant collection."""
        if not self.is_available or self._staged_collection is None:
            return

        if self._staged_collection_name is None:
            return
        self._write_active_collection_name(self._staged_collection_name)
        self._collection = self._staged_collection
        self._staged_collection = None
        self._staged_collection_name = None

    def discard_staged_reindex(self) -> None:
        """Drops a failed staged collection and retains the previous live collection."""
        if self._staged_collection is None:
            return
        staged_name = self._staged_collection_name
        try:
            if staged_name:
                self._client.delete_collection(name=staged_name)
        except Exception:
            logger.warning("Could not remove incomplete staged Chroma collection %s", staged_name, exc_info=True)
        self._staged_collection = None
        self._staged_collection_name = None


def get_vector_store(
    persist_dir: str | None = None,
    collection_name: str | None = None,
    tenant_id: str = "grand-meridian",
    force_memory: bool = False,
    embedding_function: Any | None = None,
) -> BaseVectorStore:
    """Returns a vector store instance, prioritizing ChromaDB with automatic memory fallback."""
    if force_memory:
        return MemoryVectorStore()

    resolved_dir = persist_dir
    resolved_collection = collection_name
    resolved_embedding = embedding_function
    if resolved_dir is None or resolved_embedding is None:
        try:
            from backend.config.settings import get_settings

            settings = get_settings()
            resolved_dir = resolved_dir or settings.chroma_persist_dir
            if resolved_embedding is None:
                resolved_embedding = get_embedding_function(
                    provider=settings.embedding_provider,
                    ollama_base_url=settings.ollama_base_url,
                    embedding_model=settings.embedding_model,
                )
        except Exception as exc:
            logger.warning("Could not load RAG settings (%s); using local defaults.", exc)
            resolved_dir = resolved_dir or "./data/chroma_db"
            resolved_embedding = resolved_embedding or DefaultLocalEmbeddingFunction()

    store = ChromaVectorStore(
        persist_dir=resolved_dir,
        collection_name=resolved_collection or collection_name_for_tenant(tenant_id),
        tenant_id=tenant_id,
        embedding_function=resolved_embedding,
    )
    if store.is_available:
        return store

    logger.info("Falling back to MemoryVectorStore.")
    return MemoryVectorStore()
