"""Unit tests for the Knowledge Base and Dynamic RAG Subsystem.

Tests:
1. VoiceChunker: Markdown stripping, table conversion, sentence-aware chunking, token overlap.
2. MemoryVectorStore & ChromaVectorStore: Indexing, TF-IDF / cosine retrieval, metadata isolation.
3. RAGContextRetriever: Similarity thresholding, out-of-scope fallback, prompt formatting.
4. DocumentIngestionPipeline: End-to-end ingestion and Grand Meridian hotel policy verification.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.chunker import DocumentChunk, VoiceChunker
from backend.rag.embeddings import DefaultLocalEmbeddingFunction, get_embedding_function
from backend.rag.ingestion import DocumentIngestionPipeline
from backend.rag.retriever import (
    FALLBACK_SPOKEN_RESPONSE,
    RAGContextFrameProcessor,
    RAGContextRetriever,
    clamp_top_k,
    limit_spoken_sentences,
    select_spoken_sentences,
)
from backend.rag.vector_store import (
    ChromaVectorStore,
    MemoryVectorStore,
    VectorStoreError,
    collection_name_for_tenant,
    get_vector_store,
)


class TestVoiceChunker(unittest.TestCase):
    """Tests for voice-tailored chunking and markdown stripping."""

    def setUp(self):
        self.chunker = VoiceChunker(chunk_size_tokens=50, overlap_tokens=10)

    def test_clean_markdown_syntax(self):
        """Validates removal of bold, italics, links, and headings."""
        raw = (
            "# Hotel Amenities\n"
            "We offer **free high-speed Wi-Fi** and *complimentary breakfast*.\n"
            "For bookings, visit [Grand Meridian](https://example.com/book)."
        )
        cleaned = VoiceChunker.clean_markdown_for_speech(raw)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("free high-speed Wi-Fi", cleaned)
        self.assertIn("complimentary breakfast", cleaned)
        self.assertIn("Grand Meridian", cleaned)

    def test_clean_markdown_tables(self):
        """Markdown tables must convert into natural spoken declarative lines."""
        raw_table = (
            "| Room Type | Nightly Rate |\n"
            "|---|---|\n"
            "| Standard King | $120 |\n"
            "| Deluxe Queen | $180 |\n"
        )
        cleaned = VoiceChunker.clean_markdown_for_speech(raw_table)
        self.assertNotIn("|", cleaned)
        self.assertIn("Room Type is Standard King", cleaned)
        self.assertIn("Nightly Rate is $120", cleaned)

    def test_sentence_boundary_integrity(self):
        """Verifies chunks never split a sentence in half."""
        text = (
            "Check-in begins at 3:00 PM. Check-out is strictly 11:00 AM. "
            "Late check-out is available upon request for a small fee. "
            "Valet parking is located at the main entrance. "
            "The rooftop pool is open until 10:00 PM every evening."
        )
        chunks = self.chunker.chunk_text(text, doc_name="test_policy")
        self.assertTrue(len(chunks) >= 2)
        for chunk in chunks:
            # Chunk must end with sentence-ending punctuation
            self.assertTrue(chunk.text.endswith((".", "!", "?")))

    def test_empty_text_handling(self):
        """Empty or whitespace text returns no chunks."""
        self.assertEqual(self.chunker.chunk_text(""), [])
        self.assertEqual(self.chunker.chunk_text("   \n\t  "), [])

    def test_numbered_list_and_overlap(self):
        """Numbered lists become speech, and overlap retains prior sentences."""
        raw = (
            "1. Valet parking is thirty five dollars per night.\n"
            "2. Self parking is twenty dollars per night.\n"
            "The rooftop pool is open until ten in the evening. "
            "Breakfast is served every morning in the terrace restaurant. "
            "Spa reservations should be made twenty four hours in advance."
        )
        cleaned = VoiceChunker.clean_markdown_for_speech(raw)
        self.assertNotIn("1.", cleaned)
        chunker = VoiceChunker(chunk_size_tokens=20, overlap_tokens=12)
        chunks = chunker.chunk_text(raw, doc_name="overlap_doc")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any(chunks[0].text.split()[-1] in later.text for later in chunks[1:]))


class TestMemoryVectorStore(unittest.TestCase):
    """Tests for in-memory TF-IDF vector store."""

    def setUp(self):
        self.store = MemoryVectorStore()
        self.chunks = [
            DocumentChunk(
                id="c1",
                text="The swimming pool is located on the rooftop and open from 6 AM to 10 PM.",
                metadata={"tenant_id": "grand-meridian", "category": "amenity", "doc_name": "pool_info"},
            ),
            DocumentChunk(
                id="c2",
                text="Valet parking costs $35 per night at the main entrance with in and out privileges.",
                metadata={"tenant_id": "grand-meridian", "category": "parking", "doc_name": "parking_info"},
            ),
            DocumentChunk(
                id="c3",
                text="Other tenant policy for pets.",
                metadata={"tenant_id": "other-hotel", "category": "pets", "doc_name": "other_info"},
            ),
        ]
        self.store.add_chunks(self.chunks)

    def test_search_retrieves_relevant_chunk(self):
        """Query for pool returns rooftop pool chunk with high score."""
        results = self.store.search("rooftop swimming pool hours", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "c1")
        self.assertTrue(results[0].score > 0.0)

    def test_metadata_isolation(self):
        """Filtering by tenant_id isolates chunks across tenants."""
        results = self.store.search(
            "pets",
            top_k=2,
            metadata_filter={"tenant_id": "grand-meridian"},
        )
        for r in results:
            self.assertEqual(r.metadata["tenant_id"], "grand-meridian")

    def test_delete_by_doc(self):
        """Deleting by doc_name removes all its chunks."""
        deleted = self.store.delete_by_doc("pool_info")
        self.assertEqual(deleted, 1)
        results = self.store.search("swimming pool", top_k=1)
        self.assertEqual(len(results), 0)


class TestChromaVectorStore(unittest.TestCase):
    """Tests for ChromaDB embedded vector store."""

    def test_chroma_store_operations(self):
        """Validates ChromaDB lifecycle: creation, indexing, querying, clearing."""
        import shutil
        test_dir = "./data/test_chroma_db"
        shutil.rmtree(test_dir, ignore_errors=True)
        store = ChromaVectorStore(
            persist_dir=test_dir,
            collection_name="test_collection",
            embedding_function=DefaultLocalEmbeddingFunction(),
        )
        if not store.is_available:
            self.skipTest("ChromaDB not available in current environment")

        try:
            store.clear()
            chunks = [
                DocumentChunk(
                    id="chr_1",
                    text="Free high-speed Wi-Fi is available across all guest rooms.",
                    metadata={"tenant_id": "grand-meridian", "category": "internet"},
                ),
                DocumentChunk(
                    id="chr_2",
                    text="Gourmet breakfast is served from 6:30 AM to 10:30 AM daily.",
                    metadata={"tenant_id": "grand-meridian", "category": "dining"},
                ),
            ]
            added = store.add_chunks(chunks)
            self.assertEqual(added, 2)

            results = store.search("breakfast hours", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].id, "chr_2")
            self.assertTrue(results[0].score > 0.0)
        finally:
            store.clear()

    def test_local_hash_retrieval_handles_paraphrases_and_rejects_near_misses(self):
        """Real Chroma local-hash retrieval must rank relevant facts over hash collisions."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            store = ChromaVectorStore(
                persist_dir=str(Path(temp_dir) / "chroma_local_hash_eval"),
                collection_name="local_hash_eval",
                embedding_function=DefaultLocalEmbeddingFunction(),
            )
            store.add_chunks(
                [
                    DocumentChunk(
                        id="spa",
                        text="The Meridian Wellness Spa opens at 8:00 AM daily.",
                        metadata={"tenant_id": "grand-meridian", "doc_name": "spa-hours.md"},
                    ),
                    DocumentChunk(
                        id="parking",
                        text="Valet parking costs 35 dollars per night.",
                        metadata={"tenant_id": "grand-meridian", "doc_name": "parking.md"},
                    ),
                    DocumentChunk(
                        id="pool",
                        text="The rooftop pool closes at 10:00 PM each day.",
                        metadata={"tenant_id": "grand-meridian", "doc_name": "pool.md"},
                    ),
                    DocumentChunk(
                        id="breakfast",
                        text="Breakfast is included with Deluxe and Suite bookings.",
                        metadata={"tenant_id": "grand-meridian", "doc_name": "breakfast.md"},
                    ),
                ]
            )
            retriever = RAGContextRetriever(vector_store=store)
            answerable_queries = {
                "What time does the spa open?": "spa-hours.md",
                "When can I visit the wellness spa?": "spa-hours.md",
                "How much does overnight valet parking cost?": "parking.md",
                "What time does the rooftop pool close?": "pool.md",
                "Which room bookings include breakfast?": "breakfast.md",
            }

            for query, expected_document in answerable_queries.items():
                with self.subTest(query=query):
                    result = retriever.query_with_context(
                        query,
                        metadata_filter={"tenant_id": "grand-meridian"},
                    )
                    self.assertEqual(result["status"], "matched")
                    self.assertEqual(result["chunks"][0].metadata["doc_name"], expected_document)

            unsupported_queries = [
                "Does the spa offer massages for free?",
                "Does the hotel have a gym that stays open all night?",
            ]
            for query in unsupported_queries:
                with self.subTest(query=query):
                    result = retriever.query_with_context(
                        query,
                        metadata_filter={"tenant_id": "grand-meridian"},
                    )
                    self.assertEqual(result["status"], "no_match")
            store._client = None
            store._collection = None


class TestRAGContextRetriever(unittest.TestCase):
    """Tests for retriever guardrails and spoken formatting."""

    def setUp(self):
        self.store = MemoryVectorStore()
        self.store.add_chunks([
            DocumentChunk(
                id="p1",
                text="Check-in time is 3:00 PM and check-out time is strictly 11:00 AM.",
                metadata={"doc_name": "policies", "tenant_id": "grand-meridian"},
            ),
            DocumentChunk(
                id="p2",
                text="Pets under 50 pounds are welcome with a one-time cleaning fee of $75.",
                metadata={"doc_name": "policies", "tenant_id": "grand-meridian"},
            ),
        ])
        self.retriever = RAGContextRetriever(
            vector_store=self.store,
            similarity_threshold=0.30,  # TF-IDF cosine threshold
        )

    def test_in_scope_retrieval(self):
        """In-scope query retrieves policy context."""
        ctx = self.retriever.query_with_context("What time is check-in and check-out?")
        self.assertEqual(ctx["status"], "matched")
        self.assertTrue(ctx["has_context"])
        self.assertIn("3:00 PM", ctx["context_prompt"])
        self.assertIn("11:00 AM", ctx["context_prompt"])
        self.assertIn("Never use bullet points", ctx["context_prompt"])
        self.assertIsNone(ctx["fallback_spoken"])

    def test_default_threshold_retrieves_uploaded_fact_with_local_hash_scores(self):
        """Default settings admit a paraphrased fact scored by the local-first store."""
        store = MemoryVectorStore()
        store.add_chunks(
            [
                DocumentChunk(
                    id="spa-hours",
                    text="The Meridian Wellness Spa opens at 8:00 AM daily.",
                    metadata={"tenant_id": "grand-meridian", "doc_name": "spa-hours.md"},
                )
            ]
        )
        retriever = RAGContextRetriever(vector_store=store)

        result = retriever.query_with_context(
            "What time does the spa open?",
            metadata_filter={"tenant_id": "grand-meridian"},
        )
        self.assertTrue(result["has_context"])
        self.assertIn("8:00 AM", result["context_prompt"])

        unrelated = retriever.query_with_context(
            "Can I park my submarine on Mars?",
            metadata_filter={"tenant_id": "grand-meridian"},
        )
        self.assertFalse(unrelated["has_context"])

    def test_out_of_scope_fallback(self):
        """Unrelated out-of-scope query delivers graceful spoken fallback."""
        ctx = self.retriever.query_with_context("Can I land my spaceship or submarine in your backyard?")
        self.assertEqual(ctx["status"], "no_match")
        self.assertFalse(ctx["has_context"])
        self.assertEqual(ctx["context_prompt"], "")
        self.assertEqual(ctx["fallback_spoken"], FALLBACK_SPOKEN_RESPONSE)

    def test_empty_query(self):
        """Empty query returns no context and fallback."""
        ctx = self.retriever.query_with_context("")
        self.assertEqual(ctx["status"], "no_match")
        self.assertFalse(ctx["has_context"])
        self.assertEqual(ctx["fallback_spoken"], FALLBACK_SPOKEN_RESPONSE)

    def test_top_k_clamp_and_spoken_brevity(self):
        """Voice retrieval never exceeds two chunks or two spoken sentences."""
        self.assertEqual(clamp_top_k(8), 2)
        retriever = RAGContextRetriever(vector_store=self.store, top_k=9, similarity_threshold=0.10)
        self.assertEqual(retriever.top_k, 2)
        long_text = (
            "Check-in begins at 3:00 PM. Check-out is 11:00 AM. "
            "Late checkout costs forty five dollars. Pets are welcome downstairs."
        )
        trimmed = limit_spoken_sentences(long_text, max_sentences=2)
        self.assertEqual(trimmed.count("."), 2)
        self.assertNotIn("Pets are welcome", trimmed)
        selected = select_spoken_sentences(long_text, "pet fee for dogs", max_sentences=2)
        self.assertIn("Pets are welcome downstairs", selected)
        self.assertNotIn("3:00 PM", selected)

        self.store.add_chunks([
            DocumentChunk(
                id="p3",
                text=long_text,
                metadata={"doc_name": "policies", "tenant_id": "grand-meridian"},
            ),
        ])
        ctx = retriever.query_with_context("What time is check-in and check-out?")
        self.assertIn("<knowledge_base>", ctx["context_prompt"])
        self.assertLessEqual(len(ctx["chunks"]), 2)
        self.assertNotIn("Pets are welcome downstairs", ctx["context_prompt"])

    def test_apply_to_messages_injects_system_context(self):
        """Knowledge is merged into the system prompt for the LLM turn."""
        messages = [{"role": "system", "content": "You are Elena."}, {"role": "user", "content": "check-in time"}]
        updated = self.retriever.apply_to_messages(messages, "What time is check-in?")
        self.assertEqual(updated[0]["content"], "You are Elena.")
        self.assertTrue(any("3:00 PM" in message["content"] for message in updated))


class TestDocumentIngestionPipeline(unittest.TestCase):
    """Tests for the end-to-end ingestion pipeline and hotel seed knowledge."""

    def setUp(self):
        self.store = MemoryVectorStore()
        self.pipeline = DocumentIngestionPipeline(vector_store=self.store)

    def test_seed_hotel_knowledge(self):
        """Seeds default Grand Meridian knowledge and verifies operational queries."""
        count = self.pipeline.seed_default_hotel_knowledge()
        self.assertTrue(count >= 3)

        retriever = RAGContextRetriever(vector_store=self.store, similarity_threshold=0.20)

        # 1. Parking query
        valet_ctx = retriever.query_with_context("How much is valet parking?")
        self.assertTrue(valet_ctx["has_context"])
        self.assertIn("$35", valet_ctx["context_prompt"])

        # 2. Pet query
        pet_ctx = retriever.query_with_context("Are dogs allowed and what is the pet fee?")
        self.assertTrue(pet_ctx["has_context"])
        self.assertIn("50 pounds", pet_ctx["context_prompt"])

        # 3. Wi-Fi query
        wifi_ctx = retriever.query_with_context("What is the Wi-Fi network name?")
        self.assertTrue(wifi_ctx["has_context"])
        self.assertIn("GrandMeridian_Guest", wifi_ctx["context_prompt"])

    def test_ingest_markdown_file(self):
        """Markdown files on disk are chunked and indexed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spa_hours.md"
            path.write_text(
                "# Spa\nThe Meridian Wellness Spa is open daily from 8:00 AM to 8:00 PM.\n",
                encoding="utf-8",
            )
            added = self.pipeline.ingest_file(path, metadata={"tenant_id": "grand-meridian"})
            self.assertGreaterEqual(added, 1)
            results = self.store.search("spa hours", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertIn("8:00 AM", results[0].text)


class TestEmbeddingsAndCollections(unittest.TestCase):
    """Tests for local embeddings and tenant collection naming."""

    def test_local_hash_is_stable(self):
        """BLAKE2 embeddings must match across instances (not Python's salted hash)."""
        first = DefaultLocalEmbeddingFunction()(["valet parking thirty five dollars"])
        second = DefaultLocalEmbeddingFunction()(["valet parking thirty five dollars"])
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 64)

    def test_embedding_factory_defaults_to_local_hash(self):
        """Default factory must not download models or call the network."""
        function = get_embedding_function(provider="local_hash")
        self.assertIsInstance(function, DefaultLocalEmbeddingFunction)

    def test_tenant_collection_name(self):
        """Tenant IDs map to Chroma-safe kb_* collection names."""
        self.assertEqual(collection_name_for_tenant("grand-meridian"), "kb_grand-meridian")
        self.assertEqual(collection_name_for_tenant("Grand Meridian!"), "kb_grand_meridian_")


class TestRAGContextFrameProcessor(unittest.IsolatedAsyncioTestCase):
    """Tests for the Pipecat-compatible RAG frame processor."""

    async def test_transcription_frame_retrieves_context(self):
        """A user transcription frame triggers retrieval and stores the last result."""
        store = MemoryVectorStore()
        store.add_chunks([
            DocumentChunk(
                id="f1",
                text="Check-in time is 3:00 PM and check-out time is strictly 11:00 AM.",
                metadata={"doc_name": "policies"},
            ),
        ])
        retriever = RAGContextRetriever(vector_store=store, similarity_threshold=0.20)
        processor = RAGContextFrameProcessor(retriever=retriever)

        messages = [
            {"role": "system", "content": "You are Elena."},
            {"role": "user", "content": "What time is check-in?"},
        ]
        result = await processor.prepare_turn(messages, "What time is check-in?")
        self.assertEqual(processor.last_query, "What time is check-in?")
        self.assertIsNotNone(processor.last_result)
        self.assertTrue(processor.last_result["has_context"])
        self.assertIn("3:00 PM", processor.last_result["context_prompt"])
        self.assertTrue(any("3:00 PM" in message["content"] for message in messages))
        self.assertEqual(result["status"], "matched")

    async def test_each_turn_replaces_prior_context_and_distinguishes_failure(self):
        """A no-match or retrieval failure removes old evidence and has an explicit status."""
        store = MemoryVectorStore()
        store.add_chunks(
            [
                DocumentChunk(
                    id="spa",
                    text="The spa opens at 8:00 AM.",
                    metadata={"tenant_id": "grand-meridian", "doc_name": "spa.md"},
                )
            ]
        )
        processor = RAGContextFrameProcessor(
            retriever=RAGContextRetriever(vector_store=store, similarity_threshold=0.1),
            metadata_filter={"tenant_id": "grand-meridian"},
        )
        messages = [{"role": "user", "content": "When does the spa open?"}]
        await processor.prepare_turn(messages, "When does the spa open?")
        self.assertIn("8:00 AM", messages[-1]["content"])

        no_match = await processor.prepare_turn(messages, "Tell me about Mars.")
        self.assertEqual(no_match["status"], "no_match")
        self.assertNotIn("8:00 AM", messages[-1]["content"])

        processor.retriever.vector_store.search = Mock(side_effect=VectorStoreError("query failed"))
        unavailable = await processor.prepare_turn(messages, "What are the spa hours?")
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIn("search is unavailable", messages[-1]["content"])
        self.assertNotIn("8:00 AM", messages[-1]["content"])


class TestForceMemoryFactory(unittest.TestCase):
    """Tests for the vector store factory fallback."""

    def test_force_memory_store(self):
        store = get_vector_store(force_memory=True)
        self.assertIsInstance(store, MemoryVectorStore)


if __name__ == "__main__":
    unittest.main()
