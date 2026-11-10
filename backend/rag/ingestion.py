"""Document Ingestion and Knowledge Base Pipeline.

Loads documents (Markdown, TXT, PDF), extracts text, chunks using VoiceChunker,
and indexes into the target vector store with tenant and persona isolation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.rag.chunker import VoiceChunker
from backend.rag.vector_store import BaseVectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Seed Knowledge Base for Grand Meridian Hotel
DEFAULT_HOTEL_KNOWLEDGE = """
# Grand Meridian Hotel & Resort Operational Knowledge

## Check-In and Check-Out Policy
Standard check-in time begins at 3:00 PM. Check-out time is strictly 11:00 AM.
Early check-in is subject to room availability upon arrival. Late check-out until 1:00 PM is complimentary for VIP and rewards guests, or available for a fee of $45 for standard bookings.

## Parking and Transportation
Valet parking is available at the main entrance for $35 per night with unlimited in-and-out privileges.
Self-parking is available in the north covered garage for $20 per night.
Complimentary electric vehicle charging stations (Tesla and standard J1772) are located on level 1 of the parking garage.

## Pet Policy
Grand Meridian is a pet-friendly resort. We welcome dogs and cats up to 50 pounds with a one-time non-refundable cleaning fee of $75 per stay.
Pets must be kept on a leash in all common areas, and are not permitted inside the pool enclosure or dining rooms.

## Pool and Spa Hours
The rooftop heated infinity pool is open daily from 6:00 AM to 10:00 PM.
The Meridian Wellness Spa and sauna are open daily from 8:00 AM to 8:00 PM. Reservations for massages and facials are recommended at least 24 hours in advance.

## Complimentary Breakfast and Dining
A gourmet hot buffet breakfast is served daily at The Grand Terrace from 6:30 AM to 10:30 AM on weekdays, and until 11:00 AM on weekends.
Breakfast is included for Deluxe and Suite bookings, or available for $28 per adult for standard room guests.
Room service dining is available 24 hours a day with orders placed via the front desk or room phone.

## High-Speed Internet Access
High-speed Wi-Fi is complimentary for all registered hotel guests throughout the property.
Connect to the network named "GrandMeridian_Guest" and enter your room number and last name to authenticate. Premium ultra-fast 500 Mbps connection is available for $10 per day.
"""


class DocumentIngestionPipeline:
    """Manages document ingestion, chunking, and indexing into the vector store."""

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        chunker: VoiceChunker | None = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.chunker = chunker or VoiceChunker()

    def ingest_text(
        self,
        text: str,
        doc_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunks and indexes raw text into the vector store."""
        if not text or not text.strip():
            return 0

        meta = metadata.copy() if metadata else {}
        meta.setdefault("doc_name", doc_name)

        chunks = self.chunker.chunk_text(text, metadata=meta, doc_name=doc_name)
        if not chunks:
            return 0

        added = self.vector_store.add_chunks(chunks)
        logger.info("Ingested '%s': created %d chunks, indexed %d", doc_name, len(chunks), added)
        return added

    def ingest_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> int:
        """Reads a file from disk, chunks it, and indexes into the vector store."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        meta = metadata.copy() if metadata else {}
        doc_name = document_id or path.stem
        meta.setdefault("doc_name", doc_name)
        meta.setdefault("file_extension", path.suffix)

        if path.suffix.lower() in (".md", ".txt"):
            content = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            content = self._extract_pdf_text(path)
        else:
            # Default to text read
            content = path.read_text(encoding="utf-8", errors="ignore")

        return self.ingest_text(content, doc_name=doc_name, metadata=meta)

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        """Extracts text from a PDF file using pypdf or PyMuPDF if installed."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                logger.warning("Neither PyMuPDF nor pypdf installed; cannot extract binary PDF %s", path)
                return ""

    def reindex_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> int:
        """Deletes previous chunks for a document and re-indexes with fresh content."""
        path = Path(file_path)
        doc_name = document_id or path.stem
        self.vector_store.delete_by_doc(doc_name)
        return self.ingest_file(path, metadata=metadata, document_id=doc_name)

    def reindex_atomically(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> int:
        """Re-indexes into a staged collection, then swaps it to live.

        Voice calls continue reading the previous collection until commit.
        """
        try:
            self.vector_store.begin_staged_reindex()
            doc_name = document_id or Path(file_path).stem
            self.vector_store.delete_by_doc(doc_name)
            added = self.ingest_file(file_path, metadata=metadata, document_id=doc_name)
            if added < 1:
                raise ValueError(f"No readable text was found in document: {file_path}")
            self.vector_store.commit_staged_reindex()
            return added
        except Exception:
            self.vector_store.discard_staged_reindex()
            raise

    def seed_default_hotel_knowledge(
        self,
        tenant_id: str = "grand-meridian",
        persona_id: str = "hotel_booking_specialist",
    ) -> int:
        """Seeds the default Grand Meridian hotel knowledge into the vector store."""
        return self.ingest_text(
            text=DEFAULT_HOTEL_KNOWLEDGE,
            doc_name="hotel_policies_and_amenities",
            metadata={
                "tenant_id": tenant_id,
                "persona_id": persona_id,
                "category": "hotel_policy",
            },
        )
