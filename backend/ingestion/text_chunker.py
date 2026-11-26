"""
Module: text_chunker.py
Description: Custom sliding-window text chunker with token/word overlap that preserves
source document metadata, exact page numbers, doc_id, and unique chunk IDs for evidence provenance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List
from backend.ingestion.pdf_loader import PDFDocument, PDFPage


@dataclass
class DocumentChunk:
    """Represents an indexed chunk of text with complete provenance metadata."""
    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    chunk_index: int
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """
    Sliding-window text chunker designed for RAG and claim-level verification.
    Uses word/token estimation (approx 1.3 words per token or direct whitespace tokenization)
    with configurable target size and overlap.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        """
        Args:
            chunk_size: Number of words per chunk (approx. 500 tokens).
            chunk_overlap: Number of overlapping words between consecutive chunks (approx. 80 tokens).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalizes excessive whitespace and fixes hyphenated line breaks."""
        # Join hyphenated words split across lines (e.g. "connec-\ntion" -> "connection")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Normalize newlines and whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def chunk_page(self, page: PDFPage, doc_name: str, doc_id: str = "", start_chunk_idx: int = 0) -> List[DocumentChunk]:
        """
        Splits a single PDFPage into overlapping chunks, maintaining strict page provenance.
        Never merges text across different pages to prevent provenance corruption.
        """
        cleaned_text = self.clean_text(page.text)
        if not cleaned_text:
            return []

        words = cleaned_text.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []
        step = self.chunk_size - self.chunk_overlap
        current_idx = start_chunk_idx

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            # Avoid tiny trailing fragments that have negligible semantic context
            if len(chunk_words) < 15 and chunks:
                prev_chunk = chunks[-1]
                prev_chunk.text = f"{prev_chunk.text} {chunk_text}"
                break

            chunk_id = f"{doc_name}_p{page.page_number}_c{current_idx}"
            metadata = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page_number": page.page_number,
                "chunk_index": current_idx,
                "word_count": len(chunk_words),
            }

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page_number=page.page_number,
                    text=chunk_text,
                    chunk_index=current_idx,
                    metadata=metadata,
                )
            )
            current_idx += 1

            if i + self.chunk_size >= len(words):
                break

        return chunks

    def chunk_document(self, doc: PDFDocument) -> List[DocumentChunk]:
        """
        Splits an entire PDFDocument page by page, ensuring chunks strictly respect page boundaries
        so that retrieved evidence always points to the exact source page.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in doc.pages:
            page_chunks = self.chunk_page(
                page=page,
                doc_name=doc.filename,
                doc_id=doc.doc_id,
                start_chunk_idx=global_chunk_idx,
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks
