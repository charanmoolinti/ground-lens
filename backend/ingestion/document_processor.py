"""
Module: document_processor.py
Description: High-level pipeline coordinator for document ingestion:
PDF validation, text extraction, sliding chunking, vector database deduplication and indexing.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from backend.ingestion.pdf_loader import PDFDocument, PDFLoader
from backend.ingestion.text_chunker import DocumentChunk, TextChunker
from backend.retrieval.vector_store import ChromaVectorStore


@dataclass
class IngestionResult:
    """Detailed summary of the document ingestion and indexing process."""
    doc_name: str
    total_pages: int
    total_chunks: int
    indexed_chunks: int
    already_indexed: bool
    warning: Optional[str] = None
    chunks: List[DocumentChunk] = None


class DocumentProcessor:
    """Orchestrates document loading, chunking, and vector store indexing."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ):
        self.loader = PDFLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = vector_store or ChromaVectorStore()

    def process_pdf(
        self,
        file_path_or_bytes: Union[str, Path, bytes],
        filename: Optional[str] = None,
        force_reindex: bool = False,
    ) -> IngestionResult:
        """
        Executes the full ingestion pipeline:
        1. Loads and validates the PDF.
        2. Checks if document is already indexed to prevent redundant work.
        3. Chunks the text preserving page boundaries.
        4. Indexes chunks and embeddings into ChromaDB.

        Returns:
            IngestionResult with processing statistics.
        """
        # Load PDF
        pdf_doc: PDFDocument = self.loader.load_pdf(file_path_or_bytes, filename=filename)
        doc_name = pdf_doc.filename

        # Check existing index status
        is_indexed = self.vector_store.is_document_indexed(doc_name)
        if is_indexed and not force_reindex:
            return IngestionResult(
                doc_name=doc_name,
                total_pages=pdf_doc.total_pages,
                total_chunks=0,
                indexed_chunks=0,
                already_indexed=True,
                warning=f"Document '{doc_name}' is already indexed in the vector store.",
                chunks=[],
            )

        # If force re-indexing, remove old chunks first
        if is_indexed and force_reindex:
            self.vector_store.delete_document(doc_name)

        if pdf_doc.is_empty:
            return IngestionResult(
                doc_name=doc_name,
                total_pages=pdf_doc.total_pages,
                total_chunks=0,
                indexed_chunks=0,
                already_indexed=False,
                warning=pdf_doc.warning,
                chunks=[],
            )

        # Chunk the document
        chunks = self.chunker.chunk_document(pdf_doc)

        # Index into ChromaDB
        indexed_count = self.vector_store.add_chunks(chunks)

        return IngestionResult(
            doc_name=doc_name,
            total_pages=pdf_doc.total_pages,
            total_chunks=len(chunks),
            indexed_chunks=indexed_count,
            already_indexed=False,
            warning=pdf_doc.warning,
            chunks=chunks,
        )
