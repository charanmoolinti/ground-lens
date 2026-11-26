"""
Module: pdf_loader.py
Description: Extracts text and page numbers from PDF documents using PyMuPDF.
Handles file size validation, empty documents, scanned/image-only PDFs, unique document ID hashing,
and metadata extraction gracefully.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import List, Optional, Union
import pymupdf


@dataclass
class PDFPage:
    """Represents text extracted from a single page of a PDF document."""
    page_number: int  # 1-indexed
    text: str
    char_count: int
    word_count: int


@dataclass
class PDFDocument:
    """Represents a loaded PDF document containing its pages and metadata."""
    doc_id: str
    filename: str
    file_size_bytes: int
    total_pages: int
    pages: List[PDFPage]
    is_empty: bool = False
    is_scanned: bool = False
    warning: Optional[str] = None


class PDFLoader:
    """PyMuPDF-based text and metadata extractor for PDF documents."""

    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit

    def __init__(self, min_char_per_page: int = 15, max_file_size_bytes: int = MAX_FILE_SIZE_BYTES):
        self.min_char_per_page = min_char_per_page
        self.max_file_size_bytes = max_file_size_bytes

    def load_pdf(self, file_path_or_bytes: Union[str, Path, bytes], filename: Optional[str] = None) -> PDFDocument:
        """
        Loads and extracts text page-by-page from a PDF file path or raw bytes.

        Args:
            file_path_or_bytes: Path to PDF or bytes buffer.
            filename: Name of the file (inferred if path provided).

        Returns:
            PDFDocument with extracted page texts and provenance metadata.
        """
        # Determine bytes, size, and document ID
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found at: {path}")
            raw_bytes = path.read_bytes()
            doc_name = filename or path.name
        elif isinstance(file_path_or_bytes, bytes):
            raw_bytes = file_path_or_bytes
            doc_name = filename or "uploaded_document.pdf"
        else:
            raise TypeError("Expected file path or bytes for PDF loading.")

        file_size = len(raw_bytes)
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds the maximum allowed limit of "
                f"{self.max_file_size_bytes / (1024 * 1024):.0f}MB."
            )

        if file_size == 0:
            raise ValueError("The provided PDF file is 0 bytes (completely empty).")

        # Deterministic document hash for duplicate identification
        doc_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

        try:
            doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted PDF file: {e}") from e

        pages: List[PDFPage] = []
        total_extracted_chars = 0

        try:
            total_pages = len(doc)
            if total_pages == 0:
                return PDFDocument(
                    doc_id=doc_id,
                    filename=doc_name,
                    file_size_bytes=file_size,
                    total_pages=0,
                    pages=[],
                    is_empty=True,
                    is_scanned=False,
                    warning=f"Document '{doc_name}' contains zero pages.",
                )

            for page_idx in range(total_pages):
                page = doc[page_idx]
                text = page.get_text("text").strip()
                char_count = len(text)
                words = text.split()
                word_count = len(words)
                total_extracted_chars += char_count

                pages.append(
                    PDFPage(
                        page_number=page_idx + 1,
                        text=text,
                        char_count=char_count,
                        word_count=word_count,
                    )
                )
        finally:
            doc.close()

        # Check for scanned or empty PDFs
        avg_char_per_page = total_extracted_chars / max(1, len(pages))
        is_scanned = (total_extracted_chars < 50 and len(pages) > 0) or (avg_char_per_page < self.min_char_per_page)
        is_empty = total_extracted_chars == 0

        warning = None
        if is_empty or is_scanned:
            warning = (
                f"Document '{doc_name}' contains very little or no extractable text "
                f"({total_extracted_chars} total characters across {len(pages)} page(s)). "
                "This document appears to be a scanned image-only PDF. "
                "OCR is not currently supported; please provide a PDF with selectable text."
            )

        return PDFDocument(
            doc_id=doc_id,
            filename=doc_name,
            file_size_bytes=file_size,
            total_pages=len(pages),
            pages=pages,
            is_empty=is_empty,
            is_scanned=is_scanned,
            warning=warning,
        )
