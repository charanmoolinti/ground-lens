"""
Module: pdf_loader.py
Description: Extracts text and page numbers from PDF documents using PyMuPDF.
Handles empty pages, scanned/unreadable documents, and metadata extraction gracefully.
"""

from __future__ import annotations
from dataclasses import dataclass
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
    filename: str
    total_pages: int
    pages: List[PDFPage]
    is_empty: bool = False
    warning: Optional[str] = None


class PDFLoader:
    """PyMuPDF-based text and metadata extractor for PDF documents."""

    def __init__(self, min_char_per_page: int = 10):
        self.min_char_per_page = min_char_per_page

    def load_pdf(self, file_path_or_bytes: Union[str, Path, bytes], filename: Optional[str] = None) -> PDFDocument:
        """
        Loads and extracts text page-by-page from a PDF file path or raw bytes.

        Args:
            file_path_or_bytes: Path to PDF or bytes buffer.
            filename: Name of the file (inferred if path provided).

        Returns:
            PDFDocument with extracted page texts and metadata.
        """
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found at: {path}")
            doc_name = filename or path.name
            doc = pymupdf.open(str(path))
        elif isinstance(file_path_or_bytes, bytes):
            if not filename:
                doc_name = "uploaded_document.pdf"
            else:
                doc_name = filename
            doc = pymupdf.open(stream=file_path_or_bytes, filetype="pdf")
        else:
            raise TypeError("Expected file path or bytes for PDF loading.")

        pages: List[PDFPage] = []
        total_extracted_chars = 0

        try:
            for page_idx in range(len(doc)):
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

        warning = None
        is_empty = total_extracted_chars < self.min_char_per_page
        if is_empty:
            warning = (
                f"Document '{doc_name}' contains very little or no extractable text ({total_extracted_chars} characters). "
                "It may be a scanned image-only PDF or empty."
            )

        return PDFDocument(
            filename=doc_name,
            total_pages=len(pages),
            pages=pages,
            is_empty=is_empty,
            warning=warning,
        )
