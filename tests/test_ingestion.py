"""
Unit tests for Module A: Document Ingestion (pdf_loader, text_chunker, document_processor).
"""

import tempfile
from pathlib import Path
import pymupdf
import pytest
from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.text_chunker import TextChunker
from backend.ingestion.document_processor import DocumentProcessor
from backend.retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def sample_pdf_path():
    """Generates a multi-page technical PDF for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    doc = pymupdf.open()

    # Page 1
    p1 = doc.new_page()
    p1.insert_text(
        pymupdf.Point(50, 72),
        "GROUND LENS: AI Hallucination Detection Platform.\n\n"
        "Abstract:\n"
        "Large language models can generate plausible yet factually incorrect statements.\n"
        "This research paper presents Ground Lens, a framework designed to verify LLM claims against source documents.\n"
        "The system achieves 94.5% verification accuracy across benchmark evaluation sets.\n"
        "Ground Lens reduces verification latency by 45% compared to baseline pipelines.",
        fontsize=12,
    )

    # Page 2
    p2 = doc.new_page()
    p2.insert_text(
        pymupdf.Point(50, 72),
        "Architecture & Methodology:\n\n"
        "The architecture consists of document ingestion, vector retrieval with ChromaDB, "
        "and a DeBERTa-v3 cross-encoder natural language inference classifier.\n"
        "Claims are classified into Supported, Contradicted, and Unverified categories.\n"
        "The embedding model utilized is sentence-transformers/all-MiniLM-L6-v2.",
        fontsize=12,
    )

    doc.save(str(tmp_path))
    doc.close()

    yield tmp_path

    if tmp_path.exists():
        tmp_path.unlink()


def test_pdf_loader(sample_pdf_path):
    loader = PDFLoader()
    doc = loader.load_pdf(sample_pdf_path)

    assert doc.total_pages == 2
    assert not doc.is_empty
    assert not doc.is_scanned
    assert doc.doc_id != ""
    assert "GROUND LENS" in doc.pages[0].text
    assert "DeBERTa-v3" in doc.pages[1].text
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 2


def test_scanned_pdf_detection():
    """Generates an empty page PDF (simulating scanned image-only PDF without OCR)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    doc = pymupdf.open()
    doc.new_page()  # Page without text
    doc.save(str(tmp_path))
    doc.close()

    try:
        loader = PDFLoader()
        pdf_doc = loader.load_pdf(tmp_path)
        assert pdf_doc.is_empty is True or pdf_doc.is_scanned is True
        assert "scanned image-only PDF" in (pdf_doc.warning or "")

        # Verify DocumentProcessor rejects scanned PDF without indexing empty chunks
        processor = DocumentProcessor()
        res = processor.process_pdf(tmp_path)
        assert res.indexed_chunks == 0
        assert res.is_scanned is True or res.total_chunks == 0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_file_size_validation(sample_pdf_path):
    loader = PDFLoader(max_file_size_bytes=10)  # artificially low limit: 10 bytes
    with pytest.raises(ValueError, match="exceeds the maximum allowed limit"):
        loader.load_pdf(sample_pdf_path)


def test_text_chunker(sample_pdf_path):
    loader = PDFLoader()
    doc = loader.load_pdf(sample_pdf_path)

    chunker = TextChunker(chunk_size=30, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.doc_name == sample_pdf_path.name
        assert chunk.page_number in [1, 2]
        assert len(chunk.text) > 0
        assert chunk.doc_id != ""
        assert "doc_name" in chunk.metadata
        assert "page_number" in chunk.metadata


def test_document_processor(sample_pdf_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        vector_store = ChromaVectorStore(persist_directory=tmp_dir, collection_name="test_ingest")
        processor = DocumentProcessor(vector_store=vector_store, chunk_size=30, chunk_overlap=10)

        # First run: should index
        result = processor.process_pdf(sample_pdf_path)
        assert result.already_indexed is False
        assert result.indexed_chunks > 0
        assert vector_store.is_document_indexed(sample_pdf_path.name) is True

        # Second run without force: should detect already indexed
        result_dupe = processor.process_pdf(sample_pdf_path, force_reindex=False)
        assert result_dupe.already_indexed is True
        assert result_dupe.indexed_chunks == 0

        # Run with force_reindex: should re-index
        result_force = processor.process_pdf(sample_pdf_path, force_reindex=True)
        assert result_force.already_indexed is False
        assert result_force.indexed_chunks > 0
