"""
Unit tests for Module B: Embeddings and ChromaVectorStore retrieval.
"""

import tempfile
import pytest
from backend.embeddings.embedding_model import EmbeddingModel
from backend.retrieval.vector_store import ChromaVectorStore
from backend.ingestion.text_chunker import DocumentChunk


@pytest.fixture
def embedding_model():
    return EmbeddingModel()


def test_embedding_model_encode(embedding_model):
    text = "Natural Language Inference checks if evidence supports a claim."
    vec = embedding_model.encode(text)

    assert vec.ndim == 1
    assert vec.shape[0] == 384  # MiniLM-L6-v2 dimension is 384

    batch = [text, "Second test sentence."]
    vecs = embedding_model.encode(batch)
    assert vecs.shape == (2, 384)


def test_vector_store_search():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ChromaVectorStore(persist_directory=tmp_dir, collection_name="test_retrieval")

        chunks = [
            DocumentChunk(
                chunk_id="doc1_p1_c0",
                doc_name="doc1.pdf",
                page_number=1,
                text="The deep learning model achieved 94.5% verification accuracy.",
                chunk_index=0,
                metadata={"doc_name": "doc1.pdf", "page_number": 1},
            ),
            DocumentChunk(
                chunk_id="doc1_p2_c1",
                doc_name="doc1.pdf",
                page_number=2,
                text="Training was conducted over 50 epochs using an AdamW optimizer.",
                chunk_index=1,
                metadata={"doc_name": "doc1.pdf", "page_number": 2},
            ),
            DocumentChunk(
                chunk_id="doc2_p1_c0",
                doc_name="doc2.pdf",
                page_number=1,
                text="Ground Lens evaluates hallucinations in large language models.",
                chunk_index=0,
                metadata={"doc_name": "doc2.pdf", "page_number": 1},
            ),
        ]

        indexed = store.add_chunks(chunks)
        assert indexed == 3
        assert store.count() == 3

        # Semantic query
        results = store.search("What was the model accuracy?", top_k=2)
        assert len(results) == 2
        # Top result should be the chunk mentioning accuracy
        assert "94.5%" in results[0].text
        assert results[0].page_number == 1
        assert results[0].doc_name == "doc1.pdf"

        # Document filtering
        results_doc2 = store.search("What was the model accuracy?", top_k=2, doc_name="doc2.pdf")
        assert len(results_doc2) == 1
        assert results_doc2[0].doc_name == "doc2.pdf"

        # Listing documents
        summary = store.list_indexed_documents()
        assert len(summary) == 2
