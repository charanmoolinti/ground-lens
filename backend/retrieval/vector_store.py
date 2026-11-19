"""
Module: vector_store.py
Description: Persistent vector database interface using ChromaDB.
Stores embeddings, text chunks, and provenance metadata (doc_name, page_number).
Provides similarity retrieval with cosine distance, metadata filtering, and deduplication.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from backend.embeddings.embedding_model import EmbeddingModel
from backend.ingestion.text_chunker import DocumentChunk


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with similarity score and provenance."""
    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    similarity_score: float  # Normalized similarity in [0, 1]
    distance: float
    metadata: Dict[str, Any]


class ChromaVectorStore:
    """Persistent ChromaDB vector store manager for Ground Lens."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "ground_lens_docs",
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model or EmbeddingModel()

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Create or retrieve collection using cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def is_document_indexed(self, doc_name: str) -> bool:
        """Checks if any chunks for the given document name exist in the collection."""
        existing = self.collection.get(
            where={"doc_name": doc_name},
            limit=1,
        )
        return len(existing["ids"]) > 0

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        """
        Generates embeddings and indexes a list of DocumentChunks into ChromaDB.
        Skips if chunks list is empty.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        total_indexed = 0
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate embeddings in batches
        embeddings = self.embedding_model.encode_chunks(texts, batch_size=batch_size)

        for i in range(0, len(chunks), batch_size):
            batch_ids = chunk_ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings,
            )
            total_indexed += len(batch_ids)

        return total_indexed

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_name: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Performs semantic similarity search against indexed document chunks.

        Args:
            query: Question or claim text to embed and query.
            top_k: Number of most similar chunks to retrieve.
            doc_name: Optional filter by document name.

        Returns:
            List of RetrievalResult ordered by descending similarity.
        """
        if not query.strip():
            return []

        query_embedding = self.embedding_model.encode(query, normalize=True).tolist()
        where_filter = {"doc_name": doc_name} if doc_name else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results: List[RetrievalResult] = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            # Therefore similarity = 1 - distance (clamped to [0, 1])
            similarity = max(0.0, min(1.0, 1.0 - dist))
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_name=meta.get("doc_name", "unknown"),
                    page_number=meta.get("page_number", 1),
                    text=doc_text,
                    similarity_score=similarity,
                    distance=dist,
                    metadata=meta,
                )
            )

        return retrieval_results

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """Returns summary list of distinct documents currently stored in the vector collection."""
        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])

        doc_stats: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            name = meta.get("doc_name", "unknown")
            page = meta.get("page_number", 1)
            if name not in doc_stats:
                doc_stats[name] = {"doc_name": name, "chunk_count": 0, "pages": set()}
            doc_stats[name]["chunk_count"] += 1
            doc_stats[name]["pages"].add(page)

        summary = []
        for name, data in doc_stats.items():
            summary.append({
                "doc_name": name,
                "chunk_count": data["chunk_count"],
                "total_pages": len(data["pages"]),
            })
        return summary

    def delete_document(self, doc_name: str) -> None:
        """Removes all indexed chunks associated with a document name."""
        self.collection.delete(where={"doc_name": doc_name})

    def count(self) -> int:
        """Returns total number of chunks in the collection."""
        return self.collection.count()
