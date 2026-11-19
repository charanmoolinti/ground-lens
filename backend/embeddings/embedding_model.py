"""
Module: embedding_model.py
Description: Generates dense semantic vector embeddings for document passages and queries
using sentence-transformers/all-MiniLM-L6-v2 with batch processing support and caching.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around SentenceTransformer with thread-safe singleton caching."""

    _instance = None
    _model_name = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None or cls._model_name != model_name:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._init_model(model_name)
        return cls._instance

    def _init_model(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Generates dense vector embeddings for text or list of texts.

        Args:
            texts: Single string or list of strings.
            batch_size: Batch size for encoding.
            normalize: Normalize embeddings to unit length (for cosine similarity).

        Returns:
            Numpy array of shape (N, embedding_dim) or (embedding_dim,).
        """
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else texts

        embeddings = self.model.encode(
            input_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings[0] if is_single else embeddings

    def encode_chunks(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Convenience method returning Python list of floats for vector database storage."""
        arr = self.encode(texts, batch_size=batch_size, normalize=True)
        if arr.ndim == 1:
            return [arr.tolist()]
        return arr.tolist()
