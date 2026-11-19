"""
Module: evidence_retriever.py
Description: Performs independent vector retrieval for each extracted claim.
Ensures verification does not merely reuse the question's retrieval context,
finding the most relevant document passages and page numbers for each specific proposition.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from backend.retrieval.vector_store import ChromaVectorStore, RetrievalResult
from backend.verification.claim_extractor import ExtractedClaim


@dataclass
class ClaimEvidence:
    """Represents an extracted claim paired with its independently retrieved evidence passages."""
    claim: ExtractedClaim
    evidence_passages: List[RetrievalResult]
    top_similarity_score: float
    has_relevant_evidence: bool


class EvidenceRetriever:
    """Retrieves targeted source evidence chunks from ChromaDB for discrete claims."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        top_k: int = 3,
        min_similarity_threshold: float = 0.20,
    ):
        """
        Args:
            vector_store: ChromaVectorStore instance.
            top_k: Number of candidate evidence passages to retrieve per claim.
            min_similarity_threshold: Minimum cosine similarity to consider evidence relevant.
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.top_k = top_k
        self.min_similarity_threshold = min_similarity_threshold

    def retrieve_for_claim(
        self,
        claim: ExtractedClaim,
        doc_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> ClaimEvidence:
        """Retrieves candidate evidence passages specifically matching the claim text."""
        k = top_k or self.top_k
        passages = self.vector_store.search(
            query=claim.claim_text,
            top_k=k,
            doc_name=doc_name,
        )

        top_score = passages[0].similarity_score if passages else 0.0
        has_relevant = (top_score >= self.min_similarity_threshold) and (len(passages) > 0)

        return ClaimEvidence(
            claim=claim,
            evidence_passages=passages,
            top_similarity_score=top_score,
            has_relevant_evidence=has_relevant,
        )

    def retrieve_for_claims(
        self,
        claims: List[ExtractedClaim],
        doc_name: Optional[str] = None,
    ) -> List[ClaimEvidence]:
        """Batch retrieval of evidence for a list of extracted claims."""
        return [self.retrieve_for_claim(c, doc_name=doc_name) for c in claims]
