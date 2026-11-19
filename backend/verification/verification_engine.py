"""
Module: verification_engine.py
Description: End-to-end claim verification pipeline.
Coordinates claim extraction, per-claim targeted evidence retrieval, DeBERTa NLI inference,
multi-passage conflict resolution, and metrics synthesis (claim-support ratio).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from backend.generation.answer_generator import GeneratedAnswer
from backend.generation.gemini_client import GeminiClient
from backend.retrieval.vector_store import ChromaVectorStore, RetrievalResult
from backend.verification.claim_extractor import ClaimExtractor, ExtractedClaim
from backend.verification.evidence_retriever import ClaimEvidence, EvidenceRetriever
from backend.verification.nli_classifier import (
    NLIClassifier,
    NLIScore,
    PassageNLIResult,
    VerificationStatus,
)


@dataclass
class VerifiedClaim:
    """Detailed verification outcome for an individual factual claim."""
    claim_id: int
    claim_text: str
    status: VerificationStatus
    confidence: float
    best_passage: Optional[str]
    doc_name: Optional[str]
    page_number: Optional[int]
    similarity_score: float
    nli_scores: Optional[NLIScore]
    has_evidence: bool = True
    explanation: str = ""
    passage_results: List[PassageNLIResult] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Complete verification report containing aggregated metrics and claim details."""
    question: str
    generated_answer: str
    total_claims: int
    supported_count: int
    contradicted_count: int
    unverified_count: int
    claim_support_ratio: float  # Percentage in [0, 100]
    verified_claims: List[VerifiedClaim]
    disclaimer: str = (
        "Ground Lens claim-support ratio measures the proportion of factual statements corroborated "
        "by retrieved document evidence. It reflects alignment with the provided sources, not absolute universal truth."
    )


class VerificationEngine:
    """Orchestrates full claim extraction and verification pipeline."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        nli_classifier: Optional[NLIClassifier] = None,
        support_threshold: float = 0.55,
        contradict_threshold: float = 0.55,
        min_similarity_threshold: float = 0.20,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.claim_extractor = ClaimExtractor(gemini_client=gemini_client)
        self.evidence_retriever = EvidenceRetriever(
            vector_store=self.vector_store,
            top_k=3,
            min_similarity_threshold=min_similarity_threshold,
        )
        self.nli_classifier = nli_classifier or NLIClassifier()
        self.support_threshold = support_threshold
        self.contradict_threshold = contradict_threshold

    @staticmethod
    def extract_passage_windows(text: str, window_size: int = 2) -> List[str]:
        """Splits a chunk into overlapping sentence windows for high-fidelity NLI premise matching."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= window_size:
            return [text.strip()]
        windows = []
        for i in range(len(sentences)):
            win = ' '.join(sentences[max(0, i - 1) : min(len(sentences), i + window_size)])
            windows.append(win.strip())
        return windows

    def evaluate_passage_for_claim(self, passage_text: str, claim_text: str) -> Tuple[NLIScore, VerificationStatus, float, str]:
        """Evaluates sentence windows within a chunk and returns the most informative evidence window."""
        windows = self.extract_passage_windows(passage_text)
        best_scores = None
        best_status = VerificationStatus.UNVERIFIED
        best_conf = -1.0
        best_window = passage_text

        # First evaluate full chunk
        full_scores, full_status, full_conf = self.nli_classifier.predict_pair(
            premise=passage_text,
            hypothesis=claim_text,
            support_threshold=self.support_threshold,
            contradict_threshold=self.contradict_threshold,
        )
        best_scores, best_status, best_conf = full_scores, full_status, full_conf

        # Track best supporting window and best contradicting window
        best_support_w = None
        best_support_score = None
        best_support_c = -1.0

        best_contra_w = None
        best_contra_score = None
        best_contra_c = -1.0

        best_neutral_w = passage_text
        best_neutral_score = full_scores
        best_neutral_c = full_conf

        for win in windows:
            if len(win) < 20:
                continue
            w_scores, w_status, w_conf = self.nli_classifier.predict_pair(
                premise=win,
                hypothesis=claim_text,
                support_threshold=self.support_threshold,
                contradict_threshold=self.contradict_threshold,
            )

            if w_status == VerificationStatus.SUPPORTED and w_conf > best_support_c:
                best_support_c = w_conf
                best_support_w = win
                best_support_score = w_scores

            elif w_status == VerificationStatus.CONTRADICTED and w_conf > best_contra_c:
                best_contra_c = w_conf
                best_contra_w = win
                best_contra_score = w_scores

            elif w_status == VerificationStatus.UNVERIFIED and w_conf > best_neutral_c:
                best_neutral_c = w_conf
                best_neutral_w = win
                best_neutral_score = w_scores

        # Priority 1: If there is clear direct support (entailment >= threshold) from an evidence window, mark SUPPORTED
        if best_support_w and best_support_c >= self.support_threshold:
            return best_support_score, VerificationStatus.SUPPORTED, best_support_c, best_support_w

        # Priority 2: If there is no clear support, but a window explicitly contradicts (contradiction >= threshold), mark CONTRADICTED
        if best_contra_w and best_contra_c >= self.contradict_threshold:
            return best_contra_score, VerificationStatus.CONTRADICTED, best_contra_c, best_contra_w

        # Priority 3: Otherwise UNVERIFIED
        return best_neutral_score, VerificationStatus.UNVERIFIED, best_neutral_c, best_neutral_w

    def verify(
        self,
        answer: Union[GeneratedAnswer, str],
        doc_name: Optional[str] = None,
    ) -> VerificationReport:
        """Primary verification entry point accepting GeneratedAnswer or raw text string."""
        return self.verify_answer(answer=answer, doc_name=doc_name)

    def verify_answer(
        self,
        answer: Union[GeneratedAnswer, str],
        doc_name: Optional[str] = None,
    ) -> VerificationReport:
        """
        Runs complete verification on a generated answer:
        1. Extracts discrete factual claims from answer.
        2. Retrieves specific evidence passages for each claim.
        3. Evaluates premise-hypothesis relationship via DeBERTa NLI.
        4. Synthesizes verification report with claim-support ratio.
        """
        if isinstance(answer, str):
            answer = GeneratedAnswer(
                question="Direct Claim Verification",
                answer_text=answer,
                retrieved_passages=[],
            )

        raw_claims = self.claim_extractor.extract_claims(answer.answer_text)

        if not raw_claims:
            return VerificationReport(
                question=answer.question,
                generated_answer=answer.answer_text,
                total_claims=0,
                supported_count=0,
                contradicted_count=0,
                unverified_count=0,
                claim_support_ratio=0.0,
                verified_claims=[],
            )

        verified_claims: List[VerifiedClaim] = []

        for claim in raw_claims:
            claim_evidence = self.evidence_retriever.retrieve_for_claim(
                claim=claim,
                doc_name=doc_name,
                top_k=3,
            )

            # Rule: If no relevant evidence was found, classify as UNVERIFIED
            if not claim_evidence.has_relevant_evidence or not claim_evidence.evidence_passages:
                verified_claims.append(
                    VerifiedClaim(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        status=VerificationStatus.UNVERIFIED,
                        confidence=0.0,
                        best_passage=None,
                        doc_name=None,
                        page_number=None,
                        similarity_score=claim_evidence.top_similarity_score,
                        nli_scores=None,
                        has_evidence=False,
                        explanation="Insufficient or no relevant source evidence found in the indexed documents.",
                        passage_results=[],
                    )
                )
                continue

            # Run NLI against each candidate passage with sentence windowing
            passage_results: List[PassageNLIResult] = []
            has_contradiction = False
            best_contra_passage_text = None
            best_contra_passage = None
            best_contra_conf = 0.0
            best_contra_scores = None

            has_support = False
            best_support_passage_text = None
            best_support_passage = None
            best_support_conf = 0.0
            best_support_scores = None

            for passage in claim_evidence.evidence_passages:
                scores, status, conf, evidence_excerpt = self.evaluate_passage_for_claim(
                    passage_text=passage.text,
                    claim_text=claim.claim_text,
                )

                res = PassageNLIResult(
                    premise_text=evidence_excerpt,
                    doc_name=passage.doc_name,
                    page_number=passage.page_number,
                    scores=scores,
                    status=status,
                    confidence=conf,
                )
                passage_results.append(res)

                if status == VerificationStatus.CONTRADICTED and conf > best_contra_conf:
                    has_contradiction = True
                    best_contra_conf = conf
                    best_contra_passage = passage
                    best_contra_passage_text = evidence_excerpt
                    best_contra_scores = scores

                if status == VerificationStatus.SUPPORTED and conf > best_support_conf:
                    has_support = True
                    best_support_conf = conf
                    best_support_passage = passage
                    best_support_passage_text = evidence_excerpt
                    best_support_scores = scores

            # Decision Logic & Conflict Detection:
            # When corroborating evidence exists with high confidence, corroborate the claim
            if has_support and (not has_contradiction or best_support_conf >= best_contra_conf):
                verified_claims.append(
                    VerifiedClaim(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        status=VerificationStatus.SUPPORTED,
                        confidence=best_support_conf,
                        best_passage=best_support_passage_text or (best_support_passage.text if best_support_passage else None),
                        doc_name=best_support_passage.doc_name if best_support_passage else None,
                        page_number=best_support_passage.page_number if best_support_passage else None,
                        similarity_score=claim_evidence.top_similarity_score,
                        nli_scores=best_support_scores,
                        explanation=f"Directly supported by source passage (Page {best_support_passage.page_number}).",
                        passage_results=passage_results,
                    )
                )
            elif has_contradiction:
                conflict_note = " (Evidence conflict detected across multiple retrieved passages)" if has_support else ""
                verified_claims.append(
                    VerifiedClaim(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        status=VerificationStatus.CONTRADICTED,
                        confidence=best_contra_conf,
                        best_passage=best_contra_passage_text or (best_contra_passage.text if best_contra_passage else None),
                        doc_name=best_contra_passage.doc_name if best_contra_passage else None,
                        page_number=best_contra_passage.page_number if best_contra_passage else None,
                        similarity_score=claim_evidence.top_similarity_score,
                        nli_scores=best_contra_scores,
                        explanation=f"Explicitly contradicted by source passage (Page {best_contra_passage.page_number}){conflict_note}.",
                        passage_results=passage_results,
                    )
                )

            else:
                # Top neutral passage
                top_p = claim_evidence.evidence_passages[0]
                top_res = passage_results[0]
                verified_claims.append(
                    VerifiedClaim(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        status=VerificationStatus.UNVERIFIED,
                        confidence=top_res.confidence,
                        best_passage=top_res.premise_text or top_p.text,
                        doc_name=top_p.doc_name,
                        page_number=top_p.page_number,
                        similarity_score=claim_evidence.top_similarity_score,
                        nli_scores=top_res.scores,
                        has_evidence=True,
                        explanation="Retrieved passages do not contain sufficient evidence to confirm or refute.",
                        passage_results=passage_results,
                    )
                )

        # Compute aggregate metrics
        total = len(verified_claims)
        supported = sum(1 for c in verified_claims if c.status == VerificationStatus.SUPPORTED)
        contradicted = sum(1 for c in verified_claims if c.status == VerificationStatus.CONTRADICTED)
        unverified = sum(1 for c in verified_claims if c.status == VerificationStatus.UNVERIFIED)

        ratio = round((supported / total * 100), 1) if total > 0 else 0.0

        return VerificationReport(
            question=answer.question,
            generated_answer=answer.answer_text,
            total_claims=total,
            supported_count=supported,
            contradicted_count=contradicted,
            unverified_count=unverified,
            claim_support_ratio=ratio,
            verified_claims=verified_claims,
        )
