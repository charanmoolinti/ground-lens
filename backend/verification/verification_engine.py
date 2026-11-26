"""
Module: verification_engine.py
Description: End-to-end claim verification pipeline.
Coordinates claim extraction, per-claim targeted evidence retrieval, DeBERTa NLI inference,
specialized numerical discrepancy verification, multi-passage conflict resolution,
and exportable metrics synthesis (claim-support ratio, CSV/JSON/Markdown exports).
"""

from __future__ import annotations
import csv
from dataclasses import dataclass, field
import io
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
from backend.verification.numerical_verifier import NumericalVerifier, NumericalCheckResult


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
    numerical_conflict: Optional[str] = None
    conflict_detected: bool = False
    passage_results: List[PassageNLIResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "status": self.status.value,
            "confidence": self.confidence,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "similarity_score": self.similarity_score,
            "has_evidence": self.has_evidence,
            "explanation": self.explanation,
            "numerical_conflict": self.numerical_conflict,
            "conflict_detected": self.conflict_detected,
            "best_passage": self.best_passage,
        }


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "generated_answer": self.generated_answer,
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "contradicted_count": self.contradicted_count,
            "unverified_count": self.unverified_count,
            "claim_support_ratio": self.claim_support_ratio,
            "verified_claims": [c.to_dict() for c in self.verified_claims],
            "disclaimer": self.disclaimer,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Claim ID", "Claim Text", "Status", "Confidence", "Document", "Page",
            "Similarity Score", "Explanation", "Numerical Conflict", "Best Passage"
        ])
        for c in self.verified_claims:
            writer.writerow([
                c.claim_id,
                c.claim_text,
                c.status.value,
                f"{c.confidence:.4f}",
                c.doc_name or "N/A",
                c.page_number or "N/A",
                f"{c.similarity_score:.4f}",
                c.explanation,
                c.numerical_conflict or "None",
                c.best_passage or "N/A",
            ])
        return output.getvalue()

    def to_markdown(self) -> str:
        md = [
            f"# GROUND LENS — Verification Audit Report\n",
            f"**Query / Question:** {self.question}\n",
            f"**Claim Support Ratio:** {self.claim_support_ratio}%\n",
            f"**Summary Metrics:** Total Claims: {self.total_claims} | "
            f"Supported: {self.supported_count} | Contradicted: {self.contradicted_count} | Unverified: {self.unverified_count}\n",
            f"---\n",
            f"### Claim-Level Audit Ledger\n",
        ]
        for c in self.verified_claims:
            md.append(f"#### Claim {c.claim_id}: [{c.status.value}] ({c.confidence * 100:.1f}% confidence)")
            md.append(f"> \"{c.claim_text}\"\n")
            md.append(f"- **Explanation:** {c.explanation}")
            if c.doc_name:
                md.append(f"- **Source Provenance:** `{c.doc_name}` (Page {c.page_number})")
            if c.numerical_conflict:
                md.append(f"- **Numerical Check:** {c.numerical_conflict}")
            if c.best_passage:
                md.append(f"- **Retrieved Evidence:**\n  > {c.best_passage}\n")
            md.append("")
        md.append(f"---\n*{self.disclaimer}*")
        return "\n".join(md)


class VerificationEngine:
    """Orchestrates full claim extraction, retrieval, numerical check, and NLI verification pipeline."""

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
            top_k=4,
            min_similarity_threshold=min_similarity_threshold,
        )
        self.nli_classifier = nli_classifier or NLIClassifier(device="cpu")
        self.support_threshold = support_threshold
        self.contradict_threshold = contradict_threshold

    @staticmethod
    def _extract_content_tokens(text: str) -> Set[str]:
        """Extracts significant content words from text for semantic relevance filtering."""
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "are", "was",
            "were", "has", "have", "had", "can", "could", "would", "its", "our",
            "all", "each", "page", "per", "use", "uses", "into", "than", "then",
            "also", "been", "which", "more", "most", "about", "such", "only",
            "does", "what", "when", "where", "which", "there", "their", "they",
            "system", "retrieval", "document", "spec", "specification", "version"
        }
        words = re.findall(r"\b[a-z0-9_]{3,}\b", text.lower())
        return {w for w in words if w not in stopwords}

    @staticmethod
    def extract_passage_windows(text: str) -> List[str]:
        """Splits a chunk into overlapping sentence windows for high-fidelity premise matching."""
        clean = re.sub(r"\n+", " ", text).strip()
        clean_norm = re.sub(r"\b(\d+)\.\s+", r"\1: ", clean)
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean_norm)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]

        windows = [clean]
        if len(sentences) > 1:
            for s in sentences:
                if len(s) >= 25:
                    windows.append(s)
            for i in range(len(sentences) - 1):
                windows.append(f"{sentences[i]} {sentences[i+1]}")
            for i in range(len(sentences) - 2):
                windows.append(f"{sentences[i]} {sentences[i+1]} {sentences[i+2]}")
        return windows

    def evaluate_passage_for_claim(
        self, passage_text: str, claim_text: str
    ) -> Tuple[NLIScore, VerificationStatus, float, str, Optional[str]]:
        """
        Evaluates a passage against a claim using both NumericalVerifier and DeBERTa NLI cross-encoder.
        Returns: (scores, status, confidence, evidence_window, numerical_conflict_note)
        """
        # Step 1: Rule-based numerical & negation check
        num_check: NumericalCheckResult = NumericalVerifier.verify_pairs(passage_text, claim_text)
        if num_check.has_conflict:
            # Concrete factual / numerical contradiction
            scores = NLIScore(entailment=0.01, contradiction=0.98, neutral=0.01)
            return scores, VerificationStatus.CONTRADICTED, 0.98, passage_text, num_check.explanation

        # Step 2: Multi-window DeBERTa NLI cross-encoder evaluation with lexical relevance filtering
        claim_tokens = self._extract_content_tokens(claim_text)
        raw_windows = self.extract_passage_windows(passage_text)

        windows: List[str] = []
        for i, win in enumerate(raw_windows):
            if i == 0:
                windows.append(win)  # Full passage always evaluated
            else:
                win_tokens = self._extract_content_tokens(win)
                # Sub-window must share at least 1 topical content token with claim to prevent noise
                if len(claim_tokens & win_tokens) >= 1:
                    windows.append(win)

        best_support_w = None
        best_support_score = None
        best_support_c = -1.0

        best_contra_w = None
        best_contra_score = None
        best_contra_c = -1.0

        best_neutral_w = passage_text
        best_neutral_score = None
        best_neutral_c = -1.0

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

        # Decision rules:
        # A. Factual entailment priority: If any relevant window or the passage entails the claim
        if best_support_w and best_support_c >= self.support_threshold:
            return best_support_score, VerificationStatus.SUPPORTED, best_support_c, best_support_w, None

        # B. Contradiction: Ensure contradictory window shares topical context (>= 2 content tokens)
        if best_contra_w and best_contra_c >= self.contradict_threshold:
            contra_tokens = self._extract_content_tokens(best_contra_w)
            if len(claim_tokens & contra_tokens) >= 2:
                return best_contra_score, VerificationStatus.CONTRADICTED, best_contra_c, best_contra_w, None

        # C. Fallback: UNVERIFIED (neutral or ambiguous)
        neutral_score = best_neutral_score or NLIScore(entailment=0.0, contradiction=0.0, neutral=1.0)
        return neutral_score, VerificationStatus.UNVERIFIED, max(0.0, best_neutral_c), best_neutral_w, None

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
        3. Evaluates premise-hypothesis relationship via NumericalVerifier and DeBERTa NLI.
        4. Synthesizes verification report with claim-support ratio and conflict detection.
        """
        if isinstance(answer, str):
            answer = GeneratedAnswer(
                question="Direct Statement Verification",
                answer_text=answer,
                retrieved_passages=[],
            )

        # Strictly bypass LLM claim extraction for Direct Statement Verification to ensure zero API calls
        if answer.question == "Direct Statement Verification":
            raw_claims = self.claim_extractor._extract_with_fallback(answer.answer_text)
            if not raw_claims and len(answer.answer_text.strip()) > 5:
                raw_claims = [ExtractedClaim(claim_id=1, claim_text=answer.answer_text.strip())]
        else:
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
                top_k=4,
            )

            # Strict Rule: If no relevant evidence was found, classify as UNVERIFIED (Never Contradicted)
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

            # Run NLI and numerical checks across all retrieved candidate passages
            passage_results: List[PassageNLIResult] = []
            has_contradiction = False
            best_contra_passage_text = None
            best_contra_passage = None
            best_contra_conf = 0.0
            best_contra_scores = None
            contra_explanation = None

            has_support = False
            best_support_passage_text = None
            best_support_passage = None
            best_support_conf = 0.0
            best_support_scores = None

            for passage in claim_evidence.evidence_passages:
                scores, status, conf, evidence_excerpt, num_conflict = self.evaluate_passage_for_claim(
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
                    contra_explanation = num_conflict

                if status == VerificationStatus.SUPPORTED and conf > best_support_conf:
                    has_support = True
                    best_support_conf = conf
                    best_support_passage = passage
                    best_support_passage_text = evidence_excerpt
                    best_support_scores = scores

            # Decision Logic & Conflict Detection:
            conflict_detected = has_support and has_contradiction

            # 1. Concrete numerical discrepancy or polarity negation takes absolute priority
            if has_contradiction and contra_explanation:
                conflict_note = " (Note: Conflicting evidence detected across retrieved passages)" if conflict_detected else ""
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
                        numerical_conflict=contra_explanation,
                        explanation=f"{contra_explanation}{conflict_note}",
                        conflict_detected=conflict_detected,
                        passage_results=passage_results,
                    )
                )
            # 2. Corroborated factual support
            elif has_support and (not has_contradiction or best_support_conf >= best_contra_conf):
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
                        explanation=f"Corroborated by source passage (Page {best_support_passage.page_number}).",
                        conflict_detected=conflict_detected,
                        passage_results=passage_results,
                    )
                )
            # 3. Textual contradiction
            elif has_contradiction:
                conflict_note = " (Note: Conflicting evidence detected across retrieved passages)" if conflict_detected else ""
                base_expl = contra_explanation or f"Explicitly contradicted by source passage (Page {best_contra_passage.page_number})."
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
                        numerical_conflict=contra_explanation,
                        explanation=f"{base_expl}{conflict_note}",
                        conflict_detected=conflict_detected,
                        passage_results=passage_results,
                    )
                )
            # 4. Unverified fallback
            else:
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
                        explanation="Retrieved passages do not contain sufficient evidence to confirm or refute the claim.",
                        conflict_detected=False,
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
