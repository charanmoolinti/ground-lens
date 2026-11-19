"""
Unit and integration tests for NLI verification engine and edge cases:
- Numerical differences (e.g. 94.5% vs 75%)
- Negations
- Missing evidence (Unverified)
- Exact support (Entailment)
"""

import tempfile
import pytest
from backend.ingestion.text_chunker import DocumentChunk
from backend.retrieval.vector_store import ChromaVectorStore
from backend.verification.nli_classifier import NLIClassifier, VerificationStatus
from backend.verification.verification_engine import VerificationEngine
from backend.generation.answer_generator import GeneratedAnswer


@pytest.fixture(scope="module")
def nli_classifier():
    return NLIClassifier()


def test_nli_classifier_entailment(nli_classifier):
    premise = "The model was evaluated on 10,000 images and achieved an accuracy of 94.5%."
    hypothesis = "The model achieved 94.5% accuracy."
    scores, status, conf = nli_classifier.predict_pair(premise, hypothesis)

    assert status == VerificationStatus.SUPPORTED
    assert scores.entailment > 0.70


def test_nli_classifier_contradiction_numerical(nli_classifier):
    premise = "The training completed in 12 hours using 8 A100 GPUs."
    hypothesis = "The training completed in 48 hours."
    scores, status, conf = nli_classifier.predict_pair(premise, hypothesis)

    assert status == VerificationStatus.CONTRADICTED
    assert scores.contradiction > 0.60


def test_nli_classifier_contradiction_negation(nli_classifier):
    premise = "Ground Lens requires an embedding model and a vector database."
    hypothesis = "Ground Lens does not require any embedding model."
    scores, status, conf = nli_classifier.predict_pair(premise, hypothesis)

    assert status == VerificationStatus.CONTRADICTED
    assert scores.contradiction > 0.60


def test_nli_classifier_neutral(nli_classifier):
    premise = "The framework is implemented in Python and runs on Linux."
    hypothesis = "The framework was originally created at Stanford University."
    scores, status, conf = nli_classifier.predict_pair(premise, hypothesis)

    assert status == VerificationStatus.UNVERIFIED
    assert scores.neutral > 0.40


def test_verification_engine_end_to_end(nli_classifier):
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ChromaVectorStore(persist_directory=tmp_dir, collection_name="test_engine")
        store.add_chunks([
            DocumentChunk(
                chunk_id="doc1_p1_c0",
                doc_name="ground_lens_spec.pdf",
                page_number=1,
                text="Ground Lens achieved 94.5% verification accuracy across benchmark evaluation sets.",
                chunk_index=0,
                metadata={"doc_name": "ground_lens_spec.pdf", "page_number": 1},
            ),
            DocumentChunk(
                chunk_id="doc1_p2_c1",
                doc_name="ground_lens_spec.pdf",
                page_number=2,
                text="The system reduced latency by 45% compared to baseline pipelines.",
                chunk_index=1,
                metadata={"doc_name": "ground_lens_spec.pdf", "page_number": 2},
            ),
        ])

        engine = VerificationEngine(
            vector_store=store,
            nli_classifier=nli_classifier,
            support_threshold=0.55,
            contradict_threshold=0.55,
        )

        # Multi-claim answer with:
        # 1. Supported claim: 94.5% accuracy
        # 2. Contradicted claim: reduced latency by 90% (actual is 45%)
        # 3. Unverified claim: trained by 100 researchers
        answer = GeneratedAnswer(
            question="What are the performance metrics of Ground Lens?",
            answer_text=(
                "Ground Lens achieved 94.5% verification accuracy. "
                "Ground Lens reduced latency by 90%. "
                "The platform was built by 100 researchers."
            ),
            retrieved_passages=[],
        )

        report = engine.verify_answer(answer)

        assert report.total_claims == 3
        assert report.supported_count >= 1
        assert report.contradicted_count >= 1
        assert report.unverified_count >= 1

        # Check claim support ratio calculation
        expected_ratio = round((report.supported_count / report.total_claims) * 100, 1)
        assert report.claim_support_ratio == expected_ratio

        # Verify page provenance is preserved on the supported claim
        sup_claim = next(c for c in report.verified_claims if c.status == VerificationStatus.SUPPORTED)
        assert sup_claim.page_number == 1
        assert sup_claim.doc_name == "ground_lens_spec.pdf"

        # Verify engine.verify alias works identically with GeneratedAnswer
        report_alias = engine.verify(answer)
        assert report_alias.total_claims == report.total_claims
        assert report_alias.supported_count == report.supported_count

        # Verify engine.verify works with raw string input
        report_str = engine.verify("Ground Lens achieved 94.5% verification accuracy.")
        assert report_str.total_claims >= 1
        assert report_str.supported_count >= 1

