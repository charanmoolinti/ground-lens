"""
Unit and integration tests for NLI verification engine and edge cases:
- Numerical differences (e.g. 94.5% vs 75%)
- Negations
- Missing evidence (Unverified)
- Exact support (Entailment)
- NumericalVerifier checks
- Report exports (CSV, JSON, Markdown)
"""

import tempfile
import json
import pytest
from backend.ingestion.text_chunker import DocumentChunk
from backend.retrieval.vector_store import ChromaVectorStore
from backend.verification.nli_classifier import NLIClassifier, VerificationStatus
from backend.verification.verification_engine import VerificationEngine
from backend.verification.numerical_verifier import NumericalVerifier
from backend.generation.answer_generator import GeneratedAnswer
from backend.health import HealthChecker, SubsystemState


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


def test_numerical_verifier_metrics():
    # Epoch count mismatch
    r1 = NumericalVerifier.verify_pairs(
        "The model was trained over 25 epochs using a batch size of 64.",
        "The model was trained over 150 epochs using a batch size of 64."
    )
    assert r1.has_conflict is True
    assert r1.conflict_type == "NUMERICAL_MISMATCH"
    assert "25" in r1.explanation and "150" in r1.explanation

    # Percentage distortion
    r2 = NumericalVerifier.verify_pairs(
        "Ground Lens reduced hallucination incidence by 42.8%.",
        "Ground Lens reduced hallucination incidence by 88.5%."
    )
    assert r2.has_conflict is True
    assert r2.conflict_type == "PERCENTAGE_DISTORTION"

    # BLEU metric distortion
    r3 = NumericalVerifier.verify_pairs(
        "Establishes a state-of-the-art BLEU score of 28.4 on WMT 2014.",
        "The transformer model scored 95.0 BLEU on English-to-German."
    )
    assert r3.has_conflict is True
    assert r3.conflict_type == "NUMERICAL_MISMATCH"

    # Date discrepancy
    r4 = NumericalVerifier.verify_pairs(
        "The final experiments were completed on October 14, 2024.",
        "The benchmark experiments were completed in January 2022."
    )
    assert r4.has_conflict is True
    assert r4.conflict_type == "DATE_MISMATCH"

    # Negation
    r5 = NumericalVerifier.verify_pairs(
        "ChromaDB requires persistent local disk storage to reload vector collections.",
        "ChromaDB never stores vector collections on local disk."
    )
    assert r5.has_conflict is True
    assert r5.conflict_type == "NEGATION_CONFLICT"


def test_health_checker():
    report = HealthChecker.audit_all()
    assert report is not None
    assert "pdf" in report.subsystems
    assert "embeddings" in report.subsystems
    assert "vector_store" in report.subsystems
    assert "nli" in report.subsystems
    assert "gemini" in report.subsystems
    assert report.subsystems["pdf"].state == SubsystemState.OPERATIONAL
    assert report.subsystems["embeddings"].state == SubsystemState.OPERATIONAL
    assert report.subsystems["nli"].state == SubsystemState.OPERATIONAL


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

        # Test exports
        csv_data = report.to_csv()
        assert "Claim ID,Claim Text,Status" in csv_data
        assert "94.5%" in csv_data

        json_data = report.to_json()
        parsed = json.loads(json_data)
        assert parsed["total_claims"] == 3
        assert parsed["claim_support_ratio"] == expected_ratio

        md_data = report.to_markdown()
        assert "# GROUND LENS — Verification Audit Report" in md_data
        assert "SUPPORTED" in md_data


def test_nli_classifier_cpu_initialization_no_meta_tensors():
    """Regression test: Verifies that NLI model initializes cleanly on CPU with real weights (no meta tensors)."""
    nli = NLIClassifier(device="cpu")
    assert nli.device == "cpu"
    assert nli.is_operational is True
    assert nli.loader_type in ("cross_encoder", "direct_transformers")

    # Check that underlying PyTorch model has all parameters on CPU and not on meta device
    model = nli.model.model if nli.loader_type == "cross_encoder" else nli.hf_model
    assert model is not None
    for name, param in model.named_parameters():
        assert param.device.type == "cpu", f"Parameter '{name}' is not on CPU: {param.device}"
        assert not getattr(param, "is_meta", False), f"Parameter '{name}' is a meta tensor!"
        assert param.numel() > 0, f"Parameter '{name}' has no data!"


def test_nli_classifier_singleton_caching():
    """Verifies that subsequent instantiations return the exact same cached instance without reloading."""
    nli1 = NLIClassifier(device="cpu")
    nli2 = NLIClassifier(device="cpu")
    assert nli1 is nli2
    assert id(nli1) == id(nli2)


def test_nli_classifier_direct_transformers_fallback():
    """Verifies that the direct Transformers fallback activates and works if CrossEncoder encounters an error."""
    import sentence_transformers
    from backend.verification import nli_classifier as nli_mod

    class BrokenCE:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()...")

    orig_ce = sentence_transformers.CrossEncoder
    sentence_transformers.CrossEncoder = BrokenCE
    try:
        # Reset singleton to force re-init
        nli_mod.NLIClassifier._instance = None
        nli_mod.NLIClassifier._loaded_key = None

        fallback_nli = nli_mod.NLIClassifier(device="cpu")
        assert fallback_nli.loader_type == "direct_transformers"
        assert fallback_nli.is_operational is True

        p = "The Aster embedding model produces vectors with 384 dimensions."
        h = "The Aster embedding model produces vectors with 384 dimensions."
        scores, status, conf = fallback_nli.predict_pair(p, h)
        assert status == VerificationStatus.SUPPORTED
        assert scores.entailment > 0.80
    finally:
        sentence_transformers.CrossEncoder = orig_ce
        # Reset singleton back to clean instance
        nli_mod.NLIClassifier._instance = None
        nli_mod.NLIClassifier._loaded_key = None

