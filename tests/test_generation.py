"""
Unit tests for Module C & D: Gemini Client, AnswerGenerator, and ClaimExtractor.
"""

from unittest.mock import MagicMock
import pytest
from backend.generation.answer_generator import AnswerGenerator
from backend.generation.gemini_client import GeminiClient
from backend.retrieval.vector_store import ChromaVectorStore, RetrievalResult
from backend.verification.claim_extractor import ClaimExtractor, ExtractedClaim


def test_claim_extractor_fallback():
    extractor = ClaimExtractor(gemini_client=MagicMock(is_configured=False))
    sample_text = (
        "Hello! Ground Lens is a hallucination detection system. "
        "The model achieves 94.5% verification accuracy across benchmark sets. "
        "It reduced verification latency by 45% compared to existing baselines. "
        "I hope this helps you!"
    )

    claims = extractor.extract_claims(sample_text)
    assert len(claims) >= 2
    claim_texts = [c.claim_text for c in claims]
    assert any("94.5%" in t for t in claim_texts)
    assert any("45%" in t for t in claim_texts)
    # Filler sentences should not be primary claims
    assert not any("I hope this helps" in t for t in claim_texts)


def test_claim_extractor_mocked_gemini():
    mock_client = MagicMock(spec=GeminiClient)
    mock_client.is_configured = True
    mock_client.generate_json.return_value = """
    {
      "claims": [
        {"claim_id": 1, "claim_text": "Ground Lens achieved 94.5% verification accuracy."},
        {"claim_id": 2, "claim_text": "Ground Lens reduced latency by 45%."}
      ]
    }
    """

    extractor = ClaimExtractor(gemini_client=mock_client)
    claims = extractor.extract_claims("Some answer text")

    assert len(claims) == 2
    assert claims[0].claim_text == "Ground Lens achieved 94.5% verification accuracy."
    assert claims[1].claim_text == "Ground Lens reduced latency by 45%."


def test_answer_generator_prompt_and_fallback():
    mock_store = MagicMock(spec=ChromaVectorStore)
    mock_store.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_1",
            doc_name="paper.pdf",
            page_number=1,
            text="Ground Lens uses a DeBERTa model for NLI verification.",
            similarity_score=0.92,
            distance=0.08,
            metadata={"doc_name": "paper.pdf", "page_number": 1},
        )
    ]

    mock_client = MagicMock(spec=GeminiClient)
    mock_client.is_configured = True
    mock_client.generate_content.return_value = "Ground Lens uses a DeBERTa model to perform NLI verification."

    generator = AnswerGenerator(gemini_client=mock_client, vector_store=mock_store)
    result = generator.generate_answer("What model does Ground Lens use?")

    assert "DeBERTa" in result.answer_text
    assert len(result.retrieved_passages) == 1
    assert result.retrieved_passages[0].page_number == 1
    assert result.retrieved_passages[0].doc_name == "paper.pdf"
