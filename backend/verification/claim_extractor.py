"""
Module: claim_extractor.py
Description: Decomposes LLM-generated answers into discrete, testable factual claims.
Uses Gemini structured JSON output with Pydantic validation and includes a heuristic fallback.
Filters out greetings, opinions, and conversational filler.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from backend.generation.gemini_client import GeminiClient


class ExtractedClaim(BaseModel):
    """Represents a single testable factual statement."""
    claim_id: int = Field(description="Sequential identifier for the claim (1-indexed)")
    claim_text: str = Field(description="Atomic, self-contained factual proposition")


class ClaimExtractionResponse(BaseModel):
    """Container schema for the list of extracted factual claims."""
    claims: List[ExtractedClaim] = Field(default_factory=list)


class ClaimExtractor:
    """Decomposes answers into atomic testable claims using Gemini or heuristic backup."""

    SYSTEM_INSTRUCTION = (
        "You are an expert fact-checking claim extraction analyst. "
        "Your task is to decompose an answer into atomic, testable, self-contained factual claims.\n\n"
        "Guidelines:\n"
        "1. Extract ONLY specific factual assertions (e.g. metrics, dates, model names, methodologies, technical claims).\n"
        "2. Make each claim self-contained: resolve pronouns (e.g., replace 'it' with the actual subject/entity).\n"
        "3. Exclude greetings, conversational phrases ('sure', 'as mentioned'), opinions, and non-verifiable filler.\n"
        "4. If a sentence contains multiple distinct facts, split them into separate claims.\n"
        "5. If there are no factual claims, return an empty list."
    )

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.client = gemini_client or GeminiClient()

    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extracts factual claims from answer text using Gemini structured JSON.
        Falls back to rule-based sentence decomposition if Gemini is unavailable.
        """
        cleaned = text.strip()
        if not cleaned or cleaned.startswith("[Answer Generation Notice]"):
            return []

        if self.client.is_configured:
            try:
                return self._extract_with_gemini(cleaned)
            except Exception as e:
                # Fallback to rule-based extraction on API failure
                return self._extract_with_fallback(cleaned)
        else:
            return self._extract_with_fallback(cleaned)

    def _extract_with_gemini(self, text: str) -> List[ExtractedClaim]:
        """Uses Gemini structured JSON mode with ClaimExtractionResponse schema."""
        prompt = (
            f"Analyze the following text and extract all discrete, atomic factual claims:\n\n"
            f"TEXT TO ANALYZE:\n{text}\n\n"
            f"Return JSON matching the schema."
        )

        json_str = self.client.generate_json(
            prompt=prompt,
            response_schema=ClaimExtractionResponse,
            system_instruction=self.SYSTEM_INSTRUCTION,
            temperature=0.1,
        )

        parsed = json.loads(json_str)
        response_obj = ClaimExtractionResponse(**parsed)
        # Normalize IDs to 1..N
        for idx, claim in enumerate(response_obj.claims, 1):
            claim.claim_id = idx
            claim.claim_text = claim.claim_text.strip()

        return [c for c in response_obj.claims if len(c.claim_text) > 5]

    def _extract_with_fallback(self, text: str) -> List[ExtractedClaim]:
        """
        Rule-based heuristic fallback that splits sentences and filters out non-factual clauses.
        Used when Gemini API key is not yet set or during offline testing.
        """
        # Split on sentence terminals (. ! ?) while handling abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
        claims: List[ExtractedClaim] = []
        claim_id = 1

        filler_patterns = [
            r'^(hello|hi|sure|here is|certainly|in summary|based on|overall|to conclude)',
            r'^(i hope this helps|let me know|feel free)',
        ]

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue

            # Check if sentence is filler
            is_filler = any(re.search(pat, s_clean, re.IGNORECASE) for pat in filler_patterns)
            if is_filler and len(s_clean) < 40:
                continue

            # Check for bullet point markers
            s_clean = re.sub(r'^[\*\-\•\d+\.\)]\s*', '', s_clean)

            # Split compound clauses containing "and", "while", "furthermore" if multiple metrics exist
            claims.append(
                ExtractedClaim(
                    claim_id=claim_id,
                    claim_text=s_clean,
                )
            )
            claim_id += 1

        return claims
