"""
Module: gemini_client.py
Description: Wrapper around Google Gen AI Python SDK (google-genai) for grounded answer
generation and structured JSON output. Supports runtime key configuration and graceful error handling.
"""

import os
from typing import Any, Dict, Optional, Type
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()


class GeminiClient:
    """Manages Gemini API authentication, requests, and structured outputs."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY", None)
            except Exception:
                pass
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self._client: Optional[genai.Client] = None

        if self.api_key and self.api_key.strip() and self.api_key != "your_gemini_api_key_here":
            try:
                self._client = genai.Client(api_key=self.api_key.strip())
            except Exception as e:
                self._client = None
                self._init_error = str(e)
        else:
            self._init_error = "API key not configured"

    @property
    def is_configured(self) -> bool:
        """Returns True if the client is initialized with an active API key."""
        return self._client is not None

    def update_api_key(self, api_key: str):
        """Updates the API key at runtime (e.g., from the UI)."""
        self.api_key = api_key.strip()
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)
            self._init_error = None

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generates plain text response using Gemini."""
        if not self.is_configured:
            raise RuntimeError(
                "Gemini API key is not configured. Please provide a valid GEMINI_API_KEY in .env or via the sidebar."
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        if not response.text:
            return ""
        return response.text.strip()

    def generate_json(
        self,
        prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Generates structured JSON response guaranteed to adhere to JSON format or schema.
        """
        if not self.is_configured:
            raise RuntimeError(
                "Gemini API key is not configured. Please provide a valid GEMINI_API_KEY in .env or via the sidebar."
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction,
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        if not response.text:
            return "{}"
        return response.text.strip()
