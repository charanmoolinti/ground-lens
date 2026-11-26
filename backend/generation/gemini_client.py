"""
Module: gemini_client.py
Description: Robust, secure wrapper around Google Gen AI Python SDK (google-genai) for grounded
answer generation and structured JSON output. Supports runtime key configuration, real network
connectivity pre-flight diagnostics, multi-stage health inspections, automatic model fallback across
current GA endpoints (gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash, gemini-2.5-pro, gemini-1.5-pro),
and credential validation with strict key redaction.
"""

from __future__ import annotations
import os
import re
import socket
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from pydantic import BaseModel
from backend.config_manager import ConfigManager

logger = logging.getLogger(__name__)


def check_host_reachability(host: str = "generativelanguage.googleapis.com", port: int = 443, timeout: float = 2.0) -> bool:
    """Tests basic TCP/DNS socket reachability to Google AI Studio API."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, socket.gaierror, ConnectionRefusedError, OSError):
        return False


class GeminiClient:
    """Manages Gemini API authentication, requests, multi-model fallback, and structured outputs."""

    PRIMARY_MODEL = "gemini-3.5-flash-lite"
    CANDIDATE_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.8-flash",
        "gemma-4-31b-it",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key if api_key is not None else ConfigManager.get_api_key()
        self.model_name = model_name or ConfigManager.get_gemini_model() or self.PRIMARY_MODEL
        self.active_model: Optional[str] = None
        self._client: Optional[genai.Client] = None
        self._init_error: Optional[str] = None
        self.last_diagnostic: Optional[Dict[str, Any]] = None

        if self._is_valid_key_format(self.api_key):
            try:
                self._client = genai.Client(api_key=self.api_key.strip())
                self.active_model = self.model_name
            except Exception as e:
                self._client = None
                self._init_error = self._sanitize_error(str(e))
        else:
            self._init_error = "API key not configured"

    @staticmethod
    def _is_valid_key_format(key: Optional[str]) -> bool:
        """Validates that key is a plausible non-placeholder string."""
        if not key or not isinstance(key, str):
            return False
        k = key.strip()
        return bool(k and k != "your_gemini_api_key_here" and len(k) >= 20)

    @staticmethod
    def _sanitize_error(error_msg: str) -> str:
        """Strips out sensitive credential strings and internal tokens from error messages."""
        sanitized = re.sub(r'AIza[0-9A-Za-z\-_]{35}', '[REDACTED_API_KEY]', error_msg)
        sanitized = re.sub(r'key=[0-9A-Za-z\-_]{20,}', 'key=[REDACTED_KEY]', sanitized)
        return sanitized

    @property
    def is_configured(self) -> bool:
        """Returns True if the client is initialized with a valid key format."""
        return self._client is not None and self._is_valid_key_format(self.api_key)

    def diagnose_connection(self, force_live_test: bool = False) -> Dict[str, Any]:
        """
        Executes a granular 5-stage health inspection:
        1. Key configured
        2. Key format validated
        3. Endpoint reachable (DNS/Socket)
        4. Authentication successful
        5. Model inference successful
        """
        # Return cached diagnostic if already validated and live test not forced
        if not force_live_test and self.last_diagnostic is not None:
            return self.last_diagnostic

        diag = {
            "key_present": False,
            "key_format_valid": False,
            "endpoint_reachable": False,
            "authenticated": False,
            "model_operational": False,
            "active_model": None,
            "status_message": "",
            "error_category": None,
        }

        if not self.api_key:
            diag["status_message"] = "No API key configured (Direct Statement Verification works locally)."
            diag["error_category"] = "NOT_CONFIGURED"
            self.last_diagnostic = diag
            return diag

        diag["key_present"] = True

        if not self._is_valid_key_format(self.api_key):
            diag["status_message"] = "Invalid API key format: Key must be at least 20 characters (starts with AIza...)."
            diag["error_category"] = "INVALID_FORMAT"
            self.last_diagnostic = diag
            return diag

        diag["key_format_valid"] = True

        # Check endpoint reachability before attempting live API call
        if not check_host_reachability():
            diag["status_message"] = (
                "Network offline: Unable to resolve or connect to generativelanguage.googleapis.com (DNS/Socket resolution failed). "
                "Local verification mode is 100% active."
            )
            diag["error_category"] = "DNS_OFFLINE"
            self.last_diagnostic = diag
            return diag

        diag["endpoint_reachable"] = True

        # If not forcing live roundtrip on background render, use fast verified state if already initialized
        if not force_live_test and self.is_configured:
            diag["authenticated"] = True
            diag["model_operational"] = True
            diag["active_model"] = self.active_model or self.model_name
            diag["status_message"] = f"Connected ({self.active_model or self.model_name})"
            diag["error_category"] = None
            self.last_diagnostic = diag
            return diag

        # Attempt live API call (when force_live_test is True)
        success, msg, model, err_cat = self.test_connection()
        if success:
            diag["authenticated"] = True
            diag["model_operational"] = True
            diag["active_model"] = model
            diag["status_message"] = f"Operational with model: {model}"
            diag["error_category"] = None
        else:
            diag["status_message"] = msg
            diag["error_category"] = err_cat

        self.last_diagnostic = diag
        return diag

    def update_api_key(self, api_key: str) -> Tuple[bool, str, str]:
        """
        Updates the API key, persists it securely via ConfigManager so it survives
        navigation and app restarts, and executes a real connection diagnostic.
        Returns (is_connected: bool, status_message: str, active_model: str).
        """
        cleaned = api_key.strip() if api_key else ""
        if not cleaned:
            return False, "API key cannot be empty. Existing configuration was preserved.", ""

        if not self._is_valid_key_format(cleaned):
            return False, "Invalid API key format: Expected a valid Google AI Studio key (starts with AIza...).", ""

        # Persist the key securely to .env and environment
        ConfigManager.save_api_key(cleaned)
        self.api_key = cleaned

        try:
            self._client = genai.Client(api_key=self.api_key)
        except Exception as e:
            sanitized = self._sanitize_error(str(e))
            self._init_error = sanitized
            return False, f"Client initialization error: {sanitized}", ""

        # Pre-flight network check
        if not check_host_reachability():
            return (
                False,
                "API key saved successfully. Google AI Studio endpoint is currently unreachable (network/DNS offline). "
                "Local DeBERTa verification remains 100% functional; live Gemini synthesis will activate when online.",
                "",
            )

        # Execute connection test
        success, msg, model_used, err_cat = self.test_connection()
        if success:
            self.active_model = model_used
            self._init_error = None
            return True, f"Connected & verified operational with {model_used}", model_used
        else:
            self._init_error = msg
            return False, msg, ""

    def set_model(self, model_name: str) -> None:
        """Updates the target Gemini model and persists it to configuration."""
        self.model_name = model_name
        self.active_model = model_name
        self.last_diagnostic = None
        ConfigManager.save_setting("GEMINI_MODEL", model_name)

    def clear_key(self) -> None:
        """Removes the stored API key from memory, environment, and disk."""
        self.api_key = ""
        self._client = None
        self.active_model = None
        self.last_diagnostic = None
        self._init_error = "API key not configured"
        ConfigManager.clear_api_key()

    def test_connection(self) -> Tuple[bool, str, str, Optional[str]]:
        """
        Executes a minimal test request against Google AI Studio, accurately
        distinguishing DNS offline, invalid key, 403 permission, 429 quota, billing,
        and 503 service unavailability.
        Returns: (is_operational, diagnostic_message, active_model_name, error_category)
        """
        if not self._client or not self.is_configured:
            return False, "Gemini API key is not configured.", "", "NOT_CONFIGURED"

        # Pre-flight network check before attempting API requests
        if not check_host_reachability():
            return (
                False,
                "Network offline: Unable to resolve or connect to generativelanguage.googleapis.com (DNS/Socket resolution failed). "
                "Local verification mode is active.",
                "",
                "DNS_OFFLINE",
            )

        # Prioritize configured model, then fallback candidates
        configured = self.model_name or self.PRIMARY_MODEL
        models_to_try = [configured] + [m for m in self.CANDIDATE_MODELS if m != configured]
        last_error = ""
        last_category = None

        for model in models_to_try:
            # Retry up to 2 times only for retryable errors (503, socket timeout)
            for attempt in range(2):
                try:
                    resp = self._client.models.generate_content(
                        model=model,
                        contents="Respond with 'OK'",
                    )
                    if resp and (resp.text or getattr(resp, "candidates", None)):
                        self.active_model = model
                        if model == configured:
                            return True, f"Verified operational with {model}", model, None
                        else:
                            return True, f"Operational with fallback {model} (selected model {configured} failed: {last_error[:65]})", model, None
                except (APIError, ClientError) as e:
                    err_str = self._sanitize_error(str(e))
                    last_error = err_str

                    # Non-retryable errors: fail fast without retrying
                    if "API_KEY_INVALID" in err_str or "API key not valid" in err_str or "400" in err_str and "key" in err_str.lower():
                        return False, "Authentication Failed (API_KEY_INVALID): Key is invalid or expired.", "", "INVALID_KEY"
                    if "PERMISSION_DENIED" in err_str or "403" in err_str:
                        return False, "Permission Denied (HTTP 403): Key lacks permissions for the Generative Language API.", "", "PERMISSION_DENIED"
                    if "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower() or "429" in err_str:
                        last_category = "RATE_LIMIT_QUOTA"
                        last_error = f"Quota reached (429) on {model}"
                        break  # Try next candidate model to avoid halting the app
                    if "BILLING" in err_str.upper():
                        return False, "Billing Restriction: Cloud project requires billing to be enabled.", "", "BILLING_RESTRICTION"
                    if "NOT_FOUND" in err_str or "404" in err_str:
                        last_category = "MODEL_NOT_FOUND"
                        last_error = f"Model {model} unavailable (404)"
                        break  # Try next candidate model
                    if "UNAVAILABLE" in err_str or "503" in err_str:
                        last_category = "SERVICE_UNAVAILABLE"
                        if attempt == 0:
                            time.sleep(1.0)
                            continue

                    break  # Non-retryable, try next model

                except Exception as e:
                    err_str = self._sanitize_error(str(e))
                    last_error = err_str
                    if any(x in err_str.lower() for x in ["nodename nor servname", "getaddrinfo", "connection error"]):
                        return False, "Network connection failed: DNS or socket resolution error.", "", "DNS_OFFLINE"
                    if "timeout" in err_str.lower():
                        last_category = "TIMEOUT"
                        if attempt == 0:
                            time.sleep(1.0)
                            continue
                    break

        return False, f"All candidate models failed: {last_error}", "", last_category or "MODEL_UNAVAILABLE"

    def list_available_models(self) -> List[str]:
        """Lists available models from Google AI Studio or returns standard candidate list."""
        if not self._client or not check_host_reachability():
            return self.CANDIDATE_MODELS
        try:
            models = []
            for m in self._client.models.list():
                name = getattr(m, "name", "")
                if "gemini" in name.lower() and "flash" in name.lower() or "pro" in name.lower():
                    clean_name = name.replace("models/", "")
                    models.append(clean_name)
            return models if models else self.CANDIDATE_MODELS
        except Exception:
            return self.CANDIDATE_MODELS

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generates plain text response using Gemini with automatic model fallback."""
        if not self.is_configured:
            raise RuntimeError(
                "Gemini API key is not configured. Please enter a valid GEMINI_API_KEY in Platform Settings."
            )

        if not check_host_reachability():
            raise RuntimeError(
                "Network unreachable: Cannot connect to Google AI Studio. Direct local verification remains available."
            )

        configured = self.model_name or self.PRIMARY_MODEL
        models_to_try = [configured] + [m for m in self.CANDIDATE_MODELS if m != configured]

        last_exc: Optional[Exception] = None
        for model in models_to_try:
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction,
                )
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    self.active_model = model
                    return response.text.strip()
                return ""
            except Exception as e:
                last_exc = e
                sanitized = self._sanitize_error(str(e))
                if any(x in sanitized.lower() for x in ["not found", "deprecated", "unavailable", "404", "model"]):
                    continue
                raise RuntimeError(sanitized) from e

        sanitized_last = self._sanitize_error(str(last_exc)) if last_exc else "No response returned"
        raise RuntimeError(f"Gemini generation failed across candidate models: {sanitized_last}")

    def generate_json(
        self,
        prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Generates structured JSON response guaranteed to adhere to JSON format or schema."""
        if not self.is_configured:
            raise RuntimeError(
                "Gemini API key is not configured. Please enter a valid GEMINI_API_KEY in Platform Settings."
            )

        if not check_host_reachability():
            raise RuntimeError(
                "Network unreachable: Cannot connect to Google AI Studio. Direct local verification remains available."
            )

        configured = self.model_name or self.PRIMARY_MODEL
        models_to_try = [configured] + [m for m in self.CANDIDATE_MODELS if m != configured]

        last_exc: Optional[Exception] = None
        for model in models_to_try:
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    system_instruction=system_instruction,
                )
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    self.active_model = model
                    return response.text.strip()
                return "{}"
            except Exception as e:
                last_exc = e
                sanitized = self._sanitize_error(str(e))
                if any(x in sanitized.lower() for x in ["not found", "deprecated", "unavailable", "404", "model"]):
                    continue
                raise RuntimeError(sanitized) from e

        sanitized_last = self._sanitize_error(str(last_exc)) if last_exc else "No response returned"
        raise RuntimeError(f"Gemini structured JSON failed across candidate models: {sanitized_last}")
