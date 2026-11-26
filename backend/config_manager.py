"""
Module: config_manager.py
Description: Centralized, secure configuration and credential persistence manager for GROUND LENS.
Handles loading and persisting:
- Gemini API Key (masked in UI, persisted in .env, synchronized across Streamlit session state)
- Selected Gemini Model (gemini-2.5-flash, gemini-2.0-flash, gemini-2.5-pro, etc.)
- ChromaDB persistence directory
- NLI & Embedding hyperparameters and decision thresholds
Guarantees credentials are never exposed in logs or plaintext output.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Initial load
load_dotenv(dotenv_path=ENV_PATH)


class ConfigManager:
    """Manages application-level settings, environment variables, and persistent .env storage."""

    DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
    SUPPORTED_GEMINI_MODELS = [
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
    ]

    DEFAULT_SUPPORT_THRESHOLD = 0.55
    DEFAULT_CONTRADICT_THRESHOLD = 0.55
    DEFAULT_TOP_K = 4

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Returns a dictionary of all core platform configuration parameters."""
        s_thresh, c_thresh = cls.get_thresholds()
        return {
            "GEMINI_API_KEY": cls.get_api_key(),
            "GEMINI_MODEL": cls.get_gemini_model(),
            "SUPPORT_THRESHOLD": s_thresh,
            "CONTRADICT_THRESHOLD": c_thresh,
            "TOP_K": cls.get_top_k(),
            "CHROMA_PERSIST_DIRECTORY": cls.get_setting("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
        }

    @classmethod
    def get_api_key(cls) -> str:
        """
        Retrieves the configured Gemini API key in priority order:
        1. os.environ["GEMINI_API_KEY"]
        2. Streamlit secrets (if available)
        3. .env file directly
        Returns clean string, or empty string if placeholder or unconfigured.
        """
        key = os.environ.get("GEMINI_API_KEY", "")

        # Try streamlit secrets if empty
        if not key or key == "your_gemini_api_key_here":
            try:
                import streamlit as st
                key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass

        # Try reading .env directly if still empty
        if not key or key == "your_gemini_api_key_here":
            key = cls._read_env_value("GEMINI_API_KEY")

        if key and key != "your_gemini_api_key_here":
            return key.strip()
        return ""

    @classmethod
    def save_api_key(cls, api_key: str) -> bool:
        """
        Persists the API key securely to the .env file, updates the runtime environment,
        and synchronizes Streamlit session state if active.
        """
        cleaned = api_key.strip() if api_key else ""
        if not cleaned:
            return False

        # Update runtime environment
        os.environ["GEMINI_API_KEY"] = cleaned

        # Update .env file
        cls._write_env_value("GEMINI_API_KEY", cleaned)

        # Synchronize Streamlit session state if available
        try:
            import streamlit as st
            st.session_state["gemini_api_key"] = cleaned
        except Exception:
            pass

        return True

    @classmethod
    def has_api_key(cls) -> bool:
        """Returns True if a non-placeholder API key is configured."""
        return bool(cls.get_api_key())

    @classmethod
    def remove_api_key(cls) -> bool:
        """Removes the API key from environment, resets in .env, and clears Streamlit state."""
        os.environ["GEMINI_API_KEY"] = ""
        cls._write_env_value("GEMINI_API_KEY", "")
        try:
            import streamlit as st
            st.session_state["gemini_api_key"] = ""
        except Exception:
            pass
        return True

    @classmethod
    def clear_api_key(cls) -> bool:
        """Alias for remove_api_key."""
        return cls.remove_api_key()

    @classmethod
    def get_masked_api_key(cls, key: Optional[str] = None) -> str:
        """
        Returns a secure masked representation of the API key, e.g. 'AIzaSy...********...XYZ9'.
        Never returns full secret.
        """
        target_key = key if key is not None else cls.get_api_key()
        if not target_key or target_key == "your_gemini_api_key_here":
            return "Not Configured"
        k = target_key.strip()
        if len(k) < 12:
            return "********"
        return f"{k[:6]}...{'*' * 8}...{k[-4:]}"

    @classmethod
    def get_setting(cls, key: str, default: str = "") -> str:
        """Retrieves a configuration value from env or .env file."""
        val = os.environ.get(key, "")
        if not val:
            val = cls._read_env_value(key)
        return val.strip() if val else default

    @classmethod
    def save_setting(cls, key: str, value: str) -> bool:
        """Saves a setting to environment and persists to .env file."""
        cleaned = str(value).strip()
        os.environ[key] = cleaned
        cls._write_env_value(key, cleaned)
        try:
            import streamlit as st
            st.session_state[key.lower()] = cleaned
        except Exception:
            pass
        return True

    @classmethod
    def get_gemini_model(cls) -> str:
        """Retrieves configured Gemini runtime model."""
        model = os.environ.get("GEMINI_MODEL", "")
        if not model:
            model = cls._read_env_value("GEMINI_MODEL")
        if not model:
            model = cls.DEFAULT_GEMINI_MODEL
        return model.strip()

    @classmethod
    def save_gemini_model(cls, model_name: str) -> bool:
        """Saves selected Gemini model to .env, runtime env, and Streamlit session state."""
        cleaned = model_name.strip()
        if not cleaned:
            return False
        os.environ["GEMINI_MODEL"] = cleaned
        cls._write_env_value("GEMINI_MODEL", cleaned)
        try:
            import streamlit as st
            st.session_state["gemini_model"] = cleaned
        except Exception:
            pass
        return True

    @classmethod
    def get_thresholds(cls) -> Tuple[float, float]:
        """Returns (support_threshold, contradict_threshold)."""
        try:
            s_val = float(os.environ.get("NLI_SUPPORT_THRESHOLD", cls.DEFAULT_SUPPORT_THRESHOLD))
        except ValueError:
            s_val = cls.DEFAULT_SUPPORT_THRESHOLD

        try:
            c_val = float(os.environ.get("NLI_CONTRADICT_THRESHOLD", cls.DEFAULT_CONTRADICT_THRESHOLD))
        except ValueError:
            c_val = cls.DEFAULT_CONTRADICT_THRESHOLD

        return s_val, c_val

    @classmethod
    def save_thresholds(cls, support_threshold: float, contradict_threshold: float) -> bool:
        """Persists NLI decision thresholds."""
        os.environ["NLI_SUPPORT_THRESHOLD"] = str(round(support_threshold, 2))
        os.environ["NLI_CONTRADICT_THRESHOLD"] = str(round(contradict_threshold, 2))
        cls._write_env_value("NLI_SUPPORT_THRESHOLD", str(round(support_threshold, 2)))
        cls._write_env_value("NLI_CONTRADICT_THRESHOLD", str(round(contradict_threshold, 2)))
        return True

    @classmethod
    def get_top_k(cls) -> int:
        """Returns retrieval top-k."""
        try:
            return int(os.environ.get("RETRIEVAL_TOP_K", cls.DEFAULT_TOP_K))
        except ValueError:
            return cls.DEFAULT_TOP_K

    @classmethod
    def save_top_k(cls, top_k: int) -> bool:
        """Persists retrieval top-k."""
        os.environ["RETRIEVAL_TOP_K"] = str(top_k)
        cls._write_env_value("RETRIEVAL_TOP_K", str(top_k))
        return True

    # --------------------------------------------------------------------------
    # Internal .env Helpers
    # --------------------------------------------------------------------------
    @classmethod
    def _read_env_value(cls, key: str) -> str:
        """Reads a specific key's value directly from .env file."""
        if not ENV_PATH.exists():
            return ""
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{key}=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip()
                        return val.strip("\"'")
        except Exception:
            pass
        return ""

    @classmethod
    def _write_env_value(cls, key: str, value: str) -> bool:
        """
        Safely writes or updates a single key in the .env file while preserving
        all comments and other variables.
        """
        lines = []
        key_found = False

        if ENV_PATH.exists():
            try:
                with open(ENV_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped.startswith(f"{key}=") and not stripped.startswith("#"):
                            lines.append(f"{key}={value}\n")
                            key_found = True
                        else:
                            lines.append(line)
            except Exception:
                pass

        if not key_found:
            lines.append(f"{key}={value}\n")

        try:
            with open(ENV_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return True
        except Exception:
            return False
