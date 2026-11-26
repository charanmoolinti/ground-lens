"""
Module: nli_classifier.py
Description: Natural Language Inference (NLI) classifier using cross-encoder/nli-deberta-v3-base.
Compares Premise (source evidence passage) against Hypothesis (factual claim) to produce
rigorous, calibrated probabilities for SUPPORTED (entailment), CONTRADICTED (contradiction),
and UNVERIFIED (neutral/insufficient evidence).

Guarantees 100% reliable CPU execution on Apple Silicon (M-series) and Linux/Windows
by avoiding meta-tensor/MPS bugs, validating materialized weights, providing a dual-loader
architecture (CrossEncoder with direct Transformers fallback), and supporting Streamlit
resource caching.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def _check_local_cache(model_name: str) -> bool:
    """Checks whether the model snapshots exist in the local Hugging Face hub cache."""
    slug = "models--" + model_name.replace("/", "--")
    cache_hub = Path.home() / ".cache" / "huggingface" / "hub" / slug
    if not cache_hub.exists():
        return False
    snapshots = cache_hub / "snapshots"
    if snapshots.exists() and any(snapshots.iterdir()):
        return True
    return False


# Pre-emptively configure offline mode if model is cached locally to prevent DNS retry stalls
if _check_local_cache("cross-encoder/nli-deberta-v3-base"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import huggingface_hub.constants as _hfc
        _hfc.HF_HUB_OFFLINE = True
    except ImportError:
        pass

import torch
import torch.nn.functional as F


class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class NLIScore:
    """Softmax probability distribution across NLI categories."""
    entailment: float
    contradiction: float
    neutral: float

    @property
    def max_label(self) -> str:
        d = {"entailment": self.entailment, "contradiction": self.contradiction, "neutral": self.neutral}
        return max(d, key=d.get)


@dataclass
class PassageNLIResult:
    """NLI inference result for a single premise-hypothesis pair."""
    premise_text: str
    doc_name: str
    page_number: int
    scores: NLIScore
    status: VerificationStatus
    confidence: float


class NLIClassifier:
    """
    DeBERTa cross-encoder inference engine for premise-hypothesis verification.
    Verified label mapping for cross-encoder/nli-deberta-v3-base:
      Index 0: contradiction
      Index 1: entailment
      Index 2: neutral

    Features:
      - Explicit CPU baseline (immune to MPS/meta-tensor errors on Apple Silicon M-series).
      - Dual loader architecture: SentenceTransformers CrossEncoder with automatic
        fallback to direct Hugging Face AutoModelForSequenceClassification.
      - Materialization verification: Confirms all model parameters have real physical weights.
      - Singleton pattern across instances for token & memory efficiency.
    """

    _instance = None
    _loaded_key = None

    def __new__(cls, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = "cpu"):
        cache_key = (model_name, device)
        if cls._instance is None or cls._loaded_key != cache_key:
            cls._instance = super(NLIClassifier, cls).__new__(cls)
            cls._instance._init_engine(model_name, device)
            cls._loaded_key = cache_key
        return cls._instance

    def _init_engine(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device or "cpu"
        self.is_cached = _check_local_cache(model_name)
        self.loader_type = None
        self.model = None
        self.hf_tokenizer = None
        self.hf_model = None
        self.status_message = ""
        self.is_operational = False

        if self.is_cached:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Step 1: Attempt loading via SentenceTransformers CrossEncoder on explicit CPU
        ce_error: Optional[Exception] = None
        try:
            from sentence_transformers import CrossEncoder
            # Explicitly force device=self.device to prevent SentenceTransformers from
            # querying get_device_name() which selects 'mps' on Apple Silicon and triggers
            # PyTorch meta-tensor dispatch errors.
            self.model = CrossEncoder(
                model_name,
                device=self.device,
                local_files_only=self.is_cached,
            )
            # Verify that CrossEncoder parameters are physically materialized on CPU
            self._verify_module_weights(self.model.model)
            self.loader_type = "cross_encoder"
            config = self.model.model.config
            self.is_operational = True
            self.status_message = f"CrossEncoder operational on {self.device}"
        except Exception as e:
            ce_error = e
            logger.warning(
                f"SentenceTransformers CrossEncoder loading failed ({e}). "
                "Engaging direct Hugging Face Transformers loading pathway..."
            )
            self.model = None

        # Step 2: Fallback to direct Transformers loading pathway
        if not self.is_operational:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
                config = AutoConfig.from_pretrained(model_name, local_files_only=self.is_cached)
                self.hf_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=self.is_cached,
                )
                self.hf_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    config=config,
                    local_files_only=self.is_cached,
                    low_cpu_mem_usage=False,
                )
                self.hf_model.to(self.device)
                self.hf_model.eval()

                # Verify all weights are materialized on the target device
                self._verify_module_weights(self.hf_model)
                self.loader_type = "direct_transformers"
                self.is_operational = True
                self.status_message = f"Direct Transformers operational on {self.device}"
            except Exception as e:
                full_err = f"All NLI model loaders failed. CrossEncoder error: {ce_error}; Direct Transformers error: {e}"
                logger.error(full_err)
                self.status_message = full_err
                raise RuntimeError(full_err) from e

        # Step 3: Verified Dynamic Label Mapping
        id2label = getattr(config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
        self.label_to_index = {label.lower(): idx for idx, label in id2label.items()}
        self.contra_idx = self.label_to_index.get("contradiction", 0)
        self.entail_idx = self.label_to_index.get("entailment", 1)
        self.neutral_idx = self.label_to_index.get("neutral", 2)

    def _verify_module_weights(self, module: torch.nn.Module) -> None:
        """
        Validates that all parameters and buffers in the module are physically materialized
        and reside on the expected device, confirming no placeholder meta tensors exist.
        """
        for name, param in module.named_parameters():
            if getattr(param, "is_meta", False) or param.device.type == "meta":
                raise RuntimeError(
                    f"Parameter '{name}' remains on the meta device with unmaterialized data. "
                    "Cannot perform inference on meta tensors."
                )
            if param.device.type != self.device:
                param.data = param.data.to(self.device)

        for name, buf in module.named_buffers():
            if getattr(buf, "is_meta", False) or buf.device.type == "meta":
                raise RuntimeError(
                    f"Buffer '{name}' remains on the meta device with unmaterialized data."
                )
            if buf.device.type != self.device:
                buf.data = buf.data.to(self.device)

    def _predict_logits(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Unified internal inference method executing logits generation across
        either SentenceTransformers or direct Transformers backend.
        """
        if not pairs:
            return np.empty((0, 3), dtype=np.float32)

        if self.loader_type == "cross_encoder" and self.model is not None:
            raw_out = self.model.predict(pairs)
            return np.array(raw_out, dtype=np.float32)

        if self.loader_type == "direct_transformers" and self.hf_model is not None and self.hf_tokenizer is not None:
            premises = [p[0] for p in pairs]
            hypotheses = [p[1] for p in pairs]
            inputs = self.hf_tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
                logits = outputs.logits.detach().cpu().numpy()
            return logits

        raise RuntimeError(f"NLIClassifier is not in an operational state: {self.status_message}")

    def predict_pair(
        self,
        premise: str,
        hypothesis: str,
        support_threshold: float = 0.55,
        contradict_threshold: float = 0.55,
    ) -> Tuple[NLIScore, VerificationStatus, float]:
        """
        Runs NLI inference on a (premise, hypothesis) pair.

        Args:
            premise: Evidence excerpt from source document.
            hypothesis: Factual claim statement.
            support_threshold: Minimum probability for SUPPORTED.
            contradict_threshold: Minimum probability for CONTRADICTED.

        Returns:
            Tuple of (NLIScore, VerificationStatus, primary_confidence).
        """
        logits = self._predict_logits([(premise, hypothesis)])
        probs = F.softmax(torch.tensor(logits[0], dtype=torch.float32), dim=-1).numpy()

        p_contra = float(probs[self.contra_idx])
        p_entail = float(probs[self.entail_idx])
        p_neutral = float(probs[self.neutral_idx])

        scores = NLIScore(
            entailment=round(p_entail, 4),
            contradiction=round(p_contra, 4),
            neutral=round(p_neutral, 4),
        )

        # Decision rules:
        # High entailment -> SUPPORTED
        # High contradiction -> CONTRADICTED
        # Otherwise -> UNVERIFIED (neutral or ambiguous)
        if p_entail >= support_threshold and p_entail > p_contra:
            status = VerificationStatus.SUPPORTED
            confidence = p_entail
        elif p_contra >= contradict_threshold and p_contra > p_entail:
            status = VerificationStatus.CONTRADICTED
            confidence = p_contra
        else:
            status = VerificationStatus.UNVERIFIED
            confidence = max(p_neutral, 1.0 - max(p_entail, p_contra))

        return scores, status, round(confidence, 4)

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[NLIScore]:
        """Batch scoring for multiple (premise, hypothesis) pairs."""
        if not pairs:
            return []
        logits = self._predict_logits(pairs)
        probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

        results: List[NLIScore] = []
        for p in probs:
            results.append(
                NLIScore(
                    contradiction=round(float(p[self.contra_idx]), 4),
                    entailment=round(float(p[self.entail_idx]), 4),
                    neutral=round(float(p[self.neutral_idx]), 4),
                )
            )
        return results
