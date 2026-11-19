"""
Module: nli_classifier.py
Description: Natural Language Inference (NLI) classifier using cross-encoder/nli-deberta-v3-base.
Compares Premise (source evidence passage) against Hypothesis (factual claim) to produce
rigorous, calibrated probabilities for SUPPORTED (entailment), CONTRADICTED (contradiction),
and UNVERIFIED (neutral/insufficient evidence).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder


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
    """

    _instance = None
    _model_name = None

    def __new__(cls, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        if cls._instance is None or cls._model_name != model_name:
            cls._instance = super(NLIClassifier, cls).__new__(cls)
            cls._instance._init_model(model_name)
        return cls._instance

    def _init_model(self, model_name: str):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        # Verify and document mapping
        id2label = getattr(self.model.model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
        self.label_to_index = {label.lower(): idx for idx, label in id2label.items()}
        self.contra_idx = self.label_to_index.get("contradiction", 0)
        self.entail_idx = self.label_to_index.get("entailment", 1)
        self.neutral_idx = self.label_to_index.get("neutral", 2)

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
        logits = self.model.predict([(premise, hypothesis)])
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
        logits = self.model.predict(pairs)
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
