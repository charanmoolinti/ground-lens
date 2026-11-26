"""
Module: numerical_verifier.py
Description: Rigorous rule-based and entity-aware numerical verification engine.
Inspects numbers, percentages, benchmark scores, dates, units, and negations between
premise and hypothesis to prevent subtle hallucination of numerical metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class NumericalEntity:
    raw_text: str
    value: float
    unit: str
    entity_type: str  # "percentage", "metric", "count", "date", "duration", "dimension"
    context_words: Set[str] = field(default_factory=set)


@dataclass
class NumericalCheckResult:
    has_conflict: bool
    conflict_type: Optional[str] = None  # "NUMERICAL_MISMATCH", "PERCENTAGE_DISTORTION", "DATE_MISMATCH", "NEGATION_CONFLICT"
    explanation: str = ""
    premise_evidence: Optional[str] = None
    hypothesis_assertion: Optional[str] = None


class NumericalVerifier:
    """Specialized engine for numerical, percentage, date, and negation discrepancy detection."""

    UNIT_PATTERNS = [
        # Percentages: 94.5%, 42.8 percent
        (r'(\d+(?:\.\d+)?)\s*%', 'percentage', '%'),
        (r'(\d+(?:\.\d+)?)\s*(?:percent|pct)\b', 'percentage', '%'),
        # BLEU scores: 28.4 BLEU, BLEU score of 28.4, BLEU of 28.4
        (r'(\d+(?:\.\d+)?)\s*bleu\b', 'metric', 'BLEU'),
        (r'\bbleu\s*(?:score)?\s*(?:of|is|established|reach)?\s*(\d+(?:\.\d+)?)\b', 'metric', 'BLEU'),
        # Accuracy: accuracy of 94.5%, 94.5% accuracy
        (r'\baccuracy\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*%', 'percentage', '%'),
        # Counts with units: 25 epochs, 150 epochs, trained for 25 epochs
        (r'(\d+(?:\.\d+)?)\s*(?:epoch|epochs)\b', 'count', 'epochs'),
        (r'(\d+(?:\.\d+)?)\s*(?:step|steps)\b', 'count', 'steps'),
        (r'(\d+(?:\.\d+)?)\s*(?:gpu|gpus)\b', 'count', 'GPUs'),
        (r'(\d+(?:\.\d+)?)\s*(?:head|heads)\b', 'count', 'heads'),
        (r'(\d+(?:\.\d+)?)\s*(?:layer|layers)\b', 'count', 'layers'),
        (r'(\d+(?:\.\d+)?)\s*(?:token|tokens)\b', 'count', 'tokens'),
        (r'(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)\b', 'duration', 'hours'),
        (r'(\d+(?:\.\d+)?)\s*(?:day|days)\b', 'duration', 'days'),
        (r'(\d+(?:\.\d+)?)\s*(?:millisecond|milliseconds|ms)\b', 'duration', 'ms'),
        # Dimensions: 384 dimensions, dimension of 384, dimension is 384, d_model = 512
        (r'\b(?:dimension|dimensions|dim|dims|d_model|d_k|d_v)\s*(?:of|=|:|is)?\s*(\d+(?:\.\d+)?)\b', 'dimension', 'dimension'),
        (r'(\d+(?:\.\d+)?)\s*(?:dimension|dimensions|dim|dims)\b', 'dimension', 'dimension'),
        # Users / concurrent users
        (r'(\d+(?:\.\d+)?)\s*(?:concurrent\s+users?|users?)\b', 'count', 'users'),
    ]

    MONTHS = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]

    NEGATION_WORDS = {
        "never", "no", "not", "without", "neither", "none", "unable", "cannot", "does not", "did not"
    }

    @classmethod
    def _normalize_word(cls, w: str) -> str:
        w = w.lower()
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        if w.endswith("ing") and len(w) > 5:
            return w[:-3]
        return w

    @classmethod
    def _extract_context(cls, text: str, start: int, end: int, window: int = 50) -> Set[str]:
        w_start = max(0, start - window)
        w_end = min(len(text), end + window)
        raw_words = re.findall(r'\b[a-z0-9_]{2,}\b', text[w_start:w_end].lower())
        return {cls._normalize_word(w) for w in raw_words}

    @classmethod
    def extract_entities(cls, text: str) -> List[NumericalEntity]:
        """Extracts numerical, percentage, date, and metric entities with surrounding context."""
        entities: List[NumericalEntity] = []
        lower = text.lower().replace(",", "")

        for pattern, entity_type, standard_unit in cls.UNIT_PATTERNS:
            for match in re.finditer(pattern, lower):
                try:
                    num_val = float(match.group(1))
                    context = cls._extract_context(lower, match.start(), match.end())
                    entities.append(
                        NumericalEntity(
                            raw_text=match.group(0),
                            value=num_val,
                            unit=standard_unit,
                            entity_type=entity_type,
                            context_words=context,
                        )
                    )
                except (ValueError, IndexError):
                    continue

        return entities

    @classmethod
    def extract_dates(cls, text: str) -> List[str]:
        """Finds month-year or full dates like 'October 14, 2024' or 'January 2022'."""
        dates = []
        lower = text.lower()
        for month in cls.MONTHS:
            pattern = rf'\b{month}\s+(?:\d{{1,2}},?\s+)?\d{{4}}\b'
            for match in re.finditer(pattern, lower):
                dates.append(match.group(0).strip())
        return dates

    @classmethod
    def check_negation_conflict(cls, premise: str, hypothesis: str) -> Optional[NumericalCheckResult]:
        """Detects contradictions where claim negates an explicit fact affirmed in premise."""
        p_lower = premise.lower()
        h_lower = hypothesis.lower()

        # Specific negation phrases in hypothesis
        neg_patterns = [
            r'\bnever\s+\w+',
            r'\bwithout\s+any\s+\w+',
            r'\bdoes\s+not\s+(?:require|have|use|store)',
            r'\bnever\s+stores\b',
        ]
        has_hypo_neg = any(re.search(pat, h_lower) for pat in neg_patterns)
        has_premise_neg = any(re.search(pat, p_lower) for pat in neg_patterns)

        if has_hypo_neg and not has_premise_neg:
            p_words = set(re.findall(r'\b[a-z]{4,}\b', p_lower)) - cls.NEGATION_WORDS
            h_words = set(re.findall(r'\b[a-z]{4,}\b', h_lower)) - cls.NEGATION_WORDS
            overlap = p_words & h_words

            if len(overlap) >= 2:
                overlap_list = sorted(list(overlap))[:3]
                return NumericalCheckResult(
                    has_conflict=True,
                    conflict_type="NEGATION_CONFLICT",
                    explanation=f"Direct polarity negation: Claim asserts a negation regarding '{', '.join(overlap_list)}' while the source premise positively affirms it.",
                    premise_evidence=premise.strip(),
                    hypothesis_assertion=hypothesis.strip(),
                )

        return None

    @classmethod
    def verify_pairs(cls, premise: str, hypothesis: str) -> NumericalCheckResult:
        """
        Cross-checks numerical entities, dates, and percentages between premise and hypothesis.
        Returns a NumericalCheckResult with conflict flag and detailed explanation.
        """
        # 1. Polarity / Negation check
        neg_check = cls.check_negation_conflict(premise, hypothesis)
        if neg_check:
            return neg_check

        # 2. Date discrepancy check
        p_dates = cls.extract_dates(premise)
        h_dates = cls.extract_dates(hypothesis)
        if p_dates and h_dates:
            # If hypothesis has a specific date not present in premise
            for hd in h_dates:
                if not any(hd in pd or pd in hd for pd in p_dates):
                    return NumericalCheckResult(
                        has_conflict=True,
                        conflict_type="DATE_MISMATCH",
                        explanation=f"Temporal discrepancy: Premise documents '{p_dates[0]}', but claim states '{hd}'.",
                        premise_evidence=p_dates[0],
                        hypothesis_assertion=hd,
                    )

        # 3. Numerical & percentage entity comparison
        p_entities = cls.extract_entities(premise)
        h_entities = cls.extract_entities(hypothesis)

        if not h_entities or not p_entities:
            return NumericalCheckResult(has_conflict=False)

        for he in h_entities:
            # Find all premise entities of the same unit
            matching_unit_pes = [pe for pe in p_entities if pe.unit == he.unit]
            if not matching_unit_pes:
                continue

            # Check if ANY premise entity of this unit matches the hypothesis value
            has_matching_value = any(abs(pe.value - he.value) <= 0.01 for pe in matching_unit_pes)
            if has_matching_value:
                # If there's an exact match in the premise, no conflict for this entity
                continue

            # If no premise entity matches the value, check if context overlaps
            for pe in matching_unit_pes:
                context_overlap = pe.context_words & he.context_words
                # High priority units or contextual overlap
                if len(context_overlap) >= 1 or he.unit in ["BLEU", "epochs", "steps", "hours", "GPUs"]:
                    c_type = "PERCENTAGE_DISTORTION" if he.entity_type == "percentage" else "NUMERICAL_MISMATCH"
                    expl = (
                        f"Numerical distortion in {he.unit}: Source passage specifies {pe.value:g} {he.unit}, "
                        f"whereas the claim asserts {he.value:g} {he.unit}."
                    )
                    return NumericalCheckResult(
                        has_conflict=True,
                        conflict_type=c_type,
                        explanation=expl,
                        premise_evidence=f"{pe.value:g} {he.unit}",
                        hypothesis_assertion=f"{he.value:g} {he.unit}",
                    )

        return NumericalCheckResult(has_conflict=False)
