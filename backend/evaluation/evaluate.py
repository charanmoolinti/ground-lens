"""
Module: evaluate.py
Description: Ground-truth benchmark evaluation framework for Ground Lens.
Evaluates the NLI verification engine against manually curated test cases covering
numerical discrepancies, negations, date errors, and unverified out-of-domain claims.
Computes Precision, Recall, F1-score, Confusion Matrix, and Overall Accuracy via scikit-learn.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from backend.verification.nli_classifier import NLIClassifier, VerificationStatus


@dataclass
class TestCase:
    """Represents a single ground-truth evaluation example."""
    test_id: int
    category: str
    premise: str
    hypothesis: str
    ground_truth: VerificationStatus
    description: str


# Manually curated evaluation benchmark covering realistic failure modes
BENCHMARK_DATASET: List[TestCase] = [
    # 1. Supported: Exact/Paraphrased Entailment
    TestCase(
        test_id=1,
        category="Direct Entailment",
        premise="Ground Lens introduces a claim-level verification layer between answer generation and final presentation.",
        hypothesis="Ground Lens verifies answers at the individual claim level before presentation.",
        ground_truth=VerificationStatus.SUPPORTED,
        description="Paraphrased statement of core architecture.",
    ),
    TestCase(
        test_id=2,
        category="Percentage & Metric Support",
        premise="The DeBERTa-v3 cross-encoder achieved 94.5% precision on the technical document verification benchmark.",
        hypothesis="The verification system achieved 94.5% precision on technical benchmarks.",
        ground_truth=VerificationStatus.SUPPORTED,
        description="Preservation of numerical metric (94.5%).",
    ),
    TestCase(
        test_id=3,
        category="Technical Parameter Support",
        premise="The document chunker uses a window size of 500 tokens with an overlap of 80 tokens.",
        hypothesis="The chunking process applies an 80-token overlap between chunks.",
        ground_truth=VerificationStatus.SUPPORTED,
        description="Specific technical chunking parameter.",
    ),

    # 2. Contradicted: Numerical Discrepancies
    TestCase(
        test_id=4,
        category="Numerical Hallucination",
        premise="The model was trained over 25 epochs using a batch size of 64 on NVIDIA A100 GPUs.",
        hypothesis="The model was trained over 150 epochs using a batch size of 64.",
        ground_truth=VerificationStatus.CONTRADICTED,
        description="LLM hallucinated epoch count (150 vs 25).",
    ),
    TestCase(
        test_id=5,
        category="Percentage Distortion",
        premise="Ground Lens reduced hallucination incidence in generated answers by 42.8%.",
        hypothesis="Ground Lens reduced hallucination incidence by 88.5%.",
        ground_truth=VerificationStatus.CONTRADICTED,
        description="Exaggerated reduction metric (88.5% vs 42.8%).",
    ),
    TestCase(
        test_id=6,
        category="Date Discrepancy",
        premise="The final benchmark experiments were completed on October 14, 2024.",
        hypothesis="The benchmark experiments were completed in January 2022.",
        ground_truth=VerificationStatus.CONTRADICTED,
        description="Temporal contradiction in study completion date.",
    ),

    # 3. Contradicted: Negation & Reversal
    TestCase(
        test_id=7,
        category="Factual Negation",
        premise="ChromaDB requires persistent local disk storage to reload indexed vector collections across sessions.",
        hypothesis="ChromaDB never stores vector collections on local disk.",
        ground_truth=VerificationStatus.CONTRADICTED,
        description="Direct negation of storage persistence requirement.",
    ),
    TestCase(
        test_id=8,
        category="Entity Inversion",
        premise="The Gemini LLM generates the initial answer while the DeBERTa model verifies the individual claims.",
        hypothesis="The DeBERTa model generates the answers and Gemini verifies the claims.",
        ground_truth=VerificationStatus.CONTRADICTED,
        description="Inverted functional roles between generation and verification.",
    ),

    # 4. Unverified: Missing Evidence & Out-of-Domain
    TestCase(
        test_id=9,
        category="Missing Evidence",
        premise="The system extracts text from PDF documents using PyMuPDF and preserves page numbers.",
        hypothesis="The system was awarded the Best Paper Award at NeurIPS 2024.",
        ground_truth=VerificationStatus.UNVERIFIED,
        description="Unrelated external accolade not mentioned in text.",
    ),
    TestCase(
        test_id=10,
        category="Extrapolated Conclusion",
        premise="All evaluation experiments were executed on a dedicated Linux virtual machine with 8GB RAM.",
        hypothesis="The framework is the most popular hallucination detection tool in the healthcare industry.",
        ground_truth=VerificationStatus.UNVERIFIED,
        description="Unsubstantiated domain extrapolation.",
    ),
    TestCase(
        test_id=11,
        category="Ambiguous Specification",
        premise="The pipeline retrieves the top-4 most semantically similar passages from ChromaDB.",
        hypothesis="The retrieval pipeline always takes under 5 milliseconds to respond.",
        ground_truth=VerificationStatus.UNVERIFIED,
        description="Latency claim not discussed in the premise.",
    ),
    TestCase(
        test_id=12,
        category="Non-sequitur",
        premise="Sentences are converted into 384-dimensional dense vectors using all-MiniLM-L6-v2.",
        hypothesis="The developers prefer using Vim over Visual Studio Code.",
        ground_truth=VerificationStatus.UNVERIFIED,
        description="Subjective non-sequitur completely absent from premise.",
    ),
    TestCase(
        test_id=13,
        category="Architectural Metric Support",
        premise="In this work we employ h = 8 parallel attention layers, or heads. For each of these we use d_k = d_v = 64.",
        hypothesis="Each attention head uses a key and value dimension of 64.",
        ground_truth=VerificationStatus.SUPPORTED,
        description="Exact architectural hyperparameter verification.",
    ),
    TestCase(
        test_id=14,
        category="BLEU Metric Hallucination",
        premise="On the WMT 2014 English-to-German translation task, the big transformer model establishes a state-of-the-art BLEU score of 28.4.",
        hypothesis="The transformer model scored 95.0 BLEU on English-to-German translation.",
        ground_truth=VerificationStatus.CONTRADICTED,
        description="Severe numerical benchmark hallucination (95.0 vs 28.4).",
    ),
    TestCase(
        test_id=15,
        category="Hardware Negation",
        premise="We trained the standard model on 8 NVIDIA P100 GPUs.",
        hypothesis="The standard model was trained without any GPU hardware.",
        ground_truth=VerificationStatus.CONTRADICTED,
        description="Negation of explicit hardware training specification.",
    ),
    TestCase(
        test_id=16,
        category="Future Prediction (Unverified)",
        premise="The big models were trained for 300,000 steps (3.5 days).",
        hypothesis="Version 2.0 of the architecture will be released in late 2028.",
        ground_truth=VerificationStatus.UNVERIFIED,
        description="Future roadmap claim not documented in source text.",
    ),
]


@dataclass
class EvaluationResults:
    """Contains formal scikit-learn evaluation metrics."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: np.ndarray
    labels: List[str]
    report_dict: Dict
    test_cases_evaluated: int
    case_predictions: List[Dict]


def run_benchmark_evaluation(
    classifier: Optional[NLIClassifier] = None,
    dataset: Optional[List[TestCase]] = None,
    support_threshold: float = 0.55,
    contradict_threshold: float = 0.55,
) -> EvaluationResults:
    """
    Executes the benchmark evaluation and generates scikit-learn metrics.
    """
    model = classifier or NLIClassifier()
    cases = dataset or BENCHMARK_DATASET

    y_true: List[str] = []
    y_pred: List[str] = []
    case_predictions: List[Dict] = []

    for case in cases:
        scores, pred_status, conf = model.predict_pair(
            premise=case.premise,
            hypothesis=case.hypothesis,
            support_threshold=support_threshold,
            contradict_threshold=contradict_threshold,
        )

        y_true.append(case.ground_truth.value)
        y_pred.append(pred_status.value)

        case_predictions.append({
            "test_id": case.test_id,
            "category": case.category,
            "hypothesis": case.hypothesis,
            "ground_truth": case.ground_truth.value,
            "predicted": pred_status.value,
            "confidence": conf,
            "is_correct": (case.ground_truth.value == pred_status.value),
            "scores": {
                "entailment": scores.entailment,
                "contradiction": scores.contradiction,
                "neutral": scores.neutral,
            },
        })

    labels = [
        VerificationStatus.SUPPORTED.value,
        VerificationStatus.CONTRADICTED.value,
        VerificationStatus.UNVERIFIED.value,
    ]

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return EvaluationResults(
        accuracy=round(acc * 100, 2),
        precision_macro=round(float(report["macro avg"]["precision"]) * 100, 2),
        recall_macro=round(float(report["macro avg"]["recall"]) * 100, 2),
        f1_macro=round(float(report["macro avg"]["f1-score"]) * 100, 2),
        confusion_matrix=cm,
        labels=labels,
        report_dict=report,
        test_cases_evaluated=len(cases),
        case_predictions=case_predictions,
    )


if __name__ == "__main__":
    print("Executing Ground Lens NLI Benchmark Evaluation...")
    results = run_benchmark_evaluation()
    print(f"\n--- GROUND LENS NLI BENCHMARK RESULTS ---")
    print(f"Test Cases: {results.test_cases_evaluated}")
    print(f"Overall Accuracy: {results.accuracy}%")
    print(f"Macro Precision:  {results.precision_macro}%")
    print(f"Macro Recall:     {results.recall_macro}%")
    print(f"Macro F1-Score:   {results.f1_macro}%\n")
    print("Confusion Matrix:")
    print(f"Labels: {results.labels}")
    print(results.confusion_matrix)
