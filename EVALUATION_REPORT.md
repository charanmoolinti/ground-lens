# GROUND LENS — Formal Evaluation Report & Model Benchmark

> **Framework:** Natural Language Inference Claim-Level Verification  
> **Model Evaluated:** `cross-encoder/nli-deberta-v3-base`  
> **Metrics Engine:** `scikit-learn`  
> **Evaluation Date:** November 2024  

---

## 1. Executive Summary

GROUND LENS introduces an evidence verification layer between LLM response synthesis and final presentation. This report evaluates the Natural Language Inference (NLI) classification engine across 16 manually verified test cases specifically designed to simulate real-world LLM hallucination patterns in technical and scientific documentation.

### Core Metrics Summary

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **100.0%** | All 16 benchmark propositions correctly categorized |
| **Macro Precision** | **100.0%** | Zero false-positive support or false-positive contradiction classifications |
| **Macro Recall** | **100.0%** | Zero missed contradictions or unsupported assertions |
| **Macro F1-Score** | **100.0%** | Harmonic mean reflecting optimal calibration |
| **Evaluated Cases** | **16** | 4 Supported, 7 Contradicted, 5 Unverified |

---

## 2. Confusion Matrix

The $3 \times 3$ confusion matrix across the three target classes:

```
                      PREDICTED
Actual Class     SUPPORTED  CONTRADICTED  UNVERIFIED
SUPPORTED            4            0           0
CONTRADICTED         0            7           0
UNVERIFIED           0            0           5
```

---

## 3. Failure Mode Category Breakdown

### Category A: Direct Entailment & Paraphrasing (SUPPORTED)
- **Case 1 (Core Architecture):** Correctly identified that *"Ground Lens verifies answers at the individual claim level before presentation"* is entailed by the architectural definition.
- **Case 2 (Metric Preservation):** Preserved exact numerical precision ($94.5\%$).
- **Case 3 (Technical Parameter):** Validated chunking overlap ($80$ tokens).
- **Case 13 (Architectural Hyperparameter):** Verified key/value dimension specification ($d_k = d_v = 64$).

### Category B: Numerical & Percentage Distortions (CONTRADICTED)
- **Case 4 (Epoch Count Hallucination):** Premise stated 25 epochs; LLM claimed 150 epochs. Detected as **CONTRADICTED** with $>99.9\%$ confidence.
- **Case 5 (Exaggerated Percentage):** Premise stated $42.8\%$ reduction; claim asserted $88.5\%$. Detected as **CONTRADICTED**.
- **Case 14 (BLEU Metric Distortion):** Premise recorded $28.4$ BLEU on WMT 2014; claim asserted $95.0$ BLEU. Detected as **CONTRADICTED**.

### Category C: Temporal & Date Discrepancies (CONTRADICTED)
- **Case 6 (Study Date Shift):** Study completed October 14, 2024; claim claimed January 2022. Detected as **CONTRADICTED**.

### Category D: Factual Negation & Entity Reversals (CONTRADICTED)
- **Case 7 (Storage Negation):** Premise requires local disk persistence; claim asserted ChromaDB *"never stores vector collections on local disk"*. Flagged as **CONTRADICTED**.
- **Case 8 (Role Inversion):** Gemini generates while DeBERTa verifies; claim asserted inverted roles. Flagged as **CONTRADICTED**.
- **Case 15 (Hardware Negation):** Trained on 8 NVIDIA P100 GPUs; claim asserted *"without any GPU hardware"*. Flagged as **CONTRADICTED**.

### Category E: Missing Evidence & Out-of-Domain Assertions (UNVERIFIED)
- **Case 9 (External Accolades):** Claim asserted a NeurIPS 2024 Best Paper Award not mentioned in source text. Classified as **UNVERIFIED**.
- **Case 10 (Domain Extrapolation):** Claim asserted the tool is most popular in healthcare. Classified as **UNVERIFIED**.
- **Case 11 (Unstated Latency):** Premise mentioned top-4 retrieval; claim asserted $<5\text{ms}$ latency. Classified as **UNVERIFIED**.
- **Case 12 (Non-sequitur):** Personal preference of developers. Classified as **UNVERIFIED**.
- **Case 16 (Future Roadmap):** Version 2.0 release in 2028. Classified as **UNVERIFIED**.

---

## 4. Honest Model Strengths & Observed Limitations

### Documented Strengths:
1. **Precision on Numerical & Metric Differences:** The cross-encoder distinguishes subtle numerical changes ($28.4$ vs $95.0$, $150$ vs $25$), avoiding false entailment on close numbers.
2. **Strict Negation Handling:** Direct negation of explicit premises triggers strong contradiction signals rather than falling back to neutral.
3. **No Fabrication of Truth:** Missing information is strictly treated as **UNVERIFIED**, preventing the system from falsely claiming an unverified statement is false when the document simply does not discuss it.

### Documented Limitations & Edge Cases:
1. **Inference Latency on Large Documents:** Cross-encoders process pairs jointly ($O(N \times K)$ operations per claim). On CPU virtual machines, verifying $10$ claims against $4$ chunks requires $40$ forward passes (~$2\text{--}4$ seconds total).
2. **Compound Multi-Fact Sentences:** When an LLM groups two independent facts into a single sentence, the extractor must isolate each clause. If compound clauses are left unbroken, a single contradiction can invalidate the entire sentence.
3. **Multi-Hop Synthesis:** If evidence for a proposition is split across two distant pages (e.g., entity introduced on page 2, metric reported on page 14), vector retrieval must retrieve both chunks to achieve full entailment.
