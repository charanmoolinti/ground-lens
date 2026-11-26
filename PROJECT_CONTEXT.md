# GROUND LENS — Project Context & System Architecture

**Tagline:** Ground Every Answer. Trust Only the Evidence.  
**System Class:** Standalone AI Claim-Level Hallucination Verification & Factual Intelligence Platform.

---

## 1. Executive Summary & Problem Space

Standard Retrieval-Augmented Generation (RAG) displays LLM-generated answers directly to end-users without checking whether the individual assertions contained within those answers are actually corroborated by the source documentation. 

LLMs routinely interpolate:
* Distorted numerical parameters (e.g., claiming 150 epochs when the paper documented 25).
* Fabricated benchmark scores (e.g., claiming 98.4 BLEU when the paper scored 28.4).
* Inverted polarities (e.g., asserting a system never requires persistent disk storage when it does).
* Speculative future milestones or uncorroborated accolades.

**GROUND LENS** establishes an explicit, independent verification firewall. It decomposes generated answers into atomic, testable factual claims and evaluates each claim independently using a fine-grained pretrained Natural Language Inference (NLI) Cross-Encoder (`cross-encoder/nli-deberta-v3-base`) augmented with rule-based numerical discrepancy analysis.

---

## 2. Core Epistemic Principles

1. **Semantic Similarity $\neq$ NLI Entailment:**  
   High cosine similarity indicates that two text passages discuss the same topical domain (e.g., BLEU scores or training hyperparameters). It does NOT prove that the premise logically entails the claim. Entailment requires semantic compatibility and metric preservation.
2. **Absence of Evidence is Not Contradiction:**  
   If a document does not discuss a claim (e.g. an out-of-domain assertion about healthcare popularity or awards), the claim is classified as `UNVERIFIED` (Neutral). It is never classified as `CONTRADICTED` simply because evidence is missing.
3. **Multi-Passage Conflict Detection:**  
   When multiple candidate passages are retrieved, the engine reconciles evidence across all passages rather than picking the top similarity score blindly. If conflicting evidence exists, it is surfaced to the auditor.
4. **Decoupled Truthful Health Telemetry:**  
   Subsystems report real readiness. When cloud LLMs are unconfigured or offline, local verification remains fully operational.

---

## 3. End-to-End System Pipeline

```
[ User Technical Document (PDF) ]
         │
         ▼
[ PyMuPDF Text & Page Extraction ]
  ├── Empty document detection & 50MB file size limit
  ├── Scanned image-only detection (OCR warning)
  └── Deterministic SHA-256 doc_id generation
         │
         ▼
[ Page-Bounded Word Sliding Chunking ]
  ├── 400 words (~500 tokens), 80 word overlap
  └── Strict page boundary preservation ({doc_name}_p{page}_c{idx})
         │
         ▼
[ Dense Vector Embedding ]
  └── sentence-transformers/all-MiniLM-L6-v2 (384-dimensional dense vectors)
         │
         ▼
[ ChromaDB Persistent Vector Database ]
  └── Persistent SQLite storage, cosine distance metric (1 - distance)
         │
         ├── Workflow A: Direct Statement Verification (100% Local / Offline)
         │       │
         │       ▼ (User enters direct factual claims)
         └── Workflow B: Full RAG Synthesis & Audit (Gemini API)
                 │
                 ▼ (Retrieval -> Grounded Gemini 2.0 Flash -> Structured JSON Decomposition)

[ Atomic Factual Claims (1..N) ]
         │
         ▼ (Per Claim)
[ Independent ChromaDB Evidence Retrieval ] (Top-4 passages per claim)
         │
         ▼
[ NumericalVerifier & DeBERTa-v3 Cross-Encoder Inference ]
  ├── 1. Numerical & Unit Discrepancy Check (Epochs, %, Dates, BLEU, GPUs)
  ├── 2. Polarity & Negation Check (Affirmation vs. Reversal)
  ├── 3. Multi-Window Cross-Encoder Premise Matching
  └── 4. Multi-Passage Conflict Resolution
         │
         ▼
[ Three-Way Classification Synthesis ]
  ├── SUPPORTED    (High entailment & verified numerical consistency)
  ├── CONTRADICTED (Explicit negation, metric mismatch, or cross-encoder contradiction)
  └── UNVERIFIED   (Insufficient, ambiguous, or neutral evidence)
         │
         ▼
[ Visual Verification Report, KPI Dashboard & Multi-Format Audit Export (CSV, JSON, Markdown) ]
```

---

## 4. Subsystems & Module Breakdown

* **`backend/health.py`**: Truthful health auditor evaluating PDF Ingestion, Embeddings, ChromaDB, NLI Cross-Encoder, and Gemini API. Derives overall system operational state.
* **`backend/ingestion/`**:
  * `pdf_loader.py`: High-fidelity PyMuPDF page text extraction, scanned PDF detection, file size validation, and SHA-256 document hashing.
  * `text_chunker.py`: Sliding-window word chunker with metadata (`doc_id`, `doc_name`, `page_number`, `chunk_index`).
  * `document_processor.py`: Ingestion coordinator with deduplication and index state tracking.
* **`backend/embeddings/`**:
  * `embedding_model.py`: Singleton wrapper around `all-MiniLM-L6-v2` with local cache pre-flight detection for instant offline loading.
* **`backend/retrieval/`**:
  * `vector_store.py`: Persistent ChromaDB interface with cosine similarity normalization, metadata filtering, and document inventory management.
* **`backend/generation/`**:
  * `gemini_client.py`: Robust Google GenAI SDK client with socket connectivity diagnostics, candidate model fallback (`gemini-2.0-flash`, `gemini-1.5-flash`), and structured JSON generation.
  * `answer_generator.py`: Context-grounded response synthesis.
* **`backend/verification/`**:
  * `claim_extractor.py`: Structured JSON decomposition with heuristic fallback.
  * `evidence_retriever.py`: Independent vector search per claim.
  * `numerical_verifier.py`: Rule-based verification for numbers, percentages, dates, and polarity negations.
  * `nli_classifier.py`: DeBERTa-v3 cross-encoder inference with verified label mapping (0: contradiction, 1: entailment, 2: neutral).
  * `verification_engine.py`: Full pipeline coordinator, multi-passage conflict resolution, and report export (CSV, JSON, Markdown).
* **`backend/evaluation/`**:
  * `evaluate.py`: Standardized 16-case benchmark measuring Precision, Recall, F1, and Confusion Matrix.
* **`frontend/`**:
  * `styles.py`: Enterprise Cyberpunk Dark CSS design system (#080B12, #101621, #00D9FF, #8B5CF6) with accessible contrast across all controls.
  * `assets.py`: Pure SVG iconography and brand lockups.
  * `components.py`: Dynamic health banner, KPI cards, claim cards with numerical discrepancy callouts, and export actions.
* **`app.py`**: 6-view Streamlit application (Executive Overview, Document Library, Verification Workspace, Audit Ledger, Engine Telemetry, Platform Settings).

---

## 5. Decision Logic & Calibrated Thresholds

* **Support Decision Threshold:** Entailment probability $\ge 0.55$, with no unresolvable numerical mismatch.
* **Contradiction Decision Threshold:** Contradiction probability $\ge 0.55$, or an explicit numerical/percentage/date/negation conflict identified by `NumericalVerifier`.
* **Unverified Fallback:** Any proposition whose evidence yields neutral scores or lacks sufficient candidate evidence.
