# GROUND LENS — Project Context & Architecture

**Tagline:** Ground Every Answer. Trust Only the Evidence.

## 1. Executive Summary
GROUND LENS is a standalone AI hallucination detection and claim verification platform designed for technical and AI documentation (research papers, technical manuals, specifications). It introduces a rigorous verification layer between LLM answer generation and final delivery by extracting individual factual claims and verifying each claim against retrieved source evidence using Natural Language Inference (NLI).

## 2. Core Problem & Value Proposition
Standard Retrieval-Augmented Generation (RAG) displays LLM-generated answers without verifying whether individual statements in the generated text are genuinely backed by the retrieved context. LLMs frequently interpolate numbers, dates, claims, or unsupported conclusions even when provided accurate context.

GROUND LENS breaks down answers into discrete factual claims and evaluates each claim independently:
- **SUPPORTED (Entailment):** The document evidence explicitly supports the factual claim.
- **CONTRADICTED (Contradiction):** The document evidence explicitly conflicts with the factual claim.
- **UNVERIFIED (Neutral / Insufficient Evidence):** The available document evidence neither proves nor disproves the claim.

## 3. System Architecture Diagram

```
[ User PDF Upload ]
        │
        ▼
[ PyMuPDF Text & Page Extraction ]
        │
        ▼
[ Overlapping Token Chunking (500 tokens / 80 overlap) ]
        │
        ▼
[ Sentence Transformers (all-MiniLM-L6-v2) ]
        │
        ▼
[ ChromaDB Persistent Vector Store ]
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
[ User Question ]                        [ Question Embedding ]
        │                                        │
        ▼                                        ▼
[ Question Context Retrieval ] ◄─────────────────┘
        │
        ▼
[ Gemini LLM Answer Generation ]
        │
        ▼
[ Raw Generated Answer ]
        │
        ▼
[ Gemini Claim Extractor (Structured JSON) ]
        │
        ▼
[ List of Factual Claims ]
        │
        ▼ (Per Claim)
[ Claim Embedding & Evidence Retrieval from ChromaDB ]
        │
        ▼
[ DeBERTa NLI Cross-Encoder Verification ]
   (Premise: Retrieved Source Evidence | Hypothesis: Factual Claim)
        │
        ▼
[ Classification: SUPPORTED / CONTRADICTED / UNVERIFIED ]
        │
        ▼
[ Claim-Support Ratio & Visual Verification Report ]
```

## 4. Key Modules
- **`backend/ingestion/`**: PDF page-by-page extraction, chunking with metadata (page number, file name).
- **`backend/embeddings/`**: `sentence-transformers/all-MiniLM-L6-v2` dense vector representations.
- **`backend/retrieval/`**: ChromaDB vector store for semantic similarity search with persistent storage.
- **`backend/generation/`**: Gemini API integration with context-grounded prompting and structured JSON claim extraction.
- **`backend/verification/`**: Independent claim-level evidence retrieval and DeBERTa cross-encoder NLI inference with calibrated decision thresholds.
- **`backend/evaluation/`**: Ground-truth test suite measuring Precision, Recall, F1, and Confusion Matrix.
- **`frontend/`**: Streamlit dashboard with a dark, futuristic cyberpunk-inspired theme (#080B12 background, #00D9FF electric cyan, #8B5CF6 violet).
