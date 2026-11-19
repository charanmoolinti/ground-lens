# GROUND LENS

> **Ground Every Answer. Trust Only the Evidence.**

GROUND LENS is a standalone AI hallucination detection and claim verification platform. It analyzes LLM answers generated from technical PDF documents, decomposes them into discrete factual claims, independently retrieves evidence passages for each claim, and verifies them using a Natural Language Inference (NLI) model.

---

## Key Features

- **Document Ingestion & Chunking:** Page-aware PDF parsing with PyMuPDF and token-level chunking preserving document names and page metadata.
- **Semantic Vector Storage:** ChromaDB integration with `sentence-transformers/all-MiniLM-L6-v2` dense vector embeddings.
- **RAG Answer Generation:** High-fidelity context-grounded responses generated via Google Gemini.
- **Claim Extraction:** Structured JSON extraction decomposing answers into individual testable factual claims.
- **Claim-Level Evidence Retrieval:** Independent premise retrieval from the vector store for every claim (not just relying on the question context).
- **Three-Way NLI Verification:** Cross-encoder DeBERTa classification into **SUPPORTED**, **CONTRADICTED**, and **UNVERIFIED** with calibrated confidence scores.
- **Visual Evidence Matrix:** Dark, futuristic Streamlit interface displaying claim status badges, confidence meters, and exact source text excerpts with page references.
- **Rigorous Evaluation:** Ground-truth testing measuring Precision, Recall, and F1 across multiple failure modes (numeric discrepancies, date errors, negated facts).

---

## Project Structure

```
GROUND-LENS/
├── app.py                      # Main Streamlit dashboard
├── backend/
│   ├── ingestion/              # PDF loading, chunking, and metadata extraction
│   ├── embeddings/             # Sentence Transformers embedding generator
│   ├── retrieval/              # ChromaDB vector store
│   ├── generation/             # Gemini API client & answer generator
│   ├── verification/           # Claim extractor, claim evidence retriever, NLI classifier
│   └── evaluation/             # Ground-truth evaluation framework
├── frontend/                   # UI components and custom CSS styling
├── data/                       # Local uploads and ChromaDB persistent directory
├── tests/                      # Automated unit and integration tests
├── requirements.txt            # Project dependencies
├── .env.example                # Template for environment variables
└── README.md
```

---

## Setup & Quickstart

### 1. Environment Setup
```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your Google Gemini API key:
```bash
cp .env.example .env
# Edit .env and enter your GEMINI_API_KEY
```

### 3. Run Application
```bash
streamlit run app.py
```
