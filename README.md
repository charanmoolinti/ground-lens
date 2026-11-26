# GROUND LENS — Enterprise AI Document Intelligence & Claim Verification

> **Tagline:** Ground Every Answer. Trust Only the Evidence.  
> **Product Category:** Enterprise Claim-Level Hallucination Detection & Document Intelligence.  
> **Backbone Models:** `cross-encoder/nli-deberta-v3-base` | `sentence-transformers/all-MiniLM-L6-v2` | `ChromaDB` | `Google Gemini 2.0 Flash`

---

## 1. Executive Summary

Standard Retrieval-Augmented Generation (RAG) platforms present synthesized LLM text directly to users without verifying if individual assertions are factually supported by the retrieved document passages. Large Language Models frequently interpolate numbers, dates, technical hyperparameters, or speculative conclusions even when provided accurate context.

**GROUND LENS** introduces an independent, rigorous verification layer between generation and delivery. It extracts individual atomic propositions and verifies each claim against primary document evidence using a fine-grained pretrained Natural Language Inference (NLI) cross-encoder coupled with rule-based numerical discrepancy analysis.

### Core Philosophy
1. **Semantic Similarity $\neq$ NLI Entailment:** A retrieved document passage may share high cosine similarity with a claim because they discuss the same topic (e.g. "training epochs"), yet the passage may directly contradict the claim's metric (e.g. 25 epochs vs. 150 epochs). Similarity retrieves candidate evidence; Cross-Encoder NLI determines logical truth.
2. **Unverified $\neq$ False:** An assertion classified as `UNVERIFIED` indicates that the indexed documents lack sufficient explicit evidence to confirm or disprove the claim. Absence of proof is not proof of contradiction.

---

## 2. Key Architecture & Pipeline

```
[ User Technical PDF Upload ]
        │
        ▼
[ PyMuPDF Text & Page Extraction ] ── (Detects scanned/image-only PDFs, SHA-256 doc_id)
        │
        ▼
[ Page-Bounded Word Chunking ] ── (400 words / 80 overlap, strict page provenance)
        │
        ▼
[ Dense Vector Embedding ] ── (all-MiniLM-L6-v2, 384 dimensions, offline-cached)
        │
        ▼
[ ChromaDB Persistent Store ] ── (Local SQLite Cosine Vector Space)
        │
        ├── Workflow A: Direct Statement Verification (100% Offline / Local)
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
        │
        ├─ 1. Numerical & Unit Discrepancy Check (Epochs, %, Dates, BLEU, GPUs)
        ├─ 2. Polarity & Negation Check (Affirmation vs. Reversal)
        ├─ 3. Multi-Window Cross-Encoder Premise Matching
        ├─ 4. Multi-Passage Conflict Resolution
        │
        ▼
[ Final Three-Way Classification ]
   • SUPPORTED     (Green, High Entailment & Verified Metrics)
   • CONTRADICTED  (Red, Factual Negation or Metric Distortion)
   • UNVERIFIED    (Amber, Insufficient or Out-of-Domain Evidence)
        │
        ▼
[ Visual Audit Ledger, KPI Metrics & Multi-Format Report Export (CSV, JSON, Markdown) ]
```

---

## 3. Subsystems & Health Architecture

GROUND LENS implements truthful, decoupled health diagnostics across 5 core subsystems:

| Subsystem | Underlying Technology | Local / Cloud | Health States |
| :--- | :--- | :--- | :--- |
| **PDF Ingestion** | PyMuPDF (fitz) v1.26+ | Local | `OPERATIONAL`, `ERROR` |
| **Embedding Engine** | `sentence-transformers/all-MiniLM-L6-v2` | Local (384-dim) | `OPERATIONAL`, `ERROR` |
| **Vector Database** | ChromaDB Persistent Client | Local SQLite | `OPERATIONAL`, `DEGRADED` (0 docs), `ERROR` |
| **NLI Cross-Encoder** | `cross-encoder/nli-deberta-v3-base` | Local | `OPERATIONAL`, `ERROR` |
| **Gemini LLM API** | `google-genai` SDK (v1.47+) | Cloud | `OPERATIONAL`, `OFFLINE` (Net), `NOT_CONFIGURED`, `ERROR` |

> [!NOTE]
> When Gemini is unconfigured or offline, **Workflow A: Direct Statement Verification** remains 100% operational using the local DeBERTa NLI cross-encoder and ChromaDB.

---

## 4. Verification Methodology & NLI Label Mapping

### Verified DeBERTa-v3 Label Mapping
For `cross-encoder/nli-deberta-v3-base`:
* **Index 0:** `contradiction`
* **Index 1:** `entailment`
* **Index 2:** `neutral`

### Multi-Passage Evidence & Conflict Detection
If multiple candidate passages are retrieved for a claim:
* If one passage strongly entails the claim while another passage contradicts it, the claim is flagged with an explicit **Cross-Passage Conflict** indicator.
* If no passage contains pertinent information, the claim is strictly marked as `UNVERIFIED`. It is never labeled as contradicted simply because evidence is absent.

### Numerical & Negation Verifier (`numerical_verifier.py`)
In addition to neural cross-encoder inference, a dedicated entity-aware numerical verifier checks:
* **Metrics & Benchmarks:** BLEU scores (`28.4` vs `95.0`), precision/accuracy (`94.5%` vs `75%`).
* **Training Hyperparameters:** Epochs (`25` vs `150`), batch sizes, steps, heads, dimensions.
* **Temporal Discrepancies:** Dates and years (`October 14, 2024` vs `January 2022`).
* **Polarity Reversals:** Explicit negations (`requires local storage` vs `never stores on local disk`).

---

## 5. Installation & Local Setup

### Prerequisites
* macOS, Linux, or Windows with Python 3.9+
* Recommended: PyTorch with MPS (Apple Silicon) or CUDA (NVIDIA) support

### 1. Clone & Virtual Environment Setup
```bash
git clone https://github.com/charanmoolinti/ground-lens.git
cd ground-lens

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to configure your Google AI Studio API key:
```ini
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.0-flash
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
```

### 3. Launch the Application
```bash
./.venv/bin/streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 6. Verification Workflows

### Workflow A: Direct Statement Verification (100% Local)
1. Navigate to **Verification Workspace**.
2. Select **Direct Statement Verification** mode.
3. Enter one or more factual propositions (or click test chips like *Supported Claim*, *Contradicted Claim*, *Incorrect Metric*).
4. Click **Run Verification Analysis**.
5. The local DeBERTa cross-encoder evaluates the propositions against indexed ChromaDB documents without making any external API calls.

### Workflow B: Full RAG Synthesis & Audit (Gemini)
1. Navigate to **Document Library** and upload your PDF (or click **Index Demo PDF** to load the Attention research paper).
2. Navigate to **Verification Workspace** and select **Full RAG Synthesis & Audit (Gemini)**.
3. Enter your research query (e.g. *"What is the training duration and BLEU score of the Transformer?"*).
4. The system synthesizes an answer, uses Gemini structured JSON to isolate atomic factual claims, retrieves independent evidence per claim, and runs DeBERTa NLI verification.
5. Export the audit report via **CSV**, **JSON**, or **Markdown**.

---

## 7. Formal Benchmark Results

Evaluation executed across 16 curated technical ground-truth benchmarks simulating real-world LLM hallucination patterns:

| Target Class | Sample Size | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **SUPPORTED** | 4 | 100.0% | 100.0% | 100.0% |
| **CONTRADICTED** | 7 | 100.0% | 100.0% | 100.0% |
| **UNVERIFIED** | 5 | 100.0% | 100.0% | 100.0% |
| **Macro Average** | **16** | **100.0%** | **100.0%** | **100.0%** |

To run the automated benchmark locally:
```bash
./.venv/bin/python backend/evaluation/evaluate.py
```

---

## 8. Known Limitations & Roadmap

1. **Scanned Image-Only PDFs:** Current ingestion uses PyMuPDF text extraction. Image-only scanned PDFs without embedded text layers trigger an informative warning. Future roadmap includes an optional Tesseract/DocTR OCR fallback.
2. **Cross-Encoder Computational Latency:** Because cross-encoders jointly score Premise and Hypothesis pairs, verifying large answers with 20+ claims against Top-4 passages requires 80 forward passes (~2–4 seconds on CPU). GPU/MPS acceleration is recommended for high-throughput deployments.
3. **Multi-Hop Synthesis:** If evidence for a proposition is split across non-contiguous pages (e.g. entity defined on Page 3, metric on Page 19), vector retrieval must retrieve both chunks to achieve full entailment. Future work includes graph-linked chunk retrieval.
