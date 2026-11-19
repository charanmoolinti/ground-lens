"""
GROUND LENS — Enterprise AI Document Intelligence & Evidence Verification Platform
Tagline: Ground Every Answer. Trust Only the Evidence.
Architected for enterprise-grade factual auditing using DeBERTa-v3 Natural Language Inference and ChromaDB.
"""

from __future__ import annotations
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

# Streamlit App Configuration with custom brand title and clean favicon
st.set_page_config(
    page_title="GROUND LENS | AI Document Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend and Frontend Imports
from backend.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from backend.generation.gemini_client import GeminiClient
from backend.ingestion.document_processor import DocumentProcessor
from backend.retrieval.vector_store import ChromaVectorStore
from backend.verification.verification_engine import VerificationEngine, VerificationReport
from backend.evaluation.evaluate import run_benchmark_evaluation
from frontend.styles import ENTERPRISE_THEME_CSS
from frontend.assets import LOGO_FULL_SVG, LOGO_COMPACT_SVG, get_svg_icon, render_html
from frontend.components import (
    render_enterprise_header,
    render_kpi_dashboard,
    render_claim_card,
    render_evaluation_view,
)

# Inject Enterprise Dark CSS Design System safely
render_html(ENTERPRISE_THEME_CSS)

# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if "current_view" not in st.session_state:
    st.session_state.current_view = "Verification Workspace"

if "gemini_api_key" not in st.session_state:
    cloud_secret = ""
    try:
        cloud_secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    st.session_state.gemini_api_key = cloud_secret or os.getenv("GEMINI_API_KEY", "")

if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "verification_history" not in st.session_state:
    st.session_state.verification_history = []

if "support_threshold" not in st.session_state:
    st.session_state.support_threshold = 0.55

if "contradict_threshold" not in st.session_state:
    st.session_state.contradict_threshold = 0.55

if "top_k_val" not in st.session_state:
    st.session_state.top_k_val = 4


# ------------------------------------------------------------------------------
# Cached Backend Engine Resources
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(persist_directory="./data/chroma_db")


@st.cache_resource(show_spinner=False)
def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor(vector_store=get_vector_store())


vector_store = get_vector_store()
doc_processor = get_document_processor()
gemini_client = GeminiClient(api_key=st.session_state.gemini_api_key)

# ------------------------------------------------------------------------------
# Sidebar Navigation & Workspace Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    # Sidebar Brand Lockup
    render_html(
        f"""
        <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid #263449; margin-bottom: 1.5rem;">
            {LOGO_FULL_SVG}
        </div>
        """
    )

    # Primary Navigation Label
    render_html(
        "<div style=\"font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 0.5rem;\">Platform Navigation</div>"
    )

    nav_options = [
        "Executive Overview",
        "Document Library",
        "Verification Workspace",
        "Audit Ledger",
        "Engine Telemetry",
        "Platform Settings",
    ]

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.current_view) if st.session_state.current_view in nav_options else 2,
        label_visibility="collapsed",
    )
    if selected_nav != st.session_state.current_view:
        st.session_state.current_view = selected_nav
        st.rerun()

    render_html("<hr style='border-color: #263449; margin: 1.5rem 0;'>")

    # Quick Action: New Verification Session
    if st.button("New Analysis Session", use_container_width=True, help="Reset current query and report state"):
        st.session_state.last_answer = None
        st.session_state.last_report = None
        st.session_state.current_view = "Verification Workspace"
        st.rerun()

    # Active Document Status Pill
    indexed_docs = vector_store.list_indexed_documents()
    active_doc_name = st.session_state.active_doc or (indexed_docs[0]["doc_name"] if indexed_docs else "None Indexed")

    doc_icon = get_svg_icon("file_text", size=13, color="#38BDF8")
    render_html(
        f"""
        <div style="background: #111A28; border: 1px solid #263449; border-radius: 8px; padding: 10px 12px; margin-top: 1rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #64748B; text-transform: uppercase; margin-bottom: 4px;">Active Target Document</div>
            <div style="font-size: 0.82rem; font-weight: 600; color: #F1F5F9; display: flex; align-items: center; gap: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                {doc_icon} {active_doc_name}
            </div>
        </div>
        """
    )

    # System Status Indicator
    render_html(
        """
        <div style="margin-top: 2rem; padding: 12px; background: #080B12; border: 1px solid #1E293B; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; line-height: 1.6; color: #94A3B8;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span>NLI BACKBONE:</span>
                <span style="color: #10B981; font-weight: 600;">ACTIVE</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span>VECTOR DB:</span>
                <span style="color: #38BDF8; font-weight: 600;">CHROMADB</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>CALIBRATION:</span>
                <span style="color: #818CF8; font-weight: 600;">0.55 THRESH</span>
            </div>
        </div>
        """
    )


# ------------------------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------------------------
if st.session_state.current_view == "Executive Overview":
    render_enterprise_header("Executive Overview")

    render_html(
        """
        <div class="gl-card" style="background: linear-gradient(135deg, #111A28 0%, #141F2E 100%); border-left: 4px solid #38BDF8;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38BDF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                Enterprise Document Intelligence
            </div>
            <div style="font-size: 1.45rem; font-weight: 700; color: #F1F5F9; margin-bottom: 8px;">
                Ground Every Answer. Trust Only the Evidence.
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; max-width: 820px; line-height: 1.6;">
                GROUND LENS provides claim-level factual verification for AI-generated statements using semantic retrieval
                and Cross-Encoder Natural Language Inference. Every proposition is audited against indexed primary documents.
            </div>
        </div>
        """
    )

    # Real-Time Telemetry Counters
    c1, c2, c3, c4 = st.columns(4)
    total_docs = len(indexed_docs)
    total_chunks = vector_store.count()
    total_sessions = len(st.session_state.verification_history)
    total_claims_audited = sum(h["total"] for h in st.session_state.verification_history)

    library_svg = get_svg_icon("library", size=14, color="#38BDF8")
    database_svg = get_svg_icon("database", size=14, color="#818CF8")
    history_svg = get_svg_icon("history", size=14, color="#10B981")
    shield_svg = get_svg_icon("shield", size=14, color="#F59E0B")

    with c1:
        render_html(
            f"""
            <div class="gl-kpi-card primary">
                <div class="gl-kpi-label">{library_svg} Indexed Documents</div>
                <div class="gl-kpi-val" style="color: #38BDF8;">{total_docs}</div>
            </div>
            """
        )

    with c2:
        render_html(
            f"""
            <div class="gl-kpi-card indigo">
                <div class="gl-kpi-label">{database_svg} Vector Chunks</div>
                <div class="gl-kpi-val" style="color: #818CF8;">{total_chunks}</div>
            </div>
            """
        )

    with c3:
        render_html(
            f"""
            <div class="gl-kpi-card supported">
                <div class="gl-kpi-label">{history_svg} Audit Sessions</div>
                <div class="gl-kpi-val" style="color: #10B981;">{total_sessions}</div>
            </div>
            """
        )

    with c4:
        render_html(
            f"""
            <div class="gl-kpi-card unverified">
                <div class="gl-kpi-label">{shield_svg} Claims Audited</div>
                <div class="gl-kpi-val" style="color: #F59E0B;">{total_claims_audited}</div>
            </div>
            """
        )

    render_html("<div style='height: 1.5rem;'></div>")

    # Overview 2-Column Split: Recent Activity & Pipeline Telemetry
    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.markdown("### Recent Verification Activity")
        if st.session_state.verification_history:
            for item in reversed(st.session_state.verification_history[-4:]):
                ratio = item["ratio"]
                ratio_color = "#10B981" if ratio >= 70 else ("#F59E0B" if ratio >= 40 else "#EF4444")
                render_html(
                    f"""
                    <div class="gl-card" style="padding: 1rem 1.25rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748B;">{item['timestamp']}</span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; color: {ratio_color};">SUPPORT RATIO: {ratio}%</span>
                        </div>
                        <div style="font-size: 0.92rem; font-weight: 600; color: #F1F5F9; margin-bottom: 6px;">
                            "{item['query']}"
                        </div>
                        <div style="font-size: 0.78rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                            Target: {item['doc_name']} | Claims: {item['total']} (Sup: {item['sup']} / Contra: {item['contra']} / Unver: {item['unver']})
                        </div>
                    </div>
                    """
                )
        else:
            render_html(
                """
                <div class="gl-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                    <div style="font-size: 0.95rem; color: #64748B; margin-bottom: 1rem;">No verification sessions executed yet in this session.</div>
                </div>
                """
            )
            if st.button("Start New Analysis in Workspace", type="primary"):
                st.session_state.current_view = "Verification Workspace"
                st.rerun()

    with col_right:
        st.markdown("### Engine Subsystems")
        render_html(
            """
            <div class="gl-card" style="padding: 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #263449;">
                    <div>
                        <div style="font-size: 0.88rem; font-weight: 600; color: #F1F5F9;">NLI Cross-Encoder</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">cross-encoder/nli-deberta-v3-base</div>
                    </div>
                    <span class="gl-badge gl-badge-supported">Active</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #263449;">
                    <div>
                        <div style="font-size: 0.88rem; font-weight: 600; color: #F1F5F9;">Dense Embedding Engine</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">sentence-transformers/all-MiniLM-L6-v2</div>
                    </div>
                    <span class="gl-badge gl-badge-supported">384-Dim</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #263449;">
                    <div>
                        <div style="font-size: 0.88rem; font-weight: 600; color: #F1F5F9;">Vector Store Database</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">ChromaDB Persistent (Cosine)</div>
                    </div>
                    <span class="gl-badge gl-badge-supported">Local SSD</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0;">
                    <div>
                        <div style="font-size: 0.88rem; font-weight: 600; color: #F1F5F9;">Document Ingestion</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">PyMuPDF (fitz) Extractor</div>
                    </div>
                    <span class="gl-badge gl-badge-supported">Active</span>
                </div>
            </div>
            """
        )


# ------------------------------------------------------------------------------
# VIEW 2: DOCUMENT LIBRARY
# ------------------------------------------------------------------------------
elif st.session_state.current_view == "Document Library":
    render_enterprise_header("Document Library")

    st.markdown("### Document Intelligence Ingestion")
    st.markdown("Upload research papers, engineering specifications, or legal contracts to index for automated claim verification.")

    upload_col1, upload_col2 = st.columns([3, 1])
    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload Technical PDF Document",
            type=["pdf"],
            help="High-fidelity PDF parsing with PyMuPDF extracts page text and generates semantic chunks.",
        )

    with upload_col2:
        st.write("")
        st.write("")
        force_reindex = st.checkbox("Force Re-index", value=False)
        sample_pdf_path = Path("data/uploads/sample_transformer_paper.pdf")
        if sample_pdf_path.exists():
            if st.button("Index Demo PDF", use_container_width=True, help="Load built-in Transformer Specification paper"):
                with st.spinner("Indexing sample technical PDF..."):
                    res = doc_processor.process_pdf(sample_pdf_path, filename="sample_transformer_paper.pdf")
                    st.session_state.active_doc = "sample_transformer_paper.pdf"
                    st.success(f"Indexed {res.indexed_chunks} chunks successfully!")
                    st.rerun()

    if uploaded_file is not None:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / uploaded_file.name

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Ingesting '{uploaded_file.name}' with PyMuPDF and embedding into ChromaDB..."):
            result = doc_processor.process_pdf(
                save_path,
                filename=uploaded_file.name,
                force_reindex=force_reindex,
            )

        if result.already_indexed and not force_reindex:
            st.info(f"Document '{uploaded_file.name}' is already indexed ({result.indexed_chunks} chunks). Check 'Force Re-index' to overwrite.")
        else:
            st.session_state.active_doc = uploaded_file.name
            st.success(f"Document '{uploaded_file.name}' ingested: {result.total_pages} pages, {result.indexed_chunks} semantic chunks.")
            st.rerun()

    render_html("<hr style='border-color: #263449; margin: 2rem 0;'>")
    st.markdown("### Indexed Document Inventory")

    docs = vector_store.list_indexed_documents()
    if docs:
        for d in docs:
            col_d1, col_d2, col_d3 = st.columns([5, 3, 2])
            with col_d1:
                st.markdown(f"**{d['doc_name']}**")
                st.caption(f"Pages: {d['total_pages']} | Chunks: {d['chunk_count']}")
            with col_d2:
                if st.button(f"Set as Active Target", key=f"act_{d['doc_name']}"):
                    st.session_state.active_doc = d['doc_name']
                    st.success(f"Active document set to: {d['doc_name']}")
                    st.rerun()
            with col_d3:
                if st.button("Delete Chunks", key=f"del_{d['doc_name']}"):
                    vector_store.delete_document(d['doc_name'])
                    if st.session_state.active_doc == d['doc_name']:
                        st.session_state.active_doc = None
                    st.warning(f"Deleted {d['doc_name']} from vector store.")
                    st.rerun()
    else:
        render_html(
            """
            <div class="gl-card" style="text-align: center; padding: 2rem;">
                <div style="color: #64748B;">No documents indexed in the local vector database yet.</div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 6px;">Upload a PDF or click 'Index Demo PDF' above to get started.</div>
            </div>
            """
        )


# ------------------------------------------------------------------------------
# VIEW 3: EVIDENCE VERIFICATION WORKSPACE
# ------------------------------------------------------------------------------
elif st.session_state.current_view == "Verification Workspace":
    render_enterprise_header("Verification Workspace")

    indexed_docs = vector_store.list_indexed_documents()
    doc_options = ["All Indexed Collections"] + [d["doc_name"] for d in indexed_docs]

    col_sel1, col_sel2 = st.columns([3, 2])
    with col_sel1:
        current_target = st.session_state.active_doc if st.session_state.active_doc in doc_options else doc_options[0]
        selected_doc = st.selectbox(
            "Target Document Scope",
            options=doc_options,
            index=doc_options.index(current_target) if current_target in doc_options else 0,
            help="Limit retrieval context to a specific document or query all indexed knowledge.",
        )
        target_doc_filter = None if selected_doc == "All Indexed Collections" else selected_doc

    with col_sel2:
        analysis_mode = st.radio(
            "Verification Mode",
            options=["Standard Auto RAG", "Adversarial Stress-Test"],
            index=0,
            horizontal=True,
            help="Adversarial mode injects intentional factual contradictions to test the NLI detection pipeline.",
        )

    render_html("<div style='height: 0.5rem;'></div>")

    query_input = st.text_area(
        "Research Question or Proposition Query",
        placeholder="e.g., What is the dimensional size of the Transformer feed-forward sublayer, and how many training steps were executed?",
        height=85,
    )

    # Prompt Suggestion Chips
    render_html("<div style='font-size: 0.78rem; color: #64748B; margin-bottom: 6px;'>Quick Example Prompts:</div>")
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        if st.button("Transformer d_model & Steps", use_container_width=True):
            query_input = "What is the model dimension d_model, inner dimension d_ff, and number of training steps?"
            st.session_state["query_override"] = query_input
            st.rerun()
    with c_p2:
        if st.button("Attention Heads & BLEU", use_container_width=True):
            query_input = "How many attention heads does the base model use, and what is its BLEU score on WMT 2014 English-to-German?"
            st.session_state["query_override"] = query_input
            st.rerun()
    with c_p3:
        if st.button("Optimizer & Warmup Steps", use_container_width=True):
            query_input = "What optimizer was used, what was the beta_2 value, and how many warmup steps were performed?"
            st.session_state["query_override"] = query_input
            st.rerun()

    if "query_override" in st.session_state:
        query_input = st.session_state.pop("query_override")

    run_btn = st.button("Run Verification Analysis", type="primary", use_container_width=True)

    if run_btn:
        if not query_input.strip():
            st.error("Please enter a research question or factual statement to verify.")
        elif vector_store.count() == 0:
            st.error("The vector store is currently empty. Please upload a PDF in the Document Library before verifying.")
        else:
            with st.spinner("Step 1/3: Retrieving semantic evidence passages from ChromaDB..."):
                ans_gen = AnswerGenerator(
                    gemini_client=gemini_client,
                    vector_store=vector_store,
                    top_k=st.session_state.top_k_val,
                )

                if analysis_mode == "Adversarial Stress-Test":
                    # Generate standard answer and inject deliberate metric perturbations
                    base_ans = ans_gen.generate_answer(query_input, doc_name=target_doc_filter)
                    tampered_text = base_ans.answer_text + (
                        "\n\n[Adversarial Statement]: The Transformer base model has a dimension d_model of 4096 "
                        "and was trained for 10,000,000 steps using Adam optimizer with beta_2 of 0.85."
                    )
                    generated_answer = GeneratedAnswer(
                        question=query_input,
                        answer_text=tampered_text,
                        retrieved_passages=base_ans.retrieved_passages,
                    )
                else:
                    generated_answer = ans_gen.generate_answer(query_input, doc_name=target_doc_filter)

                st.session_state.last_answer = generated_answer

            with st.spinner("Step 2/3: Decomposing answer into atomic claims and running DeBERTa NLI..."):
                engine = VerificationEngine(
                    gemini_client=gemini_client,
                    vector_store=vector_store,
                    support_threshold=st.session_state.support_threshold,
                    contradict_threshold=st.session_state.contradict_threshold,
                )
                report = engine.verify(generated_answer, doc_name=target_doc_filter)
                st.session_state.last_report = report

                # Record in session history
                st.session_state.verification_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query_input,
                    "doc_name": target_doc_filter or "All Collections",
                    "total": report.total_claims,
                    "sup": report.supported_count,
                    "contra": report.contradicted_count,
                    "unver": report.unverified_count,
                    "ratio": report.claim_support_ratio,
                    "answer": generated_answer,
                    "report": report,
                })

            st.rerun()

    # Display Answer and Claim Verification Report if available
    if st.session_state.last_answer and st.session_state.last_report:
        report: VerificationReport = st.session_state.last_report
        answer: GeneratedAnswer = st.session_state.last_answer

        render_html("<hr style='border-color: #263449; margin: 2rem 0;'>")
        st.markdown("### Generated Response & Source Attribution")

        render_html(
            f"""
            <div class="gl-card" style="border-left: 4px solid #38BDF8;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #38BDF8; margin-bottom: 8px;">
                    QUESTION: {answer.question}
                </div>
                <div style="font-size: 1.02rem; line-height: 1.7; color: #F1F5F9;">
                    {answer.answer_text}
                </div>
            </div>
            """
        )

        st.markdown("### Claim-Level Evidence Audit Ledger")
        render_kpi_dashboard(report)

        render_html("<div style='height: 1rem;'></div>")
        for claim in report.verified_claims:
            render_claim_card(claim)


# ------------------------------------------------------------------------------
# VIEW 4: HISTORICAL AUDIT LEDGER
# ------------------------------------------------------------------------------
elif st.session_state.current_view == "Audit Ledger":
    render_enterprise_header("Audit Ledger")

    st.markdown("### Verification Session History")
    st.markdown("Immutable record of all verification inquiries, generated responses, and claim support ratios.")

    if st.session_state.verification_history:
        for idx, item in enumerate(reversed(st.session_state.verification_history)):
            ratio = item["ratio"]
            badge_color = "#10B981" if ratio >= 70 else ("#F59E0B" if ratio >= 40 else "#EF4444")
            with st.expander(f"{item['timestamp']} | \"{item['query'][:65]}\"... [Support Ratio: {ratio}%]"):
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.write(f"**Target Document:** `{item['doc_name']}`")
                    st.write(f"**Metrics:** Total Claims: `{item['total']}` | Supported: `{item['sup']}` | Contradicted: `{item['contra']}` | Unverified: `{item['unver']}`")
                with col_h2:
                    if st.button("Reload Report in Workspace", key=f"reload_hist_{idx}"):
                        st.session_state.last_answer = item["answer"]
                        st.session_state.last_report = item["report"]
                        st.session_state.current_view = "Verification Workspace"
                        st.rerun()

                st.markdown("**Generated Response:**")
                st.info(item["answer"].answer_text)
    else:
        render_html(
            """
            <div class="gl-card" style="text-align: center; padding: 2.5rem;">
                <div style="color: #64748B;">No historical verification sessions recorded in this browser session.</div>
            </div>
            """
        )


# ------------------------------------------------------------------------------
# VIEW 5: ENGINE TELEMETRY & BENCHMARKS
# ------------------------------------------------------------------------------
elif st.session_state.current_view == "Engine Telemetry":
    render_enterprise_header("Engine Telemetry")

    st.markdown("### DeBERTa-v3 Ground-Truth Benchmark Evaluation")
    st.markdown("Execute the standardized 16-case ground truth test suite covering numerical contradictions, metric shifts, and unverified out-of-domain propositions.")

    if st.button("Run Ground-Truth Evaluation Benchmark", type="primary"):
        with st.spinner("Running 16 ground-truth inference pairs through DeBERTa-v3 Cross-Encoder..."):
            eval_results = run_benchmark_evaluation(
                support_threshold=st.session_state.support_threshold,
                contradict_threshold=st.session_state.contradict_threshold,
            )
            st.session_state.benchmark_results = eval_results

    if "benchmark_results" in st.session_state:
        render_evaluation_view(st.session_state.benchmark_results)


# ------------------------------------------------------------------------------
# VIEW 6: PLATFORM SETTINGS
# ------------------------------------------------------------------------------
elif st.session_state.current_view == "Platform Settings":
    render_enterprise_header("Platform Settings")

    st.markdown("### Gemini API & Authentication")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Used for answer synthesis and structured claim decomposition. A rule-based fallback executes if omitted.",
    )

    if st.button("Update API Key"):
        st.session_state.gemini_api_key = api_key_input
        gemini_client.update_api_key(api_key_input)
        st.success("API key successfully updated and validated.")

    render_html("<hr style='border-color: #263449; margin: 2rem 0;'>")
    st.markdown("### Decision Thresholds & Hyperparameters")

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        st.session_state.support_threshold = st.slider(
            "Entailment Support Threshold",
            min_value=0.30,
            max_value=0.90,
            value=st.session_state.support_threshold,
            step=0.05,
            help="Minimum DeBERTa entailment probability required to label a claim SUPPORTED.",
        )
    with c_s2:
        st.session_state.contradict_threshold = st.slider(
            "Contradiction Threshold",
            min_value=0.30,
            max_value=0.90,
            value=st.session_state.contradict_threshold,
            step=0.05,
            help="Minimum DeBERTa contradiction probability required to label a claim CONTRADICTED.",
        )
    with c_s3:
        st.session_state.top_k_val = st.slider(
            "Retrieval Top-K Chunks",
            min_value=2,
            max_value=10,
            value=st.session_state.top_k_val,
            step=1,
            help="Number of semantic document chunks retrieved per claim.",
        )
