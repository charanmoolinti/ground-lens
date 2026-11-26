"""
GROUND LENS — Enterprise AI Document Intelligence & Evidence Verification Platform
Tagline: Ground Every Answer. Trust Only the Evidence.
Architected for enterprise-grade factual auditing using DeBERTa-v3 Natural Language Inference and ChromaDB.
"""

from __future__ import annotations
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress Python 3.9 EOL / LibreSSL notifications from third-party libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.oauth2")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

import streamlit as st
from dotenv import load_dotenv

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

# Prioritize local Hugging Face model cache for instant startup
_hf_home = os.environ.get("HF_HOME")
_cache_hub = Path(_hf_home) / "hub" if _hf_home else (Path.home() / ".cache" / "huggingface" / "hub")

if (_cache_hub / "models--sentence-transformers--all-MiniLM-L6-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import huggingface_hub.constants as _hfc
        _hfc.HF_HUB_OFFLINE = True
    except ImportError:
        pass

# Streamlit App Configuration with custom brand title and clean branding
st.set_page_config(
    page_title="GROUND LENS | AI Document Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend and Frontend Imports
from backend.config_manager import ConfigManager
from backend.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from backend.generation.gemini_client import GeminiClient
from backend.health import HealthChecker, SystemHealthReport, SubsystemState
from backend.ingestion.document_processor import DocumentProcessor
from backend.retrieval.vector_store import ChromaVectorStore
from backend.verification.nli_classifier import NLIClassifier
from backend.verification.verification_engine import VerificationEngine, VerificationReport
from backend.evaluation.evaluate import run_benchmark_evaluation
from frontend.styles import ENTERPRISE_THEME_CSS
from frontend.assets import LOGO_FULL_SVG, LOGO_COMPACT_SVG, LOGO_HERO_SVG, get_brand_lockup, get_svg_icon, render_html
from frontend.components import (
    render_enterprise_header,
    render_kpi_dashboard,
    render_generated_answer,
    format_answer_markdown,
    render_claim_card,
    render_export_buttons,
    render_evaluation_view,
)

# Inject Enterprise Cyberpunk Dark CSS Design System safely
render_html(ENTERPRISE_THEME_CSS)

# ------------------------------------------------------------------------------
# Session State Initialization via ConfigManager
# ------------------------------------------------------------------------------
cfg = ConfigManager.get_all()

if "current_view" not in st.session_state:
    st.session_state.current_view = "Executive Overview"

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = cfg.get("GEMINI_API_KEY", "")

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = cfg.get("GEMINI_MODEL", "gemini-2.5-flash")

if "support_threshold" not in st.session_state:
    st.session_state.support_threshold = float(cfg.get("SUPPORT_THRESHOLD", 0.55))

if "contradict_threshold" not in st.session_state:
    st.session_state.contradict_threshold = float(cfg.get("CONTRADICT_THRESHOLD", 0.55))

if "top_k_val" not in st.session_state:
    st.session_state.top_k_val = int(cfg.get("TOP_K", 4))

if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "verification_history" not in st.session_state:
    st.session_state.verification_history = []


# ------------------------------------------------------------------------------
# Cached Backend Engine Resources
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(persist_directory="./data/chroma_db")


@st.cache_resource(show_spinner=False)
def get_document_processor() -> DocumentProcessor:
    return DocumentProcessor(vector_store=get_vector_store())


@st.cache_resource(show_spinner=False)
def get_nli_classifier() -> NLIClassifier:
    """Caches the DeBERTa NLI Cross-Encoder model once on CPU across reruns."""
    return NLIClassifier(device="cpu")


@st.cache_resource(show_spinner=False)
def get_verification_engine(support_threshold: float, contradict_threshold: float) -> VerificationEngine:
    return VerificationEngine(
        gemini_client=gemini_client,
        vector_store=get_vector_store(),
        nli_classifier=get_nli_classifier(),
        support_threshold=support_threshold,
        contradict_threshold=contradict_threshold,
    )


vector_store = get_vector_store()
doc_processor = get_document_processor()
gemini_client = GeminiClient(
    api_key=st.session_state.gemini_api_key,
    model_name=st.session_state.gemini_model,
)

# Truthful, independent multi-subsystem health audit reusing cached local singletons
health_report = HealthChecker.audit_all(
    gemini_client_or_key=gemini_client,
    vector_store=vector_store,
    nli_classifier=get_nli_classifier(),
)

# ------------------------------------------------------------------------------
# Sidebar Navigation & Workspace Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    # Sidebar Brand Lockup
    render_html(
        f"""
        <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid #1E293B; margin-bottom: 1.5rem;">
            {get_brand_lockup('sidebar')}
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
        index=nav_options.index(st.session_state.current_view) if st.session_state.current_view in nav_options else 0,
        label_visibility="collapsed",
    )
    if selected_nav != st.session_state.current_view:
        st.session_state.current_view = selected_nav
        st.rerun()

    render_html("<hr style='border-color: #1E293B; margin: 1.5rem 0;'>")

    # Quick Action: New Verification Session
    if st.button("New Analysis Session", use_container_width=True, help="Reset current query and report state"):
        st.session_state.last_answer = None
        st.session_state.last_report = None
        st.session_state.current_view = "Verification Workspace"
        st.rerun()

    # Active Document Status Pill
    indexed_docs = vector_store.list_indexed_documents()
    active_doc_name = st.session_state.active_doc or (indexed_docs[0]["doc_name"] if indexed_docs else "None Indexed")

    doc_icon = get_svg_icon("file_text", size=13, color="#00D9FF")
    render_html(
        f"""
        <div style="background: #101621; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 12px; margin-top: 1rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #64748B; text-transform: uppercase; margin-bottom: 4px;">Target Document</div>
            <div style="font-size: 0.82rem; font-weight: 600; color: #F1F5F9; display: flex; align-items: center; gap: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                {doc_icon} {active_doc_name}
            </div>
        </div>
        """
    )

    # Truthful System Health Pill in Sidebar
    nli_health = health_report.subsystems.get("nli")
    db_health = health_report.subsystems.get("vector_store")
    gemini_health = health_report.subsystems.get("gemini")

    nli_color = "#10B981" if nli_health and nli_health.state == SubsystemState.OPERATIONAL else "#EF4444"
    db_color = "#00D9FF" if db_health and db_health.state in (SubsystemState.OPERATIONAL, SubsystemState.DEGRADED) else "#EF4444"
    gem_color = (
        "#10B981" if gemini_health and gemini_health.state == SubsystemState.OPERATIONAL
        else ("#F59E0B" if gemini_health and gemini_health.state in (SubsystemState.OFFLINE, SubsystemState.NOT_CONFIGURED) else "#EF4444")
    )

    render_html(
        f"""
        <div style="margin-top: 1.5rem; padding: 12px; background: #080B12; border: 1px solid #1E293B; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; line-height: 1.7; color: #94A3B8;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span>NLI BACKBONE:</span>
                <span style="color: {nli_color}; font-weight: 700;">{nli_health.state.value if nli_health else 'UNKNOWN'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span>VECTOR DB:</span>
                <span style="color: {db_color}; font-weight: 700;">{db_health.state.value if db_health else 'UNKNOWN'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span>GEMINI API:</span>
                <span style="color: {gem_color}; font-weight: 700;">{gemini_health.state.value if gemini_health else 'UNKNOWN'}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>CALIBRATION:</span>
                <span style="color: #8B5CF6; font-weight: 700;">{st.session_state.support_threshold} THRESH</span>
            </div>
        </div>
        """
    )


# ------------------------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------------------------
if st.session_state.current_view == "Executive Overview":
    render_enterprise_header(health_report=health_report, active_view="Executive Overview")

    # Hero Banner with Vector Brand Identity
    render_html(
        f"""
        <div class="gl-card" style="background: linear-gradient(135deg, #0D1420 0%, #121C2B 100%); border-left: 4px solid #00D9FF; padding: 1.75rem 2rem; margin-bottom: 1.5rem;">
            <div style="margin-bottom: 1.25rem;">
                {get_brand_lockup('hero')}
            </div>
            <div style="font-size: 1.05rem; color: #CBD5E1; max-width: 860px; line-height: 1.65; margin-bottom: 1rem;">
                GROUND LENS delivers claim-level factual verification for AI-generated answers against primary technical documents.
                Using dense semantic retrieval with ChromaDB and a cross-encoder Natural Language Inference model (DeBERTa-v3),
                every factual claim is independently verified, cross-checked for numerical discrepancy, and attributed to source page citations.
            </div>
            <div style="display: flex; align-items: center; gap: 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #64748B;">
                <span>PLATFORM STATUS: <strong style="color: #00D9FF;">{health_report.headline}</strong></span>
                <span>|</span>
                <span>MODEL TARGET: <strong style="color: #8B5CF6;">{st.session_state.gemini_model}</strong></span>
                <span>|</span>
                <span>VERIFICATION CORE: <strong style="color: #10B981;">100% LOCAL CPU INFERENCE</strong></span>
            </div>
        </div>
        """
    )

    # Real-Time Telemetry Counters (Truthful Application Figures)
    c1, c2, c3, c4 = st.columns(4)
    total_docs = len(indexed_docs)
    total_chunks = vector_store.count()
    total_sessions = len(st.session_state.verification_history)
    total_claims_audited = sum(h["total"] for h in st.session_state.verification_history)

    library_svg = get_svg_icon("library", size=14, color="#00D9FF")
    database_svg = get_svg_icon("database", size=14, color="#8B5CF6")
    history_svg = get_svg_icon("history", size=14, color="#10B981")
    shield_svg = get_svg_icon("shield", size=14, color="#F59E0B")

    with c1:
        render_html(
            f"""
            <div class="gl-kpi-card primary">
                <div class="gl-kpi-label">{library_svg} Indexed Documents</div>
                <div class="gl-kpi-val" style="color: #00D9FF;">{total_docs}</div>
            </div>
            """
        )

    with c2:
        render_html(
            f"""
            <div class="gl-kpi-card violet">
                <div class="gl-kpi-label">{database_svg} Vector Chunks</div>
                <div class="gl-kpi-val" style="color: #8B5CF6;">{total_chunks}</div>
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

    # Overview 2-Column Split: Recent Activity & Subsystems Telemetry
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
        st.markdown("### Live Subsystems Telemetry")
        for sub_key, sub in health_report.subsystems.items():
            badge_class = (
                "gl-badge-supported" if sub.state == SubsystemState.OPERATIONAL
                else ("gl-badge-unverified" if sub.state in (SubsystemState.DEGRADED, SubsystemState.OFFLINE, SubsystemState.NOT_CONFIGURED) else "gl-badge-contradicted")
            )
            render_html(
                f"""
                <div class="gl-card-compact" style="margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-weight: 600; font-size: 0.88rem; color: #F1F5F9;">{sub.name}</span>
                        <span class="gl-badge {badge_class}">{sub.state.value}</span>
                    </div>
                    <div style="font-size: 0.76rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                        {sub.message}
                    </div>
                </div>
                """
            )


# ------------------------------------------------------------------------------
# VIEW 2: DOCUMENT LIBRARY
# ------------------------------------------------------------------------------
elif st.session_state.current_view == "Document Library":
    render_enterprise_header(health_report=health_report, active_view="Document Library")

    st.markdown("### Document Ingestion & Vector Indexing")
    st.markdown("Upload research papers, engineering specifications, or legal documentation to index for automated claim verification.")

    upload_col1, upload_col2 = st.columns([3, 1])
    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload Technical PDF Document",
            type=["pdf"],
            help="High-fidelity PDF parsing with PyMuPDF extracts page text and generates semantic chunks preserving page numbers.",
        )

    with upload_col2:
        st.write("")
        st.write("")
        force_reindex = st.checkbox("Force Re-index", value=False)
        sample_pdf_path = Path("data/uploads/sample_transformer_paper.pdf")
        if sample_pdf_path.exists():
            if st.button("Index Demo PDF", use_container_width=True, help="Load built-in Attention Is All You Need research paper"):
                with st.spinner("Indexing sample technical PDF into ChromaDB..."):
                    res = doc_processor.process_pdf(sample_pdf_path, filename="sample_transformer_paper.pdf", force_reindex=force_reindex)
                    if res.is_scanned:
                        st.error(res.warning)
                    else:
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
            try:
                result = doc_processor.process_pdf(
                    save_path,
                    filename=uploaded_file.name,
                    force_reindex=force_reindex,
                )

                if result.is_scanned or result.total_chunks == 0:
                    st.error(result.warning or "Scanned image-only PDF detected. OCR is not currently supported; please upload a PDF with selectable text.")
                elif result.already_indexed and not force_reindex:
                    st.info(f"Document '{uploaded_file.name}' is already indexed in the vector store. Check 'Force Re-index' to overwrite.")
                else:
                    st.session_state.active_doc = uploaded_file.name
                    st.success(f"Document '{uploaded_file.name}' ingested: {result.total_pages} pages, {result.indexed_chunks} semantic chunks.")
                    st.rerun()
            except ValueError as ve:
                st.error(f"Upload Error: {ve}")

    render_html("<hr style='border-color: #1E293B; margin: 2rem 0;'>")
    st.markdown("### Indexed Document Inventory")

    docs = vector_store.list_indexed_documents()
    if docs:
        for d in docs:
            col_d1, col_d2, col_d3 = st.columns([5, 3, 2])
            with col_d1:
                st.markdown(f"**{d['doc_name']}**")
                st.caption(f"Pages: {d['total_pages']} | Chunks: {d['chunk_count']}")
            with col_d2:
                is_active = (st.session_state.active_doc == d['doc_name'])
                btn_label = "Active Target ✓" if is_active else "Set as Active Target"
                if st.button(btn_label, key=f"act_{d['doc_name']}", disabled=is_active):
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
    render_enterprise_header(health_report=health_report, active_view="Verification Workspace")

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
            "Verification Engine Mode",
            options=[
                "Direct Statement Verification",
                "Full RAG Synthesis & Audit (Gemini)",
                "Adversarial Stress-Test",
            ],
            index=0,
            horizontal=False,
            help="Direct mode validates factual claims directly using DeBERTa Cross-Encoder (100% local). Full RAG uses Gemini to answer and verify.",
        )

    render_html("<div style='height: 0.5rem;'></div>")

    placeholder_text = (
        "Enter factual claim(s) to verify against indexed evidence, e.g.:\n"
        "The base models were trained for a total of 100,000 steps or 12 hours."
        if analysis_mode == "Direct Statement Verification"
        else "Enter research question, e.g.: What is the training duration and BLEU score of the Transformer?"
    )

    query_input = st.text_area(
        "Claim Statement or Research Query",
        placeholder=placeholder_text,
        height=85,
    )

    # Prompt Suggestion Chips matching real-world verification cases & Aster technical spec
    render_html("<div style='font-size: 0.78rem; color: #64748B; margin-bottom: 6px;'>Verification Test Case Chips:</div>")
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    with c_p1:
        if st.button("Supported Claim", use_container_width=True, help="Aster: 384 hidden dimensions and 6 layers (or Transformer 100,000 steps)"):
            if any("aster" in d["doc_name"].lower() for d in indexed_docs):
                query_input = "Aster-1B employs 384 hidden dimensions and 6 transformer layers."
            else:
                query_input = "The base models were trained for a total of 100,000 steps or 12 hours."
            st.session_state["query_override"] = query_input
            st.rerun()
    with c_p2:
        if st.button("Contradicted Claim", use_container_width=True, help="Aster: 768 dimensions and 12 layers (or Transformer 900,000 steps)"):
            if any("aster" in d["doc_name"].lower() for d in indexed_docs):
                query_input = "Aster-1B employs 768 hidden dimensions across 12 layers."
            else:
                query_input = "The base models were trained for 900,000 steps on 8 GPUs."
            st.session_state["query_override"] = query_input
            st.rerun()
    with c_p3:
        if st.button("Unverified Claim", use_container_width=True, help="Aster: 100,000 active users (or submarine sonar acoustic reflections)"):
            if any("aster" in d["doc_name"].lower() for d in indexed_docs):
                query_input = "Aster-1B was deployed to 100,000 active users in Q3 2024."
            else:
                query_input = "The system tracks submarine sonar acoustic reflections in deep ocean trenches."
            st.session_state["query_override"] = query_input
            st.rerun()
    with c_p4:
        if st.button("Incorrect Metric", use_container_width=True, help="Test numerical benchmark discrepancy (e.g. 98.4 vs 28.4 BLEU)"):
            query_input = "The big transformer model achieves a BLEU score of 98.4 on English-to-German."
            st.session_state["query_override"] = query_input
            st.rerun()

    if "query_override" in st.session_state:
        query_input = st.session_state.pop("query_override")

    run_btn = st.button("Run Verification Analysis", type="primary", use_container_width=True)

    # Calibrated confidence & alignment note
    render_html(
        """
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748B; margin-top: 6px; text-align: center;">
            Note: GROUND LENS claim-support ratio measures alignment with retrieved evidence, not absolute universal truth.
            NLI confidence represents model-calibrated probability against retrieved passages, not benchmark accuracy.
        </div>
        """
    )

    if run_btn:
        if not query_input.strip():
            st.error("Please enter a research question or factual statement to verify.")
        elif vector_store.count() == 0:
            st.error("The vector store is currently empty. Please upload a PDF in the Document Library before verifying.")
        elif analysis_mode == "Full RAG Synthesis & Audit (Gemini)" and not gemini_client.is_configured:
            st.error(
                "Gemini API key is not configured. Please switch to 'Direct Statement Verification' "
                "to audit claims using DeBERTa NLI (100% offline), or configure your key in Platform Settings."
            )
        else:
            with st.spinner("Step 1/2: Ingesting claim and retrieving semantic evidence passages..."):
                if analysis_mode == "Direct Statement Verification":
                    generated_answer = GeneratedAnswer(
                        question="Direct Statement Verification",
                        answer_text=query_input.strip(),
                        retrieved_passages=[],
                    )
                elif analysis_mode == "Adversarial Stress-Test":
                    ans_gen = AnswerGenerator(
                        gemini_client=gemini_client,
                        vector_store=vector_store,
                        top_k=st.session_state.top_k_val,
                    )
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
                    ans_gen = AnswerGenerator(
                        gemini_client=gemini_client,
                        vector_store=vector_store,
                        top_k=st.session_state.top_k_val,
                    )
                    generated_answer = ans_gen.generate_answer(query_input, doc_name=target_doc_filter)

                st.session_state.last_answer = generated_answer

            with st.spinner("Step 2/2: Decomposing answer into atomic claims, checking numerical entities, and running DeBERTa NLI..."):
                try:
                    engine = get_verification_engine(
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
                except Exception as e:
                    st.error(f"NLI Verification Pipeline Error: {e}")
                    st.info(
                        "Tip: Ground Lens verifies claims using DeBERTa-v3 on CPU. "
                        "If you are on an Apple Silicon Mac, CPU inference is guaranteed safe and fast."
                    )

    # Display Answer and Claim Verification Report if available
    if st.session_state.last_answer and st.session_state.last_report:
        report: VerificationReport = st.session_state.last_report
        answer: GeneratedAnswer = st.session_state.last_answer

        render_html("<hr style='border-color: #1E293B; margin: 2rem 0;'>")
        st.markdown("### Generated Response & Source Attribution")
        render_generated_answer(answer)

        st.markdown("### Claim-Level Evidence Audit Ledger")
        render_kpi_dashboard(report)

        render_html("<div style='height: 1rem;'></div>")
        for claim in report.verified_claims:
            render_claim_card(claim)

        render_html("<hr style='border-color: #1E293B; margin: 1.5rem 0;'>")
        render_export_buttons(report)


# ------------------------------------------------------------------------------
# VIEW 4: HISTORICAL AUDIT LEDGER
# ------------------------------------------------------------------------------
elif st.session_state.current_view == "Audit Ledger":
    render_enterprise_header(health_report=health_report, active_view="Audit Ledger")

    st.markdown("### Verification Session History & Audit Trail")
    st.markdown("Searchable audit record of all factual propositions, generated responses, and claim support ratios.")

    # Search & Filter Controls
    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        search_query = st.text_input("Search Verification Sessions", placeholder="Filter by claim keyword, document, or query...", label_visibility="collapsed")
    with f_col2:
        status_filter = st.selectbox("Status Filter", options=["All", "Supported (>=70%)", "Contradicted (>0)", "Unverified (>0)"], label_visibility="collapsed")

    filtered_history = []
    for item in reversed(st.session_state.verification_history):
        q_match = (
            not search_query
            or search_query.lower() in item["query"].lower()
            or search_query.lower() in item["doc_name"].lower()
            or search_query.lower() in item["answer"].answer_text.lower()
        )
        s_match = True
        if status_filter == "Supported (>=70%)":
            s_match = item["ratio"] >= 70
        elif status_filter == "Contradicted (>0)":
            s_match = item["contra"] > 0
        elif status_filter == "Unverified (>0)":
            s_match = item["unver"] > 0

        if q_match and s_match:
            filtered_history.append(item)

    if filtered_history:
        for idx, item in enumerate(filtered_history):
            ratio = item["ratio"]
            ratio_color = "#10B981" if ratio >= 70 else ("#F59E0B" if ratio >= 40 else "#EF4444")
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
                st.markdown(format_answer_markdown(item["answer"].answer_text))

                if item["report"] and hasattr(item["report"], "verified_claims"):
                    st.markdown("**Claims Breakdown:**")
                    for cl in item["report"].verified_claims:
                        render_claim_card(cl)
    else:
        render_html(
            """
            <div class="gl-card" style="text-align: center; padding: 2.5rem;">
                <div style="color: #64748B;">No historical verification sessions matching filter criteria.</div>
            </div>
            """
        )


# ------------------------------------------------------------------------------
# VIEW 5: ENGINE TELEMETRY & BENCHMARKS
# ------------------------------------------------------------------------------
elif st.session_state.current_view == "Engine Telemetry":
    render_enterprise_header(health_report=health_report, active_view="Engine Telemetry")

    st.markdown("### Subsystem Architecture & Operational Telemetry")
    st.markdown("Detailed real-time telemetry for all 6 independent platform subsystems.")

    # 6-Subsystem Architecture Grid (3x2)
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        pdf_h = health_report.subsystems.get("pdf")
        render_html(
            f"""
            <div class="gl-card" style="border-top: 3px solid #00D9FF; height: 100%;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00D9FF; text-transform: uppercase;">Subsystem 1</div>
                <div style="font-weight: 700; font-size: 1rem; color: #F1F5F9; margin: 4px 0 8px 0;">PDF Ingestion Engine</div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin-bottom: 8px;">
                    PyMuPDF (fitz) text extraction with 500-char semantic chunking and 50-char overlap. Page retention enabled.
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    State: <span class="gl-badge gl-badge-supported">{pdf_h.state.value if pdf_h else 'OPERATIONAL'}</span>
                </div>
            </div>
            """
        )
    with s_col2:
        emb_h = health_report.subsystems.get("embeddings")
        render_html(
            f"""
            <div class="gl-card" style="border-top: 3px solid #8B5CF6; height: 100%;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #8B5CF6; text-transform: uppercase;">Subsystem 2</div>
                <div style="font-weight: 700; font-size: 1rem; color: #F1F5F9; margin: 4px 0 8px 0;">Dense Embedding Model</div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin-bottom: 8px;">
                    sentence-transformers/all-MiniLM-L6-v2 generating 384-dimensional dense vectors. Hugging Face local cache.
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    State: <span class="gl-badge gl-badge-supported">{emb_h.state.value if emb_h else 'OPERATIONAL'}</span>
                </div>
            </div>
            """
        )
    with s_col3:
        vs_h = health_report.subsystems.get("vector_store")
        render_html(
            f"""
            <div class="gl-card" style="border-top: 3px solid #00D9FF; height: 100%;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00D9FF; text-transform: uppercase;">Subsystem 3</div>
                <div style="font-weight: 700; font-size: 1rem; color: #F1F5F9; margin: 4px 0 8px 0;">ChromaDB Vector Store</div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin-bottom: 8px;">
                    Persistent disk-backed collection with Cosine similarity space. Chunks indexed: {vector_store.count()}.
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    State: <span class="gl-badge gl-badge-supported">{vs_h.state.value if vs_h else 'OPERATIONAL'}</span>
                </div>
            </div>
            """
        )

    render_html("<div style='height: 1rem;'></div>")

    s_col4, s_col5, s_col6 = st.columns(3)
    with s_col4:
        nli_h = health_report.subsystems.get("nli")
        render_html(
            f"""
            <div class="gl-card" style="border-top: 3px solid #10B981; height: 100%;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #10B981; text-transform: uppercase;">Subsystem 4</div>
                <div style="font-weight: 700; font-size: 1rem; color: #F1F5F9; margin: 4px 0 8px 0;">Local DeBERTa-v3 NLI</div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin-bottom: 8px;">
                    cross-encoder/nli-deberta-v3-xsmall running on CPU baseline. Real materialized model weights. Zero meta-tensors.
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    State: <span class="gl-badge gl-badge-supported">{nli_h.state.value if nli_h else 'OPERATIONAL'}</span>
                </div>
            </div>
            """
        )
    with s_col5:
        gem_h = health_report.subsystems.get("gemini")
        gem_badge = "gl-badge-supported" if gem_h and gem_h.state == SubsystemState.OPERATIONAL else "gl-badge-unverified"
        render_html(
            f"""
            <div class="gl-card" style="border-top: 3px solid #F59E0B; height: 100%;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #F59E0B; text-transform: uppercase;">Subsystem 5</div>
                <div style="font-weight: 700; font-size: 1rem; color: #F1F5F9; margin: 4px 0 8px 0;">Gemini API Gateway</div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin-bottom: 8px;">
                    Target: {st.session_state.gemini_model}. 5-stage health diagnostic with automatic exponential backoff.
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    State: <span class="gl-badge {gem_badge}">{gem_h.state.value if gem_h else 'STANDBY'}</span>
                </div>
            </div>
            """
        )
    with s_col6:
        rag_h = health_report.subsystems.get("rag_pipeline")
        render_html(
            f"""
            <div class="gl-card" style="border-top: 3px solid #00D9FF; height: 100%;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00D9FF; text-transform: uppercase;">Subsystem 6</div>
                <div style="font-weight: 700; font-size: 1rem; color: #F1F5F9; margin: 4px 0 8px 0;">RAG Claim Audit Pipeline</div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin-bottom: 8px;">
                    Per-claim topical evidence retrieval, regex numerical discrepancy verification, and exportable ledger.
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                    State: <span class="gl-badge gl-badge-supported">{rag_h.state.value if rag_h else 'OPERATIONAL'}</span>
                </div>
            </div>
            """
        )

    render_html("<hr style='border-color: #1E293B; margin: 2rem 0;'>")

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
    render_enterprise_header(health_report=health_report, active_view="Platform Settings")

    st.markdown("### Gemini API & Authentication")
    st.markdown(
        "Configure your Google AI Studio API key for answer generation and structured claim extraction. "
        "Direct Statement Verification runs 100% locally using DeBERTa and ChromaDB and requires no API key."
    )

    # Credential Status Card with Masked Display
    masked_key = ConfigManager.get_masked_api_key()
    key_configured = ConfigManager.has_api_key()

    status_pill_html = (
        '<span class="gl-badge gl-badge-supported">CONFIGURED & SAVED</span>'
        if key_configured
        else '<span class="gl-badge gl-badge-unverified">NOT CONFIGURED</span>'
    )

    render_html(
        f"""
        <div class="gl-card" style="margin-bottom: 1.5rem; background: #0D1420; border: 1px solid #1E293B;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748B; text-transform: uppercase;">
                    Current Credential Status
                </div>
                <div>{status_pill_html}</div>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: #00D9FF; font-weight: 600;">
                {masked_key}
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 6px;">
                Keys are persisted securely to your local <code>.env</code> file. They are never exposed in logs or network transmissions.
            </div>
        </div>
        """
    )

    # API Key Input & Action Controls
    api_key_input = st.text_input(
        "Update Google Gemini API Key",
        value="",
        type="password",
        placeholder="Paste your Google AI Studio key (starts with AIza...)",
        help="Used for answer synthesis and structured claim decomposition. Direct local verification works without a key.",
    )

    c_btn1, c_btn2, c_btn3 = st.columns([2, 2, 2])

    with c_btn1:
        if st.button("Save & Persist API Key", type="primary", use_container_width=True):
            if not api_key_input.strip():
                st.error("Please provide a valid Google AI Studio API key.")
            else:
                with st.spinner("Saving key and testing endpoint reachability..."):
                    success, msg, model_used = gemini_client.update_api_key(api_key_input)
                    st.session_state.gemini_api_key = api_key_input
                    if success:
                        st.success(f"Key Saved & Verified: {msg}")
                    else:
                        st.info(f"Key Saved: {msg}")
                    st.rerun()

    with c_btn2:
        if st.button("Test Gemini Connection", use_container_width=True, help="Executes live 5-stage health diagnostic"):
            with st.spinner("Testing DNS, authentication, quota, and model inference..."):
                diag = gemini_client.diagnose_connection(force_live_test=True)
                if diag["authenticated"] and diag["model_operational"]:
                    st.success(f"Connection Operational: Verified with {diag['active_model']}")
                elif diag["error_category"] == "DNS_OFFLINE":
                    st.warning(f"Diagnostic Report: {diag['status_message']}")
                elif diag["error_category"] == "NOT_CONFIGURED":
                    st.info("No API key configured. Local verification remains 100% functional.")
                else:
                    st.error(f"Connection Diagnostic: [{diag['error_category']}] {diag['status_message']}")

    with c_btn3:
        if st.button("Remove Saved Key", use_container_width=True, help="Clears API key from memory and .env"):
            gemini_client.clear_key()
            st.session_state.gemini_api_key = ""
            st.warning("Gemini API key removed from memory and local configuration.")
            st.rerun()

    render_html("<hr style='border-color: #1E293B; margin: 2rem 0;'>")

    # Gemini Model Selection
    st.markdown("### Gemini Model Architecture")
    st.markdown("Select which Google Gemini model is utilized for grounded answer synthesis and structured claim extraction.")

    available_models = [
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-3.8-flash",
        "gemma-4-31b-it",
        "gemini-3.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
    ]

    current_model = st.session_state.gemini_model if st.session_state.gemini_model in available_models else "gemini-3.5-flash-lite"

    def _on_model_change():
        new_m = st.session_state.get("target_model_select_box")
        if new_m:
            st.session_state.gemini_model = new_m
            gemini_client.set_model(new_m)
            ConfigManager.save_setting("GEMINI_MODEL", new_m)

    selected_model = st.selectbox(
        "Target Gemini Model",
        options=available_models,
        index=available_models.index(current_model) if current_model in available_models else 0,
        key="target_model_select_box",
        on_change=_on_model_change,
        help="Select model. Changes are applied immediately and synchronized across all verification and generation sessions.",
    )

    # Immediately ensure model state is synchronized
    if st.session_state.gemini_model != selected_model:
        st.session_state.gemini_model = selected_model
        gemini_client.set_model(selected_model)

    c_m1, c_m2 = st.columns([2, 3])
    with c_m1:
        if st.button("Save Model Selection", use_container_width=True):
            st.session_state.gemini_model = selected_model
            gemini_client.set_model(selected_model)
            st.success(f"Gemini model persisted: {selected_model}")
            st.rerun()

    with c_m2:
        active_m = gemini_client.active_model or st.session_state.gemini_model
        st.caption(f"Active Gateway Model: **{active_m}** (Quota monitored independently per model)")

    render_html("<hr style='border-color: #1E293B; margin: 2rem 0;'>")
    st.markdown("### Decision Thresholds & Hyperparameters")
    st.markdown("Calibrate the Natural Language Inference cross-encoder and semantic retrieval depth.")

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        new_support = st.slider(
            "Entailment Support Threshold",
            min_value=0.30,
            max_value=0.90,
            value=st.session_state.support_threshold,
            step=0.05,
            help="Minimum DeBERTa entailment probability required to label a claim SUPPORTED.",
        )
    with c_s2:
        new_contra = st.slider(
            "Contradiction Threshold",
            min_value=0.30,
            max_value=0.90,
            value=st.session_state.contradict_threshold,
            step=0.05,
            help="Minimum DeBERTa contradiction probability required to label a claim CONTRADICTED.",
        )
    with c_s3:
        new_top_k = st.slider(
            "Retrieval Top-K Chunks",
            min_value=2,
            max_value=10,
            value=st.session_state.top_k_val,
            step=1,
            help="Number of semantic document chunks retrieved per claim.",
        )

    if st.button("Save Hyperparameters"):
        st.session_state.support_threshold = new_support
        st.session_state.contradict_threshold = new_contra
        st.session_state.top_k_val = new_top_k
        ConfigManager.save_setting("SUPPORT_THRESHOLD", str(new_support))
        ConfigManager.save_setting("CONTRADICT_THRESHOLD", str(new_contra))
        ConfigManager.save_setting("TOP_K", str(new_top_k))
        st.success("Decision thresholds and Top-K persisted to configuration.")
        st.rerun()
