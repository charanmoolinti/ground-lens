"""
Module: components.py
Description: Enterprise UI component renderers for GROUND LENS AI Document Intelligence.
Features:
- Truthful real-time engine health indicators derived directly from HealthChecker
- Zero emojis; all iconography rendered via Lucide-style SVGs from assets.py
- Safe HTML rendering via render_html
- Clear claim verification cards with audit-grade provenance and numerical discrepancy callouts
- Multi-passage conflict indicators
- Practical report export actions (CSV, JSON, Markdown)
- Evaluation benchmark matrix and component telemetry
"""

import re
from typing import List, Dict, Any, Optional
import streamlit as st

from backend.generation.answer_generator import GeneratedAnswer
from backend.health import SystemHealthReport, SubsystemState
from backend.verification.nli_classifier import VerificationStatus
from backend.verification.verification_engine import VerificationReport, VerifiedClaim
from backend.evaluation.evaluate import EvaluationResults
from frontend.assets import LOGO_FULL_SVG, LOGO_COMPACT_SVG, get_brand_lockup, get_svg_icon, render_html


def render_enterprise_header(health_report: Optional[SystemHealthReport] = None, active_view: str = "Workspace"):
    """Renders the top enterprise brand banner with truthful dynamic subsystem health."""
    shield_icon = get_svg_icon("shield", size=14, color="#10B981")
    cpu_icon = get_svg_icon("cpu", size=14, color="#00D9FF")

    if health_report:
        if health_report.can_synthesize_with_gemini and health_report.can_verify_locally:
            pill_class = "operational"
            pill_label = "SYSTEM: FULLY OPERATIONAL"
        elif health_report.can_verify_locally:
            pill_class = "operational"
            pill_label = "LOCAL ENGINE: OPERATIONAL"
        elif health_report.overall_state == SubsystemState.DEGRADED:
            pill_class = "degraded"
            pill_label = "LOCAL ENGINE: READY"
        else:
            pill_class = "error"
            pill_label = "LOCAL CORE: ERROR"
    else:
        pill_class = "degraded"
        pill_label = "ENGINE: INITIALIZING"

    render_html(
        f"""
        <div class="gl-enterprise-header">
            <div style="display: flex; align-items: center; gap: 16px;">
                {get_brand_lockup('header')}
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="gl-health-pill {pill_class}">
                    <span class="gl-pulse-dot {pill_class}"></span>
                    <span>{pill_label}</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #94A3B8; background: #080B12; padding: 6px 12px; border-radius: 8px; border: 1px solid #1E293B; display: flex; align-items: center; gap: 6px;">
                    {cpu_icon} DeBERTa-v3 | all-MiniLM-L6-v2 | ChromaDB
                </div>
            </div>
        </div>
        """
    )


def render_kpi_dashboard(report: VerificationReport):
    """Renders high-contrast KPI cards summarizing claim counts and claim-support ratio."""
    col1, col2, col3, col4, col5 = st.columns(5)

    doc_icon = get_svg_icon("file_text", size=14, color="#00D9FF")
    check_icon = get_svg_icon("check_circle", size=14, color="#10B981")
    cross_icon = get_svg_icon("x_circle", size=14, color="#EF4444")
    help_icon = get_svg_icon("help_circle", size=14, color="#F59E0B")
    ratio_icon = get_svg_icon("telemetry", size=14, color="#8B5CF6")

    with col1:
        render_html(
            f"""
            <div class="gl-kpi-card primary">
                <div class="gl-kpi-label">{doc_icon} Total Claims</div>
                <div class="gl-kpi-val" style="color: #F1F5F9;">{report.total_claims}</div>
            </div>
            """
        )

    with col2:
        render_html(
            f"""
            <div class="gl-kpi-card supported">
                <div class="gl-kpi-label" style="color: #10B981;">{check_icon} Supported</div>
                <div class="gl-kpi-val" style="color: #10B981;">{report.supported_count}</div>
            </div>
            """
        )

    with col3:
        render_html(
            f"""
            <div class="gl-kpi-card contradicted">
                <div class="gl-kpi-label" style="color: #EF4444;">{cross_icon} Contradicted</div>
                <div class="gl-kpi-val" style="color: #EF4444;">{report.contradicted_count}</div>
            </div>
            """
        )

    with col4:
        render_html(
            f"""
            <div class="gl-kpi-card unverified">
                <div class="gl-kpi-label" style="color: #F59E0B;">{help_icon} Unverified</div>
                <div class="gl-kpi-val" style="color: #F59E0B;">{report.unverified_count}</div>
            </div>
            """
        )

    with col5:
        ratio = report.claim_support_ratio
        color = "#10B981" if ratio >= 70 else ("#F59E0B" if ratio >= 40 else "#EF4444")
        render_html(
            f"""
            <div class="gl-kpi-card violet">
                <div class="gl-kpi-label" style="color: #8B5CF6;">{ratio_icon} Support Ratio</div>
                <div class="gl-kpi-val" style="color: {color};">{ratio}%</div>
            </div>
            """
        )


def format_answer_markdown(text: str) -> str:
    """
    Ensures question headings, numbered sections, and citations are
    properly formatted as clean markdown headings and paragraphs with distinct spacing.
    """
    if not text:
        return ""
    # Normalize newline escapes if any
    formatted = text.replace("\\n", "\n")
    # Ensure any markdown heading (e.g., '### 1.', '## 2.', '### Question') has double newlines before it
    formatted = re.sub(r"(?<!\n\n)[\t ]*(#{1,4}\s+)", r"\n\n\1", formatted)
    # If heading question mark is followed on the same line by capital letter answer text, split with double newline
    # e.g., '### 1. What is X? The answer is Y.' -> '### 1. What is X?\n\nThe answer is Y.'
    formatted = re.sub(r"(#{1,4}\s+[^\n\?]+\?)\s+([A-Z])", r"\1\n\n\2", formatted)
    return formatted.strip()


def render_generated_answer(answer: GeneratedAnswer):
    """
    Renders the grounded answer and question in a structured, high-contrast container
    with distinct separation between questions and answers, preserving all markdown typography.
    """
    with st.container(border=True):
        if answer.question and answer.question != "Direct Statement Verification":
            c_q1, c_q2 = st.columns([3, 1])
            with c_q1:
                st.caption("TARGET QUESTION / AUDIT QUERY")
            with c_q2:
                n_pass = len(answer.retrieved_passages) if answer.retrieved_passages else 0
                st.caption(f"CONTEXT: {n_pass} PASSAGES")

            if len(answer.question) > 200 or "\n" in answer.question:
                with st.expander("Inspect Full User Query & Prompt Instructions", expanded=False):
                    st.markdown(answer.question)
                first_line = answer.question.strip().split("\n")[0][:140]
                st.markdown(f"**Query:** *\"{first_line}...\"*")
            else:
                st.markdown(f"**\"{answer.question}\"**")

            st.divider()

        st.markdown("#### Grounded Response & Source Attribution")
        st.markdown(format_answer_markdown(answer.answer_text))


def render_claim_card(claim: VerifiedClaim):
    """
    Renders an individual verified claim with status badge, numerical check,
    provenance, and NLI confidence using 100% Streamlit-native components.
    Guarantees zero raw HTML exposure and high-contrast enterprise styling.
    """
    conf_pct = int(round(claim.confidence * 100)) if claim.confidence is not None else 0
    sim_pct = int(round(claim.similarity_score * 100)) if claim.similarity_score is not None else 0
    doc_display = claim.doc_name if claim.doc_name else "Indexed Docs"
    page_display = f"Page {claim.page_number}" if claim.page_number is not None else "Page N/A"

    with st.container(border=True):
        # 1. Header: Proposition ID & Status Badge
        c_head1, c_head2 = st.columns([2.5, 1.5])
        with c_head1:
            st.caption(f"CLAIM AUDIT PROPOSITION #{claim.claim_id}")
        with c_head2:
            if claim.status == VerificationStatus.SUPPORTED:
                st.badge("SUPPORTED", icon=":material/check_circle:", color="green")
            elif claim.status == VerificationStatus.CONTRADICTED:
                st.badge("CONTRADICTED", icon=":material/cancel:", color="red")
            else:
                st.badge("UNVERIFIED", icon=":material/help:", color="orange")
            if claim.conflict_detected:
                st.badge("CONFLICT", icon=":material/warning:", color="yellow")

        # 2. Claim Statement
        st.markdown(f"**\"{claim.claim_text}\"**")

        # 3. Discrepancy & Conflict Alerts
        if claim.numerical_conflict:
            st.error(f"**Numerical / Factual Discrepancy Detected:** {claim.numerical_conflict}", icon=":material/error:")
        elif claim.conflict_detected:
            st.warning("**Cross-Passage Conflict Detected:** Retrieved passages report conflicting factual evidence for this proposition.", icon=":material/warning:")

        # 4. Streamlit-Native Metadata & Provenance Metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1.4, 0.8])
        with m_col1:
            st.metric(label="NLI Confidence", value=f"{conf_pct}%")
        with m_col2:
            st.metric(label="Retrieval Similarity", value=f"{sim_pct}%")
        with m_col3:
            st.caption("SOURCE DOCUMENT")
            st.markdown(f"**`{doc_display}`**")
        with m_col4:
            st.caption("CITATION")
            st.markdown(f"**`{page_display}`**")

        # 5. Audit Finding
        st.info(f"**Audit Finding:** {claim.explanation}", icon=":material/fact_check:")

        # 6. Evidence Passage Inspection Expander
        if claim.best_passage:
            expander_title = f"Inspect Evidence Passage: {doc_display} [{page_display}]"
            with st.expander(expander_title, expanded=False):
                st.caption(f"COSINE DISTANCE: {1.0 - (claim.similarity_score or 0.0):.3f}")
                st.info(f'"{claim.best_passage}"')
                if claim.nli_scores:
                    col_e, col_c, col_n = st.columns(3)
                    col_e.metric("Entailment", f"{claim.nli_scores.entailment:.3f}")
                    col_c.metric("Contradiction", f"{claim.nli_scores.contradiction:.3f}")
                    col_n.metric("Neutral", f"{claim.nli_scores.neutral:.3f}")


def render_export_buttons(report: VerificationReport):
    """Renders one-click export actions for CSV, JSON, and Markdown audit reports."""
    st.markdown("##### Export Audit Report")
    c_exp1, c_exp2, c_exp3 = st.columns(3)

    with c_exp1:
        st.download_button(
            label="Download CSV Audit",
            data=report.to_csv(),
            file_name=f"ground_lens_audit_{report.total_claims}_claims.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c_exp2:
        st.download_button(
            label="Download JSON Report",
            data=report.to_json(),
            file_name=f"ground_lens_report_{report.total_claims}_claims.json",
            mime="application/json",
            use_container_width=True,
        )

    with c_exp3:
        st.download_button(
            label="Download Markdown Summary",
            data=report.to_markdown(),
            file_name=f"ground_lens_summary_{report.total_claims}_claims.md",
            mime="text/markdown",
            use_container_width=True,
        )


def render_evaluation_view(results: EvaluationResults):
    """Renders benchmark evaluation report and confusion matrix with clean SVG indicators."""
    st.markdown("### DeBERTa-v3 Ground-Truth Telemetry & Confusion Matrix")
    st.markdown(
        "Formal scikit-learn evaluation across 16 curated technical ground-truth benchmarks covering numerical "
        "discrepancies, metric distortions, temporal shifts, entity inversions, and unverified out-of-domain propositions."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{results.accuracy}%")
    c2.metric("Macro Precision", f"{results.precision_macro}%")
    c3.metric("Macro Recall", f"{results.recall_macro}%")
    c4.metric("Macro F1-Score", f"{results.f1_macro}%")

    st.markdown("#### 3 x 3 Confusion Matrix")
    cols = st.columns([1, 2])
    with cols[0]:
        st.write("Target Classes:", results.labels)
        st.table(results.confusion_matrix)

    with cols[1]:
        st.markdown(
            """
            > **Evaluation Protocol & Calibration:**  
            > - **Classification Backbone:** `cross-encoder/nli-deberta-v3-base`  
            > - **Decision Thresholds:** Support >= 0.55 | Contradiction >= 0.55  
            > - **Absence of Evidence:** Strictly classified as UNVERIFIED (never contradicted)  
            > - **Benchmark Size:** 16 verified propositions (4 Supported, 7 Contradicted, 5 Unverified)
            """
        )

    st.markdown("#### Test Case Audit Ledger")
    check_svg = get_svg_icon("check_circle", size=14, color="#10B981")
    cross_svg = get_svg_icon("x_circle", size=14, color="#EF4444")

    for case in results.case_predictions:
        icon_html = check_svg if case["is_correct"] else cross_svg
        title = f"Case #{case['test_id']} [{case['category'].upper()}]: {case['hypothesis'][:65]}..."
        with st.expander(title):
            st.write(f"**Assertion:** {case['hypothesis']}")
            st.write(f"**Ground Truth:** `{case['ground_truth']}` | **Model Prediction:** `{case['predicted']}`")
            st.write(f"**Confidence:** {case['confidence']:.2f}")
            st.json(case["scores"])
