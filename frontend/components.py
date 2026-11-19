"""
Module: components.py
Description: Enterprise UI component renderers for GROUND LENS AI Document Intelligence.
Features:
- Zero emojis; all iconography rendered via Lucide-style SVGs from assets.py
- Safe HTML rendering via render_html to prevent CommonMark indented code-block parsing
- Distinct confidence vs. retrieval similarity telemetry
- Clear claim verification cards with audit-grade evidence provenance
- Evaluation benchmark matrix and component telemetry
"""

from typing import List, Dict, Any, Optional
import streamlit as st

from backend.verification.nli_classifier import VerificationStatus
from backend.verification.verification_engine import VerificationReport, VerifiedClaim
from backend.evaluation.evaluate import EvaluationResults
from frontend.assets import LOGO_FULL_SVG, LOGO_COMPACT_SVG, get_svg_icon, render_html


def render_enterprise_header(active_view: str = "Workspace"):
    """Renders the top enterprise brand banner with status and active view indicator."""
    shield_icon = get_svg_icon("shield", size=14, color="#10B981")
    cpu_icon = get_svg_icon("cpu", size=14, color="#38BDF8")

    render_html(
        f"""
        <div class="gl-enterprise-header">
            <div style="display: flex; align-items: center; gap: 16px;">
                {LOGO_FULL_SVG}
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="gl-status-pill">
                    <span class="gl-status-dot"></span>
                    <span>{shield_icon} ENGINE: VERIFIED ACTIVE</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #94A3B8; background: #0D1420; padding: 6px 12px; border-radius: 8px; border: 1px solid #263449; display: flex; align-items: center; gap: 6px;">
                    {cpu_icon} DeBERTa-v3-base | ChromaDB
                </div>
            </div>
        </div>
        """
    )


def render_kpi_dashboard(report: VerificationReport):
    """Renders high-contrast KPI cards summarizing claim counts and claim-support ratio."""
    col1, col2, col3, col4, col5 = st.columns(5)

    doc_icon = get_svg_icon("file_text", size=14, color="#38BDF8")
    check_icon = get_svg_icon("check_circle", size=14, color="#10B981")
    cross_icon = get_svg_icon("x_circle", size=14, color="#EF4444")
    help_icon = get_svg_icon("help_circle", size=14, color="#F59E0B")
    ratio_icon = get_svg_icon("telemetry", size=14, color="#818CF8")

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
            <div class="gl-kpi-card indigo">
                <div class="gl-kpi-label" style="color: #818CF8;">{ratio_icon} Support Ratio</div>
                <div class="gl-kpi-val" style="color: {color};">{ratio}%</div>
            </div>
            """
        )


def render_claim_card(claim: VerifiedClaim):
    """Renders an individual verified claim with status badge, dual telemetry, and evidence."""
    if claim.status == VerificationStatus.SUPPORTED:
        badge_icon = get_svg_icon("check_circle", size=14, color="#10B981")
        badge_html = f'<span class="gl-badge gl-badge-supported">{badge_icon} Supported</span>'
        border_color = "rgba(16, 185, 129, 0.4)"
    elif claim.status == VerificationStatus.CONTRADICTED:
        badge_icon = get_svg_icon("x_circle", size=14, color="#EF4444")
        badge_html = f'<span class="gl-badge gl-badge-contradicted">{badge_icon} Contradicted</span>'
        border_color = "rgba(239, 68, 68, 0.4)"
    else:
        badge_icon = get_svg_icon("help_circle", size=14, color="#F59E0B")
        badge_html = f'<span class="gl-badge gl-badge-unverified">{badge_icon} Unverified</span>'
        border_color = "rgba(245, 158, 11, 0.4)"

    conf_pct = int(claim.confidence * 100) if claim.confidence else 0
    sim_pct = int(claim.similarity_score * 100) if claim.similarity_score else 0

    render_html(
        f"""
        <div class="gl-card" style="border-left: 4px solid {border_color}; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600; color: #94A3B8;">
                    CLAIM VERIFICATION PROPOSITION #{claim.claim_id}
                </div>
                <div>{badge_html}</div>
            </div>
            <div style="font-size: 1.05rem; font-weight: 500; color: #F1F5F9; line-height: 1.5; margin-bottom: 1rem;">
                "{claim.claim_text}"
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 16px; padding: 10px 14px; background: #0D1420; border-radius: 8px; border: 1px solid #263449; margin-bottom: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                <div>
                    <span style="color: #64748B;">NLI CONFIDENCE:</span>
                    <strong style="color: #38BDF8; margin-left: 6px;">{conf_pct}%</strong>
                </div>
                <div style="color: #263449;">|</div>
                <div>
                    <span style="color: #64748B;">SEMANTIC SIMILARITY:</span>
                    <strong style="color: #CBD5E1; margin-left: 6px;">{sim_pct}%</strong>
                </div>
                <div style="color: #263449;">|</div>
                <div>
                    <span style="color: #64748B;">SOURCE:</span>
                    <span style="color: #94A3B8; margin-left: 6px;">{claim.doc_name or 'Document Collection'} (Page {claim.page_number or 'N/A'})</span>
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
                <strong style="color: #CBD5E1;">Engine Assessment:</strong> {claim.explanation}
            </div>
        </div>
        """
    )

    if claim.best_passage:
        expander_label = f"Evidence Passage: {claim.doc_name or 'Source'} [Page {claim.page_number or 'N/A'}]"
        with st.expander(expander_label):
            render_html(
                f"""
                <div class="gl-evidence-box">
                    <div class="gl-evidence-header">
                        <span>CORROBORATING PASSAGE EXCERPT</span>
                        <span>SEMANTIC DISTANCE: {1.0 - (claim.similarity_score or 0.0):.3f}</span>
                    </div>
                    <div class="gl-evidence-text">
                        "{claim.best_passage}"
                    </div>
                </div>
                """
            )
            if claim.nli_scores:
                col_e, col_c, col_n = st.columns(3)
                col_e.metric("Entailment Score", f"{claim.nli_scores.entailment:.3f}")
                col_c.metric("Contradiction Score", f"{claim.nli_scores.contradiction:.3f}")
                col_n.metric("Neutral Score", f"{claim.nli_scores.neutral:.3f}")


def render_evaluation_view(results: EvaluationResults):
    """Renders benchmark evaluation report and confusion matrix with zero emojis."""
    st.markdown("### NLI Engine Ground-Truth Telemetry & Confusion Matrix")
    st.markdown(
        "Standardized evaluation conducted over 16 curated technical ground-truth benchmarks covering numerical "
        "discrepancies, metric mismatches, temporal contradictions, and unverified out-of-domain propositions."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{results.accuracy}%")
    c2.metric("Macro Precision", f"{results.precision_macro}%")
    c3.metric("Macro Recall", f"{results.recall_macro}%")
    c4.metric("Macro F1-Score", f"{results.f1_macro}%")

    st.markdown("#### 3 x 3 Confusion Matrix")
    cols = st.columns([1, 2])
    with cols[0]:
        st.write("Labels:", results.labels)
        st.table(results.confusion_matrix)

    with cols[1]:
        st.markdown(
            """
            > **Evaluation Protocol & Calibration:**  
            > - **Classification Backbone:** `cross-encoder/nli-deberta-v3-base`  
            > - **Premise Decomposition:** 2–3 sentence sliding window matching  
            > - **Entailment Decision Threshold:** >= 0.55  
            > - **Contradiction Decision Threshold:** >= 0.55  
            > - **Fallback Mode:** Neutral / Unverified
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
