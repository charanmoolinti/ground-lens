"""
Module: styles.py
Description: Enterprise Cyberpunk Dark Design System for GROUND LENS.
Guarantees 100% accessible contrast across all Streamlit components:
- Base: #080B12 (Deep Near-Black)
- Surfaces: #101621 & #141D2B
- Borders: #1E293B / #263449
- Accents: Electric Cyan #00D9FF, Violet #8B5CF6
- Text: Crisp White #F1F5F9 (Primary), Slate #CBD5E1 (Secondary), Muted #94A3B8
- Semantic: Emerald #10B981 (Supported), Crimson #EF4444 (Contradicted), Amber #F59E0B (Unverified)
- ZERO dark-text-on-dark-background or white-on-white artifacts.
"""

ENTERPRISE_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* =============================================================================
   1. GLOBAL RESET & BASE SURFACE (FORCES DARK BACKGROUND & LIGHT TEXT)
   ============================================================================= */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #F1F5F9 !important;
    background-color: #080B12 !important;
}

/* Force light text across all markdown containers and paragraphs */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] div,
[data-testid="stText"],
div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] span {
    color: #CBD5E1 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* Force high contrast on widget labels */
label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] div {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Header & Toolbar */
header[data-testid="stHeader"] {
    background: rgba(8, 11, 18, 0.92) !important;
    backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid #1E293B !important;
}

/* =============================================================================
   2. SIDEBAR STYLING & HIGH CONTRAST NAVIGATION
   ============================================================================= */
section[data-testid="stSidebar"] {
    background-color: #0D1420 !important;
    border-right: 1px solid #1E293B !important;
    padding-top: 1.5rem !important;
}

/* Sidebar navigation radios */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
    gap: 8px !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    background: #121C2B !important;
    border: 1px solid #263449 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    transition: all 0.18s ease-in-out !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
}

/* Ensure text inside sidebar radio options is bright white */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label span,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label div {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    border-color: #00D9FF !important;
    background: #1E293B !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
    color: #F1F5F9 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
    border-color: #00D9FF !important;
    background: rgba(0, 217, 255, 0.12) !important;
    box-shadow: 0 0 12px rgba(0, 217, 255, 0.2) !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
    color: #00D9FF !important;
    font-weight: 700 !important;
}

/* Main content radio options (Verification Engine Mode) */
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #CBD5E1 !important;
    font-size: 0.9rem !important;
}

div[data-testid="stRadio"] label:has(input:checked) p {
    color: #00D9FF !important;
    font-weight: 600 !important;
}

/* =============================================================================
   3. HIGH-CONTRAST BUTTONS (PREVENTS WHITE-ON-WHITE)
   ============================================================================= */
/* Primary Action Buttons */
button[kind="primary"],
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #00D9FF 0%, #0284C7 100%) !important;
    color: #080B12 !important;
    border: 1px solid #38BDF8 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(0, 217, 255, 0.3) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #38BDF8 0%, #0369A1 100%) !important;
    color: #080B12 !important;
    box-shadow: 0 6px 20px rgba(0, 217, 255, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Secondary & Standard Buttons (Chips, Nav, Action items) */
button[kind="secondary"],
.stButton > button,
div[data-testid="stButton"] > button {
    background-color: #141D2B !important;
    color: #F1F5F9 !important;
    border: 1px solid #263449 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.18s ease-in-out !important;
    cursor: pointer !important;
}

button[kind="secondary"]:hover,
.stButton > button:hover {
    background-color: #1E293B !important;
    border-color: #00D9FF !important;
    color: #00D9FF !important;
    box-shadow: 0 0 10px rgba(0, 217, 255, 0.25) !important;
}

/* Disabled buttons */
button:disabled,
.stButton > button:disabled {
    background-color: #0D1420 !important;
    color: #475569 !important;
    border-color: #1E293B !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}

/* Download Buttons */
div[data-testid="stDownloadButton"] > button {
    background-color: #141D2B !important;
    color: #F1F5F9 !important;
    border: 1px solid #263449 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.18s ease-in-out !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    border-color: #10B981 !important;
    color: #10B981 !important;
    background-color: #1E293B !important;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.25) !important;
}

/* =============================================================================
   4. FORM INPUTS, TEXTAREAS & SELECTBOXES
   ============================================================================= */
input[type="text"],
input[type="password"],
textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background-color: #121C2B !important;
    color: #F1F5F9 !important;
    border: 1px solid #263449 !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
}

input:focus, textarea:focus,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #00D9FF !important;
    box-shadow: 0 0 10px rgba(0, 217, 255, 0.3) !important;
    outline: none !important;
}

/* Placeholder Contrast */
input::placeholder, textarea::placeholder {
    color: #64748B !important;
    font-style: italic !important;
}

/* Selectbox Dropdowns */
div[data-testid="stSelectbox"] > div > div {
    background-color: #121C2B !important;
    color: #F1F5F9 !important;
    border: 1px solid #263449 !important;
    border-radius: 8px !important;
}

div[data-baseweb="popover"] ul {
    background-color: #121C2B !important;
    border: 1px solid #263449 !important;
}

div[data-baseweb="popover"] li {
    color: #F1F5F9 !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #172334 !important;
    color: #00D9FF !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #121C2B !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    margin-bottom: 0.75rem !important;
}

div[data-testid="stExpander"] summary span {
    color: #F1F5F9 !important;
    font-weight: 600 !important;
}

/* Streamlit Container (Border=True) */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #101621 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.5rem !important;
    margin-bottom: 1.25rem !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35) !important;
}

/* Streamlit Native Metric Cards */
div[data-testid="stMetric"] {
    background-color: #080B12 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}

div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] span {
    color: #94A3B8 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 2px !important;
}

div[data-testid="stMetricValue"] div {
    color: #00D9FF !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}

/* Sliders */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #00D9FF !important;
    box-shadow: 0 0 8px #00D9FF !important;
}

/* =============================================================================
   5. CARDS, SURFACES & CONTAINERS
   ============================================================================= */
.gl-card {
    background: #121C2B;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.gl-card:hover {
    border-color: rgba(0, 217, 255, 0.4);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.45);
}

.gl-card-secondary {
    background: #172334;
    border: 1px solid #263449;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.gl-card-compact {
    background: #101621;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
}

/* Enterprise Header Bar */
.gl-enterprise-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem 1.75rem;
    background: #101621;
    border: 1px solid #1E293B;
    border-radius: 12px;
    margin-bottom: 1.75rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

/* Brand Lockup & SVG Logo Dimensions */
.gl-brand-lockup {
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
}

.gl-logo-full {
    width: 340px !important;
    height: 52px !important;
    display: block !important;
    flex-shrink: 0 !important;
}

.gl-logo-hero {
    width: 440px !important;
    height: 64px !important;
    display: block !important;
    flex-shrink: 0 !important;
}

.gl-logo-compact {
    width: 44px !important;
    height: 44px !important;
    display: block !important;
    flex-shrink: 0 !important;
}

/* Subsystem Health Pill */
.gl-health-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.gl-health-pill.operational {
    background: rgba(16, 185, 129, 0.12);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.gl-health-pill.degraded {
    background: rgba(245, 158, 11, 0.12);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.gl-health-pill.error {
    background: rgba(239, 68, 68, 0.12);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.gl-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.gl-pulse-dot.operational {
    background-color: #10B981;
    box-shadow: 0 0 8px #10B981;
}

.gl-pulse-dot.degraded {
    background-color: #F59E0B;
    box-shadow: 0 0 8px #F59E0B;
}

.gl-pulse-dot.error {
    background-color: #EF4444;
    box-shadow: 0 0 8px #EF4444;
}

/* KPI Metric Cards */
.gl-kpi-card {
    background: #101621;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: left;
    position: relative;
    overflow: hidden;
}

.gl-kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
}

.gl-kpi-card.supported::before { background-color: #10B981; }
.gl-kpi-card.contradicted::before { background-color: #EF4444; }
.gl-kpi-card.unverified::before { background-color: #F59E0B; }
.gl-kpi-card.primary::before { background-color: #00D9FF; }
.gl-kpi-card.violet::before { background-color: #8B5CF6; }

.gl-kpi-label {
    font-size: 0.74rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

.gl-kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1;
}

/* Status Badges */
.gl-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.gl-badge-supported {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.gl-badge-contradicted {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.gl-badge-unverified {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

/* Evidence Box */
.gl-evidence-box {
    background: #080B12;
    border: 1px solid #1E293B;
    border-left: 3px solid #00D9FF;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 0.85rem;
}

.gl-evidence-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00D9FF;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
}

.gl-evidence-text {
    font-size: 0.9rem;
    color: #CBD5E1;
    line-height: 1.6;
    font-style: italic;
}

/* Numerical Conflict Callout */
.gl-conflict-alert {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    font-size: 0.85rem;
    color: #FCA5A5;
}
</style>
"""
