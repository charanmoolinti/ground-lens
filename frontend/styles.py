"""
Module: styles.py
Description: Enterprise dark design system for GROUND LENS AI Document Intelligence Platform.
Palette:
- Main background: #080B12
- Sidebar: #0D1420
- Secondary surface: #111A28
- Card surface: #141F2E
- Primary accent: #38BDF8 (Electric Sky Blue)
- Secondary accent: #818CF8 (Indigo)
- Primary text: #F1F5F9
- Secondary text: #94A3B8
- Supported (Entailed): #10B981 (Emerald)
- Contradicted: #EF4444 (Crimson)
- Unverified (Neutral): #F59E0B (Amber)
- Borders: #263449
"""

ENTERPRISE_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Global Reset & Base */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #F1F5F9 !important;
    background-color: #080B12 !important;
}

/* Streamlit Header & Toolbar cleanup */
header[data-testid="stHeader"] {
    background: rgba(8, 11, 18, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid #263449 !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0D1420 !important;
    border-right: 1px solid #263449 !important;
    padding-top: 1.5rem !important;
}

section[data-testid="stSidebar"] div.stRadio > div {
    gap: 6px;
}

section[data-testid="stSidebar"] div.stRadio label {
    background: #111A28;
    border: 1px solid #263449;
    border-radius: 8px;
    padding: 10px 14px;
    color: #94A3B8 !important;
    font-weight: 500;
    font-size: 0.88rem;
    transition: all 0.15s ease-in-out;
    cursor: pointer;
}

section[data-testid="stSidebar"] div.stRadio label:hover {
    border-color: #38BDF8;
    color: #F1F5F9 !important;
    background: #141F2E;
}

section[data-testid="stSidebar"] div.stRadio label[data-checked="true"] {
    border-color: #38BDF8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    color: #38BDF8 !important;
    font-weight: 600;
}

/* Typography Hierarchy */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: #F1F5F9 !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    margin-top: 0 !important;
}

p, span, label, div {
    color: #CBD5E1;
}

.mono-text {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Enterprise Header Bar */
.gl-enterprise-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem 1.5rem;
    background: #111A28;
    border: 1px solid #263449;
    border-radius: 12px;
    margin-bottom: 1.75rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.gl-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    background: rgba(16, 185, 129, 0.1);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.gl-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10B981;
    box-shadow: 0 0 8px #10B981;
}

/* Cards & Surfaces */
.gl-card {
    background: #141F2E;
    border: 1px solid #263449;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.gl-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
}

/* KPI Metric Cards */
.gl-kpi-card {
    background: #111A28;
    border: 1px solid #263449;
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
.gl-kpi-card.primary::before { background-color: #38BDF8; }
.gl-kpi-card.indigo::before { background-color: #818CF8; }

.gl-kpi-label {
    font-size: 0.75rem;
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
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.gl-badge-supported {
    background: rgba(16, 185, 129, 0.12);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.gl-badge-contradicted {
    background: rgba(239, 68, 68, 0.12);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.gl-badge-unverified {
    background: rgba(245, 158, 11, 0.12);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Evidence Inspection Panel */
.gl-evidence-box {
    background: #0D1420;
    border: 1px solid #263449;
    border-left: 3px solid #38BDF8;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-top: 0.85rem;
}

.gl-evidence-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #38BDF8;
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

/* Input Fields & Textareas */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background-color: #111A28 !important;
    border: 1px solid #263449 !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

/* Buttons */
.stButton > button {
    background: #141F2E !important;
    border: 1px solid #263449 !important;
    color: #F1F5F9 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #1E2D42 !important;
    border-color: #38BDF8 !important;
    color: #38BDF8 !important;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
    border: 1px solid #38BDF8 !important;
    color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0369A1 0%, #1D4ED8 100%) !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.35) !important;
}

/* Progress and Metrics */
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #F1F5F9 !important;
}

/* Expander Overrides */
div[data-testid="stExpander"] {
    background: #111A28 !important;
    border: 1px solid #263449 !important;
    border-radius: 8px !important;
    margin-top: 0.75rem !important;
}

/* Tables */
div[data-testid="stTable"] table {
    background-color: #111A28 !important;
    border: 1px solid #263449 !important;
    border-radius: 8px !important;
}

div[data-testid="stTable"] th {
    background-color: #141F2E !important;
    color: #94A3B8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    border-bottom: 1px solid #263449 !important;
}

div[data-testid="stTable"] td {
    color: #E2E8F0 !important;
    border-bottom: 1px solid rgba(38, 52, 73, 0.6) !important;
    font-size: 0.88rem !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #080B12;
}
::-webkit-scrollbar-thumb {
    background: #263449;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #38BDF8;
}
</style>
"""
