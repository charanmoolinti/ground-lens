"""
Module: assets.py
Description: Enterprise SVG asset library and bespoke brand identity for GROUND LENS.
Contains:
- Bespoke Metallic Silver 'G' + Precision Optical Lens + Document Core logo system
- Lucide-style single-color SVG icons for navigation, status, telemetry, and actions
- Zero emoji dependencies
- Safe Streamlit HTML renderer to prevent CommonMark indented code-block parsing
"""

import re
import streamlit as st


def render_html(html_str: str) -> None:
    """
    Safely renders custom HTML/SVG in Streamlit by stripping comments,
    removing empty lines, and eliminating leading 4-space indentations
    that would otherwise trigger CommonMark <pre><code> block formatting.
    """
    clean = re.sub(r"<!--.*?-->", "", html_str, flags=re.DOTALL)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    compact = " ".join(lines)
    st.markdown(compact, unsafe_allow_html=True)


# ==============================================================================
# BESPOKE BRAND LOGO ASSETS (COMPACT SINGLE-LINE SVGS WITHOUT COMMENTS)
# ==============================================================================

LOGO_FULL_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 64" fill="none" class="gl-logo-full" style="max-height: 48px; width: auto; display: block;">'
    '<defs>'
    '<linearGradient id="silverG" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#FFFFFF" />'
    '<stop offset="45%" stop-color="#CBD5E1" />'
    '<stop offset="70%" stop-color="#94A3B8" />'
    '<stop offset="100%" stop-color="#64748B" />'
    '</linearGradient>'
    '<linearGradient id="lensGlow" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#38BDF8" stop-opacity="0.8" />'
    '<stop offset="100%" stop-color="#818CF8" stop-opacity="0.3" />'
    '</linearGradient>'
    '<filter id="electricGlow" x="-20%" y="-20%" width="140%" height="140%">'
    '<feGaussianBlur stdDeviation="3" result="blur" />'
    '<feComposite in="SourceGraphic" in2="blur" operator="over" />'
    '</filter>'
    '</defs>'
    '<g transform="translate(6, 6)">'
    '<circle cx="26" cy="26" r="24" fill="url(#lensGlow)" opacity="0.25" filter="url(#electricGlow)" />'
    '<circle cx="26" cy="26" r="23" stroke="#263449" stroke-width="1.5" stroke-dasharray="2 4" />'
    '<path d="M 40 16 C 36 10, 28 8, 20 11 C 12 14, 7 22, 9 31 C 11 39, 19 45, 28 44 C 36 43, 42 37, 43 29 L 26 29" stroke="url(#silverG)" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" fill="none" />'
    '<circle cx="26" cy="26" r="14" stroke="#38BDF8" stroke-width="1.2" opacity="0.6" />'
    '<line x1="26" y1="8" x2="26" y2="12" stroke="#38BDF8" stroke-width="1.5" />'
    '<line x1="26" y1="40" x2="26" y2="44" stroke="#38BDF8" stroke-width="1.5" />'
    '<line x1="8" y1="26" x2="12" y2="26" stroke="#38BDF8" stroke-width="1.5" />'
    '<line x1="40" y1="26" x2="44" y2="26" stroke="#38BDF8" stroke-width="1.5" />'
    '<rect x="22" y="20" width="8" height="11" rx="1" fill="#0D1420" stroke="#38BDF8" stroke-width="1.2" />'
    '<line x1="24" y1="23" x2="28" y2="23" stroke="#CBD5E1" stroke-width="0.8" />'
    '<line x1="24" y1="25.5" x2="28" y2="25.5" stroke="#CBD5E1" stroke-width="0.8" />'
    '<line x1="24" y1="28" x2="27" y2="28" stroke="#38BDF8" stroke-width="0.8" />'
    '</g>'
    '<g transform="translate(68, 14)">'
    '<text x="0" y="22" font-family="\'Inter\', sans-serif" font-size="20" font-weight="800" letter-spacing="3" fill="#F1F5F9">'
    'GROUND<tspan fill="#38BDF8">LENS</tspan>'
    '</text>'
    '<text x="1" y="37" font-family="\'JetBrains Mono\', monospace" font-size="7.5" font-weight="600" letter-spacing="1.8" fill="#94A3B8">'
    'AI DOCUMENT INTELLIGENCE'
    '</text>'
    '</g>'
    '</svg>'
)

LOGO_COMPACT_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52" fill="none" class="gl-logo-compact" style="max-height: 40px; width: auto; display: block;">'
    '<defs>'
    '<linearGradient id="silverGCompact" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#FFFFFF" />'
    '<stop offset="50%" stop-color="#CBD5E1" />'
    '<stop offset="100%" stop-color="#64748B" />'
    '</linearGradient>'
    '<filter id="glowCompact" x="-20%" y="-20%" width="140%" height="140%">'
    '<feGaussianBlur stdDeviation="2" result="blur" />'
    '<feComposite in="SourceGraphic" in2="blur" operator="over" />'
    '</filter>'
    '</defs>'
    '<circle cx="26" cy="26" r="24" fill="#111A28" stroke="#263449" stroke-width="1.5" />'
    '<circle cx="26" cy="26" r="20" stroke="#38BDF8" stroke-width="1" opacity="0.3" filter="url(#glowCompact)" />'
    '<path d="M 38 17 C 34 11, 27 9, 20 12 C 13 15, 9 22, 10 30 C 12 37, 19 43, 27 42 C 34 41, 39 36, 40 29 L 26 29" stroke="url(#silverGCompact)" stroke-width="3.8" stroke-linecap="round" stroke-linejoin="round" fill="none" />'
    '<rect x="23" y="21" width="7" height="10" rx="1" fill="#080B12" stroke="#38BDF8" stroke-width="1.2" />'
    '<line x1="25" y1="24" x2="28" y2="24" stroke="#CBD5E1" stroke-width="0.8" />'
    '<line x1="25" y1="26.5" x2="28" y2="26.5" stroke="#CBD5E1" stroke-width="0.8" />'
    '</svg>'
)


# ==============================================================================
# ENTERPRISE SVG ICONS (LUCIDE-STYLE, MONOCHROME, NO EMOJIS)
# ==============================================================================

def get_svg_icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    """Returns clean inline SVG vector icon path by name without newlines."""
    icons = {
        "dashboard": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/></svg>',
        "library": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="m16 6 4 14"/><path d="M12 6v14"/><path d="M8 8v12"/><path d="M4 4v16"/></svg>',
        "verification": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/><path d="m8 11 2 2 4-4"/></svg>',
        "history": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>',
        "telemetry": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
        "settings": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>',
        "check_circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>',
        "x_circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>',
        "help_circle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>',
        "file_text": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
        "upload": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
        "trash": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>',
        "refresh": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg>',
        "cpu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
        "database": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>',
        "shield": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>',
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
        "chevron_right": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle;"><path d="m9 18 6-6-6-6"/></svg>',
    }
    return icons.get(name, "")
