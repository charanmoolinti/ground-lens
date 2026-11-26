"""
Module: assets.py
Description: Enterprise SVG asset library and bespoke brand identity for GROUND LENS.
Contains:
- Bespoke Optical Verification Lens Reticle + High-Contrast Brand Typography
- Resilient HTML+SVG Brand Lockups guaranteed to render without layout collapsing
- Lucide-style single-color SVG icons for navigation, status, telemetry, and actions
- Zero emoji dependencies
- Safe Streamlit HTML renderer injecting directly into DOM
"""

from __future__ import annotations
import re
import streamlit as st


def render_html(html_str: str) -> None:
    """
    Safely renders custom HTML/SVG in Streamlit by injecting directly into the DOM
    using markdown with unsafe_allow_html=True. Completely avoids iframe isolation
    and prevents CommonMark indented <pre><code> code-block parsing by stripping
    leading indentation from each line.
    """
    clean = re.sub(r"<!--.*?-->", "", html_str, flags=re.DOTALL)
    stripped_lines = [line.strip() for line in clean.splitlines() if line.strip()]
    compact = "\n".join(stripped_lines)
    st.markdown(compact, unsafe_allow_html=True)


# ==============================================================================
# BESPOKE BRAND LOGO & TAGLINE ASSETS
# ==============================================================================

def get_brand_lockup(size: str = "header") -> str:
    """
    Renders high-contrast, resilient brand lockup combining precision SVG vector lens
    with crisp HTML typography and product tagline. Guaranteed never to collapse.
    """
    if size == "hero":
        icon_size = 54
        title_size = "1.85rem"
        sub_size = "0.95rem"
        margin_sub = "4px"
    elif size == "sidebar":
        icon_size = 36
        title_size = "1.25rem"
        sub_size = "0.72rem"
        margin_sub = "2px"
    else:  # header
        icon_size = 42
        title_size = "1.45rem"
        sub_size = "0.76rem"
        margin_sub = "3px"

    icon_svg = (
        f'<svg width="{icon_size}" height="{icon_size}" viewBox="0 0 56 56" fill="none" style="display:inline-block; vertical-align:middle; flex-shrink:0;">'
        '<circle cx="28" cy="28" r="26" fill="#0D1420" stroke="#1E293B" stroke-width="1.5" />'
        '<circle cx="28" cy="28" r="23" stroke="#263449" stroke-width="1.2" stroke-dasharray="3 4" />'
        '<circle cx="28" cy="28" r="18" stroke="#00D9FF" stroke-width="2.5" fill="none" />'
        '<line x1="28" y1="5" x2="28" y2="11" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
        '<line x1="28" y1="45" x2="28" y2="51" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
        '<line x1="5" y1="28" x2="11" y2="28" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
        '<line x1="45" y1="28" x2="51" y2="28" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
        '<line x1="6" y1="50" x2="50" y2="50" stroke="#00D9FF" stroke-width="2" stroke-linecap="round" opacity="0.8" />'
        '<line x1="16" y1="50" x2="16" y2="54" stroke="#00D9FF" stroke-width="1.2" />'
        '<line x1="40" y1="50" x2="40" y2="54" stroke="#8B5CF6" stroke-width="1.2" />'
        '<circle cx="28" cy="28" r="5" fill="#080B12" stroke="#00D9FF" stroke-width="2" />'
        '<circle cx="28" cy="28" r="2.2" fill="#00D9FF" />'
        '<path d="M 23 28 L 26.5 32 L 35 22" stroke="#10B981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none" />'
        '</svg>'
    )

    return (
        f'<div class="gl-brand-lockup" style="display:flex; align-items:center; gap:14px; user-select:none;">'
        f'{icon_svg}'
        '<div>'
        f'<div style="font-family: -apple-system, BlinkMacSystemFont, \'Inter\', sans-serif; font-size:{title_size}; font-weight:800; letter-spacing:2px; color:#F1F5F9; line-height:1.1;">'
        'GROUND<span style="color:#00D9FF; text-shadow:0 0 14px rgba(0,217,255,0.45);">LENS</span>'
        '</div>'
        f'<div style="font-family: -apple-system, BlinkMacSystemFont, \'Inter\', sans-serif; font-size:{sub_size}; font-weight:500; color:#94A3B8; letter-spacing:0.4px; margin-top:{margin_sub}; white-space:nowrap;">'
        'Ground Every Answer. Trust Only the Evidence.'
        '</div>'
        '</div>'
        '</div>'
    )


LOGO_FULL_SVG = (
    '<svg width="340" height="52" viewBox="0 0 340 52" fill="none" xmlns="http://www.w3.org/2000/svg" class="gl-logo-full" style="display:block;">'
    '<defs>'
    '<linearGradient id="glLensGradFull" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#00D9FF" />'
    '<stop offset="100%" stop-color="#8B5CF6" />'
    '</linearGradient>'
    '</defs>'
    '<g transform="translate(4, 2)">'
    '<circle cx="24" cy="24" r="22" stroke="#1E293B" stroke-width="1.5" />'
    '<circle cx="24" cy="24" r="19" stroke="#263449" stroke-width="1.2" stroke-dasharray="3 3" />'
    '<circle cx="24" cy="24" r="15" stroke="url(#glLensGradFull)" stroke-width="2.2" fill="none" />'
    '<line x1="24" y1="5" x2="24" y2="10" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
    '<line x1="24" y1="38" x2="24" y2="43" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
    '<line x1="5" y1="24" x2="10" y2="24" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
    '<line x1="38" y1="24" x2="43" y2="24" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
    '<line x1="4" y1="42" x2="44" y2="42" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" opacity="0.8" />'
    '<circle cx="24" cy="24" r="4.5" fill="#080B12" stroke="#00D9FF" stroke-width="1.8" />'
    '<circle cx="24" cy="24" r="2" fill="#00D9FF" />'
    '<path d="M 20 24 L 23 27 L 30 19" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />'
    '</g>'
    '<text x="60" y="27" font-family="-apple-system, BlinkMacSystemFont, \'Inter\', sans-serif" font-size="20" font-weight="800" letter-spacing="2" fill="#F1F5F9">'
    'GROUND<tspan fill="#00D9FF">LENS</tspan>'
    '</text>'
    '<text x="61" y="42" font-family="-apple-system, BlinkMacSystemFont, \'Inter\', sans-serif" font-size="8.5" font-weight="500" letter-spacing="0.4" fill="#94A3B8">'
    'Ground Every Answer. Trust Only the Evidence.'
    '</text>'
    '</svg>'
)

LOGO_COMPACT_SVG = (
    '<svg width="44" height="44" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg" class="gl-logo-compact" style="display:block;">'
    '<defs>'
    '<linearGradient id="glLensCompactGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#00D9FF" />'
    '<stop offset="100%" stop-color="#8B5CF6" />'
    '</linearGradient>'
    '</defs>'
    '<circle cx="28" cy="28" r="26" fill="#0D1420" stroke="#1E293B" stroke-width="1.5" />'
    '<circle cx="28" cy="28" r="18" stroke="url(#glLensCompactGrad)" stroke-width="2.2" fill="none" />'
    '<line x1="6" y1="48" x2="50" y2="48" stroke="#00D9FF" stroke-width="1.5" stroke-linecap="round" opacity="0.7" />'
    '<circle cx="28" cy="28" r="4" fill="#080B12" stroke="#00D9FF" stroke-width="1.8" />'
    '<circle cx="28" cy="28" r="1.8" fill="#00D9FF" />'
    '<path d="M 23 28 L 26.5 31.5 L 34 23" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" />'
    '</svg>'
)

LOGO_HERO_SVG = (
    '<svg width="440" height="64" viewBox="0 0 440 64" fill="none" xmlns="http://www.w3.org/2000/svg" class="gl-logo-hero" style="display:block;">'
    '<defs>'
    '<linearGradient id="glHeroLens" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#00D9FF" />'
    '<stop offset="100%" stop-color="#8B5CF6" />'
    '</linearGradient>'
    '</defs>'
    '<g transform="translate(6, 4)">'
    '<circle cx="28" cy="28" r="26" stroke="#1E293B" stroke-width="1.6" />'
    '<circle cx="28" cy="28" r="23" stroke="#263449" stroke-width="1.2" stroke-dasharray="3 4" />'
    '<circle cx="28" cy="28" r="18" stroke="url(#glHeroLens)" stroke-width="2.5" fill="none" />'
    '<line x1="28" y1="5" x2="28" y2="11" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
    '<line x1="28" y1="45" x2="28" y2="51" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
    '<line x1="5" y1="28" x2="11" y2="28" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
    '<line x1="45" y1="28" x2="51" y2="28" stroke="#00D9FF" stroke-width="1.8" stroke-linecap="round" />'
    '<line x1="4" y1="50" x2="52" y2="50" stroke="#00D9FF" stroke-width="2" stroke-linecap="round" opacity="0.8" />'
    '<circle cx="28" cy="28" r="5" fill="#080B12" stroke="#00D9FF" stroke-width="2" />'
    '<circle cx="28" cy="28" r="2.2" fill="#00D9FF" />'
    '<path d="M 23 28 L 26.5 32 L 35 22" stroke="#10B981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none" />'
    '</g>'
    '<text x="72" y="32" font-family="-apple-system, BlinkMacSystemFont, \'Inter\', sans-serif" font-size="24" font-weight="800" letter-spacing="2.5" fill="#F1F5F9">'
    'GROUND<tspan fill="#00D9FF">LENS</tspan>'
    '</text>'
    '<text x="73" y="50" font-family="-apple-system, BlinkMacSystemFont, \'Inter\', sans-serif" font-size="10" font-weight="500" letter-spacing="0.4" fill="#94A3B8">'
    'Ground Every Answer. Trust Only the Evidence.'
    '</text>'
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
