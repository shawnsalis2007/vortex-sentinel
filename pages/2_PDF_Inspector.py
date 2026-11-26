"""PDF Inspector — Safe Static Document & Attachment Triage."""

from __future__ import annotations

import streamlit as st

from src.triage_views import render_pdf_inspector_view
from src.ui_theme import inject_vortex_theme, render_page_header

st.set_page_config(page_title="PDF Inspector | Vortex Sentinel", page_icon="assets/vortex_sentinel_logo.png", layout="wide")
inject_vortex_theme()

render_page_header(
    "📄 PDF Inspector (Zero-Execution Static Triage)",
    "Browser auto-preview is disabled (preview_allowed=False). JavaScript and embedded files are never executed.",
    badge="PREVIEW_ALLOWED=FALSE",
)

render_pdf_inspector_view(key_prefix="standalone_pdf")
