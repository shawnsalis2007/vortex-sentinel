"""Executable File Scanner (EXE / DLL / SYS / SCR) — Zero-Execution."""

from __future__ import annotations

import streamlit as st

from src.triage_views import render_pe_scanner_view
from src.ui_theme import inject_vortex_theme, render_page_header

st.set_page_config(page_title="Executable Scanner | Vortex Sentinel", page_icon="assets/vortex_sentinel_logo.png", layout="wide")
inject_vortex_theme()

render_page_header(
    "🛡️ Executable File Scanner (PE Static Analysis)",
    "Inspects Windows PE headers, entropy, sections, imports, and strings strictly in memory without execution.",
    badge="ZERO-EXECUTION PE ENGINE",
)

render_pe_scanner_view(key_prefix="standalone_pe")
