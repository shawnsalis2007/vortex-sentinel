"""Batch Analysis — CSV Static Feature Validation & Priority Triage."""

from __future__ import annotations

import streamlit as st

from src.triage_views import render_batch_analysis_view
from src.ui_theme import inject_vortex_theme, render_page_header

st.set_page_config(page_title="Batch Analysis | Vortex Sentinel", page_icon="assets/vortex_sentinel_logo.png", layout="wide")
inject_vortex_theme()

render_page_header(
    "📦 Batch Static Feature Analysis (CSV)",
    "Validates column schema and numeric types strictly before scoring; incompatible rows are never guessed.",
    badge="STRICT SCHEMA VALIDATION",
)

render_batch_analysis_view(key_prefix="standalone_csv")
