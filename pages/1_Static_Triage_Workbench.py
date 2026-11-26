"""Unified Static Triage Workbench — Multi-File Mixed Triage (PE, PDF, CSV)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.triage_views import (
    PE_EXTENSIONS,
    process_mixed_uploaded_files,
    render_mixed_triage_results,
)
from src.ui_theme import inject_vortex_theme, render_page_header

st.set_page_config(
    page_title="Static Triage Workbench | Vortex Sentinel",
    page_icon="assets/vortex_sentinel_logo.png",
    layout="wide",
)

inject_vortex_theme()

render_page_header(
    "🧪 Unified Static Triage Workbench",
    "Upload multiple and mixed files (Windows PE binaries, PDF documents, and CSV feature batches) in one dropzone and inspect combined + per-file results.",
    badge="MULTI-FILE MIXED TRIAGE",
)

st.markdown(
    '<div class="vortex-bento-cyan" style="padding:10px 14px;margin-bottom:10px;">'
    '<span class="vortex-kpi-label">End-to-End Triage Flow:</span> &nbsp;'
    '<span class="vortex-mono">1. Select File</span> → '
    '<span class="vortex-mono">2. Static Feature Extraction</span> → '
    '<span class="vortex-mono">3. ML Risk Score</span> → '
    '<span class="vortex-mono">4. SHAP + Analyst Brief</span> → '
    '<span class="vortex-mono">5. Inspection Report</span>'
    '</div>',
    unsafe_allow_html=True,
)

with st.expander("Static Triage Scope & Limitations"):
    st.markdown(
        "- **Entropy & Packing Context:** High Shannon entropy or packed sections raise triage priority for deeper inspection "
        "but do **not** guarantee maliciousness (legitimate installers and compressed binaries can have high entropy).\n"
        "- **Zero-Execution Queue Prioritization:** Static triage prioritizes analyst queues in milliseconds without execution "
        "(`preview_allowed=False`) before deeper sandbox review."
    )

ov1, ov2, ov3 = st.columns(3)
with ov1:
    st.markdown(
        """
        <div class="vortex-bento">
            <div class="vortex-bento-header">
                <span>PE Binary Scanner</span>
                <span class="vortex-mono">.EXE .DLL .SYS .SCR</span>
            </div>
            <p style="margin:0;font-size:0.84rem;color:#CBD5E1;">
                In-memory PE header, entropy, Authenticode, and SHAP risk scoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with ov2:
    st.markdown(
        """
        <div class="vortex-bento-cyan">
            <div class="vortex-bento-header">
                <span>PDF Inspector</span>
                <span class="vortex-mono">.PDF • PREVIEW=FALSE</span>
            </div>
            <p style="margin:0;font-size:0.84rem;color:#CBD5E1;">
                Page/time-bounded static inspection with JS &amp; attachment routing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with ov3:
    st.markdown(
        """
        <div class="vortex-bento">
            <div class="vortex-bento-header">
                <span>Batch CSV Analysis</span>
                <span class="vortex-mono">.CSV • STRICT SCHEMA</span>
            </div>
            <p style="margin:0;font-size:0.84rem;color:#CBD5E1;">
                Strict column &amp; numeric validation before batch SHAP scoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

ctrl1, ctrl2, ctrl3 = st.columns([1.4, 0.8, 0.8])
with ctrl1:
    model_key = st.selectbox(
        "Classifier Model (PE & CSV Scoring)",
        options=["random_forest", "hist_gradient_boosting"],
        format_func=lambda k: "RandomForestClassifier" if k == "random_forest" else "HistGradientBoostingClassifier",
        key="wb_mixed_model",
    )
with ctrl2:
    max_pages = st.number_input("PDF Max Pages", min_value=1, max_value=200, value=50, key="wb_pdf_pages")
with ctrl3:
    max_seconds = st.number_input("PDF Time Limit (s)", min_value=1.0, max_value=30.0, value=10.0, key="wb_pdf_secs")

uploaded_files = st.file_uploader(
    "Upload Mixed Files (.exe, .dll, .sys, .scr, .pdf, .csv)",
    type=["exe", "dll", "sys", "scr", "pdf", "csv"],
    accept_multiple_files=True,
    key="wb_mixed_uploader",
)

if uploaded_files:
    pe_q = sum(1 for f in uploaded_files if Path(f.name).suffix.lower() in PE_EXTENSIONS)
    pdf_q = sum(1 for f in uploaded_files if Path(f.name).suffix.lower() == ".pdf")
    csv_q = sum(1 for f in uploaded_files if Path(f.name).suffix.lower() == ".csv")
    other_q = len(uploaded_files) - (pe_q + pdf_q + csv_q)
    st.markdown(
        f'<div class="vortex-bento-cyan" style="padding:10px 14px;">'
        f'<span class="vortex-kpi-label">Queued for Triage:</span> '
        f'<strong>{len(uploaded_files)} file(s)</strong> &nbsp;|&nbsp; '
        f'<span class="vortex-mono">PE: {pe_q}</span> '
        f'<span class="vortex-mono">PDF: {pdf_q}</span> '
        f'<span class="vortex-mono">CSV: {csv_q}</span> '
        + (f'<span class="vortex-mono">Other: {other_q}</span>' if other_q else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    if st.button("🚀 Run Triage on All Uploaded Files", type="primary", use_container_width=True):
        with st.spinner("Running zero-execution static triage across all uploaded files..."):
            st.session_state["wb_mixed_results"] = process_mixed_uploaded_files(
                uploaded_files,
                model_key=model_key,
                max_pages=int(max_pages),
                max_seconds=float(max_seconds),
            )

if "wb_mixed_results" in st.session_state and st.session_state["wb_mixed_results"]:
    render_mixed_triage_results(st.session_state["wb_mixed_results"], key_prefix="wb_mixed")
